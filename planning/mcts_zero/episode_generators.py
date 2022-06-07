import _pickle as cPickle
import bz2
import os
from multiprocessing import Queue
from typing import Dict, Optional, Tuple

import numpy as np

from mapping.grid_maps import GridMap
from mapping.mappings import Mapping
from planning.common.actions import action_costs, action_dict_to_np_array, enumerate_actions
from planning.common.features import EpisodeHistory, generate_input_feature_planes
from planning.common.optimization import simulate_prediction_step
from planning.common.rewards import scale_value_target
from planning.mcts_zero.mcts import MCTS, Node
from sensors import Sensor


class EpisodeGenerator:
    def __init__(
        self,
        hyper_params: Dict,
        sensor: Sensor,
        grid_map: GridMap,
        episode_counter: int,
        meta_data: Dict,
        request_queue: Queue,
        reply_queue: Queue,
        worker_id: int,
        train_data_dir: str,
    ):
        self.hyper_params = hyper_params
        self.grid_map = grid_map
        self.episode_counter = episode_counter
        self.meta_data = meta_data
        self.worker_id = worker_id
        self.train_data_dir = train_data_dir

        self.initial_budget = meta_data["initial_budget"]
        self.episode_horizon = meta_data["episode_horizon"]
        self.max_episode_steps = meta_data["max_episode_steps"]
        self.min_altitude = meta_data["min_altitude"]
        self.max_altitude = meta_data["max_altitude"]
        self.altitude_spacing = meta_data["altitude_spacing"]
        self.uav_specifications = meta_data["uav_specifications"]
        self.adaptive = meta_data["scenario_info"] is not None

        self.actions = enumerate_actions(self.grid_map, self.min_altitude, self.max_altitude, self.altitude_spacing)
        self.actions_np = action_dict_to_np_array(self.actions)
        self.actions_counter = np.zeros(self.actions_np.shape[0])
        self.init_action = self.sample_init_action()
        self.budget = self.sample_budget()
        self.mapping = Mapping(grid_map, sensor, shuffle_prior_cov=self.hyper_params["shuffle_prior_cov"])
        self.mapping.sensor.sensor_simulation.create_ground_truth_map()

        self.request_queue = request_queue
        self.reply_queue = reply_queue
        self.mcts = self.create_mcts()

    def sample_init_action(self) -> np.array:
        action_idx = np.random.choice(list(range(self.actions_np.shape[0])))
        return self.actions_np[action_idx, :]

    def sample_budget(self) -> float:
        if self.hyper_params["shuffle_budget"]:
            return int(np.random.uniform(low=10, high=self.initial_budget))

        return self.initial_budget

    def create_mcts(self) -> MCTS:
        meta_data = {
            "budget": self.budget,
            "initial_budget": self.initial_budget,
            "episode_horizon": self.episode_horizon,
            "min_altitude": self.min_altitude,
            "max_altitude": self.max_altitude,
            "altitude_spacing": self.altitude_spacing,
            "uav_specifications": self.uav_specifications,
            "scenario_info": self.get_adaptive_info(),
        }

        return MCTS(
            self.mapping,
            self.hyper_params,
            meta_data,
            self.request_queue,
            self.reply_queue,
            self.episode_counter,
            self.worker_id,
        )

    def get_adaptive_info(self) -> Optional[Dict]:
        if not self.adaptive:
            return None

        return {
            "mean": self.mapping.grid_map.mean,
            "value_threshold": self.meta_data["scenario_info"]["value_threshold"],
            "interval_factor": self.meta_data["scenario_info"]["interval_factor"],
        }

    def execute(self, num_episode: int) -> float:
        train_examples = []
        node = Node(self.mapping.grid_map.cov_matrix)
        previous_action = self.init_action
        episode_history = EpisodeHistory(self.hyper_params["input_history_length"])

        depth = 0
        remaining_budget = self.budget
        rewards = []
        adaptive_info_history = []
        while depth < self.max_episode_steps and remaining_budget >= self.grid_map.resolution:
            episode_history.push(node.state, previous_action, remaining_budget / self.initial_budget)

            if self.hyper_params["reset_mcts_each_step"]:
                self.mcts = self.create_mcts()

            temperature = self.hyper_params["temperature_scale"] * int(
                depth < self.hyper_params["temperature_threshold"]
            )
            policy, valid_actions_msk = self.mcts.get_policy(
                node,
                0,
                previous_action,
                remaining_budget,
                episode_history,
                temperature=temperature,
            )

            if policy is None:
                break

            adaptive_info_history.append(self.get_adaptive_info())
            train_examples.append((node.state, previous_action, remaining_budget, valid_actions_msk, policy))
            action_idx = np.random.choice(len(policy), p=policy)
            action = self.actions_np[action_idx, :]
            reward, _, next_state = simulate_prediction_step(
                node.state,
                previous_action,
                action,
                self.mapping,
                self.uav_specifications,
                self.get_adaptive_info(),
            )
            simulated_raw_measurement = self.mapping.sensor.take_measurement(action, verbose=False)
            self.mapping.update_grid_map(action, simulated_raw_measurement)

            remaining_budget -= action_costs(action, previous_action, self.uav_specifications)
            next_node = Node(next_state)
            previous_action = action
            rewards.append(reward)
            node = next_node
            depth += 1

            self.actions_counter[action_idx] += 1

        episode_history = EpisodeHistory(self.hyper_params["input_history_length"])
        total_episode_value = sum([self.hyper_params["gamma"] ** j * rewards[j] for j in range(len(rewards))])
        for i, reward in enumerate(rewards):
            state, action, budget, valid_actions_msk, policy = train_examples[i]
            normalized_budget = budget / self.initial_budget
            bootstrapped_idx = min(i + self.episode_horizon, len(rewards))
            value = sum([self.hyper_params["gamma"] ** j * rewards[j] for j in range(i, bootstrapped_idx)])
            value = scale_value_target(value)

            min_altitude, max_altitude = self.min_altitude, self.max_altitude
            if self.hyper_params["use_fov_input"]:
                min_altitude, max_altitude = None, None

            episode_history.push(state, action, normalized_budget)
            input_feature_planes = generate_input_feature_planes(
                self.mapping,
                episode_history,
                min_altitude,
                max_altitude,
                adaptive_info=adaptive_info_history[i],
                uav_specifications=self.uav_specifications,
                use_action_costs_input=self.hyper_params["use_action_costs_input"],
            )
            train_examples[i] = (input_feature_planes, policy, value, reward, valid_actions_msk)

            self.save_sample_to_disk(train_examples[i], num_episode, i)

        return total_episode_value

    def save_sample_to_disk(self, train_sample: Tuple, num_episode: int, episode_step: int):
        sample_file_path = os.path.join(
            self.train_data_dir, f"sample_{self.worker_id}_{num_episode}_{episode_step}.pkl"
        )

        with bz2.BZ2File(sample_file_path, "wb") as file:
            cPickle.dump(train_sample, file)
