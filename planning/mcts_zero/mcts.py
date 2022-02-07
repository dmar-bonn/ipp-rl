import copy
import logging
from multiprocessing import Queue
from typing import Dict, List, Optional, Tuple

import numpy as np

from mapping.mappings import Mapping
from planning.common.actions import action_costs, action_dict_to_np_array, compute_flight_times, enumerate_actions
from planning.common.features import EpisodeHistory, generate_input_feature_planes
from planning.common.optimization import simulate_prediction_step

logger = logging.getLogger(__name__)


class Node:
    def __init__(self, state: np.array):
        self.state = state

    def state_representation(self) -> int:
        return hash(str(self.state))


class MCTS:
    def __init__(
        self,
        mapping: Mapping,
        hyper_params: Dict,
        meta_data: Dict,
        request_queue: Queue,
        reply_queue: Queue,
        episode_counter: int = -1,
        worker_id: int = 0,
    ):
        self.mapping = mapping
        self.hyper_params = hyper_params
        self.meta_data = meta_data
        self.request_queue = request_queue
        self.reply_queue = reply_queue
        self.budget = meta_data["budget"]
        self.initial_budget = meta_data["initial_budget"]
        self.uav_specifications = meta_data["uav_specifications"]
        self.puct_init = hyper_params["puct_init"]
        self.puct_base = hyper_params["puct_base"]
        self.episode_horizon = meta_data["episode_horizon"]
        self.num_simulations = hyper_params["num_mcts_simulations"]
        self.gamma = hyper_params["gamma"]
        self.dirichlet_alpha = hyper_params["dirichlet_alpha"]
        self.dirichlet_eps = hyper_params["dirichlet_eps"]
        self.actions = enumerate_actions(
            self.mapping.grid_map, meta_data["min_altitude"], meta_data["max_altitude"], meta_data["altitude_spacing"]
        )
        self.actions_np = action_dict_to_np_array(self.actions)
        self.adaptive = meta_data["scenario_info"] is not None

        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Vs = {}
        self.Ps = {}

        self.episode_counter = episode_counter
        self.worker_id = worker_id
        self.policy_counter = 0
        self.inference_counter = 0
        self.revisits_counter = 0
        self.new_visits_counter = 0

    @property
    def num_actions(self):
        return self.actions_np.shape[0]

    def get_adaptive_info(self) -> Optional[Dict]:
        if not self.adaptive:
            return None

        return {
            "mean": self.mapping.grid_map.mean,
            "value_threshold": self.meta_data["scenario_info"]["value_threshold"],
            "interval_factor": self.meta_data["scenario_info"]["interval_factor"],
        }

    def get_policy(
        self,
        root: Node,
        depth: int,
        previous_action: np.array,
        budget: float,
        episode_history: EpisodeHistory,
        temperature: float = 1,
        deploy_time: bool = False,
    ) -> Optional[Tuple[List, np.array]]:
        for i in range(self.num_simulations):
            self.simulate(root, depth, budget, previous_action, episode_history, num_sim=i)

        root_rep = root.state_representation()
        actions_visits = np.array([self.Nsa[root_rep][action] for action in range(self.num_actions)])

        if not deploy_time:
            argmax_action_visit = np.random.choice((actions_visits == np.max(actions_visits)).nonzero()[0])
            no_playouts_msk = self.Nsa[root_rep] == 0
            num_forced_playouts = np.ceil(
                np.sqrt(self.hyper_params["forced_playout_factor"] * self.Ps[root_rep] * self.Ns[root_rep])
            )
            num_forced_playouts[no_playouts_msk] = 0
            max_puct = self.compute_uct(root_rep, force_playouts=False)[argmax_action_visit]

            for action_idx in range(len(num_forced_playouts)):
                if action_idx == argmax_action_visit:
                    continue

                if num_forced_playouts[action_idx] <= 0:
                    continue

                for i in range(int(num_forced_playouts[action_idx])):
                    actions_visits[action_idx] = actions_visits[action_idx] - 1
                    qsa_normalized = self.normalize_q_values(self.Qsa[root_rep])[action_idx]
                    prior_score = self.puct_init + np.log((self.Ns[root_rep] + self.puct_base + 1) / self.puct_base)
                    prior_score *= self.Ps[root_rep][action_idx] * (
                        np.sqrt(self.Ns[root_rep] + 1) / (1 + actions_visits[action_idx])
                    )
                    pruned_puct = qsa_normalized + prior_score

                    if pruned_puct >= max_puct:
                        actions_visits[action_idx] = actions_visits[action_idx] + 1
                        break

            actions_visits[actions_visits == 1] = 0

        if np.sum(actions_visits) == 0:
            logger.error(f"No valid action from current root node visited. Try increasing number of MCTS simulations!")
            return None

        if temperature == 0:
            best_action = np.random.choice(np.array(np.argwhere(actions_visits == np.max(actions_visits))).flatten())
            policy = [0] * len(actions_visits)
            policy[best_action] = 1
            return policy, actions_visits

        actions_visits_temp = np.array([action_visits ** (1.0 / temperature) for action_visits in actions_visits])
        policy = actions_visits_temp / np.sum(actions_visits_temp)

        return policy.tolist(), self.get_next_actions_mask(previous_action, budget)

    def is_leaf_node(self, node_rep: int) -> bool:
        return node_rep not in self.Ps.keys()

    def get_next_actions_mask(self, position: np.array, budget: float, uav_specificaion: Dict = None) -> np.array:
        distances = np.linalg.norm(self.actions_np - position, ord=2, axis=1)
        if uav_specificaion is None:
            return (
                (distances > 0) & (distances <= budget) & (distances < self.hyper_params["max_valid_action_distance"])
            )

        flight_times = compute_flight_times(self.actions_np, position, uav_specificaion)
        return (
            (flight_times > 0) & (flight_times <= budget) & (distances < self.hyper_params["max_valid_action_distance"])
        )

    def add_exploration_noise(self, node_rep: int):
        self.Ps[node_rep] = (1 - self.dirichlet_eps) * self.Ps[node_rep] + self.dirichlet_eps * np.random.dirichlet(
            [self.dirichlet_alpha] * self.num_actions
        )
        self.Ps[node_rep] = self.Ps[node_rep] / np.sum(self.Ps[node_rep])

    def simulate(
        self,
        node: Node,
        depth: int,
        budget: float,
        previous_action: np.array,
        episode_history: EpisodeHistory,
        num_sim: int = -1,
    ) -> float:
        if depth > self.episode_horizon or budget <= 0:
            return 0

        episode_history.push(node.state, copy.deepcopy(previous_action), budget / self.initial_budget)

        node_rep = node.state_representation()
        if node_rep not in self.Nsa:
            self.Nsa[node_rep] = np.zeros(self.num_actions)
            self.Qsa[node_rep] = np.zeros(self.num_actions)

        if self.is_leaf_node(node_rep):
            min_altitude, max_altitude = self.meta_data["min_altitude"], self.meta_data["max_altitude"]
            if self.hyper_params["use_fov_input"]:
                min_altitude, max_altitude = None, None
            input_feature_planes = generate_input_feature_planes(
                self.mapping,
                episode_history,
                min_altitude,
                max_altitude,
                adaptive_info=self.get_adaptive_info(),
                uav_specifications=self.uav_specifications,
                use_action_costs_input=self.hyper_params["use_action_costs_input"],
            )

            next_actions_msk = self.get_next_actions_mask(previous_action, budget)
            if next_actions_msk.sum() == 0:
                return 0

            request_msg = {"id": self.worker_id, "input": input_feature_planes, "action_msk": next_actions_msk}
            self.request_queue.put(request_msg)
            while True:
                if self.hyper_params["non_blocking_read"]:
                    if self.reply_queue.empty():
                        continue

                    msg = self.reply_queue.get_nowait()
                else:
                    msg = self.reply_queue.get()

                if "policy" in msg and "value" in msg:
                    self.Ps[node_rep], value = msg["policy"], msg["value"]
                    break

            self.Ps[node_rep] = self.Ps[node_rep] * next_actions_msk
            self.inference_counter += 1

            if depth == 0 and num_sim == 0:
                self.add_exploration_noise(node_rep)

            if np.sum(self.Ps[node_rep]) > 0:
                self.Ps[node_rep] /= np.sum(self.Ps[node_rep])
            else:
                logger.error("All valid moves have 0 probability")
                self.Ps[node_rep] = self.Ps[node_rep] + next_actions_msk
                self.Ps[node_rep] /= np.sum(self.Ps[node_rep])

            self.Vs[node_rep] = next_actions_msk
            self.Ns[node_rep] = 0
            return value

        uct = self.compute_uct(node_rep, force_playouts=(depth == 0))
        action_idx = np.random.choice((uct == np.max(uct)).nonzero()[0])
        sampled_action = copy.deepcopy(self.actions_np[action_idx, :])

        reward, _, next_state = simulate_prediction_step(
            node.state,
            previous_action,
            sampled_action,
            self.mapping,
            self.uav_specifications,
            self.get_adaptive_info(),
        )
        next_node = Node(next_state)
        del node
        budget -= action_costs(sampled_action, previous_action, self.uav_specifications)
        value = reward + self.gamma * self.simulate(
            next_node, depth + 1, budget, sampled_action, episode_history, num_sim=num_sim
        )
        if self.Nsa[node_rep][action_idx] > 0:
            self.Qsa[node_rep][action_idx] = (
                self.Nsa[node_rep][action_idx] * self.Qsa[node_rep][action_idx] + value
            ) / (self.Nsa[node_rep][action_idx] + 1)
            self.Nsa[node_rep][action_idx] += 1
            self.revisits_counter += 1
        else:
            self.Qsa[node_rep][action_idx] = value
            self.Nsa[node_rep][action_idx] = 1
            self.new_visits_counter += 1

        self.Ns[node_rep] += 1
        return value

    @staticmethod
    def normalize_q_values(values: np.array) -> np.array:
        if np.all(values == 0):
            return values

        min_value = np.min(values)
        max_value = np.max(values)

        if min_value == max_value:
            return values / max_value

        return (values - min_value) / (max_value - min_value)

    def compute_uct(self, node_rep: int, force_playouts: bool = False) -> np.array:
        qsa_normalized = self.normalize_q_values(self.Qsa[node_rep])
        prior_score = self.puct_init + np.log((self.Ns[node_rep] + self.puct_base + 1) / self.puct_base)
        prior_score *= self.Ps[node_rep] * (np.sqrt(self.Ns[node_rep] + 1) / (1 + self.Nsa[node_rep]))
        uct = qsa_normalized + prior_score

        if force_playouts:
            no_playouts_msk = self.Nsa[node_rep] == 0
            num_forced_playouts = np.ceil(
                np.sqrt(self.hyper_params["forced_playout_factor"] * self.Ps[node_rep] * self.Ns[node_rep])
            )
            num_forced_playouts[no_playouts_msk] = 0
            to_be_forced_playouts_msk = self.Nsa[node_rep] < num_forced_playouts
            uct[to_be_forced_playouts_msk] = np.inf

        uct[~self.Vs[node_rep]] = -np.inf
        return uct
