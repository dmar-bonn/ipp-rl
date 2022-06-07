import logging
import os
import shutil
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.multiprocessing import Manager, Pool, Queue
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from constants import CHECKPOINTS_DIR, LOG_DIR, MissionType, TELEGRAM_CHAT_ID, TELEGRAM_TOKEN, TRAIN_DATA_DIR
from experiments.notifications import TelegramNotifier
from mapping.mappings import Mapping
from planning.common.actions import (
    action_costs,
    action_dict_to_np_array,
    compute_flight_time,
    compute_flight_times,
    enumerate_actions,
)
from planning.common.features import EpisodeHistory, generate_input_feature_planes
from planning.mcts_zero.arenas import Arena
from planning.mcts_zero.episode_generators import EpisodeGenerator
from planning.mcts_zero.inference_workers import inference_worker
from planning.mcts_zero.mcts import MCTS, Node
from planning.mcts_zero.network_wrappers.policy_network_wrappers import PolicyNetworkWrapper
from planning.mcts_zero.network_wrappers.policy_value_network_wrappers import PolicyValueNetworkWrapper
from planning.mcts_zero.network_wrappers.value_network_wrappers import ValueNetworkWrapper
from planning.missions import Mission

logger = logging.getLogger(__name__)


def run_deploy_time_mcts_worker(
    mcts: MCTS,
    root: Node,
    previous_action: np.array,
    budget: float,
    episode_history: EpisodeHistory,
    planning_step: int,
):
    np.random.seed(42 * planning_step + mcts.worker_id)
    policy, _ = mcts.get_policy(
        root,
        0,
        previous_action,
        budget,
        episode_history,
        temperature=1,
        deploy_time=True,
    )

    return policy


def run_episode_worker(
    hyper_params: Dict,
    mapping: Mapping,
    episode_index: int,
    meta_data: Dict,
    num_episodes: int,
    request_queue: Queue,
    reply_queue: Queue,
    worker_id: int,
    self_play_iter: int,
) -> List:
    np.random.seed(42 * self_play_iter + worker_id)
    episode_values = []
    for i in range(num_episodes):
        episode_generator = EpisodeGenerator(
            hyper_params,
            mapping.sensor,
            mapping.grid_map,
            episode_index + i,
            meta_data,
            request_queue,
            reply_queue,
            worker_id,
            os.path.join(TRAIN_DATA_DIR, f"iter_{self_play_iter}"),
        )
        episode_value = episode_generator.execute(i)
        episode_values.append(episode_value)

        logger.info(f"GENERATED EPISODE {i} IN WORKER {worker_id} WITH VALUE {episode_value}")

        del episode_value
        del episode_generator

    return episode_values


class MCTSZeroMission(Mission):
    def __init__(
        self,
        mapping: Mapping,
        uav_specifications: Dict,
        hyper_params: Dict,
        dist_to_boundaries: float = 10,
        min_altitude: float = 5,
        max_altitude: float = 30,
        episode_horizon: int = 10,
        altitude_spacing: float = 5,
        budget: float = 100,
        model_deployment_filename: str = "best.pth.tar",
        train_examples_iter: int = 0,
        restart_training: bool = False,
        adaptive: bool = False,
        value_threshold: float = 0.5,
        interval_factor: float = 2,
        telegram_notifications: bool = False,
        config_name: str = "standard",
        use_effective_mission_time: bool = False,
    ):
        """
        Defines a mission greedily optimizing measurement positions offline.

        Args:
            mapping (Mapping): mapping algorithm with related sensor and grid map specification
            uav_specifications (dict): uav parameters defining max_v, max_a, sampling_time
            dist_to_boundaries (float): minimal distance [m] of a waypoint to map boundaries
            min_altitude (float): lowest altitude level [m] of waypoints above ground
            max_altitude (float): highest altitude level [m] of waypoints above ground
            episode_horizon (int): total number of greedily sampled waypoints
            altitude_spacing (float): spacing [m] between consecutive altitude levels
            budget (float): total distance budget for mission
            hyper_params (dict): includes all hyper-parameters for self-play MCTS learning algorithm
            model_deployment_filename (str): filename of trained and saved policy-value network
            train_examples_iter (int): iteration number of already produced train data by self-play
            restart_training (bool): if true, load pretrained model before continuing training
            adaptive (bool): indicates if mission should be executed as adaptive or non-adaptive IPP
            value_threshold (float): grid cells with upper CI bounds above this threshold are of interest
            interval_factor (float): defines the width of the CI used to decide if a grid cell is of interest
            telegram_notifications (bool): telegram start, progress, and failure notifications are sent
            config_name (str): descriptive name of chosen mission's hyper-parameter configuration
            use_effective_mission_time (bool): if true, decrease remaining budget additionally by thinking time
        """
        super().__init__(
            mapping,
            uav_specifications,
            dist_to_boundaries,
            min_altitude,
            max_altitude,
            budget,
            adaptive,
            value_threshold,
            interval_factor,
            config_name,
            use_effective_mission_time,
        )

        self.episode_horizon = episode_horizon
        self.altitude_spacing = altitude_spacing
        self.hyper_params = hyper_params
        self.initial_budget = budget
        self.model_deployment_filename = model_deployment_filename
        self.train_examples_iter = train_examples_iter
        self.train_examples_filename = os.path.join(TRAIN_DATA_DIR, f"iter_{train_examples_iter}")
        self.restart_training = restart_training

        self.actions = enumerate_actions(
            self.mapping.grid_map, self.min_altitude, self.max_altitude, self.altitude_spacing
        )
        self.actions_np = action_dict_to_np_array(self.actions)
        self.skip_first_self_play = False

        self.mission_name = "Ours"
        self.mission_type = MissionType.MCTS_ZERO
        self.prev_network_wins = 0
        self.actions_counter = np.zeros(self.mapping.grid_map.num_grid_cells)
        self.meta_data = self.get_meta_data()

        hparams_metrics = {f"hparam/{key}": value for key, value in self.hyper_params.items()}

        if self.hyper_params["shared_network"]:
            self.shared_network = PolicyValueNetworkWrapper(self.hyper_params, self.meta_data)
            self.shared_network.set_summary_writer(SummaryWriter(os.path.join(LOG_DIR, "tensorboard")))
            self.shared_network.writer.add_hparams(self.hyper_params, hparams_metrics)
            self.prev_shared_network = self.shared_network.__class__(self.hyper_params, self.meta_data)
        else:
            self.policy_network = PolicyNetworkWrapper(self.hyper_params, self.meta_data)
            self.policy_network.set_summary_writer(SummaryWriter(os.path.join(LOG_DIR, "tensorboard")))
            self.policy_network.writer.add_hparams(self.hyper_params, hparams_metrics)

            self.value_network = ValueNetworkWrapper(self.hyper_params)
            self.value_network.set_summary_writer(SummaryWriter(os.path.join(LOG_DIR, "tensorboard")))

            self.prev_policy_network = self.policy_network.__class__(self.hyper_params, self.meta_data)
            self.prev_value_network = self.value_network.__class__(self.hyper_params)

        self.mcts = None
        self.telegram_notifications = telegram_notifications
        self.telegram_notifier = TelegramNotifier(
            self.mission_label,
            TELEGRAM_TOKEN,
            TELEGRAM_CHAT_ID,
            hyper_params=self.hyper_params,
            verbose=True,
        )

    def get_meta_data(self) -> Dict:
        return {
            "budget": self.budget,
            "initial_budget": self.initial_budget,
            "episode_horizon": self.episode_horizon,
            "max_episode_steps": self.hyper_params["max_episode_steps"],
            "min_altitude": self.min_altitude,
            "max_altitude": self.max_altitude,
            "altitude_spacing": self.altitude_spacing,
            "cov_matrix_shape": self.mapping.grid_map.cov_matrix.shape,
            "num_grid_cells": self.mapping.grid_map.num_grid_cells,
            "uav_specifications": self.uav_specifications,
            "scenario_info": self.get_adaptive_info(),
        }

    def create_waypoints(self) -> np.array:
        raise NotImplementedError("MCTS zero planning mission does not implement 'create_waypoints' function!")

    def get_network_filenames(self) -> Tuple[str, Optional[str]]:
        network_filename = (
            f"shared_net.{self.model_deployment_filename}"
            if self.hyper_params["shared_network"]
            else f"policy_net.{self.model_deployment_filename}"
        )
        value_network_filename = (
            None if self.hyper_params["shared_network"] else f"value_net.{self.model_deployment_filename}"
        )
        return network_filename, value_network_filename

    def schedule_exploration_parmeters(self, iteration: int):
        if iteration > 0:
            self.hyper_params["puct_init"] = max(
                self.hyper_params["puct_init_min"],
                self.hyper_params["puct_init"] * self.hyper_params["puct_init_decay"],
            )
            self.hyper_params["dirichlet_alpha"] = max(
                self.hyper_params["dirichlet_alpha_min"],
                self.hyper_params["dirichlet_alpha"] * self.hyper_params["dirichlet_alpha_decay"],
            )

        logger.info(f"SCHEDULED PUCT-CONSTANT: {self.hyper_params['puct_init']}")
        logger.info(f"SCHEDULED DIRICHLET ALPHA: {self.hyper_params['dirichlet_alpha']}")

    def schedule_off_policy_window_size(self, iteration: int) -> int:
        return min(
            int(
                self.hyper_params["start_train_examples_history"]
                + iteration / self.hyper_params["train_examples_history_step"]
            ),
            self.hyper_params["max_train_examples_history"],
        )

    def learn(self):
        shared_net_filepath = os.path.join(CHECKPOINTS_DIR, f"shared_net.{self.model_deployment_filename}")
        policy_net_filepath = os.path.join(CHECKPOINTS_DIR, f"policy_net.{self.model_deployment_filename}")
        value_net_filepath = os.path.join(CHECKPOINTS_DIR, f"value_net.{self.model_deployment_filename}")

        if not os.path.exists(TRAIN_DATA_DIR):
            os.mkdir(TRAIN_DATA_DIR)

        models_exist = os.path.exists(policy_net_filepath) and os.path.exists(value_net_filepath)
        if self.hyper_params["shared_network"]:
            models_exist = os.path.exists(shared_net_filepath)

        if not models_exist:
            if self.hyper_params["shared_network"]:
                self.shared_network.save_checkpoint(filename=f"shared_net.{self.model_deployment_filename}")
            else:
                self.policy_network.save_checkpoint(filename=f"policy_net.{self.model_deployment_filename}")
                self.value_network.save_checkpoint(filename=f"value_net.{self.model_deployment_filename}")

        generated_episodes = 0
        total_train_examples = 0
        net_writer = self.shared_network.writer if self.hyper_params["shared_network"] else self.value_network.writer
        network_filename, value_network_filename = self.get_network_filenames()

        manager = Manager()
        network_queue = manager.Queue()
        request_queue = manager.Queue()
        reply_queues = {}
        for i in range(self.hyper_params["num_workers"]):
            reply_queues[i] = manager.Queue()

        inference_process = torch.multiprocessing.spawn(
            inference_worker,
            args=(
                network_queue,
                request_queue,
                reply_queues,
                self.hyper_params,
                self.meta_data,
                network_filename,
                self.hyper_params["max_inference_batch_size"],
                self.hyper_params["max_waiting_time"],
                value_network_filename,
                self.hyper_params["log_network_parameters"],
            ),
            nprocs=1,
            join=False,
            daemon=True,
        )

        start_iteration = self.train_examples_iter if self.skip_first_self_play else 0
        for iteration in range(start_iteration, self.hyper_params["num_self_play_iterations"]):
            logger.info(f"Self-Play Iteration {iteration}")
            iteration_start_time = time.time()

            train_data_iter_path = os.path.join(TRAIN_DATA_DIR, f"iter_{iteration}")
            off_policy_window_size = self.schedule_off_policy_window_size(iteration)
            logger.info(f"Off-Policy Window Size: {off_policy_window_size}")

            if not self.skip_first_self_play or iteration > 0:
                self.schedule_exploration_parmeters(iteration)

                if os.path.isdir(train_data_iter_path):
                    shutil.rmtree(train_data_iter_path)
                os.mkdir(train_data_iter_path)

                t = tqdm(
                    range(self.hyper_params["num_episodes"] * self.hyper_params["num_workers"]), desc="Execute episodes"
                )
                params = []
                for i in range(self.hyper_params["num_workers"]):
                    param = (
                        self.hyper_params,
                        self.mapping,
                        generated_episodes,
                        self.meta_data,
                        self.hyper_params["num_episodes"],
                        request_queue,
                        reply_queues[i],
                        i,
                        iteration,
                    )
                    params.append(param)
                    generated_episodes += self.hyper_params["num_episodes"]

                with Pool(processes=self.hyper_params["num_workers"]) as pool:
                    results = pool.starmap(run_episode_worker, params)

                pool.close()
                pool.terminate()
                pool.join()

                episode_values = []
                for result in results:
                    episode_values_worker = result
                    episode_values.extend(episode_values_worker)
                    del episode_values_worker

                t.update(self.hyper_params["num_episodes"] * self.hyper_params["num_workers"])
                num_train_examples = len(os.listdir(os.path.join(TRAIN_DATA_DIR, f"iter_{iteration}")))

                net_writer.add_scalar(f"SelfPlay/GeneratedEpisodes", generated_episodes, iteration)
                net_writer.add_scalar(f"SelfPlay/TrainExamples", num_train_examples, generated_episodes)

                for i in range(len(episode_values)):
                    net_writer.add_scalar("SelfPlay/ValueEpisode", episode_values[i], generated_episodes + i)

                total_train_examples += num_train_examples
                net_writer.add_scalar(f"SelfPlay/GeneratedTrainSamples", total_train_examples, iteration)

            if iteration - self.hyper_params["max_train_examples_history"] >= 0:
                outdated_train_data_iter = iteration - self.hyper_params["max_train_examples_history"]
                outdated_train_data_dir = os.path.join(TRAIN_DATA_DIR, f"iter_{outdated_train_data_iter}")
                if os.path.isdir(outdated_train_data_dir):
                    shutil.rmtree(outdated_train_data_dir)

            if self.hyper_params["shared_network"]:
                self.shared_network.save_checkpoint(filename="shared_net.temp.pth.tar")
                self.prev_shared_network.load_checkpoint(filename="shared_net.temp.pth.tar")

                self.shared_network.train(self_play_iteration=iteration, window_size=off_policy_window_size)
                self.shared_network.save_checkpoint(filename=f"shared_net.snapshot_{iteration}.pth.tar")
            else:
                self.policy_network.save_checkpoint(filename="policy_net.temp.pth.tar")
                self.prev_policy_network.load_checkpoint(filename="policy_net.temp.pth.tar")

                self.value_network.save_checkpoint(filename="value_net.temp.pth.tar")
                self.prev_value_network.load_checkpoint(filename="value_net.temp.pth.tar")

                self.policy_network.train(self_play_iteration=iteration, window_size=off_policy_window_size)
                self.value_network.train(self_play_iteration=iteration, window_size=off_policy_window_size)

                self.policy_network.save_checkpoint(filename=f"policy_net.snapshot_{iteration}.pth.tar")
                self.value_network.save_checkpoint(filename=f"value_net.snapshot_{iteration}.pth.tar")

            if self.hyper_params["continuous_network_update"]:
                if self.hyper_params["shared_network"]:
                    self.shared_network.save_checkpoint(filename=f"shared_net.{self.model_deployment_filename}")
                else:
                    self.policy_network.save_checkpoint(filename=f"policy_net.{self.model_deployment_filename}")
                    self.value_network.save_checkpoint(filename=f"value_net.{self.model_deployment_filename}")

                network_queue.put("LOAD")
            else:
                self.evaluate_trained_agent(iteration)

            iteration_finish_time = time.time()
            if self.telegram_notifications:
                try:
                    iteration_id = f"Self-Play Iteration {iteration}"
                    iteration_info = {
                        "collected_new_episodes": not self.skip_first_self_play or iteration > 0,
                        "generated_episodes": generated_episodes,
                        "total_train_examples": total_train_examples,
                        "iteration_run_time": f"{round((iteration_finish_time - iteration_start_time) / 3600, 2)}h",
                    }
                    self.telegram_notifier.finished_iteration(iteration_id, additional_info=iteration_info)
                except Exception as e:
                    logger.error(f"Failed sending telegram iteration finished message.\n{e}")

        network_queue.put("END")
        inference_process.join()

    def evaluate_trained_agent(self, iteration: int):
        prev_mcts = self.mcts
        mcts = self.mcts

        arena = Arena(
            lambda node, depth, previous_action, budget: np.argmax(
                prev_mcts.get_policy(node, 0, previous_action, budget, temperature=0)
            ),
            lambda node, depth, previous_action, budget: np.argmax(
                mcts.get_policy(node, 0, previous_action, budget, temperature=0)
            ),
            self,
        )
        total_rewards_prev, total_rewards_curr = arena.play_games(self.hyper_params["num_arena_games"])
        logger.info(
            f"PREV TOTAL ARENA REWARDS: {total_rewards_prev}, CURRENT TOTAL ARENA REWARDS: {total_rewards_curr}"
        )

        relative_total_rewards_curr = total_rewards_curr / (total_rewards_prev + total_rewards_curr)
        if relative_total_rewards_curr < self.hyper_params["network_update_threshold"]:
            logger.info(f"REJECTED NEW NETWORK")
            self.prev_network_wins += 1
            if self.hyper_params["shared_network"]:
                self.shared_network.load_checkpoint(filename="shared_net.temp.pth.tar")
            else:
                self.policy_network.load_checkpoint(filename="policy_net.temp.pth.tar")
                self.value_network.load_checkpoint(filename="value_net.temp.pth.tar")
        else:
            logger.info(f"ACCEPTED NEW MODEL")
            if self.hyper_params["shared_network"]:
                self.shared_network.save_checkpoint(filename=f"shared_net.{self.model_deployment_filename}")
            else:
                self.policy_network.save_checkpoint(filename=f"policy_net.{self.model_deployment_filename}")
                self.value_network.save_checkpoint(filename=f"value_net.{self.model_deployment_filename}")

        net_writer = self.shared_network.writer if self.hyper_params["shared_network"] else self.policy_network.writer
        avg_cumulated_reward = total_rewards_curr / self.hyper_params["num_arena_games"]
        net_writer.writer.add_scalar("Performance/Reward", avg_cumulated_reward, iteration)
        net_writer.writer.add_scalar("Performance/PrevNetWins", self.prev_network_wins, iteration)

    def get_next_actions_mask(self, position: np.array, budget: float) -> np.array:
        distances = np.linalg.norm(self.actions_np - position, ord=2, axis=1)
        if not self.adaptive:
            return (
                (distances > 0) & (distances <= budget) & (distances < self.hyper_params["max_valid_action_distance"])
            )

        flight_times = compute_flight_times(self.actions_np, position, self.uav_specifications)
        return (
            (flight_times > 0) & (flight_times <= budget) & (distances < self.hyper_params["max_valid_action_distance"])
        )

    def replan(
        self,
        root: Node,
        budget: float,
        previous_action: np.array,
        episode_history: EpisodeHistory,
        mcts_instances: Dict,
        planning_step: int,
    ) -> np.array:
        if self.hyper_params["num_mcts_simulations"] <= 0:
            input_feature_planes = generate_input_feature_planes(
                self.mapping,
                episode_history,
                self.min_altitude,
                self.max_altitude,
                adaptive_info=self.get_adaptive_info(),
                uav_specifications=self.uav_specifications,
                use_action_costs_input=self.hyper_params["use_action_costs_input"],
            )
            next_actions_msk = self.get_next_actions_mask(previous_action, budget)
            input_feature_planes = input_feature_planes[np.newaxis, :]
            next_actions_msk = next_actions_msk[np.newaxis, :]

            if self.hyper_params["shared_network"]:
                policy, _ = self.shared_network.predict(input_feature_planes, next_actions_msk)
            else:
                policy = self.policy_network.predict(input_feature_planes, next_actions_msk)

            policy = policy[0]
            next_actions_msk = next_actions_msk[0]
            policy = policy * next_actions_msk
            policy /= np.sum(policy)
            max_action_idx = np.argmax(policy)
            return self.actions_np[max_action_idx, :]

        params = []
        for i, mcts in mcts_instances.items():

            params.append((mcts, root, previous_action, budget, episode_history, planning_step))

        with Pool(processes=self.hyper_params["num_workers"]) as pool:
            results = pool.starmap(run_deploy_time_mcts_worker, params)

        pool.close()
        pool.terminate()
        pool.join()

        policy_total = np.zeros(len(self.actions.keys()))
        for i, policy in enumerate(results):
            policy = np.array(policy)
            policy_total += policy

        policy_total /= np.sum(policy_total)
        max_action_idx = np.argmax(policy_total)
        return self.actions_np[max_action_idx, :]

    def check_for_train_examples(self):
        if not os.path.isdir(self.train_examples_filename):
            logger.error(f"Directory '{self.train_examples_filename}' with train examples not found!")
            self.skip_first_self_play = False
        else:
            logger.error(f"Found directory '{self.train_examples_filename}' with train examples!")
            self.skip_first_self_play = True

    def execute(self):
        try:
            if self.telegram_notifications:
                try:
                    self.telegram_notifier.start_experiment()
                except Exception as e:
                    logger.error(f"Failed sending telegram experiment started message.\n{e}")

            shared_net_filepath = os.path.join(CHECKPOINTS_DIR, f"shared_net.{self.model_deployment_filename}")
            policy_net_filepath = os.path.join(CHECKPOINTS_DIR, f"policy_net.{self.model_deployment_filename}")
            value_net_filepath = os.path.join(CHECKPOINTS_DIR, f"value_net.{self.model_deployment_filename}")

            models_exist = os.path.exists(policy_net_filepath) and os.path.exists(value_net_filepath)
            if self.hyper_params["shared_network"]:
                models_exist = os.path.exists(shared_net_filepath)

            if not models_exist or self.restart_training:
                if models_exist:
                    if self.hyper_params["shared_network"]:
                        logger.info(f"Restart training of model {shared_net_filepath}")
                        self.shared_network.load_checkpoint(filename=shared_net_filepath)
                    else:
                        logger.info(f"Restart training of models {policy_net_filepath} and {value_net_filepath}")
                        self.policy_network.load_checkpoint(filename=policy_net_filepath)
                        self.value_network.load_checkpoint(filename=value_net_filepath)
                else:
                    logger.info(f"No trained model exists. Start training of network(s) via MCTS self-play.")

                self.check_for_train_examples()
                self.learn()

            if self.hyper_params["shared_network"]:
                self.shared_network.load_checkpoint(filename=shared_net_filepath)
            else:
                self.policy_network.load_checkpoint(filename=policy_net_filepath)
                self.value_network.load_checkpoint(filename=value_net_filepath)

            network_filename, value_network_filename = self.get_network_filenames()

            manager = Manager()
            network_queue = manager.Queue()
            request_queue = manager.Queue()
            reply_queues = {}
            mcts_instances = {}
            for i in range(self.hyper_params["num_workers"]):
                reply_queues[i] = manager.Queue()
                mcts_instances[i] = MCTS(
                    self.mapping,
                    self.hyper_params,
                    self.meta_data,
                    request_queue,
                    reply_queues[i],
                    worker_id=i,
                )

            inference_process = torch.multiprocessing.spawn(
                inference_worker,
                args=(
                    network_queue,
                    request_queue,
                    reply_queues,
                    self.hyper_params,
                    self.meta_data,
                    network_filename,
                    self.hyper_params["max_inference_batch_size"],
                    self.hyper_params["max_waiting_time"],
                    value_network_filename,
                    self.hyper_params["log_network_parameters"],
                ),
                nprocs=1,
                join=False,
                daemon=True,
            )

            remaining_budget = self.budget
            previous_action = self.init_action
            root = Node(self.mapping.grid_map.cov_matrix)
            episode_history = EpisodeHistory(self.hyper_params["input_history_length"])
            self.eval(run_time=0, flight_time=0)

            while remaining_budget >= self.mapping.grid_map.resolution:
                logger.info(f"\nREMAINING BUDGET: {remaining_budget}")

                if self.hyper_params["reset_mcts_each_step"]:
                    for i in range(self.hyper_params["num_workers"]):
                        mcts_instances[i] = MCTS(
                            self.mapping,
                            self.hyper_params,
                            self.meta_data,
                            request_queue,
                            reply_queues[i],
                            worker_id=i,
                        )

                episode_history.push(root.state, previous_action, remaining_budget / self.meta_data["initial_budget"])
                start_time = time.time()
                action = self.replan(
                    root,
                    remaining_budget,
                    previous_action,
                    episode_history,
                    mcts_instances,
                    planning_step=len(self.waypoints),
                )
                finish_time = time.time()

                simulated_raw_measurement = self.mapping.sensor.take_measurement(action)
                self.mapping.update_grid_map(action, simulated_raw_measurement)
                self.waypoints = np.vstack((self.waypoints, action))

                flight_time = compute_flight_time(action, previous_action, self.uav_specifications)
                run_time = finish_time - start_time
                remaining_budget -= action_costs(action, previous_action, self.uav_specifications)
                if self.use_effective_mission_time:
                    remaining_budget -= run_time

                root = Node(self.mapping.grid_map.cov_matrix)
                previous_action = action

                self.eval(run_time=run_time, flight_time=flight_time)

            network_queue.put("END")
            inference_process.join()

            if self.telegram_notifications:
                try:
                    self.telegram_notifier.finish_experiment()
                except Exception as e:
                    logger.error(f"Failed sending telegram experiment finished message.\n{e}")
        except Exception as e:
            logger.exception(f"MCTSZero mission failed with error: {e}")
            if self.telegram_notifications:
                self.telegram_notifier.failed_experiment(e)
            raise Exception(e)
