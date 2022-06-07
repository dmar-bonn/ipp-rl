import concurrent.futures
import logging
import random
import time
from typing import Dict

import numpy as np

from constants import MissionType
from mapping.mappings import Mapping
from planning.common.actions import (
    action_costs,
    action_dict_to_np_array,
    compute_flight_time,
    compute_flight_times,
    enumerate_actions,
)
from planning.common.rewards import compute_adaptive_msk, compute_reward
from planning.missions import Mission

logger = logging.getLogger(__name__)


class Node:
    def __init__(self, state: np.array, parent=None, action=None):
        self.parent = parent
        self.state = state
        self.action = action
        self.value_sum = 0
        self.visits = 0
        self.children = []

    @property
    def value(self):
        return self.value_sum / self.visits

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @staticmethod
    def uct(node, min_val: float, max_val: float, c: float = 2.0):
        if node.visits == 0:
            return np.inf

        exploration = c * np.sqrt(np.log(node.parent.visits) / node.visits)
        if max_val == 0:
            return node.value + exploration

        if max_val == min_val:
            normalized_value = node.value / max_val
        else:
            normalized_value = node.value - min_val / (max_val - min_val)

        return normalized_value + exploration

    def select_child(self, budget: float, c: float = 2.0, uav_specifications: Dict = None):
        max_children = []
        max_uct = -np.inf

        children_vals = [child.value for child in self.children]
        min_child_val, max_child_val = min(children_vals), max(children_vals)
        for i in range(len(self.children)):
            uct = self.uct(self.children[i], min_child_val, max_child_val, c=c)
            next_action_costs = action_costs(self.children[i].action, self.action, uav_specifications)
            if next_action_costs == 0 or next_action_costs >= budget:
                uct = -np.inf

            if max_uct == uct:
                max_children.append(self.children[i])
            elif uct > max_uct:
                max_uct = uct
                max_children = [self.children[i]]

        return random.choice(max_children)


class MCTSMission(Mission):
    def __init__(
        self,
        mapping: Mapping,
        uav_specifications: Dict,
        dist_to_boundaries: float = 10,
        min_altitude: float = 5,
        max_altitude: float = 30,
        budget: float = 200.0,
        altitude_spacing: float = 5,
        num_simulations: int = 100,
        gamma: float = 0.95,
        c: float = 2.0,
        episode_horizon: int = 5,
        k: float = 4.0,
        alpha: float = 0.75,
        epsilon_expand: float = 0.2,
        epsilon_rollout: float = 0.5,
        max_greedy_radius: int = 3,
        use_gcb_rollout: bool = False,
        adaptive: bool = False,
        value_threshold: float = 0.5,
        interval_factor: float = 2,
        config_name: str = None,
        use_effective_mission_time: bool = False,
    ):
        """
        Defines a mission optimizing measurement positions offline by Monte Carlo tree search with random rollout
        and expansion policy.

        Args:
            mapping (Mapping): mapping algorithm with related sensor and grid map specification
            uav_specifications (dict): uav parameters defining max_v, max_a, sampling_time
            dist_to_boundaries (float): minimal distance [m] of a waypoint to map boundaries
            min_altitude (float): lowest altitude level [m] of waypoints above ground
            max_altitude (float): highest altitude level [m] of waypoints above ground
            altitude_spacing (float): spacing [m] between consecutive altitude levels
            budget (float): total distance budget for mission
            num_simulations (int): number of performed MCTS simulations
            gamma (float): future reward discount factor
            episode_horizon (int): max planning horizon, i.e. max tree search depth
            k (float): multiplicative factor of progressive widening applied to action space
            alpha (float): exponent of progressive widening applied to action space
            epsilon_expand (float): probability of choosing a random expansion action instead of the greedy one
            epsilon_rollout (float): probability of choosing a random rollout action instead of the greedy one
            max_greedy_radius (int): max. Manhattan distance of next greedily expanded cell from current cell
            use_gcb_rollout (bool): if true, perform generalized cost-benefit rollout policy
            adaptive (bool): indicates if mission should be executed as adaptive or non-adaptive IPP
            value_threshold (float): grid cells with upper CI bounds above this threshold are of interest
            interval_factor (float): defines the width of the CI used to decide if a grid cell is of interest
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

        self.altitude_spacing = altitude_spacing
        self.num_simulations = num_simulations
        self.mission_name = "MCTS"
        self.mission_type = MissionType.MCTS
        self.gamma = gamma
        self.c = c
        self.episode_horizon = episode_horizon
        self.k = k
        self.alpha = alpha
        self.epsilon_expand = epsilon_expand
        self.epsilon_rollout = epsilon_rollout
        self.max_greedy_radius = max_greedy_radius
        self.use_gcb_rollout = use_gcb_rollout

        self.actions = enumerate_actions(
            self.mapping.grid_map, self.min_altitude, self.max_altitude, self.altitude_spacing
        )
        self.actions_np = action_dict_to_np_array(self.actions)

    def create_waypoints(self) -> np.array:
        raise NotImplementedError("MCTS planning mission does not implement 'create_waypoints' function!")

    def get_next_actions_mask(self, position: np.array, budget: float, uav_specificaion: Dict = None) -> np.array:
        distances = np.linalg.norm(self.actions_np - position, ord=2, axis=1)
        if uav_specificaion is None:
            return (distances > 0) & (distances <= budget) & (distances < self.max_greedy_radius)

        flight_times = compute_flight_times(self.actions_np, position, uav_specificaion)
        return (flight_times > 0) & (flight_times <= budget) & (distances < self.max_greedy_radius)

    def gcb_rollout(self, node: Node, remaining_budget: float, previous_action: np.array, depth: int) -> float:
        if depth == 0 or remaining_budget < self.mapping.grid_map.resolution:
            return 0

        sampled_action = self.gcb_policy(node, remaining_budget)
        next_state = self.prediction_step(node.state, sampled_action)
        remaining_budget -= action_costs(sampled_action, previous_action, self.uav_specifications)
        next_node = Node(state=next_state, parent=node, action=sampled_action)
        adaptive_msk = compute_adaptive_msk(
            self.mapping.grid_map.mean, node.state, self.value_threshold, self.interval_factor
        )
        reward = compute_reward(
            node.state, next_node.state, node.action, next_node.action, self.uav_specifications, adaptive_msk
        )

        return reward + self.gamma * self.gcb_rollout(next_node, remaining_budget, node.action, depth - 1)

    def gcb_policy(self, node, remaining_budget):
        available_actions_msk = self.get_next_actions_mask(node.action, remaining_budget, self.uav_specifications)
        available_actions = self.actions_np[available_actions_msk]
        adaptive_msk = compute_adaptive_msk(
            self.mapping.grid_map.mean, node.state, self.value_threshold, self.interval_factor
        )

        benefit_to_cost_values = []
        for action in available_actions:
            next_state = self.prediction_step(node.state, action)
            reward = compute_reward(node.state, next_state, node.action, action, self.uav_specifications, adaptive_msk)
            benefit_to_cost_values.append(reward)

        benefit_to_cost_values = np.array(benefit_to_cost_values)
        softmax_benefit_to_cost_values = np.exp(benefit_to_cost_values) / np.sum(np.exp(benefit_to_cost_values))
        action_idx = np.random.choice(len(available_actions), p=softmax_benefit_to_cost_values)

        return available_actions[action_idx]

    def rollout(self, node: Node, remaining_budget: float, previous_action: np.array, depth: int) -> float:
        if depth == 0 or remaining_budget < self.mapping.grid_map.resolution:
            return 0

        sampled_action = self.eps_greedy_policy(node, remaining_budget, self.epsilon_rollout)
        next_state = self.prediction_step(node.state, sampled_action)
        remaining_budget -= action_costs(sampled_action, previous_action, self.uav_specifications)
        next_node = Node(state=next_state, parent=node, action=sampled_action)
        adaptive_msk = compute_adaptive_msk(
            self.mapping.grid_map.mean, node.state, self.value_threshold, self.interval_factor
        )
        reward = compute_reward(
            node.state, next_node.state, node.action, next_node.action, self.uav_specifications, adaptive_msk
        )

        return reward + self.gamma * self.rollout(next_node, remaining_budget, node.action, depth - 1)

    def prediction_step(self, state: np.array, action: np.array) -> np.array:
        _, next_state = self.mapping.update_grid_map(action, cov_only=True, predict_only=True, current_cov_matrix=state)
        return next_state

    def greedy_action(self, node: Node, actions: np.array) -> np.array:
        greedy_action = None
        max_reward = -np.inf
        adaptive_msk = compute_adaptive_msk(
            self.mapping.grid_map.mean, node.state, self.value_threshold, self.interval_factor
        )

        for action in actions:
            next_state = self.prediction_step(node.state, action)
            reward = compute_reward(node.state, next_state, node.action, action, self.uav_specifications, adaptive_msk)
            if reward > max_reward:
                greedy_action = action
                max_reward = reward

        return greedy_action

    def eps_greedy_policy(self, node: Node, remaining_budget: float, epsilon: float) -> np.array:
        available_actions_msk = self.get_next_actions_mask(node.action, remaining_budget, self.uav_specifications)
        available_actions = self.actions_np[available_actions_msk]

        if np.random.uniform(0, 1) > epsilon and available_actions_msk.sum() > 0:
            return self.greedy_action(node, available_actions)
        else:
            action_idx = np.random.choice(len(available_actions))
            return available_actions[action_idx]

    def expand(self, node: Node, remaining_budget):
        sampled_action = self.eps_greedy_policy(node, remaining_budget, self.epsilon_expand)
        next_state = self.prediction_step(node.state, sampled_action)
        return Node(state=next_state, parent=node, action=sampled_action)

    def progressive_widening(self, node: Node, remaining_budget: float):
        available_actions_msk = self.get_next_actions_mask(node.action, remaining_budget, self.uav_specifications)

        if len(node.children) == 0 or (
            len(node.children) <= self.k * node.visits ** self.alpha
            and len(node.children) < available_actions_msk.sum()
        ):
            return self.expand(node, remaining_budget), True

        return node.select_child(remaining_budget, c=self.c, uav_specifications=self.uav_specifications), False

    def simulate(self, node: Node, remaining_budget: float, depth: int) -> float:
        if depth == 0 or remaining_budget < self.mapping.grid_map.resolution:
            return 0

        if node.visits == 0:
            if self.use_gcb_rollout:
                value = self.gcb_rollout(node, remaining_budget, node.action, depth)
            else:
                value = self.rollout(node, remaining_budget, node.action, depth)
            node.visits += 1
            node.value_sum += value
            return value

        next_node, expanded = self.progressive_widening(node, remaining_budget)
        if expanded:
            node.children.append(next_node)

        remaining_budget -= action_costs(next_node.action, node.action, self.uav_specifications)
        adaptive_msk = compute_adaptive_msk(
            self.mapping.grid_map.mean, node.state, self.value_threshold, self.interval_factor
        )
        reward = compute_reward(
            node.state, next_node.state, node.action, next_node.action, self.uav_specifications, adaptive_msk
        )
        value = reward + self.simulate(next_node, remaining_budget, depth - 1)

        node.visits += 1
        next_node.visits += 1
        node.value_sum += value

        return value

    def get_depth(self, node: Node) -> float:
        if len(node.children) == 0:
            return 0

        return max([self.get_depth(child) for child in node.children]) + 1

    def run_simulations_proxy(
        self, root: Node, budget: float, depth: int, num_simulations: int, worker_id: int
    ) -> Node:
        np.random.seed(worker_id * 42 + 1)
        for i in range(num_simulations):
            self.simulate(node=root, remaining_budget=budget, depth=depth)
        return root

    @staticmethod
    def merge_roots(root_a: Node, root_b: Node) -> Node:
        child_actions_a = {str(list(child.action)): child for child in root_a.children}
        child_actions_b = {str(list(child.action)): child for child in root_b.children}

        for child_action_b in child_actions_b.keys():
            child_b = child_actions_b[child_action_b]

            if child_action_b in child_actions_a.keys():
                child_a = child_actions_a[child_action_b]
                child_a.visits += child_b.visits
                child_a.value_sum += child_b.value_sum
            else:
                root_a.children.append(child_b)

            root_a.visits += child_b.visits
            root_a.value_sum += child_b.value_sum

        return root_a

    @staticmethod
    def select_best_child(root: Node) -> Node:
        best_child = None
        for child in root.children:
            if best_child is None:
                best_child = child
                continue
            if child.value > best_child.value:
                best_child = child

        return best_child

    def replan(self, root: Node, remaining_budget: float) -> np.array:
        """
        Generates waypoints by root parallelized Monte Carlo tree search with eps-greedy expansion and rollout policy
        while performing progressive widening in action space.

        Returns:
            (np.array): next best waypoint chosen to optimize future cumulated rewards
        """
        num_workers = 1
        num_simulations_per_process = int(self.num_simulations / num_workers)
        futures = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i in range(num_workers):
                futures.append(
                    executor.submit(
                        self.run_simulations_proxy,
                        root=root,
                        budget=remaining_budget,
                        depth=self.episode_horizon,
                        num_simulations=num_simulations_per_process,
                        worker_id=i,
                    )
                )

        roots = []
        for future in concurrent.futures.as_completed(futures):
            root = future.result()
            roots.append(root)

        merged_root = roots[0]
        for i in range(1, len(roots)):
            merged_root = self.merge_roots(merged_root, roots[i])

        best_child = self.select_best_child(merged_root)
        next_waypoint = best_child.action

        return next_waypoint

    def execute(self):
        self.eval(run_time=0, flight_time=0)
        remaining_budget = self.budget
        previous_waypoint = self.init_action
        root = Node(state=self.mapping.grid_map.cov_matrix, parent=None, action=previous_waypoint)
        while remaining_budget >= self.mapping.grid_map.resolution:
            logger.info(f"\nREMAINING BUDGET: {remaining_budget}")
            start_run_time = time.time()
            waypoint = self.replan(root, remaining_budget)
            finish_run_time = time.time()

            run_time = finish_run_time - start_run_time
            remaining_budget -= action_costs(waypoint, previous_waypoint, self.uav_specifications)
            if self.use_effective_mission_time:
                remaining_budget -= run_time

            flight_time = compute_flight_time(waypoint, previous_waypoint, self.uav_specifications)
            self.waypoints = np.vstack((self.waypoints, waypoint))

            simulated_raw_measurement = self.mapping.sensor.take_measurement(waypoint)
            self.mapping.update_grid_map(waypoint, simulated_raw_measurement)
            previous_waypoint = waypoint
            root = Node(state=self.mapping.grid_map.cov_matrix, parent=None, action=waypoint)
            self.eval(run_time=run_time, flight_time=flight_time)
