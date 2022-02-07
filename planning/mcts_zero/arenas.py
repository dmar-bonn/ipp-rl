import logging
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

from planning.common.actions import action_costs, enumerate_actions
from planning.common.optimization import simulate_prediction_step
from planning.mcts_zero.mcts import Node

logger = logging.getLogger(__name__)


class Arena:
    def __init__(self, exec_mcts_prev: callable, exec_mcts_curr: callable, mission):
        self.exec_mcts_prev = exec_mcts_prev
        self.exec_mcts_curr = exec_mcts_curr
        self.mission = mission
        self.episode_horizon = mission.episode_horizon
        self.gamma = mission.hyper_params["gamma"]
        self.actions = enumerate_actions(
            mission.mapping.grid_map, mission.min_altitude, mission.max_altitude, mission.altitude_spacing
        )

    def play_game(self, exec_mcts: callable) -> float:
        node = Node(self.mission.mapping.grid_map.cov_matrix)
        depth = 0
        remaining_budget = self.mission.budget
        previous_action = np.array([0, 0, 10])
        total_reward = 0

        while depth < self.episode_horizon or remaining_budget > 0:
            action_idx = exec_mcts(node, depth, previous_action, remaining_budget)
            action = self.actions[action_idx]
            reward, _, next_state = simulate_prediction_step(
                node.state, previous_action, action, self.mission.mapping, self.mission.uav_specifications
            )
            total_reward += self.gamma ** depth * reward
            remaining_budget -= action_costs(action, previous_action, self.mission.uav_specifications)
            previous_action = action
            depth += 1
            next_node = Node(next_state)
            node = next_node

        return total_reward

    def play_games(self, num_games: int) -> Tuple[float, float]:
        logger.info(f"Play {num_games} arena games with previous and current network respectively")
        total_reward_prev = 0
        total_reward_curr = 0

        for _ in tqdm(range(num_games), desc="Play games"):
            total_reward_prev += self.play_game(self.exec_mcts_prev)
            total_reward_curr += self.play_game(self.exec_mcts_curr)

        return total_reward_prev, total_reward_curr
