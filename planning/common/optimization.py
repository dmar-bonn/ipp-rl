import os
import time
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np

from mapping.mappings import Mapping
from planning.common.actions import action_costs
from planning.common.actions import get_actions
from planning.common.rewards import compute_reward, compute_adaptive_msk


def simulate_prediction_step(
    current_state: np.array,
    previous_action: np.array,
    action: np.array,
    mapping: Mapping,
    uav_specifications: Dict = None,
    adaptive_info: Dict = None,
) -> Tuple[float, np.array, np.array]:
    adaptive_msk = None
    if adaptive_info is not None:
        adaptive_msk = compute_adaptive_msk(
            adaptive_info["mean"], current_state, adaptive_info["value_threshold"], adaptive_info["interval_factor"]
        )

    _, next_state = mapping.update_grid_map(action, cov_only=True, predict_only=True, current_cov_matrix=current_state)
    reward = compute_reward(current_state, next_state, previous_action, action, uav_specifications, adaptive_msk)
    return reward, action, next_state


def greedy_search(
    previous_action: np.array,
    remaining_budget: float,
    current_state: np.array,
    episode_horizon: int,
    mapping: Mapping,
    min_altitude: float,
    max_altitude: float,
    altitude_spacing: float,
    uav_specifications: Dict = None,
    adaptive_info: Dict = None,
) -> List:
    """
    Generates episode_horizon waypoints greedily chosen to maximize reward.

    Args:
        previous_action (np.array): last visited waypoint
        remaining_budget (float): currently remaining travel budget
        current_state (float): current Kalman filter covariance matrix
        episode_horizon (int): total number of greedily sampled waypoints
        altitude_spacing (float): spacing [m] between consecutive altitude levels
        min_altitude (float): lowest altitude level [m] of waypoints above ground
        max_altitude (float): highest altitude level [m] of waypoints above ground
        mapping (Mapping): performs map prediction step and stores grid map to optimize over
        uav_specifications (Dict): optional UAV max acceleration and speed for flight time calculation
        adaptive_info (Dict): current grid map mean belief, interesting value threshold and CI width factor

    Returns:
        (List): waypoints chosen to greedily optimize the reward
    """
    waypoints = []

    for i in range(episode_horizon):
        action_set = get_actions(
            previous_action,
            remaining_budget,
            mapping.grid_map,
            min_altitude,
            max_altitude,
            altitude_spacing,
            uav_specifications,
        )
        if len(action_set) == 0:
            break

        argmax_action = None
        argmax_state = None
        max_reward = -np.inf

        simulation_args = [
            (current_state, previous_action, action, mapping, uav_specifications, adaptive_info)
            for action in action_set
        ]
        num_processes = 4
        approx_chunksize = max(1, int(len(action_set) // num_processes))

        with Pool(processes=num_processes) as pool:
            simulation_values = pool.starmap(simulate_prediction_step, simulation_args, chunksize=approx_chunksize)

        for simulation_value in simulation_values:
            reward, action, next_state = simulation_value
            if reward > max_reward:
                max_reward = reward
                argmax_action = action
                argmax_state = next_state

        remaining_budget -= action_costs(argmax_action, previous_action, uav_specifications)
        previous_action = argmax_action
        current_state = argmax_state
        waypoints.append(argmax_action)

    return waypoints
