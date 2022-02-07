from typing import Dict, Union

import numpy as np

from planning.common.actions import action_costs


def compute_adaptive_msk(
    grid_mean: np.array, grid_covariance: np.array, value_threshold: float, interval_factor: float
):
    msk = grid_mean.flatten(order="C") + interval_factor * np.diag(grid_covariance) >= value_threshold
    return msk


def compute_reward(
    current_state: np.array,
    next_state: np.array,
    previous_action: np.array,
    action: np.array,
    uav_specifications: Dict = None,
    adaptive_msk: np.array = None,
) -> float:
    P_curr_diag = np.diag(current_state)
    P_next_diag = np.diag(next_state)

    if adaptive_msk is not None:
        P_curr_diag = P_curr_diag[adaptive_msk]
        P_next_diag = P_next_diag[adaptive_msk]

    action_utility = np.sum(P_curr_diag) - np.sum(P_next_diag)
    return action_utility / (action_costs(action, previous_action, uav_specifications) + 1)


def scale_value_target(value: float) -> float:
    return np.sqrt(value + 1) - 1


def invert_scaled_value_target(value: Union[float, np.array]) -> Union[float, np.array]:
    return np.square(value) + 2 * value
