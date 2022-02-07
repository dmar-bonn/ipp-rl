import math
from typing import Dict, List, Tuple

import numpy as np

from mapping.mappings import Mapping
from planning.common.actions import action_costs, enumerate_actions, flatten_grid_index
from planning.common.optimization import compute_adaptive_msk


class EpisodeHistory:
    def __init__(self, max_history_length: int):
        self.max_history_length = max_history_length
        self.states = []
        self.positions = []
        self.budgets = []

    def push(self, state: np.array, action: np.array, budget: float):
        self.states.insert(0, state)
        self.positions.insert(0, action)
        self.budgets.insert(0, budget)

        if len(self.states) > self.max_history_length:
            self.states.pop()
            self.positions.pop()
            self.budgets.pop()

    def pop(self) -> Tuple[np.array, np.array, float]:
        return self.states.pop(), self.positions.pop(), self.budgets.pop()

    def __len__(self):
        return len(self.states)


def generate_fov_feature_plane(mapping: Mapping, position: np.array) -> np.array:
    feature_plane = np.zeros((mapping.grid_map.num_grid_cells, mapping.grid_map.num_grid_cells))
    flattened_fov_indices = get_field_of_view_indices(mapping, position)

    for i in range(mapping.grid_map.num_grid_cells):
        for j in range(mapping.grid_map.num_grid_cells):
            if i in flattened_fov_indices and j in flattened_fov_indices:
                feature_plane[i, j] = 1

    return feature_plane


def generate_position_feature_planes(
    mapping: Mapping, position: np.array, min_altitude: float, max_altitude: float
) -> Tuple[np.array, np.array, np.array]:
    x_pos_relative = position[0] / (mapping.grid_map.x_dim * mapping.grid_map.resolution)
    y_pos_relative = position[1] / (mapping.grid_map.x_dim * mapping.grid_map.resolution)
    z_pos_relative = (position[2] - min_altitude) / (max_altitude - min_altitude)

    x_feature_plane = x_pos_relative * np.ones((mapping.grid_map.num_grid_cells, mapping.grid_map.num_grid_cells))
    y_feature_plane = y_pos_relative * np.ones((mapping.grid_map.num_grid_cells, mapping.grid_map.num_grid_cells))
    z_feature_plane = z_pos_relative * np.ones((mapping.grid_map.num_grid_cells, mapping.grid_map.num_grid_cells))

    return x_feature_plane, y_feature_plane, z_feature_plane


def generate_costs_feature_plane(
    mapping: Mapping, current_action: np.array, min_altitude: float, uav_specifications: Dict = None
) -> np.array:
    current_action[-1] = min_altitude
    costs_feature_plane = np.zeros((mapping.grid_map.num_grid_cells, mapping.grid_map.num_grid_cells))
    actions = enumerate_actions(mapping.grid_map, min_altitude, min_altitude, 1)
    for i, action in actions.items():
        costs_feature_plane[i, :] = action_costs(current_action, action, uav_specifications=uav_specifications)

    return min_max_normalize(costs_feature_plane)


def min_max_normalize(x: np.array) -> np.array:
    min_value = np.min(x)
    max_value = np.max(x)

    if min_value == max_value:
        return x / max_value

    return (x - min_value) / (max_value - min_value)


def generate_input_feature_planes(
    mapping: Mapping,
    episode_history: EpisodeHistory,
    min_altitude: float = None,
    max_altitude: float = None,
    adaptive_info: Dict = None,
    uav_specifications: Dict = None,
    use_action_costs_input: bool = False,
) -> np.array:
    state_planes = []
    for state in episode_history.states:
        if adaptive_info is not None:
            adaptive_msk = compute_adaptive_msk(
                adaptive_info["mean"], state, adaptive_info["value_threshold"], adaptive_info["interval_factor"]
            )
            state[~adaptive_msk, :] = 0
            state[:, ~adaptive_msk] = 0

        state_planes.append(min_max_normalize(state))

    budget_planes = []
    for budget in episode_history.budgets:
        budget_planes.append(budget * np.ones_like(episode_history.states[0]))

    total_feature_planes = []

    if min_altitude is None or max_altitude is None:
        fov_planes = []
        for position in episode_history.positions:
            fov_planes.append(generate_fov_feature_plane(mapping, position))

        for i in range(len(episode_history)):
            state, fov_plane, budget_plane = state_planes[i], fov_planes[i], budget_planes[i]
            total_feature_planes.extend([state, fov_plane, budget_plane])
        for _ in range(episode_history.max_history_length - len(episode_history)):
            pad_feature_plane = np.zeros_like(episode_history.states[0])
            total_feature_planes.extend([pad_feature_plane] * 3)

        return np.array(total_feature_planes)

    x_feature_planes, y_feature_planes, z_feature_planes = [], [], []
    for position in episode_history.positions:
        x_feature_plane, y_feature_plane, z_feature_plane = generate_position_feature_planes(
            mapping, position, min_altitude, max_altitude
        )
        x_feature_planes.append(x_feature_plane)
        y_feature_planes.append(y_feature_plane)
        z_feature_planes.append(z_feature_plane)

    for i in range(len(episode_history)):
        state, x_feature_plane, y_feature_plane, z_feature_plane, budget_plane = (
            state_planes[i],
            x_feature_planes[i],
            y_feature_planes[i],
            z_feature_planes[i],
            budget_planes[i],
        )
        total_feature_planes.extend([state, x_feature_plane, y_feature_plane, z_feature_plane, budget_plane])

    for _ in range(episode_history.max_history_length - len(episode_history)):
        pad_feature_plane = np.zeros_like(episode_history.states[0])
        total_feature_planes.extend([pad_feature_plane] * 5)

    if use_action_costs_input:
        total_feature_planes.append(
            generate_costs_feature_plane(mapping, episode_history.positions[0], min_altitude, uav_specifications)
        )

    return np.array(total_feature_planes)


def get_field_of_view_indices(mapping: Mapping, position: np.array) -> List:
    xl, xr, yu, yd = mapping.sensor.project_field_of_view(position)

    measurement_indices_x = np.linspace(xl, xr, math.ceil(xr - xl + 1))[:-1]
    measurement_indices_y = np.linspace(yu, yd, math.ceil(yd - yu + 1))[:-1]
    indices_x_meshed, indices_y_meshed = np.meshgrid(measurement_indices_x, measurement_indices_y)
    measurement_indices = np.array([indices_x_meshed.ravel(), indices_y_meshed.ravel()]).T

    flattened_fov_indices = []
    for i in range(measurement_indices.shape[0]):
        measurement_idx = measurement_indices[i, :]
        flattened_fov_indices.append(flatten_grid_index(mapping.grid_map, measurement_idx))

    return flattened_fov_indices
