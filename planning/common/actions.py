from typing import Dict, List

import numpy as np

from mapping.grid_maps import GridMap


def action_costs(action: np.array, previous_action: np.array, uav_specifications: Dict = None) -> float:
    if uav_specifications is None:
        return compute_distance(action, previous_action)

    return compute_flight_time(action, previous_action, uav_specifications)


def compute_distance(action: np.array, previous_action: np.array) -> float:
    return np.linalg.norm(action - previous_action, ord=2)


def compute_flight_times(actions: np.array, previous_action: np.array, uav_specifications: Dict = None) -> float:
    dists_total = np.linalg.norm(actions - previous_action, ord=2, axis=1)
    dists_acc = np.array([np.square(uav_specifications["max_v"]) / (2 * uav_specifications["max_a"])] * len(actions))
    dists_acc = np.clip(dists_acc, a_min=None, a_max=0.5 * dists_total)
    dists_const = dists_total - 2 * dists_acc

    times_acc = np.sqrt(2 * dists_acc / uav_specifications["max_a"])
    times_const = dists_const / uav_specifications["max_v"]
    times_total = times_const + 2 * times_acc

    return times_total


def compute_flight_time(action: np.array, previous_action: np.array, uav_specifications: Dict = None) -> float:
    dist_total = np.linalg.norm(action - previous_action, ord=2)
    dist_acc = min(dist_total * 0.5, np.square(uav_specifications["max_v"]) / (2 * uav_specifications["max_a"]))
    dist_const = dist_total - 2 * dist_acc

    time_acc = np.sqrt(2 * dist_acc / uav_specifications["max_a"])
    time_const = dist_const / uav_specifications["max_v"]
    time_total = time_const + 2 * time_acc

    return time_total


def get_actions(
    previous_action: np.array,
    remaining_budget: float,
    grid_map: GridMap,
    min_altitude: float,
    max_altitude: float,
    altitude_spacing: float,
    uav_specifications: Dict = None,
) -> List:
    action_set = []
    height_levels = int((max_altitude - min_altitude) / altitude_spacing) + 1

    for i in range(grid_map.y_dim):
        for j in range(grid_map.x_dim):
            for k in range(height_levels):
                x_pos = grid_map.resolution * j + 0.5 * grid_map.resolution
                y_pos = grid_map.resolution * i + 0.5 * grid_map.resolution
                z_pos = min_altitude + altitude_spacing * k
                action = np.array([x_pos, y_pos, z_pos])
                if 0 < action_costs(action, previous_action, uav_specifications) <= remaining_budget:
                    action_set.append(action)

    return action_set


def flatten_grid_index(grid_map: GridMap, index_2d: np.array) -> int:
    return int(grid_map.x_dim * index_2d[0] + index_2d[1])


def enumerate_actions(grid_map: GridMap, min_altitude: float, max_altitude: float, altitude_spacing: float) -> np.array:
    altitude_levels = np.linspace(min_altitude, max_altitude, int((max_altitude - min_altitude) / altitude_spacing) + 1)
    x_dim = grid_map.x_dim
    y_dim = grid_map.y_dim
    resolution = grid_map.resolution

    x_meshed, y_meshed = np.meshgrid(np.arange(x_dim) * resolution, np.arange(y_dim) * resolution)
    positions = np.array([x_meshed.ravel(), y_meshed.ravel()], dtype=np.float64).T
    cell_offset = np.array([0.5 * grid_map.resolution, 0.5 * grid_map.resolution], dtype=np.float64)
    positions += cell_offset

    actions = {}
    for h, altitude in enumerate(altitude_levels):
        for pos in positions:
            pos_idx2d = (pos - cell_offset) / resolution
            i = flatten_grid_index(grid_map, pos_idx2d)
            actions[h * grid_map.num_grid_cells + i] = np.array([pos[0], pos[1], altitude])

    return actions


def action_dict_to_np_array(actions: Dict) -> np.array:
    actions_np = np.zeros((len(actions), 3))
    for idx, action in actions.items():
        actions_np[idx, :] = action

    return actions_np


def out_of_bounds(waypoint, grid_map: GridMap, min_altitude: float, max_altitude: float):
    in_x_dim = 0 <= waypoint[1] <= grid_map.x_dim * grid_map.resolution
    in_y_dim = 0 <= waypoint[0] <= grid_map.y_dim * grid_map.resolution
    in_z_dim = min_altitude <= waypoint[2] <= max_altitude
    return not (in_x_dim and in_y_dim and in_z_dim)
