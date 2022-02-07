import time
from typing import Dict, List, Tuple

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
from planning.missions import Mission


class RandomDiscreteMission(Mission):
    def __init__(
        self,
        mapping: Mapping,
        uav_specifications: Dict,
        dist_to_boundaries: float = 10,
        min_altitude: float = 5,
        max_altitude: float = 30,
        budget: float = 100,
        altitude_spacing: float = 5,
        adaptive: bool = False,
        value_threshold: float = 0.5,
        interval_factor: float = 2,
        config_name: str = "standard",
        use_effective_mission_time: bool = False,
    ):
        """
        Defines a static planning mission performing sensor measurements at randomly sampled waypoints.

        Args:
            mapping (Mapping): mapping algorithm with related sensor and grid map specification
            uav_specifications (dict): uav parameters defining max_v, max_a, sampling_time
            dist_to_boundaries (float): minimal distance [m] of a waypoint to map boundaries
            min_altitude (float): lowest altitude level [m] of waypoints above ground
            max_altitude (float): highest altitude level [m] of waypoints above ground
            budget (float): total distance budget for mission
            altitude_spacing (float): spacing [m] between consecutive altitude levels
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

        self.mission_name = "Random"
        self.mission_type = MissionType.RANDOM_DISCRETE
        self.altitude_spacing = altitude_spacing
        self.actions = enumerate_actions(
            self.mapping.grid_map, self.min_altitude, self.max_altitude, self.altitude_spacing
        )
        self.actions_np = action_dict_to_np_array(self.actions)

    def get_next_actions_mask(self, position: np.array, budget: float) -> np.array:
        distances = np.linalg.norm(self.actions_np - position, ord=2, axis=1)
        if not self.adaptive:
            return (distances > 0) & (distances <= budget) & (distances < 11.5)

        flight_times = compute_flight_times(self.actions_np, position, self.uav_specifications)
        return (flight_times > 0) & (flight_times <= budget)

    def create_waypoints(self) -> Tuple[np.array, List]:
        """
        Generates waypoints sampled uniformly at random from all positions in 3D grid over map.

        Returns:
            (np.array): sampled waypoints uniformly at random from discrete 3D grid
        """
        waypoints = []
        run_times = []
        last_run_time = time.time()
        previous_waypoint = self.init_action
        remaining_budget = self.budget
        while remaining_budget >= 0:
            msk = self.get_next_actions_mask(previous_waypoint, remaining_budget)
            if msk.sum() == 0:
                break

            remaining_actions = self.actions_np[msk, :]
            action_idx = np.random.choice(len(remaining_actions))
            waypoint = remaining_actions[action_idx]
            remaining_budget -= action_costs(waypoint, previous_waypoint, self.uav_specifications)
            waypoints.append(waypoint)
            previous_waypoint = waypoint
            run_times.append(time.time() - last_run_time)
            last_run_time = time.time()

        return np.array(waypoints), run_times

    def execute(self):
        self.eval(run_time=0, flight_time=0)
        self.waypoints, run_times = self.create_waypoints()
        """
        self.waypoints = self.mav_trajectory_generator.plan_uav_trajectory(
            self.waypoints, sampling_time=self.uav_specifications["sampling_time"]
        )
        """

        previous_measurement_position = self.init_action
        remaining_budget = self.budget
        for i, measurement_position in enumerate(self.waypoints):
            next_action_costs = action_costs(
                measurement_position, previous_measurement_position, self.uav_specifications
            )
            if remaining_budget <= next_action_costs:
                break

            simulated_raw_measurement = self.mapping.sensor.take_measurement(measurement_position)
            self.mapping.update_grid_map(measurement_position, simulated_raw_measurement)
            flight_time = compute_flight_time(
                measurement_position, previous_measurement_position, self.uav_specifications
            )
            previous_measurement_position = measurement_position
            remaining_budget -= next_action_costs
            self.eval(run_time=run_times[i], flight_time=flight_time)
