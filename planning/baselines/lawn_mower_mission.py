import time
from typing import Dict, List, Tuple

import numpy as np

from constants import MissionType
from mapping.mappings import Mapping
from planning.common.actions import action_costs, compute_flight_time
from planning.missions import Mission


class LawnMowerMission(Mission):
    def __init__(
        self,
        mapping: Mapping,
        uav_specifications: Dict,
        dist_to_boundaries: float = 10,
        min_altitude: float = 5,
        max_altitude: float = 30,
        step_size: float = 5,
        altitude_spacing: float = 5,
        budget: float = 100,
        adaptive: bool = False,
        value_threshold: float = 0.5,
        interval_factor: float = 2,
        config_name: str = "standard",
        use_effective_mission_time: bool = False,
    ):
        """
        Defines a static planning mission performing sensor measurements along a created lawnmower path.

        Args:
            mapping (Mapping): mapping algorithm with related sensor and grid map specification
            uav_specifications (dict): uav parameters defining max_v, max_a, sampling_time
            dist_to_boundaries (float): minimal distance [m] of a waypoint to map boundaries
            min_altitude (float): lowest altitude level [m] of waypoints above ground
            max_altitude (float): highest altitude level [m] of waypoints above ground
            step_size (float): spacing [m] between sampled waypoints along lawnmower path
            altitude_spacing (float): spacing [m] between consecutive altitude levels
            budget (float): total distance budget for mission
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

        self.step_size = step_size
        self.altitude_spacing = altitude_spacing
        self.mission_name = "Coverage"
        self.mission_type = MissionType.LAWNMOWER

    def create_waypoints(self) -> Tuple[np.array, List]:
        """
        Generates waypoints each step_size meters along a lawnmower path covering the grid map up to a certain
        distance [m] to the map's boundaries from min_altitude to min_altitude level.

        Returns:
            (np.array): sampled waypoints on lawnmower path
        """
        min_x_pos = self.dist_to_boundaries
        max_x_pos = self.mapping.grid_map.x_dim * self.mapping.grid_map.resolution - self.dist_to_boundaries
        x_measurement_coords = np.linspace(min_x_pos, max_x_pos, int((max_x_pos - min_x_pos) / self.step_size) + 1)

        min_y_pos = self.dist_to_boundaries
        max_y_pos = self.mapping.grid_map.y_dim * self.mapping.grid_map.resolution - self.dist_to_boundaries
        y_measurement_coords = np.linspace(min_y_pos, max_y_pos, int((max_y_pos - min_y_pos) / self.step_size) + 1)

        measurement_positions = []
        run_times = []
        last_run_time = time.time()
        remaining_budget = self.budget
        previous_waypoint = self.init_action
        altitude_levels = np.linspace(
            self.min_altitude,
            self.max_altitude,
            int((self.max_altitude - self.min_altitude) / self.altitude_spacing) + 1,
        )
        for i, altitude_level in enumerate(altitude_levels):
            if remaining_budget < self.step_size:
                break
            for j, y_measurement_coord in enumerate(y_measurement_coords):
                if remaining_budget < self.step_size:
                    break
                for k, x_measurement_coord in enumerate(x_measurement_coords):
                    if remaining_budget < self.step_size:
                        break

                    if j % 2 == 1:
                        x_measurement_coord = (
                            self.mapping.grid_map.x_dim * self.mapping.grid_map.resolution - x_measurement_coord
                        )

                    waypoint = np.array([y_measurement_coord, x_measurement_coord, altitude_level], dtype=int)
                    remaining_budget -= action_costs(waypoint, previous_waypoint, self.uav_specifications)
                    previous_waypoint = waypoint
                    measurement_positions.append(waypoint)
                    run_times.append(time.time() - last_run_time)
                    last_run_time = time.time()

        return np.array(measurement_positions), run_times

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
