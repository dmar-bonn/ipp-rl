import time
from typing import Dict, Tuple

import numpy as np

from constants import MissionType
from mapping.mappings import Mapping
from planning.common.actions import action_costs, compute_flight_time
from planning.missions import Mission


class ConicalSpiralMission(Mission):
    def __init__(
        self,
        mapping: Mapping,
        uav_specifications: Dict,
        dist_to_boundaries: float = 10,
        min_altitude: float = 5,
        max_altitude: float = 30,
        num_waypoints: int = 200,
        slope_factor: float = 10,
        budget: float = 100,
        adaptive: bool = False,
        value_threshold: float = 0.5,
        interval_factor: float = 2,
        config_name: str = "standard",
        use_effective_mission_time: bool = False,
    ):
        """
        Defines a static planning mission performing num_waypoints sensor measurements along
        a conical 3D spiral shrinking in radius with increasing altitude.

        Args:
            mapping (Mapping): mapping algorithm with related sensor and grid map specification
            uav_specifications (dict): uav parameters defining max_v, max_a, sampling_time
            dist_to_boundaries (float): minimal distance [m] of a waypoint to map boundaries
            min_altitude (float): lowest altitude level [m] of waypoints above ground
            max_altitude (float): highest altitude level [m] of waypoints above ground
            num_waypoints (int): total number of randomly sampled waypoints
            slope_factor (float): more spirals with higher slope factor
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

        self.num_waypoints = num_waypoints
        self.slope_factor = slope_factor
        self.mission_name = "Conical Spiral"
        self.mission_type = MissionType.CONICAL_SPIRAL

    def create_waypoints(self) -> Tuple[np.array, float]:
        """
        Generates num_waypoints waypoints sampled with fixed step size from 3D conical spiral with decreasing
        radius as the altitude increases. The spiral is bounded by dist_to_boundaries in its xy-stretch-out,
        and it is bounded by min_altitude and max_altitude in z-direction.

        Returns:
            (np.array): sampled waypoints from 3D conical spiral
        """
        start_run_time = time.time()
        t_max = (
            0.5
            * np.minimum(
                self.mapping.grid_map.x_dim * self.mapping.grid_map.resolution,
                self.mapping.grid_map.y_dim * self.mapping.grid_map.resolution,
            )
            - self.dist_to_boundaries
        )
        c = (self.max_altitude - self.min_altitude) / t_max ** 2
        t = np.linspace(0, t_max, self.num_waypoints)

        x_coords = (
            t * np.cos(self.slope_factor * t) + 0.5 * self.mapping.grid_map.y_dim * self.mapping.grid_map.resolution
        )
        y_coords = (
            t * np.sin(self.slope_factor * t) + 0.5 * self.mapping.grid_map.x_dim * self.mapping.grid_map.resolution
        )
        z_coords = c * np.square(t) + self.min_altitude
        z_coords = np.flip(z_coords)

        tmp_waypoints = np.transpose(np.array([x_coords, y_coords, z_coords]))

        waypoints = [tmp_waypoints[0, :]]
        remaining_budget = self.budget - action_costs(self.init_action, tmp_waypoints[0, :], self.uav_specifications)
        for i in range(1, tmp_waypoints.shape[0]):
            remaining_budget -= action_costs(tmp_waypoints[i - 1, :], tmp_waypoints[i, :], self.uav_specifications)
            if remaining_budget >= 0:
                waypoints.append(tmp_waypoints[i, :])
            else:
                break

        return np.array(waypoints), time.time() - start_run_time

    def execute(self):
        self.eval(run_time=0, flight_time=0)
        self.waypoints, total_run_time = self.create_waypoints()
        """
        self.waypoints = self.mav_trajectory_generator.plan_uav_trajectory(
            self.waypoints, sampling_time=self.uav_specifications["sampling_time"]
        )
        """

        previous_measurement_position = self.init_action
        for measurement_position in self.waypoints:
            simulated_raw_measurement = self.mapping.sensor.take_measurement(measurement_position)
            self.mapping.update_grid_map(measurement_position, simulated_raw_measurement)
            flight_time = compute_flight_time(
                measurement_position, previous_measurement_position, self.uav_specifications
            )
            previous_measurement_position = measurement_position
            self.eval(run_time=total_run_time / len(self.waypoints), flight_time=flight_time)
