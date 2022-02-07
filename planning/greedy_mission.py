import logging
import time
from typing import Dict

import numpy as np

from constants import MissionType
from mapping.mappings import Mapping
from planning.common.actions import action_costs, compute_flight_time
from planning.common.optimization import greedy_search
from planning.missions import Mission

logger = logging.getLogger(__name__)


class GreedyMission(Mission):
    def __init__(
        self,
        mapping: Mapping,
        uav_specifications: Dict,
        dist_to_boundaries: float = 10,
        min_altitude: float = 5,
        max_altitude: float = 30,
        num_waypoints: int = 100,
        altitude_spacing: float = 5,
        budget: float = 400,
        adaptive: bool = False,
        value_threshold: float = 0.5,
        interval_factor: float = 2,
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
                num_waypoints (int): total number of greedily sampled waypoints
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

        self.num_waypoints = num_waypoints
        self.altitude_spacing = altitude_spacing
        self.mission_name = "Greedy"
        self.mission_type = MissionType.GREEDY

    def create_waypoints(self) -> np.array:
        raise NotImplementedError("Greedy planning mission does not implement 'create_waypoints' function!")

    def execute(self):
        """Generates num_waypoints waypoints greedily chosen to maximize reward."""
        self.eval(run_time=0, flight_time=0)
        previous_action = self.init_action
        remaining_budget = self.budget

        while remaining_budget >= 0:
            logger.info(f"Remaining budget: {remaining_budget}")
            start_time = time.time()
            greedy_waypoints = greedy_search(
                previous_action,
                remaining_budget,
                self.mapping.grid_map.cov_matrix,
                1,
                self.mapping,
                self.min_altitude,
                self.max_altitude,
                self.altitude_spacing,
                self.uav_specifications,
                adaptive_info=self.get_adaptive_info(),
            )
            finish_time = time.time()
            if len(greedy_waypoints) == 0:
                break

            next_waypoint = np.array(greedy_waypoints[0])
            simulated_raw_measurement = self.mapping.sensor.take_measurement(next_waypoint)
            self.mapping.update_grid_map(next_waypoint, simulated_raw_measurement)
            self.waypoints = np.vstack((self.waypoints, next_waypoint))

            run_time = finish_time - start_time
            remaining_budget -= action_costs(next_waypoint, previous_action, self.uav_specifications)
            if self.use_effective_mission_time:
                remaining_budget -= run_time

            flight_time = compute_flight_time(next_waypoint, previous_action, self.uav_specifications)
            previous_action = next_waypoint
            self.eval(run_time=run_time, flight_time=flight_time)
