import logging
import os
import time
from typing import Dict, List, Tuple

import cma
import numpy as np

from constants import MissionType
from mapping.mappings import Mapping
from planning.common.actions import action_costs, compute_flight_time, out_of_bounds
from planning.common.optimization import greedy_search, simulate_prediction_step
from planning.missions import Mission

logger = logging.getLogger(__name__)


class IPPMashaMission(Mission):
    def __init__(
        self,
        mapping: Mapping,
        uav_specifications: Dict,
        dist_to_boundaries: float = 10,
        min_altitude: float = 5,
        max_altitude: float = 30,
        episode_horizon: int = 10,
        altitude_spacing: float = 5,
        budget: float = 100,
        cmaes_sigma0: List = [2.0, 2.0, 0.5],
        cmaes_max_iter: int = 32,
        cmaes_population_size: int = 12,
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
                episode_horizon (int): total number of greedily sampled waypoints
                altitude_spacing (float): spacing [m] between consecutive altitude levels
                budget (float): total distance budget for mission
                cmaes_sigma0 (list): initial standard deviation for CMA-ES in x, y, and z coordinates
                cmaes_max_iter (int): maximal number of generations until CMA-ES termination
                cmaes_population_size (int): number of offsprings sampled in each CMA-ES iteration
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

        self.episode_horizon = episode_horizon
        self.altitude_spacing = altitude_spacing
        self.remaining_budget = budget
        self.previous_replan_action = self.init_action
        self.cmaes_sigma0 = cmaes_sigma0
        self.cmaes_max_iter = cmaes_max_iter
        self.cmaes_population_size = cmaes_population_size
        self.mission_name = "CMA-ES"
        self.mission_type = MissionType.IPP_MASHA

    def create_waypoints(self) -> np.array:
        raise NotImplementedError("Masha's IPP planning mission does not implement 'create_waypoints' function!")

    @staticmethod
    def flatten_waypoints(waypoints: List):
        flattened = []
        for waypoint in waypoints:
            flattened.extend([waypoint[0], waypoint[1], waypoint[2]])

        return flattened

    @staticmethod
    def stacked_waypoints(waypoints: List) -> List:
        stacked = []
        for i in range(len(waypoints) // 3):
            stacked.append(np.array([waypoints[3 * i], waypoints[3 * i + 1], waypoints[3 * i + 2]]))

        return stacked

    def simulate_trajectory(self, waypoints: List) -> float:
        waypoints = self.stacked_waypoints(waypoints)
        for waypoint in waypoints:
            if out_of_bounds(waypoint, self.mapping.grid_map, self.min_altitude, self.max_altitude):
                return 100

        path_consumed_budget = 0
        previous_action = self.previous_replan_action
        for waypoint in waypoints:
            path_consumed_budget += action_costs(waypoint, previous_action, self.uav_specifications)
            previous_action = waypoint

        if path_consumed_budget <= 0:
            return 100

        total_reward = 0
        remaining_budget = self.remaining_budget
        previous_action = self.previous_replan_action
        current_state = self.mapping.grid_map.cov_matrix

        for waypoint in waypoints:
            action_cost = action_costs(waypoint, previous_action, self.uav_specifications)
            if action_cost > remaining_budget:
                break

            reward, _, next_state = simulate_prediction_step(
                current_state,
                previous_action,
                waypoint,
                self.mapping,
                self.uav_specifications,
                adaptive_info=self.get_adaptive_info(),
            )
            total_reward += reward * (action_cost + 1)
            current_state = next_state
            previous_action = waypoint
            remaining_budget -= action_cost

        return -total_reward / path_consumed_budget

    def calculate_parameter_bounds_and_scales(self, num_waypoints: int) -> Tuple[List, List, List]:
        lower_bounds = []
        upper_bounds = []
        sigma_scales = []

        lower_z = self.min_altitude
        upper_x = self.mapping.grid_map.y_dim * self.mapping.grid_map.resolution
        upper_y = self.mapping.grid_map.x_dim * self.mapping.grid_map.resolution
        upper_z = self.max_altitude
        sigma_scale_z = min(self.cmaes_sigma0[2], (self.max_altitude - self.min_altitude) / 2)

        for i in range(num_waypoints):
            lower_bounds.extend([0, 0, lower_z])
            upper_bounds.extend([upper_x, upper_y, upper_z])
            sigma_scales.extend([self.cmaes_sigma0[0], self.cmaes_sigma0[1], sigma_scale_z])

        return lower_bounds, upper_bounds, sigma_scales

    def cma_es_optimization(self, init_waypoints: np.array) -> List:
        lower_bounds, upper_bounds, sigma_scales = self.calculate_parameter_bounds_and_scales(len(init_waypoints))
        cma_es = cma.CMAEvolutionStrategy(
            self.flatten_waypoints(init_waypoints),
            sigma0=1,
            inopts={
                "bounds": [lower_bounds, upper_bounds],
                "maxiter": self.cmaes_max_iter,
                "popsize": self.cmaes_population_size,
                "CMA_stds": sigma_scales,
            },
        )
        with cma.optimization_tools.EvalParallel2(self.simulate_trajectory, number_of_processes=4) as eval_all:
            while not cma_es.stop():
                offspring_waypoints = cma_es.ask()
                cma_es.tell(offspring_waypoints, eval_all(offspring_waypoints))
                cma_es.disp()

        return list(cma_es.result.xbest)

    def replan(self, previous_action: np.array, remaining_budget: float) -> np.array:
        """
        Generates episode_horizon waypoints greedily chosen to maximize reward. The greedy solution is then refined
        by a gradient-free evolutionary CMA-ES optimization.

        Args:
            previous_action (np.array): last measurement position of UAV
            remaining_budget (float): budget left after previous_action has been executed

        Returns:
            (np.array): waypoints chosen to optimize the reward
        """
        greedy_waypoints = greedy_search(
            previous_action,
            remaining_budget,
            self.mapping.grid_map.cov_matrix,
            self.episode_horizon,
            self.mapping,
            self.min_altitude,
            self.max_altitude,
            self.altitude_spacing,
            self.uav_specifications,
            adaptive_info=self.get_adaptive_info(),
        )
        greedy_waypoints_utility = -self.simulate_trajectory(self.flatten_waypoints(greedy_waypoints))
        logger.info(f"\nGREEDY SOLUTION total reward {greedy_waypoints_utility}")

        if len(greedy_waypoints) == 0:
            return np.array([])

        waypoints = self.cma_es_optimization(greedy_waypoints)
        cma_es_waypoints_utility = -self.simulate_trajectory(waypoints)
        logger.info(f"\nCMA-ES TUNED SOLUTION total reward {cma_es_waypoints_utility}")

        if greedy_waypoints_utility > cma_es_waypoints_utility:
            waypoints = self.flatten_waypoints(greedy_waypoints)

        waypoints = self.stacked_waypoints(waypoints)

        return np.array(waypoints)

    def execute(self):
        self.eval(run_time=0, flight_time=0)
        previous_action = self.previous_replan_action
        remaining_budget = self.budget

        while remaining_budget >= 0:
            logger.info(f"\nREMAINING BUDGET: {remaining_budget}")
            start_run_time = time.time()
            next_waypoints = self.replan(previous_action, remaining_budget)
            finish_run_time = time.time()
            if len(next_waypoints) == 0:
                break

            if self.adaptive:
                next_waypoints = np.expand_dims(next_waypoints[0, :], axis=0)

            self.waypoints = np.vstack((self.waypoints, next_waypoints))
            for measurement_position in next_waypoints:
                if remaining_budget <= 0:
                    break

                run_time = (finish_run_time - start_run_time) / len(next_waypoints)
                remaining_budget -= action_costs(measurement_position, previous_action, self.uav_specifications)
                if self.use_effective_mission_time:
                    remaining_budget -= run_time

                simulated_raw_measurement = self.mapping.sensor.take_measurement(measurement_position)
                self.mapping.update_grid_map(measurement_position, simulated_raw_measurement)
                flight_time = compute_flight_time(measurement_position, previous_action, self.uav_specifications)
                previous_action = measurement_position
                self.eval(run_time=run_time, flight_time=flight_time)
