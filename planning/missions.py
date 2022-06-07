from typing import Dict, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from mapping.mappings import Mapping
from planning.evaluation_metrics import (
    map_uncertainty,
    map_uncertainty_difference,
    mean_log_loss,
    root_mean_squared_error,
    weighted_mean_log_loss,
    weighted_root_mean_squared_error,
)


# from planning.trajectory_generation.mav_trajectory_generation import MavTrajectoryGenerator


class Mission:
    def __init__(
        self,
        mapping: Mapping,
        uav_specifications: Dict,
        dist_to_boundaries: float = 10,
        min_altitude: float = 5,
        max_altitude: float = 30,
        budget: float = 100,
        adaptive: bool = False,
        value_threshold: float = 0.5,
        interval_factor: float = 2,
        config_name: str = "standard",
        use_effective_mission_time: bool = False,
    ):
        """
        Defines the common interface each planning mission shares.

        Args:
            mapping (Mapping): mapping algorithm with related sensor and grid map specification
            uav_specifications (dict): uav parameters defining max_v, max_a, sampling_time
            dist_to_boundaries (float): minimal distance [m] of a waypoint to map boundaries
            min_altitude (float): lowest altitude level [m] of waypoints above ground
            max_altitude (float): highest altitude level [m] of waypoints above ground
            budget (float): total distance budget for mission
            adaptive (bool): indicates if mission should be executed as adaptive or non-adaptive IPP
            value_threshold (float): grid cells with upper CI bounds above this threshold are of interest
            interval_factor (float): defines the width of the CI used to decide if a grid cell is of interest
            config_name (str): descriptive name of chosen mission's hyper-parameter configuration
            use_effective_mission_time (bool): if true, decrease remaining budget additionally by thinking time
        """
        super(Mission, self).__init__()

        self.mapping = mapping
        self.uav_specifications = uav_specifications
        self.dist_to_boundaries = dist_to_boundaries
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.budget = budget
        self.adaptive = adaptive
        self.value_threshold = value_threshold
        self.interval_factor = interval_factor
        self.waypoints = np.empty((0, 3))
        self.config_name = config_name
        self.use_effective_mission_time = use_effective_mission_time
        self.mission_type = None
        self.mission_name = None
        self.init_action = np.array([2, 2, 14])

        """
        self.mav_trajectory_generator = MavTrajectoryGenerator(
            self.uav_specifications["max_v"], self.uav_specifications["max_a"]
        )
        """

        self.root_mean_squared_errors = []
        self.weighted_root_mean_squared_errors = []
        self.mean_log_losses = []
        self.weighted_mean_log_losses = []
        self.map_uncertainties = []
        self.map_uncertainty_differences = []
        self.run_times = []
        self.flight_times = []

    def create_waypoints(self) -> np.array:
        raise NotImplementedError("Planning mission does not implement 'create_waypoints' function!")

    def execute(self):
        raise NotImplementedError("Planning mission does not implement 'execute' function!")

    def get_adaptive_info(self) -> Optional[Dict]:
        if not self.adaptive:
            return None

        return {
            "mean": self.mapping.grid_map.mean,
            "value_threshold": self.value_threshold,
            "interval_factor": self.interval_factor,
        }

    def visualize_path(self):
        """Visualizes path with waypoints, i.e. measurement positions"""
        ax = plt.axes(projection="3d")
        ax.plot3D(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2], "rx-")
        plt.show()

    def visualize_grid_map(self, position: np.array = None):
        plt.title(f"{self.mission_label} - Grid map ground truth - mean - variance")

        plt.subplot(1, 3, 1)
        plt.imshow(
            self.mapping.sensor.sensor_simulation.ground_truth_map,
            extent=[
                0,
                self.mapping.grid_map.x_dim * self.mapping.grid_map.resolution,
                self.mapping.grid_map.y_dim * self.mapping.grid_map.resolution,
                0,
            ],
            vmin=0,
            vmax=1,
            cmap="plasma",
            aspect="equal",
        )
        plt.colorbar()

        ax = plt.subplot(1, 3, 2)
        plt.imshow(
            self.mapping.grid_map.mean,
            extent=[
                0,
                self.mapping.grid_map.x_dim * self.mapping.grid_map.resolution,
                self.mapping.grid_map.y_dim * self.mapping.grid_map.resolution,
                0,
            ],
            vmin=0,
            vmax=1,
            cmap="plasma",
            aspect="equal",
        )
        plt.colorbar()

        if position is not None:
            xl, xr, yu, yd = self.mapping.sensor.project_field_of_view(position)
            resolution = self.mapping.grid_map.resolution
            rect = patches.Rectangle(
                (xl * resolution, yu * resolution),
                (xr - xl + 1) * resolution,
                (yd - yu + 1) * resolution,
                linewidth=3,
                edgecolor="red",
                facecolor="none",
            )

            # Add the patch to the Axes
            ax.add_patch(rect)

        plt.subplot(1, 3, 3)
        plt.imshow(
            np.diag(self.mapping.grid_map.cov_matrix).reshape(self.mapping.grid_map.y_dim, self.mapping.grid_map.x_dim),
            extent=[
                0,
                self.mapping.grid_map.x_dim * self.mapping.grid_map.resolution,
                self.mapping.grid_map.y_dim * self.mapping.grid_map.resolution,
                0,
            ],
            vmin=0,
            vmax=1,
            cmap="plasma",
            aspect="equal",
        )
        plt.colorbar()

        plt.show()

    def eval(self, run_time: float = None, flight_time: float = None):
        """Computes evaluation metrics for mean and covariance of map estimate and adds these to metrics history"""
        ground_truth_map = self.mapping.sensor.sensor_simulation.ground_truth_map
        adaptive_msk = ground_truth_map.flatten(order="C") >= self.value_threshold if self.adaptive else None

        self.root_mean_squared_errors.append(
            root_mean_squared_error(ground_truth_map, self.mapping.grid_map.mean, adaptive_msk)
        )
        self.weighted_root_mean_squared_errors.append(
            weighted_root_mean_squared_error(ground_truth_map, self.mapping.grid_map.mean)
        )
        self.mean_log_losses.append(
            mean_log_loss(ground_truth_map, self.mapping.grid_map.mean, self.mapping.grid_map.cov_matrix)
        )
        self.weighted_mean_log_losses.append(
            weighted_mean_log_loss(ground_truth_map, self.mapping.grid_map.mean, self.mapping.grid_map.cov_matrix)
        )
        self.map_uncertainties.append(map_uncertainty(self.mapping.grid_map.cov_matrix, adaptive_msk))
        if self.adaptive:
            self.map_uncertainty_differences.append(
                map_uncertainty_difference(self.mapping.grid_map.cov_matrix, adaptive_msk)
            )

        if run_time is not None:
            self.run_times.append(run_time)

        if flight_time is not None:
            self.flight_times.append(flight_time)

    def visualize_eval(self):
        """Plot tracked evaluation metric histories as 2D line plot"""
        plt.title(f"RMSE - {self.mission_label}")
        plt.xlabel("number of measurements")
        plt.ylabel("RMSE")
        plt.plot(self.root_mean_squared_errors, "b-^")
        plt.show()

        plt.title(f"WRMSE - {self.mission_label}")
        plt.xlabel("number of measurements")
        plt.ylabel("WRSME")
        plt.plot(self.weighted_root_mean_squared_errors, "b-^")
        plt.show()

        plt.title(f"MLL - {self.mission_label}")
        plt.xlabel("number of measurements")
        plt.ylabel("MLL")
        plt.plot(self.mean_log_losses, "b-^")
        plt.show()

        plt.title(f"WMLL - {self.mission_label}")
        plt.xlabel("number of measurements")
        plt.ylabel("WMLL")
        plt.plot(self.weighted_mean_log_losses, "b-^")
        plt.show()

        plt.title(f"tr(P) - {self.mission_label}")
        plt.xlabel("number of measurements")
        plt.ylabel("tr(P)")
        plt.plot(self.map_uncertainties, "b-^")
        plt.show()

    @property
    def mission_label(self):
        return f"{self.mission_name} ({self.config_name})"
