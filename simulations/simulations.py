import logging
import os

import cv2
import imageio
import numpy as np

from constants import DATASETS_DIR
from sensors import Sensor
from simulations import Simulation
from simulations import ground_truths
from simulations import sensor_manipulations

logger = logging.getLogger(__name__)


class ScalarFieldSimulation(Simulation):
    def __init__(self, sensor: Sensor, cluster_radius: float = None):
        super().__init__(sensor)

        self.cluster_radius = cluster_radius

    def create_ground_truth_map(self) -> np.array:
        raise NotImplementedError("Scalar field simulation has no function implemented to create ground truth map")

    def take_measurement(self, position: np.array, verbose: bool = True) -> np.array:
        xl, xr, yu, yd = self.sensor.project_field_of_view(position)
        ground_truth_submap = self.get_ground_truth_submap(xl, xr, yu, yd)
        downsampled_submap = sensor_manipulations.downsample_measurement(position, ground_truth_submap, self.sensor)
        model_dependent_noisy_supmap = sensor_manipulations.add_model_dependent_gaussian_noise(
            position, downsampled_submap, self.sensor
        )

        return model_dependent_noisy_supmap


class GaussianRandomField(ScalarFieldSimulation):
    def __init__(self, sensor: Sensor, cluster_radius: float):
        super().__init__(sensor, cluster_radius)

        self.ground_truth_map = self.create_ground_truth_map()

    def create_ground_truth_map(self) -> np.array:
        """Generate random 2D gaussian scalar field with values between 0 and 1"""
        return ground_truths.gaussian_random_field(
            lambda k: k ** (-self.cluster_radius), self.sensor.grid_map.y_dim, self.sensor.grid_map.x_dim
        )


class HotspotRandomField(ScalarFieldSimulation):
    def __init__(self, sensor: Sensor, cluster_radius: float):
        super().__init__(sensor, cluster_radius)

        self.ground_truth_map = self.create_ground_truth_map()

    def create_ground_truth_map(self) -> np.array:
        """Generate random 2D two-cluster hotspot scalar field with values between 0 and 1"""
        high_interest_value = np.random.uniform(low=0.7, high=1)
        low_interest_value = np.random.uniform(low=0.0, high=0.3)

        y_center_idx = np.random.randint(low=self.cluster_radius, high=self.sensor.grid_map.y_dim)
        x_center_idx = np.random.randint(low=self.cluster_radius, high=self.sensor.grid_map.x_dim)

        min_y_idx = int(max(y_center_idx - self.cluster_radius, 0))
        max_y_idx = int(min(y_center_idx + self.cluster_radius, self.sensor.grid_map.y_dim))
        min_x_idx = int(max(x_center_idx - self.cluster_radius, 0))
        max_x_idx = int(min(x_center_idx + self.cluster_radius, self.sensor.grid_map.x_dim))

        ground_truth_map = np.ones((self.sensor.grid_map.y_dim, self.sensor.grid_map.x_dim)) * low_interest_value
        ground_truth_map[min_y_idx:max_y_idx, min_x_idx:max_x_idx] = high_interest_value

        while True:
            tmp_y_center_idx = np.random.randint(low=self.cluster_radius, high=self.sensor.grid_map.y_dim)
            tmp_x_center_idx = np.random.randint(low=self.cluster_radius, high=self.sensor.grid_map.x_dim)

            if (
                np.abs(tmp_y_center_idx - y_center_idx) <= self.cluster_radius
                or np.abs(tmp_x_center_idx - x_center_idx) <= self.cluster_radius
            ):
                continue

            min_y_idx = int(max(tmp_y_center_idx - self.cluster_radius, 0))
            max_y_idx = int(min(tmp_y_center_idx + self.cluster_radius, self.sensor.grid_map.y_dim))
            min_x_idx = int(max(tmp_x_center_idx - self.cluster_radius, 0))
            max_x_idx = int(min(tmp_x_center_idx + self.cluster_radius, self.sensor.grid_map.x_dim))

            ground_truth_map[min_y_idx:max_y_idx, min_x_idx:max_x_idx] = high_interest_value
            break

        return ground_truth_map


class SplitRandomField(ScalarFieldSimulation):
    def __init__(self, sensor: Sensor, cluster_radius: float):
        super().__init__(sensor, cluster_radius)

        self.ground_truth_map = self.create_ground_truth_map()

    def create_ground_truth_map(self) -> np.array:
        """Generate random 2D scalar field split in two parts with high and low values between 0 and 1"""
        high_interest_value = np.random.uniform(low=0.65, high=1)
        low_interest_value = np.random.uniform(low=0.0, high=0.35)

        ground_truth_map = np.ones((self.sensor.grid_map.y_dim, self.sensor.grid_map.x_dim))

        first_value, second_value = high_interest_value, low_interest_value
        if np.random.rand() > 0.5:
            first_value, second_value = low_interest_value, high_interest_value

        if np.random.rand() > 0.5:
            lower_bound_y_idx = np.ceil(self.sensor.grid_map.y_dim * 0.33)
            upper_bound_y_idx = np.ceil(self.sensor.grid_map.y_dim * 0.66)
            y_split_idx = np.random.randint(low=lower_bound_y_idx, high=upper_bound_y_idx + 1)
            ground_truth_map[:y_split_idx, :] = first_value
            ground_truth_map[y_split_idx:, :] = second_value
        else:
            lower_bound_x_idx = np.floor(self.sensor.grid_map.x_dim * 0.33)
            upper_bound_x_idx = np.ceil(self.sensor.grid_map.x_dim * 0.66)
            x_split_idx = np.random.randint(low=lower_bound_x_idx, high=upper_bound_x_idx + 1)
            ground_truth_map[:, :x_split_idx] = first_value
            ground_truth_map[:, x_split_idx:] = second_value

        return ground_truth_map


class TemperatureDataField(ScalarFieldSimulation):
    def __init__(self, sensor: Sensor, filename: str):
        super().__init__(sensor)

        self.raw_data = self.load_raw_data(filename)
        self.ground_truth_map = self.create_ground_truth_map()

    @staticmethod
    def load_raw_data(filename: str) -> np.array:
        file_path = os.path.join(DATASETS_DIR, filename)
        if not os.path.exists(file_path):
            logger.error(f"Cannot find temperature ground truth data! File {file_path} does not exist!")
            raise ValueError

        return imageio.imread(file_path)

    @staticmethod
    def rgba_to_temperature(rbga_map: np.array) -> np.array:
        """Maps rgba data map to real temperature value map"""
        return -1 * (rbga_map[:, :, 0] - rbga_map[:, :, 2])

    @staticmethod
    def normalize_temperature_map(temperature_map: np.array) -> np.array:
        """min-max normalizes temperature map values between 0 and 1"""
        min_temp = np.min(temperature_map)
        max_temp = np.max(temperature_map)

        if min_temp == max_temp:
            return temperature_map / max_temp

        return (temperature_map - min_temp) / (max_temp - min_temp)

    def create_ground_truth_map(self) -> np.array:
        temperature_map = self.rgba_to_temperature(self.raw_data)
        normalized_temperature_map = self.normalize_temperature_map(temperature_map)
        downsampled_temperature_map = cv2.resize(
            normalized_temperature_map,
            dsize=(self.sensor.grid_map.y_dim, self.sensor.grid_map.x_dim),
            interpolation=cv2.INTER_AREA,
        )
        normalized_temperature_map = self.normalize_temperature_map(downsampled_temperature_map)

        return normalized_temperature_map
