import logging
import math
from typing import Tuple

import numpy as np

from mapping.grid_maps import GridMap
from planning.common.actions import flatten_grid_index
from sensors.models import SensorModel

logger = logging.getLogger(__name__)


class AltitudeSensorModel(SensorModel):
    def __init__(self, coeff_a: float, coeff_b: float):
        """

        Args:
            coeff_a (float): altitude scaling factor
            coeff_b (float): total noise scaling factor
        """
        super().__init__()

        self.coeff_a = coeff_a
        self.coeff_b = coeff_b

    def get_noise_variance(self, position: np.array) -> float:
        """Returns sensor measurement noise scaling with altitude"""
        altitude = position[2]
        return self.coeff_a * (1 - np.exp(-self.coeff_b * altitude))

    def measurement_variance_matrix(
        self, position: np.array, num_measurements: int, resolution_factor: float
    ) -> np.array:
        """Diagonal matrix with measurement variance scaling with altitude of measurement position"""
        return resolution_factor ** 3 * self.get_noise_variance(position) * np.identity(num_measurements)

    def measurement_model_matrix(
        self, grid_map: GridMap, field_of_view_indices: Tuple, num_measurements: np.array, resolution_factor: int
    ) -> np.array:
        """
        Constructs Kalman filter measurement model matrix. Selects measurements influencing a grid cell with some
        measurement's resolution factor dependent importance weight.

        Args:
            grid_map (GridMap): grid map representation with environment dimensions
            field_of_view_indices (Tuple): tuple of four integers defining the camera's FoV vertices
            num_measurements (int): number of grid cells affected by sensor measurement
            resolution_factor (int): altitude-dependent measurement resolution factor

        Returns:
            (np.array): altitude-dependent measurement model matrix measurements x grid cells
        """
        xl, xr, yu, yd = field_of_view_indices
        H = np.zeros((num_measurements, grid_map.num_grid_cells))

        num_measurement_indices_x = math.floor((xr - xl) / resolution_factor) + 1
        grid_indices_x = np.linspace(xl, xr, math.floor((xr - xl)) + 1)
        grid_indices_y = np.linspace(yu, yd, math.floor((yd - yu)) + 1)

        for i in range(num_measurements):
            y_start = int(i / num_measurement_indices_x)
            x_start = int(i - num_measurement_indices_x * y_start)
            x_end = min(x_start * resolution_factor + resolution_factor, len(grid_indices_x))
            y_end = min(y_start * resolution_factor + resolution_factor, len(grid_indices_y))
            x_start = min(x_start * resolution_factor, x_end)
            y_start = min(y_start * resolution_factor, y_end)
            sub_measurement_indices_x = grid_indices_x[x_start:x_end]
            sub_measurement_indices_y = grid_indices_y[y_start:y_end]
            sub_indices_x_meshed, sub_indices_y_meshed = np.meshgrid(
                sub_measurement_indices_y, sub_measurement_indices_x
            )
            measurement_indices = np.array([sub_indices_x_meshed.ravel(), sub_indices_y_meshed.ravel()]).T

            flattened_measurement_indices = self.flatten_2d_indices(measurement_indices, grid_map.x_dim)
            importance_weight = 1 / resolution_factor ** 2
            if len(flattened_measurement_indices) < resolution_factor ** 2:
                importance_weight = 1 / resolution_factor
            H[i, flattened_measurement_indices] = importance_weight

        return H

    @staticmethod
    def flatten_2d_indices(indices_2d: np.array, x_dim: int) -> np.array:
        return (x_dim * indices_2d[:, 0] + indices_2d[:, 1]).astype(int)
