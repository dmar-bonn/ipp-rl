import logging

import matplotlib.pyplot as plt
import numpy as np

from sensors import Sensor

logger = logging.getLogger(__name__)


class Simulation:
    def __init__(self, sensor: Sensor):
        super(Simulation, self).__init__()

        self.sensor = sensor
        self.ground_truth_map = None

    def create_ground_truth_map(self):
        raise NotImplementedError("Sensor simulation has no function implemented to create ground truth map")

    def take_measurement(self, position: np.array, verbose: bool = True):
        raise NotImplementedError("Sensor simulation has no function implemented to take measurement")

    def get_ground_truth_submap(self, xl: int, xr: int, yu: int, yd: int) -> np.array:
        return self.ground_truth_map[yu : yd + 1, xl : xr + 1]

    def visualize_ground_truth_map(self):
        plt.title("Ground truth map")
        plt.imshow(self.ground_truth_map, vmin=0, vmax=1, cmap="cividis")
        plt.colorbar()
        plt.show()
