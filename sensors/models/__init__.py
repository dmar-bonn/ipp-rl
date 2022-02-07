import numpy as np


class SensorModel:
    def __init__(self):
        super(SensorModel, self).__init__()

    def get_noise_variance(self, position: np.array) -> float:
        raise NotImplementedError("Sensor has no noise variance function implemented")
