import cv2
import numpy as np

from sensors import Sensor


def downsample_measurement(position: np.array, original_data: np.array, sensor: Sensor) -> np.array:
    """
    Measured sensor data downsampled by measurement's altitude-dependent resolution.

    Args:
        position (np.array): position at which sensor data is measured
        original_data (np.array): sensor data measured
        sensor (Sensor): sensor which measured the data

    Returns:
        (np.array): altitude-dependently downsampled sensor data
    """
    resolution_factor = sensor.get_resolution_factor(position)
    y_dim, x_dim = np.ceil(np.array(original_data.shape) / resolution_factor).astype(int)
    if resolution_factor > 1:
        downsampled_data = cv2.resize(original_data, dsize=(y_dim, x_dim), interpolation=cv2.INTER_AREA)
    else:
        downsampled_data = original_data

    return downsampled_data


def add_random_gaussian_noise(original_data: np.array, noise_factor: float = 0.05) -> np.array:
    """
    Adds gaussian noise to sensor data with variance relative to absolute value range of measurement.

    Args:
        original_data (np.array): measured sensor data
        noise_factor (float): between 0 and 1, scales variance relative to absolute value range of measurement

    Returns:
        (np.array): noisy sensor data
    """
    noise_variance = noise_factor * (np.max(original_data) - np.min(original_data))
    return np.clip(original_data + np.random.normal(0, noise_variance, original_data.shape), a_min=0, a_max=1)


def add_model_dependent_gaussian_noise(position: np.array, original_data: np.array, sensor: Sensor) -> np.array:
    """
    Adds gaussian noise to sensor data with variance defined by the sensor's underlying model.

    Args:
        position (np.array): position at which sensor data is measured
        original_data (np.array): measured sensor data
        sensor (Sensor): sensor which measured the data

    Returns:
        (np.array): noisy sensor data
    """
    noise_variance = sensor.sensor_model.get_noise_variance(position)
    return np.clip(original_data + np.random.normal(0, noise_variance, original_data.shape), a_min=0, a_max=1)
