import numpy as np


def root_mean_squared_error(
    ground_truth_map: np.array, estimated_map: np.array, adaptive_msk: np.array = None
) -> float:
    """Returns root-mean-squared- error (RMSE) between ground truth and estimated map"""
    if adaptive_msk is None:
        return np.sqrt(np.mean(np.square(ground_truth_map - estimated_map)))

    adaptive_ground_truth_map = ground_truth_map.flatten(order="C")[adaptive_msk]
    adaptive_estimated_map = estimated_map.flatten(order="C")[adaptive_msk]
    return np.sqrt(np.mean(np.square(adaptive_ground_truth_map - adaptive_estimated_map)))


def map_uncertainty(estimated_map_covariance_matrix: np.array, adaptive_msk: np.array = None) -> float:
    """Returns trace of covariance matrix over all grid cells as a proxy for map estimation uncertainty"""
    if adaptive_msk is None:
        return np.trace(estimated_map_covariance_matrix)

    return np.sum(np.diag(estimated_map_covariance_matrix)[adaptive_msk])


def map_uncertainty_difference(estimated_map_covariance_matrix: np.array, adaptive_msk: np.array) -> float:
    """Returns difference between mean variance over region of interest and over uninteresting regions"""
    var_interesting = np.diag(estimated_map_covariance_matrix)[adaptive_msk]
    var_uninteresting = np.diag(estimated_map_covariance_matrix)[~adaptive_msk]
    return (np.mean(var_uninteresting) - np.mean(var_interesting)) / np.mean(var_uninteresting)


def weighted_root_mean_squared_error(ground_truth_map: np.array, estimated_map: np.array) -> float:
    """Returns weighted root-mean-squared-error (WRMSE) between ground truth and estimated map"""
    ground_truth_value_range = np.max(ground_truth_map) - np.min(ground_truth_map)
    weights = (ground_truth_map - np.min(estimated_map)) / ground_truth_value_range
    weights = weights / np.sum(weights)
    return np.sqrt(np.mean(weights * np.square(ground_truth_map - estimated_map)))


def mean_log_loss(
    ground_truth_map: np.array, estimated_map: np.array, estimated_map_covariance_matrix: np.array
) -> float:
    """Returns mean log-loss between ground truth and estimated map scaled by map uncertainties"""
    P_diag = np.diag(estimated_map_covariance_matrix).reshape(estimated_map.shape)
    log_loss = 0.5 * np.log(2 * np.pi * P_diag) + np.square(ground_truth_map - estimated_map) / 2 * P_diag
    return np.mean(log_loss)


def weighted_mean_log_loss(
    ground_truth_map: np.array, estimated_map: np.array, estimated_map_covariance_matrix: np.array
) -> float:
    """Returns mean log-loss scaled by map uncertainties scaled by map uncertainties"""
    ground_truth_value_range = np.max(ground_truth_map) - np.min(ground_truth_map)
    weights = (ground_truth_map - np.min(estimated_map)) / ground_truth_value_range
    weights = weights / np.sum(weights)

    P_diag = np.diag(estimated_map_covariance_matrix).reshape(estimated_map.shape)
    log_loss = 0.5 * np.log(2 * np.pi * P_diag) + np.square(ground_truth_map - estimated_map) / 2 * P_diag
    return np.mean(weights * log_loss)
