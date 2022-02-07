import itertools
import logging
from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from mapping.grid_maps import GridMap
from sensors import Sensor

logger = logging.getLogger(__name__)


class Mapping:
    def __init__(self, grid_map: GridMap, sensor: Sensor, shuffle_prior_cov: bool = False):
        self.grid_map = grid_map
        self.sensor = sensor
        self.shuffle_prior_cov = shuffle_prior_cov
        self.init_priors()

    @property
    def signal_variance(self) -> float:
        """Returns signal variance scaling the Matern kernel"""
        if "mapping" not in self.grid_map.params.keys():
            logger.error(f"Cannot find mapping specification in config file!")
            raise ValueError

        if "signal_variance" not in self.grid_map.params["mapping"].keys():
            logger.error(f"Cannot find mapping's 'signal_variance' specification in config file!")
            raise ValueError

        return self.grid_map.params["mapping"]["signal_variance"]

    @property
    def noise_variance(self) -> float:
        """Returns noise variance added to GP's training input's covariance matrix"""
        if "mapping" not in self.grid_map.params.keys():
            logger.error(f"Cannot find mapping specification in config file!")
            raise ValueError

        if "noise_variance" not in self.grid_map.params["mapping"].keys():
            logger.error(f"Cannot find mapping's 'noise_variance' specification in config file!")
            raise ValueError

        return self.grid_map.params["mapping"]["noise_variance"]

    @property
    def length_scale(self) -> float:
        """Returns length scale scaling the input data points' distance to each other"""
        if "mapping" not in self.grid_map.params.keys():
            logger.error(f"Cannot find mapping specification in config file!")
            raise ValueError

        if "length_scale" not in self.grid_map.params["mapping"].keys():
            logger.error(f"Cannot find mapping's 'length_scale' specification in config file!")
            raise ValueError

        return self.grid_map.params["mapping"]["length_scale"]

    @property
    def nu(self) -> float:
        """Returns nu parameter defining Matern kernel's smoothness"""
        if "mapping" not in self.grid_map.params.keys():
            logger.error(f"Cannot find mapping specification in config file!")
            raise ValueError

        if "nu" not in self.grid_map.params["mapping"].keys():
            logger.error(f"Cannot find mapping's 'nu' specification in config file!")
            raise ValueError

        return self.grid_map.params["mapping"]["nu"]

    @property
    def fit_gaussian_process(self):
        """Flag indicating if GP should be used to initialize a prior mean and covariance for the mapping routine"""
        if "mapping" not in self.grid_map.params.keys():
            logger.error(f"Cannot find mapping specification in config file!")
            raise ValueError

        if "fit_gaussian_process" not in self.grid_map.params["mapping"].keys():
            logger.error(f"Cannot find mapping's 'fit_gaussian_process' specification in config file!")
            raise ValueError

        return self.grid_map.params["mapping"]["fit_gaussian_process"]

    @property
    def prior_cov_mean(self):
        """Returns mean of random normally initialized prior covariance matrix"""
        if "mapping" not in self.grid_map.params.keys():
            logger.error(f"Cannot find mapping specification in config file!")
            raise ValueError

        if "prior_cov_mean" not in self.grid_map.params["mapping"].keys():
            logger.error(f"Cannot find mapping's 'prior_cov_mean' specification in config file!")
            raise ValueError

        return self.grid_map.params["mapping"]["prior_cov_mean"]

    @property
    def prior_cov_std(self):
        """Returns standard deviation of random normally initialized prior covariance matrix"""
        if "mapping" not in self.grid_map.params.keys():
            logger.error(f"Cannot find mapping specification in config file!")
            raise ValueError

        if "prior_cov_std" not in self.grid_map.params["mapping"].keys():
            logger.error(f"Cannot find mapping's 'prior_cov_std' specification in config file!")
            raise ValueError

        return self.grid_map.params["mapping"]["prior_cov_std"]

    def update_grid_map(
        self,
        measurement_position: np.array,
        measurement_data: np.array = None,
        cov_only: bool = False,
        predict_only: bool = False,
        current_cov_matrix: np.array = None,
    ):
        resolution_factor = self.sensor.get_resolution_factor(measurement_position)
        measurement_field_of_view = self.sensor.project_field_of_view(measurement_position)

        xl, xr, yu, yd = measurement_field_of_view
        num_measurements = int(np.ceil((xr - xl + 1) / resolution_factor) * np.ceil((yd - yu + 1) / resolution_factor))

        R = self.sensor.sensor_model.measurement_variance_matrix(
            measurement_position, num_measurements, resolution_factor
        )
        H = self.sensor.sensor_model.measurement_model_matrix(
            self.grid_map, measurement_field_of_view, num_measurements, resolution_factor
        )

        if current_cov_matrix is None:
            current_cov_matrix = self.grid_map.cov_matrix

        x, P = self.kalman_filter_update(
            current_cov_matrix, H, R, grid_mean=self.grid_map.mean, observation=measurement_data, cov_only=cov_only,
        )
        if predict_only:
            if cov_only:
                return None, P
            else:
                return x.reshape(self.grid_map.mean.shape), P

        self.grid_map.mean = x.reshape(self.grid_map.mean.shape)
        self.grid_map.cov_matrix = P

    @staticmethod
    def kalman_filter_update(
        P: np.array,
        H: np.array,
        R: np.array,
        grid_mean: np.array = None,
        observation: np.array = None,
        cov_only: bool = False,
    ) -> Tuple[Optional[np.array], np.array]:
        """
        Computes kalman filter update for grid map mean and covariance based on sensor model H and measurement noise R.
        Args:
            P (np.array): current grid map covariance matrix
            H (np.array): altitude-dependent measurement model matrix
            R (np.array): altitude-dependent measurement noise matrix
            grid_mean (np.array): current grid map mean
            observation (np.array): either simulated or real sensor measurement
            cov_only (bool): if True, compute only covariance matrix update

        Returns:
            (np.array): Kalman filter update of grid map's mean flattened to 1D
            (np.array): Kalman filter update of grid map cells' covariance matrix
        """
        msk = ~np.all(H == 0, axis=0)
        H_pruned = H[:, msk]
        P_pruned = P[msk][:, msk]
        PHt_pruned = np.dot(P_pruned, H_pruned.transpose())
        S = np.dot(H_pruned, PHt_pruned) + R
        S = 0.5 * np.add(S, np.transpose(S))  # ensure symmetry of S required for Cholesky decomposition
        try:
            L = np.transpose(np.linalg.cholesky(S))  # numpy returns lower instead of upper triangular, S = Sc^T*Sc
            L_inv = np.linalg.inv(L)

            Wc = np.dot(P, np.dot(H.transpose(), L_inv))
            W = np.dot(Wc, np.transpose(L_inv))  # Kalman gain K, KHP = WHP = WcWc^T
            P = P - np.dot(Wc, np.transpose(Wc))

            if not cov_only:
                x = grid_mean.flatten(order="C")
                z = observation.flatten(order="C")
                v = z - np.dot(H, x)
                x = x + np.dot(W, v)
                return x, P

            return None, P
        except np.linalg.LinAlgError as e:
            logger.error(f"Cholesky decomposition failed with the following error:\n {e}")
            logger.info("Fallback to classical matrix inversion")

            PHt = np.dot(P, H.transpose())
            S_inv = np.linalg.inv(S)
            P = P - np.dot(PHt, np.dot(S_inv, np.dot(H, P)))

            if not cov_only:
                x = grid_mean.flatten(order="C")
                z = observation.flatten(order="C")
                v = z - np.dot(H, x)
                x = x + np.dot(np.dot(PHt, S_inv), v)
                return x, P

            return None, P

    def init_priors(self):
        """Sets grid map's prior mean and covariance matrix for bayesian filtering"""
        if not self.fit_gaussian_process:
            prior_cov_mean, prior_cov_std = self.prior_cov_mean, self.prior_cov_std
            if self.shuffle_prior_cov:
                prior_cov_mean = np.random.uniform(low=0.1, high=self.prior_cov_mean)
                prior_cov_std = prior_cov_mean

            self.grid_map.mean = 0.5 * np.ones((self.grid_map.y_dim, self.grid_map.x_dim))
            self.grid_map.cov_matrix = np.random.normal(
                prior_cov_mean, prior_cov_std, (self.grid_map.num_grid_cells, self.grid_map.num_grid_cells)
            )
            # A*A^T ensures covariance matrix to positive semidefinite
            # normalized by Frobenius norm to approx. preserve covariance magnitudes of original normal random matrix
            self.grid_map.cov_matrix = (1 / np.linalg.norm(self.grid_map.cov_matrix, ord="fro")) * np.dot(
                self.grid_map.cov_matrix, np.transpose(self.grid_map.cov_matrix)
            )
            return

        signal_variance = self.signal_variance
        length_scale = self.length_scale
        if self.shuffle_prior_cov:
            signal_variance = np.random.uniform(low=0.8 * self.signal_variance, high=1.2 * self.signal_variance)
            length_scale = np.random.uniform(low=0.8 * self.length_scale, high=1.2 * self.length_scale)

        matern_kernel = ConstantKernel(constant_value=signal_variance, constant_value_bounds="fixed") * Matern(
            length_scale=length_scale, nu=self.nu, length_scale_bounds="fixed"
        )
        gpr = GaussianProcessRegressor(kernel=matern_kernel, n_restarts_optimizer=0, alpha=self.noise_variance)

        train_map = self.sensor.sensor_simulation.ground_truth_map
        cell_offset = np.array([0.5 * self.grid_map.resolution, 0.5 * self.grid_map.resolution], dtype=np.float64)
        X_test = (
            np.array(
                [x for x in itertools.product(np.arange(train_map.shape[0]), np.arange(train_map.shape[1]))],
                dtype=np.float64,
            )
            * self.grid_map.resolution
            + cell_offset
        )

        _, y_cov = gpr.predict(X_test, return_cov=True)
        y_mean = 0.5 * np.ones(self.grid_map.num_grid_cells)
        self.grid_map.mean = y_mean.reshape((self.grid_map.y_dim, self.grid_map.x_dim))
        self.grid_map.cov_matrix = y_cov
