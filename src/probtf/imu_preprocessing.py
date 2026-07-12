"""Local polynomial preprocessing for asynchronous raw IMU samples."""

from collections import deque

import numpy as np

from probtf.models import ImuKinematics


def _covariance(values, name):
    matrix = np.asarray(values, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("{} must be finite and symmetric.".format(name))
    if np.min(np.linalg.eigvalsh(matrix)) < -1e-10:
        raise ValueError("{} must be positive semidefinite.".format(name))
    return 0.5 * (matrix + matrix.T)


class ImuKinematicsPreprocessor:
    """Fit a polynomial window and emit synchronized kinematic derivatives."""

    def __init__(self, frame_id, window_size=9, polynomial_order=2, minimum_samples=None):
        self.frame_id = str(frame_id).lstrip("/")
        if not self.frame_id:
            raise ValueError("frame_id must not be empty.")
        self.window_size = int(window_size)
        self.polynomial_order = int(polynomial_order)
        if self.window_size < 3:
            raise ValueError("window_size must be at least three.")
        if self.polynomial_order < 1 or self.polynomial_order >= self.window_size:
            raise ValueError("polynomial_order must be in [1, window_size).")
        self.minimum_samples = int(
            self.polynomial_order + 2 if minimum_samples is None else minimum_samples
        )
        if not self.polynomial_order + 1 <= self.minimum_samples <= self.window_size:
            raise ValueError(
                "minimum_samples must exceed polynomial_order and not exceed window_size."
            )
        self.samples = deque(maxlen=self.window_size)

    def reset(self):
        self.samples.clear()

    def update(
        self,
        stamp,
        angular_velocity,
        specific_force,
        angular_velocity_covariance,
        specific_force_covariance,
    ):
        stamp = float(stamp)
        angular_velocity = np.asarray(angular_velocity, dtype=float).reshape(3)
        specific_force = np.asarray(specific_force, dtype=float).reshape(3)
        if not np.isfinite(stamp) or stamp < 0.0:
            raise ValueError("stamp must be finite and non-negative.")
        if not np.all(np.isfinite(angular_velocity)) or not np.all(np.isfinite(specific_force)):
            raise ValueError("IMU vectors must contain only finite values.")
        gyro_covariance = _covariance(
            angular_velocity_covariance,
            "angular_velocity_covariance",
        )
        force_covariance = _covariance(
            specific_force_covariance,
            "specific_force_covariance",
        )
        if self.samples and stamp <= self.samples[-1][0]:
            raise ValueError("IMU sample timestamps must be strictly increasing.")
        self.samples.append(
            (
                stamp,
                angular_velocity.copy(),
                specific_force.copy(),
                gyro_covariance,
                force_covariance,
            )
        )
        if len(self.samples) < self.minimum_samples:
            return None
        return self._fit()

    @staticmethod
    def _fit_components(times, values, order):
        design = np.vander(times, N=order + 1, increasing=True)
        coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
        fitted = design @ coefficients
        residual = values - fitted
        degrees_of_freedom = max(values.shape[0] - design.shape[1], 1)
        residual_covariance = residual.T @ residual / float(degrees_of_freedom)
        coefficient_covariance_scale = np.linalg.pinv(design.T @ design)
        return coefficients, residual_covariance, coefficient_covariance_scale

    def _fit(self):
        samples = tuple(self.samples)
        stamp = samples[-1][0]
        times = np.array([sample[0] - stamp for sample in samples], dtype=float)
        angular_velocities = np.stack([sample[1] for sample in samples])
        forces = np.stack([sample[2] for sample in samples])
        gyro_covariance_mean = np.mean(np.stack([sample[3] for sample in samples]), axis=0)
        force_covariance_mean = np.mean(np.stack([sample[4] for sample in samples]), axis=0)

        gyro_coefficients, gyro_residual, gyro_scale = self._fit_components(
            times,
            angular_velocities,
            self.polynomial_order,
        )
        force_coefficients, force_residual, _ = self._fit_components(
            times,
            forces,
            self.polynomial_order,
        )
        angular_velocity_covariance = gyro_covariance_mean + gyro_residual
        angular_acceleration_covariance = (
            gyro_covariance_mean + gyro_residual
        ) * max(float(gyro_scale[1, 1]), 1e-12)
        specific_force_covariance = force_covariance_mean + force_residual

        return ImuKinematics(
            frame_id=self.frame_id,
            angular_velocity=gyro_coefficients[0],
            angular_acceleration=gyro_coefficients[1],
            specific_force=force_coefficients[0],
            angular_velocity_covariance=angular_velocity_covariance,
            angular_acceleration_covariance=angular_acceleration_covariance,
            specific_force_covariance=specific_force_covariance,
            stamp=stamp,
        )
