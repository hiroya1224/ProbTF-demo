"""ROS-independent kinematic observations consumed by IMU estimators."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _frame_id(value):
    result = str(value).strip().lstrip("/")
    if not result:
        raise ValueError("frame_id must not be empty.")
    return result


def _vector(values, name):
    array = np.asarray(values, dtype=float)
    if array.size != 3:
        raise ValueError("{} must contain 3 values.".format(name))
    array = array.reshape(3)
    if not np.isfinite(array).all():
        raise ValueError("{} must contain only finite values.".format(name))
    return array


def _covariance(values, name):
    matrix = np.asarray(values, dtype=float)
    if matrix.size != 9:
        raise ValueError("{} must contain 9 values.".format(name))
    matrix = matrix.reshape(3, 3)
    if not np.isfinite(matrix).all():
        raise ValueError("{} must contain only finite values.".format(name))
    # Same allclose inequality (default rtol=1e-5), specialized to the already
    # finite 3x3 inputs. Avoid repeated generic broadcasting/NaN machinery in
    # every raw sample and every SG value/derivative covariance.
    if not (np.abs(matrix-matrix.T) <= 1e-10 + 1e-5*np.abs(matrix.T)).all():
        raise ValueError("{} must be symmetric.".format(name))
    matrix = 0.5 * (matrix + matrix.T)
    if np.min(np.linalg.eigvalsh(matrix)) < -1e-10:
        raise ValueError("{} must be positive semidefinite.".format(name))
    return matrix


@dataclass
class ImuKinematics:
    """Locally fitted IMU kinematics used by relative-pose producers."""

    frame_id: str
    angular_velocity: np.ndarray
    angular_acceleration: np.ndarray
    specific_force: np.ndarray
    angular_velocity_covariance: np.ndarray
    angular_acceleration_covariance: np.ndarray
    specific_force_covariance: np.ndarray
    stamp: Optional[float] = None

    def __post_init__(self):
        self.frame_id = _frame_id(self.frame_id)
        self.angular_velocity = _vector(self.angular_velocity, "angular_velocity")
        self.angular_acceleration = _vector(self.angular_acceleration, "angular_acceleration")
        self.specific_force = _vector(self.specific_force, "specific_force")
        self.angular_velocity_covariance = _covariance(
            self.angular_velocity_covariance,
            "angular_velocity_covariance",
        )
        self.angular_acceleration_covariance = _covariance(
            self.angular_acceleration_covariance,
            "angular_acceleration_covariance",
        )
        self.specific_force_covariance = _covariance(
            self.specific_force_covariance,
            "specific_force_covariance",
        )
        if self.stamp is not None:
            self.stamp = float(self.stamp)
            if not np.isfinite(self.stamp) or self.stamp < 0.0:
                raise ValueError("stamp must be a finite non-negative time in seconds.")
