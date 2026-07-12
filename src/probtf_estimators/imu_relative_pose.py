"""Relative-pose production from a pair of rigidly mounted IMUs.

The orientation posterior is represented directly as a quaternion Bingham
parameter. Translation is estimated from

    R f_child - f_parent = ([omega]x^2 + [alpha]x) r,

where ``r`` is the child IMU origin expressed in the parent IMU frame. The
module intentionally contains no ROS imports; transport adapters live under
``ros/core``.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from probtf.bingham import (
    bingham_fourth_moment,
    bingham_mode,
    bingham_second_moment,
    canonical_bingham_parameter,
)
from probtf.geometry import quat_left_matrix, quat_right_matrix, quat_to_rotmat
from probtf.models import (
    BinghamRotation,
    GaussianPosition,
    ImuKinematics,
    ProbabilisticTransform,
)


def skew(vector):
    x_value, y_value, z_value = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -z_value, y_value],
            [z_value, 0.0, -x_value],
            [-y_value, x_value, 0.0],
        ],
        dtype=float,
    )


def rigid_point_acceleration_operator(angular_velocity, angular_acceleration):
    omega_cross = skew(angular_velocity)
    return omega_cross @ omega_cross + skew(angular_acceleration)


def vector_alignment_bingham(
    vector_before,
    vector_after,
    covariance_before=None,
    covariance_after=None,
    variance_floor=1e-8,
):
    """Return a Bingham likelihood for ``R(q) before == after``.

    The quaternion basis is ``[w, x, y, z]``. The covariance is reduced to an
    isotropic residual variance because the exact transformed covariance
    depends on the unknown rotation; sequential updates recover anisotropy
    through the observed vector directions.
    """

    before = np.asarray(vector_before, dtype=float).reshape(3)
    after = np.asarray(vector_after, dtype=float).reshape(3)
    if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        raise ValueError("Alignment vectors must contain only finite values.")
    if np.linalg.norm(before) < 1e-10 or np.linalg.norm(after) < 1e-10:
        raise ValueError("Alignment vectors must be non-zero.")

    residual_variance = float(variance_floor)
    for covariance in (covariance_before, covariance_after):
        if covariance is None:
            continue
        matrix = np.asarray(covariance, dtype=float).reshape(3, 3)
        if not np.allclose(matrix, matrix.T, atol=1e-10):
            raise ValueError("Alignment covariance must be symmetric.")
        if np.min(np.linalg.eigvalsh(matrix)) < -1e-10:
            raise ValueError("Alignment covariance must be positive semidefinite.")
        residual_variance += max(float(np.trace(matrix)) / 3.0, 0.0)

    pure_before = np.concatenate(([0.0], before))
    pure_after = np.concatenate(([0.0], after))
    residual_matrix = quat_left_matrix(
        pure_after,
        normalize_input=False,
    ) - quat_right_matrix(pure_before, normalize_input=False)
    parameter = -0.5 * (residual_matrix.T @ residual_matrix) / residual_variance
    return canonical_bingham_parameter(parameter)


class RecursiveGaussianLeastSquares:
    """Information-form least squares with exponential forgetting."""

    def __init__(self, dimension=3, prior_variance=1e6):
        self.dimension = int(dimension)
        if self.dimension <= 0:
            raise ValueError("dimension must be positive.")
        self.prior_variance = float(prior_variance)
        if not np.isfinite(self.prior_variance) or self.prior_variance <= 0.0:
            raise ValueError("prior_variance must be positive and finite.")
        self.reset()

    def reset(self):
        self.information = np.eye(self.dimension, dtype=float) / self.prior_variance
        self.information_vector = np.zeros(self.dimension, dtype=float)
        self.update_count = 0

    def update(self, coefficient, observation, covariance, forgetting_factor=1.0):
        coefficient = np.asarray(coefficient, dtype=float).reshape(-1, self.dimension)
        observation = np.asarray(observation, dtype=float).reshape(coefficient.shape[0])
        covariance = np.asarray(covariance, dtype=float).reshape(
            coefficient.shape[0],
            coefficient.shape[0],
        )
        if not 0.0 < float(forgetting_factor) <= 1.0:
            raise ValueError("forgetting_factor must be in (0, 1].")
        covariance = 0.5 * (covariance + covariance.T)
        if np.min(np.linalg.eigvalsh(covariance)) < -1e-10:
            raise ValueError("covariance must be positive semidefinite.")
        precision = np.linalg.pinv(covariance + 1e-12 * np.eye(covariance.shape[0]))

        self.information *= float(forgetting_factor)
        self.information_vector *= float(forgetting_factor)
        self.information += coefficient.T @ precision @ coefficient
        self.information_vector += coefficient.T @ precision @ observation
        self.information = 0.5 * (self.information + self.information.T)
        self.update_count += 1

    @property
    def covariance(self):
        return np.linalg.pinv(self.information)

    @property
    def mean(self):
        return self.covariance @ self.information_vector


@dataclass
class JointGeometry:
    parent_to_joint: np.ndarray
    child_to_joint: np.ndarray
    parent_covariance: np.ndarray
    child_covariance: np.ndarray

    def __post_init__(self):
        self.parent_to_joint = np.asarray(self.parent_to_joint, dtype=float).reshape(3)
        self.child_to_joint = np.asarray(self.child_to_joint, dtype=float).reshape(3)
        self.parent_covariance = np.asarray(self.parent_covariance, dtype=float).reshape(3, 3)
        self.child_covariance = np.asarray(self.child_covariance, dtype=float).reshape(3, 3)
        for name, value in (
            ("parent_covariance", self.parent_covariance),
            ("child_covariance", self.child_covariance),
        ):
            if not np.allclose(value, value.T, atol=1e-10):
                raise ValueError("{} must be symmetric.".format(name))
            if np.min(np.linalg.eigvalsh(value)) < -1e-10:
                raise ValueError("{} must be positive semidefinite.".format(name))


class ImuRelativePoseEstimator:
    """Online producer of a Gaussian/Bingham relative transform summary."""

    def __init__(
        self,
        parent_frame_id,
        child_frame_id,
        rotation_forgetting_factor=1.0,
        position_forgetting_factor=1.0,
        prior_position_variance=1e6,
        integration_steps=120,
        source_id="imu_relative_pose",
    ):
        self.parent_frame_id = str(parent_frame_id).lstrip("/")
        self.child_frame_id = str(child_frame_id).lstrip("/")
        if not self.parent_frame_id or not self.child_frame_id:
            raise ValueError("IMU frame identifiers must not be empty.")
        if self.parent_frame_id == self.child_frame_id:
            raise ValueError("Parent and child IMU frames must differ.")
        self.rotation_forgetting_factor = self._forgetting_factor(
            rotation_forgetting_factor,
            "rotation_forgetting_factor",
        )
        self.position_forgetting_factor = self._forgetting_factor(
            position_forgetting_factor,
            "position_forgetting_factor",
        )
        self.integration_steps = int(integration_steps)
        if self.integration_steps < 20:
            raise ValueError("integration_steps must be at least 20.")
        self.source_id = str(source_id).strip()
        if not self.source_id:
            raise ValueError("source_id must not be empty.")
        self.position_estimator = RecursiveGaussianLeastSquares(
            dimension=3,
            prior_variance=prior_position_variance,
        )
        self.joint_geometry = None  # type: Optional[JointGeometry]
        self.reset()

    @staticmethod
    def _forgetting_factor(value, name):
        value = float(value)
        if not 0.0 < value <= 1.0:
            raise ValueError("{} must be in (0, 1].".format(name))
        return value

    def register_joint_geometry(
        self,
        parent_to_joint,
        child_to_joint,
        parent_covariance=None,
        child_covariance=None,
    ):
        if (parent_to_joint is None) != (child_to_joint is None):
            raise ValueError("Both joint offsets must be provided together.")
        if parent_to_joint is None:
            self.joint_geometry = None
            return
        default_covariance = np.eye(3, dtype=float) * 4e-6
        self.joint_geometry = JointGeometry(
            parent_to_joint,
            child_to_joint,
            default_covariance if parent_covariance is None else parent_covariance,
            default_covariance if child_covariance is None else child_covariance,
        )

    def reset(self):
        self.rotation_parameter = np.zeros((4, 4), dtype=float)
        self.position_estimator.reset()
        self.last_stamp = None
        self.rotation_update_count = 0

    def _check_observations(self, parent, child):
        if not isinstance(parent, ImuKinematics) or not isinstance(child, ImuKinematics):
            raise TypeError("parent and child observations must be ImuKinematics.")
        if parent.frame_id != self.parent_frame_id:
            raise ValueError("Unexpected parent IMU frame: {}".format(parent.frame_id))
        if child.frame_id != self.child_frame_id:
            raise ValueError("Unexpected child IMU frame: {}".format(child.frame_id))
        if parent.stamp is not None and child.stamp is not None:
            if abs(parent.stamp - child.stamp) > 0.1:
                raise ValueError("IMU observations are not time synchronized.")
            stamp = max(parent.stamp, child.stamp)
            if self.last_stamp is not None and stamp < self.last_stamp:
                raise ValueError("IMU observations must be time ordered.")
            self.last_stamp = stamp
        elif parent.stamp is not None or child.stamp is not None:
            raise ValueError("Both IMU observations must carry a timestamp or neither may.")

    def _add_rotation_likelihood(self, parameter):
        self.rotation_parameter = canonical_bingham_parameter(
            self.rotation_forgetting_factor * self.rotation_parameter + parameter
        )
        self.rotation_update_count += 1

    def _representative_rotation(self, parameter=None):
        if parameter is None:
            parameter = self.rotation_parameter
        return quat_to_rotmat(bingham_mode(parameter))

    def _rotation_is_observable(self):
        eigenvalues = np.linalg.eigvalsh(self.rotation_parameter)
        return self.rotation_update_count >= 2 and float(eigenvalues[-1] - eigenvalues[-2]) > 1e-6

    def _update_rotation_from_gyros(self, parent, child):
        if np.linalg.norm(parent.angular_velocity) > 1e-8 and np.linalg.norm(child.angular_velocity) > 1e-8:
            self._add_rotation_likelihood(
                vector_alignment_bingham(
                    child.angular_velocity,
                    parent.angular_velocity,
                    child.angular_velocity_covariance,
                    parent.angular_velocity_covariance,
                )
            )
        if (
            np.linalg.norm(parent.angular_acceleration) > 1e-8
            and np.linalg.norm(child.angular_acceleration) > 1e-8
        ):
            self._add_rotation_likelihood(
                vector_alignment_bingham(
                    child.angular_acceleration,
                    parent.angular_acceleration,
                    child.angular_acceleration_covariance,
                    parent.angular_acceleration_covariance,
                )
            )

    def _update_position(self, parent, child, expected_rotation):
        omega_operator = rigid_point_acceleration_operator(
            parent.angular_velocity,
            parent.angular_acceleration,
        )
        transformed_child_force = expected_rotation @ child.specific_force
        force_delta = transformed_child_force - parent.specific_force
        force_covariance = (
            parent.specific_force_covariance
            + expected_rotation @ child.specific_force_covariance @ expected_rotation.T
        )
        if np.linalg.norm(omega_operator) > 1e-10:
            self.position_estimator.update(
                omega_operator,
                force_delta,
                force_covariance,
                forgetting_factor=self.position_forgetting_factor,
            )

    def _update_registered_joint(self, parent, child, expected_rotation):
        geometry = self.joint_geometry
        parent_operator = rigid_point_acceleration_operator(
            parent.angular_velocity,
            parent.angular_acceleration,
        )
        child_operator = rigid_point_acceleration_operator(
            child.angular_velocity,
            child.angular_acceleration,
        )
        force_at_joint_parent = parent.specific_force + parent_operator @ geometry.parent_to_joint
        force_at_joint_child = child.specific_force + child_operator @ geometry.child_to_joint
        if np.linalg.norm(force_at_joint_parent) > 1e-8 and np.linalg.norm(force_at_joint_child) > 1e-8:
            self._add_rotation_likelihood(
                vector_alignment_bingham(
                    force_at_joint_child,
                    force_at_joint_parent,
                    child.specific_force_covariance,
                    parent.specific_force_covariance,
                )
            )
            expected_rotation = self._representative_rotation()
        position_mean = geometry.parent_to_joint - expected_rotation @ geometry.child_to_joint
        position_covariance = (
            geometry.parent_covariance
            + expected_rotation @ geometry.child_covariance @ expected_rotation.T
        )
        return GaussianPosition(position_mean, position_covariance)

    def update(self, parent, child):
        self._check_observations(parent, child)
        self._update_rotation_from_gyros(parent, child)

        expected_rotation = self._representative_rotation()
        if self.joint_geometry is None:
            if self._rotation_is_observable():
                self._update_position(parent, child, expected_rotation)
        else:
            self._update_registered_joint(parent, child, expected_rotation)
        return self.result()

    def result(self):
        parameter = canonical_bingham_parameter(self.rotation_parameter)
        second_moment = bingham_second_moment(
            parameter,
            integration_steps=self.integration_steps,
        )
        fourth_moment = bingham_fourth_moment(
            parameter,
            integration_steps=self.integration_steps,
        )
        orientation = BinghamRotation(
            parameter=parameter,
            mode_wxyz=bingham_mode(parameter),
            second_moment=second_moment,
            fourth_moment=fourth_moment,
        )
        expected_rotation = self._representative_rotation(parameter)
        if self.joint_geometry is None:
            position = GaussianPosition(
                self.position_estimator.mean,
                self.position_estimator.covariance,
            )
        else:
            geometry = self.joint_geometry
            position = GaussianPosition(
                geometry.parent_to_joint - expected_rotation @ geometry.child_to_joint,
                geometry.parent_covariance
                + expected_rotation @ geometry.child_covariance @ expected_rotation.T,
            )
        return ProbabilisticTransform(
            parent_frame_id=self.parent_frame_id,
            child_frame_id=self.child_frame_id,
            position=position,
            orientation=orientation,
            stamp=self.last_stamp,
            source_id=self.source_id,
            evidence_source_ids=("angular_velocity", "specific_force"),
            approximation_type="extended_least_squares_bingham_gaussian",
            closure_approximation=False,
        )
