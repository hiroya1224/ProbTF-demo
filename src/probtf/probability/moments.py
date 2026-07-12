"""Optional exact first/second-moment summaries for Prob-TF point action."""

from dataclasses import dataclass

import numpy as np

from probtf.bingham import (
    bingham_fourth_moment,
    bingham_second_moment,
    rotation_first_moment,
    rotation_vector_second_moment,
)
from probtf.distributions import BinghamOrientation, OrientationKind, TransformComponent
from probtf.distributions.validation import immutable_array, immutable_symmetric_matrix
from probtf.geometry import quat_to_rotmat, rotation_action_matrix, rotation_vector_from_quaternion


@dataclass(frozen=True)
class RotationVectorMoments:
    mean_rotation: np.ndarray
    mean_vector: np.ndarray
    second_vector: np.ndarray

    def __post_init__(self):
        object.__setattr__(
            self,
            "mean_rotation",
            immutable_array(self.mean_rotation, (3, 3), "mean_rotation"),
        )
        object.__setattr__(self, "mean_vector", immutable_array(self.mean_vector, (9,), "mean_vector"))
        object.__setattr__(
            self,
            "second_vector",
            immutable_symmetric_matrix(self.second_vector, 9, "second_vector", positive_semidefinite=True),
        )


@dataclass(frozen=True)
class PointMomentSummary:
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "mean", immutable_array(self.mean, (3,), "mean"))
        covariance = np.asarray(self.covariance, dtype=float)
        covariance = 0.5 * (covariance + covariance.T)
        object.__setattr__(
            self,
            "covariance",
            immutable_symmetric_matrix(covariance, 3, "covariance", positive_semidefinite=True),
        )


def _uniform_quaternion_fourth_moment():
    moment = np.zeros((4, 4, 4, 4), dtype=float)
    identity = np.eye(4, dtype=float)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for ell in range(4):
                    moment[i, j, k, ell] = (
                        identity[i, j] * identity[k, ell]
                        + identity[i, k] * identity[j, ell]
                        + identity[i, ell] * identity[j, k]
                    ) / 24.0
    return moment


def rotation_vector_moments(orientation, integration_steps=120):
    if not isinstance(orientation, BinghamOrientation):
        raise TypeError("orientation must be BinghamOrientation.")
    if orientation.kind is OrientationKind.DIRAC:
        rotation = quat_to_rotmat(orientation.mode_wxyz)
        vector = rotation.reshape(9, order="F")
        return RotationVectorMoments(rotation, vector, np.outer(vector, vector))
    if orientation.kind is OrientationKind.UNIFORM:
        quaternion_second = 0.25 * np.eye(4, dtype=float)
        quaternion_fourth = _uniform_quaternion_fourth_moment()
    else:
        parameter = orientation.backend_parameter_matrix()
        quaternion_second = bingham_second_moment(parameter, integration_steps)
        quaternion_fourth = bingham_fourth_moment(parameter, integration_steps)
    mean_rotation = rotation_first_moment(quaternion_second)
    mean_vector = mean_rotation.reshape(9, order="F")
    second_vector = rotation_vector_second_moment(quaternion_fourth)
    return RotationVectorMoments(mean_rotation, mean_vector, second_vector)


def expected_rotated_covariance(rotation_moments, covariance):
    sigma = np.asarray(covariance, dtype=float)
    if sigma.shape != (3, 3) or not np.all(np.isfinite(sigma)):
        raise ValueError("covariance must be a finite 3x3 matrix.")
    output = np.zeros((3, 3), dtype=float)
    second = rotation_moments.second_vector
    for row in range(3):
        for column in range(3):
            for input_row in range(3):
                for input_column in range(3):
                    output[row, column] += (
                        sigma[input_row, input_column]
                        * second[row + 3 * input_row, column + 3 * input_column]
                    )
    return 0.5 * (output + output.T)


def forward_component_point_moments(component, input_moments, integration_steps=120):
    """Compute exact first two point moments for one forward component."""

    if not isinstance(component, TransformComponent):
        raise TypeError("component must be TransformComponent.")
    if not isinstance(input_moments, PointMomentSummary):
        raise TypeError("input_moments must be PointMomentSummary.")
    rotation_moments = rotation_vector_moments(component.orientation, integration_steps)
    coupling = component.translation.rotation_coupling
    action = rotation_action_matrix(input_moments.mean)
    linear = action + coupling
    reference_vector = rotation_vector_from_quaternion(
        component.orientation.reference_quaternion_wxyz
    )
    offset = component.translation.mean_at_reference - coupling @ reference_vector
    output_mean = offset + linear @ rotation_moments.mean_vector
    centered_rotation = (
        rotation_moments.second_vector
        - np.outer(rotation_moments.mean_vector, rotation_moments.mean_vector)
    )
    output_covariance = (
        linear @ centered_rotation @ linear.T
        + expected_rotated_covariance(rotation_moments, input_moments.covariance)
        + component.translation.residual_covariance
    )
    return PointMomentSummary(output_mean, output_covariance)


def mixture_point_moments(weighted_summaries):
    entries = tuple(weighted_summaries)
    if not entries:
        raise ValueError("weighted_summaries must not be empty.")
    total = float(sum(weight for weight, _ in entries))
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("weights must sum to one.")
    mean = sum(weight * summary.mean for weight, summary in entries)
    covariance = np.zeros((3, 3), dtype=float)
    for weight, summary in entries:
        difference = summary.mean - mean
        covariance += weight * (summary.covariance + np.outer(difference, difference))
    return PointMomentSummary(mean, covariance)
