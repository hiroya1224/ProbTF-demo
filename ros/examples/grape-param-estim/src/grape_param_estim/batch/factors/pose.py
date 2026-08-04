"""Asynchronous sensor-pose factors with candidate-dependent CoG geometry."""

from typing import Tuple

import numpy as np

from grape_param_estim.batch.factor import FactorEvaluation, JacobianBlock
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.geometry import (
    skew,
    so3_geodesic_interpolation_with_right_jacobians,
    so3_log,
    so3_right_jacobian_inverse,
)


_LOG_BRANCH_WARNING_DISTANCE = 1.0e-5


def _finite_vector3(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite three-vector".format(name))
    return result


def _proper_rotation(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite 3 by 3 matrix".format(name))
    if not np.allclose(result.T @ result, np.eye(3), rtol=0.0, atol=1.0e-9):
        raise ValueError("{} must be orthonormal".format(name))
    if not np.isclose(np.linalg.det(result), 1.0, rtol=0.0, atol=1.0e-9):
        raise ValueError("{} must have determinant one".format(name))
    return result


def _sqrt_information3(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite 3 by 3 matrix".format(name))
    if np.linalg.matrix_rank(result) != 3:
        raise ValueError("{} must have full rank".format(name))
    return result


def _interpolation_fraction(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError("interpolation_fraction must be finite and in [0, 1]")
    return result


def _knot_key(kind: VariableKind, bag_id: str, knot_index: int) -> VariableKey:
    return VariableKey(kind, bag_id=bag_id, knot_index=knot_index)


def evaluate_pose_observation_factors(
    bag_id: str,
    left_knot_index: int,
    interpolation_fraction: float,
    position_left: np.ndarray,
    position_right: np.ndarray,
    rotation_left: np.ndarray,
    rotation_right: np.ndarray,
    observed_sensor_position: np.ndarray,
    observed_sensor_rotation: np.ndarray,
    sensor_position_in_body: np.ndarray,
    sensor_to_body_rotation: np.ndarray,
    cog_offset_in_body: np.ndarray,
    cog_offset_chart_jacobian: np.ndarray,
    position_square_root_information: np.ndarray,
    orientation_square_root_information: np.ndarray,
) -> Tuple[FactorEvaluation, FactorEvaluation]:
    """Return separate position and orientation factors at one sensor time.

    The latent position is the vehicle CoG.  The measured sensor origin is
    predicted as ``p_WC + R_WB (r_BS - r_BC)``; consequently a raw FC/mocap
    pose is never mislabeled as a direct CoG observation.  Rotations use the
    convention ``R_WS = R_WB C_BS`` and right-tangent state perturbations.
    """

    alpha = _interpolation_fraction(interpolation_fraction)
    position0 = _finite_vector3(position_left, "position_left")
    position1 = _finite_vector3(position_right, "position_right")
    rotation0 = _proper_rotation(rotation_left, "rotation_left")
    rotation1 = _proper_rotation(rotation_right, "rotation_right")
    observed_position = _finite_vector3(
        observed_sensor_position, "observed_sensor_position"
    )
    observed_rotation = _proper_rotation(
        observed_sensor_rotation, "observed_sensor_rotation"
    )
    sensor_position = _finite_vector3(
        sensor_position_in_body, "sensor_position_in_body"
    )
    sensor_to_body = _proper_rotation(
        sensor_to_body_rotation, "sensor_to_body_rotation"
    )
    cog_offset = _finite_vector3(cog_offset_in_body, "cog_offset_in_body")
    cog_jacobian = np.asarray(cog_offset_chart_jacobian, dtype=float)
    if cog_jacobian.shape != (3, 18) or not np.all(np.isfinite(cog_jacobian)):
        raise ValueError(
            "cog_offset_chart_jacobian must be a finite 3 by 18 matrix"
        )
    position_whitening = _sqrt_information3(
        position_square_root_information,
        "position_square_root_information",
    )
    orientation_whitening = _sqrt_information3(
        orientation_square_root_information,
        "orientation_square_root_information",
    )

    interpolated_position = (1.0 - alpha) * position0 + alpha * position1
    (
        interpolated_rotation,
        rotation_left_interpolation_jacobian,
        rotation_right_interpolation_jacobian,
    ) = so3_geodesic_interpolation_with_right_jacobians(
        rotation0,
        rotation1,
        alpha,
    )
    cog_to_sensor = sensor_position - cog_offset
    predicted_position = (
        interpolated_position + interpolated_rotation @ cog_to_sensor
    )
    raw_position_residual = observed_position - predicted_position
    position_residual = position_whitening @ raw_position_residual

    position_rotation_jacobian = (
        interpolated_rotation @ skew(cog_to_sensor)
    )
    left_index = int(left_knot_index)
    right_index = left_index + 1
    position_blocks = (
        JacobianBlock(
            _knot_key(VariableKind.POSITION, bag_id, left_knot_index),
            -(1.0 - alpha) * position_whitening,
        ),
        JacobianBlock(
            _knot_key(VariableKind.POSITION, bag_id, right_index),
            -alpha * position_whitening,
        ),
        JacobianBlock(
            _knot_key(
                VariableKind.ORIENTATION_TANGENT,
                bag_id,
                left_knot_index,
            ),
            position_whitening
            @ position_rotation_jacobian
            @ rotation_left_interpolation_jacobian,
        ),
        JacobianBlock(
            _knot_key(
                VariableKind.ORIENTATION_TANGENT,
                bag_id,
                right_index,
            ),
            position_whitening
            @ position_rotation_jacobian
            @ rotation_right_interpolation_jacobian,
        ),
        JacobianBlock(
            VariableKey(VariableKind.STATIC_PARAMETERS),
            position_whitening
            @ interpolated_rotation
            @ cog_jacobian,
        ),
    )
    position_factor = FactorEvaluation(
        residual=position_residual,
        jacobian_blocks=position_blocks,
        squared_error=float(position_residual @ position_residual),
        active_set={},
    )

    predicted_sensor_rotation = interpolated_rotation @ sensor_to_body
    raw_orientation_residual = so3_log(
        observed_rotation.T @ predicted_sensor_rotation
    )
    residual_log_jacobian = so3_right_jacobian_inverse(
        raw_orientation_residual
    )
    body_tangent_to_sensor_tangent = sensor_to_body.T
    orientation_residual = orientation_whitening @ raw_orientation_residual
    orientation_blocks = (
        JacobianBlock(
            _knot_key(
                VariableKind.ORIENTATION_TANGENT,
                bag_id,
                left_knot_index,
            ),
            orientation_whitening
            @ residual_log_jacobian
            @ body_tangent_to_sensor_tangent
            @ rotation_left_interpolation_jacobian,
        ),
        JacobianBlock(
            _knot_key(
                VariableKind.ORIENTATION_TANGENT,
                bag_id,
                right_index,
            ),
            orientation_whitening
            @ residual_log_jacobian
            @ body_tangent_to_sensor_tangent
            @ rotation_right_interpolation_jacobian,
        ),
    )
    distance_to_branch = np.pi - float(
        np.linalg.norm(raw_orientation_residual)
    )
    orientation_factor = FactorEvaluation(
        residual=orientation_residual,
        jacobian_blocks=orientation_blocks,
        squared_error=float(orientation_residual @ orientation_residual),
        active_set={
            "rotation_log_near_pi": np.asarray(
                (distance_to_branch <= _LOG_BRANCH_WARNING_DISTANCE,),
                dtype=bool,
            )
        },
    )
    return position_factor, orientation_factor


__all__ = ["evaluate_pose_observation_factors"]
