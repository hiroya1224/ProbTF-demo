"""Asynchronous world-velocity factor for an offset body-fixed sensor."""

import numpy as np

from grape_param_estim.batch.factor import FactorEvaluation, JacobianBlock
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.geometry import (
    skew,
    so3_geodesic_interpolation_with_right_jacobians,
)


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


def _fraction(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError("interpolation_fraction must be finite and in [0, 1]")
    return result


def _whitening(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("square_root_information must be a finite 3 by 3 matrix")
    if np.linalg.matrix_rank(result) != 3:
        raise ValueError("square_root_information must have full rank")
    return result


def _key(kind: VariableKind, bag_id: str, knot_index: int) -> VariableKey:
    return VariableKey(kind, bag_id=bag_id, knot_index=knot_index)


def evaluate_world_sensor_velocity_factor(
    bag_id: str,
    left_knot_index: int,
    interpolation_fraction: float,
    velocity_left: np.ndarray,
    velocity_right: np.ndarray,
    rotation_left: np.ndarray,
    rotation_right: np.ndarray,
    angular_velocity_left: np.ndarray,
    angular_velocity_right: np.ndarray,
    observed_sensor_velocity_world: np.ndarray,
    sensor_position_in_body: np.ndarray,
    cog_offset_in_body: np.ndarray,
    cog_offset_chart_jacobian: np.ndarray,
    square_root_information: np.ndarray,
) -> FactorEvaluation:
    """Evaluate a world-frame sensor-origin linear velocity likelihood.

    The latent linear velocity is the CoG velocity.  For a body-fixed sensor
    at lever ``r_CS``, the prediction is ``v_WC + R_WB (omega_B x r_CS)``.
    This explicitly represents the audited mixed odometry contract instead of
    interpreting its linear velocity as a child-frame vector.
    """

    alpha = _fraction(interpolation_fraction)
    velocity0 = _finite_vector3(velocity_left, "velocity_left")
    velocity1 = _finite_vector3(velocity_right, "velocity_right")
    rotation0 = _proper_rotation(rotation_left, "rotation_left")
    rotation1 = _proper_rotation(rotation_right, "rotation_right")
    omega0 = _finite_vector3(
        angular_velocity_left, "angular_velocity_left"
    )
    omega1 = _finite_vector3(
        angular_velocity_right, "angular_velocity_right"
    )
    observation = _finite_vector3(
        observed_sensor_velocity_world,
        "observed_sensor_velocity_world",
    )
    sensor_position = _finite_vector3(
        sensor_position_in_body, "sensor_position_in_body"
    )
    cog_offset = _finite_vector3(cog_offset_in_body, "cog_offset_in_body")
    cog_jacobian = np.asarray(cog_offset_chart_jacobian, dtype=float)
    if cog_jacobian.shape != (3, 18) or not np.all(np.isfinite(cog_jacobian)):
        raise ValueError(
            "cog_offset_chart_jacobian must be a finite 3 by 18 matrix"
        )
    whitening = _whitening(square_root_information)

    velocity = (1.0 - alpha) * velocity0 + alpha * velocity1
    omega = (1.0 - alpha) * omega0 + alpha * omega1
    rotation, rotation_left_jacobian, rotation_right_jacobian = (
        so3_geodesic_interpolation_with_right_jacobians(
            rotation0,
            rotation1,
            alpha,
        )
    )
    lever = sensor_position - cog_offset
    rotational_velocity_body = np.cross(omega, lever)
    prediction = velocity + rotation @ rotational_velocity_body
    residual = whitening @ (observation - prediction)

    orientation_jacobian = rotation @ skew(rotational_velocity_body)
    omega_jacobian = rotation @ skew(lever)
    cog_jacobian_physical = rotation @ skew(omega)
    right_index = int(left_knot_index) + 1
    blocks = (
        JacobianBlock(
            _key(VariableKind.LINEAR_VELOCITY, bag_id, left_knot_index),
            -(1.0 - alpha) * whitening,
        ),
        JacobianBlock(
            _key(VariableKind.LINEAR_VELOCITY, bag_id, right_index),
            -alpha * whitening,
        ),
        JacobianBlock(
            _key(
                VariableKind.ORIENTATION_TANGENT,
                bag_id,
                left_knot_index,
            ),
            whitening @ orientation_jacobian @ rotation_left_jacobian,
        ),
        JacobianBlock(
            _key(
                VariableKind.ORIENTATION_TANGENT,
                bag_id,
                right_index,
            ),
            whitening @ orientation_jacobian @ rotation_right_jacobian,
        ),
        JacobianBlock(
            _key(
                VariableKind.ANGULAR_VELOCITY,
                bag_id,
                left_knot_index,
            ),
            (1.0 - alpha) * whitening @ omega_jacobian,
        ),
        JacobianBlock(
            _key(
                VariableKind.ANGULAR_VELOCITY,
                bag_id,
                right_index,
            ),
            alpha * whitening @ omega_jacobian,
        ),
        JacobianBlock(
            VariableKey(VariableKind.STATIC_PARAMETERS),
            whitening @ cog_jacobian_physical @ cog_jacobian,
        ),
    )
    return FactorEvaluation(
        residual=residual,
        jacobian_blocks=blocks,
        squared_error=float(residual @ residual),
        active_set={},
    )


__all__ = ["evaluate_world_sensor_velocity_factor"]
