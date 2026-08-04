"""Asynchronous IMU observation factors with explicit frame transforms."""

import numpy as np

from grape_param_estim.batch.factor import FactorEvaluation, JacobianBlock
from grape_param_estim.batch.variables import VariableKey, VariableKind


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


def evaluate_gyro_factor(
    bag_id: str,
    left_knot_index: int,
    interpolation_fraction: float,
    angular_velocity_left: np.ndarray,
    angular_velocity_right: np.ndarray,
    gyro_bias_sensor: np.ndarray,
    observed_angular_velocity_sensor: np.ndarray,
    body_to_sensor_rotation: np.ndarray,
    square_root_information: np.ndarray,
) -> FactorEvaluation:
    """Evaluate ``y_S - C_SB omega_B - b_S`` at one IMU timestamp."""

    alpha = float(interpolation_fraction)
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("interpolation_fraction must be finite and in [0, 1]")
    omega0 = _finite_vector3(
        angular_velocity_left, "angular_velocity_left"
    )
    omega1 = _finite_vector3(
        angular_velocity_right, "angular_velocity_right"
    )
    bias = _finite_vector3(gyro_bias_sensor, "gyro_bias_sensor")
    observation = _finite_vector3(
        observed_angular_velocity_sensor,
        "observed_angular_velocity_sensor",
    )
    body_to_sensor = _proper_rotation(
        body_to_sensor_rotation, "body_to_sensor_rotation"
    )
    whitening = np.asarray(square_root_information, dtype=float)
    if whitening.shape != (3, 3) or not np.all(np.isfinite(whitening)):
        raise ValueError(
            "square_root_information must be a finite 3 by 3 matrix"
        )
    if np.linalg.matrix_rank(whitening) != 3:
        raise ValueError("square_root_information must have full rank")

    omega = (1.0 - alpha) * omega0 + alpha * omega1
    residual = whitening @ (observation - body_to_sensor @ omega - bias)
    right_index = int(left_knot_index) + 1
    blocks = (
        JacobianBlock(
            VariableKey(
                VariableKind.ANGULAR_VELOCITY,
                bag_id=bag_id,
                knot_index=left_knot_index,
            ),
            -(1.0 - alpha) * whitening @ body_to_sensor,
        ),
        JacobianBlock(
            VariableKey(
                VariableKind.ANGULAR_VELOCITY,
                bag_id=bag_id,
                knot_index=right_index,
            ),
            -alpha * whitening @ body_to_sensor,
        ),
        JacobianBlock(
            VariableKey(VariableKind.GYRO_BIAS, bag_id=bag_id),
            -whitening,
        ),
    )
    return FactorEvaluation(
        residual=residual,
        jacobian_blocks=blocks,
        squared_error=float(residual @ residual),
        active_set={},
    )


__all__ = ["evaluate_gyro_factor"]
