"""Asynchronous IMU observation factors with explicit frame transforms."""

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


def _whitening3(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite 3 by 3 matrix".format(name))
    if np.linalg.matrix_rank(result) != 3:
        raise ValueError("{} must have full rank".format(name))
    return result


def _fraction(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(
            "interpolation_fraction must be finite and in [0, 1]"
        )
    return result


def _positive_time_step(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("time_step must be finite and positive")
    return result


def _knot_key(
    kind: VariableKind,
    bag_id: str,
    knot_index: int,
) -> VariableKey:
    return VariableKey(kind, bag_id=bag_id, knot_index=knot_index)


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

    alpha = _fraction(interpolation_fraction)
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
    whitening = _whitening3(
        square_root_information, "square_root_information"
    )

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


def evaluate_accelerometer_factor(
    bag_id: str,
    left_knot_index: int,
    interpolation_fraction: float,
    rotation_left: np.ndarray,
    rotation_right: np.ndarray,
    linear_velocity_left: np.ndarray,
    linear_velocity_right: np.ndarray,
    angular_velocity_left: np.ndarray,
    angular_velocity_right: np.ndarray,
    accelerometer_bias_sensor: np.ndarray,
    observed_specific_force_sensor: np.ndarray,
    body_to_sensor_rotation: np.ndarray,
    sensor_position_in_body: np.ndarray,
    cog_offset_in_body: np.ndarray,
    cog_offset_chart_jacobian: np.ndarray,
    gravity_world: np.ndarray,
    time_step: float,
    square_root_information: np.ndarray,
) -> FactorEvaluation:
    """Evaluate one asynchronous calibrated specific-force observation.

    ``body_to_sensor_rotation`` is :math:`C_{SB}`: it maps estimator-body
    components into accelerometer-sensor components.  The sensor lever arm is
    ``sensor_position_in_body - cog_offset_in_body`` and therefore changes
    analytically with the shared CoG chart coordinates.  Over one variable-
    duration knot interval, world linear acceleration and body angular
    acceleration use adjacent-state differences, while orientation follows
    the same geodesic interpolation used by the other asynchronous factors.

    The predicted measurement is

    ``C_SB [R_WB.T (dv_W/dt - g_W) + domega_B/dt x r_BS``
    ``+ omega_B x (omega_B x r_BS)] + b_S``.
    """

    alpha = _fraction(interpolation_fraction)
    dt = _positive_time_step(time_step)
    rotation0 = _proper_rotation(rotation_left, "rotation_left")
    rotation1 = _proper_rotation(rotation_right, "rotation_right")
    velocity0 = _finite_vector3(
        linear_velocity_left, "linear_velocity_left"
    )
    velocity1 = _finite_vector3(
        linear_velocity_right, "linear_velocity_right"
    )
    omega0 = _finite_vector3(
        angular_velocity_left, "angular_velocity_left"
    )
    omega1 = _finite_vector3(
        angular_velocity_right, "angular_velocity_right"
    )
    bias = _finite_vector3(
        accelerometer_bias_sensor, "accelerometer_bias_sensor"
    )
    observation = _finite_vector3(
        observed_specific_force_sensor,
        "observed_specific_force_sensor",
    )
    body_to_sensor = _proper_rotation(
        body_to_sensor_rotation, "body_to_sensor_rotation"
    )
    sensor_position = _finite_vector3(
        sensor_position_in_body, "sensor_position_in_body"
    )
    cog_offset = _finite_vector3(
        cog_offset_in_body, "cog_offset_in_body"
    )
    gravity = _finite_vector3(gravity_world, "gravity_world")
    cog_jacobian = np.asarray(cog_offset_chart_jacobian, dtype=float)
    if cog_jacobian.shape != (3, 18) or not np.all(
        np.isfinite(cog_jacobian)
    ):
        raise ValueError(
            "cog_offset_chart_jacobian must be a finite 3 by 18 matrix"
        )
    whitening = _whitening3(
        square_root_information, "square_root_information"
    )

    (
        rotation,
        rotation_left_jacobian,
        rotation_right_jacobian,
    ) = so3_geodesic_interpolation_with_right_jacobians(
        rotation0,
        rotation1,
        alpha,
    )
    acceleration_world = (velocity1 - velocity0) / dt - gravity
    translational_body = rotation.T @ acceleration_world
    angular_acceleration = (omega1 - omega0) / dt
    omega = (1.0 - alpha) * omega0 + alpha * omega1
    lever = sensor_position - cog_offset
    tangential = np.cross(angular_acceleration, lever)
    centripetal = np.cross(omega, np.cross(omega, lever))
    modeled_body = translational_body + tangential + centripetal
    prediction = body_to_sensor @ modeled_body + bias
    residual = whitening @ (observation - prediction)

    orientation_model_jacobian = skew(translational_body)
    centripetal_omega_jacobian = (
        -skew(np.cross(omega, lever))
        - skew(omega) @ skew(lever)
    )
    tangential_omega_jacobian = -skew(lever)
    omega_left_model_jacobian = (
        -tangential_omega_jacobian / dt
        + (1.0 - alpha) * centripetal_omega_jacobian
    )
    omega_right_model_jacobian = (
        tangential_omega_jacobian / dt
        + alpha * centripetal_omega_jacobian
    )
    lever_model_jacobian = (
        skew(angular_acceleration) + skew(omega) @ skew(omega)
    )
    sensor_whitening = whitening @ body_to_sensor
    right_index = int(left_knot_index) + 1
    blocks = (
        JacobianBlock(
            _knot_key(
                VariableKind.ORIENTATION_TANGENT,
                bag_id,
                left_knot_index,
            ),
            -sensor_whitening
            @ orientation_model_jacobian
            @ rotation_left_jacobian,
        ),
        JacobianBlock(
            _knot_key(
                VariableKind.ORIENTATION_TANGENT,
                bag_id,
                right_index,
            ),
            -sensor_whitening
            @ orientation_model_jacobian
            @ rotation_right_jacobian,
        ),
        JacobianBlock(
            _knot_key(
                VariableKind.LINEAR_VELOCITY,
                bag_id,
                left_knot_index,
            ),
            sensor_whitening @ rotation.T / dt,
        ),
        JacobianBlock(
            _knot_key(
                VariableKind.LINEAR_VELOCITY,
                bag_id,
                right_index,
            ),
            -sensor_whitening @ rotation.T / dt,
        ),
        JacobianBlock(
            _knot_key(
                VariableKind.ANGULAR_VELOCITY,
                bag_id,
                left_knot_index,
            ),
            -sensor_whitening @ omega_left_model_jacobian,
        ),
        JacobianBlock(
            _knot_key(
                VariableKind.ANGULAR_VELOCITY,
                bag_id,
                right_index,
            ),
            -sensor_whitening @ omega_right_model_jacobian,
        ),
        JacobianBlock(
            VariableKey(VariableKind.ACCELEROMETER_BIAS, bag_id=bag_id),
            -whitening,
        ),
        JacobianBlock(
            VariableKey(VariableKind.STATIC_PARAMETERS),
            sensor_whitening @ lever_model_jacobian @ cog_jacobian,
        ),
    )
    return FactorEvaluation(
        residual=residual,
        jacobian_blocks=blocks,
        squared_error=float(residual @ residual),
        active_set={},
    )


__all__ = ["evaluate_accelerometer_factor", "evaluate_gyro_factor"]
