"""Analytic midpoint kinematic-consistency factors for batch estimation."""

from typing import Tuple

import numpy as np

from grape_param_estim.batch.factor import FactorEvaluation, JacobianBlock
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.geometry import (
    so3_exp,
    so3_log,
    so3_right_jacobian,
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


def _sqrt_information3(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError(
            "square_root_information must be a finite 3 by 3 matrix"
        )
    if np.linalg.matrix_rank(result) != 3:
        raise ValueError("square_root_information must have full rank")
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


def evaluate_position_kinematic_factor(
    bag_id: str,
    knot_index: int,
    position_left: np.ndarray,
    position_right: np.ndarray,
    velocity_left: np.ndarray,
    velocity_right: np.ndarray,
    time_step: float,
    square_root_information: np.ndarray,
) -> FactorEvaluation:
    """Evaluate the whitened trapezoidal position-consistency defect."""

    position0 = _finite_vector3(position_left, "position_left")
    position1 = _finite_vector3(position_right, "position_right")
    velocity0 = _finite_vector3(velocity_left, "velocity_left")
    velocity1 = _finite_vector3(velocity_right, "velocity_right")
    dt = _positive_time_step(time_step)
    whitening = _sqrt_information3(square_root_information)
    next_index = int(knot_index) + 1

    raw_residual = (
        position1 - position0 - 0.5 * dt * (velocity0 + velocity1)
    )
    residual = whitening @ raw_residual
    identity = np.eye(3)
    blocks = (
        JacobianBlock(
            _knot_key(VariableKind.POSITION, bag_id, knot_index),
            -whitening @ identity,
        ),
        JacobianBlock(
            _knot_key(VariableKind.POSITION, bag_id, next_index),
            whitening @ identity,
        ),
        JacobianBlock(
            _knot_key(VariableKind.LINEAR_VELOCITY, bag_id, knot_index),
            -0.5 * dt * whitening,
        ),
        JacobianBlock(
            _knot_key(VariableKind.LINEAR_VELOCITY, bag_id, next_index),
            -0.5 * dt * whitening,
        ),
    )
    return FactorEvaluation(
        residual=residual,
        jacobian_blocks=blocks,
        squared_error=float(residual @ residual),
        active_set={},
    )


def _orientation_defect_and_raw_jacobians(
    rotation_left: np.ndarray,
    rotation_right: np.ndarray,
    angular_velocity_left: np.ndarray,
    angular_velocity_right: np.ndarray,
    time_step: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotation0 = _proper_rotation(rotation_left, "rotation_left")
    rotation1 = _proper_rotation(rotation_right, "rotation_right")
    omega0 = _finite_vector3(
        angular_velocity_left, "angular_velocity_left"
    )
    omega1 = _finite_vector3(
        angular_velocity_right, "angular_velocity_right"
    )
    dt = _positive_time_step(time_step)

    relative_rotation = rotation0.T @ rotation1
    integration_vector = -0.5 * dt * (omega0 + omega1)
    rotation_defect = so3_exp(integration_vector) @ relative_rotation
    raw_residual = so3_log(rotation_defect)

    log_jacobian = so3_right_jacobian_inverse(raw_residual)
    endpoint_transport = relative_rotation.T
    integration_jacobian = so3_right_jacobian(integration_vector)
    orientation_left_jacobian = -log_jacobian @ endpoint_transport
    orientation_right_jacobian = log_jacobian
    angular_velocity_jacobian = (
        -0.5
        * dt
        * log_jacobian
        @ endpoint_transport
        @ integration_jacobian
    )
    return (
        raw_residual,
        orientation_left_jacobian,
        orientation_right_jacobian,
        angular_velocity_jacobian,
        angular_velocity_jacobian.copy(),
    )


def evaluate_orientation_kinematic_factor(
    bag_id: str,
    knot_index: int,
    rotation_left: np.ndarray,
    rotation_right: np.ndarray,
    angular_velocity_left: np.ndarray,
    angular_velocity_right: np.ndarray,
    time_step: float,
    square_root_information: np.ndarray,
) -> FactorEvaluation:
    """Evaluate the whitened SO(3) midpoint-integration defect.

    Orientation Jacobian columns use right-tangent endpoint perturbations.
    The principal logarithm supplies the finite convention at its branch cut;
    callers receive an explicit diagnostic before that derivative becomes
    non-unique.
    """

    whitening = _sqrt_information3(square_root_information)
    (
        raw_residual,
        orientation_left_jacobian,
        orientation_right_jacobian,
        angular_velocity_left_jacobian,
        angular_velocity_right_jacobian,
    ) = _orientation_defect_and_raw_jacobians(
        rotation_left,
        rotation_right,
        angular_velocity_left,
        angular_velocity_right,
        time_step,
    )
    residual = whitening @ raw_residual
    next_index = int(knot_index) + 1
    blocks = (
        JacobianBlock(
            _knot_key(
                VariableKind.ORIENTATION_TANGENT, bag_id, knot_index
            ),
            whitening @ orientation_left_jacobian,
        ),
        JacobianBlock(
            _knot_key(
                VariableKind.ORIENTATION_TANGENT, bag_id, next_index
            ),
            whitening @ orientation_right_jacobian,
        ),
        JacobianBlock(
            _knot_key(
                VariableKind.ANGULAR_VELOCITY, bag_id, knot_index
            ),
            whitening @ angular_velocity_left_jacobian,
        ),
        JacobianBlock(
            _knot_key(
                VariableKind.ANGULAR_VELOCITY, bag_id, next_index
            ),
            whitening @ angular_velocity_right_jacobian,
        ),
    )
    distance_to_branch = np.pi - float(np.linalg.norm(raw_residual))
    return FactorEvaluation(
        residual=residual,
        jacobian_blocks=blocks,
        squared_error=float(residual @ residual),
        active_set={
            "rotation_log_near_pi": np.asarray(
                (distance_to_branch <= _LOG_BRANCH_WARNING_DISTANCE,),
                dtype=bool,
            )
        },
    )


__all__ = [
    "evaluate_orientation_kinematic_factor",
    "evaluate_position_kinematic_factor",
]
