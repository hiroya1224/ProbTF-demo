"""Analytic fixed-schedule controller factors for sparse batch estimation."""

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np

from grape_param_estim.batch.factor import FactorEvaluation, JacobianBlock
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.controller import (
    ControllerLocalJacobian,
    ControllerStepDiagnostics,
    GrapeController,
)
from grape_param_estim.geometry import matrix_to_quaternion
from grape_param_estim.system import (
    ControllerState,
    ReferenceState,
    RigidBodyState,
)


_CURRENT_KINDS = (
    VariableKind.POSITION,
    VariableKind.ORIENTATION_TANGENT,
    VariableKind.LINEAR_VELOCITY,
    VariableKind.ANGULAR_VELOCITY,
    VariableKind.CONTROLLER_INTEGRAL,
    VariableKind.GIMBAL_ANGLE,
)


def _finite_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(
            "{} must contain {} finite values".format(name, size)
        )
    return result


def _proper_rotation(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("rotation must be a finite 3 by 3 matrix")
    if not np.allclose(result.T @ result, np.eye(3), rtol=0.0, atol=1.0e-9):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(result), 1.0, rtol=0.0, atol=1.0e-9):
        raise ValueError("rotation must have determinant one")
    return result


def _whitening(value: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size, size) or not np.all(np.isfinite(result)):
        raise ValueError(
            "{} must be a finite {} by {} matrix".format(name, size, size)
        )
    if np.linalg.matrix_rank(result) != size:
        raise ValueError("{} must have full rank".format(name))
    return result


def _key(
    kind: VariableKind,
    bag_id: str,
    knot_index: int,
) -> VariableKey:
    return VariableKey(kind, bag_id=bag_id, knot_index=knot_index)


def evaluate_controller_integral_observation_factor(
    bag_id: str,
    left_knot_index: int,
    interpolation_fraction: float,
    integral_left: np.ndarray,
    integral_right: np.ndarray,
    observed_integral: np.ndarray,
    square_root_information: np.ndarray,
) -> FactorEvaluation:
    """Connect a recorded PID-integral proxy at its asynchronous time."""

    alpha = float(interpolation_fraction)
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError(
            "interpolation_fraction must be finite and in [0, 1]"
        )
    left = _finite_vector(integral_left, 6, "integral_left")
    right = _finite_vector(integral_right, 6, "integral_right")
    observation = _finite_vector(
        observed_integral, 6, "observed_integral"
    )
    whitening = _whitening(
        square_root_information,
        6,
        "square_root_information",
    )
    prediction = (1.0 - alpha) * left + alpha * right
    residual = whitening @ (observation - prediction)
    right_index = int(left_knot_index) + 1
    return FactorEvaluation(
        residual=residual,
        jacobian_blocks=(
            JacobianBlock(
                _key(
                    VariableKind.CONTROLLER_INTEGRAL,
                    bag_id,
                    left_knot_index,
                ),
                -(1.0 - alpha) * whitening,
            ),
            JacobianBlock(
                _key(
                    VariableKind.CONTROLLER_INTEGRAL,
                    bag_id,
                    right_index,
                ),
                -alpha * whitening,
            ),
        ),
        squared_error=float(residual @ residual),
        active_set={},
    )


def _active_set(
    diagnostics: ControllerStepDiagnostics,
) -> Mapping[str, np.ndarray]:
    result = {
        "pid_near_kink": diagnostics.pid_near_kink,
        "rpy_gimbal_lock_near_kink": np.asarray(
            (diagnostics.rpy_gimbal_lock_near_kink,), dtype=bool
        ),
        "yaw_wrap_near_kink": np.asarray(
            (diagnostics.yaw_wrap_near_kink,), dtype=bool
        ),
        "thrust_atan2_near_kink": diagnostics.thrust_atan2_near_kink,
        "allocation_near_singular": np.asarray(
            (diagnostics.allocation.allocation_near_singular,), dtype=bool
        ),
        "roll_pitch_integration_active": np.asarray(
            (diagnostics.roll_pitch_integration_active,), dtype=bool
        ),
        "source_compatible_gyro_term": np.asarray(
            (diagnostics.source_compatible_gyro_term,), dtype=bool
        ),
    }
    clamp_names = (
        "error_p",
        "error_d",
        "integral",
        "p_term",
        "i_term",
        "d_term",
        "output_sum",
        "nonnegative_integral",
    )
    for clamp_name in clamp_names:
        states = tuple(
            getattr(axis, clamp_name) for axis in diagnostics.pid_active_sets
        )
        result["pid_{}_saturated".format(clamp_name)] = np.asarray(
            tuple(state.saturated for state in states), dtype=bool
        )
        result["pid_{}_near_kink".format(clamp_name)] = np.asarray(
            tuple(state.near_kink for state in states), dtype=bool
        )
    return result


def _current_blocks(
    bag_id: str,
    knot_index: int,
    local_jacobian: ControllerLocalJacobian,
    left_multiplier: np.ndarray,
) -> Tuple[JacobianBlock, ...]:
    local_blocks = (
        local_jacobian.position,
        local_jacobian.orientation_right_tangent,
        local_jacobian.world_velocity,
        local_jacobian.body_omega,
        local_jacobian.integral_error,
        local_jacobian.current_gimbal_angle,
    )
    return tuple(
        JacobianBlock(
            _key(kind, bag_id, knot_index),
            left_multiplier @ value,
        )
        for kind, value in zip(_CURRENT_KINDS, local_blocks)
    )


def _observation_factor(
    bag_id: str,
    knot_index: int,
    observation: np.ndarray,
    model_value: np.ndarray,
    model_jacobian: ControllerLocalJacobian,
    square_root_information: np.ndarray,
    active_set: Mapping[str, np.ndarray],
) -> FactorEvaluation:
    residual = square_root_information @ (observation - model_value)
    blocks = _current_blocks(
        bag_id,
        knot_index,
        model_jacobian,
        -square_root_information,
    )
    return FactorEvaluation(
        residual=residual,
        jacobian_blocks=blocks,
        squared_error=float(residual @ residual),
        active_set=active_set,
    )


@dataclass(frozen=True)
class ControllerFactorSet:
    """Controller transition plus the available issued-command factors."""

    integral_transition: FactorEvaluation
    issued_thrust: Optional[FactorEvaluation]
    issued_gimbal_angle: Optional[FactorEvaluation]
    diagnostics: ControllerStepDiagnostics
    next_roll_pitch_integration_active: bool

    def __post_init__(self) -> None:
        if not isinstance(self.integral_transition, FactorEvaluation):
            raise TypeError("integral_transition must be a FactorEvaluation")
        for name in ("issued_thrust", "issued_gimbal_angle"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, FactorEvaluation):
                raise TypeError("{} must be a FactorEvaluation or None".format(name))
        if not isinstance(self.diagnostics, ControllerStepDiagnostics):
            raise TypeError("diagnostics must be ControllerStepDiagnostics")
        if not isinstance(
            self.next_roll_pitch_integration_active, (bool, np.bool_)
        ):
            raise TypeError(
                "next_roll_pitch_integration_active must be boolean"
            )
        object.__setattr__(
            self,
            "next_roll_pitch_integration_active",
            bool(self.next_roll_pitch_integration_active),
        )

    @property
    def factors(self) -> Tuple[FactorEvaluation, ...]:
        """Return factors in deterministic transition/thrust/gimbal order."""

        result = [self.integral_transition]
        if self.issued_thrust is not None:
            result.append(self.issued_thrust)
        if self.issued_gimbal_angle is not None:
            result.append(self.issued_gimbal_angle)
        return tuple(result)


def evaluate_controller_step_factors(
    controller: GrapeController,
    bag_id: str,
    knot_index: int,
    position: np.ndarray,
    rotation: np.ndarray,
    world_velocity: np.ndarray,
    body_omega: np.ndarray,
    integral_left: np.ndarray,
    integral_right: np.ndarray,
    current_gimbal_angle: np.ndarray,
    reference: ReferenceState,
    time_step: float,
    roll_pitch_integration_active: bool,
    integral_square_root_information: np.ndarray,
    issued_thrust_observation: Optional[np.ndarray] = None,
    issued_thrust_square_root_information: Optional[np.ndarray] = None,
    issued_gimbal_observation: Optional[np.ndarray] = None,
    issued_gimbal_square_root_information: Optional[np.ndarray] = None,
) -> ControllerFactorSet:
    """Evaluate a fixed-discrete-schedule controller graph interval.

    The controller snapshot, reference, and current discrete integration flag
    are fixed provenance.  Continuous batch variables are the six current
    state blocks and the next integral block.  Recorded issued commands remain
    separate observation factors with independent covariance.
    """

    if not isinstance(controller, GrapeController):
        raise TypeError("controller must be a GrapeController")
    if not isinstance(reference, ReferenceState):
        raise TypeError("reference must be a ReferenceState")
    if not isinstance(roll_pitch_integration_active, (bool, np.bool_)):
        raise TypeError("roll_pitch_integration_active must be boolean")
    point = _finite_vector(position, 3, "position")
    attitude = _proper_rotation(rotation)
    velocity = _finite_vector(world_velocity, 3, "world_velocity")
    omega = _finite_vector(body_omega, 3, "body_omega")
    integral0 = _finite_vector(integral_left, 6, "integral_left")
    integral1 = _finite_vector(integral_right, 6, "integral_right")
    gimbal = _finite_vector(current_gimbal_angle, 4, "current_gimbal_angle")
    integral_whitening = _whitening(
        integral_square_root_information,
        6,
        "integral_square_root_information",
    )

    state = RigidBodyState(
        point,
        matrix_to_quaternion(attitude),
        velocity,
        omega,
    )
    controller_state = ControllerState(
        integral0,
        bool(roll_pitch_integration_active),
    )
    step = controller.step_with_jacobian(
        state,
        reference,
        controller_state,
        time_step,
        gimbal,
    )
    active_set = _active_set(step.diagnostics)

    integral_residual = integral_whitening @ (
        integral1 - step.next_state.integral_error
    )
    integral_blocks = _current_blocks(
        bag_id,
        knot_index,
        step.jacobian.next_integral,
        -integral_whitening,
    ) + (
        JacobianBlock(
            _key(
                VariableKind.CONTROLLER_INTEGRAL,
                bag_id,
                int(knot_index) + 1,
            ),
            integral_whitening,
        ),
    )
    integral_factor = FactorEvaluation(
        residual=integral_residual,
        jacobian_blocks=integral_blocks,
        squared_error=float(integral_residual @ integral_residual),
        active_set=active_set,
    )

    thrust_value_present = issued_thrust_observation is not None
    thrust_weight_present = issued_thrust_square_root_information is not None
    if thrust_value_present != thrust_weight_present:
        raise ValueError(
            "issued thrust observation and square-root information "
            "must be provided together"
        )
    thrust_factor = None
    if thrust_value_present:
        thrust_observation = _finite_vector(
            issued_thrust_observation, 4, "issued_thrust_observation"
        )
        thrust_whitening = _whitening(
            issued_thrust_square_root_information,
            4,
            "issued_thrust_square_root_information",
        )
        thrust_factor = _observation_factor(
            bag_id,
            knot_index,
            thrust_observation,
            step.command.thrust,
            step.jacobian.issued_thrust,
            thrust_whitening,
            active_set,
        )

    gimbal_value_present = issued_gimbal_observation is not None
    gimbal_weight_present = issued_gimbal_square_root_information is not None
    if gimbal_value_present != gimbal_weight_present:
        raise ValueError(
            "issued gimbal observation and square-root information "
            "must be provided together"
        )
    gimbal_factor = None
    if gimbal_value_present:
        gimbal_observation = _finite_vector(
            issued_gimbal_observation, 4, "issued_gimbal_observation"
        )
        gimbal_whitening = _whitening(
            issued_gimbal_square_root_information,
            4,
            "issued_gimbal_square_root_information",
        )
        gimbal_factor = _observation_factor(
            bag_id,
            knot_index,
            gimbal_observation,
            step.command.gimbal_angle,
            step.jacobian.issued_gimbal_angle,
            gimbal_whitening,
            active_set,
        )

    return ControllerFactorSet(
        integral_transition=integral_factor,
        issued_thrust=thrust_factor,
        issued_gimbal_angle=gimbal_factor,
        diagnostics=step.diagnostics,
        next_roll_pitch_integration_active=(
            step.next_state.roll_pitch_integration_active
        ),
    )


__all__ = [
    "ControllerFactorSet",
    "evaluate_controller_integral_observation_factor",
    "evaluate_controller_step_factors",
]
