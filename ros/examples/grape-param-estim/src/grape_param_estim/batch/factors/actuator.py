"""Piecewise analytic latent-actuator transition factors."""

import numpy as np

from grape_param_estim.batch.factor import FactorEvaluation, JacobianBlock
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.dynamics import advance_actuators_with_jacobian
from grape_param_estim.system import (
    ActuatorCommand,
    ActuatorParameters,
    ActuatorState,
)


def _finite_vector4(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (4,) or not np.all(np.isfinite(result)):
        raise ValueError("{} must contain four finite values".format(name))
    return result


def _whitening4(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite 4 by 4 matrix".format(name))
    if np.linalg.matrix_rank(result) != 4:
        raise ValueError("{} must have full rank".format(name))
    return result


def _key(kind: VariableKind, bag_id: str, knot_index: int) -> VariableKey:
    return VariableKey(kind, bag_id=bag_id, knot_index=knot_index)


def evaluate_actuator_transition_factor(
    bag_id: str,
    left_knot_index: int,
    thrust_left: np.ndarray,
    thrust_right: np.ndarray,
    gimbal_left: np.ndarray,
    gimbal_right: np.ndarray,
    issued_command: ActuatorCommand,
    actuator_parameters: ActuatorParameters,
    time_step: float,
    thrust_square_root_information: np.ndarray,
    gimbal_square_root_information: np.ndarray,
) -> FactorEvaluation:
    """Evaluate one constant-command actuator transition defect.

    This is the one-segment convenience wrapper for
    :func:`evaluate_actuator_interval_factor`.
    """

    return evaluate_actuator_interval_factor(
        bag_id=bag_id,
        left_knot_index=left_knot_index,
        thrust_left=thrust_left,
        thrust_right=thrust_right,
        gimbal_left=gimbal_left,
        gimbal_right=gimbal_right,
        command_segments=((issued_command, time_step),),
        actuator_parameters=actuator_parameters,
        thrust_square_root_information=thrust_square_root_information,
        gimbal_square_root_information=gimbal_square_root_information,
    )


def evaluate_actuator_interval_factor(
    bag_id: str,
    left_knot_index: int,
    thrust_left: np.ndarray,
    thrust_right: np.ndarray,
    gimbal_left: np.ndarray,
    gimbal_right: np.ndarray,
    command_segments: tuple,
    actuator_parameters: ActuatorParameters,
    thrust_square_root_information: np.ndarray,
    gimbal_square_root_information: np.ndarray,
) -> FactorEvaluation:
    """Evaluate an interval split at every delayed ZOH command switch.

    ``command_segments`` is a non-empty tuple of ``(ActuatorCommand, dt)``
    pairs in chronological order.  Segment transition Jacobians are composed
    analytically, so delay-switch event times do not get rounded to a knot and
    no intermediate actuator state needs to become an optimization variable.
    The fixed transition covariance is independent of dynamics Q.
    """

    thrust0 = _finite_vector4(thrust_left, "thrust_left")
    thrust1 = _finite_vector4(thrust_right, "thrust_right")
    gimbal0 = _finite_vector4(gimbal_left, "gimbal_left")
    gimbal1 = _finite_vector4(gimbal_right, "gimbal_right")
    if not isinstance(actuator_parameters, ActuatorParameters):
        raise TypeError("actuator_parameters must be ActuatorParameters")
    if type(command_segments) is not tuple or not command_segments:
        raise TypeError("command_segments must be a non-empty tuple")
    thrust_whitening = _whitening4(
        thrust_square_root_information,
        "thrust_square_root_information",
    )
    gimbal_whitening = _whitening4(
        gimbal_square_root_information,
        "gimbal_square_root_information",
    )
    current = ActuatorState(thrust0, gimbal0)
    thrust_previous = np.eye(4)
    gimbal_previous = np.eye(4)
    active_set = {}
    multiple_segments = len(command_segments) > 1
    for segment_index, segment in enumerate(command_segments):
        if type(segment) is not tuple or len(segment) != 2:
            raise TypeError(
                "each command segment must be an (ActuatorCommand, dt) tuple"
            )
        issued_command, segment_duration = segment
        if not isinstance(issued_command, ActuatorCommand):
            raise TypeError("segment command must be an ActuatorCommand")
        transition = advance_actuators_with_jacobian(
            current,
            issued_command,
            actuator_parameters,
            segment_duration,
        )
        thrust_previous = (
            transition.jacobian.thrust_previous @ thrust_previous
        )
        gimbal_previous = (
            transition.jacobian.gimbal_previous @ gimbal_previous
        )
        for name, mask in transition.active_set.items():
            active_name = (
                "segment_{:03d}/{}".format(segment_index, name)
                if multiple_segments
                else name
            )
            active_set[active_name] = mask
        current = transition.next_state
    residual = np.concatenate(
        (
            thrust_whitening @ (thrust1 - current.thrust),
            gimbal_whitening @ (gimbal1 - current.gimbal_angle),
        )
    )
    zero = np.zeros((4, 4), dtype=float)
    right_index = int(left_knot_index) + 1
    blocks = (
        JacobianBlock(
            _key(VariableKind.ACTUATOR_THRUST, bag_id, left_knot_index),
            np.vstack(
                (
                    -thrust_whitening
                    @ thrust_previous,
                    zero,
                )
            ),
        ),
        JacobianBlock(
            _key(VariableKind.ACTUATOR_THRUST, bag_id, right_index),
            np.vstack((thrust_whitening, zero)),
        ),
        JacobianBlock(
            _key(VariableKind.GIMBAL_ANGLE, bag_id, left_knot_index),
            np.vstack(
                (
                    zero,
                    -gimbal_whitening
                    @ gimbal_previous,
                )
            ),
        ),
        JacobianBlock(
            _key(VariableKind.GIMBAL_ANGLE, bag_id, right_index),
            np.vstack((zero, gimbal_whitening)),
        ),
    )
    return FactorEvaluation(
        residual=residual,
        jacobian_blocks=blocks,
        squared_error=float(residual @ residual),
        active_set=active_set,
    )


def evaluate_gimbal_position_factor(
    bag_id: str,
    left_knot_index: int,
    interpolation_fraction: float,
    gimbal_left: np.ndarray,
    gimbal_right: np.ndarray,
    observed_gimbal_position: np.ndarray,
    square_root_information: np.ndarray,
) -> FactorEvaluation:
    """Evaluate an asynchronous actual joint-angle observation factor."""

    alpha = float(interpolation_fraction)
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("interpolation_fraction must be finite and in [0, 1]")
    gimbal0 = _finite_vector4(gimbal_left, "gimbal_left")
    gimbal1 = _finite_vector4(gimbal_right, "gimbal_right")
    observation = _finite_vector4(
        observed_gimbal_position,
        "observed_gimbal_position",
    )
    whitening = _whitening4(
        square_root_information,
        "square_root_information",
    )
    residual = whitening @ (
        observation - (1.0 - alpha) * gimbal0 - alpha * gimbal1
    )
    right_index = int(left_knot_index) + 1
    return FactorEvaluation(
        residual=residual,
        jacobian_blocks=(
            JacobianBlock(
                _key(VariableKind.GIMBAL_ANGLE, bag_id, left_knot_index),
                -(1.0 - alpha) * whitening,
            ),
            JacobianBlock(
                _key(VariableKind.GIMBAL_ANGLE, bag_id, right_index),
                -alpha * whitening,
            ),
        ),
        squared_error=float(residual @ residual),
        active_set={},
    )


__all__ = [
    "evaluate_actuator_interval_factor",
    "evaluate_actuator_transition_factor",
    "evaluate_gimbal_position_factor",
]
