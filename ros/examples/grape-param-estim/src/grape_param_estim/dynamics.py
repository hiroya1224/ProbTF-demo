"""Continuous full six-DoF plant and closed-loop trajectory forecast."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.controller import GrapeController
from grape_param_estim.geometry import (
    normalise_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
    skew,
)
from grape_param_estim.system import (
    ActuatorCommand,
    ActuatorParameters,
    ActuatorState,
    ClosedLoopTrajectory,
    ControllerState,
    GRAVITY,
    GrapeGeometry,
    ReferenceState,
    RigidBodyState,
    VehicleParameters,
)
from grape_param_estim.timing import ZeroOrderHoldCommandHistory


ModelDiscrepancyWrench = Callable[[float, RigidBodyState], np.ndarray]


@dataclass(frozen=True)
class ActuatorWrenchJacobian:
    """Analytic Jacobian blocks of the six-dimensional actuator wrench.

    The blocks are with respect to actual thrust ``(6, 4)``, actual gimbal
    angle ``(6, 4)``, CoG offset ``(6, 3)``, force effectiveness ``(6, 4)``,
    and torque effectiveness ``(6, 4)``.  Mass, inertia, and drag have no
    direct dependency at this actuator-only API boundary.
    """

    actual_thrust: np.ndarray
    actual_gimbal_angle: np.ndarray
    cog_offset: np.ndarray
    force_effectiveness: np.ndarray
    torque_effectiveness: np.ndarray

    def __post_init__(self) -> None:
        for name, shape in (
            ("actual_thrust", (6, 4)),
            ("actual_gimbal_angle", (6, 4)),
            ("cog_offset", (6, 3)),
            ("force_effectiveness", (6, 4)),
            ("torque_effectiveness", (6, 4)),
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(
                    "{} must be a finite {} array".format(name, shape)
                )
            object.__setattr__(self, name, value.copy())


@dataclass(frozen=True)
class ActuatorTransitionJacobian:
    """Analytic blocks of one piecewise-smooth actuator transition.

    Each matrix is diagonal because the four actuator channels evolve
    independently.  Time-step derivatives are column vectors represented as
    length-four arrays.
    """

    thrust_previous: np.ndarray
    thrust_command: np.ndarray
    thrust_time_step: np.ndarray
    gimbal_previous: np.ndarray
    gimbal_command: np.ndarray
    gimbal_time_step: np.ndarray

    def __post_init__(self) -> None:
        for name, shape in (
            ("thrust_previous", (4, 4)),
            ("thrust_command", (4, 4)),
            ("thrust_time_step", (4,)),
            ("gimbal_previous", (4, 4)),
            ("gimbal_command", (4, 4)),
            ("gimbal_time_step", (4,)),
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(
                    "{} must be a finite {} array".format(name, shape)
                )
            result = value.copy()
            result.setflags(write=False)
            object.__setattr__(self, name, result)


@dataclass(frozen=True)
class ActuatorTransitionEvaluation:
    """Next actuator state, analytic derivative blocks, and branch masks."""

    next_state: ActuatorState
    jacobian: ActuatorTransitionJacobian
    active_set: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if not isinstance(self.next_state, ActuatorState):
            raise TypeError("next_state must be an ActuatorState")
        if not isinstance(self.jacobian, ActuatorTransitionJacobian):
            raise TypeError("jacobian must be an ActuatorTransitionJacobian")
        copied = {}
        for name, mask in self.active_set.items():
            if type(name) is not str or not name:
                raise ValueError("active-set names must be non-empty strings")
            value = np.asarray(mask)
            if value.shape != (4,) or value.dtype != np.bool_:
                raise ValueError("actuator active-set masks must be boolean (4,)")
            result = value.copy()
            result.setflags(write=False)
            copied[name] = result
        object.__setattr__(self, "active_set", MappingProxyType(copied))


@dataclass(frozen=True)
class RigidBodyStepJacobian:
    """Exact active-branch Jacobian of one RK4 rigid-body step.

    ``state`` maps the 12-dimensional local rigid-body chart
    ``(p, right-tangent rotation, v, omega)`` to the same chart at the end of
    the step.  The actuator blocks differentiate with respect to the actual
    midpoint actuator state used by :meth:`FullSixDofPlant.step`.
    """

    state: np.ndarray
    actual_thrust: np.ndarray
    actual_gimbal_angle: np.ndarray

    def __post_init__(self) -> None:
        for name, shape in (
            ("state", (12, 12)),
            ("actual_thrust", (12, 4)),
            ("actual_gimbal_angle", (12, 4)),
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError("{} must be a finite {} array".format(name, shape))
            object.__setattr__(self, name, value.copy())


@dataclass(frozen=True)
class RigidBodyStepEvaluation:
    """Forward RK4 rigid-body step plus its exact local Jacobian."""

    next_state: RigidBodyState
    jacobian: RigidBodyStepJacobian

    def __post_init__(self) -> None:
        if not isinstance(self.next_state, RigidBodyState):
            raise TypeError("next_state must be a RigidBodyState")
        if not isinstance(self.jacobian, RigidBodyStepJacobian):
            raise TypeError("jacobian must be a RigidBodyStepJacobian")


def _rotate_z(vector: np.ndarray, angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.asarray(
        (
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
            vector[2],
        ),
        dtype=float,
    )


def _validate_actuator_wrench_inputs(
    actuator_state: ActuatorState,
    parameters: VehicleParameters,
    geometry: GrapeGeometry,
) -> None:
    if not isinstance(actuator_state, ActuatorState):
        raise TypeError("actuator_state must be an ActuatorState instance")
    if not isinstance(parameters, VehicleParameters):
        raise TypeError("parameters must be a VehicleParameters instance")
    if not isinstance(geometry, GrapeGeometry):
        raise TypeError("geometry must be a GrapeGeometry instance")
    for value, shape, name in (
        (actuator_state.thrust, (4,), "actual thrust"),
        (actuator_state.gimbal_angle, (4,), "actual gimbal angle"),
        (parameters.cog_offset, (3,), "CoG offset"),
        (parameters.force_effectiveness, (4,), "force effectiveness"),
        (parameters.torque_effectiveness, (4,), "torque effectiveness"),
        (geometry.rotor_origins, (4, 3), "rotor origins"),
        (geometry.arm_yaws, (4,), "arm yaws"),
        (geometry.rotor_directions, (4,), "rotor directions"),
    ):
        array = np.asarray(value, dtype=float)
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError("{} must be a finite {} array".format(name, shape))
    geometry_scalars = np.asarray(
        (geometry.moment_force_rate, geometry.thrust_offset), dtype=float
    )
    if (
        not np.all(np.isfinite(geometry_scalars))
        or geometry.thrust_offset < 0.0
    ):
        raise ValueError("geometry wrench scalars must be finite and valid")


def _rotor_force_primitives(
    actual_thrust: float,
    force_effectiveness: float,
    gimbal_angle: float,
    arm_yaw: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return force and shared direction/gimbal derivative primitives."""

    sine = float(np.sin(gimbal_angle))
    cosine = float(np.cos(gimbal_angle))
    unit_direction = _rotate_z(
        np.asarray((0.0, -sine, cosine), dtype=float),
        arm_yaw,
    )
    unit_direction_derivative = _rotate_z(
        np.asarray((0.0, -cosine, -sine), dtype=float),
        arm_yaw,
    )
    effective_thrust = actual_thrust * force_effectiveness
    # Retain the original forward multiplication order for actuator_wrench.
    local_force = np.asarray(
        (
            0.0,
            -effective_thrust * sine,
            effective_thrust * cosine,
        ),
        dtype=float,
    )
    force = _rotate_z(local_force, arm_yaw)
    force_gimbal_derivative = (
        effective_thrust * unit_direction_derivative
    )
    return (
        force,
        unit_direction,
        unit_direction_derivative,
        force_gimbal_derivative,
    )


def actuator_wrench(
    actuator_state: ActuatorState,
    parameters: VehicleParameters,
    geometry: GrapeGeometry,
) -> np.ndarray:
    """Map actual rotor thrust and gimbal angle to a body-frame wrench."""

    _validate_actuator_wrench_inputs(actuator_state, parameters, geometry)
    wrench = np.zeros(6, dtype=float)
    thrust_origins = geometry.thrust_origins(
        actuator_state.gimbal_angle
    )
    for rotor in range(4):
        angle = actuator_state.gimbal_angle[rotor]
        force, _, _, _ = _rotor_force_primitives(
            actual_thrust=float(actuator_state.thrust[rotor]),
            force_effectiveness=float(parameters.force_effectiveness[rotor]),
            gimbal_angle=float(angle),
            arm_yaw=float(geometry.arm_yaws[rotor]),
        )
        origin = (
            thrust_origins[rotor]
            - parameters.cog_offset
        )
        reaction_torque = (
            parameters.torque_effectiveness[rotor]
            * geometry.rotor_directions[rotor]
            * geometry.moment_force_rate
            * force
        )
        wrench[:3] += force
        wrench[3:] += np.cross(origin, force) + reaction_torque
    if not np.all(np.isfinite(wrench)):
        raise ValueError("actuator wrench is not finite")
    return wrench


def actuator_wrench_with_jacobian(
    actuator_state: ActuatorState,
    parameters: VehicleParameters,
    geometry: GrapeGeometry,
) -> Tuple[np.ndarray, ActuatorWrenchJacobian]:
    """Return :func:`actuator_wrench` and its exact analytic Jacobian blocks."""

    # The public forward function remains the single source of truth for the
    # returned wrench.  The derivative loop shares its per-rotor primitives.
    wrench = actuator_wrench(actuator_state, parameters, geometry)
    actual_thrust = np.zeros((6, 4), dtype=float)
    actual_gimbal_angle = np.zeros((6, 4), dtype=float)
    cog_offset = np.zeros((6, 3), dtype=float)
    force_effectiveness = np.zeros((6, 4), dtype=float)
    torque_effectiveness = np.zeros((6, 4), dtype=float)
    thrust_origins = geometry.thrust_origins(
        actuator_state.gimbal_angle
    )

    for rotor in range(4):
        thrust = float(actuator_state.thrust[rotor])
        effectiveness = float(parameters.force_effectiveness[rotor])
        angle = float(actuator_state.gimbal_angle[rotor])
        (
            force,
            unit_direction,
            unit_direction_derivative,
            force_gimbal_derivative,
        ) = _rotor_force_primitives(
            actual_thrust=thrust,
            force_effectiveness=effectiveness,
            gimbal_angle=angle,
            arm_yaw=float(geometry.arm_yaws[rotor]),
        )
        origin = thrust_origins[rotor] - parameters.cog_offset
        reaction_scale = (
            parameters.torque_effectiveness[rotor]
            * geometry.rotor_directions[rotor]
            * geometry.moment_force_rate
        )

        force_thrust_derivative = effectiveness * unit_direction
        actual_thrust[:3, rotor] = force_thrust_derivative
        actual_thrust[3:, rotor] = (
            np.cross(origin, force_thrust_derivative)
            + reaction_scale * force_thrust_derivative
        )

        origin_gimbal_derivative = (
            geometry.thrust_offset * unit_direction_derivative
        )
        actual_gimbal_angle[:3, rotor] = force_gimbal_derivative
        actual_gimbal_angle[3:, rotor] = (
            np.cross(origin_gimbal_derivative, force)
            + np.cross(origin, force_gimbal_derivative)
            + reaction_scale * force_gimbal_derivative
        )

        cog_offset[3:, :] += skew(force)

        force_effectiveness_derivative = thrust * unit_direction
        force_effectiveness[:3, rotor] = force_effectiveness_derivative
        force_effectiveness[3:, rotor] = (
            np.cross(origin, force_effectiveness_derivative)
            + reaction_scale * force_effectiveness_derivative
        )

        torque_effectiveness[3:, rotor] = (
            geometry.rotor_directions[rotor]
            * geometry.moment_force_rate
            * force
        )

    jacobian = ActuatorWrenchJacobian(
        actual_thrust=actual_thrust,
        actual_gimbal_angle=actual_gimbal_angle,
        cog_offset=cog_offset,
        force_effectiveness=force_effectiveness,
        torque_effectiveness=torque_effectiveness,
    )
    return wrench, jacobian


@dataclass(frozen=True)
class _ActuatorTransitionPrimitives:
    next_state: ActuatorState
    thrust_target: np.ndarray
    gimbal_target: np.ndarray
    thrust_fraction: float
    gimbal_fraction: float
    thrust_fraction_time_derivative: float
    gimbal_fraction_time_derivative: float
    thrust_command_derivative: np.ndarray
    gimbal_command_derivative: np.ndarray
    rate_input_derivative: np.ndarray
    rate_time_derivative: np.ndarray
    final_angle_derivative: np.ndarray
    active_set: Mapping[str, np.ndarray]


def _clip_with_active_set(
    value: np.ndarray,
    lower: float,
    upper: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(value, dtype=float)
    clipped = np.clip(values, lower, upper)
    lower_active = values <= lower
    upper_active = values >= upper
    derivative = (~lower_active & ~upper_active).astype(float)
    scale = max(1.0, abs(float(lower)), abs(float(upper)))
    tolerance = 64.0 * np.finfo(float).eps * scale
    near_kink = (
        (np.abs(values - lower) <= tolerance)
        | (np.abs(values - upper) <= tolerance)
    )
    return clipped, derivative, lower_active, upper_active, near_kink


def _response_fraction(time_step: float, time_constant: float) -> Tuple[float, float]:
    if time_constant <= 0.0:
        return 1.0, 0.0
    decay = float(np.exp(-time_step / time_constant))
    return 1.0 - decay, decay / time_constant


def _actuator_transition_primitives(
    state: ActuatorState,
    command: ActuatorCommand,
    parameters: ActuatorParameters,
    time_step: float,
) -> _ActuatorTransitionPrimitives:
    if not isinstance(state, ActuatorState):
        raise TypeError("state must be an ActuatorState")
    if not isinstance(command, ActuatorCommand):
        raise TypeError("command must be an ActuatorCommand")
    if not isinstance(parameters, ActuatorParameters):
        raise TypeError("parameters must be ActuatorParameters")
    dt = float(time_step)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("actuator time step must be positive")

    (
        target_thrust,
        thrust_command_derivative,
        thrust_lower_active,
        thrust_upper_active,
        thrust_near_kink,
    ) = _clip_with_active_set(
        command.thrust,
        parameters.minimum_thrust,
        parameters.maximum_thrust,
    )
    (
        target_gimbal,
        gimbal_command_derivative,
        command_gimbal_lower_active,
        command_gimbal_upper_active,
        command_gimbal_near_kink,
    ) = _clip_with_active_set(
        command.gimbal_angle,
        -parameters.maximum_gimbal_angle,
        parameters.maximum_gimbal_angle,
    )
    thrust_fraction, thrust_fraction_time_derivative = _response_fraction(
        dt,
        parameters.thrust_time_constant,
    )
    if parameters.thrust_time_constant <= 0.0:
        thrust = target_thrust.copy()
    else:
        thrust = state.thrust + thrust_fraction * (
            target_thrust - state.thrust
        )
    gimbal_fraction, gimbal_fraction_time_derivative = _response_fraction(
        dt,
        parameters.gimbal_time_constant,
    )
    if parameters.gimbal_time_constant <= 0.0:
        unconstrained_gimbal = target_gimbal.copy()
    else:
        unconstrained_gimbal = state.gimbal_angle + gimbal_fraction * (
            target_gimbal - state.gimbal_angle
        )
    maximum_step = parameters.maximum_gimbal_rate * dt
    (
        limited_step,
        rate_input_derivative,
        rate_lower_active,
        rate_upper_active,
        rate_near_kink,
    ) = _clip_with_active_set(
        unconstrained_gimbal - state.gimbal_angle,
        -maximum_step,
        maximum_step,
    )
    rate_time_derivative = np.where(
        rate_lower_active,
        -parameters.maximum_gimbal_rate,
        np.where(rate_upper_active, parameters.maximum_gimbal_rate, 0.0),
    )
    gimbal_before_angle_limit = state.gimbal_angle + limited_step
    (
        gimbal,
        final_angle_derivative,
        final_gimbal_lower_active,
        final_gimbal_upper_active,
        final_gimbal_near_kink,
    ) = _clip_with_active_set(
        gimbal_before_angle_limit,
        -parameters.maximum_gimbal_angle,
        parameters.maximum_gimbal_angle,
    )
    active_set = {
        "thrust_command_lower": thrust_lower_active,
        "thrust_command_upper": thrust_upper_active,
        "thrust_command_near_kink": thrust_near_kink,
        "gimbal_command_lower": command_gimbal_lower_active,
        "gimbal_command_upper": command_gimbal_upper_active,
        "gimbal_command_near_kink": command_gimbal_near_kink,
        "gimbal_rate_lower": rate_lower_active,
        "gimbal_rate_upper": rate_upper_active,
        "gimbal_rate_near_kink": rate_near_kink,
        "gimbal_angle_lower": final_gimbal_lower_active,
        "gimbal_angle_upper": final_gimbal_upper_active,
        "gimbal_angle_near_kink": final_gimbal_near_kink,
    }
    return _ActuatorTransitionPrimitives(
        next_state=ActuatorState(thrust, gimbal),
        thrust_target=target_thrust,
        gimbal_target=target_gimbal,
        thrust_fraction=thrust_fraction,
        gimbal_fraction=gimbal_fraction,
        thrust_fraction_time_derivative=thrust_fraction_time_derivative,
        gimbal_fraction_time_derivative=gimbal_fraction_time_derivative,
        thrust_command_derivative=thrust_command_derivative,
        gimbal_command_derivative=gimbal_command_derivative,
        rate_input_derivative=rate_input_derivative,
        rate_time_derivative=rate_time_derivative,
        final_angle_derivative=final_angle_derivative,
        active_set=active_set,
    )


def advance_actuators(
    state: ActuatorState,
    command: ActuatorCommand,
    parameters: ActuatorParameters,
    time_step: float,
) -> ActuatorState:
    """Advance the source-compatible piecewise actuator response."""

    return _actuator_transition_primitives(
        state,
        command,
        parameters,
        time_step,
    ).next_state


def advance_actuators_with_jacobian(
    state: ActuatorState,
    command: ActuatorCommand,
    parameters: ActuatorParameters,
    time_step: float,
) -> ActuatorTransitionEvaluation:
    """Advance actuators and return the exact active-branch derivatives.

    At an exact clamp boundary the saturated convention is used: the
    derivative with respect to the clamp input is zero and ``near_kink`` is
    true.  For an active rate limit, the time-step derivative follows the
    moving rate bound.
    """

    dt = float(time_step)
    primitive = _actuator_transition_primitives(
        state,
        command,
        parameters,
        dt,
    )
    thrust_previous_diagonal = np.full(
        4, 1.0 - primitive.thrust_fraction, dtype=float
    )
    thrust_command_diagonal = (
        primitive.thrust_fraction * primitive.thrust_command_derivative
    )
    thrust_time_step = (
        primitive.thrust_fraction_time_derivative
        * (primitive.thrust_target - state.thrust)
    )

    raw_gimbal_step_previous = np.full(
        4, -primitive.gimbal_fraction, dtype=float
    )
    raw_gimbal_step_command = (
        primitive.gimbal_fraction * primitive.gimbal_command_derivative
    )
    raw_gimbal_step_time = (
        primitive.gimbal_fraction_time_derivative
        * (primitive.gimbal_target - state.gimbal_angle)
    )
    limited_step_previous = (
        primitive.rate_input_derivative * raw_gimbal_step_previous
    )
    limited_step_command = (
        primitive.rate_input_derivative * raw_gimbal_step_command
    )
    limited_step_time = (
        primitive.rate_input_derivative * raw_gimbal_step_time
        + primitive.rate_time_derivative
    )
    gimbal_previous_diagonal = primitive.final_angle_derivative * (
        1.0 + limited_step_previous
    )
    gimbal_command_diagonal = (
        primitive.final_angle_derivative * limited_step_command
    )
    gimbal_time_step = primitive.final_angle_derivative * limited_step_time
    jacobian = ActuatorTransitionJacobian(
        thrust_previous=np.diag(thrust_previous_diagonal),
        thrust_command=np.diag(thrust_command_diagonal),
        thrust_time_step=thrust_time_step,
        gimbal_previous=np.diag(gimbal_previous_diagonal),
        gimbal_command=np.diag(gimbal_command_diagonal),
        gimbal_time_step=gimbal_time_step,
    )
    return ActuatorTransitionEvaluation(
        next_state=primitive.next_state,
        jacobian=jacobian,
        active_set=primitive.active_set,
    )


def _advance_plant_and_actuators(
    start_time: float,
    end_time: float,
    state: RigidBodyState,
    actuator_state: ActuatorState,
    command_history: ZeroOrderHoldCommandHistory,
    actuator_parameters: ActuatorParameters,
    plant: "FullSixDofPlant",
    interval_model_discrepancy_wrench: Optional[Sequence[float]] = None,
) -> Tuple[RigidBodyState, ActuatorState]:
    """Advance one controller interval without quantising actuator delay."""

    boundaries = [float(start_time), float(end_time)]
    boundaries.extend(command_history.switch_times(start_time, end_time))
    boundaries = sorted(set(boundaries))
    current_state = state
    current_actuators = actuator_state
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        step = right - left
        command = command_history.value_at(0.5 * (left + right))
        midpoint_actuators = advance_actuators(
            current_actuators, command, actuator_parameters, 0.5 * step
        )
        current_state = plant.step(
            left,
            current_state,
            midpoint_actuators,
            step,
            interval_model_discrepancy_wrench,
        )
        current_actuators = advance_actuators(
            midpoint_actuators, command, actuator_parameters, 0.5 * step
        )
    return current_state, current_actuators


def _normalise_quaternion_with_jacobian(
    quaternion_xyzw: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Canonical quaternion normalization and its active-branch derivative."""

    raw = np.asarray(quaternion_xyzw, dtype=float)
    if raw.shape != (4,) or not np.all(np.isfinite(raw)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(raw))
    if norm <= np.finfo(float).eps:
        raise ValueError("quaternion norm must be positive")
    unit = raw / norm
    sign = -1.0 if unit[3] < 0.0 else 1.0
    canonical = sign * unit
    jacobian = sign * (np.eye(4) - np.outer(unit, unit)) / norm
    return canonical, jacobian


def _quaternion_right_tangent_lift(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    """Map a right SO(3) tangent perturbation to an ``xyzw`` quaternion."""

    quaternion = normalise_quaternion(quaternion_xyzw)
    vector = quaternion[:3]
    scalar = float(quaternion[3])
    return 0.5 * np.vstack(
        (
            scalar * np.eye(3) + skew(vector),
            -vector[None, :],
        )
    )


def _quaternion_right_tangent_projection(
    quaternion_xyzw: Sequence[float],
) -> np.ndarray:
    """Map a tangent quaternion variation back to the right SO(3) chart."""

    # For a unit quaternion E^T E = I / 4, where E is the lift above.
    return 4.0 * _quaternion_right_tangent_lift(quaternion_xyzw).T


class FullSixDofPlant:
    """Rigid-body dynamics with optional future model-discrepancy wrench."""

    def __init__(
        self,
        parameters: VehicleParameters,
        geometry: GrapeGeometry,
        model_discrepancy_wrench: Optional[ModelDiscrepancyWrench] = None,
    ):
        self.parameters = parameters
        self.geometry = geometry
        self.model_discrepancy_wrench = model_discrepancy_wrench
        # Ridge/near-singular numerical states must not require an ordinary
        # matrix inverse.  VehicleParameters still validates the physical
        # inertia; pinv is used here only as the stable numerical operator.

    def total_body_wrench(
        self,
        time: float,
        state: RigidBodyState,
        actuators: ActuatorState,
        interval_model_discrepancy_wrench: Optional[
            Sequence[float]
        ] = None,
    ) -> np.ndarray:
        wrench = actuator_wrench(actuators, self.parameters, self.geometry)
        rotation = quaternion_to_matrix(state.orientation_xyzw)
        body_velocity = rotation.T @ state.linear_velocity
        wrench[:3] -= self.parameters.linear_drag * body_velocity
        wrench[3:] -= (
            self.parameters.angular_drag * state.angular_velocity
        )
        if self.model_discrepancy_wrench is not None:
            discrepancy = np.asarray(
                self.model_discrepancy_wrench(float(time), state), dtype=float
            )
            if discrepancy.shape != (6,) or not np.all(
                np.isfinite(discrepancy)
            ):
                raise ValueError(
                    "model discrepancy wrench must contain six finite values"
                )
            wrench += discrepancy
        if interval_model_discrepancy_wrench is not None:
            interval_discrepancy = np.asarray(
                interval_model_discrepancy_wrench, dtype=float
            )
            if (
                interval_discrepancy.shape != (6,)
                or not np.all(np.isfinite(interval_discrepancy))
            ):
                raise ValueError(
                    "interval model discrepancy wrench must contain six "
                    "finite values"
                )
            wrench += interval_discrepancy
        return wrench

    def derivative(
        self,
        time: float,
        state_vector: Sequence[float],
        actuators: ActuatorState,
        interval_model_discrepancy_wrench: Optional[
            Sequence[float]
        ] = None,
    ) -> np.ndarray:
        state = RigidBodyState.from_vector(state_vector)
        quaternion = state.orientation_xyzw
        rotation = quaternion_to_matrix(quaternion)
        wrench = self.total_body_wrench(
            time, state, actuators, interval_model_discrepancy_wrench
        )
        pure_omega = np.concatenate(
            (state.angular_velocity, np.asarray((0.0,), dtype=float))
        )
        quaternion_rate = 0.5 * quaternion_multiply(
            quaternion, pure_omega
        )
        linear_acceleration = (
            np.asarray((0.0, 0.0, -GRAVITY))
            + rotation @ wrench[:3] / self.parameters.mass
        )
        angular_acceleration = np.linalg.solve(
            self.parameters.inertia,
            wrench[3:]
            - np.cross(
                state.angular_velocity,
                self.parameters.inertia @ state.angular_velocity,
            ),
        )
        return np.concatenate(
            (
                state.linear_velocity,
                quaternion_rate,
                linear_acceleration,
                angular_acceleration,
            )
        )

    def derivative_with_jacobian(
        self,
        time: float,
        state_vector: Sequence[float],
        actuators: ActuatorState,
        interval_model_discrepancy_wrench: Optional[
            Sequence[float]
        ] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the continuous derivative and exact active-branch Jacobians.

        The state Jacobian is with respect to the ambient 13-vector used by
        the RK4 implementation.  Quaternion normalization performed by
        :class:`RigidBodyState` is differentiated explicitly.  The two
        actuator Jacobians are with respect to actual thrust and gimbal angle.

        A state-dependent ``model_discrepancy_wrench`` callback has no
        derivative contract and is therefore rejected here.  A fixed interval
        discrepancy wrench is constant with respect to the state and remains
        supported.
        """

        if self.model_discrepancy_wrench is not None:
            raise ValueError(
                "analytic plant Jacobian requires model_discrepancy_wrench=None"
            )
        raw_state = np.asarray(state_vector, dtype=float)
        if raw_state.shape != (13,) or not np.all(np.isfinite(raw_state)):
            raise ValueError("rigid-body state vector must contain 13 finite values")
        state = RigidBodyState.from_vector(raw_state)
        quaternion, quaternion_normalization = _normalise_quaternion_with_jacobian(
            raw_state[3:7]
        )
        rotation = quaternion_to_matrix(quaternion)
        velocity = state.linear_velocity
        omega = state.angular_velocity

        actuator_body_wrench, wrench_jacobian = actuator_wrench_with_jacobian(
            actuators, self.parameters, self.geometry
        )
        body_velocity = rotation.T @ velocity
        wrench = actuator_body_wrench.copy()
        wrench[:3] -= self.parameters.linear_drag * body_velocity
        wrench[3:] -= self.parameters.angular_drag * omega
        if interval_model_discrepancy_wrench is not None:
            discrepancy = np.asarray(interval_model_discrepancy_wrench, dtype=float)
            if discrepancy.shape != (6,) or not np.all(np.isfinite(discrepancy)):
                raise ValueError("interval model discrepancy wrench must contain six finite values")
            wrench += discrepancy

        pure_omega = np.concatenate((omega, np.asarray((0.0,), dtype=float)))
        quaternion_rate = 0.5 * quaternion_multiply(quaternion, pure_omega)
        linear_acceleration = (
            np.asarray((0.0, 0.0, -GRAVITY))
            + rotation @ wrench[:3] / self.parameters.mass
        )
        inertia = self.parameters.inertia
        inertia_omega = inertia @ omega
        angular_acceleration = np.linalg.solve(
            inertia, wrench[3:] - np.cross(omega, inertia_omega)
        )
        derivative = np.concatenate(
            (velocity, quaternion_rate, linear_acceleration, angular_acceleration)
        )

        state_jacobian = np.zeros((13, 13), dtype=float)
        thrust_jacobian = np.zeros((13, 4), dtype=float)
        gimbal_jacobian = np.zeros((13, 4), dtype=float)
        state_jacobian[:3, 7:10] = np.eye(3)

        quaternion_vector = quaternion[:3]
        quaternion_scalar = float(quaternion[3])
        right_pure_omega = np.block(
            [
                [-skew(omega), omega[:, None]],
                [-omega[None, :], np.zeros((1, 1), dtype=float)],
            ]
        )
        omega_quaternion_jacobian = np.vstack(
            (
                quaternion_scalar * np.eye(3) + skew(quaternion_vector),
                -quaternion_vector[None, :],
            )
        )
        state_jacobian[3:7, 3:7] = (
            0.5 * right_pure_omega @ quaternion_normalization
        )
        state_jacobian[3:7, 10:13] = 0.5 * omega_quaternion_jacobian

        quaternion_to_right_tangent = (
            _quaternion_right_tangent_projection(quaternion)
            @ quaternion_normalization
        )
        linear_drag = np.diag(self.parameters.linear_drag)
        acceleration_right_tangent = (
            rotation
            @ (-skew(wrench[:3]) - linear_drag @ skew(body_velocity))
            / self.parameters.mass
        )
        state_jacobian[7:10, 3:7] = (
            acceleration_right_tangent @ quaternion_to_right_tangent
        )
        state_jacobian[7:10, 7:10] = (
            -rotation @ linear_drag @ rotation.T / self.parameters.mass
        )
        thrust_jacobian[7:10] = (
            rotation @ wrench_jacobian.actual_thrust[:3] / self.parameters.mass
        )
        gimbal_jacobian[7:10] = (
            rotation @ wrench_jacobian.actual_gimbal_angle[:3] / self.parameters.mass
        )

        angular_drag = np.diag(self.parameters.angular_drag)
        state_jacobian[10:13, 10:13] = np.linalg.solve(
            inertia,
            -angular_drag + skew(inertia_omega) - skew(omega) @ inertia,
        )
        thrust_jacobian[10:13] = np.linalg.solve(
            inertia, wrench_jacobian.actual_thrust[3:]
        )
        gimbal_jacobian[10:13] = np.linalg.solve(
            inertia, wrench_jacobian.actual_gimbal_angle[3:]
        )
        return derivative, state_jacobian, thrust_jacobian, gimbal_jacobian

    def step(
        self,
        time: float,
        state: RigidBodyState,
        actuators: ActuatorState,
        time_step: float,
        interval_model_discrepancy_wrench: Optional[
            Sequence[float]
        ] = None,
    ) -> RigidBodyState:
        """Advance one controller interval with fourth-order Runge--Kutta."""

        dt = float(time_step)
        vector = state.as_vector()
        k1 = self.derivative(
            time, vector, actuators, interval_model_discrepancy_wrench
        )
        k2 = self.derivative(time + 0.5 * dt, vector + 0.5 * dt * k1,
                             actuators, interval_model_discrepancy_wrench)
        k3 = self.derivative(time + 0.5 * dt, vector + 0.5 * dt * k2,
                             actuators, interval_model_discrepancy_wrench)
        k4 = self.derivative(
            time + dt,
            vector + dt * k3,
            actuators,
            interval_model_discrepancy_wrench,
        )
        result = vector + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        result[3:7] = normalise_quaternion(result[3:7])
        return RigidBodyState.from_vector(result)

    def step_with_jacobian(
        self,
        time: float,
        state: RigidBodyState,
        actuators: ActuatorState,
        time_step: float,
        interval_model_discrepancy_wrench: Optional[
            Sequence[float]
        ] = None,
    ) -> RigidBodyStepEvaluation:
        """Advance RK4 and differentiate the implemented discrete step exactly."""

        dt = float(time_step)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("plant time step must be positive")
        vector = state.as_vector()
        identity = np.eye(13, dtype=float)

        k1, a1, bt1, bg1 = self.derivative_with_jacobian(
            time, vector, actuators, interval_model_discrepancy_wrench
        )
        k1_x = a1
        k1_t = bt1
        k1_g = bg1

        stage2 = vector + 0.5 * dt * k1
        stage2_x = identity + 0.5 * dt * k1_x
        stage2_t = 0.5 * dt * k1_t
        stage2_g = 0.5 * dt * k1_g
        k2, a2, bt2, bg2 = self.derivative_with_jacobian(
            time + 0.5 * dt,
            stage2,
            actuators,
            interval_model_discrepancy_wrench,
        )
        k2_x = a2 @ stage2_x
        k2_t = a2 @ stage2_t + bt2
        k2_g = a2 @ stage2_g + bg2

        stage3 = vector + 0.5 * dt * k2
        stage3_x = identity + 0.5 * dt * k2_x
        stage3_t = 0.5 * dt * k2_t
        stage3_g = 0.5 * dt * k2_g
        k3, a3, bt3, bg3 = self.derivative_with_jacobian(
            time + 0.5 * dt,
            stage3,
            actuators,
            interval_model_discrepancy_wrench,
        )
        k3_x = a3 @ stage3_x
        k3_t = a3 @ stage3_t + bt3
        k3_g = a3 @ stage3_g + bg3

        stage4 = vector + dt * k3
        stage4_x = identity + dt * k3_x
        stage4_t = dt * k3_t
        stage4_g = dt * k3_g
        k4, a4, bt4, bg4 = self.derivative_with_jacobian(
            time + dt,
            stage4,
            actuators,
            interval_model_discrepancy_wrench,
        )
        k4_x = a4 @ stage4_x
        k4_t = a4 @ stage4_t + bt4
        k4_g = a4 @ stage4_g + bg4

        raw_result = vector + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        ambient_state_jacobian = identity + dt / 6.0 * (
            k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x
        )
        ambient_thrust_jacobian = dt / 6.0 * (
            k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t
        )
        ambient_gimbal_jacobian = dt / 6.0 * (
            k1_g + 2.0 * k2_g + 2.0 * k3_g + k4_g
        )

        normalized_quaternion, output_normalization = (
            _normalise_quaternion_with_jacobian(raw_result[3:7])
        )
        output_normalization_matrix = np.eye(13, dtype=float)
        output_normalization_matrix[3:7, 3:7] = output_normalization
        ambient_state_jacobian = output_normalization_matrix @ ambient_state_jacobian
        ambient_thrust_jacobian = output_normalization_matrix @ ambient_thrust_jacobian
        ambient_gimbal_jacobian = output_normalization_matrix @ ambient_gimbal_jacobian
        raw_result[3:7] = normalized_quaternion
        next_state = RigidBodyState.from_vector(raw_result)

        input_lift = np.zeros((13, 12), dtype=float)
        input_lift[:3, :3] = np.eye(3)
        input_lift[3:7, 3:6] = _quaternion_right_tangent_lift(
            state.orientation_xyzw
        )
        input_lift[7:10, 6:9] = np.eye(3)
        input_lift[10:13, 9:12] = np.eye(3)

        output_projection = np.zeros((12, 13), dtype=float)
        output_projection[:3, :3] = np.eye(3)
        output_projection[3:6, 3:7] = _quaternion_right_tangent_projection(
            next_state.orientation_xyzw
        )
        output_projection[6:9, 7:10] = np.eye(3)
        output_projection[9:12, 10:13] = np.eye(3)

        return RigidBodyStepEvaluation(
            next_state=next_state,
            jacobian=RigidBodyStepJacobian(
                state=output_projection @ ambient_state_jacobian @ input_lift,
                actual_thrust=output_projection @ ambient_thrust_jacobian,
                actual_gimbal_angle=output_projection @ ambient_gimbal_jacobian,
            ),
        )


def simulate_closed_loop(
    times: Sequence[float],
    references: Sequence[ReferenceState],
    initial_state: RigidBodyState,
    initial_controller_state: ControllerState,
    controller: GrapeController,
    plant: FullSixDofPlant,
    actuator_parameters: ActuatorParameters,
    initial_actuator_state: Optional[ActuatorState] = None,
    interval_model_discrepancy_wrench: Optional[np.ndarray] = None,
) -> ClosedLoopTrajectory:
    """Run one continuous episode without any observation-state resets."""

    times = np.asarray(times, dtype=float)
    if (
        times.ndim != 1
        or times.size < 2
        or np.any(~np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("times must be strictly increasing")
    if len(references) != times.size:
        raise ValueError("one reference state is required per time")
    discrepancy_path = None
    if interval_model_discrepancy_wrench is not None:
        discrepancy_path = np.asarray(
            interval_model_discrepancy_wrench, dtype=float
        )
        if (
            discrepancy_path.shape != (times.size - 1, 6)
            or not np.all(np.isfinite(discrepancy_path))
        ):
            raise ValueError(
                "interval model discrepancy wrench must have shape "
                "(N - 1, 6)"
            )

    sample_count = times.size
    position = np.empty((sample_count, 3), dtype=float)
    orientation = np.empty((sample_count, 4), dtype=float)
    velocity = np.empty((sample_count, 3), dtype=float)
    omega = np.empty((sample_count, 3), dtype=float)
    integral = np.empty((sample_count, 6), dtype=float)
    commanded_thrust = np.empty((sample_count, 4), dtype=float)
    commanded_gimbal = np.empty((sample_count, 4), dtype=float)
    actual_thrust = np.empty((sample_count, 4), dtype=float)
    actual_gimbal = np.empty((sample_count, 4), dtype=float)
    body_wrench = np.empty((sample_count, 6), dtype=float)

    state = initial_state
    controller_state = initial_controller_state
    actuator_state = initial_actuator_state
    command_history = ZeroOrderHoldCommandHistory[ActuatorCommand](
        actuator_parameters.delay
    )
    for index, time in enumerate(times):
        dt = (
            times[index + 1] - time
            if index + 1 < sample_count
            else time - times[index - 1]
        )
        # Store c_k beside x_k.  The command call returns c_{k+1}; recording
        # it at index k would discard the initial controller state and shift
        # the complete controller trajectory by one sample.
        integral[index] = controller_state.integral_error
        command, next_controller_state = controller.step(
            state,
            references[index],
            controller_state,
            dt,
            None if actuator_state is None else actuator_state.gimbal_angle,
        )
        command_history.append(float(time), command)
        if actuator_state is None:
            actuator_state = ActuatorState(
                np.clip(
                    command.thrust,
                    actuator_parameters.minimum_thrust,
                    actuator_parameters.maximum_thrust,
                ),
                np.clip(
                    command.gimbal_angle,
                    -actuator_parameters.maximum_gimbal_angle,
                    actuator_parameters.maximum_gimbal_angle,
                ),
            )

        position[index] = state.position
        orientation[index] = state.orientation_xyzw
        velocity[index] = state.linear_velocity
        omega[index] = state.angular_velocity
        commanded_thrust[index] = command.thrust
        commanded_gimbal[index] = command.gimbal_angle
        actual_thrust[index] = actuator_state.thrust
        actual_gimbal[index] = actuator_state.gimbal_angle
        body_wrench[index] = plant.total_body_wrench(
            float(time),
            state,
            actuator_state,
            None
            if discrepancy_path is None
            else discrepancy_path[
                min(index, discrepancy_path.shape[0] - 1)
            ],
        )
        if index + 1 == sample_count:
            break

        controller_state = next_controller_state
        state, actuator_state = _advance_plant_and_actuators(
            start_time=float(time),
            end_time=float(times[index + 1]),
            state=state,
            actuator_state=actuator_state,
            command_history=command_history,
            actuator_parameters=actuator_parameters,
            plant=plant,
            interval_model_discrepancy_wrench=(
                None
                if discrepancy_path is None
                else discrepancy_path[index]
            ),
        )

    return ClosedLoopTrajectory(
        times=times,
        position=position,
        orientation_xyzw=orientation,
        linear_velocity=velocity,
        angular_velocity=omega,
        controller_integral=integral,
        commanded_thrust=commanded_thrust,
        commanded_gimbal_angle=commanded_gimbal,
        actuator_thrust=actual_thrust,
        actuator_gimbal_angle=actual_gimbal,
        body_wrench=body_wrench,
    )
