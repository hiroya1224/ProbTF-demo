"""Continuous full six-DoF plant and closed-loop trajectory forecast."""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

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


ResidualWrench = Callable[[float, RigidBodyState], np.ndarray]


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


def advance_actuators(
    state: ActuatorState,
    command: ActuatorCommand,
    parameters: ActuatorParameters,
    time_step: float,
) -> ActuatorState:
    dt = float(time_step)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("actuator time step must be positive")

    def response(current: np.ndarray, target: np.ndarray, tau: float):
        if tau <= 0.0:
            return target.copy()
        fraction = 1.0 - np.exp(-dt / tau)
        return current + fraction * (target - current)

    target_thrust = np.clip(
        command.thrust,
        parameters.minimum_thrust,
        parameters.maximum_thrust,
    )
    target_gimbal = np.clip(
        command.gimbal_angle,
        -parameters.maximum_gimbal_angle,
        parameters.maximum_gimbal_angle,
    )
    thrust = response(
        state.thrust,
        target_thrust,
        parameters.thrust_time_constant,
    )
    unconstrained_gimbal = response(
        state.gimbal_angle,
        target_gimbal,
        parameters.gimbal_time_constant,
    )
    maximum_step = parameters.maximum_gimbal_rate * dt
    gimbal = state.gimbal_angle + np.clip(
        unconstrained_gimbal - state.gimbal_angle,
        -maximum_step,
        maximum_step,
    )
    gimbal = np.clip(
        gimbal,
        -parameters.maximum_gimbal_angle,
        parameters.maximum_gimbal_angle,
    )
    return ActuatorState(thrust, gimbal)


def _advance_plant_and_actuators(
    start_time: float,
    end_time: float,
    state: RigidBodyState,
    actuator_state: ActuatorState,
    command_history: ZeroOrderHoldCommandHistory,
    actuator_parameters: ActuatorParameters,
    plant: "FullSixDofPlant",
    interval_residual_wrench: Optional[Sequence[float]] = None,
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
            interval_residual_wrench,
        )
        current_actuators = advance_actuators(
            midpoint_actuators, command, actuator_parameters, 0.5 * step
        )
    return current_state, current_actuators


class FullSixDofPlant:
    """Rigid-body dynamics with optional drag and additive residual wrench."""

    def __init__(
        self,
        parameters: VehicleParameters,
        geometry: GrapeGeometry,
        residual_wrench: Optional[ResidualWrench] = None,
    ):
        self.parameters = parameters
        self.geometry = geometry
        self.residual_wrench = residual_wrench
        self._inverse_inertia = np.linalg.inv(parameters.inertia)

    def total_body_wrench(
        self,
        time: float,
        state: RigidBodyState,
        actuators: ActuatorState,
        interval_residual_wrench: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        wrench = actuator_wrench(actuators, self.parameters, self.geometry)
        rotation = quaternion_to_matrix(state.orientation_xyzw)
        body_velocity = rotation.T @ state.linear_velocity
        wrench[:3] -= self.parameters.linear_drag * body_velocity
        wrench[3:] -= (
            self.parameters.angular_drag * state.angular_velocity
        )
        if self.residual_wrench is not None:
            residual = np.asarray(
                self.residual_wrench(float(time), state), dtype=float
            )
            if residual.shape != (6,) or not np.all(np.isfinite(residual)):
                raise ValueError("residual wrench must contain six finite values")
            wrench += residual
        if interval_residual_wrench is not None:
            interval_residual = np.asarray(
                interval_residual_wrench, dtype=float
            )
            if (
                interval_residual.shape != (6,)
                or not np.all(np.isfinite(interval_residual))
            ):
                raise ValueError(
                    "interval residual wrench must contain six finite values"
                )
            wrench += interval_residual
        return wrench

    def derivative(
        self,
        time: float,
        state_vector: Sequence[float],
        actuators: ActuatorState,
        interval_residual_wrench: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        state = RigidBodyState.from_vector(state_vector)
        quaternion = state.orientation_xyzw
        rotation = quaternion_to_matrix(quaternion)
        wrench = self.total_body_wrench(
            time, state, actuators, interval_residual_wrench
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
        angular_acceleration = self._inverse_inertia @ (
            wrench[3:]
            - np.cross(
                state.angular_velocity,
                self.parameters.inertia @ state.angular_velocity,
            )
        )
        return np.concatenate(
            (
                state.linear_velocity,
                quaternion_rate,
                linear_acceleration,
                angular_acceleration,
            )
        )

    def step(
        self,
        time: float,
        state: RigidBodyState,
        actuators: ActuatorState,
        time_step: float,
        interval_residual_wrench: Optional[Sequence[float]] = None,
    ) -> RigidBodyState:
        """Advance one controller interval with fourth-order Runge--Kutta."""

        dt = float(time_step)
        vector = state.as_vector()
        k1 = self.derivative(
            time, vector, actuators, interval_residual_wrench
        )
        k2 = self.derivative(time + 0.5 * dt, vector + 0.5 * dt * k1,
                             actuators, interval_residual_wrench)
        k3 = self.derivative(time + 0.5 * dt, vector + 0.5 * dt * k2,
                             actuators, interval_residual_wrench)
        k4 = self.derivative(
            time + dt,
            vector + dt * k3,
            actuators,
            interval_residual_wrench,
        )
        result = vector + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        result[3:7] = normalise_quaternion(result[3:7])
        return RigidBodyState.from_vector(result)


def simulate_closed_loop(
    times: Sequence[float],
    references: Sequence[ReferenceState],
    initial_state: RigidBodyState,
    initial_controller_state: ControllerState,
    controller: GrapeController,
    plant: FullSixDofPlant,
    actuator_parameters: ActuatorParameters,
    initial_actuator_state: Optional[ActuatorState] = None,
    interval_residual_wrench: Optional[np.ndarray] = None,
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
    residual_path = None
    if interval_residual_wrench is not None:
        residual_path = np.asarray(interval_residual_wrench, dtype=float)
        if (
            residual_path.shape != (times.size - 1, 6)
            or not np.all(np.isfinite(residual_path))
        ):
            raise ValueError(
                "interval residual wrench must have shape (N - 1, 6)"
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
            if residual_path is None
            else residual_path[min(index, residual_path.shape[0] - 1)],
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
            interval_residual_wrench=(
                None if residual_path is None else residual_path[index]
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
