"""Continuous full six-DoF plant and closed-loop trajectory forecast."""

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.controller import GrapeController
from grape_param_estim.geometry import (
    normalise_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
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


ResidualWrench = Callable[[float, RigidBodyState], np.ndarray]


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


def actuator_wrench(
    actuator_state: ActuatorState,
    parameters: VehicleParameters,
    geometry: GrapeGeometry,
) -> np.ndarray:
    """Map actual rotor thrust and gimbal angle to a body-frame wrench."""

    wrench = np.zeros(6, dtype=float)
    thrust_origins = geometry.thrust_origins(
        actuator_state.gimbal_angle
    )
    for rotor in range(4):
        angle = actuator_state.gimbal_angle[rotor]
        effective_thrust = (
            actuator_state.thrust[rotor]
            * parameters.force_effectiveness[rotor]
        )
        local_force = np.asarray(
            (
                0.0,
                -effective_thrust * np.sin(angle),
                effective_thrust * np.cos(angle),
            ),
            dtype=float,
        )
        force = _rotate_z(local_force, geometry.arm_yaws[rotor])
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
    return wrench


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


def _delayed_command(
    query_time: float,
    command_history: Sequence[Tuple[float, ActuatorCommand]],
    delay: float,
) -> ActuatorCommand:
    """Evaluate a zero-order-held command after a continuous time delay."""

    effective_time = float(query_time) - float(delay)
    selected = command_history[0][1]
    for issued_time, command in command_history:
        if issued_time <= effective_time + 1.0e-12:
            selected = command
        else:
            break
    return selected


def _advance_plant_and_actuators(
    start_time: float,
    end_time: float,
    state: RigidBodyState,
    actuator_state: ActuatorState,
    command_history: Sequence[Tuple[float, ActuatorCommand]],
    actuator_parameters: ActuatorParameters,
    plant: "FullSixDofPlant",
    interval_residual_wrench: Optional[Sequence[float]] = None,
) -> Tuple[RigidBodyState, ActuatorState]:
    """Advance one controller interval without quantising actuator delay."""

    boundaries = [float(start_time), float(end_time)]
    for issued_time, _command in command_history[1:]:
        switch_time = float(issued_time) + actuator_parameters.delay
        if start_time + 1.0e-12 < switch_time < end_time - 1.0e-12:
            boundaries.append(switch_time)
    boundaries = sorted(set(boundaries))
    current_state = state
    current_actuators = actuator_state
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        step = right - left
        command = _delayed_command(
            0.5 * (left + right),
            command_history,
            actuator_parameters.delay,
        )
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
    command_history: List[Tuple[float, ActuatorCommand]] = []
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
        command_history.append((float(time), command))
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
