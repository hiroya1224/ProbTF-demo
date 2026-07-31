"""Stateful Python port of the Grape six-axis PID and allocation path."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.geometry import (
    euler_xyz_to_matrix,
    matrix_to_euler_xyz,
    quaternion_to_matrix,
    wrap_angle,
)
from grape_param_estim.system import (
    ActuatorCommand,
    ControllerState,
    GRAVITY,
    GrapeGeometry,
    ReferenceState,
    RigidBodyState,
    VehicleParameters,
)


POSITION_CONTROL = "position"
VELOCITY_CONTROL = "velocity"
ACCELERATION_CONTROL = "acceleration"


@dataclass(frozen=True)
class PIDConfig:
    p_gain: float
    i_gain: float
    d_gain: float
    limit_sum: float = 1.0e6
    limit_p: float = 1.0e6
    limit_i: float = 1.0e6
    limit_d: float = 1.0e6
    limit_error_p: float = 1.0e6
    limit_error_i: float = 1.0e6
    limit_error_d: float = 1.0e6

    def __post_init__(self) -> None:
        values = np.asarray(tuple(self.__dict__.values()), dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("PID gains and limits must be finite and non-negative")


@dataclass(frozen=True)
class ControllerConfig:
    """Snapshot of the branch used by the Grape flight experiments."""

    pid: Tuple[PIDConfig, ...]
    xy_control_mode: str = POSITION_CONTROL
    need_yaw_d_control: bool = True
    start_roll_pitch_integration_height: float = 0.01
    initial_height: float = 0.0
    source_compatible_gyro_term: bool = True

    def __post_init__(self) -> None:
        if len(self.pid) != 6:
            raise ValueError("controller requires x, y, z, roll, pitch and yaw PID")
        if self.xy_control_mode not in (
            POSITION_CONTROL,
            VELOCITY_CONTROL,
            ACCELERATION_CONTROL,
        ):
            raise ValueError("unknown xy control mode")
        limits = np.asarray(
            (
                self.start_roll_pitch_integration_height,
            )
        )
        if np.any(~np.isfinite(limits)):
            raise ValueError("controller limits must be finite")

    @classmethod
    def grape(cls):
        """Load the non-simulation Grape YAML snapshot into pure Python."""

        xy = PIDConfig(4.0, 0.1, 2.0, 4.0, 12.0, 12.0, 12.0)
        z_axis = PIDConfig(
            5.0,
            1.0,
            2.5,
            25.0,
            25.0,
            25.0,
            25.0,
            limit_error_p=1.0,
        )
        roll_pitch = PIDConfig(
            13.0, 1.0, 20.0, 20.0, 20.0, 20.0, 20.0
        )
        yaw = PIDConfig(
            6.0,
            1.0,
            2.0,
            20.0,
            20.0,
            20.0,
            20.0,
            limit_error_p=0.4,
        )
        return cls(pid=(xy, xy, z_axis, roll_pitch, roll_pitch, yaw))


def initial_controller_state(
    configuration: ControllerConfig,
    trim_hover: bool = True,
) -> ControllerState:
    """Return a controller state; optionally initialise the hover integrator."""

    integral = np.zeros(6, dtype=float)
    if trim_hover and configuration.pid[2].i_gain > 0.0:
        integral[2] = GRAVITY / configuration.pid[2].i_gain
    # ``trim_hover=False`` represents the reset/take-off state used by the
    # C++ core: roll/pitch integration is enabled only after the height gate.
    return ControllerState(
        integral, roll_pitch_integration_active=bool(trim_hover)
    )


def _pid_update(
    configuration: PIDConfig,
    integral_error: float,
    error_p: float,
    integration_time: float,
    error_d: float,
    feedforward: float,
) -> Tuple[float, float]:
    error_p = float(
        np.clip(error_p, -configuration.limit_error_p,
                configuration.limit_error_p)
    )
    integral = float(
        np.clip(
            integral_error + error_p * integration_time,
            -configuration.limit_error_i,
            configuration.limit_error_i,
        )
    )
    error_d = float(
        np.clip(error_d, -configuration.limit_error_d,
                configuration.limit_error_d)
    )
    p_term = float(
        np.clip(
            error_p * configuration.p_gain,
            -configuration.limit_p,
            configuration.limit_p,
        )
    )
    i_term = float(
        np.clip(
            integral * configuration.i_gain,
            -configuration.limit_i,
            configuration.limit_i,
        )
    )
    d_term = float(
        np.clip(
            error_d * configuration.d_gain,
            -configuration.limit_d,
            configuration.limit_d,
        )
    )
    result = float(
        np.clip(
            p_term + i_term + d_term + feedforward,
            -configuration.limit_sum,
            configuration.limit_sum,
        )
    )
    return result, integral


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


def acceleration_allocation_matrix(
    parameters: VehicleParameters,
    geometry: GrapeGeometry,
    gimbal_angles: np.ndarray = None,
) -> np.ndarray:
    """Port the fully actuated one-gimbal-DoF allocation matrix."""

    wrench_map = np.zeros((6, 8), dtype=float)
    local_basis = (
        np.asarray((0.0, 1.0, 0.0)),
        np.asarray((0.0, 0.0, 1.0)),
    )
    inverse_inertia = np.linalg.inv(parameters.inertia)
    origins = geometry.thrust_origins(
        np.zeros(4) if gimbal_angles is None else gimbal_angles
    )
    for rotor in range(4):
        origin = origins[rotor]
        for component, local_force in enumerate(local_basis):
            column = 2 * rotor + component
            force = _rotate_z(local_force, geometry.arm_yaws[rotor])
            torque = np.cross(origin, force) + (
                geometry.rotor_directions[rotor]
                * geometry.moment_force_rate
                * force
            )
            wrench_map[:3, column] = force / parameters.mass
            wrench_map[3:, column] = inverse_inertia @ torque
    return wrench_map


def _source_pseudoinverse(matrix: np.ndarray) -> np.ndarray:
    """Match aerial_robot_model's absolute 1e-4 SVD threshold."""

    left, singular_values, right_transpose = np.linalg.svd(
        np.asarray(matrix, dtype=float), full_matrices=False
    )
    inverse = np.asarray(
        [1.0 / value if value > 1.0e-4 else 0.0
         for value in singular_values]
    )
    return right_transpose.T @ np.diag(inverse) @ left.T


class GrapeController:
    """Stateful six-axis PID followed by Grape rotor/gimbal allocation."""

    def __init__(
        self,
        configuration: ControllerConfig,
        nominal_parameters: VehicleParameters,
        geometry: GrapeGeometry,
        articulated_model: Optional[GrapeArticulatedModel] = None,
    ):
        self.configuration = configuration
        self.nominal_parameters = nominal_parameters
        self.geometry = geometry
        self.articulated_model = articulated_model
        self._allocation = acceleration_allocation_matrix(
            nominal_parameters, geometry
        )
        if np.linalg.matrix_rank(self._allocation) != 6:
            raise ValueError("Grape allocation matrix must span all six axes")
        self._allocation_inverse = _source_pseudoinverse(self._allocation)

    @property
    def allocation_matrix(self) -> np.ndarray:
        return self._allocation.copy()

    def step(
        self,
        state: RigidBodyState,
        reference: ReferenceState,
        controller_state: ControllerState,
        time_step: float,
        gimbal_angles: np.ndarray = None,
    ) -> Tuple[ActuatorCommand, ControllerState]:
        dt = float(time_step)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("controller time step must be positive")

        rotation = quaternion_to_matrix(state.orientation_xyzw)
        current_rpy = matrix_to_euler_xyz(rotation)
        target_rotation = euler_xyz_to_matrix(reference.rpy)
        target_omega = rotation.T @ target_rotation @ (
            reference.angular_velocity
        )
        integral = controller_state.integral_error.copy()
        result = np.zeros(6, dtype=float)

        for axis in (0, 1):
            if self.configuration.xy_control_mode == POSITION_CONTROL:
                error_p = reference.position[axis] - state.position[axis]
                error_d = (
                    reference.linear_velocity[axis]
                    - state.linear_velocity[axis]
                )
            elif self.configuration.xy_control_mode == VELOCITY_CONTROL:
                error_p = 0.0
                error_d = (
                    reference.linear_velocity[axis]
                    - state.linear_velocity[axis]
                )
            else:
                error_p = 0.0
                error_d = 0.0
            result[axis], integral[axis] = _pid_update(
                self.configuration.pid[axis],
                integral[axis],
                error_p,
                dt,
                error_d,
                reference.linear_acceleration[axis],
            )

        result[2], integral[2] = _pid_update(
            self.configuration.pid[2],
            integral[2],
            reference.position[2] - state.position[2],
            dt,
            reference.linear_velocity[2] - state.linear_velocity[2],
            reference.linear_acceleration[2],
        )
        integral[2] = max(0.0, integral[2])

        roll_pitch_was_active = (
            controller_state.roll_pitch_integration_active
        )
        roll_pitch_active = roll_pitch_was_active
        if (
            not roll_pitch_was_active
            and state.position[2] - self.configuration.initial_height
            > self.configuration.start_roll_pitch_integration_height
        ):
            roll_pitch_active = True
        # The live wrapper enables I control after evaluating the height gate,
        # but the activation tick itself still uses zero integration time.
        roll_pitch_dt = dt if roll_pitch_was_active else 0.0
        for axis in (3, 4):
            local_axis = axis - 3
            result[axis], integral[axis] = _pid_update(
                self.configuration.pid[axis],
                integral[axis],
                reference.rpy[local_axis] - current_rpy[local_axis],
                roll_pitch_dt,
                target_omega[local_axis] - state.angular_velocity[local_axis],
                reference.angular_acceleration[local_axis],
            )

        yaw_error_d = target_omega[2] - state.angular_velocity[2]
        if not self.configuration.need_yaw_d_control:
            yaw_error_d = target_omega[2]
        result[5], integral[5] = _pid_update(
            self.configuration.pid[5],
            integral[5],
            wrap_angle(reference.rpy[2] - current_rpy[2]),
            dt,
            yaw_error_d,
            reference.angular_acceleration[2],
        )

        desired = np.zeros(6, dtype=float)
        desired[:3] = rotation.T @ result[:3]
        desired[3:] = result[3:]
        allocation_parameters = self.nominal_parameters
        allocation_geometry = self.geometry
        allocation_angles = gimbal_angles
        if self.articulated_model is not None:
            angles = (
                np.zeros(4, dtype=float)
                if gimbal_angles is None
                else np.asarray(gimbal_angles, dtype=float)
            )
            allocation_parameters, allocation_geometry = (
                self.articulated_model.at(angles)
            )
            # The articulated geometry already contains the current thrust
            # origins and is expressed about the current aggregate CoG.
            allocation_angles = None
        gyro = np.cross(
            state.angular_velocity,
            allocation_parameters.inertia @ state.angular_velocity,
        )
        if self.configuration.source_compatible_gyro_term:
            # This deliberately mirrors gimbalrotor_controller.cpp.  The
            # source adds its gyro vector before applying the acceleration
            # allocation inverse; keeping it here is required by the golden
            # command test even though its units are unconventional.
            desired[3:] += gyro
        else:
            desired[3:] += np.linalg.solve(
                allocation_parameters.inertia, gyro
            )

        if gimbal_angles is None and self.articulated_model is None:
            allocation_inverse = self._allocation_inverse
        else:
            allocation_inverse = _source_pseudoinverse(
                acceleration_allocation_matrix(
                    allocation_parameters,
                    allocation_geometry,
                    allocation_angles,
                )
            )
        virtual_force = allocation_inverse @ desired
        thrust = np.empty(4, dtype=float)
        gimbal = np.empty(4, dtype=float)
        for rotor in range(4):
            lateral, axial = virtual_force[2 * rotor:2 * rotor + 2]
            thrust[rotor] = np.hypot(lateral, axial)
            gimbal[rotor] = np.arctan2(-lateral, axial)
        next_state = ControllerState(integral, roll_pitch_active)
        command = ActuatorCommand(thrust, gimbal, virtual_force, desired)
        return command, next_state
