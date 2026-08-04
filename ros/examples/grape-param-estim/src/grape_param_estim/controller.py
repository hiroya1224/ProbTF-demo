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
PID_UPDATE_INPUTS = (
    "integral_error",
    "error_p",
    "error_d",
    "feedforward",
    "time_step",
)

_PID_KINK_RELATIVE_TOLERANCE = 1.0e-9
_PID_CLAMP_REGIONS = ("interior", "lower", "upper")


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
class PIDClampState:
    """One piecewise-smooth clamp state and its derivative convention.

    Exact boundaries use the saturated ``lower`` or ``upper`` region, whose
    local slope is zero.  ``applied=False`` represents the optional
    non-negative-integral clamp being disabled.
    """

    applied: bool
    region: str
    near_kink: bool

    def __post_init__(self) -> None:
        if not isinstance(self.applied, (bool, np.bool_)):
            raise TypeError("applied must be boolean")
        if self.region not in _PID_CLAMP_REGIONS:
            raise ValueError("unknown PID clamp region")
        if not isinstance(self.near_kink, (bool, np.bool_)):
            raise TypeError("near_kink must be boolean")
        if not self.applied and (
            self.region != "interior" or self.near_kink
        ):
            raise ValueError("a disabled clamp must be an interior non-kink")

    @property
    def saturated(self) -> bool:
        return bool(self.applied and self.region != "interior")

    @property
    def slope(self) -> float:
        return 0.0 if self.saturated else 1.0


@dataclass(frozen=True)
class PIDActiveSet:
    """Active-set diagnostics for every clamp in one PID update."""

    error_p: PIDClampState
    error_d: PIDClampState
    integral: PIDClampState
    p_term: PIDClampState
    i_term: PIDClampState
    d_term: PIDClampState
    output_sum: PIDClampState
    nonnegative_integral: PIDClampState

    def __post_init__(self) -> None:
        for name in (
            "error_p",
            "error_d",
            "integral",
            "p_term",
            "i_term",
            "d_term",
            "output_sum",
            "nonnegative_integral",
        ):
            if not isinstance(getattr(self, name), PIDClampState):
                raise TypeError("{} must be a PIDClampState".format(name))

    @property
    def near_kink(self) -> bool:
        return any(
            getattr(self, name).near_kink
            for name in (
                "error_p",
                "error_d",
                "integral",
                "p_term",
                "i_term",
                "d_term",
                "output_sum",
                "nonnegative_integral",
            )
        )


@dataclass(frozen=True)
class PIDUpdateJacobian:
    """PID output rows in :data:`PID_UPDATE_INPUTS` column order."""

    output: np.ndarray
    next_integral: np.ndarray

    def __post_init__(self) -> None:
        for name in ("output", "next_integral"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (len(PID_UPDATE_INPUTS),) or not np.all(
                np.isfinite(value)
            ):
                raise ValueError(
                    "{} must contain five finite derivatives".format(name)
                )
            object.__setattr__(self, name, value.copy())


@dataclass(frozen=True)
class PIDUpdateResult:
    """Forward PID values, analytic Jacobian, and active-set diagnostics."""

    output: float
    next_integral: float
    jacobian: PIDUpdateJacobian
    active_set: PIDActiveSet

    def __post_init__(self) -> None:
        values = np.asarray((self.output, self.next_integral), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("PID update result must be finite")
        if not isinstance(self.jacobian, PIDUpdateJacobian):
            raise TypeError("jacobian must be a PIDUpdateJacobian")
        if not isinstance(self.active_set, PIDActiveSet):
            raise TypeError("active_set must be a PIDActiveSet")
        object.__setattr__(self, "output", float(self.output))
        object.__setattr__(self, "next_integral", float(self.next_integral))

    @property
    def near_kink(self) -> bool:
        return self.active_set.near_kink


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


def _near_pid_kink(value: float, boundary: float) -> bool:
    scale = max(1.0, abs(value), abs(boundary))
    return bool(
        abs(value - boundary)
        <= _PID_KINK_RELATIVE_TOLERANCE * scale
    )


def _symmetric_pid_clamp(
    value: float,
    limit: float,
) -> Tuple[float, PIDClampState]:
    clamped = float(np.clip(value, -limit, limit))
    if value <= -limit:
        region = "lower"
    elif value >= limit:
        region = "upper"
    else:
        region = "interior"
    near_kink = _near_pid_kink(value, -limit) or _near_pid_kink(
        value, limit
    )
    return clamped, PIDClampState(True, region, near_kink)


def _nonnegative_integral_clamp(
    value: float,
    applied: bool,
) -> Tuple[float, PIDClampState]:
    if not applied:
        return value, PIDClampState(False, "interior", False)
    next_integral = float(max(0.0, value))
    region = "lower" if value <= 0.0 else "interior"
    return next_integral, PIDClampState(
        True,
        region,
        _near_pid_kink(value, 0.0),
    )


def _pid_update_primitive(
    configuration: PIDConfig,
    integral_error: float,
    error_p: float,
    time_step: float,
    error_d: float,
    feedforward: float,
    nonnegative_integral: bool,
) -> PIDUpdateResult:
    if not isinstance(configuration, PIDConfig):
        raise TypeError("configuration must be a PIDConfig instance")
    if not isinstance(nonnegative_integral, (bool, np.bool_)):
        raise TypeError("nonnegative_integral must be boolean")
    inputs = np.asarray(
        (integral_error, error_p, error_d, feedforward, time_step),
        dtype=float,
    )
    if not np.all(np.isfinite(inputs)):
        raise ValueError("PID update inputs must be finite")
    if inputs[4] < 0.0:
        raise ValueError("PID update time_step must be non-negative")
    integral_error, error_p, error_d, feedforward, time_step = (
        float(value) for value in inputs
    )

    error_p, error_p_state = _symmetric_pid_clamp(
        error_p,
        configuration.limit_error_p,
    )
    error_p_jacobian = np.zeros(len(PID_UPDATE_INPUTS), dtype=float)
    error_p_jacobian[1] = error_p_state.slope

    integral_candidate = integral_error + error_p * time_step
    integral_candidate_jacobian = np.zeros(
        len(PID_UPDATE_INPUTS), dtype=float
    )
    integral_candidate_jacobian[0] = 1.0
    integral_candidate_jacobian += time_step * error_p_jacobian
    integral_candidate_jacobian[4] += error_p
    integral, integral_state = _symmetric_pid_clamp(
        integral_candidate,
        configuration.limit_error_i,
    )
    integral_jacobian = (
        integral_state.slope * integral_candidate_jacobian
    )

    error_d, error_d_state = _symmetric_pid_clamp(
        error_d,
        configuration.limit_error_d,
    )
    error_d_jacobian = np.zeros(len(PID_UPDATE_INPUTS), dtype=float)
    error_d_jacobian[2] = error_d_state.slope

    p_candidate = error_p * configuration.p_gain
    p_candidate_jacobian = configuration.p_gain * error_p_jacobian
    p_term, p_term_state = _symmetric_pid_clamp(
        p_candidate,
        configuration.limit_p,
    )
    p_term_jacobian = p_term_state.slope * p_candidate_jacobian

    i_candidate = integral * configuration.i_gain
    i_candidate_jacobian = configuration.i_gain * integral_jacobian
    i_term, i_term_state = _symmetric_pid_clamp(
        i_candidate,
        configuration.limit_i,
    )
    i_term_jacobian = i_term_state.slope * i_candidate_jacobian

    d_candidate = error_d * configuration.d_gain
    d_candidate_jacobian = configuration.d_gain * error_d_jacobian
    d_term, d_term_state = _symmetric_pid_clamp(
        d_candidate,
        configuration.limit_d,
    )
    d_term_jacobian = d_term_state.slope * d_candidate_jacobian

    output_candidate = p_term + i_term + d_term + feedforward
    output_candidate_jacobian = (
        p_term_jacobian + i_term_jacobian + d_term_jacobian
    )
    output_candidate_jacobian[3] += 1.0
    output, output_sum_state = _symmetric_pid_clamp(
        output_candidate,
        configuration.limit_sum,
    )
    output_jacobian = (
        output_sum_state.slope * output_candidate_jacobian
    )

    next_integral, nonnegative_integral_state = (
        _nonnegative_integral_clamp(
            integral,
            bool(nonnegative_integral),
        )
    )
    next_integral_jacobian = (
        nonnegative_integral_state.slope * integral_jacobian
    )

    active_set = PIDActiveSet(
        error_p=error_p_state,
        error_d=error_d_state,
        integral=integral_state,
        p_term=p_term_state,
        i_term=i_term_state,
        d_term=d_term_state,
        output_sum=output_sum_state,
        nonnegative_integral=nonnegative_integral_state,
    )
    jacobian = PIDUpdateJacobian(
        output=output_jacobian,
        next_integral=next_integral_jacobian,
    )
    return PIDUpdateResult(
        output=output,
        next_integral=next_integral,
        jacobian=jacobian,
        active_set=active_set,
    )


def pid_update_with_jacobian(
    configuration: PIDConfig,
    integral_error: float,
    error_p: float,
    time_step: float,
    error_d: float,
    feedforward: float,
    *,
    nonnegative_integral: bool = False,
) -> PIDUpdateResult:
    """Return one PID update and its analytic active-set derivative.

    Exact clamp boundaries use the saturated, zero-slope convention.  The
    Jacobian columns follow :data:`PID_UPDATE_INPUTS`; ``near_kink`` warns
    callers that the selected piece may change under a small perturbation.
    """

    return _pid_update_primitive(
        configuration=configuration,
        integral_error=integral_error,
        error_p=error_p,
        time_step=time_step,
        error_d=error_d,
        feedforward=feedforward,
        nonnegative_integral=nonnegative_integral,
    )


def _pid_update(
    configuration: PIDConfig,
    integral_error: float,
    error_p: float,
    time_step: float,
    error_d: float,
    feedforward: float,
    *,
    nonnegative_integral: bool = False,
) -> Tuple[float, float]:
    result = pid_update_with_jacobian(
        configuration=configuration,
        integral_error=integral_error,
        error_p=error_p,
        time_step=time_step,
        error_d=error_d,
        feedforward=feedforward,
        nonnegative_integral=nonnegative_integral,
    )
    return result.output, result.next_integral


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
            nonnegative_integral=True,
        )

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
