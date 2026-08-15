#!/usr/bin/env python3
"""Core single-bag SG rigid-body parameter estimation in SI units."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from scipy.linalg import expm, expm_frechet, null_space
from scipy.optimize import least_squares


# Make the repository's canonical implementation importable when an entry
# script is launched directly without a sourced catkin Python path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from grape_param_estim.dynamics import (  # noqa: E402
    actuator_wrench_with_jacobian,
    advance_actuators_with_jacobian,
)
from grape_param_estim.geometry import skew as canonical_skew  # noqa: E402
from grape_param_estim.system import (  # noqa: E402
    ActuatorCommand,
    ActuatorParameters,
    ActuatorState,
    GrapeGeometry,
    VehicleParameters,
)

try:  # noqa: E402
    from .savgol_trajectory import GeometricSavitzkyGolayPose, PoseSgEvaluation
    from .single_bag_savgol_covariance import (
        COVARIANCE_MODES,
        ParameterCovarianceResult,
        SgCovarianceEvaluation,
        build_sg_covariance,
        parameter_covariances,
    )
    from .smooth_command import QuinticSmoothZoh
except ImportError:  # pragma: no cover - direct CLI import
    from savgol_trajectory import (  # type: ignore
        GeometricSavitzkyGolayPose,
        PoseSgEvaluation,
    )
    from single_bag_savgol_covariance import (  # type: ignore
        COVARIANCE_MODES,
        ParameterCovarianceResult,
        SgCovarianceEvaluation,
        build_sg_covariance,
        parameter_covariances,
    )
    from smooth_command import QuinticSmoothZoh  # type: ignore


BASE_PLAN_COMMIT = "fb45718f1f9a4d3d4b94c35d4061fa17c07bd8d8"
DEFAULT_SMOOTH_MAX_NFEV = 2000
DEFAULT_STRICT_MAX_NFEV = 2000
PHYSICAL_DIMENSION = 14
GLOBAL_SPLIT_DIMENSION = 16
ROTOR_LAG_INDEX = 14
GIMBAL_LAG_INDEX = 15
GRAVITY_WORLD = np.asarray((0.0, 0.0, -9.80665), dtype=float)
INERTIA_COMPONENTS = (
    (0, 0),
    (1, 1),
    (2, 2),
    (0, 1),
    (0, 2),
    (1, 2),
)
SQRT_TWO = math.sqrt(2.0)
SYMMETRIC_BASIS = (
    np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
    np.asarray(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0))),
    np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
    np.asarray(
        (
            (0.0, 1.0 / SQRT_TWO, 0.0),
            (1.0 / SQRT_TWO, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
    ),
    np.asarray(
        (
            (0.0, 0.0, 1.0 / SQRT_TWO),
            (0.0, 0.0, 0.0),
            (1.0 / SQRT_TWO, 0.0, 0.0),
        )
    ),
    np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0 / SQRT_TWO),
            (0.0, 1.0 / SQRT_TWO, 0.0),
        )
    ),
)
COMMON_SCALE_DIRECTION = np.asarray(
    (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
    dtype=float,
)
ACTUATOR_ACTIVE_SET_NAMES = (
    "thrust_command_lower",
    "thrust_command_upper",
    "thrust_command_near_kink",
    "gimbal_command_lower",
    "gimbal_command_upper",
    "gimbal_command_near_kink",
    "gimbal_rate_lower",
    "gimbal_rate_upper",
    "gimbal_rate_near_kink",
    "gimbal_angle_lower",
    "gimbal_angle_upper",
    "gimbal_angle_near_kink",
)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or np.any(~np.isfinite(result)):
        raise ValueError("{} must contain {} finite values".format(name, size))
    return result


def _read_json_object(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = path.expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("{} JSON cannot be read: {}".format(label, source)) from error
    if not isinstance(value, dict):
        raise ValueError("{} JSON root must be an object".format(label))
    return source, value


@dataclass(frozen=True)
class VehicleModelInput:
    source_path: Path
    parameters: VehicleParameters
    geometry: GrapeGeometry
    raw: Mapping[str, Any]


def load_vehicle_model(path: Path) -> VehicleModelInput:
    """Load the existing vehicle-model JSON contract without a prior."""

    source, raw = _read_json_object(path, "vehicle model")
    required = (
        "mass_kg",
        "inertia_kg_m2",
        "cog_position_body_m",
        "force_effectiveness",
        "torque_effectiveness",
        "linear_drag",
        "angular_drag",
        "geometry",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError("vehicle model is missing: {}".format(", ".join(missing)))
    inertia = np.asarray(raw["inertia_kg_m2"], dtype=float)
    if (
        inertia.shape != (3, 3)
        or np.any(~np.isfinite(inertia))
        or not np.allclose(inertia, inertia.T, atol=1.0e-12, rtol=0.0)
    ):
        raise ValueError("inertia_kg_m2 must be a finite symmetric 3x3 matrix")
    parameters = VehicleParameters(
        mass=float(raw["mass_kg"]),
        inertia=inertia,
        cog_offset=_finite_vector(raw["cog_position_body_m"], 3, "cog"),
        force_effectiveness=_finite_vector(
            raw["force_effectiveness"], 4, "force_effectiveness"
        ),
        torque_effectiveness=_finite_vector(
            raw["torque_effectiveness"], 4, "torque_effectiveness"
        ),
        linear_drag=_finite_vector(raw["linear_drag"], 3, "linear_drag"),
        angular_drag=_finite_vector(raw["angular_drag"], 3, "angular_drag"),
    )
    geometry_raw = raw["geometry"]
    if not isinstance(geometry_raw, dict):
        raise ValueError("vehicle-model geometry must be an object")
    geometry = GrapeGeometry(
        rotor_origins=np.asarray(
            geometry_raw["rotor_origins_body_m"], dtype=float
        ),
        arm_yaws=_finite_vector(
            geometry_raw["arm_yaws_rad"], 4, "arm_yaws_rad"
        ),
        rotor_directions=_finite_vector(
            geometry_raw["rotor_directions"], 4, "rotor_directions"
        ),
        moment_force_rate=float(geometry_raw["moment_force_rate_m"]),
        thrust_offset=float(geometry_raw["thrust_offset_m"]),
    )
    return VehicleModelInput(source, parameters, geometry, raw)


@dataclass(frozen=True)
class PhysicalParameterJacobian:
    mass: np.ndarray
    inertia: np.ndarray
    cog_offset: np.ndarray
    force_effectiveness: np.ndarray


def physical_parameter_vector(parameters: VehicleParameters) -> np.ndarray:
    inertia = np.asarray(parameters.inertia, dtype=float)
    return np.concatenate(
        (
            np.asarray((parameters.mass,), dtype=float),
            np.asarray([inertia[row, column] for row, column in INERTIA_COMPONENTS]),
            np.asarray(parameters.cog_offset, dtype=float),
            np.asarray(parameters.force_effectiveness, dtype=float),
        )
    )


def physical_parameter_jacobian(value: PhysicalParameterJacobian) -> np.ndarray:
    rows = [np.asarray(value.mass, dtype=float)]
    rows.extend(value.inertia[row, column] for row, column in INERTIA_COMPONENTS)
    rows.extend(value.cog_offset[index] for index in range(3))
    rows.extend(value.force_effectiveness[index] for index in range(4))
    return np.vstack(rows)


class SiParameterChart:
    """14-D SI chart with a matrix-exponential second-moment coordinate."""

    def __init__(self, reference: VehicleParameters) -> None:
        if not isinstance(reference, VehicleParameters):
            raise TypeError("reference must be VehicleParameters")
        self.reference = reference
        inertia = np.asarray(reference.inertia, dtype=float)
        second_moment = 0.5 * float(np.trace(inertia)) * np.eye(3) - inertia
        second_moment = 0.5 * (second_moment + second_moment.T)
        eigenvalues, eigenvectors = np.linalg.eigh(second_moment)
        if np.any(~np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0.0):
            raise ValueError("reference inertia has a non-positive second moment")
        self.reference_second_moment = _readonly(second_moment)
        self.reference_second_moment_sqrt = _readonly(
            eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
        )

    @staticmethod
    def common_scale_direction() -> np.ndarray:
        return COMMON_SCALE_DIRECTION.copy()

    @staticmethod
    def _symmetric_log_matrix(coordinate: Sequence[float]) -> np.ndarray:
        value = np.asarray(coordinate, dtype=float)
        if value.shape != (6,):
            raise ValueError("second-moment coordinate must be 6-D")
        return sum(
            (float(coefficient) * basis for coefficient, basis in zip(value, SYMMETRIC_BASIS)),
            start=np.zeros((3, 3), dtype=float),
        )

    def decode_with_jacobian(
        self, coordinate: Sequence[float]
    ) -> tuple[VehicleParameters, PhysicalParameterJacobian]:
        value = np.asarray(coordinate, dtype=float)
        if value.shape != (PHYSICAL_DIMENSION,) or np.any(~np.isfinite(value)):
            raise ValueError("physical chart coordinate must be finite and 14-D")
        mass = float(self.reference.mass) * math.exp(float(value[0]))
        log_matrix = self._symmetric_log_matrix(value[1:7])
        exponential = expm(log_matrix)
        base = self.reference_second_moment_sqrt
        second_moment = base @ exponential @ base
        second_moment = 0.5 * (second_moment + second_moment.T)
        inertia = np.trace(second_moment) * np.eye(3) - second_moment
        inertia = 0.5 * (inertia + inertia.T)
        cog = np.asarray(self.reference.cog_offset) + value[7:10]
        effectiveness = np.asarray(self.reference.force_effectiveness) * np.exp(
            value[10:14]
        )
        parameters = VehicleParameters(
            mass=mass,
            inertia=inertia,
            cog_offset=cog,
            force_effectiveness=effectiveness,
            torque_effectiveness=self.reference.torque_effectiveness,
            linear_drag=self.reference.linear_drag,
            angular_drag=self.reference.angular_drag,
        )
        mass_jacobian = np.zeros(PHYSICAL_DIMENSION)
        mass_jacobian[0] = mass
        inertia_jacobian = np.zeros((3, 3, PHYSICAL_DIMENSION))
        for local_index, basis in enumerate(SYMMETRIC_BASIS):
            derivative = base @ expm_frechet(
                log_matrix, basis, compute_expm=False
            ) @ base
            derivative = 0.5 * (derivative + derivative.T)
            inertia_jacobian[:, :, 1 + local_index] = (
                np.trace(derivative) * np.eye(3) - derivative
            )
        cog_jacobian = np.zeros((3, PHYSICAL_DIMENSION))
        cog_jacobian[:, 7:10] = np.eye(3)
        effectiveness_jacobian = np.zeros((4, PHYSICAL_DIMENSION))
        effectiveness_jacobian[:, 10:14] = np.diag(effectiveness)
        return parameters, PhysicalParameterJacobian(
            mass_jacobian,
            inertia_jacobian,
            cog_jacobian,
            effectiveness_jacobian,
        )

    def decode(self, coordinate: Sequence[float]) -> VehicleParameters:
        return self.decode_with_jacobian(coordinate)[0]


@dataclass(frozen=True)
class ActuatorHistory:
    time: np.ndarray
    actual_thrust: np.ndarray
    actual_gimbal: np.ndarray
    actual_thrust_lag_jacobian: np.ndarray
    actual_gimbal_lag_jacobian: np.ndarray
    active_set_counts: Mapping[str, int]
    propagation_mode: str
    command_mode: str
    strict_final: bool

    def __post_init__(self) -> None:
        time_axis = np.asarray(self.time, dtype=float)
        count = time_axis.size
        thrust = np.asarray(self.actual_thrust, dtype=float)
        gimbal = np.asarray(self.actual_gimbal, dtype=float)
        thrust_lag = np.asarray(self.actual_thrust_lag_jacobian, dtype=float)
        gimbal_lag = np.asarray(self.actual_gimbal_lag_jacobian, dtype=float)
        if (
            time_axis.ndim != 1
            or count < 1
            or thrust.shape != (count, 4)
            or gimbal.shape != (count, 4)
            or thrust_lag.shape != (count, 4, 2)
            or gimbal_lag.shape != (count, 4, 2)
            or any(
                np.any(~np.isfinite(x))
                for x in (time_axis, thrust, gimbal, thrust_lag, gimbal_lag)
            )
            or self.propagation_mode not in ("stateful", "direct_command")
            or self.command_mode not in ("strict", "smooth")
        ):
            raise ValueError("actuator history is invalid")
        object.__setattr__(self, "time", _readonly(time_axis))
        object.__setattr__(self, "actual_thrust", _readonly(thrust))
        object.__setattr__(self, "actual_gimbal", _readonly(gimbal))
        object.__setattr__(
            self, "actual_thrust_lag_jacobian", _readonly(thrust_lag)
        )
        object.__setattr__(
            self, "actual_gimbal_lag_jacobian", _readonly(gimbal_lag)
        )
        object.__setattr__(
            self,
            "active_set_counts",
            {str(name): int(value) for name, value in self.active_set_counts.items()},
        )


def _command(thrust: np.ndarray, gimbal: np.ndarray) -> ActuatorCommand:
    return ActuatorCommand(
        thrust=np.asarray(thrust, dtype=float),
        gimbal_angle=np.asarray(gimbal, dtype=float),
        virtual_force=np.zeros(8),
        desired_acceleration=np.zeros(6),
    )


def _count_mask(counts: dict[str, int], name: str, mask: np.ndarray) -> None:
    counts[name] = counts.get(name, 0) + int(np.count_nonzero(mask))


def generate_actuator_history(
    *,
    time_axis: Sequence[float],
    rotor_history: QuinticSmoothZoh,
    gimbal_history: QuinticSmoothZoh,
    initial_gimbal: Sequence[float],
    rotor_lag_seconds: float,
    gimbal_lag_seconds: float,
    actuator_parameters: ActuatorParameters,
    propagation_mode: str = "stateful",
    command_mode: str = "strict",
    smooth_width_fraction: Optional[float] = None,
) -> ActuatorHistory:
    """Generate actual actuator states for fitting and standardized rollout."""

    times = np.asarray(time_axis, dtype=float)
    if (
        times.ndim != 1
        or times.size < 1
        or np.any(~np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
        or propagation_mode not in ("stateful", "direct_command")
        or command_mode not in ("strict", "smooth")
    ):
        raise ValueError("actuator-history request is invalid")
    if command_mode == "smooth" and (
        smooth_width_fraction is None
        or not np.isfinite(smooth_width_fraction)
        or smooth_width_fraction <= 0.0
    ):
        raise ValueError("smooth actuator history requires a positive width")
    initial = _finite_vector(initial_gimbal, 4, "initial_gimbal")
    thrust = np.empty((times.size, 4))
    gimbal = np.empty((times.size, 4))
    thrust_lag_jacobian = np.zeros((times.size, 4, 2))
    gimbal_lag_jacobian = np.zeros((times.size, 4, 2))
    counts: dict[str, int] = {}

    def complete_counts() -> dict[str, int]:
        return {name: int(counts.get(name, 0)) for name in ACTUATOR_ACTIVE_SET_NAMES}

    def command_at(
        query: float,
    ) -> tuple[ActuatorCommand, np.ndarray, np.ndarray]:
        if command_mode == "strict":
            rotor = rotor_history.exact_zoh(query, rotor_lag_seconds)
            gimbal_value = gimbal_history.exact_zoh(query, gimbal_lag_seconds)
            rotor_derivative = np.zeros(4)
            gimbal_derivative = np.zeros(4)
        else:
            assert smooth_width_fraction is not None
            rotor_evaluation = rotor_history.evaluate(
                query, rotor_lag_seconds, smooth_width_fraction
            )
            gimbal_evaluation = gimbal_history.evaluate(
                query, gimbal_lag_seconds, smooth_width_fraction
            )
            rotor = rotor_evaluation.value
            gimbal_value = gimbal_evaluation.value
            rotor_derivative = rotor_evaluation.delay_derivative
            gimbal_derivative = gimbal_evaluation.delay_derivative
        thrust_command_jacobian = np.zeros((4, 2))
        gimbal_command_jacobian = np.zeros((4, 2))
        thrust_command_jacobian[:, 0] = rotor_derivative
        gimbal_command_jacobian[:, 1] = gimbal_derivative
        return (
            _command(rotor, gimbal_value),
            thrust_command_jacobian,
            gimbal_command_jacobian,
        )

    if propagation_mode == "direct_command":
        for index, sample_time in enumerate(times):
            command, thrust_command_jacobian, gimbal_command_jacobian = (
                command_at(float(sample_time))
            )
            thrust[index] = np.clip(
                command.thrust,
                actuator_parameters.minimum_thrust,
                actuator_parameters.maximum_thrust,
            )
            gimbal[index] = np.clip(
                command.gimbal_angle,
                -actuator_parameters.maximum_gimbal_angle,
                actuator_parameters.maximum_gimbal_angle,
            )
            thrust_interior = (
                (command.thrust > actuator_parameters.minimum_thrust)
                & (command.thrust < actuator_parameters.maximum_thrust)
            )
            gimbal_interior = (
                (command.gimbal_angle > -actuator_parameters.maximum_gimbal_angle)
                & (command.gimbal_angle < actuator_parameters.maximum_gimbal_angle)
            )
            thrust_lag_jacobian[index] = (
                thrust_interior[:, None] * thrust_command_jacobian
            )
            gimbal_lag_jacobian[index] = (
                gimbal_interior[:, None] * gimbal_command_jacobian
            )
            _count_mask(
                counts,
                "thrust_command_lower",
                command.thrust <= actuator_parameters.minimum_thrust,
            )
            _count_mask(
                counts,
                "thrust_command_upper",
                command.thrust >= actuator_parameters.maximum_thrust,
            )
            _count_mask(
                counts,
                "gimbal_command_lower",
                command.gimbal_angle <= -actuator_parameters.maximum_gimbal_angle,
            )
            _count_mask(
                counts,
                "gimbal_command_upper",
                command.gimbal_angle >= actuator_parameters.maximum_gimbal_angle,
            )
        return ActuatorHistory(
            times,
            thrust,
            gimbal,
            thrust_lag_jacobian,
            gimbal_lag_jacobian,
            complete_counts(),
            propagation_mode,
            command_mode,
            command_mode == "strict",
        )

    raw_initial_thrust = rotor_history.exact_zoh(
        float(times[0]), rotor_lag_seconds
    )
    _count_mask(
        counts,
        "thrust_command_lower",
        raw_initial_thrust <= actuator_parameters.minimum_thrust,
    )
    _count_mask(
        counts,
        "thrust_command_upper",
        raw_initial_thrust >= actuator_parameters.maximum_thrust,
    )
    initial_thrust = np.clip(
        raw_initial_thrust,
        actuator_parameters.minimum_thrust,
        actuator_parameters.maximum_thrust,
    )
    state = ActuatorState(initial_thrust, initial)
    state_thrust_lag_jacobian = np.zeros((4, 2))
    state_gimbal_lag_jacobian = np.zeros((4, 2))
    thrust[0], gimbal[0] = state.thrust, state.gimbal_angle
    rotor_switches = np.asarray(rotor_history.times[1:]) + rotor_lag_seconds
    gimbal_switches = np.asarray(gimbal_history.times[1:]) + gimbal_lag_seconds
    for index, (left, right) in enumerate(zip(times[:-1], times[1:])):
        if command_mode == "strict":
            switches = np.concatenate(
                (
                    rotor_switches[(rotor_switches > left) & (rotor_switches < right)],
                    gimbal_switches[(gimbal_switches > left) & (gimbal_switches < right)],
                )
            )
            boundaries = np.concatenate(
                (np.asarray((left,)), np.unique(switches), np.asarray((right,)))
            )
        else:
            boundaries = np.asarray((left, 0.5 * (left + right), right))
        for sub_left, sub_right in zip(boundaries[:-1], boundaries[1:]):
            step = float(sub_right - sub_left)
            if step <= 0.0:
                continue
            command, thrust_command_jacobian, gimbal_command_jacobian = (
                command_at(0.5 * (float(sub_left) + float(sub_right)))
            )
            transition = advance_actuators_with_jacobian(
                state,
                command,
                actuator_parameters,
                step,
            )
            state_thrust_lag_jacobian = (
                transition.jacobian.thrust_previous
                @ state_thrust_lag_jacobian
                + transition.jacobian.thrust_command
                @ thrust_command_jacobian
            )
            state_gimbal_lag_jacobian = (
                transition.jacobian.gimbal_previous
                @ state_gimbal_lag_jacobian
                + transition.jacobian.gimbal_command
                @ gimbal_command_jacobian
            )
            state = transition.next_state
            for name, mask in transition.active_set.items():
                _count_mask(counts, name, mask)
        thrust[index + 1], gimbal[index + 1] = state.thrust, state.gimbal_angle
        thrust_lag_jacobian[index + 1] = state_thrust_lag_jacobian
        gimbal_lag_jacobian[index + 1] = state_gimbal_lag_jacobian
    return ActuatorHistory(
        times,
        thrust,
        gimbal,
        thrust_lag_jacobian,
        gimbal_lag_jacobian,
        complete_counts(),
        propagation_mode,
        command_mode,
        command_mode == "strict",
    )


@dataclass(frozen=True)
class SingleBagDataset:
    bag_id: str
    time: np.ndarray
    sg: PoseSgEvaluation
    covariance: SgCovarianceEvaluation
    reference_sg: PoseSgEvaluation
    reference_covariance: SgCovarianceEvaluation
    rotor_history: QuinticSmoothZoh
    gimbal_history: QuinticSmoothZoh
    initial_gimbal: np.ndarray
    pose_sensor_position_in_body: np.ndarray
    pose_sensor_to_body_rotation: np.ndarray
    gyro_sensor_position_in_body: np.ndarray
    body_to_gyro_sensor_rotation: np.ndarray
    gyro_bias: np.ndarray
    accelerometer_bias: np.ndarray
    measured_gyro: np.ndarray
    measured_specific_force: np.ndarray
    flight: Any = None

    def __post_init__(self) -> None:
        time_axis = np.asarray(self.time, dtype=float)
        count = time_axis.size
        expected = {
            "initial_gimbal": (4,),
            "pose_sensor_position_in_body": (3,),
            "pose_sensor_to_body_rotation": (3, 3),
            "gyro_sensor_position_in_body": (3,),
            "body_to_gyro_sensor_rotation": (3, 3),
            "gyro_bias": (3,),
            "accelerometer_bias": (3,),
            "measured_gyro": (count, 3),
            "measured_specific_force": (count, 3),
        }
        if time_axis.shape != self.sg.time.shape or not np.allclose(
            time_axis, self.sg.time, rtol=0.0, atol=0.0
        ):
            raise ValueError("dataset time and SG evaluation disagree")
        object.__setattr__(self, "time", _readonly(time_axis))
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape:
                raise ValueError("{} has invalid shape".format(name))
            # Missing diagnostic streams are represented by NaN, while every
            # quantity used by the objective is required to be finite.
            if name not in ("measured_gyro", "measured_specific_force") and np.any(
                ~np.isfinite(value)
            ):
                raise ValueError("{} must be finite".format(name))
            object.__setattr__(self, name, _readonly(value))


def _linear_interpolate(
    source_time: Sequence[float], source_value: np.ndarray, query: np.ndarray
) -> np.ndarray:
    times = np.asarray(source_time, dtype=float)
    values = np.asarray(source_value, dtype=float)
    if values.ndim != 2 or values.shape[0] != times.size:
        raise ValueError("interpolation source is invalid")
    return np.column_stack(
        [np.interp(query, times, values[:, column]) for column in range(values.shape[1])]
    )


def prepare_single_bag_dataset(
    *,
    flight: Any,
    window_seconds: float,
    degree: int,
    covariance_mode: str,
    geometric_correction: bool,
    maximum_lag_seconds: float,
) -> SingleBagDataset:
    """Prepare one loaded ``FlightData`` object; IMU remains diagnostic only."""

    if covariance_mode not in COVARIANCE_MODES:
        raise ValueError("unknown covariance mode")
    for name in ("rotor_command", "gimbal_command", "gimbal_position"):
        if getattr(flight, name) is None:
            raise ValueError("flight is missing {}".format(name))
    extrinsics = flight.sensor_extrinsics
    trajectory = GeometricSavitzkyGolayPose(
        time_axis=flight.pose.times,
        sensor_position=flight.pose.positions,
        sensor_orientation_xyzw=flight.pose.orientations_xyzw,
        pose_sensor_to_body_rotation=extrinsics.pose_sensor_to_body_rotation,
        window_seconds=window_seconds,
        degree=degree,
    )
    support_start = max(
        trajectory.valid_start_time,
        float(flight.rotor_command.all_times[0]) + float(maximum_lag_seconds),
        float(flight.gimbal_command.all_times[0]) + float(maximum_lag_seconds),
        float(flight.gimbal_position.times[0]),
    )
    support_end = trajectory.valid_end_time
    time_axis = trajectory.centered_raw_times(
        support_start=support_start,
        support_end=support_end,
        require_covariance_dof=True,
    )
    reference_sg = trajectory.evaluate(
        time_axis, centered=True, geometric_correction=True
    )
    sg = (
        reference_sg
        if geometric_correction
        else trajectory.evaluate(
            time_axis, centered=True, geometric_correction=False
        )
    )
    covariance = build_sg_covariance(
        sg,
        degree=degree,
        mode=covariance_mode,
        geometric_correction=geometric_correction,
    )
    reference_covariance = build_sg_covariance(
        reference_sg, degree=degree, mode="full", geometric_correction=True
    )
    initial_gimbal = _linear_interpolate(
        flight.gimbal_position.times,
        flight.gimbal_position.values,
        np.asarray((time_axis[0],)),
    )[0]

    def diagnostic(stream: Any) -> np.ndarray:
        if stream is None:
            return np.full((time_axis.size, 3), np.nan)
        return _linear_interpolate(stream.times, stream.values, time_axis)

    accel_bias = flight.imu_preflight.accelerometer_bias
    return SingleBagDataset(
        bag_id=str(flight.bag_id),
        time=time_axis,
        sg=sg,
        covariance=covariance,
        reference_sg=reference_sg,
        reference_covariance=reference_covariance,
        rotor_history=QuinticSmoothZoh(
            flight.rotor_command.all_times, flight.rotor_command.all_values
        ),
        gimbal_history=QuinticSmoothZoh(
            flight.gimbal_command.all_times, flight.gimbal_command.all_values
        ),
        initial_gimbal=initial_gimbal,
        pose_sensor_position_in_body=extrinsics.pose_sensor_position_in_body,
        pose_sensor_to_body_rotation=extrinsics.pose_sensor_to_body_rotation,
        gyro_sensor_position_in_body=extrinsics.gyro_sensor_position_in_body,
        body_to_gyro_sensor_rotation=extrinsics.body_to_gyro_sensor_rotation,
        gyro_bias=np.asarray(flight.imu_preflight.gyro_bias, dtype=float),
        accelerometer_bias=(
            np.zeros(3) if accel_bias is None else np.asarray(accel_bias, dtype=float)
        ),
        measured_gyro=diagnostic(flight.gyro),
        measured_specific_force=diagnostic(flight.accelerometer),
        flight=flight,
    )


@dataclass(frozen=True)
class DynamicsEvaluation:
    physical_coordinate: np.ndarray
    rotor_lag_seconds: float
    gimbal_lag_seconds: float
    parameters: VehicleParameters
    parameter_jacobian: PhysicalParameterJacobian
    actuator_history: ActuatorHistory
    acceleration_residual: np.ndarray
    acceleration_jacobian: np.ndarray
    acceleration_lag_jacobian: np.ndarray
    whitened_residual: np.ndarray
    whitened_jacobian: np.ndarray
    whitened_lag_jacobian: np.ndarray
    modeled_wrench: np.ndarray
    required_wrench: np.ndarray
    raw_residual_wrench: np.ndarray
    predicted_specific_acceleration: np.ndarray
    predicted_angular_acceleration: np.ndarray

    @property
    def residual_vector(self) -> np.ndarray:
        return self.whitened_residual.reshape(-1)

    @property
    def jacobian_matrix(self) -> np.ndarray:
        return self.whitened_jacobian.reshape(-1, PHYSICAL_DIMENSION)

    @property
    def cost(self) -> float:
        residual = self.residual_vector
        return 0.5 * float(residual @ residual)


class SingleBagDynamicsProblem:
    """Prior-free Newton--Euler acceleration objective for exactly one bag."""

    def __init__(
        self,
        dataset: SingleBagDataset,
        model: VehicleModelInput,
        actuator_parameters: ActuatorParameters,
        *,
        actuator_propagation: str = "stateful",
    ) -> None:
        if actuator_propagation not in ("stateful", "direct_command"):
            raise ValueError("unknown actuator propagation mode")
        self.dataset = dataset
        self.model = model
        self.actuator_parameters = actuator_parameters
        self.actuator_propagation = actuator_propagation
        self.chart = SiParameterChart(model.parameters)

    def actuator_history(
        self,
        rotor_lag_seconds: float,
        gimbal_lag_seconds: float,
        *,
        command_mode: str,
        smooth_width_fraction: Optional[float] = None,
    ) -> ActuatorHistory:
        return generate_actuator_history(
            time_axis=self.dataset.time,
            rotor_history=self.dataset.rotor_history,
            gimbal_history=self.dataset.gimbal_history,
            initial_gimbal=self.dataset.initial_gimbal,
            rotor_lag_seconds=rotor_lag_seconds,
            gimbal_lag_seconds=gimbal_lag_seconds,
            actuator_parameters=self.actuator_parameters,
            propagation_mode=self.actuator_propagation,
            command_mode=command_mode,
            smooth_width_fraction=smooth_width_fraction,
        )

    def _evaluate_analytic(
        self,
        coordinate: np.ndarray,
        rotor_lag_seconds: float,
        gimbal_lag_seconds: float,
        *,
        command_mode: str,
        smooth_width_fraction: Optional[float],
        reference: bool,
    ) -> DynamicsEvaluation:
        parameters, parameter_jacobian = self.chart.decode_with_jacobian(coordinate)
        history = self.actuator_history(
            rotor_lag_seconds,
            gimbal_lag_seconds,
            command_mode=command_mode,
            smooth_width_fraction=smooth_width_fraction,
        )
        sg = self.dataset.reference_sg if reference else self.dataset.sg
        covariance = (
            self.dataset.reference_covariance if reference else self.dataset.covariance
        )
        count = self.dataset.time.size
        residual = np.empty((count, 6))
        residual_jacobian = np.empty((count, 6, PHYSICAL_DIMENSION))
        residual_lag_jacobian = np.empty((count, 6, 2))
        modeled = np.empty((count, 6))
        required = np.empty((count, 6))
        predicted_s = np.empty((count, 3))
        predicted_alpha = np.empty((count, 3))
        lever = (
            np.asarray(self.dataset.pose_sensor_position_in_body)
            - np.asarray(parameters.cog_offset)
        )
        lever_jacobian = -np.asarray(parameter_jacobian.cog_offset)
        for index in range(count):
            omega = np.asarray(sg.body_angular_velocity[index])
            observed_alpha = np.asarray(sg.body_angular_acceleration[index])
            state = ActuatorState(
                history.actual_thrust[index], history.actual_gimbal[index]
            )
            wrench, actuator_jacobian = actuator_wrench_with_jacobian(
                state, parameters, self.model.geometry
            )
            wrench_jacobian = (
                actuator_jacobian.cog_offset @ parameter_jacobian.cog_offset
                + actuator_jacobian.force_effectiveness
                @ parameter_jacobian.force_effectiveness
            )
            wrench_lag_jacobian = (
                actuator_jacobian.actual_thrust
                @ history.actual_thrust_lag_jacobian[index]
                + actuator_jacobian.actual_gimbal_angle
                @ history.actual_gimbal_lag_jacobian[index]
            )
            force, torque = wrench[:3], wrench[3:]
            force_jacobian, torque_jacobian = (
                wrench_jacobian[:3],
                wrench_jacobian[3:],
            )
            inertia = np.asarray(parameters.inertia)
            inertia_jacobian = np.asarray(parameter_jacobian.inertia)
            angular_rhs = torque - np.cross(omega, inertia @ omega)
            alpha_hat = np.linalg.solve(inertia, angular_rhs)
            inertia_omega_jacobian = np.einsum(
                "ijk,j->ik", inertia_jacobian, omega
            )
            inertia_alpha_jacobian = np.einsum(
                "ijk,j->ik", inertia_jacobian, alpha_hat
            )
            alpha_jacobian = np.linalg.solve(
                inertia,
                torque_jacobian
                - canonical_skew(omega) @ inertia_omega_jacobian
                - inertia_alpha_jacobian,
            )
            alpha_lag_jacobian = np.linalg.solve(
                inertia, wrench_lag_jacobian[3:]
            )
            centripetal_matrix = canonical_skew(omega) @ canonical_skew(omega)
            sensor_specific = (
                force / parameters.mass
                + np.cross(alpha_hat, lever)
                + centripetal_matrix @ lever
            )
            sensor_specific_jacobian = (
                force_jacobian / parameters.mass
                - np.outer(force, parameter_jacobian.mass) / parameters.mass**2
                - canonical_skew(lever) @ alpha_jacobian
                + (canonical_skew(alpha_hat) + centripetal_matrix)
                @ lever_jacobian
            )
            sensor_specific_lag_jacobian = (
                wrench_lag_jacobian[:3] / parameters.mass
                - canonical_skew(lever) @ alpha_lag_jacobian
            )
            residual[index, :3] = covariance.z[index, :3] - sensor_specific
            residual[index, 3:] = covariance.z[index, 3:] - alpha_hat
            residual_jacobian[index, :3] = -sensor_specific_jacobian
            residual_jacobian[index, 3:] = -alpha_jacobian
            residual_lag_jacobian[index, :3] = -sensor_specific_lag_jacobian
            residual_lag_jacobian[index, 3:] = -alpha_lag_jacobian
            modeled[index] = wrench
            required_force = parameters.mass * (
                covariance.z[index, :3]
                - np.cross(observed_alpha, lever)
                - centripetal_matrix @ lever
            )
            required_torque = (
                inertia @ observed_alpha
                + np.cross(omega, inertia @ omega)
            )
            required[index] = np.concatenate((required_force, required_torque))
            predicted_s[index] = sensor_specific
            predicted_alpha[index] = alpha_hat
        whitened_residual = np.einsum(
            "nij,nj->ni", covariance.whitening, residual
        )
        whitened_jacobian = np.einsum(
            "nij,njk->nik", covariance.whitening, residual_jacobian
        )
        whitened_lag_jacobian = np.einsum(
            "nij,njk->nik", covariance.whitening, residual_lag_jacobian
        )
        arrays = (
            residual,
            residual_jacobian,
            residual_lag_jacobian,
            whitened_residual,
            whitened_jacobian,
            whitened_lag_jacobian,
            modeled,
            required,
            predicted_s,
            predicted_alpha,
        )
        if any(np.any(~np.isfinite(value)) for value in arrays):
            raise FloatingPointError("Newton--Euler evaluation became non-finite")
        return DynamicsEvaluation(
            physical_coordinate=_readonly(coordinate),
            rotor_lag_seconds=float(rotor_lag_seconds),
            gimbal_lag_seconds=float(gimbal_lag_seconds),
            parameters=parameters,
            parameter_jacobian=parameter_jacobian,
            actuator_history=history,
            acceleration_residual=_readonly(residual),
            acceleration_jacobian=_readonly(residual_jacobian),
            acceleration_lag_jacobian=_readonly(residual_lag_jacobian),
            whitened_residual=_readonly(whitened_residual),
            whitened_jacobian=_readonly(whitened_jacobian),
            whitened_lag_jacobian=_readonly(whitened_lag_jacobian),
            modeled_wrench=_readonly(modeled),
            required_wrench=_readonly(required),
            raw_residual_wrench=_readonly(required - modeled),
            predicted_specific_acceleration=_readonly(predicted_s),
            predicted_angular_acceleration=_readonly(predicted_alpha),
        )

    def evaluate_physical(
        self,
        coordinate: Sequence[float],
        rotor_lag_seconds: float,
        gimbal_lag_seconds: float,
        *,
        command_mode: str = "strict",
        smooth_width_fraction: Optional[float] = None,
        reference: bool = False,
    ) -> DynamicsEvaluation:
        value = np.asarray(coordinate, dtype=float)
        return self._evaluate_analytic(
            value,
            rotor_lag_seconds,
            gimbal_lag_seconds,
            command_mode=command_mode,
            smooth_width_fraction=smooth_width_fraction,
            reference=reference,
        )

    def global_residual_jacobian(
        self,
        coordinate: Sequence[float],
        *,
        lag_layout: str,
        command_mode: str,
        smooth_width_fraction: Optional[float],
    ) -> tuple[np.ndarray, np.ndarray, DynamicsEvaluation]:
        value = np.asarray(coordinate, dtype=float)
        if lag_layout == "split" and value.shape == (16,):
            rotor_lag, gimbal_lag = float(value[14]), float(value[15])
        elif lag_layout == "common" and value.shape == (15,):
            rotor_lag = gimbal_lag = float(value[14])
        else:
            raise ValueError("global coordinate/lag layout mismatch")
        evaluation = self.evaluate_physical(
            value[:14],
            rotor_lag,
            gimbal_lag,
            command_mode=command_mode,
            smooth_width_fraction=smooth_width_fraction,
        )
        dimension = value.size
        jacobian = np.zeros((evaluation.residual_vector.size, dimension))
        jacobian[:, :14] = evaluation.jacobian_matrix
        lag_jacobian = evaluation.whitened_lag_jacobian.reshape(-1, 2)
        if lag_layout == "split":
            jacobian[:, 14:16] = lag_jacobian
        else:
            jacobian[:, 14] = lag_jacobian[:, 0] + lag_jacobian[:, 1]
        return evaluation.residual_vector, jacobian, evaluation


@dataclass(frozen=True)
class LmSettings:
    initial_damping: float = 1.0e-3
    initial_trust_radius: float = 1.0
    maximum_trust_radius: float = 8.0
    minimum_trust_radius: float = 1.0e-10
    acceptance_ratio: float = 1.0e-4
    gtol: float = 1.0e-8
    ftol: float = math.sqrt(np.finfo(float).eps)
    xtol: float = math.sqrt(np.finfo(float).eps)

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.initial_damping,
                self.initial_trust_radius,
                self.maximum_trust_radius,
                self.minimum_trust_radius,
                self.acceptance_ratio,
                self.gtol,
                self.ftol,
                self.xtol,
            )
        )
        if (
            np.any(~np.isfinite(values))
            or np.any(values <= 0.0)
            or self.maximum_trust_radius < self.initial_trust_radius
            or self.initial_trust_radius < self.minimum_trust_radius
        ):
            raise ValueError("LM settings are invalid")


@dataclass(frozen=True)
class ObjectiveEvaluation:
    residual: np.ndarray
    jacobian: np.ndarray
    payload: Any = None


@dataclass(frozen=True)
class SolverResult:
    coordinate: np.ndarray
    evaluation: ObjectiveEvaluation
    success: bool
    status: str
    message: str
    nfev: int
    elapsed_seconds: float
    diagnostics: tuple[Mapping[str, Any], ...] = ()


def solve_kkt_lm_step(
    jacobian: np.ndarray,
    residual: np.ndarray,
    damping: float,
    gauge_direction: Optional[np.ndarray],
) -> tuple[np.ndarray, float, float, float]:
    """One LM normal-equation step with an optional exact KKT section."""

    value = np.asarray(jacobian, dtype=float)
    r = np.asarray(residual, dtype=float)
    hessian = value.T @ value
    gradient = value.T @ r
    dimension = hessian.shape[0]
    regularized = hessian + float(damping) * np.eye(dimension)
    if gauge_direction is None:
        step = np.linalg.solve(regularized, -gradient)
        multiplier = 0.0
        violation = 0.0
    else:
        direction = np.asarray(gauge_direction, dtype=float)
        kkt = np.zeros((dimension + 1, dimension + 1))
        kkt[:dimension, :dimension] = regularized
        kkt[:dimension, dimension] = direction
        kkt[dimension, :dimension] = direction
        solution = np.linalg.solve(
            kkt, np.concatenate((-gradient, np.asarray((0.0,))))
        )
        step = solution[:dimension]
        multiplier = float(solution[-1])
        violation = abs(float(direction @ step))
    predicted = -float(gradient @ step + 0.5 * step @ hessian @ step)
    return step, predicted, multiplier, violation


def _project_gauge(value: np.ndarray, direction: Optional[np.ndarray]) -> np.ndarray:
    if direction is None:
        return np.asarray(value, dtype=float).copy()
    unit = direction / np.linalg.norm(direction)
    return np.asarray(value, dtype=float) - unit * float(unit @ value)


def adaptive_kkt_lm(
    evaluator: Callable[[np.ndarray], ObjectiveEvaluation],
    initial: Sequence[float],
    *,
    settings: LmSettings,
    max_nfev: int,
    gauge_direction: Optional[Sequence[float]],
    lower: Optional[Sequence[float]] = None,
    upper: Optional[Sequence[float]] = None,
) -> SolverResult:
    """Custom trust-region LM; damping never enters reported information."""

    started = time.perf_counter()
    coordinate = np.asarray(initial, dtype=float).copy()
    dimension = coordinate.size
    direction = None
    if gauge_direction is not None:
        direction = np.asarray(gauge_direction, dtype=float)
        if direction.shape != (dimension,) or np.linalg.norm(direction) == 0.0:
            raise ValueError("gauge direction has invalid dimension")
        direction = direction / np.linalg.norm(direction)
        coordinate = _project_gauge(coordinate, direction)
    lower_value = (
        np.full(dimension, -np.inf) if lower is None else np.asarray(lower, dtype=float)
    )
    upper_value = (
        np.full(dimension, np.inf) if upper is None else np.asarray(upper, dtype=float)
    )
    if lower_value.shape != coordinate.shape or upper_value.shape != coordinate.shape:
        raise ValueError("LM bound dimension mismatch")
    if np.any(coordinate < lower_value) or np.any(coordinate > upper_value):
        raise ValueError("initial point violates explicit lag bounds")
    current = evaluator(coordinate)
    nfev = 1
    cost = 0.5 * float(current.residual @ current.residual)
    diagonal_scale = max(
        float(np.max(np.diag(current.jacobian.T @ current.jacobian))), 1.0
    )
    damping = settings.initial_damping * diagonal_scale
    trust = settings.initial_trust_radius
    diagnostics: list[Mapping[str, Any]] = []
    success, status, message = False, "max_nfev", "maximum evaluations exceeded"
    while nfev < int(max_nfev):
        gradient = current.jacobian.T @ current.residual
        projected_gradient = _project_gauge(gradient, direction)
        if np.linalg.norm(projected_gradient, ord=np.inf) <= settings.gtol:
            success, status, message = True, "gtol", "gradient tolerance satisfied"
            break
        accepted = False
        old_cost = cost
        for _ in range(48):
            try:
                step, predicted, multiplier, violation = solve_kkt_lm_step(
                    current.jacobian,
                    current.residual,
                    damping,
                    direction,
                )
            except np.linalg.LinAlgError:
                damping *= 10.0
                trust = max(settings.minimum_trust_radius, 0.5 * trust)
                continue
            step_norm = float(np.linalg.norm(step))
            if (
                not np.isfinite(step_norm)
                or step_norm > trust
                or not np.isfinite(predicted)
                or predicted <= 0.0
            ):
                damping *= 4.0
                continue
            trial_coordinate = _project_gauge(coordinate + step, direction)
            if np.any(trial_coordinate < lower_value) or np.any(
                trial_coordinate > upper_value
            ):
                damping *= 4.0
                trust = max(settings.minimum_trust_radius, 0.5 * trust)
                continue
            try:
                trial = evaluator(trial_coordinate)
                nfev += 1
                trial_cost = 0.5 * float(trial.residual @ trial.residual)
                if not np.isfinite(trial_cost):
                    raise FloatingPointError("non-finite trial cost")
            except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
                nfev += 1
                diagnostics.append(
                    {
                        "accepted": False,
                        "reason": "trial_evaluation_failure",
                        "exception_type": type(error).__name__,
                    }
                )
                damping *= 4.0
                trust = max(settings.minimum_trust_radius, 0.5 * trust)
                if nfev >= int(max_nfev):
                    break
                continue
            actual = old_cost - trial_cost
            ratio = actual / predicted
            accepted = bool(actual > 0.0 and ratio > settings.acceptance_ratio)
            diagnostics.append(
                {
                    "accepted": accepted,
                    "cost_before": old_cost,
                    "cost_after": trial_cost,
                    "actual_reduction": actual,
                    "predicted_reduction": predicted,
                    "reduction_ratio": ratio,
                    "damping": damping,
                    "trust_radius": trust,
                    "step_l2": step_norm,
                    "kkt_multiplier": multiplier,
                    "gauge_step_constraint_abs": violation,
                }
            )
            if not accepted:
                damping *= 4.0
                trust = max(settings.minimum_trust_radius, 0.5 * trust)
                continue
            coordinate, current, cost = trial_coordinate, trial, trial_cost
            if ratio > 0.75:
                damping = max(damping / 3.0, np.finfo(float).tiny)
                trust = min(
                    settings.maximum_trust_radius,
                    max(trust, 2.0 * max(step_norm, settings.minimum_trust_radius)),
                )
            elif ratio < 0.25:
                damping *= 4.0
                trust = max(settings.minimum_trust_radius, 0.5 * trust)
            relative_reduction = actual / max(old_cost, 1.0)
            if relative_reduction <= settings.ftol:
                success, status, message = True, "ftol", "cost tolerance satisfied"
            elif step_norm <= settings.xtol * (
                settings.xtol + np.linalg.norm(coordinate)
            ):
                success, status, message = True, "xtol", "step tolerance satisfied"
            break
        if success:
            break
        if not accepted and trust <= settings.minimum_trust_radius:
            status, message = "trust_region_collapse", "trust region collapsed"
            break
        if not accepted and nfev >= int(max_nfev):
            break
    return SolverResult(
        coordinate=_readonly(coordinate),
        evaluation=current,
        success=success,
        status=status,
        message=message,
        nfev=nfev,
        elapsed_seconds=time.perf_counter() - started,
        diagnostics=tuple(diagnostics),
    )


def _objective_from_problem(
    problem: SingleBagDynamicsProblem,
    *,
    lag_layout: str,
    command_mode: str,
    smooth_width_fraction: Optional[float],
) -> Callable[[np.ndarray], ObjectiveEvaluation]:
    def evaluate(coordinate: np.ndarray) -> ObjectiveEvaluation:
        residual, jacobian, payload = problem.global_residual_jacobian(
            coordinate,
            lag_layout=lag_layout,
            command_mode=command_mode,
            smooth_width_fraction=smooth_width_fraction,
        )
        return ObjectiveEvaluation(residual, jacobian, payload)

    return evaluate


def _physical_objective(
    problem: SingleBagDynamicsProblem,
    rotor_lag: float,
    gimbal_lag: float,
) -> Callable[[np.ndarray], ObjectiveEvaluation]:
    def evaluate(coordinate: np.ndarray) -> ObjectiveEvaluation:
        payload = problem.evaluate_physical(
            coordinate, rotor_lag, gimbal_lag, command_mode="strict"
        )
        return ObjectiveEvaluation(
            payload.residual_vector, payload.jacobian_matrix, payload
        )

    return evaluate


def _standard_gauge_least_squares(
    evaluator: Callable[[np.ndarray], ObjectiveEvaluation],
    initial: np.ndarray,
    gauge_direction: np.ndarray,
    *,
    max_nfev: int,
    settings: LmSettings,
    lower: np.ndarray,
    upper: np.ndarray,
) -> SolverResult:
    """Standard least-squares on an explicit 13-D exact gauge section."""

    started = time.perf_counter()
    direction = np.asarray(gauge_direction, dtype=float)
    basis = null_space(direction.reshape(1, -1))
    initial_section = _project_gauge(initial, direction)
    reduced_initial = basis.T @ initial_section

    def full(reduced: np.ndarray) -> np.ndarray:
        return basis @ reduced

    def residual(reduced: np.ndarray) -> np.ndarray:
        return evaluator(full(reduced)).residual

    def jacobian(reduced: np.ndarray) -> np.ndarray:
        return evaluator(full(reduced)).jacobian @ basis

    # Bounds concern lag only.  Transforming finite lag bounds through a dense
    # gauge basis is awkward, so the supplied gauge vector is constructed with
    # zero lag entries and scipy bounds are applied in reduced coordinates by
    # identifying the unchanged lag basis rows.
    reduced_lower = np.full(reduced_initial.size, -np.inf)
    reduced_upper = np.full(reduced_initial.size, np.inf)
    for full_index in np.flatnonzero(np.isfinite(lower) | np.isfinite(upper)):
        row = basis[full_index]
        column = int(np.argmax(np.abs(row)))
        if not np.isclose(abs(row[column]), 1.0, atol=1e-12):
            raise RuntimeError("lag coordinate was mixed into gauge basis")
        sign = float(row[column])
        candidates = np.asarray((lower[full_index] / sign, upper[full_index] / sign))
        reduced_lower[column] = float(np.min(candidates))
        reduced_upper[column] = float(np.max(candidates))
    result = least_squares(
        residual,
        reduced_initial,
        jac=jacobian,
        bounds=(reduced_lower, reduced_upper),
        max_nfev=int(max_nfev),
        ftol=settings.ftol,
        xtol=settings.xtol,
        gtol=settings.gtol,
        method="trf",
    )
    coordinate = full(result.x)
    evaluation = evaluator(coordinate)
    return SolverResult(
        coordinate=_readonly(coordinate),
        evaluation=evaluation,
        success=bool(result.success),
        status="scipy_{}".format(result.status),
        message=str(result.message),
        nfev=int(result.nfev),
        elapsed_seconds=time.perf_counter() - started,
    )


def strict_lag_candidates(
    evaluation_time: Sequence[float],
    command_time: Sequence[float],
    bounds: tuple[float, float],
    current: float,
) -> np.ndarray:
    """A data-period local strict-ZOH screen around one current lag."""

    lower, upper = map(float, bounds)
    if not lower <= current <= upper or lower > upper:
        raise ValueError("strict lag bounds/current value are invalid")
    commands = np.asarray(command_time, dtype=float)
    del evaluation_time
    positive = np.diff(commands)
    positive = positive[positive > 0.0]
    if positive.size == 0:
        raise ValueError("strict lag screen needs command timestamp intervals")
    period = float(np.median(positive))
    return np.unique(
        np.clip(
            np.asarray((current - period, current, current + period)),
            lower,
            upper,
        )
    )


def strict_lag_screen(
    problem: SingleBagDynamicsProblem,
    physical_coordinate: np.ndarray,
    rotor_lag: float,
    gimbal_lag: float,
    *,
    lag_layout: str,
    bounds: tuple[float, float],
    alternations: int,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Data-period strict screen extracted from the prior implementation."""

    trace: list[dict[str, Any]] = []
    del alternations
    lower, upper = map(float, bounds)

    def period(history: QuinticSmoothZoh) -> float:
        positive = np.diff(history.times)
        positive = positive[positive > 0.0]
        if positive.size == 0:
            raise ValueError("strict lag screen requires command intervals")
        return float(np.median(positive))

    rotor_period = period(problem.dataset.rotor_history)
    gimbal_period = period(problem.dataset.gimbal_history)

    def cost(rotor: float, gimbal: float) -> float:
        return problem.evaluate_physical(
            physical_coordinate, rotor, gimbal, command_mode="strict"
        ).cost

    if lag_layout == "common":
        # The finer measured publication period defines a data-derived grid;
        # no hand-selected lag increment is introduced.
        screen_period = min(rotor_period, gimbal_period)
        grid_count = int(math.ceil((upper - lower) / screen_period))
        candidates = lower + screen_period * np.arange(grid_count + 1)
        candidates = np.unique(
            np.clip(
                np.concatenate(
                    (candidates, np.asarray((lower, rotor_lag, gimbal_lag, upper)))
                ),
                lower,
                upper,
            )
        )
        costs = np.asarray([cost(value, value) for value in candidates])
        selected_index = int(np.argmin(costs))
        selected = float(candidates[selected_index])
        candidate_table = [
            {
                "lag_seconds": float(value),
                "strict_cost": float(candidate_cost),
                "selected": bool(index == selected_index),
                "at_lower_bound": bool(value == lower),
                "at_upper_bound": bool(value == upper),
            }
            for index, (value, candidate_cost) in enumerate(
                zip(candidates, costs)
            )
        ]
        trace.append(
            {
                "channel": "common",
                "candidate_count": int(candidates.size),
                "selected": selected,
                "cost": float(np.min(costs)),
                "bounds_seconds": (lower, upper),
                "candidate_table": candidate_table,
            }
        )
        return selected, selected, trace

    rotor_values = set(
        float(value)
        for value in strict_lag_candidates(
            problem.dataset.time,
            problem.dataset.rotor_history.times,
            bounds,
            rotor_lag,
        )
    )
    gimbal_values = set(
        float(value)
        for value in strict_lag_candidates(
            problem.dataset.time,
            problem.dataset.gimbal_history.times,
            bounds,
            gimbal_lag,
        )
    )
    costs: dict[tuple[float, float], float] = {}
    expansions: list[dict[str, Any]] = []

    def evaluate_new() -> None:
        for rotor in sorted(rotor_values):
            for gimbal_value in sorted(gimbal_values):
                pair = (rotor, gimbal_value)
                if pair not in costs:
                    costs[pair] = cost(rotor, gimbal_value)

    for _ in range(256):
        evaluate_new()
        best = min(costs, key=lambda pair: (costs[pair], pair[0], pair[1]))
        rotor_order = sorted(rotor_values)
        gimbal_order = sorted(gimbal_values)
        additions: list[dict[str, Any]] = []
        if best[0] == rotor_order[0] and best[0] > lower:
            candidate = max(lower, best[0] - rotor_period)
            if candidate not in rotor_values:
                rotor_values.add(candidate)
                additions.append({"channel": "rotor", "lag_seconds": candidate})
        elif best[0] == rotor_order[-1] and best[0] < upper:
            candidate = min(upper, best[0] + rotor_period)
            if candidate not in rotor_values:
                rotor_values.add(candidate)
                additions.append({"channel": "rotor", "lag_seconds": candidate})
        if best[1] == gimbal_order[0] and best[1] > lower:
            candidate = max(lower, best[1] - gimbal_period)
            if candidate not in gimbal_values:
                gimbal_values.add(candidate)
                additions.append({"channel": "gimbal", "lag_seconds": candidate})
        elif best[1] == gimbal_order[-1] and best[1] < upper:
            candidate = min(upper, best[1] + gimbal_period)
            if candidate not in gimbal_values:
                gimbal_values.add(candidate)
                additions.append({"channel": "gimbal", "lag_seconds": candidate})
        if not additions:
            break
        expansions.extend(additions)
    else:
        raise RuntimeError("strict lag screen did not terminate")
    evaluate_new()
    best = min(costs, key=lambda pair: (costs[pair], pair[0], pair[1]))
    rotor_lag, gimbal_lag = best
    candidate_table = [
        {
            "rotor_lag_seconds": float(pair[0]),
            "gimbal_lag_seconds": float(pair[1]),
            "strict_cost": float(costs[pair]),
            "selected": bool(pair == best),
            "rotor_at_lower_bound": bool(pair[0] == lower),
            "rotor_at_upper_bound": bool(pair[0] == upper),
            "gimbal_at_lower_bound": bool(pair[1] == lower),
            "gimbal_at_upper_bound": bool(pair[1] == upper),
        }
        for pair in sorted(costs)
    ]
    trace.append(
        {
            "channel": "split_pair",
            "candidate_count": len(costs),
            "rotor_candidate_count": len(rotor_values),
            "gimbal_candidate_count": len(gimbal_values),
            "selected_rotor_lag_seconds": rotor_lag,
            "selected_gimbal_lag_seconds": gimbal_lag,
            "cost": costs[best],
            "bounds_seconds": (lower, upper),
            "candidate_table": candidate_table,
            "expansions": expansions,
        }
    )
    return rotor_lag, gimbal_lag, trace


@dataclass(frozen=True)
class EstimatorConfig:
    covariance_mode: str = "full"
    geometric_correction: bool = True
    actuator_propagation: str = "stateful"
    lag_mode: str = "split_estimated"
    initial_rotor_lag: float = 0.0
    initial_gimbal_lag: float = 0.0
    fixed_rotor_lag: Optional[float] = None
    fixed_gimbal_lag: Optional[float] = None
    lag_bounds: Optional[tuple[float, float]] = None
    smooth_width_schedule: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5)
    # These are safety ceilings; gtol/ftol/xtol remain the normal termination
    # criteria.  Real-bag validation required more than the former 80/120.
    smooth_max_nfev: int = DEFAULT_SMOOTH_MAX_NFEV
    strict_max_nfev: int = DEFAULT_STRICT_MAX_NFEV
    strict_alternations: int = 8
    kkt_enabled: bool = True
    solver_type: str = "custom_kkt_lm"
    initial_physical_coordinate: np.ndarray = field(
        default_factory=lambda: np.zeros(PHYSICAL_DIMENSION)
    )
    scale_initial_offset: float = 0.0
    lm: LmSettings = field(default_factory=LmSettings)

    def __post_init__(self) -> None:
        if self.covariance_mode not in COVARIANCE_MODES:
            raise ValueError("unknown covariance mode")
        if self.lag_mode not in (
            "zero",
            "fixed",
            "common_estimated",
            "split_estimated",
            "split_strict_only",
        ):
            raise ValueError("unknown lag mode")
        if self.solver_type not in ("custom_kkt_lm", "standard_least_squares"):
            raise ValueError("unknown solver type")
        coordinate = np.asarray(self.initial_physical_coordinate, dtype=float)
        if coordinate.shape != (14,) or np.any(~np.isfinite(coordinate)):
            raise ValueError("initial physical coordinate must be finite 14-D")
        object.__setattr__(self, "initial_physical_coordinate", _readonly(coordinate))
        if self.lag_mode in ("common_estimated", "split_estimated", "split_strict_only"):
            if self.lag_bounds is None:
                raise ValueError("estimated lag mode requires explicit lag bounds")
            lower, upper = self.lag_bounds
            if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
                raise ValueError("lag bounds are invalid")
            if not (
                lower <= self.initial_rotor_lag <= upper
                and lower <= self.initial_gimbal_lag <= upper
            ):
                raise ValueError("initial lag lies outside bounds")
        if self.lag_mode == "fixed" and (
            self.fixed_rotor_lag is None or self.fixed_gimbal_lag is None
        ):
            raise ValueError("fixed lag mode requires both fixed values")
        if (
            self.smooth_max_nfev < 1
            or self.strict_max_nfev < 1
            or self.strict_alternations < 1
            or any(
                (not np.isfinite(value) or value <= 0.0)
                for value in self.smooth_width_schedule
            )
        ):
            raise ValueError("estimator termination/smoothing settings are invalid")


@dataclass(frozen=True)
class EstimationResult:
    physical_coordinate: np.ndarray
    rotor_lag_seconds: float
    gimbal_lag_seconds: float
    evaluation: DynamicsEvaluation
    reference_evaluation: DynamicsEvaluation
    success: bool
    status: str
    message: str
    stages: tuple[Mapping[str, Any], ...]
    total_nfev: int
    elapsed_seconds: float
    ridge: Mapping[str, Any]
    uncertainty: ParameterCovarianceResult
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    first_failure_stage: Optional[str] = None
    first_failure_status: Optional[str] = None
    first_failure_message: Optional[str] = None


def _solve_stage(
    problem: SingleBagDynamicsProblem,
    initial: np.ndarray,
    evaluator: Callable[[np.ndarray], ObjectiveEvaluation],
    *,
    config: EstimatorConfig,
    max_nfev: int,
    lag_columns: int,
) -> SolverResult:
    gauge = np.zeros(initial.size)
    gauge[:14] = COMMON_SCALE_DIRECTION
    lower, upper = np.full(initial.size, -np.inf), np.full(initial.size, np.inf)
    if lag_columns:
        assert config.lag_bounds is not None
        lower[-lag_columns:] = config.lag_bounds[0]
        upper[-lag_columns:] = config.lag_bounds[1]
    if config.solver_type == "standard_least_squares":
        return _standard_gauge_least_squares(
            evaluator,
            initial,
            gauge,
            max_nfev=max_nfev,
            settings=config.lm,
            lower=lower,
            upper=upper,
        )
    return adaptive_kkt_lm(
        evaluator,
        initial,
        settings=config.lm,
        max_nfev=max_nfev,
        gauge_direction=gauge if config.kkt_enabled else None,
        lower=lower,
        upper=upper,
    )


def ridge_analysis(
    whitened_jacobian: np.ndarray,
    gauge_direction: Sequence[float],
    unwhitened_jacobian: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """14-D physical ridge products with machine-precision rank only."""

    jacobian = np.asarray(whitened_jacobian, dtype=float)
    direction = np.asarray(gauge_direction, dtype=float)
    if (
        jacobian.ndim != 2
        or jacobian.shape[1] != PHYSICAL_DIMENSION
        or direction.shape != (PHYSICAL_DIMENSION,)
        or np.any(~np.isfinite(jacobian))
        or np.any(~np.isfinite(direction))
    ):
        raise ValueError("scientific ridge inputs must be finite and 14-D")
    _u, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
    tolerance = (
        0.0
        if singular.size == 0
        else max(jacobian.shape) * np.finfo(float).eps * float(singular[0])
    )
    rank = int(np.count_nonzero(singular > tolerance))
    result = {
        "dimension": PHYSICAL_DIMENSION,
        "raw_whitened_jacobian": _readonly(jacobian),
        "jtj": _readonly(jacobian.T @ jacobian),
        "singular_values": _readonly(singular),
        "whitened_singular_values": _readonly(singular),
        "right_singular_vectors": _readonly(vt),
        "whitened_right_singular_vectors": _readonly(vt),
        "machine_rank_tolerance": tolerance,
        "machine_numerical_rank": rank,
        "nullity": int(jacobian.shape[1] - rank),
        "exact_scale_gauge_direction": _readonly(direction),
        "j_v_scale": _readonly(jacobian @ direction),
        "j_v_scale_norm": float(np.linalg.norm(jacobian @ direction)),
    }
    if unwhitened_jacobian is not None:
        raw = np.asarray(unwhitened_jacobian, dtype=float)
        if raw.shape != jacobian.shape or np.any(~np.isfinite(raw)):
            raise ValueError("unwhitened ridge Jacobian shape mismatch")
        _raw_u, raw_singular, raw_vt = np.linalg.svd(
            raw, full_matrices=False
        )
        result.update(
            {
                "raw_acceleration_jacobian": _readonly(raw),
                "unwhitened_diagnostic_singular_values": _readonly(
                    raw_singular
                ),
                "unwhitened_diagnostic_right_singular_vectors": _readonly(
                    raw_vt
                ),
            }
        )
    return result


def covariance_weighting_diagnostics(
    covariance: SgCovarianceEvaluation,
    residual: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return exact eigenmode decomposition of the weighted objective."""

    acceleration_residual = np.asarray(residual, dtype=float)
    if acceleration_residual.shape != covariance.z.shape:
        raise ValueError("covariance diagnostic residual shape mismatch")
    count = acceleration_residual.shape[0]
    eigenvalues = np.empty((count, 6))
    ranks = np.empty(count, dtype=int)
    condition = np.empty(count)
    gains = np.zeros((count, 6))
    eigenmode_residual = np.empty((count, 6))
    contributions = np.zeros((count, 6))
    for index, matrix in enumerate(covariance.local_sigma_z):
        values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
        scale = float(np.max(np.abs(values))) if values.size else 0.0
        tolerance = max(matrix.shape) * np.finfo(float).eps * scale
        retained = values > tolerance
        projected = vectors.T @ acceleration_residual[index]
        eigenvalues[index] = values
        ranks[index] = int(np.count_nonzero(retained))
        condition[index] = (
            float(np.max(values[retained]) / np.min(values[retained]))
            if np.any(retained)
            else np.nan
        )
        gains[index, retained] = 1.0 / np.sqrt(values[retained])
        eigenmode_residual[index] = projected
        contributions[index, retained] = (
            projected[retained] ** 2 / values[retained]
        )
    mahalanobis = np.sum(contributions, axis=1)
    whitened = np.einsum(
        "nij,nj->ni", covariance.whitening, acceleration_residual
    )
    if not np.allclose(
        mahalanobis,
        np.sum(whitened**2, axis=1),
        rtol=2.0e-10,
        atol=2.0e-10,
    ):
        raise RuntimeError("covariance eigenmode contributions are inconsistent")
    return {
        "sigma_z_eigenvalues": _readonly(eigenvalues),
        "sigma_z_machine_rank": _readonly(ranks),
        "sigma_z_retained_condition_number": _readonly(condition),
        "whitening_gain": _readonly(gains),
        "mahalanobis_contribution_per_time": _readonly(mahalanobis),
        "covariance_eigenmode_residual": _readonly(eigenmode_residual),
        "covariance_eigenmode_mahalanobis_contribution": _readonly(
            contributions
        ),
    }


def metric_cross_evaluation(
    evaluation: DynamicsEvaluation,
) -> dict[str, float]:
    """Evaluate one physical point under full and identity metrics."""

    residual = np.asarray(evaluation.acceleration_residual)
    whitened = np.asarray(evaluation.whitened_residual)
    return {
        "full_covariance_objective_sum": 0.5
        * float(np.sum(whitened**2)),
        "identity_objective_sum": 0.5 * float(np.sum(residual**2)),
        "specific_acceleration_rmse_m_per_s2": float(
            np.sqrt(np.mean(np.sum(residual[:, :3] ** 2, axis=1)))
        ),
        "angular_acceleration_rmse_rad_per_s2": float(
            np.sqrt(np.mean(np.sum(residual[:, 3:] ** 2, axis=1)))
        ),
    }


def acceleration_wrench_closure(
    dataset: SingleBagDataset,
    evaluation: DynamicsEvaluation,
) -> dict[str, np.ndarray | float]:
    """Check the exact Newton--Euler residual identities used by the model."""

    residual = np.asarray(evaluation.acceleration_residual)
    raw = np.asarray(evaluation.raw_residual_wrench)
    parameters = evaluation.parameters
    inertia = np.asarray(parameters.inertia)
    lever = (
        np.asarray(dataset.pose_sensor_position_in_body)
        - np.asarray(parameters.cog_offset)
    )
    reconstructed_force = float(parameters.mass) * (
        residual[:, :3] + np.cross(lever, residual[:, 3:])
    )
    reconstructed_torque = np.einsum(
        "ij,nj->ni", inertia, residual[:, 3:]
    )
    force_error = raw[:, :3] - reconstructed_force
    torque_error = raw[:, 3:] - reconstructed_torque
    return {
        "force_acceleration_closure_error": _readonly(force_error),
        "torque_acceleration_closure_error": _readonly(torque_error),
        "force_acceleration_closure_error_max_abs": float(
            np.max(np.abs(force_error))
        ),
        "force_acceleration_closure_error_rms": float(
            np.sqrt(np.mean(force_error**2))
        ),
        "torque_acceleration_closure_error_max_abs": float(
            np.max(np.abs(torque_error))
        ),
        "torque_acceleration_closure_error_rms": float(
            np.sqrt(np.mean(torque_error**2))
        ),
    }


def _lag_diagnostics(
    lag_layout: Optional[str],
    trace: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_table: list[Mapping[str, Any]] = []
    if trace:
        table = trace[-1].get("candidate_table", [])
        if isinstance(table, list):
            candidate_table = table
    selected = [row for row in candidate_table if bool(row.get("selected"))]
    if candidate_table and len(selected) != 1:
        raise RuntimeError("strict lag candidate table must select exactly once")
    row = selected[0] if selected else {}
    if lag_layout == "common":
        rotor_lower = gimbal_lower = bool(row.get("at_lower_bound", False))
        rotor_upper = gimbal_upper = bool(row.get("at_upper_bound", False))
    else:
        rotor_lower = bool(row.get("rotor_at_lower_bound", False))
        rotor_upper = bool(row.get("rotor_at_upper_bound", False))
        gimbal_lower = bool(row.get("gimbal_at_lower_bound", False))
        gimbal_upper = bool(row.get("gimbal_at_upper_bound", False))
    return {
        "lag_layout": lag_layout if lag_layout is not None else "fixed_physical",
        "candidate_table": candidate_table,
        "rotor_at_lower_bound": rotor_lower,
        "rotor_at_upper_bound": rotor_upper,
        "gimbal_at_lower_bound": gimbal_lower,
        "gimbal_at_upper_bound": gimbal_upper,
    }


def estimation_diagnostics(
    *,
    problem: SingleBagDynamicsProblem,
    final: DynamicsEvaluation,
    nominal_reference: DynamicsEvaluation,
    estimated_reference: DynamicsEvaluation,
    ridge: Mapping[str, Any],
    uncertainty: ParameterCovarianceResult,
    lag_layout: Optional[str],
    lag_trace: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build post-fit diagnostics without feeding them back to optimization."""

    covariance = covariance_weighting_diagnostics(
        problem.dataset.covariance, final.acceleration_residual
    )
    right = np.asarray(ridge["whitened_right_singular_vectors"])
    coordinate = np.asarray(final.physical_coordinate)
    displacement = right @ coordinate
    energy = float(coordinate @ coordinate)
    displacement_fraction = (
        np.zeros_like(displacement)
        if energy == 0.0
        else displacement**2 / energy
    )
    naive_variance = np.einsum(
        "ij,jk,ik->i", right, uncertainty.naive, right
    )
    overlap_variance = np.einsum(
        "ij,jk,ik->i", right, uncertainty.overlap_corrected, right
    )
    inflation = np.full_like(naive_variance, np.nan)
    np.divide(
        overlap_variance,
        naive_variance,
        out=inflation,
        where=naive_variance > 0.0,
    )
    closure = acceleration_wrench_closure(problem.dataset, final)
    nominal_moments, nominal_axes = np.linalg.eigh(
        np.asarray(problem.model.parameters.inertia)
    )
    estimated_moments, estimated_axes = np.linalg.eigh(
        np.asarray(final.parameters.inertia)
    )
    lag = _lag_diagnostics(lag_layout, lag_trace)
    return {
        "covariance": covariance,
        "metric_cross_evaluation": {
            "nominal": metric_cross_evaluation(nominal_reference),
            "estimated": metric_cross_evaluation(estimated_reference),
        },
        "ridge": {
            "parameter_displacement_ridge_coordinates": _readonly(
                displacement
            ),
            "parameter_displacement_ridge_energy_fraction": _readonly(
                displacement_fraction
            ),
        },
        "lag": lag,
        "closure": closure,
        "inertia": {
            "nominal_principal_moments_kg_m2": _readonly(nominal_moments),
            "nominal_principal_axes_body": _readonly(nominal_axes),
            "estimated_principal_moments_kg_m2": _readonly(
                estimated_moments
            ),
            "estimated_principal_axes_body": _readonly(estimated_axes),
            "estimated_to_nominal_ratio": _readonly(
                estimated_moments / nominal_moments
            ),
        },
        "overlap_correction": {
            "cross_time_covariance_model": (
                "pairwise_mean_local_raw_pose_covariance"
            ),
            "uncertainty_variance_naive_in_ridge_basis": _readonly(
                naive_variance
            ),
            "uncertainty_variance_overlap_in_ridge_basis": _readonly(
                overlap_variance
            ),
            "uncertainty_variance_inflation_in_ridge_basis": _readonly(
                inflation
            ),
        },
    }


def estimate_single_bag(
    problem: SingleBagDynamicsProblem, config: EstimatorConfig
) -> EstimationResult:
    """Run smooth continuation, strict screen, then strict physical refinement."""

    started = time.perf_counter()
    stages: list[Mapping[str, Any]] = []
    physical = np.asarray(config.initial_physical_coordinate).copy()
    physical += (
        float(config.scale_initial_offset)
        * COMMON_SCALE_DIRECTION
        / np.linalg.norm(COMMON_SCALE_DIRECTION)
    )
    if config.lag_mode == "zero":
        rotor_lag = gimbal_lag = 0.0
        lag_layout = None
    elif config.lag_mode == "fixed":
        rotor_lag = float(config.fixed_rotor_lag)  # type: ignore[arg-type]
        gimbal_lag = float(config.fixed_gimbal_lag)  # type: ignore[arg-type]
        lag_layout = None
    else:
        rotor_lag = float(config.initial_rotor_lag)
        gimbal_lag = float(config.initial_gimbal_lag)
        lag_layout = "common" if config.lag_mode == "common_estimated" else "split"

    total_nfev = 0
    stage_success = True
    first_failure_stage: Optional[str] = None
    first_failure_status: Optional[str] = None
    first_failure_message: Optional[str] = None
    lag_trace_for_final: Sequence[Mapping[str, Any]] = ()
    if lag_layout is not None and config.lag_mode != "split_strict_only":
        coordinate = (
            np.concatenate((physical, np.asarray((rotor_lag,))))
            if lag_layout == "common"
            else np.concatenate((physical, np.asarray((rotor_lag, gimbal_lag))))
        )
        for width in config.smooth_width_schedule:
            objective = _objective_from_problem(
                problem,
                lag_layout=lag_layout,
                command_mode="smooth",
                smooth_width_fraction=float(width),
            )
            solved = _solve_stage(
                problem,
                coordinate,
                objective,
                config=config,
                max_nfev=config.smooth_max_nfev,
                lag_columns=1 if lag_layout == "common" else 2,
            )
            total_nfev += solved.nfev
            coordinate = np.asarray(solved.coordinate).copy()
            stages.append(
                {
                    "stage": "smooth_continuation",
                    "width_fraction": float(width),
                    "success": solved.success,
                    "status": solved.status,
                    "message": solved.message,
                    "nfev": solved.nfev,
                    "elapsed_seconds": solved.elapsed_seconds,
                }
            )
            stage_success = stage_success and solved.success
            if not solved.success:
                if first_failure_stage is None:
                    first_failure_stage = "smooth_continuation"
                    first_failure_status = solved.status
                    first_failure_message = solved.message
                break
        physical = coordinate[:14]
        if lag_layout == "common":
            rotor_lag = gimbal_lag = float(coordinate[14])
        else:
            rotor_lag, gimbal_lag = float(coordinate[14]), float(coordinate[15])

    if lag_layout is not None:
        assert config.lag_bounds is not None
        rotor_lag, gimbal_lag, screen_trace = strict_lag_screen(
            problem,
            physical,
            rotor_lag,
            gimbal_lag,
            lag_layout=lag_layout,
            bounds=config.lag_bounds,
            alternations=config.strict_alternations,
        )
        lag_trace_for_final = screen_trace
        stages.append(
            {
                "stage": "strict_lag_screen",
                "success": True,
                "rotor_lag_seconds": rotor_lag,
                "gimbal_lag_seconds": gimbal_lag,
                "trace": screen_trace,
            }
        )
    strict_iteration_limit = config.strict_alternations if lag_layout is not None else 1
    visited_lags: set[tuple[float, float]] = set()
    for strict_iteration in range(strict_iteration_limit):
        pair = (round(rotor_lag, 12), round(gimbal_lag, 12))
        if pair in visited_lags:
            stages.append(
                {
                    "stage": "strict_cycle_guard",
                    "success": True,
                    "iteration": strict_iteration + 1,
                    "rotor_lag_seconds": rotor_lag,
                    "gimbal_lag_seconds": gimbal_lag,
                }
            )
            break
        visited_lags.add(pair)
        physical_result = _solve_stage(
            problem,
            physical,
            _physical_objective(problem, rotor_lag, gimbal_lag),
            config=config,
            max_nfev=config.strict_max_nfev,
            lag_columns=0,
        )
        total_nfev += physical_result.nfev
        physical = np.asarray(physical_result.coordinate).copy()
        stages.append(
            {
                "stage": "strict_physical_refinement",
                "iteration": strict_iteration + 1,
                "success": physical_result.success,
                "status": physical_result.status,
                "message": physical_result.message,
                "nfev": physical_result.nfev,
                "elapsed_seconds": physical_result.elapsed_seconds,
                "rotor_lag_seconds": rotor_lag,
                "gimbal_lag_seconds": gimbal_lag,
            }
        )
        stage_success = stage_success and physical_result.success
        if not physical_result.success and first_failure_stage is None:
            first_failure_stage = "strict_physical_refinement"
            first_failure_status = physical_result.status
            first_failure_message = physical_result.message
        if not physical_result.success or lag_layout is None:
            break
        assert config.lag_bounds is not None
        next_rotor, next_gimbal, verify_trace = strict_lag_screen(
            problem,
            physical,
            rotor_lag,
            gimbal_lag,
            lag_layout=lag_layout,
            bounds=config.lag_bounds,
            alternations=1,
        )
        fixed_point = bool(
            np.isclose(next_rotor, rotor_lag, rtol=0.0, atol=5.0e-13)
            and np.isclose(next_gimbal, gimbal_lag, rtol=0.0, atol=5.0e-13)
        )
        stages.append(
            {
                "stage": "strict_post_refinement_screen",
                "iteration": strict_iteration + 1,
                "success": True,
                "fixed_point": fixed_point,
                "rotor_lag_seconds": next_rotor,
                "gimbal_lag_seconds": next_gimbal,
                "trace": verify_trace,
            }
        )
        if fixed_point:
            lag_trace_for_final = verify_trace
            break
        next_pair = (round(next_rotor, 12), round(next_gimbal, 12))
        if next_pair in visited_lags:
            stages.append(
                {
                    "stage": "strict_cycle_guard",
                    "success": True,
                    "iteration": strict_iteration + 1,
                    "rejected_revisited_pair": next_pair,
                }
            )
            break
        if strict_iteration + 1 >= strict_iteration_limit:
            stages.append(
                {
                    "stage": "strict_alternation_limit",
                    "success": True,
                    "iteration_limit": strict_iteration_limit,
                    "unrefined_next_pair": next_pair,
                }
            )
            break
        lag_trace_for_final = verify_trace
        rotor_lag, gimbal_lag = next_rotor, next_gimbal
    final = problem.evaluate_physical(
        physical, rotor_lag, gimbal_lag, command_mode="strict"
    )
    if not final.actuator_history.strict_final:
        raise RuntimeError("final estimator result was not evaluated with strict ZOH")
    reference = problem.evaluate_physical(
        physical, rotor_lag, gimbal_lag, command_mode="strict", reference=True
    )
    nominal_reference = problem.evaluate_physical(
        np.zeros(PHYSICAL_DIMENSION),
        rotor_lag,
        gimbal_lag,
        command_mode="strict",
        reference=True,
    )
    ridge = ridge_analysis(
        final.jacobian_matrix,
        COMMON_SCALE_DIRECTION,
        final.acceleration_jacobian.reshape(-1, PHYSICAL_DIMENSION),
    )
    uncertainty = parameter_covariances(
        final.acceleration_jacobian,
        problem.dataset.covariance,
        COMMON_SCALE_DIRECTION,
    )
    diagnostics = estimation_diagnostics(
        problem=problem,
        final=final,
        nominal_reference=nominal_reference,
        estimated_reference=reference,
        ridge=ridge,
        uncertainty=uncertainty,
        lag_layout=lag_layout,
        lag_trace=lag_trace_for_final,
    )
    ridge.update(diagnostics["ridge"])
    final_status = "completed" if stage_success else str(first_failure_status)
    final_message = (
        "all stages completed" if stage_success else str(first_failure_message)
    )
    return EstimationResult(
        physical_coordinate=_readonly(physical),
        rotor_lag_seconds=rotor_lag,
        gimbal_lag_seconds=gimbal_lag,
        evaluation=final,
        reference_evaluation=reference,
        success=bool(stage_success),
        status=final_status,
        message=final_message,
        stages=tuple(stages),
        total_nfev=total_nfev,
        elapsed_seconds=time.perf_counter() - started,
        ridge=ridge,
        uncertainty=uncertainty,
        diagnostics=diagnostics,
        first_failure_stage=first_failure_stage,
        first_failure_status=first_failure_status,
        first_failure_message=first_failure_message,
    )
