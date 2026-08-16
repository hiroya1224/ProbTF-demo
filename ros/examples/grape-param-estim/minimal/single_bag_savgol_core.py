#!/usr/bin/env python3
"""Core single-bag SG rigid-body parameter estimation in SI units."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Optional, Sequence
import warnings

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
    from .gimbal_savgol import IrregularSavitzkyGolayGimbal
    from .rotor_lag import (
        StrictLagCell,
        StrictZohCellGrid,
        local_strict_cell_descent,
        power_of_two_epsilon,
    )
    from .single_bag_savgol_covariance import (
        COVARIANCE_MODES,
        ParameterCovarianceResult,
        ResidualWrenchUncertainty,
        SgCovarianceEvaluation,
        build_sg_covariance,
        machine_pseudoinverse_symmetric,
        parameter_covariances,
        residual_wrench_uncertainty,
    )
    from .smooth_command import QuinticSmoothZoh
except ImportError:  # pragma: no cover - direct CLI import
    from savgol_trajectory import (  # type: ignore
        GeometricSavitzkyGolayPose,
        PoseSgEvaluation,
    )
    from gimbal_savgol import IrregularSavitzkyGolayGimbal  # type: ignore
    from rotor_lag import (  # type: ignore
        StrictLagCell,
        StrictZohCellGrid,
        local_strict_cell_descent,
        power_of_two_epsilon,
    )
    from single_bag_savgol_covariance import (  # type: ignore
        COVARIANCE_MODES,
        ParameterCovarianceResult,
        ResidualWrenchUncertainty,
        SgCovarianceEvaluation,
        build_sg_covariance,
        machine_pseudoinverse_symmetric,
        parameter_covariances,
        residual_wrench_uncertainty,
    )
    from smooth_command import QuinticSmoothZoh  # type: ignore


BASE_PLAN_COMMIT = "ac03f309efbd659a9cc1a85f23bde5715cdd4934"
DEFAULT_SMOOTH_MAX_NFEV = 2000
DEFAULT_STRICT_MAX_NFEV = 2000
PHYSICAL_DIMENSION = 14
ROTOR_LAG_INDEX = 14
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
PHYSICAL_CHART_LABELS = (
    "log_mass",
    "second_moment_diag_1",
    "second_moment_diag_2",
    "second_moment_diag_3",
    "second_moment_offdiag_12",
    "second_moment_offdiag_13",
    "second_moment_offdiag_23",
    "cog_x",
    "cog_y",
    "cog_z",
    "log_force_eff_1",
    "log_force_eff_2",
    "log_force_eff_3",
    "log_force_eff_4",
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
    """Direct rotor command and selected observed/replayed gimbal trajectory."""

    time: np.ndarray
    actual_thrust: np.ndarray
    actual_gimbal: np.ndarray
    actual_thrust_lag_jacobian: np.ndarray
    active_set_counts: Mapping[str, int]
    gimbal_source: str
    command_mode: str
    strict_final: bool

    def __post_init__(self) -> None:
        time_axis = np.asarray(self.time, dtype=float)
        count = time_axis.size
        thrust = np.asarray(self.actual_thrust, dtype=float)
        gimbal = np.asarray(self.actual_gimbal, dtype=float)
        thrust_lag = np.asarray(self.actual_thrust_lag_jacobian, dtype=float)
        if (
            time_axis.ndim != 1
            or count < 1
            or thrust.shape != (count, 4)
            or gimbal.shape != (count, 4)
            or thrust_lag.shape != (count, 4)
            or any(
                np.any(~np.isfinite(x))
                for x in (time_axis, thrust, gimbal, thrust_lag)
            )
            or self.gimbal_source
            not in ("measured_sg", "measured_linear", "command_replay")
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
            self,
            "active_set_counts",
            {str(name): int(value) for name, value in self.active_set_counts.items()},
        )


@dataclass(frozen=True)
class GimbalCommandReplay:
    time: np.ndarray
    angle: np.ndarray
    active_set_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float)
        angle = np.asarray(self.angle, dtype=float)
        if (
            time.ndim != 1
            or angle.shape != (time.size, 4)
            or np.any(~np.isfinite(time))
            or np.any(~np.isfinite(angle))
        ):
            raise ValueError("gimbal command replay is invalid")
        object.__setattr__(self, "time", _readonly(time))
        object.__setattr__(self, "angle", _readonly(angle))
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


def generate_gimbal_command_replay(
    *,
    time_axis: Sequence[float],
    gimbal_history: QuinticSmoothZoh,
    initial_gimbal: Sequence[float],
    actuator_parameters: ActuatorParameters,
) -> GimbalCommandReplay:
    """Replay recorded gimbal commands from the first measured angle.

    This is diagnostic-only for the default estimator.  The
    ``gimbal_command_replay`` ablation may explicitly select it as its gimbal
    source, but it never introduces an estimated gimbal lag.
    """

    times = np.asarray(time_axis, dtype=float)
    if (
        times.ndim != 1
        or times.size < 1
        or np.any(~np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("gimbal replay time axis is invalid")
    initial = _finite_vector(initial_gimbal, 4, "initial_gimbal")
    gimbal = np.empty((times.size, 4))
    counts: dict[str, int] = {}
    dummy_thrust = np.full(4, actuator_parameters.minimum_thrust, dtype=float)
    state = ActuatorState(dummy_thrust, initial)
    gimbal[0] = state.gimbal_angle
    gimbal_switches = np.asarray(gimbal_history.times[1:])
    for index, (left, right) in enumerate(zip(times[:-1], times[1:])):
        switches = gimbal_switches[
            (gimbal_switches > left) & (gimbal_switches < right)
        ]
        boundaries = np.concatenate(
            (np.asarray((left,)), switches, np.asarray((right,)))
        )
        for sub_left, sub_right in zip(boundaries[:-1], boundaries[1:]):
            step = float(sub_right - sub_left)
            if step <= 0.0:
                continue
            midpoint = 0.5 * (float(sub_left) + float(sub_right))
            command = _command(
                dummy_thrust, gimbal_history.exact_zoh(midpoint, 0.0)
            )
            transition = advance_actuators_with_jacobian(
                state,
                command,
                actuator_parameters,
                step,
            )
            state = transition.next_state
            for name, mask in transition.active_set.items():
                _count_mask(counts, name, mask)
        gimbal[index + 1] = state.gimbal_angle
    return GimbalCommandReplay(
        times,
        gimbal,
        {
            name: int(counts.get(name, 0))
            for name in ACTUATOR_ACTIVE_SET_NAMES
            if name.startswith("gimbal_")
        },
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
    gimbal_raw_time: np.ndarray
    gimbal_raw_angle: np.ndarray
    gimbal_sg_angle: np.ndarray
    gimbal_linear_angle: np.ndarray
    initial_gimbal: np.ndarray
    rotor_lag_data_support_upper: float
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
            "gimbal_raw_time": (np.asarray(self.gimbal_raw_time).size,),
            "gimbal_raw_angle": (np.asarray(self.gimbal_raw_time).size, 4),
            "gimbal_sg_angle": (count, 4),
            "gimbal_linear_angle": (count, 4),
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
        raw_time = np.asarray(self.gimbal_raw_time, dtype=float)
        if (
            raw_time.ndim != 1
            or raw_time.size < 1
            or np.any(~np.isfinite(raw_time))
            or np.any(np.diff(raw_time) <= 0.0)
        ):
            raise ValueError("raw gimbal time is invalid")
        if (
            not np.isfinite(self.rotor_lag_data_support_upper)
            or self.rotor_lag_data_support_upper <= 0.0
        ):
            raise ValueError("rotor lag data-support upper bound is invalid")
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
    gimbal_smoother = IrregularSavitzkyGolayGimbal(
        flight.gimbal_position.times,
        flight.gimbal_position.values,
        window_seconds=window_seconds,
        degree=degree,
    )
    support_start = max(
        trajectory.valid_start_time,
        gimbal_smoother.valid_start_time,
        float(flight.rotor_command.all_times[0]),
    )
    support_end = min(trajectory.valid_end_time, gimbal_smoother.valid_end_time)
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
    gimbal_sg = gimbal_smoother.evaluate(time_axis)
    gimbal_linear = _linear_interpolate(
        gimbal_smoother.time,
        gimbal_smoother.raw_angle,
        time_axis,
    )
    initial_gimbal = _linear_interpolate(
        gimbal_smoother.time,
        gimbal_smoother.raw_angle,
        np.asarray((time_axis[0],)),
    )[0]
    rotor_history = QuinticSmoothZoh(
        flight.rotor_command.all_times, flight.rotor_command.all_values
    )
    data_support_upper = float(time_axis[0] - rotor_history.times[0])

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
        rotor_history=rotor_history,
        gimbal_history=QuinticSmoothZoh(
            flight.gimbal_command.all_times, flight.gimbal_command.all_values
        ),
        gimbal_raw_time=gimbal_smoother.time,
        gimbal_raw_angle=gimbal_smoother.raw_angle,
        gimbal_sg_angle=gimbal_sg.angle,
        gimbal_linear_angle=gimbal_linear,
        initial_gimbal=initial_gimbal,
        rotor_lag_data_support_upper=data_support_upper,
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
    """Newton--Euler acceleration objective for exactly one bag."""

    def __init__(
        self,
        dataset: SingleBagDataset,
        model: VehicleModelInput,
        actuator_parameters: ActuatorParameters,
        *,
        gimbal_source: str = "measured_sg",
        parameter_prior: Optional[Any] = None,
    ) -> None:
        if gimbal_source not in (
            "measured_sg",
            "measured_linear",
            "command_replay",
        ):
            raise ValueError("unknown gimbal source")
        self.dataset = dataset
        self.model = model
        self.actuator_parameters = actuator_parameters
        self.gimbal_source = gimbal_source
        self.chart = SiParameterChart(model.parameters)
        self.parameter_prior = parameter_prior
        self.strict_lag_grid = StrictZohCellGrid(
            dataset.time, dataset.rotor_history.times
        )
        if not np.isclose(
            self.strict_lag_grid.data_support_upper,
            dataset.rotor_lag_data_support_upper,
            rtol=0.0,
            atol=2.0e-12,
        ):
            raise RuntimeError("rotor lag support calculations disagree")
        self.gimbal_replay = generate_gimbal_command_replay(
            time_axis=dataset.time,
            gimbal_history=dataset.gimbal_history,
            initial_gimbal=dataset.initial_gimbal,
            actuator_parameters=actuator_parameters,
        )

    def actuator_history(
        self,
        rotor_lag_seconds: float,
        *,
        command_mode: str,
        epsilon: Optional[float] = None,
    ) -> ActuatorHistory:
        if command_mode not in ("strict", "smooth"):
            raise ValueError("unknown rotor command mode")
        if command_mode == "smooth" and (
            epsilon is None or not np.isfinite(epsilon) or epsilon <= 0.0
        ):
            raise ValueError("smooth rotor evaluation requires positive epsilon")
        lag = float(rotor_lag_seconds)
        if (
            not np.isfinite(lag)
            or lag < 0.0
            or lag > self.strict_lag_grid.data_support_upper
        ):
            raise ValueError("rotor lag lies outside recorded data support")
        thrust = np.empty((self.dataset.time.size, 4))
        derivative = np.zeros_like(thrust)
        counts: dict[str, int] = {}
        for index, sample_time in enumerate(self.dataset.time):
            if command_mode == "strict":
                raw = self.dataset.rotor_history.exact_zoh(sample_time, lag)
                raw_derivative = np.zeros(4)
            else:
                assert epsilon is not None
                command = self.dataset.rotor_history.evaluate(
                    sample_time, lag, epsilon
                )
                raw = command.value
                raw_derivative = command.delay_derivative
            thrust[index] = np.clip(
                raw,
                self.actuator_parameters.minimum_thrust,
                self.actuator_parameters.maximum_thrust,
            )
            interior = (
                (raw > self.actuator_parameters.minimum_thrust)
                & (raw < self.actuator_parameters.maximum_thrust)
            )
            derivative[index] = interior * raw_derivative
            _count_mask(
                counts,
                "thrust_command_lower",
                raw <= self.actuator_parameters.minimum_thrust,
            )
            _count_mask(
                counts,
                "thrust_command_upper",
                raw >= self.actuator_parameters.maximum_thrust,
            )
        if self.gimbal_source == "measured_sg":
            gimbal = self.dataset.gimbal_sg_angle
        elif self.gimbal_source == "measured_linear":
            gimbal = self.dataset.gimbal_linear_angle
        else:
            gimbal = self.gimbal_replay.angle
        combined_counts = {
            name: int(counts.get(name, 0))
            + (
                int(self.gimbal_replay.active_set_counts.get(name, 0))
                if name.startswith("gimbal_")
                else 0
            )
            for name in ACTUATOR_ACTIVE_SET_NAMES
        }
        return ActuatorHistory(
            time=self.dataset.time,
            actual_thrust=thrust,
            actual_gimbal=gimbal,
            actual_thrust_lag_jacobian=derivative,
            active_set_counts=combined_counts,
            gimbal_source=self.gimbal_source,
            command_mode=command_mode,
            strict_final=command_mode == "strict",
        )

    def _evaluate_analytic(
        self,
        coordinate: np.ndarray,
        rotor_lag_seconds: float,
        *,
        command_mode: str,
        epsilon: Optional[float],
        reference: bool,
    ) -> DynamicsEvaluation:
        parameters, parameter_jacobian = self.chart.decode_with_jacobian(coordinate)
        history = self.actuator_history(
            rotor_lag_seconds,
            command_mode=command_mode,
            epsilon=epsilon,
        )
        sg = self.dataset.reference_sg if reference else self.dataset.sg
        covariance = (
            self.dataset.reference_covariance if reference else self.dataset.covariance
        )
        count = self.dataset.time.size
        residual = np.empty((count, 6))
        residual_jacobian = np.empty((count, 6, PHYSICAL_DIMENSION))
        residual_lag_jacobian = np.empty((count, 6))
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
            "nij,nj->ni", covariance.whitening, residual_lag_jacobian
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
        *,
        command_mode: str = "strict",
        epsilon: Optional[float] = None,
        reference: bool = False,
    ) -> DynamicsEvaluation:
        value = np.asarray(coordinate, dtype=float)
        return self._evaluate_analytic(
            value,
            rotor_lag_seconds,
            command_mode=command_mode,
            epsilon=epsilon,
            reference=reference,
        )

    def global_residual_jacobian(
        self,
        coordinate: Sequence[float],
        *,
        command_mode: str,
        epsilon: Optional[float],
    ) -> tuple[np.ndarray, np.ndarray, DynamicsEvaluation]:
        value = np.asarray(coordinate, dtype=float)
        if value.shape != (15,):
            raise ValueError("global coordinate/lag layout mismatch")
        rotor_lag = float(value[14])
        evaluation = self.evaluate_physical(
            value[:14],
            rotor_lag,
            command_mode=command_mode,
            epsilon=epsilon,
        )
        dimension = value.size
        jacobian = np.zeros((evaluation.residual_vector.size, dimension))
        jacobian[:, :14] = evaluation.jacobian_matrix
        jacobian[:, 14] = evaluation.whitened_lag_jacobian.reshape(-1)
        if self.parameter_prior is None:
            return evaluation.residual_vector, jacobian, evaluation
        prior = self.parameter_prior.evaluate(self.chart, value[:14])
        prior_jacobian = np.zeros((prior.residual.size, dimension))
        prior_jacobian[:, :14] = prior.jacobian
        return (
            np.concatenate((evaluation.residual_vector, prior.residual)),
            np.vstack((jacobian, prior_jacobian)),
            evaluation,
        )

    def physical_residual_jacobian(
        self, coordinate: Sequence[float], rotor_lag_seconds: float
    ) -> tuple[np.ndarray, np.ndarray, DynamicsEvaluation]:
        """Return strict data rows plus optional parameter-prior rows."""

        value = np.asarray(coordinate, dtype=float)
        evaluation = self.evaluate_physical(
            value, rotor_lag_seconds, command_mode="strict"
        )
        if self.parameter_prior is None:
            return evaluation.residual_vector, evaluation.jacobian_matrix, evaluation
        prior = self.parameter_prior.evaluate(self.chart, value)
        return (
            np.concatenate((evaluation.residual_vector, prior.residual)),
            np.vstack((evaluation.jacobian_matrix, prior.jacobian)),
            evaluation,
        )


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
    command_mode: str,
    epsilon: Optional[float],
) -> Callable[[np.ndarray], ObjectiveEvaluation]:
    def evaluate(coordinate: np.ndarray) -> ObjectiveEvaluation:
        residual, jacobian, payload = problem.global_residual_jacobian(
            coordinate,
            command_mode=command_mode,
            epsilon=epsilon,
        )
        return ObjectiveEvaluation(residual, jacobian, payload)

    return evaluate


def _physical_objective(
    problem: SingleBagDynamicsProblem,
    rotor_lag: float,
) -> Callable[[np.ndarray], ObjectiveEvaluation]:
    def evaluate(coordinate: np.ndarray) -> ObjectiveEvaluation:
        residual, jacobian, payload = problem.physical_residual_jacobian(
            coordinate, rotor_lag
        )
        return ObjectiveEvaluation(residual, jacobian, payload)

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

    initial_evaluation = evaluator(full(reduced_initial))
    residual_shape = initial_evaluation.residual.shape
    jacobian_shape = (initial_evaluation.residual.size, reduced_initial.size)
    if (
        initial_evaluation.residual.ndim != 1
        or initial_evaluation.jacobian.shape
        != (initial_evaluation.residual.size, initial.size)
        or np.any(~np.isfinite(initial_evaluation.residual))
        or np.any(~np.isfinite(initial_evaluation.jacobian))
    ):
        raise ValueError("initial standard-solver evaluation is invalid")
    # scipy.optimize.least_squares does not provide an exception boundary for
    # trial residual evaluations.  The exponential inertia chart is physical
    # for every finite coordinate, but sufficiently extreme rejected trials
    # can lose SPD through floating-point cancellation.  Return a finite cost
    # that is guaranteed to exceed the initial cost so TRF rejects such a
    # trial, matching the rejection semantics of adaptive_kkt_lm.
    maximum_initial_component = max(
        1.0, float(np.max(np.abs(initial_evaluation.residual)))
    )
    maximum_safe_component = math.sqrt(
        np.finfo(float).max / max(initial_evaluation.residual.size, 1)
    ) / 4.0
    penalty_component = min(
        maximum_safe_component, 1.0e6 * maximum_initial_component
    )
    invalid_residual = np.full(residual_shape, penalty_component)
    invalid_jacobian = np.zeros(jacobian_shape)
    invalid_trial_count = 0
    invalid_exception_types: set[str] = set()

    def safe_evaluation(reduced: np.ndarray) -> ObjectiveEvaluation:
        nonlocal invalid_trial_count
        try:
            # Promote numerical warnings from extreme matrix-exponential
            # trials to a rejected evaluation without polluting production
            # logs or continuing with NaNs.
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                evaluation = evaluator(full(reduced))
            if (
                evaluation.residual.shape != residual_shape
                or evaluation.jacobian.shape
                != (initial_evaluation.residual.size, initial.size)
                or np.any(~np.isfinite(evaluation.residual))
                or np.any(~np.isfinite(evaluation.jacobian))
            ):
                raise FloatingPointError("non-finite standard-solver trial")
            return ObjectiveEvaluation(
                evaluation.residual,
                evaluation.jacobian @ basis,
                evaluation.payload,
            )
        except (
            ValueError,
            FloatingPointError,
            OverflowError,
            RuntimeWarning,
            np.linalg.LinAlgError,
        ) as error:
            invalid_trial_count += 1
            invalid_exception_types.add(type(error).__name__)
            return ObjectiveEvaluation(invalid_residual, invalid_jacobian)

    def residual(reduced: np.ndarray) -> np.ndarray:
        return safe_evaluation(reduced).residual

    def jacobian(reduced: np.ndarray) -> np.ndarray:
        return safe_evaluation(reduced).jacobian

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
        diagnostics=(
            {
                "invalid_trial_count": invalid_trial_count,
                "invalid_trial_exception_types": sorted(
                    invalid_exception_types
                ),
            },
        ),
    )


@dataclass(frozen=True)
class EstimatorConfig:
    covariance_mode: str = "identity"
    geometric_correction: bool = True
    gimbal_source: str = "measured_sg"
    lag_mode: str = "estimated"
    initial_rotor_lag: Optional[float] = None
    initial_rotor_lag_multiplier: float = 1.0
    fixed_rotor_lag: Optional[float] = None
    lag_continuation_depth: int = 9
    lag_continuation_schedule: Optional[tuple[float, ...]] = None
    lag_continuation_enabled: bool = True
    # These are safety ceilings; gtol/ftol/xtol remain the normal termination
    # criteria.  Real-bag validation required more than the former 80/120.
    smooth_max_nfev: int = DEFAULT_SMOOTH_MAX_NFEV
    strict_max_nfev: int = DEFAULT_STRICT_MAX_NFEV
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
        if self.gimbal_source not in (
            "measured_sg",
            "measured_linear",
            "command_replay",
        ):
            raise ValueError("unknown gimbal source")
        if self.lag_mode not in ("estimated", "zero", "fixed"):
            raise ValueError("unknown lag mode")
        if self.solver_type not in ("custom_kkt_lm", "standard_least_squares"):
            raise ValueError("unknown solver type")
        coordinate = np.asarray(self.initial_physical_coordinate, dtype=float)
        if coordinate.shape != (14,) or np.any(~np.isfinite(coordinate)):
            raise ValueError("initial physical coordinate must be finite 14-D")
        object.__setattr__(self, "initial_physical_coordinate", _readonly(coordinate))
        if self.initial_rotor_lag is not None and (
            not np.isfinite(self.initial_rotor_lag)
            or self.initial_rotor_lag < 0.0
        ):
            raise ValueError("initial rotor lag must be finite and non-negative")
        if (
            not np.isfinite(self.initial_rotor_lag_multiplier)
            or self.initial_rotor_lag_multiplier <= 0.0
        ):
            raise ValueError("initial rotor lag multiplier must be positive")
        if self.fixed_rotor_lag is not None and (
            not np.isfinite(self.fixed_rotor_lag) or self.fixed_rotor_lag < 0.0
        ):
            raise ValueError("fixed rotor lag must be finite and non-negative")
        schedule = self.lag_continuation_schedule
        if schedule is not None:
            schedule = tuple(float(value) for value in schedule)
            if not schedule or any(
                not np.isfinite(value) or value <= 0.0 for value in schedule
            ):
                raise ValueError("lag continuation schedule is invalid")
            object.__setattr__(self, "lag_continuation_schedule", schedule)
        if (
            self.smooth_max_nfev < 1
            or self.strict_max_nfev < 1
            or self.lag_continuation_depth < 0
        ):
            raise ValueError("estimator termination/smoothing settings are invalid")

    @property
    def continuation_epsilon(self) -> tuple[float, ...]:
        return (
            power_of_two_epsilon(self.lag_continuation_depth)
            if self.lag_continuation_schedule is None
            else self.lag_continuation_schedule
        )


@dataclass(frozen=True)
class EstimationResult:
    physical_coordinate: np.ndarray
    rotor_lag_seconds: float
    final_smooth_rotor_lag_seconds: float
    evaluation: DynamicsEvaluation
    reference_evaluation: DynamicsEvaluation
    success: bool
    status: str
    message: str
    stages: tuple[Mapping[str, Any], ...]
    total_nfev: int
    elapsed_seconds: float
    ridge: Mapping[str, Any]
    uncertainty: Optional[ParameterCovarianceResult]
    residual_wrench_uncertainty: Optional[ResidualWrenchUncertainty]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    first_failure_stage: Optional[str] = None
    first_failure_status: Optional[str] = None
    first_failure_message: Optional[str] = None
    prior_evaluation: Optional[Any] = None
    prior_diagnostics: Mapping[str, Any] = field(
        default_factory=lambda: {"active": False}
    )
    postfit_uncertainty_status: str = "completed"
    postfit_uncertainty_failure: Optional[Mapping[str, Any]] = None

    @property
    def optimization_status(self) -> str:
        return "completed" if self.success else "failed"

    @property
    def overall_case_status(self) -> str:
        if not self.success:
            return "failed"
        if self.postfit_uncertainty_status == "completed":
            return "completed"
        return "point_estimate_completed"


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
        if lag_columns != 1:
            raise ValueError("scientific objective has exactly one lag column")
        lower[-1] = 0.0
        upper[-1] = problem.strict_lag_grid.data_support_upper
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


def common_scale_quotient_basis() -> np.ndarray:
    """Return one deterministic orthonormal basis for scale-quotient space."""

    basis = null_space(COMMON_SCALE_DIRECTION.reshape(1, -1))
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    if basis.shape != (PHYSICAL_DIMENSION, PHYSICAL_DIMENSION - 1):
        raise RuntimeError("scale quotient basis has unexpected dimension")
    return _readonly(basis)


def _covariance_component_std(covariance: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (
        np.asarray(covariance, dtype=float)
        + np.asarray(covariance, dtype=float).T
    )
    diagonal = np.diag(matrix)
    scale = float(np.max(np.abs(diagonal))) if diagonal.size else 0.0
    tolerance = max(matrix.shape) * np.finfo(float).eps * scale
    if np.any(diagonal < -tolerance):
        raise RuntimeError("physical uncertainty has negative variance")
    return _readonly(np.sqrt(np.maximum(diagonal, 0.0)))


def nominal_mass_gauge_uncertainty(
    problem: SingleBagDynamicsProblem,
    final: DynamicsEvaluation,
    uncertainty: ParameterCovarianceResult,
) -> dict[str, Any]:
    """Transform chart covariance to nominal mass and physical summaries."""

    coordinate = np.asarray(final.physical_coordinate)
    transform = np.eye(PHYSICAL_DIMENSION)
    transform[:, 0] -= COMMON_SCALE_DIRECTION
    fixed_coordinate = coordinate - coordinate[0] * COMMON_SCALE_DIRECTION
    fixed_parameters, fixed_jacobian = problem.chart.decode_with_jacobian(
        fixed_coordinate
    )
    if not np.isclose(
        fixed_parameters.mass,
        problem.model.parameters.mass,
        rtol=3.0e-15,
        atol=3.0e-15,
    ):
        raise RuntimeError("nominal-mass chart transformation is inconsistent")
    moments, axes = np.linalg.eigh(np.asarray(fixed_parameters.inertia))
    moment_jacobian = np.empty((3, PHYSICAL_DIMENSION), dtype=float)
    for index in range(3):
        axis = axes[:, index]
        moment_jacobian[index] = np.einsum(
            "i,ijp,j->p", axis, fixed_jacobian.inertia, axis
        )

    covariance_inputs = {
        "overlap_corrected": uncertainty.overlap_corrected,
        "wrench_corrected": uncertainty.wrench_corrected,
        "conservative_fusion": uncertainty.conservative_fusion,
    }
    result: dict[str, Any] = {
        "transform": _readonly(transform),
        "coordinate": _readonly(fixed_coordinate),
        "mass_kg": float(fixed_parameters.mass),
        "force_effectiveness": _readonly(
            np.asarray(fixed_parameters.force_effectiveness)
        ),
        "cog_offset_m": _readonly(np.asarray(fixed_parameters.cog_offset)),
        "principal_inertia_moments_kg_m2": _readonly(moments),
    }
    for name, covariance in covariance_inputs.items():
        fixed_covariance = transform @ np.asarray(covariance) @ transform.T
        fixed_covariance = 0.5 * (fixed_covariance + fixed_covariance.T)
        if abs(float(fixed_covariance[0, 0])) > (
            50.0
            * np.finfo(float).eps
            * max(float(np.max(np.abs(fixed_covariance))), np.finfo(float).tiny)
        ):
            raise RuntimeError("fixed-mass covariance retains mass variance")
        force_covariance = (
            fixed_jacobian.force_effectiveness
            @ fixed_covariance
            @ fixed_jacobian.force_effectiveness.T
        )
        cog_covariance = (
            fixed_jacobian.cog_offset
            @ fixed_covariance
            @ fixed_jacobian.cog_offset.T
        )
        moment_covariance = (
            moment_jacobian @ fixed_covariance @ moment_jacobian.T
        )
        result["covariance_{}".format(name)] = _readonly(fixed_covariance)
        result["force_effectiveness_std_{}".format(name)] = (
            _covariance_component_std(force_covariance)
        )
        result["cog_offset_std_{}".format(name)] = (
            _covariance_component_std(cog_covariance)
        )
        result["principal_inertia_moments_std_{}".format(name)] = (
            _covariance_component_std(moment_covariance)
        )
    return result


def estimation_diagnostics(
    *,
    problem: SingleBagDynamicsProblem,
    final: DynamicsEvaluation,
    nominal_reference: DynamicsEvaluation,
    estimated_reference: DynamicsEvaluation,
    ridge: Mapping[str, Any],
    uncertainty: ParameterCovarianceResult,
    residual_wrench: ResidualWrenchUncertainty,
    strict_lag: Mapping[str, Any],
    continuation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build post-fit diagnostics without feeding them back to optimization."""

    covariance = covariance_weighting_diagnostics(
        problem.dataset.covariance, final.acceleration_residual
    )
    reference_full_covariance = covariance_weighting_diagnostics(
        problem.dataset.reference_covariance,
        estimated_reference.acceleration_residual,
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
    wrench_variance = np.einsum(
        "ij,jk,ik->i", right, uncertainty.wrench_corrected, right
    )
    conservative_variance = np.einsum(
        "ij,jk,ik->i", right, uncertainty.conservative_fusion, right
    )
    conservative_ratio = np.full_like(overlap_variance, np.nan)
    overlap_scale = float(np.max(np.abs(overlap_variance)))
    overlap_tolerance = (
        overlap_variance.size * np.finfo(float).eps * overlap_scale
    )
    np.divide(
        conservative_variance,
        overlap_variance,
        out=conservative_ratio,
        where=overlap_variance > overlap_tolerance,
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
    quotient = common_scale_quotient_basis()
    quotient_coordinate = quotient.T @ coordinate
    quotient_jtj = (
        quotient.T @ final.jacobian_matrix.T @ final.jacobian_matrix @ quotient
    )
    quotient_conservative = (
        quotient.T @ uncertainty.conservative_fusion @ quotient
    )
    quotient_overlap = quotient.T @ uncertainty.overlap_corrected @ quotient
    conservative_order_eigenvalues = np.linalg.eigvalsh(
        0.5
        * (
            quotient_conservative
            - quotient_overlap
            + (quotient_conservative - quotient_overlap).T
        )
    )
    normalized_scale = COMMON_SCALE_DIRECTION / np.linalg.norm(
        COMMON_SCALE_DIRECTION
    )
    scale_alignment = np.abs(right @ normalized_scale)
    gauge_ridge_index = int(np.argmax(scale_alignment))
    non_gauge_indices = [
        index for index in range(right.shape[0]) if index != gauge_ridge_index
    ]
    top_indices = sorted(
        non_gauge_indices,
        key=lambda index: float(conservative_variance[index]),
        reverse=True,
    )[:3]
    singular_values = np.asarray(ridge["whitened_singular_values"])
    top_directions = []
    for index in top_indices:
        top_directions.append(
            {
                "ridge_direction_index": index,
                "singular_value": float(singular_values[index]),
                "conservative_variance": float(conservative_variance[index]),
                "conservative_to_overlap_variance_ratio": float(
                    conservative_ratio[index]
                ),
                "physical_chart_components": {
                    label: float(value)
                    for label, value in zip(PHYSICAL_CHART_LABELS, right[index])
                },
            }
        )
    residual_mean = np.mean(final.acceleration_residual, axis=0)
    acceleration_second_moment = (
        np.asarray(final.acceleration_residual).T
        @ np.asarray(final.acceleration_residual)
        / final.acceleration_residual.shape[0]
    )
    acceleration_second_moment = 0.5 * (
        acceleration_second_moment + acceleration_second_moment.T
    )
    recovered_acceleration_residual = np.einsum(
        "ij,nj->ni",
        residual_wrench.wrench_to_acceleration,
        residual_wrench.wrench,
    )
    acceleration_recovery_error = (
        recovered_acceleration_residual - final.acceleration_residual
    )
    recovery_scale = max(
        float(np.max(np.abs(final.acceleration_residual))), 1.0
    )
    recovery_tolerance = (
        100.0 * np.finfo(float).eps * recovery_scale
    )
    if np.max(np.abs(acceleration_recovery_error)) > recovery_tolerance:
        raise RuntimeError(
            "nominal-mass residual wrench does not recover acceleration residual"
        )
    sg_middle_trace = float(np.trace(uncertainty.sandwich_middle))
    residual_middle_trace = float(
        np.trace(uncertainty.sandwich_middle_residual_uncentered)
    )
    trace_scale = max(abs(sg_middle_trace), abs(residual_middle_trace))
    trace_tolerance = (
        PHYSICAL_DIMENSION * np.finfo(float).eps * trace_scale
    )
    residual_to_sg_trace_ratio = (
        residual_middle_trace / sg_middle_trace
        if sg_middle_trace > trace_tolerance
        else np.nan
    )
    nominal_mass_uncertainty = nominal_mass_gauge_uncertainty(
        problem, final, uncertainty
    )
    gimbal_error = (
        np.asarray(problem.gimbal_replay.angle)
        - np.asarray(problem.dataset.gimbal_sg_angle)
    )
    return {
        "covariance": covariance,
        "reference_full_covariance": reference_full_covariance,
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
        "lag": dict(strict_lag),
        "continuation": dict(continuation),
        "gimbal": {
            "objective_source": problem.gimbal_source,
            "command_replay_angle": _readonly(problem.gimbal_replay.angle),
            "command_replay_rmse_rad": _readonly(
                np.sqrt(np.mean(gimbal_error**2, axis=0))
            ),
            "command_replay_max_abs_error_rad": _readonly(
                np.max(np.abs(gimbal_error), axis=0)
            ),
            "command_replay_rate_limit_active_counts": dict(
                problem.gimbal_replay.active_set_counts
            ),
        },
        "quotient": {
            "basis": quotient,
            "coordinate": _readonly(quotient_coordinate),
            "jtj": _readonly(quotient_jtj),
            "covariance_naive": _readonly(
                quotient.T @ uncertainty.naive @ quotient
            ),
            "covariance_overlap_corrected": _readonly(
                quotient.T @ uncertainty.overlap_corrected @ quotient
            ),
            "covariance_wrench_corrected": _readonly(
                quotient.T @ uncertainty.wrench_corrected @ quotient
            ),
            "covariance_conservative_fusion": _readonly(
                quotient_conservative
            ),
        },
        "residual_wrench": {
            "mass_gauge_scale": residual_wrench.mass_gauge_scale,
            "fixed_mass_kg": residual_wrench.fixed_mass_kg,
            "mean": residual_wrench.mean,
            "uncentered_second_moment": (
                residual_wrench.uncentered_second_moment
            ),
            "empirical_covariance": residual_wrench.empirical_covariance,
            "empirical_std": residual_wrench.empirical_std,
            "empirical_correlation": residual_wrench.empirical_correlation,
            "sg_covariance_mean": residual_wrench.sg_covariance_mean,
            "sg_std": residual_wrench.sg_std,
            "sg_correlation": residual_wrench.sg_correlation,
            "excess_covariance_raw": residual_wrench.excess_covariance_raw,
            "excess_covariance_raw_eigenvalues": (
                residual_wrench.excess_covariance_raw_eigenvalues
            ),
            "appreciably_negative_raw_eigenvalues": (
                residual_wrench.appreciably_negative_raw_eigenvalues
            ),
            "raw_negative_eigenvalue_tolerance": (
                residual_wrench.raw_negative_eigenvalue_tolerance
            ),
            "model_discrepancy_covariance": (
                residual_wrench.model_discrepancy_covariance
            ),
            "model_discrepancy_eigenvalues": (
                residual_wrench.model_discrepancy_eigenvalues
            ),
            "model_discrepancy_std": residual_wrench.model_discrepancy_std,
            "model_discrepancy_correlation": (
                residual_wrench.model_discrepancy_correlation
            ),
            "acceleration_model_discrepancy_covariance": (
                residual_wrench.acceleration_model_discrepancy_covariance
            ),
            "acceleration_model_discrepancy_eigenvalues": (
                residual_wrench.acceleration_model_discrepancy_eigenvalues
            ),
            "acceleration_model_discrepancy_std": (
                residual_wrench.acceleration_model_discrepancy_std
            ),
            "acceleration_model_discrepancy_correlation": (
                residual_wrench.acceleration_model_discrepancy_correlation
            ),
            "closure_inverse_error_max_abs": float(
                np.max(
                    np.abs(
                        residual_wrench.wrench_to_acceleration
                        @ residual_wrench.acceleration_to_wrench
                        - np.eye(6)
                    )
                )
            ),
        },
        "conservative_fusion": {
            "interpretation": (
                "The conservative fusion covariance deliberately retains the "
                "existing SG-overlap uncertainty and adds the uncentered "
                "empirical residual score second moment without subtracting "
                "the SG contribution. It is intended as a conservative fusion "
                "distribution, not as a calibrated generative noise covariance."
            ),
            "residual_mean": _readonly(residual_mean),
            "residual_uncentered_second_moment": _readonly(
                acceleration_second_moment
            ),
            "residual_recovered_from_nominal_mass_wrench": _readonly(
                recovered_acceleration_residual
            ),
            "residual_recovery_error_from_nominal_mass_wrench": _readonly(
                acceleration_recovery_error
            ),
            "residual_recovery_error_max_abs": float(
                np.max(np.abs(acceleration_recovery_error))
            ),
            "sandwich_middle_residual_trace": residual_middle_trace,
            "sandwich_middle_sg_trace": sg_middle_trace,
            "sandwich_middle_residual_to_sg_trace_ratio": (
                residual_to_sg_trace_ratio
            ),
            "sandwich_middle_existing_wrench_trace": float(
                np.trace(uncertainty.sandwich_middle_wrench)
            ),
            "sandwich_middle_conservative_trace": float(
                np.trace(uncertainty.sandwich_middle_conservative_fusion)
            ),
            "covariance_psd_order_min_eigenvalue": float(
                conservative_order_eigenvalues[0]
            ),
            "covariance_psd_order_eigenvalues": _readonly(
                conservative_order_eigenvalues
            ),
            "variance_overlap_in_ridge_basis": _readonly(overlap_variance),
            "variance_wrench_corrected_in_ridge_basis": _readonly(
                wrench_variance
            ),
            "variance_conservative_fusion_in_ridge_basis": _readonly(
                conservative_variance
            ),
            "conservative_to_overlap_variance_ratio_in_ridge_basis": (
                _readonly(conservative_ratio)
            ),
            "exact_scale_gauge_ridge_direction_index": gauge_ridge_index,
            "exact_scale_gauge_ridge_alignment": float(
                scale_alignment[gauge_ridge_index]
            ),
            "top_ambiguous_non_gauge_directions": top_directions,
            "nominal_mass_gauge": nominal_mass_uncertainty,
        },
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
            "uncertainty_variance_wrench_corrected_in_ridge_basis": (
                _readonly(wrench_variance)
            ),
            "uncertainty_variance_conservative_fusion_in_ridge_basis": (
                _readonly(conservative_variance)
            ),
            "uncertainty_variance_inflation_in_ridge_basis": _readonly(
                inflation
            ),
        },
    }


def _point_estimate_diagnostics(
    *,
    problem: SingleBagDynamicsProblem,
    final: DynamicsEvaluation,
    nominal_reference: DynamicsEvaluation,
    estimated_reference: DynamicsEvaluation,
    ridge: Mapping[str, Any],
    strict_lag: Mapping[str, Any],
    continuation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build diagnostics that do not depend on post-fit uncertainty."""

    coordinate = np.asarray(final.physical_coordinate)
    right = np.asarray(ridge["whitened_right_singular_vectors"])
    displacement = right @ coordinate
    energy = float(coordinate @ coordinate)
    displacement_fraction = (
        np.zeros_like(displacement)
        if energy == 0.0
        else displacement**2 / energy
    )
    quotient = common_scale_quotient_basis()
    gimbal_error = (
        np.asarray(problem.gimbal_replay.angle)
        - np.asarray(problem.dataset.gimbal_sg_angle)
    )
    nominal_moments, nominal_axes = np.linalg.eigh(
        np.asarray(problem.model.parameters.inertia)
    )
    estimated_moments, estimated_axes = np.linalg.eigh(
        np.asarray(final.parameters.inertia)
    )
    diagnostics: dict[str, Any] = {
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
        "lag": dict(strict_lag),
        "continuation": dict(continuation),
        "gimbal": {
            "objective_source": problem.gimbal_source,
            "command_replay_angle": _readonly(problem.gimbal_replay.angle),
            "command_replay_rmse_rad": _readonly(
                np.sqrt(np.mean(gimbal_error**2, axis=0))
            ),
            "command_replay_max_abs_error_rad": _readonly(
                np.max(np.abs(gimbal_error), axis=0)
            ),
            "command_replay_rate_limit_active_counts": dict(
                problem.gimbal_replay.active_set_counts
            ),
        },
        "quotient": {
            "basis": quotient,
            "coordinate": _readonly(quotient.T @ coordinate),
            "jtj": _readonly(
                quotient.T
                @ final.jacobian_matrix.T
                @ final.jacobian_matrix
                @ quotient
            ),
        },
        "closure": acceleration_wrench_closure(problem.dataset, final),
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
    }
    try:
        diagnostics["covariance"] = covariance_weighting_diagnostics(
            problem.dataset.covariance, final.acceleration_residual
        )
        diagnostics["reference_full_covariance"] = (
            covariance_weighting_diagnostics(
                problem.dataset.reference_covariance,
                estimated_reference.acceleration_residual,
            )
        )
    except Exception as error:
        diagnostics["covariance_diagnostic_failure"] = {
            "exception_type": type(error).__name__,
            "message": str(error),
        }
    return diagnostics


def _parameter_prior_diagnostics(
    problem: SingleBagDynamicsProblem,
    final: DynamicsEvaluation,
) -> tuple[Optional[Any], dict[str, Any]]:
    prior = problem.parameter_prior
    if prior is None:
        return None, {"active": False}
    evaluation = prior.evaluate(problem.chart, final.physical_coordinate)
    data_jacobian = np.asarray(final.jacobian_matrix)
    prior_jacobian = np.asarray(evaluation.jacobian)
    data_information = data_jacobian.T @ data_jacobian
    prior_information = prior_jacobian.T @ prior_jacobian
    total_curvature = data_information + prior_information
    quotient = common_scale_quotient_basis()
    reduced = quotient.T @ total_curvature @ quotient
    reduced_covariance = machine_pseudoinverse_symmetric(reduced)
    augmented_covariance = quotient @ reduced_covariance @ quotient.T
    augmented_covariance = 0.5 * (
        augmented_covariance + augmented_covariance.T
    )
    metadata = dict(prior.metadata(evaluation))
    metadata.update(
        {
            "prior_residual": evaluation.residual,
            "prior_objective_sum": evaluation.objective,
            "data_objective_sum": final.cost,
            "total_objective_sum": final.cost + evaluation.objective,
            "prior_jacobian": evaluation.jacobian,
            "prior_information_matrix": _readonly(prior_information),
            "data_local_curvature": _readonly(data_information),
            "total_local_curvature": _readonly(total_curvature),
            "prior_augmented_local_curvature": _readonly(total_curvature),
            "parameter_covariance_prior_augmented_local_curvature": (
                _readonly(augmented_covariance)
            ),
            "prior_augmented_covariance_interpretation": (
                "local prior-augmented curvature / pseudo-posterior diagnostic"
            ),
            "prior_jacobian_scale_gauge_norm": float(
                np.linalg.norm(prior_jacobian @ COMMON_SCALE_DIRECTION)
            ),
            "total_curvature_scale_gauge_norm": float(
                np.linalg.norm(total_curvature @ COMMON_SCALE_DIRECTION)
            ),
        }
    )
    return evaluation, metadata


def estimate_single_bag(
    problem: SingleBagDynamicsProblem, config: EstimatorConfig
) -> EstimationResult:
    """Run one-rotor-lag continuation and exact strict-cell refinement."""

    started = time.perf_counter()
    stages: list[Mapping[str, Any]] = []
    physical = np.asarray(config.initial_physical_coordinate).copy()
    physical += (
        float(config.scale_initial_offset)
        * COMMON_SCALE_DIRECTION
        / np.linalg.norm(COMMON_SCALE_DIRECTION)
    )
    period = float(problem.dataset.rotor_history.median_period)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("rotor command history has no positive median period")
    default_initial = period * float(config.initial_rotor_lag_multiplier)
    initial_lag = (
        default_initial
        if config.initial_rotor_lag is None
        else float(config.initial_rotor_lag)
    )
    if config.lag_mode == "zero":
        rotor_lag = 0.0
    elif config.lag_mode == "fixed":
        rotor_lag = (
            period
            if config.fixed_rotor_lag is None
            else float(config.fixed_rotor_lag)
        )
    else:
        rotor_lag = initial_lag
    if not 0.0 <= rotor_lag <= problem.strict_lag_grid.data_support_upper:
        raise ValueError("initial/fixed rotor lag lies outside recorded data support")

    total_nfev = 0
    stage_success = True
    first_failure_stage: Optional[str] = None
    first_failure_status: Optional[str] = None
    first_failure_message: Optional[str] = None

    def record_failure(stage: str, solved: SolverResult) -> None:
        nonlocal stage_success, first_failure_stage
        nonlocal first_failure_status, first_failure_message
        stage_success = stage_success and solved.success
        if not solved.success and first_failure_stage is None:
            first_failure_stage = stage
            first_failure_status = solved.status
            first_failure_message = solved.message

    continuation_epsilon: list[float] = []
    continuation_lag: list[float] = []
    continuation_smooth_cost: list[float] = []
    continuation_strict_cost: list[float] = []
    continuation_command_error: list[float] = []
    continuation_lag_step: list[float] = []
    continuation_physical_step: list[float] = []
    continuation_lag_jacobian_norm: list[float] = []
    continuation_physical: list[np.ndarray] = []

    if config.lag_mode == "estimated" and config.lag_continuation_enabled:
        coordinate = np.concatenate((physical, np.asarray((rotor_lag,))))
        previous = coordinate.copy()
        for stage_index, epsilon in enumerate(config.continuation_epsilon):
            solved = _solve_stage(
                problem,
                coordinate,
                _objective_from_problem(
                    problem, command_mode="smooth", epsilon=float(epsilon)
                ),
                config=config,
                max_nfev=config.smooth_max_nfev,
                lag_columns=1,
            )
            total_nfev += solved.nfev
            coordinate = np.asarray(solved.coordinate).copy()
            smooth_evaluation = problem.evaluate_physical(
                coordinate[:14],
                float(coordinate[14]),
                command_mode="smooth",
                epsilon=float(epsilon),
            )
            strict_evaluation = problem.evaluate_physical(
                coordinate[:14], float(coordinate[14]), command_mode="strict"
            )
            smooth_prior_cost = (
                0.0
                if problem.parameter_prior is None
                else problem.parameter_prior.evaluate(
                    problem.chart, coordinate[:14]
                ).objective
            )
            command_error = float(
                np.max(
                    np.abs(
                        smooth_evaluation.actuator_history.actual_thrust
                        - strict_evaluation.actuator_history.actual_thrust
                    )
                )
            )
            continuation_epsilon.append(float(epsilon))
            continuation_lag.append(float(coordinate[14]))
            continuation_smooth_cost.append(float(smooth_evaluation.cost))
            continuation_strict_cost.append(float(strict_evaluation.cost))
            continuation_command_error.append(command_error)
            continuation_lag_step.append(float(abs(coordinate[14] - previous[14])))
            continuation_physical_step.append(
                float(np.linalg.norm(coordinate[:14] - previous[:14]))
            )
            continuation_lag_jacobian_norm.append(
                float(np.linalg.norm(smooth_evaluation.whitened_lag_jacobian))
            )
            continuation_physical.append(coordinate[:14].copy())
            stages.append(
                {
                    "stage": "smooth_continuation",
                    "index": stage_index,
                    "epsilon": float(epsilon),
                    "transition_full_width_seconds": float(epsilon * period),
                    "success": solved.success,
                    "status": solved.status,
                    "message": solved.message,
                    "nfev": solved.nfev,
                    "elapsed_seconds": solved.elapsed_seconds,
                    "rotor_lag_seconds": float(coordinate[14]),
                    "smooth_cost": float(smooth_evaluation.cost),
                    "smooth_data_cost": float(smooth_evaluation.cost),
                    "smooth_prior_cost": float(smooth_prior_cost),
                    "smooth_total_cost": float(
                        smooth_evaluation.cost + smooth_prior_cost
                    ),
                    "strict_cost_at_same_point": float(strict_evaluation.cost),
                    "command_max_error": command_error,
                }
            )
            record_failure("smooth_continuation", solved)
            previous = coordinate.copy()
            if not solved.success:
                break
        physical = coordinate[:14].copy()
        rotor_lag = float(coordinate[14])

    final_smooth_lag = float(rotor_lag)
    strict_initial_physical = physical.copy()
    candidate_rows: list[dict[str, Any]] = []
    final_neighbor_rows: list[dict[str, Any]] = []

    if config.lag_mode == "estimated":
        def profile_cell(cell: StrictLagCell) -> tuple[float, SolverResult]:
            nonlocal total_nfev
            solved = _solve_stage(
                problem,
                strict_initial_physical.copy(),
                _physical_objective(problem, cell.representative),
                config=config,
                max_nfev=config.strict_max_nfev,
                lag_columns=0,
            )
            total_nfev += solved.nfev
            record_failure("strict_cell_physical_refinement", solved)
            cost = 0.5 * float(solved.evaluation.residual @ solved.evaluation.residual)
            return cost, solved

        profile = local_strict_cell_descent(
            problem.strict_lag_grid, rotor_lag, profile_cell
        )
        selected_cell = profile.selected.cell
        selected_solver = profile.selected.payload
        physical = np.asarray(selected_solver.coordinate).copy()
        rotor_lag = float(selected_cell.representative)
        neighbor_indices = {item.cell.index for item in profile.final_neighbors}
        for item in profile.evaluated:
            solved = item.payload
            row = {
                "cell_index": item.cell.index,
                "cell_lower_seconds": item.cell.lower,
                "cell_upper_seconds": item.cell.upper,
                "representative_seconds": item.cell.representative,
                "strict_cost": item.cost,
                "strict_data_cost": float(solved.evaluation.payload.cost),
                "strict_prior_cost": float(
                    item.cost - solved.evaluation.payload.cost
                ),
                "selected": item.cell.index == selected_cell.index,
                "final_neighbor": item.cell.index in neighbor_indices,
                "optimizer_success": solved.success,
                "optimizer_status": solved.status,
                "optimizer_message": solved.message,
                "nfev": solved.nfev,
            }
            candidate_rows.append(row)
            if row["final_neighbor"]:
                final_neighbor_rows.append(row)
        stages.append(
            {
                "stage": "strict_zoh_cell_refinement",
                "success": all(
                    item.payload.success for item in profile.evaluated
                ),
                "initial_lag_seconds": final_smooth_lag,
                "selected_cell_index": selected_cell.index,
                "selected_cell_lower_seconds": selected_cell.lower,
                "selected_cell_upper_seconds": selected_cell.upper,
                "selected_representative_seconds": selected_cell.representative,
                "evaluated_cell_count": len(profile.evaluated),
                "same_physical_initial_for_every_cell": True,
            }
        )
    else:
        selected_cell = problem.strict_lag_grid.containing(rotor_lag)
        solved = _solve_stage(
            problem,
            physical,
            _physical_objective(problem, rotor_lag),
            config=config,
            max_nfev=config.strict_max_nfev,
            lag_columns=0,
        )
        total_nfev += solved.nfev
        record_failure("strict_physical_refinement", solved)
        physical = np.asarray(solved.coordinate).copy()
        cost = 0.5 * float(solved.evaluation.residual @ solved.evaluation.residual)
        candidate_rows.append(
            {
                "cell_index": selected_cell.index,
                "cell_lower_seconds": selected_cell.lower,
                "cell_upper_seconds": selected_cell.upper,
                "representative_seconds": selected_cell.representative,
                "evaluated_lag_seconds": rotor_lag,
                "strict_cost": cost,
                "strict_data_cost": float(solved.evaluation.payload.cost),
                "strict_prior_cost": float(
                    cost - solved.evaluation.payload.cost
                ),
                "selected": True,
                "final_neighbor": False,
                "optimizer_success": solved.success,
                "optimizer_status": solved.status,
                "optimizer_message": solved.message,
                "nfev": solved.nfev,
            }
        )
        stages.append(
            {
                "stage": "strict_physical_refinement",
                "success": solved.success,
                "status": solved.status,
                "message": solved.message,
                "nfev": solved.nfev,
                "elapsed_seconds": solved.elapsed_seconds,
                "rotor_lag_seconds": rotor_lag,
            }
        )

    final = problem.evaluate_physical(
        physical, rotor_lag, command_mode="strict"
    )
    if not final.actuator_history.strict_final:
        raise RuntimeError("final estimator result was not evaluated with strict ZOH")
    reference = problem.evaluate_physical(
        physical, rotor_lag, command_mode="strict", reference=True
    )
    nominal_reference = problem.evaluate_physical(
        np.zeros(PHYSICAL_DIMENSION),
        rotor_lag,
        command_mode="strict",
        reference=True,
    )
    ridge = ridge_analysis(
        final.jacobian_matrix,
        COMMON_SCALE_DIRECTION,
        final.acceleration_jacobian.reshape(-1, PHYSICAL_DIMENSION),
    )
    prior_evaluation, prior_diagnostics = _parameter_prior_diagnostics(
        problem, final
    )
    continuation = {
        "epsilon": _readonly(np.asarray(continuation_epsilon)),
        "rotor_lag_seconds": _readonly(np.asarray(continuation_lag)),
        "smooth_cost": _readonly(np.asarray(continuation_smooth_cost)),
        "strict_cost": _readonly(np.asarray(continuation_strict_cost)),
        "absolute_cost_difference": _readonly(
            np.abs(
                np.asarray(continuation_smooth_cost)
                - np.asarray(continuation_strict_cost)
            )
        ),
        "command_max_error": _readonly(np.asarray(continuation_command_error)),
        "rotor_lag_step": _readonly(np.asarray(continuation_lag_step)),
        "physical_step_norm": _readonly(np.asarray(continuation_physical_step)),
        "lag_jacobian_norm": _readonly(
            np.asarray(continuation_lag_jacobian_norm)
        ),
        "physical_coordinate": _readonly(
            np.asarray(continuation_physical).reshape(-1, PHYSICAL_DIMENSION)
        ),
    }
    strict_lag = {
        "mode": config.lag_mode,
        "data_support_lower_seconds": 0.0,
        "data_support_upper_seconds": problem.strict_lag_grid.data_support_upper,
        "lag_reached_data_support_boundary": bool(
            selected_cell.at_data_support_boundary
        ),
        "final_smooth_rotor_lag_seconds": final_smooth_lag,
        "strict_lag_cell_lower_seconds": selected_cell.lower,
        "strict_lag_cell_upper_seconds": selected_cell.upper,
        "strict_lag_cell_representative_seconds": rotor_lag,
        "strict_lag_cell_width_seconds": selected_cell.width,
        "candidate_table": candidate_rows,
        "neighbor_cells": final_neighbor_rows,
    }
    postfit_status = "completed"
    postfit_failure: Optional[dict[str, Any]] = None
    postfit_stage = "residual_wrench_uncertainty"
    postfit_residual_wrench: Optional[ResidualWrenchUncertainty] = None
    uncertainty: Optional[ParameterCovarianceResult] = None
    try:
        postfit_residual_wrench = residual_wrench_uncertainty(
            raw_residual_wrench=final.raw_residual_wrench,
            modeled_wrench=final.modeled_wrench,
            required_wrench=final.required_wrench,
            estimated_mass_kg=final.parameters.mass,
            estimated_inertia_kg_m2=final.parameters.inertia,
            fixed_mass_kg=problem.model.parameters.mass,
            lever_arm_m=(
                np.asarray(problem.dataset.pose_sensor_position_in_body)
                - np.asarray(final.parameters.cog_offset)
            ),
            reference_sigma_z=(
                problem.dataset.reference_covariance.local_sigma_z
            ),
        )
        postfit_stage = "parameter_covariance"
        uncertainty = parameter_covariances(
            final.acceleration_jacobian,
            problem.dataset.covariance,
            COMMON_SCALE_DIRECTION,
            additional_residual_covariance=(
                postfit_residual_wrench.acceleration_model_discrepancy_covariance
            ),
            uncentered_residual=final.acceleration_residual,
        )
        postfit_stage = "postfit_diagnostics"
        diagnostics = estimation_diagnostics(
            problem=problem,
            final=final,
            nominal_reference=nominal_reference,
            estimated_reference=reference,
            ridge=ridge,
            uncertainty=uncertainty,
            residual_wrench=postfit_residual_wrench,
            strict_lag=strict_lag,
            continuation=continuation,
        )
        diagnostics["postfit_uncertainty"] = {"status": "completed"}
    except Exception as error:
        postfit_status = "failed"
        postfit_failure = {
            "status": "failed",
            "failure_stage": postfit_stage,
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        diagnostics = _point_estimate_diagnostics(
            problem=problem,
            final=final,
            nominal_reference=nominal_reference,
            estimated_reference=reference,
            ridge=ridge,
            strict_lag=strict_lag,
            continuation=continuation,
        )
        diagnostics["postfit_uncertainty"] = dict(postfit_failure)
    ridge.update(diagnostics["ridge"])
    final_status = "completed" if stage_success else str(first_failure_status)
    final_message = (
        "all stages completed" if stage_success else str(first_failure_message)
    )
    return EstimationResult(
        physical_coordinate=_readonly(physical),
        rotor_lag_seconds=rotor_lag,
        final_smooth_rotor_lag_seconds=final_smooth_lag,
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
        residual_wrench_uncertainty=postfit_residual_wrench,
        diagnostics=diagnostics,
        first_failure_stage=first_failure_stage,
        first_failure_status=first_failure_status,
        first_failure_message=first_failure_message,
        prior_evaluation=prior_evaluation,
        prior_diagnostics=prior_diagnostics,
        postfit_uncertainty_status=postfit_status,
        postfit_uncertainty_failure=postfit_failure,
    )
