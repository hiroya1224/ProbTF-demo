"""Auditable fixed-grid initialization for asynchronous flight data.

This module maps :class:`~grape_param_estim.sensor_models.FlightData` onto the
canonical nonlinear batch state.  Interpolation here is initialization-only:
the original asynchronous observation series remain untouched and continue to
define likelihood factors at their own measurement times.
"""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable, Optional, Tuple

import numpy as np

from grape_param_estim.batch.layout import VariableLayout
from grape_param_estim.batch.state import BatchState
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.geometry import (
    quaternion_to_matrix,
    so3_exp,
    so3_geodesic_interpolation,
    so3_log,
)
from grape_param_estim.sensor_models import (
    FlightData,
    TimeInterval,
    TimestampSource,
)


_TIME_TOLERANCE = 2.0e-7
_PID_AXIS_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
_ROTOR_NAMES = ("rotor_1", "rotor_2", "rotor_3", "rotor_4")
_GIMBAL_NAMES = ("gimbal1", "gimbal2", "gimbal3", "gimbal4")
_KNOT_KINDS = (
    VariableKind.POSITION,
    VariableKind.ORIENTATION_TANGENT,
    VariableKind.LINEAR_VELOCITY,
    VariableKind.ANGULAR_VELOCITY,
    VariableKind.CONTROLLER_INTEGRAL,
    VariableKind.ACTUATOR_THRUST,
    VariableKind.GIMBAL_ANGLE,
)

IntegralReplay = Callable[[FlightData, np.ndarray], np.ndarray]


def _canonical_text(value: object, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError("{} must be non-empty canonical text".format(name))
    return value


def _optional_text(value: object, name: str) -> Optional[str]:
    if value is None:
        return None
    return _canonical_text(value, name)


def _positive_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("{} must be a finite positive scalar".format(name))
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("{} must be a finite positive scalar".format(name))
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("{} must be a non-negative integer".format(name))
    result = int(value)
    if result < 0:
        raise ValueError("{} must be a non-negative integer".format(name))
    return result


def _readonly_times(value: object, name: str, minimum_size: int = 2) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.ndim != 1
        or result.size < minimum_size
        or np.any(~np.isfinite(result))
        or np.any(np.diff(result) <= 0.0)
    ):
        raise ValueError(
            "{} must contain at least {} strictly increasing finite times"
            .format(name, minimum_size)
        )
    copied = result.copy()
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class InitializationFieldProvenance:
    """Exact source and fallback decision for one initialized state field."""

    field: str
    source: str
    method: str
    timestamp_source: Optional[TimestampSource]
    source_sample_count: int
    maximum_time_offset_seconds: Optional[float]
    fallback_reason: Optional[str] = None

    def __post_init__(self) -> None:
        field = _canonical_text(self.field, "field")
        source = _canonical_text(self.source, "source")
        method = _canonical_text(self.method, "method")
        if self.timestamp_source is not None and not isinstance(
            self.timestamp_source, TimestampSource
        ):
            raise TypeError(
                "timestamp_source must be a TimestampSource or None"
            )
        count = _nonnegative_integer(
            self.source_sample_count, "source_sample_count"
        )
        maximum_offset = self.maximum_time_offset_seconds
        if maximum_offset is not None:
            if (
                isinstance(maximum_offset, (bool, np.bool_))
                or not isinstance(maximum_offset, Real)
            ):
                raise TypeError(
                    "maximum_time_offset_seconds must be non-negative"
                )
            maximum_offset = float(maximum_offset)
            if not np.isfinite(maximum_offset) or maximum_offset < 0.0:
                raise ValueError(
                    "maximum_time_offset_seconds must be non-negative"
                )
            if count == 0:
                raise ValueError(
                    "a time offset requires at least one source sample"
                )
        fallback_reason = _optional_text(
            self.fallback_reason, "fallback_reason"
        )
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "source_sample_count", count)
        object.__setattr__(
            self, "maximum_time_offset_seconds", maximum_offset
        )
        object.__setattr__(self, "fallback_reason", fallback_reason)


@dataclass(frozen=True)
class InitializationProvenance:
    """Immutable per-field initialization audit without copied observations."""

    fields: Tuple[InitializationFieldProvenance, ...]
    observation_policy: str

    def __post_init__(self) -> None:
        if not isinstance(self.fields, tuple) or not self.fields:
            raise TypeError("fields must be a non-empty tuple")
        if any(
            not isinstance(value, InitializationFieldProvenance)
            for value in self.fields
        ):
            raise TypeError(
                "fields must contain InitializationFieldProvenance values"
            )
        names = tuple(value.field for value in self.fields)
        if len(set(names)) != len(names):
            raise ValueError("initialization provenance fields must be unique")
        object.__setattr__(
            self,
            "observation_policy",
            _canonical_text(self.observation_policy, "observation_policy"),
        )

    def for_field(self, field: str) -> InitializationFieldProvenance:
        """Return the exact provenance record for one canonical field."""

        for value in self.fields:
            if value.field == field:
                return value
        raise KeyError(field)


@dataclass(frozen=True)
class FixedKnotGrid:
    """Flight-phase-aligned fixed grid inside common initialization support."""

    times: np.ndarray
    period_seconds: float
    flight_interval: TimeInterval
    common_support: TimeInterval

    def __post_init__(self) -> None:
        if not isinstance(self.flight_interval, TimeInterval):
            raise TypeError("flight_interval must be a TimeInterval")
        if not isinstance(self.common_support, TimeInterval):
            raise TypeError("common_support must be a TimeInterval")
        period = _positive_scalar(self.period_seconds, "period_seconds")
        times = _readonly_times(self.times, "times")
        if not np.allclose(
            np.diff(times), period, rtol=1.0e-10, atol=2.0e-12
        ):
            raise ValueError("knot times must use one fixed period")
        if (
            self.common_support.start
            < self.flight_interval.start - _TIME_TOLERANCE
            or self.common_support.end
            > self.flight_interval.end + _TIME_TOLERANCE
        ):
            raise ValueError("common support must lie inside flight interval")
        if (
            times[0] < self.common_support.start - _TIME_TOLERANCE
            or times[-1] > self.common_support.end + _TIME_TOLERANCE
        ):
            raise ValueError("knot grid requires source extrapolation")
        phase = (times - self.flight_interval.start) / period
        if not np.allclose(
            phase, np.rint(phase), rtol=0.0, atol=2.0e-8
        ):
            raise ValueError("knot grid is not aligned to the flight interval")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "period_seconds", period)

    @property
    def count(self) -> int:
        return int(self.times.size)

    @property
    def interval(self) -> TimeInterval:
        return TimeInterval(float(self.times[0]), float(self.times[-1]))


@dataclass(frozen=True)
class FlightInitialization:
    """One immutable canonical batch initial state and its source audit."""

    bag_id: str
    grid: FixedKnotGrid
    state: BatchState
    provenance: InitializationProvenance

    def __post_init__(self) -> None:
        bag_id = _canonical_text(self.bag_id, "bag_id")
        if not isinstance(self.grid, FixedKnotGrid):
            raise TypeError("grid must be a FixedKnotGrid")
        if not isinstance(self.state, BatchState):
            raise TypeError("state must be a BatchState")
        if not isinstance(self.provenance, InitializationProvenance):
            raise TypeError("provenance must be InitializationProvenance")
        if self.state.layout.bag_ids != (bag_id,):
            raise ValueError("initial state must contain exactly its one bag")
        knot_indices = {
            key.knot_index
            for key in self.state.layout.variable_keys
            if key.bag_id == bag_id and key.knot_index is not None
        }
        if knot_indices != set(range(self.grid.count)):
            raise ValueError("initial state knots do not match the fixed grid")
        static = self.state.value(
            VariableKey(VariableKind.STATIC_PARAMETERS)
        )
        if not np.array_equal(static, np.zeros(18, dtype=float)):
            raise ValueError("static parameter initialization must be chart zero")
        required_provenance = {"knot_grid"}
        required_provenance.update(kind.value for kind in VariableKind)
        actual_provenance = {
            value.field for value in self.provenance.fields
        }
        if actual_provenance != required_provenance:
            raise ValueError(
                "provenance must describe the grid and every state field"
            )
        object.__setattr__(self, "bag_id", bag_id)

    @property
    def layout(self) -> VariableLayout:
        return self.state.layout


def _series_width(series: object, width: int, name: str) -> None:
    values = np.asarray(series.values, dtype=float)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError("{} must contain {} columns".format(name, width))


def _common_support(
    flight_interval: TimeInterval,
    sources: Tuple[Tuple[str, np.ndarray], ...],
) -> TimeInterval:
    start = max(
        (flight_interval.start,) + tuple(float(value[0]) for _, value in sources)
    )
    end = min(
        (flight_interval.end,) + tuple(float(value[-1]) for _, value in sources)
    )
    if end <= start + _TIME_TOLERANCE:
        raise ValueError(
            "required initialization streams have no common support: {}"
            .format(", ".join(name for name, _times in sources))
        )
    return TimeInterval(start, end)


def _fixed_grid_times(
    flight_interval: TimeInterval,
    common_support: TimeInterval,
    period_seconds: float,
    knot_interval: Optional[TimeInterval],
) -> np.ndarray:
    desired = common_support if knot_interval is None else knot_interval
    if not isinstance(desired, TimeInterval):
        raise TypeError("knot_interval must be a TimeInterval or None")
    if (
        desired.start < flight_interval.start - _TIME_TOLERANCE
        or desired.end > flight_interval.end + _TIME_TOLERANCE
    ):
        raise ValueError("knot_interval must lie inside FlightData.interval")
    origin = flight_interval.start
    first_index = int(np.ceil(
        (desired.start - origin - _TIME_TOLERANCE) / period_seconds
    ))
    last_index = int(np.floor(
        (desired.end - origin + _TIME_TOLERANCE) / period_seconds
    ))
    if last_index - first_index + 1 < 2:
        raise ValueError("fixed knot grid has fewer than two knots")
    result = origin + np.arange(
        first_index, last_index + 1, dtype=float
    ) * period_seconds
    if (
        result[0] < common_support.start - _TIME_TOLERANCE
        or result[-1] > common_support.end + _TIME_TOLERANCE
    ):
        raise ValueError("requested knot interval requires extrapolation")
    return result


def _check_interpolation_support(
    source_times: np.ndarray,
    target_times: np.ndarray,
    name: str,
) -> None:
    if source_times.size < 2:
        raise ValueError("{} needs at least two source samples".format(name))
    if (
        target_times[0] < source_times[0] - _TIME_TOLERANCE
        or target_times[-1] > source_times[-1] + _TIME_TOLERANCE
    ):
        raise ValueError("{} initialization requires extrapolation".format(name))


def _linear_interpolate(
    source_times: np.ndarray,
    source_values: np.ndarray,
    target_times: np.ndarray,
    name: str,
) -> np.ndarray:
    _check_interpolation_support(source_times, target_times, name)
    values = np.asarray(source_values, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] != source_times.size
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("{} values do not align with source times".format(name))
    return np.column_stack(
        [
            np.interp(target_times, source_times, values[:, column])
            for column in range(values.shape[1])
        ]
    )


def _so3_interpolate(
    source_times: np.ndarray,
    source_quaternions: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    _check_interpolation_support(
        source_times, target_times, "pose orientation SO(3)"
    )
    quaternions = np.asarray(source_quaternions, dtype=float)
    if quaternions.shape != (source_times.size, 4):
        raise ValueError("pose orientation quaternions do not align with times")
    source_rotations = np.asarray(
        tuple(quaternion_to_matrix(value) for value in quaternions)
    )
    result = np.empty((target_times.size, 3, 3), dtype=float)
    right_indices = np.searchsorted(source_times, target_times, side="right")
    right_indices = np.clip(right_indices, 1, source_times.size - 1)
    for row, (time, right) in enumerate(zip(target_times, right_indices)):
        left = right - 1
        span = source_times[right] - source_times[left]
        fraction = float((time - source_times[left]) / span)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        result[row] = so3_geodesic_interpolation(
            source_rotations[left], source_rotations[right], fraction
        )
    return result


def _zoh_interpolate(
    source_times: np.ndarray,
    source_values: np.ndarray,
    target_times: np.ndarray,
    name: str,
) -> np.ndarray:
    _check_interpolation_support(source_times, target_times, name)
    indices = np.searchsorted(source_times, target_times, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("{} initialization requires extrapolation".format(name))
    return np.asarray(source_values, dtype=float)[indices].copy()


def _maximum_nearest_offset(
    source_times: np.ndarray, target_times: np.ndarray
) -> float:
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, source_times.size - 1)
    left = np.clip(right - 1, 0, source_times.size - 1)
    offsets = np.minimum(
        np.abs(target_times - source_times[left]),
        np.abs(source_times[right] - target_times),
    )
    return float(np.max(offsets))


def _maximum_zoh_age(
    source_times: np.ndarray, target_times: np.ndarray
) -> float:
    indices = np.searchsorted(source_times, target_times, side="right") - 1
    return float(np.max(target_times - source_times[indices]))


def _smoothing_window(value: object) -> int:
    result = _nonnegative_integer(value, "pose_smoothing_window")
    if result == 0 or result % 2 == 0:
        raise ValueError("pose_smoothing_window must be a positive odd integer")
    return result


def _smooth_positions(values: np.ndarray, window: int) -> np.ndarray:
    if window == 1:
        return np.asarray(values, dtype=float).copy()
    half = window // 2
    result = np.empty_like(values, dtype=float)
    for index in range(values.shape[0]):
        start = max(0, index - half)
        end = min(values.shape[0], index + half + 1)
        result[index] = np.mean(values[start:end], axis=0)
    return result


def _smooth_rotations(values: np.ndarray, window: int) -> np.ndarray:
    if window == 1:
        return np.asarray(values, dtype=float).copy()
    half = window // 2
    result = np.empty_like(values, dtype=float)
    for index in range(values.shape[0]):
        start = max(0, index - half)
        end = min(values.shape[0], index + half + 1)
        center = values[index]
        tangent_mean = np.mean(
            np.asarray(
                tuple(
                    so3_log(center.T @ values[neighbor])
                    for neighbor in range(start, end)
                )
            ),
            axis=0,
        )
        result[index] = center @ so3_exp(tangent_mean)
    return result


def _position_velocity(
    positions: np.ndarray, knot_times: np.ndarray
) -> np.ndarray:
    edge_order = 2 if knot_times.size >= 3 else 1
    return np.gradient(
        positions, knot_times, axis=0, edge_order=edge_order
    )


def _orientation_angular_velocity(
    rotations: np.ndarray, period_seconds: float
) -> np.ndarray:
    forward = np.asarray(
        tuple(
            so3_log(rotations[index].T @ rotations[index + 1])
            / period_seconds
            for index in range(rotations.shape[0] - 1)
        )
    )
    result = np.empty((rotations.shape[0], 3), dtype=float)
    result[0] = forward[0]
    result[-1] = (
        rotations[-1].T @ rotations[-2] @ forward[-1]
    )
    for index in range(1, rotations.shape[0] - 1):
        backward_in_current = (
            rotations[index].T @ rotations[index - 1] @ forward[index - 1]
        )
        result[index] = 0.5 * (backward_in_current + forward[index])
    return result


def _pid_integral_proxy(
    flight_data: FlightData,
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
    debug = flight_data.pid_debug
    if debug is None:
        return None, "PID debug is unavailable", None
    if debug.axis_names != _PID_AXIS_NAMES:
        return None, "PID debug axis order is not canonical", None
    axis_gains_method = getattr(
        flight_data.controller_snapshot, "axis_gains", None
    )
    if not callable(axis_gains_method):
        return None, "controller snapshot has no axis_gains replay seam", None
    try:
        gains = np.asarray(axis_gains_method(), dtype=float)
    except (TypeError, ValueError) as error:
        return None, "controller snapshot axis gains are invalid: {}".format(
            error
        ), None
    if gains.shape != (6, 3) or np.any(~np.isfinite(gains)):
        return None, "controller snapshot axis gains are invalid", None
    integral_gains = gains[:, 1]
    active = np.abs(integral_gains) > 1.0e-12
    if np.any(np.abs(debug.i_term[:, ~active]) > 1.0e-10):
        return (
            None,
            "nonzero PID I term cannot be decoded with zero I gain",
            None,
        )
    result = np.zeros_like(debug.i_term, dtype=float)
    result[:, active] = debug.i_term[:, active] / integral_gains[active]
    method = (
        "linear interpolation of PID debug i_term/i_gain proxy; zero-gain "
        "axes with zero I term initialize to zero; saturation cannot be "
        "disambiguated without controller limits"
    )
    return result, None, method


def _field_provenance(
    field: str,
    source: str,
    method: str,
    timestamp_source: Optional[TimestampSource],
    source_sample_count: int,
    maximum_time_offset_seconds: Optional[float] = None,
    fallback_reason: Optional[str] = None,
) -> InitializationFieldProvenance:
    return InitializationFieldProvenance(
        field=field,
        source=source,
        method=method,
        timestamp_source=timestamp_source,
        source_sample_count=source_sample_count,
        maximum_time_offset_seconds=maximum_time_offset_seconds,
        fallback_reason=fallback_reason,
    )


def build_flight_initialization(
    flight_data: FlightData,
    knot_period_seconds: float,
    *,
    knot_interval: Optional[TimeInterval] = None,
    pose_smoothing_window: int = 5,
    integral_replay: Optional[IntegralReplay] = None,
    allow_zero_integral_fallback: bool = False,
) -> FlightInitialization:
    """Build one canonical initial state without changing observations.

    The default grid is the phase-aligned subset of the common support of all
    source streams used for initialization.  Passing ``knot_interval`` asks
    for a narrower explicit interval; any knot outside common support is
    rejected rather than extrapolated.  PID debug is the primary controller
    integral source.  A deterministic replay callback is the first fallback;
    all-zero integrals require the explicit ``allow_zero_integral_fallback``
    opt-in and retain the reason in provenance.
    """

    if not isinstance(flight_data, FlightData):
        raise TypeError("flight_data must be a FlightData")
    period = _positive_scalar(knot_period_seconds, "knot_period_seconds")
    smoothing_window = _smoothing_window(pose_smoothing_window)
    if knot_interval is not None and not isinstance(knot_interval, TimeInterval):
        raise TypeError("knot_interval must be a TimeInterval or None")
    if integral_replay is not None and not callable(integral_replay):
        raise TypeError("integral_replay must be callable or None")
    if not isinstance(
        allow_zero_integral_fallback, (bool, np.bool_)
    ):
        raise TypeError("allow_zero_integral_fallback must be boolean")

    if flight_data.rotor_command is None:
        raise ValueError("recorded rotor command is required for initialization")
    if flight_data.gimbal_position is None:
        raise ValueError("recorded gimbal joint position is required")
    _series_width(flight_data.rotor_command, 4, "rotor_command")
    _series_width(flight_data.gimbal_position, 4, "gimbal_position")
    if flight_data.rotor_command.field_names != _ROTOR_NAMES:
        raise ValueError("rotor_command must use canonical rotor_1..4 order")
    if flight_data.rotor_command.timestamp_source is not TimestampSource.RECORD:
        raise ValueError("rotor_command issue time must use rosbag record time")
    if flight_data.gimbal_position.field_names != _GIMBAL_NAMES:
        raise ValueError("gimbal_position must use canonical gimbal1..4 order")
    if flight_data.velocity is not None:
        _series_width(flight_data.velocity, 3, "velocity")
        if flight_data.velocity.field_names != ("x", "y", "z"):
            raise ValueError("velocity must use canonical x,y,z order")
    if flight_data.gyro is not None:
        _series_width(flight_data.gyro, 3, "gyro")
        if flight_data.gyro.field_names != ("x", "y", "z"):
            raise ValueError("gyro must use canonical x,y,z order")

    pid_integral, pid_fallback_reason, pid_method = _pid_integral_proxy(
        flight_data
    )
    if (
        pid_integral is not None
        and flight_data.pid_debug.timestamp_source is not TimestampSource.RECORD
    ):
        pid_integral = None
        pid_method = None
        pid_fallback_reason = "PID debug must use rosbag record time"
    if pid_integral is not None:
        integral_source = "pid_debug"
    elif integral_replay is not None:
        integral_source = "deterministic_replay"
    elif bool(allow_zero_integral_fallback):
        integral_source = "explicit_zero"
    else:
        raise ValueError(
            "controller integral initialization is unavailable: {}"
            .format(pid_fallback_reason)
        )

    support_sources = [
        ("pose", flight_data.pose.times),
        ("rotor_command", flight_data.rotor_command.times),
        ("gimbal_position", flight_data.gimbal_position.times),
    ]
    if flight_data.velocity is not None:
        support_sources.append(("velocity", flight_data.velocity.times))
    if flight_data.gyro is not None:
        support_sources.append(("gyro", flight_data.gyro.times))
    if integral_source == "pid_debug":
        support_sources.append(("pid_debug", flight_data.pid_debug.times))
    common_support = _common_support(
        flight_data.interval, tuple(support_sources)
    )
    knot_times = _fixed_grid_times(
        flight_data.interval,
        common_support,
        period,
        knot_interval,
    )
    grid = FixedKnotGrid(
        times=knot_times,
        period_seconds=period,
        flight_interval=flight_data.interval,
        common_support=common_support,
    )

    interpolated_position = _linear_interpolate(
        flight_data.pose.times,
        flight_data.pose.positions,
        grid.times,
        "pose position",
    )
    interpolated_rotation = _so3_interpolate(
        flight_data.pose.times,
        flight_data.pose.orientations_xyzw,
        grid.times,
    )
    position = _smooth_positions(
        interpolated_position, smoothing_window
    )
    orientation = _smooth_rotations(
        interpolated_rotation, smoothing_window
    )

    if flight_data.velocity is not None:
        linear_velocity = _linear_interpolate(
            flight_data.velocity.times,
            flight_data.velocity.values,
            grid.times,
            "direct linear velocity",
        )
        velocity_source = "FlightData.velocity.values"
        velocity_method = "linear interpolation for latent initialization"
        velocity_timestamp = flight_data.velocity.timestamp_source
        velocity_count = int(flight_data.velocity.times.size)
        velocity_offset = _maximum_nearest_offset(
            flight_data.velocity.times, grid.times
        )
        velocity_fallback = None
    else:
        linear_velocity = _position_velocity(position, grid.times)
        velocity_source = "smoothed pose initial path"
        velocity_method = "finite difference on fixed knot grid"
        velocity_timestamp = flight_data.pose.timestamp_source
        velocity_count = grid.count
        velocity_offset = None
        velocity_fallback = "direct velocity observation is unavailable"

    if flight_data.gyro is not None:
        measured_omega = _linear_interpolate(
            flight_data.gyro.times,
            flight_data.gyro.values,
            grid.times,
            "gyro",
        )
        sensor_omega = (
            measured_omega - flight_data.imu_preflight.gyro_bias
        )
        # FlightData stores the numeric C_SB convention explicitly:
        # y_S - b_S = C_SB omega_B.  Apply the inverse rotation here instead
        # of assuming that the sensor axes equal the estimator body axes.
        angular_velocity = (
            sensor_omega
            @ flight_data.sensor_extrinsics.body_to_gyro_sensor_rotation
        )
        omega_source = "FlightData.gyro.values"
        omega_method = (
            "linear interpolation minus sensor-frame preflight gyro bias, "
            "then C_SB transpose into the estimator body frame"
        )
        omega_timestamp = flight_data.gyro.timestamp_source
        omega_count = int(flight_data.gyro.times.size)
        omega_offset = _maximum_nearest_offset(
            flight_data.gyro.times, grid.times
        )
        omega_fallback = None
    else:
        angular_velocity = _orientation_angular_velocity(
            orientation, grid.period_seconds
        )
        omega_source = "smoothed pose orientation initial path"
        omega_method = "SO(3) finite difference on fixed knot grid"
        omega_timestamp = flight_data.pose.timestamp_source
        omega_count = grid.count
        omega_offset = None
        omega_fallback = "gyro observation is unavailable"

    if integral_source == "pid_debug":
        controller_integral = _linear_interpolate(
            flight_data.pid_debug.times,
            pid_integral,
            grid.times,
            "PID integral proxy",
        )
        integral_source_name = "FlightData.pid_debug.i_term"
        integral_timestamp = flight_data.pid_debug.timestamp_source
        integral_count = int(flight_data.pid_debug.times.size)
        integral_offset = _maximum_nearest_offset(
            flight_data.pid_debug.times, grid.times
        )
        integral_fallback = None
        integral_method = pid_method
    elif integral_source == "deterministic_replay":
        replayed = np.asarray(
            integral_replay(flight_data, grid.times), dtype=float
        )
        if replayed.shape != (grid.count, 6) or np.any(~np.isfinite(replayed)):
            raise ValueError(
                "integral_replay must return finite shape ({}, 6)"
                .format(grid.count)
            )
        controller_integral = replayed.copy()
        integral_source_name = "integral_replay callback"
        integral_timestamp = None
        integral_count = grid.count
        integral_offset = None
        integral_fallback = pid_fallback_reason
        integral_method = "caller-supplied deterministic controller replay"
    else:
        controller_integral = np.zeros((grid.count, 6), dtype=float)
        integral_source_name = "explicit all-zero fallback"
        integral_timestamp = None
        integral_count = 0
        integral_offset = None
        integral_fallback = (
            "{}; deterministic replay was not provided"
            .format(pid_fallback_reason)
        )
        integral_method = "initialize every controller integral axis to zero"

    actuator_thrust = _zoh_interpolate(
        flight_data.rotor_command.times,
        flight_data.rotor_command.values,
        grid.times,
        "recorded rotor command",
    )
    gimbal_angle = _linear_interpolate(
        flight_data.gimbal_position.times,
        flight_data.gimbal_position.values,
        grid.times,
        "recorded gimbal joint position",
    )

    include_gyro_bias = flight_data.gyro is not None
    include_accelerometer_bias = flight_data.accelerometer is not None
    if include_accelerometer_bias:
        _series_width(flight_data.accelerometer, 3, "accelerometer")
        if flight_data.accelerometer.field_names != ("x", "y", "z"):
            raise ValueError("accelerometer must use canonical x,y,z order")
        if flight_data.imu_preflight.accelerometer_bias is None:
            raise ValueError(
                "accelerometer observation requires preflight bias calibration"
            )

    keys = [VariableKey(VariableKind.STATIC_PARAMETERS)]
    if include_gyro_bias:
        keys.append(
            VariableKey(VariableKind.GYRO_BIAS, bag_id=flight_data.bag_id)
        )
    if include_accelerometer_bias:
        keys.append(
            VariableKey(
                VariableKind.ACCELEROMETER_BIAS,
                bag_id=flight_data.bag_id,
            )
        )
    for knot_index in range(grid.count):
        keys.extend(
            VariableKey(
                kind,
                bag_id=flight_data.bag_id,
                knot_index=knot_index,
            )
            for kind in _KNOT_KINDS
        )
    layout = VariableLayout(tuple(keys))
    values = {
        VariableKey(VariableKind.STATIC_PARAMETERS): np.zeros(18, dtype=float)
    }
    if include_gyro_bias:
        values[
            VariableKey(VariableKind.GYRO_BIAS, bag_id=flight_data.bag_id)
        ] = flight_data.imu_preflight.gyro_bias
    if include_accelerometer_bias:
        values[
            VariableKey(
                VariableKind.ACCELEROMETER_BIAS,
                bag_id=flight_data.bag_id,
            )
        ] = flight_data.imu_preflight.accelerometer_bias
    knot_values = {
        VariableKind.POSITION: position,
        VariableKind.ORIENTATION_TANGENT: orientation,
        VariableKind.LINEAR_VELOCITY: linear_velocity,
        VariableKind.ANGULAR_VELOCITY: angular_velocity,
        VariableKind.CONTROLLER_INTEGRAL: controller_integral,
        VariableKind.ACTUATOR_THRUST: actuator_thrust,
        VariableKind.GIMBAL_ANGLE: gimbal_angle,
    }
    for knot_index in range(grid.count):
        for kind in _KNOT_KINDS:
            values[
                VariableKey(
                    kind,
                    bag_id=flight_data.bag_id,
                    knot_index=knot_index,
                )
            ] = knot_values[kind][knot_index]
    state = BatchState(layout, values)

    pose_offset = _maximum_nearest_offset(
        flight_data.pose.times, grid.times
    )
    grid_fallback = None
    if (
        common_support.start > flight_data.interval.start + _TIME_TOLERANCE
        or common_support.end < flight_data.interval.end - _TIME_TOLERANCE
    ):
        grid_fallback = (
            "FlightData.interval was trimmed to common initialization support"
        )
    provenance_fields = [
        _field_provenance(
            "knot_grid",
            "FlightData.interval and required initialization stream support",
            "fixed period aligned to FlightData.interval start",
            None,
            grid.count,
            fallback_reason=grid_fallback,
        ),
        _field_provenance(
            VariableKind.STATIC_PARAMETERS.value,
            "nominal physical parameter center",
            "18-D static parameter chart zero",
            None,
            0,
        ),
        _field_provenance(
            VariableKind.POSITION.value,
            "FlightData.pose.positions",
            "linear interpolation then centered moving-average window {}"
            .format(smoothing_window),
            flight_data.pose.timestamp_source,
            int(flight_data.pose.times.size),
            pose_offset,
        ),
        _field_provenance(
            VariableKind.ORIENTATION_TANGENT.value,
            "FlightData.pose.orientations_xyzw",
            "SO(3) geodesic interpolation then local tangent-mean window {}"
            .format(smoothing_window),
            flight_data.pose.timestamp_source,
            int(flight_data.pose.times.size),
            pose_offset,
        ),
        _field_provenance(
            VariableKind.LINEAR_VELOCITY.value,
            velocity_source,
            velocity_method,
            velocity_timestamp,
            velocity_count,
            velocity_offset,
            velocity_fallback,
        ),
        _field_provenance(
            VariableKind.ANGULAR_VELOCITY.value,
            omega_source,
            omega_method,
            omega_timestamp,
            omega_count,
            omega_offset,
            omega_fallback,
        ),
        _field_provenance(
            VariableKind.CONTROLLER_INTEGRAL.value,
            integral_source_name,
            integral_method,
            integral_timestamp,
            integral_count,
            integral_offset,
            integral_fallback,
        ),
        _field_provenance(
            VariableKind.ACTUATOR_THRUST.value,
            "FlightData.rotor_command.values",
            "causal zero-order hold at recorded command issue time",
            flight_data.rotor_command.timestamp_source,
            int(flight_data.rotor_command.times.size),
            _maximum_zoh_age(
                flight_data.rotor_command.times, grid.times
            ),
        ),
        _field_provenance(
            VariableKind.GIMBAL_ANGLE.value,
            "FlightData.gimbal_position.values",
            "linear interpolation of recorded actual joint position",
            flight_data.gimbal_position.timestamp_source,
            int(flight_data.gimbal_position.times.size),
            _maximum_nearest_offset(
                flight_data.gimbal_position.times, grid.times
            ),
        ),
    ]
    if include_gyro_bias:
        provenance_fields.append(
            _field_provenance(
                VariableKind.GYRO_BIAS.value,
                flight_data.imu_preflight.source_topic,
                flight_data.imu_preflight.method,
                flight_data.imu_preflight.timestamp_source,
                flight_data.imu_preflight.imu_sample_count,
            )
        )
    else:
        provenance_fields.append(
            _field_provenance(
                VariableKind.GYRO_BIAS.value,
                "bag-local bias block omitted",
                "no gyro likelihood variable is initialized",
                None,
                0,
                fallback_reason="gyro observation is unavailable",
            )
        )
    if include_accelerometer_bias:
        provenance_fields.append(
            _field_provenance(
                VariableKind.ACCELEROMETER_BIAS.value,
                flight_data.imu_preflight.source_topic,
                flight_data.imu_preflight.method,
                flight_data.imu_preflight.timestamp_source,
                flight_data.imu_preflight.accelerometer_sample_count,
            )
        )
    else:
        provenance_fields.append(
            _field_provenance(
                VariableKind.ACCELEROMETER_BIAS.value,
                "bag-local bias block omitted",
                "no accelerometer likelihood variable is initialized",
                None,
                0,
                fallback_reason=(
                    flight_data.imu_preflight.accelerometer_unavailable_reason
                    or "accelerometer observation is disabled"
                ),
            )
        )
    provenance = InitializationProvenance(
        fields=tuple(provenance_fields),
        observation_policy=(
            "initialization interpolation only; FlightData observation "
            "series remain asynchronous and unchanged"
        ),
    )
    return FlightInitialization(
        bag_id=flight_data.bag_id,
        grid=grid,
        state=state,
        provenance=provenance,
    )


__all__ = [
    "FixedKnotGrid",
    "FlightInitialization",
    "InitializationFieldProvenance",
    "InitializationProvenance",
    "IntegralReplay",
    "build_flight_initialization",
]
