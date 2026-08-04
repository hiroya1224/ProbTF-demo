"""Immutable, ROS-free contracts for asynchronous flight measurements.

The adapter keeps every stream on its own authoritative timestamp sequence.
In particular, these contracts do not imply nearest-neighbour resampling or
sample-index alignment between sensors.  Bag record times are retained beside
header-derived measurement times so command issue times and transport offsets
remain available to downstream factors and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import TYPE_CHECKING, Iterator, Optional, Tuple

import numpy as np


if TYPE_CHECKING:
    # real_rosbag owns the legacy snapshot for now.  Importing it at runtime
    # would pull controller, geometry, and bag-adapter implementation into this
    # otherwise lightweight domain-model module.
    from grape_param_estim.real_rosbag import ControllerGainSnapshot
    from grape_param_estim.controller import ControllerConfig


class TimestampSource(Enum):
    """Clock used as the authoritative measurement timestamp."""

    HEADER = "header"
    RECORD = "record"


class UsageDecision(Enum):
    """Planned estimator role of one inspected topic."""

    USED = "used"
    INPUT = "input"
    INITIALIZATION = "initialization"
    DIAGNOSTIC = "diagnostic"
    DISABLED = "disabled"


def _canonical_text(value: object, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(
            "{} must be a non-empty canonical string".format(name)
        )
    return value


def _optional_text(value: object, name: str) -> Optional[str]:
    if value is None:
        return None
    return _canonical_text(value, name)


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("{} must be a finite scalar".format(name))
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _positive_optional_scalar(value: object, name: str) -> Optional[float]:
    if value is None:
        return None
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError("{} must be positive".format(name))
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < 0
    ):
        raise ValueError("{} must be a non-negative integer".format(name))
    return int(value)


def _canonical_names(
    value: object, name: str, unique: bool = True
) -> Tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise TypeError("{} must be a non-empty tuple".format(name))
    result = tuple(
        _canonical_text(member, "{} member".format(name)) for member in value
    )
    if unique and len(set(result)) != len(result):
        raise ValueError("{} must not contain duplicates".format(name))
    return result


def _readonly_times(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.ndim != 1
        or result.size == 0
        or not np.all(np.isfinite(result))
        or (result.size > 1 and np.any(np.diff(result) <= 0.0))
    ):
        raise ValueError(
            "{} must be a finite, non-empty, strictly increasing 1-D array"
            .format(name)
        )
    copied = result.copy()
    copied.setflags(write=False)
    return copied


def _readonly_matrix(
    value: object,
    rows: int,
    columns: int,
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (rows, columns) or not np.all(np.isfinite(result)):
        raise ValueError(
            "{} must be a finite array with shape ({}, {})".format(
                name, rows, columns
            )
        )
    copied = result.copy()
    copied.setflags(write=False)
    return copied


def _readonly_vector(value: object, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(
            "{} must be a finite array with shape ({},)".format(name, size)
        )
    copied = result.copy()
    copied.setflags(write=False)
    return copied


def _series_times(
    timestamp_source: object,
    times: object,
    record_times: object,
) -> Tuple[TimestampSource, np.ndarray, np.ndarray]:
    if not isinstance(timestamp_source, TimestampSource):
        raise TypeError("timestamp_source must be a TimestampSource")
    measurement = _readonly_times(times, "times")
    records = _readonly_times(record_times, "record_times")
    if measurement.shape != records.shape:
        raise ValueError(
            "times and record_times must contain the same samples"
        )
    if timestamp_source is TimestampSource.RECORD and measurement.size > 1:
        # ``times`` may be made flight-local by subtracting one common epoch;
        # its increments must nevertheless remain the bag record increments.
        if not np.allclose(
            np.diff(measurement),
            np.diff(records),
            rtol=1.0e-10,
            atol=2.0e-7,
        ):
            raise ValueError(
                "record-timestamped times must preserve record-time increments"
            )
    return timestamp_source, measurement, records


@dataclass(frozen=True)
class TimeInterval:
    """Ordered bounds in the flight data's common time coordinate.

    Measurement windows use inclusive support checks; transition-delimited
    calibration intervals may document an exclusive end in their provenance.
    """

    start: float
    end: float

    def __post_init__(self) -> None:
        start = _finite_scalar(self.start, "start")
        end = _finite_scalar(self.end, "end")
        if end <= start:
            raise ValueError("end must be greater than start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class TopicSensorContract:
    """Inspection result and estimator decision for one concrete ROS topic.

    Timing counts describe the raw inspected topic.  They may therefore report
    duplicate or non-monotonic header stamps even though every retained series
    below is required to have a strictly increasing authoritative time axis.
    Covariance provenance is descriptive rather than inferred: adapters record
    whether covariance came from a message, a preflight interval, a sensor
    specification, or was unavailable.
    """

    topic: str
    message_type: str
    timestamp_source: TimestampSource
    usage: UsageDecision
    frame_id: Optional[str]
    fields: Tuple[str, ...]
    units: Tuple[str, ...]
    sample_rate_hz: Optional[float]
    median_gap_seconds: Optional[float]
    maximum_gap_seconds: Optional[float]
    duplicate_timestamp_count: int
    nonmonotonic_timestamp_count: int
    covariance_provenance: Optional[str] = None
    unavailable_reason: Optional[str] = None
    mixed_frame_notes: Optional[str] = None

    def __post_init__(self) -> None:
        topic = _canonical_text(self.topic, "topic")
        message_type = _canonical_text(self.message_type, "message_type")
        if not isinstance(self.timestamp_source, TimestampSource):
            raise TypeError("timestamp_source must be a TimestampSource")
        if not isinstance(self.usage, UsageDecision):
            raise TypeError("usage must be a UsageDecision")
        frame_id = _optional_text(self.frame_id, "frame_id")
        fields = _canonical_names(self.fields, "fields")
        units = _canonical_names(self.units, "units", unique=False)
        if len(fields) != len(units):
            raise ValueError("fields and units must have equal length")

        sample_rate_hz = _positive_optional_scalar(
            self.sample_rate_hz, "sample_rate_hz"
        )
        median_gap_seconds = _positive_optional_scalar(
            self.median_gap_seconds, "median_gap_seconds"
        )
        maximum_gap_seconds = _positive_optional_scalar(
            self.maximum_gap_seconds, "maximum_gap_seconds"
        )
        if (
            median_gap_seconds is not None
            and maximum_gap_seconds is not None
            and maximum_gap_seconds < median_gap_seconds
        ):
            raise ValueError(
                "maximum_gap_seconds cannot be smaller than the median gap"
            )

        duplicate_count = _nonnegative_integer(
            self.duplicate_timestamp_count, "duplicate_timestamp_count"
        )
        nonmonotonic_count = _nonnegative_integer(
            self.nonmonotonic_timestamp_count,
            "nonmonotonic_timestamp_count",
        )
        covariance_provenance = _optional_text(
            self.covariance_provenance, "covariance_provenance"
        )
        unavailable_reason = _optional_text(
            self.unavailable_reason, "unavailable_reason"
        )
        mixed_frame_notes = _optional_text(
            self.mixed_frame_notes, "mixed_frame_notes"
        )
        if self.usage is UsageDecision.DISABLED and unavailable_reason is None:
            raise ValueError("disabled topics require an unavailable_reason")

        object.__setattr__(self, "topic", topic)
        object.__setattr__(self, "message_type", message_type)
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "sample_rate_hz", sample_rate_hz)
        object.__setattr__(self, "median_gap_seconds", median_gap_seconds)
        object.__setattr__(self, "maximum_gap_seconds", maximum_gap_seconds)
        object.__setattr__(
            self, "duplicate_timestamp_count", duplicate_count
        )
        object.__setattr__(
            self, "nonmonotonic_timestamp_count", nonmonotonic_count
        )
        object.__setattr__(
            self, "covariance_provenance", covariance_provenance
        )
        object.__setattr__(self, "unavailable_reason", unavailable_reason)
        object.__setattr__(self, "mixed_frame_notes", mixed_frame_notes)


@dataclass(frozen=True)
class SensorContract:
    """Immutable collection of unique per-topic sensor contracts."""

    topics: Tuple[TopicSensorContract, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.topics, tuple) or not self.topics:
            raise TypeError("topics must be a non-empty tuple")
        for contract in self.topics:
            if not isinstance(contract, TopicSensorContract):
                raise TypeError(
                    "topics must contain only TopicSensorContract values"
                )
        names = tuple(contract.topic for contract in self.topics)
        if len(set(names)) != len(names):
            raise ValueError("sensor contract topics must be unique")

    def __iter__(self) -> Iterator[TopicSensorContract]:
        return iter(self.topics)

    def __len__(self) -> int:
        return len(self.topics)

    def for_topic(self, topic: str) -> TopicSensorContract:
        """Return one exact topic contract or raise ``KeyError``."""

        for contract in self.topics:
            if contract.topic == topic:
                return contract
        raise KeyError(topic)


@dataclass(frozen=True)
class SensorExtrinsics:
    """Numeric fixed-frame geometry used by observation factors.

    ``sensor_to_body_rotation`` matrices map sensor-frame coordinates into
    the estimator body frame.  ``body_to_gyro_sensor_rotation`` maps body
    angular velocity into the gyro measurement frame.  Keeping both names
    explicit prevents an apparently harmless transpose from changing the
    likelihood.
    """

    body_frame: str
    pose_sensor_frame: str
    velocity_sensor_frame: str
    gyro_sensor_frame: str
    pose_sensor_position_in_body: np.ndarray
    pose_sensor_to_body_rotation: np.ndarray
    velocity_sensor_position_in_body: np.ndarray
    velocity_sensor_to_body_rotation: np.ndarray
    gyro_sensor_position_in_body: np.ndarray
    body_to_gyro_sensor_rotation: np.ndarray
    source: str

    def __post_init__(self) -> None:
        for name in (
            "body_frame",
            "pose_sensor_frame",
            "velocity_sensor_frame",
            "gyro_sensor_frame",
            "source",
        ):
            object.__setattr__(
                self, name, _canonical_text(getattr(self, name), name)
            )
        for name in (
            "pose_sensor_position_in_body",
            "velocity_sensor_position_in_body",
            "gyro_sensor_position_in_body",
        ):
            object.__setattr__(
                self,
                name,
                _readonly_vector(getattr(self, name), 3, name),
            )
        for name in (
            "pose_sensor_to_body_rotation",
            "velocity_sensor_to_body_rotation",
            "body_to_gyro_sensor_rotation",
        ):
            rotation = _readonly_matrix(getattr(self, name), 3, 3, name)
            if not np.allclose(
                rotation.T @ rotation,
                np.eye(3),
                rtol=0.0,
                atol=1.0e-10,
            ) or not np.isclose(
                np.linalg.det(rotation), 1.0, rtol=0.0, atol=1.0e-10
            ):
                raise ValueError("{} must be a proper rotation".format(name))
            object.__setattr__(self, name, rotation)


@dataclass(frozen=True)
class VectorSeries:
    """One labelled vector stream on its original measurement time axis."""

    times: np.ndarray
    record_times: np.ndarray
    values: np.ndarray
    field_names: Tuple[str, ...]
    timestamp_source: TimestampSource

    def __post_init__(self) -> None:
        source, times, record_times = _series_times(
            self.timestamp_source, self.times, self.record_times
        )
        field_names = _canonical_names(self.field_names, "field_names")
        values = _readonly_matrix(
            self.values, times.size, len(field_names), "values"
        )
        object.__setattr__(self, "timestamp_source", source)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "record_times", record_times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "field_names", field_names)


@dataclass(frozen=True)
class CausalVectorSeries(VectorSeries):
    """A record-time command series with explicit pre-window history.

    ``times``/``values`` remain the selected-interval samples used for
    inspection and command-observation factors.  ``history_*`` stores every
    strictly earlier issue event retained by the adapter so a delayed ZOH
    input can be reconstructed without extrapolating the first selected
    value backwards in time.
    """

    history_times: np.ndarray
    history_record_times: np.ndarray
    history_values: np.ndarray

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.timestamp_source is not TimestampSource.RECORD:
            raise ValueError("causal command history must use record time")
        _, history_times, history_record_times = _series_times(
            TimestampSource.RECORD,
            self.history_times,
            self.history_record_times,
        )
        history_values = _readonly_matrix(
            self.history_values,
            history_times.size,
            len(self.field_names),
            "history_values",
        )
        if history_times[-1] >= self.times[0]:
            raise ValueError(
                "command history must be strictly earlier than selected samples"
            )
        selected_epoch = self.record_times[0] - self.times[0]
        history_epoch = history_record_times[-1] - history_times[-1]
        if not np.isclose(
            selected_epoch, history_epoch, rtol=0.0, atol=2.0e-7
        ):
            raise ValueError(
                "command history and selected samples must share one bag epoch"
            )
        object.__setattr__(self, "history_times", history_times)
        object.__setattr__(self, "history_record_times", history_record_times)
        object.__setattr__(self, "history_values", history_values)

    @property
    def all_times(self) -> np.ndarray:
        result = np.concatenate((self.history_times, self.times))
        result.setflags(write=False)
        return result

    @property
    def all_record_times(self) -> np.ndarray:
        result = np.concatenate(
            (self.history_record_times, self.record_times)
        )
        result.setflags(write=False)
        return result

    @property
    def all_values(self) -> np.ndarray:
        result = np.vstack((self.history_values, self.values))
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class PoseSeries:
    """Position and orientation observations at their measurement times.

    ``orientations_xyzw`` stores unit quaternions, but quaternion sign is not
    part of the observation: ``q`` and ``-q`` represent the same rotation.
    Consumers must form sign-invariant SO(3) residuals and must not compare
    quaternion components directly or assume temporal sign continuity.
    """

    times: np.ndarray
    record_times: np.ndarray
    positions: np.ndarray
    orientations_xyzw: np.ndarray
    timestamp_source: TimestampSource

    def __post_init__(self) -> None:
        source, times, record_times = _series_times(
            self.timestamp_source, self.times, self.record_times
        )
        positions = _readonly_matrix(
            self.positions, times.size, 3, "positions"
        )
        orientations = _readonly_matrix(
            self.orientations_xyzw,
            times.size,
            4,
            "orientations_xyzw",
        )
        norms = np.linalg.norm(orientations, axis=1)
        if not np.allclose(norms, 1.0, rtol=1.0e-6, atol=1.0e-6):
            raise ValueError("orientations_xyzw must contain unit quaternions")
        object.__setattr__(self, "timestamp_source", source)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "record_times", record_times)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "orientations_xyzw", orientations)


@dataclass(frozen=True)
class PidDebugSeries:
    """Per-axis controller debug fields without reference-time resampling."""

    times: np.ndarray
    record_times: np.ndarray
    axis_names: Tuple[str, ...]
    target_p: np.ndarray
    error_p: np.ndarray
    target_d: np.ndarray
    error_d: np.ndarray
    total: np.ndarray
    p_term: np.ndarray
    i_term: np.ndarray
    d_term: np.ndarray
    timestamp_source: TimestampSource

    def __post_init__(self) -> None:
        source, times, record_times = _series_times(
            self.timestamp_source, self.times, self.record_times
        )
        axis_names = _canonical_names(self.axis_names, "axis_names")
        object.__setattr__(self, "timestamp_source", source)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "record_times", record_times)
        object.__setattr__(self, "axis_names", axis_names)
        for name in (
            "target_p",
            "error_p",
            "target_d",
            "error_d",
            "total",
            "p_term",
            "i_term",
            "d_term",
        ):
            object.__setattr__(
                self,
                name,
                _readonly_matrix(
                    getattr(self, name), times.size, len(axis_names), name
                ),
            )


@dataclass(frozen=True)
class ReferenceSeries:
    """Six-axis controller reference on its own timestamp sequence."""

    times: np.ndarray
    record_times: np.ndarray
    position: np.ndarray
    linear_velocity: np.ndarray
    linear_acceleration: np.ndarray
    rpy: np.ndarray
    angular_velocity: np.ndarray
    angular_acceleration: np.ndarray
    timestamp_source: TimestampSource

    def __post_init__(self) -> None:
        source, times, record_times = _series_times(
            self.timestamp_source, self.times, self.record_times
        )
        object.__setattr__(self, "timestamp_source", source)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "record_times", record_times)
        for name in (
            "position",
            "linear_velocity",
            "linear_acceleration",
            "rpy",
            "angular_velocity",
            "angular_acceleration",
        ):
            object.__setattr__(
                self,
                name,
                _readonly_matrix(getattr(self, name), times.size, 3, name),
            )


@dataclass(frozen=True)
class FlightModeSeries:
    """Recorded discrete controller-mode schedule with a causal anchor.

    ``times`` and ``states`` retain every in-interval flight-state sample.
    ``initial_*`` identifies the latest causal sample at or before the flight
    interval start, so downstream ZOH mode lookup never needs extrapolation.
    """

    times: np.ndarray
    record_times: np.ndarray
    states: np.ndarray
    initial_time: float
    initial_record_time: float
    initial_state: int
    timestamp_source: TimestampSource
    source_topic: str
    state_semantics: str

    def __post_init__(self) -> None:
        source, times, record_times = _series_times(
            self.timestamp_source, self.times, self.record_times
        )
        if source is not TimestampSource.RECORD:
            raise ValueError("flight mode schedule must use record time")
        raw_states = np.asarray(self.states)
        if (
            raw_states.shape != times.shape
            or raw_states.dtype == np.bool_
            or not np.issubdtype(raw_states.dtype, np.integer)
            or np.any(raw_states < 0)
        ):
            raise ValueError(
                "states must be a non-negative integer array aligned to times"
            )
        states = raw_states.astype(np.int64, copy=True)
        states.setflags(write=False)
        initial_time = _finite_scalar(self.initial_time, "initial_time")
        initial_record_time = _finite_scalar(
            self.initial_record_time, "initial_record_time"
        )
        initial_state = _nonnegative_integer(
            self.initial_state, "initial_state"
        )
        if initial_time > times[0] or initial_record_time > record_times[0]:
            raise ValueError("flight mode anchor must be causal")
        if not np.isclose(
            initial_record_time - initial_time,
            record_times[0] - times[0],
            rtol=0.0,
            atol=2.0e-7,
        ):
            raise ValueError("flight mode times must share one bag epoch")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "record_times", record_times)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "initial_time", initial_time)
        object.__setattr__(self, "initial_record_time", initial_record_time)
        object.__setattr__(self, "initial_state", initial_state)
        object.__setattr__(self, "timestamp_source", source)
        object.__setattr__(
            self,
            "source_topic",
            _canonical_text(self.source_topic, "source_topic"),
        )
        object.__setattr__(
            self,
            "state_semantics",
            _canonical_text(self.state_semantics, "state_semantics"),
        )


@dataclass(frozen=True)
class ImuPreflightCalibration:
    """Immutable estimates from the initial contiguous ARM_OFF interval."""

    interval: TimeInterval
    state_value: int
    imu_sample_count: int
    gyro_bias: np.ndarray
    gyro_standard_deviation: np.ndarray
    specific_force_mean: np.ndarray
    specific_force_standard_deviation: np.ndarray
    specific_force_norm_mean: float
    accelerometer_bias: Optional[np.ndarray]
    accelerometer_sample_count: int
    gravity_magnitude: float
    frame_id: str
    timestamp_source: TimestampSource
    source_topic: str
    state_topic: str
    orientation_topic: Optional[str]
    method: str
    accelerometer_unavailable_reason: Optional[str]

    def __post_init__(self) -> None:
        if not isinstance(self.interval, TimeInterval):
            raise TypeError("interval must be a TimeInterval")
        state_value = _nonnegative_integer(self.state_value, "state_value")
        imu_sample_count = _nonnegative_integer(
            self.imu_sample_count, "imu_sample_count"
        )
        if imu_sample_count < 2:
            raise ValueError("imu_sample_count must be at least two")
        gyro_bias = _readonly_vector(self.gyro_bias, 3, "gyro_bias")
        gyro_standard_deviation = _readonly_vector(
            self.gyro_standard_deviation,
            3,
            "gyro_standard_deviation",
        )
        specific_force_mean = _readonly_vector(
            self.specific_force_mean, 3, "specific_force_mean"
        )
        specific_force_standard_deviation = _readonly_vector(
            self.specific_force_standard_deviation,
            3,
            "specific_force_standard_deviation",
        )
        if np.any(gyro_standard_deviation < 0.0) or np.any(
            specific_force_standard_deviation < 0.0
        ):
            raise ValueError(
                "preflight standard deviations cannot be negative"
            )
        specific_force_norm_mean = _finite_scalar(
            self.specific_force_norm_mean, "specific_force_norm_mean"
        )
        gravity_magnitude = _finite_scalar(
            self.gravity_magnitude, "gravity_magnitude"
        )
        if specific_force_norm_mean <= 0.0 or gravity_magnitude <= 0.0:
            raise ValueError(
                "specific force norm and gravity must be positive"
            )

        accelerometer_sample_count = _nonnegative_integer(
            self.accelerometer_sample_count, "accelerometer_sample_count"
        )
        accelerometer_bias = None
        if self.accelerometer_bias is not None:
            accelerometer_bias = _readonly_vector(
                self.accelerometer_bias, 3, "accelerometer_bias"
            )
        orientation_topic = _optional_text(
            self.orientation_topic, "orientation_topic"
        )
        unavailable_reason = _optional_text(
            self.accelerometer_unavailable_reason,
            "accelerometer_unavailable_reason",
        )
        if accelerometer_bias is None:
            if accelerometer_sample_count != 0 or unavailable_reason is None:
                raise ValueError(
                    "disabled accelerometer calibration requires zero samples "
                    "and an unavailable reason"
                )
            if orientation_topic is not None:
                raise ValueError(
                    "disabled accelerometer calibration has no orientation "
                    "topic"
                )
        elif (
            accelerometer_sample_count < 2
            or unavailable_reason is not None
            or orientation_topic is None
        ):
            raise ValueError(
                "accelerometer bias requires samples and orientation "
                "provenance"
            )

        if not isinstance(self.timestamp_source, TimestampSource):
            raise TypeError("timestamp_source must be a TimestampSource")
        object.__setattr__(self, "state_value", state_value)
        object.__setattr__(self, "imu_sample_count", imu_sample_count)
        object.__setattr__(self, "gyro_bias", gyro_bias)
        object.__setattr__(
            self, "gyro_standard_deviation", gyro_standard_deviation
        )
        object.__setattr__(self, "specific_force_mean", specific_force_mean)
        object.__setattr__(
            self,
            "specific_force_standard_deviation",
            specific_force_standard_deviation,
        )
        object.__setattr__(
            self, "specific_force_norm_mean", specific_force_norm_mean
        )
        object.__setattr__(self, "accelerometer_bias", accelerometer_bias)
        object.__setattr__(
            self, "accelerometer_sample_count", accelerometer_sample_count
        )
        object.__setattr__(self, "gravity_magnitude", gravity_magnitude)
        object.__setattr__(
            self, "frame_id", _canonical_text(self.frame_id, "frame_id")
        )
        object.__setattr__(
            self,
            "source_topic",
            _canonical_text(self.source_topic, "source_topic"),
        )
        object.__setattr__(
            self,
            "state_topic",
            _canonical_text(self.state_topic, "state_topic"),
        )
        object.__setattr__(self, "orientation_topic", orientation_topic)
        object.__setattr__(
            self, "method", _canonical_text(self.method, "method")
        )
        object.__setattr__(
            self,
            "accelerometer_unavailable_reason",
            unavailable_reason,
        )


@dataclass(frozen=True)
class FlightProvenance:
    """Factual identity and record-time extent of the source bag."""

    bag_path: str
    bag_sha256: str
    bag_size_bytes: int
    bag_record_start: float
    bag_record_end: float
    adapter_revision: Optional[str] = None
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        bag_path = _canonical_text(self.bag_path, "bag_path")
        bag_sha256 = _canonical_text(self.bag_sha256, "bag_sha256")
        bag_size_bytes = _nonnegative_integer(
            self.bag_size_bytes, "bag_size_bytes"
        )
        bag_record_start = _finite_scalar(
            self.bag_record_start, "bag_record_start"
        )
        bag_record_end = _finite_scalar(self.bag_record_end, "bag_record_end")
        if bag_record_end <= bag_record_start:
            raise ValueError(
                "bag_record_end must be greater than bag_record_start"
            )
        adapter_revision = _optional_text(
            self.adapter_revision, "adapter_revision"
        )
        if not isinstance(self.notes, tuple):
            raise TypeError("notes must be a tuple")
        notes = tuple(
            _canonical_text(note, "notes member") for note in self.notes
        )
        object.__setattr__(self, "bag_path", bag_path)
        object.__setattr__(self, "bag_sha256", bag_sha256)
        object.__setattr__(self, "bag_size_bytes", bag_size_bytes)
        object.__setattr__(self, "bag_record_start", bag_record_start)
        object.__setattr__(self, "bag_record_end", bag_record_end)
        object.__setattr__(self, "adapter_revision", adapter_revision)
        object.__setattr__(self, "notes", notes)


@dataclass(frozen=True)
class FlightData:
    """Asynchronous estimator input for one selected flight interval.

    Optional streams remain absent rather than being synthesized from pose or
    projected onto another stream's time grid.  ``controller_snapshot`` is the
    existing adapter-owned :class:`ControllerGainSnapshot`; it is referenced
    only for static type checking to keep this module's runtime import light.
    """

    bag_id: str
    interval: TimeInterval
    pose: PoseSeries
    velocity: Optional[VectorSeries]
    gyro: Optional[VectorSeries]
    accelerometer: Optional[VectorSeries]
    gimbal_position: Optional[VectorSeries]
    gimbal_command: Optional[VectorSeries]
    rotor_command: Optional[VectorSeries]
    pid_debug: Optional[PidDebugSeries]
    reference: ReferenceSeries
    flight_mode: FlightModeSeries
    imu_preflight: ImuPreflightCalibration
    controller_snapshot: "ControllerGainSnapshot"
    controller_configuration: "ControllerConfig"
    sensor_extrinsics: SensorExtrinsics
    sensor_contract: SensorContract
    provenance: FlightProvenance

    def __post_init__(self) -> None:
        bag_id = _canonical_text(self.bag_id, "bag_id")
        if not isinstance(self.interval, TimeInterval):
            raise TypeError("interval must be a TimeInterval")
        if not isinstance(self.pose, PoseSeries):
            raise TypeError("pose must be a PoseSeries")
        for name in (
            "velocity",
            "gyro",
            "accelerometer",
            "gimbal_position",
            "gimbal_command",
            "rotor_command",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, VectorSeries):
                raise TypeError(
                    "{} must be a VectorSeries or None".format(name)
                )
        if self.pid_debug is not None and not isinstance(
            self.pid_debug, PidDebugSeries
        ):
            raise TypeError("pid_debug must be a PidDebugSeries or None")
        if not isinstance(self.reference, ReferenceSeries):
            raise TypeError("reference must be a ReferenceSeries")
        if not isinstance(self.flight_mode, FlightModeSeries):
            raise TypeError("flight_mode must be a FlightModeSeries")
        if not isinstance(self.imu_preflight, ImuPreflightCalibration):
            raise TypeError(
                "imu_preflight must be an ImuPreflightCalibration"
            )
        if self.controller_snapshot is None:
            raise TypeError("controller_snapshot cannot be None")
        if self.controller_configuration is None:
            raise TypeError("controller_configuration cannot be None")
        if not isinstance(self.sensor_extrinsics, SensorExtrinsics):
            raise TypeError("sensor_extrinsics must be SensorExtrinsics")
        if not isinstance(self.sensor_contract, SensorContract):
            raise TypeError("sensor_contract must be a SensorContract")
        if not isinstance(self.provenance, FlightProvenance):
            raise TypeError("provenance must be a FlightProvenance")
        object.__setattr__(self, "bag_id", bag_id)


__all__ = [
    "CausalVectorSeries",
    "FlightData",
    "FlightModeSeries",
    "FlightProvenance",
    "ImuPreflightCalibration",
    "PidDebugSeries",
    "PoseSeries",
    "ReferenceSeries",
    "SensorContract",
    "SensorExtrinsics",
    "TimeInterval",
    "TimestampSource",
    "TopicSensorContract",
    "UsageDecision",
    "VectorSeries",
]
