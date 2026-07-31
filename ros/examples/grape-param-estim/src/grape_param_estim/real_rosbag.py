"""Record-time adapter from a Grape rosbag to the estimator contracts.

ROS is deliberately imported only by :func:`read_grape_rosbag_arrays`.
Everything from episode selection through interpolation and covariance
calibration is an array-level API and can be tested without a ROS install.
Only CoG position and baselink orientation enter ``PoseObservations``;
odometry twist, IMU, commands and joint data never enter the likelihood.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple
import hashlib

import numpy as np

from grape_param_estim.controller import (
    ControllerConfig,
    PIDConfig,
)
from grape_param_estim.geometry import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_vector_from_matrix,
)
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    ControllerState,
    PoseObservations,
    ReferenceState,
)


DEFAULT_GRAPE_BAG = (
    "/home/leus/catkin_ws/bags/grape-drone/20260613_grape_hovering/"
    "20260613_grape_hovering_3_2026-06-13-15-12-51.bag"
)

COG_ODOM_TOPIC = "/gimbalrotor/uav/cog/odom"
BASELINK_ODOM_TOPIC = "/gimbalrotor/uav/baselink/odom"
PID_TOPIC = "/gimbalrotor/debug/pose/pid"
FLIGHT_STATE_TOPIC = "/gimbalrotor/flight_state"
JOINT_STATE_TOPIC = "/gimbalrotor/joint_states"
FOUR_AXIS_COMMAND_TOPIC = "/gimbalrotor/four_axes/command"
GAIN_TOPICS = (
    ("xy", "/gimbalrotor/controller/xy/parameter_updates"),
    ("z", "/gimbalrotor/controller/z/parameter_updates"),
    (
        "roll_pitch",
        "/gimbalrotor/controller/roll_pitch/parameter_updates",
    ),
    ("yaw", "/gimbalrotor/controller/yaw/parameter_updates"),
)
GIMBAL_JOINT_NAMES = ("gimbal1", "gimbal2", "gimbal3", "gimbal4")
ARM_OFF_FLIGHT_STATE = 0
TAKEOFF_FLIGHT_STATE = 3
LAND_FLIGHT_STATE = 4
HOVER_FLIGHT_STATE = 5
STOP_FLIGHT_STATE = 6

TOPIC_TYPE_CONTRACT = (
    (COG_ODOM_TOPIC, "nav_msgs/Odometry"),
    (BASELINK_ODOM_TOPIC, "nav_msgs/Odometry"),
    (PID_TOPIC, "aerial_robot_msgs/PoseControlPid"),
    (FLIGHT_STATE_TOPIC, "std_msgs/UInt8"),
    (JOINT_STATE_TOPIC, "sensor_msgs/JointState"),
    (FOUR_AXIS_COMMAND_TOPIC, "spinal/FourAxisCommand"),
) + tuple(
    (topic, "dynamic_reconfigure/Config") for _group, topic in GAIN_TOPICS
)

PID_AXIS_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
PID_CONFIG_FIELD_NAMES = (
    "p_gain",
    "i_gain",
    "d_gain",
    "limit_sum",
    "limit_p",
    "limit_i",
    "limit_d",
    "limit_error_p",
    "limit_error_i",
    "limit_error_d",
)


def _finite_times(value, name, minimum_size=1):
    result = np.asarray(value, dtype=float)
    if (
        result.ndim != 1
        or result.size < minimum_size
        or np.any(~np.isfinite(result))
        or (result.size > 1 and np.any(np.diff(result) <= 0.0))
    ):
        raise ValueError("{} must be strictly increasing record times".format(
            name
        ))
    return result.copy()


def _finite_matrix(value, rows, columns, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (rows, columns) or np.any(~np.isfinite(result)):
        raise ValueError(
            "{} must have finite shape ({}, {})".format(
                name, rows, columns
            )
        )
    return result.copy()


@dataclass(frozen=True)
class TimedVectorSeries:
    """A strictly ordered record-time vector stream."""

    record_times: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        times = _finite_times(self.record_times, "series record_times")
        values = np.asarray(self.values, dtype=float)
        if (
            values.ndim != 2
            or values.shape[0] != times.size
            or values.shape[1] < 1
            or np.any(~np.isfinite(values))
        ):
            raise ValueError("series values must be a finite member matrix")
        object.__setattr__(self, "record_times", times)
        object.__setattr__(self, "values", values.copy())


@dataclass(frozen=True)
class PidReferenceSeries:
    """PoseControlPid fields sampled on rosbag record time."""

    record_times: np.ndarray
    target_position: np.ndarray
    target_linear_velocity: np.ndarray
    target_rpy: np.ndarray
    target_angular_velocity: np.ndarray
    total: np.ndarray
    p_term: np.ndarray
    i_term: np.ndarray
    d_term: np.ndarray

    def __post_init__(self) -> None:
        times = _finite_times(self.record_times, "PID record_times")
        object.__setattr__(self, "record_times", times)
        for name in (
            "target_position",
            "target_linear_velocity",
            "target_rpy",
            "target_angular_velocity",
        ):
            object.__setattr__(
                self,
                name,
                _finite_matrix(getattr(self, name), times.size, 3, name),
            )
        for name in ("total", "p_term", "i_term", "d_term"):
            object.__setattr__(
                self,
                name,
                _finite_matrix(getattr(self, name), times.size, 6, name),
            )


@dataclass(frozen=True)
class FlightStateSeries:
    """Recorded flight state samples; values are not estimator observations."""

    record_times: np.ndarray
    states: np.ndarray

    def __post_init__(self) -> None:
        times = _finite_times(self.record_times, "flight-state record_times")
        states = np.asarray(self.states, dtype=np.int64)
        if states.shape != times.shape:
            raise ValueError("flight states must align with record times")
        object.__setattr__(self, "record_times", times)
        object.__setattr__(self, "states", states.copy())


@dataclass(frozen=True)
class ControllerGainEvents:
    """Dynamic-reconfigure gain updates from the four controller groups."""

    record_times: np.ndarray
    groups: Tuple[str, ...]
    gains: np.ndarray
    pid_control_flags: np.ndarray

    def __post_init__(self) -> None:
        times = _finite_times(self.record_times, "gain-event record_times")
        groups = tuple(str(value) for value in self.groups)
        gains = _finite_matrix(self.gains, times.size, 3, "gain events")
        flags = np.asarray(self.pid_control_flags, dtype=bool)
        if len(groups) != times.size or flags.shape != times.shape:
            raise ValueError("gain event fields must align")
        allowed = {value[0] for value in GAIN_TOPICS}
        if any(group not in allowed for group in groups):
            raise ValueError("unknown controller gain group")
        object.__setattr__(self, "record_times", times)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "gains", gains)
        object.__setattr__(self, "pid_control_flags", flags.copy())


@dataclass(frozen=True)
class RosbagArrayData:
    """ROS-free arrays parsed from the topics required by one episode."""

    bag_path: str
    bag_sha256: str
    bag_size_bytes: int
    bag_record_start: float
    bag_record_end: float
    topic_names: Tuple[str, ...]
    topic_types: Tuple[str, ...]
    cog_position: TimedVectorSeries
    baselink_orientation: TimedVectorSeries
    pid: PidReferenceSeries
    flight_state: FlightStateSeries
    controller_gain_events: ControllerGainEvents
    joint_position: TimedVectorSeries
    joint_names: Tuple[str, str, str, str]
    commanded_thrust: TimedVectorSeries

    def __post_init__(self) -> None:
        start = float(self.bag_record_start)
        end = float(self.bag_record_end)
        size = int(self.bag_size_bytes)
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError("bag record range is invalid")
        if size < 0:
            raise ValueError("bag size cannot be negative")
        names = tuple(str(value) for value in self.topic_names)
        types = tuple(str(value) for value in self.topic_types)
        if len(names) != len(types) or not names:
            raise ValueError("topic provenance must contain names and types")
        if tuple(self.joint_names) != GIMBAL_JOINT_NAMES:
            raise ValueError("joint series must use canonical gimbal order")
        if self.cog_position.values.shape[1] != 3:
            raise ValueError("CoG position must contain xyz")
        if self.baselink_orientation.values.shape[1] != 4:
            raise ValueError("baselink orientation must contain xyzw")
        if self.joint_position.values.shape[1] != 4:
            raise ValueError("joint position must contain four gimbals")
        if self.commanded_thrust.values.shape[1] != 4:
            raise ValueError("commanded thrust must contain four rotors")
        object.__setattr__(self, "bag_path", str(self.bag_path))
        object.__setattr__(self, "bag_sha256", str(self.bag_sha256))
        object.__setattr__(self, "bag_size_bytes", size)
        object.__setattr__(self, "bag_record_start", start)
        object.__setattr__(self, "bag_record_end", end)
        object.__setattr__(self, "topic_names", names)
        object.__setattr__(self, "topic_types", types)
        object.__setattr__(self, "joint_names", GIMBAL_JOINT_NAMES)


@dataclass(frozen=True)
class ControllerGainSnapshot:
    """Effective gains at the window start and how each became effective."""

    groups: Tuple[str, ...]
    record_times: np.ndarray
    gains: np.ndarray
    pid_control_flags: np.ndarray
    source_kinds: Tuple[str, ...]

    def __post_init__(self) -> None:
        groups = tuple(str(value) for value in self.groups)
        expected = tuple(value[0] for value in GAIN_TOPICS)
        if groups != expected:
            raise ValueError("snapshot groups must use canonical order")
        times = np.asarray(self.record_times, dtype=float)
        gains = np.asarray(self.gains, dtype=float)
        flags = np.asarray(self.pid_control_flags, dtype=bool)
        source_kinds = tuple(str(value) for value in self.source_kinds)
        if (
            times.shape != (4,)
            or gains.shape != (4, 3)
            or flags.shape != (4,)
            or len(source_kinds) != 4
            or np.any(~np.isfinite(times))
            or np.any(~np.isfinite(gains))
            or np.any(gains < 0.0)
        ):
            raise ValueError("controller snapshot is invalid")
        allowed_sources = {
            "static_controller_configuration",
            "dynamic_reconfigure_applied",
        }
        if any(value not in allowed_sources for value in source_kinds):
            raise ValueError("controller snapshot source is invalid")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "record_times", times.copy())
        object.__setattr__(self, "gains", gains.copy())
        object.__setattr__(self, "pid_control_flags", flags.copy())
        object.__setattr__(self, "source_kinds", source_kinds)

    def axis_gains(self) -> np.ndarray:
        """Return p/i/d rows ordered x,y,z,roll,pitch,yaw."""

        return np.asarray(
            (
                self.gains[0],
                self.gains[0],
                self.gains[1],
                self.gains[2],
                self.gains[2],
                self.gains[3],
            )
        )


@dataclass(frozen=True)
class EpisodeProvenance:
    """Factual source, timing, calibration and anchor provenance."""

    bag_path: str
    bag_sha256: str
    bag_size_bytes: int
    bag_record_start: float
    bag_record_end: float
    time_basis: str
    requested_window_start: float
    requested_window_end: float
    source_available_start: float
    source_available_end: float
    resample_period: float
    selected_flight_state: int
    flight_transition_record_times: np.ndarray
    flight_transition_states: np.ndarray
    static_window_start: float
    static_window_end: float
    static_position_samples: int
    static_position_inliers: int
    static_orientation_samples: int
    static_orientation_inliers: int
    static_position_center: np.ndarray
    static_orientation_xyzw: np.ndarray
    covariance_outlier_threshold: float
    covariance_eigenvalue_floor: float
    controller_state_anchor_record_time: float
    joint_anchor_record_time: float
    thrust_anchor_record_time: float
    thrust_anchor_kind: str
    reference_acceleration_kind: str
    controller_static_source: str
    controller_source_revision: str
    topic_names: Tuple[str, ...]
    topic_types: Tuple[str, ...]

    def __post_init__(self) -> None:
        numeric = np.asarray(
            (
                self.bag_record_start,
                self.bag_record_end,
                self.requested_window_start,
                self.requested_window_end,
                self.source_available_start,
                self.source_available_end,
                self.resample_period,
                self.static_window_start,
                self.static_window_end,
                self.covariance_outlier_threshold,
                self.covariance_eigenvalue_floor,
                self.controller_state_anchor_record_time,
                self.joint_anchor_record_time,
                self.thrust_anchor_record_time,
            ),
            dtype=float,
        )
        if np.any(~np.isfinite(numeric)) or self.resample_period <= 0.0:
            raise ValueError("episode provenance contains invalid numbers")
        transition_times = _finite_times(
            self.flight_transition_record_times,
            "flight transition record_times",
        )
        transition_states = np.asarray(
            self.flight_transition_states, dtype=np.int64
        )
        if transition_states.shape != transition_times.shape:
            raise ValueError("flight transition provenance must align")
        center = np.asarray(self.static_position_center, dtype=float)
        quaternion = np.asarray(self.static_orientation_xyzw, dtype=float)
        if (
            center.shape != (3,)
            or quaternion.shape != (4,)
            or np.any(~np.isfinite(center))
            or np.any(~np.isfinite(quaternion))
        ):
            raise ValueError("static pose provenance is invalid")
        if len(self.topic_names) != len(self.topic_types):
            raise ValueError("topic provenance must align")
        object.__setattr__(
            self, "selected_flight_state", int(self.selected_flight_state)
        )
        object.__setattr__(
            self, "flight_transition_record_times", transition_times
        )
        object.__setattr__(
            self, "flight_transition_states", transition_states.copy()
        )
        object.__setattr__(self, "static_position_center", center.copy())
        object.__setattr__(
            self, "static_orientation_xyzw", quaternion.copy()
        )
        object.__setattr__(
            self, "topic_names", tuple(str(v) for v in self.topic_names)
        )
        object.__setattr__(
            self, "topic_types", tuple(str(v) for v in self.topic_types)
        )


@dataclass(frozen=True)
class RealFlightEpisode:
    """Direct input contract for the real-data strong/weak problems."""

    record_times: np.ndarray
    window_start_record_time: float
    window_end_record_time: float
    window_start_local_time: float
    window_end_local_time: float
    observations: PoseObservations
    references: Tuple[ReferenceState, ...]
    controller_configuration: ControllerConfig
    initial_controller_state: ControllerState
    initial_actuator_state: ActuatorState
    controller_snapshot: ControllerGainSnapshot
    provenance: EpisodeProvenance

    def __post_init__(self) -> None:
        times = _finite_times(
            self.record_times, "resampled record_times", minimum_size=2
        )
        if times.size != self.observations.times.size:
            raise ValueError("record and observation times must align")
        if len(self.references) != times.size:
            raise ValueError("reference and observation lengths must agree")
        if not np.allclose(
            self.observations.times,
            times - times[0],
            atol=2.0e-7,
            rtol=0.0,
        ):
            raise ValueError("observation time must be episode-relative")
        if not np.isclose(
            self.window_start_record_time, times[0], atol=2.0e-7, rtol=0.0
        ):
            raise ValueError("window start must equal first record time")
        if not np.isclose(
            self.window_end_record_time, times[-1], atol=2.0e-7, rtol=0.0
        ):
            raise ValueError("window end must equal last record time")
        object.__setattr__(self, "record_times", times)


def _compress_flight_states(series: FlightStateSeries):
    changed = np.concatenate(
        ((True,), series.states[1:] != series.states[:-1])
    )
    return series.record_times[changed], series.states[changed]


def _flight_episodes(series: FlightStateSeries):
    times, states = _compress_flight_states(series)
    allowed_airborne = {
        TAKEOFF_FLIGHT_STATE,
        LAND_FLIGHT_STATE,
        HOVER_FLIGHT_STATE,
    }
    episodes = []
    for start_index in np.flatnonzero(states == TAKEOFF_FLIGHT_STATE):
        end_index = None
        for index in range(start_index + 1, states.size):
            if states[index] == STOP_FLIGHT_STATE:
                end_index = index
                break
            if states[index] not in allowed_airborne:
                break
        if end_index is not None:
            episodes.append(
                (
                    float(times[start_index]),
                    float(times[end_index]),
                    start_index,
                    end_index,
                )
            )
    if not episodes:
        raise ValueError("no complete TAKEOFF-to-STOP flight episode found")
    return times, states, tuple(episodes)


def select_continuous_flight_window(
    series: FlightStateSeries,
    bag_record_start: float,
    episode_index: int = 0,
    start_local: Optional[float] = None,
    end_local: Optional[float] = None,
    window_state: Optional[int] = HOVER_FLIGHT_STATE,
):
    """Select one continuous state interval inside one complete flight.

    State 5 (HOVER) is the deliberately conservative default: the target bag
    has already reached a controlled airborne condition there.  Passing
    ``window_state=None`` exposes the complete TAKEOFF-to-STOP interval for
    diagnostics, but callers must opt in explicitly.
    """

    transitions, states, episodes = _flight_episodes(series)
    index = int(episode_index)
    if index < 0:
        index += len(episodes)
    if index < 0 or index >= len(episodes):
        raise ValueError("episode_index is outside the complete flights")
    episode_start, episode_end, first, last = episodes[index]
    interval_start = episode_start
    interval_end = episode_end
    if window_state is not None:
        selected_state = int(window_state)
        matching = [
            transition_index
            for transition_index in range(first, last)
            if states[transition_index] == selected_state
        ]
        if not matching:
            raise ValueError(
                "flight episode has no continuous state={} interval".format(
                    selected_state
                )
            )
        if len(matching) != 1:
            raise ValueError(
                "flight episode has multiple state={} intervals; select a "
                "local window explicitly with window_state=None".format(
                    selected_state
                )
            )
        state_index = matching[0]
        interval_start = float(transitions[state_index])
        interval_end = float(transitions[state_index + 1])
    requested_start = (
        interval_start
        if start_local is None
        else float(bag_record_start) + float(start_local)
    )
    requested_end = (
        interval_end
        if end_local is None
        else float(bag_record_start) + float(end_local)
    )
    tolerance = 2.0e-7
    if (
        requested_start < interval_start - tolerance
        or requested_end > interval_end + tolerance
        or requested_end <= requested_start
    ):
        raise ValueError(
            "requested window must stay inside one continuous selected-state "
            "interval of one complete flight episode; flight segments cannot "
            "be concatenated"
        )
    return (
        requested_start,
        requested_end,
        transitions[first:last + 1].copy(),
        states[first:last + 1].copy(),
        episode_start,
        episode_end,
    )


def _static_window_before_episode(
    series: FlightStateSeries, episode_start: float
):
    times, states = _compress_flight_states(series)
    candidates = []
    for index in range(times.size - 1):
        if (
            states[index] == ARM_OFF_FLIGHT_STATE
            and times[index + 1] <= episode_start
        ):
            candidates.append((float(times[index]), float(times[index + 1])))
    if not candidates:
        raise ValueError("no preflight ARM_OFF calibration interval found")
    return candidates[-1]


def robust_covariance(
    samples: np.ndarray,
    outlier_threshold: float = 6.0,
    eigenvalue_floor: float = 1.0e-12,
):
    """Median/MAD-filtered covariance with only a numerical SPD floor."""

    values = np.asarray(samples, dtype=float)
    threshold = float(outlier_threshold)
    floor = float(eigenvalue_floor)
    if (
        values.ndim != 2
        or values.shape[0] < max(4, values.shape[1] + 1)
        or values.shape[1] < 1
        or np.any(~np.isfinite(values))
        or not np.isfinite(threshold)
        or threshold <= 0.0
        or not np.isfinite(floor)
        or floor <= 0.0
    ):
        raise ValueError("robust covariance input is invalid")
    center = np.median(values, axis=0)
    deviation = np.abs(values - center)
    scale = 1.4826 * np.median(deviation, axis=0)
    scale = np.maximum(scale, np.sqrt(floor))
    inliers = np.all(deviation <= threshold * scale, axis=1)
    if np.count_nonzero(inliers) < values.shape[1] + 1:
        raise ValueError("too few robust covariance inliers")
    covariance = np.cov(values[inliers], rowvar=False)
    covariance = np.atleast_2d(covariance)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (
        eigenvectors
        @ np.diag(np.maximum(eigenvalues, floor))
        @ eigenvectors.T
    )
    return center, covariance, inliers


def robust_pose_covariances(
    position_samples: np.ndarray,
    orientation_samples: np.ndarray,
    outlier_threshold: float = 6.0,
    eigenvalue_floor: float = 1.0e-12,
):
    """Calibrate translation and SO(3)-tangent covariance preflight."""

    position_center, translation_covariance, position_inliers = (
        robust_covariance(
            position_samples, outlier_threshold, eigenvalue_floor
        )
    )
    quaternions = np.asarray(orientation_samples, dtype=float)
    if (
        quaternions.ndim != 2
        or quaternions.shape[0] < 4
        or quaternions.shape[1] != 4
        or np.any(~np.isfinite(quaternions))
    ):
        raise ValueError("orientation calibration samples are invalid")
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms <= 1.0e-12):
        raise ValueError("orientation calibration contains zero quaternion")
    quaternions = quaternions / norms[:, None]
    quaternions[quaternions @ quaternions[0] < 0.0] *= -1.0
    _eigenvalues, eigenvectors = np.linalg.eigh(
        quaternions.T @ quaternions
    )
    mean_quaternion = eigenvectors[:, -1]
    if mean_quaternion[3] < 0.0:
        mean_quaternion *= -1.0
    mean_rotation = quaternion_to_matrix(mean_quaternion)
    tangent = np.asarray(
        [
            rotation_vector_from_matrix(
                mean_rotation.T @ quaternion_to_matrix(value)
            )
            for value in quaternions
        ]
    )
    _rotation_center, rotation_covariance, orientation_inliers = (
        robust_covariance(tangent, outlier_threshold, eigenvalue_floor)
    )
    return (
        position_center,
        mean_quaternion,
        translation_covariance,
        rotation_covariance,
        position_inliers,
        orientation_inliers,
    )


def linear_resample(
    source_times: Sequence[float],
    source_values: np.ndarray,
    target_times: Sequence[float],
) -> np.ndarray:
    """Linearly interpolate a finite vector stream without extrapolation."""

    source = _finite_times(source_times, "source record_times")
    target = _finite_times(target_times, "target record_times")
    values = np.asarray(source_values, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] != source.size
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("source values do not align with record times")
    tolerance = 2.0e-7
    if target[0] < source[0] - tolerance or target[-1] > source[-1] + tolerance:
        raise ValueError("interpolation target requires extrapolation")
    return np.column_stack(
        [np.interp(target, source, values[:, index])
         for index in range(values.shape[1])]
    )


def quaternion_slerp_resample(
    source_times: Sequence[float],
    source_quaternions: np.ndarray,
    target_times: Sequence[float],
) -> np.ndarray:
    """Sign-continuous xyzw SLERP on rosbag record time."""

    source = _finite_times(source_times, "quaternion source record_times")
    target = _finite_times(target_times, "quaternion target record_times")
    quaternions = np.asarray(source_quaternions, dtype=float)
    if (
        quaternions.shape != (source.size, 4)
        or np.any(~np.isfinite(quaternions))
    ):
        raise ValueError("source quaternion shape is invalid")
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms <= 1.0e-12):
        raise ValueError("source contains a zero quaternion")
    quaternions = quaternions / norms[:, None]
    for index in range(1, quaternions.shape[0]):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
            quaternions[index] *= -1.0
    tolerance = 2.0e-7
    if target[0] < source[0] - tolerance or target[-1] > source[-1] + tolerance:
        raise ValueError("SLERP target requires extrapolation")
    result = np.empty((target.size, 4), dtype=float)
    right_indices = np.searchsorted(source, target, side="right")
    right_indices = np.clip(right_indices, 1, source.size - 1)
    for row, (time, right) in enumerate(zip(target, right_indices)):
        left = right - 1
        fraction = (time - source[left]) / (source[right] - source[left])
        fraction = float(np.clip(fraction, 0.0, 1.0))
        first = quaternions[left]
        second = quaternions[right]
        cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
        if cosine > 0.9995:
            value = first + fraction * (second - first)
        else:
            angle = float(np.arccos(cosine))
            sine = float(np.sin(angle))
            value = (
                np.sin((1.0 - fraction) * angle) / sine * first
                + np.sin(fraction * angle) / sine * second
            )
        result[row] = value / np.linalg.norm(value)
    return result


def _select_controller_snapshot(
    events: ControllerGainEvents,
    window_start: float,
    window_end: float,
    initial_group_gains: np.ndarray,
    initial_record_time: float,
) -> ControllerGainSnapshot:
    """Reconstruct effective gains, treating ``pid_control_flag`` as a gate.

    The controller PID objects load their initial gains from ROS parameters.
    A dynamic-reconfigure message with ``pid_control_flag=false`` does not
    apply its displayed values to those objects.  Consequently the static
    controller configuration is the initial truth and only true events can
    alter it.
    """

    groups = tuple(value[0] for value in GAIN_TOPICS)
    initial = _finite_matrix(
        initial_group_gains, len(groups), 3, "initial controller gains"
    )
    initial_time = float(initial_record_time)
    if not np.isfinite(initial_time) or initial_time > window_start:
        raise ValueError("initial controller snapshot time is invalid")
    selected_times = []
    selected_gains = []
    selected_flags = []
    source_kinds = []
    for group_index, group in enumerate(groups):
        gains = initial[group_index].copy()
        snapshot_time = initial_time
        source_kind = "static_controller_configuration"
        group_indices = [
            index
            for index, value in enumerate(events.groups)
            if value == group and events.record_times[index] <= window_end
        ]
        before_window = [
            index
            for index in group_indices
            if events.record_times[index] <= window_start
        ]
        if before_window:
            first = before_window[0]
            if (
                not events.pid_control_flags[first]
                and not np.allclose(
                    events.gains[first], gains, atol=1.0e-12, rtol=0.0
                )
            ):
                raise ValueError(
                    "inactive startup {} gain event disagrees with the "
                    "static controller configuration".format(group)
                )
            if not events.pid_control_flags[first]:
                snapshot_time = float(events.record_times[first])
        for event_index in before_window:
            if events.pid_control_flags[event_index]:
                gains = events.gains[event_index].copy()
                snapshot_time = float(events.record_times[event_index])
                source_kind = "dynamic_reconfigure_applied"
        for event_index in group_indices:
            event_time = events.record_times[event_index]
            if event_time <= window_start:
                continue
            if (
                events.pid_control_flags[event_index]
                and not np.array_equal(events.gains[event_index], gains)
            ):
                raise ValueError(
                    "applied controller gains change inside the selected "
                    "episode"
                )
        selected_times.append(snapshot_time)
        selected_gains.append(gains)
        selected_flags.append(source_kind == "dynamic_reconfigure_applied")
        source_kinds.append(source_kind)
    return ControllerGainSnapshot(
        groups,
        np.asarray(selected_times),
        np.asarray(selected_gains),
        np.asarray(selected_flags),
        tuple(source_kinds),
    )


def _controller_group_gains(configuration: ControllerConfig) -> np.ndarray:
    axis_gains = np.asarray(
        [
            (pid.p_gain, pid.i_gain, pid.d_gain)
            for pid in configuration.pid
        ],
        dtype=float,
    )
    if axis_gains.shape != (6, 3) or np.any(~np.isfinite(axis_gains)):
        raise ValueError("base controller gains are invalid")
    if (
        not np.array_equal(axis_gains[0], axis_gains[1])
        or not np.array_equal(axis_gains[3], axis_gains[4])
    ):
        raise ValueError(
            "base controller must share gains across xy and roll/pitch"
        )
    return axis_gains[[0, 2, 3, 5]].copy()


def _controller_configuration(
    snapshot: ControllerGainSnapshot,
    initial_height: float,
    base: Optional[ControllerConfig],
) -> ControllerConfig:
    selected = ControllerConfig.grape() if base is None else base
    gains = snapshot.axis_gains()
    pids = []
    for index, original in enumerate(selected.pid):
        values = dict(original.__dict__)
        values.update(
            p_gain=float(gains[index, 0]),
            i_gain=float(gains[index, 1]),
            d_gain=float(gains[index, 2]),
        )
        pids.append(PIDConfig(**values))
    return ControllerConfig(
        pid=tuple(pids),
        xy_control_mode=selected.xy_control_mode,
        need_yaw_d_control=selected.need_yaw_d_control,
        start_roll_pitch_integration_height=(
            selected.start_roll_pitch_integration_height
        ),
        initial_height=float(initial_height),
        source_compatible_gyro_term=selected.source_compatible_gyro_term,
    )


def _latest_index_at_or_before(times: np.ndarray, query: float, name: str):
    index = int(np.searchsorted(times, query, side="right") - 1)
    if index < 0:
        raise ValueError("{} has no causal anchor before the window".format(name))
    return index


def _check_stream_gap(
    times: np.ndarray, start: float, end: float, maximum_gap: float, name: str
):
    left = max(0, int(np.searchsorted(times, start, side="right") - 1))
    right = min(times.size, int(np.searchsorted(times, end, side="left") + 1))
    selected = times[left:right]
    if selected.size < 2 or np.max(np.diff(selected)) > maximum_gap:
        raise ValueError("{} contains a gap inside the flight window".format(name))


def build_real_flight_episode(
    arrays: RosbagArrayData,
    sample_period: float = 0.04,
    episode_index: int = 0,
    start_local: Optional[float] = None,
    end_local: Optional[float] = None,
    window_state: Optional[int] = HOVER_FLIGHT_STATE,
    covariance_outlier_threshold: float = 6.0,
    covariance_eigenvalue_floor: float = 1.0e-12,
    base_controller_configuration: Optional[ControllerConfig] = None,
    actuator_parameters: Optional[ActuatorParameters] = None,
    controller_static_source: str = "ControllerConfig.grape",
    controller_source_revision: str = "",
) -> RealFlightEpisode:
    """Build one resampled, continuous, pose-only real flight episode."""

    period = float(sample_period)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("sample_period must be positive")
    (
        requested_start,
        requested_end,
        transition_times,
        transition_states,
        complete_start,
        _complete_end,
    ) = select_continuous_flight_window(
        arrays.flight_state,
        arrays.bag_record_start,
        episode_index,
        start_local,
        end_local,
        window_state,
    )
    static_start, static_end = _static_window_before_episode(
        arrays.flight_state, complete_start
    )
    static_position_mask = (
        (arrays.cog_position.record_times >= static_start)
        & (arrays.cog_position.record_times < static_end)
    )
    static_orientation_mask = (
        (arrays.baselink_orientation.record_times >= static_start)
        & (arrays.baselink_orientation.record_times < static_end)
    )
    (
        static_position,
        static_orientation,
        translation_covariance,
        rotation_covariance,
        position_inliers,
        orientation_inliers,
    ) = robust_pose_covariances(
        arrays.cog_position.values[static_position_mask],
        arrays.baselink_orientation.values[static_orientation_mask],
        covariance_outlier_threshold,
        covariance_eigenvalue_floor,
    )
    base_configuration = (
        ControllerConfig.grape()
        if base_controller_configuration is None
        else base_controller_configuration
    )
    snapshot = _select_controller_snapshot(
        arrays.controller_gain_events,
        requested_start,
        requested_end,
        _controller_group_gains(base_configuration),
        arrays.bag_record_start,
    )
    configuration = _controller_configuration(
        snapshot, static_position[2], base_configuration
    )

    available_start = max(
        requested_start,
        arrays.cog_position.record_times[0],
        arrays.baselink_orientation.record_times[0],
        arrays.pid.record_times[0],
        arrays.commanded_thrust.record_times[0],
    )
    available_end = min(
        requested_end,
        arrays.cog_position.record_times[-1],
        arrays.baselink_orientation.record_times[-1],
        arrays.pid.record_times[-1],
    )
    if available_end - available_start < period:
        raise ValueError("common record-time flight window is too short")
    maximum_gap = 0.10
    for times, name in (
        (arrays.cog_position.record_times, "CoG odometry"),
        (arrays.baselink_orientation.record_times, "baselink odometry"),
        (arrays.pid.record_times, "PoseControlPid"),
    ):
        _check_stream_gap(
            times, available_start, available_end, maximum_gap, name
        )
    # Treat the requested end as exclusive.  In the default state-selected
    # case it is exactly the transition into the next flight state.
    exclusive_end = np.nextafter(available_end, available_start)
    sample_count = int(np.floor(
        (exclusive_end - available_start) / period
    )) + 1
    if sample_count < 2:
        raise ValueError("common record-time flight window has too few samples")
    relative_times = np.arange(sample_count, dtype=float) * period
    record_times = available_start + relative_times
    position = linear_resample(
        arrays.cog_position.record_times,
        arrays.cog_position.values,
        record_times,
    )
    orientation = quaternion_slerp_resample(
        arrays.baselink_orientation.record_times,
        arrays.baselink_orientation.values,
        record_times,
    )
    observations = PoseObservations(
        times=relative_times,
        position=position,
        orientation_xyzw=orientation,
        translation_covariance=translation_covariance,
        rotation_covariance=rotation_covariance,
    )

    target_position = linear_resample(
        arrays.pid.record_times, arrays.pid.target_position, record_times
    )
    target_velocity = linear_resample(
        arrays.pid.record_times,
        arrays.pid.target_linear_velocity,
        record_times,
    )
    unwrapped_rpy = np.unwrap(arrays.pid.target_rpy, axis=0)
    target_rpy = linear_resample(
        arrays.pid.record_times, unwrapped_rpy, record_times
    )
    target_omega = linear_resample(
        arrays.pid.record_times,
        arrays.pid.target_angular_velocity,
        record_times,
    )
    feedforward = (
        arrays.pid.total
        - arrays.pid.p_term
        - arrays.pid.i_term
        - arrays.pid.d_term
    )
    target_acceleration = linear_resample(
        arrays.pid.record_times, feedforward, record_times
    )
    references = tuple(
        ReferenceState(
            position=target_position[index],
            linear_velocity=target_velocity[index],
            linear_acceleration=target_acceleration[index, :3],
            rpy=target_rpy[index],
            angular_velocity=target_omega[index],
            angular_acceleration=target_acceleration[index, 3:],
        )
        for index in range(sample_count)
    )

    controller_state_index = _latest_index_at_or_before(
        arrays.pid.record_times,
        record_times[0],
        "PoseControlPid controller state",
    )
    initial_i_term = arrays.pid.i_term[controller_state_index]
    axis_gains = snapshot.axis_gains()
    integral = np.zeros(6, dtype=float)
    for axis in range(6):
        gain = axis_gains[axis, 1]
        limit_i = configuration.pid[axis].limit_i
        if abs(initial_i_term[axis]) >= limit_i - 1.0e-10:
            raise ValueError(
                "saturated PID I term cannot uniquely reconstruct the "
                "controller integral state"
            )
        if gain > 0.0:
            integral[axis] = initial_i_term[axis] / gain
        elif abs(initial_i_term[axis]) > 1.0e-12:
            raise ValueError("nonzero I term cannot be decoded with zero gain")
    history_mask = (
        (arrays.cog_position.record_times >= complete_start)
        & (arrays.cog_position.record_times <= record_times[0])
    )
    roll_pitch_active = bool(
        np.any(
            arrays.cog_position.values[history_mask, 2]
            - configuration.initial_height
            > configuration.start_roll_pitch_integration_height
        )
    )
    controller_state = ControllerState(integral, roll_pitch_active)

    joint_index = _latest_index_at_or_before(
        arrays.joint_position.record_times,
        record_times[0],
        "joint_states",
    )
    thrust_index = _latest_index_at_or_before(
        arrays.commanded_thrust.record_times,
        record_times[0],
        "four_axes command",
    )
    selected_actuator_parameters = (
        ActuatorParameters()
        if actuator_parameters is None
        else actuator_parameters
    )
    thrust = np.clip(
        arrays.commanded_thrust.values[thrust_index],
        selected_actuator_parameters.minimum_thrust,
        selected_actuator_parameters.maximum_thrust,
    )
    gimbal = np.clip(
        arrays.joint_position.values[joint_index],
        -selected_actuator_parameters.maximum_gimbal_angle,
        selected_actuator_parameters.maximum_gimbal_angle,
    )
    actuator_state = ActuatorState(thrust, gimbal)

    provenance = EpisodeProvenance(
        bag_path=arrays.bag_path,
        bag_sha256=arrays.bag_sha256,
        bag_size_bytes=arrays.bag_size_bytes,
        bag_record_start=arrays.bag_record_start,
        bag_record_end=arrays.bag_record_end,
        time_basis="rosbag_record_time",
        requested_window_start=requested_start,
        requested_window_end=requested_end,
        source_available_start=available_start,
        source_available_end=available_end,
        resample_period=period,
        selected_flight_state=(-1 if window_state is None else window_state),
        flight_transition_record_times=transition_times,
        flight_transition_states=transition_states,
        static_window_start=static_start,
        static_window_end=static_end,
        static_position_samples=int(np.count_nonzero(static_position_mask)),
        static_position_inliers=int(np.count_nonzero(position_inliers)),
        static_orientation_samples=int(
            np.count_nonzero(static_orientation_mask)
        ),
        static_orientation_inliers=int(
            np.count_nonzero(orientation_inliers)
        ),
        static_position_center=static_position,
        static_orientation_xyzw=static_orientation,
        covariance_outlier_threshold=covariance_outlier_threshold,
        covariance_eigenvalue_floor=covariance_eigenvalue_floor,
        controller_state_anchor_record_time=(
            arrays.pid.record_times[controller_state_index]
        ),
        joint_anchor_record_time=(
            arrays.joint_position.record_times[joint_index]
        ),
        thrust_anchor_record_time=(
            arrays.commanded_thrust.record_times[thrust_index]
        ),
        thrust_anchor_kind="clipped_four_axes_command_proxy",
        reference_acceleration_kind="PoseControlPid_total_minus_P_I_D",
        controller_static_source=str(controller_static_source),
        controller_source_revision=str(controller_source_revision),
        topic_names=arrays.topic_names,
        topic_types=arrays.topic_types,
    )
    return RealFlightEpisode(
        record_times=record_times,
        window_start_record_time=record_times[0],
        window_end_record_time=record_times[-1],
        window_start_local_time=(
            record_times[0] - arrays.bag_record_start
        ),
        window_end_local_time=(
            record_times[-1] - arrays.bag_record_start
        ),
        observations=observations,
        references=references,
        controller_configuration=configuration,
        initial_controller_state=controller_state,
        initial_actuator_state=actuator_state,
        controller_snapshot=snapshot,
        provenance=provenance,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _first_scalar(values, field_name):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size < 1 or not np.isfinite(result[0]):
        raise ValueError("PoseControlPid {} is empty".format(field_name))
    return float(result[0])


def _series(times, values):
    return TimedVectorSeries(np.asarray(times), np.asarray(values))


def read_grape_rosbag_arrays(
    path: str = DEFAULT_GRAPE_BAG,
    compute_sha256: bool = True,
) -> RosbagArrayData:
    """Read factual message fields using rosbag record time, lazily importing ROS."""

    try:
        import rosbag  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise RuntimeError(
            "rosbag is required only for reading; source the ROS workspace"
        ) from error
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("rosbag does not exist: {}".format(source))
    topics = tuple(value[0] for value in TOPIC_TYPE_CONTRACT)
    gain_group_by_topic = {topic: group for group, topic in GAIN_TOPICS}
    cog_times, cog_position = [], []
    base_times, base_orientation = [], []
    pid_times = []
    pid_target_position, pid_target_velocity = [], []
    pid_target_rpy, pid_target_omega = [], []
    pid_total, pid_p, pid_i, pid_d = [], [], [], []
    flight_times, flight_states = [], []
    gain_times, gain_groups, gains, gain_flags = [], [], [], []
    joint_times, joint_position = [], []
    command_times, command_thrust = [], []

    with rosbag.Bag(str(source), "r") as bag:
        bag_start = float(bag.get_start_time())
        bag_end = float(bag.get_end_time())
        topic_info = bag.get_type_and_topic_info().topics
        for topic, expected_type in TOPIC_TYPE_CONTRACT:
            if topic not in topic_info:
                raise ValueError("required topic is missing: {}".format(topic))
            actual_type = str(topic_info[topic].msg_type)
            if actual_type != expected_type:
                raise ValueError(
                    "{} has type {}, expected {}".format(
                        topic, actual_type, expected_type
                    )
                )
        topic_types = tuple(str(topic_info[v].msg_type) for v in topics)
        for topic, message, stamp in bag.read_messages(topics=topics):
            record_time = float(stamp.to_sec())
            if topic == COG_ODOM_TOPIC:
                value = message.pose.pose.position
                cog_times.append(record_time)
                cog_position.append((value.x, value.y, value.z))
            elif topic == BASELINK_ODOM_TOPIC:
                value = message.pose.pose.orientation
                base_times.append(record_time)
                base_orientation.append((value.x, value.y, value.z, value.w))
            elif topic == PID_TOPIC:
                axes = tuple(getattr(message, name) for name in PID_AXIS_NAMES)
                pid_times.append(record_time)
                pid_target_position.append(
                    tuple(float(value.target_p) for value in axes[:3])
                )
                pid_target_velocity.append(
                    tuple(float(value.target_d) for value in axes[:3])
                )
                pid_target_rpy.append(
                    tuple(float(value.target_p) for value in axes[3:])
                )
                pid_target_omega.append(
                    tuple(float(value.target_d) for value in axes[3:])
                )
                pid_total.append(
                    tuple(_first_scalar(value.total, "total") for value in axes)
                )
                pid_p.append(
                    tuple(_first_scalar(value.p_term, "p_term") for value in axes)
                )
                pid_i.append(
                    tuple(_first_scalar(value.i_term, "i_term") for value in axes)
                )
                pid_d.append(
                    tuple(_first_scalar(value.d_term, "d_term") for value in axes)
                )
            elif topic == FLIGHT_STATE_TOPIC:
                flight_times.append(record_time)
                flight_states.append(int(message.data))
            elif topic == JOINT_STATE_TOPIC:
                lookup = {
                    str(name): float(value)
                    for name, value in zip(message.name, message.position)
                }
                if any(name not in lookup for name in GIMBAL_JOINT_NAMES):
                    raise ValueError("joint_states is missing a Grape gimbal")
                joint_times.append(record_time)
                joint_position.append(
                    tuple(lookup[name] for name in GIMBAL_JOINT_NAMES)
                )
            elif topic == FOUR_AXIS_COMMAND_TOPIC:
                value = np.asarray(message.base_thrust, dtype=float)
                if value.shape != (4,) or np.any(~np.isfinite(value)):
                    raise ValueError("FourAxisCommand base_thrust is invalid")
                command_times.append(record_time)
                command_thrust.append(value)
            elif topic in gain_group_by_topic:
                doubles = {
                    str(value.name): float(value.value)
                    for value in message.doubles
                }
                required = ("p_gain", "i_gain", "d_gain")
                if any(name not in doubles for name in required):
                    raise ValueError("dynamic controller gain is incomplete")
                bools = {
                    str(value.name): bool(value.value)
                    for value in message.bools
                }
                gain_times.append(record_time)
                gain_groups.append(gain_group_by_topic[topic])
                gains.append(tuple(doubles[name] for name in required))
                gain_flags.append(bools.get("pid_control_flag", False))

    return RosbagArrayData(
        bag_path=str(source),
        bag_sha256=_sha256(source) if compute_sha256 else "",
        bag_size_bytes=source.stat().st_size,
        bag_record_start=bag_start,
        bag_record_end=bag_end,
        topic_names=topics,
        topic_types=topic_types,
        cog_position=_series(cog_times, cog_position),
        baselink_orientation=_series(base_times, base_orientation),
        pid=PidReferenceSeries(
            np.asarray(pid_times),
            np.asarray(pid_target_position),
            np.asarray(pid_target_velocity),
            np.asarray(pid_target_rpy),
            np.asarray(pid_target_omega),
            np.asarray(pid_total),
            np.asarray(pid_p),
            np.asarray(pid_i),
            np.asarray(pid_d),
        ),
        flight_state=FlightStateSeries(
            np.asarray(flight_times), np.asarray(flight_states)
        ),
        controller_gain_events=ControllerGainEvents(
            np.asarray(gain_times),
            tuple(gain_groups),
            np.asarray(gains),
            np.asarray(gain_flags),
        ),
        joint_position=_series(joint_times, joint_position),
        joint_names=GIMBAL_JOINT_NAMES,
        commanded_thrust=_series(command_times, command_thrust),
    )


def load_grape_rosbag_episode(
    path: str = DEFAULT_GRAPE_BAG,
    sample_period: float = 0.04,
    episode_index: int = 0,
    start_local: Optional[float] = None,
    end_local: Optional[float] = None,
    window_state: Optional[int] = HOVER_FLIGHT_STATE,
    compute_sha256: bool = True,
    controller_source_revision: str = "",
) -> RealFlightEpisode:
    """Read and build one real episode; all topic alignment uses record time."""

    arrays = read_grape_rosbag_arrays(path, compute_sha256=compute_sha256)
    return build_real_flight_episode(
        arrays,
        sample_period=sample_period,
        episode_index=episode_index,
        start_local=start_local,
        end_local=end_local,
        window_state=window_state,
        controller_source_revision=controller_source_revision,
    )


def save_real_flight_episode(path: str, episode: RealFlightEpisode) -> Path:
    """Persist the adapter output as arrays only; loading never needs pickle."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    references = episode.references
    configuration = episode.controller_configuration
    pid_configuration = np.asarray(
        [
            [getattr(pid, name) for name in PID_CONFIG_FIELD_NAMES]
            for pid in configuration.pid
        ]
    )
    provenance = episode.provenance
    np.savez_compressed(
        str(destination),
        schema=np.asarray(("grape-weak-constraint/phase5-real-episode",)),
        record_times=episode.record_times,
        times=episode.observations.times,
        window_start_record_time=np.asarray(
            (episode.window_start_record_time,)
        ),
        window_end_record_time=np.asarray((episode.window_end_record_time,)),
        window_start_local_time=np.asarray((episode.window_start_local_time,)),
        window_end_local_time=np.asarray((episode.window_end_local_time,)),
        observations_position=episode.observations.position,
        observations_orientation_xyzw=(
            episode.observations.orientation_xyzw
        ),
        observation_translation_covariance=(
            episode.observations.translation_covariance
        ),
        observation_rotation_covariance=(
            episode.observations.rotation_covariance
        ),
        reference_position=np.asarray([value.position for value in references]),
        reference_linear_velocity=np.asarray(
            [value.linear_velocity for value in references]
        ),
        reference_linear_acceleration=np.asarray(
            [value.linear_acceleration for value in references]
        ),
        reference_rpy=np.asarray([value.rpy for value in references]),
        reference_angular_velocity=np.asarray(
            [value.angular_velocity for value in references]
        ),
        reference_angular_acceleration=np.asarray(
            [value.angular_acceleration for value in references]
        ),
        controller_pid_axis_names=np.asarray(PID_AXIS_NAMES),
        controller_pid_field_names=np.asarray(PID_CONFIG_FIELD_NAMES),
        controller_pid_configuration=pid_configuration,
        controller_xy_control_mode=np.asarray(
            (configuration.xy_control_mode,)
        ),
        controller_need_yaw_d_control=np.asarray(
            (configuration.need_yaw_d_control,), dtype=bool
        ),
        controller_start_roll_pitch_integration_height=np.asarray(
            (configuration.start_roll_pitch_integration_height,)
        ),
        controller_initial_height=np.asarray((configuration.initial_height,)),
        controller_source_compatible_gyro_term=np.asarray(
            (configuration.source_compatible_gyro_term,), dtype=bool
        ),
        initial_controller_integral_error=(
            episode.initial_controller_state.integral_error
        ),
        initial_roll_pitch_integration_active=np.asarray(
            (
                episode.initial_controller_state
                .roll_pitch_integration_active,
            ),
            dtype=bool,
        ),
        initial_actuator_thrust=episode.initial_actuator_state.thrust,
        initial_actuator_gimbal_angle=(
            episode.initial_actuator_state.gimbal_angle
        ),
        snapshot_group=np.asarray(episode.controller_snapshot.groups),
        snapshot_record_time=episode.controller_snapshot.record_times,
        snapshot_gain=episode.controller_snapshot.gains,
        snapshot_pid_control_flag=(
            episode.controller_snapshot.pid_control_flags
        ),
        snapshot_source_kind=np.asarray(
            episode.controller_snapshot.source_kinds
        ),
        provenance_bag_path=np.asarray((provenance.bag_path,)),
        provenance_bag_sha256=np.asarray((provenance.bag_sha256,)),
        provenance_bag_size_bytes=np.asarray(
            (provenance.bag_size_bytes,), dtype=np.int64
        ),
        provenance_bag_record_start=np.asarray(
            (provenance.bag_record_start,)
        ),
        provenance_bag_record_end=np.asarray((provenance.bag_record_end,)),
        provenance_time_basis=np.asarray((provenance.time_basis,)),
        provenance_requested_window_start=np.asarray(
            (provenance.requested_window_start,)
        ),
        provenance_requested_window_end=np.asarray(
            (provenance.requested_window_end,)
        ),
        provenance_source_available_start=np.asarray(
            (provenance.source_available_start,)
        ),
        provenance_source_available_end=np.asarray(
            (provenance.source_available_end,)
        ),
        provenance_resample_period=np.asarray(
            (provenance.resample_period,)
        ),
        provenance_selected_flight_state=np.asarray(
            (provenance.selected_flight_state,), dtype=np.int64
        ),
        provenance_flight_transition_record_times=(
            provenance.flight_transition_record_times
        ),
        provenance_flight_transition_states=(
            provenance.flight_transition_states
        ),
        provenance_static_window_start=np.asarray(
            (provenance.static_window_start,)
        ),
        provenance_static_window_end=np.asarray(
            (provenance.static_window_end,)
        ),
        provenance_static_sample_counts=np.asarray(
            (
                provenance.static_position_samples,
                provenance.static_position_inliers,
                provenance.static_orientation_samples,
                provenance.static_orientation_inliers,
            ),
            dtype=np.int64,
        ),
        provenance_static_position_center=(
            provenance.static_position_center
        ),
        provenance_static_orientation_xyzw=(
            provenance.static_orientation_xyzw
        ),
        provenance_covariance_outlier_threshold=np.asarray(
            (provenance.covariance_outlier_threshold,)
        ),
        provenance_covariance_eigenvalue_floor=np.asarray(
            (provenance.covariance_eigenvalue_floor,)
        ),
        provenance_controller_state_anchor_record_time=np.asarray(
            (provenance.controller_state_anchor_record_time,)
        ),
        provenance_joint_anchor_record_time=np.asarray(
            (provenance.joint_anchor_record_time,)
        ),
        provenance_thrust_anchor_record_time=np.asarray(
            (provenance.thrust_anchor_record_time,)
        ),
        provenance_thrust_anchor_kind=np.asarray(
            (provenance.thrust_anchor_kind,)
        ),
        provenance_reference_acceleration_kind=np.asarray(
            (provenance.reference_acceleration_kind,)
        ),
        provenance_controller_static_source=np.asarray(
            (provenance.controller_static_source,)
        ),
        provenance_controller_source_revision=np.asarray(
            (provenance.controller_source_revision,)
        ),
        provenance_topic_names=np.asarray(provenance.topic_names),
        provenance_topic_types=np.asarray(provenance.topic_types),
    )
    return destination


__all__ = [
    "ControllerGainEvents",
    "ControllerGainSnapshot",
    "DEFAULT_GRAPE_BAG",
    "EpisodeProvenance",
    "FlightStateSeries",
    "HOVER_FLIGHT_STATE",
    "PidReferenceSeries",
    "RealFlightEpisode",
    "RosbagArrayData",
    "TimedVectorSeries",
    "build_real_flight_episode",
    "linear_resample",
    "load_grape_rosbag_episode",
    "quaternion_slerp_resample",
    "read_grape_rosbag_arrays",
    "robust_covariance",
    "robust_pose_covariances",
    "save_real_flight_episode",
    "select_continuous_flight_window",
]
