"""ROS bag loading and short-segment selection for Grape replay."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Tuple

import numpy as np


TOPIC_KEYS = (
    "thrust_command",
    "gimbal_command",
    "gimbal_state",
    "imu",
    "cog_odometry",
    "body_odometry",
    "flight_state",
)
GIMBAL_JOINT_NAMES = tuple(
    "gimbal{}".format(index) for index in range(1, 5)
)


def _validate_times(name: str, times: np.ndarray, minimum: int = 1) -> None:
    values = np.asarray(times, dtype=float)
    if (
        values.ndim != 1
        or values.size < minimum
        or not np.all(np.isfinite(values))
        or (values.size > 1 and np.any(np.diff(values) <= 0.0))
    ):
        raise ValueError(
            "{} timestamps must be finite and increasing".format(name)
        )


def _validate_matrix(
    name: str, values: np.ndarray, times: np.ndarray, width: int
) -> None:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (len(times), width) or not np.all(np.isfinite(matrix)):
        raise ValueError(
            "{} must have shape ({}, {}) and contain finite values".format(
                name, len(times), width
            )
        )


def _normalise_quaternions(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=float).copy()
    norms = np.linalg.norm(result, axis=1)
    if np.any(~np.isfinite(result)) or np.any(norms <= np.finfo(float).eps):
        raise ValueError("odometry contains an invalid quaternion")
    result /= norms[:, np.newaxis]
    for index in range(1, result.shape[0]):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


@dataclass(frozen=True)
class AnalysisData:
    """Streams aligned to odometry timestamps inside one analysis interval."""

    bag_path: str
    start_time: float
    end_time: float
    segment_duration: float
    times: np.ndarray
    position: np.ndarray
    orientation_xyzw: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    specific_force: np.ndarray
    base_thrust: np.ndarray
    gimbal_target_angle: np.ndarray
    gimbal_measured_angle: np.ndarray
    flight_state: np.ndarray
    segment_id: np.ndarray

    def __post_init__(self) -> None:
        _validate_times("analysis", self.times, minimum=3)
        for name, values, width in (
            ("position", self.position, 3),
            ("orientation_xyzw", self.orientation_xyzw, 4),
            ("linear_velocity", self.linear_velocity, 3),
            ("angular_velocity", self.angular_velocity, 3),
            ("specific_force", self.specific_force, 3),
            ("base_thrust", self.base_thrust, 4),
            ("gimbal_target_angle", self.gimbal_target_angle, 4),
            ("gimbal_measured_angle", self.gimbal_measured_angle, 4),
        ):
            _validate_matrix(name, values, self.times, width)
        states = np.asarray(self.flight_state)
        segments = np.asarray(self.segment_id)
        if states.shape != self.times.shape:
            raise ValueError("flight_state must match analysis timestamps")
        if (
            segments.shape != self.times.shape
            or not np.issubdtype(segments.dtype, np.integer)
            or np.any(segments < 0)
            or np.any(np.diff(segments) < 0)
        ):
            raise ValueError("segment_id must be an ordered non-negative vector")

    @property
    def segment_count(self) -> int:
        return int(self.segment_id[-1]) + 1

    def segments(self) -> Iterator[Tuple[int, slice]]:
        """Yield contiguous slices, including a possibly short final segment."""

        boundaries = np.flatnonzero(np.diff(self.segment_id)) + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [self.times.size]))
        for start, stop in zip(starts, stops):
            yield int(self.segment_id[start]), slice(int(start), int(stop))


@dataclass(frozen=True)
class BagRecording:
    """Native streams from one bag, expressed in bag-local seconds."""

    bag_path: str
    bag_start_time: float
    bag_duration: float
    command_times: np.ndarray
    base_thrust: np.ndarray
    gimbal_target_times: np.ndarray
    gimbal_target_angle: np.ndarray
    gimbal_measured_times: np.ndarray
    gimbal_measured_angle: np.ndarray
    imu_times: np.ndarray
    specific_force: np.ndarray
    angular_velocity: np.ndarray
    state_times: np.ndarray
    position: np.ndarray
    linear_velocity: np.ndarray
    body_times: np.ndarray
    body_orientation_xyzw: np.ndarray
    body_angular_velocity: np.ndarray
    flight_state_times: np.ndarray
    flight_state: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.bag_duration) or self.bag_duration <= 0.0:
            raise ValueError("bag_duration must be positive")
        for name, times, minimum in (
            ("command", self.command_times, 3),
            ("gimbal target", self.gimbal_target_times, 3),
            ("gimbal measured", self.gimbal_measured_times, 3),
            ("imu", self.imu_times, 3),
            ("CoG state", self.state_times, 3),
            ("body state", self.body_times, 3),
            ("flight_state", self.flight_state_times, 1),
        ):
            _validate_times(name, times, minimum=minimum)
        for name, values, times, width in (
            ("base_thrust", self.base_thrust, self.command_times, 4),
            (
                "gimbal_target_angle",
                self.gimbal_target_angle,
                self.gimbal_target_times,
                4,
            ),
            (
                "gimbal_measured_angle",
                self.gimbal_measured_angle,
                self.gimbal_measured_times,
                4,
            ),
            ("specific_force", self.specific_force, self.imu_times, 3),
            ("angular_velocity", self.angular_velocity, self.imu_times, 3),
            ("position", self.position, self.state_times, 3),
            ("linear_velocity", self.linear_velocity, self.state_times, 3),
            (
                "body_orientation_xyzw",
                self.body_orientation_xyzw,
                self.body_times,
                4,
            ),
            (
                "body_angular_velocity",
                self.body_angular_velocity,
                self.body_times,
                3,
            ),
        ):
            _validate_matrix(name, values, times, width)
        if np.asarray(self.flight_state).shape != self.flight_state_times.shape:
            raise ValueError("flight_state values and timestamps must match")

    @property
    def analysis_bounds(self) -> Tuple[float, float]:
        """Return the common interval of every required physical stream."""

        start = max(
            self.command_times[0],
            self.gimbal_target_times[0],
            self.gimbal_measured_times[0],
            self.imu_times[0],
            self.state_times[0],
            self.body_times[0],
        )
        end = min(
            self.command_times[-1],
            self.gimbal_target_times[-1],
            self.gimbal_measured_times[-1],
            self.imu_times[-1],
            self.state_times[-1],
            self.body_times[-1],
        )
        if end <= start:
            raise ValueError("required bag streams do not overlap")
        return float(start), float(end)

    def select_interval(
        self,
        start_time: float,
        end_time: float,
        segment_duration: float,
    ) -> AnalysisData:
        """Align physical inputs/state to CoG timestamps and make segments.

        Translation is the measured CoG state. Orientation and angular
        velocity are the measured physical body/baselink state. Together they
        define a rigid-body frame centred at the CoG and parallel to baselink,
        which is also the frame used by the wrench geometry in ``model.py``.
        """

        start = float(start_time)
        end = float(end_time)
        duration = float(segment_duration)
        available_start, available_end = self.analysis_bounds
        if (
            not np.isfinite(start)
            or not np.isfinite(end)
            or not np.isfinite(duration)
            or start < available_start
            or end > available_end
            or end <= start
            or duration <= 0.0
        ):
            raise ValueError(
                "analysis interval must lie in [{:.3f}, {:.3f}] and "
                "segment_duration must be positive".format(
                    available_start, available_end
                )
            )

        selected = (self.state_times >= start) & (self.state_times <= end)
        times = self.state_times[selected]
        if times.size < 3:
            raise ValueError("analysis interval contains fewer than 3 states")

        segment_count = max(
            1, int(np.ceil((end - start) / duration - 1.0e-12))
        )
        segment_id = np.floor((times - start) / duration).astype(int)
        segment_id = np.minimum(segment_id, segment_count - 1)
        unique_ids = np.unique(segment_id)
        remap = {int(value): index for index, value in enumerate(unique_ids)}
        segment_id = np.asarray(
            [remap[int(value)] for value in segment_id], dtype=int
        )

        flight_indices = (
            np.searchsorted(self.flight_state_times, times, side="right") - 1
        )
        flight_indices = np.clip(
            flight_indices, 0, self.flight_state_times.size - 1
        )
        return AnalysisData(
            bag_path=self.bag_path,
            start_time=start,
            end_time=end,
            segment_duration=duration,
            times=times,
            position=self.position[selected],
            orientation_xyzw=_interpolate_quaternions(
                self.body_times, self.body_orientation_xyzw, times
            ),
            linear_velocity=self.linear_velocity[selected],
            angular_velocity=_interpolate_matrix(
                self.body_times, self.body_angular_velocity, times
            ),
            specific_force=_interpolate_matrix(
                self.imu_times, self.specific_force, times
            ),
            base_thrust=_interpolate_matrix(
                self.command_times, self.base_thrust, times
            ),
            gimbal_target_angle=_interpolate_matrix(
                self.gimbal_target_times, self.gimbal_target_angle, times
            ),
            gimbal_measured_angle=_interpolate_matrix(
                self.gimbal_measured_times,
                self.gimbal_measured_angle,
                times,
            ),
            flight_state=np.asarray(self.flight_state)[flight_indices],
            segment_id=segment_id,
        )


def _message_time(message: Any, record_time: Any, bag_start: float) -> float:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None and float(stamp.to_sec()) > 0.0:
        return float(stamp.to_sec()) - bag_start
    return float(record_time.to_sec()) - bag_start


def _ordered_unique(
    timestamps: Any, values: Any, width: int, minimum: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    times = np.asarray(timestamps, dtype=float)
    matrix = np.asarray(values, dtype=float).reshape(-1, width)
    if times.size != matrix.shape[0] or times.size < minimum:
        raise ValueError(
            "bag stream has fewer than {} usable samples".format(minimum)
        )
    order = np.argsort(times, kind="stable")
    times = times[order]
    matrix = matrix[order]
    unique_times, indices = np.unique(times, return_index=True)
    if unique_times.size < minimum:
        raise ValueError("bag stream has too few unique timestamps")
    return unique_times, matrix[indices]


def _interpolate_matrix(
    source_times: np.ndarray,
    source_values: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(query_times, source_times, source_values[:, column])
            for column in range(source_values.shape[1])
        ]
    )


def _interpolate_quaternions(
    source_times: np.ndarray,
    source_values: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Spherically interpolate a sign-continuous quaternion stream."""

    quaternions = _normalise_quaternions(source_values)
    upper = np.searchsorted(source_times, query_times, side="right")
    upper = np.clip(upper, 1, len(source_times) - 1)
    lower = upper - 1
    interval = source_times[upper] - source_times[lower]
    fraction = (query_times - source_times[lower]) / interval
    first = quaternions[lower]
    second = quaternions[upper]
    dot = np.clip(np.sum(first * second, axis=1), -1.0, 1.0)

    result = np.empty_like(first)
    nearly_equal = dot > 0.9995
    result[nearly_equal] = (
        (1.0 - fraction[nearly_equal, np.newaxis])
        * first[nearly_equal]
        + fraction[nearly_equal, np.newaxis] * second[nearly_equal]
    )
    separated = ~nearly_equal
    if np.any(separated):
        angle = np.arccos(dot[separated])
        denominator = np.sin(angle)
        first_weight = np.sin(
            (1.0 - fraction[separated]) * angle
        ) / denominator
        second_weight = np.sin(fraction[separated] * angle) / denominator
        result[separated] = (
            first_weight[:, np.newaxis] * first[separated]
            + second_weight[:, np.newaxis] * second[separated]
        )
    return _normalise_quaternions(result)


def _gimbal_positions(message: Any) -> Tuple[float, ...]:
    """Extract gimbal1..4 in physical joint order from a JointState."""

    positions = tuple(message.position)
    names = tuple(message.name)
    if all(name in names for name in GIMBAL_JOINT_NAMES):
        return tuple(
            positions[names.index(name)] for name in GIMBAL_JOINT_NAMES
        )
    if not names and len(positions) >= 4:
        return positions[:4]
    return ()


def read_bag(
    bag_path: str,
    topics: Mapping[str, str],
    progress_callback=None,
) -> BagRecording:
    """Read the physical state and command streams used by Phase 0--1."""

    if set(topics) != set(TOPIC_KEYS):
        raise ValueError(
            "topics must define exactly: {}".format(", ".join(TOPIC_KEYS))
        )
    path = Path(bag_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))

    try:
        import rosbag
    except ImportError as exc:
        raise RuntimeError("ROS 1 rosbag Python module is required") from exc

    command_times = []
    base_thrust = []
    gimbal_target_times = []
    gimbal_target_angle = []
    gimbal_measured_times = []
    gimbal_measured_angle = []
    imu_times = []
    specific_force = []
    angular_velocity = []
    state_times = []
    position = []
    linear_velocity = []
    body_times = []
    body_orientation = []
    body_angular_velocity = []
    flight_state_times = []
    flight_state = []

    selected_topics = tuple(dict.fromkeys(topics[key] for key in TOPIC_KEYS))
    with rosbag.Bag(str(path), "r") as bag:
        bag_start = float(bag.get_start_time())
        bag_end = float(bag.get_end_time())
        bag_duration = bag_end - bag_start
        next_report = 0.0
        for topic, message, record_time in bag.read_messages(
            topics=selected_topics
        ):
            record_offset = float(record_time.to_sec()) - bag_start
            fraction = float(
                np.clip(record_offset / max(bag_duration, 1.0e-12), 0.0, 1.0)
            )
            if progress_callback is not None and fraction >= next_report:
                progress_callback(fraction, "reading ROS bag")
                next_report = fraction + 0.01

            timestamp = _message_time(message, record_time, bag_start)
            if topic == topics["thrust_command"]:
                values = tuple(message.base_thrust)
                if len(values) >= 4:
                    command_times.append(timestamp)
                    base_thrust.append(values[:4])
            elif topic == topics["gimbal_command"]:
                values = _gimbal_positions(message)
                if values:
                    gimbal_target_times.append(timestamp)
                    gimbal_target_angle.append(values)
            elif topic == topics["gimbal_state"]:
                values = _gimbal_positions(message)
                if values:
                    gimbal_measured_times.append(timestamp)
                    gimbal_measured_angle.append(values)
            elif topic == topics["imu"]:
                imu_times.append(timestamp)
                specific_force.append(
                    (
                        message.linear_acceleration.x,
                        message.linear_acceleration.y,
                        message.linear_acceleration.z,
                    )
                )
                angular_velocity.append(
                    (
                        message.angular_velocity.x,
                        message.angular_velocity.y,
                        message.angular_velocity.z,
                    )
                )
            elif topic == topics["cog_odometry"]:
                state_times.append(timestamp)
                position.append(
                    (
                        message.pose.pose.position.x,
                        message.pose.pose.position.y,
                        message.pose.pose.position.z,
                    )
                )
                linear_velocity.append(
                    (
                        message.twist.twist.linear.x,
                        message.twist.twist.linear.y,
                        message.twist.twist.linear.z,
                    )
                )
            elif topic == topics["body_odometry"]:
                body_times.append(timestamp)
                body_orientation.append(
                    (
                        message.pose.pose.orientation.x,
                        message.pose.pose.orientation.y,
                        message.pose.pose.orientation.z,
                        message.pose.pose.orientation.w,
                    )
                )
                body_angular_velocity.append(
                    (
                        message.twist.twist.angular.x,
                        message.twist.twist.angular.y,
                        message.twist.twist.angular.z,
                    )
                )
            elif topic == topics["flight_state"]:
                flight_state_times.append(timestamp)
                flight_state.append((int(message.data),))
        if progress_callback is not None:
            progress_callback(1.0, "ROS bag loaded")

    command_time, thrust = _ordered_unique(
        command_times, base_thrust, width=4
    )
    gimbal_target_time, target_angle = _ordered_unique(
        gimbal_target_times, gimbal_target_angle, width=4
    )
    gimbal_measured_time, measured_angle = _ordered_unique(
        gimbal_measured_times, gimbal_measured_angle, width=4
    )
    imu_time, force = _ordered_unique(
        imu_times, specific_force, width=3
    )
    gyro_time, gyro = _ordered_unique(
        imu_times, angular_velocity, width=3
    )
    state_time, pose = _ordered_unique(state_times, position, width=3)
    velocity_time, velocity = _ordered_unique(
        state_times, linear_velocity, width=3
    )
    body_time, quaternion = _ordered_unique(
        body_times, body_orientation, width=4
    )
    body_gyro_time, body_gyro = _ordered_unique(
        body_times, body_angular_velocity, width=3
    )
    flight_time, flight_value = _ordered_unique(
        flight_state_times, flight_state, width=1, minimum=1
    )
    if not np.array_equal(imu_time, gyro_time):
        raise RuntimeError("IMU streams have diverging timestamps")
    if not np.array_equal(state_time, velocity_time):
        raise RuntimeError("CoG odometry fields have diverging timestamps")
    if not np.array_equal(body_time, body_gyro_time):
        raise RuntimeError("body odometry fields have diverging timestamps")

    return BagRecording(
        bag_path=str(path),
        bag_start_time=bag_start,
        bag_duration=bag_duration,
        command_times=command_time,
        base_thrust=thrust,
        gimbal_target_times=gimbal_target_time,
        gimbal_target_angle=target_angle,
        gimbal_measured_times=gimbal_measured_time,
        gimbal_measured_angle=measured_angle,
        imu_times=imu_time,
        specific_force=force,
        angular_velocity=gyro,
        state_times=state_time,
        position=pose,
        linear_velocity=velocity,
        body_times=body_time,
        body_orientation_xyzw=_normalise_quaternions(quaternion),
        body_angular_velocity=body_gyro,
        flight_state_times=flight_time,
        flight_state=flight_value[:, 0].astype(int),
    )


def scan_bag_paths(directory: str) -> Tuple[str, ...]:
    """Return recursively discovered ``.bag`` files in deterministic order."""

    root = Path(directory).expanduser()
    if root.is_file():
        return (str(root.resolve()),) if root.suffix == ".bag" else ()
    if not root.is_dir():
        return ()
    return tuple(
        str(path.resolve())
        for path in sorted(root.rglob("*.bag"))
        if path.is_file()
    )


def suggest_analysis_interval(
    recording: BagRecording,
    duration: float = 5.28,
) -> Tuple[float, float]:
    """Suggest a low-motion airborne interval for a newly selected bag.

    This is only a UI starting point. It never filters samples from an interval
    chosen by the user.
    """

    available_start, available_end = recording.analysis_bounds
    requested = float(
        np.clip(float(duration), 0.25, available_end - available_start)
    )
    times = recording.state_times
    in_bounds = (times >= available_start) & (times <= available_end)
    flight_indices = (
        np.searchsorted(recording.flight_state_times, times, side="right") - 1
    )
    flight_indices = np.clip(
        flight_indices, 0, recording.flight_state_times.size - 1
    )
    states = recording.flight_state[flight_indices]
    altitude = recording.position[:, 2]
    lower_altitude = float(np.percentile(altitude[in_bounds], 10.0))
    upper_altitude = float(np.percentile(altitude[in_bounds], 90.0))
    airborne_threshold = lower_altitude + max(
        0.10, 0.15 * (upper_altitude - lower_altitude)
    )
    airborne = altitude > airborne_threshold
    body_rate = _interpolate_matrix(
        recording.body_times,
        recording.body_angular_velocity,
        times,
    )
    motion_score = np.linalg.norm(
        recording.linear_velocity, axis=1
    ) + 0.20 * np.linalg.norm(body_rate, axis=1)

    masks = (
        in_bounds & airborne & (states == 5),
        in_bounds & airborne & np.isin(states, (3, 4, 5)),
        in_bounds & airborne,
        in_bounds,
    )
    for mask in masks:
        best = None
        invalid_prefix = np.concatenate(
            ([0], np.cumsum((~mask).astype(int)))
        )
        score_prefix = np.concatenate(([0.0], np.cumsum(motion_score)))
        for start_index in np.flatnonzero(mask):
            target_end = times[start_index] + requested
            end_index = int(np.searchsorted(times, target_end, side="left"))
            if end_index >= times.size:
                continue
            if invalid_prefix[end_index + 1] != invalid_prefix[start_index]:
                continue
            count = end_index - start_index + 1
            score = (
                score_prefix[end_index + 1] - score_prefix[start_index]
            ) / count
            candidate = (
                float(score),
                float(times[start_index]),
                float(times[end_index]),
            )
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            return best[1], best[2]

        transitions = np.diff(
            np.concatenate(([False], mask, [False])).astype(int)
        )
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1)
        shortened = []
        minimum_short_window = min(requested, 1.0)
        for start_index, stop_index in zip(starts, stops):
            end_index = int(stop_index - 1)
            run_duration = float(times[end_index] - times[start_index])
            if run_duration + 1.0e-9 < minimum_short_window:
                continue
            mean_score = float(
                np.mean(motion_score[start_index : end_index + 1])
            )
            shortened.append(
                (
                    -run_duration,
                    mean_score,
                    float(times[start_index]),
                    float(times[end_index]),
                )
            )
        if shortened:
            shortened.sort()
            return shortened[0][2], shortened[0][3]

    return available_start, available_end


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    source = Path(path).expanduser()
    with source.open("r", encoding="utf-8") as stream:
        content = yaml.safe_load(stream)
    if not isinstance(content, dict):
        raise ValueError("configuration YAML must contain a mapping")
    return content


def save_yaml(path: str, content: Mapping[str, Any]) -> str:
    import yaml

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            dict(content),
            stream,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
    return str(destination.resolve())


__all__ = [
    "AnalysisData",
    "BagRecording",
    "TOPIC_KEYS",
    "load_yaml",
    "read_bag",
    "save_yaml",
    "scan_bag_paths",
    "suggest_analysis_interval",
]
