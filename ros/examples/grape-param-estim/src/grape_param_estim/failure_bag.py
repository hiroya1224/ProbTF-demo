"""Read the streams needed for failed-flight identification and estimation."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np

from grape_param_estim.controller_sample import command_to_wrench


PID_AXES = ("x", "y", "z", "roll", "pitch", "yaw")
PID_GROUPS = ("xy", "z", "roll_pitch", "yaw")


@dataclass(frozen=True)
class ControllerRecording:
    """Recorded PID terms and dynamic-reconfigure gain snapshots."""

    times: np.ndarray
    total: np.ndarray
    proportional: np.ndarray
    integral: np.ndarray
    derivative: np.ndarray
    gain_times: Mapping[str, np.ndarray]
    gain_values: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        if (
            times.size < 3
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError(
                "controller PID timestamps must be finite and ordered"
            )
        for name in (
            "total",
            "proportional",
            "integral",
            "derivative",
        ):
            values = np.asarray(getattr(self, name), dtype=float)
            if (
                values.shape != (times.size, len(PID_AXES))
                or not np.all(np.isfinite(values))
            ):
                raise ValueError(
                    "{} PID terms have an invalid shape".format(name)
                )
        if set(self.gain_times) != set(self.gain_values):
            raise ValueError("controller gain groups do not match")
        for group in self.gain_times:
            update_times = np.asarray(
                self.gain_times[group], dtype=float
            )
            values = np.asarray(
                self.gain_values[group], dtype=float
            )
            if (
                values.shape != (update_times.size, 3)
                or update_times.size < 1
                or not np.all(np.isfinite(update_times))
                or not np.all(np.isfinite(values))
                or (
                    update_times.size > 1
                    and np.any(np.diff(update_times) <= 0.0)
                )
            ):
                raise ValueError(
                    "{} controller gains are invalid".format(group)
                )

    def gains_at(self, group: str, timestamp: float) -> Optional[Mapping]:
        """Return the latest recorded P/I/D gains at one bag-local time."""

        if group not in self.gain_times:
            return None
        times = np.asarray(self.gain_times[group], dtype=float)
        index = int(np.searchsorted(times, timestamp, side="right") - 1)
        if index < 0:
            index = 0
        values = np.asarray(self.gain_values[group], dtype=float)[index]
        return {
            "p": float(values[0]),
            "i": float(values[1]),
            "d": float(values[2]),
        }


@dataclass(frozen=True)
class FailureBagData:
    """Numerical data extracted from one failed-flight interval."""

    bag_path: str
    bag_sha256: str
    bag_start_time: float
    start_offset_s: float
    end_offset_s: float
    command_times: np.ndarray
    command_wrench: np.ndarray
    imu_times: np.ndarray
    specific_force: np.ndarray
    angular_velocity: np.ndarray
    state_times: np.ndarray
    linear_velocity: np.ndarray

    def __post_init__(self) -> None:
        matrix_shapes = (
            ("command_wrench", self.command_wrench, self.command_times, 6),
            ("specific_force", self.specific_force, self.imu_times, 3),
            (
                "angular_velocity",
                self.angular_velocity,
                self.imu_times,
                3,
            ),
            (
                "linear_velocity",
                self.linear_velocity,
                self.state_times,
                3,
            ),
        )
        for name, values, timestamps, width in matrix_shapes:
            array = np.asarray(values, dtype=float)
            times = np.asarray(timestamps, dtype=float)
            if array.shape != (times.size, width):
                raise ValueError("{} has an invalid shape".format(name))
            if (
                times.size < 3
                or not np.all(np.isfinite(array))
                or not np.all(np.isfinite(times))
                or np.any(np.diff(times) <= 0.0)
            ):
                raise ValueError("{} must be finite and time ordered".format(name))


@dataclass(frozen=True)
class FailureBagRecording:
    """Complete numerical recording used for automatic episode detection."""

    bag_path: str
    bag_sha256: str
    bag_start_time: float
    bag_duration_s: float
    command_times: np.ndarray
    command_wrench: np.ndarray
    imu_times: np.ndarray
    specific_force: np.ndarray
    angular_velocity: np.ndarray
    state_times: np.ndarray
    position: np.ndarray
    linear_velocity: np.ndarray
    flight_state_times: np.ndarray
    flight_state: np.ndarray
    orientation_xyzw: Optional[np.ndarray] = None
    controller: Optional[ControllerRecording] = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.bag_duration_s) or self.bag_duration_s <= 0.0:
            raise ValueError("bag duration must be finite and positive")
        matrix_shapes = (
            ("specific_force", self.specific_force, self.imu_times, 3),
            (
                "angular_velocity",
                self.angular_velocity,
                self.imu_times,
                3,
            ),
            ("position", self.position, self.state_times, 3),
            (
                "linear_velocity",
                self.linear_velocity,
                self.state_times,
                3,
            ),
        )
        for name, values, timestamps, width in matrix_shapes:
            array = np.asarray(values, dtype=float)
            times = np.asarray(timestamps, dtype=float)
            if array.shape != (times.size, width):
                raise ValueError("{} has an invalid shape".format(name))
            if (
                times.size < 3
                or not np.all(np.isfinite(array))
                or not np.all(np.isfinite(times))
                or np.any(np.diff(times) <= 0.0)
            ):
                raise ValueError(
                    "{} must be finite and time ordered".format(name)
                )
        command = np.asarray(self.command_wrench, dtype=float)
        command_times = np.asarray(self.command_times, dtype=float)
        if command.shape != (command_times.size, 6):
            raise ValueError("command_wrench has an invalid shape")
        if (
            not np.all(np.isfinite(command))
            or not np.all(np.isfinite(command_times))
            or (
                command_times.size > 1
                and np.any(np.diff(command_times) <= 0.0)
            )
        ):
            raise ValueError(
                "command_wrench must be finite and time ordered"
            )
        flight_times = np.asarray(self.flight_state_times, dtype=float)
        flight_values = np.asarray(self.flight_state)
        if (
            flight_values.shape != flight_times.shape
            or flight_times.size < 3
            or not np.all(np.isfinite(flight_times))
            or np.any(np.diff(flight_times) <= 0.0)
        ):
            raise ValueError("flight_state must be finite and time ordered")
        if self.orientation_xyzw is not None:
            orientation = np.asarray(
                self.orientation_xyzw, dtype=float
            )
            if (
                orientation.shape != (self.state_times.size, 4)
                or not np.all(np.isfinite(orientation))
                or np.any(
                    np.linalg.norm(orientation, axis=1)
                    <= np.finfo(float).eps
                )
            ):
                raise ValueError("orientation_xyzw is invalid")

    def estimator_data(
        self, start_offset_s: float, end_offset_s: float
    ) -> FailureBagData:
        """Return an estimator view without copying the recording arrays."""

        start = float(start_offset_s)
        end = float(end_offset_s)
        if not 0.0 <= start < end <= self.bag_duration_s:
            raise ValueError("estimator interval is outside the bag")
        return FailureBagData(
            bag_path=self.bag_path,
            bag_sha256=self.bag_sha256,
            bag_start_time=self.bag_start_time,
            start_offset_s=start,
            end_offset_s=end,
            command_times=self.command_times,
            command_wrench=self.command_wrench,
            imu_times=self.imu_times,
            specific_force=self.specific_force,
            angular_velocity=self.angular_velocity,
            state_times=self.state_times,
            linear_velocity=self.linear_velocity,
        )


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
    progress_callback=None,
) -> str:
    digest = sha256()
    total_size = max(1, path.stat().st_size)
    completed = 0
    next_report = 0.0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            completed += len(chunk)
            fraction = min(1.0, completed / total_size)
            if (
                progress_callback is not None
                and (fraction >= next_report or fraction >= 1.0)
            ):
                progress_callback(fraction, "hashing ROS bag")
                next_report = fraction + 0.01
    if progress_callback is not None:
        progress_callback(1.0, "ROS bag hash complete")
    return digest.hexdigest()


def _message_time(message, record_time, bag_start: float) -> float:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None and stamp.to_sec() > 0.0:
        return float(stamp.to_sec() - bag_start)
    return float(record_time.to_sec() - bag_start)


def _ordered_unique(
    timestamps,
    values,
    width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    times = np.asarray(timestamps, dtype=float)
    matrix = np.asarray(values, dtype=float).reshape(-1, width)
    if times.size != matrix.shape[0] or times.size < 3:
        raise ValueError("bag stream has too few usable samples")
    order = np.argsort(times, kind="stable")
    times = times[order]
    matrix = matrix[order]
    unique, indices = np.unique(times, return_index=True)
    return unique, matrix[indices]


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


def _collect_streams(
    path: Path,
    topics: Mapping[str, str],
    low: float,
    high: float,
    allow_missing_command: bool = False,
    controller_topics: Optional[Mapping[str, str]] = None,
    progress_callback=None,
):
    try:
        import rosbag
    except ImportError as exc:
        raise RuntimeError("ROS 1 rosbag Python module is required") from exc

    command_times = []
    thrust_values = []
    gimbal_times = []
    gimbal_values = []
    imu_times = []
    specific_force = []
    angular_velocity = []
    state_times = []
    position = []
    linear_velocity = []
    orientation = []
    flight_state_times = []
    flight_state = []
    pid_times = []
    pid_rows = []
    gain_times = {group: [] for group in PID_GROUPS}
    gain_values = {group: [] for group in PID_GROUPS}

    controller_topics = dict(controller_topics or {})
    selected_topics = tuple(
        dict.fromkeys(
            tuple(topics.values())
            + tuple(controller_topics.values())
        )
    )
    gain_topic_groups = {
        topic: group
        for group, topic in controller_topics.items()
        if group in PID_GROUPS
    }
    with rosbag.Bag(str(path), "r") as bag:
        bag_start = float(bag.get_start_time())
        bag_duration = float(bag.get_end_time()) - bag_start
        next_progress = 0.0
        for topic, message, record_time in bag.read_messages(
            topics=selected_topics
        ):
            progress = float(
                np.clip(
                    (
                        float(record_time.to_sec()) - bag_start
                    )
                    / max(bag_duration, np.finfo(float).eps),
                    0.0,
                    1.0,
                )
            )
            if (
                progress_callback is not None
                and progress >= next_progress
            ):
                progress_callback(progress, "reading ROS bag")
                next_progress = progress + 0.005
            timestamp = _message_time(message, record_time, bag_start)
            if timestamp < low or timestamp > high:
                continue
            if topic == topics["command"]:
                thrust = tuple(message.base_thrust)
                if len(thrust) == 4:
                    command_times.append(timestamp)
                    thrust_values.append(thrust)
            elif topic == topics["gimbal"]:
                values = tuple(message.position)
                if len(values) >= 4:
                    gimbal_times.append(timestamp)
                    gimbal_values.append(values[:4])
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
            elif topic == topics["odometry"]:
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
                orientation.append(
                    (
                        message.pose.pose.orientation.x,
                        message.pose.pose.orientation.y,
                        message.pose.pose.orientation.z,
                        message.pose.pose.orientation.w,
                    )
                )
            elif (
                "flight_state" in topics
                and topic == topics["flight_state"]
            ):
                flight_state_times.append(timestamp)
                flight_state.append((int(message.data),))
            elif (
                controller_topics.get("pid_debug") is not None
                and topic == controller_topics["pid_debug"]
            ):
                row = []
                usable = True
                for term in (
                    "total",
                    "p_term",
                    "i_term",
                    "d_term",
                ):
                    for axis in PID_AXES:
                        values = tuple(
                            getattr(getattr(message, axis), term)
                        )
                        if not values:
                            usable = False
                            break
                        row.append(float(values[0]))
                    if not usable:
                        break
                if usable and np.all(np.isfinite(row)):
                    pid_times.append(timestamp)
                    pid_rows.append(row)
            elif topic in gain_topic_groups:
                values = {
                    parameter.name: float(parameter.value)
                    for parameter in message.doubles
                }
                if all(
                    name in values
                    for name in ("p_gain", "i_gain", "d_gain")
                ):
                    group = gain_topic_groups[topic]
                    gain_times[group].append(timestamp)
                    gain_values[group].append(
                        (
                            values["p_gain"],
                            values["i_gain"],
                            values["d_gain"],
                        )
                    )
        if progress_callback is not None:
            progress_callback(1.0, "ROS bag messages loaded")

    if allow_missing_command and (
        len(command_times) < 3 or len(gimbal_times) < 3
    ):
        command_time = np.empty(0, dtype=float)
        wrench = np.empty((0, 6), dtype=float)
    else:
        command_time, thrust = _ordered_unique(
            command_times, thrust_values, 4
        )
        gimbal_time, gimbal = _ordered_unique(
            gimbal_times, gimbal_values, 4
        )
        usable = (
            (command_time >= gimbal_time[0])
            & (command_time <= gimbal_time[-1])
        )
        command_time = command_time[usable]
        thrust = thrust[usable]
        if command_time.size < 3:
            if not allow_missing_command:
                raise ValueError(
                    "command and gimbal streams do not overlap"
                )
            command_time = np.empty(0, dtype=float)
            wrench = np.empty((0, 6), dtype=float)
        else:
            command_gimbal = _interpolate_matrix(
                gimbal_time, gimbal, command_time
            )
            wrench = np.vstack(
                [
                    command_to_wrench(thrust_row, angle_row)
                    for thrust_row, angle_row in zip(
                        thrust, command_gimbal
                    )
                ]
            )
    imu_time, force = _ordered_unique(
        imu_times, specific_force, 3
    )
    gyro_time, gyro = _ordered_unique(
        imu_times, angular_velocity, 3
    )
    if not np.array_equal(imu_time, gyro_time):
        raise RuntimeError("IMU acceleration and gyro timestamps diverged")
    state_time, pose = _ordered_unique(
        state_times, position, 3
    )
    velocity_time, velocity = _ordered_unique(
        state_times, linear_velocity, 3
    )
    orientation_time, quaternion = _ordered_unique(
        state_times, orientation, 4
    )
    if not np.array_equal(state_time, velocity_time):
        raise RuntimeError("odometry pose and velocity timestamps diverged")
    if not np.array_equal(state_time, orientation_time):
        raise RuntimeError("odometry pose and orientation timestamps diverged")

    controller = None
    if len(pid_times) >= 3:
        pid_time, terms = _ordered_unique(
            pid_times, pid_rows, 4 * len(PID_AXES)
        )
        ordered_gain_times = {}
        ordered_gain_values = {}
        for group in PID_GROUPS:
            if not gain_times[group]:
                continue
            times = np.asarray(gain_times[group], dtype=float)
            values = np.asarray(gain_values[group], dtype=float)
            order = np.argsort(times, kind="stable")
            times = times[order]
            values = values[order]
            unique, indices = np.unique(times, return_index=True)
            ordered_gain_times[group] = unique
            ordered_gain_values[group] = values[indices]
        controller = ControllerRecording(
            times=pid_time,
            total=terms[:, 0:6],
            proportional=terms[:, 6:12],
            integral=terms[:, 12:18],
            derivative=terms[:, 18:24],
            gain_times=ordered_gain_times,
            gain_values=ordered_gain_values,
        )

    result = {
        "bag_start": bag_start,
        "bag_duration": bag_duration,
        "command_time": command_time,
        "wrench": wrench,
        "imu_time": imu_time,
        "force": force,
        "gyro": gyro,
        "state_time": state_time,
        "position": pose,
        "velocity": velocity,
        "orientation": quaternion,
        "controller": controller,
    }
    if "flight_state" in topics:
        flight_time, flight_value = _ordered_unique(
            flight_state_times, flight_state, 1
        )
        result["flight_time"] = flight_time
        result["flight_state"] = flight_value[:, 0].astype(int)
    return result


def read_failure_bag(
    bag_path,
    topics: Mapping[str, str],
    start_offset_s: float,
    end_offset_s: float,
    margin_s: float,
) -> FailureBagData:
    """Load command, gimbal, IMU and body-velocity streams directly."""

    path = Path(bag_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    required = {"command", "gimbal", "imu", "odometry"}
    if set(topics) != required:
        raise ValueError("topics must define command, gimbal, imu and odometry")
    start = float(start_offset_s)
    end = float(end_offset_s)
    margin = float(margin_s)
    if not (0.0 <= start < end and margin >= 0.0):
        raise ValueError("analysis offsets or margin are invalid")

    streams = _collect_streams(
        path,
        topics,
        start - margin,
        end + margin,
    )

    return FailureBagData(
        bag_path=str(path),
        bag_sha256=sha256_file(path),
        bag_start_time=streams["bag_start"],
        start_offset_s=start,
        end_offset_s=end,
        command_times=streams["command_time"],
        command_wrench=streams["wrench"],
        imu_times=streams["imu_time"],
        specific_force=streams["force"],
        angular_velocity=streams["gyro"],
        state_times=streams["state_time"],
        linear_velocity=streams["velocity"],
    )


def read_failure_recording(
    bag_path,
    topics: Mapping[str, str],
    controller_topics: Optional[Mapping[str, str]] = None,
    progress_callback=None,
) -> FailureBagRecording:
    """Load a complete bag for automatic control-episode detection."""

    path = Path(bag_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    required = {
        "command",
        "gimbal",
        "imu",
        "odometry",
        "flight_state",
    }
    if set(topics) != required:
        raise ValueError(
            "topics must define command, gimbal, imu, odometry and "
            "flight_state"
        )
    streams = _collect_streams(
        path,
        topics,
        -np.inf,
        np.inf,
        allow_missing_command=True,
        controller_topics=controller_topics,
        progress_callback=(
            None
            if progress_callback is None
            else lambda fraction, phase: progress_callback(
                0.9 * fraction, phase
            )
        ),
    )
    return FailureBagRecording(
        bag_path=str(path),
        bag_sha256=sha256_file(
            path,
            progress_callback=(
                None
                if progress_callback is None
                else lambda fraction, phase: progress_callback(
                    0.9 + 0.1 * fraction, phase
                )
            ),
        ),
        bag_start_time=streams["bag_start"],
        bag_duration_s=streams["bag_duration"],
        command_times=streams["command_time"],
        command_wrench=streams["wrench"],
        imu_times=streams["imu_time"],
        specific_force=streams["force"],
        angular_velocity=streams["gyro"],
        state_times=streams["state_time"],
        position=streams["position"],
        linear_velocity=streams["velocity"],
        flight_state_times=streams["flight_time"],
        flight_state=streams["flight_state"],
        orientation_xyzw=streams["orientation"],
        controller=streams["controller"],
    )


__all__ = [
    "ControllerRecording",
    "FailureBagData",
    "FailureBagRecording",
    "PID_AXES",
    "PID_GROUPS",
    "read_failure_bag",
    "read_failure_recording",
    "sha256_file",
]
