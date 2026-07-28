"""Read only the four streams needed for failed-flight identification."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np

from grape_param_estim.controller_sample import command_to_wrench


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


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
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


def read_failure_bag(
    bag_path,
    topics: Mapping[str, str],
    start_offset_s: float,
    end_offset_s: float,
    margin_s: float,
) -> FailureBagData:
    """Load command, gimbal, IMU and body-velocity streams directly."""

    try:
        import rosbag
    except ImportError as exc:
        raise RuntimeError("ROS 1 rosbag Python module is required") from exc

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

    command_times = []
    thrust_values = []
    gimbal_times = []
    gimbal_values = []
    imu_times = []
    specific_force = []
    angular_velocity = []
    state_times = []
    linear_velocity = []

    selected_topics = tuple(topics.values())
    with rosbag.Bag(str(path), "r") as bag:
        bag_start = float(bag.get_start_time())
        low = start - margin
        high = end + margin
        for topic, message, record_time in bag.read_messages(
            topics=selected_topics
        ):
            timestamp = _message_time(message, record_time, bag_start)
            if timestamp < low or timestamp > high:
                continue
            if topic == topics["command"]:
                thrust = tuple(message.base_thrust)
                if len(thrust) == 4:
                    command_times.append(timestamp)
                    thrust_values.append(thrust)
            elif topic == topics["gimbal"]:
                position = tuple(message.position)
                if len(position) >= 4:
                    gimbal_times.append(timestamp)
                    gimbal_values.append(position[:4])
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
                linear_velocity.append(
                    (
                        message.twist.twist.linear.x,
                        message.twist.twist.linear.y,
                        message.twist.twist.linear.z,
                    )
                )

    command_time, thrust = _ordered_unique(command_times, thrust_values, 4)
    gimbal_time, gimbal = _ordered_unique(gimbal_times, gimbal_values, 4)
    imu_time, force = _ordered_unique(imu_times, specific_force, 3)
    gyro_time, gyro = _ordered_unique(
        imu_times, angular_velocity, 3
    )
    if not np.array_equal(imu_time, gyro_time):
        raise RuntimeError("IMU acceleration and gyro timestamps diverged")
    state_time, velocity = _ordered_unique(
        state_times, linear_velocity, 3
    )

    usable = (
        (command_time >= gimbal_time[0])
        & (command_time <= gimbal_time[-1])
    )
    command_time = command_time[usable]
    thrust = thrust[usable]
    if command_time.size < 3:
        raise ValueError("command and gimbal streams do not overlap")
    command_gimbal = _interpolate_matrix(
        gimbal_time, gimbal, command_time
    )
    wrench = np.vstack(
        [
            command_to_wrench(thrust_row, angle_row)
            for thrust_row, angle_row in zip(thrust, command_gimbal)
        ]
    )

    return FailureBagData(
        bag_path=str(path),
        bag_sha256=sha256_file(path),
        bag_start_time=bag_start,
        start_offset_s=start,
        end_offset_s=end,
        command_times=command_time,
        command_wrench=wrench,
        imu_times=imu_time,
        specific_force=force,
        angular_velocity=gyro,
        state_times=state_time,
        linear_velocity=velocity,
    )


__all__ = [
    "FailureBagData",
    "read_failure_bag",
    "sha256_file",
]
