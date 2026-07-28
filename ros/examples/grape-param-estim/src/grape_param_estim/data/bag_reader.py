"""Read-only ROS bag topic inventory used by controller replay audits."""

from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .provenance import sha256_file, stable_hash, validated_sha256


def _readonly(values: Any, width: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (
        array.ndim != 2
        or array.shape[1] != width
        or not np.all(np.isfinite(array))
    ):
        raise ValueError("{} must be a finite (N, {}) matrix".format(name, width))
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy


def _times(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if (
        array.size < 2
        or not np.all(np.isfinite(array))
        or np.any(np.diff(array) <= 0.0)
    ):
        raise ValueError("{} must be finite and strictly increasing".format(name))
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy


def _deduplicate_last(rows):
    if not rows:
        return np.empty(0), np.empty((0, 0))
    rows = sorted(rows, key=lambda item: item[0])
    output = []
    for stamp, value in rows:
        if output and abs(stamp - output[-1][0]) <= 1.0e-9:
            output[-1] = (stamp, value)
        else:
            output.append((stamp, value))
    return (
        np.asarray([item[0] for item in output], dtype=float),
        np.stack([item[1] for item in output]),
    )


@dataclass(frozen=True)
class ForwardEpisodeData:
    """Bag-derived inputs for recorded-command forward identification."""

    episode_id: str
    bag_path: str
    source_bag_sha256: str
    replay_start_offset_s: float
    score_start_offset_s: float
    score_end_offset_s: float
    command_times: np.ndarray
    base_thrust: np.ndarray
    gimbal_angle: np.ndarray
    mocap_times: np.ndarray
    position_world: np.ndarray
    orientation_xyzw: np.ndarray
    imu_times: np.ndarray
    specific_force_body: np.ndarray
    angular_velocity_body: np.ndarray
    topic_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        command_times = _times(self.command_times, "command_times")
        mocap_times = _times(self.mocap_times, "mocap_times")
        imu_times = _times(self.imu_times, "imu_times")
        thrust = _readonly(self.base_thrust, 4, "base_thrust")
        gimbal = _readonly(self.gimbal_angle, 4, "gimbal_angle")
        position = _readonly(self.position_world, 3, "position_world")
        orientation = _readonly(self.orientation_xyzw, 4, "orientation_xyzw")
        specific = _readonly(
            self.specific_force_body, 3, "specific_force_body"
        )
        angular = _readonly(
            self.angular_velocity_body, 3, "angular_velocity_body"
        )
        if (
            thrust.shape[0] != command_times.size
            or gimbal.shape[0] != command_times.size
            or position.shape[0] != mocap_times.size
            or orientation.shape[0] != mocap_times.size
            or specific.shape[0] != imu_times.size
            or angular.shape[0] != imu_times.size
        ):
            raise ValueError("forward episode channels are not time aligned")
        replay = float(self.replay_start_offset_s)
        score_start = float(self.score_start_offset_s)
        score_end = float(self.score_end_offset_s)
        if replay > score_start or score_start >= score_end:
            raise ValueError(
                "forward episode requires replay_start <= score_start < score_end"
            )
        object.__setattr__(self, "command_times", command_times)
        object.__setattr__(self, "base_thrust", thrust)
        object.__setattr__(self, "gimbal_angle", gimbal)
        object.__setattr__(self, "mocap_times", mocap_times)
        object.__setattr__(self, "position_world", position)
        object.__setattr__(self, "orientation_xyzw", orientation)
        object.__setattr__(self, "imu_times", imu_times)
        object.__setattr__(self, "specific_force_body", specific)
        object.__setattr__(self, "angular_velocity_body", angular)
        object.__setattr__(
            self,
            "topic_counts",
            MappingProxyType(
                {str(key): int(value) for key, value in self.topic_counts.items()}
            ),
        )

    @property
    def normalized_episode_sha256(self) -> str:
        return stable_hash(
            {
                "episode_id": self.episode_id,
                "source_bag_sha256": self.source_bag_sha256,
                "interval": [
                    self.replay_start_offset_s,
                    self.score_start_offset_s,
                    self.score_end_offset_s,
                ],
                "command_times": self.command_times,
                "base_thrust": self.base_thrust,
                "gimbal_angle": self.gimbal_angle,
                "mocap_times": self.mocap_times,
                "position_world": self.position_world,
                "orientation_xyzw": self.orientation_xyzw,
                "imu_times": self.imu_times,
                "specific_force_body": self.specific_force_body,
                "angular_velocity_body": self.angular_velocity_body,
            }
        )

    def recorded_commands(self):
        from grape_param_estim.forward.rollout import RecordedCommandSeries

        return RecordedCommandSeries(
            timestamps=self.command_times,
            base_thrust=self.base_thrust,
            gimbal_angle=self.gimbal_angle,
            source_bag_sha256=self.source_bag_sha256,
        )


@dataclass(frozen=True)
class TopicInventoryEntry:
    """Connection-level facts available without deserializing bag payloads."""

    topic: str
    message_type: str
    message_count: int
    connection_count: int = 1

    def __post_init__(self) -> None:
        topic = str(self.topic)
        message_type = str(self.message_type)
        count = int(self.message_count)
        connections = int(self.connection_count)
        if not topic.startswith("/"):
            raise ValueError("topic inventory names must be absolute ROS topics")
        if not message_type or count < 0 or connections < 0:
            raise ValueError("topic inventory type/count fields are invalid")
        object.__setattr__(self, "topic", topic)
        object.__setattr__(self, "message_type", message_type)
        object.__setattr__(self, "message_count", count)
        object.__setattr__(self, "connection_count", connections)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BagTopicInventory:
    """Immutable, hash-bound topic inventory for one source bag."""

    bag_path: str
    source_bag_sha256: str
    start_record_time: float
    end_record_time: float
    topics: Mapping[str, TopicInventoryEntry]

    def __post_init__(self) -> None:
        bag_path = str(Path(self.bag_path).expanduser().resolve())
        digest = validated_sha256(self.source_bag_sha256, "source_bag_sha256")
        start = float(self.start_record_time)
        end = float(self.end_record_time)
        if not np.isfinite(start) or not np.isfinite(end) or end < start:
            raise ValueError("bag record-time interval must be finite and ordered")
        normalized: Dict[str, TopicInventoryEntry] = {}
        for topic, entry in self.topics.items():
            if not isinstance(entry, TopicInventoryEntry):
                if not isinstance(entry, Mapping):
                    raise TypeError("topic entries must be TopicInventoryEntry mappings")
                entry = TopicInventoryEntry(
                    topic=str(topic),
                    message_type=str(entry.get("message_type", entry.get("type", ""))),
                    message_count=int(entry.get("message_count", entry.get("count", 0))),
                    connection_count=int(
                        entry.get("connection_count", entry.get("connections", 1))
                    ),
                )
            if str(topic) != entry.topic:
                raise ValueError("topic inventory key does not match entry.topic")
            normalized[entry.topic] = entry
        object.__setattr__(self, "bag_path", bag_path)
        object.__setattr__(self, "source_bag_sha256", digest)
        object.__setattr__(self, "start_record_time", start)
        object.__setattr__(self, "end_record_time", end)
        object.__setattr__(
            self,
            "topics",
            MappingProxyType(dict(sorted(normalized.items()))),
        )

    @classmethod
    def from_mapping(
        cls,
        *,
        bag_path: Any,
        source_bag_sha256: str,
        topics: Mapping[str, Mapping[str, Any]],
        start_record_time: float = 0.0,
        end_record_time: float = 0.0,
    ) -> "BagTopicInventory":
        return cls(
            bag_path=str(bag_path),
            source_bag_sha256=source_bag_sha256,
            start_record_time=start_record_time,
            end_record_time=end_record_time,
            topics=topics,
        )

    @property
    def inventory_sha256(self) -> str:
        return stable_hash(self.to_dict(include_inventory_hash=False))

    def to_dict(self, include_inventory_hash: bool = True) -> Dict[str, Any]:
        result = {
            "bag_path": self.bag_path,
            "source_bag_sha256": self.source_bag_sha256,
            "start_record_time": self.start_record_time,
            "end_record_time": self.end_record_time,
            "topics": {
                topic: entry.to_dict() for topic, entry in self.topics.items()
            },
        }
        if include_inventory_hash:
            result["inventory_sha256"] = self.inventory_sha256
        return result


def read_bag_topic_inventory(
    path: Any,
    *,
    source_bag_sha256: Optional[str] = None,
) -> BagTopicInventory:
    """Read topic types/counts and provenance without scanning message payloads."""

    try:
        import rosbag
    except ImportError as exc:  # pragma: no cover - ROS runtime boundary
        raise RuntimeError("read_bag_topic_inventory requires ROS 1 rosbag") from exc

    bag_path = Path(path).expanduser().resolve()
    if not bag_path.is_file():
        raise FileNotFoundError(str(bag_path))
    with rosbag.Bag(str(bag_path), "r") as bag:
        type_info = bag.get_type_and_topic_info()
        topics = {
            str(topic): TopicInventoryEntry(
                topic=str(topic),
                message_type=str(item.msg_type),
                message_count=int(item.message_count),
                connection_count=int(item.connections),
            )
            for topic, item in type_info.topics.items()
        }
        start = float(bag.get_start_time())
        end = float(bag.get_end_time())
    actual_digest = sha256_file(bag_path)
    if source_bag_sha256 is not None:
        expected_digest = validated_sha256(
            source_bag_sha256, "source_bag_sha256"
        )
        if actual_digest != expected_digest:
            raise ValueError(
                "source bag SHA-256 mismatch for {}".format(bag_path)
            )
    digest = actual_digest
    return BagTopicInventory(
        bag_path=str(bag_path),
        source_bag_sha256=digest,
        start_record_time=start,
        end_record_time=end,
        topics=topics,
    )


def _event_seconds(message: Any, record_time: Any) -> float:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        value = float(stamp.to_sec())
        if np.isfinite(value) and value > 0.0:
            return value
    return float(record_time.to_sec())


def read_forward_episode(
    path: Any,
    *,
    episode_id: str,
    replay_start_offset_s: float,
    score_start_offset_s: float,
    score_end_offset_s: float,
    topics: Optional[Mapping[str, str]] = None,
    source_bag_sha256: Optional[str] = None,
) -> ForwardEpisodeData:
    """Read command/gimbal, mocap, and IMU channels without a ROS master."""

    try:
        import genpy
        import rosbag
    except ImportError as exc:  # pragma: no cover - ROS runtime boundary
        raise RuntimeError("read_forward_episode requires ROS 1 rosbag/genpy") from exc

    selected_topics = {
        "command": "/gimbalrotor/four_axes/command",
        "gimbal": "/gimbalrotor/gimbals_ctrl",
        "mocap": "/gimbalrotor/mocap/pose",
        "imu": "/gimbalrotor/sensor_plugin/imu1/ros_converted",
    }
    if topics is not None:
        selected_topics.update(
            {str(key): str(value) for key, value in topics.items()}
        )
    if set(selected_topics) != {"command", "gimbal", "mocap", "imu"}:
        raise ValueError("forward episode topics must define four declared roles")
    bag_path = Path(path).expanduser().resolve()
    if not bag_path.is_file():
        raise FileNotFoundError(str(bag_path))
    actual_digest = sha256_file(bag_path)
    if source_bag_sha256 is not None:
        expected_digest = validated_sha256(
            source_bag_sha256, "source_bag_sha256"
        )
        if actual_digest != expected_digest:
            raise ValueError(
                "source bag SHA-256 mismatch for {}".format(bag_path)
            )
    digest = actual_digest
    command_rows = []
    gimbal_rows = []
    mocap_rows = []
    imu_rows = []
    counts = {value: 0 for value in selected_topics.values()}
    with rosbag.Bag(str(bag_path), "r") as bag:
        bag_start = float(bag.get_start_time())
        start = bag_start + float(replay_start_offset_s)
        end = bag_start + float(score_end_offset_s)
        read_start = genpy.Time.from_sec(max(bag_start, start - 1.0))
        read_end = genpy.Time.from_sec(end + 0.05)
        for topic, message, record_time in bag.read_messages(
            topics=tuple(selected_topics.values()),
            start_time=read_start,
            end_time=read_end,
        ):
            counts[topic] += 1
            stamp = _event_seconds(message, record_time) - bag_start
            # Keep one causal gimbal target immediately before replay_start.
            # JointState is published on its own event stream, so discarding
            # this held value would make the first recorded FourAxisCommand
            # unusable even though the bag contains its actuator state.
            lower_bound = (
                float(replay_start_offset_s) - 1.0
                if topic == selected_topics["gimbal"]
                else float(replay_start_offset_s)
            )
            if stamp < lower_bound - 1.0e-9 or stamp > float(
                score_end_offset_s
            ) + 1.0e-9:
                continue
            if topic == selected_topics["command"]:
                thrust = np.asarray(message.base_thrust, dtype=float)
                if thrust.shape == (4,) and np.all(np.isfinite(thrust)):
                    command_rows.append((stamp, thrust))
            elif topic == selected_topics["gimbal"]:
                names = tuple(str(item) for item in message.name)
                position = np.asarray(message.position, dtype=float)
                if position.size >= 4 and np.all(np.isfinite(position)):
                    if all(
                        "gimbal{}".format(index) in names
                        for index in range(1, 5)
                    ):
                        position = np.asarray(
                            [
                                position[names.index("gimbal{}".format(index))]
                                for index in range(1, 5)
                            ]
                        )
                    else:
                        position = position[:4]
                    gimbal_rows.append((stamp, position))
            elif topic == selected_topics["mocap"]:
                pose = (
                    message.pose.pose
                    if hasattr(message.pose, "pose")
                    else message.pose
                )
                mocap_rows.append(
                    (
                        stamp,
                        np.asarray(
                            [
                                pose.position.x,
                                pose.position.y,
                                pose.position.z,
                                pose.orientation.x,
                                pose.orientation.y,
                                pose.orientation.z,
                                pose.orientation.w,
                            ],
                            dtype=float,
                        ),
                    )
                )
            elif topic == selected_topics["imu"]:
                imu_rows.append(
                    (
                        stamp,
                        np.asarray(
                            [
                                message.linear_acceleration.x,
                                message.linear_acceleration.y,
                                message.linear_acceleration.z,
                                message.angular_velocity.x,
                                message.angular_velocity.y,
                                message.angular_velocity.z,
                            ],
                            dtype=float,
                        ),
                    )
                )
    command_times, thrust = _deduplicate_last(command_rows)
    gimbal_times, gimbal = _deduplicate_last(gimbal_rows)
    mocap_times, mocap = _deduplicate_last(mocap_rows)
    imu_times, imu = _deduplicate_last(imu_rows)
    if (
        command_times.size < 2
        or gimbal_times.size < 2
        or mocap_times.size < 3
        or imu_times.size < 3
    ):
        raise ValueError("forward episode lacks required command/gimbal/mocap/IMU data")
    # Controller and gimbal targets are causal event streams.  Align the
    # latest gimbal command at or before each FourAxisCommand.
    gimbal_index = np.searchsorted(
        gimbal_times, command_times, side="right"
    ) - 1
    if np.any(gimbal_index < 0):
        raise ValueError("no causal gimbal target exists for the first command")
    aligned_gimbal = gimbal[gimbal_index]
    return ForwardEpisodeData(
        episode_id=str(episode_id),
        bag_path=str(bag_path),
        source_bag_sha256=digest,
        replay_start_offset_s=float(replay_start_offset_s),
        score_start_offset_s=float(score_start_offset_s),
        score_end_offset_s=float(score_end_offset_s),
        command_times=command_times,
        base_thrust=thrust,
        gimbal_angle=aligned_gimbal,
        mocap_times=mocap_times,
        position_world=mocap[:, :3],
        orientation_xyzw=mocap[:, 3:],
        imu_times=imu_times,
        specific_force_body=imu[:, :3],
        angular_velocity_body=imu[:, 3:],
        topic_counts=counts,
    )


__all__ = [
    "BagTopicInventory",
    "ForwardEpisodeData",
    "TopicInventoryEntry",
    "read_forward_episode",
    "read_bag_topic_inventory",
]
