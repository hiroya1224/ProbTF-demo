"""Reproducible rosbag inventory and episode metadata helpers."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .episode import (
    EVENT_TIME_HEADER,
    EVENT_TIME_RECORD,
    clock_diagnostics,
    message_header_time,
    sha256_file,
    stable_hash,
)


MANIFEST_SCHEMA = "grape_bag_manifest/v1"


def _utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _message_value(value: Any) -> Any:
    """Convert a ROS message/value to plain deterministic Python containers."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_sec"):
        return float(value.to_sec())
    slots = getattr(value, "__slots__", None)
    if slots is not None:
        return {name: _message_value(getattr(value, name)) for name in slots}
    if isinstance(value, Mapping):
        return {
            str(key): _message_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_message_value(item) for item in value]
    return repr(value)


def dynamic_reconfigure_values(message: Any) -> Dict[str, Any]:
    """Flatten a ``dynamic_reconfigure/Config`` message."""

    values: Dict[str, Any] = {}
    for field_name in ("bools", "ints", "strs", "doubles"):
        for item in getattr(message, field_name, ()):
            values[str(item.name)] = _message_value(item.value)
    return values


@dataclass(frozen=True)
class TopicInventory:
    topic: str
    message_type: str
    message_count: int
    connection_count: int
    first_record_time: Optional[float]
    last_record_time: Optional[float]
    first_event_time: Optional[float]
    last_event_time: Optional[float]
    header_time_count: int
    record_time_count: int
    header_record_offset_median_s: Optional[float]
    header_record_offset_drift_s_per_s: Optional[float]
    header_record_offset_drift_ppm: Optional[float]
    maximum_header_record_offset_step_s: Optional[float]
    header_record_offset_jump_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def inspect_bag(
    path: Any,
    metadata: Optional[Mapping[str, Any]] = None,
    source_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Inspect one bag, including per-topic event-time coverage.

    The scan also records dynamic-reconfigure history and hashes parameter-like
    singleton topics.  Missing launch/source/physical metadata is represented
    as ``UNKNOWN`` through the caller-provided metadata rather than borrowed
    from the current checkout.
    """

    try:
        import rosbag
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("inspect_bag requires the ROS 1 rosbag module") from exc

    bag_path = Path(path).expanduser().resolve()
    if not bag_path.is_file():
        raise FileNotFoundError(str(bag_path))
    with rosbag.Bag(str(bag_path), "r") as bag:
        start = float(bag.get_start_time())
        end = float(bag.get_end_time())
        type_info = bag.get_type_and_topic_info()
        buffers: Dict[str, Dict[str, Any]] = {}
        for topic, item in type_info.topics.items():
            buffers[topic] = {
                "type": item.msg_type,
                "count": int(item.message_count),
                "connections": int(item.connections),
                "first_record": None,
                "last_record": None,
                "first_event": None,
                "last_event": None,
                "header_count": 0,
                "record_count": 0,
                "header_event_times": [],
                "header_record_times": [],
            }
        parameter_history = []
        singleton_payloads: Dict[str, Any] = {}
        parameter_topics = {
            topic
            for topic, item in type_info.topics.items()
            if item.message_count <= 4
            and (
                topic.endswith("/parameter_updates")
                or topic
                in (
                    "/gimbalrotor/motor_info",
                    "/gimbalrotor/joint_profiles",
                    "/gimbalrotor/uav_info",
                    "/gimbalrotor/rpy/gain",
                )
            )
        }
        for topic, message, record_stamp in bag.read_messages():
            record_time = float(record_stamp.to_sec())
            header_time = message_header_time(message)
            event_time = record_time if header_time is None else float(header_time)
            item = buffers[topic]
            if item["first_record"] is None:
                item["first_record"] = record_time
                item["last_record"] = record_time
                item["first_event"] = event_time
                item["last_event"] = event_time
            else:
                item["first_record"] = min(item["first_record"], record_time)
                item["last_record"] = max(item["last_record"], record_time)
                item["first_event"] = min(item["first_event"], event_time)
                item["last_event"] = max(item["last_event"], event_time)
            if header_time is None:
                item["record_count"] += 1
            else:
                item["header_count"] += 1
                item["header_event_times"].append(event_time)
                item["header_record_times"].append(record_time)
            if topic.endswith("/parameter_updates"):
                parameter_history.append(
                    {
                        "topic": topic,
                        "record_time": record_time,
                        "event_time": event_time,
                        "values": dynamic_reconfigure_values(message),
                    }
                )
            if topic in parameter_topics:
                singleton_payloads.setdefault(topic, []).append(_message_value(message))

    topics = []
    for topic, item in sorted(buffers.items()):
        diagnostic = clock_diagnostics(
            np.asarray(item["header_event_times"], dtype=float),
            np.asarray(item["header_record_times"], dtype=float),
            (EVENT_TIME_HEADER,) * len(item["header_event_times"]),
        )
        topics.append(
            TopicInventory(
                topic=topic,
                message_type=str(item["type"]),
                message_count=int(item["count"]),
                connection_count=int(item["connections"]),
                first_record_time=item["first_record"],
                last_record_time=item["last_record"],
                first_event_time=item["first_event"],
                last_event_time=item["last_event"],
                header_time_count=int(item["header_count"]),
                record_time_count=int(item["record_count"]),
                header_record_offset_median_s=diagnostic.offset_median_s,
                header_record_offset_drift_s_per_s=(
                    diagnostic.offset_drift_s_per_s
                ),
                header_record_offset_drift_ppm=diagnostic.offset_drift_ppm,
                maximum_header_record_offset_step_s=(
                    diagnostic.maximum_offset_step_s
                ),
                header_record_offset_jump_count=diagnostic.offset_jump_count,
            ).to_dict()
        )

    supplied = dict(metadata or {})
    episode_id = str(supplied.pop("episode_id", bag_path.stem))
    assumptions = dict(supplied.pop("assumptions", {}))
    unknowns = list(supplied.pop("unknowns", ()))
    default_unknowns = (
        "recording_source_commit",
        "flashed_firmware_commit",
        "complete_rosparam_snapshot",
        "launch_arguments",
        "urdf_hash",
        "payload_mass",
        "propeller_ids_and_service_age",
        "rpm_thrust_calibration",
    )
    for name in default_unknowns:
        if name not in unknowns and name not in supplied:
            unknowns.append(name)
    digest = source_hash or sha256_file(bag_path)
    payload_hashes = {
        topic: stable_hash(payload)
        for topic, payload in sorted(singleton_payloads.items())
    }
    result = {
        "episode_id": episode_id,
        "absolute_path": str(bag_path),
        "sha256": digest,
        "file_size_bytes": int(bag_path.stat().st_size),
        "recorded_at_utc": _utc_iso(start),
        "start_record_time": start,
        "end_record_time": end,
        "duration_s": end - start,
        "message_count": int(sum(item["message_count"] for item in topics)),
        "topics": topics,
        "dynamic_reconfigure_history": sorted(
            parameter_history, key=lambda item: (item["record_time"], item["topic"])
        ),
        "parameter_topic_payload_hashes": payload_hashes,
        "assumptions": assumptions,
        "unknowns": sorted(set(str(item) for item in unknowns)),
    }
    result.update(supplied)
    return result


def build_manifest(
    bag_paths: Sequence[Any],
    metadata_by_name: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a deterministic manifest for a fixed list of bags."""

    metadata_lookup = dict(metadata_by_name or {})
    bags = []
    for path in sorted(Path(item).expanduser().resolve() for item in bag_paths):
        metadata = metadata_lookup.get(path.name, metadata_lookup.get(path.stem, {}))
        bags.append(inspect_bag(path, metadata=metadata))
    episode_ids = [item["episode_id"] for item in bags]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode_id must be unique across the manifest")
    split_by_hash: Dict[str, str] = {}
    for item in bags:
        split = str(item.get("split", "UNASSIGNED"))
        prior = split_by_hash.setdefault(item["sha256"], split)
        if prior != split:
            raise ValueError("the same bag hash cannot appear in multiple splits")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "bag_count": len(bags),
        "bags": bags,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    return manifest


def load_metadata_yaml(path: Any) -> Dict[str, Mapping[str, Any]]:
    import yaml

    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    entries = loaded.get("bags", loaded)
    if isinstance(entries, list):
        return {
            str(item["filename"]): {
                key: value for key, value in item.items() if key != "filename"
            }
            for item in entries
        }
    if not isinstance(entries, Mapping):
        raise ValueError("metadata YAML must contain a bag mapping or list")
    return {str(key): dict(value or {}) for key, value in entries.items()}


def write_manifest_yaml(manifest: Mapping[str, Any], path: Any) -> None:
    """Write a manifest atomically; the input bags remain untouched."""

    import os
    import tempfile
    import yaml

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(
                dict(manifest),
                stream,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(destination))
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "MANIFEST_SCHEMA",
    "TopicInventory",
    "build_manifest",
    "dynamic_reconfigure_values",
    "inspect_bag",
    "load_metadata_yaml",
    "write_manifest_yaml",
]
