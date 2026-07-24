"""Reproducible rosbag inventory and episode metadata helpers."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .episode import (
    EVENT_TIME_RECORD,
    clock_diagnostics,
    message_event_time_nanoseconds,
    sha256_file,
    stable_hash,
)


MANIFEST_SCHEMA = "grape_bag_manifest/v2"
EVENT_TIME_RULE = "header.stamp > top-level stamp > bag record time"


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
    event_time_rule: str
    event_range_scope: str
    first_record_time_ns: Optional[int]
    last_record_time_ns: Optional[int]
    first_event_time_ns: Optional[int]
    last_event_time_ns: Optional[int]
    first_record_time: Optional[float]
    last_record_time: Optional[float]
    first_event_time: Optional[float]
    last_event_time: Optional[float]
    event_time_source_counts: Mapping[str, int]
    header_time_count: int
    record_time_count: int
    header_record_offset_median_s: Optional[float]
    header_record_offset_drift_s_per_s: Optional[float]
    header_record_offset_drift_ppm: Optional[float]
    maximum_header_record_offset_step_s: Optional[float]
    header_record_offset_jump_count: int
    event_timestamp_backward_jump_count: int
    clock_segment_count: int
    clock_drift_sample_count: int
    clock_drift_method: str
    clock_diagnostic_input_order: str
    clock_warnings: Tuple[str, ...]

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
        container_start = float(bag.get_start_time())
        container_end = float(bag.get_end_time())
        type_info = bag.get_type_and_topic_info()
        bag_first_record_ns: Optional[int] = None
        bag_last_record_ns: Optional[int] = None
        buffers: Dict[str, Dict[str, Any]] = {}
        for topic, item in type_info.topics.items():
            buffers[topic] = {
                "type": item.msg_type,
                "count": int(item.message_count),
                "connections": int(item.connections),
                "first_record_ns": None,
                "last_record_ns": None,
                "first_event_ns": None,
                "last_event_ns": None,
                "header_count": 0,
                "record_count": 0,
                "event_source_counts": {},
                "event_times": [],
                "record_times": [],
                "event_sources": [],
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
            record_ns = int(record_stamp.to_nsec())
            record_time = 1.0e-9 * record_ns
            event_ns, event_source = message_event_time_nanoseconds(message)
            if event_ns is None:
                event_ns = record_ns
                event_source = EVENT_TIME_RECORD
            event_time = 1.0e-9 * event_ns
            bag_first_record_ns = (
                record_ns
                if bag_first_record_ns is None
                else min(bag_first_record_ns, record_ns)
            )
            bag_last_record_ns = (
                record_ns
                if bag_last_record_ns is None
                else max(bag_last_record_ns, record_ns)
            )
            item = buffers[topic]
            if item["first_record_ns"] is None:
                item["first_record_ns"] = record_ns
                item["last_record_ns"] = record_ns
                item["first_event_ns"] = event_ns
                item["last_event_ns"] = event_ns
            else:
                item["first_record_ns"] = min(item["first_record_ns"], record_ns)
                item["last_record_ns"] = max(item["last_record_ns"], record_ns)
                item["first_event_ns"] = min(item["first_event_ns"], event_ns)
                item["last_event_ns"] = max(item["last_event_ns"], event_ns)
            item["event_source_counts"][event_source] = (
                item["event_source_counts"].get(event_source, 0) + 1
            )
            item["event_times"].append(event_time)
            item["record_times"].append(record_time)
            item["event_sources"].append(event_source)
            if event_source == EVENT_TIME_RECORD:
                item["record_count"] += 1
            else:
                item["header_count"] += 1
            if topic.endswith("/parameter_updates"):
                parameter_history.append(
                    {
                        "topic": topic,
                        "record_time_ns": record_ns,
                        "record_time": record_time,
                        "event_time_ns": event_ns,
                        "event_time": event_time,
                        "event_time_source": event_source,
                        "values": dynamic_reconfigure_values(message),
                    }
                )
            if topic in parameter_topics:
                singleton_payloads.setdefault(topic, []).append(_message_value(message))

    topics = []
    for topic, item in sorted(buffers.items()):
        diagnostic = clock_diagnostics(
            np.asarray(item["event_times"], dtype=float),
            np.asarray(item["record_times"], dtype=float),
            tuple(item["event_sources"]),
            input_order="record",
        )
        first_record_ns = item["first_record_ns"]
        last_record_ns = item["last_record_ns"]
        first_event_ns = item["first_event_ns"]
        last_event_ns = item["last_event_ns"]
        topics.append(
            TopicInventory(
                topic=topic,
                message_type=str(item["type"]),
                message_count=int(item["count"]),
                connection_count=int(item["connections"]),
                event_time_rule=EVENT_TIME_RULE,
                event_range_scope="this_topic_only",
                first_record_time_ns=first_record_ns,
                last_record_time_ns=last_record_ns,
                first_event_time_ns=first_event_ns,
                last_event_time_ns=last_event_ns,
                first_record_time=(
                    None if first_record_ns is None else 1.0e-9 * first_record_ns
                ),
                last_record_time=(
                    None if last_record_ns is None else 1.0e-9 * last_record_ns
                ),
                first_event_time=(
                    None if first_event_ns is None else 1.0e-9 * first_event_ns
                ),
                last_event_time=(
                    None if last_event_ns is None else 1.0e-9 * last_event_ns
                ),
                event_time_source_counts=dict(
                    sorted(item["event_source_counts"].items())
                ),
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
                event_timestamp_backward_jump_count=(
                    diagnostic.timestamp_backward_jump_count
                ),
                clock_segment_count=diagnostic.segment_count,
                clock_drift_sample_count=diagnostic.drift_sample_count,
                clock_drift_method=diagnostic.drift_method,
                clock_diagnostic_input_order=diagnostic.input_order,
                clock_warnings=diagnostic.warnings,
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
        "battery_id_charge_cycles_and_state_of_charge",
        "complete_mixer_allocation_and_motor_order",
        "complete_filter_and_servo_configuration",
        "controller_integrator_and_delay_state",
        "environment_wind_ground_contact_and_tether_state",
        "operator_experiment_log",
    )
    for name in default_unknowns:
        if name not in unknowns and name not in supplied:
            unknowns.append(name)
    digest = source_hash or sha256_file(bag_path)
    payload_hashes = {
        topic: stable_hash(payload)
        for topic, payload in sorted(singleton_payloads.items())
    }
    if bag_first_record_ns is None or bag_last_record_ns is None:
        # Empty bags have no event range; retain ROS's container metadata only
        # as an explicitly approximate compatibility field.
        bag_start = container_start
        bag_end = container_end
        start_ns = None
        end_ns = None
        duration_ns = None
    else:
        start_ns = bag_first_record_ns
        end_ns = bag_last_record_ns
        duration_ns = end_ns - start_ns
        bag_start = 1.0e-9 * start_ns
        bag_end = 1.0e-9 * end_ns
    result = {
        "episode_id": episode_id,
        "absolute_path": str(bag_path),
        "sha256": digest,
        "file_size_bytes": int(bag_path.stat().st_size),
        "recorded_at_utc": _utc_iso(bag_start),
        "start_record_time_ns": start_ns,
        "end_record_time_ns": end_ns,
        "duration_ns": duration_ns,
        "start_record_time": bag_start,
        "end_record_time": bag_end,
        "duration_s": bag_end - bag_start,
        "event_time_rule": EVENT_TIME_RULE,
        "event_range_note": (
            "Bag boundaries use record time only; per-topic event ranges must "
            "not be combined because stale producer stamps may predate the bag."
        ),
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
        result = {
            str(item["filename"]): {
                key: value for key, value in item.items() if key != "filename"
            }
            for item in entries
        }
    else:
        if not isinstance(entries, Mapping):
            raise ValueError("metadata YAML must contain a bag mapping or list")
        result = {str(key): dict(value or {}) for key, value in entries.items()}
    if loaded.get("schema") == "grape_episode_metadata/v2":
        required = {"value", "status", "evidence", "provenance"}
        for filename, item in result.items():
            outcome = item.get("outcome")
            labels = item.get("labels")
            if not isinstance(outcome, Mapping) or set(outcome) != required:
                raise ValueError(
                    "{} outcome must have value/status/evidence/provenance".format(
                        filename
                    )
                )
            if not isinstance(labels, list) or any(
                not isinstance(label, Mapping) or set(label) != required
                for label in labels
            ):
                raise ValueError(
                    "{} labels must each have value/status/evidence/provenance".format(
                        filename
                    )
                )
            for label in [outcome] + labels:
                if label["status"] not in ("bag_confirmed", "provisional"):
                    raise ValueError(
                        "{} has unsupported label status".format(filename)
                    )
    return result


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
