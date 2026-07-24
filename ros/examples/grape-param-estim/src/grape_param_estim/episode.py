"""Event-time episode representation for the Grape rosbag analyses.

The classes in this module are deliberately ROS independent.  ROS messages
are converted at the boundary by :func:`read_rosbag_episode`; downstream
estimators only see immutable, timestamped numeric values plus explicit
quality events and provenance hashes.

There are two important time axes in a ROS bag:

* ``record_time`` is when rosbag received the message;
* ``header_time`` (when present) is the producing sensor's clock.

Neither is silently preferred.  Every :class:`TopicSpec` declares the rule
used for that topic and every resulting sample records which clock supplied
its event time.  This makes clock assumptions auditable and prevents an
apparently synchronized data set from hiding a record/header substitution.
"""

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


EVENT_TIME_HEADER = "header"
EVENT_TIME_RECORD = "record"
EVENT_TIME_HEADER_OR_RECORD = "header_or_record"
_EVENT_TIME_POLICIES = {
    EVENT_TIME_HEADER,
    EVENT_TIME_RECORD,
    EVENT_TIME_HEADER_OR_RECORD,
}


def _freeze_array(values: Any, dtype: Any, shape: Optional[Tuple[int, ...]], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError("{} must have shape {}".format(name, shape))
    output = np.array(array, copy=True)
    output.setflags(write=False)
    return output


def _finite_times(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values".format(name))
    return _freeze_array(array, float, None, name)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def stable_hash(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible configuration data."""

    try:
        payload = json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "stable_hash input must be finite and JSON-compatible"
        ) from exc
    return sha256(payload).hexdigest()


def sha256_file(path: Any, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it in memory."""

    source = Path(path).expanduser().resolve()
    digest = sha256()
    with source.open("rb") as stream:
        while True:
            chunk = stream.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _ros_time_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "to_sec"):
        seconds = float(value.to_sec())
    elif hasattr(value, "secs") and hasattr(value, "nsecs"):
        seconds = float(value.secs) + 1.0e-9 * float(value.nsecs)
    else:
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return None
    if not np.isfinite(seconds) or seconds <= 0.0:
        return None
    return seconds


def message_header_time(message: Any) -> Optional[float]:
    """Read a ROS-style header/stamp without importing ROS message packages."""

    header = getattr(message, "header", None)
    if header is not None:
        stamp = _ros_time_seconds(getattr(header, "stamp", None))
        if stamp is not None:
            return stamp
    # Some spinal messages have a top-level stamp rather than std_msgs/Header.
    return _ros_time_seconds(getattr(message, "stamp", None))


@dataclass(frozen=True)
class QualityEvent:
    stamp: float
    kind: str
    topic: str
    detail: str = ""
    severity: str = "warning"

    def __post_init__(self) -> None:
        stamp = float(self.stamp)
        if not np.isfinite(stamp):
            raise ValueError("quality-event stamp must be finite")
        if not self.kind or not self.topic:
            raise ValueError("quality-event kind and topic must not be empty")
        if self.severity not in ("info", "warning", "error"):
            raise ValueError("quality-event severity must be info, warning, or error")
        object.__setattr__(self, "stamp", stamp)


@dataclass(frozen=True)
class EventSample:
    role: str
    topic: str
    event_time: float
    record_time: float
    event_time_source: str
    value: Any
    valid: bool
    age: float


@dataclass(frozen=True)
class ClockDiagnostics:
    """Header-to-record clock alignment diagnostics for one topic."""

    header_sample_count: int
    record_fallback_count: int
    offset_median_s: Optional[float]
    offset_drift_s_per_s: Optional[float]
    offset_drift_ppm: Optional[float]
    maximum_offset_step_s: Optional[float]
    offset_jump_count: int
    jump_threshold_s: float


def clock_diagnostics(
    event_times: np.ndarray,
    record_times: np.ndarray,
    event_time_sources: Sequence[str],
    jump_threshold: float = 0.02,
) -> ClockDiagnostics:
    """Estimate clock offset, linear drift, and abrupt offset changes."""

    events = np.asarray(event_times, dtype=float).reshape(-1)
    records = np.asarray(record_times, dtype=float).reshape(-1)
    sources = tuple(event_time_sources)
    threshold = float(jump_threshold)
    if (
        records.shape != events.shape
        or len(sources) != events.size
        or not np.all(np.isfinite(events))
        or not np.all(np.isfinite(records))
    ):
        raise ValueError("clock diagnostic arrays must be finite and aligned")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("jump_threshold must be finite and positive")
    header_mask = np.asarray(
        [item == EVENT_TIME_HEADER for item in sources], dtype=bool
    )
    fallback_count = int(
        np.count_nonzero(
            np.asarray([item == EVENT_TIME_RECORD for item in sources], dtype=bool)
        )
    )
    count = int(np.count_nonzero(header_mask))
    if not count:
        return ClockDiagnostics(
            header_sample_count=0,
            record_fallback_count=fallback_count,
            offset_median_s=None,
            offset_drift_s_per_s=None,
            offset_drift_ppm=None,
            maximum_offset_step_s=None,
            offset_jump_count=0,
            jump_threshold_s=threshold,
        )
    header_events = events[header_mask]
    offsets = records[header_mask] - header_events
    steps = np.abs(np.diff(offsets))
    if count >= 2 and float(np.ptp(header_events)) > 0.0:
        centered_time = header_events - float(np.mean(header_events))
        centered_offset = offsets - float(np.mean(offsets))
        denominator = float(np.dot(centered_time, centered_time))
        slope = (
            float(np.dot(centered_time, centered_offset) / denominator)
            if denominator > 0.0
            else 0.0
        )
    else:
        slope = None
    return ClockDiagnostics(
        header_sample_count=count,
        record_fallback_count=fallback_count,
        offset_median_s=float(np.median(offsets)),
        offset_drift_s_per_s=slope,
        offset_drift_ppm=None if slope is None else 1.0e6 * slope,
        maximum_offset_step_s=(
            float(np.max(steps)) if steps.size else 0.0
        ),
        offset_jump_count=int(np.count_nonzero(steps > threshold)),
        jump_threshold_s=threshold,
    )


@dataclass(frozen=True)
class EventSeries:
    """One normalized topic ordered by event time."""

    role: str
    topic: str
    message_type: str
    event_times: np.ndarray
    record_times: np.ndarray
    event_time_sources: Tuple[str, ...]
    values: Tuple[Any, ...]
    valid_mask: np.ndarray
    unit: str = ""
    frame_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_hash: str = ""

    def __post_init__(self) -> None:
        if not self.role or not self.topic:
            raise ValueError("event-series role and topic must not be empty")
        event_times = _finite_times(self.event_times, "event_times")
        record_times = _finite_times(self.record_times, "record_times")
        count = event_times.size
        if record_times.shape != (count,):
            raise ValueError("record_times must match event_times")
        if count and np.any(np.diff(event_times) < 0.0):
            raise ValueError("event_times must be non-decreasing")
        sources = tuple(str(item) for item in self.event_time_sources)
        if len(sources) != count or any(
            item not in (EVENT_TIME_HEADER, EVENT_TIME_RECORD) for item in sources
        ):
            raise ValueError("event_time_sources must identify every sample clock")
        values = tuple(self.values)
        if len(values) != count:
            raise ValueError("values must contain one item per sample")
        valid = _freeze_array(self.valid_mask, bool, (count,), "valid_mask")
        metadata = MappingProxyType(dict(self.metadata))
        source_hash = str(self.source_hash) or stable_hash(
            {
                "role": self.role,
                "topic": self.topic,
                "message_type": self.message_type,
                "event_times": event_times,
                "record_times": record_times,
                "event_time_sources": sources,
                "values": values,
                "valid_mask": valid,
                "unit": self.unit,
                "frame_id": self.frame_id,
                "metadata": metadata,
            }
        )
        object.__setattr__(self, "event_times", event_times)
        object.__setattr__(self, "record_times", record_times)
        object.__setattr__(self, "event_time_sources", sources)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "source_hash", source_hash)

    @property
    def count(self) -> int:
        return int(self.event_times.size)

    def clock_diagnostics(self, jump_threshold: float = 0.02) -> ClockDiagnostics:
        return clock_diagnostics(
            self.event_times,
            self.record_times,
            self.event_time_sources,
            jump_threshold,
        )

    def nearest(
        self,
        stamp: float,
        max_age: float = float("inf"),
        causal: bool = False,
        require_valid: bool = True,
    ) -> Optional[EventSample]:
        """Return the nearest (or most recent causal) sample.

        ``age`` is absolute for nearest-neighbour queries and non-negative for
        causal queries.  A sample beyond ``max_age`` is reported as missing.
        """

        requested = float(stamp)
        limit = float(max_age)
        if not np.isfinite(requested) or limit < 0.0:
            raise ValueError("stamp must be finite and max_age non-negative")
        if not self.count:
            return None
        if causal:
            index = int(np.searchsorted(self.event_times, requested, side="right") - 1)
            if index < 0:
                return None
            age = requested - float(self.event_times[index])
        else:
            insertion = int(np.searchsorted(self.event_times, requested))
            candidates = []
            if insertion < self.count:
                candidates.append(insertion)
            if insertion:
                candidates.append(insertion - 1)
            index = min(
                candidates,
                key=lambda candidate: abs(float(self.event_times[candidate]) - requested),
            )
            age = abs(float(self.event_times[index]) - requested)
        if age > limit or (require_valid and not bool(self.valid_mask[index])):
            return None
        return EventSample(
            role=self.role,
            topic=self.topic,
            event_time=float(self.event_times[index]),
            record_time=float(self.record_times[index]),
            event_time_source=self.event_time_sources[index],
            value=self.values[index],
            valid=bool(self.valid_mask[index]),
            age=float(age),
        )


@dataclass(frozen=True)
class ModeSegment:
    start: float
    end: float
    mode: Any

    def __post_init__(self) -> None:
        start = float(self.start)
        end = float(self.end)
        if not np.isfinite(start) or not np.isfinite(end) or end < start:
            raise ValueError("mode segment must have finite start <= end")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_id: str
    source_bag: str
    source_bag_sha256: str
    start_time: float
    end_time: float
    labels: Tuple[str, ...] = ()
    split: str = "UNASSIGNED"
    assumptions: Mapping[str, Any] = field(default_factory=dict)
    unknowns: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        start = float(self.start_time)
        end = float(self.end_time)
        if not self.episode_id or not self.source_bag_sha256:
            raise ValueError("episode_id and source_bag_sha256 must not be empty")
        if not np.isfinite(start) or not np.isfinite(end) or end < start:
            raise ValueError("episode metadata requires finite start <= end")
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "labels", tuple(str(item) for item in self.labels))
        object.__setattr__(self, "unknowns", tuple(str(item) for item in self.unknowns))
        object.__setattr__(
            self, "assumptions", MappingProxyType(dict(self.assumptions))
        )


@dataclass(frozen=True)
class EpisodeDataset:
    """Immutable normalized episode with auditable source/config hashes."""

    metadata: EpisodeMetadata
    series: Mapping[str, EventSeries]
    quality_events: Tuple[QualityEvent, ...]
    config_hash: str
    dataset_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, EpisodeMetadata):
            raise TypeError("metadata must be EpisodeMetadata")
        series = dict(self.series)
        if any(role != item.role for role, item in series.items()):
            raise ValueError("series mapping keys must match EventSeries.role")
        events = tuple(
            sorted(self.quality_events, key=lambda item: (item.stamp, item.topic, item.kind))
        )
        config_hash = str(self.config_hash)
        if not config_hash:
            raise ValueError("config_hash must not be empty")
        digest = str(self.dataset_hash) or stable_hash(
            {
                "metadata": self.metadata,
                "series_hashes": {
                    role: item.source_hash for role, item in sorted(series.items())
                },
                "events": events,
                "config_hash": config_hash,
            }
        )
        object.__setattr__(self, "series", MappingProxyType(series))
        object.__setattr__(self, "quality_events", events)
        object.__setattr__(self, "config_hash", config_hash)
        object.__setattr__(self, "dataset_hash", digest)

    def query(
        self,
        role: str,
        stamp: float,
        max_age: float = float("inf"),
        causal: bool = False,
        require_valid: bool = True,
    ) -> Optional[EventSample]:
        if role not in self.series:
            raise KeyError("episode has no normalized role '{}'".format(role))
        return self.series[role].nearest(stamp, max_age, causal, require_valid)

    def common_timeline(self, roles: Optional[Iterable[str]] = None) -> np.ndarray:
        selected = tuple(roles) if roles is not None else tuple(self.series)
        if not selected:
            return _freeze_array([], float, (0,), "timeline")
        arrays = []
        for role in selected:
            if role not in self.series:
                raise KeyError("episode has no normalized role '{}'".format(role))
            arrays.append(self.series[role].event_times)
        return _freeze_array(np.unique(np.concatenate(arrays)), float, None, "timeline")

    def mode_segments(self, role: str = "flight_state") -> Tuple[ModeSegment, ...]:
        if role not in self.series:
            return ()
        item = self.series[role]
        if not item.count:
            return ()
        segments = []
        start = float(item.event_times[0])
        current = item.values[0]
        for index in range(1, item.count):
            if item.values[index] != current:
                boundary = float(item.event_times[index])
                segments.append(ModeSegment(start, boundary, current))
                start = boundary
                current = item.values[index]
        segments.append(ModeSegment(start, self.metadata.end_time, current))
        return tuple(segments)


@dataclass(frozen=True)
class TopicSpec:
    """Conversion rule for one raw ROS topic."""

    role: str
    topic: str
    extractor: Callable[[Any], Any]
    event_time_policy: str = EVENT_TIME_HEADER_OR_RECORD
    unit: str = ""
    frame_id: str = ""
    required: bool = False
    gap_threshold: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.role or not self.topic or not callable(self.extractor):
            raise ValueError("TopicSpec requires role, topic, and extractor")
        if self.event_time_policy not in _EVENT_TIME_POLICIES:
            raise ValueError("unsupported event-time policy")
        if self.gap_threshold is not None:
            threshold = float(self.gap_threshold)
            if not np.isfinite(threshold) or threshold <= 0.0:
                raise ValueError("gap_threshold must be finite and positive")
            object.__setattr__(self, "gap_threshold", threshold)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _choose_event_time(
    message: Any,
    record_time: float,
    policy: str,
) -> Tuple[float, str]:
    header_time = message_header_time(message)
    if policy == EVENT_TIME_RECORD:
        return record_time, EVENT_TIME_RECORD
    if policy == EVENT_TIME_HEADER:
        if header_time is None:
            raise ValueError("message has no valid header/stamp")
        return header_time, EVENT_TIME_HEADER
    if header_time is not None:
        return header_time, EVENT_TIME_HEADER
    return record_time, EVENT_TIME_RECORD


def _series_quality_events(
    spec: TopicSpec,
    event_times: np.ndarray,
    record_times: np.ndarray,
    sources: Sequence[str],
    clock_jump_threshold: float,
) -> Tuple[QualityEvent, ...]:
    events = []
    if event_times.size > 1:
        gaps = np.diff(event_times)
        if spec.gap_threshold is not None:
            for index in np.flatnonzero(gaps > spec.gap_threshold):
                events.append(
                    QualityEvent(
                        stamp=float(event_times[index + 1]),
                        kind="gap",
                        topic=spec.topic,
                        detail="{:.9g}s gap exceeds {:.9g}s".format(
                            float(gaps[index]), float(spec.gap_threshold)
                        ),
                    )
                )
    header_mask = np.asarray([item == EVENT_TIME_HEADER for item in sources])
    if np.count_nonzero(header_mask) > 1:
        offsets = record_times[header_mask] - event_times[header_mask]
        offset_steps = np.abs(np.diff(offsets))
        header_indices = np.flatnonzero(header_mask)
        for local_index in np.flatnonzero(offset_steps > clock_jump_threshold):
            sample_index = int(header_indices[local_index + 1])
            events.append(
                QualityEvent(
                    stamp=float(event_times[sample_index]),
                    kind="clock_offset_jump",
                    topic=spec.topic,
                    detail="record-header offset changed by {:.9g}s".format(
                        float(offset_steps[local_index])
                    ),
                )
            )
    if any(item == EVENT_TIME_RECORD for item in sources) and spec.event_time_policy == EVENT_TIME_HEADER_OR_RECORD:
        first = next(
            index for index, item in enumerate(sources) if item == EVENT_TIME_RECORD
        )
        events.append(
            QualityEvent(
                stamp=float(event_times[first]),
                kind="record_time_fallback",
                topic=spec.topic,
                detail="header/stamp unavailable; bag record time used",
                severity="info",
            )
        )
    return tuple(events)


def read_rosbag_episode(
    bag_path: Any,
    topic_specs: Sequence[TopicSpec],
    episode_id: Optional[str] = None,
    labels: Sequence[str] = (),
    split: str = "UNASSIGNED",
    assumptions: Optional[Mapping[str, Any]] = None,
    unknowns: Sequence[str] = (),
    config: Optional[Mapping[str, Any]] = None,
    source_bag_sha256: Optional[str] = None,
    clock_jump_threshold: float = 0.02,
) -> EpisodeDataset:
    """Read selected topics from a ROS 1 bag into a normalized episode.

    Raw messages are never modified or written back.  Extractor failures are
    retained as invalid samples and explicit ``decode_error`` events.
    """

    try:
        import rosbag  # Imported lazily so unit tests remain ROS independent.
    except ImportError as exc:  # pragma: no cover - depends on ROS installation.
        raise RuntimeError("read_rosbag_episode requires the ROS 1 rosbag module") from exc

    path = Path(bag_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    specs = tuple(topic_specs)
    roles = [item.role for item in specs]
    topics = [item.topic for item in specs]
    if len(set(roles)) != len(roles) or len(set(topics)) != len(topics):
        raise ValueError("topic specs require unique roles and topics")
    spec_by_topic = {item.topic: item for item in specs}
    buffers: Dict[str, Dict[str, Any]] = {
        item.role: {
            "event_times": [],
            "record_times": [],
            "sources": [],
            "values": [],
            "valid": [],
            "message_type": "",
        }
        for item in specs
    }
    quality_events = []
    with rosbag.Bag(str(path), "r") as bag:
        bag_start = float(bag.get_start_time())
        bag_end = float(bag.get_end_time())
        for topic, message, ros_record_time in bag.read_messages(topics=topics):
            spec = spec_by_topic[topic]
            record_time = float(ros_record_time.to_sec())
            try:
                event_time, source = _choose_event_time(
                    message, record_time, spec.event_time_policy
                )
            except ValueError as exc:
                quality_events.append(
                    QualityEvent(
                        stamp=record_time,
                        kind="missing_event_time",
                        topic=topic,
                        detail=str(exc),
                        severity="error" if spec.required else "warning",
                    )
                )
                continue
            try:
                value = spec.extractor(message)
                valid = True
            except Exception as exc:  # Extractors are user-provided bag boundaries.
                value = None
                valid = False
                quality_events.append(
                    QualityEvent(
                        stamp=event_time,
                        kind="decode_error",
                        topic=topic,
                        detail="{}: {}".format(type(exc).__name__, exc),
                        severity="error" if spec.required else "warning",
                    )
                )
            buffer = buffers[spec.role]
            buffer["event_times"].append(event_time)
            buffer["record_times"].append(record_time)
            buffer["sources"].append(source)
            buffer["values"].append(value)
            buffer["valid"].append(valid)
            buffer["message_type"] = str(getattr(message, "_type", type(message).__name__))

    normalized: Dict[str, EventSeries] = {}
    for spec in specs:
        buffer = buffers[spec.role]
        if not buffer["event_times"]:
            if spec.required:
                quality_events.append(
                    QualityEvent(
                        stamp=bag_start,
                        kind="missing_topic",
                        topic=spec.topic,
                        detail="required topic contains no messages",
                        severity="error",
                    )
                )
            continue
        # Header stamps can arrive out of order relative to record time.  Keep
        # each sample intact while ordering the normalized series by event time.
        order = np.argsort(np.asarray(buffer["event_times"]), kind="stable")
        event_times = np.asarray(buffer["event_times"], dtype=float)[order]
        record_times = np.asarray(buffer["record_times"], dtype=float)[order]
        sources = tuple(buffer["sources"][index] for index in order)
        values = tuple(buffer["values"][index] for index in order)
        valid = np.asarray(buffer["valid"], dtype=bool)[order]
        quality_events.extend(
            _series_quality_events(
                spec,
                event_times,
                record_times,
                sources,
                float(clock_jump_threshold),
            )
        )
        normalized[spec.role] = EventSeries(
            role=spec.role,
            topic=spec.topic,
            message_type=buffer["message_type"],
            event_times=event_times,
            record_times=record_times,
            event_time_sources=sources,
            values=values,
            valid_mask=valid,
            unit=spec.unit,
            frame_id=spec.frame_id,
            metadata=spec.metadata,
        )

    bag_hash = source_bag_sha256 or sha256_file(path)
    configuration = {
        "topic_specs": [
            {
                "role": item.role,
                "topic": item.topic,
                "event_time_policy": item.event_time_policy,
                "unit": item.unit,
                "frame_id": item.frame_id,
                "required": item.required,
                "gap_threshold": item.gap_threshold,
                "metadata": item.metadata,
            }
            for item in specs
        ],
        "normalization": dict(config or {}),
        "clock_jump_threshold": float(clock_jump_threshold),
    }
    metadata = EpisodeMetadata(
        episode_id=episode_id or path.stem,
        source_bag=str(path),
        source_bag_sha256=bag_hash,
        start_time=bag_start,
        end_time=bag_end,
        labels=tuple(labels),
        split=split,
        assumptions=dict(assumptions or {}),
        unknowns=tuple(unknowns),
    )
    return EpisodeDataset(
        metadata=metadata,
        series=normalized,
        quality_events=tuple(quality_events),
        config_hash=stable_hash(configuration),
    )


__all__ = [
    "ClockDiagnostics",
    "EVENT_TIME_HEADER",
    "EVENT_TIME_HEADER_OR_RECORD",
    "EVENT_TIME_RECORD",
    "EpisodeDataset",
    "EpisodeMetadata",
    "EventSample",
    "EventSeries",
    "ModeSegment",
    "QualityEvent",
    "TopicSpec",
    "clock_diagnostics",
    "message_header_time",
    "read_rosbag_episode",
    "sha256_file",
    "stable_hash",
]
