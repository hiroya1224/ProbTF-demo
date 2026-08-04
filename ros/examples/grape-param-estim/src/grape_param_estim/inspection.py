"""Lightweight rosbag inspection bundles for the desktop GUI boundary."""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Tuple
import hashlib
import json
import os
import re
import tempfile

import numpy as np

from grape_param_estim.artifact_io import (
    begin_bundle,
    mark_bundle_cancelled,
    mark_bundle_complete,
    write_json_atomic,
)

from grape_param_estim.progress import (
    CancellationToken,
    ProgressCallback,
    ProgressCancelled,
    ProgressTracker,
    STAGE_PREPARING_TRAJECTORY,
    STAGE_WRITING_ARTIFACTS,
)

from grape_param_estim.real_rosbag import (
    ControllerGainSnapshot,
    FlightEpisodeCandidate,
    RosbagArrayData,
    SmoothingIntervalRecommendation,
    TOPIC_TYPE_CONTRACT,
    _select_controller_snapshot,
    linear_resample,
    list_flight_episode_candidates,
    quaternion_slerp_resample,
    read_grape_rosbag_arrays,
    recommend_smoothing_interval,
)


INSPECTION_REQUEST_SCHEMA = "grape-param-estim/inspection-request/v1"
INSPECTION_BUNDLE_SCHEMA = "grape-param-estim/inspection-bundle/v1"
FLIGHT_INSPECTION_SCHEMA = "grape-param-estim/flight-inspection/v1"
INSPECTION_PREVIEW_SCHEMA = "grape-param-estim/inspection-preview/v1"
CONFIGURATION_FINGERPRINT_SCHEMA = (
    "grape-param-estim/configuration-fingerprint/v1"
)
INSPECTION_WRITER_ID = "grape_param_estim.inspection"

CONFIGURATION_FINGERPRINT_FIELDS = (
    "payload",
    "rotor_propeller",
    "geometry",
    "robot_model_revision",
    "actuator_wiring",
    "hardware_revision",
)

_BAG_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _finite_positive(value, name):
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("{} must be finite and positive".format(name))
    return result


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool):
        raise ValueError("{} must be an integer".format(name))
    result = int(value)
    if result != value or result < minimum:
        raise ValueError(
            "{} must be an integer greater than or equal to {}".format(
                name, minimum
            )
        )
    return result


def _normalise_configuration_provenance(
    value: Optional[Mapping[str, str]],
) -> Tuple[Tuple[str, str], ...]:
    supplied = {} if value is None else dict(value)
    unknown = set(supplied) - set(CONFIGURATION_FINGERPRINT_FIELDS)
    if unknown:
        raise ValueError(
            "unknown configuration provenance fields: {}".format(
                ", ".join(sorted(unknown))
            )
        )
    result = []
    for field_name in CONFIGURATION_FINGERPRINT_FIELDS:
        if field_name not in supplied:
            continue
        field_value = str(supplied[field_name]).strip()
        if field_value:
            result.append((field_name, field_value))
    return tuple(result)


@dataclass(frozen=True)
class InspectionWorkloadSettings:
    """Explicit sparse-batch work counts used only for preview estimates."""

    knot_period_seconds: float = 0.05
    maximum_solver_iterations: int = 30
    maximum_em_iterations: int = 5
    lag_profile_evaluations: int = 7
    mcmc_proposals: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "knot_period_seconds",
            _finite_positive(
                self.knot_period_seconds, "knot_period_seconds"
            ),
        )
        object.__setattr__(
            self,
            "maximum_solver_iterations",
            _positive_integer(
                self.maximum_solver_iterations,
                "maximum_solver_iterations",
            ),
        )
        object.__setattr__(
            self,
            "maximum_em_iterations",
            _positive_integer(
                self.maximum_em_iterations, "maximum_em_iterations"
            ),
        )
        object.__setattr__(
            self,
            "lag_profile_evaluations",
            _positive_integer(
                self.lag_profile_evaluations,
                "lag_profile_evaluations",
            ),
        )
        object.__setattr__(
            self,
            "mcmc_proposals",
            _positive_integer(self.mcmc_proposals, "mcmc_proposals", 0),
        )


@dataclass(frozen=True)
class InspectionBagRequest:
    bag_id: str
    path: str
    episode_index: int = 0
    configuration_provenance: Tuple[Tuple[str, str], ...] = tuple()

    def __post_init__(self) -> None:
        bag_id = str(self.bag_id)
        path = str(self.path)
        index = int(self.episode_index)
        if not _BAG_ID_PATTERN.match(bag_id):
            raise ValueError("bag_id is not a safe artifact identifier")
        if not path:
            raise ValueError("bag path cannot be empty")
        if index < 0:
            raise ValueError("episode_index cannot be negative")
        provenance = _normalise_configuration_provenance(
            dict(self.configuration_provenance)
        )
        object.__setattr__(self, "bag_id", bag_id)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "episode_index", index)
        object.__setattr__(self, "configuration_provenance", provenance)


@dataclass(frozen=True)
class InspectionRequest:
    request_id: str
    bags: Tuple[InspectionBagRequest, ...]
    preview_max_samples: int = 1200
    workload_settings: InspectionWorkloadSettings = (
        InspectionWorkloadSettings()
    )

    def __post_init__(self) -> None:
        request_id = str(self.request_id)
        bags = tuple(self.bags)
        if not request_id:
            raise ValueError("request_id cannot be empty")
        if not bags or any(not isinstance(v, InspectionBagRequest) for v in bags):
            raise ValueError("inspection request must contain bags")
        if len({value.bag_id for value in bags}) != len(bags):
            raise ValueError("inspection bag IDs must be unique")
        preview_count = _positive_integer(
            self.preview_max_samples, "preview_max_samples", 2
        )
        if not isinstance(self.workload_settings, InspectionWorkloadSettings):
            raise TypeError(
                "workload_settings must be InspectionWorkloadSettings"
            )
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "bags", bags)
        object.__setattr__(self, "preview_max_samples", preview_count)


@dataclass(frozen=True)
class ConfigurationFingerprint:
    value: str
    complete: bool
    components: Tuple[Tuple[str, str], ...]
    missing_components: Tuple[str, ...]

    def __post_init__(self) -> None:
        value = str(self.value)
        components = tuple((str(k), str(v)) for k, v in self.components)
        missing = tuple(str(v) for v in self.missing_components)
        if not value:
            raise ValueError("configuration fingerprint cannot be empty")
        if bool(self.complete) != (len(missing) == 0):
            raise ValueError("fingerprint completeness and missing fields disagree")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "complete", bool(self.complete))
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "missing_components", missing)


@dataclass(frozen=True)
class InspectionPreview:
    record_times: np.ndarray
    local_times: np.ndarray
    position: np.ndarray
    orientation_xyzw: np.ndarray
    reference_position: np.ndarray
    reference_rpy: np.ndarray
    flight_state: np.ndarray

    def __post_init__(self) -> None:
        record_times = np.asarray(self.record_times, dtype=float)
        local_times = np.asarray(self.local_times, dtype=float)
        count = record_times.size
        if (
            record_times.ndim != 1
            or count < 2
            or local_times.shape != (count,)
            or np.any(~np.isfinite(record_times))
            or np.any(~np.isfinite(local_times))
            or np.any(np.diff(record_times) <= 0.0)
        ):
            raise ValueError("inspection preview times are invalid")
        arrays = {
            "position": (self.position, (count, 3), float),
            "orientation_xyzw": (
                self.orientation_xyzw, (count, 4), float
            ),
            "reference_position": (
                self.reference_position, (count, 3), float
            ),
            "reference_rpy": (self.reference_rpy, (count, 3), float),
            "flight_state": (self.flight_state, (count,), np.int64),
        }
        object.__setattr__(self, "record_times", record_times.copy())
        object.__setattr__(self, "local_times", local_times.copy())
        for name, (raw, shape, dtype) in arrays.items():
            value = np.asarray(raw, dtype=dtype)
            if value.shape != shape or np.any(~np.isfinite(value)):
                raise ValueError("inspection preview {} is invalid".format(name))
            object.__setattr__(self, name, value.copy())


@dataclass(frozen=True)
class FlightInspection:
    bag_id: str
    source_path: str
    bag_sha256: str
    bag_size_bytes: int
    bag_mtime_ns: int
    record_start: float
    record_end: float
    topic_contract: Tuple[Mapping[str, object], ...]
    topic_contract_valid: bool
    episodes: Tuple[FlightEpisodeCandidate, ...]
    recommendation: Optional[SmoothingIntervalRecommendation]
    controller_snapshot: Optional[ControllerGainSnapshot]
    configuration_fingerprint: ConfigurationFingerprint
    estimated_work_units: Mapping[str, object]
    warnings: Tuple[str, ...]
    status: str
    preview: InspectionPreview

    def __post_init__(self) -> None:
        if self.status not in {
            "ready",
            "needs_configuration_confirmation",
            "blocked",
        }:
            raise ValueError("unknown inspection status")
        if len(str(self.bag_sha256)) != 64:
            raise ValueError("bag SHA256 must contain 64 hexadecimal characters")
        try:
            int(str(self.bag_sha256), 16)
        except ValueError as error:
            raise ValueError("bag SHA256 is not hexadecimal") from error
        object.__setattr__(self, "episodes", tuple(self.episodes))
        object.__setattr__(self, "warnings", tuple(str(v) for v in self.warnings))


def _strict_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def _check_keys(value, required, optional, name):
    if not isinstance(value, dict):
        raise ValueError("{} must be a JSON object".format(name))
    missing = set(required) - set(value)
    unknown = set(value) - set(required) - set(optional)
    if missing:
        raise ValueError(
            "{} is missing: {}".format(name, ", ".join(sorted(missing)))
        )
    if unknown:
        raise ValueError(
            "{} has unknown fields: {}".format(
                name, ", ".join(sorted(unknown))
            )
        )


def load_inspection_request(path: str) -> InspectionRequest:
    """Load and strictly validate one inspection request JSON file."""

    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read inspection request: {}".format(error)) from error
    _check_keys(
        payload,
        required=("schema", "bags"),
        optional=(
            "request_id",
            "preview_max_samples",
            "workload_settings",
        ),
        name="inspection request",
    )
    if payload["schema"] != INSPECTION_REQUEST_SCHEMA:
        raise ValueError(
            "unsupported inspection request schema: {}".format(
                payload["schema"]
            )
        )
    raw_bags = payload["bags"]
    if not isinstance(raw_bags, list) or not raw_bags:
        raise ValueError("inspection request bags must be a non-empty list")
    bags = []
    for bag_index, raw in enumerate(raw_bags):
        name = "inspection bag {}".format(bag_index)
        _check_keys(
            raw,
            required=("bag_id", "path"),
            optional=("episode_index", "configuration_provenance"),
            name=name,
        )
        provenance = raw.get("configuration_provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("{} provenance must be an object".format(name))
        bags.append(
            InspectionBagRequest(
                bag_id=raw["bag_id"],
                path=raw["path"],
                episode_index=raw.get("episode_index", 0),
                configuration_provenance=tuple(provenance.items()),
            )
        )
    raw_settings = payload.get("workload_settings", {})
    _check_keys(
        raw_settings,
        required=tuple(),
        optional=(
            "knot_period_seconds",
            "maximum_solver_iterations",
            "maximum_em_iterations",
            "lag_profile_evaluations",
            "mcmc_proposals",
        ),
        name="workload_settings",
    )
    settings = InspectionWorkloadSettings(**raw_settings)
    return InspectionRequest(
        request_id=payload.get("request_id", source.stem),
        bags=tuple(bags),
        preview_max_samples=payload.get("preview_max_samples", 1200),
        workload_settings=settings,
    )


def _configuration_fingerprint(
    provenance: Tuple[Tuple[str, str], ...],
    topic_contract: Tuple[Mapping[str, object], ...],
) -> ConfigurationFingerprint:
    supplied = dict(provenance)
    components = tuple(
        (name, supplied[name])
        for name in CONFIGURATION_FINGERPRINT_FIELDS
        if name in supplied
    )
    missing = tuple(
        name for name in CONFIGURATION_FINGERPRINT_FIELDS
        if name not in supplied
    )
    context = tuple(
        (str(value["topic"]), str(value.get("actual_type") or "missing"))
        for value in topic_contract
    )
    canonical = json.dumps(
        {"components": components, "topic_context": context},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    prefix = "complete" if not missing else "incomplete"
    return ConfigurationFingerprint(
        value="{}:{}".format(prefix, digest),
        complete=not missing,
        components=components,
        missing_components=missing,
    )


def _topic_contract(arrays: RosbagArrayData):
    actual = dict(zip(arrays.topic_names, arrays.topic_types))
    entries = []
    for topic, expected_type in TOPIC_TYPE_CONTRACT:
        actual_type = actual.get(topic)
        entries.append(
            {
                "topic": topic,
                "expected_type": expected_type,
                "actual_type": actual_type,
                "present": actual_type is not None,
                "type_matches": actual_type == expected_type,
            }
        )
    result = tuple(entries)
    return result, all(value["type_matches"] for value in result)


def _preview(arrays: RosbagArrayData, maximum_samples: int) -> InspectionPreview:
    common_start = max(
        arrays.cog_position.record_times[0],
        arrays.baselink_orientation.record_times[0],
        arrays.pid.record_times[0],
        arrays.flight_state.record_times[0],
    )
    common_end = min(
        arrays.cog_position.record_times[-1],
        arrays.baselink_orientation.record_times[-1],
        arrays.pid.record_times[-1],
        arrays.flight_state.record_times[-1],
    )
    available = np.flatnonzero(
        (arrays.cog_position.record_times >= common_start)
        & (arrays.cog_position.record_times <= common_end)
    )
    if available.size < 2:
        raise ValueError("bag has too few common samples for a preview")
    if available.size > maximum_samples:
        selected = np.unique(np.linspace(
            0, available.size - 1, maximum_samples, dtype=np.int64
        ))
        available = available[selected]
    record_times = arrays.cog_position.record_times[available]
    position = arrays.cog_position.values[available]
    orientation = quaternion_slerp_resample(
        arrays.baselink_orientation.record_times,
        arrays.baselink_orientation.values,
        record_times,
    )
    reference_position = linear_resample(
        arrays.pid.record_times,
        arrays.pid.target_position,
        record_times,
    )
    reference_rpy = linear_resample(
        arrays.pid.record_times,
        np.unwrap(arrays.pid.target_rpy, axis=0),
        record_times,
    )
    state_indices = np.searchsorted(
        arrays.flight_state.record_times, record_times, side="right"
    ) - 1
    state_indices = np.clip(
        state_indices, 0, arrays.flight_state.states.size - 1
    )
    return InspectionPreview(
        record_times=record_times,
        local_times=record_times - arrays.bag_record_start,
        position=position,
        orientation_xyzw=orientation,
        reference_position=reference_position,
        reference_rpy=reference_rpy,
        flight_state=arrays.flight_state.states[state_indices],
    )


def _estimated_work_units(
    recommendation: Optional[SmoothingIntervalRecommendation],
    settings: InspectionWorkloadSettings,
):
    if recommendation is None:
        sample_count = 0
        knot_count = 0
    else:
        duration = recommendation.interval.duration
        sample_count = (
            int(np.floor(duration / settings.knot_period_seconds)) + 1
        )
        knot_count = max(2, sample_count)
    lag_profile_units = (
        settings.lag_profile_evaluations
        * (settings.maximum_em_iterations + 1)
        if sample_count >= 2
        else 0
    )
    nonlinear_iteration_units = (
        lag_profile_units * settings.maximum_solver_iterations
    )
    return {
        "sample_count": sample_count,
        "knot_count": knot_count,
        "lag_profile_point_units": lag_profile_units,
        "nonlinear_iteration_units": nonlinear_iteration_units,
        "mcmc_proposal_units": settings.mcmc_proposals,
        "estimate_kind": (
            "upper_bound_excluding_lm_retries_and_q_backtracking"
        ),
    }


def _sha256_file(
    path: Path, checkpoint: Optional[Callable[[], None]] = None
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            if checkpoint is not None:
                checkpoint()
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def inspect_flight_arrays(
    bag_request: InspectionBagRequest,
    arrays: RosbagArrayData,
    preview_max_samples: int = 1200,
    workload_settings: Optional[InspectionWorkloadSettings] = None,
    source_path: Optional[Path] = None,
) -> FlightInspection:
    """Build one ROS-free inspection result from already parsed bag arrays."""

    if not isinstance(bag_request, InspectionBagRequest):
        raise TypeError("bag_request must be an InspectionBagRequest")
    if not isinstance(arrays, RosbagArrayData):
        raise TypeError("arrays must be RosbagArrayData")
    maximum_samples = _positive_integer(
        preview_max_samples, "preview_max_samples", 2
    )
    settings = workload_settings or InspectionWorkloadSettings()
    if not isinstance(settings, InspectionWorkloadSettings):
        raise TypeError("workload_settings has the wrong type")

    path = (
        Path(arrays.bag_path).expanduser().resolve()
        if source_path is None
        else Path(source_path).expanduser().resolve()
    )
    if path.is_file():
        stat = path.stat()
        mtime_ns = int(stat.st_mtime_ns)
        size_bytes = int(stat.st_size)
        bag_sha256 = arrays.bag_sha256 or _sha256_file(path)
    else:
        mtime_ns = 0
        size_bytes = int(arrays.bag_size_bytes)
        bag_sha256 = arrays.bag_sha256
    topic_contract, contract_valid = _topic_contract(arrays)
    fingerprint = _configuration_fingerprint(
        bag_request.configuration_provenance, topic_contract
    )
    episodes = list_flight_episode_candidates(
        arrays.flight_state, arrays.bag_record_start
    )
    warning_messages = []
    recommendation = None
    snapshot = None
    if not episodes:
        warning_messages.append(
            "no complete TAKEOFF-to-STOP flight episode found"
        )
    elif bag_request.episode_index >= len(episodes):
        warning_messages.append(
            "requested episode_index is outside the complete flights"
        )
    else:
        recommendation = recommend_smoothing_interval(
            episodes[bag_request.episode_index]
        )
        warning_messages.extend(recommendation.warnings)
        try:
            snapshot = _select_controller_snapshot(
                arrays.controller_gain_events,
                recommendation.interval.start_record_time,
                recommendation.interval.end_record_time,
            )
        except ValueError as error:
            warning_messages.append(str(error))
    if not contract_valid:
        warning_messages.append("required rosbag topic contract is incomplete")
    if not fingerprint.complete:
        warning_messages.append(
            "configuration fingerprint is incomplete: {}".format(
                ", ".join(fingerprint.missing_components)
            )
        )
    blocked = not contract_valid or recommendation is None or snapshot is None
    status = (
        "blocked"
        if blocked
        else (
            "ready"
            if fingerprint.complete
            else "needs_configuration_confirmation"
        )
    )
    return FlightInspection(
        bag_id=bag_request.bag_id,
        source_path=str(path),
        bag_sha256=bag_sha256,
        bag_size_bytes=size_bytes,
        bag_mtime_ns=mtime_ns,
        record_start=arrays.bag_record_start,
        record_end=arrays.bag_record_end,
        topic_contract=topic_contract,
        topic_contract_valid=contract_valid,
        episodes=episodes,
        recommendation=recommendation,
        controller_snapshot=snapshot,
        configuration_fingerprint=fingerprint,
        estimated_work_units=_estimated_work_units(
            recommendation, settings
        ),
        warnings=tuple(warning_messages),
        status=status,
        preview=_preview(arrays, maximum_samples),
    )


def _interval_payload(interval):
    return {
        "state": interval.state,
        "control_active": interval.control_active,
        "start_record_time": interval.start_record_time,
        "end_record_time": interval.end_record_time,
        "start_local_time": interval.start_local_time,
        "end_local_time": interval.end_local_time,
        "duration_seconds": interval.duration,
    }


def _inspection_payload(result: FlightInspection):
    snapshot = result.controller_snapshot
    recommendation = result.recommendation
    return {
        "schema": FLIGHT_INSPECTION_SCHEMA,
        "bag_id": result.bag_id,
        "bag_path": result.source_path,
        "bag_sha256": result.bag_sha256,
        "bag_size": result.bag_size_bytes,
        "bag_mtime": result.bag_mtime_ns / 1.0e9,
        "record_time_start": result.record_start,
        "record_time_end": result.record_end,
        "topic_contract": list(result.topic_contract),
        "topic_contract_valid": result.topic_contract_valid,
        "complete_episodes": [
            {
                "episode_index": episode.episode_index,
                "start_record_time": episode.start_record_time,
                "end_record_time": episode.end_record_time,
                "start_local_time": episode.start_local_time,
                "end_local_time": episode.end_local_time,
                "state_intervals": [
                    _interval_payload(value)
                    for value in episode.state_intervals
                ],
            }
            for episode in result.episodes
        ],
        "state5_intervals": [
            _interval_payload(interval)
            for episode in result.episodes
            for interval in episode.state_intervals
            if interval.state == 5
        ],
        "recommended_interval": (
            None
            if recommendation is None
            else {
                "episode_index": recommendation.episode_index,
                "reason": recommendation.reason,
                "warnings": list(recommendation.warnings),
                "interval": _interval_payload(recommendation.interval),
            }
        ),
        "controller_snapshot": (
            None
            if snapshot is None
            else {
                "groups": list(snapshot.groups),
                "record_times": snapshot.record_times.tolist(),
                "gains": snapshot.gains.tolist(),
                "pid_control_flags": snapshot.pid_control_flags.tolist(),
                "source_kinds": list(snapshot.source_kinds),
            }
        ),
        "controller_flags": {
            "available": False,
            "missing": [
                "xy_control_mode",
                "need_yaw_d_control",
                "allocation_mode",
            ],
        },
        "configuration_fingerprint": {
            "schema": CONFIGURATION_FINGERPRINT_SCHEMA,
            "value": result.configuration_fingerprint.value,
            "complete": result.configuration_fingerprint.complete,
            "components": dict(result.configuration_fingerprint.components),
            "missing_components": list(
                result.configuration_fingerprint.missing_components
            ),
        },
        "estimated_work_units": dict(result.estimated_work_units),
        "warnings": list(result.warnings),
        "status": result.status,
    }


def _write_preview_atomic(path: Path, result: FlightInspection) -> None:
    """Atomically publish one pickle-free preview payload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".{}.".format(path.name),
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as stream:
            temporary_name = stream.name
            np.savez_compressed(
                stream,
                schema=np.asarray((INSPECTION_PREVIEW_SCHEMA,)),
                bag_id=np.asarray((result.bag_id,)),
                bag_sha256=np.asarray((result.bag_sha256,)),
                record_times=result.preview.record_times,
                time=result.preview.local_times,
                position=result.preview.position,
                orientation_xyzw=result.preview.orientation_xyzw,
                reference_position=result.preview.reference_position,
                reference_rpy=result.preview.reference_rpy,
                flight_state=result.preview.flight_state,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(path))
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def inspect_flights(
    request: InspectionRequest,
    output_directory: str,
    arrays_loader: Optional[Callable[[str], RosbagArrayData]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    cancellation_token: Optional[CancellationToken] = None,
) -> Path:
    """Inspect all requested bags and write one reloadable directory bundle."""

    if not isinstance(request, InspectionRequest):
        raise TypeError("request must be an InspectionRequest")
    cancellation = (
        CancellationToken()
        if cancellation_token is None
        else cancellation_token
    )
    if not isinstance(cancellation, CancellationToken):
        raise TypeError("cancellation_token must be a CancellationToken")
    destination = Path(output_directory).expanduser().resolve()
    sources = {}
    for bag_request in request.bags:
        cancellation.raise_if_cancelled()
        source = Path(bag_request.path).expanduser().resolve()
        if not source.is_file():
            raise ValueError("rosbag does not exist: {}".format(source))
        sources[bag_request.bag_id] = source

    artifacts = {}
    manifest = {
        "schema": INSPECTION_BUNDLE_SCHEMA,
        "request_schema": INSPECTION_REQUEST_SCHEMA,
        "request_id": request.request_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "writer_id": INSPECTION_WRITER_ID,
        "writer_version": 1,
        "bag_ids": [value.bag_id for value in request.bags],
        "artifacts": {"bags": artifacts},
    }
    for value in request.bags:
        artifacts[value.bag_id] = {
            "inspection": "bags/{}.inspection.json".format(value.bag_id),
            "preview": "bags/{}.preview.npz".format(value.bag_id),
        }
    begin_bundle(destination, manifest)
    bags_directory = destination / "bags"
    bags_directory.mkdir(parents=True, exist_ok=True)
    total_units = 2 + 5 * len(request.bags)
    tracker = ProgressTracker(
        run_id=request.request_id,
        overall_total_units=total_units,
        callback=progress_callback,
        cancellation_token=cancellation,
        eta_calibration_units=min(4, max(2, total_units // 3)),
    )
    active_stage = None
    active_work_id = None

    def emit(
        work_id: str,
        work_label: str,
        bag_id: Optional[str] = None,
        message: str = "",
        advance: bool = False,
    ) -> None:
        nonlocal active_stage, active_work_id
        if active_stage is None:
            stage_id = (
                STAGE_WRITING_ARTIFACTS
                if work_id in {"artifact_writing", "artifact_validation"}
                else STAGE_PREPARING_TRAJECTORY
            )
            active_stage = tracker.begin_stage(stage_id, 1)
            active_work_id = work_id
        elif active_work_id != work_id:
            raise RuntimeError(
                "inspection progress work changed before completion"
            )
        detail = work_label
        if message:
            detail = "{} ({})".format(detail, message)
        if advance:
            active_stage.complete(bag_id=bag_id, message=detail)
            active_stage = None
            active_work_id = None
        else:
            active_stage.emit(0, bag_id=bag_id, message=detail)

    try:
        emit(
            "request_validation",
            "Inspection request validated",
            advance=True,
        )
        for bag_index, bag_request in enumerate(request.bags, start=1):
            bag_id = bag_request.bag_id
            source = sources[bag_id]
            bag_message = "bag {}/{}".format(
                bag_index, len(request.bags)
            )
            initial_stat = source.stat()
            emit(
                "bag_reading",
                "Reading rosbag",
                bag_id,
                bag_message,
            )
            arrays = (
                read_grape_rosbag_arrays(
                    str(source),
                    compute_sha256=False,
                    checkpoint=tracker.checkpoint,
                )
                if arrays_loader is None
                else arrays_loader(str(source))
            )
            tracker.checkpoint()
            emit(
                "bag_reading",
                "Rosbag read",
                bag_id,
                bag_message,
                advance=True,
            )
            emit(
                "sha256",
                "Computing rosbag SHA256",
                bag_id,
                bag_message,
            )
            bag_sha256 = _sha256_file(source, tracker.checkpoint)
            final_stat = source.stat()
            if (
                initial_stat.st_size != final_stat.st_size
                or initial_stat.st_mtime_ns != final_stat.st_mtime_ns
            ):
                raise ValueError(
                    "rosbag changed during inspection: {}".format(source)
                )
            arrays = replace(
                arrays,
                bag_path=str(source),
                bag_sha256=bag_sha256,
                bag_size_bytes=int(final_stat.st_size),
            )
            emit(
                "sha256",
                "Rosbag SHA256 complete",
                bag_id,
                bag_message,
                advance=True,
            )
            emit(
                "topic_contract_validation",
                "Validating topic contract",
                bag_id,
                bag_message,
            )
            result = inspect_flight_arrays(
                bag_request,
                arrays,
                preview_max_samples=request.preview_max_samples,
                workload_settings=request.workload_settings,
                source_path=source,
            )
            final_result_stat = source.stat()
            if (
                final_stat.st_size != final_result_stat.st_size
                or final_stat.st_mtime_ns != final_result_stat.st_mtime_ns
            ):
                raise ValueError(
                    "rosbag changed during inspection: {}".format(source)
                )
            emit(
                "topic_contract_validation",
                "Topic contract validated",
                bag_id,
                bag_message,
                advance=True,
            )
            emit(
                "interval_building",
                "Flight intervals built",
                bag_id,
                bag_message,
                advance=True,
            )
            emit(
                "artifact_writing",
                "Writing inspection artifacts",
                bag_id,
                bag_message,
            )
            inspection_name = "{}.inspection.json".format(result.bag_id)
            preview_name = "{}.preview.npz".format(result.bag_id)
            inspection_path = bags_directory / inspection_name
            preview_path = bags_directory / preview_name
            write_json_atomic(inspection_path, _inspection_payload(result))
            _write_preview_atomic(preview_path, result)
            tracker.checkpoint()
            emit(
                "artifact_writing",
                "Inspection artifacts written",
                bag_id,
                bag_message,
                advance=True,
            )
        emit(
            "artifact_validation",
            "Validating inspection bundle",
        )
        mark_bundle_complete(destination)
        emit(
            "artifact_validation",
            "Inspection bundle complete",
            advance=True,
        )
        return destination
    except ProgressCancelled as error:
        mark_bundle_cancelled(destination, error.reason)
        raise


__all__ = [
    "CONFIGURATION_FINGERPRINT_FIELDS",
    "CONFIGURATION_FINGERPRINT_SCHEMA",
    "FLIGHT_INSPECTION_SCHEMA",
    "FlightInspection",
    "INSPECTION_BUNDLE_SCHEMA",
    "INSPECTION_PREVIEW_SCHEMA",
    "INSPECTION_REQUEST_SCHEMA",
    "InspectionBagRequest",
    "InspectionWorkloadSettings",
    "InspectionPreview",
    "InspectionRequest",
    "inspect_flight_arrays",
    "inspect_flights",
    "load_inspection_request",
]
