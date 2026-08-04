"""Strict, pickle-free directory bundle I/O contracts.

The estimator and the GUI intentionally run in different Python
environments.  This module is the small, ROS- and Qt-independent boundary
between them.  A bundle is usable only when its atomic ``manifest.json`` has
``status == "complete"`` and every file named by that manifest validates.
Scratch files may exist beside a writing or cancelled manifest, but loaders
never treat them as a completed result.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


INSPECTION_BUNDLE_SCHEMA = "grape-param-estim/inspection-bundle/v1"
FLIGHT_INSPECTION_SCHEMA = "grape-param-estim/flight-inspection/v1"
MANIFEST_NAME = "manifest.json"
WRITING_STATUS = "writing"
COMPLETE_STATUS = "complete"
CANCELLED_STATUS = "cancelled"
_KNOWN_SCHEMAS = {INSPECTION_BUNDLE_SCHEMA}


class ArtifactValidationError(ValueError):
    """An artifact exists but does not satisfy its declared schema."""


class UnsupportedArtifactSchema(ArtifactValidationError):
    """The manifest schema is not understood by this loader."""


class IncompleteArtifactError(ArtifactValidationError):
    """A writing, cancelled, or otherwise incomplete bundle was loaded."""


class ArtifactStateError(ArtifactValidationError):
    """A requested manifest state transition is not allowed."""


@dataclass(frozen=True)
class InspectionBundle:
    root: Path
    manifest: Mapping[str, Any]
    inspections: Mapping[str, Mapping[str, Any]]
    previews: Mapping[str, Mapping[str, np.ndarray]]


def _reject_constant(value: str) -> None:
    raise ArtifactValidationError(
        "JSON contains non-finite numeric constant {!r}".format(value)
    )


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactValidationError(
                "JSON object contains duplicate key {!r}".format(key)
            )
        result[key] = value
    return result


def _normalise_json(value: Any, location: str = "request") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ArtifactValidationError(
                "{} contains a non-finite number".format(location)
            )
        return value
    if isinstance(value, (list, tuple)):
        return [
            _normalise_json(item, "{}[{}]".format(location, index))
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactValidationError(
                    "{} contains a non-string object key".format(location)
                )
            result[key] = _normalise_json(
                item, "{}.{}".format(location, key)
            )
        return result
    raise ArtifactValidationError(
        "{} contains unsupported JSON value type {}".format(
            location, type(value).__name__
        )
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON for JSON-compatible ``value``.

    Mapping insertion order and insignificant whitespace do not affect the
    result.  Array order remains meaningful, and non-finite values are
    rejected rather than emitted as non-standard JSON.
    """

    normalised = _normalise_json(value)
    return json.dumps(
        normalised,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def request_fingerprint(request: Mapping[str, Any]) -> str:
    """Return an algorithm-labelled fingerprint of a request mapping."""

    if not isinstance(request, Mapping):
        raise ArtifactValidationError("request must be a mapping")
    digest = hashlib.sha256(canonical_json_bytes(request)).hexdigest()
    return "sha256:{}".format(digest)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
    except ArtifactValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(
            "cannot read JSON artifact {}: {}".format(path, error)
        ) from error
    if not isinstance(value, dict):
        raise ArtifactValidationError(
            "JSON artifact {} must contain one object".format(path)
        )
    return value


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a finite, duplicate-key-free JSON object."""

    return _load_json(Path(path).expanduser().resolve())


def request_fingerprint_file(path: Union[str, Path]) -> str:
    """Fingerprint a request JSON file after parsing and canonicalization."""

    return request_fingerprint(read_json(path))


def write_json_atomic(path: Union[str, Path], value: Mapping[str, Any]) -> Path:
    """Atomically replace ``path`` with canonical, fsynced JSON."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".{}.".format(destination.name),
            suffix=".tmp",
            dir=str(destination.parent),
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(destination))
        temporary_name = None
        try:
            directory_fd = os.open(str(destination.parent), os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return destination


def write_npz_atomic(
    path: Union[str, Path], arrays: Mapping[str, np.ndarray]
) -> Path:
    """Atomically replace ``path`` with a pickle-free compressed NPZ."""

    if not isinstance(arrays, Mapping) or not arrays:
        raise ArtifactValidationError(
            "NPZ payload must be a non-empty array mapping"
        )
    payload: Dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if not isinstance(key, str) or not key:
            raise ArtifactValidationError(
                "NPZ payload keys must be non-empty strings"
            )
        selected = np.asarray(value)
        if selected.dtype.hasobject:
            raise ArtifactValidationError(
                "NPZ payload {!r} has forbidden object dtype".format(key)
            )
        payload[key] = selected

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=".{}.".format(destination.name),
            suffix=".tmp",
            dir=str(destination.parent),
            delete=False,
        ) as stream:
            temporary_name = stream.name
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(destination))
        temporary_name = None
        try:
            directory_fd = os.open(str(destination.parent), os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return destination


def _require_keys(
    mapping: Mapping[str, Any], keys: Sequence[str], location: str
) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ArtifactValidationError(
            "{} is missing required keys: {}".format(
                location, ", ".join(missing)
            )
        )


def _required_string(
    mapping: Mapping[str, Any], key: str, location: str
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ArtifactValidationError(
            "{}.{} must be a non-empty string".format(location, key)
        )
    return value


def _string_list(value: Any, location: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ArtifactValidationError(
            "{} must be a list of non-empty strings".format(location)
        )
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ArtifactValidationError(
            "{} must not contain duplicates".format(location)
        )
    return result


def _validate_interval(value: Any, location: str) -> Tuple[float, float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or isinstance(value[0], bool)
        or isinstance(value[1], bool)
    ):
        raise ArtifactValidationError(
            "{} must contain [start, end]".format(location)
        )
    try:
        start, end = float(value[0]), float(value[1])
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError(
            "{} must contain numeric bounds".format(location)
        ) from error
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ArtifactValidationError(
            "{} must contain finite increasing bounds".format(location)
        )
    return start, end


def _validate_known_schema(manifest: Mapping[str, Any]) -> str:
    schema = manifest.get("schema")
    if schema not in _KNOWN_SCHEMAS:
        raise UnsupportedArtifactSchema(
            "unsupported artifact schema {!r}".format(schema)
        )
    return str(schema)


def _validate_status(
    manifest: Mapping[str, Any], require_complete: bool
) -> str:
    status = manifest.get("status")
    if status not in {WRITING_STATUS, COMPLETE_STATUS, CANCELLED_STATUS}:
        raise ArtifactValidationError(
            "manifest.status must be writing, complete, or cancelled"
        )
    if require_complete and status != COMPLETE_STATUS:
        raise IncompleteArtifactError(
            "bundle status is {!r}; only complete bundles are loadable".format(
                status
            )
        )
    return str(status)


def _artifact_path(root: Path, relative: Any, location: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ArtifactValidationError(
            "{} must be a non-empty relative path".format(location)
        )
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ArtifactValidationError(
            "{} must stay inside the bundle".format(location)
        )
    root_resolved = root.resolve()
    candidate = (root_resolved / relative_path).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ArtifactValidationError(
            "{} resolves outside the bundle".format(location)
        )
    if not candidate.is_file():
        raise ArtifactValidationError(
            "{} does not name an existing file: {}".format(
                location, relative
            )
        )
    return candidate


def load_npz_strict(
    path: Union[str, Path], required_keys: Sequence[str] = ()
) -> Dict[str, np.ndarray]:
    """Load and detach every array while rejecting pickle/object payloads."""

    source_path = Path(path).expanduser().resolve()
    try:
        with np.load(str(source_path), allow_pickle=False) as archive:
            arrays: Dict[str, np.ndarray] = {}
            for key in archive.files:
                try:
                    value = np.asarray(archive[key])
                except ValueError as error:
                    raise ArtifactValidationError(
                        "{}:{} cannot be loaded without pickle".format(
                            source_path, key
                        )
                    ) from error
                if value.dtype.hasobject:
                    raise ArtifactValidationError(
                        "{}:{} has forbidden object dtype".format(
                            source_path, key
                        )
                    )
                arrays[key] = value.copy()
    except ArtifactValidationError:
        raise
    except (OSError, ValueError) as error:
        raise ArtifactValidationError(
            "cannot read NPZ artifact {}: {}".format(source_path, error)
        ) from error
    _require_keys(arrays, required_keys, str(source_path))
    return arrays


def _finite_array(
    arrays: Mapping[str, np.ndarray],
    key: str,
    shape: Tuple[Optional[int], ...],
    location: str,
) -> np.ndarray:
    value = arrays[key]
    if value.ndim != len(shape) or any(
        expected is not None and value.shape[index] != expected
        for index, expected in enumerate(shape)
    ):
        raise ArtifactValidationError(
            "{}:{} has shape {}; expected {}".format(
                location, key, value.shape, shape
            )
        )
    if not np.issubdtype(value.dtype, np.number) or np.any(~np.isfinite(value)):
        raise ArtifactValidationError(
            "{}:{} must be a finite numeric array".format(location, key)
        )
    return value


def _validate_preview(arrays: Mapping[str, np.ndarray], location: str) -> None:
    required = (
        "time",
        "position",
        "orientation_xyzw",
        "reference_position",
        "reference_rpy",
        "flight_state",
    )
    _require_keys(arrays, required, location)
    time = _finite_array(arrays, "time", (None,), location)
    count = time.size
    if count < 2 or np.any(np.diff(time) <= 0.0):
        raise ArtifactValidationError(
            "{}:time must be strictly increasing".format(location)
        )
    for key, width in (
        ("position", 3),
        ("orientation_xyzw", 4),
        ("reference_position", 3),
        ("reference_rpy", 3),
    ):
        _finite_array(arrays, key, (count, width), location)
    state = arrays["flight_state"]
    if state.shape != (count,) or not np.issubdtype(state.dtype, np.integer):
        raise ArtifactValidationError(
            "{}:flight_state must be an integer time series".format(location)
        )


def _validate_flight_inspection(
    value: Mapping[str, Any], bag_id: str, location: str
) -> None:
    required = (
        "schema",
        "bag_id",
        "bag_path",
        "bag_size",
        "bag_mtime",
        "bag_sha256",
        "record_time_start",
        "record_time_end",
        "topic_contract",
        "complete_episodes",
        "state5_intervals",
        "recommended_interval",
        "warnings",
        "controller_snapshot",
        "controller_flags",
        "configuration_fingerprint",
        "estimated_work_units",
        "status",
    )
    _require_keys(value, required, location)
    if value["schema"] != FLIGHT_INSPECTION_SCHEMA:
        raise UnsupportedArtifactSchema(
            "{} has unsupported inspection schema {!r}".format(
                location, value["schema"]
            )
        )
    if value["bag_id"] != bag_id:
        raise ArtifactValidationError(
            "{} bag_id does not match manifest".format(location)
        )
    _required_string(value, "bag_path", location)
    _required_string(value, "bag_sha256", location)
    fingerprint = value["configuration_fingerprint"]
    if (
        not isinstance(fingerprint, dict)
        or not isinstance(fingerprint.get("value"), str)
        or not fingerprint["value"]
        or not isinstance(fingerprint.get("complete"), bool)
        or not isinstance(fingerprint.get("missing_components"), list)
    ):
        raise ArtifactValidationError(
            "{}.configuration_fingerprint is invalid".format(location)
        )
    try:
        size = int(value["bag_size"])
        mtime = float(value["bag_mtime"])
        start = float(value["record_time_start"])
        end = float(value["record_time_end"])
        work = value["estimated_work_units"]
        if not isinstance(work, dict):
            raise TypeError("estimated_work_units must be an object")
        required_work = {
            "sample_count",
            "knot_count",
            "lag_profile_point_units",
            "nonlinear_iteration_units",
            "mcmc_proposal_units",
            "estimate_kind",
        }
        if set(work) != required_work:
            raise TypeError(
                "estimated_work_units must contain the sparse-batch fields"
            )
        count_keys = (
            "sample_count",
            "knot_count",
            "lag_profile_point_units",
            "nonlinear_iteration_units",
            "mcmc_proposal_units",
        )
        if any(
            isinstance(work[key], bool) or not isinstance(work[key], int)
            for key in count_keys
        ):
            raise TypeError("sparse-batch work counts must be integers")
        work_counts = tuple(
            int(work[key])
            for key in count_keys
        )
        estimate_kind = work["estimate_kind"]
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError(
            "{} contains invalid numeric metadata".format(location)
        ) from error
    if (
        size < 0
        or any(value < 0 for value in work_counts)
        or not isinstance(estimate_kind, str)
        or not estimate_kind
        or not np.isfinite(mtime)
        or not np.isfinite(start)
        or not np.isfinite(end)
        or end <= start
        or (work_counts[0] == 0) != (work_counts[1] == 0)
        or (
            work_counts[0] > 0
            and work_counts[1] != work_counts[0]
        )
    ):
        raise ArtifactValidationError(
            "{} contains invalid time/size/work metadata".format(location)
        )
    if not isinstance(value["topic_contract"], (dict, list)):
        raise ArtifactValidationError(
            "{}.topic_contract must be an object or list".format(location)
        )
    if not isinstance(value["complete_episodes"], list):
        raise ArtifactValidationError(
            "{}.complete_episodes must be a list".format(location)
        )
    if not isinstance(value["state5_intervals"], list):
        raise ArtifactValidationError(
            "{}.state5_intervals must be a list".format(location)
        )
    recommendation = value["recommended_interval"]
    if recommendation is not None:
        if not isinstance(recommendation, dict) or not isinstance(
            recommendation.get("interval"), dict
        ):
            raise ArtifactValidationError(
                "{}.recommended_interval is invalid".format(location)
            )
        interval = recommendation["interval"]
        _validate_interval(
            [interval.get("start_local_time"), interval.get("end_local_time")],
            "{}.recommended_interval.interval".format(location),
        )
    if not isinstance(value["warnings"], list) or any(
        not isinstance(item, str) for item in value["warnings"]
    ):
        raise ArtifactValidationError(
            "{}.warnings must be a string list".format(location)
        )
    if (
        value["controller_snapshot"] is not None
        and not isinstance(value["controller_snapshot"], dict)
    ) or not isinstance(value["controller_flags"], dict):
        raise ArtifactValidationError(
            "{} controller metadata must be objects".format(location)
        )
    if value["status"] not in {
        "ready",
        "needs_configuration_confirmation",
        "blocked",
    }:
        raise ArtifactValidationError(
            "{}.status is invalid".format(location)
        )


def _load_inspection(
    root: Path, manifest: Mapping[str, Any]
) -> InspectionBundle:
    _require_keys(manifest, ("bag_ids", "artifacts"), "manifest")
    bag_ids = _string_list(manifest["bag_ids"], "manifest.bag_ids")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or not isinstance(
        artifacts.get("bags"), dict
    ):
        raise ArtifactValidationError(
            "manifest.artifacts.bags must be an object"
        )
    bag_artifacts = artifacts["bags"]
    if set(bag_artifacts) != set(bag_ids):
        raise ArtifactValidationError(
            "inspection artifact bag IDs do not match manifest.bag_ids"
        )
    inspections: Dict[str, Mapping[str, Any]] = {}
    previews: Dict[str, Mapping[str, np.ndarray]] = {}
    for bag_id in bag_ids:
        declared = bag_artifacts[bag_id]
        if not isinstance(declared, dict):
            raise ArtifactValidationError(
                "manifest artifact entry for {} must be an object".format(
                    bag_id
                )
            )
        _require_keys(declared, ("inspection", "preview"), bag_id)
        inspection_path = _artifact_path(
            root, declared["inspection"], "{}.inspection".format(bag_id)
        )
        preview_path = _artifact_path(
            root, declared["preview"], "{}.preview".format(bag_id)
        )
        inspection = _load_json(inspection_path)
        _validate_flight_inspection(
            inspection, bag_id, str(inspection_path)
        )
        preview = load_npz_strict(preview_path)
        _validate_preview(preview, str(preview_path))
        inspections[bag_id] = inspection
        previews[bag_id] = preview
    return InspectionBundle(root, dict(manifest), inspections, previews)


def read_manifest(root: Union[str, Path]) -> Dict[str, Any]:
    """Read an inspection manifest without claiming payload completeness."""

    bundle_root = Path(root).expanduser().resolve()
    manifest = _load_json(bundle_root / MANIFEST_NAME)
    _validate_known_schema(manifest)
    _validate_status(manifest, require_complete=False)
    return manifest


def load_inspection_bundle(root: Union[str, Path]) -> InspectionBundle:
    """Strictly load one complete inspection bundle."""

    bundle_root = Path(root).expanduser().resolve()
    manifest = _load_json(bundle_root / MANIFEST_NAME)
    _validate_known_schema(manifest)
    _validate_status(manifest, require_complete=True)
    return _load_inspection(bundle_root, manifest)


def begin_bundle(
    root: Union[str, Path], manifest: Mapping[str, Any]
) -> Path:
    """Create an atomic writing manifest for a new inspection bundle.

    Existing manifests are never overwritten. Producers write all payloads
    after this call and invoke :func:`mark_bundle_complete` only after their
    files are closed.
    """

    bundle_root = Path(root).expanduser().resolve()
    bundle_root.mkdir(parents=True, exist_ok=True)
    destination = bundle_root / MANIFEST_NAME
    if destination.exists():
        raise ArtifactStateError(
            "bundle already has a manifest: {}".format(destination)
        )
    candidate = dict(_normalise_json(manifest, "manifest"))
    _validate_known_schema(candidate)
    candidate["status"] = WRITING_STATUS
    _validate_status(candidate, require_complete=False)
    write_json_atomic(destination, candidate)
    return destination


def mark_bundle_complete(
    root: Union[str, Path], updates: Optional[Mapping[str, Any]] = None
) -> Path:
    """Validate every declared inspection artifact and publish completion."""

    bundle_root = Path(root).expanduser().resolve()
    manifest = read_manifest(bundle_root)
    if manifest["status"] != WRITING_STATUS:
        raise ArtifactStateError("only a writing bundle can become complete")
    candidate = dict(manifest)
    if updates is not None:
        candidate.update(_normalise_json(updates, "manifest updates"))
    candidate["status"] = COMPLETE_STATUS
    _validate_known_schema(candidate)
    _validate_status(candidate, require_complete=True)
    _load_inspection(bundle_root, candidate)
    return write_json_atomic(bundle_root / MANIFEST_NAME, candidate)


def mark_bundle_cancelled(
    root: Union[str, Path],
    reason: str,
    updates: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically make cancellation authoritative over partial payloads."""

    if not isinstance(reason, str) or not reason:
        raise ArtifactValidationError(
            "cancellation reason must be a non-empty string"
        )
    bundle_root = Path(root).expanduser().resolve()
    manifest = read_manifest(bundle_root)
    if manifest["status"] == COMPLETE_STATUS:
        raise ArtifactStateError("a complete bundle cannot be cancelled")
    if manifest["status"] == CANCELLED_STATUS:
        if manifest.get("cancellation_reason") != reason:
            raise ArtifactStateError(
                "cancelled bundle already has a different reason"
            )
        return bundle_root / MANIFEST_NAME
    candidate = dict(manifest)
    if updates is not None:
        candidate.update(_normalise_json(updates, "manifest updates"))
    candidate["status"] = CANCELLED_STATUS
    candidate["cancellation_reason"] = reason
    return write_json_atomic(bundle_root / MANIFEST_NAME, candidate)


__all__ = [
    "ArtifactStateError",
    "ArtifactValidationError",
    "CANCELLED_STATUS",
    "COMPLETE_STATUS",
    "FLIGHT_INSPECTION_SCHEMA",
    "INSPECTION_BUNDLE_SCHEMA",
    "IncompleteArtifactError",
    "InspectionBundle",
    "MANIFEST_NAME",
    "UnsupportedArtifactSchema",
    "WRITING_STATUS",
    "begin_bundle",
    "canonical_json_bytes",
    "load_inspection_bundle",
    "load_npz_strict",
    "mark_bundle_cancelled",
    "mark_bundle_complete",
    "read_json",
    "read_manifest",
    "request_fingerprint",
    "request_fingerprint_file",
    "write_json_atomic",
    "write_npz_atomic",
]
