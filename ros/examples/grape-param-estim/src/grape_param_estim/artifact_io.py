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
ASSIMILATION_RUN_SCHEMA = "grape-param-estim/assimilation-run/v1"
PID_PROPOSAL_EVALUATION_SCHEMA = (
    "grape-param-estim/pid-proposal-evaluation/v1"
)

MANIFEST_NAME = "manifest.json"
WRITING_STATUS = "writing"
COMPLETE_STATUS = "complete"
CANCELLED_STATUS = "cancelled"
_KNOWN_SCHEMAS = {
    INSPECTION_BUNDLE_SCHEMA,
    ASSIMILATION_RUN_SCHEMA,
    PID_PROPOSAL_EVALUATION_SCHEMA,
}


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


@dataclass(frozen=True)
class AssimilationRunBundle:
    root: Path
    manifest: Mapping[str, Any]
    shared_posterior: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, np.ndarray]
    bags: Mapping[str, Mapping[str, np.ndarray]]
    warnings: Tuple[str, ...]


@dataclass(frozen=True)
class PidProposalEvaluationBundle:
    root: Path
    manifest: Mapping[str, Any]
    proposal_ensemble: Mapping[str, np.ndarray]
    summary: Mapping[str, np.ndarray]
    bags: Mapping[str, Mapping[str, np.ndarray]]
    proposed_yaml_path: Path
    proposed_diff_yaml_path: Path


Bundle = Union[
    InspectionBundle,
    AssimilationRunBundle,
    PidProposalEvaluationBundle,
]


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


def _member_ids(
    arrays: Mapping[str, np.ndarray], key: str, location: str
) -> np.ndarray:
    value = arrays[key]
    if (
        value.ndim != 1
        or value.size < 1
        or not np.issubdtype(value.dtype, np.integer)
    ):
        raise ArtifactValidationError(
            "{}:{} must be a non-empty integer vector".format(location, key)
        )
    result = value.astype(np.int64, copy=False)
    if np.unique(result).size != result.size:
        raise ArtifactValidationError(
            "{}:{} contains duplicate member IDs".format(location, key)
        )
    return result


def _same_ids(
    expected: np.ndarray,
    actual: np.ndarray,
    location: str,
) -> None:
    if not np.array_equal(expected, actual):
        raise ArtifactValidationError(
            "{} member_id order does not match the shared source law".format(
                location
            )
        )


def _scalar_bool(
    arrays: Mapping[str, np.ndarray], key: str, location: str
) -> bool:
    value = arrays[key]
    if value.size != 1 or not np.issubdtype(value.dtype, np.bool_):
        raise ArtifactValidationError(
            "{}:{} must contain one boolean".format(location, key)
        )
    return bool(value.reshape(-1)[0])


def _string_vector(
    arrays: Mapping[str, np.ndarray], key: str, size: int, location: str
) -> np.ndarray:
    value = arrays[key]
    if value.shape != (size,) or value.dtype.kind not in {"U", "S"}:
        raise ArtifactValidationError(
            "{}:{} must be a length-{} string vector".format(
                location, key, size
            )
        )
    return value.astype(str)


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
        units = int(work.get("member_bag_forecast_units", -1))
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError(
            "{} contains invalid numeric metadata".format(location)
        ) from error
    if (
        size < 0
        or units < 0
        or not np.isfinite(mtime)
        or not np.isfinite(start)
        or not np.isfinite(end)
        or end <= start
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


_RUN_SHARED_KEYS = (
    "member_id",
    "parameter_coordinates",
    "physical_parameter_coordinates",
    "constant_delay_coordinate",
    "mass",
    "inertia",
    "cog",
    "force_effectiveness",
    "torque_effectiveness",
    "constant_delay",
    "ridge_covariance",
    "ridge_eigenvalues",
    "ridge_eigenvectors",
    "ridge_expected_direction",
    "ridge_expected_variance",
    "mode_id",
    "mode_weight",
    "selected_mode_id",
)

_RUN_BAG_KEYS = (
    "member_id",
    "times",
    "record_times",
    "observed_position",
    "observed_orientation_xyzw",
    "observation_translation_covariance",
    "observation_rotation_covariance",
    "reference_position",
    "reference_linear_velocity",
    "reference_linear_acceleration",
    "reference_rpy",
    "reference_angular_velocity",
    "reference_angular_acceleration",
    "nominal_position",
    "nominal_orientation_xyzw",
    "nominal_linear_velocity",
    "nominal_angular_velocity",
    "nominal_controller_integral",
    "nominal_commanded_thrust",
    "nominal_commanded_gimbal_angle",
    "nominal_actuator_thrust",
    "nominal_actuator_gimbal_angle",
    "nominal_body_wrench",
    "posterior_position",
    "posterior_orientation_xyzw",
    "posterior_linear_velocity",
    "posterior_angular_velocity",
    "posterior_controller_integral",
    "posterior_commanded_thrust",
    "posterior_commanded_gimbal_angle",
    "posterior_actuator_thrust",
    "posterior_actuator_gimbal_angle",
    "posterior_body_wrench",
    "correction_translation",
    "correction_rotation_vector",
    "observed_correction_translation",
    "observed_correction_rotation_vector",
    "residual_wrench_interval",
    "residual_wrench_knot",
    "innovation_ensemble",
    "objective_contribution",
    "pose_component_coverage",
    "initial_position",
    "initial_orientation_xyzw",
    "initial_linear_velocity",
    "initial_angular_velocity",
    "initial_controller_integral",
    "initial_controller_roll_pitch_integration_active",
    "initial_actuator_thrust",
    "initial_actuator_gimbal_angle",
    "actuator_thrust_time_constant",
    "actuator_gimbal_time_constant",
    "actuator_minimum_thrust",
    "actuator_maximum_thrust",
    "actuator_maximum_gimbal_angle",
    "actuator_maximum_gimbal_rate",
    "q_knot_indices",
    "q_knot_times",
    "q_stationary_standard_deviation",
    "q_correlation_time",
    "q_resolution_sufficient",
    "controller_snapshot_groups",
    "controller_snapshot_record_times",
    "controller_snapshot_gains",
    "controller_snapshot_pid_control_flags",
    "controller_snapshot_source_kinds",
    "controller_pid_axis_names",
    "controller_pid_field_names",
    "controller_pid_configuration",
    "controller_xy_control_mode",
    "controller_need_yaw_d_control",
    "controller_start_roll_pitch_integration_height",
    "controller_initial_height",
    "controller_source_compatible_gyro_term",
    "provenance_bag_path",
    "provenance_bag_sha256",
    "provenance_bag_size_bytes",
    "provenance_time_basis",
    "provenance_requested_window",
    "provenance_source_available_window",
    "provenance_selected_flight_state",
    "provenance_topic_names",
    "provenance_topic_types",
)

_RUN_DIAGNOSTIC_KEYS = (
    "iteration",
    "objective",
    "accepted_objective",
    "gradient_norm",
    "step_norm",
    "accepted_fraction",
    "ensemble_rank",
    "converged",
    "termination_reason",
)

_RUN_INITIAL_PRIOR_KEYS = (
    "initial_prior_member_id",
    "requested_prior_control_ensemble",
    "effective_prior_control_ensemble",
    "initial_prior_radial_scale",
    "initial_prior_backoff_trials",
    "initial_prior_maximum_backoff_trials",
    "initial_prior_requested_rank",
    "initial_prior_effective_rank",
    "initial_prior_failed_scale",
    "initial_prior_failure_type",
    "initial_prior_failure_reason",
)


def _validate_shared_posterior(
    arrays: Mapping[str, np.ndarray], location: str
) -> np.ndarray:
    _require_keys(arrays, _RUN_SHARED_KEYS, location)
    member_id = _member_ids(arrays, "member_id", location)
    count = member_id.size
    parameter = _finite_array(
        arrays, "parameter_coordinates", (count, 19), location
    )
    physical = _finite_array(
        arrays, "physical_parameter_coordinates", (count, 18), location
    )
    delay_coordinate = _finite_array(
        arrays, "constant_delay_coordinate", (count,), location
    )
    if not np.array_equal(parameter[:, :18], physical) or not np.array_equal(
        parameter[:, 18], delay_coordinate
    ):
        raise ArtifactValidationError(
            "{} raw shared coordinates are internally misaligned".format(location)
        )
    mass = _finite_array(arrays, "mass", (count,), location)
    inertia = _finite_array(arrays, "inertia", (count, 3, 3), location)
    _finite_array(arrays, "cog", (count, 3), location)
    force = _finite_array(
        arrays, "force_effectiveness", (count, 4), location
    )
    torque = _finite_array(
        arrays, "torque_effectiveness", (count, 4), location
    )
    delay = _finite_array(arrays, "constant_delay", (count,), location)
    if (
        np.any(mass <= 0.0)
        or np.any(force <= 0.0)
        or np.any(torque <= 0.0)
        or np.any(delay < 0.0)
        or any(
            not np.allclose(value, value.T, atol=1.0e-10)
            or np.any(np.linalg.eigvalsh(value) <= 0.0)
            for value in inertia
        )
    ):
        raise ArtifactValidationError(
            "{} physical posterior contains invalid members".format(location)
        )
    covariance = _finite_array(
        arrays, "ridge_covariance", (19, 19), location
    )
    eigenvalues = _finite_array(
        arrays, "ridge_eigenvalues", (19,), location
    )
    eigenvectors = _finite_array(
        arrays, "ridge_eigenvectors", (19, 19), location
    )
    expected_direction = _finite_array(
        arrays, "ridge_expected_direction", (19,), location
    )
    expected_variance = _finite_array(
        arrays, "ridge_expected_variance", (1,), location
    )[0]
    if (
        not np.allclose(covariance, covariance.T, atol=1.0e-10)
        or np.any(eigenvalues < -1.0e-10)
        or not np.all(np.diff(eigenvalues) >= -1.0e-10)
        or not np.allclose(
            eigenvectors.T @ eigenvectors, np.eye(19), atol=1.0e-7
        )
        or not np.allclose(
            covariance,
            eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T,
            atol=1.0e-7,
            rtol=1.0e-6,
        )
        or not np.isclose(np.linalg.norm(expected_direction), 1.0)
        or expected_variance < 0.0
        or not np.isclose(
            expected_variance,
            expected_direction @ covariance @ expected_direction,
            atol=1.0e-8,
            rtol=1.0e-6,
        )
    ):
        raise ArtifactValidationError(
            "{} ridge diagnostics are inconsistent".format(location)
        )
    mode_id = arrays["mode_id"]
    if mode_id.ndim != 1 or mode_id.size < 1 or mode_id.dtype.kind not in {
        "U",
        "S",
    }:
        raise ArtifactValidationError(
            "{}:mode_id must be a non-empty string law".format(location)
        )
    modes = mode_id.astype(str)
    if any(not value for value in modes) or np.unique(modes).size != modes.size:
        raise ArtifactValidationError(
            "{}:mode_id values must be unique and non-empty".format(location)
        )
    mode_weight = _finite_array(
        arrays, "mode_weight", (modes.size,), location
    )
    selected_mode = _string_vector(
        arrays, "selected_mode_id", 1, location
    )[0]
    if (
        np.any(mode_weight < 0.0)
        or not np.isclose(np.sum(mode_weight), 1.0)
        or selected_mode not in set(modes)
    ):
        raise ArtifactValidationError(
            "{} mode law or selected mode is invalid".format(location)
        )
    return member_id


def _validate_run_bag(
    arrays: Mapping[str, np.ndarray],
    shared_member_id: np.ndarray,
    location: str,
) -> bool:
    _require_keys(arrays, _RUN_BAG_KEYS, location)
    member_id = _member_ids(arrays, "member_id", location)
    _same_ids(shared_member_id, member_id, location)
    members = member_id.size
    times = _finite_array(arrays, "times", (None,), location)
    count = times.size
    if count < 2 or np.any(np.diff(times) <= 0.0):
        raise ArtifactValidationError(
            "{}:times must be strictly increasing".format(location)
        )
    record_times = _finite_array(
        arrays, "record_times", (count,), location
    )
    if np.any(np.diff(record_times) <= 0.0) or not np.allclose(
        times, record_times - record_times[0], atol=2.0e-7, rtol=0.0
    ):
        raise ArtifactValidationError(
            "{}:record_times must be increasing and align to local time".format(
                location
            )
        )
    for key, width in (
        ("observed_position", 3),
        ("observed_orientation_xyzw", 4),
        ("reference_position", 3),
        ("reference_linear_velocity", 3),
        ("reference_linear_acceleration", 3),
        ("reference_rpy", 3),
        ("reference_angular_velocity", 3),
        ("reference_angular_acceleration", 3),
        ("nominal_position", 3),
        ("nominal_orientation_xyzw", 4),
        ("nominal_linear_velocity", 3),
        ("nominal_angular_velocity", 3),
        ("nominal_controller_integral", 6),
        ("nominal_commanded_thrust", 4),
        ("nominal_commanded_gimbal_angle", 4),
        ("nominal_actuator_thrust", 4),
        ("nominal_actuator_gimbal_angle", 4),
        ("nominal_body_wrench", 6),
        ("observed_correction_translation", 3),
        ("observed_correction_rotation_vector", 3),
    ):
        _finite_array(arrays, key, (count, width), location)
    for key in (
        "observation_translation_covariance",
        "observation_rotation_covariance",
    ):
        covariance = _finite_array(arrays, key, (3, 3), location)
        if not np.allclose(covariance, covariance.T, atol=1.0e-12) or np.any(
            np.linalg.eigvalsh(covariance) < -1.0e-12
        ):
            raise ArtifactValidationError(
                "{}:{} must be positive semidefinite".format(location, key)
            )
    for key, width in (
        ("posterior_position", 3),
        ("posterior_orientation_xyzw", 4),
        ("posterior_linear_velocity", 3),
        ("posterior_angular_velocity", 3),
        ("posterior_controller_integral", 6),
        ("posterior_commanded_thrust", 4),
        ("posterior_commanded_gimbal_angle", 4),
        ("posterior_actuator_thrust", 4),
        ("posterior_actuator_gimbal_angle", 4),
        ("posterior_body_wrench", 6),
        ("correction_translation", 3),
        ("correction_rotation_vector", 3),
    ):
        _finite_array(arrays, key, (members, count, width), location)
    for key in (
        "observed_orientation_xyzw",
        "nominal_orientation_xyzw",
    ):
        quaternion = arrays[key]
        if not np.allclose(np.linalg.norm(quaternion, axis=-1), 1.0, atol=1.0e-6):
            raise ArtifactValidationError(
                "{}:{} must contain unit quaternions".format(location, key)
            )
    if not np.allclose(
        np.linalg.norm(arrays["posterior_orientation_xyzw"], axis=-1),
        1.0,
        atol=1.0e-6,
    ):
        raise ArtifactValidationError(
            "{} posterior orientations must be unit quaternions".format(
                location
            )
        )
    _finite_array(
        arrays,
        "residual_wrench_interval",
        (members, count - 1, 6),
        location,
    )
    knot_indices = arrays["q_knot_indices"]
    if (
        knot_indices.ndim != 1
        or knot_indices.size < 2
        or not np.issubdtype(knot_indices.dtype, np.integer)
        or knot_indices[0] != 0
        or knot_indices[-1] != count - 1
        or np.any(np.diff(knot_indices) <= 0)
    ):
        raise ArtifactValidationError(
            "{}:q_knot_indices must span the time boundaries".format(location)
        )
    knots = knot_indices.size
    knot_times = _finite_array(arrays, "q_knot_times", (knots,), location)
    if not np.allclose(knot_times, times[knot_indices]):
        raise ArtifactValidationError(
            "{} Q knot times do not match indices".format(location)
        )
    _finite_array(
        arrays, "residual_wrench_knot", (members, knots, 6), location
    )
    _finite_array(
        arrays, "innovation_ensemble", (members, knots * 6), location
    )
    objective = _finite_array(
        arrays, "objective_contribution", (members,), location
    )
    coverage = _finite_array(
        arrays, "pose_component_coverage", (1,), location
    )[0]
    sigma = _finite_array(
        arrays, "q_stationary_standard_deviation", (6,), location
    )
    correlation = _finite_array(
        arrays, "q_correlation_time", (1,), location
    )[0]
    if (
        np.any(objective < 0.0)
        or not 0.0 <= coverage <= 1.0
        or np.any(sigma <= 0.0)
        or correlation <= 0.0
    ):
        raise ArtifactValidationError(
            "{} objective, coverage, or Q calibration is invalid".format(
                location
            )
        )
    for key, width in (
        ("initial_position", 3),
        ("initial_orientation_xyzw", 4),
        ("initial_linear_velocity", 3),
        ("initial_angular_velocity", 3),
        ("initial_controller_integral", 6),
        ("initial_actuator_thrust", 4),
        ("initial_actuator_gimbal_angle", 4),
    ):
        _finite_array(arrays, key, (members, width), location)
    if not np.allclose(
        np.linalg.norm(arrays["initial_orientation_xyzw"], axis=1),
        1.0,
        atol=1.0e-6,
    ):
        raise ArtifactValidationError(
            "{} initial orientations must be unit quaternions".format(location)
        )
    active = arrays["initial_controller_roll_pitch_integration_active"]
    if active.shape != (members,) or not np.issubdtype(active.dtype, np.bool_):
        raise ArtifactValidationError(
            "{} controller-state flags must align with members".format(location)
        )
    actuator_scalars = {}
    for key in (
        "actuator_thrust_time_constant",
        "actuator_gimbal_time_constant",
        "actuator_minimum_thrust",
        "actuator_maximum_thrust",
        "actuator_maximum_gimbal_angle",
        "actuator_maximum_gimbal_rate",
    ):
        actuator_scalars[key] = _finite_array(
            arrays, key, (1,), location
        )[0]
    if (
        actuator_scalars["actuator_thrust_time_constant"] < 0.0
        or actuator_scalars["actuator_gimbal_time_constant"] < 0.0
        or actuator_scalars["actuator_maximum_thrust"]
        <= actuator_scalars["actuator_minimum_thrust"]
        or actuator_scalars["actuator_maximum_gimbal_angle"] <= 0.0
        or actuator_scalars["actuator_maximum_gimbal_rate"] <= 0.0
    ):
        raise ArtifactValidationError(
            "{} actuator parameter snapshot is invalid".format(location)
        )

    groups = _string_vector(
        arrays, "controller_snapshot_groups", 4, location
    )
    if tuple(groups) != ("xy", "z", "roll_pitch", "yaw"):
        raise ArtifactValidationError(
            "{} controller snapshot group order is invalid".format(location)
        )
    _finite_array(
        arrays, "controller_snapshot_record_times", (4,), location
    )
    gains = _finite_array(
        arrays, "controller_snapshot_gains", (4, 3), location
    )
    snapshot_flags = arrays["controller_snapshot_pid_control_flags"]
    if (
        np.any(gains < 0.0)
        or snapshot_flags.shape != (4,)
        or not np.issubdtype(snapshot_flags.dtype, np.bool_)
    ):
        raise ArtifactValidationError(
            "{} controller gain snapshot is invalid".format(location)
        )
    _string_vector(
        arrays, "controller_snapshot_source_kinds", 4, location
    )
    axes = _string_vector(arrays, "controller_pid_axis_names", 6, location)
    fields = arrays["controller_pid_field_names"]
    if fields.ndim != 1 or fields.size < 3 or fields.dtype.kind not in {"U", "S"}:
        raise ArtifactValidationError(
            "{} controller PID field names are invalid".format(location)
        )
    if tuple(axes) != ("x", "y", "z", "roll", "pitch", "yaw"):
        raise ArtifactValidationError(
            "{} controller PID axis order is invalid".format(location)
        )
    _finite_array(
        arrays,
        "controller_pid_configuration",
        (6, fields.size),
        location,
    )
    _string_vector(arrays, "controller_xy_control_mode", 1, location)
    for key in (
        "controller_need_yaw_d_control",
        "controller_source_compatible_gyro_term",
    ):
        _scalar_bool(arrays, key, location)
    _finite_array(
        arrays,
        "controller_start_roll_pitch_integration_height",
        (1,),
        location,
    )
    _finite_array(arrays, "controller_initial_height", (1,), location)
    for key in (
        "provenance_bag_path",
        "provenance_bag_sha256",
        "provenance_time_basis",
    ):
        _string_vector(arrays, key, 1, location)
    size = arrays["provenance_bag_size_bytes"]
    state = arrays["provenance_selected_flight_state"]
    if (
        size.shape != (1,)
        or not np.issubdtype(size.dtype, np.integer)
        or size[0] < 0
        or state.shape != (1,)
        or not np.issubdtype(state.dtype, np.integer)
    ):
        raise ArtifactValidationError(
            "{} provenance integer fields are invalid".format(location)
        )
    for key in (
        "provenance_requested_window",
        "provenance_source_available_window",
    ):
        interval = _finite_array(arrays, key, (2,), location)
        if interval[1] <= interval[0]:
            raise ArtifactValidationError(
                "{}:{} must be increasing".format(location, key)
            )
    topic_names = arrays["provenance_topic_names"]
    topic_types = arrays["provenance_topic_types"]
    if (
        topic_names.ndim != 1
        or topic_types.shape != topic_names.shape
        or topic_names.dtype.kind not in {"U", "S"}
        or topic_types.dtype.kind not in {"U", "S"}
    ):
        raise ArtifactValidationError(
            "{} topic provenance must be aligned strings".format(location)
        )
    return _scalar_bool(arrays, "q_resolution_sufficient", location)


def _validate_run_diagnostics(
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    member_id: np.ndarray,
    location: str,
) -> Optional[Tuple[float, int]]:
    _require_keys(arrays, _RUN_DIAGNOSTIC_KEYS, location)
    iteration = arrays["iteration"]
    if (
        iteration.ndim != 1
        or iteration.size < 1
        or not np.issubdtype(iteration.dtype, np.integer)
        or not np.array_equal(
            iteration.astype(np.int64), np.arange(iteration.size)
        )
    ):
        raise ArtifactValidationError(
            "{}:iteration must be a zero-based sequence".format(location)
        )
    count = iteration.size
    values = {}
    for key in (
        "objective",
        "accepted_objective",
        "gradient_norm",
        "step_norm",
        "accepted_fraction",
    ):
        values[key] = _finite_array(arrays, key, (count,), location)
    if (
        np.any(values["objective"] < 0.0)
        or np.any(values["accepted_objective"] < 0.0)
        or np.any(
            values["accepted_objective"] > values["objective"] + 1.0e-8
        )
        or np.any(values["gradient_norm"] < 0.0)
        or np.any(values["step_norm"] < 0.0)
        or np.any(values["accepted_fraction"] < 0.0)
        or np.any(values["accepted_fraction"] > 1.0)
    ):
        raise ArtifactValidationError(
            "{} iteration diagnostics are outside their domains".format(
                location
            )
        )
    rank = arrays["ensemble_rank"]
    if (
        rank.shape != (1,)
        or not np.issubdtype(rank.dtype, np.integer)
        or rank[0] < 1
        or rank[0] >= manifest["shared_member_count"]
    ):
        raise ArtifactValidationError(
            "{} ensemble rank is invalid".format(location)
        )
    converged = _scalar_bool(arrays, "converged", location)
    reason = _string_vector(arrays, "termination_reason", 1, location)[0]
    if (
        converged != manifest["converged"]
        or reason != manifest["termination_reason"]
    ):
        raise ArtifactValidationError(
            "{} termination diagnostics do not match manifest".format(
                location
            )
        )

    present = tuple(key in arrays for key in _RUN_INITIAL_PRIOR_KEYS)
    manifest_audit = manifest.get("initial_prior_forecast")
    if any(present) and not all(present):
        raise ArtifactValidationError(
            "{} initial-prior audit fields must be all present or all absent".format(
                location
            )
        )
    if all(present) != (manifest_audit is not None):
        raise ArtifactValidationError(
            "{} initial-prior diagnostics and manifest must appear together".format(
                location
            )
        )
    if not all(present):
        return None

    initial_member_id = _member_ids(
        arrays, "initial_prior_member_id", location
    )
    _same_ids(member_id, initial_member_id, location)
    member_count = member_id.size
    requested = _finite_array(
        arrays,
        "requested_prior_control_ensemble",
        (member_count, None),
        location,
    )
    effective = _finite_array(
        arrays,
        "effective_prior_control_ensemble",
        requested.shape,
        location,
    )
    if requested.shape[1] < 1:
        raise ArtifactValidationError(
            "{} initial-prior controls cannot be empty".format(location)
        )

    radial_scale = _finite_array(
        arrays, "initial_prior_radial_scale", (1,), location
    )[0]

    def integer_scalar(key: str) -> int:
        value = arrays[key]
        if value.shape != (1,) or not np.issubdtype(
            value.dtype, np.integer
        ):
            raise ArtifactValidationError(
                "{}:{} must contain one integer".format(location, key)
            )
        return int(value[0])

    backoff_trials = integer_scalar("initial_prior_backoff_trials")
    maximum_trials = integer_scalar(
        "initial_prior_maximum_backoff_trials"
    )
    requested_rank = integer_scalar("initial_prior_requested_rank")
    effective_rank = integer_scalar("initial_prior_effective_rank")
    if (
        not 0.0 < radial_scale <= 1.0
        or not 0 <= backoff_trials <= maximum_trials <= 30
        or not np.isclose(
            radial_scale,
            2.0 ** (-backoff_trials),
            rtol=0.0,
            atol=1.0e-15,
        )
    ):
        raise ArtifactValidationError(
            "{} initial-prior radial-backoff metadata is invalid".format(
                location
            )
        )

    failed_scale = _finite_array(
        arrays,
        "initial_prior_failed_scale",
        (backoff_trials,),
        location,
    )
    failure_type = _string_vector(
        arrays,
        "initial_prior_failure_type",
        backoff_trials,
        location,
    )
    failure_reason = _string_vector(
        arrays,
        "initial_prior_failure_reason",
        backoff_trials,
        location,
    )
    expected_failed_scale = 2.0 ** (-np.arange(backoff_trials, dtype=float))
    if (
        not np.allclose(
            failed_scale,
            expected_failed_scale,
            rtol=0.0,
            atol=1.0e-15,
        )
        or any(not value for value in failure_type)
        or any(not value for value in failure_reason)
    ):
        raise ArtifactValidationError(
            "{} initial-prior failure provenance is invalid".format(location)
        )

    requested_mean = np.mean(requested, axis=0, keepdims=True)
    expected_effective = requested_mean + radial_scale * (
        requested - requested_mean
    )
    tolerance = 5.0e-13 * max(
        1.0, float(np.max(np.abs(expected_effective)))
    )
    if (
        not np.allclose(
            effective,
            expected_effective,
            rtol=5.0e-13,
            atol=tolerance,
        )
        or not np.allclose(
            np.mean(effective, axis=0),
            requested_mean[0],
            rtol=5.0e-13,
            atol=tolerance,
        )
    ):
        raise ArtifactValidationError(
            "{} effective prior is not the declared global radial transform".format(
                location
            )
        )
    calculated_requested_rank = int(
        np.linalg.matrix_rank(requested - requested_mean)
    )
    effective_mean = np.mean(effective, axis=0, keepdims=True)
    calculated_effective_rank = int(
        np.linalg.matrix_rank(effective - effective_mean)
    )
    expected_rank = min(requested.shape[1], member_count - 1)
    if (
        requested_rank != calculated_requested_rank
        or effective_rank != calculated_effective_rank
        or requested_rank != effective_rank
        or requested_rank != expected_rank
    ):
        raise ArtifactValidationError(
            "{} initial-prior member shape/rank was not preserved".format(
                location
            )
        )

    if not isinstance(manifest_audit, dict):
        raise ArtifactValidationError(
            "manifest.initial_prior_forecast must be an object"
        )
    required_manifest_audit = (
        "strategy",
        "radial_scale",
        "backoff_trials",
        "maximum_backoff_trials",
        "requested_member_count",
        "effective_member_count",
        "requested_rank",
        "effective_rank",
        "failed_attempts",
        "effective_prior_source",
    )
    _require_keys(
        manifest_audit,
        required_manifest_audit,
        "manifest.initial_prior_forecast",
    )
    failed_attempts = manifest_audit["failed_attempts"]
    if (
        manifest_audit["strategy"] != "global_radial_dyadic_backoff"
        or manifest_audit["effective_prior_source"]
        != "diagnostics.npz:effective_prior_control_ensemble"
        or manifest_audit["radial_scale"] != float(radial_scale)
        or manifest_audit["backoff_trials"] != backoff_trials
        or manifest_audit["maximum_backoff_trials"] != maximum_trials
        or manifest_audit["requested_member_count"] != member_count
        or manifest_audit["effective_member_count"] != member_count
        or manifest_audit["requested_rank"] != requested_rank
        or manifest_audit["effective_rank"] != effective_rank
        or not isinstance(failed_attempts, list)
        or len(failed_attempts) != backoff_trials
    ):
        raise ArtifactValidationError(
            "manifest initial-prior audit does not match diagnostics"
        )
    for index, attempt in enumerate(failed_attempts):
        if (
            not isinstance(attempt, dict)
            or attempt.get("radial_scale") != float(failed_scale[index])
            or attempt.get("exception_type") != failure_type[index]
            or attempt.get("reason") != failure_reason[index]
        ):
            raise ArtifactValidationError(
                "manifest initial-prior failure {} does not match diagnostics".format(
                    index
                )
            )
    return float(radial_scale), backoff_trials


def _validate_run_manifest(
    manifest: Mapping[str, Any]
) -> Tuple[Tuple[str, ...], Mapping[str, Any]]:
    required = (
        "run_id",
        "created_at",
        "estimator_revision",
        "request_path",
        "request_fingerprint",
        "project_request_fingerprint",
        "selected_bag_ids",
        "selected_intervals",
        "configuration_fingerprint",
        "shared_member_count",
        "termination_reason",
        "converged",
        "artifacts",
    )
    _require_keys(manifest, required, "manifest")
    for key in (
        "run_id",
        "created_at",
        "estimator_revision",
        "request_path",
        "request_fingerprint",
        "project_request_fingerprint",
        "configuration_fingerprint",
        "termination_reason",
    ):
        _required_string(manifest, key, "manifest")
    bag_ids = _string_list(
        manifest["selected_bag_ids"], "manifest.selected_bag_ids"
    )
    intervals = manifest["selected_intervals"]
    if not isinstance(intervals, dict) or set(intervals) != set(bag_ids):
        raise ArtifactValidationError(
            "manifest.selected_intervals must map every selected bag"
        )
    for bag_id in bag_ids:
        _validate_interval(
            intervals[bag_id],
            "manifest.selected_intervals.{}".format(bag_id),
        )
    member_count = manifest["shared_member_count"]
    if (
        isinstance(member_count, bool)
        or not isinstance(member_count, int)
        or member_count < 1
    ):
        raise ArtifactValidationError(
            "manifest.shared_member_count must be a positive integer"
        )
    if not isinstance(manifest["converged"], bool):
        raise ArtifactValidationError("manifest.converged must be boolean")
    reason = str(manifest["termination_reason"]).lower().replace("-", "_")
    if reason in {"maximum_iterations", "max_iterations"} and manifest[
        "converged"
    ]:
        raise ArtifactValidationError(
            "maximum_iterations termination cannot be labelled converged"
        )
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict):
        raise ArtifactValidationError("manifest.artifacts must be an object")
    _require_keys(
        artifacts, ("shared_posterior", "diagnostics", "bags"), "artifacts"
    )
    if not isinstance(artifacts["bags"], dict) or set(
        artifacts["bags"]
    ) != set(bag_ids):
        raise ArtifactValidationError(
            "manifest.artifacts.bags must map every selected bag"
        )
    return bag_ids, artifacts


def _load_run(
    root: Path, manifest: Mapping[str, Any]
) -> AssimilationRunBundle:
    bag_ids, artifacts = _validate_run_manifest(manifest)
    shared_path = _artifact_path(
        root, artifacts["shared_posterior"], "artifacts.shared_posterior"
    )
    diagnostics_path = _artifact_path(
        root, artifacts["diagnostics"], "artifacts.diagnostics"
    )
    shared = load_npz_strict(shared_path)
    member_id = _validate_shared_posterior(shared, str(shared_path))
    if member_id.size != manifest["shared_member_count"]:
        raise ArtifactValidationError(
            "manifest.shared_member_count does not match shared_posterior"
        )
    diagnostics = load_npz_strict(diagnostics_path)
    initial_prior_audit = _validate_run_diagnostics(
        diagnostics, manifest, member_id, str(diagnostics_path)
    )
    bags: Dict[str, Mapping[str, np.ndarray]] = {}
    warnings = []
    if initial_prior_audit is not None and initial_prior_audit[1] > 0:
        warnings.append(
            "Initial prior ensemble was globally contracted to radial scale "
            "{:.12g} after {} numerical forecast failure(s); member IDs, "
            "center, shape, and rank were preserved".format(
                initial_prior_audit[0], initial_prior_audit[1]
            )
        )
    for bag_id in bag_ids:
        path = _artifact_path(
            root,
            artifacts["bags"][bag_id],
            "artifacts.bags.{}".format(bag_id),
        )
        arrays = load_npz_strict(path)
        if not _validate_run_bag(arrays, member_id, str(path)):
            warnings.append(
                "{}: Q resolution is insufficient".format(bag_id)
            )
        bags[bag_id] = arrays
    return AssimilationRunBundle(
        root, dict(manifest), shared, diagnostics, bags, tuple(warnings)
    )


_PID_PROPOSAL_KEYS = (
    "source_run_id",
    "source_member_id",
    "proposal_source_member_id",
    "source_mode_id",
    "xy_scale",
    "z_scale",
    "roll_pitch_scale",
    "yaw_scale",
    "proposed_pid",
    "current_pid",
    "constant_delay",
    "acceleration_response",
    "proposal_range_50",
    "proposal_range_95",
)

_PID_SUMMARY_KEYS = (
    "source_run_id",
    "source_member_id",
    "source_mode_id",
    "bag_id",
    "candidate_id",
    "candidate_source",
    "candidate_source_member_id",
    "candidate_source_mode_id",
    "current_pid",
    "current_pid_baseline_bag_id",
    "current_pid_snapshot_group",
    "current_pid_snapshot_topic",
    "current_pid_snapshot_record_time",
    "current_pid_snapshot_source_kind",
    "proposed_pid",
    "difference",
    "ratio",
    "ratio_configured",
    "member_bag_position_rmse",
    "member_bag_orientation_rmse",
    "member_bag_maximum_position_error",
    "member_bag_maximum_orientation_error",
    "member_bag_forecast_completion",
    "member_bag_failure_reason",
    "member_bag_position_threshold_exceeded",
    "member_bag_orientation_threshold_exceeded",
    "position_threshold",
    "orientation_threshold",
    "position_threshold_configured",
    "orientation_threshold_configured",
    "position_threshold_metric",
    "orientation_threshold_metric",
    "cvar_level",
    "correction_coverage_interval",
    "log_gain_change",
    "forecast_completion",
    "numerical_failure_count",
    "per_bag_forecast_completion",
    "per_bag_numerical_failure_count",
    "per_bag_position_threshold_exceedance",
    "per_bag_orientation_threshold_exceedance",
    "aggregate_position_threshold_exceedance",
    "aggregate_orientation_threshold_exceedance",
    "pareto_dominated",
    "pareto_non_dominated",
    "improves_current",
    "candidate_eligible",
    "candidate_rejection_reason",
    "selected_candidate_id",
    "recommendation_available",
    "recommended_candidate_id",
    "rejection_reason",
    "improvement_rule",
    "scenario_assumption",
    "per_bag_correction_translation_zero_coverage",
    "per_bag_correction_rotation_zero_coverage",
    "per_bag_correction_transform_zero_coverage",
)

_PID_PHYSICAL_METRICS = (
    "position_rmse",
    "orientation_rmse",
    "maximum_position_error",
    "maximum_orientation_error",
)
_PID_SUMMARY_KEYS += tuple(
    "{}_{}_{}".format(scope, metric, statistic)
    for scope in ("per_bag", "aggregate")
    for metric in _PID_PHYSICAL_METRICS
    for statistic in ("mean", "upper_cvar")
)

_PID_BAG_KEYS = (
    "member_id",
    "candidate_id",
    "times",
    "reference_position",
    "reference_rpy",
    "prediction_position",
    "prediction_orientation_xyzw",
    "correction_translation",
    "correction_rotation_vector",
    "position_error",
    "orientation_error_rotation_vector",
    "position_rmse",
    "orientation_rmse",
    "maximum_position_error",
    "maximum_orientation_error",
    "forecast_success",
    "forecast_failure_reason",
    "residual_policy",
    "correction_coverage_interval",
    "correction_translation_zero_coverage",
    "correction_rotation_zero_coverage",
    "correction_transform_zero_coverage",
)


def _forbid_controller_mass(
    arrays: Mapping[str, np.ndarray], location: str
) -> None:
    forbidden = [key for key in arrays if "controller_mass" in key.lower()]
    if forbidden:
        raise ArtifactValidationError(
            "{} contains forbidden controller-mass proposal fields: {}".format(
                location, ", ".join(forbidden)
            )
        )


def _numeric_with_missing(
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
    if not np.issubdtype(value.dtype, np.number) or np.any(np.isinf(value)):
        raise ArtifactValidationError(
            "{}:{} must be numeric and cannot contain infinity".format(
                location, key
            )
        )
    return value


def _scalar_string(
    arrays: Mapping[str, np.ndarray], key: str, location: str
) -> str:
    value = arrays[key]
    if value.size != 1 or value.dtype.kind not in {"U", "S"}:
        raise ArtifactValidationError(
            "{}:{} must contain one string".format(location, key)
        )
    return str(value.reshape(-1)[0])


def _validate_pid_proposal(
    arrays: Mapping[str, np.ndarray], location: str
) -> np.ndarray:
    _require_keys(arrays, _PID_PROPOSAL_KEYS, location)
    _forbid_controller_mass(arrays, location)
    source_run_id = _scalar_string(arrays, "source_run_id", location)
    if not source_run_id:
        raise ArtifactValidationError(
            "{}:source_run_id cannot be empty".format(location)
        )
    member_id = _member_ids(arrays, "source_member_id", location)
    members = member_id.size
    proposal_source = _member_ids(
        arrays, "proposal_source_member_id", location
    )
    _same_ids(member_id, proposal_source, location)
    source_modes = _string_vector(
        arrays, "source_mode_id", members, location
    )
    if any(not value for value in source_modes) or np.unique(
        source_modes
    ).size != 1:
        raise ArtifactValidationError(
            "{} PID proposal must retain one non-empty mode law".format(
                location
            )
        )
    scales = []
    for key in ("xy_scale", "z_scale", "roll_pitch_scale", "yaw_scale"):
        scale = _finite_array(arrays, key, (members,), location)
        if np.any(scale <= 0.0):
            raise ArtifactValidationError(
                "{}:{} must contain positive scales".format(location, key)
            )
        scales.append(scale)
    proposed = _finite_array(
        arrays, "proposed_pid", (members, 4, 3), location
    )
    current = _finite_array(arrays, "current_pid", (4, 3), location)
    delay = _finite_array(arrays, "constant_delay", (members,), location)
    _finite_array(
        arrays, "acceleration_response", (members, 6, 6), location
    )
    range_50 = _finite_array(
        arrays, "proposal_range_50", (2, 4, 3), location
    )
    range_95 = _finite_array(
        arrays, "proposal_range_95", (2, 4, 3), location
    )
    scale_matrix = np.stack(scales, axis=1)
    if (
        np.any(proposed < 0.0)
        or np.any(current < 0.0)
        or np.any(delay < 0.0)
        or not np.allclose(
            proposed,
            current[None, :, :] * scale_matrix[:, :, None],
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        or np.any(range_50[0] > range_50[1])
        or np.any(range_95[0] > range_95[1])
        or np.any(range_95[0] > range_50[0])
        or np.any(range_50[1] > range_95[1])
    ):
        raise ArtifactValidationError(
            "{} PID proposal values/ranges are inconsistent".format(location)
        )
    return member_id


def _validate_pid_summary(
    arrays: Mapping[str, np.ndarray],
    source_member_id: np.ndarray,
    bag_ids: Sequence[str],
    source_run_id: str,
    location: str,
) -> np.ndarray:
    _require_keys(arrays, _PID_SUMMARY_KEYS, location)
    _forbid_controller_mass(arrays, location)
    if _scalar_string(arrays, "source_run_id", location) != source_run_id:
        raise ArtifactValidationError(
            "{} source_run_id does not match manifest".format(location)
        )
    summary_member = _member_ids(arrays, "source_member_id", location)
    _same_ids(source_member_id, summary_member, location)
    members = source_member_id.size
    source_modes = _string_vector(
        arrays, "source_mode_id", members, location
    )
    summary_bags = _string_vector(
        arrays, "bag_id", len(bag_ids), location
    )
    if not np.array_equal(summary_bags, np.asarray(bag_ids)):
        raise ArtifactValidationError(
            "{} bag_id order does not match manifest".format(location)
        )
    candidate = arrays["candidate_id"]
    if (
        candidate.ndim != 1
        or candidate.size < 1
        or candidate.dtype.kind not in {"U", "S"}
    ):
        raise ArtifactValidationError(
            "{}:candidate_id must be a non-empty string vector".format(
                location
            )
        )
    candidate = candidate.astype(str)
    if any(not item for item in candidate) or np.unique(candidate).size != (
        candidate.size
    ):
        raise ArtifactValidationError(
            "{}:candidate_id must be non-empty and unique".format(location)
        )
    count = candidate.size
    if candidate[0] != "current":
        raise ArtifactValidationError(
            "{} current baseline must be the first candidate".format(location)
        )
    sources = _string_vector(arrays, "candidate_source", count, location)
    if sources[0] != "current" or np.count_nonzero(
        sources == "current"
    ) != 1 or any(
        value not in {"current", "member-derived", "user"}
        for value in sources
    ):
        raise ArtifactValidationError(
            "{} candidate sources are invalid".format(location)
        )
    candidate_member = arrays["candidate_source_member_id"]
    if candidate_member.shape != (count,) or not np.issubdtype(
        candidate_member.dtype, np.integer
    ):
        raise ArtifactValidationError(
            "{} candidate source member IDs are invalid".format(location)
        )
    candidate_mode = _string_vector(
        arrays, "candidate_source_mode_id", count, location
    )
    known_members = set(int(value) for value in source_member_id)
    known_modes = set(source_modes)
    for index, source in enumerate(sources):
        if source == "member-derived":
            if (
                int(candidate_member[index]) not in known_members
                or candidate_mode[index] not in known_modes
            ):
                raise ArtifactValidationError(
                    "{} member-derived candidate provenance is invalid".format(
                        location
                    )
                )
        elif candidate_member[index] != -1 or candidate_mode[index]:
            raise ArtifactValidationError(
                "{} only member-derived candidates may name a source member".format(
                    location
                )
            )
    current = _finite_array(arrays, "current_pid", (4, 3), location)
    proposed = _finite_array(
        arrays, "proposed_pid", (count, 4, 3), location
    )
    difference = _finite_array(
        arrays, "difference", (count, 4, 3), location
    )
    ratio = _numeric_with_missing(
        arrays, "ratio", (count, 4, 3), location
    )
    ratio_configured = arrays["ratio_configured"]
    if ratio_configured.shape != ratio.shape or not np.issubdtype(
        ratio_configured.dtype, np.bool_
    ):
        raise ArtifactValidationError(
            "{} ratio_configured must be a boolean candidate/gain array".format(
                location
            )
        )
    expected_configured = np.broadcast_to(current != 0.0, ratio.shape)
    expected_ratio = np.full(ratio.shape, np.nan)
    np.divide(
        proposed,
        current[None, :, :],
        out=expected_ratio,
        where=expected_configured,
    )
    if (
        np.any(current < 0.0)
        or np.any(proposed < 0.0)
        or not np.array_equal(proposed[0], current)
        or not np.allclose(difference, proposed - current[None, :, :])
        or not np.array_equal(ratio_configured, expected_configured)
        or not np.allclose(ratio, expected_ratio, equal_nan=True)
    ):
        raise ArtifactValidationError(
            "{} exact PID comparison arrays are inconsistent".format(location)
        )
    baseline_bag = _scalar_string(
        arrays, "current_pid_baseline_bag_id", location
    )
    snapshot_group = _string_vector(
        arrays, "current_pid_snapshot_group", 4, location
    )
    snapshot_topic = _string_vector(
        arrays, "current_pid_snapshot_topic", 4, location
    )
    _finite_array(
        arrays, "current_pid_snapshot_record_time", (4,), location
    )
    snapshot_source = _string_vector(
        arrays, "current_pid_snapshot_source_kind", 4, location
    )
    if (
        baseline_bag not in set(bag_ids)
        or tuple(snapshot_group)
        != ("xy", "z", "roll_pitch", "yaw")
        or any(not value for value in snapshot_topic)
        or any(not value for value in snapshot_source)
    ):
        raise ArtifactValidationError(
            "{} current PID snapshot provenance is invalid".format(location)
        )
    bags = len(bag_ids)
    completion = arrays["member_bag_forecast_completion"]
    reason = arrays["member_bag_failure_reason"]
    expected_member_shape = (count, bags, members)
    if completion.shape != expected_member_shape or not np.issubdtype(
        completion.dtype, np.bool_
    ):
        raise ArtifactValidationError(
            "{} member-bag completion shape is invalid".format(location)
        )
    if reason.shape != expected_member_shape or reason.dtype.kind not in {
        "U",
        "S",
    }:
        raise ArtifactValidationError(
            "{} member-bag failure reasons are invalid".format(location)
        )
    reason_text = reason.astype(str)
    if np.any(reason_text[completion] != "") or np.any(
        reason_text[~completion] == ""
    ):
        raise ArtifactValidationError(
            "{} completion and failure reasons disagree".format(location)
        )

    raw_metrics = {}
    for metric in _PID_PHYSICAL_METRICS:
        key = "member_bag_{}".format(metric)
        values = _numeric_with_missing(
            arrays, key, expected_member_shape, location
        )
        if (
            np.any(~np.isfinite(values[completion]))
            or np.any(values[completion] < 0.0)
            or np.any(~np.isnan(values[~completion]))
        ):
            raise ArtifactValidationError(
                "{}:{} must be finite for completed forecasts and NaN for "
                "failures".format(location, key)
            )
        raw_metrics[metric] = values

    def configured_threshold(name: str) -> bool:
        configured = _scalar_bool(
            arrays, "{}_threshold_configured".format(name), location
        )
        threshold = _numeric_with_missing(
            arrays, "{}_threshold".format(name), (1,), location
        )[0]
        if configured:
            if not np.isfinite(threshold) or threshold <= 0.0:
                raise ArtifactValidationError(
                    "{} configured {} threshold must be positive".format(
                        location, name
                    )
                )
        elif not np.isnan(threshold):
            raise ArtifactValidationError(
                "{} unconfigured {} threshold must be NaN".format(
                    location, name
                )
            )
        return configured

    position_configured = configured_threshold("position")
    orientation_configured = configured_threshold("orientation")
    position_metric = _scalar_string(
        arrays, "position_threshold_metric", location
    )
    orientation_metric = _scalar_string(
        arrays, "orientation_threshold_metric", location
    )
    if position_metric not in {
        "position_rmse",
        "maximum_position_error",
    } or orientation_metric not in {
        "orientation_rmse",
        "maximum_orientation_error",
    }:
        raise ArtifactValidationError(
            "{} threshold metric names are invalid".format(location)
        )
    for name, configured in (
        ("position", position_configured),
        ("orientation", orientation_configured),
    ):
        values = _numeric_with_missing(
            arrays,
            "member_bag_{}_threshold_exceeded".format(name),
            expected_member_shape,
            location,
        )
        if configured:
            if np.any(~np.isin(values[completion], (0.0, 1.0))) or np.any(
                ~np.isnan(values[~completion])
            ):
                raise ArtifactValidationError(
                    "{} {} threshold flags are invalid".format(location, name)
                )
        elif np.any(~np.isnan(values)):
            raise ArtifactValidationError(
                "{} unconfigured {} threshold flags must be NaN".format(
                    location, name
                )
            )

    cvar_level = _finite_array(arrays, "cvar_level", (1,), location)[0]
    coverage_interval = _finite_array(
        arrays, "correction_coverage_interval", (1,), location
    )[0]
    if not 0.0 <= cvar_level < 1.0 or not 0.0 < coverage_interval < 1.0:
        raise ArtifactValidationError(
            "{} CVaR/coverage levels are invalid".format(location)
        )

    def upper_cvar(values: np.ndarray) -> float:
        ordered = np.sort(values)
        sample_count = ordered.size
        left = np.arange(sample_count, dtype=float) / sample_count
        right = np.arange(1, sample_count + 1, dtype=float) / sample_count
        mass = np.maximum(0.0, right - np.maximum(left, cvar_level))
        return float(np.dot(mass, ordered) / (1.0 - cvar_level))

    for metric, raw in raw_metrics.items():
        per_bag_mean = _numeric_with_missing(
            arrays, "per_bag_{}_mean".format(metric), (count, bags), location
        )
        per_bag_cvar = _numeric_with_missing(
            arrays,
            "per_bag_{}_upper_cvar".format(metric),
            (count, bags),
            location,
        )
        for candidate_index in range(count):
            for bag_index in range(bags):
                values = raw[candidate_index, bag_index][
                    completion[candidate_index, bag_index]
                ]
                expected_mean = np.nan if values.size == 0 else np.mean(values)
                expected_cvar = (
                    np.nan if values.size == 0 else upper_cvar(values)
                )
                if not np.isclose(
                    per_bag_mean[candidate_index, bag_index],
                    expected_mean,
                    equal_nan=True,
                ) or not np.isclose(
                    per_bag_cvar[candidate_index, bag_index],
                    expected_cvar,
                    equal_nan=True,
                ):
                    raise ArtifactValidationError(
                        "{} per-bag {} summaries do not match raw members".format(
                            location, metric
                        )
                    )
        for statistic, per_bag in (
            ("mean", per_bag_mean),
            ("upper_cvar", per_bag_cvar),
        ):
            aggregate = _numeric_with_missing(
                arrays,
                "aggregate_{}_{}".format(metric, statistic),
                (count,),
                location,
            )
            expected = np.asarray(
                [
                    (
                        np.nan
                        if not np.any(np.isfinite(row))
                        else np.mean(row[np.isfinite(row)])
                    )
                    for row in per_bag
                ]
            )
            if not np.allclose(aggregate, expected, equal_nan=True):
                raise ArtifactValidationError(
                    "{} aggregate {} {} is not bag-equal".format(
                        location, metric, statistic
                    )
                )

    per_bag_completion = _finite_array(
        arrays, "per_bag_forecast_completion", (count, bags), location
    )
    per_bag_failures = arrays["per_bag_numerical_failure_count"]
    numerical_failures = arrays["numerical_failure_count"]
    aggregate_completion = _finite_array(
        arrays, "forecast_completion", (count,), location
    )
    expected_per_bag_completion = np.mean(completion, axis=2)
    expected_per_bag_failures = np.count_nonzero(~completion, axis=2)
    if (
        per_bag_failures.shape != (count, bags)
        or not np.issubdtype(per_bag_failures.dtype, np.integer)
        or numerical_failures.shape != (count,)
        or not np.issubdtype(numerical_failures.dtype, np.integer)
        or not np.allclose(
            per_bag_completion, expected_per_bag_completion
        )
        or not np.array_equal(per_bag_failures, expected_per_bag_failures)
        or not np.allclose(
            aggregate_completion, np.mean(per_bag_completion, axis=1)
        )
        or not np.array_equal(
            numerical_failures, np.sum(per_bag_failures, axis=1)
        )
    ):
        raise ArtifactValidationError(
            "{} completion/failure summaries are inconsistent".format(location)
        )

    for name, configured in (
        ("position", position_configured),
        ("orientation", orientation_configured),
    ):
        raw = arrays["member_bag_{}_threshold_exceeded".format(name)]
        per_bag = _numeric_with_missing(
            arrays,
            "per_bag_{}_threshold_exceedance".format(name),
            (count, bags),
            location,
        )
        aggregate = _numeric_with_missing(
            arrays,
            "aggregate_{}_threshold_exceedance".format(name),
            (count,),
            location,
        )
        expected_per_bag = np.full((count, bags), np.nan)
        if configured:
            for candidate_index in range(count):
                for bag_index in range(bags):
                    selected = completion[candidate_index, bag_index]
                    if np.any(selected):
                        expected_per_bag[candidate_index, bag_index] = np.mean(
                            raw[candidate_index, bag_index][selected]
                        )
        expected_aggregate = np.asarray(
            [
                (
                    np.nan
                    if not np.any(np.isfinite(row))
                    else np.mean(row[np.isfinite(row)])
                )
                for row in expected_per_bag
            ]
        )
        if not np.allclose(per_bag, expected_per_bag, equal_nan=True) or not (
            np.allclose(aggregate, expected_aggregate, equal_nan=True)
        ):
            raise ArtifactValidationError(
                "{} {} threshold summaries are inconsistent".format(
                    location, name
                )
            )

    coverage_fields = (
        "per_bag_correction_translation_zero_coverage",
        "per_bag_correction_rotation_zero_coverage",
        "per_bag_correction_transform_zero_coverage",
    )
    any_completed = np.any(completion, axis=2)
    for key in coverage_fields:
        coverage = _numeric_with_missing(
            arrays, key, (count, bags), location
        )
        if (
            np.any(~np.isfinite(coverage[any_completed]))
            or np.any(coverage[any_completed] < 0.0)
            or np.any(coverage[any_completed] > 1.0)
            or np.any(~np.isnan(coverage[~any_completed]))
        ):
            raise ArtifactValidationError(
                "{}:{} coverage availability is invalid".format(location, key)
            )

    bool_fields = {}
    for key in (
        "pareto_dominated",
        "pareto_non_dominated",
        "improves_current",
        "candidate_eligible",
    ):
        value = arrays[key]
        if value.shape != (count,) or not np.issubdtype(value.dtype, np.bool_):
            raise ArtifactValidationError(
                "{}:{} must be a candidate boolean vector".format(
                    location, key
                )
            )
        bool_fields[key] = value
    if not np.array_equal(
        bool_fields["pareto_non_dominated"],
        ~bool_fields["pareto_dominated"],
    ) or np.any(
        bool_fields["candidate_eligible"]
        != (
            bool_fields["improves_current"]
            & bool_fields["pareto_non_dominated"]
            & (np.arange(count) > 0)
        )
    ):
        raise ArtifactValidationError(
            "{} Pareto/eligibility flags are inconsistent".format(location)
        )
    candidate_rejection = _string_vector(
        arrays, "candidate_rejection_reason", count, location
    )
    if any(
        (not candidate_rejection[index])
        == (not bool_fields["candidate_eligible"][index])
        for index in range(count)
    ):
        raise ArtifactValidationError(
            "{} candidate rejection reasons disagree with eligibility".format(
                location
            )
        )
    gain_change = arrays["log_gain_change"]
    if (
        gain_change.shape != (count,)
        or not np.issubdtype(gain_change.dtype, np.number)
        or np.any(np.isnan(gain_change))
        or np.any(gain_change < 0.0)
        or gain_change[0] != 0.0
    ):
        raise ArtifactValidationError(
            "{} log gain change is invalid".format(location)
        )

    selected = _scalar_string(arrays, "selected_candidate_id", location)
    recommended = _scalar_string(
        arrays, "recommended_candidate_id", location
    )
    recommendation = _scalar_bool(
        arrays, "recommendation_available", location
    )
    if selected and selected not in set(candidate):
        raise ArtifactValidationError(
            "{} selected candidate was not evaluated".format(location)
        )
    if recommendation:
        selected_index = int(np.flatnonzero(candidate == selected)[0])
        if recommended != selected or not bool_fields["candidate_eligible"][
            selected_index
        ]:
            raise ArtifactValidationError(
                "{} recommendation is not the eligible explicit selection".format(
                    location
                )
            )
    elif recommended:
        raise ArtifactValidationError(
            "{} unavailable recommendation must have an empty ID".format(
                location
            )
        )
    for key in ("rejection_reason", "improvement_rule", "scenario_assumption"):
        value = _scalar_string(arrays, key, location)
        if key != "rejection_reason" and not value:
            raise ArtifactValidationError(
                "{}:{} cannot be empty".format(location, key)
            )
    return candidate


def _validate_forecast_paths(
    arrays: Mapping[str, np.ndarray],
    candidates: int,
    members: int,
    samples: int,
    location: str,
) -> None:
    success = arrays["forecast_success"]
    reason = arrays["forecast_failure_reason"]
    if success.shape != (candidates, members) or not np.issubdtype(
        success.dtype, np.bool_
    ):
        raise ArtifactValidationError(
            "{}:forecast_success must have boolean shape (C, M)".format(
                location
            )
        )
    if reason.shape != success.shape or reason.dtype.kind not in {"U", "S"}:
        raise ArtifactValidationError(
            "{}:forecast_failure_reason must have string shape (C, M)".format(
                location
            )
        )
    paths = []
    for key, width in (
        ("prediction_position", 3),
        ("prediction_orientation_xyzw", 4),
        ("correction_translation", 3),
        ("correction_rotation_vector", 3),
        ("position_error", 3),
        ("orientation_error_rotation_vector", 3),
    ):
        value = arrays[key]
        if (
            value.shape != (candidates, members, samples, width)
            or not np.issubdtype(value.dtype, np.number)
        ):
            raise ArtifactValidationError(
                "{}:{} has invalid candidate/member/time shape".format(
                    location, key
                )
            )
        paths.append(value)
    reason_text = reason.astype(str)
    completed_orientation = arrays["prediction_orientation_xyzw"][success]
    if completed_orientation.size and not np.allclose(
        np.linalg.norm(completed_orientation, axis=2),
        1.0,
        atol=1.0e-6,
    ):
        raise ArtifactValidationError(
            "{} completed predictions must contain unit quaternions".format(
                location
            )
        )
    for candidate_index in range(candidates):
        for member_index in range(members):
            selected = [
                value[candidate_index, member_index] for value in paths
            ]
            finite = all(np.all(np.isfinite(value)) for value in selected)
            missing = all(np.all(np.isnan(value)) for value in selected)
            completed = bool(success[candidate_index, member_index])
            message = reason_text[candidate_index, member_index]
            if completed and (not finite or message):
                raise ArtifactValidationError(
                    "{} completed forecast must have finite paths and no "
                    "failure reason".format(location)
                )
            if not completed and (not missing or not message):
                raise ArtifactValidationError(
                    "{} failed forecast must have all-NaN paths and a "
                    "failure reason".format(location)
                )


def _validate_pid_bag(
    arrays: Mapping[str, np.ndarray],
    source_member_id: np.ndarray,
    candidate_id: np.ndarray,
    summary: Mapping[str, np.ndarray],
    bag_index: int,
    location: str,
) -> None:
    _require_keys(arrays, _PID_BAG_KEYS, location)
    _forbid_controller_mass(arrays, location)
    member_id = _member_ids(arrays, "member_id", location)
    _same_ids(source_member_id, member_id, location)
    candidates = _string_vector(
        arrays, "candidate_id", candidate_id.size, location
    )
    if not np.array_equal(candidates, candidate_id):
        raise ArtifactValidationError(
            "{} candidate_id order does not match summary".format(location)
        )
    times = _finite_array(arrays, "times", (None,), location)
    samples = times.size
    if samples < 2 or np.any(np.diff(times) <= 0.0):
        raise ArtifactValidationError(
            "{}:times must be strictly increasing".format(location)
        )
    _finite_array(arrays, "reference_position", (samples, 3), location)
    _finite_array(arrays, "reference_rpy", (samples, 3), location)
    candidates_count = candidate_id.size
    members = source_member_id.size
    _validate_forecast_paths(
        arrays,
        candidates_count,
        members,
        samples,
        location,
    )
    success = arrays["forecast_success"]
    reason = arrays["forecast_failure_reason"].astype(str)
    if not np.array_equal(
        success, summary["member_bag_forecast_completion"][:, bag_index]
    ) or not np.array_equal(
        reason, summary["member_bag_failure_reason"][:, bag_index].astype(str)
    ):
        raise ArtifactValidationError(
            "{} forecast status does not match summary".format(location)
        )
    if not np.allclose(
        arrays["position_error"],
        arrays["prediction_position"]
        - arrays["reference_position"][None, None, :, :],
        equal_nan=True,
    ):
        raise ArtifactValidationError(
            "{} position error does not match prediction/reference".format(
                location
            )
        )
    duration = times[-1] - times[0]
    metric_sources = {
        "position_rmse": np.sqrt(
            np.trapz(
                np.sum(arrays["position_error"] ** 2, axis=3),
                times,
                axis=2,
            )
            / duration
        ),
        "orientation_rmse": np.sqrt(
            np.trapz(
                np.sum(
                    arrays["orientation_error_rotation_vector"] ** 2,
                    axis=3,
                ),
                times,
                axis=2,
            )
            / duration
        ),
        "maximum_position_error": np.sqrt(
            np.max(np.sum(arrays["position_error"] ** 2, axis=3), axis=2)
        ),
        "maximum_orientation_error": np.sqrt(
            np.max(
                np.sum(
                    arrays["orientation_error_rotation_vector"] ** 2,
                    axis=3,
                ),
                axis=2,
            )
        ),
    }
    for metric, expected in metric_sources.items():
        values = _numeric_with_missing(
            arrays, metric, (candidates_count, members), location
        )
        if (
            np.any(~np.isfinite(values[success]))
            or np.any(values[success] < 0.0)
            or np.any(~np.isnan(values[~success]))
            or not np.allclose(values, expected, equal_nan=True)
            or not np.allclose(
                values,
                summary["member_bag_{}".format(metric)][:, bag_index],
                equal_nan=True,
            )
        ):
            raise ArtifactValidationError(
                "{}:{} metrics do not match raw paths/summary".format(
                    location, metric
                )
            )
    policies = _string_vector(
        arrays, "residual_policy", candidates_count, location
    )
    if any(value not in {"posterior_replay", "zero"} for value in policies) or (
        np.unique(policies).size != 1
    ):
        raise ArtifactValidationError(
            "{} residual policy must be one bag-wide declared policy".format(
                location
            )
        )
    coverage_interval = _finite_array(
        arrays, "correction_coverage_interval", (1,), location
    )[0]
    if not np.isclose(
        coverage_interval, summary["correction_coverage_interval"][0]
    ):
        raise ArtifactValidationError(
            "{} correction coverage interval does not match summary".format(
                location
            )
        )
    tail = 50.0 * (1.0 - coverage_interval)

    def expected_coverage(path: np.ndarray, candidate_index: int) -> float:
        completed = success[candidate_index]
        if not np.any(completed):
            return np.nan
        lower, upper = np.percentile(
            path[candidate_index, completed],
            (tail, 100.0 - tail),
            axis=0,
        )
        return float(np.mean((lower <= 0.0) & (upper >= 0.0)))

    expected_translation = np.asarray(
        [
            expected_coverage(arrays["correction_translation"], index)
            for index in range(candidates_count)
        ]
    )
    expected_rotation = np.asarray(
        [
            expected_coverage(arrays["correction_rotation_vector"], index)
            for index in range(candidates_count)
        ]
    )
    expected_transform = (
        expected_translation + expected_rotation
    ) / 2.0
    for key, expected, summary_key in (
        (
            "correction_translation_zero_coverage",
            expected_translation,
            "per_bag_correction_translation_zero_coverage",
        ),
        (
            "correction_rotation_zero_coverage",
            expected_rotation,
            "per_bag_correction_rotation_zero_coverage",
        ),
        (
            "correction_transform_zero_coverage",
            expected_transform,
            "per_bag_correction_transform_zero_coverage",
        ),
    ):
        values = _numeric_with_missing(
            arrays, key, (candidates_count,), location
        )
        if not np.allclose(values, expected, equal_nan=True) or not np.allclose(
            values, summary[summary_key][:, bag_index], equal_nan=True
        ):
            raise ArtifactValidationError(
                "{}:{} coverage does not match raw paths/summary".format(
                    location, key
                )
            )


def _validate_pid_manifest(
    manifest: Mapping[str, Any]
) -> Tuple[Tuple[str, ...], Mapping[str, Any]]:
    required = (
        "evaluation_id",
        "source_run_id",
        "created_at",
        "selected_bag_ids",
        "artifacts",
    )
    _require_keys(manifest, required, "manifest")
    for key in ("evaluation_id", "source_run_id", "created_at"):
        _required_string(manifest, key, "manifest")
    bag_ids = _string_list(
        manifest["selected_bag_ids"], "manifest.selected_bag_ids"
    )
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict):
        raise ArtifactValidationError("manifest.artifacts must be an object")
    _require_keys(
        artifacts,
        (
            "proposal_ensemble",
            "summary",
            "proposed_yaml",
            "proposed_diff_yaml",
            "bags",
        ),
        "artifacts",
    )
    if not isinstance(artifacts["bags"], dict) or set(
        artifacts["bags"]
    ) != set(bag_ids):
        raise ArtifactValidationError(
            "manifest.artifacts.bags must map every selected bag"
        )
    return bag_ids, artifacts


def _load_pid(
    root: Path, manifest: Mapping[str, Any]
) -> PidProposalEvaluationBundle:
    bag_ids, artifacts = _validate_pid_manifest(manifest)
    proposal_path = _artifact_path(
        root, artifacts["proposal_ensemble"], "artifacts.proposal_ensemble"
    )
    summary_path = _artifact_path(
        root, artifacts["summary"], "artifacts.summary"
    )
    yaml_path = _artifact_path(
        root, artifacts["proposed_yaml"], "artifacts.proposed_yaml"
    )
    diff_path = _artifact_path(
        root,
        artifacts["proposed_diff_yaml"],
        "artifacts.proposed_diff_yaml",
    )
    try:
        yaml_path.read_text(encoding="utf-8")
        diff_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactValidationError(
            "PID YAML artifacts must be readable UTF-8 text"
        ) from error
    proposal = load_npz_strict(proposal_path)
    member_id = _validate_pid_proposal(proposal, str(proposal_path))
    if _scalar_string(
        proposal, "source_run_id", str(proposal_path)
    ) != manifest["source_run_id"]:
        raise ArtifactValidationError(
            "proposal source_run_id does not match manifest"
        )
    summary = load_npz_strict(summary_path)
    candidate_id = _validate_pid_summary(
        summary,
        member_id,
        bag_ids,
        manifest["source_run_id"],
        str(summary_path),
    )
    if (
        not np.array_equal(summary["current_pid"], proposal["current_pid"])
        or not np.array_equal(
            summary["source_mode_id"], proposal["source_mode_id"]
        )
    ):
        raise ArtifactValidationError(
            "proposal and summary source laws are not aligned"
        )
    bags: Dict[str, Mapping[str, np.ndarray]] = {}
    for bag_index, bag_id in enumerate(bag_ids):
        path = _artifact_path(
            root,
            artifacts["bags"][bag_id],
            "artifacts.bags.{}".format(bag_id),
        )
        arrays = load_npz_strict(path)
        _validate_pid_bag(
            arrays,
            member_id,
            candidate_id,
            summary,
            bag_index,
            str(path),
        )
        bags[bag_id] = arrays
    return PidProposalEvaluationBundle(
        root,
        dict(manifest),
        proposal,
        summary,
        bags,
        yaml_path,
        diff_path,
    )


def _load_bundle_with_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    require_complete: bool,
) -> Bundle:
    schema = _validate_known_schema(manifest)
    _validate_status(manifest, require_complete)
    if schema == INSPECTION_BUNDLE_SCHEMA:
        return _load_inspection(root, manifest)
    if schema == ASSIMILATION_RUN_SCHEMA:
        return _load_run(root, manifest)
    return _load_pid(root, manifest)


def read_manifest(root: Union[str, Path]) -> Dict[str, Any]:
    """Read a bundle manifest without claiming that the bundle is complete."""

    bundle_root = Path(root).expanduser().resolve()
    manifest = _load_json(bundle_root / MANIFEST_NAME)
    _validate_known_schema(manifest)
    _validate_status(manifest, require_complete=False)
    return manifest


def load_bundle(root: Union[str, Path]) -> Bundle:
    """Strictly load one complete inspection, assimilation, or PID bundle."""

    bundle_root = Path(root).expanduser().resolve()
    manifest = _load_json(bundle_root / MANIFEST_NAME)
    return _load_bundle_with_manifest(
        bundle_root, manifest, require_complete=True
    )


def load_inspection_bundle(root: Union[str, Path]) -> InspectionBundle:
    bundle = load_bundle(root)
    if not isinstance(bundle, InspectionBundle):
        raise UnsupportedArtifactSchema("bundle is not an inspection bundle")
    return bundle


def load_assimilation_run(
    root: Union[str, Path]
) -> AssimilationRunBundle:
    bundle = load_bundle(root)
    if not isinstance(bundle, AssimilationRunBundle):
        raise UnsupportedArtifactSchema("bundle is not an assimilation run")
    return bundle


def load_pid_proposal_evaluation(
    root: Union[str, Path]
) -> PidProposalEvaluationBundle:
    bundle = load_bundle(root)
    if not isinstance(bundle, PidProposalEvaluationBundle):
        raise UnsupportedArtifactSchema(
            "bundle is not a PID proposal evaluation"
        )
    return bundle


def begin_bundle(
    root: Union[str, Path], manifest: Mapping[str, Any]
) -> Path:
    """Create an atomic writing manifest for a new bundle.

    Existing manifests are never overwritten.  Producers write all payloads
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
    """Validate every declared artifact, then atomically publish completion."""

    bundle_root = Path(root).expanduser().resolve()
    manifest = read_manifest(bundle_root)
    if manifest["status"] != WRITING_STATUS:
        raise ArtifactStateError(
            "only a writing bundle can become complete"
        )
    candidate = dict(manifest)
    if updates is not None:
        candidate.update(_normalise_json(updates, "manifest updates"))
    candidate["status"] = COMPLETE_STATUS
    _load_bundle_with_manifest(
        bundle_root, candidate, require_complete=True
    )
    return write_json_atomic(bundle_root / MANIFEST_NAME, candidate)


def mark_bundle_cancelled(
    root: Union[str, Path],
    reason: str,
    updates: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically make cancellation authoritative without loading partial data."""

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
    if candidate.get("schema") == ASSIMILATION_RUN_SCHEMA:
        candidate["termination_reason"] = "cancelled"
        candidate["converged"] = False
    return write_json_atomic(bundle_root / MANIFEST_NAME, candidate)


__all__ = [
    "ASSIMILATION_RUN_SCHEMA",
    "ArtifactStateError",
    "ArtifactValidationError",
    "AssimilationRunBundle",
    "CANCELLED_STATUS",
    "COMPLETE_STATUS",
    "FLIGHT_INSPECTION_SCHEMA",
    "INSPECTION_BUNDLE_SCHEMA",
    "IncompleteArtifactError",
    "InspectionBundle",
    "MANIFEST_NAME",
    "PID_PROPOSAL_EVALUATION_SCHEMA",
    "PidProposalEvaluationBundle",
    "UnsupportedArtifactSchema",
    "WRITING_STATUS",
    "begin_bundle",
    "canonical_json_bytes",
    "load_assimilation_run",
    "load_bundle",
    "load_inspection_bundle",
    "load_npz_strict",
    "load_pid_proposal_evaluation",
    "mark_bundle_cancelled",
    "mark_bundle_complete",
    "read_json",
    "read_manifest",
    "request_fingerprint",
    "request_fingerprint_file",
    "write_json_atomic",
    "write_npz_atomic",
]
