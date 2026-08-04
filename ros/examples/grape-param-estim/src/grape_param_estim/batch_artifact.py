"""Strict v1 artifacts for sparse batch estimation runs.

This module deliberately recognises exactly one estimation schema.  It does
not dispatch to, translate, or otherwise interpret the former ensemble and
stage artifacts.  A run is loadable only after its manifest says
``status == "complete"`` and every referenced file passes both its SHA-256
integrity check and its array contract.

The Q convention is intentionally data, not policy: the manifest must spell
out the residual quantity, component order, and units.  In particular, this
loader does not choose between wrench and acceleration discrepancy models.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import re
import shutil
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .artifact_io import (
    ArtifactValidationError,
    IncompleteArtifactError,
    UnsupportedArtifactSchema,
    load_npz_strict,
    read_json,
    write_json_atomic,
    write_npz_atomic,
)
from .posterior import representatives as _posterior_representatives


BATCH_ESTIMATION_RUN_SCHEMA = "grape-param-estim/batch-estimation-run/v1"
STATIC_PARAMETER_DIMENSION = 18
DYNAMICS_RESIDUAL_DIMENSION = 6
COMPLETE_STATUS = "complete"
WRITING_STATUS = "writing"

_BAG_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

_CONDITIONAL_TRAJECTORY_EVALUATION_METHOD = (
    _posterior_representatives.CONDITIONAL_TRAJECTORY_EVALUATION_METHOD
)
_CONDITIONAL_TRAJECTORY_SAMPLE_ORDER = (
    _posterior_representatives.CONDITIONAL_TRAJECTORY_SAMPLE_ORDER
)
_CONDITIONAL_TRAJECTORY_SELECTION_POLICY = (
    _posterior_representatives.CONDITIONAL_TRAJECTORY_SELECTION_POLICY
)
_CONDITIONAL_TRAJECTORY_WARM_START_POLICY = (
    _posterior_representatives.CONDITIONAL_TRAJECTORY_WARM_START_POLICY
)
_CONDITIONAL_TRAJECTORY_FEATURE_POLICY = (
    _posterior_representatives.FEATURE_POLICY
)
_CONDITIONAL_TRAJECTORY_RIDGE_POLICY = (
    _posterior_representatives.RIDGE_COORDINATE_POLICY
)
_CONDITIONAL_TRAJECTORY_ROLE_PRIORITY = _posterior_representatives.ROLE_PRIORITY
_CONDITIONAL_TRAJECTORY_ROLE_RECORD_KEYS = (
    _posterior_representatives.ROLE_RECORD_KEYS
)
_CONDITIONAL_TRAJECTORY_SELECTION_KEYS = (
    _posterior_representatives.SELECTION_MANIFEST_KEYS
)
POSTERIOR_REPRESENTATIVE_MAXIMUM_SAMPLE_COUNT = (
    _posterior_representatives.POSTERIOR_REPRESENTATIVE_MAXIMUM_SAMPLE_COUNT
)
RepresentativeRole = _posterior_representatives.RepresentativeRole
select_posterior_representatives_from_arrays = (
    _posterior_representatives.select_posterior_representatives_from_arrays
)

_MANIFEST_KEYS = (
    "schema",
    "status",
    "run_id",
    "estimator_revision",
    "selected_bag_ids",
    "selected_intervals",
    "selected_bag_sha256",
    "configuration_fingerprint",
    "controller_snapshot_fingerprint",
    "sensor_contracts",
    "observation_factors",
    "parameter_prior",
    "delay_prior",
    "actuator_model",
    "q_definition",
    "knot_policy",
    "interpolation_policy",
    "solver_settings",
    "em_settings",
    "mcmc_settings",
    "request_fingerprint",
    "substage_status",
    "warnings",
    "artifacts",
)
_WRITER_METADATA_KEYS = tuple(
    key for key in _MANIFEST_KEYS if key not in {"schema", "status", "artifacts"}
)


@dataclass(frozen=True)
class BatchEstimationRun:
    """A validated, detached sparse-batch run."""

    root: Path
    manifest: Mapping[str, Any]
    map_static: Mapping[str, np.ndarray]
    q_em: Mapping[str, np.ndarray]
    laplace: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, np.ndarray]
    bags: Mapping[str, Mapping[str, np.ndarray]]
    mcmc_samples: Optional[Mapping[str, np.ndarray]]
    trajectories: Mapping[str, Mapping[str, np.ndarray]]


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


def _reject_unknown_keys(
    mapping: Mapping[str, Any], keys: Sequence[str], location: str
) -> None:
    allowed = set(keys)
    unknown = sorted(key for key in mapping if key not in allowed)
    if unknown:
        raise ArtifactValidationError(
            "{} has unknown keys: {}".format(location, ", ".join(unknown))
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


def _required_mapping(
    mapping: Mapping[str, Any], key: str, location: str
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(
            "{}.{} must be an object".format(location, key)
        )
    return value


def _string_list(
    value: Any,
    location: str,
    nonempty: bool = True,
    unique: bool = True,
) -> Tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ArtifactValidationError(
            "{} must be a list of non-empty strings".format(location)
        )
    result = tuple(value)
    if nonempty and not result:
        raise ArtifactValidationError("{} must not be empty".format(location))
    if unique and len(set(result)) != len(result):
        raise ArtifactValidationError(
            "{} must not contain duplicates".format(location)
        )
    return result


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ArtifactValidationError(
            "{} must have form sha256:<64 lowercase hex digits>".format(
                location
            )
        )
    return value


def _safe_bag_id(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or value in {".", ".."}
        or _BAG_ID_PATTERN.fullmatch(value) is None
    ):
        raise ArtifactValidationError(
            "{} is not a safe bag ID".format(location)
        )
    return value


def _interval(value: Any, location: str) -> Tuple[float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or isinstance(value[0], bool)
        or isinstance(value[1], bool)
    ):
        raise ArtifactValidationError(
            "{} must contain [start, end]".format(location)
        )
    try:
        start = float(value[0])
        end = float(value[1])
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError(
            "{} must contain numeric bounds".format(location)
        ) from error
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ArtifactValidationError(
            "{} must contain finite increasing bounds".format(location)
        )
    return start, end


def _artifact_path(root: Path, relative: Any, location: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ArtifactValidationError(
            "{} must be a non-empty relative path".format(location)
        )
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ArtifactValidationError(
            "{} must stay inside the run directory".format(location)
        )
    root_resolved = root.resolve()
    candidate = (root_resolved / relative_path).resolve()
    if root_resolved not in candidate.parents:
        raise ArtifactValidationError(
            "{} resolves outside the run directory".format(location)
        )
    if not candidate.is_file():
        raise ArtifactValidationError(
            "{} does not name an existing file: {}".format(
                location, relative
            )
        )
    return candidate


def file_sha256(path: Union[str, Path]) -> str:
    """Return an algorithm-labelled digest without loading a file in memory."""

    digest = hashlib.sha256()
    source = Path(path)
    try:
        with source.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise ArtifactValidationError(
            "cannot hash artifact {}: {}".format(source, error)
        ) from error
    return "sha256:{}".format(digest.hexdigest())


def _artifact_descriptor(
    root: Path,
    value: Any,
    expected_path: str,
    location: str,
) -> Path:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError("{} must be an object".format(location))
    _require_keys(value, ("path", "sha256"), location)
    _reject_unknown_keys(value, ("path", "sha256"), location)
    relative = _required_string(value, "path", location)
    if relative != expected_path:
        raise ArtifactValidationError(
            "{}.path must be {!r}".format(location, expected_path)
        )
    expected_digest = _sha256(value["sha256"], "{}.sha256".format(location))
    path = _artifact_path(root, relative, "{}.path".format(location))
    actual_digest = file_sha256(path)
    if actual_digest != expected_digest:
        raise ArtifactValidationError(
            "{} SHA-256 mismatch: manifest has {}, file has {}".format(
                location, expected_digest, actual_digest
            )
        )
    return path


def _array(
    arrays: Mapping[str, np.ndarray],
    key: str,
    shape: Tuple[Optional[int], ...],
    location: str,
    kind: str = "numeric",
    finite: bool = True,
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
    if kind == "numeric":
        valid_dtype = np.issubdtype(value.dtype, np.number) and not np.issubdtype(
            value.dtype, np.bool_
        )
    elif kind == "integer":
        valid_dtype = np.issubdtype(value.dtype, np.integer) and not np.issubdtype(
            value.dtype, np.bool_
        )
    elif kind == "boolean":
        valid_dtype = np.issubdtype(value.dtype, np.bool_)
    elif kind == "string":
        valid_dtype = value.dtype.kind in {"U", "S"}
    else:  # pragma: no cover - private programming error
        raise AssertionError("unknown array kind {!r}".format(kind))
    if not valid_dtype:
        raise ArtifactValidationError(
            "{}:{} has invalid {} dtype {}".format(
                location, key, kind, value.dtype
            )
        )
    if finite and kind == "numeric" and np.any(~np.isfinite(value)):
        raise ArtifactValidationError(
            "{}:{} must contain only finite values".format(location, key)
        )
    return value


def _strings(
    arrays: Mapping[str, np.ndarray],
    key: str,
    size: Optional[int],
    location: str,
    unique: bool = False,
) -> np.ndarray:
    value = _array(arrays, key, (size,), location, kind="string")
    decoded = np.asarray(
        [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in value.tolist()
        ]
    )
    if any(not item for item in decoded.tolist()):
        raise ArtifactValidationError(
            "{}:{} contains an empty string".format(location, key)
        )
    if unique and np.unique(decoded).size != decoded.size:
        raise ArtifactValidationError(
            "{}:{} contains duplicates".format(location, key)
        )
    return decoded


def _strictly_increasing(value: np.ndarray, location: str) -> None:
    if value.size > 1 and np.any(np.diff(value) <= 0.0):
        raise ArtifactValidationError(
            "{} must be strictly increasing".format(location)
        )


def _unit_quaternions(value: np.ndarray, location: str) -> None:
    if value.shape[0] and not np.allclose(
        np.linalg.norm(value, axis=-1), 1.0, rtol=1.0e-7, atol=1.0e-9
    ):
        raise ArtifactValidationError(
            "{} must contain unit quaternions".format(location)
        )


def _positive(value: np.ndarray, location: str) -> None:
    if np.any(value <= 0.0):
        raise ArtifactValidationError("{} must be positive".format(location))


def _nonnegative(value: np.ndarray, location: str) -> None:
    if np.any(value < 0.0):
        raise ArtifactValidationError(
            "{} must be non-negative".format(location)
        )


def _positive_semidefinite(value: np.ndarray, location: str) -> None:
    if value.shape[0] == 0:
        return
    eigenvalues = np.linalg.eigvalsh(value)
    scale = np.maximum(1.0, np.max(np.abs(eigenvalues), axis=-1))
    if np.any(np.min(eigenvalues, axis=-1) < -1.0e-10 * scale):
        raise ArtifactValidationError(
            "{} must be positive semidefinite".format(location)
        )


def _id_vector(
    arrays: Mapping[str, np.ndarray], key: str, location: str
) -> np.ndarray:
    value = arrays[key]
    if value.ndim != 1 or value.size == 0:
        raise ArtifactValidationError(
            "{}:{} must be a non-empty vector".format(location, key)
        )
    if np.issubdtype(value.dtype, np.integer) and not np.issubdtype(
        value.dtype, np.bool_
    ):
        normalized = value.astype(np.int64, copy=False)
    elif value.dtype.kind in {"U", "S"}:
        normalized = _strings(arrays, key, value.size, location)
    else:
        raise ArtifactValidationError(
            "{}:{} must contain integer or string IDs".format(location, key)
        )
    if np.unique(normalized).size != normalized.size:
        raise ArtifactValidationError(
            "{}:{} contains duplicate IDs".format(location, key)
        )
    return normalized


def _validate_conditional_trajectory_selection(
    mcmc_settings: Mapping[str, Any],
    bag_ids: Tuple[str, ...],
    trajectory_bag_ids: Tuple[str, ...],
) -> Optional[Mapping[str, Any]]:
    """Validate manifest provenance for any stored trajectory subset."""

    selection = mcmc_settings.get("conditional_trajectory_selection")
    if not trajectory_bag_ids:
        if selection is not None:
            raise ArtifactValidationError(
                "manifest trajectory selection requires trajectory artifacts"
            )
        return None
    if not isinstance(selection, Mapping):
        raise ArtifactValidationError(
            "manifest.mcmc_settings.conditional_trajectory_selection is required"
        )
    location = "manifest.mcmc_settings.conditional_trajectory_selection"
    _require_keys(selection, _CONDITIONAL_TRAJECTORY_SELECTION_KEYS, location)
    _reject_unknown_keys(
        selection, _CONDITIONAL_TRAJECTORY_SELECTION_KEYS, location
    )
    expected_literals = {
        "policy": _CONDITIONAL_TRAJECTORY_SELECTION_POLICY,
        "sample_order": _CONDITIONAL_TRAJECTORY_SAMPLE_ORDER,
        "conditional_evaluation_method": (
            _CONDITIONAL_TRAJECTORY_EVALUATION_METHOD
        ),
        "warm_start_policy": _CONDITIONAL_TRAJECTORY_WARM_START_POLICY,
    }
    for name, expected in expected_literals.items():
        if selection[name] != expected:
            raise ArtifactValidationError(
                "{}.{} must be {!r}".format(location, name, expected)
            )
    fixed_policies = {
        "role_priority": list(_CONDITIONAL_TRAJECTORY_ROLE_PRIORITY),
        "feature_policy": dict(_CONDITIONAL_TRAJECTORY_FEATURE_POLICY),
        "ridge_coordinate_policy": dict(
            _CONDITIONAL_TRAJECTORY_RIDGE_POLICY
        ),
    }
    for name, expected in fixed_policies.items():
        if selection[name] != expected:
            raise ArtifactValidationError(
                "{}.{} does not match role-union v2".format(location, name)
            )
    available = selection["available_sample_count"]
    if (
        isinstance(available, bool)
        or not isinstance(available, (int, np.integer))
        or int(available) <= 0
    ):
        raise ArtifactValidationError(
            location + ".available_sample_count must be a positive integer"
        )
    maximum = selection["maximum_sample_count"]
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, np.integer))
        or int(maximum)
        != POSTERIOR_REPRESENTATIVE_MAXIMUM_SAMPLE_COUNT
    ):
        raise ArtifactValidationError(
            location + ".maximum_sample_count must be the fixed v2 bound"
        )
    raw_sample_ids = selection["selected_sample_ids"]
    if (
        not isinstance(raw_sample_ids, list)
        or not raw_sample_ids
        or any(
            type(value) is not str
            or not value
            or value.strip() != value
            for value in raw_sample_ids
        )
    ):
        raise ArtifactValidationError(
            location
            + ".selected_sample_ids must be a non-empty canonical string list"
        )
    sample_ids = tuple(raw_sample_ids)
    if len(set(sample_ids)) != len(sample_ids):
        raise ArtifactValidationError(
            location + ".selected_sample_ids must not contain duplicates"
        )
    selected_bags = _string_list(
        selection["selected_bag_ids"],
        location + ".selected_bag_ids",
    )
    if selected_bags != bag_ids or trajectory_bag_ids != bag_ids:
        raise ArtifactValidationError(
            "conditional trajectory selection must cover every selected bag in order"
        )
    if len(sample_ids) > min(int(available), int(maximum)):
        raise ArtifactValidationError(
            "conditional trajectory selection exceeds its recorded bound"
        )
    raw_roles = selection["role_records"]
    if not isinstance(raw_roles, list) or not raw_roles:
        raise ArtifactValidationError(
            location + ".role_records must be a non-empty list"
        )
    roles = []
    for index, raw_role in enumerate(raw_roles):
        role_location = "{}.role_records[{}]".format(location, index)
        if not isinstance(raw_role, Mapping):
            raise ArtifactValidationError(
                role_location + " must be an object"
            )
        _require_keys(
            raw_role, _CONDITIONAL_TRAJECTORY_ROLE_RECORD_KEYS, role_location
        )
        _reject_unknown_keys(
            raw_role, _CONDITIONAL_TRAJECTORY_ROLE_RECORD_KEYS, role_location
        )
        try:
            role = RepresentativeRole(**dict(raw_role))
        except (TypeError, ValueError) as error:
            raise ArtifactValidationError(
                role_location + " is not a canonical v2 role record"
            ) from error
        if dict(role.manifest_payload) != dict(raw_role):
            raise ArtifactValidationError(
                role_location + " is not in canonical form"
            )
        if role.sample_id not in sample_ids:
            raise ArtifactValidationError(
                role_location + " references an unselected sample"
            )
        roles.append(role)
    role_ids = tuple(value.role_id for value in roles)
    if len(set(role_ids)) != len(role_ids):
        raise ArtifactValidationError(
            location + ".role_records contains duplicate role IDs"
        )
    if (
        roles[0].role_class != "highest_log_posterior"
        or roles[0].sample_id != sample_ids[0]
    ):
        raise ArtifactValidationError(
            location + " must place the primary role and sample first"
        )
    return selection


def _validate_manifest(
    root: Path, manifest: Mapping[str, Any]
) -> Tuple[
    Tuple[str, ...],
    Mapping[str, Path],
    Mapping[str, Path],
    Optional[Path],
    Mapping[str, Path],
]:
    schema = manifest.get("schema")
    if schema != BATCH_ESTIMATION_RUN_SCHEMA:
        raise UnsupportedArtifactSchema(
            "unsupported artifact schema {!r}; expected {!r}".format(
                schema, BATCH_ESTIMATION_RUN_SCHEMA
            )
        )
    status = manifest.get("status")
    if status != COMPLETE_STATUS:
        raise IncompleteArtifactError(
            "batch run status is {!r}; only complete runs are loadable".format(
                status
            )
        )
    _require_keys(manifest, _MANIFEST_KEYS, "manifest")
    _reject_unknown_keys(manifest, _MANIFEST_KEYS, "manifest")
    _required_string(manifest, "run_id", "manifest")
    _required_string(manifest, "estimator_revision", "manifest")
    bag_ids = _string_list(manifest["selected_bag_ids"], "manifest.selected_bag_ids")
    for index, bag_id in enumerate(bag_ids):
        _safe_bag_id(bag_id, "manifest.selected_bag_ids[{}]".format(index))

    intervals = _required_mapping(manifest, "selected_intervals", "manifest")
    bag_hashes = _required_mapping(manifest, "selected_bag_sha256", "manifest")
    sensor_contracts = _required_mapping(manifest, "sensor_contracts", "manifest")
    observation_factors = _required_mapping(
        manifest, "observation_factors", "manifest"
    )
    expected_bags = set(bag_ids)
    for name, values in (
        ("selected_intervals", intervals),
        ("selected_bag_sha256", bag_hashes),
        ("sensor_contracts", sensor_contracts),
        ("observation_factors", observation_factors),
    ):
        if set(values) != expected_bags:
            raise ArtifactValidationError(
                "manifest.{} keys must exactly match selected_bag_ids".format(
                    name
                )
            )
    for bag_id in bag_ids:
        _interval(intervals[bag_id], "manifest.selected_intervals.{}".format(bag_id))
        _sha256(
            bag_hashes[bag_id],
            "manifest.selected_bag_sha256.{}".format(bag_id),
        )
        if (
            not isinstance(sensor_contracts[bag_id], Mapping)
            or not sensor_contracts[bag_id]
        ):
            raise ArtifactValidationError(
                "manifest.sensor_contracts.{} must be a non-empty object".format(
                    bag_id
                )
            )
        factors = observation_factors[bag_id]
        if not isinstance(factors, Mapping) or not factors:
            raise ArtifactValidationError(
                "manifest.observation_factors.{} must be a non-empty object".format(
                    bag_id
                )
            )
        for factor_name, factor in factors.items():
            if not isinstance(factor_name, str) or not factor_name:
                raise ArtifactValidationError(
                    "manifest.observation_factors.{} has an invalid factor name".format(
                        bag_id
                    )
                )
            location = "manifest.observation_factors.{}.{}".format(
                bag_id, factor_name
            )
            if not isinstance(factor, Mapping):
                raise ArtifactValidationError("{} must be an object".format(location))
            _require_keys(factor, ("enabled", "disabled_reason"), location)
            _reject_unknown_keys(
                factor, ("enabled", "disabled_reason"), location
            )
            enabled = factor["enabled"]
            reason = factor["disabled_reason"]
            if not isinstance(enabled, bool):
                raise ArtifactValidationError(
                    "{}.enabled must be boolean".format(location)
                )
            if not enabled and (not isinstance(reason, str) or not reason):
                raise ArtifactValidationError(
                    "{}.disabled_reason must explain a disabled factor".format(
                        location
                    )
                )
            if enabled and reason is not None:
                raise ArtifactValidationError(
                    "{}.disabled_reason must be null for an enabled factor".format(
                        location
                    )
                )

    for key in (
        "configuration_fingerprint",
        "controller_snapshot_fingerprint",
        "request_fingerprint",
    ):
        _sha256(manifest[key], "manifest.{}".format(key))
    for key in (
        "parameter_prior",
        "delay_prior",
        "actuator_model",
        "knot_policy",
        "interpolation_policy",
        "solver_settings",
        "em_settings",
        "mcmc_settings",
    ):
        value = _required_mapping(manifest, key, "manifest")
        if not value:
            raise ArtifactValidationError(
                "manifest.{} must be explicit and non-empty".format(key)
            )

    actuator_model = _required_mapping(manifest, "actuator_model", "manifest")
    actuator_keys = (
        "source",
        "thrust_time_constant_seconds",
        "gimbal_time_constant_seconds",
        "minimum_thrust_newtons",
        "maximum_thrust_newtons",
        "maximum_gimbal_angle_radians",
        "maximum_gimbal_rate_radians_per_second",
    )
    _require_keys(actuator_model, actuator_keys, "manifest.actuator_model")
    _reject_unknown_keys(
        actuator_model, actuator_keys, "manifest.actuator_model"
    )
    _required_string(actuator_model, "source", "manifest.actuator_model")
    actuator_values = {}
    for key in actuator_keys[1:]:
        value = actuator_model[key]
        if isinstance(value, bool):
            raise ArtifactValidationError(
                "manifest.actuator_model.{} must be numeric".format(key)
            )
        try:
            selected = float(value)
        except (TypeError, ValueError) as error:
            raise ArtifactValidationError(
                "manifest.actuator_model.{} must be numeric".format(key)
            ) from error
        if not np.isfinite(selected):
            raise ArtifactValidationError(
                "manifest.actuator_model.{} must be finite".format(key)
            )
        actuator_values[key] = selected
    for key in (
        "thrust_time_constant_seconds",
        "gimbal_time_constant_seconds",
        "maximum_gimbal_angle_radians",
        "maximum_gimbal_rate_radians_per_second",
    ):
        if actuator_values[key] <= 0.0:
            raise ArtifactValidationError(
                "manifest.actuator_model.{} must be positive".format(key)
            )
    if actuator_values["minimum_thrust_newtons"] < 0.0:
        raise ArtifactValidationError(
            "manifest.actuator_model.minimum_thrust_newtons cannot be negative"
        )
    if (
        actuator_values["maximum_thrust_newtons"]
        <= actuator_values["minimum_thrust_newtons"]
    ):
        raise ArtifactValidationError(
            "manifest.actuator_model thrust bounds are inconsistent"
        )

    q_definition = _required_mapping(manifest, "q_definition", "manifest")
    _require_keys(
        q_definition,
        ("definition", "components", "units"),
        "manifest.q_definition",
    )
    _reject_unknown_keys(
        q_definition,
        ("definition", "components", "units"),
        "manifest.q_definition",
    )
    _required_string(q_definition, "definition", "manifest.q_definition")
    components = _string_list(
        q_definition["components"], "manifest.q_definition.components"
    )
    units = _string_list(
        q_definition["units"],
        "manifest.q_definition.units",
        nonempty=True,
        unique=False,
    )
    if len(components) != DYNAMICS_RESIDUAL_DIMENSION:
        raise ArtifactValidationError(
            "manifest.q_definition.components must have length {}".format(
                DYNAMICS_RESIDUAL_DIMENSION
            )
        )
    if len(units) != DYNAMICS_RESIDUAL_DIMENSION:
        raise ArtifactValidationError(
            "manifest.q_definition.units must have length {}".format(
                DYNAMICS_RESIDUAL_DIMENSION
            )
        )

    substages = _required_mapping(manifest, "substage_status", "manifest")
    required_stages = {"map", "laplace_em", "laplace"}
    if not required_stages.issubset(set(substages)):
        raise ArtifactValidationError(
            "manifest.substage_status must include map, laplace_em, and laplace"
        )
    for name, stage in substages.items():
        if not isinstance(name, str) or not name or not isinstance(stage, Mapping):
            raise ArtifactValidationError(
                "manifest.substage_status entries must be named objects"
            )
        location = "manifest.substage_status.{}".format(name)
        _require_keys(stage, ("converged", "termination_reason"), location)
        _reject_unknown_keys(
            stage, ("converged", "termination_reason"), location
        )
        if not isinstance(stage["converged"], bool):
            raise ArtifactValidationError(
                "{}.converged must be boolean".format(location)
            )
        _required_string(stage, "termination_reason", location)

    warnings = manifest["warnings"]
    if not isinstance(warnings, list) or any(
        not isinstance(item, str) or not item for item in warnings
    ):
        raise ArtifactValidationError(
            "manifest.warnings must be a list of non-empty strings"
        )

    mcmc_settings = manifest["mcmc_settings"]
    if "enabled" not in mcmc_settings or not isinstance(
        mcmc_settings["enabled"], bool
    ):
        raise ArtifactValidationError(
            "manifest.mcmc_settings.enabled must be explicit boolean"
        )
    mcmc_enabled = bool(mcmc_settings["enabled"])
    if mcmc_enabled and "mcmc" not in substages:
        raise ArtifactValidationError(
            "manifest.substage_status.mcmc is required when MCMC is enabled"
        )

    artifacts = _required_mapping(manifest, "artifacts", "manifest")
    allowed_artifacts = {
        "map_static",
        "q_em",
        "laplace",
        "diagnostics",
        "bags",
        "mcmc_samples",
        "trajectories",
    }
    _require_keys(
        artifacts,
        ("map_static", "q_em", "laplace", "diagnostics", "bags"),
        "manifest.artifacts",
    )
    _reject_unknown_keys(artifacts, tuple(allowed_artifacts), "manifest.artifacts")

    core_paths: Dict[str, Path] = {}
    for name in ("map_static", "q_em", "laplace", "diagnostics"):
        core_paths[name] = _artifact_descriptor(
            root,
            artifacts[name],
            "{}.npz".format(name),
            "manifest.artifacts.{}".format(name),
        )

    bag_artifacts = artifacts["bags"]
    if not isinstance(bag_artifacts, Mapping) or set(bag_artifacts) != expected_bags:
        raise ArtifactValidationError(
            "manifest.artifacts.bags keys must exactly match selected_bag_ids"
        )
    bag_paths: Dict[str, Path] = {}
    for bag_id in bag_ids:
        bag_paths[bag_id] = _artifact_descriptor(
            root,
            bag_artifacts[bag_id],
            "bags/{}.npz".format(bag_id),
            "manifest.artifacts.bags.{}".format(bag_id),
        )

    mcmc_path: Optional[Path] = None
    if mcmc_enabled:
        if "mcmc_samples" not in artifacts:
            raise ArtifactValidationError(
                "manifest.artifacts.mcmc_samples is required when MCMC is enabled"
            )
        mcmc_path = _artifact_descriptor(
            root,
            artifacts["mcmc_samples"],
            "mcmc_samples.npz",
            "manifest.artifacts.mcmc_samples",
        )
    elif "mcmc_samples" in artifacts:
        raise ArtifactValidationError(
            "manifest.artifacts.mcmc_samples is forbidden when MCMC is disabled"
        )

    trajectory_paths: Dict[str, Path] = {}
    if "trajectories" in artifacts:
        if not mcmc_enabled:
            raise ArtifactValidationError(
                "trajectory subsets require enabled MCMC"
            )
        trajectories = artifacts["trajectories"]
        if not isinstance(trajectories, Mapping):
            raise ArtifactValidationError(
                "manifest.artifacts.trajectories must be an object"
            )
        if not set(trajectories).issubset(expected_bags):
            raise ArtifactValidationError(
                "manifest.artifacts.trajectories contains an unknown bag ID"
            )
        for bag_id, descriptor in trajectories.items():
            trajectory_paths[bag_id] = _artifact_descriptor(
                root,
                descriptor,
                "trajectories/{}/selected_samples.npz".format(bag_id),
                "manifest.artifacts.trajectories.{}".format(bag_id),
            )
    if mcmc_enabled and set(trajectory_paths) != expected_bags:
        raise ArtifactValidationError(
            "completed MCMC requires selected trajectories for every bag"
        )

    _validate_conditional_trajectory_selection(
        mcmc_settings, bag_ids, tuple(trajectory_paths)
    )

    return bag_ids, core_paths, bag_paths, mcmc_path, trajectory_paths


_MAP_STATIC_KEYS = (
    "parameter_coordinate_map",
    "mass",
    "inertia",
    "cog",
    "force_effectiveness",
    "torque_effectiveness",
    "delay",
    "q_diagonal",
    "objective_component_names",
    "objective_component_values",
    "prior_objective",
    "likelihood_objective",
    "bag_id",
    "bag_objective",
)


def _validate_map_static(
    arrays: Mapping[str, np.ndarray], bag_ids: Tuple[str, ...], location: str
) -> None:
    _require_keys(arrays, _MAP_STATIC_KEYS, location)
    _reject_unknown_keys(arrays, _MAP_STATIC_KEYS, location)
    _array(
        arrays,
        "parameter_coordinate_map",
        (STATIC_PARAMETER_DIMENSION,),
        location,
    )
    mass = _array(arrays, "mass", (1,), location)
    _positive(mass, "{}:mass".format(location))
    inertia = _array(arrays, "inertia", (3, 3), location)
    if not np.allclose(inertia, inertia.T, rtol=1.0e-10, atol=1.0e-12):
        raise ArtifactValidationError("{}:inertia must be symmetric".format(location))
    if np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
        raise ArtifactValidationError(
            "{}:inertia must be positive definite".format(location)
        )
    _array(arrays, "cog", (3,), location)
    force = _array(arrays, "force_effectiveness", (4,), location)
    torque = _array(arrays, "torque_effectiveness", (4,), location)
    _positive(force, "{}:force_effectiveness".format(location))
    _positive(torque, "{}:torque_effectiveness".format(location))
    delay = _array(arrays, "delay", (1,), location)
    _nonnegative(delay, "{}:delay".format(location))
    q = _array(arrays, "q_diagonal", (DYNAMICS_RESIDUAL_DIMENSION,), location)
    _positive(q, "{}:q_diagonal".format(location))
    component_names = _strings(
        arrays, "objective_component_names", None, location, unique=True
    )
    if component_names.size == 0:
        raise ArtifactValidationError(
            "{}:objective_component_names must not be empty".format(location)
        )
    _array(
        arrays,
        "objective_component_values",
        (component_names.size,),
        location,
    )
    _array(arrays, "prior_objective", (1,), location)
    _array(arrays, "likelihood_objective", (1,), location)
    actual_bags = _strings(arrays, "bag_id", len(bag_ids), location, unique=True)
    if tuple(actual_bags.tolist()) != bag_ids:
        raise ArtifactValidationError(
            "{}:bag_id must match selected_bag_ids in order".format(location)
        )
    _array(arrays, "bag_objective", (len(bag_ids),), location)


_Q_EM_KEYS = (
    "iteration",
    "input_q",
    "target_q",
    "accepted_q",
    "alpha",
    "log_q_change",
    "map_objective",
    "approximate_marginal_objective",
    "lag",
    "accepted",
    "reason",
    "floor_activation",
    "expected_residual_second_moment",
    "map_residual_second_moment",
    "covariance_correction",
)


def _validate_q_em(arrays: Mapping[str, np.ndarray], location: str) -> None:
    _require_keys(arrays, _Q_EM_KEYS, location)
    _reject_unknown_keys(arrays, _Q_EM_KEYS, location)
    iteration = _array(arrays, "iteration", (None,), location, kind="integer")
    count = iteration.size
    if count == 0 or not np.array_equal(iteration, np.arange(count)):
        raise ArtifactValidationError(
            "{}:iteration must be contiguous from zero".format(location)
        )
    for key in ("input_q", "target_q", "accepted_q"):
        value = _array(
            arrays,
            key,
            (count, DYNAMICS_RESIDUAL_DIMENSION),
            location,
        )
        _positive(value, "{}:{}".format(location, key))
    for key in (
        "alpha",
        "log_q_change",
        "map_objective",
        "approximate_marginal_objective",
        "lag",
    ):
        _array(arrays, key, (count,), location)
    if np.any(arrays["alpha"] < 0.0) or np.any(arrays["alpha"] > 1.0):
        raise ArtifactValidationError("{}:alpha must lie in [0, 1]".format(location))
    _nonnegative(arrays["log_q_change"], "{}:log_q_change".format(location))
    _nonnegative(arrays["lag"], "{}:lag".format(location))
    _array(arrays, "accepted", (count,), location, kind="boolean")
    _strings(arrays, "reason", count, location)
    _array(
        arrays,
        "floor_activation",
        (count, DYNAMICS_RESIDUAL_DIMENSION),
        location,
        kind="boolean",
    )
    for key in (
        "expected_residual_second_moment",
        "map_residual_second_moment",
        "covariance_correction",
    ):
        value = _array(
            arrays,
            key,
            (count, DYNAMICS_RESIDUAL_DIMENSION),
            location,
        )
        _nonnegative(value, "{}:{}".format(location, key))
    if not np.allclose(
        arrays["expected_residual_second_moment"],
        arrays["map_residual_second_moment"] + arrays["covariance_correction"],
        rtol=1.0e-9,
        atol=1.0e-12,
    ):
        raise ArtifactValidationError(
            "{}:expected residual second moment must equal MAP moment plus "
            "covariance correction".format(location)
        )


_LAPLACE_KEYS = (
    "reduced_likelihood_hessian",
    "reduced_posterior_hessian",
    "covariance",
    "static_covariance_conditioning",
    "eigenvalues",
    "eigenvectors",
    "effective_rank",
    "exact_ridge_direction",
    "ridge_alignment",
    "condition_number",
    "delay_profile_available",
    "delay_profile_grid",
    "delay_profile_objective",
    "delay_profile_approximate_marginal_objective",
    "delay_profile_static_coordinate",
    "delay_local_uncertainty",
    "delay_uncertainty_source",
    "delay_profile_curvature",
    "delay_profile_curvature_valid",
    "delay_local_geometry_valid",
    "delay_local_geometry_method",
    "delay_local_geometry_reason",
    "delay_profile_support_lag",
    "delay_profile_support_map_objective",
    "delay_profile_support_static_coordinate",
    "delay_profile_gradient",
    "delay_static_sensitivity",
    "parameter_delay_cross_covariance",
    "joint_parameter_delay_information",
    "joint_parameter_delay_covariance",
    "mcmc_quadratic_surrogate_method",
)


def _validate_laplace(arrays: Mapping[str, np.ndarray], location: str) -> None:
    _require_keys(arrays, _LAPLACE_KEYS, location)
    _reject_unknown_keys(arrays, _LAPLACE_KEYS, location)
    dimension = STATIC_PARAMETER_DIMENSION
    for key in (
        "reduced_likelihood_hessian",
        "reduced_posterior_hessian",
        "covariance",
    ):
        value = _array(arrays, key, (dimension, dimension), location)
        if not np.allclose(value, value.T, rtol=1.0e-9, atol=1.0e-11):
            raise ArtifactValidationError(
                "{}:{} must be symmetric".format(location, key)
            )
    conditioning = _strings(
        arrays, "static_covariance_conditioning", 1, location
    )
    if conditioning[0] != "fixed_delay_conditional":
        raise ArtifactValidationError(
            "{}:covariance must be labelled fixed_delay_conditional".format(
                location
            )
        )
    posterior = arrays["reduced_posterior_hessian"]
    conditional_covariance = arrays["covariance"]
    if np.min(np.linalg.eigvalsh(posterior)) <= 0.0 or np.min(
        np.linalg.eigvalsh(conditional_covariance)
    ) <= 0.0:
        raise ArtifactValidationError(
            "{}:posterior Hessian and conditional covariance must be positive definite".format(
                location
            )
        )
    if not np.allclose(
        posterior @ conditional_covariance,
        np.eye(dimension),
        rtol=2.0e-8,
        atol=2.0e-8,
    ):
        raise ArtifactValidationError(
            "{}:fixed-delay conditional covariance must invert posterior Hessian".format(
                location
            )
        )
    _array(arrays, "eigenvalues", (dimension,), location)
    eigenvectors = _array(arrays, "eigenvectors", (dimension, dimension), location)
    if not np.allclose(
        eigenvectors.T.dot(eigenvectors),
        np.eye(dimension),
        rtol=1.0e-7,
        atol=1.0e-8,
    ):
        raise ArtifactValidationError(
            "{}:eigenvectors must be orthonormal".format(location)
        )
    rank = _array(arrays, "effective_rank", (1,), location, kind="integer")
    if int(rank[0]) < 0 or int(rank[0]) > dimension:
        raise ArtifactValidationError(
            "{}:effective_rank is outside [0, {}]".format(location, dimension)
        )
    ridge = _array(arrays, "exact_ridge_direction", (dimension,), location)
    if not np.isclose(np.linalg.norm(ridge), 1.0, rtol=1.0e-7, atol=1.0e-9):
        raise ArtifactValidationError(
            "{}:exact_ridge_direction must have unit norm".format(location)
        )
    alignment = _array(arrays, "ridge_alignment", (1,), location)
    if alignment[0] < 0.0 or alignment[0] > 1.0:
        raise ArtifactValidationError(
            "{}:ridge_alignment must lie in [0, 1]".format(location)
        )
    condition = _array(
        arrays, "condition_number", (1,), location, finite=False
    )
    if np.isnan(condition[0]) or condition[0] < 0.0:
        raise ArtifactValidationError(
            "{}:condition_number must be non-negative or +inf".format(location)
        )
    available = _array(
        arrays,
        "delay_profile_available",
        (1,),
        location,
        kind="boolean",
    )
    grid = _array(arrays, "delay_profile_grid", (None,), location)
    if available[0] and grid.size == 0:
        raise ArtifactValidationError(
            "{}:delay_profile_grid must not be empty when the final-Q "
            "profile is available".format(location)
        )
    if not available[0] and grid.size != 0:
        raise ArtifactValidationError(
            "{}:delay_profile_grid must be empty when the final-Q profile "
            "is unavailable".format(location)
        )
    if grid.size:
        _strictly_increasing(grid, "{}:delay_profile_grid".format(location))
        _nonnegative(grid, "{}:delay_profile_grid".format(location))
    objective = _array(
        arrays,
        "delay_profile_objective",
        (grid.size,),
        location,
    )
    marginal = _array(
        arrays,
        "delay_profile_approximate_marginal_objective",
        (grid.size,),
        location,
    )
    coordinate = _array(
        arrays,
        "delay_profile_static_coordinate",
        (grid.size, dimension),
        location,
    )
    delay_standard_deviation = _array(
        arrays, "delay_local_uncertainty", (1,), location
    )
    _positive(
        delay_standard_deviation,
        "{}:delay_local_uncertainty".format(location),
    )
    _strings(arrays, "delay_uncertainty_source", 1, location)
    curvature = _array(
        arrays, "delay_profile_curvature", (1,), location
    )
    curvature_valid = _array(
        arrays,
        "delay_profile_curvature_valid",
        (1,),
        location,
        kind="boolean",
    )
    geometry_valid = _array(
        arrays,
        "delay_local_geometry_valid",
        (1,),
        location,
        kind="boolean",
    )
    if not np.array_equal(curvature_valid, geometry_valid):
        raise ArtifactValidationError(
            "{}:delay curvature and local geometry validity must agree".format(
                location
            )
        )
    method = _strings(arrays, "delay_local_geometry_method", 1, location)
    reason = _strings(arrays, "delay_local_geometry_reason", 1, location)
    surrogate_method = _strings(
        arrays, "mcmc_quadratic_surrogate_method", 1, location
    )
    if method[0] != "nonuniform_three_point_map_profile_v1":
        raise ArtifactValidationError(
            "{}:unknown delay local geometry method".format(location)
        )
    support_lag = arrays["delay_profile_support_lag"]
    support_objective = arrays["delay_profile_support_map_objective"]
    support_coordinate = arrays["delay_profile_support_static_coordinate"]
    gradient = arrays["delay_profile_gradient"]
    sensitivity = arrays["delay_static_sensitivity"]
    cross = arrays["parameter_delay_cross_covariance"]
    joint_information = arrays["joint_parameter_delay_information"]
    joint_covariance = arrays["joint_parameter_delay_covariance"]
    if curvature_valid[0]:
        _positive(curvature, "{}:delay_profile_curvature".format(location))
        if not available[0]:
            raise ArtifactValidationError(
                "{}:delay profile curvature cannot be valid without a "
                "final-Q profile".format(location)
            )
        if reason[0] != "valid":
            raise ArtifactValidationError(
                "{}:valid local geometry requires reason 'valid'".format(location)
            )
        if surrogate_method[0] != "joint_profile_information_v1":
            raise ArtifactValidationError(
                "{}:valid local geometry must use joint MCMC information".format(
                    location
                )
            )
        _array(arrays, "delay_profile_support_lag", (3,), location)
        _array(
            arrays, "delay_profile_support_map_objective", (3,), location
        )
        _array(
            arrays,
            "delay_profile_support_static_coordinate",
            (3, dimension),
            location,
        )
        _array(arrays, "delay_profile_gradient", (1,), location)
        _array(arrays, "delay_static_sensitivity", (dimension,), location)
        _array(
            arrays,
            "parameter_delay_cross_covariance",
            (dimension,),
            location,
        )
        _array(
            arrays,
            "joint_parameter_delay_information",
            (dimension + 1, dimension + 1),
            location,
        )
        _array(
            arrays,
            "joint_parameter_delay_covariance",
            (dimension + 1, dimension + 1),
            location,
        )
        _strictly_increasing(
            support_lag, "{}:delay_profile_support_lag".format(location)
        )
        support_indices = []
        for value in support_lag:
            match = np.flatnonzero(
                np.isclose(grid, value, rtol=0.0, atol=1.0e-14)
            )
            if match.size != 1:
                raise ArtifactValidationError(
                    "{}:local support lag is absent from profile grid".format(
                        location
                    )
                )
            support_indices.append(int(match[0]))
        if not np.allclose(
            support_objective,
            objective[support_indices],
            rtol=2.0e-10,
            atol=2.0e-10,
        ) or not np.allclose(
            support_coordinate,
            coordinate[support_indices],
            rtol=2.0e-10,
            atol=2.0e-10,
        ):
            raise ArtifactValidationError(
                "{}:local support does not reproduce the MAP profile".format(
                    location
                )
            )
        left = support_lag[1] - support_lag[0]
        right = support_lag[2] - support_lag[1]
        first = np.asarray(
            (
                -right / (left * (left + right)),
                (right - left) / (left * right),
                left / (right * (left + right)),
            )
        )
        second = np.asarray(
            (
                2.0 / (left * (left + right)),
                -2.0 / (left * right),
                2.0 / (right * (left + right)),
            )
        )
        if not np.isclose(
            gradient[0], first @ support_objective, rtol=2.0e-10, atol=1.0e-10
        ) or not np.isclose(
            curvature[0], second @ support_objective, rtol=2.0e-10, atol=1.0e-8
        ) or not np.allclose(
            sensitivity,
            first @ support_coordinate,
            rtol=2.0e-10,
            atol=1.0e-9,
        ):
            raise ArtifactValidationError(
                "{}:nonuniform three-point derivatives do not reproduce".format(
                    location
                )
            )
        expected_information = np.empty_like(joint_information)
        information_times_sensitivity = posterior @ sensitivity
        expected_information[:-1, :-1] = posterior
        expected_information[:-1, -1] = -information_times_sensitivity
        expected_information[-1, :-1] = -information_times_sensitivity
        expected_information[-1, -1] = curvature[0] + float(
            sensitivity @ information_times_sensitivity
        )
        expected_covariance = np.empty_like(joint_covariance)
        expected_cross = sensitivity / curvature[0]
        expected_covariance[:-1, :-1] = conditional_covariance + np.outer(
            sensitivity, sensitivity
        ) / curvature[0]
        expected_covariance[:-1, -1] = expected_cross
        expected_covariance[-1, :-1] = expected_cross
        expected_covariance[-1, -1] = 1.0 / curvature[0]
        if not np.allclose(
            joint_information,
            expected_information,
            rtol=2.0e-9,
            atol=2.0e-8,
        ) or not np.allclose(
            joint_covariance,
            expected_covariance,
            rtol=2.0e-9,
            atol=2.0e-9,
        ) or not np.allclose(
            cross, expected_cross, rtol=2.0e-9, atol=2.0e-9
        ):
            raise ArtifactValidationError(
                "{}:joint static-delay algebra is inconsistent".format(location)
            )
        if not np.isclose(
            delay_standard_deviation[0] ** 2,
            1.0 / curvature[0],
            rtol=2.0e-9,
            atol=1.0e-14,
        ):
            raise ArtifactValidationError(
                "{}:delay uncertainty disagrees with curvature".format(location)
            )
    elif curvature[0] != 0.0:
        raise ArtifactValidationError(
            "{}:delay_profile_curvature must use canonical zero when "
            "unavailable".format(location)
        )
    else:
        if surrogate_method[0] != "proposal_only_block_diagonal_fallback_v1":
            raise ArtifactValidationError(
                "{}:invalid geometry must audit proposal-only fallback".format(
                    location
                )
            )
        empty_shapes = {
            "delay_profile_support_lag": (0,),
            "delay_profile_support_map_objective": (0,),
            "delay_profile_support_static_coordinate": (0, dimension),
            "delay_profile_gradient": (0,),
            "delay_static_sensitivity": (0,),
            "parameter_delay_cross_covariance": (0,),
            "joint_parameter_delay_information": (0, 0),
            "joint_parameter_delay_covariance": (0, 0),
        }
        for key, shape in empty_shapes.items():
            _array(arrays, key, shape, location)


_DIAGNOSTIC_KEYS = (
    "bag_id",
    "knot_count",
    "factor_count",
    "residual_dimension",
    "jacobian_nnz",
    "assembly_seconds",
    "factorization_seconds",
    "schur_solve_seconds",
    "nonlinear_iteration_seconds",
    "em_iteration_seconds",
    "mcmc_target_seconds",
    "peak_memory_bytes",
)

_MCMC_DIAGNOSTIC_KEYS = (
    "mcmc_chain_id",
    "mcmc_mode_id",
    "mcmc_draws_per_chain",
    "mcmc_split_rhat",
    "mcmc_effective_sample_size",
    "mcmc_integrated_autocorrelation_time",
    "mcmc_ridge_coordinate_trace",
    "mcmc_delay_trace",
    "mcmc_log_posterior_trace",
    "mcmc_kernel_names",
    "mcmc_kernel_attempts",
    "mcmc_kernel_stage_one_accepted",
    "mcmc_kernel_stage_two_attempted",
    "mcmc_kernel_stage_two_accepted",
    "mcmc_kernel_full_target_cache_hits",
    "mcmc_kernel_inner_solve_failures",
    "mcmc_kernel_inner_iterations",
    "mcmc_completed",
    "mcmc_converged",
    "mcmc_rhat_threshold",
    "mcmc_minimum_effective_sample_size",
)


def _validate_diagnostics(
    arrays: Mapping[str, np.ndarray],
    bag_ids: Tuple[str, ...],
    mcmc_enabled: bool,
    location: str,
) -> None:
    _require_keys(arrays, _DIAGNOSTIC_KEYS, location)
    allowed = _DIAGNOSTIC_KEYS + _MCMC_DIAGNOSTIC_KEYS
    _reject_unknown_keys(arrays, allowed, location)
    actual_bags = _strings(arrays, "bag_id", len(bag_ids), location, unique=True)
    if tuple(actual_bags.tolist()) != bag_ids:
        raise ArtifactValidationError(
            "{}:bag_id must match selected_bag_ids in order".format(location)
        )
    for key in ("knot_count", "factor_count", "residual_dimension", "jacobian_nnz"):
        value = _array(
            arrays, key, (len(bag_ids),), location, kind="integer"
        )
        if np.any(value <= 0):
            raise ArtifactValidationError(
                "{}:{} must be positive".format(location, key)
            )
    for key in ("assembly_seconds", "factorization_seconds", "schur_solve_seconds"):
        value = _array(arrays, key, (len(bag_ids),), location)
        _nonnegative(value, "{}:{}".format(location, key))
    for key in (
        "nonlinear_iteration_seconds",
        "em_iteration_seconds",
        "mcmc_target_seconds",
    ):
        value = _array(arrays, key, (None,), location)
        _nonnegative(value, "{}:{}".format(location, key))
    if mcmc_enabled and arrays["mcmc_target_seconds"].size == 0:
        raise ArtifactValidationError(
            "{}:mcmc_target_seconds must not be empty when MCMC is enabled".format(
                location
            )
        )
    if not mcmc_enabled and arrays["mcmc_target_seconds"].size != 0:
        raise ArtifactValidationError(
            "{}:mcmc_target_seconds must be empty when MCMC is disabled".format(
                location
            )
        )
    supplied_mcmc = set(arrays).intersection(_MCMC_DIAGNOSTIC_KEYS)
    if mcmc_enabled:
        _require_keys(arrays, _MCMC_DIAGNOSTIC_KEYS, location)
        chain_ids = _strings(
            arrays, "mcmc_chain_id", None, location, unique=True
        )
        if chain_ids.size < 2:
            raise ArtifactValidationError(
                "{}:mcmc_chain_id must contain at least two chains".format(
                    location
                )
            )
        _strings(arrays, "mcmc_mode_id", 1, location)
        draws = _array(
            arrays,
            "mcmc_draws_per_chain",
            (1,),
            location,
            kind="integer",
        )
        if draws[0] < 4:
            raise ArtifactValidationError(
                "{}:mcmc_draws_per_chain must be at least four".format(location)
            )
        posterior_dimension = STATIC_PARAMETER_DIMENSION + 1
        for key in (
            "mcmc_split_rhat",
            "mcmc_effective_sample_size",
            "mcmc_integrated_autocorrelation_time",
        ):
            _array(arrays, key, (posterior_dimension,), location, finite=False)
            if np.any(np.isnan(arrays[key])):
                raise ArtifactValidationError(
                    "{}:{} must not contain NaN".format(location, key)
                )
        _positive(
            arrays["mcmc_effective_sample_size"],
            "{}:mcmc_effective_sample_size".format(location),
        )
        if np.any(arrays["mcmc_integrated_autocorrelation_time"] < 1.0):
            raise ArtifactValidationError(
                "{}:mcmc_integrated_autocorrelation_time must be >= 1".format(
                    location
                )
            )
        trace_shape = (chain_ids.size, int(draws[0]))
        for key in (
            "mcmc_ridge_coordinate_trace",
            "mcmc_delay_trace",
            "mcmc_log_posterior_trace",
        ):
            _array(arrays, key, trace_shape, location)
        kernels = _strings(
            arrays, "mcmc_kernel_names", None, location, unique=True
        )
        if kernels.size == 0:
            raise ArtifactValidationError(
                "{}:mcmc_kernel_names must not be empty".format(location)
            )
        count_keys = (
            "mcmc_kernel_attempts",
            "mcmc_kernel_stage_one_accepted",
            "mcmc_kernel_stage_two_attempted",
            "mcmc_kernel_stage_two_accepted",
            "mcmc_kernel_full_target_cache_hits",
            "mcmc_kernel_inner_solve_failures",
            "mcmc_kernel_inner_iterations",
        )
        for key in count_keys:
            values = _array(
                arrays, key, (kernels.size,), location, kind="integer"
            )
            if np.any(values < 0):
                raise ArtifactValidationError(
                    "{}:{} must be non-negative".format(location, key)
                )
        if np.any(
            arrays["mcmc_kernel_stage_two_accepted"]
            > arrays["mcmc_kernel_stage_two_attempted"]
        ) or np.any(
            arrays["mcmc_kernel_stage_two_attempted"]
            > arrays["mcmc_kernel_stage_one_accepted"]
        ) or np.any(
            arrays["mcmc_kernel_stage_one_accepted"]
            > arrays["mcmc_kernel_attempts"]
        ):
            raise ArtifactValidationError(
                "{} MCMC kernel acceptance counts are not nested".format(
                    location
                )
            )
        completed = _array(
            arrays, "mcmc_completed", (1,), location, kind="boolean"
        )
        converged = _array(
            arrays, "mcmc_converged", (1,), location, kind="boolean"
        )
        if converged[0] and not completed[0]:
            raise ArtifactValidationError(
                "{}: an incomplete MCMC run cannot be converged".format(
                    location
                )
            )
        for key in (
            "mcmc_rhat_threshold",
            "mcmc_minimum_effective_sample_size",
        ):
            value = _array(arrays, key, (1,), location)
            _positive(value, "{}:{}".format(location, key))
        if arrays["mcmc_rhat_threshold"][0] <= 1.0:
            raise ArtifactValidationError(
                "{}:mcmc_rhat_threshold must exceed one".format(location)
            )
    elif supplied_mcmc:
        raise ArtifactValidationError(
            "{} contains MCMC diagnostics while MCMC is disabled".format(
                location
            )
        )
    peak = _array(arrays, "peak_memory_bytes", (1,), location, kind="integer")
    if peak[0] < 0:
        raise ArtifactValidationError(
            "{}:peak_memory_bytes must be non-negative".format(location)
        )


_BAG_KEYS = (
    "bag_id",
    "knot_time",
    "knot_record_time",
    "reference_time",
    "reference_record_time",
    "reference_position",
    "reference_linear_velocity",
    "reference_linear_acceleration",
    "reference_rpy",
    "reference_angular_velocity",
    "reference_angular_acceleration",
    "pose_time",
    "pose_record_time",
    "pose_position",
    "pose_orientation_xyzw",
    "pose_valid",
    "pose_covariance",
    "pose_covariance_valid",
    "velocity_time",
    "velocity_record_time",
    "velocity",
    "velocity_valid",
    "velocity_covariance",
    "velocity_covariance_valid",
    "gyro_time",
    "gyro_record_time",
    "gyro",
    "gyro_valid",
    "gyro_covariance",
    "gyro_covariance_valid",
    "accelerometer_time",
    "accelerometer_record_time",
    "accelerometer",
    "accelerometer_valid",
    "accelerometer_covariance",
    "accelerometer_covariance_valid",
    "thrust_command_time",
    "thrust_command_record_time",
    "thrust_command",
    "thrust_command_valid",
    "thrust_command_covariance",
    "thrust_command_covariance_valid",
    "gimbal_command_time",
    "gimbal_command_record_time",
    "gimbal_command",
    "gimbal_command_valid",
    "gimbal_command_covariance",
    "gimbal_command_covariance_valid",
    "gimbal_observation_time",
    "gimbal_observation_record_time",
    "gimbal_observation",
    "gimbal_observation_valid",
    "gimbal_observation_covariance",
    "gimbal_observation_covariance_valid",
    "controller_integral_time",
    "controller_integral_record_time",
    "controller_integral_observation",
    "controller_integral_valid",
    "controller_integral_covariance",
    "controller_integral_covariance_valid",
    "nominal_position",
    "nominal_orientation_xyzw",
    "nominal_linear_velocity",
    "nominal_angular_velocity",
    "nominal_controller_integral",
    "nominal_actuator_thrust",
    "nominal_actuator_gimbal",
    "map_position",
    "map_orientation_xyzw",
    "map_linear_velocity",
    "map_angular_velocity",
    "map_controller_integral",
    "map_actuator_thrust",
    "map_actuator_gimbal",
    "map_dynamics_residual",
    "map_dynamics_residual_valid",
    "correction_translation",
    "correction_rotation_vector",
    "factor_names",
    "factor_residual_history",
    "factor_normalized_residual_history",
    "objective_component_names",
    "objective_component_values",
    "numerical_diagnostic_names",
    "numerical_diagnostic_values",
)


def _stream(
    arrays: Mapping[str, np.ndarray],
    prefix: str,
    value_name: str,
    value_dimension: int,
    covariance_dimension: Optional[int],
    location: str,
) -> None:
    time = _array(arrays, "{}_time".format(prefix), (None,), location)
    count = time.size
    _strictly_increasing(time, "{}:{}_time".format(location, prefix))
    record = _array(
        arrays, "{}_record_time".format(prefix), (count,), location
    )
    _strictly_increasing(
        record, "{}:{}_record_time".format(location, prefix)
    )
    _array(arrays, value_name, (count, value_dimension), location)
    _array(
        arrays, "{}_valid".format(prefix), (count,), location, kind="boolean"
    )
    if covariance_dimension is not None:
        covariance = _array(
            arrays,
            "{}_covariance".format(prefix),
            (count, covariance_dimension, covariance_dimension),
            location,
        )
        if count and not np.allclose(
            covariance,
            np.swapaxes(covariance, 1, 2),
            rtol=1.0e-9,
            atol=1.0e-11,
        ):
            raise ArtifactValidationError(
                "{}:{}_covariance must be symmetric".format(location, prefix)
            )
        _positive_semidefinite(
            covariance, "{}:{}_covariance".format(location, prefix)
        )
        covariance_valid = _array(
            arrays,
            "{}_covariance_valid".format(prefix),
            (count,),
            location,
            kind="boolean",
        )
        if np.any(~covariance_valid) and np.any(
            covariance[~covariance_valid] != 0.0
        ):
            raise ArtifactValidationError(
                "{}:{} covariance must use canonical zeros where unavailable"
                .format(location, prefix)
            )
        if np.any(covariance_valid):
            eigenvalues = np.linalg.eigvalsh(covariance[covariance_valid])
            if np.any(eigenvalues <= 0.0):
                raise ArtifactValidationError(
                    "{}:{} available covariance must be positive definite"
                    .format(location, prefix)
                )


def _validate_bag(
    arrays: Mapping[str, np.ndarray], bag_id: str, location: str
) -> None:
    _require_keys(arrays, _BAG_KEYS, location)
    _reject_unknown_keys(arrays, _BAG_KEYS, location)
    stored_id = _strings(arrays, "bag_id", 1, location)
    if stored_id[0] != bag_id:
        raise ArtifactValidationError(
            "{}:bag_id does not match manifest key {!r}".format(
                location, bag_id
            )
        )
    time = _array(arrays, "knot_time", (None,), location)
    count = time.size
    if count < 2:
        raise ArtifactValidationError(
            "{}:knot_time must contain at least two knots".format(location)
        )
    _strictly_increasing(time, "{}:knot_time".format(location))
    record_time = _array(arrays, "knot_record_time", (count,), location)
    _strictly_increasing(record_time, "{}:knot_record_time".format(location))

    reference_time = _array(arrays, "reference_time", (None,), location)
    reference_count = reference_time.size
    if reference_count == 0:
        raise ArtifactValidationError(
            "{}:reference_time must not be empty".format(location)
        )
    _strictly_increasing(reference_time, "{}:reference_time".format(location))
    reference_record_time = _array(
        arrays, "reference_record_time", (reference_count,), location
    )
    _strictly_increasing(
        reference_record_time, "{}:reference_record_time".format(location)
    )
    for key in (
        "reference_position",
        "reference_linear_velocity",
        "reference_linear_acceleration",
        "reference_rpy",
        "reference_angular_velocity",
        "reference_angular_acceleration",
    ):
        _array(arrays, key, (reference_count, 3), location)

    _stream(arrays, "pose", "pose_position", 3, 6, location)
    pose_count = arrays["pose_time"].size
    orientation = _array(
        arrays, "pose_orientation_xyzw", (pose_count, 4), location
    )
    _unit_quaternions(orientation, "{}:pose_orientation_xyzw".format(location))
    _stream(arrays, "velocity", "velocity", 3, 3, location)
    _stream(arrays, "gyro", "gyro", 3, 3, location)
    _stream(arrays, "accelerometer", "accelerometer", 3, 3, location)
    _stream(arrays, "gimbal_observation", "gimbal_observation", 4, 4, location)
    _stream(
        arrays,
        "controller_integral",
        "controller_integral_observation",
        6,
        6,
        location,
    )

    for prefix, value_name in (
        ("thrust_command", "thrust_command"),
        ("gimbal_command", "gimbal_command"),
    ):
        stream_time = _array(arrays, "{}_time".format(prefix), (None,), location)
        stream_count = stream_time.size
        _strictly_increasing(
            stream_time, "{}:{}_time".format(location, prefix)
        )
        stream_record = _array(
            arrays, "{}_record_time".format(prefix), (stream_count,), location
        )
        _strictly_increasing(
            stream_record, "{}:{}_record_time".format(location, prefix)
        )
        _array(arrays, value_name, (stream_count, 4), location)
        _array(
            arrays,
            "{}_valid".format(prefix),
            (stream_count,),
            location,
            kind="boolean",
        )
        covariance = _array(
            arrays,
            "{}_covariance".format(prefix),
            (stream_count, 4, 4),
            location,
        )
        covariance_valid = _array(
            arrays,
            "{}_covariance_valid".format(prefix),
            (stream_count,),
            location,
            kind="boolean",
        )
        if stream_count and not np.allclose(
            covariance,
            np.swapaxes(covariance, 1, 2),
            rtol=1.0e-9,
            atol=1.0e-11,
        ):
            raise ArtifactValidationError(
                "{}:{}_covariance must be symmetric".format(location, prefix)
            )
        if np.any(~covariance_valid) and np.any(
            covariance[~covariance_valid] != 0.0
        ):
            raise ArtifactValidationError(
                "{}:{} covariance must use canonical zeros where unavailable"
                .format(location, prefix)
            )
        if np.any(covariance_valid) and np.any(
            np.linalg.eigvalsh(covariance[covariance_valid]) <= 0.0
        ):
            raise ArtifactValidationError(
                "{}:{} available covariance must be positive definite".format(
                    location, prefix
                )
            )

    state_shapes = {
        "position": (count, 3),
        "orientation_xyzw": (count, 4),
        "linear_velocity": (count, 3),
        "angular_velocity": (count, 3),
        "controller_integral": (count, 6),
        "actuator_thrust": (count, 4),
        "actuator_gimbal": (count, 4),
    }
    for state_prefix in ("nominal", "map"):
        for suffix, shape in state_shapes.items():
            value = _array(
                arrays, "{}_{}".format(state_prefix, suffix), shape, location
            )
            if suffix == "orientation_xyzw":
                _unit_quaternions(
                    value,
                    "{}:{}_{}".format(location, state_prefix, suffix),
                )
    _array(
        arrays,
        "map_dynamics_residual",
        (count - 1, DYNAMICS_RESIDUAL_DIMENSION),
        location,
    )
    _array(
        arrays,
        "map_dynamics_residual_valid",
        (count - 1,),
        location,
        kind="boolean",
    )
    _array(arrays, "correction_translation", (count, 3), location)
    _array(arrays, "correction_rotation_vector", (count, 3), location)

    factor_names = _strings(arrays, "factor_names", None, location, unique=True)
    if factor_names.size == 0:
        raise ArtifactValidationError(
            "{}:factor_names must not be empty".format(location)
        )
    residual_history = _array(
        arrays,
        "factor_residual_history",
        (None, factor_names.size),
        location,
    )
    _array(
        arrays,
        "factor_normalized_residual_history",
        residual_history.shape,
        location,
    )
    objective_names = _strings(
        arrays, "objective_component_names", None, location, unique=True
    )
    _array(
        arrays,
        "objective_component_values",
        (objective_names.size,),
        location,
    )
    diagnostic_names = _strings(
        arrays, "numerical_diagnostic_names", None, location, unique=True
    )
    _array(
        arrays,
        "numerical_diagnostic_values",
        (diagnostic_names.size,),
        location,
    )


_MCMC_KEYS = (
    "sample_id",
    "chain_id",
    "draw_index",
    "parameter_coordinate",
    "mass",
    "inertia",
    "cog",
    "force_effectiveness",
    "torque_effectiveness",
    "delay",
    "log_posterior",
    "log_likelihood_approximation",
    "log_determinant_term",
    "accepted_kernel",
    "source_mode_id",
)


def _validate_mcmc(
    arrays: Mapping[str, np.ndarray], location: str
) -> np.ndarray:
    _require_keys(arrays, _MCMC_KEYS, location)
    forbidden = {"member_id", "weight", "weights", "particle_weight"}.intersection(
        arrays
    )
    if forbidden:
        raise ArtifactValidationError(
            "{} contains forbidden particle fields: {}".format(
                location, ", ".join(sorted(forbidden))
            )
        )
    _reject_unknown_keys(arrays, _MCMC_KEYS, location)
    sample_ids = _id_vector(arrays, "sample_id", location)
    count = sample_ids.size
    chain_id = _strings(arrays, "chain_id", count, location)
    draw_index = _array(
        arrays, "draw_index", (count,), location, kind="integer"
    )
    if np.any(draw_index < 0):
        raise ArtifactValidationError(
            "{}:draw_index must be non-negative".format(location)
        )
    chain_draw = tuple(zip(chain_id.tolist(), draw_index.tolist()))
    if len(set(chain_draw)) != count:
        raise ArtifactValidationError(
            "{} contains duplicate (chain_id, draw_index) pairs".format(location)
        )
    _array(
        arrays,
        "parameter_coordinate",
        (count, STATIC_PARAMETER_DIMENSION),
        location,
    )
    mass = _array(arrays, "mass", (count,), location)
    _positive(mass, "{}:mass".format(location))
    inertia = _array(arrays, "inertia", (count, 3, 3), location)
    if not np.allclose(
        inertia, np.swapaxes(inertia, 1, 2), rtol=1.0e-9, atol=1.0e-11
    ):
        raise ArtifactValidationError("{}:inertia must be symmetric".format(location))
    if np.any(np.linalg.eigvalsh(inertia) <= 0.0):
        raise ArtifactValidationError(
            "{}:inertia must be positive definite".format(location)
        )
    _array(arrays, "cog", (count, 3), location)
    force = _array(arrays, "force_effectiveness", (count, 4), location)
    torque = _array(arrays, "torque_effectiveness", (count, 4), location)
    _positive(force, "{}:force_effectiveness".format(location))
    _positive(torque, "{}:torque_effectiveness".format(location))
    delay = _array(arrays, "delay", (count,), location)
    _nonnegative(delay, "{}:delay".format(location))
    for key in (
        "log_posterior",
        "log_likelihood_approximation",
        "log_determinant_term",
    ):
        _array(arrays, key, (count,), location)
    accepted_kernel = _array(
        arrays, "accepted_kernel", (count,), location, kind="string"
    )
    del accepted_kernel
    _strings(arrays, "source_mode_id", count, location)
    return sample_ids


_TRAJECTORY_KEYS = (
    "sample_id",
    "knot_time",
    "conditional_position",
    "conditional_orientation_xyzw",
    "conditional_linear_velocity",
    "conditional_angular_velocity",
    "conditional_controller_integral",
    "conditional_actuator_thrust",
    "conditional_actuator_gimbal",
    "correction_translation",
    "correction_rotation_vector",
    "dynamics_residual",
    "dynamics_residual_valid",
    "conditional_objective",
)


def _validate_trajectory_subset(
    arrays: Mapping[str, np.ndarray],
    mcmc_sample_ids: np.ndarray,
    location: str,
) -> None:
    _require_keys(arrays, _TRAJECTORY_KEYS, location)
    _reject_unknown_keys(arrays, _TRAJECTORY_KEYS, location)
    selected_ids = _id_vector(arrays, "sample_id", location)
    if selected_ids.dtype.kind != mcmc_sample_ids.dtype.kind:
        raise ArtifactValidationError(
            "{}:sample_id dtype kind does not match mcmc_samples".format(location)
        )
    available = set(mcmc_sample_ids.tolist())
    missing = [
        sample_id
        for sample_id in selected_ids.tolist()
        if sample_id not in available
    ]
    if missing:
        raise ArtifactValidationError(
            "{}:sample_id contains IDs absent from mcmc_samples: {}".format(
                location, missing
            )
        )
    sample_count = selected_ids.size
    knot_time = _array(arrays, "knot_time", (None,), location)
    knot_count = knot_time.size
    if knot_count < 2:
        raise ArtifactValidationError(
            "{}:knot_time must contain at least two knots".format(location)
        )
    _strictly_increasing(knot_time, "{}:knot_time".format(location))
    shapes = {
        "conditional_position": (sample_count, knot_count, 3),
        "conditional_orientation_xyzw": (sample_count, knot_count, 4),
        "conditional_linear_velocity": (sample_count, knot_count, 3),
        "conditional_angular_velocity": (sample_count, knot_count, 3),
        "conditional_controller_integral": (sample_count, knot_count, 6),
        "conditional_actuator_thrust": (sample_count, knot_count, 4),
        "conditional_actuator_gimbal": (sample_count, knot_count, 4),
        "correction_translation": (sample_count, knot_count, 3),
        "correction_rotation_vector": (sample_count, knot_count, 3),
        "dynamics_residual": (
            sample_count,
            knot_count - 1,
            DYNAMICS_RESIDUAL_DIMENSION,
        ),
        "conditional_objective": (sample_count,),
    }
    for key, shape in shapes.items():
        value = _array(arrays, key, shape, location)
        if key == "conditional_orientation_xyzw":
            _unit_quaternions(
                value.reshape((-1, 4)),
                "{}:conditional_orientation_xyzw".format(location),
            )
    _array(
        arrays,
        "dynamics_residual_valid",
        (sample_count, knot_count - 1),
        location,
        kind="boolean",
    )


def _freeze(arrays: Dict[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    for value in arrays.values():
        value.setflags(write=False)
    return arrays


def _load_validated_run(
    run_root: Path, manifest: Mapping[str, Any]
) -> BatchEstimationRun:
    (
        bag_ids,
        core_paths,
        bag_paths,
        mcmc_path,
        trajectory_paths,
    ) = _validate_manifest(run_root, manifest)

    map_static = load_npz_strict(core_paths["map_static"])
    q_em = load_npz_strict(core_paths["q_em"])
    laplace = load_npz_strict(core_paths["laplace"])
    diagnostics = load_npz_strict(core_paths["diagnostics"])
    _validate_map_static(map_static, bag_ids, str(core_paths["map_static"]))
    _validate_q_em(q_em, str(core_paths["q_em"]))
    _validate_laplace(laplace, str(core_paths["laplace"]))
    mcmc_enabled = mcmc_path is not None
    _validate_diagnostics(
        diagnostics,
        bag_ids,
        mcmc_enabled,
        str(core_paths["diagnostics"]),
    )

    bags: Dict[str, Mapping[str, np.ndarray]] = {}
    for bag_id in bag_ids:
        arrays = load_npz_strict(bag_paths[bag_id])
        _validate_bag(arrays, bag_id, str(bag_paths[bag_id]))
        bags[bag_id] = _freeze(arrays)

    mcmc_arrays: Optional[Dict[str, np.ndarray]] = None
    mcmc_sample_ids: Optional[np.ndarray] = None
    if mcmc_path is not None:
        mcmc_arrays = load_npz_strict(mcmc_path)
        mcmc_sample_ids = _validate_mcmc(mcmc_arrays, str(mcmc_path))

    trajectories: Dict[str, Mapping[str, np.ndarray]] = {}
    for bag_id, path in trajectory_paths.items():
        if mcmc_sample_ids is None:  # pragma: no cover - manifest prevents it
            raise ArtifactValidationError(
                "trajectory subsets require mcmc_samples"
            )
        arrays = load_npz_strict(path)
        _validate_trajectory_subset(arrays, mcmc_sample_ids, str(path))
        if not np.array_equal(arrays["knot_time"], bags[bag_id]["knot_time"]):
            raise ArtifactValidationError(
                "{}:knot_time must exactly match bags/{}/knot_time".format(
                    path, bag_id
                )
            )
        trajectories[bag_id] = _freeze(arrays)

    if trajectories:
        if mcmc_arrays is None or mcmc_sample_ids is None:  # pragma: no cover
            raise ArtifactValidationError(
                "conditional trajectories require retained MCMC arrays"
            )
        selection = manifest["mcmc_settings"][
            "conditional_trajectory_selection"
        ]
        parameter_prior = manifest["parameter_prior"]
        delay_prior = manifest["delay_prior"]
        mcmc_settings = manifest["mcmc_settings"]
        try:
            expected_representatives = (
                select_posterior_representatives_from_arrays(
                    mcmc_arrays,
                    prior_mean_coordinate=parameter_prior[
                        "mean_coordinate"
                    ],
                    prior_covariance=parameter_prior["covariance"],
                    delay_bounds_seconds=delay_prior["bounds_seconds"],
                    delay_scale_seconds=mcmc_settings[
                        "delay_scale_seconds"
                    ],
                    exact_ridge_direction=laplace[
                        "exact_ridge_direction"
                    ],
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactValidationError(
                "conditional trajectory role-union inputs are invalid"
            ) from error
        expected_selection = expected_representatives.manifest_payload(
            bag_ids
        )
        if selection != expected_selection:
            raise ArtifactValidationError(
                "conditional trajectory roles, IDs, or order disagree with "
                "recomputed role-union v2"
            )
        selected_ids = tuple(
            str(value) for value in selection["selected_sample_ids"]
        )
        if int(selection["available_sample_count"]) != mcmc_sample_ids.size:
            raise ArtifactValidationError(
                "conditional trajectory population disagrees with MCMC sample count"
            )
        available_ids = tuple(str(value) for value in mcmc_sample_ids.tolist())
        index_by_id = {
            sample_id: index for index, sample_id in enumerate(available_ids)
        }
        if not set(selected_ids).issubset(index_by_id):
            raise ArtifactValidationError(
                "conditional trajectory selection contains unknown MCMC IDs"
            )
        if delay_prior.get("prior_kind") != "uniform":
            raise ArtifactValidationError(
                "conditional objective audit requires the declared uniform delay prior"
            )
        lower, upper = _interval(
            delay_prior.get("bounds_seconds"),
            "manifest.delay_prior.bounds_seconds",
        )
        delay_log_prior = -float(np.log(upper - lower))
        solver_relative_tolerance = manifest["solver_settings"].get(
            "relative_objective_tolerance", 0.0
        )
        if isinstance(solver_relative_tolerance, bool):
            raise ArtifactValidationError(
                "manifest solver relative objective tolerance must be numeric"
            )
        try:
            solver_relative_tolerance = float(solver_relative_tolerance)
        except (TypeError, ValueError) as error:
            raise ArtifactValidationError(
                "manifest solver relative objective tolerance must be numeric"
            ) from error
        if (
            not np.isfinite(solver_relative_tolerance)
            or solver_relative_tolerance < 0.0
        ):
            raise ArtifactValidationError(
                "manifest solver relative objective tolerance is invalid"
            )
        objective_relative_tolerance = max(
            2.0e-8, 5.0 * solver_relative_tolerance
        )
        for bag_id, arrays in trajectories.items():
            actual_ids = tuple(str(value) for value in arrays["sample_id"].tolist())
            if actual_ids != selected_ids:
                raise ArtifactValidationError(
                    "trajectory {} sample order disagrees with manifest policy"
                    .format(bag_id)
                )
            indices = np.asarray(
                tuple(index_by_id[value] for value in actual_ids),
                dtype=np.int64,
            )
            expected_objective = (
                delay_log_prior
                + mcmc_arrays["log_determinant_term"][indices]
                - mcmc_arrays["log_posterior"][indices]
            )
            if not np.allclose(
                arrays["conditional_objective"],
                expected_objective,
                rtol=objective_relative_tolerance,
                atol=objective_relative_tolerance
                * np.maximum(1.0, np.abs(expected_objective)),
            ):
                raise ArtifactValidationError(
                    "trajectory {} conditional objective is not the retained target"
                    .format(bag_id)
                )

    if not np.allclose(
        map_static["q_diagonal"],
        q_em["accepted_q"][-1],
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ArtifactValidationError(
            "map_static:q_diagonal must match the final accepted Q in q_em"
        )
    em_status = manifest["substage_status"]["laplace_em"]
    if em_status["termination_reason"] == "fixed_by_request":
        fixed_q_record = (
            q_em["iteration"].size == 1
            and not bool(q_em["accepted"][0])
            and float(q_em["alpha"][0]) == 0.0
            and float(q_em["log_q_change"][0]) == 0.0
            and np.array_equal(q_em["input_q"][0], q_em["accepted_q"][0])
            and str(q_em["reason"][0]) == "fixed_by_request"
            and not bool(em_status["converged"])
            and em_status["termination_reason"] == "fixed_by_request"
        )
        if not fixed_q_record:
            raise ArtifactValidationError(
                "fixed Q policy requires one non-updating diagnostic record"
            )
    elif np.any(q_em["reason"] == "fixed_by_request"):
        raise ArtifactValidationError(
            "Q history fixed_by_request reason disagrees with substage status"
        )
    if not np.isclose(
        map_static["delay"][0],
        q_em["lag"][-1],
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ArtifactValidationError(
            "map_static:delay must match the final lag in q_em"
        )
    if bool(laplace["delay_profile_available"][0]):
        grid = laplace["delay_profile_grid"]
        objective = laplace["delay_profile_objective"]
        best_index = min(
            range(grid.size), key=lambda index: (objective[index], grid[index])
        )
        expected_map_objective = float(map_static["prior_objective"][0]) + float(
            map_static["likelihood_objective"][0]
        )
        if not np.isclose(
            q_em["map_objective"][-1],
            expected_map_objective,
            rtol=2.0e-9,
            atol=2.0e-9 * max(1.0, abs(expected_map_objective)),
        ):
            raise ArtifactValidationError(
                "final q_em MAP objective must reproduce map_static"
            )
        if not np.isclose(
            grid[best_index],
            map_static["delay"][0],
            rtol=1.0e-12,
            atol=1.0e-14,
        ) or not np.isclose(
            objective[best_index],
            expected_map_objective,
            rtol=2.0e-9,
            atol=2.0e-9 * max(1.0, abs(expected_map_objective)),
        ) or not np.allclose(
            laplace["delay_profile_static_coordinate"][best_index],
            map_static["parameter_coordinate_map"],
            rtol=2.0e-9,
            atol=2.0e-10,
        ):
            raise ArtifactValidationError(
                "final-Q MAP profile center must reproduce map_static"
            )
        marginal = q_em["approximate_marginal_objective"][-1]
        if not np.isclose(
            laplace["delay_profile_approximate_marginal_objective"][best_index],
            marginal,
            rtol=2.0e-9,
            atol=2.0e-9 * max(1.0, abs(marginal)),
        ):
            raise ArtifactValidationError(
                "final-Q profile marginal audit must reproduce q_em"
            )
        if bool(laplace["delay_local_geometry_valid"][0]) and not np.isclose(
            laplace["delay_profile_support_lag"][1],
            map_static["delay"][0],
            rtol=1.0e-12,
            atol=1.0e-14,
        ):
            raise ArtifactValidationError(
                "valid local geometry support must center on the final MAP delay"
            )

    return BatchEstimationRun(
        root=run_root,
        manifest=manifest,
        map_static=_freeze(map_static),
        q_em=_freeze(q_em),
        laplace=_freeze(laplace),
        diagnostics=_freeze(diagnostics),
        bags=bags,
        mcmc_samples=(
            None if mcmc_arrays is None else _freeze(mcmc_arrays)
        ),
        trajectories=trajectories,
    )


def _copy_array_mapping(
    arrays: Mapping[str, np.ndarray], location: str
) -> Dict[str, np.ndarray]:
    if not isinstance(arrays, Mapping) or not arrays:
        raise ArtifactValidationError(
            "{} must be a non-empty array mapping".format(location)
        )
    copied: Dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if not isinstance(key, str) or not key:
            raise ArtifactValidationError(
                "{} keys must be non-empty strings".format(location)
            )
        selected = np.asarray(value)
        if selected.dtype.hasobject:
            raise ArtifactValidationError(
                "{}:{} has forbidden object dtype".format(location, key)
            )
        copied[key] = np.array(selected, copy=True)
    return copied


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_writing_directory(
    destination: Path, manifest: Mapping[str, Any]
) -> None:
    if destination.exists() or destination.is_symlink():
        raise ArtifactValidationError(
            "batch run destination already exists: {}".format(destination)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".{}-".format(destination.name),
            suffix=".writing",
            dir=str(destination.parent),
        )
    )
    published = False
    try:
        write_json_atomic(staging / "manifest.json", manifest)
        os.replace(str(staging), str(destination))
        published = True
        _fsync_directory(destination.parent)
    finally:
        if not published and staging.exists():
            manifest_path = staging / "manifest.json"
            if manifest_path.is_file():
                manifest_path.unlink()
            staging.rmdir()


def _written_descriptor(destination: Path, relative: str) -> Dict[str, str]:
    return {
        "path": relative,
        "sha256": file_sha256(destination / relative),
    }


def write_batch_estimation_run(
    root: Union[str, Path],
    *,
    manifest_metadata: Mapping[str, Any],
    map_static: Mapping[str, np.ndarray],
    q_em: Mapping[str, np.ndarray],
    laplace: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, np.ndarray],
    bags: Mapping[str, Mapping[str, np.ndarray]],
    mcmc_samples: Optional[Mapping[str, np.ndarray]] = None,
    trajectories: Optional[
        Mapping[str, Mapping[str, np.ndarray]]
    ] = None,
) -> BatchEstimationRun:
    """Atomically publish one immutable strict-v1 estimation directory.

    ``manifest_metadata`` contains every manifest field except ``schema``,
    ``status``, and ``artifacts``.  Those three fields are writer-owned so a
    caller cannot advertise completion before the payload and SHA-256
    descriptors have validated.  The destination must not already exist.

    The directory first appears atomically with ``status == "writing"``.
    Every NPZ is then replaced atomically and validated from disk.  The final
    atomic manifest replacement is the sole completion commit point.  An I/O
    or validation failure therefore leaves an explicitly incomplete run that
    the public loader refuses to consume.
    """

    if not isinstance(manifest_metadata, Mapping):
        raise ArtifactValidationError("manifest_metadata must be an object")
    _require_keys(
        manifest_metadata, _WRITER_METADATA_KEYS, "manifest_metadata"
    )
    _reject_unknown_keys(
        manifest_metadata, _WRITER_METADATA_KEYS, "manifest_metadata"
    )
    bag_ids = _string_list(
        manifest_metadata["selected_bag_ids"],
        "manifest_metadata.selected_bag_ids",
    )
    for index, bag_id in enumerate(bag_ids):
        _safe_bag_id(
            bag_id,
            "manifest_metadata.selected_bag_ids[{}]".format(index),
        )
    if not isinstance(bags, Mapping) or set(bags) != set(bag_ids):
        raise ArtifactValidationError(
            "bags keys must exactly match manifest_metadata.selected_bag_ids"
        )
    selected_trajectories = {} if trajectories is None else trajectories
    if not isinstance(selected_trajectories, Mapping):
        raise ArtifactValidationError("trajectories must be an object")
    if not set(selected_trajectories).issubset(set(bag_ids)):
        raise ArtifactValidationError(
            "trajectories contains an unknown bag ID"
        )

    mcmc_settings = _required_mapping(
        manifest_metadata, "mcmc_settings", "manifest_metadata"
    )
    if not isinstance(mcmc_settings.get("enabled"), bool):
        raise ArtifactValidationError(
            "manifest_metadata.mcmc_settings.enabled must be explicit boolean"
        )
    mcmc_enabled = bool(mcmc_settings["enabled"])
    if mcmc_enabled != (mcmc_samples is not None):
        raise ArtifactValidationError(
            "mcmc_samples presence must exactly match mcmc_settings.enabled"
        )
    if selected_trajectories and not mcmc_enabled:
        raise ArtifactValidationError(
            "trajectory subsets require enabled MCMC"
        )

    payload_map_static = _copy_array_mapping(map_static, "map_static")
    payload_q_em = _copy_array_mapping(q_em, "q_em")
    payload_laplace = _copy_array_mapping(laplace, "laplace")
    payload_diagnostics = _copy_array_mapping(diagnostics, "diagnostics")
    payload_bags = {
        bag_id: _copy_array_mapping(bags[bag_id], "bags.{}".format(bag_id))
        for bag_id in bag_ids
    }
    payload_mcmc = (
        None
        if mcmc_samples is None
        else _copy_array_mapping(mcmc_samples, "mcmc_samples")
    )
    payload_trajectories = {
        bag_id: _copy_array_mapping(
            selected_trajectories[bag_id],
            "trajectories.{}".format(bag_id),
        )
        for bag_id in bag_ids
        if bag_id in selected_trajectories
    }

    destination = Path(root).expanduser().resolve()
    writing_manifest = dict(manifest_metadata)
    writing_manifest.update(
        {
            "schema": BATCH_ESTIMATION_RUN_SCHEMA,
            "status": WRITING_STATUS,
            "artifacts": {},
        }
    )
    _publish_writing_directory(destination, writing_manifest)

    write_npz_atomic(destination / "map_static.npz", payload_map_static)
    write_npz_atomic(destination / "q_em.npz", payload_q_em)
    write_npz_atomic(destination / "laplace.npz", payload_laplace)
    write_npz_atomic(destination / "diagnostics.npz", payload_diagnostics)
    for bag_id in bag_ids:
        write_npz_atomic(
            destination / "bags" / "{}.npz".format(bag_id),
            payload_bags[bag_id],
        )
    if payload_mcmc is not None:
        write_npz_atomic(destination / "mcmc_samples.npz", payload_mcmc)
    for bag_id, arrays in payload_trajectories.items():
        write_npz_atomic(
            destination
            / "trajectories"
            / bag_id
            / "selected_samples.npz",
            arrays,
        )

    artifacts: Dict[str, Any] = {
        "map_static": _written_descriptor(destination, "map_static.npz"),
        "q_em": _written_descriptor(destination, "q_em.npz"),
        "laplace": _written_descriptor(destination, "laplace.npz"),
        "diagnostics": _written_descriptor(destination, "diagnostics.npz"),
        "bags": {
            bag_id: _written_descriptor(
                destination, "bags/{}.npz".format(bag_id)
            )
            for bag_id in bag_ids
        },
    }
    if payload_mcmc is not None:
        artifacts["mcmc_samples"] = _written_descriptor(
            destination, "mcmc_samples.npz"
        )
    if payload_trajectories:
        artifacts["trajectories"] = {
            bag_id: _written_descriptor(
                destination,
                "trajectories/{}/selected_samples.npz".format(bag_id),
            )
            for bag_id in bag_ids
            if bag_id in payload_trajectories
        }

    complete_manifest = dict(manifest_metadata)
    complete_manifest.update(
        {
            "schema": BATCH_ESTIMATION_RUN_SCHEMA,
            "status": COMPLETE_STATUS,
            "artifacts": artifacts,
        }
    )
    # Validate the bytes read back from disk and their descriptors before the
    # completion manifest can become authoritative.
    _load_validated_run(destination, complete_manifest)
    write_json_atomic(destination / "manifest.json", complete_manifest)
    return load_batch_estimation_run(destination)


def load_batch_estimation_run(
    root: Union[str, Path]
) -> BatchEstimationRun:
    """Load and validate a completed v1 sparse-batch estimation run."""

    run_root = Path(root).expanduser().resolve()
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactValidationError(
            "batch run has no manifest.json: {}".format(run_root)
        )
    manifest = read_json(manifest_path)
    return _load_validated_run(run_root, manifest)


def replace_batch_estimation_run(
    root: Union[str, Path],
    *,
    expected_request_fingerprint: str,
    manifest_metadata: Mapping[str, Any],
    map_static: Mapping[str, np.ndarray],
    q_em: Mapping[str, np.ndarray],
    laplace: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, np.ndarray],
    bags: Mapping[str, Mapping[str, np.ndarray]],
    mcmc_samples: Optional[Mapping[str, np.ndarray]] = None,
    trajectories: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None,
) -> BatchEstimationRun:
    """Publish an upgraded complete run with rollback to the original."""

    destination = Path(root).expanduser().resolve()
    original = load_batch_estimation_run(destination)
    if original.manifest["request_fingerprint"] != expected_request_fingerprint:
        raise ArtifactValidationError(
            "current run request fingerprint changed before replacement"
        )
    if original.mcmc_samples is not None:
        raise ArtifactValidationError(
            "current run already contains posterior samples"
        )
    staging_parent = Path(
        tempfile.mkdtemp(
            prefix=".{}-upgrade-".format(destination.name),
            dir=str(destination.parent),
        )
    )
    candidate = staging_parent / "complete-run"
    backup_parent = Path(
        tempfile.mkdtemp(
            prefix=".{}-rollback-".format(destination.name),
            dir=str(destination.parent),
        )
    )
    backup = backup_parent / "original-run"
    original_moved = False
    replacement_published = False
    try:
        write_batch_estimation_run(
            candidate,
            manifest_metadata=manifest_metadata,
            map_static=map_static,
            q_em=q_em,
            laplace=laplace,
            diagnostics=diagnostics,
            bags=bags,
            mcmc_samples=mcmc_samples,
            trajectories=trajectories,
        )
        original_moved = True
        _publish_replacement_directory(destination, candidate, backup)
        original_moved = False
        replacement_published = True
        return load_batch_estimation_run(destination)
    finally:
        if replacement_published and backup.exists():
            shutil.rmtree(str(backup))
            original_moved = False
        if original_moved and not destination.exists() and backup.exists():
            os.replace(str(backup), str(destination))
        if staging_parent.exists():
            shutil.rmtree(str(staging_parent))
        if backup_parent.exists():
            shutil.rmtree(str(backup_parent))


def _publish_replacement_directory(
    destination: Path, candidate: Path, backup: Path
) -> None:
    """Swap two complete directories and restore the original on failure."""

    os.replace(str(destination), str(backup))
    try:
        os.replace(str(candidate), str(destination))
        _fsync_directory(destination.parent)
    except Exception:
        os.replace(str(backup), str(destination))
        _fsync_directory(destination.parent)
        raise


__all__ = [
    "BATCH_ESTIMATION_RUN_SCHEMA",
    "BatchEstimationRun",
    "file_sha256",
    "load_batch_estimation_run",
    "replace_batch_estimation_run",
    "write_batch_estimation_run",
]
