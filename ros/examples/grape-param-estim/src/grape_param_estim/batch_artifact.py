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
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .artifact_io import (
    ArtifactValidationError,
    IncompleteArtifactError,
    UnsupportedArtifactSchema,
    load_npz_strict,
    read_json,
)


BATCH_ESTIMATION_RUN_SCHEMA = "grape-param-estim/batch-estimation-run/v1"
STATIC_PARAMETER_DIMENSION = 18
DYNAMICS_RESIDUAL_DIMENSION = 6
COMPLETE_STATUS = "complete"

_BAG_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


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
    required = (
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
    _require_keys(manifest, required, "manifest")
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

    q_definition = _required_mapping(manifest, "q_definition", "manifest")
    _require_keys(
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
    _array(arrays, "force_effectiveness", (4,), location)
    _array(arrays, "torque_effectiveness", (4,), location)
    _array(arrays, "delay", (1,), location)
    q = _array(arrays, "q_diagonal", (DYNAMICS_RESIDUAL_DIMENSION,), location)
    _positive(q, "{}:q_diagonal".format(location))
    component_names = _strings(
        arrays, "objective_component_names", None, location, unique=True
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


_LAPLACE_KEYS = (
    "reduced_likelihood_hessian",
    "reduced_posterior_hessian",
    "covariance",
    "eigenvalues",
    "eigenvectors",
    "effective_rank",
    "exact_ridge_direction",
    "ridge_alignment",
    "condition_number",
    "delay_profile_grid",
    "delay_profile_objective",
    "delay_local_uncertainty",
)


def _validate_laplace(arrays: Mapping[str, np.ndarray], location: str) -> None:
    _require_keys(arrays, _LAPLACE_KEYS, location)
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
    grid = _array(arrays, "delay_profile_grid", (None,), location)
    if grid.size == 0:
        raise ArtifactValidationError(
            "{}:delay_profile_grid must not be empty".format(location)
        )
    _strictly_increasing(grid, "{}:delay_profile_grid".format(location))
    objective = _array(
        arrays,
        "delay_profile_objective",
        (grid.size,),
        location,
        finite=False,
    )
    if np.any(np.isnan(objective)):
        raise ArtifactValidationError(
            "{}:delay_profile_objective must not contain NaN".format(location)
        )
    uncertainty = _array(arrays, "delay_local_uncertainty", (1,), location)
    _positive(uncertainty, "{}:delay_local_uncertainty".format(location))


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


def _validate_diagnostics(
    arrays: Mapping[str, np.ndarray], bag_ids: Tuple[str, ...], location: str
) -> None:
    _require_keys(arrays, _DIAGNOSTIC_KEYS, location)
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
    "velocity_time",
    "velocity_record_time",
    "velocity",
    "velocity_valid",
    "velocity_covariance",
    "gyro_time",
    "gyro_record_time",
    "gyro",
    "gyro_valid",
    "gyro_covariance",
    "accelerometer_time",
    "accelerometer_record_time",
    "accelerometer",
    "accelerometer_valid",
    "accelerometer_covariance",
    "thrust_command_time",
    "thrust_command_record_time",
    "thrust_command",
    "thrust_command_valid",
    "gimbal_command_time",
    "gimbal_command_record_time",
    "gimbal_command",
    "gimbal_command_valid",
    "gimbal_observation_time",
    "gimbal_observation_record_time",
    "gimbal_observation",
    "gimbal_observation_valid",
    "gimbal_observation_covariance",
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


def _validate_bag(
    arrays: Mapping[str, np.ndarray], bag_id: str, location: str
) -> None:
    _require_keys(arrays, _BAG_KEYS, location)
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
    sample_ids = _id_vector(arrays, "sample_id", location)
    count = sample_ids.size
    _array(arrays, "chain_id", (count,), location, kind="integer")
    _array(arrays, "draw_index", (count,), location, kind="integer")
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
    _array(arrays, "cog", (count, 3), location)
    _array(arrays, "force_effectiveness", (count, 4), location)
    _array(arrays, "torque_effectiveness", (count, 4), location)
    _array(arrays, "delay", (count,), location)
    for key in (
        "log_posterior",
        "log_likelihood_approximation",
        "log_determinant_term",
    ):
        _array(arrays, key, (count,), location)
    _strings(arrays, "accepted_kernel", count, location)
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
    "conditional_objective",
)


def _validate_trajectory_subset(
    arrays: Mapping[str, np.ndarray],
    mcmc_sample_ids: np.ndarray,
    location: str,
) -> None:
    _require_keys(arrays, _TRAJECTORY_KEYS, location)
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


def _freeze(arrays: Dict[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    for value in arrays.values():
        value.setflags(write=False)
    return arrays


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
    _validate_diagnostics(diagnostics, bag_ids, str(core_paths["diagnostics"]))

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
        trajectories[bag_id] = _freeze(arrays)

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


__all__ = [
    "BATCH_ESTIMATION_RUN_SCHEMA",
    "BatchEstimationRun",
    "file_sha256",
    "load_batch_estimation_run",
]
