"""Strict stage-boundary artifact for a diagonal body-wrench ``Q`` estimate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, Union
import zipfile

import numpy as np

from grape_param_estim.artifact_io import (
    ArtifactStateError,
    ArtifactValidationError,
    CANCELLED_STATUS,
    COMPLETE_STATUS,
    IncompleteArtifactError,
    MANIFEST_NAME,
    UnsupportedArtifactSchema,
    WRITING_STATUS,
    canonical_json_bytes,
    read_json,
    request_fingerprint as compute_request_fingerprint,
    write_json_atomic,
    write_npz_atomic,
)
from grape_param_estim.diagonal_q import (
    BODY_WRENCH_COMPONENT_ORDER,
    BODY_WRENCH_DIMENSION,
    BODY_WRENCH_FRAME,
    BODY_WRENCH_VARIANCE_UNITS,
    BodyWrenchDiagonalCovariance,
    DiagonalQEmSufficientStatistics,
    shared_diagonal_q_m_step,
)
from grape_param_estim.diagonal_q_em import (
    LOG_Q_TOLERANCE_TERMINATION,
    MAXIMUM_ITERATIONS_TERMINATION,
    DiagonalQBagExpectation,
    DiagonalQEmResult,
    DiagonalQInitialPilot,
    initial_diagonal_q_from_pilots,
)


DIAGONAL_Q_ESTIMATE_SCHEMA = (
    "grape-param-estim/diagonal-wrench-q-estimate/v1"
)

_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONFIGURATION_FINGERPRINT = re.compile(
    r"(?:(?:complete|incomplete):[0-9a-f]{64}"
    r"|manual-group:sha256:[0-9a-f]{64})\Z"
)
_DIGEST = _FINGERPRINT
_TRACE_KEYS = (
    "iteration",
    "input_stationary_variance",
    "raw_stationary_variance",
    "output_stationary_variance",
    "floor_applied",
    "maximum_absolute_log_q_change",
    "approx_log_likelihood",
    "bag_approx_log_likelihood",
)
_BAG_ARRAY_KEYS = (
    "bag_id",
    "source_path",
    "source_sha256",
    "source_size_bytes",
    "selected_interval_local_seconds",
    "effective_interval_local_seconds",
    "episode_index",
    "configuration_fingerprint",
    "fixed_model_fingerprint",
    "constant_delay_seconds",
    "times",
    "correlation_time",
    "observation_covariance_fingerprint",
    "observation_translation_covariance",
    "observation_rotation_covariance",
    "last_em_member_count",
    "last_em_times",
    "last_em_correlation_time",
    "last_em_initial_second_moment",
    "last_em_transition_second_moment",
    "smoothed_wrench_input_stationary_variance",
    "approx_log_likelihood",
    "smoothed_wrench",
)
_MANIFEST_KEYS = {
    "schema",
    "status",
    "run_id",
    "stage_id",
    "request_fingerprint",
    "project_fingerprint",
    "stage_input_fingerprint",
    "implementation",
    "selected_bag_ids",
    "selected_intervals",
    "body_wrench",
    "bags",
    "em",
    "initial_stationary_variance",
    "final_stationary_variance",
    "final_stationary_standard_deviation",
    "smoothed_wrench_input_stationary_variance",
    "terminal_implied_raw_stationary_variance",
    "terminal_implied_stationary_variance",
    "artifacts",
}
_BAG_METADATA_KEYS = (
    "source_path",
    "source_sha256",
    "source_size_bytes",
    "selected_interval_local_seconds",
    "effective_interval_local_seconds",
    "episode_index",
    "configuration_fingerprint",
    "fixed_model_fingerprint",
    "fixed_model_provenance",
    "constant_delay_seconds",
    "time_basis",
    "fixed_observation_covariance",
    "boundary_count",
    "member_count",
    "last_em_member_count",
    "correlation_time",
    "pilot_stationary_standard_deviation",
    "final_approx_log_likelihood",
)
_TIME_BASIS = "episode_relative_seconds"


@dataclass(frozen=True)
class DiagonalQArtifactBagInput:
    """Immutable scientific input provenance for one selected bag."""

    bag_id: str
    source_path: str
    source_sha256: str
    source_size_bytes: int
    selected_interval_local_seconds: Tuple[float, float]
    effective_interval_local_seconds: Tuple[float, float]
    episode_index: int
    configuration_fingerprint: str
    fixed_model_provenance: Mapping[str, Any]
    constant_delay_seconds: float
    translation_covariance: np.ndarray
    rotation_covariance: np.ndarray
    fixed_r_provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        identifier = _string(self.bag_id, "bag_id")
        if "\x00" in identifier:
            raise ArtifactValidationError("bag_id cannot contain NUL")
        source_path = _string(self.source_path, "source_path")
        if "\x00" in source_path:
            raise ArtifactValidationError("source_path cannot contain NUL")
        source_sha256 = _raw_sha256(self.source_sha256, "source_sha256")
        source_size_bytes = _positive_integer(
            self.source_size_bytes, "source_size_bytes"
        )
        if source_size_bytes > np.iinfo(np.int64).max:
            raise ArtifactValidationError(
                "source_size_bytes exceeds the portable int64 range"
            )
        raw_interval = self.selected_interval_local_seconds
        if not isinstance(raw_interval, (list, tuple)) or len(raw_interval) != 2:
            raise ArtifactValidationError(
                "selected_interval_local_seconds must contain two bounds"
            )
        interval = _interval(
            list(raw_interval), "selected_interval_local_seconds"
        )
        raw_effective_interval = self.effective_interval_local_seconds
        if (
            not isinstance(raw_effective_interval, (list, tuple))
            or len(raw_effective_interval) != 2
        ):
            raise ArtifactValidationError(
                "effective_interval_local_seconds must contain two bounds"
            )
        effective_interval = _interval(
            list(raw_effective_interval),
            "effective_interval_local_seconds",
        )
        tolerance = 2.0e-7
        if (
            effective_interval[0] < interval[0] - tolerance
            or effective_interval[1] > interval[1] + tolerance
        ):
            raise ArtifactValidationError(
                "effective interval must lie inside the selected interval"
            )
        episode_index = _nonnegative_integer(
            self.episode_index, "episode_index"
        )
        if episode_index > np.iinfo(np.int64).max:
            raise ArtifactValidationError(
                "episode_index exceeds the portable int64 range"
            )
        configuration_fingerprint = _string(
            self.configuration_fingerprint, "configuration_fingerprint"
        )
        if _CONFIGURATION_FINGERPRINT.fullmatch(
            configuration_fingerprint
        ) is None:
            raise ArtifactValidationError(
                "configuration_fingerprint must be a complete, incomplete, "
                "or manual-group SHA256"
            )
        fixed_model_provenance = _normalised_json_mapping(
            self.fixed_model_provenance, "fixed_model_provenance"
        )
        constant_delay_seconds = _finite_float(
            self.constant_delay_seconds, "constant_delay_seconds"
        )
        if constant_delay_seconds < 0.0:
            raise ArtifactValidationError(
                "constant_delay_seconds cannot be negative"
            )
        translation_covariance = _covariance3(
            self.translation_covariance, "translation_covariance"
        )
        rotation_covariance = _covariance3(
            self.rotation_covariance, "rotation_covariance"
        )
        provenance = _normalised_json_mapping(
            self.fixed_r_provenance, "fixed_r_provenance"
        )
        object.__setattr__(self, "bag_id", identifier)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "source_sha256", source_sha256)
        object.__setattr__(self, "source_size_bytes", source_size_bytes)
        object.__setattr__(
            self, "selected_interval_local_seconds", interval
        )
        object.__setattr__(
            self, "effective_interval_local_seconds", effective_interval
        )
        object.__setattr__(self, "episode_index", episode_index)
        object.__setattr__(
            self, "configuration_fingerprint", configuration_fingerprint
        )
        object.__setattr__(
            self, "fixed_model_provenance", fixed_model_provenance
        )
        object.__setattr__(
            self, "constant_delay_seconds", constant_delay_seconds
        )
        object.__setattr__(
            self, "translation_covariance", translation_covariance
        )
        object.__setattr__(
            self, "rotation_covariance", rotation_covariance
        )
        object.__setattr__(self, "fixed_r_provenance", provenance)

    @property
    def observation_covariance_fingerprint(self) -> str:
        return _observation_covariance_fingerprint(
            self.fixed_r_provenance,
            self.translation_covariance,
            self.rotation_covariance,
        )

    @property
    def fixed_model_fingerprint(self) -> str:
        return compute_request_fingerprint(self.fixed_model_provenance)


@dataclass(frozen=True)
class DiagonalQArtifactBundle:
    """A fully validated, detached diagonal-Q stage-boundary bundle."""

    root: Path
    manifest: Mapping[str, Any]
    trace: Mapping[str, np.ndarray]
    bag_inputs: Tuple[DiagonalQArtifactBagInput, ...]
    pilots: Tuple[DiagonalQInitialPilot, ...]
    last_em_statistics: Tuple[DiagonalQEmSufficientStatistics, ...]
    expectations: Tuple[DiagonalQBagExpectation, ...]
    covariance: BodyWrenchDiagonalCovariance

    @property
    def bag_ids(self) -> Tuple[str, ...]:
        return tuple(value.bag_id for value in self.pilots)


def _exact_keys(
    value: Any, expected: Sequence[str], location: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(
            "{} must be an object".format(location)
        )
    expected_set = set(expected)
    actual = set(value)
    missing = sorted(expected_set - actual)
    extra = sorted(actual - expected_set)
    if missing or extra:
        raise ArtifactValidationError(
            "{} keys differ from schema; missing={}, extra={}".format(
                location, missing, extra
            )
        )
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactValidationError(
            "{} must be a non-empty string".format(location)
        )
    return value


def _fingerprint(value: Any, location: str) -> str:
    selected = _string(value, location)
    if _FINGERPRINT.fullmatch(selected) is None:
        raise ArtifactValidationError(
            "{} must be a lowercase SHA256 fingerprint".format(location)
        )
    return selected


def _positive_integer(value: Any, location: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ArtifactValidationError(
            "{} must be a positive integer".format(location)
        )
    selected = int(value)
    if selected <= 0:
        raise ArtifactValidationError(
            "{} must be a positive integer".format(location)
        )
    return selected


def _nonnegative_integer(value: Any, location: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ArtifactValidationError(
            "{} must be a non-negative integer".format(location)
        )
    selected = int(value)
    if selected < 0:
        raise ArtifactValidationError(
            "{} must be a non-negative integer".format(location)
        )
    return selected


def _raw_sha256(value: Any, location: str) -> str:
    selected = _string(value, location)
    if _RAW_SHA256.fullmatch(selected) is None:
        raise ArtifactValidationError(
            "{} must be a lowercase raw SHA256 digest".format(location)
        )
    return selected


def _positive_float(value: Any, location: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ArtifactValidationError(
            "{} must be finite and positive".format(location)
        )
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ArtifactValidationError(
            "{} must be finite and positive".format(location)
        ) from error
    if not np.isfinite(selected) or selected <= 0.0:
        raise ArtifactValidationError(
            "{} must be finite and positive".format(location)
        )
    return selected


def _finite_float(value: Any, location: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ArtifactValidationError(
            "{} must be finite".format(location)
        )
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ArtifactValidationError(
            "{} must be finite".format(location)
        ) from error
    if not np.isfinite(selected):
        raise ArtifactValidationError(
            "{} must be finite".format(location)
        )
    return selected


def _covariance3(value: Any, location: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ArtifactValidationError(
            "{} must be a finite symmetric positive-definite 3x3 matrix".format(
                location
            )
        ) from error
    if (
        result.shape != (3, 3)
        or np.any(~np.isfinite(result))
        or not np.allclose(result, result.T, rtol=0.0, atol=1.0e-12)
    ):
        raise ArtifactValidationError(
            "{} must be a finite symmetric positive-definite 3x3 matrix".format(
                location
            )
        )
    symmetric = 0.5 * (result + result.T)
    try:
        eigenvalues = np.linalg.eigvalsh(symmetric)
    except np.linalg.LinAlgError as error:
        raise ArtifactValidationError(
            "{} must be positive definite".format(location)
        ) from error
    if np.any(~np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0.0):
        raise ArtifactValidationError(
            "{} must be positive definite".format(location)
        )
    return symmetric.copy()


def _component_vector(
    value: Any,
    location: str,
    *,
    positive: bool,
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        qualifier = "positive" if positive else "non-negative"
        raise ArtifactValidationError(
            "{} must contain six finite {} values".format(
                location, qualifier
            )
        ) from error
    invalid_sign = result <= 0.0 if positive else result < 0.0
    if (
        result.shape != (BODY_WRENCH_DIMENSION,)
        or np.any(~np.isfinite(result))
        or np.any(invalid_sign)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ArtifactValidationError(
            "{} must contain six finite {} values".format(
                location, qualifier
            )
        )
    return result.copy()


def _interval(value: Any, location: str) -> Tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ArtifactValidationError(
            "{} must contain [start, end]".format(location)
        )
    start = _finite_float(value[0], "{}[0]".format(location))
    end = _finite_float(value[1], "{}[1]".format(location))
    if end <= start or not np.isfinite(end - start):
        raise ArtifactValidationError(
            "{} must contain increasing bounds with finite duration".format(
                location
            )
        )
    return start, end


def _normalised_json_mapping(
    value: Mapping[str, Any], location: str
) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ArtifactValidationError(
            "{} must be a non-empty object".format(location)
        )
    try:
        normalised = json.loads(canonical_json_bytes(value).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(
            "{} cannot be canonicalised".format(location)
        ) from error
    if not isinstance(normalised, dict) or not normalised:
        raise ArtifactValidationError(
            "{} must be a non-empty object".format(location)
        )
    return normalised


def _implementation_provenance(
    value: Any, location: str
) -> Mapping[str, Any]:
    selected = _normalised_json_mapping(value, location)
    _exact_keys(
        selected,
        ("algorithm_version", "source_revision", "source_dirty"),
        location,
    )
    for key in ("algorithm_version", "source_revision"):
        text = _string(selected[key], "{}.{}".format(location, key))
        if "\x00" in text:
            raise ArtifactValidationError(
                "{}.{} cannot contain NUL".format(location, key)
            )
    if not isinstance(selected["source_dirty"], bool):
        raise ArtifactValidationError(
            "{}.source_dirty must be boolean".format(location)
        )
    return selected


def _observation_covariance_fingerprint(
    provenance: Mapping[str, Any],
    translation_covariance: np.ndarray,
    rotation_covariance: np.ndarray,
) -> str:
    return compute_request_fingerprint(
        {
            "provenance": _normalised_json_mapping(
                provenance, "fixed observation covariance provenance"
            ),
            "translation_covariance": _covariance3(
                translation_covariance, "translation_covariance"
            ).tolist(),
            "rotation_covariance": _covariance3(
                rotation_covariance, "rotation_covariance"
            ).tolist(),
        }
    )


def _descriptor(
    value: Any, location: str, require_digest: bool
) -> Mapping[str, Any]:
    selected = _exact_keys(
        value, ("path", "sha256", "size_bytes"), location
    )
    relative = _string(selected["path"], "{}.path".format(location))
    relative_path = Path(relative)
    raw_parts = relative.split("/")
    if (
        "\x00" in relative
        or relative_path.is_absolute()
        or relative_path.as_posix() != relative
        or any(part in {"", ".", ".."} for part in raw_parts)
        or any("\\" in part or ":" in part for part in raw_parts)
    ):
        raise ArtifactValidationError(
            "{}.path must stay inside the bundle".format(location)
        )
    digest = selected["sha256"]
    size = selected["size_bytes"]
    if require_digest:
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            raise ArtifactValidationError(
                "{}.sha256 must contain a SHA256 digest".format(location)
            )
        _positive_integer(size, "{}.size_bytes".format(location))
    elif not (
        (digest is None and size is None)
        or (
            isinstance(digest, str)
            and _DIGEST.fullmatch(digest) is not None
            and isinstance(size, int)
            and not isinstance(size, bool)
            and size > 0
        )
    ):
        raise ArtifactValidationError(
            "{} writing descriptor must be empty or complete".format(location)
        )
    return selected


def _validate_manifest(
    manifest: Mapping[str, Any], *, require_complete: bool
):
    if not isinstance(manifest, Mapping):
        raise ArtifactValidationError("manifest must be an object")
    if manifest.get("schema") != DIAGONAL_Q_ESTIMATE_SCHEMA:
        raise UnsupportedArtifactSchema(
            "unsupported artifact schema {!r}".format(
                manifest.get("schema")
            )
        )
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
    expected_keys = set(_MANIFEST_KEYS)
    if status == CANCELLED_STATUS:
        expected_keys.add("cancellation_reason")
    _exact_keys(manifest, tuple(expected_keys), "manifest")
    if status == CANCELLED_STATUS:
        _string(manifest["cancellation_reason"], "manifest.cancellation_reason")

    _string(manifest["run_id"], "manifest.run_id")
    _string(manifest["stage_id"], "manifest.stage_id")
    _fingerprint(
        manifest["request_fingerprint"], "manifest.request_fingerprint"
    )
    _fingerprint(
        manifest["project_fingerprint"], "manifest.project_fingerprint"
    )
    _fingerprint(
        manifest["stage_input_fingerprint"],
        "manifest.stage_input_fingerprint",
    )
    implementation = _exact_keys(
        manifest["implementation"],
        ("provenance", "fingerprint"),
        "manifest.implementation",
    )
    implementation_provenance = _implementation_provenance(
        implementation["provenance"],
        "manifest.implementation.provenance",
    )
    implementation_fingerprint = _fingerprint(
        implementation["fingerprint"],
        "manifest.implementation.fingerprint",
    )
    if compute_request_fingerprint(
        implementation_provenance
    ) != implementation_fingerprint:
        raise ArtifactValidationError(
            "manifest implementation fingerprint is inconsistent"
        )
    raw_bag_ids = manifest["selected_bag_ids"]
    if not isinstance(raw_bag_ids, list) or not raw_bag_ids:
        raise ArtifactValidationError(
            "manifest.selected_bag_ids must be a non-empty list"
        )
    bag_ids = tuple(
        _string(value, "manifest.selected_bag_ids")
        for value in raw_bag_ids
    )
    if bag_ids != tuple(sorted(bag_ids)) or len(set(bag_ids)) != len(bag_ids):
        raise ArtifactValidationError(
            "manifest.selected_bag_ids must be sorted and unique"
        )

    intervals = _exact_keys(
        manifest["selected_intervals"], bag_ids, "manifest.selected_intervals"
    )
    for bag_id in bag_ids:
        _interval(
            intervals[bag_id],
            "manifest.selected_intervals.{!r}".format(bag_id),
        )

    body_wrench = _exact_keys(
        manifest["body_wrench"],
        ("frame", "component_order", "variance_units"),
        "manifest.body_wrench",
    )
    if body_wrench["frame"] != BODY_WRENCH_FRAME:
        raise ArtifactValidationError("body-wrench frame is not canonical")
    if body_wrench["component_order"] != list(BODY_WRENCH_COMPONENT_ORDER):
        raise ArtifactValidationError("body-wrench component order changed")
    if body_wrench["variance_units"] != list(BODY_WRENCH_VARIANCE_UNITS):
        raise ArtifactValidationError("body-wrench variance units changed")

    bag_metadata = _exact_keys(manifest["bags"], bag_ids, "manifest.bags")
    for bag_id in bag_ids:
        location = "manifest.bags.{!r}".format(bag_id)
        metadata = _exact_keys(
            bag_metadata[bag_id],
            _BAG_METADATA_KEYS,
            location,
        )
        source_path = _string(
            metadata["source_path"], "{}.source_path".format(location)
        )
        if "\x00" in source_path:
            raise ArtifactValidationError(
                "{}.source_path cannot contain NUL".format(location)
            )
        _raw_sha256(
            metadata["source_sha256"],
            "{}.source_sha256".format(location),
        )
        source_size_bytes = _positive_integer(
            metadata["source_size_bytes"],
            "{}.source_size_bytes".format(location),
        )
        if source_size_bytes > np.iinfo(np.int64).max:
            raise ArtifactValidationError(
                "{}.source_size_bytes exceeds int64".format(location)
            )
        selected_interval = _interval(
            metadata["selected_interval_local_seconds"],
            "{}.selected_interval_local_seconds".format(location),
        )
        if selected_interval != tuple(intervals[bag_id]):
            raise ArtifactValidationError(
                "{}.selected_interval_local_seconds does not match "
                "manifest.selected_intervals".format(location)
            )
        effective_interval = _interval(
            metadata["effective_interval_local_seconds"],
            "{}.effective_interval_local_seconds".format(location),
        )
        tolerance = 2.0e-7
        if (
            effective_interval[0] < selected_interval[0] - tolerance
            or effective_interval[1] > selected_interval[1] + tolerance
        ):
            raise ArtifactValidationError(
                "{}.effective interval must lie inside selected interval".format(
                    location
                )
            )
        episode_index = _nonnegative_integer(
            metadata["episode_index"],
            "{}.episode_index".format(location),
        )
        if episode_index > np.iinfo(np.int64).max:
            raise ArtifactValidationError(
                "{}.episode_index exceeds int64".format(location)
            )
        configuration_fingerprint = _string(
            metadata["configuration_fingerprint"],
            "{}.configuration_fingerprint".format(location),
        )
        if _CONFIGURATION_FINGERPRINT.fullmatch(
            configuration_fingerprint
        ) is None:
            raise ArtifactValidationError(
                "{}.configuration_fingerprint must be a complete, incomplete, "
                "or manual-group SHA256".format(location)
            )
        fixed_model_provenance = _normalised_json_mapping(
            metadata["fixed_model_provenance"],
            "{}.fixed_model_provenance".format(location),
        )
        fixed_model_fingerprint = _fingerprint(
            metadata["fixed_model_fingerprint"],
            "{}.fixed_model_fingerprint".format(location),
        )
        if compute_request_fingerprint(
            fixed_model_provenance
        ) != fixed_model_fingerprint:
            raise ArtifactValidationError(
                "{}.fixed model fingerprint is inconsistent".format(location)
            )
        constant_delay = _finite_float(
            metadata["constant_delay_seconds"],
            "{}.constant_delay_seconds".format(location),
        )
        if constant_delay < 0.0:
            raise ArtifactValidationError(
                "{}.constant_delay_seconds cannot be negative".format(location)
            )
        if metadata["time_basis"] != _TIME_BASIS:
            raise ArtifactValidationError(
                "{}.time_basis must be {!r}".format(location, _TIME_BASIS)
            )
        fixed_r = _exact_keys(
            metadata["fixed_observation_covariance"],
            ("fixed", "provenance", "covariance_fingerprint"),
            "{}.fixed_observation_covariance".format(location),
        )
        if fixed_r["fixed"] is not True:
            raise ArtifactValidationError(
                "{}.fixed_observation_covariance must be fixed".format(
                    location
                )
            )
        _normalised_json_mapping(
            fixed_r["provenance"],
            "{}.fixed_observation_covariance.provenance".format(location),
        )
        _fingerprint(
            fixed_r["covariance_fingerprint"],
            "{}.fixed_observation_covariance.covariance_fingerprint".format(
                location
            ),
        )
        boundary_count = _positive_integer(
            metadata["boundary_count"], "{}.boundary_count".format(location)
        )
        if boundary_count < 2:
            raise ArtifactValidationError(
                "{}.boundary_count must include a positive-duration interval".format(
                    location
                )
            )
        _positive_integer(
            metadata["member_count"], "{}.member_count".format(location)
        )
        _positive_integer(
            metadata["last_em_member_count"],
            "{}.last_em_member_count".format(location),
        )
        _positive_float(
            metadata["correlation_time"],
            "{}.correlation_time".format(location),
        )
        _component_vector(
            metadata["pilot_stationary_standard_deviation"],
            "{}.pilot_stationary_standard_deviation".format(location),
            positive=False,
        )
        _finite_float(
            metadata["final_approx_log_likelihood"],
            "{}.final_approx_log_likelihood".format(location),
        )

    em = _exact_keys(
        manifest["em"],
        (
            "maximum_iterations",
            "log_q_tolerance",
            "component_floor",
            "completed_iterations",
            "converged",
            "termination_reason",
            "smoothed_wrench_semantics",
        ),
        "manifest.em",
    )
    maximum_iterations = _positive_integer(
        em["maximum_iterations"], "manifest.em.maximum_iterations"
    )
    _positive_float(em["log_q_tolerance"], "manifest.em.log_q_tolerance")
    _component_vector(
        em["component_floor"],
        "manifest.em.component_floor",
        positive=True,
    )
    completed_iterations = _positive_integer(
        em["completed_iterations"], "manifest.em.completed_iterations"
    )
    if completed_iterations > maximum_iterations:
        raise ArtifactValidationError(
            "completed EM iterations exceed maximum_iterations"
        )
    if not isinstance(em["converged"], bool):
        raise ArtifactValidationError("manifest.em.converged must be boolean")
    termination = _string(
        em["termination_reason"], "manifest.em.termination_reason"
    )
    if (
        em["smoothed_wrench_semantics"]
        != "terminal_e_step_conditioned_on_final_q"
    ):
        raise ArtifactValidationError(
            "manifest.em.smoothed_wrench_semantics is unsupported"
        )
    if em["converged"]:
        if termination != LOG_Q_TOLERANCE_TERMINATION:
            raise ArtifactValidationError(
                "converged EM must terminate by log_q_tolerance"
            )
    elif (
        termination != MAXIMUM_ITERATIONS_TERMINATION
        or completed_iterations != maximum_iterations
    ):
        raise ArtifactValidationError(
            "non-converged EM must exhaust maximum_iterations"
        )

    initial_variance = _component_vector(
        manifest["initial_stationary_variance"],
        "manifest.initial_stationary_variance",
        positive=True,
    )
    final_variance = _component_vector(
        manifest["final_stationary_variance"],
        "manifest.final_stationary_variance",
        positive=True,
    )
    final_standard_deviation = _component_vector(
        manifest["final_stationary_standard_deviation"],
        "manifest.final_stationary_standard_deviation",
        positive=True,
    )
    if not np.array_equal(final_standard_deviation, np.sqrt(final_variance)):
        raise ArtifactValidationError(
            "final standard deviation does not match final variance"
        )
    smoothed_wrench_input_variance = _component_vector(
        manifest["smoothed_wrench_input_stationary_variance"],
        "manifest.smoothed_wrench_input_stationary_variance",
        positive=True,
    )
    if not np.array_equal(
        smoothed_wrench_input_variance, final_variance
    ):
        raise ArtifactValidationError(
            "terminal smoothed wrench must be conditioned on final Q"
        )
    terminal_implied_raw_variance = _component_vector(
        manifest["terminal_implied_raw_stationary_variance"],
        "manifest.terminal_implied_raw_stationary_variance",
        positive=False,
    )
    terminal_implied_variance = _component_vector(
        manifest["terminal_implied_stationary_variance"],
        "manifest.terminal_implied_stationary_variance",
        positive=True,
    )
    if not np.array_equal(
        terminal_implied_variance,
        np.maximum(
            terminal_implied_raw_variance,
            np.asarray(em["component_floor"], dtype=float),
        ),
    ):
        raise ArtifactValidationError(
            "terminal implied Q does not match its raw value and floor"
        )

    artifacts = _exact_keys(
        manifest["artifacts"], ("em_trace", "bags"), "manifest.artifacts"
    )
    require_digest = status == COMPLETE_STATUS
    _descriptor(
        artifacts["em_trace"],
        "manifest.artifacts.em_trace",
        require_digest,
    )
    bag_artifacts = _exact_keys(
        artifacts["bags"], bag_ids, "manifest.artifacts.bags"
    )
    for bag_id in bag_ids:
        _descriptor(
            bag_artifacts[bag_id],
            "manifest.artifacts.bags.{!r}".format(bag_id),
            require_digest,
        )

    return (
        status,
        bag_ids,
        bag_metadata,
        em,
        initial_variance,
        final_variance,
        smoothed_wrench_input_variance,
        terminal_implied_raw_variance,
        terminal_implied_variance,
        artifacts,
    )


def _artifact_path(root: Path, descriptor: Mapping[str, Any], location: str) -> Path:
    relative = Path(str(descriptor["path"]))
    root_resolved = root.resolve()
    lexical = root_resolved / relative
    if lexical.is_symlink():
        raise ArtifactValidationError(
            "{}.path must not be a symbolic link".format(location)
        )
    candidate = lexical.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ArtifactValidationError(
            "{}.path resolves outside the bundle".format(location)
        )
    if not candidate.is_file():
        raise ArtifactValidationError(
            "{}.path does not name an existing file".format(location)
        )
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise ArtifactValidationError(
            "cannot hash artifact {}: {}".format(path, error)
        ) from error
    return "sha256:{}".format(digest.hexdigest())


def _verified_artifact_payload(
    root: Path, descriptor: Mapping[str, Any], location: str
) -> Tuple[Path, bytes]:
    path = _artifact_path(root, descriptor, location)
    descriptor_fd = None
    try:
        descriptor_fd = os.open(
            str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        metadata = os.fstat(descriptor_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactValidationError(
                "{} must be a regular payload file".format(location)
            )
        size = metadata.st_size
        chunks = []
        while True:
            block = os.read(descriptor_fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
    except OSError as error:
        raise ArtifactValidationError(
            "cannot read artifact {}: {}".format(path, error)
        ) from error
    finally:
        if descriptor_fd is not None:
            os.close(descriptor_fd)
    if size != descriptor["size_bytes"]:
        raise ArtifactValidationError(
            "{} size does not match the manifest".format(location)
        )
    payload = b"".join(chunks)
    digest = "sha256:{}".format(hashlib.sha256(payload).hexdigest())
    if digest != descriptor["sha256"]:
        raise ArtifactValidationError(
            "{} SHA256 digest does not match the manifest".format(location)
        )
    return path, payload


def _load_npz_exact(
    payload: bytes, path: Path, keys: Sequence[str]
) -> Dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile) as error:
        raise ArtifactValidationError(
            "cannot inspect NPZ artifact {}: {}".format(path, error)
        ) from error
    expected_names = {"{}.npy".format(key) for key in keys}
    if len(names) != len(set(names)) or set(names) != expected_names:
        raise ArtifactValidationError(
            "{} ZIP members differ from the NPZ schema".format(path)
        )
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != set(keys):
                raise ArtifactValidationError(
                    "{} arrays differ from the NPZ schema".format(path)
                )
            arrays = {key: np.asarray(archive[key]).copy() for key in keys}
    except ArtifactValidationError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ArtifactValidationError(
            "cannot load NPZ artifact {}: {}".format(path, error)
        ) from error
    _exact_keys(arrays, keys, str(path))
    return arrays


def _numeric_array(
    arrays: Mapping[str, np.ndarray],
    key: str,
    shape: Tuple[int, ...],
    location: str,
) -> np.ndarray:
    value = arrays[key]
    if (
        value.shape != shape
        or not np.issubdtype(value.dtype, np.floating)
        or np.any(~np.isfinite(value))
    ):
        raise ArtifactValidationError(
            "{}:{} must be a finite floating-point {} array".format(
                location, key, shape
            )
        )
    return value


def _unicode_scalar_array(
    arrays: Mapping[str, np.ndarray], key: str, location: str
) -> str:
    value = arrays[key]
    if value.shape != (1,) or value.dtype.kind != "U":
        raise ArtifactValidationError(
            "{}:{} must be a one-element Unicode array".format(location, key)
        )
    return str(value[0])


def _integer_scalar_array(
    arrays: Mapping[str, np.ndarray], key: str, location: str
) -> int:
    value = arrays[key]
    if value.shape != (1,) or not np.issubdtype(value.dtype, np.integer):
        raise ArtifactValidationError(
            "{}:{} must be a one-element integer array".format(location, key)
        )
    return int(value[0])


def _validate_episode_relative_times(
    times: np.ndarray,
    selected_interval: Tuple[float, float],
    location: str,
) -> None:
    if times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
        raise ArtifactValidationError(
            "{} must start at zero and be strictly increasing".format(location)
        )
    duration = selected_interval[1] - selected_interval[0]
    tolerance = max(2.0e-7, 1.0e-9 * max(1.0, abs(duration)))
    if abs(float(times[-1]) - duration) > tolerance:
        raise ArtifactValidationError(
            "{} duration does not match selected_interval_local_seconds".format(
                location
            )
        )


def _validate_trace(
    arrays: Mapping[str, np.ndarray],
    bag_ids: Tuple[str, ...],
    em: Mapping[str, Any],
    initial_variance: np.ndarray,
    final_variance: np.ndarray,
    location: str,
) -> None:
    iteration_count = int(em["completed_iterations"])
    bag_count = len(bag_ids)
    iteration = arrays["iteration"]
    if (
        iteration.shape != (iteration_count,)
        or not np.issubdtype(iteration.dtype, np.integer)
        or not np.array_equal(
            iteration, np.arange(1, iteration_count + 1, dtype=iteration.dtype)
        )
    ):
        raise ArtifactValidationError(
            "{}:iteration must be contiguous and one-based".format(location)
        )
    input_variance = _numeric_array(
        arrays,
        "input_stationary_variance",
        (iteration_count, BODY_WRENCH_DIMENSION),
        location,
    )
    raw_variance = _numeric_array(
        arrays,
        "raw_stationary_variance",
        (iteration_count, BODY_WRENCH_DIMENSION),
        location,
    )
    output_variance = _numeric_array(
        arrays,
        "output_stationary_variance",
        (iteration_count, BODY_WRENCH_DIMENSION),
        location,
    )
    log_change = _numeric_array(
        arrays,
        "maximum_absolute_log_q_change",
        (iteration_count,),
        location,
    )
    total_likelihood = _numeric_array(
        arrays,
        "approx_log_likelihood",
        (iteration_count,),
        location,
    )
    bag_likelihood = _numeric_array(
        arrays,
        "bag_approx_log_likelihood",
        (iteration_count, bag_count),
        location,
    )
    floor_applied = arrays["floor_applied"]
    if floor_applied.shape != (
        iteration_count,
        BODY_WRENCH_DIMENSION,
    ) or not np.issubdtype(floor_applied.dtype, np.bool_):
        raise ArtifactValidationError(
            "{}:floor_applied must be a boolean trace".format(location)
        )
    if (
        np.any(input_variance <= 0.0)
        or np.any(raw_variance < 0.0)
        or np.any(output_variance <= 0.0)
        or np.any(log_change < 0.0)
    ):
        raise ArtifactValidationError(
            "{} contains invalid Q values".format(location)
        )
    floor = np.asarray(em["component_floor"], dtype=float)
    expected_floor = raw_variance < floor[None, :]
    expected_output = np.maximum(raw_variance, floor[None, :])
    if not np.array_equal(floor_applied, expected_floor) or not np.array_equal(
        output_variance, expected_output
    ):
        raise ArtifactValidationError(
            "{} floor provenance is inconsistent".format(location)
        )
    if not np.array_equal(input_variance[0], initial_variance):
        raise ArtifactValidationError(
            "{} first input Q does not match the manifest".format(location)
        )
    if iteration_count > 1 and not np.array_equal(
        input_variance[1:], output_variance[:-1]
    ):
        raise ArtifactValidationError(
            "{} Q iteration trace is not contiguous".format(location)
        )
    if not np.array_equal(output_variance[-1], final_variance):
        raise ArtifactValidationError(
            "{} final Q does not match the manifest".format(location)
        )
    expected_log_change = np.max(
        np.abs(np.log(output_variance) - np.log(input_variance)), axis=1
    )
    if not np.array_equal(log_change, expected_log_change):
        raise ArtifactValidationError(
            "{} log-Q change trace is inconsistent".format(location)
        )
    expected_likelihood = np.asarray(
        [
            math.fsum(float(value) for value in row)
            for row in bag_likelihood
        ],
        dtype=float,
    )
    if not np.array_equal(total_likelihood, expected_likelihood):
        raise ArtifactValidationError(
            "{} likelihood trace is inconsistent".format(location)
        )
    converged = bool(em["converged"])
    tolerance = float(em["log_q_tolerance"])
    if converged != bool(log_change[-1] <= tolerance):
        raise ArtifactValidationError(
            "{} convergence flag is inconsistent".format(location)
        )
    if np.any(log_change[:-1] <= tolerance):
        raise ArtifactValidationError(
            "{} continued after an earlier converged iteration".format(location)
        )


def _load_complete(
    root: Path, manifest: Mapping[str, Any]
) -> DiagonalQArtifactBundle:
    (
        _status,
        bag_ids,
        bag_metadata,
        em,
        initial_variance,
        final_variance,
        smoothed_wrench_input_variance,
        terminal_implied_raw_variance,
        terminal_implied_variance,
        artifacts,
    ) = _validate_manifest(manifest, require_complete=True)
    trace_descriptor = artifacts["em_trace"]
    trace_path, trace_payload = _verified_artifact_payload(
        root, trace_descriptor, "manifest.artifacts.em_trace"
    )
    used_paths = {trace_path}
    trace = _load_npz_exact(trace_payload, trace_path, _TRACE_KEYS)
    _validate_trace(
        trace,
        bag_ids,
        em,
        initial_variance,
        final_variance,
        str(trace_path),
    )

    bag_inputs = []
    pilots = []
    last_em_statistics = []
    expectations = []
    for bag_id in bag_ids:
        metadata = bag_metadata[bag_id]
        descriptor = artifacts["bags"][bag_id]
        location = "manifest.artifacts.bags.{!r}".format(bag_id)
        path, payload = _verified_artifact_payload(
            root, descriptor, location
        )
        if path in used_paths:
            raise ArtifactValidationError(
                "artifact paths must be unique inside the bundle"
            )
        used_paths.add(path)
        arrays = _load_npz_exact(payload, path, _BAG_ARRAY_KEYS)
        boundary_count = int(metadata["boundary_count"])
        member_count = int(metadata["member_count"])
        last_member_count = int(metadata["last_em_member_count"])
        if _unicode_scalar_array(arrays, "bag_id", str(path)) != bag_id:
            raise ArtifactValidationError(
                "{}:bag_id does not match the manifest".format(path)
            )
        source_path = _unicode_scalar_array(arrays, "source_path", str(path))
        if source_path != metadata["source_path"]:
            raise ArtifactValidationError(
                "{}:source_path does not match the manifest".format(path)
            )
        source_sha256 = _unicode_scalar_array(
            arrays, "source_sha256", str(path)
        )
        if (
            _raw_sha256(source_sha256, "{}:source_sha256".format(path))
            != metadata["source_sha256"]
        ):
            raise ArtifactValidationError(
                "{}:source_sha256 does not match the manifest".format(path)
            )
        source_size_bytes = _integer_scalar_array(
            arrays, "source_size_bytes", str(path)
        )
        if (
            _positive_integer(
                source_size_bytes, "{}:source_size_bytes".format(path)
            )
            != metadata["source_size_bytes"]
        ):
            raise ArtifactValidationError(
                "{}:source_size_bytes does not match the manifest".format(path)
            )
        stored_interval = _numeric_array(
            arrays,
            "selected_interval_local_seconds",
            (2,),
            str(path),
        )
        selected_interval = tuple(
            float(value)
            for value in metadata["selected_interval_local_seconds"]
        )
        if not np.array_equal(stored_interval, np.asarray(selected_interval)):
            raise ArtifactValidationError(
                "{}:selected interval does not match the manifest".format(path)
            )
        stored_effective_interval = _numeric_array(
            arrays,
            "effective_interval_local_seconds",
            (2,),
            str(path),
        )
        effective_interval = tuple(
            float(value)
            for value in metadata["effective_interval_local_seconds"]
        )
        if not np.array_equal(
            stored_effective_interval, np.asarray(effective_interval)
        ):
            raise ArtifactValidationError(
                "{}:effective interval does not match the manifest".format(
                    path
                )
            )
        episode_index = _integer_scalar_array(
            arrays, "episode_index", str(path)
        )
        if (
            _nonnegative_integer(
                episode_index, "{}:episode_index".format(path)
            )
            != metadata["episode_index"]
        ):
            raise ArtifactValidationError(
                "{}:episode_index does not match the manifest".format(path)
            )
        configuration_fingerprint = _unicode_scalar_array(
            arrays, "configuration_fingerprint", str(path)
        )
        if configuration_fingerprint != metadata["configuration_fingerprint"]:
            raise ArtifactValidationError(
                "{}:configuration_fingerprint does not match the manifest".format(
                    path
                )
            )
        fixed_model_fingerprint = _unicode_scalar_array(
            arrays, "fixed_model_fingerprint", str(path)
        )
        if (
            _fingerprint(
                fixed_model_fingerprint,
                "{}:fixed_model_fingerprint".format(path),
            )
            != metadata["fixed_model_fingerprint"]
        ):
            raise ArtifactValidationError(
                "{}:fixed_model_fingerprint does not match the manifest".format(
                    path
                )
            )
        stored_delay = _numeric_array(
            arrays, "constant_delay_seconds", (1,), str(path)
        )[0]
        if stored_delay != metadata["constant_delay_seconds"]:
            raise ArtifactValidationError(
                "{}:constant_delay_seconds does not match the manifest".format(
                    path
                )
            )
        fixed_r = metadata["fixed_observation_covariance"]
        covariance_fingerprint = _unicode_scalar_array(
            arrays, "observation_covariance_fingerprint", str(path)
        )
        if covariance_fingerprint != fixed_r["covariance_fingerprint"]:
            raise ArtifactValidationError(
                "{}:observation covariance fingerprint does not match the "
                "manifest".format(path)
            )
        translation_covariance = _covariance3(
            _numeric_array(
                arrays,
                "observation_translation_covariance",
                (3, 3),
                str(path),
            ),
            "{}:observation_translation_covariance".format(path),
        )
        rotation_covariance = _covariance3(
            _numeric_array(
                arrays,
                "observation_rotation_covariance",
                (3, 3),
                str(path),
            ),
            "{}:observation_rotation_covariance".format(path),
        )
        recomputed_r_fingerprint = _observation_covariance_fingerprint(
            fixed_r["provenance"],
            translation_covariance,
            rotation_covariance,
        )
        if recomputed_r_fingerprint != covariance_fingerprint:
            raise ArtifactValidationError(
                "{}:fixed observation covariance payload changed".format(path)
            )
        times = _numeric_array(
            arrays, "times", (boundary_count,), str(path)
        )
        _validate_episode_relative_times(
            times, effective_interval, "{}:times".format(path)
        )
        stored_correlation_time = _numeric_array(
            arrays, "correlation_time", (1,), str(path)
        )
        if stored_correlation_time[0] != metadata["correlation_time"]:
            raise ArtifactValidationError(
                "{}:correlation_time does not match the manifest".format(path)
            )
        stored_last_member_count = _integer_scalar_array(
            arrays, "last_em_member_count", str(path)
        )
        if stored_last_member_count != last_member_count:
            raise ArtifactValidationError(
                "{}:last_em_member_count does not match the manifest".format(
                    path
                )
            )
        if last_member_count != member_count:
            raise ArtifactValidationError(
                "{}:last and terminal E-step member counts differ".format(path)
            )
        last_times = _numeric_array(
            arrays, "last_em_times", (boundary_count,), str(path)
        )
        if not np.array_equal(last_times, times):
            raise ArtifactValidationError(
                "{}:last and terminal E-step time grids differ".format(path)
            )
        _validate_episode_relative_times(
            last_times, effective_interval, "{}:last_em_times".format(path)
        )
        last_correlation_time = _numeric_array(
            arrays, "last_em_correlation_time", (1,), str(path)
        )[0]
        if last_correlation_time != metadata["correlation_time"]:
            raise ArtifactValidationError(
                "{}:last EM correlation time does not match the manifest".format(
                    path
                )
            )
        last_initial_second_moment = _numeric_array(
            arrays,
            "last_em_initial_second_moment",
            (BODY_WRENCH_DIMENSION,),
            str(path),
        )
        last_transition_second_moment = _numeric_array(
            arrays,
            "last_em_transition_second_moment",
            (boundary_count - 1, BODY_WRENCH_DIMENSION),
            str(path),
        )
        if np.any(last_initial_second_moment < 0.0) or np.any(
            last_transition_second_moment < 0.0
        ):
            raise ArtifactValidationError(
                "{}:last EM second moments must be non-negative".format(path)
            )
        try:
            statistics = DiagonalQEmSufficientStatistics(
                bag_id=bag_id,
                member_count=last_member_count,
                times=last_times,
                correlation_time=last_correlation_time,
                initial_second_moment=last_initial_second_moment,
                transition_second_moment=last_transition_second_moment,
            )
        except (TypeError, ValueError) as error:
            raise ArtifactValidationError(
                "{}:last EM sufficient statistics are invalid".format(path)
            ) from error
        stored_input_variance = _numeric_array(
            arrays,
            "smoothed_wrench_input_stationary_variance",
            (BODY_WRENCH_DIMENSION,),
            str(path),
        )
        if not np.array_equal(
            stored_input_variance, smoothed_wrench_input_variance
        ):
            raise ArtifactValidationError(
                "{}:smoothed-wrench conditioning Q changed".format(path)
            )
        stored_likelihood = _numeric_array(
            arrays, "approx_log_likelihood", (1,), str(path)
        )
        if stored_likelihood[0] != metadata["final_approx_log_likelihood"]:
            raise ArtifactValidationError(
                "{}:approx_log_likelihood does not match the manifest".format(
                    path
                )
            )
        wrench = _numeric_array(
            arrays,
            "smoothed_wrench",
            (member_count, boundary_count, BODY_WRENCH_DIMENSION),
            str(path),
        )
        bag_inputs.append(
            DiagonalQArtifactBagInput(
                bag_id=bag_id,
                source_path=source_path,
                source_sha256=source_sha256,
                source_size_bytes=source_size_bytes,
                selected_interval_local_seconds=selected_interval,
                effective_interval_local_seconds=effective_interval,
                episode_index=episode_index,
                configuration_fingerprint=configuration_fingerprint,
                fixed_model_provenance=metadata["fixed_model_provenance"],
                constant_delay_seconds=stored_delay,
                translation_covariance=translation_covariance,
                rotation_covariance=rotation_covariance,
                fixed_r_provenance=fixed_r["provenance"],
            )
        )
        pilots.append(
            DiagonalQInitialPilot(
                bag_id,
                boundary_count,
                metadata["pilot_stationary_standard_deviation"],
            )
        )
        last_em_statistics.append(statistics)
        expectations.append(
            DiagonalQBagExpectation(
                bag_id,
                times,
                metadata["correlation_time"],
                wrench,
                stored_likelihood[0],
            )
        )
    last_update = shared_diagonal_q_m_step(
        last_em_statistics, em["component_floor"]
    )
    if (
        not np.array_equal(
            last_update.raw_stationary_variance,
            trace["raw_stationary_variance"][-1],
        )
        or not np.array_equal(
            last_update.covariance.stationary_variance,
            trace["output_stationary_variance"][-1],
        )
        or not np.array_equal(
            last_update.floor_applied, trace["floor_applied"][-1]
        )
        or not np.array_equal(
            last_update.covariance.stationary_variance, final_variance
        )
    ):
        raise ArtifactValidationError(
            "last EM sufficient statistics do not reproduce the final M-step"
        )
    terminal_update = shared_diagonal_q_m_step(
        tuple(value.sufficient_statistics for value in expectations),
        em["component_floor"],
    )
    if (
        not np.array_equal(
            terminal_update.raw_stationary_variance,
            terminal_implied_raw_variance,
        )
        or not np.array_equal(
            terminal_update.covariance.stationary_variance,
            terminal_implied_variance,
        )
    ):
        raise ArtifactValidationError(
            "terminal smoothed paths do not reproduce their diagnostic Q"
        )
    expected_initial = initial_diagonal_q_from_pilots(
        pilots, em["component_floor"]
    )
    if not np.array_equal(
        expected_initial.stationary_variance, initial_variance
    ):
        raise ArtifactValidationError(
            "pilot scales do not reproduce the initial covariance"
        )
    detached_manifest = json.loads(
        canonical_json_bytes(manifest).decode("utf-8")
    )
    return DiagonalQArtifactBundle(
        root=root,
        manifest=detached_manifest,
        trace={key: value.copy() for key, value in trace.items()},
        bag_inputs=tuple(bag_inputs),
        pilots=tuple(pilots),
        last_em_statistics=tuple(last_em_statistics),
        expectations=tuple(expectations),
        covariance=BodyWrenchDiagonalCovariance(final_variance),
    )


def read_diagonal_q_manifest(
    root: Union[str, Path]
) -> Dict[str, Any]:
    """Read and validate manifest structure without claiming completion."""

    bundle_root = Path(root).expanduser().resolve()
    manifest_path = bundle_root / MANIFEST_NAME
    if manifest_path.is_symlink():
        raise ArtifactValidationError("manifest must not be a symbolic link")
    manifest = read_json(manifest_path)
    _validate_manifest(manifest, require_complete=False)
    return manifest


def load_diagonal_q_artifact(
    root: Union[str, Path]
) -> DiagonalQArtifactBundle:
    """Load only a complete, digest-valid, pickle-free diagonal-Q bundle."""

    bundle_root = Path(root).expanduser().resolve()
    manifest_path = bundle_root / MANIFEST_NAME
    if manifest_path.is_symlink():
        raise ArtifactValidationError("manifest must not be a symbolic link")
    manifest = read_json(manifest_path)
    return _load_complete(bundle_root, manifest)


def _ordered_expectations(
    result: DiagonalQEmResult,
    expectations: Iterable[DiagonalQBagExpectation],
) -> Tuple[DiagonalQBagExpectation, ...]:
    try:
        values = tuple(expectations)
    except TypeError as error:
        raise TypeError(
            "expectations must be an iterable of DiagonalQBagExpectation"
        ) from error
    if any(not isinstance(value, DiagonalQBagExpectation) for value in values):
        raise TypeError(
            "expectations must contain DiagonalQBagExpectation values"
        )
    ordered = tuple(sorted(values, key=lambda value: value.bag_id))
    identifiers = tuple(value.bag_id for value in ordered)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("expectations must have unique bag IDs")
    if identifiers != result.bag_ids:
        raise ValueError("expectation bag IDs must match the EM result")
    result_by_id = {
        value.bag_id: value for value in result.final_expectations
    }
    for value in ordered:
        previous = result_by_id[value.bag_id]
        if (
            value.boundary_count != previous.boundary_count
            or value.member_count != previous.member_count
            or value.correlation_time != previous.correlation_time
            or not np.array_equal(value.times, previous.times)
            or value.approx_log_likelihood != previous.approx_log_likelihood
            or not np.array_equal(
                value.smoothed_wrench, previous.smoothed_wrench
            )
        ):
            raise ValueError(
                "final expectation layout changed for bag {!r}".format(
                    value.bag_id
                )
            )
    return ordered


def _ordered_bag_inputs(
    values: Iterable[DiagonalQArtifactBagInput],
    bag_ids: Tuple[str, ...],
) -> Tuple[DiagonalQArtifactBagInput, ...]:
    try:
        raw_values = tuple(values)
    except TypeError as error:
        raise TypeError(
            "bag_inputs must be an iterable of DiagonalQArtifactBagInput"
        ) from error
    if any(
        not isinstance(value, DiagonalQArtifactBagInput)
        for value in raw_values
    ):
        raise TypeError(
            "bag_inputs must contain DiagonalQArtifactBagInput values"
        )
    ordered = tuple(sorted(raw_values, key=lambda value: value.bag_id))
    identifiers = tuple(value.bag_id for value in ordered)
    if len(set(identifiers)) != len(identifiers):
        raise ArtifactValidationError("bag_inputs must have unique bag IDs")
    if identifiers != bag_ids:
        raise ArtifactValidationError(
            "bag input IDs must match the diagonal-Q EM result"
        )
    # Reconstruct the values because frozen dataclasses can still contain
    # externally mutated NumPy arrays.
    return tuple(
        DiagonalQArtifactBagInput(
            bag_id=value.bag_id,
            source_path=value.source_path,
            source_sha256=value.source_sha256,
            source_size_bytes=value.source_size_bytes,
            selected_interval_local_seconds=(
                value.selected_interval_local_seconds
            ),
            effective_interval_local_seconds=(
                value.effective_interval_local_seconds
            ),
            episode_index=value.episode_index,
            configuration_fingerprint=value.configuration_fingerprint,
            fixed_model_provenance=value.fixed_model_provenance,
            constant_delay_seconds=value.constant_delay_seconds,
            translation_covariance=value.translation_covariance,
            rotation_covariance=value.rotation_covariance,
            fixed_r_provenance=value.fixed_r_provenance,
        )
        for value in ordered
    )


def _empty_descriptor(path: str) -> Dict[str, Any]:
    return {"path": path, "sha256": None, "size_bytes": None}


def _complete_descriptor(path: Path, root: Path) -> Dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _safe_new_payload_path(root: Path, relative: str) -> Path:
    lexical = root / relative
    lexical.parent.mkdir(parents=True, exist_ok=True)
    root_resolved = root.resolve()
    parent_resolved = lexical.parent.resolve()
    if (
        parent_resolved != root_resolved
        and root_resolved not in parent_resolved.parents
    ):
        raise ArtifactStateError(
            "artifact output parent resolves outside the bundle"
        )
    if lexical.exists() or lexical.is_symlink():
        raise ArtifactStateError(
            "artifact output already exists: {}".format(lexical)
        )
    return lexical


def _trace_arrays(result: DiagonalQEmResult) -> Dict[str, np.ndarray]:
    trace = result.iterations
    bag_ids = result.bag_ids
    return {
        "iteration": np.asarray(
            [value.iteration for value in trace], dtype=np.int64
        ),
        "input_stationary_variance": np.asarray(
            [value.input_covariance.stationary_variance for value in trace],
            dtype=float,
        ),
        "raw_stationary_variance": np.asarray(
            [value.update.raw_stationary_variance for value in trace],
            dtype=float,
        ),
        "output_stationary_variance": np.asarray(
            [value.output_covariance.stationary_variance for value in trace],
            dtype=float,
        ),
        "floor_applied": np.asarray(
            [value.update.floor_applied for value in trace], dtype=bool
        ),
        "maximum_absolute_log_q_change": np.asarray(
            [value.maximum_absolute_log_q_change for value in trace],
            dtype=float,
        ),
        "approx_log_likelihood": np.asarray(
            [value.approx_log_likelihood for value in trace], dtype=float
        ),
        "bag_approx_log_likelihood": np.asarray(
            [
                [dict(value.bag_approx_log_likelihoods)[bag_id] for bag_id in bag_ids]
                for value in trace
            ],
            dtype=float,
        ),
    }


def _validate_result_last_m_step(result: DiagonalQEmResult) -> None:
    statistics = tuple(
        value.sufficient_statistics for value in result.last_expectations
    )
    expected = shared_diagonal_q_m_step(
        statistics, result.config.component_floor
    )
    recorded = result.iterations[-1].update
    if (
        not np.array_equal(
            expected.raw_stationary_variance,
            recorded.raw_stationary_variance,
        )
        or not np.array_equal(
            expected.covariance.stationary_variance,
            recorded.covariance.stationary_variance,
        )
        or not np.array_equal(expected.floor_applied, recorded.floor_applied)
    ):
        raise ArtifactValidationError(
            "last EM expectations do not reproduce the final trace update"
        )


def write_diagonal_q_artifact(
    root: Union[str, Path],
    *,
    run_id: str,
    stage_id: str,
    request_fingerprint: str,
    project_fingerprint: str,
    stage_input_fingerprint: str,
    implementation_provenance: Mapping[str, Any],
    bag_inputs: Iterable[DiagonalQArtifactBagInput],
    result: DiagonalQEmResult,
    expectations: Iterable[DiagonalQBagExpectation],
) -> Path:
    """Atomically publish a complete diagonal-Q stage-boundary bundle."""

    if not isinstance(result, DiagonalQEmResult):
        raise TypeError("result must be a DiagonalQEmResult")
    _validate_result_last_m_step(result)
    selected_run_id = _string(run_id, "run_id")
    selected_stage_id = _string(stage_id, "stage_id")
    selected_request_fingerprint = _fingerprint(
        request_fingerprint, "request_fingerprint"
    )
    selected_project_fingerprint = _fingerprint(
        project_fingerprint, "project_fingerprint"
    )
    selected_stage_input_fingerprint = _fingerprint(
        stage_input_fingerprint, "stage_input_fingerprint"
    )
    selected_implementation_provenance = _implementation_provenance(
        implementation_provenance, "implementation_provenance"
    )
    selected_implementation_fingerprint = compute_request_fingerprint(
        selected_implementation_provenance
    )
    bag_ids = result.bag_ids
    ordered_expectations = _ordered_expectations(result, expectations)
    ordered_bag_inputs = _ordered_bag_inputs(bag_inputs, bag_ids)
    input_by_id = {value.bag_id: value for value in ordered_bag_inputs}
    intervals = {
        bag_id: list(
            input_by_id[bag_id].selected_interval_local_seconds
        )
        for bag_id in bag_ids
    }
    last_expectation_by_id = {
        value.bag_id: value for value in result.last_expectations
    }
    expectation_by_id = {
        value.bag_id: value for value in ordered_expectations
    }
    for bag_id in bag_ids:
        bag_input = input_by_id[bag_id]
        terminal = expectation_by_id[bag_id]
        last = last_expectation_by_id[bag_id]
        _validate_episode_relative_times(
            terminal.times,
            bag_input.effective_interval_local_seconds,
            "terminal expectation {!r} times".format(bag_id),
        )
        _validate_episode_relative_times(
            last.times,
            bag_input.effective_interval_local_seconds,
            "last EM expectation {!r} times".format(bag_id),
        )
        if last.member_count != terminal.member_count:
            raise ArtifactValidationError(
                "last and terminal E-step member counts differ for bag "
                "{!r}".format(bag_id)
            )
    terminal_update = shared_diagonal_q_m_step(
        tuple(value.sufficient_statistics for value in ordered_expectations),
        result.config.component_floor,
    )
    destination = Path(root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / MANIFEST_NAME
    if manifest_path.exists() or manifest_path.is_symlink():
        raise ArtifactStateError(
            "bundle already has a manifest: {}".format(manifest_path)
        )

    trace_relative = "em_trace.npz"
    bag_relatives = {
        bag_id: "bags/{:04d}.npz".format(index)
        for index, bag_id in enumerate(bag_ids)
    }
    trace_path = _safe_new_payload_path(destination, trace_relative)
    bag_paths = {
        bag_id: _safe_new_payload_path(destination, bag_relatives[bag_id])
        for bag_id in bag_ids
    }
    pilot_by_id = {value.bag_id: value for value in result.pilots}
    manifest: Dict[str, Any] = {
        "schema": DIAGONAL_Q_ESTIMATE_SCHEMA,
        "status": WRITING_STATUS,
        "run_id": selected_run_id,
        "stage_id": selected_stage_id,
        "request_fingerprint": selected_request_fingerprint,
        "project_fingerprint": selected_project_fingerprint,
        "stage_input_fingerprint": selected_stage_input_fingerprint,
        "implementation": {
            "provenance": selected_implementation_provenance,
            "fingerprint": selected_implementation_fingerprint,
        },
        "selected_bag_ids": list(bag_ids),
        "selected_intervals": intervals,
        "body_wrench": {
            "frame": BODY_WRENCH_FRAME,
            "component_order": list(BODY_WRENCH_COMPONENT_ORDER),
            "variance_units": list(BODY_WRENCH_VARIANCE_UNITS),
        },
        "bags": {
            bag_id: {
                "source_path": input_by_id[bag_id].source_path,
                "source_sha256": input_by_id[bag_id].source_sha256,
                "source_size_bytes": input_by_id[bag_id].source_size_bytes,
                "selected_interval_local_seconds": intervals[bag_id],
                "effective_interval_local_seconds": list(
                    input_by_id[bag_id].effective_interval_local_seconds
                ),
                "episode_index": input_by_id[bag_id].episode_index,
                "configuration_fingerprint": (
                    input_by_id[bag_id].configuration_fingerprint
                ),
                "fixed_model_fingerprint": (
                    input_by_id[bag_id].fixed_model_fingerprint
                ),
                "fixed_model_provenance": (
                    input_by_id[bag_id].fixed_model_provenance
                ),
                "constant_delay_seconds": (
                    input_by_id[bag_id].constant_delay_seconds
                ),
                "time_basis": _TIME_BASIS,
                "fixed_observation_covariance": {
                    "fixed": True,
                    "provenance": (
                        input_by_id[bag_id].fixed_r_provenance
                    ),
                    "covariance_fingerprint": (
                        input_by_id[bag_id]
                        .observation_covariance_fingerprint
                    ),
                },
                "boundary_count": expectation_by_id[bag_id].boundary_count,
                "member_count": expectation_by_id[bag_id].member_count,
                "last_em_member_count": (
                    last_expectation_by_id[bag_id].member_count
                ),
                "correlation_time": (
                    expectation_by_id[bag_id].correlation_time
                ),
                "pilot_stationary_standard_deviation": (
                    pilot_by_id[bag_id].stationary_standard_deviation.tolist()
                ),
                "final_approx_log_likelihood": (
                    expectation_by_id[bag_id].approx_log_likelihood
                ),
            }
            for bag_id in bag_ids
        },
        "em": {
            "maximum_iterations": result.config.maximum_iterations,
            "log_q_tolerance": result.config.log_q_tolerance,
            "component_floor": result.config.component_floor.tolist(),
            "completed_iterations": len(result.iterations),
            "converged": result.converged,
            "termination_reason": result.termination_reason,
            "smoothed_wrench_semantics": (
                "terminal_e_step_conditioned_on_final_q"
            ),
        },
        "initial_stationary_variance": (
            result.initial_covariance.stationary_variance.tolist()
        ),
        "final_stationary_variance": (
            result.covariance.stationary_variance.tolist()
        ),
        "final_stationary_standard_deviation": (
            result.covariance.stationary_standard_deviation.tolist()
        ),
        "smoothed_wrench_input_stationary_variance": (
            result.final_expectation_input_covariance
            .stationary_variance.tolist()
        ),
        "terminal_implied_raw_stationary_variance": (
            terminal_update.raw_stationary_variance.tolist()
        ),
        "terminal_implied_stationary_variance": (
            terminal_update.covariance.stationary_variance.tolist()
        ),
        "artifacts": {
            "em_trace": _empty_descriptor(trace_relative),
            "bags": {
                bag_id: _empty_descriptor(bag_relatives[bag_id])
                for bag_id in bag_ids
            },
        },
    }
    _validate_manifest(manifest, require_complete=False)
    write_json_atomic(manifest_path, manifest)

    write_npz_atomic(trace_path, _trace_arrays(result))
    for bag_id in bag_ids:
        expectation = expectation_by_id[bag_id]
        last_expectation = last_expectation_by_id[bag_id]
        last_statistics = last_expectation.sufficient_statistics
        bag_input = input_by_id[bag_id]
        path = bag_paths[bag_id]
        write_npz_atomic(
            path,
            {
                "bag_id": np.asarray((bag_id,)),
                "source_path": np.asarray((bag_input.source_path,)),
                "source_sha256": np.asarray((bag_input.source_sha256,)),
                "source_size_bytes": np.asarray(
                    (bag_input.source_size_bytes,), dtype=np.int64
                ),
                "selected_interval_local_seconds": np.asarray(
                    bag_input.selected_interval_local_seconds, dtype=float
                ),
                "effective_interval_local_seconds": np.asarray(
                    bag_input.effective_interval_local_seconds, dtype=float
                ),
                "episode_index": np.asarray(
                    (bag_input.episode_index,), dtype=np.int64
                ),
                "configuration_fingerprint": np.asarray(
                    (bag_input.configuration_fingerprint,)
                ),
                "fixed_model_fingerprint": np.asarray(
                    (bag_input.fixed_model_fingerprint,)
                ),
                "constant_delay_seconds": np.asarray(
                    (bag_input.constant_delay_seconds,), dtype=float
                ),
                "times": expectation.times,
                "correlation_time": np.asarray(
                    (expectation.correlation_time,)
                ),
                "observation_covariance_fingerprint": np.asarray(
                    (bag_input.observation_covariance_fingerprint,)
                ),
                "observation_translation_covariance": (
                    bag_input.translation_covariance
                ),
                "observation_rotation_covariance": (
                    bag_input.rotation_covariance
                ),
                "last_em_member_count": np.asarray(
                    (last_statistics.member_count,), dtype=np.int64
                ),
                "last_em_times": last_statistics.times,
                "last_em_correlation_time": np.asarray(
                    (last_statistics.correlation_time,), dtype=float
                ),
                "last_em_initial_second_moment": (
                    last_statistics.initial_second_moment
                ),
                "last_em_transition_second_moment": (
                    last_statistics.transition_second_moment
                ),
                "smoothed_wrench_input_stationary_variance": (
                    result.final_expectation_input_covariance
                    .stationary_variance
                ),
                "approx_log_likelihood": np.asarray(
                    (expectation.approx_log_likelihood,)
                ),
                "smoothed_wrench": expectation.smoothed_wrench,
            },
        )

    candidate = json.loads(canonical_json_bytes(manifest).decode("utf-8"))
    candidate["status"] = COMPLETE_STATUS
    candidate["artifacts"]["em_trace"] = _complete_descriptor(
        trace_path, destination
    )
    candidate["artifacts"]["bags"] = {
        bag_id: _complete_descriptor(bag_paths[bag_id], destination)
        for bag_id in bag_ids
    }
    _load_complete(destination, candidate)
    current = read_diagonal_q_manifest(destination)
    if current["status"] != WRITING_STATUS or canonical_json_bytes(
        current
    ) != canonical_json_bytes(manifest):
        raise ArtifactStateError(
            "writing manifest changed before completion"
        )
    write_json_atomic(manifest_path, candidate)
    return destination


def mark_diagonal_q_artifact_cancelled(
    root: Union[str, Path], reason: str
) -> Path:
    """Atomically make cancellation authoritative for an incomplete bundle."""

    selected_reason = _string(reason, "cancellation reason")
    destination = Path(root).expanduser().resolve()
    manifest = read_diagonal_q_manifest(destination)
    if manifest["status"] == COMPLETE_STATUS:
        raise ArtifactStateError("a complete bundle cannot be cancelled")
    if manifest["status"] == CANCELLED_STATUS:
        if manifest["cancellation_reason"] != selected_reason:
            raise ArtifactStateError(
                "cancelled bundle already has a different reason"
            )
        return destination / MANIFEST_NAME
    candidate = dict(manifest)
    candidate["status"] = CANCELLED_STATUS
    candidate["cancellation_reason"] = selected_reason
    _validate_manifest(candidate, require_complete=False)
    return write_json_atomic(destination / MANIFEST_NAME, candidate)


__all__ = [
    "DIAGONAL_Q_ESTIMATE_SCHEMA",
    "DiagonalQArtifactBagInput",
    "DiagonalQArtifactBundle",
    "load_diagonal_q_artifact",
    "mark_diagonal_q_artifact_cancelled",
    "read_diagonal_q_manifest",
    "write_diagonal_q_artifact",
]
