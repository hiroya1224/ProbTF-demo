"""Strict, pickle-free artifacts for posterior PID cross-evaluation."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from grape_param_estim.artifact_io import (
    ArtifactValidationError,
    IncompleteArtifactError,
    UnsupportedArtifactSchema,
    load_npz_strict,
    read_json,
    write_json_atomic,
    write_npz_atomic,
)
from grape_param_estim.controller_config import (
    PidGainComparison,
    render_pid_diff_yaml,
    render_proposed_pid_yaml,
)
from grape_param_estim.pid.metrics import FORECAST_COST_METRICS
from grape_param_estim.pid.particle_search import (
    MODEL_DISCREPANCY_POLICIES,
    MODEL_DISCREPANCY_QUANTITIES,
    MODEL_DISCREPANCY_INTERVAL_MODELS,
    PidCandidateEvaluation,
)
from grape_param_estim.pid.proposal import PhysicalPlantPosterior


PID_PROPOSAL_EVALUATION_SCHEMA = (
    "grape-param-estim/pid-proposal-evaluation/v1"
)
_COMPLETE_STATUS = "complete"
_SAFE_FILE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MANIFEST_KEYS = (
    "schema",
    "status",
    "evaluation_id",
    "estimation_run_id",
    "estimation_request_fingerprint",
    "request_fingerprint",
    "model_discrepancy_policy",
    "model_discrepancy_residual_quantity",
    "model_discrepancy_interval_model",
    "model_discrepancy_q_diagonal",
    "model_discrepancy_base_seed",
    "model_discrepancy_replicates",
    "plant_sample_subset_method",
    "plant_sample_ids",
    "bag_ids",
    "candidate_ids",
    "selection_policy",
    "nondominated_candidate_ids",
    "recommended_candidate_ids",
    "recommendation_available",
    "rejection_reason",
    "selected_candidate_id",
    "artifacts",
)


def _canonical(value: object, name: str) -> str:
    selected = str(value)
    if not selected or selected.strip() != selected or "\x00" in selected:
        raise ArtifactValidationError(
            "{} must be a canonical non-empty string".format(name)
        )
    return selected


def _sha256_identifier(value: object, name: str) -> str:
    selected = _canonical(value, name)
    if (
        len(selected) != 71
        or not selected.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in selected[7:])
    ):
        raise ArtifactValidationError(
            "{} must have form sha256:<64 lowercase hex>".format(name)
        )
    return selected


def _safe_file_identifier(value: object, name: str) -> str:
    selected = _canonical(value, name)
    if _SAFE_FILE_ID.fullmatch(selected) is None:
        raise ArtifactValidationError(
            "{} is not a safe file identifier".format(name)
        )
    return selected


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _strict_keys(
    value: Mapping[str, Any],
    required: Sequence[str],
    location: str,
) -> None:
    missing = sorted(set(required) - set(value))
    unknown = sorted(set(value) - set(required))
    if missing or unknown:
        raise ArtifactValidationError(
            "{} keys disagree; missing={}, unknown={}".format(
                location, missing, unknown
            )
        )


def _string_list(
    value: object,
    location: str,
    *,
    allow_empty: bool = False,
) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise ArtifactValidationError("{} must be a list".format(location))
    selected = tuple(_canonical(item, location) for item in value)
    if (not allow_empty and not selected) or len(set(selected)) != len(selected):
        raise ArtifactValidationError(
            "{} must contain unique strings".format(location)
        )
    return selected


def _descriptor(root: Path, value: object, expected: str) -> Path:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError("artifact descriptor must be an object")
    _strict_keys(value, ("path", "sha256"), "artifact descriptor")
    if value["path"] != expected:
        raise ArtifactValidationError(
            "artifact path must be {!r}".format(expected)
        )
    digest = _sha256_identifier(value["sha256"], "artifact sha256")
    relative = Path(expected)
    if relative.is_absolute() or ".." in relative.parts:
        raise ArtifactValidationError("artifact path escapes its root")
    path = root / relative
    if not path.is_file() or _file_sha256(path) != digest:
        raise ArtifactValidationError(
            "artifact file is missing or failed SHA-256: {}".format(expected)
        )
    return path


def _arrays(path: Path, keys: Sequence[str]) -> Dict[str, np.ndarray]:
    values = load_npz_strict(path)
    _strict_keys(values, keys, str(path))
    return values


def _string_array(
    arrays: Mapping[str, np.ndarray],
    key: str,
    shape: Tuple[int, ...],
    *,
    unique: bool = False,
    allow_empty: bool = False,
) -> np.ndarray:
    value = arrays[key]
    if value.shape != shape or value.dtype.kind not in "US":
        raise ArtifactValidationError("{} must be a string array".format(key))
    result = value.astype(str)
    if (not allow_empty and np.any(result == "")) or (
        unique and np.unique(result).size != result.size
    ):
        raise ArtifactValidationError("{} has invalid identifiers".format(key))
    return result


def _numeric_array(
    arrays: Mapping[str, np.ndarray],
    key: str,
    shape: Tuple[int, ...],
    *,
    integer: bool = False,
) -> np.ndarray:
    value = arrays[key]
    valid = (
        np.issubdtype(value.dtype, np.integer)
        if integer
        else np.issubdtype(value.dtype, np.number)
    )
    if (
        value.shape != shape
        or not valid
        or np.issubdtype(value.dtype, np.bool_)
        or np.any(~np.isfinite(value))
    ):
        raise ArtifactValidationError("{} has an invalid numeric array".format(key))
    return value


@dataclass(frozen=True)
class PidEvaluationArtifactIdentity:
    evaluation_id: str
    estimation_run_id: str
    estimation_request_fingerprint: str
    request_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evaluation_id", _canonical(self.evaluation_id, "evaluation_id")
        )
        object.__setattr__(
            self,
            "estimation_run_id",
            _canonical(self.estimation_run_id, "estimation_run_id"),
        )
        object.__setattr__(
            self,
            "estimation_request_fingerprint",
            _sha256_identifier(
                self.estimation_request_fingerprint,
                "estimation_request_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "request_fingerprint",
            _sha256_identifier(self.request_fingerprint, "request_fingerprint"),
        )


@dataclass(frozen=True)
class PidProposalEvaluationArtifact:
    root: Path
    manifest: Mapping[str, Any]
    source_samples: Mapping[str, np.ndarray]
    candidate_particles: Mapping[str, np.ndarray]
    summary: Mapping[str, np.ndarray]
    bags: Mapping[str, Mapping[str, np.ndarray]]
    proposed_yaml: Optional[str]
    proposed_diff_yaml: Optional[str]


_SOURCE_KEYS = (
    "sample_id",
    "source_mode_id",
    "delay",
    "mass",
    "inertia",
    "cog",
    "force_effectiveness",
    "torque_effectiveness",
)
_CANDIDATE_KEYS = (
    "candidate_id",
    "source",
    "source_sample_id",
    "source_mode_id",
    "generation",
    "parent_candidate_id",
    "gain_values",
)
_SUMMARY_KEYS = (
    "metric_names",
    "candidate_id",
    "record_count",
    "mean",
    "quantile",
    "upper_cvar",
    "forecast_completion_mean",
    "forecast_completion_lower_quantile",
    "forecast_completion_lower_cvar",
    "gain_change_magnitude",
    "quantile_level",
    "cvar_level",
    "nondominated_candidate_id",
    "recommended_candidate_id",
)
_BAG_KEYS = (
    "candidate_id",
    "sample_id",
    "replicate_index",
    "discrepancy_seed",
) + FORECAST_COST_METRICS + ("forecast_completion",)


def _source_payload(posterior: PhysicalPlantPosterior) -> Dict[str, np.ndarray]:
    return {
        "sample_id": posterior.sample_id,
        "source_mode_id": np.asarray(
            tuple(value.source_mode_id for value in posterior.samples), dtype=str
        ),
        "delay": posterior.delay,
        "mass": np.asarray(tuple(value.parameters.mass for value in posterior.samples)),
        "inertia": np.asarray(
            tuple(value.parameters.inertia for value in posterior.samples)
        ),
        "cog": np.asarray(
            tuple(value.parameters.cog_offset for value in posterior.samples)
        ),
        "force_effectiveness": np.asarray(
            tuple(value.parameters.force_effectiveness for value in posterior.samples)
        ),
        "torque_effectiveness": np.asarray(
            tuple(value.parameters.torque_effectiveness for value in posterior.samples)
        ),
    }


def _candidate_payload(evaluation: PidCandidateEvaluation) -> Dict[str, np.ndarray]:
    return {
        "candidate_id": np.asarray(
            tuple(value.candidate_id for value in evaluation.candidates), dtype=str
        ),
        "source": np.asarray(
            tuple(value.source for value in evaluation.candidates), dtype=str
        ),
        "source_sample_id": np.asarray(
            tuple(value.source_sample_id for value in evaluation.candidates), dtype=str
        ),
        "source_mode_id": np.asarray(
            tuple(value.source_mode_id for value in evaluation.candidates), dtype=str
        ),
        "generation": np.asarray(
            tuple(value.generation for value in evaluation.candidates), dtype=np.int64
        ),
        "parent_candidate_id": np.asarray(
            tuple(value.parent_candidate_id for value in evaluation.candidates),
            dtype=str,
        ),
        "gain_values": np.asarray(
            tuple(value.configuration.values for value in evaluation.candidates)
        ),
    }


def _summary_payload(evaluation: PidCandidateEvaluation) -> Dict[str, np.ndarray]:
    summaries = evaluation.summaries
    return {
        "metric_names": np.asarray(FORECAST_COST_METRICS, dtype=str),
        "candidate_id": np.asarray(
            tuple(value.candidate_id for value in summaries), dtype=str
        ),
        "record_count": np.asarray(
            tuple(value.record_count for value in summaries), dtype=np.int64
        ),
        "mean": np.asarray(tuple(value.mean for value in summaries)),
        "quantile": np.asarray(tuple(value.quantile for value in summaries)),
        "upper_cvar": np.asarray(tuple(value.upper_cvar for value in summaries)),
        "forecast_completion_mean": np.asarray(
            tuple(value.forecast_completion_mean for value in summaries)
        ),
        "forecast_completion_lower_quantile": np.asarray(
            tuple(value.forecast_completion_lower_quantile for value in summaries)
        ),
        "forecast_completion_lower_cvar": np.asarray(
            tuple(value.forecast_completion_lower_cvar for value in summaries)
        ),
        "gain_change_magnitude": np.asarray(
            tuple(value.gain_change_magnitude for value in summaries)
        ),
        "quantile_level": np.asarray(
            tuple(value.quantile_level for value in summaries)
        ),
        "cvar_level": np.asarray(tuple(value.cvar_level for value in summaries)),
        "nondominated_candidate_id": np.asarray(
            evaluation.decision.nondominated_candidate_ids, dtype=str
        ),
        "recommended_candidate_id": np.asarray(
            evaluation.decision.recommended_candidate_ids, dtype=str
        ),
    }


def _bag_payload(
    evaluation: PidCandidateEvaluation, bag_id: str
) -> Dict[str, np.ndarray]:
    records = tuple(value for value in evaluation.records if value.bag_id == bag_id)
    payload = {
        "candidate_id": np.asarray(
            tuple(value.candidate_id for value in records), dtype=str
        ),
        "sample_id": np.asarray(tuple(value.sample_id for value in records), dtype=str),
        "replicate_index": np.asarray(
            tuple(value.replicate_index for value in records), dtype=np.int64
        ),
        "discrepancy_seed": np.asarray(
            tuple(value.discrepancy_seed for value in records), dtype=np.uint64
        ),
        "forecast_completion": np.asarray(
            tuple(value.metrics.forecast_completion for value in records)
        ),
    }
    for name in FORECAST_COST_METRICS:
        payload[name] = np.asarray(
            tuple(getattr(value.metrics, name) for value in records)
        )
    return payload


def _validate_payloads(
    manifest: Mapping[str, Any],
    source: Mapping[str, np.ndarray],
    candidates: Mapping[str, np.ndarray],
    summary: Mapping[str, np.ndarray],
    bags: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    source_count = source["sample_id"].size
    source_ids = _string_array(
        source, "sample_id", (source_count,), unique=True
    )
    _string_array(source, "source_mode_id", (source_count,))
    for key, shape in (
        ("delay", (source_count,)),
        ("mass", (source_count,)),
        ("inertia", (source_count, 3, 3)),
        ("cog", (source_count, 3)),
        ("force_effectiveness", (source_count, 4)),
        ("torque_effectiveness", (source_count, 4)),
    ):
        _numeric_array(source, key, shape)
    if (
        np.any(source["delay"] < 0.0)
        or np.any(source["mass"] <= 0.0)
        or np.any(source["force_effectiveness"] <= 0.0)
        or np.any(source["torque_effectiveness"] <= 0.0)
        or np.any(np.linalg.eigvalsh(source["inertia"]) <= 0.0)
        or not np.allclose(
            source["inertia"],
            np.swapaxes(source["inertia"], 1, 2),
            rtol=1.0e-9,
            atol=1.0e-11,
        )
    ):
        raise ArtifactValidationError("source sample physical values are invalid")

    candidate_count = candidates["candidate_id"].size
    candidate_ids = _string_array(
        candidates, "candidate_id", (candidate_count,), unique=True
    )
    _string_array(candidates, "source", (candidate_count,))
    _string_array(
        candidates, "source_sample_id", (candidate_count,), allow_empty=True
    )
    _string_array(
        candidates, "source_mode_id", (candidate_count,), allow_empty=True
    )
    _string_array(
        candidates, "parent_candidate_id", (candidate_count,), allow_empty=True
    )
    generation = _numeric_array(
        candidates, "generation", (candidate_count,), integer=True
    )
    gains = _numeric_array(candidates, "gain_values", (candidate_count, 4, 3))
    if candidate_ids[0] != "current" or np.any(generation < 0) or np.any(gains < 0.0):
        raise ArtifactValidationError("candidate particle values are invalid")

    metric_count = len(FORECAST_COST_METRICS)
    if tuple(_string_array(summary, "metric_names", (metric_count,))) != tuple(
        FORECAST_COST_METRICS
    ):
        raise ArtifactValidationError("summary metric order is not canonical")
    if not np.array_equal(
        _string_array(summary, "candidate_id", (candidate_count,), unique=True),
        candidate_ids,
    ):
        raise ArtifactValidationError("summary candidates are not aligned")
    for key, shape in (
        ("record_count", (candidate_count,)),
        ("mean", (candidate_count, metric_count)),
        ("quantile", (candidate_count, metric_count)),
        ("upper_cvar", (candidate_count, metric_count)),
        ("forecast_completion_mean", (candidate_count,)),
        ("forecast_completion_lower_quantile", (candidate_count,)),
        ("forecast_completion_lower_cvar", (candidate_count,)),
        ("gain_change_magnitude", (candidate_count,)),
        ("quantile_level", (candidate_count,)),
        ("cvar_level", (candidate_count,)),
    ):
        _numeric_array(summary, key, shape, integer=(key == "record_count"))
    expected_record_count = (
        len(manifest["plant_sample_ids"])
        * len(manifest["bag_ids"])
        * int(manifest["model_discrepancy_replicates"])
    )
    if np.any(summary["record_count"] != expected_record_count):
        raise ArtifactValidationError("summary record counts are incomplete")
    nondominated = _string_array(
        summary,
        "nondominated_candidate_id",
        (summary["nondominated_candidate_id"].size,),
        unique=True,
    )
    recommended = _string_array(
        summary,
        "recommended_candidate_id",
        (summary["recommended_candidate_id"].size,),
        unique=True,
        allow_empty=True,
    )
    if not set(nondominated).issubset(set(candidate_ids)) or not set(
        recommended
    ).issubset(set(nondominated)):
        raise ArtifactValidationError("recommendation identifiers are invalid")

    expected_candidate_ids = tuple(manifest["candidate_ids"])
    expected_sample_ids = tuple(manifest["plant_sample_ids"])
    expected_bags = tuple(manifest["bag_ids"])
    if tuple(candidate_ids) != expected_candidate_ids:
        raise ArtifactValidationError("manifest candidate IDs are not aligned")
    if not set(expected_sample_ids).issubset(set(source_ids)):
        raise ArtifactValidationError("manifest plant samples are unavailable")
    if set(bags) != set(expected_bags):
        raise ArtifactValidationError("bag payloads are not aligned")
    expected_per_bag = (
        len(expected_candidate_ids)
        * len(expected_sample_ids)
        * int(manifest["model_discrepancy_replicates"])
    )
    for bag_id in expected_bags:
        arrays = bags[bag_id]
        count = arrays["candidate_id"].size
        if count != expected_per_bag:
            raise ArtifactValidationError("bag forecast Cartesian product is incomplete")
        bag_candidates = _string_array(arrays, "candidate_id", (count,))
        bag_samples = _string_array(arrays, "sample_id", (count,))
        replicate = _numeric_array(
            arrays, "replicate_index", (count,), integer=True
        )
        seed = _numeric_array(arrays, "discrepancy_seed", (count,), integer=True)
        del seed
        for key in FORECAST_COST_METRICS + ("forecast_completion",):
            value = _numeric_array(
                arrays,
                key,
                (count,),
                integer=(key == "numerical_failure_count"),
            )
            if np.any(value < 0.0):
                raise ArtifactValidationError("bag metrics must be non-negative")
        if (
            np.any(arrays["forecast_completion"] > 1.0)
            or np.any(arrays["actuator_saturation_rate"] > 1.0)
        ):
            raise ArtifactValidationError("bag rates must remain in [0, 1]")
        identities = set(
            zip(bag_candidates.tolist(), bag_samples.tolist(), replicate.tolist())
        )
        expected = {
            (candidate_id, sample_id, replicate_index)
            for candidate_id in expected_candidate_ids
            for sample_id in expected_sample_ids
            for replicate_index in range(
                int(manifest["model_discrepancy_replicates"])
            )
        }
        if identities != expected or len(identities) != count:
            raise ArtifactValidationError("bag forecast identities are incomplete")


def _manifest(
    identity: PidEvaluationArtifactIdentity,
    evaluation: PidCandidateEvaluation,
    selected_candidate_id: Optional[str],
    artifacts: Mapping[str, Any],
) -> Dict[str, Any]:
    discrepancy = evaluation.discrepancy
    return {
        "schema": PID_PROPOSAL_EVALUATION_SCHEMA,
        "status": _COMPLETE_STATUS,
        "evaluation_id": identity.evaluation_id,
        "estimation_run_id": identity.estimation_run_id,
        "estimation_request_fingerprint": identity.estimation_request_fingerprint,
        "request_fingerprint": identity.request_fingerprint,
        "model_discrepancy_policy": discrepancy.policy,
        "model_discrepancy_residual_quantity": discrepancy.residual_quantity,
        "model_discrepancy_interval_model": discrepancy.interval_model,
        "model_discrepancy_q_diagonal": discrepancy.diagonal_q.tolist(),
        "model_discrepancy_base_seed": discrepancy.base_seed,
        "model_discrepancy_replicates": discrepancy.replicates,
        "plant_sample_subset_method": evaluation.plant_sample_subset_method,
        "plant_sample_ids": list(evaluation.plant_sample_ids),
        "bag_ids": list(evaluation.bag_ids),
        "candidate_ids": [value.candidate_id for value in evaluation.candidates],
        "selection_policy": evaluation.decision.selection_policy,
        "nondominated_candidate_ids": list(
            evaluation.decision.nondominated_candidate_ids
        ),
        "recommended_candidate_ids": list(
            evaluation.decision.recommended_candidate_ids
        ),
        "recommendation_available": evaluation.decision.recommendation_available,
        "rejection_reason": evaluation.decision.rejection_reason,
        "selected_candidate_id": selected_candidate_id,
        "artifacts": dict(artifacts),
    }


def _load(root: Path) -> PidProposalEvaluationArtifact:
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema") != PID_PROPOSAL_EVALUATION_SCHEMA:
        raise UnsupportedArtifactSchema("unsupported PID evaluation schema")
    if manifest.get("status") != _COMPLETE_STATUS:
        raise IncompleteArtifactError("only complete PID evaluations are loadable")
    _strict_keys(manifest, _MANIFEST_KEYS, "manifest")
    for key in (
        "evaluation_id",
        "estimation_run_id",
        "model_discrepancy_policy",
        "model_discrepancy_residual_quantity",
        "model_discrepancy_interval_model",
        "plant_sample_subset_method",
        "selection_policy",
    ):
        _canonical(manifest[key], "manifest.{}".format(key))
    if manifest["model_discrepancy_policy"] not in MODEL_DISCREPANCY_POLICIES:
        raise ArtifactValidationError("manifest model discrepancy policy is invalid")
    if (
        manifest["model_discrepancy_residual_quantity"]
        not in MODEL_DISCREPANCY_QUANTITIES
    ):
        raise ArtifactValidationError("manifest model discrepancy quantity is invalid")
    if manifest["model_discrepancy_interval_model"] not in MODEL_DISCREPANCY_INTERVAL_MODELS:
        raise ArtifactValidationError("manifest model discrepancy interval model is invalid")
    _sha256_identifier(
        manifest["estimation_request_fingerprint"],
        "manifest.estimation_request_fingerprint",
    )
    _sha256_identifier(manifest["request_fingerprint"], "manifest.request_fingerprint")
    candidate_ids = _string_list(manifest["candidate_ids"], "candidate_ids")
    sample_ids = _string_list(manifest["plant_sample_ids"], "plant_sample_ids")
    bag_ids = _string_list(manifest["bag_ids"], "bag_ids")
    for bag_id in bag_ids:
        _safe_file_identifier(bag_id, "bag_id")
    nondominated = _string_list(
        manifest["nondominated_candidate_ids"], "nondominated_candidate_ids"
    )
    recommended = _string_list(
        manifest["recommended_candidate_ids"],
        "recommended_candidate_ids",
        allow_empty=True,
    )
    if (
        candidate_ids[0] != "current"
        or not set(nondominated).issubset(set(candidate_ids))
        or not set(recommended).issubset(set(nondominated))
        or bool(recommended) != bool(manifest["recommendation_available"])
        or (bool(recommended) == bool(manifest["rejection_reason"]))
    ):
        raise ArtifactValidationError("manifest recommendation is inconsistent")
    selected = manifest["selected_candidate_id"]
    if selected is not None and selected not in recommended:
        raise ArtifactValidationError("selected candidate must be recommended")
    q = np.asarray(manifest["model_discrepancy_q_diagonal"], dtype=float)
    if q.shape != (6,) or np.any(~np.isfinite(q)) or np.any(q < 0.0):
        raise ArtifactValidationError("manifest Q diagonal is invalid")
    for key in ("model_discrepancy_base_seed", "model_discrepancy_replicates"):
        value = manifest[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ArtifactValidationError("manifest {} is invalid".format(key))
    if manifest["model_discrepancy_base_seed"] >= 2 ** 64:
        raise ArtifactValidationError("model discrepancy seed exceeds uint64")
    if manifest["model_discrepancy_replicates"] < 1:
        raise ArtifactValidationError("replicates must be positive")

    artifacts = manifest["artifacts"]
    required_artifacts = ("source_samples", "candidate_particles", "summary", "bags")
    optional_artifacts = ("proposed_yaml", "proposed_diff_yaml")
    if not isinstance(artifacts, Mapping):
        raise ArtifactValidationError("manifest.artifacts must be an object")
    if set(artifacts) - set(required_artifacts + optional_artifacts) or set(
        required_artifacts
    ) - set(artifacts):
        raise ArtifactValidationError("manifest artifact keys are invalid")
    if (selected is None) != ("proposed_yaml" not in artifacts):
        raise ArtifactValidationError("selected candidate YAML contract is invalid")
    if (selected is None) != ("proposed_diff_yaml" not in artifacts):
        raise ArtifactValidationError("selected candidate diff contract is invalid")
    source_path = _descriptor(root, artifacts["source_samples"], "source_samples.npz")
    candidate_path = _descriptor(
        root, artifacts["candidate_particles"], "candidate_particles.npz"
    )
    summary_path = _descriptor(root, artifacts["summary"], "summary.npz")
    bag_descriptors = artifacts["bags"]
    if not isinstance(bag_descriptors, Mapping) or set(bag_descriptors) != set(bag_ids):
        raise ArtifactValidationError("bag artifact descriptors are invalid")
    bag_paths = {
        bag_id: _descriptor(
            root, bag_descriptors[bag_id], "bags/{}.npz".format(bag_id)
        )
        for bag_id in bag_ids
    }
    source = _arrays(source_path, _SOURCE_KEYS)
    candidates = _arrays(candidate_path, _CANDIDATE_KEYS)
    summary = _arrays(summary_path, _SUMMARY_KEYS)
    bags = {bag_id: _arrays(path, _BAG_KEYS) for bag_id, path in bag_paths.items()}
    _validate_payloads(manifest, source, candidates, summary, bags)
    proposed = None
    proposed_diff = None
    if selected is not None:
        proposed_path = _descriptor(
            root, artifacts["proposed_yaml"], "proposed_GimbalrotorControl.yaml"
        )
        diff_path = _descriptor(
            root,
            artifacts["proposed_diff_yaml"],
            "proposed_GimbalrotorControl.diff.yaml",
        )
        proposed = proposed_path.read_text(encoding="utf-8")
        proposed_diff = diff_path.read_text(encoding="utf-8")
        if not proposed or not proposed_diff:
            raise ArtifactValidationError("proposed PID YAML files must not be empty")
    for mapping in (source, candidates, summary, *bags.values()):
        for value in mapping.values():
            value.setflags(write=False)
    return PidProposalEvaluationArtifact(
        root=root,
        manifest=MappingProxyType(dict(manifest)),
        source_samples=MappingProxyType(source),
        candidate_particles=MappingProxyType(candidates),
        summary=MappingProxyType(summary),
        bags=MappingProxyType(
            {key: MappingProxyType(value) for key, value in bags.items()}
        ),
        proposed_yaml=proposed,
        proposed_diff_yaml=proposed_diff,
    )


def load_pid_proposal_evaluation(
    root: Union[str, Path]
) -> PidProposalEvaluationArtifact:
    selected = Path(root).expanduser().resolve()
    if not selected.is_dir():
        raise ArtifactValidationError("PID evaluation root is not a directory")
    return _load(selected)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def write_pid_proposal_evaluation(
    root: Union[str, Path],
    *,
    identity: PidEvaluationArtifactIdentity,
    posterior: PhysicalPlantPosterior,
    evaluation: PidCandidateEvaluation,
    selected_candidate_id: Optional[str] = None,
) -> PidProposalEvaluationArtifact:
    """Atomically publish a complete immutable PID evaluation directory."""

    if not isinstance(identity, PidEvaluationArtifactIdentity):
        raise TypeError("identity has the wrong type")
    if not isinstance(posterior, PhysicalPlantPosterior):
        raise TypeError("posterior has the wrong type")
    if not isinstance(evaluation, PidCandidateEvaluation):
        raise TypeError("evaluation has the wrong type")
    selected = None if selected_candidate_id is None else str(selected_candidate_id)
    recommended = set(evaluation.decision.recommended_candidate_ids)
    if selected is not None and selected not in recommended:
        raise ArtifactValidationError(
            "selected_candidate_id must name a recommended candidate"
        )
    destination = Path(root).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ArtifactValidationError("PID evaluation destination already exists")
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
        source = _source_payload(posterior)
        candidates = _candidate_payload(evaluation)
        summary = _summary_payload(evaluation)
        bags = {
            bag_id: _bag_payload(evaluation, bag_id)
            for bag_id in evaluation.bag_ids
        }
        write_npz_atomic(staging / "source_samples.npz", source)
        write_npz_atomic(staging / "candidate_particles.npz", candidates)
        write_npz_atomic(staging / "summary.npz", summary)
        for bag_id, payload in bags.items():
            write_npz_atomic(staging / "bags" / "{}.npz".format(bag_id), payload)
        artifact_descriptors: Dict[str, Any] = {
            "source_samples": {
                "path": "source_samples.npz",
                "sha256": _file_sha256(staging / "source_samples.npz"),
            },
            "candidate_particles": {
                "path": "candidate_particles.npz",
                "sha256": _file_sha256(staging / "candidate_particles.npz"),
            },
            "summary": {
                "path": "summary.npz",
                "sha256": _file_sha256(staging / "summary.npz"),
            },
            "bags": {
                bag_id: {
                    "path": "bags/{}.npz".format(bag_id),
                    "sha256": _file_sha256(
                        staging / "bags" / "{}.npz".format(bag_id)
                    ),
                }
                for bag_id in evaluation.bag_ids
            },
        }
        if selected is not None:
            candidate = next(
                value
                for value in evaluation.candidates
                if value.candidate_id == selected
            )
            current = evaluation.candidates[0].configuration
            proposed_path = staging / "proposed_GimbalrotorControl.yaml"
            diff_path = staging / "proposed_GimbalrotorControl.diff.yaml"
            _write_text(
                proposed_path, render_proposed_pid_yaml(candidate.configuration)
            )
            _write_text(
                diff_path,
                render_pid_diff_yaml(
                    PidGainComparison.from_configurations(
                        current, candidate.configuration
                    )
                ),
            )
            artifact_descriptors["proposed_yaml"] = {
                "path": proposed_path.name,
                "sha256": _file_sha256(proposed_path),
            }
            artifact_descriptors["proposed_diff_yaml"] = {
                "path": diff_path.name,
                "sha256": _file_sha256(diff_path),
            }
        manifest = _manifest(identity, evaluation, selected, artifact_descriptors)
        _validate_payloads(manifest, source, candidates, summary, bags)
        write_json_atomic(staging / "manifest.json", manifest)
        _load(staging)
        os.replace(str(staging), str(destination))
        published = True
        return _load(destination)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


__all__ = [
    "PID_PROPOSAL_EVALUATION_SCHEMA",
    "PidEvaluationArtifactIdentity",
    "PidProposalEvaluationArtifact",
    "load_pid_proposal_evaluation",
    "write_pid_proposal_evaluation",
]
