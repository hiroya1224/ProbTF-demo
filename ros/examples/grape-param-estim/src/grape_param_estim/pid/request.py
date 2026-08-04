"""Strict GUI-to-worker request for posterior PID cross-evaluation."""

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from grape_param_estim.artifact_io import (
    ArtifactValidationError,
    read_json,
    request_fingerprint,
)
from grape_param_estim.pid.particle_search import MODEL_DISCREPANCY_POLICIES


PID_EVALUATION_REQUEST_SCHEMA = (
    "grape-param-estim/pid-proposal-evaluation-request/v1"
)
PLANT_SAMPLE_SUBSET_METHODS = (
    "all_equal_weight_mcmc_samples",
    "explicit_equal_weight_mcmc_subset",
)
PID_CANDIDATE_SOURCES = ("current", "sample-derived", "user")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_SAFE_BAG_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def _error(location: str, message: str) -> None:
    raise ArtifactValidationError("{} {}".format(location, message))


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error(location, "must be an object")
    return value


def _keys(value: Mapping[str, Any], expected: Sequence[str], location: str) -> None:
    missing = sorted(set(expected) - set(value))
    unknown = sorted(set(value) - set(expected))
    if missing or unknown:
        _error(
            location,
            "keys disagree; missing={}, unknown={}".format(missing, unknown),
        )


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value or "\x00" in value:
        _error(location, "must be a canonical non-empty string")
    return value


def _identifier(value: Any, location: str, *, bag: bool = False) -> str:
    selected = _string(value, location)
    pattern = _SAFE_BAG_ID if bag else _SAFE_ID
    if pattern.fullmatch(selected) is None:
        _error(location, "must be a safe identifier")
    return selected


def _choice(value: Any, choices: Sequence[str], location: str) -> str:
    selected = _string(value, location)
    if selected not in choices:
        _error(location, "must be one of {}".format(tuple(choices)))
    return selected


def _number(value: Any, location: str, *, lower: float, upper: float = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _error(location, "must be a finite number")
    selected = float(value)
    if not np.isfinite(selected) or selected < lower or (
        upper is not None and selected > upper
    ):
        _error(location, "is outside its allowed range")
    return selected


def _integer(value: Any, location: str, *, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        _error(location, "must be an integer in [{}, {}]".format(minimum, maximum))
    return value


def _vector(value: Any, size: int, location: str, *, nonnegative: bool) -> np.ndarray:
    if not isinstance(value, list) or len(value) != size:
        _error(location, "must contain {} numbers".format(size))
    try:
        selected = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError(
            "{} must contain {} finite numbers".format(location, size)
        ) from error
    if (
        selected.shape != (size,)
        or np.any(~np.isfinite(selected))
        or (nonnegative and np.any(selected < 0.0))
    ):
        _error(location, "must contain {} valid numbers".format(size))
    selected.setflags(write=False)
    return selected


def _absolute_path(value: Any, location: str, *, must_exist: bool) -> Path:
    selected = Path(_string(value, location)).expanduser()
    if not selected.is_absolute():
        _error(location, "must be an absolute path")
    selected = selected.resolve()
    if must_exist and not selected.exists():
        _error(location, "does not exist")
    return selected


@dataclass(frozen=True)
class PidEvaluationBagRequest:
    bag_id: str
    path: Path
    sha256: str
    roll_pitch_integration_active: bool


@dataclass(frozen=True)
class PidEvaluationCandidateRequest:
    candidate_id: str
    source: str
    source_sample_id: Optional[str]
    gain_values: Optional[np.ndarray]


@dataclass(frozen=True)
class PidEvaluationRequest:
    source_path: Path
    payload: Mapping[str, Any]
    fingerprint: str
    evaluation_id: str
    estimation_run: Path
    output_directory: Path
    resume: bool
    forecast_workers: Union[str, int]
    baseline_bag_id: str
    selected_mode_id: Optional[str]
    bags: Tuple[PidEvaluationBagRequest, ...]
    fixed_linear_drag: np.ndarray
    fixed_angular_drag: np.ndarray
    discrepancy_policy: str
    discrepancy_base_seed: int
    discrepancy_replicates: int
    plant_sample_subset_method: str
    plant_sample_ids: Optional[Tuple[str, ...]]
    candidates: Tuple[PidEvaluationCandidateRequest, ...]
    quantile_level: float
    cvar_level: float
    selected_candidate_id: Optional[str]
    maximum_reference_age_seconds: float


def _bags(value: Any) -> Tuple[PidEvaluationBagRequest, ...]:
    if not isinstance(value, list) or not value:
        _error("request.bags", "must be a non-empty list")
    result = []
    for index, raw in enumerate(value):
        location = "request.bags[{}]".format(index)
        item = _mapping(raw, location)
        _keys(
            item,
            ("bag_id", "path", "sha256", "roll_pitch_integration_active"),
            location,
        )
        sha256 = _string(item["sha256"], location + ".sha256")
        if _SHA256.fullmatch(sha256) is None:
            _error(location + ".sha256", "must have form sha256:<64 lowercase hex>")
        active = item["roll_pitch_integration_active"]
        if not isinstance(active, bool):
            _error(location + ".roll_pitch_integration_active", "must be boolean")
        result.append(
            PidEvaluationBagRequest(
                bag_id=_identifier(item["bag_id"], location + ".bag_id", bag=True),
                path=_absolute_path(item["path"], location + ".path", must_exist=True),
                sha256=sha256,
                roll_pitch_integration_active=active,
            )
        )
    if len({value.bag_id for value in result}) != len(result):
        _error("request.bags", "contains duplicate bag IDs")
    return tuple(result)


def _candidates(value: Any) -> Tuple[PidEvaluationCandidateRequest, ...]:
    if not isinstance(value, list) or not value:
        _error("request.candidates", "must be a non-empty list")
    result = []
    for index, raw in enumerate(value):
        location = "request.candidates[{}]".format(index)
        item = _mapping(raw, location)
        _keys(
            item,
            ("candidate_id", "source", "source_sample_id", "gain_values"),
            location,
        )
        candidate_id = _identifier(item["candidate_id"], location + ".candidate_id")
        source = _choice(item["source"], PID_CANDIDATE_SOURCES, location + ".source")
        sample_id = item["source_sample_id"]
        gains = item["gain_values"]
        if source == "current":
            if candidate_id != "current" or sample_id is not None or gains is not None:
                _error(location, "current must use ID current and null source/gains")
            selected_sample = None
            selected_gains = None
        elif source == "sample-derived":
            if sample_id is None or gains is not None:
                _error(location, "sample-derived requires source_sample_id and null gains")
            selected_sample = _string(sample_id, location + ".source_sample_id")
            selected_gains = None
        else:
            if sample_id is not None or not isinstance(gains, list) or len(gains) != 4:
                _error(location, "user requires null source sample and 4x3 gains")
            rows = tuple(
                _vector(row, 3, location + ".gain_values", nonnegative=True)
                for row in gains
            )
            selected_sample = None
            selected_gains = np.vstack(rows)
            selected_gains.setflags(write=False)
        result.append(
            PidEvaluationCandidateRequest(
                candidate_id=candidate_id,
                source=source,
                source_sample_id=selected_sample,
                gain_values=selected_gains,
            )
        )
    identifiers = tuple(value.candidate_id for value in result)
    if len(set(identifiers)) != len(identifiers) or identifiers.count("current") != 1:
        _error("request.candidates", "must contain unique IDs and current exactly once")
    return tuple(result)


def validate_pid_evaluation_request(
    payload: Mapping[str, Any],
    source_path: Union[str, Path] = "<memory>",
) -> PidEvaluationRequest:
    request = _mapping(payload, "request")
    expected = (
        "schema",
        "evaluation_id",
        "estimation_run",
        "output_directory",
        "resume",
        "forecast_workers",
        "baseline_bag_id",
        "selected_mode_id",
        "bags",
        "fixed_plant_parameters",
        "model_discrepancy",
        "plant_sample_subset",
        "candidates",
        "quantile_level",
        "cvar_level",
        "selected_candidate_id",
        "maximum_reference_age_seconds",
    )
    _keys(request, expected, "request")
    _choice(request["schema"], (PID_EVALUATION_REQUEST_SCHEMA,), "request.schema")
    evaluation_id = _identifier(request["evaluation_id"], "request.evaluation_id")
    estimation_run = _absolute_path(
        request["estimation_run"], "request.estimation_run", must_exist=True
    )
    output_directory = _absolute_path(
        request["output_directory"], "request.output_directory", must_exist=False
    )
    resume = request["resume"]
    if not isinstance(resume, bool):
        _error("request.resume", "must be boolean")
    raw_workers = request["forecast_workers"]
    if raw_workers == "auto":
        forecast_workers: Union[str, int] = "auto"
    else:
        forecast_workers = _integer(
            raw_workers,
            "request.forecast_workers",
            minimum=1,
            maximum=32,
        )
    bags = _bags(request["bags"])
    baseline = _identifier(
        request["baseline_bag_id"], "request.baseline_bag_id", bag=True
    )
    if baseline not in {value.bag_id for value in bags}:
        _error("request.baseline_bag_id", "must name a requested bag")
    mode = request["selected_mode_id"]
    selected_mode = None if mode is None else _string(mode, "request.selected_mode_id")
    fixed = _mapping(request["fixed_plant_parameters"], "request.fixed_plant_parameters")
    _keys(fixed, ("linear_drag", "angular_drag"), "request.fixed_plant_parameters")
    linear_drag = _vector(
        fixed["linear_drag"], 3, "request.fixed_plant_parameters.linear_drag", nonnegative=True
    )
    angular_drag = _vector(
        fixed["angular_drag"], 3, "request.fixed_plant_parameters.angular_drag", nonnegative=True
    )
    discrepancy = _mapping(request["model_discrepancy"], "request.model_discrepancy")
    _keys(discrepancy, ("policy", "base_seed", "replicates"), "request.model_discrepancy")
    policy = _choice(
        discrepancy["policy"], MODEL_DISCREPANCY_POLICIES, "request.model_discrepancy.policy"
    )
    base_seed = _integer(
        discrepancy["base_seed"],
        "request.model_discrepancy.base_seed",
        minimum=0,
        maximum=2 ** 64 - 1,
    )
    replicates = _integer(
        discrepancy["replicates"],
        "request.model_discrepancy.replicates",
        minimum=1,
        maximum=10 ** 6,
    )
    subset = _mapping(request["plant_sample_subset"], "request.plant_sample_subset")
    _keys(subset, ("method", "sample_ids"), "request.plant_sample_subset")
    subset_method = _choice(
        subset["method"], PLANT_SAMPLE_SUBSET_METHODS, "request.plant_sample_subset.method"
    )
    raw_sample_ids = subset["sample_ids"]
    if subset_method == "all_equal_weight_mcmc_samples":
        if raw_sample_ids is not None:
            _error("request.plant_sample_subset.sample_ids", "must be null for all samples")
        sample_ids = None
    else:
        if not isinstance(raw_sample_ids, list) or not raw_sample_ids:
            _error("request.plant_sample_subset.sample_ids", "must be a non-empty list")
        sample_ids = tuple(
            _string(value, "request.plant_sample_subset.sample_ids")
            for value in raw_sample_ids
        )
        if len(set(sample_ids)) != len(sample_ids):
            _error("request.plant_sample_subset.sample_ids", "contains duplicates")
    candidates = _candidates(request["candidates"])
    quantile = _number(request["quantile_level"], "request.quantile_level", lower=0.0, upper=1.0)
    if quantile in (0.0, 1.0):
        _error("request.quantile_level", "must be strictly between zero and one")
    cvar = _number(request["cvar_level"], "request.cvar_level", lower=0.0, upper=1.0)
    if cvar == 1.0:
        _error("request.cvar_level", "must be less than one")
    selected = request["selected_candidate_id"]
    selected_candidate = (
        None
        if selected is None
        else _identifier(selected, "request.selected_candidate_id")
    )
    if selected_candidate is not None and selected_candidate not in {
        value.candidate_id for value in candidates
    }:
        _error("request.selected_candidate_id", "must name a requested candidate")
    maximum_reference_age = _number(
        request["maximum_reference_age_seconds"],
        "request.maximum_reference_age_seconds",
        lower=0.0,
    )
    if maximum_reference_age == 0.0:
        _error("request.maximum_reference_age_seconds", "must be positive")
    frozen = MappingProxyType(dict(request))
    fingerprint_payload = dict(request)
    # ``resume`` changes process control only.  As in the batch-estimation
    # request, the interrupted and resumed invocations must retain one exact
    # request identity.  The explicit worker setting remains part of that
    # identity even though it is recorded as a non-scientific runtime setting.
    fingerprint_payload["resume"] = False
    return PidEvaluationRequest(
        source_path=Path(source_path),
        payload=frozen,
        fingerprint=request_fingerprint(fingerprint_payload),
        evaluation_id=evaluation_id,
        estimation_run=estimation_run,
        output_directory=output_directory,
        resume=resume,
        forecast_workers=forecast_workers,
        baseline_bag_id=baseline,
        selected_mode_id=selected_mode,
        bags=bags,
        fixed_linear_drag=linear_drag,
        fixed_angular_drag=angular_drag,
        discrepancy_policy=policy,
        discrepancy_base_seed=base_seed,
        discrepancy_replicates=replicates,
        plant_sample_subset_method=subset_method,
        plant_sample_ids=sample_ids,
        candidates=candidates,
        quantile_level=quantile,
        cvar_level=cvar,
        selected_candidate_id=selected_candidate,
        maximum_reference_age_seconds=maximum_reference_age,
    )


def load_pid_evaluation_request(path: Union[str, Path]) -> PidEvaluationRequest:
    source = Path(path).expanduser().resolve()
    return validate_pid_evaluation_request(read_json(source), source)


__all__ = [
    "PID_CANDIDATE_SOURCES",
    "PID_EVALUATION_REQUEST_SCHEMA",
    "PLANT_SAMPLE_SUBSET_METHODS",
    "PidEvaluationBagRequest",
    "PidEvaluationCandidateRequest",
    "PidEvaluationRequest",
    "load_pid_evaluation_request",
    "validate_pid_evaluation_request",
]
