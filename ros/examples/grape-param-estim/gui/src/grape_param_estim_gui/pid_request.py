"""Qt-free construction of strict posterior PID-evaluation requests."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from .artifact_loader import BatchEstimationRun


PID_EVALUATION_REQUEST_SCHEMA = (
    "grape-param-estim/pid-proposal-evaluation-request/v2"
)
BATCH_ESTIMATION_RUN_SCHEMA = "grape-param-estim/batch-estimation-run/v1"
COMPLETE_STATUS = "complete"
MODEL_DISCREPANCY_POLICIES = (
    "zero_model_discrepancy",
    "sample_model_discrepancy",
)
SELECTION_TARGETS = ("sample-derived", "user")
USER_CANDIDATE_ID = "user-exact"
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


def sample_candidate_id(sample_id: str) -> str:
    selected = str(sample_id)
    if not selected or selected.strip() != selected or "\x00" in selected:
        raise ValueError("source_sample_id must be a canonical non-empty string")
    encoded = base64.urlsafe_b64encode(selected.encode("utf-8")).decode("ascii")
    return "sample_{}".format(encoded.rstrip("="))


def _vector3(value: object, name: str) -> tuple[float, float, float]:
    selected = np.asarray(value, dtype=float)
    if selected.shape != (3,) or np.any(~np.isfinite(selected)) or np.any(selected < 0.0):
        raise ValueError("{} must contain three finite non-negative values".format(name))
    return tuple(float(item) for item in selected)


def _bag_inputs(
    values: object,
) -> tuple[tuple[str, str, str, bool], ...]:
    result = []
    for raw in tuple(values):  # type: ignore[arg-type]
        if len(raw) != 4:
            raise ValueError("each bag input must contain ID, path, SHA256, and integration flag")
        bag_id, path, sha256, active = raw
        source = Path(path).expanduser().resolve()
        digest = str(sha256)
        if len(digest) == 64:
            digest = "sha256:" + digest
        if (
            not str(bag_id)
            or not source.is_file()
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        ):
            raise ValueError("bag input identity is invalid")
        if not isinstance(active, (bool, np.bool_)):
            raise ValueError("roll/pitch integration flag must be boolean")
        result.append((str(bag_id), str(source), digest, bool(active)))
    if not result or len({item[0] for item in result}) != len(result):
        raise ValueError("bag inputs must have unique non-empty IDs")
    return tuple(result)


@dataclass(frozen=True)
class PidEvaluationLaunchOptions:
    """Every GUI-controlled assumption required by the strict worker."""

    source_sample_id: str
    baseline_bag_id: str
    selected_mode_id: str
    bags: tuple[tuple[str, str, str, bool], ...]
    fixed_linear_drag: tuple[float, float, float]
    fixed_angular_drag: tuple[float, float, float]
    model_discrepancy_policy: str
    maximum_derived_candidates: int | None = 12
    quantile_level: float = 0.95
    cvar_level: float = 0.9
    base_seed: int = 0
    replicates: int = 1
    maximum_reference_age_seconds: float = 0.5
    forecast_workers: str | int = "auto"
    user_candidate_values: tuple[tuple[float, ...], ...] | None = None
    selected_candidate_source: str | None = None

    def __post_init__(self) -> None:
        sample = str(self.source_sample_id)
        sample_candidate_id(sample)
        baseline = str(self.baseline_bag_id)
        mode = str(self.selected_mode_id)
        if not baseline or not mode:
            raise ValueError("baseline_bag_id and selected_mode_id cannot be empty")
        policy = str(self.model_discrepancy_policy)
        if policy not in MODEL_DISCREPANCY_POLICIES:
            raise ValueError("unknown model discrepancy policy")
        maximum_candidates = self.maximum_derived_candidates
        if maximum_candidates is not None and (
            isinstance(maximum_candidates, (bool, np.bool_))
            or not isinstance(maximum_candidates, (int, np.integer))
            or int(maximum_candidates) < 1
            or int(maximum_candidates) > 10 ** 6
        ):
            raise ValueError(
                "maximum_derived_candidates must be null or a positive integer"
            )
        quantile = float(self.quantile_level)
        cvar = float(self.cvar_level)
        maximum_age = float(self.maximum_reference_age_seconds)
        if not math.isfinite(quantile) or not 0.0 < quantile < 1.0:
            raise ValueError("quantile_level must be finite and in (0, 1)")
        if not math.isfinite(cvar) or not 0.0 <= cvar < 1.0:
            raise ValueError("cvar_level must be finite and in [0, 1)")
        if not math.isfinite(maximum_age) or maximum_age <= 0.0:
            raise ValueError("maximum_reference_age_seconds must be positive")
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, (int, np.integer)) or not 0 <= int(self.base_seed) < 2**64:
            raise ValueError("base_seed must be an unsigned 64-bit integer")
        if isinstance(self.replicates, bool) or not isinstance(self.replicates, (int, np.integer)) or int(self.replicates) < 1:
            raise ValueError("replicates must be a positive integer")
        workers = self.forecast_workers
        if workers != "auto" and (
            isinstance(workers, (bool, np.bool_))
            or not isinstance(workers, (int, np.integer))
            or not 1 <= int(workers) <= 32
        ):
            raise ValueError("forecast_workers must be auto or an integer in [1, 32]")
        user_values = self.user_candidate_values
        if user_values is not None:
            values = np.asarray(user_values, dtype=float)
            if values.shape != (4, 3) or np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError("user_candidate_values must be a finite non-negative 4x3 array")
            user_values = tuple(tuple(float(item) for item in row) for row in values)
        selection = self.selected_candidate_source
        if selection is not None and selection not in SELECTION_TARGETS:
            raise ValueError("unknown selected candidate source")
        if selection == "user" and user_values is None:
            raise ValueError("the exact user candidate must be included before selecting it")
        object.__setattr__(self, "source_sample_id", sample)
        object.__setattr__(self, "baseline_bag_id", baseline)
        object.__setattr__(self, "selected_mode_id", mode)
        object.__setattr__(self, "bags", _bag_inputs(self.bags))
        object.__setattr__(self, "fixed_linear_drag", _vector3(self.fixed_linear_drag, "fixed_linear_drag"))
        object.__setattr__(self, "fixed_angular_drag", _vector3(self.fixed_angular_drag, "fixed_angular_drag"))
        object.__setattr__(self, "model_discrepancy_policy", policy)
        object.__setattr__(
            self,
            "maximum_derived_candidates",
            None if maximum_candidates is None else int(maximum_candidates),
        )
        object.__setattr__(self, "quantile_level", quantile)
        object.__setattr__(self, "cvar_level", cvar)
        object.__setattr__(self, "base_seed", int(self.base_seed))
        object.__setattr__(self, "replicates", int(self.replicates))
        object.__setattr__(
            self,
            "forecast_workers",
            "auto" if workers == "auto" else int(workers),
        )
        object.__setattr__(self, "maximum_reference_age_seconds", maximum_age)
        object.__setattr__(self, "user_candidate_values", user_values)

    @property
    def candidate_id(self) -> str:
        return sample_candidate_id(self.source_sample_id)

    @property
    def selected_candidate_id(self) -> str | None:
        if self.selected_candidate_source == "sample-derived":
            return self.candidate_id
        if self.selected_candidate_source == "user":
            return USER_CANDIDATE_ID
        return None


def build_pid_evaluation_request(
    run: BatchEstimationRun,
    evaluation_id: str,
    output_directory: str | Path,
    options: PidEvaluationLaunchOptions,
) -> dict[str, Any]:
    """Build the backend request without inventing a representative plant."""

    if not isinstance(run, BatchEstimationRun):
        raise TypeError("run must be a loaded BatchEstimationRun")
    identifier = str(evaluation_id)
    if _SAFE_IDENTIFIER.fullmatch(identifier) is None:
        raise ValueError("evaluation_id is not a safe identifier")
    if not isinstance(options, PidEvaluationLaunchOptions):
        raise TypeError("options must be PidEvaluationLaunchOptions")
    if run.manifest.get("schema") != BATCH_ESTIMATION_RUN_SCHEMA or run.manifest.get("status") != COMPLETE_STATUS:
        raise ValueError("source is not a complete batch estimation run")
    if run.mcmc is None or run.mcmc.size == 0:
        raise ValueError("PID evaluation requires retained MCMC samples")
    selected_bags = tuple(str(value) for value in run.manifest.get("selected_bag_ids", ()))
    if options.baseline_bag_id not in selected_bags:
        raise ValueError("baseline_bag_id is not in the source run")
    if set(item[0] for item in options.bags) != set(selected_bags):
        raise ValueError("PID evaluation bags must exactly match the source run")
    try:
        source = run.mcmc.sample(options.source_sample_id)
    except KeyError as error:
        raise ValueError("selected sample is not in the source run") from error
    if source.source_mode_id != options.selected_mode_id:
        raise ValueError("selected sample does not belong to selected_mode_id")
    source_root = Path(run.root).expanduser().resolve()
    output_root = Path(output_directory).expanduser().resolve()
    if not source_root.is_dir():
        raise ValueError("source batch estimation run directory is unavailable")
    candidates: list[dict[str, Any]] = [
        {"candidate_id": "current", "source": "current", "source_sample_id": None, "gain_values": None},
    ]
    if options.user_candidate_values is not None:
        candidates.append({"candidate_id": USER_CANDIDATE_ID, "source": "user", "source_sample_id": None, "gain_values": [list(row) for row in options.user_candidate_values]})
    return {
        "schema": PID_EVALUATION_REQUEST_SCHEMA,
        "evaluation_id": identifier,
        "estimation_run": str(source_root),
        "output_directory": str(output_root),
        "resume": False,
        "forecast_workers": options.forecast_workers,
        "baseline_bag_id": options.baseline_bag_id,
        "selected_mode_id": options.selected_mode_id,
        "bags": [{"bag_id": bag_id, "path": path, "sha256": sha256, "roll_pitch_integration_active": active} for bag_id, path, sha256, active in options.bags],
        "fixed_plant_parameters": {"linear_drag": list(options.fixed_linear_drag), "angular_drag": list(options.fixed_angular_drag)},
        "model_discrepancy": {"policy": options.model_discrepancy_policy, "base_seed": options.base_seed, "replicates": options.replicates},
        "plant_sample_subset": {"method": "all_equal_weight_mcmc_samples", "sample_ids": None},
        "derived_candidate_population": {
            "method": (
                "all_raw_mcmc_samples"
                if options.maximum_derived_candidates is None
                else "deterministic_k_medoids"
            ),
            "maximum_candidates": options.maximum_derived_candidates,
            "required_source_sample_ids": [options.source_sample_id],
        },
        "candidates": candidates,
        "quantile_level": options.quantile_level,
        "cvar_level": options.cvar_level,
        "selected_candidate_id": options.selected_candidate_id,
        "maximum_reference_age_seconds": options.maximum_reference_age_seconds,
    }


__all__ = [
    "BATCH_ESTIMATION_RUN_SCHEMA",
    "COMPLETE_STATUS",
    "MODEL_DISCREPANCY_POLICIES",
    "PID_EVALUATION_REQUEST_SCHEMA",
    "PidEvaluationLaunchOptions",
    "SELECTION_TARGETS",
    "USER_CANDIDATE_ID",
    "build_pid_evaluation_request",
    "sample_candidate_id",
]
