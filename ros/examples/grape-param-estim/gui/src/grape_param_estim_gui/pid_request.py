"""Qt-free construction of strict posterior-predictive PID requests."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .artifact_loader import AssimilationRun


PID_EVALUATION_REQUEST_SCHEMA = "grape-param-estim/pid-evaluation-request/v1"
ASSIMILATION_RUN_SCHEMA = "grape-param-estim/assimilation-run/v1"
COMPLETE_STATUS = "complete"
RESIDUAL_POLICIES = ("posterior_replay", "zero")
SELECTION_TARGETS = ("member-derived", "user")
USER_CANDIDATE_ID = "user-exact"
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class PidEvaluationLaunchOptions:
    """User-visible choices for exact member-derived and optional user gains."""

    source_member_id: int
    baseline_bag_id: str
    residual_policy: str = "posterior_replay"
    cvar_level: float = 0.9
    user_candidate_values: tuple[tuple[float, ...], ...] | None = None
    selected_candidate_source: str | None = None

    def __post_init__(self) -> None:
        member_id = self.source_member_id
        if isinstance(member_id, bool) or not isinstance(member_id, (int, np.integer)):
            raise ValueError("source_member_id must be an integer")
        member_id = int(member_id)
        if member_id < 0:
            raise ValueError("source_member_id cannot be negative")
        baseline = str(self.baseline_bag_id)
        if not baseline:
            raise ValueError("baseline_bag_id cannot be empty")
        policy = str(self.residual_policy)
        if policy not in RESIDUAL_POLICIES:
            raise ValueError("unknown residual policy")
        level = float(self.cvar_level)
        if not math.isfinite(level) or not 0.0 <= level < 1.0:
            raise ValueError("cvar_level must be finite and in [0, 1)")
        user_values = self.user_candidate_values
        if user_values is not None:
            values = np.asarray(user_values, dtype=float)
            if (
                values.shape != (4, 3)
                or np.any(~np.isfinite(values))
                or np.any(values < 0.0)
            ):
                raise ValueError(
                    "user_candidate_values must be a finite non-negative 4x3 array"
                )
            user_values = tuple(
                tuple(float(value) for value in row) for row in values
            )
        selection = self.selected_candidate_source
        if selection is not None:
            selection = str(selection)
            if selection not in SELECTION_TARGETS:
                raise ValueError("unknown selected candidate source")
            if selection == "user" and user_values is None:
                raise ValueError(
                    "the exact user candidate must be included before selecting it"
                )
        object.__setattr__(self, "source_member_id", member_id)
        object.__setattr__(self, "baseline_bag_id", baseline)
        object.__setattr__(self, "residual_policy", policy)
        object.__setattr__(self, "cvar_level", level)
        object.__setattr__(self, "user_candidate_values", user_values)
        object.__setattr__(self, "selected_candidate_source", selection)

    @property
    def candidate_id(self) -> str:
        return "member-{}-exact".format(self.source_member_id)

    @property
    def selected_candidate_id(self) -> str | None:
        if self.selected_candidate_source == "member-derived":
            return self.candidate_id
        if self.selected_candidate_source == "user":
            return USER_CANDIDATE_ID
        return None


def _selected_bag_ids(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    raw = manifest.get("selected_bag_ids")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("assimilation manifest has no selected bags")
    result = tuple(str(value) for value in raw)
    if any(not value for value in result) or len(set(result)) != len(result):
        raise ValueError("assimilation manifest selected bags are invalid")
    return result


def build_pid_evaluation_request(
    run: AssimilationRun,
    evaluation_id: str,
    options: PidEvaluationLaunchOptions,
) -> dict[str, Any]:
    """Build the backend v1 request without inventing a representative member.

    Thresholds deliberately remain unconfigured.  The member-derived proposal
    follows the shared GUI selection, an exact user configuration is optional,
    and the current controller is included exactly once as the baseline.
    """

    if not isinstance(run, AssimilationRun):
        raise TypeError("run must be a loaded AssimilationRun")
    identifier = str(evaluation_id)
    if not _SAFE_IDENTIFIER.fullmatch(identifier):
        raise ValueError("evaluation_id is not a safe identifier")
    if not isinstance(options, PidEvaluationLaunchOptions):
        raise TypeError("options must be PidEvaluationLaunchOptions")
    manifest = run.manifest
    if manifest.get("schema") != ASSIMILATION_RUN_SCHEMA:
        raise ValueError("source is not an assimilation run")
    if manifest.get("status") != COMPLETE_STATUS:
        raise ValueError("source assimilation run is not complete")
    selected_bags = _selected_bag_ids(manifest)
    if options.baseline_bag_id not in selected_bags:
        raise ValueError("baseline_bag_id is not in the source run")
    member_ids = np.asarray(run.shared_posterior.member_id)
    if member_ids.ndim != 1 or options.source_member_id not in set(
        int(value) for value in member_ids.tolist()
    ):
        raise ValueError("selected member is not in the source run")
    source_root = Path(run.root).expanduser().resolve()
    if not source_root.is_dir():
        raise ValueError("source assimilation run directory is unavailable")

    candidates: list[dict[str, Any]] = [
        {"candidate_id": "current", "source": "current"},
        {
            "candidate_id": options.candidate_id,
            "source": "member-derived",
            "source_member_id": options.source_member_id,
        },
    ]
    if options.user_candidate_values is not None:
        candidates.append(
            {
                "candidate_id": USER_CANDIDATE_ID,
                "source": "user",
                "values": [
                    list(row) for row in options.user_candidate_values
                ],
            }
        )
    return {
        "schema": PID_EVALUATION_REQUEST_SCHEMA,
        "evaluation_id": identifier,
        "assimilation_run": str(source_root),
        "baseline_bag_id": options.baseline_bag_id,
        "residual_policy": options.residual_policy,
        "cvar_level": options.cvar_level,
        "thresholds": {
            "position": None,
            "orientation": None,
            "position_metric": "position_rmse",
            "orientation_metric": "orientation_rmse",
        },
        "candidates": candidates,
        "selected_candidate_id": options.selected_candidate_id,
    }


__all__ = [
    "ASSIMILATION_RUN_SCHEMA",
    "COMPLETE_STATUS",
    "PID_EVALUATION_REQUEST_SCHEMA",
    "PidEvaluationLaunchOptions",
    "RESIDUAL_POLICIES",
    "SELECTION_TARGETS",
    "USER_CANDIDATE_ID",
    "build_pid_evaluation_request",
]
