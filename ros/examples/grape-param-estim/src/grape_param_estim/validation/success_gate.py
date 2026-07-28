"""Held-out success-episode posterior-predictive sanity gate."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.episode import stable_hash

from .failure_event import FailureEvent
from .posterior_predictive import (
    ValidationDatasetIdentity,
    posterior_failure_probability,
    trajectory_envelope,
    validate_trajectory_envelope,
)


SUCCESS_VALIDATION_ROLE = "validation_success"


def _success_validation_result_payload(
    *,
    episode_id: str,
    role: str,
    credible_probability: float,
    trajectory_coverage: float,
    posterior_failure_probability: float,
    passed: bool,
    reasons: Sequence[str],
) -> Mapping[str, Any]:
    return {
        "episode_id": str(episode_id),
        "role": str(role),
        "credible_probability": float(credible_probability),
        "trajectory_coverage": float(trajectory_coverage),
        "posterior_failure_probability": float(
            posterior_failure_probability
        ),
        "passed": bool(passed),
        "reasons": tuple(str(item) for item in reasons),
    }


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def assert_success_episodes_validation_only(
    episodes: Sequence[Any],
) -> None:
    """Reject any successful episode assigned to an inference role."""

    for item in episodes:
        outcome_value = _field(item, "outcome", "")
        if isinstance(outcome_value, Mapping):
            outcome_value = outcome_value.get("value", "")
        outcome = str(outcome_value).lower()
        role = str(_field(item, "role", ""))
        if outcome == "success" and role != SUCCESS_VALIDATION_ROLE:
            raise ValueError(
                "success episode '{}' must have role '{}'".format(
                    _field(item, "episode_id", "UNKNOWN"),
                    SUCCESS_VALIDATION_ROLE,
                )
            )


@dataclass(frozen=True)
class SuccessGateConfig:
    credible_probability: float = 0.95
    minimum_trajectory_coverage: float = 0.95
    maximum_false_failure_probability: float = 0.05

    def __post_init__(self) -> None:
        credible = float(self.credible_probability)
        coverage = float(self.minimum_trajectory_coverage)
        false_failure = float(self.maximum_false_failure_probability)
        if (
            not 0.0 < credible < 1.0
            or not 0.0 <= coverage <= 1.0
            or not 0.0 <= false_failure <= 1.0
        ):
            raise ValueError("success-gate probabilities are invalid")
        object.__setattr__(self, "credible_probability", credible)
        object.__setattr__(self, "minimum_trajectory_coverage", coverage)
        object.__setattr__(
            self,
            "maximum_false_failure_probability",
            false_failure,
        )


@dataclass(frozen=True)
class SuccessEpisodeValidation:
    episode_id: str
    role: str
    credible_probability: float
    trajectory_coverage: float
    posterior_failure_probability: float
    passed: bool
    reasons: Tuple[str, ...]
    dataset_identity: ValidationDatasetIdentity

    def __post_init__(self) -> None:
        episode_id = str(self.episode_id).strip()
        role = str(self.role)
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "role", role)
        for name in (
            "credible_probability",
            "trajectory_coverage",
            "posterior_failure_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError("{} must lie in [0, 1]".format(name))
            object.__setattr__(self, name, value)
        if type(self.passed) is not bool:
            raise TypeError("passed must be a built-in bool")
        object.__setattr__(
            self, "reasons", tuple(str(item) for item in self.reasons)
        )
        if not isinstance(
            self.dataset_identity, ValidationDatasetIdentity
        ):
            raise TypeError(
                "dataset_identity must be ValidationDatasetIdentity"
            )
        if (
            self.dataset_identity.schema
            != "grape_held_out_success_dataset/v1"
            or self.dataset_identity.validation_result_sha256
            != stable_hash(
                _success_validation_result_payload(
                    episode_id=self.episode_id,
                    role=self.role,
                    credible_probability=self.credible_probability,
                    trajectory_coverage=self.trajectory_coverage,
                    posterior_failure_probability=(
                        self.posterior_failure_probability
                    ),
                    passed=self.passed,
                    reasons=self.reasons,
                )
            )
        ):
            raise ValueError(
                "success dataset/result identity mismatch"
            )

    @property
    def dataset_sha256(self) -> str:
        return self.dataset_identity.content_sha256

    @property
    def validation_only(self) -> bool:
        return self.role == SUCCESS_VALIDATION_ROLE


def evaluate_success_episode(
    *,
    episode_id: str,
    role: str,
    timestamps: Sequence[float],
    observed_trajectory: np.ndarray,
    predictive_trajectories: np.ndarray,
    predicted_failures: Sequence[Optional[FailureEvent]],
    weights: Optional[Sequence[float]] = None,
    config: SuccessGateConfig = SuccessGateConfig(),
    dataset_provenance_sha256: Optional[str] = None,
) -> SuccessEpisodeValidation:
    """Check 95% predictive coverage and false-failure probability."""

    if not isinstance(config, SuccessGateConfig):
        raise TypeError("config must be SuccessGateConfig")
    samples = np.asarray(predictive_trajectories)
    events = tuple(predicted_failures)
    provenance_hash = None
    if dataset_provenance_sha256 is not None:
        provenance_hash = str(dataset_provenance_sha256).lower()
        if (
            len(provenance_hash) != 64
            or any(
                item not in "0123456789abcdef"
                for item in provenance_hash
            )
        ):
            raise ValueError(
                "dataset_provenance_sha256 must be a lowercase SHA-256"
            )
    if samples.ndim < 2 or len(events) != samples.shape[0]:
        raise ValueError(
            "predicted failures and trajectories must share particle IDs"
        )
    envelope = trajectory_envelope(
        timestamps,
        samples,
        weights,
        config.credible_probability,
    )
    coverage = validate_trajectory_envelope(
        observed_trajectory,
        envelope,
        config.minimum_trajectory_coverage,
    )
    failure_probability = posterior_failure_probability(
        events, weights
    )
    reasons = []
    normalized_role = str(role)
    if normalized_role != SUCCESS_VALIDATION_ROLE:
        reasons.append("success_episode_not_validation_only")
    if not coverage.passed:
        reasons.append("posterior_predictive_coverage")
    if (
        failure_probability
        > config.maximum_false_failure_probability
    ):
        reasons.append("false_failure_probability")
    result_payload = _success_validation_result_payload(
        episode_id=str(episode_id),
        role=normalized_role,
        credible_probability=config.credible_probability,
        trajectory_coverage=coverage.element_coverage_fraction,
        posterior_failure_probability=failure_probability,
        passed=not reasons,
        reasons=tuple(reasons),
    )
    return SuccessEpisodeValidation(
        **result_payload,
        dataset_identity=ValidationDatasetIdentity._from_evaluation(
            "grape_held_out_success_dataset/v1",
            {
                "dataset_provenance_sha256": provenance_hash,
                "episode_id": str(episode_id),
                "role": normalized_role,
                "timestamps": np.asarray(timestamps, dtype=float),
                "observed_trajectory": np.asarray(
                    observed_trajectory, dtype=float
                ),
            },
            result_payload,
        ),
    )


@dataclass(frozen=True)
class SuccessGateReport:
    episodes: Tuple[SuccessEpisodeValidation, ...]
    passed: bool
    reasons: Tuple[str, ...]

    def __post_init__(self) -> None:
        episodes = tuple(self.episodes)
        if not episodes or any(
            not isinstance(item, SuccessEpisodeValidation)
            for item in episodes
        ):
            raise ValueError(
                "episodes must contain SuccessEpisodeValidation values"
            )
        if len({item.episode_id for item in episodes}) != len(episodes):
            raise ValueError("success validation episode IDs must be unique")
        if type(self.passed) is not bool:
            raise TypeError("passed must be a built-in bool")
        expected = all(item.passed for item in episodes)
        if self.passed != expected:
            raise ValueError("aggregate success gate is inconsistent")
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(
            self, "reasons", tuple(str(item) for item in self.reasons)
        )

    @property
    def dataset_sha256(self) -> Optional[str]:
        return stable_hash(
            {
                "schema": "grape_held_out_success_dataset_set/v1",
                "episodes": tuple(
                    {
                        "episode_id": item.episode_id,
                        "dataset_sha256": item.dataset_sha256,
                    }
                    for item in self.episodes
                ),
            }
        )


def evaluate_success_gate(
    episodes: Sequence[SuccessEpisodeValidation],
) -> SuccessGateReport:
    values = tuple(episodes)
    reasons = tuple(
        "{}:{}".format(item.episode_id, reason)
        for item in values
        for reason in item.reasons
    )
    return SuccessGateReport(
        episodes=values,
        passed=bool(values and all(item.passed for item in values)),
        reasons=reasons,
    )


__all__ = [
    "SUCCESS_VALIDATION_ROLE",
    "SuccessEpisodeValidation",
    "SuccessGateConfig",
    "SuccessGateReport",
    "assert_success_episodes_validation_only",
    "evaluate_success_episode",
    "evaluate_success_gate",
]
