"""Posterior-predictive trajectory and held-out failure validation."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.episode import stable_hash

from .failure_event import FailureEvent, censor_after_failure


_VALIDATION_DATASET_FACTORY_TOKEN = object()


@dataclass(frozen=True, init=False)
class ValidationDatasetIdentity:
    """Content identity measured from held-out observations, not a label."""

    schema: str
    content_sha256: str
    validation_result_sha256: str

    def __init__(
        self,
        schema: str,
        payload: Mapping[str, Any],
        validation_result: Mapping[str, Any],
        *,
        _factory_token: Any = None,
    ) -> None:
        if _factory_token is not _VALIDATION_DATASET_FACTORY_TOKEN:
            raise TypeError(
                "validation dataset identities are created only by "
                "held-out evaluation functions"
            )
        normalized_schema = str(schema)
        if normalized_schema not in (
            "grape_held_out_failure_dataset/v1",
            "grape_held_out_success_dataset/v1",
        ):
            raise ValueError("unsupported validation dataset schema")
        if not isinstance(payload, Mapping):
            raise TypeError("validation dataset payload must be a mapping")
        if not isinstance(validation_result, Mapping):
            raise TypeError(
                "validation result payload must be a mapping"
            )
        result_hash = stable_hash(dict(validation_result))
        object.__setattr__(self, "schema", normalized_schema)
        object.__setattr__(
            self, "validation_result_sha256", result_hash
        )
        object.__setattr__(
            self,
            "content_sha256",
            stable_hash(
                {
                    "schema": normalized_schema,
                    "payload": dict(payload),
                    "validation_result_sha256": result_hash,
                }
            ),
        )

    @classmethod
    def _from_evaluation(
        cls,
        schema: str,
        payload: Mapping[str, Any],
        validation_result: Mapping[str, Any],
    ) -> "ValidationDatasetIdentity":
        return cls(
            schema,
            payload,
            validation_result,
            _factory_token=_VALIDATION_DATASET_FACTORY_TOKEN,
        )

    def to_mapping(self) -> Mapping[str, str]:
        return {
            "schema": self.schema,
            "content_sha256": self.content_sha256,
            "validation_result_sha256": (
                self.validation_result_sha256
            ),
        }


def _failure_validation_result_payload(
    trajectory: Any, failure: Any, passed: bool
) -> Mapping[str, Any]:
    envelope = trajectory.envelope
    observed_failure = failure.observed_failure
    return {
        "trajectory": {
            "envelope": {
                "timestamps": envelope.timestamps,
                "lower": envelope.lower,
                "median": envelope.median,
                "upper": envelope.upper,
                "credible_probability": (
                    envelope.credible_probability
                ),
            },
            "element_coverage_fraction": (
                trajectory.element_coverage_fraction
            ),
            "time_coverage_fraction": (
                trajectory.time_coverage_fraction
            ),
            "minimum_coverage_fraction": (
                trajectory.minimum_coverage_fraction
            ),
            "evaluated_time_count": trajectory.evaluated_time_count,
            "passed": trajectory.passed,
        },
        "failure": {
            "observed_failure": {
                "failure_type": observed_failure.failure_type,
                "stamp": observed_failure.stamp,
                "detector_id": observed_failure.detector_id,
                "metadata": dict(observed_failure.metadata),
            },
            "predicted_failure_probability": (
                failure.predicted_failure_probability
            ),
            "matching_type_probability": (
                failure.matching_type_probability
            ),
            "conditional_matching_type_probability": (
                failure.conditional_matching_type_probability
            ),
            "predicted_time_lower": failure.predicted_time_lower,
            "predicted_time_upper": failure.predicted_time_upper,
            "observed_time_covered": failure.observed_time_covered,
            "credible_probability": failure.credible_probability,
            "passed": failure.passed,
            "reasons": tuple(failure.reasons),
        },
        "passed": bool(passed),
    }


def _normalize_weights(
    weights: Optional[Sequence[float]], count: int
) -> np.ndarray:
    values = (
        np.full(count, 1.0 / count)
        if weights is None
        else np.asarray(weights, dtype=float).reshape(-1)
    )
    if (
        values.shape != (count,)
        or np.any(values < 0.0)
        or not np.all(np.isfinite(values))
        or float(np.sum(values)) <= 0.0
    ):
        raise ValueError("weights must be a finite positive measure")
    output = np.array(values / np.sum(values), copy=True)
    output.setflags(write=False)
    return output


def _weighted_quantile(
    samples: np.ndarray, weights: np.ndarray, probability: float
) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    if values.ndim < 1 or values.shape[0] != weights.size:
        raise ValueError("weighted samples must have particles on axis zero")
    flattened = values.reshape(values.shape[0], -1)
    output = np.empty(flattened.shape[1], dtype=float)
    for column in range(flattened.shape[1]):
        order = np.argsort(flattened[:, column], kind="stable")
        cumulative = np.cumsum(weights[order])
        cumulative[-1] = 1.0
        index = min(
            int(np.searchsorted(cumulative, probability, side="left")),
            len(order) - 1,
        )
        output[column] = flattened[order[index], column]
    return output.reshape(values.shape[1:])


@dataclass(frozen=True)
class TrajectoryEnvelope:
    timestamps: np.ndarray
    lower: np.ndarray
    median: np.ndarray
    upper: np.ndarray
    credible_probability: float

    def __post_init__(self) -> None:
        times = np.asarray(self.timestamps, dtype=float).reshape(-1)
        if (
            times.size == 0
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("envelope timestamps must be finite and increasing")
        shape = np.asarray(self.lower).shape
        if not shape or shape[0] != times.size:
            raise ValueError("envelope arrays must use time on axis zero")
        arrays = []
        for name in ("lower", "median", "upper"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError("envelope arrays must be finite and aligned")
            copy = np.array(values, copy=True)
            copy.setflags(write=False)
            arrays.append(copy)
        if np.any(arrays[0] > arrays[1]) or np.any(arrays[1] > arrays[2]):
            raise ValueError("trajectory envelope quantiles are not ordered")
        probability = float(self.credible_probability)
        if not 0.0 < probability < 1.0:
            raise ValueError("credible_probability must lie in (0, 1)")
        times_copy = np.array(times, copy=True)
        times_copy.setflags(write=False)
        object.__setattr__(self, "timestamps", times_copy)
        object.__setattr__(self, "lower", arrays[0])
        object.__setattr__(self, "median", arrays[1])
        object.__setattr__(self, "upper", arrays[2])
        object.__setattr__(self, "credible_probability", probability)


def trajectory_envelope(
    timestamps: Sequence[float],
    predictive_trajectories: np.ndarray,
    weights: Optional[Sequence[float]] = None,
    credible_probability: float = 0.95,
) -> TrajectoryEnvelope:
    """Build a weighted pointwise credible envelope over trajectories."""

    samples = np.asarray(predictive_trajectories, dtype=float)
    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if (
        samples.ndim < 2
        or samples.shape[0] == 0
        or samples.shape[1] != times.size
        or not np.all(np.isfinite(samples))
    ):
        raise ValueError(
            "predictive_trajectories must have finite shape (P, T, ...)"
        )
    probability = float(credible_probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("credible_probability must lie in (0, 1)")
    normalized = _normalize_weights(weights, samples.shape[0])
    tail = 0.5 * (1.0 - probability)
    return TrajectoryEnvelope(
        timestamps=times,
        lower=_weighted_quantile(samples, normalized, tail),
        median=_weighted_quantile(samples, normalized, 0.5),
        upper=_weighted_quantile(samples, normalized, 1.0 - tail),
        credible_probability=probability,
    )


@dataclass(frozen=True)
class TrajectoryEnvelopeValidation:
    envelope: TrajectoryEnvelope
    element_coverage_fraction: float
    time_coverage_fraction: float
    minimum_coverage_fraction: float
    evaluated_time_count: int
    passed: bool

    def __post_init__(self) -> None:
        for name in (
            "element_coverage_fraction",
            "time_coverage_fraction",
            "minimum_coverage_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError("{} must lie in [0, 1]".format(name))
            object.__setattr__(self, name, value)
        if int(self.evaluated_time_count) < 1:
            raise ValueError("at least one time must be evaluated")
        if type(self.passed) is not bool:
            raise TypeError("passed must be a built-in bool")


def validate_trajectory_envelope(
    observed_trajectory: np.ndarray,
    envelope: TrajectoryEnvelope,
    minimum_coverage_fraction: float = 0.95,
    score_mask: Optional[Sequence[bool]] = None,
) -> TrajectoryEnvelopeValidation:
    """Check held-out observations against a predictive trajectory envelope."""

    if not isinstance(envelope, TrajectoryEnvelope):
        raise TypeError("envelope must be TrajectoryEnvelope")
    observed = np.asarray(observed_trajectory, dtype=float)
    if observed.shape != envelope.lower.shape or not np.all(
        np.isfinite(observed)
    ):
        raise ValueError("observed trajectory must match the envelope")
    threshold = float(minimum_coverage_fraction)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("minimum_coverage_fraction must lie in [0, 1]")
    mask = (
        np.ones(envelope.timestamps.shape, dtype=bool)
        if score_mask is None
        else np.asarray(score_mask, dtype=bool).reshape(-1)
    )
    if mask.shape != envelope.timestamps.shape or not np.any(mask):
        raise ValueError("score_mask must select aligned trajectory times")
    inside = (observed >= envelope.lower) & (observed <= envelope.upper)
    selected = inside[mask]
    element_coverage = float(np.mean(selected))
    per_time = (
        selected
        if selected.ndim == 1
        else np.all(selected.reshape(selected.shape[0], -1), axis=1)
    )
    time_coverage = float(np.mean(per_time))
    return TrajectoryEnvelopeValidation(
        envelope=envelope,
        element_coverage_fraction=element_coverage,
        time_coverage_fraction=time_coverage,
        minimum_coverage_fraction=threshold,
        evaluated_time_count=int(np.count_nonzero(mask)),
        passed=bool(element_coverage >= threshold),
    )


def posterior_failure_probability(
    predicted_failures: Sequence[Optional[FailureEvent]],
    weights: Optional[Sequence[float]] = None,
) -> float:
    events = tuple(predicted_failures)
    if not events:
        raise ValueError("predicted_failures must not be empty")
    if any(
        item is not None and not isinstance(item, FailureEvent)
        for item in events
    ):
        raise TypeError("predicted failures must be FailureEvent or None")
    normalized = _normalize_weights(weights, len(events))
    return float(
        np.sum(
            normalized[
                np.asarray([item is not None for item in events], dtype=bool)
            ]
        )
    )


@dataclass(frozen=True)
class HeldOutFailureValidation:
    observed_failure: FailureEvent
    predicted_failure_probability: float
    matching_type_probability: float
    conditional_matching_type_probability: float
    predicted_time_lower: Optional[float]
    predicted_time_upper: Optional[float]
    observed_time_covered: bool
    credible_probability: float
    passed: bool
    reasons: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.observed_failure, FailureEvent):
            raise TypeError("observed_failure must be FailureEvent")
        for name in (
            "predicted_failure_probability",
            "matching_type_probability",
            "conditional_matching_type_probability",
            "credible_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError("{} must lie in [0, 1]".format(name))
            object.__setattr__(self, name, value)
        if type(self.observed_time_covered) is not bool or type(
            self.passed
        ) is not bool:
            raise TypeError("validation gates must be built-in bool values")
        object.__setattr__(
            self, "reasons", tuple(str(item) for item in self.reasons)
        )


def validate_held_out_failure(
    observed_failure: FailureEvent,
    predicted_failures: Sequence[Optional[FailureEvent]],
    weights: Optional[Sequence[float]] = None,
    credible_probability: float = 0.95,
    minimum_failure_probability: float = 0.5,
    minimum_matching_type_probability: float = 0.5,
) -> HeldOutFailureValidation:
    """Validate occurrence, type, and time on an inference-held-out failure."""

    if not isinstance(observed_failure, FailureEvent):
        raise TypeError("observed_failure must be FailureEvent")
    events = tuple(predicted_failures)
    if not events:
        raise ValueError("predicted_failures must not be empty")
    if any(
        item is not None and not isinstance(item, FailureEvent)
        for item in events
    ):
        raise TypeError("predicted failures must be FailureEvent or None")
    normalized = _normalize_weights(weights, len(events))
    probability = float(credible_probability)
    failure_threshold = float(minimum_failure_probability)
    type_threshold = float(minimum_matching_type_probability)
    if (
        not 0.0 < probability < 1.0
        or not 0.0 <= failure_threshold <= 1.0
        or not 0.0 <= type_threshold <= 1.0
    ):
        raise ValueError("failure validation probabilities are invalid")
    failure_mask = np.asarray(
        [item is not None for item in events], dtype=bool
    )
    type_mask = np.asarray(
        [
            item is not None
            and item.failure_type == observed_failure.failure_type
            for item in events
        ],
        dtype=bool,
    )
    failure_probability = float(np.sum(normalized[failure_mask]))
    matching_probability = float(np.sum(normalized[type_mask]))
    conditional_type = (
        0.0
        if failure_probability <= 0.0
        else matching_probability / failure_probability
    )
    if np.any(type_mask):
        selected_weights = normalized[type_mask]
        selected_weights = selected_weights / np.sum(selected_weights)
        selected_times = np.asarray(
            [item.stamp for item, keep in zip(events, type_mask) if keep],
            dtype=float,
        )
        tail = 0.5 * (1.0 - probability)
        lower = float(
            _weighted_quantile(selected_times, selected_weights, tail)
        )
        upper = float(
            _weighted_quantile(selected_times, selected_weights, 1.0 - tail)
        )
        time_covered = bool(
            lower <= observed_failure.stamp <= upper
        )
    else:
        lower = upper = None
        time_covered = False
    reasons = []
    if failure_probability < failure_threshold:
        reasons.append("failure_occurrence_probability")
    if matching_probability < type_threshold:
        reasons.append("failure_type_probability")
    if not time_covered:
        reasons.append("failure_time_coverage")
    return HeldOutFailureValidation(
        observed_failure=observed_failure,
        predicted_failure_probability=failure_probability,
        matching_type_probability=matching_probability,
        conditional_matching_type_probability=float(conditional_type),
        predicted_time_lower=lower,
        predicted_time_upper=upper,
        observed_time_covered=time_covered,
        credible_probability=probability,
        passed=not reasons,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class PosteriorPredictiveValidation:
    trajectory: TrajectoryEnvelopeValidation
    failure: HeldOutFailureValidation
    passed: bool
    dataset_identity: ValidationDatasetIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, TrajectoryEnvelopeValidation):
            raise TypeError("trajectory must be TrajectoryEnvelopeValidation")
        if not isinstance(self.failure, HeldOutFailureValidation):
            raise TypeError("failure must be HeldOutFailureValidation")
        if type(self.passed) is not bool:
            raise TypeError("passed must be a built-in bool")
        if self.passed != (self.trajectory.passed and self.failure.passed):
            raise ValueError("combined validation gate is inconsistent")
        if not isinstance(
            self.dataset_identity, ValidationDatasetIdentity
        ):
            raise TypeError(
                "dataset_identity must be ValidationDatasetIdentity"
            )
        if (
            self.dataset_identity.schema
            != "grape_held_out_failure_dataset/v1"
            or self.dataset_identity.validation_result_sha256
            != stable_hash(
                _failure_validation_result_payload(
                    self.trajectory,
                    self.failure,
                    self.passed,
                )
            )
        ):
            raise ValueError(
                "failure dataset/result identity mismatch"
            )

    @property
    def dataset_sha256(self) -> str:
        return self.dataset_identity.content_sha256


def validate_posterior_predictive(
    *,
    timestamps: Sequence[float],
    observed_trajectory: np.ndarray,
    predictive_trajectories: np.ndarray,
    observed_failure: FailureEvent,
    predicted_failures: Sequence[Optional[FailureEvent]],
    weights: Optional[Sequence[float]] = None,
    credible_probability: float = 0.95,
    minimum_coverage_fraction: float = 0.95,
    score_mask: Optional[Sequence[bool]] = None,
    minimum_failure_probability: float = 0.5,
    minimum_matching_type_probability: float = 0.5,
    dataset_provenance_sha256: Optional[str] = None,
) -> PosteriorPredictiveValidation:
    """Validate trajectory and event claims on a held-out failure episode.

    Trajectory coverage is evaluated only through the observed failure sample.
    A caller-provided ``score_mask`` is treated as a base mask and intersected
    with that causal horizon; post-failure motion cannot rescue or invalidate
    the trajectory component.
    """

    samples = np.asarray(predictive_trajectories)
    observed = np.asarray(observed_trajectory)
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
    censoring = censor_after_failure(
        timestamps,
        observed_failure,
        base_mask=score_mask,
        include_failure_sample=True,
    )
    if (
        samples.ndim < 2
        or len(events) != samples.shape[0]
        or samples.shape[1] != censoring.timestamps.size
        or observed.ndim < 1
        or observed.shape[0] != censoring.timestamps.size
    ):
        raise ValueError(
            "failure trajectories must align by particle and timestamp"
        )
    selected = censoring.score_mask
    envelope = trajectory_envelope(
        censoring.timestamps[selected],
        samples[:, selected],
        weights,
        credible_probability,
    )
    trajectory = validate_trajectory_envelope(
        observed[selected],
        envelope,
        minimum_coverage_fraction,
    )
    failure = validate_held_out_failure(
        observed_failure,
        events,
        weights,
        credible_probability,
        minimum_failure_probability,
        minimum_matching_type_probability,
    )
    passed = bool(trajectory.passed and failure.passed)
    result_payload = _failure_validation_result_payload(
        trajectory, failure, passed
    )
    return PosteriorPredictiveValidation(
        trajectory=trajectory,
        failure=failure,
        passed=passed,
        dataset_identity=ValidationDatasetIdentity._from_evaluation(
            "grape_held_out_failure_dataset/v1",
            {
                "dataset_provenance_sha256": provenance_hash,
                "timestamps": np.asarray(timestamps, dtype=float),
                "observed_trajectory": observed,
                "observed_failure": {
                    "failure_type": observed_failure.failure_type,
                    "stamp": observed_failure.stamp,
                    "detector_id": observed_failure.detector_id,
                    "metadata": dict(observed_failure.metadata),
                },
                "score_mask": censoring.score_mask,
            },
            result_payload,
        ),
    )


__all__ = [
    "HeldOutFailureValidation",
    "PosteriorPredictiveValidation",
    "TrajectoryEnvelope",
    "TrajectoryEnvelopeValidation",
    "ValidationDatasetIdentity",
    "posterior_failure_probability",
    "trajectory_envelope",
    "validate_held_out_failure",
    "validate_posterior_predictive",
    "validate_trajectory_envelope",
]
