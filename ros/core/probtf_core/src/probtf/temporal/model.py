"""Public contracts for model-based temporal transform evaluation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import hashlib
import json
import math

import numpy as np

from probtf.distributions import OrientationKind, TransformDistributionStamped
from probtf.provenance import ApproximationInfo
from probtf.temporal.policy import TemporalPolicy


class TemporalQueryMode(Enum):
    """Whether a query may condition on observations after its requested time."""

    ONLINE = "online"
    OFFLINE_SMOOTHING = "offline_smoothing"


class TemporalEvaluationKind(Enum):
    SAMPLE_SELECTION = "sample_selection"
    STATIC = "static"
    MODEL_INTERPOLATION = "model_interpolation"
    MODEL_PREDICTION = "model_prediction"


class TemporalUncertaintyBackend(Enum):
    MOMENT = "moment"
    SAMPLE = "sample"


class TemporalDiagnosticCode(Enum):
    EXACT_SAMPLE = "exact_sample"
    STATIC_EDGE = "static_edge"
    MODEL_INTERPOLATION = "model_interpolation"
    MODEL_PREDICTION = "model_prediction"
    ENDPOINT_CONDITIONED = "endpoint_conditioned"
    DEPENDENCE_APPROXIMATED = "dependence_approximated"
    UNCERTAINTY_LIMIT_EXCEEDED = "uncertainty_limit_exceeded"
    DISCRETE_PROCESS_NOISE_ADAPTED = "discrete_process_noise_adapted"


def _finite_nonnegative(value, name, allow_none=False):
    if value is None and allow_none:
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        suffix = " or None" if allow_none else ""
        raise ValueError("{} must be finite and non-negative{}.".format(name, suffix))
    return result


def _identifier(value, name, allow_empty=False):
    result = str(value).strip()
    if not result and not allow_empty:
        raise ValueError("{} must not be empty.".format(name))
    return result


def _unique_strings(values, name):
    result = tuple(_identifier(value, name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError("{} must contain unique values.".format(name))
    return result


def _spectral_density(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
        raise ValueError("process_noise_spectral_density must be a finite 6x6 matrix.")
    matrix = 0.5 * (matrix + matrix.T)
    if float(np.linalg.eigvalsh(matrix)[0]) < -1.0e-10:
        raise ValueError("process_noise_spectral_density must be positive semidefinite.")
    matrix = matrix.copy()
    matrix.setflags(write=False)
    return matrix


@dataclass(frozen=True)
class TemporalEvaluationRequest:
    """One immutable temporal-model invocation.

    Process noise is always continuous-time spectral density in the model.
    ``max_prediction_horizon`` and ``max_age`` are intentionally optional in
    the data type because interpolation does not need them; the buffer rejects
    prediction requests that omit either safety limit.
    """

    requested_stamp: float
    policy: TemporalPolicy
    anchors: Tuple[TransformDistributionStamped, ...]
    model_selector: str
    max_prediction_horizon: Optional[float] = None
    max_age: Optional[float] = None
    random_seed: Optional[int] = None
    random_stream: str = ""
    query_mode: TemporalQueryMode = TemporalQueryMode.ONLINE
    max_uncertainty_trace: Optional[float] = None
    allow_degraded: bool = False

    def __post_init__(self):
        requested = _finite_nonnegative(self.requested_stamp, "requested_stamp")
        if not isinstance(self.policy, TemporalPolicy):
            raise TypeError("policy must be TemporalPolicy.")
        if self.policy not in (
            TemporalPolicy.INTERPOLATE_WITH_MODEL,
            TemporalPolicy.PREDICT_WITH_MODEL,
        ):
            raise ValueError("A temporal model request requires a model-based policy.")
        anchors = tuple(self.anchors)
        if any(not isinstance(item, TransformDistributionStamped) for item in anchors):
            raise TypeError("anchors must contain TransformDistributionStamped records.")
        if not isinstance(self.query_mode, TemporalQueryMode):
            raise TypeError("query_mode must be TemporalQueryMode.")
        seed = self.random_seed
        if seed is not None:
            if isinstance(seed, (bool, np.bool_)):
                raise ValueError("random_seed must be a non-negative integer or None.")
            seed = int(seed)
            if seed < 0 or seed != self.random_seed:
                raise ValueError("random_seed must be a non-negative integer or None.")
        object.__setattr__(self, "requested_stamp", requested)
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "model_selector", _identifier(self.model_selector, "model_selector"))
        object.__setattr__(
            self,
            "max_prediction_horizon",
            _finite_nonnegative(
                self.max_prediction_horizon,
                "max_prediction_horizon",
                allow_none=True,
            ),
        )
        object.__setattr__(
            self,
            "max_age",
            _finite_nonnegative(self.max_age, "max_age", allow_none=True),
        )
        object.__setattr__(
            self,
            "max_uncertainty_trace",
            _finite_nonnegative(
                self.max_uncertainty_trace,
                "max_uncertainty_trace",
                allow_none=True,
            ),
        )
        object.__setattr__(self, "random_seed", seed)
        object.__setattr__(
            self,
            "random_stream",
            _identifier(self.random_stream, "random_stream", allow_empty=True),
        )
        object.__setattr__(self, "allow_degraded", bool(self.allow_degraded))


@dataclass(frozen=True)
class TemporalEvaluationResult:
    """A model output together with replayable provenance and diagnostics."""

    record: TransformDistributionStamped
    requested_stamp: float
    evaluated_stamp: float
    source_stamps: Tuple[float, ...]
    model_id: str
    model_version: str
    config_fingerprint: str
    evaluation_kind: TemporalEvaluationKind
    horizon: float
    dependency_ids: Tuple[str, ...]
    backend: TemporalUncertaintyBackend
    approximation: ApproximationInfo = field(default_factory=ApproximationInfo)
    diagnostics: Tuple[TemporalDiagnosticCode, ...] = ()
    warnings: Tuple[str, ...] = ()
    random_seed: Optional[int] = None
    random_stream: str = ""
    initial_uncertainty_trace: float = 0.0
    result_uncertainty_trace: float = 0.0

    def __post_init__(self):
        if not isinstance(self.record, TransformDistributionStamped):
            raise TypeError("record must be TransformDistributionStamped.")
        if not isinstance(self.evaluation_kind, TemporalEvaluationKind):
            raise TypeError("evaluation_kind must be TemporalEvaluationKind.")
        if not isinstance(self.backend, TemporalUncertaintyBackend):
            raise TypeError("backend must be TemporalUncertaintyBackend.")
        if not isinstance(self.approximation, ApproximationInfo):
            raise TypeError("approximation must be ApproximationInfo.")
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(item, TemporalDiagnosticCode) for item in diagnostics):
            raise TypeError("diagnostics must contain TemporalDiagnosticCode values.")
        source_stamps = tuple(
            _finite_nonnegative(value, "source_stamps") for value in self.source_stamps
        )
        initial = _finite_nonnegative(
            self.initial_uncertainty_trace,
            "initial_uncertainty_trace",
        )
        result = _finite_nonnegative(
            self.result_uncertainty_trace,
            "result_uncertainty_trace",
        )
        object.__setattr__(
            self,
            "requested_stamp",
            _finite_nonnegative(self.requested_stamp, "requested_stamp"),
        )
        object.__setattr__(
            self,
            "evaluated_stamp",
            _finite_nonnegative(self.evaluated_stamp, "evaluated_stamp"),
        )
        object.__setattr__(self, "source_stamps", source_stamps)
        object.__setattr__(self, "model_id", _identifier(self.model_id, "model_id"))
        object.__setattr__(
            self,
            "model_version",
            _identifier(self.model_version, "model_version"),
        )
        object.__setattr__(
            self,
            "config_fingerprint",
            _identifier(self.config_fingerprint, "config_fingerprint"),
        )
        object.__setattr__(self, "horizon", _finite_nonnegative(self.horizon, "horizon"))
        object.__setattr__(
            self,
            "dependency_ids",
            _unique_strings(self.dependency_ids, "dependency_ids"),
        )
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "warnings", tuple(str(value) for value in self.warnings))
        object.__setattr__(
            self,
            "random_stream",
            _identifier(self.random_stream, "random_stream", allow_empty=True),
        )
        object.__setattr__(self, "initial_uncertainty_trace", initial)
        object.__setattr__(self, "result_uncertainty_trace", result)
        if not np.isclose(self.record.stamp, self.evaluated_stamp, rtol=0.0, atol=1.0e-12):
            raise ValueError("record.stamp must equal evaluated_stamp.")

    @property
    def uncertainty_increase(self):
        return self.result_uncertainty_trace - self.initial_uncertainty_trace


class TemporalModel(ABC):
    """Base class for explicitly registered edge/authority temporal models."""

    def __init__(
        self,
        model_id,
        version,
        process_noise_spectral_density,
        maximum_horizon,
        backend=TemporalUncertaintyBackend.MOMENT,
        supported_orientation_kinds=tuple(OrientationKind),
        minimum_history=2,
        config=None,
    ):
        if not isinstance(backend, TemporalUncertaintyBackend):
            raise TypeError("backend must be TemporalUncertaintyBackend.")
        kinds = tuple(supported_orientation_kinds)
        if any(not isinstance(item, OrientationKind) for item in kinds):
            raise TypeError("supported_orientation_kinds must contain OrientationKind values.")
        history = int(minimum_history)
        if history < 1 or history != minimum_history:
            raise ValueError("minimum_history must be a positive integer.")
        self.model_id = _identifier(model_id, "model_id")
        self.version = _identifier(version, "version")
        self.process_noise_spectral_density = _spectral_density(
            process_noise_spectral_density
        )
        self.maximum_horizon = _finite_nonnegative(
            maximum_horizon,
            "maximum_horizon",
        )
        self.backend = backend
        self.supported_orientation_kinds = kinds
        self.minimum_history = history
        payload = {
            "model_id": self.model_id,
            "version": self.version,
            "backend": self.backend.value,
            "minimum_history": history,
            "maximum_horizon": self.maximum_horizon,
            "process_noise_spectral_density": self.process_noise_spectral_density.tolist(),
            "config": {} if config is None else config,
        }
        self.config_fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @property
    def supports_interpolation(self):
        return True

    @property
    def supports_prediction(self):
        return True

    def validate_distribution(self, record):
        for component in record.distribution.components:
            if component.orientation.kind not in self.supported_orientation_kinds:
                return False
        return True

    @abstractmethod
    def interpolate(self, left, right, request):
        """Condition on the two endpoint records and evaluate at the request."""

    @abstractmethod
    def predict(self, history_at_or_before_t, request):
        """Causally predict from history whose stamps do not exceed the query."""


@dataclass(frozen=True)
class ResolvedEdgeRecord:
    """Backward-compatible resolved record plus optional model diagnostics."""

    record: TransformDistributionStamped
    requested_stamp: float
    sample_stamp: float
    policy: TemporalPolicy
    diagnostic: str = ""
    evaluation: Optional[TemporalEvaluationResult] = None

    @property
    def time_offset(self):
        return self.sample_stamp - self.requested_stamp


def discrete_process_noise_to_spectral_density(discrete_covariance, sample_period):
    """Explicit compatibility adapter from per-step ``Qd`` to canonical ``Qc``."""

    period = float(sample_period)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("sample_period must be finite and positive.")
    covariance = _spectral_density(discrete_covariance)
    output = np.asarray(covariance / period, dtype=float)
    output.setflags(write=False)
    return output


__all__ = [
    "ResolvedEdgeRecord",
    "TemporalDiagnosticCode",
    "TemporalEvaluationKind",
    "TemporalEvaluationRequest",
    "TemporalEvaluationResult",
    "TemporalModel",
    "TemporalQueryMode",
    "TemporalUncertaintyBackend",
    "discrete_process_noise_to_spectral_density",
]
