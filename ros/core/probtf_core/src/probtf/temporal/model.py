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


def _nonnegative_extended(value, name):
    result = float(value)
    if math.isnan(result) or result < 0.0:
        raise ValueError("{} must be non-negative and not NaN.".format(name))
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
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("process_noise_spectral_density must be symmetric.")
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
        if type(self.allow_degraded) is not bool:
            raise TypeError("allow_degraded must be a built-in bool.")
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
        initial = _nonnegative_extended(
            self.initial_uncertainty_trace,
            "initial_uncertainty_trace",
        )
        result = _nonnegative_extended(
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
        if (
            math.isinf(self.initial_uncertainty_trace)
            and math.isinf(self.result_uncertainty_trace)
            and self.evaluation_kind
            in (
                TemporalEvaluationKind.STATIC,
                TemporalEvaluationKind.SAMPLE_SELECTION,
            )
        ):
            return 0.0
        return self.result_uncertainty_trace - self.initial_uncertainty_trace


class TemporalModel(ABC):
    """Base class for explicitly registered edge/authority temporal models."""

    def __setattr__(self, name, value):
        if getattr(self, "_configuration_frozen", False):
            raise AttributeError(
                "TemporalModel configuration is immutable after construction."
            )
        object.__setattr__(self, name, value)

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
        object.__setattr__(self, "_configuration_frozen", True)

    @property
    def supports_interpolation(self):
        return True

    @property
    def supports_prediction(self):
        return True

    def validate_distribution(self, record):
        if not isinstance(record, TransformDistributionStamped):
            return False
        normalized = record.distribution.normalize_weights()
        if not normalized.components:
            return False
        for weighted in normalized.components:
            component = weighted.component
            if component.orientation.kind not in self.supported_orientation_kinds:
                return False
            if self.backend is TemporalUncertaintyBackend.MOMENT:
                # FINITE_BINGHAM includes axial/plateau laws for which a local
                # Gaussian moment is not defined.  Model support is a
                # capability test, not merely an enum test.
                try:
                    from probtf.temporal.backends import component_pose_covariance

                    component_pose_covariance(component)
                except (ValueError, np.linalg.LinAlgError):
                    return False
        return True

    def _validate_request_anchors(self, records, request):
        if not isinstance(request, TemporalEvaluationRequest):
            raise TypeError("request must be TemporalEvaluationRequest.")
        if request.model_selector != self.model_id:
            raise ValueError("MODEL_SELECTOR_MISMATCH: request selects another model.")
        records = tuple(records)
        if len(records) != len(request.anchors):
            raise ValueError("ANCHOR_MISMATCH: request anchors do not match model inputs.")
        from probtf.temporal.provenance import source_record_dependency_id

        if any(
            source_record_dependency_id(record)
            != source_record_dependency_id(request_record)
            for record, request_record in zip(records, request.anchors)
        ):
            raise ValueError("ANCHOR_MISMATCH: request anchors do not match model inputs.")
        if records:
            edge = (
                records[0].parent_frame_id,
                records[0].child_frame_id,
                records[0].edge_id,
                records[0].authority,
            )
            if any(
                (
                    record.parent_frame_id,
                    record.child_frame_id,
                    record.edge_id,
                    record.authority,
                )
                != edge
                for record in records[1:]
            ):
                raise ValueError(
                    "ANCHOR_INCONSISTENT: anchors must share edge, frames, and authority."
                )
        return records

    def validate_interpolation_request(self, left, right, request):
        records = self._validate_request_anchors((left, right), request)
        if request.policy is not TemporalPolicy.INTERPOLATE_WITH_MODEL:
            raise ValueError("POLICY_MISMATCH: interpolation requires INTERPOLATE_WITH_MODEL.")
        if not self.supports_interpolation:
            raise ValueError("MODEL_SUPPORT_EXCEEDED: model does not support interpolation.")
        if len(records) != 2 or left.stamp >= right.stamp:
            raise ValueError("ANCHOR_INCONSISTENT: interpolation stamps must increase.")
        if not left.stamp < request.requested_stamp < right.stamp:
            raise ValueError(
                "MODEL_SUPPORT_EXCEEDED: interpolation requires strict bracketing; "
                "endpoints are sample-selection queries."
            )
        if request.query_mode is not TemporalQueryMode.OFFLINE_SMOOTHING:
            raise ValueError(
                "NON_CAUSAL_INPUT_REJECTED: interpolation requires offline_smoothing."
            )
        if (
            request.max_age is not None
            and request.requested_stamp - left.stamp > request.max_age
        ):
            raise ValueError("TEMPORAL_STALE: interpolation anchor exceeds max_age.")
        if not all(self.validate_distribution(record) for record in records):
            raise ValueError(
                "MODEL_SUPPORT_EXCEEDED: an interpolation distribution is unsupported."
            )

    def validate_prediction_request(self, history, request):
        records = self._validate_request_anchors(history, request)
        if request.policy is not TemporalPolicy.PREDICT_WITH_MODEL:
            raise ValueError("POLICY_MISMATCH: prediction requires PREDICT_WITH_MODEL.")
        if not self.supports_prediction:
            raise ValueError("MODEL_SUPPORT_EXCEEDED: model does not support prediction.")
        if len(records) < self.minimum_history:
            raise ValueError("INSUFFICIENT_HISTORY: prediction history is too short.")
        if any(
            left.stamp >= right.stamp
            for left, right in zip(records[:-1], records[1:])
        ):
            raise ValueError("ANCHOR_INCONSISTENT: prediction stamps must increase.")
        if any(record.stamp > request.requested_stamp for record in records):
            raise ValueError("NON_CAUSAL_INPUT_REJECTED: prediction history contains future data.")
        if request.max_prediction_horizon is None:
            raise ValueError(
                "PREDICTION_HORIZON_REQUIRED: max_prediction_horizon is mandatory."
            )
        if request.max_age is None:
            raise ValueError("MAX_AGE_REQUIRED: max_age is mandatory.")
        horizon = request.requested_stamp - records[-1].stamp
        if horizon <= 0.0:
            raise ValueError(
                "MODEL_SUPPORT_EXCEEDED: a prediction requires a future stamp; "
                "the anchor stamp is a sample-selection query."
            )
        if horizon > request.max_prediction_horizon:
            raise ValueError(
                "PREDICTION_HORIZON_EXCEEDED: request exceeds max_prediction_horizon."
            )
        if horizon > request.max_age:
            raise ValueError("TEMPORAL_STALE: prediction anchor exceeds max_age.")
        if horizon > self.maximum_horizon:
            raise ValueError("MODEL_SUPPORT_EXCEEDED: prediction exceeds model horizon.")
        if not all(
            self.validate_distribution(record)
            for record in records[-self.minimum_history :]
        ):
            raise ValueError(
                "MODEL_SUPPORT_EXCEEDED: a prediction distribution is unsupported."
            )

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


@dataclass(frozen=True)
class DiscreteProcessNoiseAdaptation:
    """Traceable compatibility result for converting per-step ``Qd``."""

    spectral_density: np.ndarray
    sample_period: float
    diagnostic: TemporalDiagnosticCode = (
        TemporalDiagnosticCode.DISCRETE_PROCESS_NOISE_ADAPTED
    )
    detail: str = (
        "Converted legacy per-step covariance Qd to canonical continuous-time "
        "spectral density Qc=Qd/dt."
    )

    def __post_init__(self):
        covariance = _spectral_density(self.spectral_density)
        period = float(self.sample_period)
        if not math.isfinite(period) or period <= 0.0:
            raise ValueError("sample_period must be finite and positive.")
        if self.diagnostic is not TemporalDiagnosticCode.DISCRETE_PROCESS_NOISE_ADAPTED:
            raise ValueError("diagnostic must identify discrete process-noise adaptation.")
        object.__setattr__(self, "spectral_density", covariance)
        object.__setattr__(self, "sample_period", period)
        object.__setattr__(self, "detail", str(self.detail))


def adapt_discrete_process_noise(discrete_covariance, sample_period):
    """Return ``Qc`` together with mandatory compatibility provenance."""

    period = float(sample_period)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("sample_period must be finite and positive.")
    covariance = _spectral_density(discrete_covariance)
    return DiscreteProcessNoiseAdaptation(covariance / period, period)


def discrete_process_noise_to_spectral_density(discrete_covariance, sample_period):
    """Explicit compatibility adapter from per-step ``Qd`` to canonical ``Qc``."""

    return adapt_discrete_process_noise(
        discrete_covariance,
        sample_period,
    ).spectral_density


__all__ = [
    "DiscreteProcessNoiseAdaptation",
    "ResolvedEdgeRecord",
    "TemporalDiagnosticCode",
    "TemporalEvaluationKind",
    "TemporalEvaluationRequest",
    "TemporalEvaluationResult",
    "TemporalModel",
    "TemporalQueryMode",
    "TemporalUncertaintyBackend",
    "adapt_discrete_process_noise",
    "discrete_process_noise_to_spectral_density",
]
