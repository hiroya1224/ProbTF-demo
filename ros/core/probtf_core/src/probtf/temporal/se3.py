"""Reference SE(3) temporal models with explicit uncertainty backends."""

import numpy as np

from probtf.geometry import body_twist_between, se3_exp
from probtf.temporal.backends import (
    moment_interpolate,
    moment_predict,
    record_representative,
    record_uncertainty_trace,
    sample_interpolate,
    sample_predict,
)
from probtf.temporal.model import (
    TemporalDiagnosticCode,
    TemporalEvaluationKind,
    TemporalEvaluationResult,
    TemporalModel,
    TemporalUncertaintyBackend,
)
from probtf.temporal.provenance import (
    source_record_dependency_ids,
    temporal_detail,
)


def _positive_count(value):
    count = int(value)
    if count < 1 or count != value:
        raise ValueError("sample_count must be a positive integer.")
    return count


def _acceleration(value):
    array = np.asarray(value, dtype=float)
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise ValueError("body_acceleration must be a finite vector with shape (6,).")
    array = array.copy()
    array.setflags(write=False)
    return array


def _nonempty(value, name):
    result = str(value).strip()
    if not result:
        raise ValueError("{} must not be empty.".format(name))
    return result


class ConstantBodyTwistModel(TemporalModel):
    """SE(3) constant-body-twist with continuous-time spectral density ``Qc``."""

    CONVENTION = "T(t+h)=T(t)*Exp(h*xi_body), xi=[rho,phi]"

    def __init__(
        self,
        process_noise_spectral_density,
        maximum_horizon,
        *,
        model_id="se3_constant_body_twist",
        version="1",
        backend=TemporalUncertaintyBackend.MOMENT,
        sample_count=256,
    ):
        self.sample_count = _positive_count(sample_count)
        super().__init__(
            model_id=model_id,
            version=version,
            process_noise_spectral_density=process_noise_spectral_density,
            maximum_horizon=maximum_horizon,
            backend=backend,
            minimum_history=2,
            config={
                "convention": self.CONVENTION,
                "sample_count": self.sample_count,
            },
        )

    def _detail(
        self,
        records,
        request,
        kind,
        horizon,
        diagnostics,
    ):
        dependencies = source_record_dependency_ids(records)
        detail = temporal_detail(
            model_id=self.model_id,
            model_version=self.version,
            config_fingerprint=self.config_fingerprint,
            source_stamps=tuple(record.stamp for record in records),
            dependency_ids=dependencies,
            authority=records[-1].authority,
            backend=self.backend,
            evaluation_kind=kind,
            horizon=horizon,
            random_seed=request.random_seed,
            random_stream=request.random_stream,
            diagnostics=diagnostics,
        )
        return dependencies, detail

    def interpolate(self, left, right, request):
        if left.stamp >= right.stamp:
            raise ValueError("Interpolation anchors must have strictly increasing stamps.")
        if left.authority != right.authority:
            raise ValueError("Interpolation anchors must have one authority.")
        diagnostics = [
            TemporalDiagnosticCode.MODEL_INTERPOLATION,
            TemporalDiagnosticCode.ENDPOINT_CONDITIONED,
        ]
        warnings = []
        if self.backend is TemporalUncertaintyBackend.MOMENT and (
            left.distribution.deterministic_transform() is None
            or right.distribution.deterministic_transform() is None
        ):
            diagnostics.append(TemporalDiagnosticCode.DEPENDENCE_APPROXIMATED)
            warnings.append(
                "Cross-time covariance between stochastic endpoints is unavailable; "
                "the tangent moment backend uses an explicitly diagnosed approximation."
            )
        dependencies, detail = self._detail(
            (left, right),
            request,
            TemporalEvaluationKind.MODEL_INTERPOLATION,
            0.0,
            tuple(diagnostics),
        )
        if self.backend is TemporalUncertaintyBackend.MOMENT:
            record = moment_interpolate(
                left,
                right,
                request.requested_stamp,
                detail,
            )
        else:
            record = sample_interpolate(
                left,
                right,
                request.requested_stamp,
                self.sample_count,
                request.random_seed,
                request.random_stream,
                detail,
            )
        alpha = (request.requested_stamp - left.stamp) / (right.stamp - left.stamp)
        initial_trace = (
            (1.0 - alpha) * record_uncertainty_trace(left)
            + alpha * record_uncertainty_trace(right)
        )
        return TemporalEvaluationResult(
            record=record,
            requested_stamp=request.requested_stamp,
            evaluated_stamp=request.requested_stamp,
            source_stamps=(left.stamp, right.stamp),
            model_id=self.model_id,
            model_version=self.version,
            config_fingerprint=self.config_fingerprint,
            evaluation_kind=TemporalEvaluationKind.MODEL_INTERPOLATION,
            horizon=0.0,
            dependency_ids=dependencies,
            backend=self.backend,
            approximation=record.approximation,
            diagnostics=tuple(diagnostics),
            warnings=tuple(warnings),
            random_seed=request.random_seed,
            random_stream=request.random_stream,
            initial_uncertainty_trace=initial_trace,
            result_uncertainty_trace=record_uncertainty_trace(record),
        )

    def _body_acceleration(self, history, request):
        del history, request
        return np.zeros(6, dtype=float)

    def predict(self, history_at_or_before_t, request):
        history = tuple(history_at_or_before_t)
        if len(history) < self.minimum_history:
            raise ValueError("Constant-body-twist prediction needs at least two records.")
        previous, anchor = history[-2:]
        if previous.stamp >= anchor.stamp:
            raise ValueError("Prediction history must have strictly increasing stamps.")
        horizon = request.requested_stamp - anchor.stamp
        if horizon < 0.0:
            raise ValueError("Prediction anchor must not be later than the request.")
        acceleration = self._body_acceleration(history, request)
        diagnostics = (TemporalDiagnosticCode.MODEL_PREDICTION,)
        dependencies, detail = self._detail(
            (previous, anchor),
            request,
            TemporalEvaluationKind.MODEL_PREDICTION,
            horizon,
            diagnostics,
        )
        if self.backend is TemporalUncertaintyBackend.MOMENT:
            twist = body_twist_between(
                record_representative(previous),
                record_representative(anchor),
                anchor.stamp - previous.stamp,
            )
            increment = twist * horizon + 0.5 * acceleration * horizon ** 2
            record = moment_predict(
                (previous, anchor),
                request.requested_stamp,
                increment,
                self.process_noise_spectral_density * horizon,
                detail,
            )
        else:
            record = sample_predict(
                (previous, anchor),
                request.requested_stamp,
                acceleration,
                self.process_noise_spectral_density,
                self.sample_count,
                request.random_seed,
                request.random_stream,
                detail,
            )
        initial_trace = record_uncertainty_trace(anchor)
        result_trace = record_uncertainty_trace(record)
        return TemporalEvaluationResult(
            record=record,
            requested_stamp=request.requested_stamp,
            evaluated_stamp=request.requested_stamp,
            source_stamps=(previous.stamp, anchor.stamp),
            model_id=self.model_id,
            model_version=self.version,
            config_fingerprint=self.config_fingerprint,
            evaluation_kind=TemporalEvaluationKind.MODEL_PREDICTION,
            horizon=horizon,
            dependency_ids=dependencies,
            backend=self.backend,
            approximation=record.approximation,
            diagnostics=diagnostics,
            random_seed=request.random_seed,
            random_stream=request.random_stream,
            initial_uncertainty_trace=initial_trace,
            result_uncertainty_trace=result_trace,
        )


class ConstantBodyAccelerationModel(ConstantBodyTwistModel):
    """Constant body acceleration with mandatory source/frame metadata."""

    def __init__(
        self,
        body_acceleration,
        acceleration_source,
        acceleration_frame,
        process_noise_spectral_density,
        maximum_horizon,
        *,
        model_id="se3_constant_body_acceleration",
        version="1",
        backend=TemporalUncertaintyBackend.MOMENT,
        sample_count=256,
    ):
        self.body_acceleration = _acceleration(body_acceleration)
        self.acceleration_source = _nonempty(
            acceleration_source,
            "acceleration_source",
        )
        self.acceleration_frame = _nonempty(
            acceleration_frame,
            "acceleration_frame",
        )
        self.sample_count = _positive_count(sample_count)
        TemporalModel.__init__(
            self,
            model_id=model_id,
            version=version,
            process_noise_spectral_density=process_noise_spectral_density,
            maximum_horizon=maximum_horizon,
            backend=backend,
            minimum_history=2,
            config={
                "convention": self.CONVENTION,
                "sample_count": self.sample_count,
                "body_acceleration": self.body_acceleration.tolist(),
                "acceleration_source": self.acceleration_source,
                "acceleration_frame": self.acceleration_frame,
            },
        )

    def _body_acceleration(self, history, request):
        del history, request
        return self.body_acceleration


class EndpointConditionedSampleInterpolationModel(ConstantBodyTwistModel):
    """Non-Gaussian endpoint-conditioned interpolation reference backend."""

    def __init__(
        self,
        *,
        model_id="se3_endpoint_conditioned_sample_interpolation",
        version="1",
        sample_count=512,
    ):
        super().__init__(
            np.zeros((6, 6)),
            maximum_horizon=0.0,
            model_id=model_id,
            version=version,
            backend=TemporalUncertaintyBackend.SAMPLE,
            sample_count=sample_count,
        )

    @property
    def supports_prediction(self):
        return False

    def predict(self, history_at_or_before_t, request):
        del history_at_or_before_t, request
        raise NotImplementedError(
            "EndpointConditionedSampleInterpolationModel does not predict."
        )


__all__ = [
    "ConstantBodyAccelerationModel",
    "ConstantBodyTwistModel",
    "EndpointConditionedSampleInterpolationModel",
]
