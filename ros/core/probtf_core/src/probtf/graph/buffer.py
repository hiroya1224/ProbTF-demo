"""Timestamp-ordered histories for physical Prob-TF edges."""

from bisect import bisect_left, bisect_right
from dataclasses import replace
from threading import RLock

import numpy as np

from probtf.distributions import TransformDistributionStamped
from probtf.graph.status import GraphErrorCode, TemporalResolutionError
from probtf.temporal import (
    AuthorityConflictPolicy,
    ResolvedEdgeRecord,
    TemporalDiagnosticCode,
    TemporalEvaluationKind,
    TemporalEvaluationRequest,
    TemporalEvaluationResult,
    TemporalModel,
    TemporalPolicy,
    TemporalQueryMode,
    TemporalUncertaintyBackend,
    parse_temporal_detail,
    source_record_dependency_id,
    source_record_dependency_ids,
)
from probtf.temporal.backends import copy_record_at_stamp, record_uncertainty_trace
from probtf.temporal.provenance import make_transform_provenance, temporal_detail


def _same_component(left, right):
    return (
        left.component_id == right.component_id
        and left.raw_weight == right.raw_weight
        and left.orientation.kind is right.orientation.kind
        and left.orientation.inverse_concentration == right.orientation.inverse_concentration
        and np.array_equal(left.orientation.shape_matrix, right.orientation.shape_matrix)
        and np.array_equal(
            left.orientation.reference_quaternion_wxyz,
            right.orientation.reference_quaternion_wxyz,
        )
        and np.array_equal(
            left.translation.mean_at_reference,
            right.translation.mean_at_reference,
        )
        and np.array_equal(
            left.translation.residual_covariance,
            right.translation.residual_covariance,
        )
        and np.array_equal(
            left.translation.rotation_coupling,
            right.translation.rotation_coupling,
        )
        and left.provenance == right.provenance
        and left.approximation == right.approximation
    )


def _same_deterministic_transform(left, right):
    if left is None or right is None:
        return left is right
    return np.array_equal(left.translation, right.translation) and np.array_equal(
        left.rotation_wxyz,
        right.rotation_wxyz,
    )


def _same_static_payload(left, right):
    return (
        len(left.distribution.components) == len(right.distribution.components)
        and all(
            _same_component(left_component, right_component)
            for left_component, right_component in zip(
                left.distribution.components,
                right.distribution.components,
            )
        )
        and left.representative_kind is right.representative_kind
        and _same_deterministic_transform(left.representative, right.representative)
        and left.provenance == right.provenance
        and left.approximation == right.approximation
    )


class EdgeTimeBuffer:
    def __init__(self, max_records=None, conflict_policy=AuthorityConflictPolicy.REJECT):
        if max_records is not None and int(max_records) < 1:
            raise ValueError("max_records must be positive or None.")
        if not isinstance(conflict_policy, AuthorityConflictPolicy):
            raise TypeError("conflict_policy must be AuthorityConflictPolicy.")
        self.max_records = None if max_records is None else int(max_records)
        self.conflict_policy = conflict_policy
        self._records = []
        self._lock = RLock()
        self._edge_id = None
        self._endpoints = None
        self._static = None

    def __len__(self):
        with self._lock:
            return len(self._records)

    @property
    def records(self):
        with self._lock:
            return tuple(self._records)

    @property
    def earliest_stamp(self):
        with self._lock:
            if not self._records:
                raise TemporalResolutionError(GraphErrorCode.TEMPORAL_OUT_OF_RANGE, "Edge buffer is empty.")
            return self._records[0].stamp

    @property
    def latest_stamp(self):
        with self._lock:
            if not self._records:
                raise TemporalResolutionError(GraphErrorCode.TEMPORAL_OUT_OF_RANGE, "Edge buffer is empty.")
            return self._records[-1].stamp

    @property
    def is_static(self):
        return bool(self._static)

    def insert(self, record):
        if not isinstance(record, TransformDistributionStamped):
            raise TypeError("record must be a TransformDistributionStamped.")
        endpoints = (record.parent_frame_id, record.child_frame_id)
        with self._lock:
            if self._edge_id is None:
                self._edge_id = record.edge_id
                self._endpoints = endpoints
                self._static = record.is_static
            elif record.edge_id != self._edge_id or endpoints != self._endpoints:
                raise ValueError("All records in an EdgeTimeBuffer must describe one physical edge.")
            elif record.is_static != self._static:
                raise ValueError("A physical edge cannot mix static and dynamic records.")

            if self._static and self._records:
                existing = self._records[-1]
                if existing.authority != record.authority:
                    raise TemporalResolutionError(
                        GraphErrorCode.AUTHORITY_CONFLICT,
                        "Static edge authorities '{}' and '{}' conflict.".format(
                            existing.authority,
                            record.authority,
                        ),
                    )
                if _same_static_payload(existing, record):
                    return
                raise TemporalResolutionError(
                    GraphErrorCode.STATIC_EDGE_CONFLICT,
                    "A static edge is time invariant and cannot change payload.",
                )

            stamps = [item.stamp for item in self._records]
            index = bisect_left(stamps, record.stamp)
            if index < len(self._records) and self._records[index].stamp == record.stamp:
                existing = self._records[index]
                if existing.authority != record.authority:
                    if self.conflict_policy is AuthorityConflictPolicy.REJECT:
                        raise TemporalResolutionError(
                            GraphErrorCode.AUTHORITY_CONFLICT,
                            "Authorities '{}' and '{}' conflict at stamp {}.".format(
                                existing.authority,
                                record.authority,
                                record.stamp,
                            ),
                        )
                    if self.conflict_policy is AuthorityConflictPolicy.KEEP_FIRST:
                        return
                self._records[index] = record
            else:
                self._records.insert(index, record)
            if self.max_records is not None and len(self._records) > self.max_records:
                del self._records[: len(self._records) - self.max_records]

    def record_at_sample_stamp(self, stamp):
        with self._lock:
            stamps = [item.stamp for item in self._records]
            index = bisect_left(stamps, float(stamp))
            if index == len(self._records) or self._records[index].stamp != float(stamp):
                raise KeyError("No sample at stamp {}.".format(stamp))
            return self._records[index]

    def model_authority(self, stamp, policy):
        """Return the authority whose explicitly bound model would be used."""

        if not isinstance(policy, TemporalPolicy):
            raise TypeError("policy must be TemporalPolicy.")
        if policy not in (
            TemporalPolicy.INTERPOLATE_WITH_MODEL,
            TemporalPolicy.PREDICT_WITH_MODEL,
        ):
            raise ValueError("model_authority requires a model-based policy.")
        if stamp is None:
            raise ValueError("stamp is required for model-based temporal policies.")
        requested = float(stamp)
        if not np.isfinite(requested) or requested < 0.0:
            raise ValueError("stamp must be finite and non-negative.")
        with self._lock:
            if not self._records:
                raise TemporalResolutionError(
                    GraphErrorCode.TEMPORAL_OUT_OF_RANGE,
                    "Edge buffer is empty.",
                )
            if self._static:
                return self._records[-1].authority
            stamps = [item.stamp for item in self._records]
            exact = bisect_left(stamps, requested)
            if exact < len(stamps) and stamps[exact] == requested:
                return self._records[exact].authority
            if policy is TemporalPolicy.INTERPOLATE_WITH_MODEL:
                if exact == 0 or exact == len(stamps):
                    raise TemporalResolutionError(
                        GraphErrorCode.MODEL_SUPPORT_EXCEEDED,
                        "Interpolation requires samples strictly bracketing the requested stamp.",
                    )
                left = self._records[exact - 1]
                right = self._records[exact]
                if left.authority != right.authority:
                    raise TemporalResolutionError(
                        GraphErrorCode.AUTHORITY_CONFLICT,
                        "Interpolation endpoints have different authorities.",
                    )
                return left.authority
            index = bisect_right(stamps, requested) - 1
            if index < 0:
                raise TemporalResolutionError(
                    GraphErrorCode.MODEL_SUPPORT_EXCEEDED,
                    "Prediction requires an anchor at or before the requested stamp.",
                )
            return self._records[index].authority

    @staticmethod
    def _safe_uncertainty_trace(record):
        try:
            value = record_uncertainty_trace(record)
        except ValueError:
            return float("inf")
        return value if not np.isnan(value) else float("inf")

    @staticmethod
    def _selection_evaluation(record, requested, policy, diagnostic):
        dependency = source_record_dependency_id(record)
        code = (
            TemporalDiagnosticCode.STATIC_EDGE
            if diagnostic == "STATIC_EDGE"
            else TemporalDiagnosticCode.EXACT_SAMPLE
        )
        kind = (
            TemporalEvaluationKind.STATIC
            if diagnostic == "STATIC_EDGE"
            else TemporalEvaluationKind.SAMPLE_SELECTION
        )
        output = record
        if kind is TemporalEvaluationKind.STATIC:
            detail = temporal_detail(
                model_id="static_identity",
                model_version="1",
                config_fingerprint="not_applicable",
                source_stamps=(record.stamp,),
                dependency_ids=(dependency,),
                authority=record.authority,
                backend=TemporalUncertaintyBackend.MOMENT,
                evaluation_kind=kind,
                requested_stamp=requested,
                horizon=0.0,
                random_seed=None,
                random_stream="",
                diagnostics=(code,),
                warnings=(),
            )
            output = replace(
                copy_record_at_stamp(record, requested),
                provenance=make_transform_provenance((record,), detail),
            )
        uncertainty = EdgeTimeBuffer._safe_uncertainty_trace(record)
        return TemporalEvaluationResult(
            record=output,
            requested_stamp=requested,
            evaluated_stamp=output.stamp,
            source_stamps=(record.stamp,),
            model_id="static_identity" if kind is TemporalEvaluationKind.STATIC else "sample_selection",
            model_version="1",
            config_fingerprint="not_applicable",
            evaluation_kind=kind,
            horizon=0.0,
            dependency_ids=(dependency,),
            backend=TemporalUncertaintyBackend.MOMENT,
            approximation=record.approximation,
            diagnostics=(code,),
            initial_uncertainty_trace=uncertainty,
            result_uncertainty_trace=uncertainty,
        )

    @staticmethod
    def _validate_model_result(
        result,
        request,
        edge_id,
        endpoints,
        authority,
        temporal_model,
    ):
        if not isinstance(result, TemporalEvaluationResult):
            raise TypeError("TemporalModel methods must return TemporalEvaluationResult.")
        if result.model_id != request.model_selector:
            raise ValueError("Temporal result model_id does not match the selected model.")
        if (
            result.model_id != temporal_model.model_id
            or result.model_version != temporal_model.version
            or result.config_fingerprint != temporal_model.config_fingerprint
            or result.backend is not temporal_model.backend
        ):
            raise ValueError("Temporal result does not match the registered model configuration.")
        expected_kind = {
            TemporalPolicy.INTERPOLATE_WITH_MODEL:
                TemporalEvaluationKind.MODEL_INTERPOLATION,
            TemporalPolicy.PREDICT_WITH_MODEL:
                TemporalEvaluationKind.MODEL_PREDICTION,
        }[request.policy]
        if result.evaluation_kind is not expected_kind:
            raise ValueError("Temporal result kind does not match the requested policy.")
        if result.record.edge_id != edge_id:
            raise ValueError("Temporal result changed the physical edge_id.")
        if (
            result.record.parent_frame_id,
            result.record.child_frame_id,
        ) != endpoints:
            raise ValueError("Temporal result changed the physical edge endpoints.")
        if result.record.authority != authority:
            raise ValueError("Temporal result changed the bound authority.")
        if not np.isclose(
            result.requested_stamp,
            request.requested_stamp,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("Temporal result requested_stamp does not match the request.")
        if not np.isclose(
            result.evaluated_stamp,
            request.requested_stamp,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("Temporal result must evaluate at the requested stamp.")
        records_by_stamp = {record.stamp: record for record in request.anchors}
        if len(records_by_stamp) != len(request.anchors) or any(
            stamp not in records_by_stamp for stamp in result.source_stamps
        ):
            raise ValueError("Temporal result cites a source stamp absent from the request.")
        if request.policy is TemporalPolicy.INTERPOLATE_WITH_MODEL:
            expected_stamps = tuple(record.stamp for record in request.anchors)
            if result.source_stamps != expected_stamps:
                raise ValueError("Interpolation result must cite both request anchors.")
        else:
            available_stamps = tuple(record.stamp for record in request.anchors)
            if (
                len(result.source_stamps) < temporal_model.minimum_history
                or result.source_stamps
                != available_stamps[-len(result.source_stamps) :]
            ):
                raise ValueError(
                    "Prediction result source stamps must be a sufficient causal suffix."
                )
        source_records = tuple(
            records_by_stamp[stamp] for stamp in result.source_stamps
        )
        if result.dependency_ids != source_record_dependency_ids(source_records):
            raise ValueError("Temporal result dependency lineage does not match its sources.")
        actual_trace = record_uncertainty_trace(result.record)
        if (
            np.isinf(actual_trace)
            and not np.isinf(result.result_uncertainty_trace)
        ) or (
            np.isfinite(actual_trace)
            and result.result_uncertainty_trace + 1.0e-10 < actual_trace
        ):
            raise ValueError("Temporal result understates the record uncertainty trace.")
        expected_payload = {
            "authority": authority,
            "backend": result.backend.value,
            "config_fingerprint": result.config_fingerprint,
            "dependency_ids": list(result.dependency_ids),
            "diagnostics": [item.value for item in result.diagnostics],
            "evaluation_kind": result.evaluation_kind.value,
            "horizon": result.horizon,
            "model_id": result.model_id,
            "model_version": result.model_version,
            "random_seed": result.random_seed,
            "random_stream": result.random_stream,
            "requested_stamp": result.requested_stamp,
            "source_stamps": list(result.source_stamps),
            "warnings": list(result.warnings),
        }
        provenance_details = (result.record.provenance.detail,) + tuple(
            component.provenance.detail
            for component in result.record.distribution.components
        )
        if any(
            parse_temporal_detail(detail) != expected_payload
            for detail in provenance_details
        ):
            raise ValueError(
                "Temporal result provenance payload is missing or inconsistent."
            )

    @staticmethod
    def _apply_uncertainty_limit(result, max_uncertainty_trace, allow_degraded):
        limit = max_uncertainty_trace
        if limit is None or result.result_uncertainty_trace <= limit:
            return result
        if not allow_degraded:
            raise TemporalResolutionError(
                GraphErrorCode.UNCERTAINTY_LIMIT_EXCEEDED,
                "Temporal model uncertainty trace exceeds the configured limit.",
            )
        diagnostics = tuple(result.diagnostics)
        if TemporalDiagnosticCode.UNCERTAINTY_LIMIT_EXCEEDED not in diagnostics:
            diagnostics += (TemporalDiagnosticCode.UNCERTAINTY_LIMIT_EXCEEDED,)
        warnings = result.warnings + (
            "Uncertainty limit exceeded; returning an explicitly degraded result.",
        )
        detail = temporal_detail(
            model_id=result.model_id,
            model_version=result.model_version,
            config_fingerprint=result.config_fingerprint,
            source_stamps=result.source_stamps,
            dependency_ids=result.dependency_ids,
            authority=result.record.authority,
            backend=result.backend,
            evaluation_kind=result.evaluation_kind,
            requested_stamp=result.requested_stamp,
            horizon=result.horizon,
            random_seed=result.random_seed,
            random_stream=result.random_stream,
            diagnostics=diagnostics,
            warnings=warnings,
        )
        record = replace(
            result.record,
            provenance=replace(
                result.record.provenance,
                method="temporal_model_evaluation",
                detail=detail,
            ),
            distribution=replace(
                result.record.distribution,
                components=tuple(
                    replace(
                        component,
                        provenance=replace(
                            component.provenance,
                            method="temporal_model_evaluation",
                            detail=detail,
                        ),
                    )
                    for component in result.record.distribution.components
                ),
            ),
        )
        return replace(
            result,
            record=record,
            diagnostics=diagnostics,
            warnings=warnings,
        )

    def resolve(
        self,
        stamp,
        policy,
        tolerance=0.0,
        max_age=None,
        *,
        temporal_model=None,
        model_selector=None,
        max_prediction_horizon=None,
        random_seed=None,
        random_stream="",
        query_mode=TemporalQueryMode.ONLINE,
        max_uncertainty_trace=None,
        allow_degraded=False,
    ):
        if not isinstance(policy, TemporalPolicy):
            raise TypeError("policy must be TemporalPolicy.")
        if policy is TemporalPolicy.LATEST_COMMON:
            raise TemporalResolutionError(
                GraphErrorCode.UNSUPPORTED_TEMPORAL_POLICY,
                "LATEST_COMMON is a path-level policy.",
            )
        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        if max_age is not None:
            max_age = float(max_age)
            if not np.isfinite(max_age) or max_age < 0.0:
                raise ValueError("max_age must be finite and non-negative or None.")
        if not isinstance(query_mode, TemporalQueryMode):
            raise TypeError("query_mode must be TemporalQueryMode.")
        if temporal_model is not None and not isinstance(temporal_model, TemporalModel):
            raise TypeError("temporal_model must be TemporalModel or None.")

        with self._lock:
            if not self._records:
                raise TemporalResolutionError(GraphErrorCode.TEMPORAL_OUT_OF_RANGE, "Edge buffer is empty.")
            if self._static:
                requested = self._records[-1].stamp if stamp is None else float(stamp)
                record = self._records[-1]
                if policy in (
                    TemporalPolicy.INTERPOLATE_WITH_MODEL,
                    TemporalPolicy.PREDICT_WITH_MODEL,
                ):
                    evaluation = self._selection_evaluation(
                        record,
                        requested,
                        policy,
                        "STATIC_EDGE",
                    )
                    evaluation = self._apply_uncertainty_limit(
                        evaluation,
                        max_uncertainty_trace,
                        allow_degraded,
                    )
                    return ResolvedEdgeRecord(
                        evaluation.record,
                        requested,
                        evaluation.record.stamp,
                        policy,
                        "STATIC_EDGE",
                        evaluation,
                    )
                return ResolvedEdgeRecord(record, requested, record.stamp, policy, "STATIC_EDGE")

            if policy is TemporalPolicy.LATEST and stamp is None:
                record = self._records[-1]
                return ResolvedEdgeRecord(record, record.stamp, record.stamp, policy, "LATEST_SAMPLE")
            if stamp is None:
                raise ValueError("stamp is required for this temporal policy.")
            requested = float(stamp)
            if not np.isfinite(requested) or requested < 0.0:
                raise ValueError("stamp must be finite and non-negative.")
            stamps = [item.stamp for item in self._records]

            if policy in (
                TemporalPolicy.INTERPOLATE_WITH_MODEL,
                TemporalPolicy.PREDICT_WITH_MODEL,
            ):
                exact = bisect_left(stamps, requested)
                if exact < len(stamps) and stamps[exact] == requested:
                    record = self._records[exact]
                    evaluation = self._selection_evaluation(
                        record,
                        requested,
                        policy,
                        "EXACT_SAMPLE",
                    )
                    evaluation = self._apply_uncertainty_limit(
                        evaluation,
                        max_uncertainty_trace,
                        allow_degraded,
                    )
                    return ResolvedEdgeRecord(
                        record,
                        requested,
                        record.stamp,
                        policy,
                        "EXACT_SAMPLE",
                        evaluation,
                    )

                if policy is TemporalPolicy.INTERPOLATE_WITH_MODEL:
                    if exact == 0 or exact == len(stamps):
                        raise TemporalResolutionError(
                            GraphErrorCode.MODEL_SUPPORT_EXCEEDED,
                            "Interpolation requires samples strictly bracketing the requested stamp.",
                        )
                    left = self._records[exact - 1]
                    right = self._records[exact]
                    if left.authority != right.authority:
                        raise TemporalResolutionError(
                            GraphErrorCode.AUTHORITY_CONFLICT,
                            "Interpolation endpoints have different authorities.",
                        )
                    if query_mode is not TemporalQueryMode.OFFLINE_SMOOTHING:
                        raise TemporalResolutionError(
                            GraphErrorCode.NON_CAUSAL_INPUT_REJECTED,
                            "Interpolation uses a future endpoint and requires offline_smoothing mode.",
                        )
                    if max_age is not None and requested - left.stamp > max_age:
                        raise TemporalResolutionError(
                            GraphErrorCode.TEMPORAL_STALE,
                            "Interpolation's causal-side anchor is older than max_age.",
                        )
                    if temporal_model is None:
                        raise TemporalResolutionError(
                            GraphErrorCode.MODEL_NOT_REGISTERED,
                            "No named temporal model is registered for this edge and authority.",
                        )
                    if not temporal_model.supports_interpolation:
                        raise TemporalResolutionError(
                            GraphErrorCode.MODEL_SUPPORT_EXCEEDED,
                            "The selected model does not support interpolation.",
                        )
                    anchors = (left, right)
                    if any(
                        not temporal_model.validate_distribution(record)
                        for record in anchors
                    ):
                        raise TemporalResolutionError(
                            GraphErrorCode.MODEL_SUPPORT_EXCEEDED,
                            "An interpolation endpoint distribution is outside model support.",
                        )
                    selector = temporal_model.model_id if model_selector is None else str(model_selector)
                    if selector != temporal_model.model_id:
                        raise ValueError("model_selector does not match temporal_model.model_id.")
                    request = TemporalEvaluationRequest(
                        requested_stamp=requested,
                        policy=policy,
                        anchors=anchors,
                        model_selector=selector,
                        max_prediction_horizon=max_prediction_horizon,
                        max_age=max_age,
                        random_seed=random_seed,
                        random_stream=random_stream,
                        query_mode=query_mode,
                        max_uncertainty_trace=max_uncertainty_trace,
                        allow_degraded=allow_degraded,
                    )
                    result = temporal_model.interpolate(left, right, request)
                    self._validate_model_result(
                        result,
                        request,
                        self._edge_id,
                        self._endpoints,
                        left.authority,
                        temporal_model,
                    )
                    result = self._apply_uncertainty_limit(
                        result,
                        request.max_uncertainty_trace,
                        request.allow_degraded,
                    )
                    return ResolvedEdgeRecord(
                        result.record,
                        requested,
                        result.evaluated_stamp,
                        policy,
                        "MODEL_INTERPOLATION",
                        result,
                    )

                if max_prediction_horizon is None:
                    raise ValueError(
                        "max_prediction_horizon is mandatory for PREDICT_WITH_MODEL."
                    )
                maximum = float(max_prediction_horizon)
                if not np.isfinite(maximum) or maximum < 0.0:
                    raise ValueError(
                        "max_prediction_horizon must be finite and non-negative."
                    )
                if max_age is None:
                    raise ValueError("max_age is mandatory for PREDICT_WITH_MODEL.")
                index = bisect_right(stamps, requested) - 1
                if index < 0:
                    raise TemporalResolutionError(
                        GraphErrorCode.MODEL_SUPPORT_EXCEEDED,
                        "Prediction requires an anchor at or before the requested stamp.",
                    )
                anchor = self._records[index]
                age = requested - anchor.stamp
                if age > max_age:
                    raise TemporalResolutionError(
                        GraphErrorCode.TEMPORAL_STALE,
                        "Prediction anchor is older than max_age.",
                    )
                if age > maximum:
                    raise TemporalResolutionError(
                        GraphErrorCode.PREDICTION_HORIZON_EXCEEDED,
                        "Prediction exceeds max_prediction_horizon.",
                    )
                if temporal_model is None:
                    raise TemporalResolutionError(
                        GraphErrorCode.MODEL_NOT_REGISTERED,
                        "No named temporal model is registered for this edge and authority.",
                    )
                if age > temporal_model.maximum_horizon:
                    raise TemporalResolutionError(
                        GraphErrorCode.MODEL_SUPPORT_EXCEEDED,
                        "Prediction exceeds the selected model's support.",
                    )
                if not temporal_model.supports_prediction:
                    raise TemporalResolutionError(
                        GraphErrorCode.MODEL_SUPPORT_EXCEEDED,
                        "The selected model does not support prediction.",
                    )
                history = tuple(
                    record
                    for record in self._records[: index + 1]
                    if record.authority == anchor.authority
                    and record.stamp <= requested
                )
                if len(history) < temporal_model.minimum_history:
                    raise TemporalResolutionError(
                        GraphErrorCode.INSUFFICIENT_HISTORY,
                        "Prediction history is shorter than the model requirement.",
                    )
                if any(
                    not temporal_model.validate_distribution(record)
                    for record in history[-temporal_model.minimum_history :]
                ):
                    raise TemporalResolutionError(
                        GraphErrorCode.MODEL_SUPPORT_EXCEEDED,
                        "A prediction anchor distribution is outside model support.",
                    )
                selector = temporal_model.model_id if model_selector is None else str(model_selector)
                if selector != temporal_model.model_id:
                    raise ValueError("model_selector does not match temporal_model.model_id.")
                request = TemporalEvaluationRequest(
                    requested_stamp=requested,
                    policy=policy,
                    anchors=history,
                    model_selector=selector,
                    max_prediction_horizon=maximum,
                    max_age=max_age,
                    random_seed=random_seed,
                    random_stream=random_stream,
                    query_mode=query_mode,
                    max_uncertainty_trace=max_uncertainty_trace,
                    allow_degraded=allow_degraded,
                )
                result = temporal_model.predict(history, request)
                self._validate_model_result(
                    result,
                    request,
                    self._edge_id,
                    self._endpoints,
                    anchor.authority,
                    temporal_model,
                )
                if any(source_stamp > requested for source_stamp in result.source_stamps):
                    raise TemporalResolutionError(
                        GraphErrorCode.NON_CAUSAL_INPUT_REJECTED,
                        "Prediction result cites a future source sample.",
                    )
                result = self._apply_uncertainty_limit(
                    result,
                    request.max_uncertainty_trace,
                    request.allow_degraded,
                )
                return ResolvedEdgeRecord(
                    result.record,
                    requested,
                    result.evaluated_stamp,
                    policy,
                    "MODEL_PREDICTION",
                    result,
                )

            if policy is TemporalPolicy.EXACT:
                index = bisect_left(stamps, requested)
                if index == len(stamps) or stamps[index] != requested:
                    raise TemporalResolutionError(
                        GraphErrorCode.TEMPORAL_OUT_OF_RANGE,
                        "No exact edge sample exists at stamp {}.".format(requested),
                    )
                record = self._records[index]
                return ResolvedEdgeRecord(record, requested, record.stamp, policy, "EXACT_SAMPLE")

            if policy is TemporalPolicy.NEAREST_WITHIN_TOLERANCE:
                right = bisect_left(stamps, requested)
                candidates = []
                if right > 0:
                    candidates.append(self._records[right - 1])
                if right < len(self._records):
                    candidates.append(self._records[right])
                record = min(candidates, key=lambda item: (abs(item.stamp - requested), item.stamp))
                if abs(record.stamp - requested) > tolerance:
                    raise TemporalResolutionError(
                        GraphErrorCode.TEMPORAL_OUT_OF_RANGE,
                        "Nearest edge sample is outside tolerance.",
                    )
                return ResolvedEdgeRecord(record, requested, record.stamp, policy, "NEAREST_SAMPLE")

            if policy is TemporalPolicy.LATEST:
                index = bisect_right(stamps, requested) - 1
                if index < 0:
                    raise TemporalResolutionError(
                        GraphErrorCode.TEMPORAL_OUT_OF_RANGE,
                        "No edge sample exists at or before stamp {}.".format(requested),
                    )
                record = self._records[index]
                if max_age is not None and requested - record.stamp > max_age:
                    raise TemporalResolutionError(
                        GraphErrorCode.TEMPORAL_STALE,
                        "Latest edge sample is older than max_age.",
                    )
                return ResolvedEdgeRecord(
                    record,
                    requested,
                    record.stamp,
                    policy,
                    "LATEST_AT_OR_BEFORE",
                )

        raise TemporalResolutionError(
            GraphErrorCode.UNSUPPORTED_TEMPORAL_POLICY,
            "Unsupported temporal policy '{}'.".format(policy.value),
        )
