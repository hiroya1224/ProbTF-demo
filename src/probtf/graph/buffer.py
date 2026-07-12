"""Timestamp-ordered histories for physical Prob-TF edges."""

from bisect import bisect_left, bisect_right
from threading import RLock

import numpy as np

from probtf.distributions import TransformDistributionStamped
from probtf.graph.status import GraphErrorCode, TemporalResolutionError
from probtf.temporal import AuthorityConflictPolicy, ResolvedEdgeRecord, TemporalPolicy


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

    def resolve(self, stamp, policy, tolerance=0.0, max_age=None):
        if not isinstance(policy, TemporalPolicy):
            raise TypeError("policy must be TemporalPolicy.")
        if policy in (TemporalPolicy.INTERPOLATE_WITH_MODEL, TemporalPolicy.PREDICT_WITH_MODEL):
            raise TemporalResolutionError(
                GraphErrorCode.UNSUPPORTED_TEMPORAL_POLICY,
                "Temporal model evaluation is not implemented.",
            )
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

        with self._lock:
            if not self._records:
                raise TemporalResolutionError(GraphErrorCode.TEMPORAL_OUT_OF_RANGE, "Edge buffer is empty.")
            if self._static:
                requested = self._records[-1].stamp if stamp is None else float(stamp)
                record = self._records[-1]
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
