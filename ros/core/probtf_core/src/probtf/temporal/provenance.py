"""Stable source-record identities and wire-preserving temporal provenance."""

import hashlib
import json

import numpy as np

from probtf.distributions import TransformDistributionStamped
from probtf.provenance import ComponentProvenance, TransformProvenance


_DETAIL_PREFIX = "probtf.temporal/v1:"


def _update_array(digest, value):
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def _update_approximation(digest, approximation):
    for value in (
        approximation.kind.value,
        bool(approximation.lossy),
        approximation.detail,
        approximation.source,
        approximation.error_bound,
    ):
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")


def source_record_dependency_id(record):
    """Return a deterministic content identity for one immutable source record."""

    if not isinstance(record, TransformDistributionStamped):
        raise TypeError("record must be TransformDistributionStamped.")
    digest = hashlib.sha256()
    for value in (
        record.edge_id,
        record.parent_frame_id,
        record.child_frame_id,
        "{:.17g}".format(record.stamp),
        record.authority,
        str(bool(record.is_static)),
        record.representative_kind.value,
        record.provenance.source_ids,
        record.provenance.derived_from_edge_ids,
        record.provenance.method,
        record.provenance.detail,
    ):
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
    _update_approximation(digest, record.approximation)
    if record.representative is None:
        digest.update(b"representative:none\0")
    else:
        digest.update(b"representative:present\0")
        _update_array(digest, record.representative.translation)
        _update_array(digest, record.representative.rotation_wxyz)
    for component in record.distribution.components:
        for value in (
            component.component_id,
            "{:.17g}".format(component.raw_weight),
            component.orientation.kind.value,
            "{:.17g}".format(component.orientation.inverse_concentration),
            component.provenance.source_ids,
            component.provenance.derived_from_edge_ids,
            component.provenance.method,
            component.provenance.detail,
        ):
            digest.update(repr(value).encode("utf-8"))
            digest.update(b"\0")
        _update_array(digest, component.orientation.shape_matrix)
        _update_array(digest, component.orientation.reference_quaternion_wxyz)
        _update_array(digest, component.translation.mean_at_reference)
        _update_array(digest, component.translation.residual_covariance)
        _update_array(digest, component.translation.rotation_coupling)
        _update_approximation(digest, component.approximation)
    return "record:{}".format(digest.hexdigest())


def source_record_dependency_ids(records):
    output = []
    for record in records:
        nested = temporal_dependency_ids(record)
        dependencies = nested or (source_record_dependency_id(record),)
        for dependency_id in dependencies:
            if dependency_id not in output:
                output.append(dependency_id)
    return tuple(output)


def temporal_detail(
    *,
    model_id,
    model_version,
    config_fingerprint,
    source_stamps,
    dependency_ids,
    authority,
    backend,
    evaluation_kind,
    requested_stamp,
    horizon,
    random_seed,
    random_stream,
    diagnostics=(),
    warnings=(),
):
    """Encode temporal metadata in existing v2 provenance wire fields.

    The current ROS v2 message already round-trips ``method`` and ``detail``.
    Encoding the extension here therefore avoids silently losing temporal
    provenance when an evaluated record crosses a ROS bridge.
    """

    payload = {
        "authority": str(authority),
        "backend": backend.value,
        "config_fingerprint": str(config_fingerprint),
        "dependency_ids": list(dependency_ids),
        "diagnostics": [item.value for item in diagnostics],
        "evaluation_kind": evaluation_kind.value,
        "horizon": float(horizon),
        "model_id": str(model_id),
        "model_version": str(model_version),
        "random_seed": random_seed,
        "random_stream": str(random_stream),
        "requested_stamp": float(requested_stamp),
        "source_stamps": [float(stamp) for stamp in source_stamps],
        "warnings": [str(warning) for warning in warnings],
    }
    return _DETAIL_PREFIX + json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_temporal_detail(detail):
    text = str(detail)
    if not text.startswith(_DETAIL_PREFIX):
        return None
    return json.loads(text[len(_DETAIL_PREFIX) :])


def temporal_dependency_ids(record):
    """Recover all versioned temporal source identities from a record."""

    if not isinstance(record, TransformDistributionStamped):
        raise TypeError("record must be TransformDistributionStamped.")
    output = []
    for detail in (record.provenance.detail,) + tuple(
        component.provenance.detail
        for component in record.distribution.components
    ):
        payload = parse_temporal_detail(detail)
        if payload is None:
            continue
        for dependency_id in payload.get("dependency_ids", ()):
            dependency_id = str(dependency_id)
            if dependency_id and dependency_id not in output:
                output.append(dependency_id)
    return tuple(output)


def merged_source_ids(records):
    output = []
    for record in records:
        for source_id in record.provenance.source_ids:
            if source_id not in output:
                output.append(source_id)
        for component in record.distribution.components:
            for source_id in component.provenance.source_ids:
                if source_id not in output:
                    output.append(source_id)
    return tuple(output)


def merged_edge_ids(records):
    output = []
    for record in records:
        for edge_id in (record.edge_id,) + tuple(record.provenance.derived_from_edge_ids):
            if edge_id not in output:
                output.append(edge_id)
    return tuple(output)


def make_transform_provenance(records, detail):
    return TransformProvenance(
        source_ids=merged_source_ids(records),
        derived_from_edge_ids=merged_edge_ids(records),
        method="temporal_model_evaluation",
        detail=detail,
    )


def make_component_provenance(records, detail):
    return ComponentProvenance(
        source_ids=merged_source_ids(records),
        derived_from_edge_ids=merged_edge_ids(records),
        method="temporal_model_evaluation",
        detail=detail,
    )


__all__ = [
    "make_component_provenance",
    "make_transform_provenance",
    "merged_edge_ids",
    "merged_source_ids",
    "parse_temporal_detail",
    "source_record_dependency_id",
    "source_record_dependency_ids",
    "temporal_dependency_ids",
    "temporal_detail",
]
