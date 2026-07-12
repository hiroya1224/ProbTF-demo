"""ROS v1/v2 adapters with explicit loss diagnostics."""

from dataclasses import dataclass

import numpy as np

from probtf.compatibility import (
    LegacyProjectionPolicy,
    distribution_to_legacy_transform,
    legacy_transform_to_stamped,
)
from probtf.models import ProbabilisticTransform
from probtf_ros.conversions import probabilistic_transform_to_msg


@dataclass(frozen=True)
class LegacyRosConversionResult:
    value: object
    diagnostics: tuple


def legacy_message_to_v2_record(message, authority="legacy_ros_adapter"):
    if not message.has_position or not message.has_orientation:
        raise ValueError(
            "A complete v2 transform requires both position and orientation; missing fields are not zero-filled."
        )
    header_parent = str(message.header.frame_id).lstrip("/")
    explicit_parent = str(message.parent_frame_id).lstrip("/")
    if header_parent and explicit_parent and header_parent != explicit_parent:
        raise ValueError("v1 header.frame_id and parent_frame_id disagree.")
    parent = header_parent or explicit_parent
    mode = np.array(
        [
            message.orientation_mode.w,
            message.orientation_mode.x,
            message.orientation_mode.y,
            message.orientation_mode.z,
        ],
        dtype=float,
    )
    legacy = ProbabilisticTransform.from_arrays(
        parent_frame_id=parent,
        child_frame_id=message.child_frame_id,
        position_mean=[message.position_mean.x, message.position_mean.y, message.position_mean.z],
        position_covariance=np.asarray(message.position_covariance, dtype=float).reshape(3, 3),
        orientation_bingham=np.asarray(message.orientation_bingham.matrix, dtype=float).reshape(4, 4),
        orientation_mode_wxyz=mode,
        stamp=(
            float(message.header.stamp.to_sec())
            if hasattr(message.header.stamp, "to_sec")
            else float(message.header.stamp)
        ),
        edge_id=message.edge_id,
        source_id=message.source_id,
        evidence_source_ids=tuple(message.evidence_source_ids),
        approximation_type=message.approximation_type,
        closure_approximation=message.closure_approximation,
    )
    converted = legacy_transform_to_stamped(legacy, authority)
    return LegacyRosConversionResult(converted.value, converted.diagnostics)


def v2_record_to_legacy_message(
    record,
    message_type=None,
    time_factory=None,
    policy=LegacyProjectionPolicy.EXACT_SINGLE_UNCOUPLED,
):
    projected = distribution_to_legacy_transform(record, policy)
    message = probabilistic_transform_to_msg(projected.value, message_type, time_factory)
    return LegacyRosConversionResult(message, projected.diagnostics)

