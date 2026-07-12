"""Explicit deterministic TF import and representative-only TF export."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    RepresentativeKind,
    RepresentativePolicy,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.provenance import (
    ApproximationInfo,
    ApproximationKind,
    ComponentProvenance,
    TransformProvenance,
)


class TfExportPolicy(Enum):
    EXACT_ONLY = "exact_only"
    STORED_REPRESENTATIVE = "stored_representative"
    HIGHEST_WEIGHT_COMPONENT_MODE = "highest_weight_component_mode"


@dataclass(frozen=True)
class TfExportResult:
    message: object
    approximation: ApproximationInfo


def _stamp_to_seconds(stamp):
    if hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        return float(stamp.secs) + float(stamp.nsecs) * 1e-9
    return float(stamp)


def _time_from_seconds(seconds, time_factory=None):
    if time_factory is None:
        import rospy

        time_factory = rospy.Time.from_sec
    return time_factory(float(seconds))


def _quaternion_wxyz(message):
    return np.array([message.w, message.x, message.y, message.z], dtype=float)


def _assign_quaternion(message, quaternion):
    message.w = float(quaternion[0])
    message.x = float(quaternion[1])
    message.y = float(quaternion[2])
    message.z = float(quaternion[3])


def _vector(message):
    return np.array([message.x, message.y, message.z], dtype=float)


def _assign_vector(message, vector):
    message.x, message.y, message.z = (float(value) for value in vector)


def deterministic_tf_to_record(message, authority, is_static=False, edge_id=None):
    """Import ``geometry_msgs/TransformStamped`` as one exact Dirac component."""

    parent = str(message.header.frame_id).lstrip("/")
    child = str(message.child_frame_id).lstrip("/")
    identifier = str(edge_id or "{}__to__{}".format(parent, child))
    quaternion = _quaternion_wxyz(message.transform.rotation)
    translation = _vector(message.transform.translation)
    component = TransformComponent(
        component_id="{}:tf".format(identifier),
        raw_weight=1.0,
        orientation=BinghamOrientation.dirac(quaternion),
        translation=ConditionalGaussianTranslation(
            translation,
            np.zeros((3, 3)),
            np.zeros((3, 9)),
        ),
        provenance=ComponentProvenance(
            source_ids=(str(authority),),
            method="tf_import",
        ),
    )
    deterministic = component.deterministic_transform()
    return TransformDistributionStamped(
        parent_frame_id=parent,
        child_frame_id=child,
        stamp=_stamp_to_seconds(message.header.stamp),
        edge_id=identifier,
        authority=str(authority),
        distribution=TransformDistribution((component,)),
        representative=deterministic,
        representative_kind=RepresentativeKind.EXACT_MAP,
        provenance=TransformProvenance(
            source_ids=(str(authority),),
            method="tf_import",
        ),
        is_static=is_static,
    )


def _export_transform(record, policy):
    exact = record.distribution.deterministic_transform()
    if exact is not None:
        return exact, ApproximationInfo()
    if policy is TfExportPolicy.EXACT_ONLY:
        raise ValueError("Stochastic Prob-TF edges require an explicit representative export policy.")
    if policy is TfExportPolicy.STORED_REPRESENTATIVE:
        if record.representative is None:
            raise ValueError("No stored representative is available.")
        return record.representative, ApproximationInfo(
            ApproximationKind.REPRESENTATIVE_PROJECTION,
            True,
            "Exported the record's explicitly typed stored representative.",
        )
    projection = record.distribution.representative(
        RepresentativePolicy.HIGHEST_WEIGHT_COMPONENT_MODE
    )
    return projection.transform, projection.approximation


def record_to_deterministic_tf(
    record,
    message_type=None,
    time_factory=None,
    policy=TfExportPolicy.EXACT_ONLY,
):
    if not isinstance(policy, TfExportPolicy):
        raise TypeError("policy must be TfExportPolicy.")
    if message_type is None:
        from geometry_msgs.msg import TransformStamped

        message_type = TransformStamped
    transform, approximation = _export_transform(record, policy)
    message = message_type()
    message.header.frame_id = record.parent_frame_id
    message.header.stamp = _time_from_seconds(record.stamp, time_factory)
    message.child_frame_id = record.child_frame_id
    _assign_vector(message.transform.translation, transform.translation)
    _assign_quaternion(message.transform.rotation, transform.rotation_wxyz)
    return TfExportResult(message, approximation)


def _tf_signature(message, is_static):
    transform = message.transform
    return (
        str(message.header.frame_id).lstrip("/"),
        str(message.child_frame_id).lstrip("/"),
        round(_stamp_to_seconds(message.header.stamp), 9),
        bool(is_static),
        tuple(
            round(float(value), 12)
            for value in (
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w,
            )
        ),
    )


class ProbTfTfBridge:
    """Stateful loop prevention around TF import/export conversion."""

    def __init__(self, listener, own_authority="probtf_tf_bridge"):
        self.listener = listener
        self.own_authority = str(own_authority)
        self._exported_signatures = set()

    def import_transform(self, message, authority, is_static=False):
        signature = _tf_signature(message, is_static)
        if str(authority) == self.own_authority or signature in self._exported_signatures:
            return None
        record = deterministic_tf_to_record(message, authority, is_static)
        self.listener.graph.insert(record)
        return record

    def import_tf_array(self, message, authority, is_static=False):
        return tuple(
            record
            for record in (
                self.import_transform(transform, authority, is_static)
                for transform in message.transforms
            )
            if record is not None
        )

    def export_transform(self, record, **kwargs):
        result = record_to_deterministic_tf(record, **kwargs)
        self._exported_signatures.add(_tf_signature(result.message, record.is_static))
        return result

