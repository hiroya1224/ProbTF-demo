"""Lossless v2 message conversion for the Prob-TF joint component model."""

from dataclasses import dataclass

import numpy as np

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    OrientationKind,
    RepresentativeKind,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.geometry import DeterministicTransform, pack_symmetric_upper, unpack_symmetric_upper
from probtf.provenance import (
    ApproximationInfo,
    ApproximationKind,
    ComponentProvenance,
    TransformProvenance,
)


_ORIENTATION_TO_WIRE = {
    OrientationKind.FINITE_BINGHAM: 0,
    OrientationKind.DIRAC: 1,
    OrientationKind.UNIFORM: 2,
}
_ORIENTATION_FROM_WIRE = {value: key for key, value in _ORIENTATION_TO_WIRE.items()}
_APPROXIMATION_ORDER = (
    ApproximationKind.EXACT,
    ApproximationKind.PRODUCER_SUPPLIED,
    ApproximationKind.LEGACY_ADAPTER,
    ApproximationKind.TANGENT_SURROGATE,
    ApproximationKind.NUMERICAL_INTEGRATION,
    ApproximationKind.MONTE_CARLO,
    ApproximationKind.MOMENT_SUMMARY,
    ApproximationKind.MIXTURE_REDUCTION,
    ApproximationKind.BINGHAM_CLOSURE,
    ApproximationKind.REPRESENTATIVE_PROJECTION,
    ApproximationKind.UNAVAILABLE,
)
_APPROXIMATION_TO_WIRE = {kind: index for index, kind in enumerate(_APPROXIMATION_ORDER)}
_REPRESENTATIVE_ORDER = (
    RepresentativeKind.NONE,
    RepresentativeKind.EXACT_MAP,
    RepresentativeKind.COMPONENT_MODE_APPROXIMATION,
    RepresentativeKind.PRODUCER_SUPPLIED,
    RepresentativeKind.MOMENT_REPRESENTATIVE,
)
_REPRESENTATIVE_TO_WIRE = {kind: index for index, kind in enumerate(_REPRESENTATIVE_ORDER)}


@dataclass(frozen=True)
class V2MessageTypes:
    orientation: object
    translation: object
    component: object
    stamped: object
    array: object
    approximation: object
    provenance: object

    @classmethod
    def defaults(cls):
        from probtf_msgs.msg import (
            ApproximationInfo as ApproximationInfoMsg,
            BinghamOrientation as BinghamOrientationMsg,
            ConditionalGaussianTranslation as ConditionalGaussianTranslationMsg,
            ProbabilisticTransformArray as ProbabilisticTransformArrayMsg,
            ProbabilisticTransformComponent as ProbabilisticTransformComponentMsg,
            ProbabilisticTransformStamped as ProbabilisticTransformStampedMsg,
            Provenance as ProvenanceMsg,
        )

        return cls(
            BinghamOrientationMsg,
            ConditionalGaussianTranslationMsg,
            ProbabilisticTransformComponentMsg,
            ProbabilisticTransformStampedMsg,
            ProbabilisticTransformArrayMsg,
            ApproximationInfoMsg,
            ProvenanceMsg,
        )


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


def _assign_vector3(message, values):
    message.x, message.y, message.z = (float(value) for value in values)


def _vector3(message):
    return np.array([message.x, message.y, message.z], dtype=float)


def _assign_quaternion_xyzw(message, quat_wxyz):
    message.w = float(quat_wxyz[0])
    message.x = float(quat_wxyz[1])
    message.y = float(quat_wxyz[2])
    message.z = float(quat_wxyz[3])


def _quaternion_wxyz(message):
    return np.array([message.w, message.x, message.y, message.z], dtype=float)


def approximation_to_msg(value, message_type):
    if not isinstance(value, ApproximationInfo):
        raise TypeError("value must be ApproximationInfo.")
    message = message_type()
    message.kind = _APPROXIMATION_TO_WIRE[value.kind]
    message.lossy = value.lossy
    message.detail = value.detail
    message.source = value.source
    message.has_error_bound = value.error_bound is not None
    message.error_bound = 0.0 if value.error_bound is None else value.error_bound
    return message


def approximation_from_msg(message):
    index = int(message.kind)
    if index < 0 or index >= len(_APPROXIMATION_ORDER):
        raise ValueError("Unknown approximation kind {}.".format(message.kind))
    try:
        kind = _APPROXIMATION_ORDER[index]
    except (IndexError, ValueError) as exc:
        raise ValueError("Unknown approximation kind {}.".format(message.kind)) from exc
    return ApproximationInfo(
        kind,
        bool(message.lossy),
        message.detail,
        message.source,
        float(message.error_bound) if message.has_error_bound else None,
    )


def provenance_to_msg(value, message_type):
    message = message_type()
    message.source_ids = list(value.source_ids)
    message.derived_from_edge_ids = list(value.derived_from_edge_ids)
    message.method = value.method
    message.detail = value.detail
    return message


def provenance_from_msg(message, provenance_type):
    return provenance_type(
        source_ids=tuple(message.source_ids),
        derived_from_edge_ids=tuple(message.derived_from_edge_ids),
        method=message.method,
        detail=message.detail,
    )


def orientation_to_msg(orientation, message_type):
    if not isinstance(orientation, BinghamOrientation):
        raise TypeError("orientation must be BinghamOrientation.")
    message = message_type()
    message.kind = _ORIENTATION_TO_WIRE[orientation.kind]
    message.inverse_concentration = orientation.inverse_concentration
    message.shape_upper_wxyz = pack_symmetric_upper(orientation.shape_matrix).tolist()
    _assign_quaternion_xyzw(
        message.reference_quaternion,
        orientation.reference_quaternion_wxyz,
    )
    return message


def orientation_from_msg(message):
    try:
        kind = _ORIENTATION_FROM_WIRE[int(message.kind)]
    except KeyError as exc:
        raise ValueError("Unknown orientation kind {}.".format(message.kind)) from exc
    return BinghamOrientation(
        kind,
        float(message.inverse_concentration),
        unpack_symmetric_upper(message.shape_upper_wxyz, 4),
        _quaternion_wxyz(message.reference_quaternion),
    )


def translation_to_msg(translation, message_type):
    if not isinstance(translation, ConditionalGaussianTranslation):
        raise TypeError("translation must be ConditionalGaussianTranslation.")
    message = message_type()
    _assign_vector3(message.mean_at_reference, translation.mean_at_reference)
    message.residual_covariance_upper = pack_symmetric_upper(
        translation.residual_covariance
    ).tolist()
    message.rotation_coupling = translation.rotation_coupling.reshape(-1, order="C").tolist()
    return message


def translation_from_msg(message):
    return ConditionalGaussianTranslation(
        _vector3(message.mean_at_reference),
        unpack_symmetric_upper(message.residual_covariance_upper, 3),
        np.asarray(message.rotation_coupling, dtype=float).reshape(3, 9, order="C"),
    )


def component_to_msg(component, types):
    message = types.component()
    message.component_id = component.component_id
    message.weight = component.raw_weight
    message.orientation = orientation_to_msg(component.orientation, types.orientation)
    message.translation = translation_to_msg(component.translation, types.translation)
    message.approximation = approximation_to_msg(component.approximation, types.approximation)
    message.provenance = provenance_to_msg(component.provenance, types.provenance)
    return message


def component_from_msg(message):
    return TransformComponent(
        message.component_id,
        float(message.weight),
        orientation_from_msg(message.orientation),
        translation_from_msg(message.translation),
        provenance_from_msg(message.provenance, ComponentProvenance),
        approximation_from_msg(message.approximation),
    )


def transform_distribution_to_msg(record, message_types=None, time_factory=None):
    if not isinstance(record, TransformDistributionStamped):
        raise TypeError("record must be TransformDistributionStamped.")
    types = V2MessageTypes.defaults() if message_types is None else message_types
    message = types.stamped()
    message.header.frame_id = record.parent_frame_id
    message.header.stamp = _time_from_seconds(record.stamp, time_factory)
    message.child_frame_id = record.child_frame_id
    message.edge_id = record.edge_id
    message.authority = record.authority
    message.is_static = record.is_static
    message.representative_kind = _REPRESENTATIVE_TO_WIRE[record.representative_kind]
    if record.representative is not None:
        _assign_vector3(message.representative.translation, record.representative.translation)
        _assign_quaternion_xyzw(
            message.representative.rotation,
            record.representative.rotation_wxyz,
        )
    message.components = [component_to_msg(component, types) for component in record.distribution.components]
    message.approximation = approximation_to_msg(record.approximation, types.approximation)
    message.provenance = provenance_to_msg(record.provenance, types.provenance)
    return message


def transform_distribution_from_msg(message):
    representative_index = int(message.representative_kind)
    if representative_index < 0 or representative_index >= len(_REPRESENTATIVE_ORDER):
        raise ValueError("Unknown representative kind {}.".format(message.representative_kind))
    try:
        representative_kind = _REPRESENTATIVE_ORDER[representative_index]
    except (IndexError, ValueError) as exc:
        raise ValueError("Unknown representative kind {}.".format(message.representative_kind)) from exc
    representative = None
    if representative_kind is not RepresentativeKind.NONE:
        representative = DeterministicTransform(
            _vector3(message.representative.translation),
            _quaternion_wxyz(message.representative.rotation),
        )
    return TransformDistributionStamped(
        parent_frame_id=message.header.frame_id,
        child_frame_id=message.child_frame_id,
        stamp=_stamp_to_seconds(message.header.stamp),
        edge_id=message.edge_id,
        authority=message.authority,
        distribution=TransformDistribution(tuple(component_from_msg(item) for item in message.components)),
        representative=representative,
        representative_kind=representative_kind,
        provenance=provenance_from_msg(message.provenance, TransformProvenance),
        is_static=bool(message.is_static),
        approximation=approximation_from_msg(message.approximation),
    )


def transform_array_to_msg(records, message_types=None, time_factory=None):
    types = V2MessageTypes.defaults() if message_types is None else message_types
    message = types.array()
    message.transforms = [
        transform_distribution_to_msg(record, types, time_factory) for record in records
    ]
    return message


def transform_array_from_msg(message):
    return tuple(transform_distribution_from_msg(item) for item in message.transforms)
