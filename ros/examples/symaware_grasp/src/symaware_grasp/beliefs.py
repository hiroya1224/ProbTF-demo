"""Native ProbTF v2 belief construction and component summaries."""

import numpy as np

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    DistributionStatus,
    RepresentativeKind,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.geometry import DeterministicTransform
from probtf.probability import (
    PointMomentSummary,
    forward_component_point_moments,
    mixture_point_moments,
)
from probtf.provenance import ApproximationInfo, ComponentProvenance, TransformProvenance


def make_transform_record(
    parent_frame_id,
    child_frame_id,
    stamp,
    edge_id,
    authority,
    position_mean,
    position_covariance,
    orientation_parameter,
    orientation_reference_wxyz,
    source_id,
    component_id=None,
    is_static=False,
    approximation=None,
):
    """Build a one-component v2 joint transform law without a v1 adapter."""

    if approximation is None:
        approximation = ApproximationInfo()
    if not isinstance(approximation, ApproximationInfo):
        raise TypeError("approximation must be ApproximationInfo or None.")
    orientation = BinghamOrientation.from_parameter_matrix(
        orientation_parameter,
        reference_quaternion_wxyz=orientation_reference_wxyz,
    )
    component_id = component_id or "{}:belief".format(edge_id)
    component = TransformComponent(
        component_id=component_id,
        raw_weight=1.0,
        orientation=orientation,
        translation=ConditionalGaussianTranslation(
            np.asarray(position_mean, dtype=float),
            np.asarray(position_covariance, dtype=float),
            np.zeros((3, 9), dtype=float),
        ),
        provenance=ComponentProvenance(
            source_ids=(str(source_id),),
            method="native_v2_belief",
        ),
        approximation=approximation,
    )
    representative = DeterministicTransform(
        component.conditional_translation_mean(orientation.mode_wxyz),
        orientation.mode_wxyz,
    )
    return TransformDistributionStamped(
        parent_frame_id=parent_frame_id,
        child_frame_id=child_frame_id,
        stamp=float(stamp),
        edge_id=edge_id,
        authority=authority,
        distribution=TransformDistribution((component,)),
        representative=representative,
        representative_kind=RepresentativeKind.COMPONENT_MODE_APPROXIMATION,
        provenance=TransformProvenance(
            source_ids=(str(source_id),),
            method="native_v2_belief",
        ),
        is_static=bool(is_static),
        approximation=approximation,
    )


def distribution_point_moments(record, point=(0.0, 0.0, 0.0), integration_steps=120):
    """Evaluate point moments over every usable v2 mixture component."""

    if not isinstance(record, TransformDistributionStamped):
        raise TypeError("record must be a TransformDistributionStamped.")
    normalized = record.distribution.normalize_weights()
    if normalized.status is not DistributionStatus.OK:
        raise ValueError(
            "Cannot evaluate a {} transform distribution.".format(normalized.status.value)
        )
    input_moments = PointMomentSummary(
        np.asarray(point, dtype=float),
        np.zeros((3, 3), dtype=float),
    )
    return mixture_point_moments(
        tuple(
            (
                item.weight,
                forward_component_point_moments(
                    item.component,
                    input_moments,
                    integration_steps,
                ),
            )
            for item in normalized.components
        )
    )


def representative_component(record):
    """Return the highest normalized-weight component and its display pose."""

    if not isinstance(record, TransformDistributionStamped):
        raise TypeError("record must be a TransformDistributionStamped.")
    normalized = record.distribution.normalize_weights()
    if normalized.status is not DistributionStatus.OK:
        raise ValueError(
            "Cannot select from a {} transform distribution.".format(normalized.status.value)
        )
    selected = max(normalized.components, key=lambda item: item.weight)
    quaternion = selected.component.orientation.mode_wxyz
    translation = selected.component.conditional_translation_mean(quaternion)
    return selected.component, translation, quaternion
