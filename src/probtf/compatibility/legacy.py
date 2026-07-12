"""Explicit adapters for the v1 independent Gaussian/Bingham model."""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    DistributionStatus,
    OrientationKind,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.models import BinghamRotation, GaussianPosition, ProbabilisticTransform
from probtf.provenance import (
    ApproximationInfo,
    ApproximationKind,
    ComponentProvenance,
    TransformProvenance,
)


class LegacyProjectionPolicy(Enum):
    EXACT_SINGLE_UNCOUPLED = "exact_single_uncoupled"
    HIGHEST_WEIGHT_COMPONENT_MODE = "highest_weight_component_mode"


@dataclass(frozen=True)
class LegacyConversionResult:
    value: object
    diagnostics: Tuple[str, ...] = ()


def legacy_transform_to_distribution(transform):
    """Embed a v1 independent pose summary as one uncoupled component."""

    if not isinstance(transform, ProbabilisticTransform):
        raise TypeError("transform must be a legacy ProbabilisticTransform.")
    orientation = BinghamOrientation.from_parameter_matrix(
        transform.orientation_bingham,
        transform.orientation_mode_wxyz,
    )
    component = TransformComponent(
        component_id="{}:legacy".format(transform.edge_id),
        raw_weight=1.0,
        orientation=orientation,
        translation=ConditionalGaussianTranslation(
            transform.position_mean,
            transform.position_covariance,
            np.zeros((3, 9), dtype=float),
        ),
        provenance=ComponentProvenance(
            source_ids=tuple(filter(None, (transform.source_id,))) + transform.evidence_source_ids,
            method="legacy_v1_independent_embedding",
        ),
        approximation=ApproximationInfo(
            kind=ApproximationKind.LEGACY_ADAPTER,
            lossy=False,
            detail="The v1 independence assumption is represented explicitly by zero rotation coupling.",
        ),
    )
    return LegacyConversionResult(
        TransformDistribution((component,)),
        ("LEGACY_INDEPENDENCE_ASSUMPTION",),
    )


def legacy_transform_to_stamped(transform, authority="legacy_adapter", stamp_if_missing=0.0):
    converted = legacy_transform_to_distribution(transform)
    stamp = stamp_if_missing if transform.stamp is None else transform.stamp
    record = TransformDistributionStamped(
        parent_frame_id=transform.parent_frame_id,
        child_frame_id=transform.child_frame_id,
        stamp=stamp,
        edge_id=transform.edge_id,
        authority=authority,
        distribution=converted.value,
        provenance=TransformProvenance(
            source_ids=tuple(filter(None, (transform.source_id,))),
            derived_from_edge_ids=(transform.edge_id,),
            method="legacy_v1_adapter",
        ),
    )
    return LegacyConversionResult(record, converted.diagnostics)


def _select_component(distribution, policy):
    normalized = distribution.normalize_weights()
    if normalized.status is not DistributionStatus.OK:
        raise ValueError("Only an OK distribution can be projected to v1.")
    if policy is LegacyProjectionPolicy.EXACT_SINGLE_UNCOUPLED:
        if len(normalized.components) != 1:
            raise ValueError("v1 conversion requires one component or an explicit lossy policy.")
        return normalized.components[0].component, False
    selected = max(normalized.components, key=lambda item: item.weight).component
    return selected, True


def distribution_to_legacy_transform(record, policy=LegacyProjectionPolicy.EXACT_SINGLE_UNCOUPLED):
    """Project a v2 edge to v1 under an explicit loss policy.

    Dirac orientations have no finite v1 Bingham encoding and are rejected.
    A coupled component is accepted only by the explicit mode-projection
    policy, which evaluates translation at the selected component mode.
    """

    if not isinstance(record, TransformDistributionStamped):
        raise TypeError("record must be a TransformDistributionStamped.")
    if not isinstance(policy, LegacyProjectionPolicy):
        raise TypeError("policy must be a LegacyProjectionPolicy.")
    component, projected = _select_component(record.distribution, policy)
    if component.orientation.kind is OrientationKind.DIRAC:
        raise ValueError("A Dirac orientation cannot be encoded by the finite v1 Bingham message.")

    coupled = not np.allclose(component.translation.rotation_coupling, 0.0, rtol=0.0, atol=0.0)
    if coupled and policy is LegacyProjectionPolicy.EXACT_SINGLE_UNCOUPLED:
        raise ValueError("v1 conversion would discard rotation/translation coupling.")

    mode = component.orientation.mode_wxyz
    mean = component.conditional_translation_mean(mode) if coupled else component.translation.mean_at_reference
    parameter = component.orientation.parameter_matrix()
    diagnostics = []
    if projected:
        diagnostics.append("MIXTURE_COMPONENT_MODE_PROJECTION")
    if coupled:
        diagnostics.append("ROTATION_COUPLING_EVALUATED_AT_MODE")
    lossy = bool(diagnostics)
    legacy = ProbabilisticTransform(
        parent_frame_id=record.parent_frame_id,
        child_frame_id=record.child_frame_id,
        position=GaussianPosition(mean, component.translation.residual_covariance),
        orientation=BinghamRotation(parameter, mode),
        stamp=record.stamp,
        edge_id=record.edge_id,
        source_id=record.authority,
        evidence_source_ids=component.provenance.source_ids,
        approximation_type=(
            "legacy_component_mode_projection" if lossy else "legacy_exact_single_uncoupled"
        ),
        closure_approximation=lossy,
    )
    return LegacyConversionResult(legacy, tuple(diagnostics))

