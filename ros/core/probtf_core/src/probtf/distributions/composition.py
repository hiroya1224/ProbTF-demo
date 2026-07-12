"""Closed operations on the v2 transform-component representation."""

import numpy as np

from probtf.distributions.bingham_orientation import BinghamOrientation
from probtf.distributions.conditional_translation import ConditionalGaussianTranslation
from probtf.distributions.stamped import TransformDistributionStamped
from probtf.distributions.transform_component import TransformComponent
from probtf.distributions.transform_distribution import TransformDistribution
from probtf.geometry import (
    DeterministicTransform,
    quat_mul,
    quat_right_matrix,
    quat_to_rotmat,
    rotation_action_matrix,
)
from probtf.provenance import ComponentProvenance, TransformProvenance


def _append_unique(values, value):
    return tuple(values) if value in values else tuple(values) + (value,)


def _right_composed_orientation(orientation, rotation_wxyz):
    right_action = quat_right_matrix(rotation_wxyz)
    return BinghamOrientation(
        kind=orientation.kind,
        inverse_concentration=orientation.inverse_concentration,
        shape_matrix=right_action @ orientation.shape_matrix @ right_action.T,
        reference_quaternion_wxyz=quat_mul(
            orientation.reference_quaternion_wxyz,
            rotation_wxyz,
        ),
    )


def _right_composed_component(component, fixed_transform, component_id, source_edge_id):
    fixed_rotation = quat_to_rotmat(fixed_transform.rotation_wxyz)
    old_reference_rotation = quat_to_rotmat(
        component.orientation.reference_quaternion_wxyz
    )

    # R_new = R_old R_fixed, hence vec(R_old) =
    # (R_fixed kron I) vec(R_new) for column-major vectorization.
    old_from_new_rotation_vector = np.kron(fixed_rotation, np.eye(3))
    offset_in_new_child = fixed_rotation.T @ fixed_transform.translation
    coupling = (
        component.translation.rotation_coupling @ old_from_new_rotation_vector
        + rotation_action_matrix(offset_in_new_child)
    )
    mean_at_reference = (
        component.translation.mean_at_reference
        + old_reference_rotation @ fixed_transform.translation
    )

    provenance = component.provenance
    return TransformComponent(
        component_id=component_id,
        raw_weight=component.raw_weight,
        orientation=_right_composed_orientation(
            component.orientation,
            fixed_transform.rotation_wxyz,
        ),
        translation=ConditionalGaussianTranslation(
            mean_at_reference,
            component.translation.residual_covariance,
            coupling,
        ),
        provenance=ComponentProvenance(
            source_ids=provenance.source_ids,
            derived_from_edge_ids=_append_unique(
                provenance.derived_from_edge_ids,
                source_edge_id,
            ),
            method="deterministic_right_composition",
            detail=provenance.detail,
        ),
        approximation=component.approximation,
    )


def compose_with_deterministic_right(
    record,
    fixed_transform,
    child_frame_id,
    edge_id,
    authority=None,
):
    """Return ``record * fixed_transform`` without losing rotation coupling.

    ``record`` maps its child coordinates into its parent. ``fixed_transform``
    maps the new child coordinates into ``record.child_frame_id``. The result
    therefore maps the new child directly into ``record.parent_frame_id``.
    Mixture weights and residual covariance are preserved component by
    component; the deterministic offset is represented exactly by the affine
    ``vec(R)`` coupling term.
    """

    if not isinstance(record, TransformDistributionStamped):
        raise TypeError("record must be a TransformDistributionStamped.")
    if not isinstance(fixed_transform, DeterministicTransform):
        raise TypeError("fixed_transform must be a DeterministicTransform.")

    components = tuple(
        _right_composed_component(
            component,
            fixed_transform,
            "{}:{}".format(edge_id, component.component_id),
            record.edge_id,
        )
        for component in record.distribution.components
    )
    representative = (
        None
        if record.representative is None
        else fixed_transform.then(record.representative)
    )
    provenance = record.provenance
    return TransformDistributionStamped(
        parent_frame_id=record.parent_frame_id,
        child_frame_id=child_frame_id,
        stamp=record.stamp,
        edge_id=edge_id,
        authority=record.authority if authority is None else authority,
        distribution=TransformDistribution(components),
        representative=representative,
        representative_kind=record.representative_kind,
        provenance=TransformProvenance(
            source_ids=provenance.source_ids,
            derived_from_edge_ids=_append_unique(
                provenance.derived_from_edge_ids,
                record.edge_id,
            ),
            method="deterministic_right_composition",
            detail=provenance.detail,
        ),
        is_static=record.is_static,
        approximation=record.approximation,
    )
