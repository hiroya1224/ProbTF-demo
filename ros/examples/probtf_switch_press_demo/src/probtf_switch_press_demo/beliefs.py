"""Native control distributions and deterministic changes of coordinates."""
from dataclasses import dataclass, replace
import json

import numpy as np

from probtf.distributions import (
    BinghamOrientation, ConditionalGaussianTranslation, RepresentativeKind,
    TransformComponent, TransformDistribution, TransformDistributionStamped,
)
from probtf.geometry import DeterministicTransform, quat_left_matrix, quat_mul, rotmat_to_quat
from probtf.provenance import ApproximationInfo, ApproximationKind, ComponentProvenance, TransformProvenance
from probtf.joint import JointTransformEvaluation


BODY_FRAMES = tuple("link_{}".format(index) for index in range(1, 7)) + ("tool0", "rod_tip")


@dataclass(frozen=True)
class JointMeasure(JointTransformEvaluation):
    """Display adapter for a derived query; the source law lives in expression."""

    @classmethod
    def from_query(cls, result):
        return cls(result.body_ids, result.weights, result.local_transforms,
                   result.expression, result.metadata)

    def relative_records(self, before, stamp):
        relative = self.relative_transforms(before)
        metadata = dict(self.metadata, reference_query_id=before.metadata.get("query_id"))
        return tuple(finite_pose_record("observation_hand", "predicted/" + name, stamp,
                     relative[:, index], self.body_ids, self.weights, metadata=metadata)
                     for index, name in enumerate(BODY_FRAMES))

    def root_records(self, stamp):
        return tuple(finite_pose_record("base_link", "root_predicted/" + name, stamp,
                     self.root_transforms[:, index], self.body_ids, self.weights,
                     metadata=self.metadata) for index, name in enumerate(BODY_FRAMES))


def finite_pose_record(parent, child, stamp, poses, body_ids, weights,
                       method="joint_transform_pushforward", metadata=None):
    """Serialize a derived numerical query for display, with source references."""
    metadata = dict(metadata or {}, source_observation_stamp=stamp)
    metadata.setdefault("result_kind", "joint_finite_pushforward")
    metadata.setdefault("source_law_id", "explicit_finite_body_law")
    metadata.setdefault("source_revision", 1)
    metadata.setdefault("query_id", child)
    metadata.setdefault("order", None)
    metadata.setdefault("integration_method", "finite_sum")
    detail = json.dumps(dict(metadata, representative="weighted_translation_and_quaternion_moments"), sort_keys=True)
    source = metadata["source_law_id"]
    is_finite = metadata["integration_method"] == "finite_sum"
    approximation = ApproximationInfo(kind=ApproximationKind.EXACT if is_finite else ApproximationKind.MONTE_CARLO,
                                     lossy=not is_finite, detail=detail, source=source)
    components = tuple(TransformComponent(name, float(weight),
        BinghamOrientation.dirac(rotmat_to_quat(pose[:3, :3])),
        ConditionalGaussianTranslation(pose[:3, 3], np.zeros((3, 3)), np.zeros((3, 9))),
        provenance=ComponentProvenance(source_ids=(name,), derived_from_edge_ids=(source,),
            method=method), approximation=approximation)
        for pose, name, weight in zip(poses, body_ids, weights))
    # Equal empirical weights do not identify a modal body. In particular,
    # argmax would select an arbitrary first candidate as the axes.
    # Show moments of this SAME law; neither supports nor weights are moved.
    mass = np.asarray(weights, dtype=float) / np.sum(weights)
    quaternions = np.array([component.orientation.reference_quaternion_wxyz for component in components])
    scatter = np.einsum("s,si,sj->ij", mass, quaternions, quaternions)
    _values, directions = np.linalg.eigh(scatter)
    rotation = directions[:, -1]
    if rotation[np.argmax(np.abs(rotation))] < 0:
        rotation = -rotation
    mean_translation = mass @ np.asarray(poses)[:, :3, 3]
    return TransformDistributionStamped(parent, child, stamp, child,
        "switch_press_control", TransformDistribution(components),
        representative=DeterministicTransform(mean_translation, rotation),
        representative_kind=RepresentativeKind.MOMENT_REPRESENTATIVE,
        provenance=TransformProvenance(source_ids=tuple(body_ids), derived_from_edge_ids=(source,),
            method=method, detail=detail), approximation=approximation)


def left_compose_record(source, pose, parent, child=None):
    """Exact deterministic left action, preserving the coupled probability law."""
    rotation, translation = pose[:3, :3], pose[:3, 3]
    left_quaternion = rotmat_to_quat(rotation)
    left_action = quat_left_matrix(left_quaternion)
    old_from_new_rotation = np.kron(np.eye(3), rotation.T)
    components = []
    for component in source.distribution.components:
        old = component.orientation
        orientation = BinghamOrientation(old.kind, old.inverse_concentration,
            left_action @ old.shape_matrix @ left_action.T,
            quat_mul(left_quaternion, old.reference_quaternion_wxyz))
        offset = component.translation
        components.append(replace(component, orientation=orientation,
            translation=ConditionalGaussianTranslation(rotation @ offset.mean_at_reference + translation,
                rotation @ offset.residual_covariance @ rotation.T,
                rotation @ offset.rotation_coupling @ old_from_new_rotation)))
    representative = None
    if source.representative is not None:
        representative = DeterministicTransform(rotation @ source.representative.translation + translation,
            quat_mul(left_quaternion, source.representative.rotation_wxyz))
    return replace(source, parent_frame_id=parent, child_frame_id=child or source.child_frame_id,
        distribution=TransformDistribution(tuple(components)), representative=representative)


def display_record(source, reference, stamp, child=None):
    """Frozen-law display query: placement heartbeat does not retime source law."""
    placed = (replace(source, child_frame_id=child or source.child_frame_id)
              if source.parent_frame_id == "base_link" and np.array_equal(reference, np.eye(4))
              else left_compose_record(source, reference, "base_link", child))
    try:
        metadata = json.loads(source.provenance.detail)
        if not isinstance(metadata, dict):
            metadata = {}
    except (ValueError, TypeError):
        metadata = {"source_detail": source.provenance.detail}
    metadata.update(display_result_kind="frozen_law_query", source_observation_stamp=source.stamp,
                    temporal_prediction=False)
    return replace(placed, edge_id="display_" + placed.child_frame_id,
        authority="switch_press_display",
        provenance=replace(source.provenance, method="display_of_control_distribution",
                           detail=json.dumps(metadata, sort_keys=True)))
