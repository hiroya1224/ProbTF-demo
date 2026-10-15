"""Existing joint-law display and exact placement of the observed marker."""
from dataclasses import replace

import numpy as np

from probtf.distributions import (
    BinghamOrientation, ConditionalGaussianTranslation, RepresentativeKind,
    TransformComponent, TransformDistribution, TransformDistributionStamped,
)
from probtf.geometry import DeterministicTransform, quat_left_matrix, quat_mul, rotmat_to_quat
from probtf.provenance import ComponentProvenance, TransformProvenance
from symaware_grasp.probtf_config import config_from_mapping


BODY_FRAMES = tuple("link_{}".format(index) for index in range(1, 7)) + ("tool0", "rod_tip")


def pose_component(pose, name, weight):
    return TransformComponent(name, float(weight), BinghamOrientation.dirac(rotmat_to_quat(pose[:3, :3])),
        ConditionalGaussianTranslation(pose[:3, 3], np.zeros((3, 3)), np.zeros((3, 9))),
        provenance=ComponentProvenance(source_ids=(name,), method="fixed_display_geometry"))


def pose_record(parent, child, stamp, components, representative):
    return TransformDistributionStamped(parent, child, stamp, "display_" + child.replace("/", "_"),
        "switch_press_display", TransformDistribution(tuple(components)),
        representative=DeterministicTransform(representative[:3, 3], rotmat_to_quat(representative[:3, :3])),
        representative_kind=RepresentativeKind.PRODUCER_SUPPLIED,
        provenance=TransformProvenance(method="display_marginal", detail="No feedback to IK"))


def body_frame_records(mapping, pose, rod_length, stamp):
    """Follow the visible joints using symaware_grasp's existing Bingham joint laws.

    This is a configured kinematic display error model, separate from the IK's
    finite stiffness hypotheses. The representative chain is the visible arm.
    """
    configured = config_from_mapping(mapping, stamp=stamp, authority="switch_press_display",
        source_id="configured_display_joint_uncertainty", joint_positions=pose, dynamic_joints=True)
    records = [replace(record, is_static=False) for record in configured.records]
    tip = np.eye(4)
    tip[0, 3] = rod_length
    records.append(pose_record("tool0", "rod_tip", stamp, [pose_component(tip, "rigid_rod", 1.)], tip))
    return records


def observation_pose_record(preview, hand_camera, stamp):
    """Place the unchanged sensor law in base, without drawing a helper camera frame.

    The deterministic left action rotates the coupled law exactly. Its stamp
    is the render snapshot time; provenance retains the original source time.
    """
    source = preview["marker_record"]
    camera = preview["hand"] @ hand_camera
    rotation, translation = camera[:3, :3], camera[:3, 3]
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
    return replace(source, parent_frame_id="base_link", child_frame_id="observed_ar_marker",
        edge_id="display_ar_marker", stamp=stamp, distribution=TransformDistribution(tuple(components)),
        representative=representative, authority="switch_press_display",
        provenance=replace(source.provenance, method="observation_display_snapshot",
            detail="source_stamp={:.9f}; frozen observation camera; {}".format(source.stamp, source.provenance.detail)))
