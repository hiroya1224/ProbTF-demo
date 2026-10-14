"""Finite upright C4 box hypotheses and checks against one fixed true box.

The native transform law keeps all orientation components. True geometry is
specified independently of a component chosen for display or IK evaluation.
The arm model, trajectory construction and IK solver are reused unchanged.
"""
from dataclasses import replace
from itertools import product
import math

import numpy as np

from probtf.distributions import TransformDistribution
from probtf.geometry import quat_mul, quat_normalize, quat_to_rotmat, rpy_to_quat
from probtf.provenance import ApproximationInfo, ApproximationKind
from symaware_grasp.arm_kinematics import ToyArm6DOF
from symaware_grasp.beliefs import make_transform_record
from symaware_grasp.geometry_utils import axially_symmetric_bingham_parameter
from symaware_grasp.grasp_collision import BottleCollisionModel
from symaware_grasp.grasp_trial import gripper_joint_position


BOX_DIMENSIONS = (0.100, 0.100, 0.180)
BOX_POSITION = (0.78, -0.12, 0.46)
BOX_YAW_DEG = 67.5
BOX_CONCENTRATION = 5000.0


def _vector(values, name, positive=False):
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(name + " must contain three finite values")
    if positive and np.any(values <= 0.0):
        raise ValueError(name + " must contain positive values")
    return values.copy()


def _rotation(values, name):
    values = np.asarray(values, dtype=float)
    if values.shape != (3, 3) or not np.all(np.isfinite(values)):
        raise ValueError(name + " must be a finite 3 by 3 matrix")
    if not np.allclose(values.T @ values, np.eye(3), rtol=0.0, atol=1e-9) or not np.isclose(np.linalg.det(values), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(name + " must be a proper rotation")
    return values.copy()


def box_symmetry_quaternions(reference_wxyz):
    """Return the four upright body-z quarter turns of a square box, in wxyz.

    Each mode keeps z upward and points its x axis toward one side normal.
    The quaternion modes are distinct rotations; they need not be orthogonal.
    """
    reference = quat_normalize(reference_wxyz)
    rotation = _rotation(quat_to_rotmat(reference), "reference rotation")
    if not np.allclose(rotation[:, 2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-9):
        raise ValueError("box reference z axis must point upward")
    return np.asarray([quat_mul(reference, rpy_to_quat(0.0, 0.0, index * math.pi / 2.0))
                       for index in range(4)], dtype=float)


def box_mode_quaternions(yaw_degrees=BOX_YAW_DEG):
    yaw_degrees = float(yaw_degrees)
    if not math.isfinite(yaw_degrees):
        raise ValueError("yaw_degrees must be finite")
    return box_symmetry_quaternions(rpy_to_quat(0.0, 0.0, math.radians(yaw_degrees)))


def box_corners(position, rotation, dimensions=BOX_DIMENSIONS):
    position = _vector(position, "position")
    rotation = _rotation(rotation, "rotation")
    half_size = 0.5 * _vector(dimensions, "dimensions", positive=True)
    corners = np.asarray(tuple(product((-1.0, 1.0), repeat=3))) * half_size
    return corners @ rotation.T + position


def make_box_record(
    distribution="mixture", position=BOX_POSITION, position_covariance=None,
    mode_rpy_deg=(0.0, 0.0, BOX_YAW_DEG), dimensions=BOX_DIMENSIONS,
    concentration=BOX_CONCENTRATION, parent_frame_id="base_link",
    child_frame_id="demo_box", stamp=0.0, edge_id="symaware_box_pose",
    authority="box_pose_node",
):
    """Construct one native law, with four equal finite components or one mode.

    The box has a square base, a greater height and an upward z axis. All
    modes share its center and preserve its physical corners; the complete
    mixture remains independent of the fixed geometry used for rendering.
    """
    if distribution not in ("mixture", "single_pose"):
        raise ValueError("distribution must be mixture or single_pose")
    position = _vector(position, "position")
    dimensions = _vector(dimensions, "dimensions", positive=True)
    if not np.isclose(dimensions[0], dimensions[1], rtol=0.0, atol=1e-12):
        raise ValueError("box orientation law requires a square base")
    if dimensions[2] <= max(dimensions[:2]):
        raise ValueError("box orientation law requires a height greater than its base side")
    rpy = _vector(mode_rpy_deg, "mode_rpy_deg")
    concentration = float(concentration)
    if not math.isfinite(concentration) or concentration <= 0.0:
        raise ValueError("concentration must be finite and positive")
    covariance = np.zeros((3, 3)) if position_covariance is None else np.asarray(position_covariance, dtype=float).reshape(3, 3)
    reference = rpy_to_quat(*(math.radians(value) for value in rpy))
    modes = box_symmetry_quaternions(reference)
    if distribution == "single_pose":
        modes = modes[:1]
    approximation = ApproximationInfo(
        kind=ApproximationKind.PRODUCER_SUPPLIED,
        detail="Configured finite upright box orientation candidates; C4 body-z quarter turns preserve the fixed square box dimensions {} m.".format(tuple(dimensions)),
        source="box_pose_node",
    )
    records = []
    for index, mode in enumerate(modes):
        records.append(make_transform_record(
            parent_frame_id=parent_frame_id, child_frame_id=child_frame_id,
            stamp=stamp, edge_id=edge_id, authority=authority,
            position_mean=position, position_covariance=covariance,
            orientation_parameter=axially_symmetric_bingham_parameter(mode, (0.0, 0.0, 1.0), (concentration,) * 3),
            orientation_reference_wxyz=mode,
            source_id="configured_box_pose", component_id="{}:C4:{}".format(child_frame_id, index),
            approximation=approximation,
        ))
    first = records[0]
    components = tuple(replace(record.distribution.components[0], raw_weight=1.0 / len(records)) for record in records)
    return replace(
        first, distribution=TransformDistribution(components),
        provenance=replace(first.provenance, method="configured_box_" + distribution),
    )


def box_box_gap(center_a, rotation_a, half_size_a, center_b, rotation_b, half_size_b):
    """Certified separating-axis lower bound for two oriented boxes, in metres.

    Fifteen possible separating axes are used: three axes of each box and
    their nine cross products. A positive gap certifies separation. Contact
    and overlap return a nonpositive value; this is not penetration depth.
    """
    center_a = _vector(center_a, "center_a")
    center_b = _vector(center_b, "center_b")
    rotation_a = _rotation(rotation_a, "rotation_a")
    rotation_b = _rotation(rotation_b, "rotation_b")
    half_size_a = _vector(half_size_a, "half_size_a", positive=True)
    half_size_b = _vector(half_size_b, "half_size_b", positive=True)
    cross_axes = np.cross(rotation_a.T[:, None, :], rotation_b.T[None, :, :]).reshape(9, 3)
    axes = np.vstack((rotation_a.T, rotation_b.T, cross_axes))
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1e-12] / norms[norms > 1e-12, None]
    extent_a = np.abs(axes @ rotation_a) @ half_size_a
    extent_b = np.abs(axes @ rotation_b) @ half_size_b
    gaps = np.abs(axes @ (center_b - center_a)) - extent_a - extent_b
    gap = float(np.max(gaps))
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, float(np.linalg.norm(center_b - center_a)), float(np.max(half_size_a + half_size_b)))
    return 0.0 if abs(gap) <= tolerance else gap


class BoxCollisionModel(BottleCollisionModel):
    """Reuse the arm's URDF bodies, FK and path sampling for a fixed true box.

    ``axis`` is accepted solely for compatibility with inherited validate_path;
    the full fixed object_rotation determines the box geometry.
    """
    def __init__(self, urdf, dimensions=BOX_DIMENSIONS, object_rotation=None):
        super().__init__(urdf)
        self.dimensions = _vector(dimensions, "dimensions", positive=True)
        reference = box_mode_quaternions()[0]
        self.object_rotation = _rotation(quat_to_rotmat(reference) if object_rotation is None else object_rotation, "object_rotation")
        self.dimensions.setflags(write=False)
        self.object_rotation.setflags(write=False)

    def clearance(self, theta, opening, position, axis=None):
        transforms = self.transforms(theta, opening)
        position = _vector(position, "position")
        smallest, pair = float("inf"), None
        for link, name, local, half_size in self.bodies:
            body = transforms[link] @ local
            gap = box_box_gap(body[:3, 3], body[:3, :3], half_size,
                              position, self.object_rotation, self.dimensions / 2.0)
            if gap < smallest:
                smallest, pair = gap, "{}:{} / box".format(link, name)
        return smallest, pair


def evaluate_box_grasp(theta, opening, position, mode_quaternions, robot=None, dimensions=BOX_DIMENSIONS, object_rotation=None):
    """Geometry and orientation diagnostics, independent of the IK objective.

    Mode error is the smallest physical quaternion angle over the supplied
    grasp modes. Jaw gap uses the fixed true box's support width along the
    hand's opening axis, and does not imply contact or successful holding.
    """
    robot = robot or ToyArm6DOF()
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (robot.dof,) or not np.all(np.isfinite(theta)) or not robot.within_limits(theta):
        raise ValueError("invalid arm joint state")
    gripper_joint_position(opening)
    position = _vector(position, "position")
    modes = np.asarray(mode_quaternions, dtype=float)
    if modes.ndim != 2 or modes.shape[1] != 4 or not len(modes):
        raise ValueError("mode_quaternions must contain one or more wxyz quaternions")
    modes = np.asarray([quat_normalize(mode) for mode in modes])
    hand_position, hand_quaternion, _ = robot.forward_kinematics(theta)
    rotation = _rotation(quat_to_rotmat(modes[0]) if object_rotation is None else object_rotation, "object_rotation")
    hand_rotation = quat_to_rotmat(hand_quaternion)
    width = float(np.abs(hand_rotation[:, 1] @ rotation) @ _vector(dimensions, "dimensions", positive=True))
    angles = 2.0 * np.arccos(np.clip(np.abs(modes @ hand_quaternion), 0.0, 1.0))
    return {
        "tcp_x": float(hand_position[0]), "tcp_y": float(hand_position[1]), "tcp_z": float(hand_position[2]),
        "tcp_object_distance_mm": 1000.0 * float(np.linalg.norm(hand_position - position)),
        "nearest_mode_error_deg": math.degrees(float(np.min(angles))),
        "reference_mode_error_deg": math.degrees(float(angles[0])),
        "wrist_bend_deg": math.degrees(abs(float(theta[4]))),
        "elbow_flexion_deg": math.degrees(abs(float(theta[2]))),
        "gripper_opening_mm": 1000.0 * float(opening),
        "jaw_axis_box_width_mm": 1000.0 * width,
        "jaw_gap_mm": 1000.0 * (float(opening) - width),
    }
