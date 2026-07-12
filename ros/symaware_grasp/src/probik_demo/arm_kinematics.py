import math

import numpy as np

from probik_demo.ptf_utils import quaternion_from_rotation_matrix, normalize_wxyz


def _translation_matrix(translation_xyz):
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = np.asarray(translation_xyz, dtype=float)
    return transform


def _rotation_matrix_from_axis_angle(axis_xyz, angle_rad):
    axis = np.asarray(axis_xyz, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0.0:
        raise ValueError("Joint axis must be non-zero.")
    axis = axis / axis_norm
    x_value, y_value, z_value = axis
    skew = np.array(
        [
            [0.0, -z_value, y_value],
            [z_value, 0.0, -x_value],
            [-y_value, x_value, 0.0],
        ],
        dtype=float,
    )
    return (
        np.eye(3, dtype=float)
        + math.sin(angle_rad) * skew
        + (1.0 - math.cos(angle_rad)) * (skew @ skew)
    )


def _rotation_transform(axis_xyz, angle_rad):
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _rotation_matrix_from_axis_angle(axis_xyz, angle_rad)
    return transform


class ToyArm6DOF:
    def __init__(self):
        self.joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]
        self.lower_limits = np.array(
            [-math.pi, -2.3562, -2.7925, -math.pi, -2.3562, -math.pi],
            dtype=float,
        )
        self.upper_limits = np.array(
            [math.pi, 2.3562, 2.7925, math.pi, 2.3562, math.pi],
            dtype=float,
        )
        self._joint_origins = [
            np.array([0.0, 0.0, 0.12], dtype=float),
            np.array([0.0, 0.0, 0.18], dtype=float),
            np.array([0.36, 0.0, 0.0], dtype=float),
            np.array([0.30, 0.0, 0.0], dtype=float),
            np.array([0.14, 0.0, 0.0], dtype=float),
            np.array([0.12, 0.0, 0.0], dtype=float),
        ]
        self._joint_axes = [
            np.array([0.0, 0.0, 1.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([1.0, 0.0, 0.0], dtype=float),
        ]
        self._tool_offset = np.array([0.10, 0.0, 0.0], dtype=float)

    @property
    def dof(self):
        return len(self.joint_names)

    def clip_to_limits(self, joint_positions):
        joint_positions = np.asarray(joint_positions, dtype=float)
        return np.clip(joint_positions, self.lower_limits, self.upper_limits)

    def within_limits(self, joint_positions):
        joint_positions = np.asarray(joint_positions, dtype=float)
        return np.all(joint_positions >= self.lower_limits) and np.all(joint_positions <= self.upper_limits)

    def forward_transform(self, joint_positions):
        joint_positions = self.clip_to_limits(joint_positions)
        transform = np.eye(4, dtype=float)
        for origin_xyz, axis_xyz, joint_value in zip(
            self._joint_origins, self._joint_axes, joint_positions
        ):
            transform = transform @ _translation_matrix(origin_xyz)
            transform = transform @ _rotation_transform(axis_xyz, joint_value)
        transform = transform @ _translation_matrix(self._tool_offset)
        return transform

    def forward_kinematics(self, joint_positions):
        transform = self.forward_transform(joint_positions)
        position = transform[:3, 3].copy()
        quaternion_wxyz = normalize_wxyz(quaternion_from_rotation_matrix(transform[:3, :3]))
        return position, quaternion_wxyz, transform

    def joint_limit_cost(self, joint_positions, barrier_scale=0.02):
        joint_positions = np.asarray(joint_positions, dtype=float)
        if not self.within_limits(joint_positions):
            return float("inf")
        center = 0.5 * (self.lower_limits + self.upper_limits)
        half_range = 0.5 * (self.upper_limits - self.lower_limits)
        normalized = (joint_positions - center) / np.maximum(half_range, 1e-6)
        return float(barrier_scale * np.sum(1.0 / np.maximum(1.0 - normalized**2, 1e-4) - 1.0))

    def heuristic_seed(self, target_position_xyz):
        target = np.asarray(target_position_xyz, dtype=float)
        base_yaw = math.atan2(target[1], target[0])

        shoulder_height = 0.12 + 0.18
        planar_x = math.hypot(target[0], target[1]) - 0.22
        planar_z = target[2] - shoulder_height
        upper_length = 0.36
        fore_length = 0.30 + 0.14 + 0.12 + 0.10
        distance = math.hypot(planar_x, planar_z)
        distance = min(max(distance, 1e-6), upper_length + fore_length - 1e-6)

        cos_elbow = (distance**2 - upper_length**2 - fore_length**2) / (2.0 * upper_length * fore_length)
        cos_elbow = min(max(cos_elbow, -1.0), 1.0)
        elbow = math.acos(cos_elbow) - math.pi

        shoulder = math.atan2(planar_z, planar_x)
        shoulder -= math.atan2(
            fore_length * math.sin(math.acos(cos_elbow)),
            upper_length + fore_length * cos_elbow,
        )

        seed = np.array([base_yaw, shoulder, -elbow, 0.0, 0.0, 0.0], dtype=float)
        return self.clip_to_limits(seed)
