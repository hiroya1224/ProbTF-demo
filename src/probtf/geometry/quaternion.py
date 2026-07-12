"""Quaternion operations using the internal ``[w, x, y, z]`` convention."""

import math

import numpy as np


def normalize_vec(vec, eps=1e-12):
    vector = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < eps:
        raise ValueError("Vector norm is too small to normalize.")
    return vector / norm


def quat_normalize(q, eps=1e-12):
    quaternion = np.asarray(q, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("Quaternion must be a finite vector with shape (4,).")
    norm = float(np.linalg.norm(quaternion))
    if norm < eps:
        raise ValueError("Quaternion norm is too small to normalize.")
    return quaternion / norm


def quat_conj(q):
    quaternion = quat_normalize(q)
    return np.array(
        [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]],
        dtype=float,
    )


def quat_mul(q1, q2):
    w1, x1, y1, z1 = quat_normalize(q1)
    w2, x2, y2, z2 = quat_normalize(q2)
    return quat_normalize(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_left_matrix(q, normalize_input=True):
    quaternion = quat_normalize(q) if normalize_input else np.asarray(q, dtype=float)
    w_value, x_value, y_value, z_value = quaternion
    return np.array(
        [
            [w_value, -x_value, -y_value, -z_value],
            [x_value, w_value, -z_value, y_value],
            [y_value, z_value, w_value, -x_value],
            [z_value, -y_value, x_value, w_value],
        ],
        dtype=float,
    )


def quat_right_matrix(q, normalize_input=True):
    quaternion = quat_normalize(q) if normalize_input else np.asarray(q, dtype=float)
    w_value, x_value, y_value, z_value = quaternion
    return np.array(
        [
            [w_value, -x_value, -y_value, -z_value],
            [x_value, w_value, z_value, -y_value],
            [y_value, -z_value, w_value, x_value],
            [z_value, y_value, -x_value, w_value],
        ],
        dtype=float,
    )


def quat_to_rotmat(q):
    w_value, x_value, y_value, z_value = quat_normalize(q)
    return np.array(
        [
            [
                w_value * w_value + x_value * x_value - y_value * y_value - z_value * z_value,
                2.0 * (x_value * y_value - w_value * z_value),
                2.0 * (x_value * z_value + w_value * y_value),
            ],
            [
                2.0 * (x_value * y_value + w_value * z_value),
                w_value * w_value - x_value * x_value + y_value * y_value - z_value * z_value,
                2.0 * (y_value * z_value - w_value * x_value),
            ],
            [
                2.0 * (x_value * z_value - w_value * y_value),
                2.0 * (y_value * z_value + w_value * x_value),
                w_value * w_value - x_value * x_value - y_value * y_value + z_value * z_value,
            ],
        ],
        dtype=float,
    )


def rotmat_to_quat(rotation):
    """Return a normalized ``wxyz`` quaternion for a proper rotation matrix."""

    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix.")
    if not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0.0, atol=1e-8):
        raise ValueError("rotation must be orthogonal.")
    if not np.isclose(np.linalg.det(matrix), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("rotation must have determinant one.")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion = quat_normalize(quaternion)
    pivot = int(np.argmax(np.abs(quaternion)))
    return -quaternion if quaternion[pivot] < 0.0 else quaternion


def axis_angle_to_quat(axis, angle):
    axis_vector = normalize_vec(axis)
    half_angle = 0.5 * float(angle)
    return quat_normalize(
        [
            math.cos(half_angle),
            axis_vector[0] * math.sin(half_angle),
            axis_vector[1] * math.sin(half_angle),
            axis_vector[2] * math.sin(half_angle),
        ]
    )


def rpy_to_quat(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return quat_normalize(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )

