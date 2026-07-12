import math

import numpy as np


def normalize_vec(vec, eps=1e-12):
    vector = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        raise ValueError("Vector norm is too small to normalize.")
    return vector / norm


def quat_normalize(q, eps=1e-12):
    quaternion = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm < eps:
        raise ValueError("Quaternion norm is too small to normalize.")
    return quaternion / norm


def quat_conj(q):
    quaternion = quat_normalize(q)
    return np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]], dtype=float)


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


def complete_orthonormal_basis(first_vec):
    basis_vectors = [normalize_vec(first_vec)]
    dimension = basis_vectors[0].shape[0]
    for candidate in np.eye(dimension, dtype=float):
        work = candidate.copy()
        for vector in basis_vectors:
            work -= float(np.dot(work, vector)) * vector
        norm = float(np.linalg.norm(work))
        if norm > 1e-8:
            basis_vectors.append(work / norm)
        if len(basis_vectors) == dimension:
            break
    if len(basis_vectors) != dimension:
        raise ValueError("Could not construct an orthonormal basis.")
    return np.column_stack(basis_vectors)


def tangent_projector(v):
    unit_v = normalize_vec(v)
    return np.eye(unit_v.shape[0], dtype=float) - np.outer(unit_v, unit_v)


def tangent_basis(v):
    unit_v = normalize_vec(v)
    projector = tangent_projector(unit_v)
    eigenvalues, eigenvectors = np.linalg.eigh(projector)
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order[:2]]
    basis[:, 0] = normalize_vec(basis[:, 0])
    residual = basis[:, 1] - float(np.dot(basis[:, 0], basis[:, 1])) * basis[:, 0]
    basis[:, 1] = normalize_vec(residual)
    return basis


def exp_s2(v, u, eps=1e-12):
    base = normalize_vec(v)
    tangent = np.asarray(u, dtype=float)
    norm = float(np.linalg.norm(tangent))
    if norm < eps:
        return base.copy()
    return normalize_vec(math.cos(norm) * base + math.sin(norm) * tangent / norm)
