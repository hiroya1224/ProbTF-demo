import math
import os
import sys

import numpy as np


DEFAULT_BINGHAM_SOURCE_DIR = os.environ.get("BINGHAM_SOURCE_DIR", "/home/leus/BinghamNLL/src")
if os.path.isdir(DEFAULT_BINGHAM_SOURCE_DIR) and DEFAULT_BINGHAM_SOURCE_DIR not in sys.path:
    sys.path.insert(0, DEFAULT_BINGHAM_SOURCE_DIR)

try:
    import quaternion
    from bingham.distribution import BinghamDistribution
except ImportError as exc:
    raise ImportError(
        "probik_demo requires numpy-quaternion and the local BinghamNLL source. "
        "Set BINGHAM_SOURCE_DIR or add /home/leus/BinghamNLL/src to PYTHONPATH."
    ) from exc


def normalize_wxyz(quat_wxyz):
    quat = np.asarray(quat_wxyz, dtype=float)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("Quaternion must be non-zero.")
    return quat / norm


def quaternion_from_rpy(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return normalize_wxyz(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def orthonormal_columns_from_mode(mode_wxyz):
    mode = normalize_wxyz(mode_wxyz)
    basis = []
    for candidate in np.eye(4):
        work = candidate.copy()
        work -= np.dot(work, mode) * mode
        for vector in basis:
            work -= np.dot(work, vector) * vector
        norm = np.linalg.norm(work)
        if norm > 1e-8:
            basis.append(work / norm)
        if len(basis) == 3:
            break
    if len(basis) != 3:
        raise ValueError("Could not create a full quaternion basis from the requested mode.")
    basis.append(mode)
    return np.column_stack(basis)


def demo_bingham_matrix(mode_wxyz, concentrations):
    if len(concentrations) != 3:
        raise ValueError("Expected three concentration values.")
    mode = normalize_wxyz(mode_wxyz)
    eigenvectors = orthonormal_columns_from_mode(mode)
    eigenvalues = np.array(
        [-abs(concentrations[0]), -abs(concentrations[1]), -abs(concentrations[2]), 0.0],
        dtype=float,
    )
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def symmetric_matrix_from_flat(values, size):
    matrix = np.asarray(values, dtype=float).reshape(size, size)
    return 0.5 * (matrix + matrix.T)


def vector3_from_msg(msg):
    return np.array([msg.x, msg.y, msg.z], dtype=float)


def quaternion_msg_fields_from_wxyz(quat_wxyz):
    quat = normalize_wxyz(quat_wxyz)
    return {
        "w": quat[0],
        "x": quat[1],
        "y": quat[2],
        "z": quat[3],
    }


def quaternion_array_to_wxyz(quat_value):
    return quaternion.as_float_array(quat_value)


def rotation_matrix_from_quaternion(quat_value):
    return quaternion.as_rotation_matrix(quat_value)


def make_bingham_distribution(matrix_values):
    return BinghamDistribution(A=symmetric_matrix_from_flat(matrix_values, 4))


def pack_rgb(r, g, b):
    return (int(r) << 16) | (int(g) << 8) | int(b)
