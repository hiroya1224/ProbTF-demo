"""Small SO(3) helpers used by the closed-loop estimator models."""

from typing import Sequence, Tuple

import numpy as np


def skew(vector: Sequence[float]) -> np.ndarray:
    """Return the cross-product matrix of a three-vector."""

    x_value, y_value, z_value = np.asarray(vector, dtype=float)
    return np.asarray(
        (
            (0.0, -z_value, y_value),
            (z_value, 0.0, -x_value),
            (-y_value, x_value, 0.0),
        ),
        dtype=float,
    )


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""

    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def normalise_quaternion(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(float).eps:
        raise ValueError("quaternion norm must be positive")
    result = quaternion / norm
    if result[3] < 0.0:
        result = -result
    return result


def quaternion_multiply(
    left_xyzw: Sequence[float], right_xyzw: Sequence[float]
) -> np.ndarray:
    left = np.asarray(left_xyzw, dtype=float)
    right = np.asarray(right_xyzw, dtype=float)
    vector = (
        left[3] * right[:3]
        + right[3] * left[:3]
        + np.cross(left[:3], right[:3])
    )
    scalar = left[3] * right[3] - np.dot(left[:3], right[:3])
    return np.concatenate((vector, np.asarray((scalar,), dtype=float)))


def quaternion_to_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    x_value, y_value, z_value, w_value = normalise_quaternion(
        quaternion_xyzw
    )
    return np.asarray(
        (
            (
                1.0 - 2.0 * (y_value**2 + z_value**2),
                2.0 * (x_value * y_value - z_value * w_value),
                2.0 * (x_value * z_value + y_value * w_value),
            ),
            (
                2.0 * (x_value * y_value + z_value * w_value),
                1.0 - 2.0 * (x_value**2 + z_value**2),
                2.0 * (y_value * z_value - x_value * w_value),
            ),
            (
                2.0 * (x_value * z_value - y_value * w_value),
                2.0 * (y_value * z_value + x_value * w_value),
                1.0 - 2.0 * (x_value**2 + y_value**2),
            ),
        ),
        dtype=float,
    )


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to an ``xyzw`` quaternion."""

    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3 by 3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.asarray(
            (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            )
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(
                max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            )
            quaternion = np.asarray(
                (
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                )
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(
                max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            )
            quaternion = np.asarray(
                (
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                )
            )
        else:
            scale = 2.0 * np.sqrt(
                max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            )
            quaternion = np.asarray(
                (
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                )
            )
    return normalise_quaternion(quaternion)


def euler_xyz_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=float,
    )


def matrix_to_euler_xyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float)
    pitch = float(np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(np.cos(pitch)) > 1.0e-8:
        roll = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    else:
        roll = 0.0
        yaw = float(np.arctan2(-matrix[0, 1], matrix[1, 1]))
    return np.asarray((roll, pitch, yaw), dtype=float)


def rotation_matrix_from_vector(rotation_vector: Sequence[float]) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=float)
    angle = float(np.linalg.norm(vector))
    omega = skew(vector)
    if angle < 1.0e-8:
        return np.eye(3) + omega + 0.5 * omega @ omega
    return (
        np.eye(3)
        + np.sin(angle) / angle * omega
        + (1.0 - np.cos(angle)) / (angle * angle) * omega @ omega
    )


def rotation_vector_from_matrix(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    vee = np.asarray(
        (
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ),
        dtype=float,
    )
    if angle < 1.0e-7:
        return 0.5 * vee
    if np.pi - angle < 1.0e-5:
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (matrix + np.eye(3))
        )
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        if np.dot(axis, vee) < 0.0:
            axis = -axis
        return angle * axis
    return 0.5 * angle / np.sin(angle) * vee


def correction_transform_path(
    nominal_position: np.ndarray,
    nominal_orientation_xyzw: np.ndarray,
    candidate_position: np.ndarray,
    candidate_orientation_xyzw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ``T_nominal^-1 T_candidate`` along a complete path."""

    nominal_position = np.asarray(nominal_position, dtype=float)
    candidate_position = np.asarray(candidate_position, dtype=float)
    nominal_orientation = np.asarray(nominal_orientation_xyzw, dtype=float)
    candidate_orientation = np.asarray(candidate_orientation_xyzw, dtype=float)
    if (
        nominal_position.shape != candidate_position.shape
        or nominal_position.ndim != 2
        or nominal_position.shape[1] != 3
        or nominal_orientation.shape != candidate_orientation.shape
        or nominal_orientation.shape != (nominal_position.shape[0], 4)
    ):
        raise ValueError("nominal and candidate pose paths must align")
    translation = np.empty_like(nominal_position)
    rotation_vector = np.empty_like(nominal_position)
    for index in range(nominal_position.shape[0]):
        nominal_rotation = quaternion_to_matrix(nominal_orientation[index])
        candidate_rotation = quaternion_to_matrix(candidate_orientation[index])
        translation[index] = nominal_rotation.T @ (
            candidate_position[index] - nominal_position[index]
        )
        rotation_vector[index] = rotation_vector_from_matrix(
            nominal_rotation.T @ candidate_rotation
        )
    return translation, rotation_vector
