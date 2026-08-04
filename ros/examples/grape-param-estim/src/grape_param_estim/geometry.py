"""Small SO(3) helpers used by the closed-loop estimator models."""

from typing import Sequence, Tuple

import numpy as np


_SO3_SERIES_ANGLE_SQUARED = 1.0e-8
_SO3_LOG_NEAR_PI = 1.0e-3


def _finite_vector3(value: Sequence[float], *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values")
    return vector


def _finite_matrix3(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 3 by 3 matrix")
    return matrix


def _so3_coefficients(angle_squared: float) -> Tuple[float, float, float]:
    """Return stable Rodrigues and left-Jacobian scalar coefficients."""

    if angle_squared < _SO3_SERIES_ANGLE_SQUARED:
        angle_fourth = angle_squared * angle_squared
        angle_sixth = angle_fourth * angle_squared
        sine_over_angle = (
            1.0
            - angle_squared / 6.0
            + angle_fourth / 120.0
            - angle_sixth / 5040.0
        )
        one_minus_cosine_over_angle_squared = (
            0.5
            - angle_squared / 24.0
            + angle_fourth / 720.0
            - angle_sixth / 40320.0
        )
        angle_minus_sine_over_angle_cubed = (
            1.0 / 6.0
            - angle_squared / 120.0
            + angle_fourth / 5040.0
            - angle_sixth / 362880.0
        )
        return (
            sine_over_angle,
            one_minus_cosine_over_angle_squared,
            angle_minus_sine_over_angle_cubed,
        )

    angle = float(np.sqrt(angle_squared))
    return (
        float(np.sin(angle) / angle),
        float((1.0 - np.cos(angle)) / angle_squared),
        float((angle - np.sin(angle)) / (angle_squared * angle)),
    )


def _so3_inverse_jacobian_coefficient(angle_squared: float) -> float:
    if angle_squared < _SO3_SERIES_ANGLE_SQUARED:
        angle_fourth = angle_squared * angle_squared
        angle_sixth = angle_fourth * angle_squared
        return float(
            1.0 / 12.0
            + angle_squared / 720.0
            + angle_fourth / 30240.0
            + angle_sixth / 1209600.0
        )

    angle = float(np.sqrt(angle_squared))
    sine_half_angle = float(np.sin(0.5 * angle))
    if abs(sine_half_angle) <= np.finfo(float).eps * max(1.0, angle):
        raise ValueError("SO(3) Jacobian is singular at this rotation vector")
    half_angle_cotangent = (
        0.5 * angle * float(np.cos(0.5 * angle)) / sine_half_angle
    )
    return float((1.0 - half_angle_cotangent) / angle_squared)


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


def so3_exp(rotation_vector: Sequence[float]) -> np.ndarray:
    """Return the SO(3) exponential of a finite three-vector."""

    vector = _finite_vector3(rotation_vector, name="rotation_vector")
    angle_squared = float(np.dot(vector, vector))
    if not np.isfinite(angle_squared):
        raise ValueError("rotation_vector norm must be finite")
    sine_coefficient, cosine_coefficient, _ = _so3_coefficients(
        angle_squared
    )
    omega = skew(vector)
    return (
        np.eye(3)
        + sine_coefficient * omega
        + cosine_coefficient * omega @ omega
    )


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Return the principal SO(3) logarithm as a finite three-vector.

    The logarithm has norm in ``[0, pi]``.  At exactly ``pi`` the axis sign
    is inherently ambiguous; this implementation chooses a deterministic
    finite sign while preserving the approached sign away from the kink.
    """

    matrix = _finite_matrix3(rotation, name="rotation")
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
        angle_squared = angle * angle
        coefficient = (
            0.5
            + angle_squared / 12.0
            + 7.0 * angle_squared * angle_squared / 720.0
        )
        return coefficient * vee
    if np.pi - angle < _SO3_LOG_NEAR_PI:
        symmetric_rotation = 0.5 * (matrix + matrix.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric_rotation)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        axis_norm = float(np.linalg.norm(axis))
        if not np.isfinite(axis_norm) or axis_norm <= np.finfo(float).eps:
            raise ValueError("rotation does not have a finite near-pi axis")
        axis = axis / axis_norm
        alignment = float(np.dot(axis, vee))
        if alignment < 0.0:
            axis = -axis
        elif abs(alignment) <= 64.0 * np.finfo(float).eps:
            pivot = int(np.argmax(np.abs(axis)))
            if axis[pivot] < 0.0:
                axis = -axis
        return angle * axis
    return 0.5 * angle / np.sin(angle) * vee


def so3_left_jacobian(rotation_vector: Sequence[float]) -> np.ndarray:
    """Return the SO(3) left Jacobian of the exponential map.

    It follows ``Exp(phi + d) = Exp(J_l(phi) d) Exp(phi) + O(||d||^2)``.
    """

    vector = _finite_vector3(rotation_vector, name="rotation_vector")
    angle_squared = float(np.dot(vector, vector))
    if not np.isfinite(angle_squared):
        raise ValueError("rotation_vector norm must be finite")
    _, linear_coefficient, quadratic_coefficient = _so3_coefficients(
        angle_squared
    )
    omega = skew(vector)
    return (
        np.eye(3)
        + linear_coefficient * omega
        + quadratic_coefficient * omega @ omega
    )


def so3_right_jacobian(rotation_vector: Sequence[float]) -> np.ndarray:
    """Return the SO(3) right Jacobian of the exponential map.

    It follows ``Exp(phi + d) = Exp(phi) Exp(J_r(phi) d) + O(||d||^2)``.
    """

    vector = _finite_vector3(rotation_vector, name="rotation_vector")
    return so3_left_jacobian(-vector)


def so3_left_jacobian_inverse(
    rotation_vector: Sequence[float],
) -> np.ndarray:
    """Return the analytic inverse SO(3) left Jacobian."""

    vector = _finite_vector3(rotation_vector, name="rotation_vector")
    angle_squared = float(np.dot(vector, vector))
    if not np.isfinite(angle_squared):
        raise ValueError("rotation_vector norm must be finite")
    coefficient = _so3_inverse_jacobian_coefficient(angle_squared)
    omega = skew(vector)
    return np.eye(3) - 0.5 * omega + coefficient * omega @ omega


def so3_right_jacobian_inverse(
    rotation_vector: Sequence[float],
) -> np.ndarray:
    """Return the analytic inverse SO(3) right Jacobian."""

    vector = _finite_vector3(rotation_vector, name="rotation_vector")
    return so3_left_jacobian_inverse(-vector)


def right_tangent_rotation_action_jacobian(
    rotation: np.ndarray,
    vector: Sequence[float],
) -> np.ndarray:
    """Differentiate ``R Exp(delta) v`` with respect to ``delta`` at zero."""

    matrix = _finite_matrix3(rotation, name="rotation")
    action_vector = _finite_vector3(vector, name="vector")
    return -matrix @ skew(action_vector)


def _so3_geodesic_midpoint_components(
    left_rotation: np.ndarray,
    right_rotation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left = _finite_matrix3(left_rotation, name="left_rotation")
    right = _finite_matrix3(right_rotation, name="right_rotation")
    relative_rotation = left.T @ right
    relative_vector = so3_log(relative_rotation)
    half_step = so3_exp(0.5 * relative_vector)
    midpoint = left @ half_step
    return midpoint, relative_rotation, relative_vector, half_step


def so3_geodesic_midpoint(
    left_rotation: np.ndarray,
    right_rotation: np.ndarray,
) -> np.ndarray:
    """Return the principal geodesic midpoint of two rotations."""

    midpoint, _, _, _ = _so3_geodesic_midpoint_components(
        left_rotation,
        right_rotation,
    )
    return midpoint


def so3_geodesic_midpoint_with_right_jacobians(
    left_rotation: np.ndarray,
    right_rotation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a midpoint and its right-tangent endpoint Jacobian blocks.

    For endpoint perturbations ``R0 Exp(d0)`` and ``R1 Exp(d1)``, the returned
    blocks satisfy
    ``midpoint' = midpoint Exp(J0 d0 + J1 d1) + O(||d||^2)``.
    The principal-log branch determines the finite convention at ``pi``.
    """

    midpoint, _, relative_vector, half_step = (
        _so3_geodesic_midpoint_components(left_rotation, right_rotation)
    )
    half_vector = 0.5 * relative_vector
    half_right_jacobian = so3_right_jacobian(half_vector)
    relative_left_inverse = so3_left_jacobian_inverse(relative_vector)
    relative_right_inverse = so3_right_jacobian_inverse(relative_vector)
    left_block = (
        half_step.T
        - 0.5 * half_right_jacobian @ relative_left_inverse
    )
    right_block = 0.5 * half_right_jacobian @ relative_right_inverse
    return midpoint, left_block, right_block


def rotation_matrix_from_vector(rotation_vector: Sequence[float]) -> np.ndarray:
    """Backward-compatible direct wrapper for :func:`so3_exp`."""

    return so3_exp(rotation_vector)


def rotation_vector_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Backward-compatible direct wrapper for :func:`so3_log`."""

    return so3_log(rotation)


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
