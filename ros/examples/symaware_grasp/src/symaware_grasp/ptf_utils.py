import numpy as np

from probtf.geometry import (
    axis_angle_to_quat as quaternion_from_axis_angle,
    quat_left_matrix as quaternion_left_matrix,
    quat_mul as quaternion_multiply_wxyz,
    quat_normalize as normalize_wxyz,
    quat_right_matrix as quaternion_right_matrix,
    rpy_to_quat as quaternion_from_rpy,
)


try:
    import quaternion
    from bingham.distribution import BinghamDistribution as BinghamDistributionImpl
except ImportError as exc:
    raise ImportError(
        "symaware_grasp requires numpy-quaternion and the pinned BinghamNLL submodule. "
        "Initialize submodules before installing the root project."
    ) from exc


def quaternion_from_rotation_matrix(rotation_matrix):
    quat = quaternion.from_rotation_matrix(np.asarray(rotation_matrix, dtype=float))
    return normalize_wxyz(quaternion.as_float_array(quat))


def rotation_matrix_from_quaternion(quat_value):
    if isinstance(quat_value, np.ndarray) or isinstance(quat_value, list) or isinstance(quat_value, tuple):
        quat_value = quaternion.as_quat_array(normalize_wxyz(quat_value))
    return quaternion.as_rotation_matrix(quat_value)


def orthonormal_columns_from_mode(mode_wxyz, preferred_tangent=None):
    mode = normalize_wxyz(mode_wxyz)
    basis = []
    if preferred_tangent is not None:
        tangent = np.asarray(preferred_tangent, dtype=float)
        tangent -= np.dot(tangent, mode) * mode
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-8:
            basis.append(tangent / tangent_norm)
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


def quaternion_from_approach_and_finger_axes(approach_axis_xyz, finger_axis_xyz):
    x_axis = np.asarray(approach_axis_xyz, dtype=float)
    z_axis = np.asarray(finger_axis_xyz, dtype=float)

    x_norm = np.linalg.norm(x_axis)
    z_norm = np.linalg.norm(z_axis)
    if x_norm == 0.0 or z_norm == 0.0:
        raise ValueError("Approach axis and finger axis must both be non-zero.")

    x_axis = x_axis / x_norm
    z_axis = z_axis / z_norm
    z_axis = z_axis - np.dot(z_axis, x_axis) * x_axis
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        raise ValueError("Approach axis and finger axis must not be parallel.")
    z_axis = z_axis / z_norm

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    return quaternion_from_rotation_matrix(rotation_matrix)


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


def axially_symmetric_bingham_matrix(mode_wxyz, symmetry_axis_body, concentrations):
    if len(concentrations) != 3:
        raise ValueError("Expected three concentration values.")
    epsilon = 1e-4
    positive = quaternion_multiply_wxyz(mode_wxyz, quaternion_from_axis_angle(symmetry_axis_body, epsilon))
    negative = quaternion_multiply_wxyz(mode_wxyz, quaternion_from_axis_angle(symmetry_axis_body, -epsilon))
    symmetry_tangent = positive - negative
    eigenvectors = orthonormal_columns_from_mode(mode_wxyz, preferred_tangent=symmetry_tangent)
    eigenvectors = np.column_stack(
        [
            eigenvectors[:, 1],
            eigenvectors[:, 2],
            eigenvectors[:, 0],
            eigenvectors[:, 3],
        ]
    )
    eigenvalues = np.array(
        [-abs(concentrations[0]), -abs(concentrations[1]), -abs(concentrations[2]), 0.0],
        dtype=float,
    )
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def symmetric_matrix_from_flat(values, size):
    matrix = np.asarray(values, dtype=float).reshape(size, size)
    return 0.5 * (matrix + matrix.T)


def regularized_inverse_covariance(covariance_matrix, epsilon=1e-6):
    covariance_matrix = symmetric_matrix_from_flat(covariance_matrix, 3)
    return np.linalg.inv(covariance_matrix + epsilon * np.eye(3, dtype=float))


def quaternion_array_to_wxyz(quat_value):
    return normalize_wxyz(quaternion.as_float_array(quat_value))


def make_bingham_distribution(matrix_values):
    if hasattr(matrix_values, "matrix"):
        matrix_values = matrix_values.matrix
    return BinghamDistributionImpl(A=symmetric_matrix_from_flat(matrix_values, 4))


def pushforward_bingham_right(matrix_values, rhs_wxyz):
    matrix = symmetric_matrix_from_flat(matrix_values, 4)
    right_matrix = quaternion_right_matrix(rhs_wxyz)
    return right_matrix @ matrix @ right_matrix.T


def pack_rgb(r, g, b):
    return (int(r) << 16) | (int(g) << 8) | int(b)
