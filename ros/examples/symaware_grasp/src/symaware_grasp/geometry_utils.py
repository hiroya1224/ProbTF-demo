"""Small grasp-specific geometry helpers built on the core conventions."""

import numpy as np

from probtf.geometry import (
    axis_angle_to_quat,
    quat_mul,
    quat_normalize,
    rotmat_to_quat,
)


def quaternion_from_approach_and_finger_axes(approach_axis_xyz, finger_axis_xyz):
    x_axis = np.asarray(approach_axis_xyz, dtype=float)
    z_axis = np.asarray(finger_axis_xyz, dtype=float)
    x_norm = float(np.linalg.norm(x_axis))
    z_norm = float(np.linalg.norm(z_axis))
    if not np.isfinite(x_norm) or not np.isfinite(z_norm) or x_norm < 1e-8 or z_norm < 1e-8:
        raise ValueError("Approach and finger axes must be finite and non-zero.")
    x_axis = x_axis / x_norm
    z_axis -= float(np.dot(z_axis, x_axis)) * x_axis
    z_norm = float(np.linalg.norm(z_axis))
    if not np.isfinite(z_norm) or z_norm < 1e-8:
        raise ValueError("Approach and finger axes must be finite, non-zero, and non-parallel.")
    z_axis /= z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    return rotmat_to_quat(np.column_stack((x_axis, y_axis, z_axis)))


def axially_symmetric_bingham_parameter(mode_wxyz, symmetry_axis_body, concentrations):
    """Construct a finite Bingham parameter with a named weak symmetry tangent."""

    values = np.asarray(concentrations, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("concentrations must contain three finite values.")
    mode = quat_normalize(mode_wxyz)
    epsilon = 1e-4
    positive = quat_mul(mode, axis_angle_to_quat(symmetry_axis_body, epsilon))
    negative = quat_mul(mode, axis_angle_to_quat(symmetry_axis_body, -epsilon))
    tangent = positive - negative
    tangent -= float(np.dot(tangent, mode)) * mode
    tangent /= np.linalg.norm(tangent)

    tangent_basis = [tangent]
    for candidate in np.eye(4, dtype=float):
        work = candidate - float(np.dot(candidate, mode)) * mode
        for existing in tangent_basis:
            work -= float(np.dot(work, existing)) * existing
        norm = float(np.linalg.norm(work))
        if norm > 1e-8:
            tangent_basis.append(work / norm)
        if len(tangent_basis) == 3:
            break
    basis = np.column_stack((tangent_basis[1], tangent_basis[2], tangent_basis[0], mode))
    eigenvalues = np.array(
        [-abs(values[0]), -abs(values[1]), -abs(values[2]), 0.0],
        dtype=float,
    )
    parameter = basis @ np.diag(eigenvalues) @ basis.T
    return 0.5 * (parameter + parameter.T)
