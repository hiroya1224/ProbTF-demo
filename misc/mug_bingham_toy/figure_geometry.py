"""Deterministic, display-only geometry for the learned mug Bingham figure.

All quaternions are active Hamilton ``wxyz`` rotations.  The points returned by
``half_height_quaternions`` lie on a density level set; they are neither random
draws nor a quadrature rule for the distribution.  No concentration parameter,
training output, or covariance is changed here.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation


HALF_HEIGHT_DROP = float(np.log(2.0))
FIGURE_ANGLES_DEG = (0, 60, 90, 120, 180)


def _matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("A must be a finite 4 by 4 matrix")
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(matrix))))
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
        raise ValueError("A must be symmetric")
    return matrix


def _unit_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape[-1:] != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("Hamilton quaternions must be finite with last axis 4")
    length = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(length == 0.0):
        raise ValueError("A quaternion must be nonzero")
    return quaternion / length


def quaternion_left_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return L(p) such that L(p) @ q is the Hamilton product p * q.

    This algebraic map deliberately does not normalize its argument.  For a
    unit p it is orthogonal, and represents a change of destination frame.
    Batched inputs with shape ``(..., 4)`` are supported.
    """
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape[-1:] != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("Hamilton quaternions must be finite with last axis 4")
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack((w, -x, -y, -z,
                     x, w, -z, y,
                     y, z, w, -x,
                     z, -y, x, w), axis=-1).reshape(
                         quaternion.shape[:-1] + (4, 4))


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert Hamilton wxyz object-to-destination rotations to matrices."""
    quaternion = _unit_quaternion(quaternion)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack((1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
                     2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
                     2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)),
                    axis=-1).reshape(quaternion.shape[:-1] + (3, 3))


def _quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    quaternion = Rotation.from_matrix(matrix).as_quat()[..., [3, 0, 1, 2]]
    # The sign has no physical effect; choose one deterministic representation.
    dominant = np.argmax(np.abs(quaternion), axis=-1)
    sign = np.take_along_axis(quaternion, dominant[..., None], axis=-1)
    return quaternion * np.where(sign < 0.0, -1.0, 1.0)


def mode_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Return a unit maximum-density eigenvector, with deterministic sign.

    When the largest eigenvalue is repeated there is an entire modal
    eigenspace; this selects one representative without modifying the law.
    """
    _, basis = np.linalg.eigh(_matrix(matrix))
    quaternion = basis[:, -1].copy()
    if quaternion[np.argmax(np.abs(quaternion))] < 0.0:
        quaternion *= -1.0
    return quaternion


def frame_change_bingham(matrix_camera: np.ndarray,
                        rotation_world_camera: np.ndarray) -> np.ndarray:
    """Express the same Bingham law in world coordinates.

    If q_WO = q_WC * q_CO, then A_W = L(q_WC) A_C L(q_WC).T, so
    q_WO.T A_W q_WO = q_CO.T A_C q_CO.  Neither eigenvalues nor density
    concentrations are changed, and trace zero is preserved when present.
    """
    matrix_camera = _matrix(matrix_camera)
    rotation_world_camera = np.asarray(rotation_world_camera, dtype=np.float64)
    if (rotation_world_camera.shape != (3, 3)
            or not np.isfinite(rotation_world_camera).all()
            or not np.allclose(rotation_world_camera.T @ rotation_world_camera,
                               np.eye(3), rtol=0.0, atol=1e-11)
            or not np.isclose(np.linalg.det(rotation_world_camera), 1.0,
                              rtol=0.0, atol=1e-11)):
        raise ValueError("R_world_camera must be a proper rotation matrix")
    left = quaternion_left_matrix(_quaternion_from_matrix(rotation_world_camera))
    return left @ matrix_camera @ left.T


def camera_up_world_poses(angles=FIGURE_ANGLES_DEG,
                          elevation_degrees: float = 10.0) -> dict:
    """Upright mug rotations for the five reference-style figure panels.

    The native mesh axes are retained: object +Z is up and the handle extends
    toward object +X.  The camera lies in the world's negative-Y half-plane,
    elevated above the horizontal, with OpenCV x right, y down, z forward.
    The object rotation is world yaw 180 degrees minus the displayed angle:
    the handle is left at 0, behind the cup at 90, and right at 180 degrees.
    Object translation, the learned renderer's distance/intrinsics, and the
    frozen bounding-box center are not part of this rotation-only helper.
    """
    angles = np.asarray(angles, dtype=np.float64)
    if angles.ndim != 1 or not np.isfinite(angles).all():
        raise ValueError("Figure angles must be a finite one-dimensional array")
    if not np.isfinite(elevation_degrees):
        raise ValueError("Camera elevation must be finite")
    beta = np.radians(elevation_degrees)
    rotation_world_camera = np.array([[1.0, 0.0, 0.0],
                                      [0.0, -np.sin(beta), np.cos(beta)],
                                      [0.0, -np.cos(beta), -np.sin(beta)]])
    rotation_world_object = Rotation.from_euler(
        "z", 180.0 - angles, degrees=True).as_matrix()
    rotation_camera_object = rotation_world_camera.T @ rotation_world_object
    return {
        "angles_deg": angles,
        "R_world_camera": rotation_world_camera,
        "R_camera_world": rotation_world_camera.T,
        "R_world_object": rotation_world_object,
        "R_camera_object": rotation_camera_object,
        "q_world_camera_wxyz": _quaternion_from_matrix(rotation_world_camera),
        "q_world_object_wxyz": _quaternion_from_matrix(rotation_world_object),
        "q_camera_object_wxyz": _quaternion_from_matrix(rotation_camera_object),
        "elevation_degrees": float(elevation_degrees),
    }


def _fibonacci_directions(count: int) -> np.ndarray:
    # This is the reference figure's exact index/phase construction.
    index = np.arange(count) + 0.5
    z = 1 - 2 * index / len(index)
    phase = np.pi * (3 - np.sqrt(5)) * index
    radius = np.sqrt(1 - z*z)
    return np.column_stack((radius * np.cos(phase),
                            radius * np.sin(phase), z))


def half_height_quaternions(matrix: np.ndarray, count: int = 960
                             ) -> tuple[np.ndarray, dict]:
    """Deterministically represent q.T A q = lambda_max - log(2).

    For all three mode gaps greater than log(2), use exactly the reference
    figure's 960-point (by default) principal-ellipsoid/Fibonacci construction.
    For wider laws, probe great circles from a selected mode and retain only
    directions that reach half height within a physical rotation of 180 deg.
    Their omitted indices are reported, rather than changing the matrix to
    force a closed ellipsoid.  If a finite direction grid misses an existing
    narrow contour, exact principal-direction crossings provide finite points.

    A uniform law, or any law whose full log-density range is less than log(2),
    has no half-height level set and returns an empty (0,4) array.  Exactly
    critical levels at the minimum eigenspace are represented directly there.
    The diagnostics are JSON serializable.  These are display points, without
    statistical sampling or quadrature weights.
    """
    matrix = _matrix(matrix)
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("count must be a positive integer")
    count = int(count)
    eigenvalues, basis = np.linalg.eigh(matrix)
    gaps = eigenvalues[-1] - eigenvalues[:3]
    directions = _fibonacci_directions(count)
    radial_gaps = (directions**2) @ gaps
    crossing = radial_gaps >= HALF_HEIGHT_DROP
    omitted = np.flatnonzero(~crossing)
    anchors = 0
    axial_roundoff_corrections = 0

    if np.min(gaps) > HALF_HEIGHT_DROP:
        # Keep this calculation identical to half_height_rotations in
        # dthesis/scripts/render_mug_learning_figure.py.
        transverse = directions * np.sqrt(HALF_HEIGHT_DROP / gaps)
        axial_squared = 1 - np.sum(transverse**2, axis=1)
        negative_roundoff = axial_squared < 0.0
        axial_roundoff_corrections = int(np.count_nonzero(negative_roundoff))
        if axial_roundoff_corrections:
            # With gaps only one floating-point step above log(2), the
            # subtractive expression can round below zero.  Evaluate the
            # same nonnegative quantity as a sum of principal-axis terms;
            # no eigenvalue or density level is replaced or bounded.
            axial_squared[negative_roundoff] = (
                directions[negative_roundoff]**2) @ ((gaps-HALF_HEIGHT_DROP)/gaps)
        axial = np.sqrt(axial_squared)
        quaternions = transverse @ basis[:, :3].T + axial[:, None] * basis[:, -1]
        method = "principal_ellipsoid_fibonacci"
    elif np.max(gaps) < HALF_HEIGHT_DROP:
        quaternions = np.empty((0, 4), dtype=np.float64)
        method = "no_half_height_level_set"
    elif np.max(gaps) == HALF_HEIGHT_DROP:
        # At the exact limiting level only the minimum eigenspace remains.
        indices = np.flatnonzero(gaps == HALF_HEIGHT_DROP)
        dimension = len(indices)
        if dimension == 1:
            coordinates = np.ones((1, 1))
        elif dimension == 2:
            phase = 2*np.pi*(np.arange(count) + 0.5)/count
            coordinates = np.column_stack((np.cos(phase), np.sin(phase)))
        else:
            coordinates = directions
        quaternions = coordinates @ basis[:, indices].T
        method = "critical_minimum_eigenspace"
    else:
        selected = directions[crossing]
        selected_gaps = radial_gaps[crossing]
        if len(selected) == 0:
            # This is a geometric coverage fallback, not a quality condition
            # on the inferred law.  The six axes are deterministic probes.
            indices = np.flatnonzero(gaps >= HALF_HEIGHT_DROP)
            selected = np.concatenate((np.eye(3)[indices], -np.eye(3)[indices]))
            selected_gaps = np.tile(gaps[indices], 2)
            anchors = len(selected)
        radial = np.sqrt(HALF_HEIGHT_DROP / selected_gaps)
        transverse = selected * radial[:, None]
        axial = np.sqrt(1 - radial**2)
        quaternions = transverse @ basis[:, :3].T + axial[:, None] * basis[:, -1]
        method = "great_circle_fibonacci_crossings"

    density = np.einsum("ni,ij,nj->n", quaternions, matrix, quaternions)
    diagnostics = {
        "method": method,
        "point_interpretation": "deterministic half-height level set; not random samples or quadrature",
        "quaternion_order": "Hamilton wxyz",
        "count_requested": count,
        "point_count": int(len(quaternions)),
        "direction_count": count,
        "crossing_direction_count": int(np.count_nonzero(crossing)),
        "non_crossing_direction_count": int(len(omitted)),
        "non_crossing_direction_indices": omitted.tolist(),
        "principal_crossing_anchor_count": int(anchors),
        "axial_roundoff_correction_count": axial_roundoff_corrections,
        "eigenvalues": eigenvalues.tolist(),
        "mode_gaps": gaps.tolist(),
        "peak_log_density_unnormalized": float(eigenvalues[-1]),
        "half_height_log_density_unnormalized": float(eigenvalues[-1] - HALF_HEIGHT_DROP),
        "target_log_density_drop": -HALF_HEIGHT_DROP,
        "level_set_exists": bool(np.max(gaps) >= HALF_HEIGHT_DROP),
        "uniform_law": bool(np.max(gaps) == 0.0),
        "max_unit_norm_error": (float(np.max(np.abs(np.linalg.norm(quaternions, axis=1)-1)))
                                if len(quaternions) else None),
        "max_log_density_drop_residual": (
            float(np.max(np.abs(density - eigenvalues[-1] + HALF_HEIGHT_DROP)))
            if len(quaternions) else None),
    }
    return quaternions, diagnostics


def yaw_half_width(matrix: np.ndarray, mode: np.ndarray | None = None,
                   axis=(0.0, 0.0, 1.0)) -> dict:
    """Half-height one-sided physical angle about a fixed destination axis.

    The path is left multiplication by (cos(theta/2), axis*sin(theta/2)).
    For an exact mode its log-density drop is gap*sin(theta/2)^2, reproducing
    the reference figure's 2*asin(sqrt(log(2)/gap)) result.  If the path never
    reaches half height on [0,180] degrees, width_deg is None.  The tiny mode
    cross term is retained in the root equation rather than changing A.
    """
    matrix = _matrix(matrix)
    quaternion = mode_quaternion(matrix) if mode is None else _unit_quaternion(mode)
    if quaternion.shape != (4,):
        raise ValueError("mode must be one Hamilton quaternion")
    axis = np.asarray(axis, dtype=np.float64)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0.0:
        raise ValueError("axis must be a finite nonzero three-vector")
    axis = axis / np.linalg.norm(axis)
    tangent = quaternion_left_matrix(np.r_[0.0, axis]) @ quaternion
    peak = float(quaternion @ matrix @ quaternion)
    gap = float(peak - tangent @ matrix @ tangent)
    cross = float(quaternion @ matrix @ tangent)

    def residual(theta):
        if theta == 0.0:
            return HALF_HEIGHT_DROP
        if theta == np.pi:
            return HALF_HEIGHT_DROP - gap
        half = theta / 2.0
        return float(HALF_HEIGHT_DROP - gap*np.sin(half)**2
                     + cross*np.sin(theta))

    # The quadratic restricted to this circle has at most one interior
    # minimum over a physical half-turn.  Checking that extremum also handles
    # a supplied approximate mode without a scan or stochastic approximation.
    probes = [0.0, float(np.pi)]
    stationary = float(np.arctan2(2.0*cross, gap))
    for multiple in (-1, 0, 1, 2):
        candidate = stationary + multiple*np.pi
        if 0.0 < candidate < np.pi:
            probes.append(candidate)
    probes.sort()
    angle = None
    for start, stop in zip(probes[:-1], probes[1:]):
        if residual(stop) <= 0.0:
            angle = (stop if residual(stop) == 0.0 else
                     float(brentq(residual, start, stop,
                                  xtol=5e-15, rtol=4*np.finfo(float).eps)))
            break
    return {
        "width_deg": None if angle is None else float(np.degrees(angle)),
        "crosses_half_height_within_180_deg": angle is not None,
        "gap": gap,
        "mode_cross_term": cross,
        "axis": axis.tolist(),
        "peak_log_density_unnormalized": peak,
        "target_log_density_drop": -HALF_HEIGHT_DROP,
        "log_density_drop_residual": None if angle is None else residual(angle),
        "mode_eigenvector_residual": float(np.linalg.norm(matrix @ quaternion - peak*quaternion)),
    }
