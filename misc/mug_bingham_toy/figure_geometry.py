"""Deterministic, display-only geometry for the learned mug Bingham figure.

All quaternions are active Hamilton ``wxyz`` rotations.  The points returned by
``half_height_quaternions`` lie on a density level set; they are neither random
draws nor a quadrature rule for the distribution.  No concentration parameter,
training output, or covariance is changed here.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import i0e
from scipy.spatial.transform import Rotation


HALF_HEIGHT_DROP = float(np.log(2.0))
FIGURE_ANGLES_DEG = (0, 60, 90, 120, 180)
SPHERICAL_DENSITY_PROVENANCE = {
    "formula_source": "/home/leus/dthesis/chapter/4-transform-law-propagation.tex",
    "formula_source_sha256": "6e78e24a1e2ed629297ab88bfbbc45d9536e95d53f9ea1027165b88d0156bc04",
    "formula_labels": ["jmaa:eq:Lglobal", "jmaa:eq:fiber-bessel-density", "jmaa:eq:globaldensity"],
    "independent_existing_check": "/home/leus/dthesis/scripts/verify_ch4_math.py",
    "independent_existing_check_sha256": "8cb4602fc6826a5b0e60ed8b20d7f4b48b0fdef933a92739a8a676a991bca78e",
    "stable_bessel_source": "/home/leus/dthesis/references/jmaa_bingham_3dvec/v0.5.0/arxiv/numerical_examples/utils/induced_density.py",
    "stable_bessel_source_sha256": "7a4e72cadd91326d1d6bef6c9eb584da557bee67b4c02b5a935bbf50ee7af6ea",
}


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


def quaternion_right_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return R(p) such that R(p) @ q is the Hamilton product q * p."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape[-1:] != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("Hamilton quaternions must be finite with last axis 4")
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack((w, -x, -y, -z,
                     x, w, z, -y,
                     y, -z, w, x,
                     z, y, -x, w), axis=-1).reshape(
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
    fixed cylinder midpoint are not part of this rotation-only helper.
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


def triangulated_unit_sphere(subdivisions: int = 5) -> dict:
    """A closed icosphere covering both camera-facing and far hemispheres.

    Camera coordinates are OpenCV x right, y image-down, z forward.  Thus a
    viewer looking along camera +z sees the negative-z surface first.  The
    returned camera_xy is (x,y) with no frame rotation or sign change: a plot
    using these coordinates must put increasing y downward.  All faces are
    retained, with front_faces and face_depth supplied for display styling.

    areas are exact spherical triangle areas; centers are normalized face
    centers at which an analytic density can be evaluated.  The mesh is a
    deterministic display discretization, without stochastic samples.
    """
    if (isinstance(subdivisions, bool)
            or not isinstance(subdivisions, (int, np.integer)) or subdivisions < 0):
        raise ValueError("subdivisions must be a nonnegative integer")
    golden = (1.0 + np.sqrt(5.0))/2.0
    vertices = np.array([[-1., golden, 0.], [1., golden, 0.],
                         [-1., -golden, 0.], [1., -golden, 0.],
                         [0., -1., golden], [0., 1., golden],
                         [0., -1., -golden], [0., 1., -golden],
                         [golden, 0., -1.], [golden, 0., 1.],
                         [-golden, 0., -1.], [-golden, 0., 1.]])
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                      [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                      [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                      [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], dtype=np.int64)
    for _ in range(int(subdivisions)):
        expanded = list(vertices)
        midpoint_indices = {}

        def midpoint(a, b):
            edge = tuple(sorted((int(a), int(b))))
            if edge not in midpoint_indices:
                point = vertices[edge[0]] + vertices[edge[1]]
                point /= np.linalg.norm(point)
                midpoint_indices[edge] = len(expanded)
                expanded.append(point)
            return midpoint_indices[edge]

        refined = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        vertices, faces = np.asarray(expanded), np.asarray(refined, dtype=np.int64)
    triangles = vertices[faces]
    centers = triangles.sum(axis=1)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    determinant = np.einsum("ni,ni->n", a, np.cross(b, c))
    denominator = (1.0 + np.einsum("ni,ni->n", a, b)
                   + np.einsum("ni,ni->n", b, c) + np.einsum("ni,ni->n", c, a))
    areas = 2.0*np.arctan2(np.abs(determinant), denominator)
    return {
        "vertices": vertices,
        "faces": faces,
        "centers": centers,
        "areas": areas,
        "camera_xy": vertices[:, :2].copy(),
        "face_camera_xy": centers[:, :2].copy(),
        "face_depth": centers[:, 2].copy(),
        "front_faces": centers[:, 2] < 0.0,
        "limb_faces": (triangles[:, :, 2].min(axis=1) < 0.0)
                      & (triangles[:, :, 2].max(axis=1) > 0.0),
    }


def _unit_directions(directions: np.ndarray) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float64)
    if directions.shape[-1:] != (3,) or not np.isfinite(directions).all():
        raise ValueError("Directions must be finite with last axis three")
    norm = np.linalg.norm(directions, axis=-1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("Directions must be nonzero")
    return directions/norm


def axis_direction_log_density(matrix_camera: np.ndarray, directions: np.ndarray,
                               log_normalizer: float | None = None,
                               body_axes: np.ndarray | None = None) -> np.ndarray:
    """Exact Bingham-induced S2 log-density for the object's three axes.

    Default output columns correspond to R(q)e_x, R(q)e_y, R(q)e_z, expressed
    in the same camera frame as matrix_camera and directions.  body_axes may
    instead supply an array of input directions, one per row.

    This adapts the existing JMAA fiber/Bessel formula recorded in
    SPHERICAL_DENSITY_PROVENANCE:

        p_v(x) = pi/(2 Z(A)) exp(T) I0(sqrt(S)),
        T = (q1.T A q1 + q2.T A q2)/2,
        sqrt(S) = hypot((q1.T A q1 - q2.T A q2)/2, q1.T A q2).

    q1,q2 are an orthonormal basis of the great-circle quaternion fiber
    R(q)v=x.  Its global projector L=1/2(I-L(pure(x))R(pure(v))) supplies a
    well-conditioned basis even at x=-v.  The equivalent sum-of-squares
    Bessel argument avoids a cancellation-prone T^2+tr(AKAK)/2 evaluation;
    no density, eigenvalue, or S floor is applied.  scipy.special.i0e is the
    existing source's stable evaluation of log I0.

    Pass the existing S3 Bingham log Z(A) for an absolute density per unit S2
    area.  If omitted, only the common -log Z(A) is omitted; density ratios,
    camera-frame shape, and later display-only color normalization are exact.
    The formula uses the original full A, without tangent Gaussian, fitted
    marginal, Monte Carlo, or concentration rescaling.
    """
    matrix_camera = _matrix(matrix_camera)
    directions = _unit_directions(directions)
    shape = directions.shape[:-1]
    points = directions.reshape(-1, 3)
    axes = np.eye(3) if body_axes is None else _unit_directions(body_axes)
    if axes.ndim != 2 or axes.shape[0] == 0:
        raise ValueError("body_axes must have shape (axis_count, 3)")
    if log_normalizer is not None and not np.isfinite(log_normalizer):
        raise ValueError("log_normalizer must be finite when supplied")
    left_output = quaternion_left_matrix(np.column_stack((np.zeros(len(points)), points)))
    result = np.empty((len(points), len(axes)))
    for index, axis in enumerate(axes):
        right_input = quaternion_right_matrix(np.r_[0.0, axis])
        projector = 0.5*(np.eye(4) - left_output @ right_input)
        diagonal = np.diagonal(projector, axis1=-2, axis2=-1)
        pivot = np.argmax(diagonal, axis=1)
        q1 = projector[np.arange(len(points)), :, pivot]
        q1 /= np.linalg.norm(q1, axis=1, keepdims=True)
        q2 = q1 @ right_input.T
        h11 = np.einsum("ni,ij,nj->n", q1, matrix_camera, q1)
        h22 = np.einsum("ni,ij,nj->n", q2, matrix_camera, q2)
        h12 = np.einsum("ni,ij,nj->n", q1, matrix_camera, q2)
        mean = 0.5*(h11+h22)
        argument = np.hypot(0.5*(h11-h22), h12)
        result[:, index] = np.log(np.pi/2.0) + mean + argument + np.log(i0e(argument))
    if log_normalizer is not None:
        result -= float(log_normalizer)
    return result.reshape(shape + (len(axes),))


def camera_sphere_density_mesh(matrix_camera: np.ndarray, subdivisions: int = 5,
                               log_normalizer: float | None = None) -> dict:
    """Filled-sphere display packet from the same camera-frame Bingham A.

    Both hemispheres and all triangle faces are retained.  Raw log-densities
    are evaluated analytically at vertices and spherical face centers.  RGB
    channels are the object x/y/z marginal densities divided by each channel's
    mesh maximum, as a color mapping only.  The exact offsets used for that
    color mapping are saved separately; the raw log-densities and A are not
    modified.  Face and vertex color offsets are separate because their
    deterministic evaluation points differ.
    """
    matrix_camera = _matrix(matrix_camera)
    mesh = triangulated_unit_sphere(subdivisions)
    vertex_logs = axis_direction_log_density(matrix_camera, mesh["vertices"], log_normalizer)
    face_logs = axis_direction_log_density(matrix_camera, mesh["centers"], log_normalizer)
    vertex_offsets = vertex_logs.max(axis=0)
    face_offsets = face_logs.max(axis=0)
    mesh.update({
        "vertex_log_density": vertex_logs,
        "face_log_density": face_logs,
        "vertex_rgb": np.exp(vertex_logs-vertex_offsets),
        "face_rgb": np.exp(face_logs-face_offsets),
        "vertex_color_log_offsets": vertex_offsets,
        "face_color_log_offsets": face_offsets,
        "diagnostics": {
            "density_formula": "pi/(2 Z(A)) exp(T) I0(hypot((h11-h22)/2,h12))",
            "density_provenance": dict(SPHERICAL_DENSITY_PROVENANCE),
            "normalization": ("absolute density per unit spherical area" if log_normalizer is not None
                              else "common -log Z(A) omitted from raw log-density"),
            "log_normalizer_s3": None if log_normalizer is None else float(log_normalizer),
            "axis_order": ["object +x", "object +y", "object +z"],
            "color_channels": {"red": "object +x", "green": "object +y", "blue": "object +z"},
            "color_mapping": "each density channel divided by its mesh maximum; display only",
            "direction_frame": "OpenCV camera: x right, y image-down, z forward",
            "projection": "camera_xy=(x,y); display increasing y downward; viewer looks along +z",
            "front_hemisphere": "camera z < 0; all far-hemisphere faces also retained",
            "subdivisions": int(subdivisions),
            "vertex_count": int(len(mesh["vertices"])),
            "face_count": int(len(mesh["faces"])),
            "front_face_count": int(np.count_nonzero(mesh["front_faces"])),
            "far_face_count": int(np.count_nonzero(~mesh["front_faces"])),
            "limb_face_count": int(np.count_nonzero(mesh["limb_faces"])),
            "sphere_area": float(mesh["areas"].sum()),
            "matrix_modified": False,
            "monte_carlo": False,
        },
    })
    return mesh


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
