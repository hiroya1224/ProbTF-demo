"""Coordinate conventions shared by sampling, rendering, and annotations.

``T_A_B`` maps coordinates in frame B to frame A.  Cameras use the OpenCV
optical convention (x right, y down, z forward).  pyrender cameras use the
OpenGL convention (x right, y up, z backward).
"""

from typing import Iterable

import numpy as np


T_GL_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def as_transform(value: Iterable[Iterable[float]]) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("a homogeneous transform must have shape (4, 4)")
    if not np.all(np.isfinite(transform)):
        raise ValueError("transform contains non-finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("invalid homogeneous transform bottom row")
    return transform


def make_transform(rotation: np.ndarray, translation: Iterable[float]) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("rotation and translation must have shapes (3,3) and (3,)")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = as_transform(transform)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3].dot(transform[:3, 3])
    return result


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = as_transform(transform)
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("points must end in dimension 3")
    return points.dot(transform[:3, :3].T) + transform[:3, 3]


def opencv_camera_pose_to_pyrender(T_W_C: np.ndarray) -> np.ndarray:
    """Return ``T_W_GL`` for a given OpenCV optical ``T_W_C`` pose."""
    return as_transform(T_W_C).dot(T_GL_CV)


def pyrender_camera_pose_to_opencv(T_W_GL: np.ndarray) -> np.ndarray:
    return as_transform(T_W_GL).dot(T_GL_CV)


def marker_corners_ippe(size_m: float) -> np.ndarray:
    """OpenCV SOLVEPNP_IPPE_SQUARE object-point order.

    The tag frame has x toward image-right, y toward image-up, and z along the
    printed front-face normal.  This order is TL, TR, BR, BL in that frame.
    """
    if size_m <= 0.0:
        raise ValueError("marker size must be positive")
    half = 0.5 * float(size_m)
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0],
         [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


def normalize(vector: Iterable[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError("cannot normalize a near-zero vector")
    return value / norm


def look_at_opencv(position_W: Iterable[float], target_W: Iterable[float],
                   down_hint_W: Iterable[float] = (0.0, 1.0, 0.0)) -> np.ndarray:
    """Build a camera-to-world pose whose optical z axis looks at a target."""
    position = np.asarray(position_W, dtype=np.float64)
    z_axis = normalize(np.asarray(target_W, dtype=np.float64) - position)
    down_hint = normalize(down_hint_W)
    x_axis = np.cross(down_hint, z_axis)
    if np.linalg.norm(x_axis) < 1e-9:
        x_axis = np.cross((0.0, 0.0, 1.0), z_axis)
    x_axis = normalize(x_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    return make_transform(np.column_stack((x_axis, y_axis, z_axis)), position)


def projected_edge_lengths(corners_px: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners_px, dtype=np.float64)
    if corners.shape != (4, 2):
        raise ValueError("corners must have shape (4, 2)")
    return np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
