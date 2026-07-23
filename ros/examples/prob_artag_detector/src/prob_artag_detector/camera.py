"""Projection and local right-perturbation Jacobians."""

import cv2
import numpy as np

from probtf.geometry import skew

from prob_artag_detector.models import CameraModel


def ippe_square_object_points(tag_size_m):
    """Return the strict object-point order required by IPPE_SQUARE.

    The corresponding image order is top-left, top-right, bottom-right,
    bottom-left as returned by OpenCV's ArUco detector.
    """

    size = float(tag_size_m)
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("tag_size_m must be positive and finite.")
    half = 0.5 * size
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def _rotation_matrix(rotation):
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix.")
    return matrix


def _translation_vector(translation):
    vector = np.asarray(translation, dtype=float).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("translation must be a finite 3-vector.")
    return vector


def transform_points(object_points, rotation, translation):
    points = np.asarray(object_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("object_points must have shape (N,3) and be finite.")
    return (_rotation_matrix(rotation) @ points.T).T + _translation_vector(translation)


def project_points(object_points, rotation, translation, camera_model):
    if not isinstance(camera_model, CameraModel):
        raise TypeError("camera_model must be CameraModel.")
    rotation = _rotation_matrix(rotation)
    translation = _translation_vector(translation)
    rotation_vector, _ = cv2.Rodrigues(rotation)
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float64),
        rotation_vector,
        translation.reshape(3, 1),
        camera_model.camera_matrix,
        camera_model.distortion,
    )
    return projected.reshape(-1, 2)


def analytic_pinhole_pose_jacobian(object_points, rotation, translation, camera_model):
    """Jacobian of stacked pixels wrt ``[delta_x, right_rotation_u]``."""

    if camera_model.has_distortion:
        raise ValueError("analytic pinhole Jacobian is invalid for nonzero distortion.")
    rotation = _rotation_matrix(rotation)
    translation = _translation_vector(translation)
    points = np.asarray(object_points, dtype=float)
    camera_points = transform_points(points, rotation, translation)
    fx = camera_model.camera_matrix[0, 0]
    fy = camera_model.camera_matrix[1, 1]
    rows = []
    for object_point, camera_point in zip(points, camera_points):
        x_value, y_value, z_value = camera_point
        if z_value <= 0.0:
            raise ValueError("projection Jacobian requires positive corner depth.")
        projection = np.array(
            [
                [fx / z_value, 0.0, -fx * x_value / (z_value * z_value)],
                [0.0, fy / z_value, -fy * y_value / (z_value * z_value)],
            ]
        )
        rows.append(np.hstack((projection, -projection @ rotation @ skew(object_point))))
    return np.vstack(rows)


def finite_difference_pose_jacobian(
    object_points,
    rotation,
    translation,
    camera_model,
    step=1e-6,
):
    """Central difference in the same translation/right-rotation chart."""

    step = float(step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("finite-difference step must be positive and finite.")
    rotation = _rotation_matrix(rotation)
    translation = _translation_vector(translation)
    jacobian = np.zeros((2 * len(object_points), 6), dtype=float)
    for index in range(6):
        plus_translation = translation.copy()
        minus_translation = translation.copy()
        plus_rotation = rotation
        minus_rotation = rotation
        if index < 3:
            plus_translation[index] += step
            minus_translation[index] -= step
        else:
            axis = np.zeros(3, dtype=float)
            axis[index - 3] = step
            plus_delta, _ = cv2.Rodrigues(axis)
            minus_delta, _ = cv2.Rodrigues(-axis)
            plus_rotation = rotation @ plus_delta
            minus_rotation = rotation @ minus_delta
        plus = project_points(
            object_points, plus_rotation, plus_translation, camera_model
        ).reshape(-1)
        minus = project_points(
            object_points, minus_rotation, minus_translation, camera_model
        ).reshape(-1)
        jacobian[:, index] = (plus - minus) / (2.0 * step)
    return jacobian


def pose_jacobian(
    object_points,
    rotation,
    translation,
    camera_model,
    finite_difference_step=1e-6,
    verify=False,
    verification_rtol=2e-4,
    verification_atol=2e-4,
):
    """Use analytic pinhole derivatives, falling back for distorted cameras."""

    if camera_model.has_distortion:
        return finite_difference_pose_jacobian(
            object_points,
            rotation,
            translation,
            camera_model,
            finite_difference_step,
        )
    analytic = analytic_pinhole_pose_jacobian(
        object_points, rotation, translation, camera_model
    )
    if verify:
        numerical = finite_difference_pose_jacobian(
            object_points,
            rotation,
            translation,
            camera_model,
            finite_difference_step,
        )
        if not np.allclose(
            analytic,
            numerical,
            rtol=float(verification_rtol),
            atol=float(verification_atol),
        ):
            raise ValueError("analytic pinhole Jacobian failed finite-difference verification.")
    return analytic
