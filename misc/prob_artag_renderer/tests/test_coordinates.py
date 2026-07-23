import numpy as np

from prob_artag_renderer.coordinates import (
    T_GL_CV, invert_transform, look_at_opencv, make_transform,
    marker_corners_ippe, opencv_camera_pose_to_pyrender,
    pyrender_camera_pose_to_opencv, transform_points,
)


def test_ippe_square_corner_order_is_exact():
    np.testing.assert_allclose(
        marker_corners_ippe(2.0),
        [[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]],
    )


def test_opencv_to_pyrender_pose_preserves_physical_points():
    angle = 0.31
    rotation = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    T_W_C = make_transform(rotation, [0.2, -0.3, 1.1])
    T_W_GL = opencv_camera_pose_to_pyrender(T_W_C)
    point_C = np.array([[0.4, -0.2, 2.0]])
    point_GL = transform_points(T_GL_CV, point_C)
    np.testing.assert_allclose(
        transform_points(T_W_C, point_C), transform_points(T_W_GL, point_GL), atol=1e-12
    )
    np.testing.assert_allclose(pyrender_camera_pose_to_opencv(T_W_GL), T_W_C)


def test_transform_inverse_and_non_axis_aligned_look_at():
    pose = look_at_opencv([0.2, -0.1, -0.3], [0.4, 0.2, 1.7])
    np.testing.assert_allclose(invert_transform(pose).dot(pose), np.eye(4), atol=1e-12)
    expected_forward = np.array([0.2, 0.3, 2.0])
    expected_forward /= np.linalg.norm(expected_forward)
    np.testing.assert_allclose(pose[:3, 2], expected_forward, atol=1e-12)
    assert np.linalg.det(pose[:3, :3]) > 0.999999
