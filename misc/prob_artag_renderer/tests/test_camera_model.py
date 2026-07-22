import numpy as np

from prob_artag_renderer.camera_model import CameraModel


def test_pinhole_project_unproject_round_trip():
    camera = CameraModel()
    points = np.array([[0.1, -0.2, 1.0], [-0.3, 0.25, 2.0], [0.5, 0.1, 3.0]])
    pixels = camera.project(points)
    reconstructed = camera.unproject(pixels, points[:, 2])
    np.testing.assert_allclose(reconstructed, points, atol=1e-12)


def test_distorted_projection_matches_opencv():
    cv2 = __import__("cv2")
    camera = CameraModel(distortion=np.array([-0.1, 0.02, 0.001, -0.002, 0.0]))
    points = np.array([[0.2, 0.1, 1.3], [-0.4, 0.2, 2.1]])
    expected, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3),
                                    camera.matrix, camera.distortion)
    np.testing.assert_allclose(camera.project(points), expected.reshape(-1, 2), atol=1e-10)
