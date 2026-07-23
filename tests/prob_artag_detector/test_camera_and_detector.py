import cv2
import numpy as np
import pytest
from types import SimpleNamespace

from prob_artag_detector import (
    ArucoCornerDetector,
    CameraModel,
    analytic_pinhole_pose_jacobian,
    finite_difference_pose_jacobian,
    ippe_square_object_points,
    pose_jacobian,
    draw_debug_image,
)
from prob_artag_detector.models import MarkerObservation


def _camera(distortion=None):
    return CameraModel(
        np.array([[610.0, 0.0, 321.0], [0.0, 605.0, 239.0], [0.0, 0.0, 1.0]]),
        np.zeros(5) if distortion is None else distortion,
        640,
        480,
    )


def _pose():
    rotation, _ = cv2.Rodrigues(np.array([0.21, -0.14, 0.08]))
    return rotation, np.array([0.04, -0.03, 0.86])


def test_ippe_square_object_points_have_strict_documented_order():
    np.testing.assert_allclose(
        ippe_square_object_points(0.12),
        [
            [-0.06, 0.06, 0.0],
            [0.06, 0.06, 0.0],
            [0.06, -0.06, 0.0],
            [-0.06, -0.06, 0.0],
        ],
    )
    with pytest.raises(ValueError, match="positive"):
        ippe_square_object_points(0.0)


def test_analytic_right_perturbation_jacobian_matches_central_difference():
    camera = _camera()
    rotation, translation = _pose()
    points = ippe_square_object_points(0.12)
    analytic = analytic_pinhole_pose_jacobian(points, rotation, translation, camera)
    numerical = finite_difference_pose_jacobian(
        points, rotation, translation, camera, step=2e-7
    )
    np.testing.assert_allclose(analytic, numerical, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(
        pose_jacobian(points, rotation, translation, camera, verify=True), analytic
    )


def test_distorted_camera_automatically_uses_finite_difference_jacobian():
    camera = _camera(np.array([0.08, -0.03, 0.002, -0.001, 0.01]))
    rotation, translation = _pose()
    points = ippe_square_object_points(0.12)
    expected = finite_difference_pose_jacobian(
        points, rotation, translation, camera, step=1e-6
    )
    actual = pose_jacobian(
        points,
        rotation,
        translation,
        camera,
        finite_difference_step=1e-6,
        verify=True,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="nonzero distortion"):
        analytic_pinhole_pose_jacobian(points, rotation, translation, camera)


def test_camera_model_rejects_invalid_calibration():
    with pytest.raises(ValueError, match="focal"):
        CameraModel(
            np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.zeros(5),
        )
    with pytest.raises(ValueError, match="coefficients"):
        CameraModel(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.zeros(3),
        )


def test_aruco_detector_recovers_id_order_and_full_covariance():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = (
        cv2.aruco.generateImageMarker(dictionary, 17, 160)
        if hasattr(cv2.aruco, "generateImageMarker")
        else cv2.aruco.drawMarker(dictionary, 17, 160)
    )
    canvas = np.full((240, 240), 255, dtype=np.uint8)
    canvas[40:200, 40:200] = marker
    basis = np.eye(8)
    basis[0, 1] = 0.2
    covariance = basis @ basis.T
    detector = ArucoCornerDetector(
        "DICT_APRILTAG_36h11",
        corner_refinement=False,
        image_covariance=covariance,
        adaptive_covariance=True,
    )
    observations = detector.detect(canvas)
    assert len(observations) == 1
    observation = observations[0]
    assert observation.marker_id == 17
    np.testing.assert_allclose(observation.image_covariance, covariance)
    np.testing.assert_allclose(
        observation.corners_px,
        [[40.0, 40.0], [199.0, 40.0], [199.0, 199.0], [40.0, 199.0]],
        atol=2.0,
    )


def test_adaptive_corner_covariance_grows_for_unstable_blurred_edges():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = (
        cv2.aruco.generateImageMarker(dictionary, 17, 80)
        if hasattr(cv2.aruco, "generateImageMarker")
        else cv2.aruco.drawMarker(dictionary, 17, 80)
    )
    sharp = np.full((180, 180), 255, dtype=np.uint8)
    sharp[50:130, 50:130] = marker
    blurred = cv2.GaussianBlur(sharp, (0, 0), 3.5)
    detector = ArucoCornerDetector(
        adaptive_covariance=True,
        bootstrap_samples=12,
        bootstrap_seed=9,
    )

    sharp_observation = detector.detect(sharp)[0]
    blurred_observation = detector.detect(blurred)[0]
    repeated_observation = detector.detect(blurred)[0]
    sharp_covariance = sharp_observation.image_covariance
    blurred_covariance = blurred_observation.image_covariance

    np.testing.assert_allclose(
        blurred_covariance,
        repeated_observation.image_covariance,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(blurred_covariance, blurred_covariance.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(blurred_covariance)) >= 0.25 - 1e-10
    assert np.trace(blurred_covariance) > 1.2 * np.trace(sharp_covariance)
    off_diagonal = blurred_covariance - np.diag(np.diag(blurred_covariance))
    assert np.max(np.abs(off_diagonal)) > 1e-4


def test_adaptive_corner_covariance_inflates_when_perturbations_lose_marker(
    monkeypatch,
):
    detector = ArucoCornerDetector(
        adaptive_covariance=True,
        bootstrap_samples=4,
        corner_refinement=False,
    )
    corners = np.array(
        [[[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]]],
        dtype=np.float32,
    )
    call_count = [0]

    def baseline_then_missing(_gray):
        call_count[0] += 1
        if call_count[0] == 1:
            return (corners,), np.array([[7]], dtype=np.int32), ()
        return (), None, ()

    monkeypatch.setattr(detector, "_detect_markers", baseline_then_missing)
    observation = detector.detect(np.full((100, 100), 255, dtype=np.uint8))[0]
    assert call_count[0] == 1 + detector.bootstrap_samples
    assert np.min(np.diag(observation.image_covariance)) > 1.0


def test_adaptive_corner_covariance_counts_repeatable_perturbation_bias(
    monkeypatch,
):
    detector = ArucoCornerDetector(
        adaptive_covariance=True,
        bootstrap_samples=4,
        bootstrap_noise_std=0.0,
        bootstrap_dither_px=0.0,
        bootstrap_blur_sigma_px=0.0,
    )
    corners = np.array(
        [[[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]]],
        dtype=np.float32,
    )
    shifted = corners.copy()
    shifted[0, 0, 0] += 1.0
    call_count = [0]

    def baseline_then_biased(_gray):
        call_count[0] += 1
        value = corners if call_count[0] == 1 else shifted
        return (value,), np.array([[7]], dtype=np.int32), ()

    monkeypatch.setattr(detector, "_detect_markers", baseline_then_biased)
    observation = detector.detect(np.full((100, 100), 255, dtype=np.uint8))[0]
    assert observation.image_covariance[0, 0] == pytest.approx(1.25)
    np.testing.assert_allclose(
        observation.image_covariance[1:, 1:],
        np.eye(7) * 0.25,
        atol=1e-12,
    )


def test_detector_validates_dictionary_covariance_and_image_type():
    with pytest.raises(ValueError, match="Unsupported"):
        ArucoCornerDetector("DICT_4X4_50")
    with pytest.raises(ValueError, match="8x8"):
        ArucoCornerDetector(image_covariance=np.eye(4))
    with pytest.raises(ValueError, match="at least 2"):
        ArucoCornerDetector(bootstrap_samples=1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ArucoCornerDetector(bootstrap_covariance_shrinkage=1.1)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        ArucoCornerDetector(bootstrap_min_success_ratio=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        ArucoCornerDetector(bootstrap_dropout_sigma_px=-1.0)
    detector = ArucoCornerDetector(corner_refinement=False)
    with pytest.raises(ValueError, match="uint8"):
        detector.detect(np.zeros((20, 20), dtype=float))
    with pytest.raises(ValueError, match="grayscale"):
        detector.detect(np.zeros((20, 20, 2), dtype=np.uint8))


def test_debug_overlay_draws_every_retained_pose_axis(monkeypatch):
    camera = _camera()
    observation = MarkerObservation(
        3,
        np.array([[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]]),
        np.eye(8),
    )
    calls = []

    def record_axis(*args, **kwargs):
        calls.append((args, kwargs))
        return args[0]

    monkeypatch.setattr(cv2, "drawFrameAxes", record_axis)
    result = SimpleNamespace(
        rotations=(np.eye(3), cv2.Rodrigues(np.array([0.1, 0.0, 0.0]))[0]),
        translations=(np.array([0.0, 0.0, 1.0]), np.array([0.01, 0.0, 1.0])),
        diagnostics=SimpleNamespace(accepted_count=2),
        weights=(0.7, 0.3),
    )
    output = draw_debug_image(
        np.zeros((100, 100, 3), dtype=np.uint8),
        (observation,),
        (result,),
        camera_model=camera,
    )
    assert output.shape == (100, 100, 3)
    assert len(calls) == 2
