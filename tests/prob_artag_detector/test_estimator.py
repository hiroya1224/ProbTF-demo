import math

import cv2
import numpy as np
import pytest

import prob_artag_detector.estimator as estimator_module
from prob_artag_detector import (
    CameraModel,
    MarkerObservation,
    PoseEstimationError,
    PoseMixtureEstimator,
    PoseMixtureResult,
    PoseSeed,
    bingham_parameter_from_tangent_precision,
    image_precision,
    ippe_square_object_points,
    local_gauss_newton_hessian,
    normalize_log_weights,
    project_points,
    reconstruct_pose_hessian,
    solve_ippe_square_candidates,
)
from probtf.distributions import BinghamOrientation
from probtf.geometry import (
    axis_angle_to_quat,
    quat_mul,
    right_perturbation_vec_rotation_jacobian,
)
from probtf.probability import (
    PointMomentSummary,
    forward_component_point_moments,
    mixture_point_moments,
)
from probtf_ros import transform_distribution_from_msg, transform_distribution_to_msg


def _camera(distortion=None):
    return CameraModel(
        np.array([[600.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]]),
        np.zeros(5) if distortion is None else distortion,
        640,
        480,
    )


def _pose():
    rotation, _ = cv2.Rodrigues(np.array([0.24, -0.13, 0.07]))
    return rotation, np.array([0.035, -0.022, 0.82])


def _observation(camera=None, covariance=None, noise=None):
    camera = _camera() if camera is None else camera
    rotation, translation = _pose()
    corners = project_points(
        ippe_square_object_points(0.12), rotation, translation, camera
    )
    if noise is not None:
        corners = corners + np.asarray(noise, dtype=float).reshape(4, 2)
    covariance = np.eye(8) * 0.25 if covariance is None else covariance
    return MarkerObservation(4, corners, covariance), rotation, translation


def _rotation_error(first, second):
    relative = first.T @ second
    return math.acos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))


def test_ippe_returns_all_candidates_and_contains_ground_truth_solution():
    camera = _camera()
    observation, truth_rotation, truth_translation = _observation(camera)
    seeds = solve_ippe_square_candidates(observation.corners_px, camera, 0.12)
    assert len(seeds) >= 2
    best = min(
        seeds,
        key=lambda seed: np.linalg.norm(seed.translation - truth_translation)
        + _rotation_error(seed.rotation, truth_rotation),
    )
    np.testing.assert_allclose(best.translation, truth_translation, atol=1e-8)
    assert _rotation_error(best.rotation, truth_rotation) < 1e-7


def test_full_image_covariance_enters_common_mahalanobis_hessian():
    camera = _camera()
    observation, rotation, translation = _observation(camera)
    factor = np.diag(np.linspace(0.3, 1.1, 8))
    factor[3, 0] = 0.15
    covariance = factor @ factor.T
    precision = image_precision(covariance)
    hessian = local_gauss_newton_hessian(
        ippe_square_object_points(0.12),
        rotation,
        translation,
        camera,
        precision,
    )
    assert hessian.shape == (6, 6)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-9)
    assert np.linalg.eigvalsh(hessian)[0] > 0.0
    with pytest.raises(ValueError, match="positive definite"):
        image_precision(np.zeros((8, 8)))


def test_larger_image_covariance_widens_joint_mixture_point_moments():
    camera = _camera()
    narrow_observation, _, _ = _observation(
        camera, covariance=np.eye(8) * 0.25
    )
    wide_observation, _, _ = _observation(
        camera, covariance=np.eye(8) * 4.0
    )
    np.testing.assert_array_equal(
        wide_observation.corners_px, narrow_observation.corners_px
    )

    estimator = PoseMixtureEstimator(0.12)
    narrow = estimator.estimate(
        narrow_observation, camera, "camera", "tag", 1.0
    )
    wide = estimator.estimate(
        wide_observation, camera, "camera", "tag", 1.0
    )

    def rgb_axis_endpoint_spread(result):
        normalized = result.record.distribution.normalize_weights()
        total = 0.0
        for axis in range(3):
            endpoint = np.zeros(3)
            endpoint[axis] = 0.06
            input_moments = PointMomentSummary(endpoint, np.zeros((3, 3)))
            summary = mixture_point_moments(
                (
                    weighted.weight,
                    forward_component_point_moments(
                        weighted.component,
                        input_moments,
                        integration_steps=40,
                    ),
                )
                for weighted in normalized.components
            )
            total += float(np.trace(summary.covariance))
        return total

    narrow_spread = rgb_axis_endpoint_spread(narrow)
    wide_spread = rgb_axis_endpoint_spread(wide)
    assert narrow_spread > 0.0
    assert wide_spread > 10.0 * narrow_spread


def test_shared_translation_prior_adds_same_global_hessian_precision():
    camera = _camera()
    _, rotation, translation = _observation(camera)
    points = ippe_square_object_points(0.12)
    precision = image_precision(np.eye(8))
    without_prior = local_gauss_newton_hessian(
        points, rotation, translation, camera, precision
    )
    with_prior = local_gauss_newton_hessian(
        points,
        rotation,
        translation,
        camera,
        precision,
        translation_prior_precision=0.25,
    )
    expected = np.zeros((6, 6))
    expected[:3, :3] = 0.25 * np.eye(3)
    np.testing.assert_allclose(with_prior - without_prior, expected, atol=1e-9)


def test_hessian_block_decomposition_reconstructs_original_precision():
    hxx = np.array([[8.0, 0.4, -0.2], [0.4, 7.0, 0.3], [-0.2, 0.3, 9.0]])
    hxu = np.array([[0.5, -0.2, 0.3], [0.1, 0.4, -0.5], [-0.3, 0.2, 0.6]])
    rotation_precision = np.array(
        [[5.0, 0.2, 0.1], [0.2, 4.0, -0.1], [0.1, -0.1, 6.0]]
    )
    covariance = np.linalg.solve(hxx, np.eye(3))
    local_map = -np.linalg.solve(hxx, hxu)
    huu = local_map.T @ hxx @ local_map + rotation_precision
    expected = np.block([[hxx, hxu], [hxu.T, huu]])
    reconstructed = reconstruct_pose_hessian(covariance, local_map, rotation_precision)
    np.testing.assert_allclose(reconstructed, expected, atol=1e-12)


def test_bingham_embedding_matches_right_quaternion_chart_curvature():
    mode = np.array([0.8, -0.2, 0.3, 0.45], dtype=float)
    mode /= np.linalg.norm(mode)
    precision = np.array([[12.0, 0.8, -0.2], [0.8, 8.0, 0.4], [-0.2, 0.4, 10.0]])
    parameter = bingham_parameter_from_tangent_precision(mode, precision)
    assert np.trace(parameter) == pytest.approx(0.0, abs=1e-10)
    orientation = BinghamOrientation.from_parameter_matrix(parameter, mode)
    np.testing.assert_allclose(orientation.parameter_matrix(), parameter, atol=1e-10)
    perturbation = np.array([2e-4, -1e-4, 1.5e-4])
    angle = np.linalg.norm(perturbation)
    query = quat_mul(mode, axis_angle_to_quat(perturbation / angle, angle))
    actual = float(query @ parameter @ query - mode @ parameter @ mode)
    expected = -0.5 * float(perturbation @ precision @ perturbation)
    assert actual == pytest.approx(expected, rel=5e-4, abs=1e-12)


def test_laplace_log_sum_exp_weights_are_stable_and_positive():
    weights = normalize_log_weights([-10000.0, -10001.0, -10010.0])
    assert np.sum(weights) == pytest.approx(1.0)
    assert np.all(weights > 0.0)
    assert weights[0] > weights[1] > weights[2]
    with pytest.raises(ValueError, match="non-empty"):
        normalize_log_weights([])
    with pytest.raises(ValueError, match="finite"):
        normalize_log_weights([0.0, np.nan])


def test_end_to_end_builds_native_dynamic_t_c_m_distribution_and_deduplicates():
    camera = _camera()
    observation, truth_rotation, truth_translation = _observation(camera)
    result = PoseMixtureEstimator(0.12, verify_jacobian=True).estimate(
        observation,
        camera,
        parent_frame_id="camera_optical",
        child_frame_id="apriltag_4",
        stamp=12.5,
        authority="test_detector",
    )
    record = result.record
    assert record.parent_frame_id == "camera_optical"
    assert record.child_frame_id == "apriltag_4"
    assert not record.is_static
    assert result.diagnostics.seed_count >= 2
    assert result.diagnostics.accepted_count == 2
    assert len(record.distribution.components) == 2
    assert any(
        item.reason.startswith("accepted_") for item in result.diagnostics.candidates
    )
    assert sum(result.weights) == pytest.approx(1.0)
    best_index = int(np.argmax(result.weights))
    np.testing.assert_allclose(result.translations[best_index], truth_translation, atol=1e-7)
    assert _rotation_error(result.rotations[best_index], truth_rotation) < 1e-6
    component = record.distribution.components[best_index]
    jacobian = right_perturbation_vec_rotation_jacobian(
        component.orientation.reference_quaternion_wxyz
    )
    assert component.translation.rotation_coupling.shape == (3, 9)
    assert component.translation.residual_covariance.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(component.translation.residual_covariance) > 0.0)
    assert np.linalg.norm(component.translation.rotation_coupling @ jacobian) > 0.0


@pytest.mark.parametrize(
    "rotation_vector",
    (
        np.array([0.035, -0.025, 0.01]),
        np.array([0.24, -0.13, 0.07]),
    ),
    ids=("near_frontal", "tilted"),
)
def test_ippe_branch_guard_preserves_two_distinct_pose_hypotheses(rotation_vector):
    camera = _camera()
    rotation, _ = cv2.Rodrigues(rotation_vector)
    translation = np.array([0.02, -0.01, 0.9])
    corners = project_points(
        ippe_square_object_points(0.12), rotation, translation, camera
    )
    observation = MarkerObservation(9, corners, np.eye(8) * 0.25)
    result = PoseMixtureEstimator(0.12).estimate(
        observation, camera, "camera", "apriltag_9", 1.0
    )
    assert result.diagnostics.seed_count == 2
    assert result.diagnostics.accepted_count == 2
    assert result.diagnostics.deduplicated_count == 0
    assert set(result.seed_indices) == {0, 1}
    assert len(result.record.distribution.components) == 2
    for seed_index, component in zip(
        result.seed_indices, result.record.distribution.components
    ):
        assert "candidate {}.".format(seed_index) in component.provenance.detail
    assert _rotation_error(result.rotations[0], result.rotations[1]) > 1e-4


def test_explicit_ippe_seeds_are_reused_and_result_keeps_legacy_constructor(monkeypatch):
    camera = _camera()
    observation, _, _ = _observation(camera)
    seeds = solve_ippe_square_candidates(observation.corners_px, camera, 0.12)

    def unexpected_resolve(*_args, **_kwargs):
        raise AssertionError("IPPE must not be solved twice")

    monkeypatch.setattr(
        estimator_module, "solve_ippe_square_candidates", unexpected_resolve
    )
    result = PoseMixtureEstimator(0.12).estimate(
        observation,
        camera,
        "camera",
        "tag",
        1.0,
        pose_seeds=seeds,
    )
    assert set(result.seed_indices) == set(range(len(seeds)))

    legacy = PoseMixtureResult(
        result.record,
        result.diagnostics,
        result.rotations,
        result.translations,
        result.log_masses,
        result.weights,
    )
    assert legacy.seed_indices == ()
    with pytest.raises(ValueError, match="unique"):
        PoseMixtureResult(
            result.record,
            result.diagnostics,
            result.rotations,
            result.translations,
            result.log_masses,
            result.weights,
            (0, 0),
        )


def test_distorted_camera_estimation_uses_numerical_jacobian():
    camera = _camera(np.array([0.06, -0.02, 0.001, -0.002, 0.005]))
    observation, _, truth_translation = _observation(camera)
    result = PoseMixtureEstimator(0.12).estimate(
        observation, camera, "camera", "tag", 2.0
    )
    best = int(np.argmax(result.weights))
    np.testing.assert_allclose(result.translations[best], truth_translation, atol=2e-6)


def test_invalid_covariance_and_cheirality_are_explicitly_rejected(monkeypatch):
    camera = _camera()
    observation, _, _ = _observation(camera, covariance=np.zeros((8, 8)))
    with pytest.raises(ValueError, match="positive definite"):
        PoseMixtureEstimator(0.12).estimate(observation, camera, "camera", "tag", 1.0)

    good_observation, _, _ = _observation(camera)
    monkeypatch.setattr(
        estimator_module,
        "solve_ippe_square_candidates",
        lambda *_args, **_kwargs: (PoseSeed(np.eye(3), np.array([0.0, 0.0, -1.0]), 0.0),),
    )
    with pytest.raises(PoseEstimationError, match="cheirality"):
        PoseMixtureEstimator(0.12).estimate(
            good_observation, camera, "camera", "tag", 1.0
        )


def test_message_roundtrip_and_core_point_kernel_accept_estimated_component():
    camera = _camera()
    observation, _, _ = _observation(camera, covariance=np.eye(8) * 4.0)
    result = PoseMixtureEstimator(0.12).estimate(
        observation, camera, "camera", "apriltag_4", 3.0
    )
    message = transform_distribution_to_msg(result.record, time_factory=lambda value: value)
    decoded = transform_distribution_from_msg(message)
    assert decoded.parent_frame_id == "camera"
    assert decoded.child_frame_id == "apriltag_4"
    assert len(decoded.distribution.components) == len(result.record.distribution.components)
    component = decoded.distribution.components[0]
    summary = forward_component_point_moments(
        component,
        PointMomentSummary(np.zeros(3), np.zeros((3, 3))),
        integration_steps=40,
    )
    assert summary.mean.shape == (3,)
    assert summary.covariance.shape == (3, 3)
    assert np.all(np.isfinite(summary.mean))
    assert np.all(np.linalg.eigvalsh(summary.covariance) >= -1e-9)


def test_estimator_configuration_and_observation_inputs_are_validated():
    with pytest.raises(ValueError, match="max_iterations"):
        PoseMixtureEstimator(0.12, max_iterations=0)
    with pytest.raises(ValueError, match="positive"):
        PoseMixtureEstimator(-0.12)
    with pytest.raises(ValueError, match="translation_prior_mean"):
        PoseMixtureEstimator(0.12, translation_prior_mean=[0.0, 0.0])
    with pytest.raises(ValueError, match="translation_prior_variance"):
        PoseMixtureEstimator(0.12, translation_prior_variance=0.0)
    assert PoseMixtureEstimator(
        0.12, translation_prior_variance=None
    ).translation_prior_precision == 0.0
    camera = _camera()
    with pytest.raises(ValueError, match="shape"):
        solve_ippe_square_candidates(np.zeros((3, 2)), camera, 0.12)
