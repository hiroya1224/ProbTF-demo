"""Detector audits retain finite usable values without inventing a law."""

import cv2
import numpy as np
import pytest

import prob_artag_detector.camera as camera_module
import prob_artag_detector.estimator as estimator_module
from prob_artag_detector import CameraModel, MarkerObservation, PoseMixtureEstimator, PoseSeed


def _fixture():
    camera = CameraModel(np.array([[600., 0., 320.], [0., 615., 240.], [0., 0., 1.]]),
                         np.zeros(5), 640, 480)
    rotation, _ = cv2.Rodrigues(np.array([.24, -.13, .07]))
    translation = np.array([.035, -.022, .82])
    points = camera_module.ippe_square_object_points(.12)
    corners = camera_module.project_points(points, rotation, translation, camera)
    observation = MarkerObservation(4, corners, .25*np.eye(8))
    return camera, points, rotation, translation, observation


def test_relative_spectral_audit_keeps_original_thin_SPD_matrix(caplog):
    covariance = np.diag([1e-20, 2., 3.])
    result = estimator_module._positive_definite(covariance, 'thin covariance', 1e-12)
    np.testing.assert_array_equal(result, covariance)
    np.testing.assert_array_equal(np.linalg.solve(result, np.eye(3)),
                                  np.diag([1e20, .5, 1./3.]))
    assert 'relative spectral audit missed' in caplog.text
    assert 'no matrix modification' in caplog.text


def test_diagnostic_eigensolver_failure_keeps_required_factorization(monkeypatch, caplog):
    covariance = np.diag([2., 3., 4.])

    def failed_audit(matrix):
        raise np.linalg.LinAlgError('diagnostic eigensolver failed')

    monkeypatch.setattr(np.linalg, 'eigvalsh', failed_audit)
    np.testing.assert_array_equal(estimator_module._positive_definite(covariance, 'covariance', 1e-12), covariance)
    assert 'spectral audit failed' in caplog.text


@pytest.mark.parametrize('matrix,message', [
    (np.diag([0., 1., 2.]), 'required Cholesky'),
    (np.diag([-1., 1., 2.]), 'required Cholesky'),
    (np.diag([float('nan'), 1., 2.]), 'finite square'),
    (np.diag([float('inf'), 1., 2.]), 'finite square')])
def test_required_covariance_operation_failures_are_not_relabelled(matrix, message):
    with pytest.raises(ValueError, match=message):
        estimator_module._positive_definite(matrix, 'covariance', 1e-12)


def test_singleton_finite_line_search_stall_keeps_pose_and_reports_no_convergence(monkeypatch):
    camera, _, rotation, translation, observation = _fixture()
    estimator = PoseMixtureEstimator(.12, translation_prior_variance=None)
    seed = PoseSeed(rotation, translation+np.array([.02, -.01, .01]), 1.)
    # A finite objective plateau admits no strict line-search improvement.
    # Keep the real projection, Jacobian, Hessian, and distribution build.
    monkeypatch.setattr(estimator, '_objective', lambda *args: 1.)
    result = estimator.estimate(observation, camera, 'camera', 'marker', 1., pose_seeds=(seed,))
    assert len(result.record.distribution.components) == 1
    np.testing.assert_array_equal(result.translations[0], seed.translation)
    np.testing.assert_array_equal(result.rotations[0], seed.rotation)
    assert result.diagnostics.candidates[0].reason == 'accepted_line_search_stalled'
    assert 'last valid IPPE-branch pose was retained' in result.record.distribution.components[0].approximation.detail
    assert np.all(np.isfinite(result.log_masses))


def test_finite_iteration_budget_result_is_not_called_converged():
    camera, _, rotation, translation, observation = _fixture()
    seed = PoseSeed(rotation, translation+np.array([.02, -.01, .01]), 1.)
    result = PoseMixtureEstimator(.12, max_iterations=1, convergence_tolerance=1e-16).estimate(
        observation, camera, 'camera', 'marker', 1., pose_seeds=(seed,))
    assert result.diagnostics.candidates[0].reason == 'accepted_iteration_budget'
    assert 'without a convergence claim' in result.record.distribution.components[0].approximation.detail


def test_jacobian_verification_miss_keeps_same_finite_analytic_derivative(monkeypatch, caplog):
    camera, points, rotation, translation, _ = _fixture()
    analytic = camera_module.analytic_pinhole_pose_jacobian(points, rotation, translation, camera)
    monkeypatch.setattr(camera_module, 'finite_difference_pose_jacobian', lambda *args: analytic+1.)
    actual = camera_module.pose_jacobian(points, rotation, translation, camera, verify=True)
    np.testing.assert_array_equal(actual, analytic)
    assert 'finite-difference audit missed' in caplog.text


def test_actual_nonfinite_jacobian_remains_an_error(monkeypatch):
    camera, points, rotation, translation, _ = _fixture()
    monkeypatch.setattr(camera_module, 'analytic_pinhole_pose_jacobian',
                        lambda *args: np.full((8, 6), float('nan')))
    with pytest.raises(ValueError, match='nonfinite'):
        camera_module.pose_jacobian(points, rotation, translation, camera)


@pytest.mark.parametrize('failure', [ValueError, np.linalg.LinAlgError, FloatingPointError, 'nonfinite'])
def test_unavailable_optional_jacobian_audit_does_not_replace_finite_derivative(monkeypatch, caplog, failure):
    camera, points, rotation, translation, _ = _fixture()
    analytic = camera_module.analytic_pinhole_pose_jacobian(points, rotation, translation, camera)

    def unavailable_reference(*args):
        if failure == 'nonfinite':
            return np.full((8, 6), float('nan'))
        raise failure('reference finite difference unavailable')

    monkeypatch.setattr(camera_module, 'finite_difference_pose_jacobian', unavailable_reference)
    actual = camera_module.pose_jacobian(points, rotation, translation, camera, verify=True)
    np.testing.assert_array_equal(actual, analytic)
    assert 'audit unavailable' in caplog.text


def test_distorted_camera_required_jacobian_failure_is_not_an_optional_audit(monkeypatch):
    camera, points, rotation, translation, _ = _fixture()
    camera = CameraModel(camera.camera_matrix, np.array([.1, 0., 0., 0., 0.]), 640, 480)

    def unavailable_reference(*args):
        raise FloatingPointError('required distorted-camera derivative unavailable')

    monkeypatch.setattr(camera_module, 'finite_difference_pose_jacobian', unavailable_reference)
    with pytest.raises(FloatingPointError, match='required distorted-camera'):
        camera_module.pose_jacobian(points, rotation, translation, camera, verify=True)


def test_actual_singular_refinement_system_remains_unavailable(monkeypatch):
    camera, _, rotation, translation, observation = _fixture()
    estimator = PoseMixtureEstimator(.12, translation_prior_variance=None)
    seed = PoseSeed(rotation, translation, 0.)
    monkeypatch.setattr(estimator_module, 'pose_jacobian', lambda *args, **kwargs: np.zeros((8, 6)))
    with pytest.raises(estimator_module.PoseEstimationError, match='singular'):
        estimator.estimate(observation, camera, 'camera', 'marker', 1., pose_seeds=(seed,))
