"""Stationary raw-vector covariance defines one-reading likelihoods."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


SCRIPT = (Path(__file__).resolve().parents[2]
          / 'ros/examples/probtf_orientation_demo/scripts/static_vector_noise.py')
SPEC = importlib.util.spec_from_file_location('orientation_static_vector_noise', SCRIPT)
noise_model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(noise_model)


def calibration_samples():
    # Six deterministic readings have exactly known mean and unbiased covariance.
    center = np.array([2., -3., 10.])
    offsets = np.array([[1., 0., 0.], [-1., 0., 0.],
                        [0., 2., 0.], [0., -2., 0.],
                        [0., 0., 3.], [0., 0., -3.]])
    return center + offsets


def test_estimate_preserves_raw_mean_and_full_unbiased_sample_covariance():
    samples = calibration_samples()
    original = samples.copy()
    result = noise_model.estimate_isotropic_vector_noise(samples)
    expected = np.diag([2., 8., 18.]) / 5.
    assert result['count'] == 6
    np.testing.assert_array_equal(result['mean'], [2., -3., 10.])
    np.testing.assert_allclose(result['sample_covariance'], expected, rtol=0, atol=1e-15)
    assert result['isotropic_variance'] == pytest.approx(28. / 15.)
    assert result['reference_magnitude'] == pytest.approx(np.sqrt(113.))
    np.testing.assert_array_equal(samples, original)


def test_covariance_describes_one_reading_and_is_not_divided_by_count():
    samples = calibration_samples()
    result = noise_model.estimate_isotropic_vector_noise(samples)
    single_reading_covariance = np.cov(samples, rowvar=False, ddof=1)
    np.testing.assert_array_equal(result['sample_covariance'], single_reading_covariance)
    assert result['isotropic_variance'] == pytest.approx(
        np.trace(single_reading_covariance) / 3.)
    assert result['isotropic_variance'] != pytest.approx(
        np.trace(single_reading_covariance) / (3. * len(samples)))


def test_concentration_uses_raw_vector_length_without_bias_subtraction():
    result = noise_model.estimate_isotropic_vector_noise(calibration_samples())
    vectors = np.array([[2., -3., 10.], [0.4, -0.7, 9.4], [8., 2., -1.]])
    original = vectors.copy()
    actual = noise_model.concentration_from_vector_noise(vectors, result)
    expected = (np.sqrt(113.) * np.linalg.norm(vectors, axis=1)
                / result['isotropic_variance'])
    assert np.asarray(actual).shape == (3,)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0)
    # In particular, a reading equal to the calibration mean still has evidence.
    assert actual[0] > 0
    np.testing.assert_array_equal(vectors, original)


@pytest.mark.parametrize('scale', [0.001, 0.1, 17.])
def test_common_sensor_unit_change_leaves_directional_concentration_unchanged(scale):
    samples = calibration_samples()
    vectors = np.array([[0.2, 0.3, 9.8], [4., -2., 30.], [-8., 3., 0.1]])
    original_samples = samples.copy()
    original_vectors = vectors.copy()
    base_noise = noise_model.estimate_isotropic_vector_noise(samples)
    scaled_noise = noise_model.estimate_isotropic_vector_noise(scale * samples)
    actual = noise_model.concentration_from_vector_noise(scale * vectors, scaled_noise)
    expected = noise_model.concentration_from_vector_noise(vectors, base_noise)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0)
    np.testing.assert_allclose(scaled_noise['sample_covariance'],
                               scale**2 * np.asarray(base_noise['sample_covariance']),
                               rtol=3e-14, atol=1e-20)
    assert scaled_noise['reference_magnitude'] == pytest.approx(
        scale * base_noise['reference_magnitude'])
    np.testing.assert_array_equal(samples, original_samples)
    np.testing.assert_array_equal(vectors, original_vectors)


def test_raw_gaussian_log_likelihood_differences_equal_directional_evidence():
    result = noise_model.estimate_isotropic_vector_noise(calibration_samples())
    vectors = np.array([[0.7, -1.2, 9.4], [-1.1, 0.2, 8.6], [5., 2., 20.]])
    kappas = noise_model.concentration_from_vector_noise(vectors, result)
    reference_unit = np.array([0.3, -0.4, np.sqrt(0.75)])
    rotations = Rotation.from_euler('xyz', [[0., 0., 0.], [23., -17., 48.],
                                           [-32., 11., -71.], [7., 44., 133.]],
                                    degrees=True).as_matrix()
    # R maps sensor coordinates to world coordinates. The raw Gaussian's mean
    # is R^T times the fixed world-reference vector, with its measured magnitude.
    predicted_sensor_means = np.einsum(
        'kji,j->ki', rotations, result['reference_magnitude'] * reference_unit)
    for vector, kappa in zip(vectors, kappas):
        gaussian_log_likelihood = -np.sum(
            (vector - predicted_sensor_means)**2, axis=1) / (2. * result['isotropic_variance'])
        observed_unit = vector / np.linalg.norm(vector)
        directional_log_likelihood = kappa * np.einsum(
            'j,kjl,l->k', reference_unit, rotations, observed_unit)
        np.testing.assert_allclose(
            gaussian_log_likelihood - gaussian_log_likelihood[0],
            directional_log_likelihood - directional_log_likelihood[0],
            rtol=2e-14, atol=2e-13)


def test_zero_sample_variance_is_rejected_without_an_invented_floor():
    with pytest.raises(ValueError):
        noise_model.estimate_isotropic_vector_noise(np.tile([1., 2., 3.], (8, 1)))


@pytest.mark.parametrize('samples', [np.empty((0, 3)), np.array([[1., 2., 3.]])])
def test_at_least_two_calibration_readings_are_required(samples):
    with pytest.raises(ValueError):
        noise_model.estimate_isotropic_vector_noise(samples)


@pytest.mark.parametrize('nonfinite', [np.nan, np.inf, -np.inf])
def test_nonfinite_calibration_reading_is_rejected(nonfinite):
    samples = calibration_samples()
    samples[2, 1] = nonfinite
    with pytest.raises(ValueError):
        noise_model.estimate_isotropic_vector_noise(samples)
