"""Keep the synthetic sensor generator separate from the existing estimator."""

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2]/'ros/examples/probtf_orientation_demo/scripts'
sys.path.insert(0, str(SCRIPTS))
import compare_synthetic_flight as experiment


def test_estimator_noise_does_not_use_the_generating_covariance_or_bias():
    _, settings = experiment.generate_flight(.02, 100.)
    before = experiment.assumed_sensor_noise(settings)
    changed = dict(settings, generation_noise_variances=object(),
                   generation_noise_covariances=object(), gyro_bias_rad_s=object())
    assert experiment.assumed_sensor_noise(changed) == before
    for sensor, variance in settings['estimation_noise_variances'].items():
        np.testing.assert_array_equal(before[sensor]['isotropic_covariance'], variance*np.eye(3))


def test_filter_uses_only_readings_initial_pose_and_assumed_noise_without_random_draws():
    data, settings = experiment.generate_flight(.10, 100.)
    allowed = {'gyro_rad_s', 'acceleration_m_s2', 'magnetic_raw', 'measurement_stamps'}

    class OnlyReadings(dict):
        def __getitem__(self, key):
            assert key in allowed, 'Estimator accessed hidden truth or generated noise: '+key
            return super().__getitem__(key)

    measurements = OnlyReadings(data)
    initial = data['truth_wxyz'][0].copy()
    noise = experiment.assumed_sensor_noise(settings)
    reference = np.asarray(settings['magnetic_reference_world_unit'])
    with patch.object(np.random, 'default_rng', side_effect=AssertionError('Randomness in inference')):
        result, acceleration_weights, magnetic_weights = experiment.estimate_measurements(
            measurements, initial, reference, noise)
    parameters, modes, carried, residuals = result
    assert parameters.shape == (11, 6, 4, 4)
    assert modes.shape == (11, 6, 4)
    assert carried.shape == (11, 3, 4, 4)
    assert np.isfinite(parameters).all() and np.isfinite(residuals).all()
    np.testing.assert_array_equal(acceleration_weights,
        experiment.static_vector_noise.concentration_from_vector_noise(data['acceleration_m_s2'], noise['acceleration']))
    np.testing.assert_array_equal(magnetic_weights,
        experiment.static_vector_noise.concentration_from_vector_noise(data['magnetic_raw'], noise['magnetic']))
    gravity, field = experiment.comparison.direction_factors(
        data['acceleration_m_s2'], data['magnetic_raw'], reference,
        acceleration_weights, magnetic_weights)
    np.testing.assert_array_equal(parameters, experiment.comparison.output_parameters(carried, gravity, field))
