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
    allowed = {'gyro_rad_s', 'acceleration_m_s2', 'magnetic_raw', 'measurement_stamps', 'magnetic_observed'}

    class OnlyReadings(dict):
        def __getitem__(self, key):
            assert key in allowed, 'Estimator accessed hidden truth or generated noise: '+key
            return super().__getitem__(key)

        def get(self, key, default=None):
            assert key in allowed, 'Estimator accessed hidden truth or generated noise: '+key
            return super().get(key, default)

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
    field = np.where(data['magnetic_observed'][:, None, None], field, 0.0)
    np.testing.assert_array_equal(parameters, experiment.comparison.output_parameters(carried, gravity, field))


def test_magnetic_publication_mask_fuses_each_observation_once_and_omits_stale_output():
    count = 7
    initial = np.array([1., 0., 0., 0.])
    acceleration = np.tile([0., 0., 1.], (count, 1))
    magnetic = np.tile([1., .1, .2], (count, 1))
    observed = np.array([True, False, False, True, False, False, True])

    def identity_propagation(parameter, *args):
        return parameter.copy(), 0.

    options = dict(acceleration_concentration=.1, magnetic_concentration=2.,
                   propagation=identity_propagation, magnetic_observed=observed)
    output, _, retained, _ = experiment.comparison.compare_six(
        np.zeros((count, 3)), acceleration, magnetic, np.arange(count)*.005,
        initial, np.array([1., .1, .2]), **options)
    gravity, factor = experiment.comparison.direction_factors(acceleration, magnetic, np.array([1., .1, .2]), .1, 2.)
    initial_parameter = 100.*(np.outer(initial, initial)-np.eye(4))
    for k in range(count):
        expected = experiment.comparison.canonical_batch(initial_parameter
            + (k+1)*gravity[k] + observed[:k+1].sum()*factor[k])
        np.testing.assert_allclose(retained[k, 2], expected, rtol=0., atol=1e-12)
        if not observed[k]:
            np.testing.assert_array_equal(retained[k, 2],
                experiment.comparison.canonical_batch(retained[k-1, 2] + gravity[k]))
            np.testing.assert_array_equal(output[k, 3], output[k, 1])
            np.testing.assert_array_equal(output[k, 4], output[k, 2])


def test_missing_publication_mask_keeps_legacy_every_sample_behavior():
    data, settings = experiment.generate_flight(.03, 100.)
    measurements = {key: data[key] for key in ('gyro_rad_s', 'acceleration_m_s2', 'magnetic_raw', 'measurement_stamps')}
    args = (data['truth_wxyz'][0], np.asarray(settings['magnetic_reference_world_unit']),
            experiment.assumed_sensor_noise(settings))
    legacy = experiment.estimate_measurements(measurements, *args)[0]
    explicit = experiment.estimate_measurements(dict(measurements, magnetic_observed=np.ones(4, bool)), *args)[0]
    for actual, expected in zip(legacy, explicit):
        np.testing.assert_array_equal(actual, expected)


def test_madgwick_omits_held_magnetic_readings_without_accessing_future_truth():
    data, settings = experiment.generate_flight(.03, 200.)
    measurements = {key: data[key] for key in (
        'gyro_rad_s', 'acceleration_m_s2', 'magnetic_raw', 'measurement_stamps', 'magnetic_observed')}
    measurements['mocap_wxyz'] = data['truth_wxyz'][:1]
    calls = []

    def record_update(q, gyro, acceleration, dt, magnetic=None, beta=.1):
        calls.append(None if magnetic is None else magnetic.copy())
        return q.copy()

    with patch.object(experiment.comparison.madgwick, 'madgwick_update', side_effect=record_update):
        experiment.comparison.madgwick.run_madgwick(measurements, settings['magnetic_reference_world_unit'])
    for k in range(1, len(data['measurement_stamps'])):
        assert calls[2*(k-1)] is None
        magnetic = calls[2*(k-1)+1]
        if data['magnetic_observed'][k]:
            np.testing.assert_array_equal(magnetic, data['magnetic_raw'][k])
        else:
            assert magnetic is None


def test_sparse_output_statistics_ignore_unavailable_magnetic_outputs():
    truth = np.tile([1., 0., 0., 0.], (4, 1))
    modes = np.repeat(truth[:, None], 6, axis=1)
    # The placeholder output at an absent magnetic publication must not enter
    # either RMS or final statistics for output-only magnetic cases.
    modes[1:, 3:5] = [0., 0., 0., 1.]
    available = np.ones((4, 6), dtype=bool)
    available[1:, 3:5] = False
    _, _, stats = experiment.comparison.evaluate(modes, truth, available)
    for case in (3, 4):
        assert stats[case]['rms_deg'] == 0.
        assert stats[case]['final_deg'] == 0.
    assert experiment.comparison.evaluate(modes, truth)[2][3]['rms_deg'] > 0.
