"""Native random pulse timing is independent of the sparse observation grid."""
from pathlib import Path
import importlib.util
import unittest

import numpy as np


HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('test_native_poisson_spike_input',
                                               HERE/'frequent_spikes_input.py')
inputs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inputs)


class FrequentNativeSpikeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = inputs.generate()

    def test_spike_schedule_does_not_depend_on_acquisition_grid(self):
        first = self.values
        changed = inputs.generate(dict(rate_hz=2.))
        for name in first[3]:
            if name.startswith('gyro_'):
                np.testing.assert_allclose(first[3][name], changed[3][name], rtol=1e-14, atol=1e-14)
            else:
                np.testing.assert_array_equal(first[3][name], changed[3][name])
        self.assertEqual(first[4], changed[4])
        np.testing.assert_array_equal(first[2]['source_index'], np.arange(20)*200)
        self.assertEqual(len(changed[2]['time']), 40)
        self.assertTrue(first[-1]['arrival_process_independent_of_acquisition'])
        self.assertFalse(first[-1]['forced_acquired_spike_count'])

    def test_same_seed_reproduces_all_sensor_values_and_event_records(self):
        second = inputs.generate()
        for position in (0, 1, 2, 3):
            for name in self.values[position]:
                a, b = self.values[position][name], second[position][name]
                if a.dtype.kind=='f':
                    np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-14)
                else:
                    np.testing.assert_array_equal(a, b)
        self.assertEqual(self.values[4], second[4])
        self.assertEqual(self.values[5], second[5])

    def test_sparse_observations_are_sampled_from_the_full_native_corruption(self):
        source, selected, acquisitions, sensor, events, metadata = self.values
        native_index = acquisitions['native_array_index']
        for name in ('gyro_white','gyro_spike','spike_mask','spike_vector_rad_s',
                     'active_spike_count'):
            np.testing.assert_array_equal(acquisitions[name], sensor[name][native_index])
        np.testing.assert_array_equal(sensor['gyro_white'], source['gyro'])
        np.testing.assert_array_equal(selected['sparse_gyro'], acquisitions['gyro_white'])
        np.testing.assert_array_equal(sensor['gyro_spike'][~sensor['spike_mask']],
                                      source['gyro'][~sensor['spike_mask']])
        self.assertFalse(acquisitions['spike_mask'][0])
        self.assertFalse(sensor['spike_mask'][0])
        self.assertEqual(metadata['spike_expected_arrivals_per_second'], 10.)
        self.assertGreater(len(events), len(acquisitions['time']))
        # These particular fixed-seed pulses include misses and hits. There
        # is no generator rejection/retry step to impose this condition.
        self.assertGreater(metadata['acquired_spiked_count'], 0)
        self.assertGreater(metadata['acquired_clean_count'], 1)
        self.assertTrue(any(not np.any((acquisitions['time']>=e['requested_onset_seconds'])
                            & (acquisitions['time']<e['requested_stop_seconds'])) for e in events))
        self.assertTrue(any(abs(e['requested_onset_seconds']-round(e['requested_onset_seconds']))>.01
                            for e in events))

    def test_logged_pulses_reconstruct_every_native_sample_including_overlaps(self):
        _, _, _, sensor, events, _ = self.values
        total = np.zeros_like(sensor['spike_vector_rad_s'])
        count = np.zeros(len(total), dtype=np.int64)
        for event in events:
            first, stop = event['first_native_array_index'], event['native_stop_array_index_exclusive']
            total[first:stop] += event['vector_body_rad_s']
            count[first:stop] += 1
            if stop>first:
                self.assertGreaterEqual(sensor['time'][first], event['requested_onset_seconds'])
                self.assertLess(sensor['time'][stop-1], event['applied_stop_seconds'])
        np.testing.assert_array_equal(total, sensor['spike_vector_rad_s'])
        np.testing.assert_array_equal(count, sensor['active_spike_count'])
        np.testing.assert_array_equal(count>0, sensor['spike_mask'])

    def test_existing_standardized_white_realization_is_reused_at_requested_sigma(self):
        source = self.values[0]
        original, _ = inputs._trajectory.generate_dataset(
            duration=5., sensor_rate=200., seed=20261113,
            motion_profile='three_frequency')
        for key in ('imu_time','imu_stamp_ns','imu_source_index',
                    'gyro_true_body_rad_s'):
            if key=='gyro_true_body_rad_s':
                # Vectorized analytic trig evaluation can differ by one
                # rounding unit across differently sized array allocations.
                np.testing.assert_allclose(source[key][:len(original[key])], original[key],
                                           rtol=1e-14, atol=1e-14)
            else:
                np.testing.assert_array_equal(source[key][:len(original[key])], original[key])
        for key in ('initial_quaternion_wxyz','gyro_bias_rad_s',
                    'inertia_body_kg_m2'):
            np.testing.assert_array_equal(source[key], original[key])
        np.testing.assert_array_equal(source['gyro_realized_noise_rad_s'][:1001],
                                      10*original['gyro_realized_noise_rad_s'])
        np.testing.assert_array_equal(source['generator_gyro_covariance_rad2_s2'],
                                      np.eye(3)*np.deg2rad(2.)**2)
        np.testing.assert_allclose(source['gyro']-source['gyro_true_body_rad_s']
                                  -source['gyro_bias_rad_s'],
                                  source['gyro_realized_noise_rad_s'], atol=1e-15)

    def test_predeclared_tail_is_clean_without_reselecting_the_poisson_realization(self):
        _, _, acquisitions, sensor, events, metadata = self.values
        self.assertTrue(np.all(~sensor['spike_mask'][sensor['time']>=15.]))
        self.assertTrue(np.all(~acquisitions['spike_mask'][acquisitions['time']>=15.]))
        self.assertEqual(metadata['clean_recovery_tail_seconds'], 5.)
        self.assertTrue(all(event['requested_onset_seconds']<15. for event in events))
        self.assertFalse(metadata['seeds_reselected_from_estimation_results'])
        self.assertTrue(metadata['clean_tail_is_a_predeclared_intervention'])


if __name__ == '__main__':
    unittest.main()
