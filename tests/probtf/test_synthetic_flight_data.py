"""Physical and timing checks for the prescribed synthetic flight."""

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT/'ros/examples/probtf_orientation_demo/scripts/synthetic_flight_data.py'
SPEC = importlib.util.spec_from_file_location('synthetic_flight_data', SCRIPT)
flight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(flight)


class ZeroNoise:
    def normal(self, loc, scale, size):
        return np.zeros(size)


class RecordedNoise:
    def __init__(self):
        self.calls = []

    def normal(self, loc, scale, size):
        self.calls.append((loc, scale, size))
        return np.broadcast_to(scale * np.array([2.0, -1.0, 0.5]), size).copy()


class SyntheticFlightDataTest(unittest.TestCase):
    def test_default_clock_includes_endpoints_and_200_hz_intervals(self):
        arrays, settings = flight.generate_flight(rng=ZeroNoise())
        self.assertEqual(len(arrays['time_seconds']), 12001)
        self.assertEqual(arrays['time_seconds'][0], 0.0)
        self.assertEqual(arrays['time_seconds'][-1], 60.0)
        np.testing.assert_array_equal(arrays['time_seconds'], arrays['measurement_stamps'])
        np.testing.assert_allclose(np.diff(arrays['time_seconds']), 1.0/200.0, rtol=0.0, atol=1e-14)
        self.assertEqual(settings['effective_sample_rate_hz'], 200.0)
        json.dumps(settings, allow_nan=False)
        self.assertTrue(all(np.isfinite(value).all() for value in arrays.values()))

    def test_exact_right_end_angular_velocity_reconstructs_rotations(self):
        arrays, _ = flight.generate_flight(duration_s=10.0, sample_rate_hz=100.0, rng=ZeroNoise())
        rotations = Rotation.from_quat(arrays['truth_wxyz'][:, [1, 2, 3, 0]])
        increments = Rotation.from_rotvec(arrays['ideal_gyro_rad_s'][1:]
                                         * np.diff(arrays['measurement_stamps'])[:, None])
        reconstructed = rotations[:-1] * increments
        errors = (reconstructed.inv() * rotations[1:]).magnitude()
        np.testing.assert_allclose(errors, 0.0, rtol=0.0, atol=1e-14)
        np.testing.assert_array_equal(arrays['ideal_gyro_rad_s'][0], np.zeros(3))

    def test_acceleration_is_specific_force_with_analytic_world_motion(self):
        arrays, settings = flight.generate_flight(rng=ZeroNoise())
        rotations = Rotation.from_quat(arrays['truth_wxyz'][:, [1, 2, 3, 0]])
        force_world = rotations.apply(arrays['ideal_acceleration_m_s2'])
        np.testing.assert_allclose(force_world + settings['gravity_world_m_s2'],
                                   arrays['acceleration_world_m_s2'], rtol=0.0, atol=2e-14)
        position_derivative = np.gradient(arrays['position_m'], arrays['time_seconds'], axis=0)
        velocity_derivative = np.gradient(arrays['velocity_m_s'], arrays['time_seconds'], axis=0)
        np.testing.assert_allclose(position_derivative[1:-1], arrays['velocity_m_s'][1:-1],
                                   rtol=0.0, atol=3e-9)
        np.testing.assert_allclose(velocity_derivative[1:-1], arrays['acceleration_world_m_s2'][1:-1],
                                   rtol=0.0, atol=3e-10)

    def test_magnetic_truth_is_one_fixed_world_vector(self):
        arrays, settings = flight.generate_flight(duration_s=2.0, rng=ZeroNoise())
        rotations = Rotation.from_quat(arrays['truth_wxyz'][:, [1, 2, 3, 0]])
        field = rotations[arrays['magnetic_source_index']].apply(arrays['ideal_magnetic_raw'])
        expected = np.broadcast_to(settings['magnetic_field_world'], field.shape)
        np.testing.assert_allclose(field, expected, rtol=0.0, atol=5e-14)
        self.assertAlmostEqual(np.linalg.norm(settings['magnetic_reference_world_unit']), 1.0)

    def test_generation_and_estimation_covariances_are_isotropic_and_distinct(self):
        rng = RecordedNoise()
        _, settings = flight.generate_flight(duration_s=0.05, rng=rng)
        self.assertEqual(len(rng.calls), 3)
        for call, sensor in zip(rng.calls, ('angular_velocity', 'acceleration', 'magnetic')):
            estimated = settings['estimation_noise_variances'][sensor]
            generated = settings['generation_noise_variances'][sensor]
            ratio = settings['generation_to_estimation_variance_ratios'][sensor]
            self.assertEqual(ratio, 2.25 if sensor == 'magnetic' else 4.0)
            self.assertEqual(generated, ratio * estimated)
            np.testing.assert_array_equal(settings['estimation_noise_covariances'][sensor], estimated*np.eye(3))
            np.testing.assert_array_equal(settings['generation_noise_covariances'][sensor], generated*np.eye(3))
            self.assertEqual(call[0], 0.0)
            self.assertEqual(call[1], np.sqrt(generated))
            self.assertEqual(call[2], (4, 3) if sensor == 'magnetic' else (11, 3))

    def test_magnetic_publications_are_one_third_rate_with_original_source_clock(self):
        arrays, settings = flight.generate_flight(rng=ZeroNoise())
        indices = np.arange(0, 12001, 3)
        np.testing.assert_array_equal(np.flatnonzero(arrays['magnetic_observed']), indices)
        np.testing.assert_array_equal(arrays['magnetic_sample_indices'], indices)
        self.assertEqual(settings['magnetic_sample_count'], 4001)
        self.assertAlmostEqual(settings['sensor_sample_rates_hz']['magnetic'], 200.0/3.0)
        source = 3 * (np.arange(12001) // 3)
        np.testing.assert_array_equal(arrays['magnetic_source_index'], source)
        np.testing.assert_array_equal(arrays['magnetic_measurement_stamps'], arrays['time_seconds'][source])
        np.testing.assert_array_equal(arrays['magnetic_sample_stamps'], arrays['time_seconds'][indices])
        for key in ('magnetic_raw', 'magnetic_noise', 'ideal_magnetic_raw'):
            np.testing.assert_array_equal(arrays[key], arrays[key][source])

    def test_roll_pitch_motion_and_quieter_magnetic_noise_match_new_conditions(self):
        arrays, settings = flight.generate_flight(rng=ZeroNoise())
        rpy = Rotation.from_quat(arrays['truth_wxyz'][:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True)
        self.assertAlmostEqual(np.max(np.abs(rpy[:, 0])), 20.0, places=8)
        self.assertAlmostEqual(np.max(np.abs(rpy[:, 1])), 15.0, places=8)
        self.assertAlmostEqual(settings['generation_noise_variances']['magnetic'], 6.042895406840119/20.0)
        self.assertAlmostEqual(settings['estimation_noise_variances']['magnetic'], 0.6714328229822354/5.0)

    def test_measurements_reconstruct_from_ideal_saved_noise_and_separate_bias(self):
        arrays, settings = flight.generate_flight(duration_s=0.05, rng=RecordedNoise())
        np.testing.assert_array_equal(arrays['gyro_rad_s'], arrays['ideal_gyro_rad_s']
            + settings['gyro_bias_rad_s'] + arrays['gyro_noise'])
        np.testing.assert_array_equal(arrays['acceleration_m_s2'],
            arrays['ideal_acceleration_m_s2'] + arrays['acceleration_noise'])
        np.testing.assert_array_equal(arrays['magnetic_raw'],
            arrays['ideal_magnetic_raw'] + arrays['magnetic_noise'])
        self.assertFalse(settings['gyro_bias_subtracted'])
        self.assertEqual(settings['gyro_bias_rad_s'], [0.0, 0.0, 0.004])
        self.assertTrue(settings['random_measurement_generation'])
        self.assertFalse(settings['monte_carlo_inference'])

    def test_nondefault_clock_records_effective_rate_without_losing_endpoints(self):
        arrays, settings = flight.generate_flight(duration_s=0.107, sample_rate_hz=100.0, rng=ZeroNoise())
        self.assertEqual(len(arrays['time_seconds']), 12)
        self.assertEqual(arrays['time_seconds'][-1], 0.107)
        self.assertAlmostEqual(settings['effective_sample_rate_hz'], 11/0.107)


if __name__ == '__main__':
    unittest.main()
