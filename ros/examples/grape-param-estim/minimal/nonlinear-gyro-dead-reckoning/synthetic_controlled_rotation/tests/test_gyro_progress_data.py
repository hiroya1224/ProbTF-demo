"""Acquisition-side retention and full-covariance display extraction checks."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('synthetic_gyro_progress_test', HERE/'gyro_progress_data.py')
progress = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = progress
spec.loader.exec_module(progress)


def fixture(linear=False):
    sigma_deg_s = 2.5
    sigma = np.deg2rad(sigma_deg_s)
    offsets = np.r_[np.sqrt(3.)*sigma*np.eye(3), -np.sqrt(3.)*sigma*np.eye(3)]
    weight = np.full(6, 1./6.)
    acquisition = dict(time=np.array([0., .5]), gyro=np.array([[.2, -.1, .3], [-.4, .7, .1]]),
                       source_index=np.array([0, 100]), stamp_ns=np.array([0, 500_000_000]))
    final_action = np.eye(3) if linear else np.array([[2., .4, 0.], [0., .5, .2], [.3, 0., 1.]])
    actions = [np.eye(3), .5*(np.eye(3)+final_action), final_action]
    means = [acquisition['gyro'][0]+np.array([t, -.2*t, .3*t]) for t in (0., .25, .5)]
    first_nodes = np.stack([mean+offsets@action.T for mean, action in zip(means, actions)])
    second_nodes = np.stack([acquisition['gyro'][1]+offsets,
                             acquisition['gyro'][1]+np.array([.2, .1, -.1])+offsets])
    first_cov = np.stack([sigma**2*action@action.T for action in actions])
    second_cov = np.stack([sigma**2*np.eye(3)]*2)
    prediction = dict(time=np.array([0., .25, .5, 1.]),
        omega_mean_rad_s=np.vstack((means[:2], acquisition['gyro'][1], acquisition['gyro'][1]+[.2, .1, -.1])),
        omega_covariance_rad2_s2=np.concatenate((first_cov[:2], second_cov)),
        block_0_time=np.array([0., .25, .5]), block_0_omega=first_nodes, block_0_weights=weight,
        block_1_time=np.array([.5, 1.]), block_1_omega=second_nodes, block_1_weights=weight)
    return prediction, acquisition, sigma_deg_s, first_cov[-1]


class GyroProgressTests(unittest.TestCase):
    def test_both_observation_sides_and_full_cross_covariance_are_retained(self):
        prediction, acquisition, sigma, before_covariance = fixture()
        result = progress.extract_progress(prediction, acquisition, sigma)
        arrays = result['arrays']
        np.testing.assert_array_equal(arrays['time'], [0., .25, .5, .5, 1.])
        np.testing.assert_array_equal(arrays['stage'], [2, 0, 1, 2, 0])
        np.testing.assert_array_equal(arrays['right_continuous_mask'], [True, True, False, True, True])
        np.testing.assert_allclose(arrays['covariance_rad2_s2'][2], before_covariance, atol=1e-17)
        self.assertNotEqual(arrays['covariance_rad2_s2'][2, 0, 1], 0.)
        event = result['events'][1]
        self.assertEqual(event['source_index'], 100)
        self.assertEqual(event['stamp_ns'], 500_000_000)
        self.assertLess(event['post_over_pre_width_ratio'][0], 1.)
        self.assertGreater(event['post_over_pre_width_ratio'][1], 1.)
        self.assertIsNone(result['events'][0]['before_covariance_rad2_s2'])
        self.assertTrue(result['metadata']['canonical_time_bitwise_equal'])
        self.assertLess(result['metadata']['canonical_mean_max_abs_difference_rad_s'], 3e-16)
        self.assertLess(result['metadata']['canonical_covariance_max_abs_difference_rad2_s2'], 1e-17)

    def test_linear_constant_covariance_does_not_acquire_artificial_waists(self):
        prediction, acquisition, sigma, _ = fixture(linear=True)
        result = progress.extract_progress(prediction, acquisition, sigma)
        expected = np.eye(3)*np.deg2rad(sigma)**2
        for value in result['arrays']['covariance_rad2_s2']:
            np.testing.assert_allclose(value, expected, atol=1e-17)
        np.testing.assert_allclose(result['events'][1]['post_over_pre_width_ratio'], np.ones(3), atol=1e-13)

    def test_saved_actual_nonlinear_case_matches_unchanged_canonical_moments(self):
        directory = HERE/'publication/smooth_rotation_5s/rate_0p5_hz'
        prediction = dict(np.load(directory/'sigma_2p5_deg_s/nonlinear.npz'))
        acquisition = dict(np.load(directory/'acquisitions.npz'))
        result = progress.extract_progress(prediction, acquisition, 2.5)
        self.assertEqual(result['metadata']['canonical_mean_max_abs_difference_rad_s'], 0.)
        self.assertEqual(result['metadata']['canonical_covariance_max_abs_difference_rad2_s2'], 0.)
        self.assertEqual(result['metadata']['retained_pre_observation_endpoints'], 2)
        np.testing.assert_allclose(result['events'][1]['after_covariance_rad2_s2'],
                                   np.eye(3)*np.deg2rad(2.5)**2, atol=1e-17)


if __name__ == '__main__':
    unittest.main()
