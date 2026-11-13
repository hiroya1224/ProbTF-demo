"""Prediction data boundary and finite-law regressions for sparse gyro replay."""
from pathlib import Path
import importlib.util
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location('nonlinear_gyro_experiment_runner', HERE/'run.py')
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def input_fixture():
    initial = Rotation.from_euler('xyz', [.03, -.04, .1]).as_quat()[[3, 0, 1, 2]]
    time = np.linspace(10., 15., 51)
    return dict(time=np.array([10., 15.]), sparse_gyro_time=np.array([10., 12., 14.]),
        sparse_gyro=np.array([[.03, -.02, .01], [-.01, .04, -.02], [.02, .01, -.03]]),
        initial_quaternion_wxyz=initial,
        dense_imu_time=time.copy(), dense_gyro=np.zeros((len(time), 3)),
        imu_time=time.copy(), gyro=np.zeros((len(time), 3)),
        mocap_time=time.copy(), mocap_quaternion_xyzw=np.tile(initial[[1, 2, 3, 0]], (len(time), 1)))


def torque_fixture():
    return dict(inertia=np.diag([.2, .3, .5]),
                torque=lambda t: np.array([.002*np.cos(t-10.), -.001, .0005]))


class SparsePredictionDataChecks(unittest.TestCase):
    def test_dense_gyro_and_noninitial_mocap_are_not_prediction_inputs(self):
        original = input_fixture()
        changed = {key: value.copy() for key, value in original.items()}
        # Keep the declared sparse readings and initial alignment exactly the
        # same. Even unusable held-out dense/mocap data must not affect predict.
        for key in ('dense_gyro', 'dense_imu_time', 'gyro', 'imu_time',
                    'mocap_time', 'mocap_quaternion_xyzw'):
            changed[key][...] = np.nan
        arguments = dict(output_dt=.1, max_step=.02)
        first = runner.predict(original, torque_fixture(), 2., 'nonlinear', **arguments)
        second = runner.predict(changed, torque_fixture(), 2., 'nonlinear', **arguments)
        self.assertEqual(set(first), set(second))
        for key in first:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)

    def test_fresh_gyro_updates_rate_without_resetting_or_reapplying_attitude(self):
        data = input_fixture()
        predicted = runner.predict(data, torque_fixture(), 2., 'nonlinear',
                                   output_dt=.1, max_step=.02)
        prior = np.outer(data['initial_quaternion_wxyz'], data['initial_quaternion_wxyz'])
        for block, observation in enumerate(data['sparse_gyro_time']):
            hit = np.flatnonzero(predicted['time'] == observation)
            self.assertEqual(len(hit), 1)
            index = hit[0]
            np.testing.assert_allclose(predicted['omega_mean_rad_s'][index], data['sparse_gyro'][block], atol=1e-16)
            np.testing.assert_allclose(predicted['omega_covariance_rad2_s2'][index],
                                      np.eye(3)*np.deg2rad(2.)**2, atol=1e-17)
            parents = predicted[f'block_{block}_parent_wxyz']
            weights = predicted[f'block_{block}_parent_weight']
            expected = np.einsum('k,ki,kj->ij', weights, parents, parents)
            np.testing.assert_allclose(predicted['quaternion_second_moment'][index], expected, atol=2e-15)
            if block:
                self.assertGreater(np.linalg.norm(expected-prior), .01)
            np.testing.assert_allclose(predicted[f'block_{block}_delta_wxyz'][0],
                                      np.tile([1., 0., 0., 0.], (6, 1)), atol=0.)

    def test_saved_moments_and_rpy_spread_use_the_same_complete_finite_law(self):
        data = input_fixture()
        predicted = runner.predict(data, torque_fixture(), 2., 'nonlinear',
                                   output_dt=.1, max_step=.02)
        expected_branches = (6, 36, 216)
        for block in range(3):
            parents = predicted[f'block_{block}_parent_wxyz']
            parent_weights = predicted[f'block_{block}_parent_weight']
            delta = predicted[f'block_{block}_delta_wxyz']
            times = predicted[f'block_{block}_time']
            weights = np.repeat(parent_weights/6., 6)
            self.assertEqual(len(weights), expected_branches[block])
            for k in (0, len(times)//2, len(times)-1):
                index = int(np.flatnonzero(predicted['time'] == times[k])[0])
                points = runner.quaternion_multiply(parents[:, None, :], delta[k, None, :, :]).reshape(-1, 4)
                second = np.einsum('k,ki,kj->ij', weights, points, points)
                np.testing.assert_allclose(second, predicted['quaternion_second_moment'][index], atol=3e-15)
                angles = Rotation.from_quat(points[:, [1, 2, 3, 0]]).as_euler('xyz')
                center = predicted['rpy_mean_rad'][index]
                angles = center+(angles-center+np.pi)%(2.*np.pi)-np.pi
                mean = weights@angles
                centered = angles-mean
                covariance = np.einsum('k,ki,kj->ij', weights, centered, centered)
                np.testing.assert_allclose(mean, center, atol=3e-15)
                np.testing.assert_allclose(covariance, predicted['rpy_covariance_rad2'][index], atol=3e-16)
        self.assertEqual(len(predicted['final_law_weights']), 216)
        np.testing.assert_allclose(predicted['final_law_weights'].sum(), 1., atol=2e-15)

    def test_full_turn_display_is_continuous_but_errors_are_periodic(self):
        time = np.linspace(0., 1., 13)
        angle = 2.*np.pi*time
        rotations = Rotation.from_rotvec(np.column_stack((angle, 0.*angle, 0.*angle)))
        quaternions = rotations.as_quat()[:, [3, 0, 1, 2]]
        moments = np.einsum('ti,tj->tij', quaternions, quaternions)
        _mode, mean, covariance, _explicit = runner.finite_law_statistics(
            quaternions[:, None, :], np.ones(1), moments, np.zeros(3))
        np.testing.assert_allclose(mean[:, 0], angle, atol=2e-15)
        data = dict(mocap_time=time, mocap_quaternion_xyzw=np.tile([0., 0., 0., 1.], (len(time), 1)),
                    initial_quaternion_wxyz=np.array([1., 0., 0., 0.]))
        result = dict(time=time, rpy_mean_rad=mean, rpy_covariance_rad2=covariance,
                      core_product_vs_enumeration_max_abs=np.array([0.]))
        errors, scores = runner.errors(data, result)
        np.testing.assert_allclose(errors['error_rpy_rad'][-1], np.zeros(3), atol=1e-15)
        self.assertLess(scores['geodesic_final_deg'], 1e-12)
        self.assertGreater(scores['geodesic_rmse_deg'], 90.)


if __name__ == '__main__':
    unittest.main()
