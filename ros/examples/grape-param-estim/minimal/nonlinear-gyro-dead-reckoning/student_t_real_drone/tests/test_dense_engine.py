"""Independent clock, causal target, and final-tail tests for dense output."""
from pathlib import Path
import importlib.util
import sys
import unittest

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, multivariate_t


HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_real_dense_tests', HERE/'dense_engine.py')
dense = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dense
spec.loader.exec_module(dense)


class DensePredictionTests(unittest.TestCase):
    def fixture(self):
        observations = np.array([0., .043, .081])
        data = dict(time=np.array([0., .13]),
            sparse_gyro_time=observations,
            sparse_gyro=np.array([[.24, -.35, .15], [.251, -.354, .160], [.268, -.368, .170]]),
            initial_quaternion_wxyz=np.array([1., 0., 0., 0.]))
        physical = dict(inertia=np.diag([.045, .07, .1]),
                        torque=np.array([.016, -.021, .012]))
        return data, physical

    def test_every_original_observation_row_is_unchanged_and_tail_is_not_held(self):
        data, physical = self.fixture()
        settings = dict(order=7, max_step=.002, mode='nonlinear', student_df=4.,
                        refresh_seconds=.02)
        original = dense.engine.predict_full_rate(data, physical, **settings)
        result = dense.predict_dense(data, physical, output_dt=.01, **settings)
        rows = result['observation_row_indices']
        for key in dense.ROW_FIELDS:
            np.testing.assert_array_equal(result[key][rows], original[key], err_msg=key)
        self.assertEqual(result['time'][-1], data['time'][-1])
        self.assertEqual(result['event_side'][-1], 0)
        self.assertEqual(result['acquisition_index'][-1], 2)
        self.assertEqual(result['target_likelihood_factor_count'][-1], 2)
        last_observation = rows[-1]
        self.assertGreater(np.linalg.norm(result['omega_mean_rad_s'][-1]
            -result['omega_mean_rad_s'][last_observation]), .001)
        self.assertGreater(np.linalg.norm(result['rotation_mean'][-1]
            -result['rotation_mean'][last_observation]), .001)

    def test_future_reading_cannot_change_earlier_dense_predictions(self):
        data, physical = self.fixture()
        settings = dict(order=7, output_dt=.01, max_step=.002, refresh_seconds=.02)
        result = dense.predict_dense(data, physical, **settings)
        changed = dict(data, sparse_gyro=data['sparse_gyro'].copy())
        changed['sparse_gyro'][-1] += np.array([.9, -.6, .3])
        other = dense.predict_dense(changed, physical, **settings)
        earlier = result['time'] < data['sparse_gyro_time'][-1]
        np.testing.assert_array_equal(result['time'][earlier], other['time'][earlier])
        for key in dense.ROW_FIELDS:
            np.testing.assert_array_equal(result[key][earlier], other[key][earlier], err_msg=key)

    def test_final_tail_matches_independent_student_all_factor_joint_moments(self):
        data, physical = self.fixture()
        result = dense.predict_dense(data, physical, output_dt=.01, order=9,
                                     refresh_seconds=.02)
        proposal = len(result['refresh_acquisition_index'])-1
        center = result['refresh_proposal_mean_rad_s'][proposal]
        covariance = result['refresh_proposal_covariance_rad2_s2'][proposal]
        nodes, quadrature = dense.engine.gaussian_tensor_points(center, covariance, 9)
        query = np.r_[data['sparse_gyro_time'], data['time'][-1]]
        flow = dense.engine._adaptive.ControlledFlow(physical,
            data['initial_quaternion_wxyz'], *data['time'], max_step=.002)
        states = flow.evaluate(nodes, query)
        marginal = result['measurement_covariance_rad2_s2']
        raw = (np.log(quadrature)
            +multivariate_normal.logpdf(nodes, mean=data['sparse_gyro'][0], cov=marginal)
            -multivariate_normal.logpdf(nodes, mean=center, cov=covariance))
        for index in range(1, len(data['sparse_gyro_time'])):
            raw += multivariate_t.logpdf(states[index, :, :3]-data['sparse_gyro'][index],
                loc=np.zeros(3), shape=.5*marginal, df=4.)
        mass = np.exp(raw-logsumexp(raw))
        expected = dense.engine.summarize_joint(states[-1, :, 3:], states[-1, :, :3],
                                                mass, initial_nodes=nodes)
        for key in ('quaternion_second_moment', 'rotation_mean', 'rotation_covariance_colmajor',
                    'omega_mean_rad_s', 'omega_covariance_rad2_s2',
                    'rotation_omega_covariance_colmajor'):
            np.testing.assert_allclose(result[key][-1], expected[key], rtol=2e-8, atol=2e-10,
                                       err_msg=key)

    def test_gaussian_linear_tail_has_analytic_mean_and_covariance(self):
        data, physical = self.fixture()
        result = dense.predict_dense(data, physical, output_dt=.01, order=7,
            refresh_seconds=.02, student_df=None, mode='linear_forced')
        acceleration = np.linalg.solve(physical['inertia'], physical['torque'])
        included = result['acquisition_index']+1
        included = np.where(result['event_side']==-1, included-1, included)
        history_initial = (data['sparse_gyro']
            -data['sparse_gyro_time'][:, None]*acceleration)
        sums = np.vstack((np.zeros(3), np.cumsum(history_initial, axis=0)))
        means = sums[included]/included[:, None]+result['time'][:, None]*acceleration
        covariances = result['measurement_covariance_rad2_s2'][None]/included[:, None, None]
        np.testing.assert_allclose(result['omega_mean_rad_s'], means, rtol=0., atol=2e-12)
        np.testing.assert_allclose(result['omega_covariance_rad2_s2'], covariances,
                                   rtol=0., atol=2e-13)


if __name__ == '__main__':
    unittest.main()
