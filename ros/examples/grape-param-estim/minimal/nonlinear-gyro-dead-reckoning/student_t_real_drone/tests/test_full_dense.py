"""Grouped causal dense output against unchanged updates and direct Bayes law."""
from pathlib import Path
import importlib.util
import sys
import unittest

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, multivariate_t


HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_real_full_dense_tests', HERE/'full_dense_engine.py')
dense = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dense
spec.loader.exec_module(dense)

JOINT_FIELDS = ('quaternion_second_moment', 'rotation_mean',
    'rotation_covariance_colmajor', 'omega_mean_rad_s',
    'omega_covariance_rad2_s2', 'rotation_omega_covariance_colmajor',
    'initial_omega_mean_rad_s', 'initial_omega_covariance_rad2_s2')


class FullDensePredictionTests(unittest.TestCase):
    def fixture(self):
        data = dict(time=np.array([0., .13]),
            sparse_gyro_time=np.array([0., .027, .043, .065, .081]),
            sparse_gyro=np.array([[.24, -.35, .15], [.247, -.352, .155],
                [.251, -.354, .160], [.260, -.360, .163], [.268, -.368, .170]]),
            initial_quaternion_wxyz=np.array([1., 0., 0., 0.]))
        physical = dict(inertia=np.diag([.045, .07, .1]),
                        torque=np.array([.016, -.021, .012]))
        settings = dict(order=7, max_step=.003, student_df=4., refresh_seconds=.07,
                        initial_dense_seconds=0., reanchor_displacement_sigma=None)
        return data, physical, settings

    def test_original_observation_rows_are_unchanged_and_dense_grouping_reuses_prefix(self):
        data, physical, settings = self.fixture()
        original = dense.engine.predict_full_rate(data, physical, **settings)
        result = dense.predict_full_dense(data, physical, output_dt=.01, **settings)
        rows = result['observation_row_indices']
        for key in dense.ROW_FIELDS:
            np.testing.assert_array_equal(result[key][rows], original[key], err_msg=key)
        self.assertEqual(result['time'][-1], data['time'][-1])
        self.assertEqual(result['event_side'][-1], 0)
        self.assertEqual(result['acquisition_index'][-1], 4)
        self.assertEqual(result['target_likelihood_factor_count'][-1], 4)
        self.assertLess(len(result['dense_group_records']), len(result['dense_interval_records']))
        last_observation = rows[-1]
        self.assertGreater(np.linalg.norm(result['omega_mean_rad_s'][-1]
            -result['omega_mean_rad_s'][last_observation]), .001)
        self.assertGreater(np.linalg.norm(result['rotation_mean'][-1]
            -result['rotation_mean'][last_observation]), .001)

    def test_grouped_predictions_match_interval_reference_to_finite_rk4_error(self):
        data, physical, settings = self.fixture()
        result = dense.predict_full_dense(data, physical, output_dt=.01, **settings)
        reference = dense.reference.predict_dense(data, physical, output_dt=.01, **settings)
        for key in dense.ROW_FIELDS:
            np.testing.assert_allclose(result[key], reference[key], rtol=1.e-7, atol=2.e-10,
                                       err_msg=key)
        maximum = max(float(np.max(np.abs(result[key]-reference[key]))) for key in JOINT_FIELDS)
        print('Grouped/interval dense maximum joint-moment RK4 difference:', maximum)

    def test_future_reading_cannot_change_earlier_dense_predictions(self):
        data, physical, settings = self.fixture()
        result = dense.predict_full_dense(data, physical, output_dt=.01, **settings)
        changed = dict(data, sparse_gyro=data['sparse_gyro'].copy())
        changed['sparse_gyro'][-1] += np.array([.9, -.6, .3])
        other = dense.predict_full_dense(changed, physical, output_dt=.01, **settings)
        earlier = result['time'] < data['sparse_gyro_time'][-1]
        for key in dense.ROW_FIELDS:
            np.testing.assert_array_equal(result[key][earlier], other[key][earlier], err_msg=key)

    def test_final_tail_contains_all_factors_and_full_joint_moments(self):
        data, physical, settings = self.fixture()
        result = dense.predict_full_dense(data, physical, output_dt=.01, **settings)
        refresh = len(result['refresh_acquisition_index'])-1
        center = result['refresh_proposal_mean_rad_s'][refresh]
        proposal_covariance = result['refresh_proposal_covariance_rad2_s2'][refresh]
        nodes, quadrature = dense.engine.gaussian_tensor_points(center, proposal_covariance, 7)
        # Directly propagate all candidate trajectories, independently compute
        # every Student-t factor, and retain the same joint q/omega moments.
        query = np.unique(np.r_[data['sparse_gyro_time'], result['dense_output_grid_time']])
        flow = dense.engine._adaptive.ControlledFlow(physical,
            data['initial_quaternion_wxyz'], *data['time'], max_step=settings['max_step'])
        states = flow.evaluate(nodes, query)
        marginal = result['measurement_covariance_rad2_s2']
        raw = (np.log(quadrature)
            +multivariate_normal.logpdf(nodes, mean=data['sparse_gyro'][0], cov=marginal)
            -multivariate_normal.logpdf(nodes, mean=center, cov=proposal_covariance))
        for index in range(1, len(data['sparse_gyro_time'])):
            row = np.searchsorted(query, data['sparse_gyro_time'][index])
            raw += multivariate_t.logpdf(states[row, :, :3]-data['sparse_gyro'][index],
                loc=np.zeros(3), shape=.5*marginal, df=4.)
        mass = np.exp(raw-logsumexp(raw))
        expected = dense.engine.summarize_joint(states[-1, :, 3:], states[-1, :, :3], mass,
                                                initial_nodes=nodes)
        for key in JOINT_FIELDS:
            np.testing.assert_allclose(result[key][-1], expected[key], rtol=1.e-7, atol=2.e-10,
                                       err_msg=key)

    def test_gaussian_linear_limit_and_grid_observation_coincidence(self):
        data, physical, settings = self.fixture()
        # Exactly coincident grid points retain only the existing pre/post
        # observations and never receive an extra likelihood or duplicate row.
        data['sparse_gyro_time'] = np.array([0., .02, .04, .06, .08])
        # Refresh each Gaussian observation so that a finite-order proposal
        # rule integrates its exact Gaussian posterior, rather than testing
        # undersampling of a repeatedly narrowed density on fixed nodes.
        settings.update(student_df=None, mode='linear_forced', initial_dense_seconds=.25)
        result = dense.predict_full_dense(data, physical, output_dt=.01, **settings)
        coincidence = np.isin(result['time'], data['sparse_gyro_time'])
        self.assertTrue(np.all(result['event_side'][coincidence] != 0))
        acceleration = np.linalg.solve(physical['inertia'], physical['torque'])
        included = result['acquisition_index']+1
        included = np.where(result['event_side'] == -1, included-1, included)
        history_initial = data['sparse_gyro']-data['sparse_gyro_time'][:, None]*acceleration
        sums = np.vstack((np.zeros(3), np.cumsum(history_initial, axis=0)))
        means = sums[included]/included[:, None]+result['time'][:, None]*acceleration
        covariances = result['measurement_covariance_rad2_s2'][None]/included[:, None, None]
        np.testing.assert_allclose(result['omega_mean_rad_s'], means, rtol=0., atol=2.e-12)
        np.testing.assert_allclose(result['omega_covariance_rad2_s2'], covariances,
                                   rtol=0., atol=2.e-13)


if __name__ == '__main__':
    unittest.main()
