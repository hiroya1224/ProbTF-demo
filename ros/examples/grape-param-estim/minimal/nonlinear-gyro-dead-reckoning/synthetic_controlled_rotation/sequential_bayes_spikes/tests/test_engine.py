"""Independent checks of sequential likelihoods, physical support and causality.

The linear controlled case has an analytic Gaussian posterior.  It detects
measurement replacement or repeated use of the first reading independently
of the nonlinear demonstration's visual appearance.
"""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation


PATH = Path(__file__).resolve().parents[1]/'engine.py'
SPEC = importlib.util.spec_from_file_location('sequential_bayes_engine_tested', PATH)
ENGINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ENGINE
SPEC.loader.exec_module(ENGINE)


class DeterministicJointBayesTests(unittest.TestCase):
    def test_tensor_rule_preserves_non_diagonal_prior_mean_and_covariance(self):
        mean = np.array([.2, -.1, .05])
        root = np.array([[.08, 0., 0.], [.025, .06, 0.], [-.02, .01, .04]])
        covariance = root@root.T
        nodes, weights = ENGINE.gaussian_tensor_points(mean, covariance, 5)
        self.assertEqual(nodes.shape, (5**3, 3))
        self.assertEqual(weights.shape, (5**3,))
        self.assertTrue(np.all(weights > 0.))
        self.assertAlmostEqual(float(weights.sum()), 1., places=14)
        actual_mean = weights@nodes
        centered = nodes-actual_mean
        actual_covariance = np.einsum('s,si,sj->ij', weights, centered, centered)
        np.testing.assert_allclose(actual_mean, mean, rtol=2e-14, atol=2e-16)
        np.testing.assert_allclose(actual_covariance, covariance, rtol=2e-14, atol=2e-16)

    def test_sequential_likelihood_product_counts_each_reading_once(self):
        nodes, weights = ENGINE.gaussian_tensor_points(np.zeros(3), np.eye(3)*.04, 7)
        first = np.array([.1, -.05, .08])
        second = np.array([-.02, .09, .12])
        covariance = np.array([[.03, .005, 0.], [.005, .05, .002], [0., .002, .04]])
        first_log, first_evidence = ENGINE.bayes_log_weight_update(
            np.log(weights), nodes, first, covariance)
        final_log, second_evidence = ENGINE.bayes_log_weight_update(
            first_log, nodes, second, covariance)

        def likelihood(observation):
            difference = nodes-observation
            squared = np.einsum('si,is->s', difference,
                                 np.linalg.solve(covariance, difference.T))
            return -.5*(squared+3*np.log(2*np.pi)+np.linalg.slogdet(covariance)[1])

        direct = np.log(weights)+likelihood(first)+likelihood(second)
        direct_normalizer = logsumexp(direct)
        np.testing.assert_allclose(final_log, direct-direct_normalizer,
                                   rtol=0., atol=2e-13)
        self.assertAlmostEqual(first_evidence+second_evidence, direct_normalizer,
                               places=12)
        self.assertAlmostEqual(float(logsumexp(final_log)), 0., places=13)

    def test_joint_summary_uses_the_same_physical_quaternions_and_full_cross(self):
        omega, weights = ENGINE.gaussian_tensor_points(
            np.array([.1, -.2, .15]), np.eye(3)*.01, 5)
        rotations = Rotation.from_rotvec(.3*omega)
        quaternion = rotations.as_quat()[:, [3, 0, 1, 2]]
        result = ENGINE.summarize_joint(quaternion, omega, weights)
        matrices = rotations.as_matrix()
        vectors = matrices.transpose(0, 2, 1).reshape(-1, 9)
        mean_rotation = np.einsum('s,sij->ij', weights, matrices)
        mean_vector = weights@vectors
        mean_omega = weights@omega
        centered_rotation = vectors-mean_vector
        centered_omega = omega-mean_omega
        covariance_rotation = np.einsum('s,si,sj->ij', weights,
                                         centered_rotation, centered_rotation)
        covariance_rate = np.einsum('s,si,sj->ij', weights,
                                     centered_omega, centered_omega)
        cross = np.einsum('s,si,sj->ij', weights,
                          centered_rotation, centered_omega)
        second_quaternion = np.einsum('s,si,sj->ij', weights, quaternion, quaternion)
        np.testing.assert_allclose(result['quaternion_second_moment'], second_quaternion,
                                   rtol=2e-13, atol=2e-15)
        np.testing.assert_allclose(result['rotation_mean'], mean_rotation,
                                   rtol=2e-13, atol=2e-15)
        np.testing.assert_allclose(result['rotation_covariance_colmajor'], covariance_rotation,
                                   rtol=2e-12, atol=2e-15)
        np.testing.assert_allclose(result['omega_covariance_rad2_s2'], covariance_rate,
                                   rtol=2e-12, atol=2e-15)
        np.testing.assert_allclose(result['rotation_omega_covariance_colmajor'], cross,
                                   rtol=2e-12, atol=2e-15)
        self.assertGreater(float(np.linalg.norm(cross)), .001)
        joint_covariance = np.block([[covariance_rotation, cross],
                                     [cross.T, covariance_rate]])
        self.assertGreaterEqual(float(np.linalg.eigvalsh(joint_covariance)[0]), -1e-14)
        self.assertAlmostEqual(float(np.trace(second_quaternion)), 1., places=14)


class SequentialPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sigma_deg_s = 5.
        cls.sigma = np.deg2rad(cls.sigma_deg_s)
        cls.observation_times = np.array([0., .01, .02])
        cls.initial_mean = np.array([.12, -.08, .04])
        # Zero input makes the linear model's exact omega(t) equal to omega0.
        # Slightly unequal observations also test the posterior mean.
        cls.white = cls.initial_mean+cls.sigma*np.array(
            [[0., 0., 0.], [.2, -.1, .15], [-.1, .15, -.2]])
        changed = cls.white.copy()
        changed[-1] += np.array([.08, -.04, .06])
        data = dict(time=np.array([0., .03]),
                    initial_quaternion_wxyz=np.array([1., 0., 0., 0.]),
                    sparse_gyro_time=cls.observation_times,
                    sparse_gyro=cls.white)
        profile = dict(inertia=np.diag([.045, .070, .100]),
                       torque=lambda _stamp: np.zeros(3))
        cls.predictions = ENGINE.predict_joint(
            data, profile, sigma_deg_s=cls.sigma_deg_s, order=21,
            output_dt=.005, max_step=.005,
            observation_sets={'white':cls.white, 'changed':changed},
            mode='linear_forced')

    def post_row(self, prediction, index):
        selected = np.flatnonzero((prediction['event_side'] == 1)
                                  & (prediction['acquisition_index'] == index))
        self.assertEqual(len(selected), 1)
        return int(selected[0])

    def test_first_reading_initializes_once_without_double_counting(self):
        prediction = self.predictions['white']
        row = self.post_row(prediction, 0)
        self.assertEqual(float(prediction['time'][row]), 0.)
        np.testing.assert_allclose(prediction['omega_mean_rad_s'][row], self.white[0],
                                   rtol=2e-12, atol=2e-15)
        np.testing.assert_allclose(prediction['omega_covariance_rad2_s2'][row],
                                   np.eye(3)*self.sigma**2, rtol=2e-12, atol=2e-15)

    def test_linear_case_matches_analytic_mean_and_accumulating_covariance(self):
        prediction = self.predictions['white']
        for index in (1, 2):
            row = self.post_row(prediction, index)
            expected_mean = self.white[:index+1].mean(axis=0)
            expected_covariance = np.eye(3)*self.sigma**2/(index+1)
            # This bound checks deterministic order-21 quadrature against
            # an independently derived Gaussian solution.  It is a test
            # assertion, not an inference acceptance or stopping threshold.
            np.testing.assert_allclose(prediction['omega_mean_rad_s'][row], expected_mean,
                                       rtol=0., atol=2e-6)
            np.testing.assert_allclose(prediction['omega_covariance_rad2_s2'][row],
                                       expected_covariance, rtol=8e-5, atol=3e-9)
        self.assertLess(float(np.trace(prediction['omega_covariance_rad2_s2'][row])),
                        1.01*self.sigma**2)

    def test_prior_and_posterior_are_retained_at_the_same_actual_time(self):
        prediction = self.predictions['white']
        for index in (1, 2):
            selected = np.flatnonzero(prediction['acquisition_index'] == index)
            np.testing.assert_array_equal(prediction['event_side'][selected], [-1, 1])
            np.testing.assert_array_equal(prediction['time'][selected],
                                           np.repeat(self.observation_times[index], 2))
            prior, posterior = selected
            self.assertLess(float(np.trace(prediction['omega_covariance_rad2_s2'][posterior])),
                            float(np.trace(prediction['omega_covariance_rad2_s2'][prior])))
            self.assertGreater(float(np.linalg.norm(
                prediction['rotation_omega_covariance_colmajor'][prior])), 0.)
            self.assertLess(float(np.trace(prediction['rotation_covariance_colmajor'][posterior])),
                            float(np.trace(prediction['rotation_covariance_colmajor'][prior])))

    def test_future_changed_reading_cannot_change_previous_online_output(self):
        white, changed = (self.predictions[name] for name in ('white', 'changed'))
        np.testing.assert_array_equal(white['time'], changed['time'])
        cutoff = self.observation_times[-1]
        before = ((white['time'] < cutoff)
                  | ((white['time'] == cutoff) & (white['event_side'] == -1)))
        for key in ('quaternion_second_moment', 'rotation_mean',
                    'rotation_covariance_colmajor', 'omega_mean_rad_s',
                    'omega_covariance_rad2_s2', 'rotation_omega_covariance_colmajor'):
            np.testing.assert_array_equal(white[key][before], changed[key][before])
        post = self.post_row(white, 2)
        self.assertGreater(float(np.linalg.norm(white['omega_mean_rad_s'][post]
                                                -changed['omega_mean_rad_s'][post])), .01)

    def test_known_control_is_retained_on_the_original_clock(self):
        start = 1.2
        times = start+self.observation_times
        acceleration0 = np.array([.4, -.25, .1])
        acceleration_slope = np.array([.1, .05, -.08])

        def rate_increment(stamp):
            return (acceleration0*(stamp-start)
                    +.5*acceleration_slope*(stamp**2-start**2))

        readings = self.white+np.stack([rate_increment(stamp) for stamp in times])
        inertia = np.diag([.045, .070, .100])
        profile = dict(inertia=inertia,
                       torque=lambda stamp: inertia@(acceleration0+acceleration_slope*stamp))
        data = dict(time=np.array([start, start+.03]),
                    initial_quaternion_wxyz=np.array([1., 0., 0., 0.]),
                    sparse_gyro_time=times, sparse_gyro=readings)
        prediction = ENGINE.predict_joint(
            data, profile, sigma_deg_s=self.sigma_deg_s, order=21,
            output_dt=.005, max_step=.005, mode='linear_forced')['default']
        for index in (1, 2):
            row = self.post_row(prediction, index)
            expected_mean = self.white[:index+1].mean(axis=0)+rate_increment(times[index])
            np.testing.assert_allclose(prediction['omega_mean_rad_s'][row], expected_mean,
                                       rtol=0., atol=2e-6)
            np.testing.assert_allclose(prediction['omega_covariance_rad2_s2'][row],
                                       np.eye(3)*self.sigma**2/(index+1),
                                       rtol=8e-5, atol=3e-9)

    def test_nonlinear_prediction_retains_the_euler_gyroscopic_term(self):
        mean = np.array([1., 2., 3.])
        inertia = np.diag([1., 2., 3.])
        torque = np.array([.4, -.3, .2])
        duration = 1e-5
        data = dict(time=np.array([0., duration]),
                    initial_quaternion_wxyz=np.array([1., 0., 0., 0.]),
                    sparse_gyro_time=np.array([0.]), sparse_gyro=mean[None, :])
        prediction = ENGINE.predict_joint(
            data, dict(inertia=inertia, torque=torque),
            sigma_deg_s=self.sigma_deg_s, order=5,
            output_dt=duration, max_step=duration, mode='nonlinear')['default']
        # For this diagonal inertia and independent isotropic initial rate,
        # E[omega cross Jomega] equals mean cross Jmean exactly.  The first
        # time derivative therefore has an independent closed-form check.
        expected_derivative = np.linalg.solve(inertia, torque-np.cross(mean, inertia@mean))
        np.testing.assert_allclose(prediction['omega_mean_rad_s'][-1],
                                   mean+duration*expected_derivative,
                                   rtol=0., atol=2e-8)


if __name__ == '__main__':
    unittest.main()
