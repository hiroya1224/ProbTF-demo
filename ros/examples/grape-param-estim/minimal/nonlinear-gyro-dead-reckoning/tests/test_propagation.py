"""Physical/measure checks for the new six-point nonlinear increment map."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from propagation import (gaussian_six_points, integrate_six_point_increment,
                         quaternion_multiply, propagate_orientation_second_moments,
                         compose_product_cubature, weighted_rpy_summary)
from probtf_estimators.orientation_imu import delta_quaternion_second_moment


class NonlinearIncrementChecks(unittest.TestCase):
    def test_six_points_recover_the_full_gaussian_first_and_second_moments(self):
        mean = np.array([.2, -.3, .5])
        root = np.array([[.1, 0., 0.], [.03, .2, 0.], [-.02, .01, .15]])
        covariance = root@root.T
        points, weights = gaussian_six_points(mean, covariance)
        np.testing.assert_allclose(weights@points, mean, atol=1e-16)
        centered = points-mean
        np.testing.assert_allclose(np.einsum('s,si,sj->ij', weights, centered, centered), covariance,
                                   atol=1e-16)
        np.testing.assert_allclose(points[:3]-mean, -(points[3:]-mean), atol=1e-16)

    def test_free_principal_axis_rotation_has_the_exact_constant_rate_limit(self):
        times = np.linspace(0., 1., 11)
        result = integrate_six_point_increment([0., 0., .6], np.zeros((3, 3)),
                    times, np.zeros(3), np.diag([.2, .3, .5]), max_step=.002)
        expected = Rotation.from_rotvec(times[:, None]*[0., 0., .6]).as_quat()[:, [3, 0, 1, 2]]
        np.testing.assert_allclose(result['delta'][:, 0], expected, atol=2e-14)
        np.testing.assert_allclose(result['omega'], np.broadcast_to([0., 0., .6], (11, 6, 3)), atol=0.)

    def test_constant_torque_spherical_inertia_matches_analytic_acceleration(self):
        times = np.linspace(2., 3., 11)
        result = integrate_six_point_increment(np.zeros(3), np.zeros((3, 3)),
                    times, [0., 0., .04], .4*np.eye(3), max_step=.002)
        elapsed = times-times[0]
        np.testing.assert_allclose(result['omega'][:, 0, 2], .1*elapsed, atol=1e-14)
        expected = Rotation.from_rotvec(np.column_stack((0.*elapsed, 0.*elapsed, .05*elapsed**2))).as_quat()[:, [3, 0, 1, 2]]
        np.testing.assert_allclose(result['delta'][:, 0], expected, atol=2e-14)

    def test_free_unequal_inertia_preserves_energy_and_world_angular_momentum(self):
        tensor = np.diag([.2, .3, .5])
        result = integrate_six_point_increment([.5, .7, .9], np.diag([.01, .02, .03]),
                    np.linspace(0., 1., 21), [0., 0., 0.], tensor, max_step=.002)
        omega = result['omega']
        energy = .5*np.einsum('tsi,ij,tsj->ts', omega, tensor, omega)
        np.testing.assert_allclose(energy, np.broadcast_to(energy[0], energy.shape), atol=2e-13)
        rotations = Rotation.from_quat(result['delta'].reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(21, 6, 3, 3)
        world_momentum = np.einsum('tsij,jk,tsk->tsi', rotations, tensor, omega)
        np.testing.assert_allclose(world_momentum, np.broadcast_to(world_momentum[0], world_momentum.shape), atol=2e-13)
        self.assertGreater(np.linalg.norm(result['covariance_delta_second_omega'][-1]), 0.)

    def test_constant_comparator_matches_existing_core_increment_cubature(self):
        mean, covariance = np.array([.1, -.2, .3]), .04*np.eye(3)
        result = integrate_six_point_increment(mean, covariance, [0., .2, .8],
                    [9., 8., 7.], np.diag([.2, .3, .5]), mode='constant')
        for index, time in enumerate(result['times']):
            np.testing.assert_allclose(result['M2delta'][index],
                 delta_quaternion_second_moment(mean, time, covariance), atol=2e-16)
        self.assertEqual(result['nonlinear_rk4_steps'], 0)

    def test_linear_forced_omits_only_the_gyroscopic_term(self):
        times = np.linspace(0., .5, 11)
        mean = np.array([.3, .4, .5])
        tensor = np.diag([.2, .3, .5])
        torque = np.array([.01, -.02, .03])
        linear = integrate_six_point_increment(mean, np.zeros((3, 3)), times,
                    torque, tensor, mode='linear_forced')
        nonlinear = integrate_six_point_increment(mean, np.zeros((3, 3)), times,
                    torque, tensor, mode='nonlinear')
        expected = mean + times[:, None]*np.linalg.solve(tensor, torque)
        np.testing.assert_allclose(linear['omega'][:, 0], expected, atol=1e-14)
        self.assertGreater(np.linalg.norm(nonlinear['omega'][-1]-linear['omega'][-1]), .01)

    def test_core_product_moment_matches_all_finite_product_nodes(self):
        parents = Rotation.from_rotvec([[.1, 0., 0.], [0., -.2, .1]]).as_quat()[:, [3, 0, 1, 2]]
        parent_weights = np.array([.3, .7])
        prior = np.einsum('s,si,sj->ij', parent_weights, parents, parents)
        result = integrate_six_point_increment([.1, .2, .3], .01*np.eye(3), [0., .3],
                    [.03, -.02, .01], np.diag([.2, .3, .5]), max_step=.002)
        moments = propagate_orientation_second_moments(prior, result['M2delta'])
        for index in range(2):
            points, weights = compose_product_cubature(parents, parent_weights,
                                  result['delta'][index], result['weights'])
            self.assertEqual(len(points), 12)
            actual = np.einsum('s,si,sj->ij', weights, points, points)
            np.testing.assert_allclose(moments[index], actual, atol=3e-16)

    def test_rpy_summary_is_the_weighted_moment_of_the_same_finite_law(self):
        angles = np.array([[.1, -.2, np.pi-.01], [.2, -.1, -np.pi+.02]])
        points = Rotation.from_euler('xyz', angles).as_quat()[:, [3, 0, 1, 2]]
        weights = np.array([.25, .75])
        summary = weighted_rpy_summary(points, weights, reference_rpy=[0., 0., np.pi])
        expected = angles.copy(); expected[1, 2] += 2.*np.pi
        np.testing.assert_allclose(summary['mean'], weights@expected, atol=1e-15)
        centered = expected-weights@expected
        np.testing.assert_allclose(summary['covariance'],
            np.einsum('s,si,sj->ij', weights, centered, centered), atol=1e-16)


if __name__ == '__main__':
    unittest.main()
