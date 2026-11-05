"""Deterministic moment oracles for the shared-gyro-increment prediction."""
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ros/examples/probtf_orientation_demo/scripts'))
import translational_prediction as prediction


def fixed_prior():
    # Four finite-law atoms are an exact algebraic test oracle, not Monte Carlo
    # inference samples or a candidate cloud used by the implementation.
    rotations = Rotation.from_euler('xyz', [[0, 0, 0], [30, -25, 14],
        [-22, 17, -32], [14, 29, 43]], degrees=True)
    weights = np.array([.35, .25, .20, .20])
    qs = rotations.as_quat()[:, [3, 0, 1, 2]]
    gravity = rotations.inv().apply(np.tile([0., 0., -9.80665], (4, 1)))
    magnetic = rotations.inv().apply(np.tile([.8, .2, .56], (4, 1)))
    eh = weights @ gravity; eb = weights @ magnetic
    ch = np.einsum('n,ni,nj->ij', weights, gravity-eh, gravity-eh)
    nh = np.einsum('n,ni,nj->ij', weights, gravity, gravity)
    nb = np.einsum('n,ni,nj->ij', weights, magnetic, magnetic)
    mq = np.einsum('n,ni,nj->ij', weights, qs, qs)
    intercept = np.array([.2, -.3, .1])
    f = np.array([[.02, -.01, .03], [.01, .04, -.02], [-.03, .01, .05]])
    s = np.array([[.05, .002, -.001], [.002, .04, .003], [-.001, .003, .06]])
    ma = intercept + f @ eh; ca = s + f @ ch @ f.T; cah = f @ ch
    return dict(Mq=mq, Eh=eh, Nh=nh, Eb=eb, Nb=nb, mean_a=ma,
        cov_a=ca, cov_ah=cah, cov_hg=ch), (intercept, f, s)


class TranslationalPredictionTests(unittest.TestCase):
    def test_zero_interval_is_exact_identity_without_an_inverse(self):
        sigma = np.zeros((3, 3))
        d = prediction.gyro_increment([.2, -.1, .3], 0., sigma)
        np.testing.assert_array_equal(d['M2delta'], np.diag([1., 0., 0., 0.]))
        x = np.arange(9.).reshape(3, 3)
        np.testing.assert_array_equal(d['average_transported_matrix'](x), x)
        np.testing.assert_array_equal(d['uncertainty_center']([1, 2, 3], [4, 5, 6]),
                                      np.zeros((3, 3)))

    def test_deterministic_increment_rotates_all_joint_moments_together(self):
        prior, _ = fixed_prior()
        gyro = np.array([.23, -.31, .12]); dt = .17; beta = .93
        process = np.diag([.002, .003, .004])
        result = prediction.predict_joint_moments(**prior, gyro=gyro, dt=dt,
            Sigma=np.zeros((3, 3)), beta=beta, Q=process)
        d = Rotation.from_rotvec(gyro*dt).as_matrix()
        np.testing.assert_allclose(result['Ehg'], d.T @ prior['Eh'], atol=2e-15)
        np.testing.assert_allclose(result['Ngg'], d.T @ prior['Nh'] @ d, atol=2e-14)
        np.testing.assert_allclose(result['Ehb'], d.T @ prior['Eb'], atol=2e-16)
        np.testing.assert_allclose(result['Nbb'], d.T @ prior['Nb'] @ d, atol=2e-16)
        np.testing.assert_allclose(result['mean_a'], beta*d.T @ prior['mean_a'], atol=2e-16)
        np.testing.assert_allclose(result['cov_a'], beta**2*d.T @ prior['cov_a'] @ d+process, atol=2e-16)
        np.testing.assert_allclose(result['cov_ah'], beta*d.T @ prior['cov_ah'] @ d, atol=2e-15)
        q = result['delta']['mean_quaternion']
        # Right multiplication q_old*delta, evaluated independently on each atom.
        from compare_madgwick_rolling import quaternion_product
        eig, vectors = np.linalg.eigh(prior['Mq'])
        expected = sum(eig[i]*np.outer(quaternion_product(vectors[:, i], q),
                                      quaternion_product(vectors[:, i], q)) for i in range(4))
        np.testing.assert_allclose(result['Mq'], expected, atol=3e-16)

    def test_deterministic_prediction_preserves_conditional_family(self):
        prior, (m,f,s) = fixed_prior()
        gyro = np.array([.23,-.31,.12]);dt=.17;beta=.93
        process = np.diag([.002,.003,.004])
        result = prediction.predict_joint_moments(**prior,gyro=gyro,dt=dt,
            Sigma=np.zeros((3,3)),beta=beta,Q=process)
        parameters = prediction.conditional_parameters(result['mean_a'],result['cov_a'],
            result['Ehg'],result['cov_hg'],result['cov_ah'])
        d = Rotation.from_rotvec(gyro*dt).as_matrix()
        np.testing.assert_allclose(parameters['m'],beta*d.T@m,atol=2e-15)
        np.testing.assert_allclose(parameters['F'],beta*d.T@f@d,atol=2e-15)
        np.testing.assert_allclose(parameters['S'],beta**2*d.T@s@d+process,atol=2e-16)

    def test_same_random_increment_creates_cross_covariance(self):
        h = np.array([1., -2., -9.]); scale = .3; beta = .8
        prior = dict(Mq=np.diag([1.,0.,0.,0.]), Eh=h, Nh=np.outer(h,h),
                     Eb=np.array([.8,.2,.5]), Nb=np.outer([.8,.2,.5],[.8,.2,.5]),
                     mean_a=scale*h, cov_a=np.zeros((3,3)),
                     cov_ah=np.zeros((3,3)), cov_hg=np.zeros((3,3)))
        result = prediction.predict_joint_moments(**prior, gyro=[.1,-.2,.3], dt=.1,
                     Sigma=np.array([[.3,.04,.02],[.04,.2,-.01],[.02,-.01,.4]]),
                     beta=beta,Q=np.zeros((3,3)))
        self.assertGreater(np.linalg.norm(result['cov_hg']), .05)
        np.testing.assert_allclose(result['cov_ah'], beta*scale*result['cov_hg'], atol=5e-17)
        np.testing.assert_allclose(result['cov_a'], (beta*scale)**2*result['cov_hg'], atol=2e-17)
        # Independent rotations of a and h would instead give zero cross here.
        self.assertGreater(np.linalg.norm(result['cov_ah']), .01)

    def test_second_fourth_moment_contracts_and_rotation_orthogonality(self):
        sigma = np.array([[.30,.04,.02],[.04,.20,-.01],[.02,-.01,.40]])
        delta = prediction.gyro_increment([.2,-.3,.1], .1, sigma)
        np.testing.assert_allclose(np.einsum('ijkk->ij', delta['noise_M4']),
                                   delta['noise_M2'], atol=2e-9)
        np.testing.assert_allclose(np.trace(delta['M2delta']),1.,atol=3e-16)
        np.testing.assert_allclose(delta['average_transported_matrix'](np.eye(3)),
                                   np.eye(3),atol=3e-9)
        np.testing.assert_allclose(delta['noise_parameter'][1:,1:],
                                   -2*np.linalg.inv(sigma*.1**2),rtol=3e-16,atol=2e-13)
        self.assertNotEqual(delta['noise_parameter'][1,2],0.)

    def test_transport_and_center_use_the_same_fourth_moment(self):
        delta = prediction.gyro_increment([.1,-.2,.3], .2, np.diag([.3,.2,.4]))
        a, b = np.array([1.,2.,3.]), np.array([-.2,.5,-.3])
        direct = delta['average_transported_matrix'](np.outer(a,b))
        direct -= np.outer(delta['mean_rotation'].T@a,delta['mean_rotation'].T@b)
        np.testing.assert_allclose(delta['uncertainty_center'](a,b),direct,atol=2e-15)

    def test_tiny_noise_keeps_nonzero_rotation_uncertainty(self):
        delta = prediction.gyro_increment([0.,0.,0.], .005,
                                          np.diag([1e-6,2e-6,3e-6]))
        value = delta['uncertainty_center']([0.,0.,-9.80665],[0.,0.,-9.80665])
        self.assertGreater(value[0,0],0.)
        self.assertGreater(value[1,1],0.)
        self.assertGreater(value[2,2],0.)
        np.testing.assert_allclose(value[:2,:2],
            9.80665**2*.005**2*np.diag([2e-6,1e-6]),rtol=1e-8,atol=1e-20)

    def test_conditional_projection_recovers_full_given_coupling(self):
        prior, expected = fixed_prior()
        parameters = prediction.conditional_parameters(prior['mean_a'], prior['cov_a'],
            prior['Eh'],prior['cov_hg'],prior['cov_ah'])
        for name,value in zip(('m','F','S'),expected):
            np.testing.assert_allclose(parameters[name],value,atol=8e-16)

    def test_no_random_candidates_are_generated(self):
        prior, _ = fixed_prior()
        with mock.patch.object(np.random,'default_rng',side_effect=AssertionError('No samples')):
            result = prediction.predict_joint_moments(**prior,gyro=[.1,-.2,.3],dt=.1,
                Sigma=np.diag([.03,.02,.04]),beta=.95,Q=np.eye(3)*.001)
        np.testing.assert_allclose(result['cov_hg'],result['cov_hg'].T,atol=0.)


if __name__ == '__main__':
    unittest.main()
