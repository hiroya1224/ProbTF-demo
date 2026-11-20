"""Independent continuous Student-t conditioning checks, without sampling."""
from pathlib import Path
import importlib.util
import sys
import unittest

import numpy as np
from scipy.integrate import quad_vec
from scipy.spatial.transform import Rotation
from scipy.stats import gamma, multivariate_normal


HERE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


engine = load('_process_student_independent_test_engine',HERE/'student_t_process_engine.py')
verify = load('_process_student_independent_test_reference',HERE/'process_verify.py')


def integrated_gaussian_student(mean,covariance,measurement,measurement_covariance,nu):
    """Integrate one latent Gamma precision with adaptive deterministic quad."""
    scale = (nu-2.)/nu*measurement_covariance
    def integrand(precision):
        if precision == 0.:
            return np.zeros(13)
        noise = scale/precision
        innovation = covariance+noise
        gain = np.linalg.solve(innovation,covariance.T).T
        updated_mean = mean+gain@(measurement-mean)
        residual = np.eye(3)-gain
        updated_covariance = residual@covariance@residual.T+gain@noise@gain.T
        evidence = (gamma.pdf(precision,a=nu/2.,scale=2./nu)
            *multivariate_normal.pdf(measurement,mean=mean,cov=innovation))
        return evidence*np.r_[1.,updated_mean,
            (updated_covariance+np.outer(updated_mean,updated_mean)).reshape(-1)]
    integrated,error = quad_vec(integrand,0.,np.inf,epsabs=1e-10,epsrel=1e-11)
    normalizer = integrated[0]
    posterior_mean = integrated[1:4]/normalizer
    posterior_covariance = integrated[4:].reshape(3,3)/normalizer-np.outer(posterior_mean,posterior_mean)
    return posterior_mean,posterior_covariance,float(np.log(normalizer)),float(error)


def component_orientation(rotations,weights):
    q = np.stack([rotation.as_quat()[[3,0,1,2]] for rotation in rotations])
    r = np.stack([rotation.as_matrix().reshape(9,order='F') for rotation in rotations])
    centered = r-weights@r
    return (q.T*weights)@q,(centered.T*weights)@centered


class IndependentStudentProcessConditioningTests(unittest.TestCase):
    def test_gaussian_times_student_matches_independent_latent_gamma_integral(self):
        mean = np.array([.01,-.02,.03])
        covariance = np.eye(3)*np.deg2rad(3.)**2
        measurement = np.array([.085,-.03,-.025])
        measurement_covariance = np.eye(3)*np.deg2rad(5.)**2
        q = np.array([1.,0.,0.,0.])
        expected_mean,expected_covariance,expected_log_z,error = integrated_gaussian_student(
            mean,covariance,measurement,measurement_covariance,4.)
        differences = []
        for order in (7,15):
            result = engine.density_quadrature(np.ones(1),mean[None],covariance,
                np.outer(q,q)[None],np.zeros((1,9,9)),measurement,measurement_covariance,
                student_df=4.,order=order)
            differences.append(max(float(np.max(np.abs(result['statistics']['omega_mean_rad_s']-expected_mean))),
                float(np.max(np.abs(result['statistics']['omega_covariance_rad2_s2']-expected_covariance)))))
        self.assertLess(differences[-1],differences[0])
        self.assertLess(differences[-1],2e-8)
        self.assertAlmostEqual(result['diagnostics']['log_predictive_evidence'],expected_log_z,places=6)
        centered = result['nodes'][:,0]-result['statistics']['omega_mean_rad_s'][0]
        fourth = float(result['weights']@(centered**4))
        gaussian_fourth = float(3*result['statistics']['omega_covariance_rad2_s2'][0,0]**2)
        self.assertGreater(abs(fourth-gaussian_fourth),1e-9)
        print('Independent latent-Gamma quadrature mean/cov errors:',differences,
              'integral error estimate:',error,flush=True)

    def test_same_mixture_student_target_preserves_conditional_orientation_and_cross(self):
        weights = np.array([.35,.65])
        means = np.array([[-.04,.02,.03],[.08,-.05,.06]])
        covariance = np.eye(3)*np.deg2rad(1.)**2
        measurement = np.array([.025,-.015,.02])
        measurement_covariance = np.eye(3)*np.deg2rad(2.5)**2
        within_weights = np.array([.3,.7])
        orientations = [component_orientation([
            Rotation.from_euler('xyz',[.2,-.1,.3]),Rotation.from_euler('xyz',[-.1,.05,-.2])],within_weights),
            component_orientation([Rotation.from_euler('xyz',[-.3,.2,-.1]),
                                  Rotation.from_euler('xyz',[.1,-.2,.4])],within_weights)]
        qmoments = np.stack([item[0] for item in orientations])
        rotation_covariances = np.stack([item[1] for item in orientations])
        result = engine.density_quadrature(weights,means,covariance,qmoments,
            rotation_covariances,measurement,measurement_covariance,student_df=4.,order=7)
        reference = verify.independent_density_rule(weights,means,covariance,qmoments,
            rotation_covariances,result['nodes'],result['base_weights'],result['proposal_mean'],
            result['proposal_covariance'],measurement,measurement_covariance,4.)
        np.testing.assert_allclose(result['weights'],reference['weights'],atol=4e-14,rtol=3e-13)
        np.testing.assert_allclose(result['quaternion_moments'],reference['quaternion_moments'],atol=2e-14,rtol=3e-13)
        np.testing.assert_allclose(result['rotation_covariances'],reference['rotation_covariances'],atol=2e-14,rtol=3e-13)
        for key,value in reference['moments'].items():
            np.testing.assert_allclose(result['statistics'][key],value,atol=2e-14,rtol=3e-13,err_msg=key)
        self.assertGreater(np.linalg.norm(reference['moments']['rotation_omega_covariance_colmajor']),1e-5)
        self.assertGreater(np.linalg.norm(reference['rotation_covariances']),1e-3)
        self.assertTrue(result['diagnostics']['gaussian_proposal_is_not_posterior'])


if __name__=='__main__':
    unittest.main()
