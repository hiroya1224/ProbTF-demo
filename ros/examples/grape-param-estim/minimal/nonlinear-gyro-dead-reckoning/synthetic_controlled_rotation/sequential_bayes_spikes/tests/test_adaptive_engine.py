"""Native flow equivalence and causal adaptive quadrature of one Bayes law."""
from pathlib import Path
import importlib.util
import unittest

import numpy as np

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('adaptive_bayes_test_engine',HERE/'adaptive_engine.py')
adaptive=importlib.util.module_from_spec(spec)
spec.loader.exec_module(adaptive)


class NativeFlowTests(unittest.TestCase):
    def test_same_nonlinear_controlled_rk4_as_python_reference(self):
        from propagation import integrate_six_point_increment,gaussian_six_points,quaternion_multiply
        q0=np.array([.8,.2,-.3,.4]);q0/=np.linalg.norm(q0)
        mean=np.array([.3,-.5,.7]);covariance=np.diag([.003,.004,.002])
        nodes,_=gaussian_six_points(mean,covariance)
        times=np.linspace(1.2,1.26,7)
        tensor=np.array([[.045,.002,-.001],[.002,.070,.003],[-.001,.003,.1]])
        torque=np.array([.03,-.02,.01])
        actual=adaptive.ControlledFlow(dict(inertia=tensor,torque=torque),q0,times[0],times[-1]).evaluate(nodes,times)
        expected=integrate_six_point_increment(mean,covariance,times,torque,tensor,max_step=.002)
        np.testing.assert_allclose(actual[:,:,:3],expected['omega'],rtol=0,atol=2e-15)
        expected_q=quaternion_multiply(q0,expected['delta'])
        np.testing.assert_allclose(actual[:,:,3:],expected_q,rtol=0,atol=2e-15)


class AdaptiveBayesTests(unittest.TestCase):
    def fixture(self):
        times=np.linspace(0.,.2,21)
        measured=np.tile([.1,-.2,.3],(21,1))
        measured[1:]+=np.array([.01,-.02,.03])
        data=dict(time=np.array([0.,.2]),initial_quaternion_wxyz=np.array([1.,0.,0.,0.]),
                  sparse_gyro_time=times,sparse_gyro=measured)
        profile=dict(inertia=np.diag([.045,.070,.100]),torque=np.zeros(3))
        return data,profile

    def test_twenty_updates_match_narrow_analytic_gaussian_without_false_certainty(self):
        data,profile=self.fixture()
        result=adaptive.predict_adaptive(data,profile,order=3,output_dt=.01,
            mode='linear_forced')['default']
        variance=np.deg2rad(5.)**2
        for event,stamp in enumerate(data['sparse_gyro_time']):
            rows=np.flatnonzero((result['time']==stamp)&(result['event_side']==1))
            self.assertEqual(len(rows),1)
            row=rows[0]
            expected_mean=data['sparse_gyro'][:event+1].mean(axis=0)
            expected_cov=np.eye(3)*variance/(event+1)
            np.testing.assert_allclose(result['omega_mean_rad_s'][row],expected_mean,atol=2e-10,rtol=0)
            np.testing.assert_allclose(result['omega_covariance_rad2_s2'][row],expected_cov,atol=2e-10,rtol=0)
        self.assertFalse(result['metadata']['earlier_online_outputs_rewritten'])
        self.assertTrue(result['metadata']['proposal_is_not_posterior_state'])

    def test_future_measurement_does_not_change_previous_online_moments(self):
        data,profile=self.fixture()
        data['sparse_gyro_time']=np.array([0.,.1,.2])
        data['sparse_gyro']=np.tile([.1,-.2,.3],(3,1))
        changed=data['sparse_gyro'].copy();changed[-1]+=[.2,-.1,.3]
        cases=adaptive.predict_adaptive(data,profile,order=5,output_dt=.02,mode='linear_forced',
            observation_sets=dict(first=data['sparse_gyro'],changed=changed))
        mask=(cases['first']['time']<.2)|((cases['first']['time']==.2)&(cases['first']['event_side']==-1))
        for key in ('rotation_mean','omega_mean_rad_s','omega_covariance_rad2_s2',
                    'rotation_omega_covariance_colmajor'):
            np.testing.assert_array_equal(cases['first'][key][mask],cases['changed'][key][mask])
        self.assertGreater(np.linalg.norm(cases['changed']['rotation_omega_covariance_colmajor'][-1]),0.)
        self.assertFalse(np.array_equal(cases['first']['rotation_mean'][-1],cases['changed']['rotation_mean'][-1]))

    def test_gaussian_proposal_ratio_preserves_non_gaussian_target(self):
        nodes,base=adaptive.gaussian_tensor_points(np.zeros(3),np.eye(3),7)
        prior=adaptive.gaussian_log_density(nodes,np.zeros(3),np.eye(3)*2)
        proposal=adaptive.gaussian_log_density(nodes,np.zeros(3),np.eye(3))
        # An observation depending quadratically on W is a non-Gaussian law.
        observation_factor=-.5*(nodes[:,0]**2-1.)**2
        logweights=np.log(base)+prior+observation_factor-proposal
        weights=np.exp(logweights-adaptive.logsumexp(logweights))
        self.assertGreater(float(weights@(nodes[:,0]**2)),.4)
        self.assertLess(float(weights@(nodes[:,0]**2)),1.7)
        self.assertFalse(np.allclose(weights,base))

    def test_adapted_nodes_retain_every_original_likelihood_and_divide_proposal(self):
        data,profile=self.fixture()
        data['sparse_gyro_time']=np.array([0.,.1,.2])
        data['sparse_gyro']=np.array([[.1,-.2,.3],[.13,-.21,.32],[.08,-.17,.28]])
        result=adaptive.predict_adaptive(data,profile,order=5,output_dt=.02)['default']
        nodes=result['event_initial_omega_nodes_rad_s'][-1]
        states=adaptive.ControlledFlow(profile,data['initial_quaternion_wxyz'],0.,.2).evaluate(
            nodes,data['sparse_gyro_time'])
        variance=np.deg2rad(5.)**2
        original_residual=nodes-data['sparse_gyro'][0]
        log_target=-.5*np.sum(original_residual**2,axis=1)/variance
        for j in (1,2):
            residual=states[j,:,:3]-data['sparse_gyro'][j]
            log_target-=.5*np.sum(residual**2,axis=1)/variance
        centered=nodes-result['event_proposal_mean_rad_s'][-1]
        inverse=np.linalg.solve(result['event_proposal_covariance_rad2_s2'][-1],np.eye(3))
        log_proposal=-.5*np.einsum('ni,ij,nj->n',centered,inverse,centered)
        _,mass=adaptive.gaussian_tensor_points(np.zeros(3),np.eye(3),5)
        expected=np.log(mass)+log_target-log_proposal
        expected-=adaptive.logsumexp(expected)
        np.testing.assert_allclose(result['event_log_weights'][-1],expected,atol=2e-12,rtol=0)


if __name__=='__main__':
    unittest.main()
