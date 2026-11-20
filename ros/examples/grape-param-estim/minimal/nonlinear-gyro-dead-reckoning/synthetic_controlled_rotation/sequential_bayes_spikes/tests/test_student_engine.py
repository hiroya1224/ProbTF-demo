"""Independent references for Student-t likelihoods and full causal Bayes law."""
from pathlib import Path
import importlib.util
import sys
import unittest

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, multivariate_t


HERE = Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


engine=load('_student_engine_test_filter',HERE/'full_rate_engine.py')
helpers=load('_student_engine_test_reference',HERE/'rate_comparison_verify.py')


class StudentBayesTests(unittest.TestCase):
    @staticmethod
    def fixture():
        times=np.array([0.,.009,.021,.033,.05])
        measurements=np.array([[.3,-.2,.1],[.306,-.194,.098],
            [.75,-.50,.30],[.299,-.204,.102],[.304,-.203,.10]])
        data=dict(time=np.array([0.,.05]),initial_quaternion_wxyz=np.array([1.,0.,0.,0.]),
                  sparse_gyro_time=times,sparse_gyro=measurements)
        physical=dict(inertia=np.array([[.045,.002,-.001],[.002,.070,.003],[-.001,.003,.100]]),
                      torque=np.array([.03,-.02,.01]))
        return data,physical

    def test_density_matches_scipy_multivariate_student_with_marginal_covariance(self):
        covariance=np.array([[.03,.004,-.002],[.004,.05,.003],[-.002,.003,.02]])
        mean=np.array([.2,-.4,.3])
        values=np.array([[.2,-.4,.3],[.21,-.45,.27],[2.,-3.,4.],[-20.,10.,30.]])
        for degrees in (3.,4.,10.):
            with self.subTest(degrees=degrees):
                expected=multivariate_t.logpdf(values,loc=mean,
                    shape=(degrees-2.)/degrees*covariance,df=degrees)
                actual=engine.student_t_log_density(values,mean,covariance,degrees)
                np.testing.assert_allclose(actual,expected,rtol=0.,atol=3e-13)

    def test_all_student_factors_and_original_prior_are_reconstructed_independently(self):
        data,physical=self.fixture()
        order,degrees=9,4.
        for mode in ('nonlinear','linear_forced'):
            with self.subTest(mode=mode):
                case=engine.predict_full_rate(data,physical,order=order,mode=mode,
                    refresh_seconds=.001,student_df=degrees)
                self.assertEqual(case['target_likelihood_factor_count'][-1],4)
                center=case['refresh_proposal_mean_rad_s'][-1]
                proposal_covariance=case['refresh_proposal_covariance_rad2_s2'][-1]
                nodes,quadrature=engine.gaussian_tensor_points(center,proposal_covariance,order)
                states=engine._adaptive.ControlledFlow(physical,data['initial_quaternion_wxyz'],
                    *data['time'],max_step=.002,mode=mode).evaluate(nodes,data['sparse_gyro_time'])
                covariance=np.eye(3)*np.deg2rad(5.)**2
                raw=(np.log(quadrature)
                    +multivariate_normal.logpdf(nodes,mean=data['sparse_gyro'][0],cov=covariance)
                    -multivariate_normal.logpdf(nodes,mean=center,cov=proposal_covariance))
                for index in range(1,len(data['sparse_gyro_time'])):
                    raw+=multivariate_t.logpdf(states[index,:,:3]-data['sparse_gyro'][index],
                        shape=(degrees-2.)/degrees*covariance,df=degrees)
                expected=raw-logsumexp(raw)
                np.testing.assert_allclose(case['refresh_log_weights'][-1],expected,rtol=0.,atol=5e-11)
                moments=helpers.independent_joint_moments(states[-1],np.exp(expected))
                for name,value in moments.items():
                    np.testing.assert_allclose(case[name][-1],value,rtol=0.,atol=5e-12,err_msg=name)
                self.assertGreater(np.linalg.norm(case['rotation_omega_covariance_colmajor'][-1]),1e-8)

    def test_first_reading_initializes_once_and_every_later_time_has_pre_and_post(self):
        data,physical=self.fixture()
        case=engine.predict_full_rate(data,physical,order=7,refresh_seconds=.001,student_df=4.)
        covariance=np.eye(3)*np.deg2rad(5.)**2
        np.testing.assert_allclose(case['omega_mean_rad_s'][0],data['sparse_gyro'][0],rtol=0.,atol=2e-14)
        np.testing.assert_allclose(case['omega_covariance_rad2_s2'][0],covariance,rtol=0.,atol=2e-14)
        acquired=dict(time=data['sparse_gyro_time'])
        check=helpers.structure(case,acquired,len(data['sparse_gyro_time']))
        for key in ('time_rows_exact','event_side_rows_exact','acquisition_indices_exact',
                    'likelihood_factor_count_all_rows_exact','event_times_exact'):
            self.assertTrue(check[key],key)
        self.assertEqual(case['target_likelihood_factor_count'][0],0)
        self.assertEqual(len(case['time']),2*len(data['sparse_gyro_time'])-1)

    def test_future_outlier_cannot_change_previous_joint_moments(self):
        data,physical=self.fixture()
        alternate={key:value.copy() for key,value in data.items()}
        alternate['sparse_gyro'][-1]+=[1.,-.7,.3]
        options=dict(order=7,refresh_seconds=.001,student_df=4.)
        base=engine.predict_full_rate(data,physical,**options)
        changed=engine.predict_full_rate(alternate,physical,**options)
        rows=(base['time']<data['time'][-1])|((base['time']==data['time'][-1])&(base['event_side']==-1))
        for key in ('quaternion_second_moment','rotation_mean','rotation_covariance_colmajor',
                    'omega_mean_rad_s','omega_covariance_rad2_s2','rotation_omega_covariance_colmajor'):
            np.testing.assert_array_equal(base[key][rows],changed[key][rows],err_msg=key)

    def test_constant_rate_posterior_matches_direct_original_prior_tensor_quadrature(self):
        # This short synthetic check has no truth or spikes labels. The only
        # dynamics are constant omega and the same quaternion kinematics.
        times=np.array([0.,.01,.02])
        data=dict(time=np.array([0.,.02]),initial_quaternion_wxyz=np.array([1.,0.,0.,0.]),
            sparse_gyro_time=times,sparse_gyro=np.array([[0.,0.,0.],[.60,-.40,.30],[0.,0.,0.]]))
        physical=dict(inertia=np.diag([.045,.07,.10]),torque=np.zeros(3))
        covariance=np.eye(3)*np.deg2rad(5.)**2
        nodes,quadrature=engine.gaussian_tensor_points(np.zeros(3),covariance,81)
        logs=np.log(quadrature)
        for observed in data['sparse_gyro'][1:]:
            logs+=multivariate_t.logpdf(nodes-observed,shape=.5*covariance,df=4.)
        mass=np.exp(logs-logsumexp(logs))
        mean=mass@nodes
        centered=nodes-mean
        expected_covariance=np.einsum('n,ni,nj->ij',mass,centered,centered)
        case=engine.predict_full_rate(data,physical,order=31,mode='linear_forced',
            refresh_seconds=.001,student_df=4.)
        np.testing.assert_allclose(case['omega_mean_rad_s'][-1],mean,rtol=0.,atol=1e-8)
        np.testing.assert_allclose(case['omega_covariance_rad2_s2'][-1],expected_covariance,rtol=0.,atol=1e-8)
        gaussian_posterior_mean=data['sparse_gyro'].mean(axis=0)
        self.assertLess(np.linalg.norm(mean),.2*np.linalg.norm(gaussian_posterior_mean))


if __name__=='__main__':
    unittest.main()
