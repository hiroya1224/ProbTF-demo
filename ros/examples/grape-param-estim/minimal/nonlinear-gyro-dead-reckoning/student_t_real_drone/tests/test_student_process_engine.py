"""Q timing, causal repeated Bayes updates and full joint moment checks."""
from pathlib import Path
import importlib.util
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_real_student_process_tests',HERE/'student_t_process_engine.py')
engine = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = engine
spec.loader.exec_module(engine)
FIELDS = ('quaternion_second_moment','rotation_mean','rotation_covariance_colmajor',
          'omega_mean_rad_s','omega_covariance_rad2_s2','rotation_omega_covariance_colmajor')


class StudentProcessTests(unittest.TestCase):
    @staticmethod
    def fixture():
        stamps = np.array([0.,.027,.065,.109])
        observed = np.array([[.03,-.02,.01],[.04,-.01,.02],[-.01,.03,.04],[.02,.01,-.02]])
        q0 = Rotation.from_euler('xyz',[.2,-.1,.3]).as_quat()[[3,0,1,2]]
        data = dict(time=np.array([0.,.123]),sparse_gyro_time=stamps,
            sparse_gyro=observed,initial_quaternion_wxyz=q0)
        physical = dict(inertia=.05*np.eye(3),torque=np.array([.0002,-.0001,.0003]))
        return data,physical

    def predict(self,data,physical,**options):
        return engine.predict_process_student(data,physical,sigma_deg_s=2.5,
            process_sigma_deg_s=1.,order=5,**options)

    def test_q_once_per_native_boundary_with_dense_points_and_tail(self):
        data,physical = self.fixture()
        case = self.predict(data,physical)
        prior = np.flatnonzero(case['event_side']==-1)
        posterior = np.flatnonzero(case['event_side']==1)
        self.assertEqual(len(prior),3)
        self.assertEqual(len(posterior),4)
        np.testing.assert_array_equal(case['time'][prior],data['sparse_gyro_time'][1:])
        np.testing.assert_array_equal(case['time'][posterior],data['sparse_gyro_time'])
        np.testing.assert_array_equal(case['process_application_count'][prior],np.arange(1,4))
        np.testing.assert_array_equal(case['process_application_count'][posterior],np.arange(4))
        np.testing.assert_array_equal(case['target_likelihood_factor_count'][prior],np.arange(3))
        np.testing.assert_array_equal(case['target_likelihood_factor_count'][posterior],np.arange(4))
        self.assertEqual(case['time'][-1],data['time'][-1])
        self.assertEqual(case['event_side'][-1],0)
        self.assertEqual(case['process_application_count'][-1],3)
        expected = np.eye(3)*(np.deg2rad(2.5)**2+np.deg2rad(1.)**2)
        np.testing.assert_allclose(case['omega_covariance_rad2_s2'][prior[0]],expected,atol=2e-16,rtol=0.)
        np.testing.assert_array_equal(case['process_covariance_rad2_s2'],np.eye(3)*np.deg2rad(1.)**2)
        # Final tail contains no fresh Q; linear torque keeps omega covariance.
        np.testing.assert_allclose(case['omega_covariance_rad2_s2'][-1],
            case['omega_covariance_rad2_s2'][posterior[-1]],atol=2e-17,rtol=0.)
        expected_shift = np.linalg.solve(physical['inertia'],physical['torque'])*(data['time'][-1]-data['sparse_gyro_time'][-1])
        np.testing.assert_allclose(case['omega_mean_rad_s'][-1]-case['omega_mean_rad_s'][posterior[-1]],
            expected_shift,atol=2e-16,rtol=0.)

    def test_future_measurement_cannot_change_any_prior_output(self):
        data,physical = self.fixture()
        first = self.predict(data,physical)
        changed = dict(data,sparse_gyro=data['sparse_gyro'].copy())
        changed['sparse_gyro'][-1] += np.array([.7,-.6,.5])
        second = self.predict(changed,physical)
        before = ((first['time']<data['sparse_gyro_time'][-1])
            | ((first['time']==data['sparse_gyro_time'][-1])&(first['event_side']==-1)))
        for key in FIELDS:
            np.testing.assert_array_equal(first[key][before],second[key][before],err_msg=key)
        after = (first['time']==data['sparse_gyro_time'][-1])&(first['event_side']==1)
        self.assertGreater(float(np.max(abs(first['omega_mean_rad_s'][after]-second['omega_mean_rad_s'][after]))),1e-5)

    def test_dense_clock_and_rk4_refinement_do_not_add_process_noise(self):
        data,physical = self.fixture()
        first = self.predict(data,physical,output_dt=.01,max_step=.002)
        second = self.predict(data,physical,output_dt=.007,max_step=.001)
        for side in (-1,1):
            a,b = first['event_side']==side,second['event_side']==side
            for key in FIELDS:
                np.testing.assert_allclose(first[key][a],second[key][b],atol=5e-13,rtol=0.,err_msg=key)
        self.assertEqual(first['metadata']['process_increment_count'],3)
        self.assertEqual(second['metadata']['process_increment_count'],3)

    def test_spherical_inertia_modes_agree_and_joint_covariance_retains_cross(self):
        data,physical = self.fixture()
        first = self.predict(data,physical,mode='nonlinear')
        second = self.predict(data,physical,mode='linear_forced')
        for key in FIELDS:
            np.testing.assert_allclose(first[key],second[key],atol=3e-14,rtol=0.,err_msg=key)
            self.assertTrue(np.isfinite(first[key]).all(),key)
        np.testing.assert_allclose(np.trace(first['quaternion_second_moment'],axis1=1,axis2=2),1.,atol=3e-14,rtol=0.)
        self.assertGreater(float(np.linalg.norm(first['rotation_omega_covariance_colmajor'][-1])),1e-5)
        for cov_r,cov_w,cross in zip(first['rotation_covariance_colmajor'],
                first['omega_covariance_rad2_s2'],first['rotation_omega_covariance_colmajor']):
            joint = np.block([[cov_r,cross],[cross.T,cov_w]])
            self.assertGreater(float(np.linalg.eigvalsh(joint).min()),-3e-14)

    def test_final_continuous_target_packet_reconstructs_last_observation(self):
        data,physical = self.fixture()
        case = self.predict(data,physical)
        rule = engine.density_quadrature(case['final_prior_component_weights'],
            case['final_prior_component_omega_means_rad_s'],
            case['final_prior_common_omega_covariance_rad2_s2'],
            case['final_prior_conditional_quaternion_second_moment'],
            case['final_prior_conditional_rotation_covariance_colmajor'],
            case['final_measurement_rad_s'],case['measurement_covariance_rad2_s2'],
            student_df=4.,order=5,proposal_mean=case['final_proposal_mean_rad_s'],
            proposal_covariance=case['final_proposal_covariance_rad2_s2'])
        last = np.flatnonzero(case['event_side']==1)[-1]
        for key in FIELDS:
            np.testing.assert_allclose(rule['statistics'][key],case[key][last],atol=2e-15,rtol=0.,err_msg=key)
        np.testing.assert_array_equal(rule['nodes'],case['final_quadrature_nodes_rad_s'])
        np.testing.assert_array_equal(rule['weights'],case['final_quadrature_posterior_weights'])

    def test_sparse_five_second_clock_has_two_q_increments_for_501_outputs(self):
        stamps = np.array([0.,2.003,4.007])
        observations = np.array([[.01,-.02,.03],[.015,-.01,.02],[-.01,.01,.025]])
        data = dict(time=np.array([0.,5.]),sparse_gyro_time=stamps,
            sparse_gyro=observations,initial_quaternion_wxyz=np.array([1.,0.,0.,0.]))
        # A read-only constant torque law provides no observation information.
        torque_value = np.array([.00002,-.00001,.00003])
        class ConstantTorque:
            def __call__(self,query):
                return np.broadcast_to(torque_value,np.asarray(query).shape+(3,)).copy()
        physical = dict(inertia=.05*np.eye(3),torque=ConstantTorque())
        original_observations = observations.copy()
        case = self.predict(data,physical,output_dt=.01,mode='linear_forced')
        prior = np.flatnonzero(case['event_side']==-1)
        posterior = np.flatnonzero(case['event_side']==1)
        self.assertEqual(len(case['dense_output_grid_time']),501)
        self.assertEqual(len(prior),2)
        self.assertEqual(len(posterior),3)
        self.assertEqual(int(case['process_noise_application_count']),2)
        np.testing.assert_array_equal(case['process_increment_times'],stamps[1:])
        np.testing.assert_array_equal(case['time'][posterior],stamps)
        np.testing.assert_array_equal(case['time'][prior],stamps[1:])
        q = np.eye(3)*np.deg2rad(1.)**2
        for before,previous_after in zip(prior,posterior[:-1]):
            np.testing.assert_allclose(case['omega_covariance_rad2_s2'][before],
                case['omega_covariance_rad2_s2'][previous_after]+q,atol=3e-16,rtol=0.)
        dense = case['event_side']==0
        np.testing.assert_array_equal(case['process_application_count'][dense],
            np.searchsorted(stamps,case['time'][dense],side='right')-1)
        tail = (case['time']>stamps[-1])&dense
        self.assertGreater(np.count_nonzero(tail),90)
        np.testing.assert_allclose(case['omega_covariance_rad2_s2'][tail],
            np.broadcast_to(case['omega_covariance_rad2_s2'][posterior[-1]],
                case['omega_covariance_rad2_s2'][tail].shape),atol=3e-16,rtol=0.)
        self.assertEqual(case['time'][-1],5.)
        np.testing.assert_array_equal(observations,original_observations)
        np.testing.assert_array_equal(torque_value,np.array([.00002,-.00001,.00003]))


if __name__ == '__main__':
    unittest.main()
