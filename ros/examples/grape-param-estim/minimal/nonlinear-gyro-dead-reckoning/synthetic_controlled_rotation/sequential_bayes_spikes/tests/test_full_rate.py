"""Causal all-reading Bayes integration against independent references.

No long experiment is needed here: exact forced-linear posterior moments,
the existing per-reading adaptive implementation, and direct full-history
importance weights expose missed/doubled observations and stale proposals.
"""
from pathlib import Path
import importlib.util
import sys
import unittest

import numpy as np
from scipy.special import logsumexp


HERE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


adaptive = load('full_rate_test_existing_adaptive', HERE/'adaptive_engine.py')
full_rate = load('full_rate_test_engine', HERE/'full_rate_engine.py')


MOMENTS = ('quaternion_second_moment','rotation_mean','rotation_covariance_colmajor',
           'omega_mean_rad_s','omega_covariance_rad2_s2',
           'rotation_omega_covariance_colmajor','initial_omega_mean_rad_s',
           'initial_omega_covariance_rad2_s2','rpy_proxy_rad',
           'rpy_covariance_delta_rad2')


class FullRateBayesTests(unittest.TestCase):
    def fixture(self, *, irregular=False, duration=.05):
        times = (np.array([0.,.005,.0098,.0152,.020,.0251,.030,.0347,.0401,.045,.05])
                 if irregular else np.arange(round(duration/.005)+1)*.005)
        tensor = np.array([[.045,.002,-.001],
                           [.002,.070,.003],
                           [-.001,.003,.100]])
        torque = np.array([.03,-.02,.01])
        q0 = np.array([.8,.2,-.3,.4]); q0 /= np.linalg.norm(q0)
        # The sensor input is deliberately deterministic. Its small changes
        # test actual likelihood fusion without generation-only truth fields.
        measurements = (np.array([.35,-.50,.80])[None,:]
            +times[:,None]*np.array([.11,-.08,.06])[None,:]
            +np.column_stack((.008*np.sin(83.*times),
                              -.006*np.sin(57.*times),
                              .009*np.sin(69.*times))))
        data = dict(time=np.array([0.,float(times[-1])]),
                    initial_quaternion_wxyz=q0,
                    sparse_gyro_time=times,sparse_gyro=measurements)
        return data, dict(inertia=tensor,torque=torque)

    @staticmethod
    def row(case, index, side):
        rows = np.flatnonzero((case['acquisition_index']==index)
                              &(case['event_side']==side))
        if len(rows)!=1:
            raise AssertionError(f'Expected one acquisition {index}, side {side}; got {rows}')
        return int(rows[0])

    @staticmethod
    def analytic_linear(data, profile, index, side):
        included = index+1 if side==1 else index
        stamp = data['sparse_gyro_time'][index]
        acceleration = np.linalg.solve(profile['inertia'],profile['torque'])
        initial_values = (data['sparse_gyro'][:included]
            -data['sparse_gyro_time'][:included,None]*acceleration)
        mean = initial_values.mean(axis=0)+stamp*acceleration
        covariance = np.eye(3)*np.deg2rad(5.)**2/included
        return mean,covariance

    def test_per_reading_refresh_matches_existing_adaptive_on_irregular_200hz_clock(self):
        data,profile = self.fixture(irregular=True)
        for mode in ('nonlinear','linear_forced'):
            with self.subTest(mode=mode):
                old = adaptive.predict_adaptive(data,profile,sigma_deg_s=5.,order=5,
                    output_dt=.05,max_step=.002,mode=mode)['default']
                case = full_rate.predict_full_rate(data,profile,sigma_deg_s=5.,order=5,
                    max_step=.002,mode=mode,refresh_seconds=.001)
                for key in ('time','event_side','acquisition_index'):
                    np.testing.assert_array_equal(case[key],old[key],err_msg=key)
                for key in MOMENTS:
                    np.testing.assert_allclose(case[key],old[key],rtol=2e-9,atol=2e-10,
                                               err_msg=key)

    def test_linear_forced_uses_every_reading_once_with_exact_analytic_moments(self):
        data,profile = self.fixture()
        case = full_rate.predict_full_rate(data,profile,order=5,mode='linear_forced',
                                           refresh_seconds=.001)
        count = len(data['sparse_gyro_time'])
        self.assertEqual(len(case['time']),2*count-1)
        self.assertEqual(case['event_side'][0],1)
        self.assertEqual(case['target_likelihood_factor_count'][0],0)
        for index,stamp in enumerate(data['sparse_gyro_time']):
            for side in ((1,) if index==0 else (-1,1)):
                row = self.row(case,index,side)
                self.assertEqual(case['time'][row],stamp)
                expected_count = index if side==1 else index-1
                self.assertEqual(case['target_likelihood_factor_count'][row],expected_count)
                mean,covariance = self.analytic_linear(data,profile,index,side)
                np.testing.assert_allclose(case['omega_mean_rad_s'][row],mean,
                                           rtol=0.,atol=3e-9)
                np.testing.assert_allclose(case['omega_covariance_rad2_s2'][row],covariance,
                                           rtol=0.,atol=3e-10)
        np.testing.assert_array_equal(case['event_time'],data['sparse_gyro_time'])

    def test_refresh_weights_are_original_prior_times_all_past_factors_divided_by_proposal(self):
        data,profile = self.fixture(irregular=True)
        order = 5
        case = full_rate.predict_full_rate(data,profile,order=order,
                                           refresh_seconds=.001)
        index = int(case['refresh_acquisition_index'][-1])
        self.assertEqual(index,len(data['sparse_gyro_time'])-1)
        nodes,quadrature = full_rate.gaussian_tensor_points(
            case['refresh_proposal_mean_rad_s'][-1],
            case['refresh_proposal_covariance_rad2_s2'][-1],order)
        states = adaptive.ControlledFlow(profile,data['initial_quaternion_wxyz'],
            data['time'][0],data['time'][1],max_step=.002).evaluate(
                nodes,data['sparse_gyro_time'][:index+1])
        variance = np.deg2rad(5.)**2
        log_target = -.5*np.sum((nodes-data['sparse_gyro'][0])**2,axis=1)/variance
        for acquired in range(1,index+1):
            residual = states[acquired,:,:3]-data['sparse_gyro'][acquired]
            log_target -= .5*np.sum(residual**2,axis=1)/variance
        centered = nodes-case['refresh_proposal_mean_rad_s'][-1]
        covariance = case['refresh_proposal_covariance_rad2_s2'][-1]
        log_proposal = -.5*np.sum(centered*np.linalg.solve(covariance,centered.T).T,axis=1)
        expected = np.log(quadrature)+log_target-log_proposal
        expected -= logsumexp(expected)
        np.testing.assert_allclose(case['refresh_log_weights'][-1],expected,
                                   rtol=0.,atol=2e-10)

    def test_grouped_proposal_refinement_preserves_the_same_analytic_target(self):
        data,profile = self.fixture(duration=.03)
        errors = []
        for order in (7,25):
            # Disable extra placement triggers only for this numerical
            # quadrature test. Every observed likelihood is still included.
            case = full_rate.predict_full_rate(data,profile,order=order,
                mode='linear_forced',refresh_seconds=.01,initial_dense_seconds=0.,
                reanchor_displacement_sigma=None)
            self.assertLess(len(case['refresh_acquisition_index']),len(data['sparse_gyro_time']))
            error = 0.
            for index in range(len(data['sparse_gyro_time'])):
                for side in ((1,) if index==0 else (-1,1)):
                    row = self.row(case,index,side)
                    mean,covariance = self.analytic_linear(data,profile,index,side)
                    error = max(error,np.max(np.abs(case['omega_mean_rad_s'][row]-mean)),
                                np.max(np.abs(case['omega_covariance_rad2_s2'][row]-covariance)))
                    self.assertEqual(case['target_likelihood_factor_count'][row],
                                     index if side==1 else index-1)
            errors.append(error)
        # The finite difference is an integration diagnostic. No observation
        # is discarded, and no covariance floor is used to force agreement.
        self.assertLess(errors[1],errors[0]*.01)
        self.assertLess(errors[1],3e-8)

    def test_grouped_nonlinear_updates_match_high_order_per_reading_reference(self):
        data,profile = self.fixture(duration=.03)
        reference = adaptive.predict_adaptive(data,profile,order=11,
            output_dt=.03,max_step=.002,mode='nonlinear')['default']
        case = full_rate.predict_full_rate(data,profile,order=21,max_step=.002,
            mode='nonlinear',refresh_seconds=.01,initial_dense_seconds=0.,
            reanchor_displacement_sigma=None)
        self.assertLess(len(case['refresh_acquisition_index']),len(data['sparse_gyro_time']))
        for key in ('time','event_side','acquisition_index'):
            np.testing.assert_array_equal(case[key],reference[key],err_msg=key)
        for key in MOMENTS:
            np.testing.assert_allclose(case[key],reference[key],rtol=0.,atol=1e-9,
                                       err_msg=key)

    def test_future_measurements_and_longer_horizon_do_not_rewrite_the_online_prefix(self):
        data,profile = self.fixture(duration=.05)
        changed = {key:value.copy() for key,value in data.items()}
        changed['sparse_gyro'][-1] += [.4,-.3,.2]
        final = data['sparse_gyro_time'][-1]
        extended,extended_profile = self.fixture(duration=.10)
        # Check both per-reading startup placement and actual grouped
        # placement. A forced refresh merely because a run ends would make
        # its final posterior depend on knowledge of the future run length.
        for dense in (.25,0.):
            with self.subTest(initial_dense_seconds=dense):
                options = dict(order=5,refresh_seconds=.02,
                               initial_dense_seconds=dense)
                base = full_rate.predict_full_rate(data,profile,**options)
                alternate = full_rate.predict_full_rate(changed,profile,**options)
                before_update = ((base['time']<final)
                    |((base['time']==final)&(base['event_side']==-1)))
                for key in MOMENTS:
                    np.testing.assert_array_equal(base[key][before_update],
                        alternate[key][before_update],err_msg=key)
                self.assertGreater(np.linalg.norm(base['omega_mean_rad_s'][-1]
                    -alternate['omega_mean_rad_s'][-1]),1e-4)
                long_case = full_rate.predict_full_rate(extended,extended_profile,**options)
                prefix = long_case['time']<=final
                for key in ('time','event_side','acquisition_index',
                            'target_likelihood_factor_count'):
                    np.testing.assert_array_equal(base[key],long_case[key][prefix],err_msg=key)
                for key in MOMENTS:
                    # Different native output allocations can change the
                    # final rounding unit; this bound is far below even
                    # the order-refinement differences measured elsewhere.
                    np.testing.assert_allclose(base[key],long_case[key][prefix],
                                               rtol=0.,atol=3e-13,err_msg=key)

    def test_joint_quaternion_rate_and_full_cross_covariance_are_retained(self):
        data,profile = self.fixture()
        case = full_rate.predict_full_rate(data,profile,order=5,refresh_seconds=.02)
        rows = len(case['time'])
        for key,shape in (('quaternion_second_moment',(4,4)),
                          ('rotation_mean',(3,3)),
                          ('rotation_covariance_colmajor',(9,9)),
                          ('omega_mean_rad_s',(3,)),
                          ('omega_covariance_rad2_s2',(3,3)),
                          ('rotation_omega_covariance_colmajor',(9,3))):
            self.assertEqual(case[key].shape,(rows,*shape),key)
            self.assertTrue(np.isfinite(case[key]).all(),key)
        np.testing.assert_allclose(np.trace(case['quaternion_second_moment'],axis1=1,axis2=2),
                                   1.,rtol=0.,atol=3e-13)
        self.assertGreater(np.linalg.norm(case['rotation_omega_covariance_colmajor'][-1]),1e-6)
        self.assertGreater(np.linalg.norm(case['omega_covariance_rad2_s2'][-1]
            -np.diag(np.diag(case['omega_covariance_rad2_s2'][-1]))),1e-8)
        for row in (0,rows//2,rows-1):
            rotation_covariance = case['rotation_covariance_colmajor'][row]
            omega_covariance = case['omega_covariance_rad2_s2'][row]
            cross = case['rotation_omega_covariance_colmajor'][row]
            joint = np.block([[rotation_covariance,cross],[cross.T,omega_covariance]])
            np.testing.assert_allclose(joint,joint.T,rtol=0.,atol=3e-13)
            self.assertGreaterEqual(np.linalg.eigvalsh(joint).min(),-3e-12)


if __name__=='__main__':
    unittest.main()
