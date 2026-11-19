"""Compare two dynamics models using exactly the same spiked observations."""
from pathlib import Path
import importlib.util
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('shared_spikes_model_comparison_runner', HERE/'frequent_run.py')
adaptive = load('shared_spikes_model_comparison_engine', HERE/'adaptive_engine.py')


class SharedSpikedInputTests(unittest.TestCase):
    def fixture(self):
        white = np.array([[.3, -.5, .7], [.31, -.49, .71], [.32, -.48, .72]])
        spiked = white.copy()
        spiked[1] += [.2, -.3, .4]
        selected = dict(time=np.array([0., .02]),
            initial_quaternion_wxyz=np.array([1., 0., 0., 0.]),
            sparse_gyro_time=np.array([0., .01, .02]), sparse_gyro=white.copy())
        profile = dict(inertia=np.array([[.045,.002,-.001],
                                        [.002,.070,.003],
                                        [-.001,.003,.100]]),
                       torque=np.array([.03,-.02,.01]))
        acquisitions = dict(time=selected['sparse_gyro_time'].copy(),
            gyro_white=white, gyro_spike=spiked,
            spike_mask=np.array([False,True,False]))
        config = dict(assumed_sigma_deg_s=5., output_dt_seconds=.005,
                      max_step_seconds=.002)
        return selected, profile, acquisitions, config

    def test_orchestration_passes_identical_spiked_values_clock_and_prior_to_both_models(self):
        selected, profile, acquisitions, config = self.fixture()
        untouched = {key:value.copy() for key,value in selected.items()}
        calls = []

        class Recorder:
            @staticmethod
            def predict_adaptive(data, physical, **kwargs):
                calls.append((data,physical,kwargs))
                return {'shared':dict(marker=kwargs['mode'])}

        result = runner.predict_models(Recorder, selected, profile, acquisitions, config, 7)
        self.assertEqual(tuple(result), ('nonlinear','linear_forced'))
        self.assertEqual([call[2]['mode'] for call in calls], ['nonlinear','linear_forced'])
        for data, physical, kwargs in calls:
            self.assertIs(physical, profile)
            np.testing.assert_array_equal(data['sparse_gyro'], acquisitions['gyro_spike'])
            np.testing.assert_array_equal(data['sparse_gyro_time'], selected['sparse_gyro_time'])
            np.testing.assert_array_equal(data['initial_quaternion_wxyz'], selected['initial_quaternion_wxyz'])
            self.assertEqual(tuple(kwargs['observation_sets']), ('shared',))
            np.testing.assert_array_equal(kwargs['observation_sets']['shared'], acquisitions['gyro_spike'])
            self.assertEqual(kwargs['sigma_deg_s'], config['assumed_sigma_deg_s'])
            self.assertEqual(kwargs['order'], 7)
            self.assertEqual(kwargs['max_step'], config['max_step_seconds'])
            self.assertEqual(kwargs['output_dt'], config['output_dt_seconds'])
        for key in selected:
            np.testing.assert_array_equal(selected[key], untouched[key])
        self.assertNotIn('white', result)
        self.assertNotIn('spike', result)

    def test_real_filters_share_initial_rate_law_but_use_different_dynamics(self):
        selected, profile, acquisitions, config = self.fixture()
        result = runner.predict_models(adaptive, selected, profile, acquisitions, config, 3)
        nonlinear, linear = result['nonlinear'], result['linear_forced']
        for key in ('time','event_side','acquisition_index','measurement_covariance_rad2_s2'):
            np.testing.assert_array_equal(nonlinear[key], linear[key])
        for key in ('rotation_mean','rotation_covariance_colmajor',
                    'omega_mean_rad_s','omega_covariance_rad2_s2'):
            np.testing.assert_array_equal(nonlinear[key][0], linear[key][0])
        first_regular = np.flatnonzero((nonlinear['time']==.005)&(nonlinear['event_side']==0))[0]
        self.assertGreater(np.linalg.norm(nonlinear['omega_mean_rad_s'][first_regular]
                                         -linear['omega_mean_rad_s'][first_regular]), 1e-5)
        for case in result.values():
            self.assertEqual([item['likelihood_applications'] for item in case['events']], [0,1,1])
            self.assertGreater(np.linalg.norm(case['rotation_omega_covariance_colmajor'][-1]), 0.)


class SharedQuaternionKinematicsTests(unittest.TestCase):
    def test_only_gyroscopic_term_is_removed_and_quaternion_equation_is_shared(self):
        tensor = np.array([[.045,.002,-.001],[.002,.070,.003],[-.001,.003,.100]])
        torque = np.array([.03,-.02,.01])
        omega = np.array([.3,-.5,.7])
        q0 = np.array([.8,.2,-.3,.4]);q0 /= np.linalg.norm(q0)
        dt = 1e-5
        from propagation import quaternion_multiply
        quaternion_derivative = .5*quaternion_multiply(q0, np.r_[0.,omega])
        derivatives = {}
        for mode in ('nonlinear','linear_forced'):
            state = adaptive.ControlledFlow(dict(inertia=tensor,torque=torque),q0,0.,dt,
                max_step=dt,mode=mode).evaluate(omega[None,:],np.array([0.,dt/2.,dt]))
            derivative = (-3.*state[0,0]+4.*state[1,0]-state[2,0])/dt
            derivatives[mode] = derivative[:3]
            expected_force = torque-(np.cross(omega,tensor@omega)
                                      if mode=='nonlinear' else np.zeros(3))
            np.testing.assert_allclose(derivatives[mode],np.linalg.solve(tensor,expected_force),
                                       rtol=0,atol=3e-9)
            np.testing.assert_allclose(derivative[3:],quaternion_derivative,
                                       rtol=0,atol=3e-9)
        np.testing.assert_allclose(derivatives['linear_forced']-derivatives['nonlinear'],
            np.linalg.solve(tensor,np.cross(omega,tensor@omega)),rtol=0,atol=3e-9)


if __name__=='__main__':
    unittest.main()
