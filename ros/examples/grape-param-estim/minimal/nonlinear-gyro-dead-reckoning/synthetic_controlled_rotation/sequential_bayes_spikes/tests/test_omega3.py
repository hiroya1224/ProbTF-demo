"""Independent physical and causal checks for the exact body-omega 3x case.

The changed signal is angular motion, not its clock, measured noise, or
Bayesian observation law.  Numerical tolerances here check finite integration
error; they are never used to accept or reject an observation.
"""
from pathlib import Path
import importlib.util
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


inputs = load('omega3_test_inputs', HERE/'omega3_input.py')
native = load('omega3_test_native_flow', HERE/'native_flow.py')
bayes = load('omega3_test_full_rate', HERE/'full_rate_engine.py')


class OmegaThreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.sensor, cls.acquired, cls.selected, cls.physical, cls.metadata = inputs.generate()
        with np.load(HERE/'frequent_publication/source_input.npz', allow_pickle=False) as saved:
            cls.previous = {name: saved[name].copy() for name in saved.files}
        with np.load(HERE/'frequent_publication/sensor_input.npz', allow_pickle=False) as saved:
            cls.old_sensor = {name: saved[name].copy() for name in saved.files}

    def test_exact_body_rate_and_acceleration_scaling_at_unchanged_times(self):
        for name in ('reference_omega_body_rad_s','reference_alpha_body_rad_s2',
                     'gyro_true_body_rad_s'):
            np.testing.assert_array_equal(self.source[name], 3.*self.previous[name],
                                          err_msg=name)
        np.testing.assert_allclose(self.source['reference_gyroscopic_torque_body_nm'],
            9.*self.previous['reference_gyroscopic_torque_body_nm'],
            rtol=3e-15, atol=3e-17)
        for name in ('time','reference_time','torque_time','imu_time','imu_stamp_ns',
                     'imu_record_ns','imu_source_index','initial_quaternion_wxyz',
                     'inertia_body_kg_m2'):
            np.testing.assert_array_equal(self.source[name], self.previous[name], err_msg=name)

    def test_white_draw_bias_spike_events_and_acquisition_indices_are_preserved(self):
        for name in ('gyro_realized_noise_rad_s','gyro_bias_rad_s',
                     'generator_gyro_covariance_rad2_s2'):
            np.testing.assert_array_equal(self.source[name], self.previous[name], err_msg=name)
        for name in self.old_sensor:
            if name not in ('gyro_white','gyro_spike'):
                np.testing.assert_array_equal(self.sensor[name], self.old_sensor[name], err_msg=name)
        movement = 2.*self.previous['gyro_true_body_rad_s']
        for name in ('gyro_white','gyro_spike'):
            np.testing.assert_allclose(self.sensor[name]-self.old_sensor[name], movement,
                                       rtol=3e-13, atol=5e-16, err_msg=name)
        index = self.acquired['native_array_index']
        np.testing.assert_array_equal(index, np.arange(0,4001,20))
        for name in ('stamp_ns','record_ns','source_index','spike_vector_rad_s','gyro_spike'):
            np.testing.assert_array_equal(self.acquired[name], self.sensor[name][index], err_msg=name)

    def test_quaternion_curve_independently_differentiates_to_threefold_body_omega(self):
        q = self.source['reference_quaternion_xyzw']
        stamps = self.source['reference_time']
        step = float(stamps[1]-stamps[0])
        np.testing.assert_allclose(np.diff(stamps), step, rtol=0., atol=4e-15)
        rotation = Rotation.from_quat(q).as_matrix()
        # Fourth-order central derivative of saved rotations; no analytic
        # body-rate function or quaternion generator is called here.
        derivative = (rotation[:-4]-8.*rotation[1:-3]
                      +8.*rotation[3:-1]-rotation[4:])/(12.*step)
        skew = np.einsum('tji,tjk->tik', rotation[2:-2], derivative)
        body_omega = .5*np.column_stack((skew[:,2,1]-skew[:,1,2],
                                        skew[:,0,2]-skew[:,2,0],
                                        skew[:,1,0]-skew[:,0,1]))
        np.testing.assert_allclose(body_omega,
            self.source['reference_omega_body_rad_s'][2:-2], rtol=0., atol=2e-6)
        # Generator integration uses finite DOP853 tolerances; the source
        # quaternion is not silently normalized to hide its residual.
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1., rtol=0., atol=1e-10)
        np.testing.assert_allclose(q[0][[3,0,1,2]],
            self.source['initial_quaternion_wxyz'], rtol=0., atol=3e-14)

    def test_saved_torque_reproduces_the_new_motion_under_nonlinear_euler(self):
        time = self.source['reference_time']
        flow = native.ControlledFlow(self.physical,
            self.source['initial_quaternion_wxyz'], time[0], time[-1],
            max_step=.001, mode='nonlinear')
        states = flow.evaluate(self.source['reference_omega_body_rad_s'][:1], time)[:,0]
        # The known torque is sampled at 1 kHz and linearly interpolated,
        # so finite error against the analytic motion is expected.
        np.testing.assert_allclose(states[:,:3],
            self.source['reference_omega_body_rad_s'], rtol=0., atol=4e-5)
        reconstructed = Rotation.from_quat(states[:,[4,5,6,3]])
        truth = Rotation.from_quat(self.source['reference_quaternion_xyzw'])
        self.assertLess(float(np.max((truth.inv()*reconstructed).magnitude())), 1e-4)

    def test_only_permitted_sparse_inputs_reach_the_estimator(self):
        self.assertEqual(set(self.selected),
            {'time','initial_quaternion_wxyz','sparse_gyro_time','sparse_gyro'})
        self.assertEqual(set(self.physical), {'inertia','torque'})
        np.testing.assert_array_equal(self.selected['sparse_gyro'], self.acquired['gyro_spike'])
        np.testing.assert_array_equal(self.selected['sparse_gyro_time'], self.acquired['time'])
        np.testing.assert_array_equal(self.selected['initial_quaternion_wxyz'],
                                       self.source['initial_quaternion_wxyz'])
        self.assertEqual(len(self.selected['sparse_gyro']), 201)

    def test_q_zero_bayes_keeps_all_prefix_likelihoods_and_matches_linear_reference(self):
        # A short prefix suffices to expose fresh-Gaussian replacement,
        # doubled observations, a nonzero process covariance, or future data.
        data = {name: value.copy() for name, value in self.selected.items()}
        data['sparse_gyro_time'] = data['sparse_gyro_time'][:5]
        data['sparse_gyro'] = data['sparse_gyro'][:5]
        data['time'] = np.array([0., data['sparse_gyro_time'][-1]])
        case = bayes.predict_full_rate(data, self.physical, order=7,
            sigma_deg_s=5., max_step=.002, mode='linear_forced',
            refresh_seconds=.1, initial_dense_seconds=.25)
        zero_flow = native.ControlledFlow(self.physical,
            self.source['initial_quaternion_wxyz'], *data['time'],
            max_step=.002, mode='linear_forced')
        drift = zero_flow.evaluate(np.zeros((1,3)), data['sparse_gyro_time'])[:,0,:3]
        variance = np.deg2rad(5.)**2
        for index in range(5):
            for side in ((1,) if index==0 else (-1,1)):
                row = np.flatnonzero((case['acquisition_index']==index)
                                     &(case['event_side']==side))
                self.assertEqual(len(row), 1)
                row = int(row[0])
                included = index+1 if side==1 else index
                values = data['sparse_gyro'][:included]-drift[:included]
                expected_mean = values.mean(axis=0)+drift[index]
                np.testing.assert_allclose(case['omega_mean_rad_s'][row],
                    expected_mean, rtol=0., atol=2e-7)
                np.testing.assert_allclose(case['omega_covariance_rad2_s2'][row],
                    np.eye(3)*variance/included, rtol=0., atol=2e-8)
                self.assertEqual(case['target_likelihood_factor_count'][row], included-1)


if __name__ == '__main__':
    unittest.main()
