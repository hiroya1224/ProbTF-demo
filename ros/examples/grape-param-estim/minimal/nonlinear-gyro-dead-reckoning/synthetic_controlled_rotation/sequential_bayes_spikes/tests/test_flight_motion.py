"""Independent kinematic and provenance checks for flight-calibrated motion.

The calibration changes the synthetic motion, not the random sensor
realization, acquisition clock, observation model, or Bayes input interface.
These tests never run the inverse calibration or inspect estimation scores.
"""
from pathlib import Path
import importlib.util
import json
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inputs = load('flight_motion_test_inputs', HERE/'frequent_spikes_input.py')
runner = load('flight_motion_test_runner', HERE/'frequent_run.py')
trajectory = inputs._trajectory


def derivative(values, step):
    """Fourth-order central difference, independent of analytic derivatives."""
    minus2, minus1, plus1, plus2 = values
    return (minus2-8.*minus1+8.*plus1-plus2)/(12.*step)


class FlightMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((HERE/'frequent_config.json').read_text())
        cls.profile_path = HERE/cls.config['motion_parameters_file']
        cls.parameters = json.loads(cls.profile_path.read_text())
        cls.values = inputs.generate(cls.config)
        cls.legacy_config = dict(cls.config, motion_profile='three_frequency')
        del cls.legacy_config['motion_parameters_file']
        cls.legacy = inputs.generate(cls.legacy_config)

    def evaluate(self, stamps):
        return trajectory.trajectory(stamps, self.config['motion_profile'], self.parameters)

    def numerical_derivatives(self):
        # Non-grid queries exercise the actual smooth curve, independently
        # of its 200 Hz sensor and 1 kHz known-torque sampling grids.
        stamps = np.linspace(.037, self.config['duration_seconds']-.041, 127)
        step = 1e-4
        value = self.evaluate(stamps)
        neighbors = [self.evaluate(stamps+offset*step) for offset in (-2.,-1.,1.,2.)]
        rotation = Rotation.from_quat(value['quaternion_xyzw']).as_matrix()
        rotation_dot = derivative([
            Rotation.from_quat(item['quaternion_xyzw']).as_matrix()
            for item in neighbors], step)
        skew = np.einsum('tji,tjk->tik', rotation, rotation_dot)
        omega = .5*np.column_stack((skew[:,2,1]-skew[:,1,2],
                                    skew[:,0,2]-skew[:,2,0],
                                    skew[:,1,0]-skew[:,0,1]))
        alpha = derivative([item['omega_body_rad_s'] for item in neighbors], step)
        return value, omega, alpha

    def test_calibrated_omega_and_alpha_match_differentiated_quaternion_rotation(self):
        value, omega, alpha = self.numerical_derivatives()
        np.testing.assert_allclose(value['omega_body_rad_s'], omega,
                                   rtol=2e-8, atol=2e-9)
        np.testing.assert_allclose(value['alpha_body_rad_s2'], alpha,
                                   rtol=2e-8, atol=2e-9)
        np.testing.assert_allclose(np.linalg.norm(value['quaternion_xyzw'], axis=1),
                                   1., rtol=0., atol=4e-16)

    def test_saved_input_torque_satisfies_nonlinear_euler_for_the_new_motion(self):
        value, omega, alpha = self.numerical_derivatives()
        source = self.values[0]
        inertia = source['inertia_body_kg_m2']
        # Use finite-difference alpha and omega, not the torque builder's
        # own analytic acceleration, as the independent balance check.
        expected = alpha@inertia.T+np.cross(omega, omega@inertia.T)
        np.testing.assert_allclose(value['torque_body_nm'], expected,
                                   rtol=3e-8, atol=3e-10)
        control = self.evaluate(source['torque_time'])
        np.testing.assert_allclose(source['torque_body_nm'], control['torque_body_nm'],
                                   rtol=3e-14, atol=3e-14)
        np.testing.assert_array_equal(inertia, trajectory.INERTIA)
        q0 = self.evaluate(0.)['quaternion_xyzw'][[3,0,1,2]]
        np.testing.assert_array_equal(source['initial_quaternion_wxyz'], q0)

    def test_generator_uses_the_frozen_calibration_at_the_declared_physical_scale(self):
        calibration = json.loads((self.profile_path.parent/'calibration.json').read_text())
        with np.load(self.profile_path.parent/'comparison.npz', allow_pickle=False) as saved:
            value = self.evaluate(saved['synthetic_local_time'])
            np.testing.assert_allclose(value['rpy_rad'], saved['new_synthetic_rpy_rad'],
                                       rtol=2e-14, atol=2e-14)
            np.testing.assert_allclose(value['omega_body_rad_s'],
                                       saved['new_synthetic_body_omega_rad_s'],
                                       rtol=2e-14, atol=2e-14)
            target = self.parameters['physical_motion_std_multiplier']*np.std(
                saved['real_sg_body_omega_rad_s'], axis=0)
            np.testing.assert_allclose(np.std(value['omega_body_rad_s'], axis=0),
                                       target, rtol=2e-12, atol=2e-14)
            for dimension, row in enumerate(calibration['spectrum']['per_axis']):
                np.testing.assert_array_equal(self.parameters['frequency_hz'][dimension],
                                              row['frequency_hz'])
        self.assertEqual(self.parameters['source_window_seconds'], [40.,60.])
        self.assertEqual(self.parameters['physical_motion_std_multiplier'], 1.5)
        self.assertFalse(self.parameters['selection_uses_reconstruction_result'])
        self.assertFalse(calibration['amplitude_calibration']['filters_or_estimated_covariances_used'])

    def test_motion_change_preserves_every_noise_draw_spike_event_and_original_clock(self):
        source, selected, acquired, sensor, events, _ = self.values
        previous, old_selected, old_acquired, old_sensor, old_events, _ = self.legacy
        for name in ('gyro_realized_noise_rad_s','gyro_bias_rad_s',
                     'generator_gyro_covariance_rad2_s2','inertia_body_kg_m2',
                     'imu_time','imu_stamp_ns','imu_record_ns','imu_source_index',
                     'reference_time','torque_time','time'):
            np.testing.assert_array_equal(source[name], previous[name], err_msg=name)
        self.assertEqual(events, old_events)
        for name in sensor:
            if not name.startswith('gyro_'):
                np.testing.assert_array_equal(sensor[name], old_sensor[name], err_msg=name)
        for name in acquired:
            if not name.startswith('gyro_'):
                np.testing.assert_array_equal(acquired[name], old_acquired[name], err_msg=name)
        np.testing.assert_array_equal(selected['sparse_gyro_time'],
                                      old_selected['sparse_gyro_time'])
        movement_change = source['gyro_true_body_rad_s']-previous['gyro_true_body_rad_s']
        for name in ('gyro_white','gyro_spike'):
            np.testing.assert_allclose(sensor[name]-old_sensor[name], movement_change,
                                       rtol=2e-14, atol=2e-15)
        # This is a changed experiment, not a metadata-only relabeling.
        self.assertGreater(np.linalg.norm(movement_change), .1)
        self.assertGreater(np.linalg.norm(source['torque_body_nm']-previous['torque_body_nm']), .1)

    def test_estimator_receives_only_same_allowed_inputs_and_shared_corrupted_measurements(self):
        source, selected, acquired, _, _, metadata = self.values
        self.assertEqual(set(selected), {'time','initial_quaternion_wxyz',
                                        'sparse_gyro_time','sparse_gyro'})
        physical = dict(inertia=source['inertia_body_kg_m2'],
            torque=runner.shared.TorqueProfile(source['torque_time'], source['torque_body_nm']))
        calls = []

        class Recorder:
            @staticmethod
            def predict_adaptive(data, model, **options):
                calls.append((data, model, options))
                return {'shared': {'recorded': True}}

        runner.predict_models(Recorder, selected, physical, acquired,
                              self.config, self.config['quadrature_orders'][0])
        self.assertEqual(len(calls), 2)
        for data, model, options in calls:
            self.assertEqual(set(data), set(selected))
            self.assertEqual(set(model), {'inertia','torque'})
            self.assertIs(model, physical)
            np.testing.assert_array_equal(data['sparse_gyro'], acquired['gyro_spike'])
            np.testing.assert_array_equal(data['sparse_gyro_time'], acquired['time'])
            np.testing.assert_array_equal(data['initial_quaternion_wxyz'],
                                          source['initial_quaternion_wxyz'])
            self.assertEqual(set(options), {'sigma_deg_s','order','output_dt',
                                           'max_step','observation_sets','mode'})
            self.assertEqual(options['sigma_deg_s'], 5.)
            np.testing.assert_array_equal(options['observation_sets']['shared'],
                                          acquired['gyro_spike'])
        self.assertEqual([options['mode'] for _, _, options in calls],
                         ['nonlinear','linear_forced'])
        self.assertFalse(metadata['generation']['motion_calibration']
                         ['motion_parameters_supplied_to_estimator'])
        self.assertTrue(metadata['native_spikes_not_supplied_to_estimator'])

    def test_original_default_profiles_replay_saved_inputs_without_calibration(self):
        publications = HERE.parent/'publication'
        for motion, directory in ((None, 'smooth_rotation_5s'),
                                  ('three_frequency', 'three_frequency_0p5hz_sigma5')):
            with self.subTest(motion=motion):
                data, _ = (trajectory.generate_dataset() if motion is None
                           else trajectory.generate_dataset(motion_profile=motion))
                with np.load(publications/directory/'input.npz', allow_pickle=False) as saved:
                    for key in saved.files:
                        if (saved[key].dtype.kind != 'f'
                                or key in ('gyro_realized_noise_rad_s','gyro_bias_rad_s',
                                           'generator_gyro_covariance_rad2_s2')):
                            np.testing.assert_array_equal(data[key], saved[key], err_msg=key)
                        else:
                            np.testing.assert_allclose(data[key], saved[key],
                                rtol=3e-14, atol=3e-14, err_msg=key)


if __name__ == '__main__':
    unittest.main()
