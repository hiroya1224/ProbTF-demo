"""All-native held-out gyro clocks, frozen physics, and inference boundary."""
from pathlib import Path
import importlib.util
import json
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_real_student_full_input_tests', HERE/'input.py')
inputs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = inputs
spec.loader.exec_module(inputs)


class FullNativeRealFlightInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = dict(np.load(inputs.ARCHIVE/'input.npz', allow_pickle=False))
        cls.metadata = json.loads((inputs.ARCHIVE/'input.json').read_text())
        cls.profile = dict(np.load(inputs.REFINED/'refined_torque_profile.npz', allow_pickle=False))
        cls.index = np.flatnonzero((cls.raw['imu_time'] >= cls.raw['time'][0])
                                  & (cls.raw['imu_time'] <= cls.raw['time'][1]))

    def test_every_native_value_and_original_clock_is_preserved(self):
        source, acquired, selected, physical, provenance = inputs.load_full_inputs()
        self.assertEqual(len(self.index), 1001)
        self.assertEqual(len(acquired['time']), len(self.index))
        for key, raw_key in (('time', 'imu_time'), ('gyro', 'gyro'),
                             ('header_stamp_ns', 'imu_stamp_ns'),
                             ('record_stamp_ns', 'imu_record_ns'),
                             ('original_source_index', 'imu_source_index')):
            np.testing.assert_array_equal(acquired[key], self.raw[raw_key][self.index], err_msg=key)
        np.testing.assert_array_equal(acquired['native_array_index'], self.index)
        np.testing.assert_array_equal(selected['sparse_gyro_time'], acquired['time'])
        np.testing.assert_array_equal(selected['sparse_gyro'], acquired['gyro'])
        np.testing.assert_array_equal(selected['time'], self.raw['time'])
        np.testing.assert_array_equal(selected['initial_quaternion_wxyz'], self.raw['initial_quaternion_wxyz'])
        self.assertTrue(np.all(np.diff(acquired['time']) > 0.))
        self.assertEqual(acquired['time'][0], selected['time'][0])
        self.assertLess(acquired['time'][-1], selected['time'][-1])
        self.assertEqual(int(np.rint(selected['time'][-1]*1e9)), 55_000_988_619)
        np.testing.assert_array_equal(source['gyro_time'], acquired['time'])
        np.testing.assert_array_equal(source['gyro_body_rad_s'], acquired['gyro'])

    def test_physics_uses_frozen_inertia_and_past_residual_exactly_once(self):
        selected, acquired, physical = inputs.prepare_full_prediction_inputs(
            self.raw, self.metadata, self.profile)
        np.testing.assert_array_equal(physical['inertia'], self.profile['inertia_body'])
        times = np.r_[self.profile['torque_profile_time'][::61], self.raw['time']]
        actuator = np.column_stack([np.interp(times, self.profile['torque_profile_time'],
            self.profile['torque_actuator_body'][:, axis]) for axis in range(3)])
        applied = np.column_stack([np.interp(times, self.profile['torque_profile_time'],
            self.profile['torque_applied_body'][:, axis]) for axis in range(3)])
        np.testing.assert_allclose(physical['torque'](times), applied, rtol=0., atol=2e-16)
        np.testing.assert_allclose(physical['torque'](times)-actuator,
            np.broadcast_to(self.profile['residual_mean_body'], actuator.shape), rtol=0., atol=2e-16)

    def test_references_and_gyro_outside_the_window_cannot_enter_inference(self):
        before, acquired, physical = inputs.prepare_full_prediction_inputs(
            self.raw, self.metadata, self.profile)
        self.assertEqual(set(before), {'time', 'initial_quaternion_wxyz', 'sparse_gyro_time', 'sparse_gyro'})
        self.assertEqual(set(physical), {'inertia', 'torque'})
        poisoned = {key: value.copy() for key, value in self.raw.items()}
        retained = np.zeros(len(poisoned['gyro']), dtype=bool)
        retained[self.index] = True
        poisoned['gyro'][~retained] = np.nan
        for key in ('mocap_quaternion_xyzw', 'mocap_position', 'dense_gyro',
                    'calibration_gyro', 'past_calibration_gyro'):
            poisoned[key][...] = np.nan
        after, evidence, changed_physical = inputs.prepare_full_prediction_inputs(
            poisoned, self.metadata, self.profile)
        for key in before:
            np.testing.assert_array_equal(after[key], before[key], err_msg=key)
        for key in acquired:
            np.testing.assert_array_equal(evidence[key], acquired[key], err_msg=key)
        np.testing.assert_array_equal(changed_physical['inertia'], physical['inertia'])
        np.testing.assert_array_equal(changed_physical['torque'](self.raw['time']),
                                      physical['torque'](self.raw['time']))

    def test_preparation_does_not_mutate_archived_arrays(self):
        raw_before = {key: value.copy() for key, value in self.raw.items()}
        profile_before = {key: value.copy() for key, value in self.profile.items()}
        selected, acquired, physical = inputs.prepare_full_prediction_inputs(
            self.raw, self.metadata, self.profile)
        selected['sparse_gyro'][0, 0] += 1.
        acquired['gyro'][0, 0] -= 1.
        physical['inertia'][0, 0] += 1.
        for key in self.raw:
            np.testing.assert_array_equal(self.raw[key], raw_before[key], err_msg=key)
        for key in self.profile:
            np.testing.assert_array_equal(self.profile[key], profile_before[key], err_msg=key)


if __name__ == '__main__':
    unittest.main()
