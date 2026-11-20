"""Clock, physical-input, and no-oracle checks for held-out real gyro replay."""
from pathlib import Path
import importlib.util
import json
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


inputs = load('_real_student_input_checks', HERE/'input.py')


class RealFlightInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = dict(np.load(inputs.ARCHIVE/'input.npz',allow_pickle=False))
        cls.metadata = json.loads((inputs.ARCHIVE/'input.json').read_text())
        cls.profile = dict(np.load(inputs.REFINED/'refined_torque_profile.npz',allow_pickle=False))

    def test_original_acquisitions_preserve_requested_bag_clock_values_and_ns(self):
        for rate,count in ((.5,3),(2.,10),(5.,25),(10.,50)):
            with self.subTest(rate=rate):
                source,acquired,selected,physical,provenance = inputs.load_inputs(rate)
                start,end = self.metadata['requested_window_seconds']
                requested = start+np.arange(0.,end-start,1./rate)
                index = np.searchsorted(self.raw['imu_time'],requested,side='left')
                self.assertEqual(len(acquired['time']),count)
                np.testing.assert_array_equal(acquired['requested_time'],requested)
                np.testing.assert_array_equal(acquired['native_array_index'],index)
                np.testing.assert_array_equal(acquired['original_source_index'],self.raw['imu_source_index'][index])
                for key,raw_key in (('time','imu_time'),('gyro','gyro'),
                        ('header_stamp_ns','imu_stamp_ns'),('record_stamp_ns','imu_record_ns')):
                    np.testing.assert_array_equal(acquired[key],self.raw[raw_key][index],err_msg=key)
                np.testing.assert_array_equal(selected['sparse_gyro'],acquired['gyro'])
                np.testing.assert_array_equal(selected['sparse_gyro_time'],acquired['time'])
                self.assertEqual(selected['time'][-1],self.raw['time'][-1])
                self.assertEqual(int(np.rint(selected['time'][-1]*1e9)),55_000_988_619)
                self.assertGreater(selected['time'][-1],acquired['time'][-1])

    def test_frozen_torque_includes_strictly_past_residual_once(self):
        selected,acquired,physical = inputs.prepare_prediction_inputs(
            self.raw,self.metadata,self.profile,5.)
        np.testing.assert_array_equal(physical['inertia'],self.profile['inertia_body'])
        times = np.r_[self.profile['torque_profile_time'][::61],self.raw['time']]
        expected = np.column_stack([np.interp(times,self.profile['torque_profile_time'],
            self.profile['torque_applied_body'][:,axis]) for axis in range(3)])
        np.testing.assert_allclose(physical['torque'](times),expected,rtol=0.,atol=2e-16)
        delta = physical['torque'](times)-np.column_stack([np.interp(times,
            self.profile['torque_profile_time'],self.profile['torque_actuator_body'][:,axis])
            for axis in range(3)])
        np.testing.assert_allclose(delta,np.broadcast_to(self.profile['residual_mean_body'],delta.shape),
                                   rtol=0.,atol=2e-16)

    def test_filter_interface_contains_only_allowed_inputs(self):
        for rate in (.5,2.,5.,10.):
            selected,acquired,physical = inputs.prepare_prediction_inputs(self.raw,self.metadata,self.profile,rate)
            self.assertEqual(set(selected),{'time','initial_quaternion_wxyz','sparse_gyro_time','sparse_gyro'})
            self.assertEqual(set(physical),{'inertia','torque'})
            np.testing.assert_array_equal(selected['initial_quaternion_wxyz'],self.raw['initial_quaternion_wxyz'])
            np.testing.assert_array_equal(selected['time'],self.raw['time'])

    def test_unacquired_gyro_and_noninitial_mocap_cannot_enter_prediction(self):
        for rate in (.5,2.,5.,10.):
            before,acquired,physical = inputs.prepare_prediction_inputs(self.raw,self.metadata,self.profile,rate)
            poisoned = {key:value.copy() for key,value in self.raw.items()}
            retained = np.zeros(len(poisoned['gyro']),dtype=bool)
            retained[acquired['native_array_index']] = True
            poisoned['gyro'][~retained] = np.nan
            for key in ('mocap_quaternion_xyzw','mocap_position','dense_gyro','calibration_gyro',
                        'past_calibration_gyro'):
                poisoned[key][...] = np.nan
            after,evidence,changed_physical = inputs.prepare_prediction_inputs(poisoned,self.metadata,self.profile,rate)
            for key in before:
                np.testing.assert_array_equal(after[key],before[key],err_msg=key)
            for key in acquired:
                np.testing.assert_array_equal(evidence[key],acquired[key],err_msg=key)
            np.testing.assert_array_equal(changed_physical['inertia'],physical['inertia'])
            np.testing.assert_array_equal(changed_physical['torque'](self.raw['time']),physical['torque'](self.raw['time']))


if __name__=='__main__':
    unittest.main()
