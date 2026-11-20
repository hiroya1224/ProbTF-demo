"""Restore the saved spikes without changing motion or measurement clocks."""
from pathlib import Path
import importlib.util
import json
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name,HERE/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load('_student_input_test_runner','student_t_run.py')


class RestoredStudentInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = HERE/'omega3_q0_publication'
        cls.config = json.loads((HERE/'student_t_config.json').read_text())
        cls.config['spike_scale'] = 1.
        cls.source,cls.sensor,cls.acquired,cls.selected,cls.physical,cls.transformation = (
            runner.prepare_inputs(cls.baseline,cls.config))
        cls.old_source = dict(np.load(cls.baseline/'source_input.npz'))
        cls.old_sensor = dict(np.load(cls.baseline/'sensor_input.npz'))
        cls.old_acquired = dict(np.load(cls.baseline/'acquisitions.npz'))
        cls.full_sensor = dict(np.load(HERE/'frequent_publication/sensor_input.npz'))

    def test_restore_changes_only_pulse_amplitudes_and_recomposes_measurements(self):
        amplitude = ('spike_vector_rad_s','event_vector_body_rad_s','event_magnitude_deg_s')
        for name in amplitude:
            np.testing.assert_array_equal(self.sensor[name],2.*self.old_sensor[name],err_msg=name)
            np.testing.assert_array_equal(self.sensor[name],self.full_sensor[name],err_msg=name)
        for name,value in self.old_sensor.items():
            if name not in (*amplitude,'gyro_spike'):
                np.testing.assert_array_equal(self.sensor[name],value,err_msg=name)
        np.testing.assert_array_equal(self.sensor['gyro_spike'],
            self.old_sensor['gyro_white']+self.full_sensor['spike_vector_rad_s'])

    def test_motion_initial_pose_torque_and_white_noise_are_exactly_preserved(self):
        self.assertEqual(set(self.source),set(self.old_source))
        for name,value in self.old_source.items():
            np.testing.assert_array_equal(self.source[name],value,err_msg=name)
        self.assertEqual(set(self.physical),{'inertia','torque'})
        np.testing.assert_array_equal(self.physical['inertia'],self.old_source['inertia_body_kg_m2'])
        np.testing.assert_array_equal(self.sensor['gyro_white'],self.old_sensor['gyro_white'])

    def test_every_acquisition_keeps_its_original_clock_index_and_pulse_mask(self):
        self.assertEqual(len(self.acquired['time']),201)
        index = self.acquired['native_array_index']
        np.testing.assert_array_equal(index,np.arange(0,4001,20))
        for name,value in self.old_acquired.items():
            if name not in ('gyro_spike','spike_vector_rad_s'):
                np.testing.assert_array_equal(self.acquired[name],value,err_msg=name)
        for name in ('time','stamp_ns','record_ns','source_index','gyro_spike',
                     'spike_mask','spike_vector_rad_s'):
            np.testing.assert_array_equal(self.acquired[name],self.sensor[name][index],err_msg=name)
        self.assertEqual(int(np.count_nonzero(self.acquired['spike_mask'])),61)

    def test_estimator_receives_only_the_same_permitted_sparse_inputs(self):
        self.assertEqual(set(self.selected),
            {'time','initial_quaternion_wxyz','sparse_gyro_time','sparse_gyro'})
        np.testing.assert_array_equal(self.selected['time'],self.source['time'])
        np.testing.assert_array_equal(self.selected['initial_quaternion_wxyz'],
            self.source['initial_quaternion_wxyz'])
        np.testing.assert_array_equal(self.selected['sparse_gyro_time'],self.acquired['time'])
        np.testing.assert_array_equal(self.selected['sparse_gyro'],self.acquired['gyro_spike'])


if __name__=='__main__':
    unittest.main()
