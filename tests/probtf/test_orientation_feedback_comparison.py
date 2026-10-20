"""Verify the distinction between displayed and recursively retained evidence."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'ros/examples/probtf_orientation_demo/scripts/compare_feedback_orientation.py'
spec = importlib.util.spec_from_file_location('orientation_feedback_comparison', SCRIPT)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class FeedbackComparisonTest(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(5, dtype=float)*0.04
        self.gyro = np.tile([0.1, -0.2, 0.3], (5, 1))
        self.acc = np.tile([0.6, -0.4, 9.8], (5, 1))
        self.mag = np.tile([0.2, 0.8, -0.5], (5, 1))
        self.q = np.array([1., 0., 0., 0.])
        self.reference = np.array([1., 0., 0.])
        self.options = dict(initial_concentration=20., gyro_variance=0.,
                            acceleration_concentration=5., magnetic_concentration=5.)

    def run_six(self, acceleration=None, magnetic=None):
        return comparison.compare_six(self.gyro, self.acc if acceleration is None else acceleration,
            self.mag if magnetic is None else magnetic, self.times, self.q, self.reference, **self.options)

    def test_output_only_evidence_is_not_accumulated(self):
        self.gyro[:] = 0
        output, _, retained, _ = self.run_six()
        initial = 20.*(np.outer(self.q, self.q)-np.eye(4))
        gravity, field = comparison.direction_factors(self.acc, self.mag, self.reference, 5., 5.)
        canonical = comparison.original.canonical_bingham_parameter
        for k in range(5):
            np.testing.assert_allclose(retained[k, 0], initial, atol=2e-12)
            np.testing.assert_allclose(output[k, 1], canonical(initial+gravity[k]), atol=2e-12)
            np.testing.assert_allclose(output[k, 3], canonical(initial+gravity[k]+field[k]), atol=2e-12)
            np.testing.assert_allclose(output[k, 4], canonical(initial+(k+1)*gravity[k]+field[k]), atol=2e-12)
            np.testing.assert_allclose(output[k, 5], canonical(initial+(k+1)*(gravity[k]+field[k])), atol=2e-12)

    def test_past_magnetic_data_cannot_change_partial_feedback_output(self):
        other_mag = self.mag.copy()
        other_mag[:-1] = [0.8, -0.2, 0.5]
        first, _, state_a, _ = self.run_six()
        second, _, state_b, _ = self.run_six(magnetic=other_mag)
        np.testing.assert_allclose(state_a[:, :2], state_b[:, :2], atol=1e-12)
        np.testing.assert_allclose(first[-1, :5], second[-1, :5], atol=1e-12)
        self.assertGreater(np.linalg.norm(first[-1, 5]-second[-1, 5]), 1.)

    def test_each_measured_noise_weight_is_applied_once_at_its_own_time(self):
        self.gyro[:] = 0
        acceleration_weights = np.array([2., 5., 3., 7., 4.])
        magnetic_weights = np.array([3., 2., 6., 4., 8.])
        options = dict(self.options, acceleration_concentration=acceleration_weights,
                       magnetic_concentration=magnetic_weights)
        output, _, retained, _ = comparison.compare_six(
            self.gyro, self.acc, self.mag, self.times, self.q, self.reference, **options)
        gravity, field = comparison.direction_factors(
            self.acc, self.mag, self.reference, acceleration_weights, magnetic_weights)
        initial = 20.*(np.outer(self.q, self.q)-np.eye(4))
        canonical = comparison.original.canonical_bingham_parameter
        for k in range(len(self.times)):
            np.testing.assert_allclose(output[k, 1], canonical(initial+gravity[k]), atol=2e-12)
            np.testing.assert_allclose(retained[k, 1], canonical(initial+gravity[:k+1].sum(axis=0)), atol=2e-12)
            np.testing.assert_allclose(retained[k, 2], canonical(initial+(gravity+field)[:k+1].sum(axis=0)), atol=2e-12)

    def test_past_acceleration_changes_only_feedback_cases_at_final_sample(self):
        other_acc = self.acc.copy()
        other_acc[:-1] = [9.8, -0.4, 0.6]
        first, _, _, _ = self.run_six()
        second, _, _, _ = self.run_six(acceleration=other_acc)
        np.testing.assert_allclose(first[-1, [0, 1, 3]], second[-1, [0, 1, 3]], atol=1e-12)
        for case in (2, 4, 5):
            self.assertGreater(np.linalg.norm(first[-1, case]-second[-1, case]), 1.)

    def test_recursion_matches_core_sensor_update_sequence(self):
        parameters,modes,residuals = comparison.original.compare_conditions(
            self.gyro,self.acc,self.mag,self.times,self.q,self.reference,**self.options)
        actual,actual_modes,retained,actual_residuals = self.run_six()
        for new,core in ((0,0),(2,1),(5,4)):
            np.testing.assert_allclose(actual[:,new],parameters[:,core],atol=2e-12)
            np.testing.assert_allclose(actual_modes[:,new],modes[:,core],atol=2e-12)
        np.testing.assert_allclose(retained,parameters[:,[0,1,4]],atol=2e-12)
        np.testing.assert_allclose(actual_residuals,residuals[:,[0,1,3]],atol=2e-12)

    def test_prediction_uses_right_endpoint_and_rejects_duplicate_time(self):
        self.gyro[0] = [100., 100., 100.]
        first, _, _, _ = self.run_six()
        self.gyro[0] = [-100., -100., -100.]
        second, _, _, _ = self.run_six()
        np.testing.assert_array_equal(first, second)
        self.gyro[1] *= -1
        third, _, _, _ = self.run_six()
        self.assertGreater(np.linalg.norm(first[1:, 0]-third[1:, 0]), 0.1)
        self.times[1] = self.times[0]
        with self.assertRaisesRegex(ValueError, 'strictly increase'):
            self.run_six()

    def test_reader_respects_common_measurement_domain(self):
        boundary = comparison.EXPERIMENT_START_STAMP
        def stamp(value):
            return SimpleNamespace(to_sec=lambda:value)
        def imu(value,data):
            return SimpleNamespace(stamp=stamp(value),gyro_data=data,acc_data=[0.,0.,9.8],mag_data=[1.,0.,0.])
        def mocap(value):
            return SimpleNamespace(header=SimpleNamespace(stamp=stamp(value),frame_id='world'),
                pose=SimpleNamespace(orientation=SimpleNamespace(w=1.,x=0.,y=0.,z=0.),
                                     position=SimpleNamespace(x=0.,y=0.,z=0.)))
        events = [('/delta/imu',imu(boundary-.1,[float('nan')]*3),stamp(boundary+.01)),
                  ('/delta/imu',imu(boundary+.02,[0.,0.,0.]),stamp(boundary+.03)),
                  ('/delta/mocap/pose',mocap(boundary+.02),stamp(boundary+.03)),
                  ('/delta/imu',imu(boundary+.04,[0.,0.,0.]),stamp(boundary+.05)),
                  ('/delta/mocap/pose',mocap(boundary+.04),stamp(boundary+.05))]
        class Bag:
            def __init__(inner,*args):pass
            def __enter__(inner):return inner
            def __exit__(inner,*args):pass
            def read_messages(inner,topics,start_time):
                self.assertAlmostEqual(start_time.to_sec(),boundary)
                return iter(events)
        with patch('rosbag.Bag',Bag):
            values,poses,frame = comparison.read_measurements(Path('20250303_delta_circle_trajectory_flight_5_2025-03-03-13-20-43.bag'))
        self.assertEqual(frame,'world')
        self.assertEqual(len(values),2)
        self.assertEqual(len(poses),2)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values[:,1] >= boundary).all())

    def test_initialization_and_raw_position_share_the_measurement_times(self):
        times = comparison.EXPERIMENT_START_STAMP+np.arange(250)*.01
        rotations = comparison.original.Rotation.from_euler('z',np.arange(250)*.001)
        quaternions = comparison.original.wxyz_from_rotations(rotations)
        reference = np.array([1.,0.,-1.])/np.sqrt(2.)
        field = rotations.inv().apply(reference)
        imu = np.column_stack((times,times,np.zeros((250,3)),np.tile([0.,0.,9.8],(250,1)),field))
        poses = np.column_stack((times,times,quaternions,np.tile([.1,.2,.3],(250,1))))
        arrays,actual_reference,metadata = comparison.prepare_measurements(imu,poses)
        np.testing.assert_allclose(actual_reference,reference,atol=1e-12)
        np.testing.assert_allclose(metadata['initial_quaternion_wxyz'],arrays['mocap_wxyz'][0],atol=1e-12)
        np.testing.assert_array_equal(arrays['position_measurement_stamps'],arrays['measurement_stamps'])
        np.testing.assert_array_equal(arrays['position_time_seconds'],arrays['time_seconds'])
        self.assertEqual(arrays['time_seconds'][0],0.)
        self.assertEqual(metadata['calibration_samples'],100)
        np.testing.assert_array_equal(arrays['acceleration_m_s2'],imu[100:,5:8])

    def test_acceleration_display_limit_preserves_sensor_values(self):
        acceleration = np.array([[0.,0.,9.8],[1.,2.,50.],[0.,0.,9.8]])
        original_values = acceleration.copy()
        arrays = dict(time_seconds=np.arange(3),position_time_seconds=np.arange(3),
            position_m=np.zeros((3,3)),gyro_rad_s=np.zeros((3,3)),acceleration_m_s2=acceleration,
            magnetic_raw=np.ones((3,3)))
        with patch.object(comparison.original,'save_figure'):
            limits = comparison.plot_raw_measurements(arrays,Path('/tmp'))
        np.testing.assert_array_equal(acceleration,original_values)
        np.testing.assert_allclose(np.array(limits['acc'])[:,1],12.)


if __name__ == '__main__':
    unittest.main()
