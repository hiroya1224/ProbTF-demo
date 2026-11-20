"""Sparse original clocks and a fixed process increment per selected gyro."""
from pathlib import Path
import importlib.util
import json
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parents[1]


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verify = load('_selected_process_native_verification_tests',HERE/'process_verify.py')
inputs = verify.inputs
engine = load('_selected_process_native_engine_tests',HERE/'student_t_process_engine.py')
RATES = (.5,2.,5.,10.)
COUNTS = (3,10,25,50)


class SelectedProcessGyroInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = dict(np.load(inputs.ARCHIVE/'input.npz',allow_pickle=False))
        cls.metadata = json.loads((inputs.ARCHIVE/'input.json').read_text())
        cls.profile = dict(np.load(inputs.REFINED/'refined_torque_profile.npz',allow_pickle=False))

    def selected(self,rate):
        return inputs.prepare_prediction_inputs(self.raw,self.metadata,self.profile,rate)

    def test_all_requested_rates_preserve_native_values_and_original_clocks(self):
        for rate,count in zip(RATES,COUNTS):
            with self.subTest(rate=rate):
                selected,acquired,_ = self.selected(rate)
                report = verify.selected_acquisition_checks(acquired,rate,self.raw,self.metadata)
                self.assertEqual(report['expected_count'],count)
                self.assertEqual(report['actual_count'],count)
                self.assertTrue(report['all_selected_native_readings_exact'])
                self.assertTrue(report['selected_readings_inside_fixed_window'])
                self.assertEqual(report['original_window_native_record_count'],1001)
                np.testing.assert_array_equal(selected['sparse_gyro'],acquired['gyro'])
                np.testing.assert_array_equal(selected['sparse_gyro_time'],acquired['time'])
                self.assertEqual(set(selected),{'time','initial_quaternion_wxyz','sparse_gyro_time','sparse_gyro'})

    def test_verifier_detects_changed_ns_or_original_source_correspondence(self):
        _,acquired,_ = self.selected(5.)
        altered = {key:value.copy() for key,value in acquired.items()}
        altered['header_stamp_ns'][4] += 1
        altered['original_source_index'][7] += 1
        report = verify.selected_acquisition_checks(altered,5.,self.raw,self.metadata)
        self.assertFalse(report['all_selected_native_readings_exact'])
        self.assertFalse(report['original_values_indices_and_clocks_exact']['header_stamp_ns'])
        self.assertFalse(report['original_values_indices_and_clocks_exact']['original_source_index'])

    def test_process_q_is_fixed_per_selected_record_not_rescaled_by_interval(self):
        # Spherical inertia and zero torque provide an independent exact
        # omega prediction, regardless of each selected native interval.
        physical = dict(inertia=.03*np.eye(3),torque=np.zeros(3))
        covariance = np.eye(3)*np.deg2rad(2.5)**2
        process = np.eye(3)*np.deg2rad(1.)**2
        for rate,count in zip(RATES,COUNTS):
            with self.subTest(rate=rate):
                selected,acquired,_ = self.selected(rate)
                case = engine.predict_process_student(selected,physical,sigma_deg_s=2.5,
                    process_sigma_deg_s=1.,student_df=4.,order=3,mode='nonlinear',output_dt=.01)
                np.testing.assert_array_equal(case['process_covariance_rad2_s2'],process)
                first_prior = np.flatnonzero((case['event_side']==-1)&(case['acquisition_index']==1))[0]
                np.testing.assert_allclose(case['omega_covariance_rad2_s2'][first_prior],
                    covariance+process,rtol=0.,atol=3e-16)
                report = verify.process_application_checks(case,acquired,dict(process_sigma_deg_s=1.))
                self.assertEqual(report['actual_process_increment_count'],count-1)
                self.assertTrue(report['applied_at_native_acquisition_endpoints_only'])
                self.assertTrue(report['no_extra_process_increment_for_dense_queries_or_last_tail'])
                self.assertTrue(report['increment_times_exact'])
                self.assertEqual(case['target_likelihood_factor_count'][-1],count-1)


if __name__=='__main__':
    unittest.main()
