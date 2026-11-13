"""Known-motion recovery and unchanged-input checks for the new experiment."""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
for directory in (HERE, HERE.parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from trajectory import generate_dataset, INERTIA
from audit import audit_dataset
from torque import TorqueProfile


class SyntheticAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, _ = generate_dataset()
        cls.original = {key: np.asarray(value).copy() for key, value in cls.data.items()}
        profile = dict(torque=TorqueProfile(cls.data['torque_time'], cls.data['torque_body_nm']),
                       inertia=INERTIA)
        cls.result = audit_dataset(cls.data, profile)

    def test_body_kinematics_and_euler_torque_agree(self):
        self.assertLess(self.result['analytic_body_omega_vs_rotation_derivative_max_abs_rad_s'], 1e-6)
        self.assertLess(self.result['analytic_body_alpha_vs_omega_derivative_max_abs_rad_s2'], 1e-6)
        self.assertLess(self.result['torque_equation_max_abs_residual_nm'], 1e-12)

    def test_noiseless_matched_model_recovers_prescribed_rotation(self):
        experiment = self.result['separate_zero_noise_no_bias_unit_experiment']
        self.assertFalse(experiment['primary_noisy_dataset_or_law_replaced'])
        for item in experiment['rates'].values():
            self.assertEqual(set(item['methods']), {'nonlinear', 'linear_forced'})
            self.assertLess(item['methods']['nonlinear']['geodesic_max_deg'], .01)

    def test_diagnostic_does_not_modify_primary_noise_or_truth(self):
        for key, value in self.original.items():
            np.testing.assert_array_equal(value, self.data[key], err_msg=key)

    def test_three_frequency_profile_uses_its_own_reference_and_exact_sparse_clock(self):
        data, _ = generate_dataset(motion_profile='three_frequency')
        original = {key: np.asarray(value).copy() for key, value in data.items()}
        profile = dict(torque=TorqueProfile(data['torque_time'], data['torque_body_nm']),
                       inertia=INERTIA)
        result = audit_dataset(data, profile, rates=(.5,))
        self.assertEqual(result['motion_profile'], 'three_frequency')
        self.assertLess(result['analytic_body_omega_vs_rotation_derivative_max_abs_rad_s'], 1e-6)
        self.assertLess(result['analytic_body_alpha_vs_omega_derivative_max_abs_rad_s2'], 1e-6)
        self.assertLess(result['torque_equation_max_abs_residual_nm'], 1e-12)
        experiment = result['separate_zero_noise_no_bias_unit_experiment']
        self.assertEqual(experiment['requested_rates_hz'], [.5])
        self.assertEqual(set(experiment['rates']), {'0.5'})
        item = experiment['rates']['0.5']
        self.assertEqual(item['acquisition_source_index'], [0, 400, 800])
        self.assertLess(item['methods']['nonlinear']['geodesic_max_deg'], .01)
        for key, value in original.items():
            np.testing.assert_array_equal(value, data[key], err_msg=key)

    def test_old_saved_dataset_without_profile_keeps_two_frequency_reference(self):
        data = dict(self.data)
        data.pop('motion_profile', None)
        profile = dict(torque=TorqueProfile(data['torque_time'], data['torque_body_nm']),
                       inertia=INERTIA)
        result = audit_dataset(data, profile, rates=(.5,))
        self.assertEqual(result['motion_profile'], 'two_frequency')
        self.assertLess(result['torque_equation_max_abs_residual_nm'], 1e-12)


if __name__ == '__main__':
    unittest.main()
