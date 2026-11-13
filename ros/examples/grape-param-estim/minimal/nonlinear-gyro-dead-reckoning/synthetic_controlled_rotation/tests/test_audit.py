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


if __name__ == '__main__':
    unittest.main()
