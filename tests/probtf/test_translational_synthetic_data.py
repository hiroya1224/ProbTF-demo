"""Physics, clocks and truth boundary of the separate synthetic IMU cases."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT/'ros/examples/probtf_orientation_demo/scripts/translational_synthetic_data.py'
spec = importlib.util.spec_from_file_location('translational_synthetic_data', SCRIPT)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class TranslationalSyntheticDataTests(unittest.TestCase):
    def test_default_clock_and_magnetic_new_observation_mask(self):
        d = module.generate_comparison_data(duration=.12)
        a = d['arrays']; t = a['time_seconds']
        np.testing.assert_allclose(np.diff(t), .005, atol=1e-16)
        np.testing.assert_array_equal(np.flatnonzero(a['magnetic_observed']), np.arange(0,len(t),3))
        idx = a['magnetic_source_index']
        np.testing.assert_array_equal(a['magnetic_measurement_stamps'], t[idx])
        np.testing.assert_array_equal(a['magnetic_raw'], a['magnetic_raw'][idx])
        np.testing.assert_array_equal(a['magnetic_noise'], a['magnetic_noise'][idx])

    def test_specific_force_keeps_translational_acceleration_and_gravity_sign(self):
        d = module.generate_comparison_data(duration=2., scenario='smooth_translation')
        a = d['arrays']; r = Rotation.from_quat(a['truth_wxyz'][:,[1,2,3,0]])
        expected = r.inv().apply(a['acceleration_world_m_s2'] - module.GRAVITY_WORLD_M_S2)
        np.testing.assert_allclose(a['ideal_acceleration_m_s2'], expected, atol=1e-14)
        np.testing.assert_allclose(a['acceleration_body_m_s2'], r.inv().apply(a['acceleration_world_m_s2']), atol=1e-14)
        self.assertGreater(np.linalg.norm(a['acceleration_world_m_s2']), .1)

    def test_gyro_integrates_true_right_end_quaternion_increments(self):
        a = module.generate_comparison_data(duration=1.)['arrays']
        r = Rotation.from_quat(a['truth_wxyz'][:,[1,2,3,0]])
        predicted = r[:-1] * Rotation.from_rotvec(a['ideal_gyro_rad_s'][1:]*np.diff(a['time_seconds'])[:,None])
        np.testing.assert_allclose((predicted.inv()*r[1:]).magnitude(), 0., atol=4e-16)
        np.testing.assert_array_equal(a['ideal_gyro_rad_s'][0], np.zeros(3))

    def test_all_cases_share_attitude_and_realised_noise(self):
        cases = [module.generate_comparison_data(duration=1.,scenario=c) for c in module.SCENARIOS]
        for d in cases[1:]:
            for key in ('truth_wxyz','gyro_rad_s','magnetic_raw','gyro_noise','acceleration_noise'):
                np.testing.assert_array_equal(cases[0]['arrays'][key], d['arrays'][key])
        np.testing.assert_array_equal(cases[0]['arrays']['acceleration_world_m_s2'], np.zeros((201,3)))
        self.assertGreater(np.max(np.abs(cases[2]['arrays']['acceleration_world_m_s2'])), 1.)

    def test_generated_measurements_equal_saved_ideal_and_noise(self):
        d = module.generate_comparison_data(duration=.2)
        a = d['arrays']
        for measured,ideal,noise in [('gyro_rad_s','ideal_gyro_rad_s','gyro_noise'),
                ('acceleration_m_s2','ideal_acceleration_m_s2','acceleration_noise'),
                ('magnetic_raw','ideal_magnetic_raw','magnetic_noise')]:
            np.testing.assert_array_equal(a[measured], a[ideal]+a[noise])

    def test_anisotropic_covariances_are_positive_and_generation_differs(self):
        d = module.generate_comparison_data(duration=.1)
        for sensor, c in d['metadata']['estimation_noise_covariances'].items():
            c = np.asarray(c); generated = np.asarray(d['metadata']['generation_noise_covariances'][sensor])
            self.assertGreater(np.linalg.eigvalsh(c).min(), 0.)
            self.assertGreater(np.max(np.linalg.eigvalsh(c))/np.min(np.linalg.eigvalsh(c)), 1.1)
            self.assertGreater(np.max(np.abs(c - np.diag(np.diag(c)))), 0.)
            self.assertFalse(np.array_equal(c, generated))

    def test_estimator_inputs_exclude_truth_noise_and_generation_covariances(self):
        d = module.generate_comparison_data(duration=.1)
        inputs = module.estimator_inputs(d)
        self.assertFalse(any('truth' in key or 'noise' in key or 'ideal' in key or 'world_m_s2' in key and key!='gravity_world_m_s2' for key in inputs))
        np.testing.assert_array_equal(inputs['initial_wxyz'], d['arrays']['truth_wxyz'][0])
        np.testing.assert_array_equal(inputs['gyro_covariance'], d['metadata']['estimation_noise_covariances']['angular_velocity'])
        self.assertTrue(all(not value.flags.writeable for value in inputs.values()))
        old = inputs['initial_wxyz'].copy()
        d['arrays']['truth_wxyz'][1:] = [0.,1.,0.,0.]
        np.testing.assert_array_equal(module.estimator_inputs(d)['initial_wxyz'], old)
        np.testing.assert_array_equal(module.estimator_inputs(d)['gyro_rad_s'], inputs['gyro_rad_s'])

    def test_sustained_acceleration_has_exact_analytic_position_integrals(self):
        t = np.linspace(0., 10., 10001)
        step, first, second = module._smooth_step_integrals(t,2.,1.)
        np.testing.assert_allclose(np.gradient(first,t)[10:-10], step[10:-10], atol=2e-6)
        np.testing.assert_allclose(np.gradient(second,t)[10:-10], first[10:-10], atol=2e-6)
        d = module.generate_comparison_data(duration=10.,scenario='sustained_translation')
        np.testing.assert_allclose(d['arrays']['velocity_m_s'][-1], 0., atol=2e-14)
        np.testing.assert_array_equal(d['arrays']['acceleration_world_m_s2'][0], [0.,0.,0.])

    def test_reproducible_and_invalid_inputs(self):
        first = module.generate_comparison_data(duration=.1,seed=17)
        second = module.generate_comparison_data(duration=.1,seed=17)
        for key in first['arrays']:np.testing.assert_array_equal(first['arrays'][key],second['arrays'][key])
        for kwargs in ({'duration':0.},{'dt':0.},{'duration':float('nan')},{'dt':float('inf')},{'scenario':'oracle'}):
            with self.assertRaises(ValueError):module.generate_comparison_data(**kwargs)


if __name__ == '__main__':unittest.main()
