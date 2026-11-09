"""The PID comparison consumes the frozen IMU law, not mocap plant values."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import ndtri

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import imu_pid_inputs as inputs
from single_bag_savgol_core import COMMON_SCALE_DIRECTION


class ImuPidInputTests(unittest.TestCase):
    def test_nodes_are_fixed_unscrambled_midpoints_with_no_random_realizations(self):
        z, unit = inputs.deterministic_gaussian_nodes(128)
        again, unit_again = inputs.deterministic_gaussian_nodes(128)
        np.testing.assert_array_equal(unit, unit_again)
        np.testing.assert_array_equal(z, again)
        np.testing.assert_array_equal(unit[0], np.full(16,.5/128))
        np.testing.assert_array_equal(z,ndtri(unit))
        self.assertTrue(np.all((unit > 0.) & (unit < 1.)))
        self.assertEqual(z.shape,(128,16))
        self.assertFalse(z.flags.writeable)

    def test_nominal_node_is_the_saved_imu_physical_point_and_three_lags(self):
        for case in ('failure1','failure2'):
            b = inputs.load_imu_pid_inputs(case,node_count=1)
            point = b['estimate']['point_estimate']
            p = b['nominal_parameters']
            np.testing.assert_allclose(p.mass,point['mass_kg'],rtol=1e-13)
            np.testing.assert_allclose(p.inertia,point['inertia_kg_m2'],rtol=1e-13,atol=1e-15)
            np.testing.assert_allclose(p.cog_offset,point['cog_position_body_m'],rtol=1e-13,atol=1e-15)
            np.testing.assert_allclose(p.force_effectiveness,point['force_effectiveness'],rtol=1e-13)
            np.testing.assert_array_equal(b['coordinates'][0],b['mean'])
            np.testing.assert_array_equal(b['taus'][0],b['nominal_taus'])
            self.assertEqual(b['actuators'][0].thrust_time_constant,b['nominal_taus'][0])
            np.testing.assert_allclose(b['plants'][0].inertia_over_mass,p.inertia/p.mass)
            np.testing.assert_allclose(b['plants'][0].force_effectiveness_over_mass,p.force_effectiveness/p.mass)
            self.assertTrue(b['valid_node_mask'][0])

    def test_full_joint_covariance_and_cross_terms_are_not_conditioned_or_diagonalized(self):
        for case in ('failure1','failure2'):
            b = inputs.load_imu_pid_inputs(case)
            original = np.asarray(b['estimate']['parameter_distribution']['covariance_reduced_log_tau'])
            np.testing.assert_array_equal(b['covariance'],original)
            np.testing.assert_array_equal(b['physical_thrust_covariance'],original[:14,:14])
            self.assertTrue(np.any(original[:13,13] != 0.))
            self.assertTrue(np.any(original[:14,14:] != 0.))
            expected = b['mean'][None,:]+np.einsum('nk,ik->ni', b['standard_nodes'],b['factor'],optimize=False)
            np.testing.assert_array_equal(b['coordinates'],expected)
            np.testing.assert_array_equal(b['taus'],np.exp(b['coordinates'][:,13:16]))
            np.testing.assert_allclose(b['factor']@b['factor'].T,original,rtol=2e-14,atol=1e-12)
            self.assertFalse(b['covariance'].flags.writeable)
            self.assertEqual(b['metadata']['full_joint_covariance_preserved'],True)

    def test_unrepresentable_nodes_keep_their_original_order_weight_and_denominator(self):
        for case in ('failure1','failure2'):
            b = inputs.load_imu_pid_inputs(case)
            self.assertEqual(len(b['plants']),128)
            self.assertEqual(len(b['node_errors']),128)
            self.assertEqual(len(b['actuators']),128)
            self.assertTrue(np.any(~b['valid_node_mask']))
            np.testing.assert_array_equal(b['weights'],np.full(128,1./128))
            self.assertEqual(float(b['weights'].sum()),1.)
            self.assertTrue(np.isfinite(b['coordinates']).all())
            for index, valid in enumerate(b['valid_node_mask']):
                if valid:
                    self.assertIsNotNone(b['plants'][index])
                    self.assertIsNone(b['node_errors'][index])
                else:
                    self.assertIsNotNone(b['node_errors'][index])
            repeated = inputs.load_imu_pid_inputs(case)
            np.testing.assert_array_equal(b['coordinates'],repeated['coordinates'])
            np.testing.assert_array_equal(b['valid_node_mask'],repeated['valid_node_mask'])
            self.assertFalse(b['metadata']['monte_carlo'])
            self.assertFalse(b['metadata']['success_input_used'])

    def test_legacy_context_supplies_no_plant_covariance_or_fitted_thrust_tau(self):
        baseline = inputs.load_imu_pid_inputs('failure1',node_count=1)
        with tempfile.TemporaryDirectory(prefix='imu-pid-context-') as directory:
            root = Path(directory)
            folder = root/'failure1'; folder.mkdir()
            context = json.loads((inputs.CONTROLLER_CONTEXT_SOURCE/'failure1'/'estimate.json').read_text())
            context['point_estimate'] = {'unused':'not a physical model'}
            context['plant_distribution'] = {'unused':'not a covariance'}
            context['actuator_model']['thrust_time_constant_seconds'] = 99999.
            (folder/'estimate.json').write_text(json.dumps(context))
            with patch.object(inputs,'CONTROLLER_CONTEXT_SOURCE',root):
                changed = inputs.load_imu_pid_inputs('failure1',node_count=1)
            np.testing.assert_array_equal(changed['coordinates'],baseline['coordinates'])
            np.testing.assert_array_equal(changed['covariance'],baseline['covariance'])
            np.testing.assert_array_equal(changed['reference_gains'],baseline['reference_gains'])
            self.assertEqual(changed['nominal_actuator'],baseline['nominal_actuator'])
            self.assertEqual(changed['metadata']['controller_context_selected_sha256'],
                             baseline['metadata']['controller_context_selected_sha256'])

    def test_different_bag_controller_context_is_rejected_without_reusing_its_values(self):
        with tempfile.TemporaryDirectory(prefix='imu-pid-context-mismatch-') as directory:
            root = Path(directory)
            folder = root/'failure1'; folder.mkdir()
            context = json.loads((inputs.CONTROLLER_CONTEXT_SOURCE/'failure1'/'estimate.json').read_text())
            context['provenance']['bag_sha256'] = 'another-recording'
            (folder/'estimate.json').write_text(json.dumps(context))
            with patch.object(inputs,'CONTROLLER_CONTEXT_SOURCE',root):
                with self.assertRaisesRegex(ValueError,'same failure bag'):
                    inputs.load_imu_pid_inputs('failure1',node_count=1)

    def test_native_command_period_is_used_and_success_is_not_a_failure_input(self):
        for case in ('failure1','failure2'):
            b = inputs.load_imu_pid_inputs(case,node_count=1)
            with np.load(inputs.SOURCE/'inputs'/(case+'.npz')) as archived:
                self.assertEqual(b['controller_dt'],float(np.asarray(archived['controller_dt']).reshape(-1)[0]))
        with self.assertRaisesRegex(ValueError,'failure-only'):
            inputs.load_imu_pid_inputs('success',node_count=1)

    def test_scale_free_decode_cancels_an_unrepresentable_common_si_scale_exactly(self):
        bundle = inputs.load_imu_pid_inputs('failure1',node_count=1)
        chart = inputs.SiParameterChart(bundle['vehicle_model'].parameters)
        expected = inputs.scale_free_chart_node(chart,np.zeros(14))
        for multiplier in (-1000.,1000.):
            coordinate = multiplier*COMMON_SCALE_DIRECTION
            with self.assertRaises((ValueError,OverflowError)):
                chart.decode(coordinate)
            actual = inputs.scale_free_chart_node(chart,coordinate)
            np.testing.assert_allclose(actual.inertia_over_mass,expected.inertia_over_mass,rtol=1e-14,atol=1e-17)
            np.testing.assert_array_equal(actual.force_effectiveness_over_mass,expected.force_effectiveness_over_mass)
            np.testing.assert_array_equal(actual.cog_position_body,expected.cog_position_body)
        # A moderate nontrivial coordinate also matches the original chart.
        coordinate = np.linspace(-.12,.12,14)
        actual = inputs.scale_free_chart_node(chart,coordinate)
        raw = chart.decode(coordinate)
        np.testing.assert_allclose(actual.inertia_over_mass,raw.inertia/raw.mass,rtol=1e-13,atol=1e-16)
        np.testing.assert_allclose(actual.force_effectiveness_over_mass,
                                   raw.force_effectiveness/raw.mass,rtol=1e-13)
        self.assertTrue(any(raw is None and plant is not None for raw,plant in zip(
            inputs.load_imu_pid_inputs('failure1')['physical_parameters'],
            inputs.load_imu_pid_inputs('failure1')['plants'])))


if __name__ == '__main__':
    unittest.main()
