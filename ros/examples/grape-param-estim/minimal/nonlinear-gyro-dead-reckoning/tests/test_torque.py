"""Control-frame agreement and causal calibration boundaries."""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from torque import (build_torque_profile, nominal_actuator_wrench,
                    load_vehicle_model, TorqueProfile)
from grape_param_estim.dynamics import actuator_wrench
from grape_param_estim.system import ActuatorState

MODEL = HERE.parent / 'grape_vehicle_model.json'


def synthetic_inputs():
    control_time = np.linspace(0., 1.5, 61)
    gyro_time = np.linspace(.1, .5, 81)
    return dict(time=np.array([1., 1.5]),
        rotor_command_time=control_time,
        rotor_command=np.column_stack([6. + .2*np.sin(control_time + j) for j in range(4)]),
        gimbal_time=control_time,
        gimbal_angle=np.column_stack([.03*np.cos(control_time + j) for j in range(4)]),
        past_calibration_gyro_time=gyro_time,
        past_calibration_gyro=np.column_stack((.02*gyro_time, -.03*gyro_time**2,
                                                .04*np.ones(len(gyro_time)))),
        gyro_noise_covariance=np.eye(3)*1e-6,
        dense_gyro=np.zeros((10, 3)), gyro=np.zeros((20, 3)),
        mocap_quaternion_xyzw=np.zeros((10, 4)))


class TorqueTests(unittest.TestCase):
    def test_vectorized_wrench_preserves_existing_body_cog_geometry(self):
        model = load_vehicle_model(MODEL)
        thrust = np.array([[3., 4., 5., 6.], [4., 3., 7., 8.]])
        gimbal = np.array([[.1, -.2, .03, .05], [-.3, .2, -.1, 0.]])
        expected = np.array([actuator_wrench(ActuatorState(t, g), model.parameters,
                                             model.body_geometry) for t,g in zip(thrust,gimbal)])
        np.testing.assert_allclose(nominal_actuator_wrench(model, thrust, gimbal),
                                   expected, rtol=2e-15, atol=2e-15)

    def test_hidden_dense_gyro_and_mocap_do_not_change_torque(self):
        data = synthetic_inputs()
        first = build_torque_profile(data, MODEL, thrust_tau=.2)
        changed = {k:v.copy() for k,v in data.items()}
        changed['gyro'][:] = 1e10
        changed['dense_gyro'][:] = -1e9
        changed['mocap_quaternion_xyzw'][:] = np.nan
        second = build_torque_profile(changed, MODEL, thrust_tau=.2)
        for key in first['arrays']:
            np.testing.assert_array_equal(first['arrays'][key], second['arrays'][key])
        self.assertEqual(first['metadata'], second['metadata'])

    def test_past_mean_residual_is_added_to_mean_and_balances_past_average(self):
        result = build_torque_profile(synthetic_inputs(), MODEL, thrust_tau=.2)
        a = result['arrays']; weights = a['calibration_time_weights']
        np.testing.assert_allclose(weights.sum(), 1., rtol=0., atol=3e-16)
        difference = a['calibration_required_torque_body'] - a['calibration_actuator_torque_body']
        np.testing.assert_allclose(weights @ difference, result['residual_mean'], atol=1e-15)
        np.testing.assert_allclose(result['torque']([1., 1.25, 1.5]),
            result['actuator_torque']([1., 1.25, 1.5]) + result['residual_mean'], atol=1e-15)
        self.assertTrue(np.any(result['residual_mean'] != 0.))
        self.assertLess(result['metadata']['calibration_time_seconds'][1], 1.)

    def test_full_nominal_inertia_and_prior_lag_value_are_retained(self):
        result = build_torque_profile(synthetic_inputs(), MODEL)
        source = load_vehicle_model(MODEL)
        np.testing.assert_array_equal(result['inertia'], source.parameters.inertia)
        self.assertFalse(result['metadata']['inertia_diagonalized'])
        self.assertEqual(result['metadata']['thrust_tau_seconds'], .19771202959689427)
        self.assertFalse(result['metadata']['evaluation_dense_gyro_used'])

    def test_support_and_past_boundary_are_not_silently_extrapolated(self):
        profile = TorqueProfile([1., 2.], [[1., 2., 3.], [2., 3., 4.]])
        np.testing.assert_array_equal(profile(1.), [1., 2., 3.])
        self.assertEqual(profile(np.ones((2, 3))).shape, (2, 3, 3))
        with self.assertRaises(ValueError):
            profile(.9)
        data = synthetic_inputs(); data['past_calibration_gyro_time'][-1] = 1.
        with self.assertRaises(ValueError):
            build_torque_profile(data, MODEL, thrust_tau=.2)


if __name__ == '__main__':
    unittest.main()
