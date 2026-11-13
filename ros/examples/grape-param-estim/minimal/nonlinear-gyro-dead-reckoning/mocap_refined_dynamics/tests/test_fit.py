"""Derivative and physical-law checks for the nominal-start refinement."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('mocap_refined_test_fit', HERE / 'fit.py')
fit = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fit
_spec.loader.exec_module(fit)
MODEL = HERE.parents[1] / 'grape_vehicle_model.json'


def example():
    model = fit._torque.load_vehicle_model(MODEL)
    command_time = np.arange(0., 6.01, .02)
    command = np.column_stack([7. + .3 * np.sin((1. + j*.31)*command_time + j)
                              for j in range(4)])
    gimbal_time = np.arange(0., 6.01, .01)
    gimbal = np.column_stack([.1 * np.sin((1.3+j*.12)*gimbal_time + j*.6)
                             for j in range(4)])
    time = np.linspace(1., 4., 43)
    omega = np.column_stack([.2 * np.sin((.7+j*.42)*time+j*.5) for j in range(3)])
    alpha = np.column_stack([.2*(.7+j*.42)*np.cos((.7+j*.42)*time+j*.5) for j in range(3)])
    data = dict(rotor_command_time=command_time, rotor_command=command,
                gimbal_time=gimbal_time, gimbal_angle=gimbal, time=np.array([5., 6.]))
    kin = dict(time=time, omega_body_rad_s=omega, alpha_body_rad_s2=alpha)
    return data, kin, model


class RotationalFitTests(unittest.TestCase):
    def test_physical_chart_nominal_and_fixed_trace(self):
        _, _, model = example()
        chart = fit.RotationalChart(model, .05)
        initial = chart.decode(chart.initial())
        np.testing.assert_allclose(initial['inertia'], model.parameters.inertia, atol=1e-15)
        for magnitude in (-3., -.5, .5, 3.):
            coordinate = chart.initial(); coordinate[:5] = magnitude*np.arange(1, 6)/5.
            p = chart.decode(coordinate)
            self.assertAlmostEqual(np.trace(p['inertia']), np.trace(model.parameters.inertia), places=14)
            second = np.trace(p['inertia'])/2.*np.eye(3)-p['inertia']
            self.assertGreater(np.linalg.eigvalsh(second)[0], 0.)
            self.assertGreater(np.linalg.eigvalsh(p['inertia'])[0], 0.)

    def test_residual_analytic_jacobian(self):
        data, kin, model = example()
        problem = fit.RotationalProblem(data, kin, model, .04)
        coordinate = problem.chart.initial()
        coordinate[:5] = [.1, -.2, .06, -.04, .03]
        coordinate[5:8] = [.003, -.002, .001]
        coordinate[8:12] = [.03, -.01, .02, -.04]
        jacobian = problem.jacobian(coordinate).copy()
        numerical = np.empty_like(jacobian)
        step = 1e-6
        for i in range(13):
            direction = np.eye(13)[i] * step
            numerical[:, i] = (problem.residual(coordinate+direction)-problem.residual(coordinate-direction))/(2*step)
        np.testing.assert_allclose(jacobian, numerical, atol=2e-9, rtol=2e-5)

    def test_physical_parameter_jacobian(self):
        _, _, model = example()
        chart = fit.RotationalChart(model, .05)
        coordinate = chart.initial(); coordinate[:5] = [.12, -.05, .02, .03, -.04]
        _, _, jacobian = chart.decode(coordinate, jacobian=True)
        numerical = np.empty_like(jacobian)
        for i in range(13):
            direction = np.eye(13)[i] * 1e-6
            numerical[:, i] = (fit._physical_vector(chart.decode(coordinate+direction))
                - fit._physical_vector(chart.decode(coordinate-direction)))/(2e-6)
        np.testing.assert_allclose(jacobian, numerical, atol=2e-10, rtol=1e-5)

    def test_nominal_wrench_matches_existing_formula(self):
        data, kin, model = example()
        problem = fit.RotationalProblem(data, kin, model, .05)
        result = problem.evaluate(problem.chart.initial())
        time = kin['time']
        gimbal = np.column_stack([np.interp(time,data['gimbal_time'],data['gimbal_angle'][:,j]) for j in range(4)])
        expected = fit._torque.nominal_actuator_wrench(model, result['thrust'], gimbal)
        np.testing.assert_allclose(result['torque'], expected[:,3:], atol=1e-14)
        np.testing.assert_allclose(result['force'], expected[:,:3], atol=1e-14)

    def test_nonlinear_required_torque_is_retained(self):
        data, kin, model = example()
        problem = fit.RotationalProblem(data, kin, model, .05)
        result = problem.evaluate(problem.chart.initial())
        inertia = np.asarray(model.parameters.inertia)
        gyroterm = np.cross(kin['omega_body_rad_s'], kin['omega_body_rad_s']@inertia.T)
        self.assertGreater(np.max(np.abs(gyroterm)), 1e-4)
        np.testing.assert_allclose(result['required'], kin['alpha_body_rad_s2']@inertia.T+gyroterm, atol=1e-15)

    def test_fit_cannot_use_prediction_window_for_training(self):
        data, kin, _ = example()
        data['time'] = np.array([3., 6.])
        with self.assertRaisesRegex(ValueError, 'before prediction'):
            fit.fit_rotational_model(data, kin, MODEL, max_nfev=1)

    def test_synthetic_physical_parameters_recovered_from_nominal_start(self):
        data, kin, model = example()
        problem = fit.RotationalProblem(data, kin, model, .05)
        target = problem.chart.initial()
        target[:5] = [.08, -.04, .03, -.02, .01]
        target[5:8] = [.002, -.001, .003]
        target[8:12] = [.02, -.01, .01, -.02]
        generated = problem.evaluate(target)
        inertia = generated['parameters']['inertia']
        omega = kin['omega_body_rad_s']
        kin['alpha_body_rad_s2'] = (generated['torque']-np.cross(omega,omega@inertia.T))@np.linalg.inv(inertia).T
        result = fit.fit_rotational_model(data, kin, MODEL, initial_tau=.05, max_nfev=100)
        np.testing.assert_allclose(result['coordinate'], target, atol=2e-8)
        self.assertEqual(result['parameters']['jacobian_rank'], 13)
        self.assertLess(result['parameters']['optimizer']['cost'], 1e-20)

    def test_finite_optimizer_budget_is_saved_without_quality_rejection(self):
        data, kin, _ = example()
        result = fit.fit_rotational_model(data, kin, MODEL, initial_tau=.05, max_nfev=1)
        self.assertEqual(result['parameters']['optimizer']['status'], 0)
        self.assertFalse(result['parameters']['optimizer']['success'])
        profile = fit.build_refined_profile(data, result, profile_step_seconds=.01)
        self.assertTrue(np.isfinite(profile['torque']([5., 5.5, 6.])).all())


if __name__ == '__main__':
    unittest.main()
