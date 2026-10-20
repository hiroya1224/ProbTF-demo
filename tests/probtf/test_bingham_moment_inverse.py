"""Check the example inverse against the unchanged public Bingham moments."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ros/core/probtf_core/src'))
SCRIPT = ROOT / 'ros/examples/probtf_orientation_demo/scripts/bingham_moment_inverse.py'
SPEC = importlib.util.spec_from_file_location('example_bingham_moment_inverse', SCRIPT)
inverse = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inverse)


class BinghamMomentInverseTest(unittest.TestCase):
    def test_public_moment_roundtrip_without_concentration_bounds(self):
        for values in ((-30., -15., -5., 0.), (-5.13e6, -5.13e6, -100., 0.)):
            target = inverse.bingham_second_moment(np.diag(values))
            before = target.copy()
            fitted, diagnostic = inverse.fit_bingham_second_moment(
                target, initial_eigenvalues=np.asarray(values) * 1.1)
            np.testing.assert_allclose(inverse.bingham_second_moment(fitted), target,
                                       rtol=0.0, atol=2e-10)
            np.testing.assert_array_equal(target, before)
            self.assertIsNone(diagnostic['concentration_bounds'])
            self.assertFalse(diagnostic['target_moment_modified'])
            if values[0] < -1e6:
                self.assertLess(np.linalg.eigvalsh(fitted)[1], -1e6)

    def test_no_mode_gap_is_invented_for_uniform_or_repeated_eigenvalues(self):
        for values in ((0., 0., 0., 0.), (-40., -40., 0., 0.)):
            target = inverse.bingham_second_moment(np.diag(values))
            fitted, diagnostic = inverse.fit_bingham_second_moment(target)
            np.testing.assert_allclose(inverse.bingham_second_moment(fitted), target,
                                       rtol=0.0, atol=1e-8)
            if values[0] == 0.:
                np.testing.assert_allclose(fitted, np.zeros((4, 4)), atol=2e-14)

    def test_parameter_is_returned_in_the_full_target_eigenbasis(self):
        v = np.array([1., 2., -1., 0.5]);v /= np.linalg.norm(v)
        rotation = np.eye(4) - 2.0 * np.outer(v, v)
        parameter = rotation @ np.diag([-8.e6, -5.e6, -80., 0.]) @ rotation.T
        parameter = 0.5 * (parameter + parameter.T)
        target = inverse.bingham_second_moment(parameter)
        fitted, diagnostic = inverse.fit_bingham_second_moment(
            target, initial_eigenvalues=np.linalg.eigvalsh(parameter))
        np.testing.assert_allclose(inverse.bingham_second_moment(fitted), target,
                                   rtol=0.0, atol=2e-10)
        self.assertGreater(np.linalg.norm(fitted - np.diag(np.diag(fitted))), 1e5)

    def test_dual_score_is_the_analytic_derivative(self):
        values = np.array([-20., -10., -3., 0.])
        target = np.diag(inverse.bingham_second_moment(np.diag(values)))
        scale = 0.5 / target[:3]
        point = np.array([-0.8, -1.1, -0.6])
        _, derivative = inverse._dual_value_gradient(point, target, scale, 120)
        actual = []
        for i in range(3):
            change = np.eye(3)[i] * 1e-5
            actual.append((inverse._dual_value_gradient(point + change, target, scale, 120)[0]
                           - inverse._dual_value_gradient(point - change, target, scale, 120)[0]) / 2e-5)
        np.testing.assert_allclose(derivative, actual, rtol=2e-5, atol=2e-8)

    def test_finite_unconverged_result_is_returned_with_its_actual_residual(self):
        target = inverse.bingham_second_moment(np.diag([-1e6, -2e5, -20., 0.]))
        before = target.copy()
        # An unusable Jacobian must leave an honest finite result rather than
        # rejecting the next filter step or claiming moment convergence.
        uniform, _ = inverse._moment_eigenvalues_and_jacobian(np.zeros(4), 120)
        with patch.object(inverse, '_moment_eigenvalues_and_jacobian',
                          return_value=(uniform, np.zeros((4, 4)))):
            fitted, diagnostic = inverse.fit_bingham_second_moment(
                target, initial_eigenvalues=np.zeros(4), max_iterations=1)
        self.assertIsNone(diagnostic['concentration_bounds'])
        self.assertFalse(diagnostic['optimizer_success'])
        self.assertTrue(diagnostic['finite_iteration_result_used'])
        self.assertGreater(diagnostic['moment_residual_frobenius'], 0.1)
        np.testing.assert_array_equal(target, before)
        np.testing.assert_allclose(fitted, np.zeros((4, 4)), atol=0.)

    def test_measured_micro_noise_is_not_erased_by_inverse_convergence(self):
        parameter = np.diag([0., -100., -100., -100.])
        covariance = np.array([[6.47814413e-7, -5.35653150e-9, -1.23244457e-8],
                               [-5.35653150e-9, 3.51899478e-7, 4.52832111e-8],
                               [-1.23244457e-8, 4.52832111e-8, 3.48278089e-7]])
        current = inverse.bingham_second_moment(parameter)
        target = inverse.quaternion_product_second_moment(current,
            inverse.delta_quaternion_second_moment([0., 0., 0.], .005, covariance))
        actual, _ = inverse.propagate(parameter, np.zeros(3), .005, covariance)
        delta_target = target-current
        delta_fitted = inverse.bingham_second_moment(actual)-current
        self.assertGreater(np.linalg.norm(delta_target), 1e-12)
        np.testing.assert_allclose(delta_fitted, delta_target, rtol=0., atol=1e-15)

    def test_repeated_small_noise_matches_direct_moment_accumulation(self):
        parameter = np.diag([0., -100., -100., -100.])
        covariance = np.array([[6.5e-7, -5e-9, -1.2e-8],
                               [-5e-9, 3.5e-7, 4.5e-8],
                               [-1.2e-8, 4.5e-8, 3.5e-7]])
        initial = inverse.bingham_second_moment(parameter)
        direct = initial.copy()
        increment = inverse.delta_quaternion_second_moment(np.zeros(3), .005, covariance)
        for _ in range(100):
            direct = inverse.quaternion_product_second_moment(direct, increment)
            parameter, _ = inverse.propagate(parameter, np.zeros(3), .005, covariance)
        self.assertGreater(np.linalg.norm(direct-initial), 5e-10)
        np.testing.assert_allclose(inverse.bingham_second_moment(parameter), direct,
                                   rtol=0., atol=5e-14)

    def test_propagation_uses_the_original_core_product_and_preceding_eigenvalues(self):
        parameter = np.diag([-5.13e6, -5.13e6, -100., 0.])
        gyro = np.array([-.0011, .0008, -.00018]);covariance = np.eye(3) * 1e-4;dt = .005
        target = inverse.quaternion_product_second_moment(
            inverse.bingham_second_moment(parameter),
            inverse.delta_quaternion_second_moment(gyro, dt, covariance))
        with patch.object(inverse, 'fit_bingham_second_moment',
                          wraps=inverse.fit_bingham_second_moment) as fit:
            predicted, residual = inverse.propagate(parameter, gyro, dt, covariance)
        np.testing.assert_array_equal(fit.call_args.args[0], target)
        np.testing.assert_array_equal(fit.call_args.kwargs['initial_eigenvalues'],
                                      np.linalg.eigvalsh(parameter))
        actual = inverse.bingham_second_moment(predicted)
        np.testing.assert_allclose(actual, target, rtol=0.0, atol=2e-10)
        small = np.linalg.eigvalsh(actual)[:2] / np.linalg.eigvalsh(target)[:2]
        np.testing.assert_allclose(small, np.ones(2), rtol=2e-5, atol=0.)
        self.assertAlmostEqual(residual, np.linalg.norm(actual - target), places=16)

    def test_zero_interval_keeps_the_same_law_without_inverse(self):
        parameter = np.diag([-5.e6, -5.e6, -100., 0.])
        with patch.object(inverse, 'fit_bingham_second_moment') as fit:
            actual, residual = inverse.propagate(parameter, [1., 2., 3.], 0., np.eye(3))
        fit.assert_not_called()
        np.testing.assert_array_equal(actual, parameter)
        self.assertEqual(residual, 0.)

    def test_zero_target_eigenvalues_are_not_replaced_by_positive_floors(self):
        target = np.diag([0., 0., 0., 1.])
        fitted, diagnostic = inverse.fit_bingham_second_moment(target, max_iterations=2)
        np.testing.assert_array_equal(diagnostic['target_moment_eigenvalues'], [0., 0., 0., 1.])
        self.assertEqual(diagnostic['unit_coordinate_scale_indices'], [0, 1, 2])
        self.assertTrue(np.isfinite(fitted).all())

    def test_nonfinite_target_cannot_enter_the_numerical_inverse(self):
        for value in (np.nan, np.inf):
            target = np.eye(4) / 4.;target[0, 0] = value
            with self.assertRaises(FloatingPointError):
                inverse.fit_bingham_second_moment(target)


if __name__ == '__main__':
    unittest.main()
