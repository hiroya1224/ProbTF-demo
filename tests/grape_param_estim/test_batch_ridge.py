import unittest

import numpy as np

from grape_param_estim.batch.ridge import (
    STATIC_PARAMETER_NAMES,
    analyze_reduced_hessian,
    analyze_reduced_information,
)


class ReducedHessianRidgeTests(unittest.TestCase):
    def test_exact_and_near_ridges_have_named_deterministic_loadings(self):
        generator = np.random.RandomState(8821)
        orthogonal, _ = np.linalg.qr(generator.normal(size=(18, 18)))
        eigenvalues = np.geomspace(2.0e-6, 20.0, 18)
        eigenvalues[0] = 0.0
        eigenvalues[1] = 8.0e-11
        hessian = orthogonal @ np.diag(eigenvalues) @ orthogonal.T

        first = analyze_reduced_hessian(
            hessian,
            relative_rank_tolerance=1.0e-10,
        )
        second = analyze_reduced_hessian(
            hessian,
            relative_rank_tolerance=1.0e-10,
        )

        self.assertEqual(first.effective_rank, 16)
        self.assertTrue(np.isinf(first.condition_number))
        self.assertTrue(np.isfinite(first.identified_condition_number))
        self.assertEqual(len(first.ridge_directions), 2)
        np.testing.assert_array_equal(first.eigenvectors, second.eigenvectors)
        np.testing.assert_allclose(
            first.hessian @ first.eigenvectors,
            first.eigenvectors * first.eigenvalues,
            rtol=2.0e-8,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            first.eigenvectors.T @ first.eigenvectors,
            np.eye(18),
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        for direction in first.ridge_directions:
            self.assertEqual(len(direction.loadings), 18)
            self.assertEqual(
                {loading.parameter_name for loading in direction.loadings},
                set(STATIC_PARAMETER_NAMES),
            )
            absolute_values = [
                abs(loading.coefficient) for loading in direction.loadings
            ]
            self.assertEqual(absolute_values, sorted(absolute_values, reverse=True))
            anchor = int(np.argmax(np.abs(direction.vector)))
            self.assertGreaterEqual(direction.vector[anchor], 0.0)

    def test_roundoff_asymmetry_and_negative_eigenvalue_are_distinguished(self):
        hessian = np.diag(np.linspace(0.0, 3.0, 18))
        hessian[2, 7] += 2.0e-13
        analysis = analyze_reduced_hessian(hessian)
        self.assertGreater(analysis.maximum_input_asymmetry, 0.0)
        np.testing.assert_array_equal(analysis.hessian, analysis.hessian.T)
        self.assertEqual(analysis.eigenvalues[0], 0.0)

        materially_asymmetric = np.eye(18)
        materially_asymmetric[0, 1] = 1.0e-3
        with self.assertRaisesRegex(ValueError, "symmetric"):
            analyze_reduced_hessian(materially_asymmetric)
        negative = np.eye(18)
        negative[0, 0] = -1.0e-4
        with self.assertRaisesRegex(ValueError, "negative"):
            analyze_reduced_hessian(negative)

    def test_degenerate_eigenspace_basis_is_canonical(self):
        generator = np.random.RandomState(72)
        orthogonal, _ = np.linalg.qr(generator.normal(size=(18, 18)))
        eigenvalues = np.asarray((0.0, 0.0) + tuple(range(1, 17)), dtype=float)
        hessian = orthogonal @ np.diag(eigenvalues) @ orthogonal.T

        analysis = analyze_reduced_hessian(hessian)
        first_projector = orthogonal[:, :2] @ orthogonal[:, :2].T
        np.testing.assert_allclose(
            analysis.eigenvectors[:, :2]
            @ analysis.eigenvectors[:, :2].T,
            first_projector,
            rtol=3.0e-12,
            atol=3.0e-12,
        )
        first_visible_coordinate = int(
            np.flatnonzero(np.diag(first_projector) > 1.0e-12)[0]
        )
        expected = first_projector[:, first_visible_coordinate]
        expected /= np.linalg.norm(expected)
        anchor = int(np.argmax(np.abs(expected)))
        if expected[anchor] < 0.0:
            expected = -expected
        np.testing.assert_allclose(
            analysis.eigenvectors[:, 0], expected, rtol=3.0e-12, atol=3.0e-12
        )

    def test_likelihood_and_posterior_are_kept_separate(self):
        likelihood = np.diag(np.asarray((0.0,) + (2.0,) * 17))
        prior = np.diag(np.linspace(0.2, 1.9, 18))
        analysis = analyze_reduced_information(likelihood, likelihood + prior)

        self.assertEqual(analysis.likelihood.effective_rank, 17)
        self.assertEqual(analysis.posterior.effective_rank, 18)
        self.assertTrue(np.isinf(analysis.likelihood.condition_number))
        self.assertTrue(np.isfinite(analysis.posterior.condition_number))
        np.testing.assert_allclose(analysis.prior_contribution, prior)

    def test_outputs_are_independent_and_read_only(self):
        source = np.diag(np.linspace(0.0, 2.0, 18))
        analysis = analyze_reduced_hessian(source)
        source[:] = 4.0
        self.assertFalse(np.all(analysis.hessian == 4.0))
        for array in (
            analysis.hessian,
            analysis.eigenvalues,
            analysis.relative_eigenvalues,
            analysis.eigenvectors,
            analysis.ridge_directions[0].vector,
        ):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.flat[0] = 0.0

    def test_invalid_shape_names_tolerance_and_zero_information(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            analyze_reduced_hessian(np.eye(17))
        with self.assertRaisesRegex(ValueError, "finite"):
            invalid = np.eye(18)
            invalid[0, 0] = np.nan
            analyze_reduced_hessian(invalid)
        with self.assertRaisesRegex(ValueError, "18 entries"):
            analyze_reduced_hessian(np.eye(18), parameter_names=("a",))
        with self.assertRaisesRegex(ValueError, "unique"):
            analyze_reduced_hessian(np.eye(18), parameter_names=("a",) * 18)
        for value in (-1.0, 1.0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                analyze_reduced_hessian(
                    np.eye(18), relative_rank_tolerance=value
                )
        with self.assertRaises(TypeError):
            analyze_reduced_hessian(
                np.eye(18), relative_rank_tolerance=True
            )

        zero = analyze_reduced_hessian(np.zeros((18, 18)))
        self.assertEqual(zero.effective_rank, 0)
        self.assertTrue(np.isinf(zero.condition_number))
        self.assertTrue(np.isinf(zero.identified_condition_number))
        self.assertEqual(len(zero.ridge_directions), 18)


if __name__ == "__main__":
    unittest.main()
