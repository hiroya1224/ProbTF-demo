import unittest

import numpy as np

from grape_param_estim.model_error import (
    GaussMarkovWrenchProcess,
    KnotGaussMarkovWrenchProcess,
)
from grape_param_estim.strong_constraint import (
    CONTROL_DIMENSION,
    IEnKSConfig,
    StrongConstraintPrior,
)
from grape_param_estim.strong_constraint_experiments import (
    _problem_from_synthetic,
)
from grape_param_estim.synthetic import run_synthetic_experiment
from grape_param_estim.weak_constraint import (
    WeakConstraintIEnKSQ,
    WeakConstraintPrior,
    WeakConstraintProblem,
)


class KnotGaussMarkovWrenchProcessTests(unittest.TestCase):
    def setUp(self):
        self.integration_times = np.asarray((0.0, 0.1, 0.4, 0.55, 1.0))
        self.knot_indices = np.asarray((0, 2, 4))
        self.sigma = np.asarray((0.8, 0.5, 0.3, 0.12, 0.09, 0.07))
        self.tau = 0.3
        self.process = KnotGaussMarkovWrenchProcess(
            self.integration_times,
            self.knot_indices,
            self.sigma,
            self.tau,
        )

    def test_sparse_dimension_times_and_factory_are_dense_compatible(self):
        self.assertEqual(self.process.innovation_dimension, 6 * 3)
        np.testing.assert_array_equal(
            self.process.times, self.integration_times[:-1]
        )
        np.testing.assert_array_equal(
            self.process.knot_times,
            self.integration_times[self.knot_indices],
        )
        decoded = self.process.decode(
            np.zeros(self.process.innovation_dimension)
        )
        self.assertEqual(decoded.shape, (self.integration_times.size - 1, 6))
        np.testing.assert_array_equal(decoded, 0.0)

        factory = GaussMarkovWrenchProcess.from_knots(
            self.integration_times,
            self.knot_indices,
            self.sigma,
            self.tau,
        )
        self.assertIsInstance(factory, KnotGaussMarkovWrenchProcess)
        self.assertEqual(
            factory.compatibility_signature,
            self.process.compatibility_signature,
        )

    def test_ou_is_decoded_only_at_knots(self):
        innovations = np.arange(
            1.0, self.process.innovation_dimension + 1.0
        ).reshape(3, 6) / 17.0
        actual = self.process.decode_knots(innovations.reshape(-1))
        expected = np.empty_like(actual)
        expected[0] = self.sigma * innovations[0]
        knot_times = self.integration_times[self.knot_indices]
        for index in range(1, knot_times.size):
            rho = np.exp(
                -(knot_times[index] - knot_times[index - 1]) / self.tau
            )
            expected[index] = (
                rho * expected[index - 1]
                + self.sigma
                * np.sqrt(1.0 - rho**2)
                * innovations[index]
            )
        np.testing.assert_allclose(actual, expected, atol=2.0e-16)

    def test_piecewise_linear_matrix_returns_exact_interval_averages(self):
        expected_matrix = np.asarray(
            (
                (0.875, 0.125, 0.0),
                (0.375, 0.625, 0.0),
                (0.0, 0.875, 0.125),
                (0.0, 0.375, 0.625),
            )
        )
        np.testing.assert_allclose(
            self.process.interpolation_matrix,
            expected_matrix,
            atol=2.0e-16,
        )
        np.testing.assert_array_equal(
            self.process.interval_average_interpolation_matrix,
            self.process.interpolation_matrix,
        )
        np.testing.assert_allclose(
            np.sum(self.process.interpolation_matrix, axis=1),
            1.0,
            atol=2.0e-16,
        )

        innovations = np.linspace(
            -0.7, 0.9, self.process.innovation_dimension
        )
        knots = self.process.decode_knots(innovations)
        np.testing.assert_allclose(
            self.process.decode(innovations),
            expected_matrix @ knots,
            atol=2.0e-16,
        )

    def test_sampling_is_reproducible_and_recentered_at_each_knot(self):
        first = self.process.sample_innovations(64, seed=31)
        repeated = self.process.sample_innovations(64, seed=31)
        different = self.process.sample_innovations(64, seed=32)
        self.assertEqual(first.shape, (64, 18))
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, different))
        np.testing.assert_allclose(
            np.mean(first, axis=0), 0.0, atol=2.0e-16
        )

    def test_signature_captures_dense_sparse_and_knot_layout(self):
        repeated = KnotGaussMarkovWrenchProcess(
            self.integration_times.copy(),
            self.knot_indices.copy(),
            self.sigma.copy(),
            self.tau,
        )
        changed_knots = KnotGaussMarkovWrenchProcess(
            self.integration_times,
            np.asarray((0, 1, 4)),
            self.sigma,
            self.tau,
        )
        dense = GaussMarkovWrenchProcess(
            self.integration_times[:-1], self.sigma, self.tau
        )
        self.assertEqual(
            repeated.compatibility_signature,
            self.process.compatibility_signature,
        )
        self.assertNotEqual(
            changed_knots.compatibility_signature,
            self.process.compatibility_signature,
        )
        self.assertNotEqual(
            dense.compatibility_signature,
            self.process.compatibility_signature,
        )

    def test_invalid_sparse_grid_knots_and_decode_are_rejected(self):
        for integration_times in (
            np.asarray((0.0,)),
            np.asarray((0.0, 0.0, 0.1)),
            np.asarray((0.0, np.nan, 0.1)),
            np.zeros((2, 2)),
        ):
            with self.assertRaisesRegex(ValueError, "integration_times"):
                KnotGaussMarkovWrenchProcess(
                    integration_times,
                    np.asarray((0, 1)),
                    self.sigma,
                    self.tau,
                )
        for knot_indices in (
            np.asarray((0,)),
            np.asarray((0.0, 4.0)),
            np.asarray((False, True)),
            np.asarray((1, 4)),
            np.asarray((0, 2, 2, 4)),
            np.asarray((0, 2, 3)),
            np.zeros((2, 2), dtype=int),
        ):
            with self.assertRaisesRegex(ValueError, "knot_indices"):
                KnotGaussMarkovWrenchProcess(
                    self.integration_times,
                    knot_indices,
                    self.sigma,
                    self.tau,
                )
        for innovations in (
            np.zeros(self.process.innovation_dimension - 1),
            np.zeros(self.process.innovation_dimension + 1),
            np.zeros((3, 6)),
            np.full(self.process.innovation_dimension, np.nan),
        ):
            with self.assertRaisesRegex(ValueError, "finite values"):
                self.process.decode(innovations)

    def test_sparse_process_runs_through_weak_problem_and_signature_guard(self):
        synthetic = run_synthetic_experiment(
            duration=0.16,
            time_step=0.04,
            truth_residual_wrench=lambda _time, _state: np.zeros(6),
            translation_noise=0.001,
            rotation_noise=0.001,
            seed=4,
        )
        strong_problem = _problem_from_synthetic(synthetic)
        integration_times = synthetic.observations.times
        process = KnotGaussMarkovWrenchProcess(
            integration_times,
            np.asarray((0, 2, 4)),
            self.sigma,
            self.tau,
        )
        weak_problem = WeakConstraintProblem(strong_problem, process)
        strong = strong_problem.forecast(np.zeros(CONTROL_DIMENSION))
        weak = weak_problem.forecast(
            np.zeros(weak_problem.control_dimension)
        )
        np.testing.assert_array_equal(weak.position, strong.position)
        np.testing.assert_array_equal(
            weak.orientation_xyzw, strong.orientation_xyzw
        )

        incompatible = KnotGaussMarkovWrenchProcess(
            integration_times,
            np.asarray((0, 1, 4)),
            self.sigma,
            self.tau,
        )
        configuration = IEnKSConfig(
            ensemble_size=weak_problem.control_dimension + 1,
            maximum_iterations=1,
            seed=4,
        )
        with self.assertRaisesRegex(ValueError, "must agree"):
            WeakConstraintIEnKSQ(configuration).fit(
                weak_problem,
                WeakConstraintPrior(
                    StrongConstraintPrior.grape(), incompatible
                ),
            )

        wrong_final_time = integration_times.copy()
        wrong_final_time[-1] += 0.01
        wrong_grid = KnotGaussMarkovWrenchProcess(
            wrong_final_time,
            np.asarray((0, 2, 4)),
            self.sigma,
            self.tau,
        )
        with self.assertRaisesRegex(ValueError, "grid"):
            WeakConstraintProblem(strong_problem, wrong_grid)


if __name__ == "__main__":
    unittest.main()
