from dataclasses import replace
import unittest

import numpy as np

from grape_param_estim.batch.em_loop import EStepPhase
from grape_param_estim.batch.lag_profile import LagProfileSettings
from grape_param_estim.batch.lm import LMSettings
from grape_param_estim.estimation import (
    EstimationCancelled,
    SparseLaplaceEStepSolver,
    make_fixed_q_laplace_problem_factory,
    restore_fixed_graph_laplace,
    solve_fixed_graph_laplace,
)
from grape_param_estim.posterior.delayed_acceptance import PosteriorPoint
from tests.grape_param_estim.test_batch_graph_builder import (
    BatchGraphBuilderTests,
)


class EstimationOrchestrationTests(unittest.TestCase):
    def setUp(self):
        helper = BatchGraphBuilderTests()
        helper.setUp()
        self.prepared = helper._prepared()
        self.calls = []

        def factory(q, delay, static_coordinate):
            self.calls.append(
                (
                    np.asarray(q).copy(),
                    float(delay),
                    np.asarray(static_coordinate).copy(),
                )
            )
            return replace(
                self.prepared,
                dynamics=replace(self.prepared.dynamics, q=q),
                fixed_delay=delay,
                initial_parameter_coordinates=static_coordinate,
            )

        self.factory = factory

    def test_fixed_graph_solve_uses_undamped_laplace_and_corrected_moments(self):
        result = solve_fixed_graph_laplace(
            self.factory,
            self.prepared.dynamics.q,
            self.prepared.fixed_delay,
            self.prepared.initial_parameter_coordinates,
        )
        self.assertTrue(result.lm.converged)
        self.assertEqual(result.dynamics.moments.interval_count, 1)
        self.assertGreater(
            float(np.max(result.dynamics.moments.covariance_correction)),
            0.0,
        )
        self.assertTrue(np.isfinite(result.marginal_objective.value))
        geometry = result.static_geometry()
        np.testing.assert_allclose(
            geometry.information.posterior.hessian,
            result.factorization.reduced_hessian,
        )

    def test_completed_map_checkpoint_restores_without_nonlinear_iterations(self):
        solved = solve_fixed_graph_laplace(
            self.factory,
            self.prepared.dynamics.q,
            self.prepared.fixed_delay,
            self.prepared.initial_parameter_coordinates,
        )
        restored = restore_fixed_graph_laplace(
            self.factory,
            solved.prepared.dynamics.q,
            solved.prepared.fixed_delay,
            {
                key: solved.lm.state.value(key)
                for key in solved.lm.state.layout.variable_keys
            },
            solved.lm.objective,
        )
        self.assertEqual(restored.lm.iterations, ())
        np.testing.assert_allclose(
            restored.factorization.reduced_hessian,
            solved.factorization.reduced_hessian,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        with self.assertRaisesRegex(ValueError, "objective"):
            restore_fixed_graph_laplace(
                self.factory,
                solved.prepared.dynamics.q,
                solved.prepared.fixed_delay,
                {
                    key: solved.lm.state.value(key)
                    for key in solved.lm.state.layout.variable_keys
                },
                solved.lm.objective + 1.0,
            )

    def test_em_adapter_profiles_delay_without_a_delay_derivative(self):
        solver = SparseLaplaceEStepSolver(
            self.factory,
            self.prepared.initial_parameter_coordinates,
            LMSettings(),
            LagProfileSettings(
                minimum_lag=0.0,
                maximum_lag=0.02,
                coarse_grid_points=3,
                refinement_tolerance=0.009,
                maximum_refinement_evaluations=2,
            ),
        )
        result = solver(
            self.prepared.dynamics.q,
            EStepPhase.WIDE_LAG_PROFILE,
            self.prepared.fixed_delay,
            None,
        )
        self.assertGreaterEqual(len(self.calls), 3)
        self.assertEqual(len(solver.profile_history), 1)
        self.assertEqual(len(solver.profile_q_history), 1)
        np.testing.assert_array_equal(
            solver.profile_q_history[0], self.prepared.dynamics.q
        )
        self.assertFalse(solver.profile_q_history[0].flags.writeable)
        self.assertEqual(result.termination_reason, "wide_lag_profile")
        self.assertTrue(
            0.0 <= result.lag <= 0.02,
        )
        solution = solver.take_solution_for_result(result)
        self.assertIs(solution.lm.state, result.state)
        self.assertEqual(
            solution.marginal_objective.value,
            result.approximate_marginal_objective,
        )
        with self.assertRaises(ValueError):
            solver.take_solution_for_result(result)

    def test_mcmc_factory_preserves_exact_proposed_static_and_delay(self):
        factory = make_fixed_q_laplace_problem_factory(
            self.factory, self.prepared.dynamics.q
        )
        coordinate = self.prepared.initial_parameter_coordinates.copy()
        coordinate[7] += 0.002
        point = PosteriorPoint(coordinate, 0.004)
        fixed = factory(point)
        self.assertEqual(fixed.fixed_delay, point.delay)
        np.testing.assert_array_equal(
            fixed.initial_state.value(
                fixed.initial_state.layout.variable_keys[0]
            ),
            coordinate,
        )
        self.assertTrue(fixed.graph_objective_includes_static_prior)

    def test_fixed_graph_cancellation_propagates_without_becoming_failure(self):
        records = []
        with self.assertRaises(EstimationCancelled):
            solve_fixed_graph_laplace(
                self.factory,
                self.prepared.dynamics.q,
                self.prepared.fixed_delay,
                self.prepared.initial_parameter_coordinates,
                cancellation_requested=lambda: bool(records),
                lm_progress=records.append,
            )
        self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
