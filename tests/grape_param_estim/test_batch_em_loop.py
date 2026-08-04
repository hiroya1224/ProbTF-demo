import unittest

import numpy as np

from grape_param_estim.batch.em_loop import (
    EStepPhase,
    LaplaceEStepFailure,
    LaplaceEStepResult,
    LaplaceEmCancelled,
    LaplaceEmSettings,
    LaplaceEmTerminationReason,
    run_laplace_em,
)
from grape_param_estim.batch.laplace_em import (
    DiagonalQDefinition,
    ExpectedResidualMoments,
    QIntervalModel,
)
from grape_param_estim.batch.layout import VariableLayout
from grape_param_estim.batch.state import BatchState
from grape_param_estim.batch.variables import VariableKey, VariableKind


KNOT_KINDS = (
    VariableKind.POSITION,
    VariableKind.ORIENTATION_TANGENT,
    VariableKind.LINEAR_VELOCITY,
    VariableKind.ANGULAR_VELOCITY,
    VariableKind.CONTROLLER_INTEGRAL,
    VariableKind.ACTUATOR_THRUST,
    VariableKind.GIMBAL_ANGLE,
)


def _state():
    keys = [VariableKey(VariableKind.STATIC_PARAMETERS)]
    keys.extend(
        VariableKey(kind, bag_id="bag-a", knot_index=0)
        for kind in KNOT_KINDS
    )
    layout = VariableLayout(tuple(keys))
    values = {}
    for key in layout.variable_keys:
        values[key] = (
            np.eye(3)
            if key.kind is VariableKind.ORIENTATION_TANGENT
            else np.zeros(key.dimension)
        )
    return BatchState(layout, values)


def _definition():
    return DiagonalQDefinition(
        residual_quantity="explicit_test_quantity",
        component_names=("a", "b", "c", "d", "e", "f"),
        component_units=("u",) * 6,
        interval_model=QIntervalModel.FIXED_INTERVAL_COVARIANCE,
    )


def _step(q, lag, marginal, residual=2.0):
    return LaplaceEStepResult(
        q=q,
        lag=lag,
        state=_state(),
        moments=ExpectedResidualMoments(
            np.full((2, 6), residual), np.zeros((2, 6))
        ),
        map_objective=marginal - 1.0,
        approximate_marginal_objective=marginal,
        inner_iterations=3,
        termination_reason="gradient_tolerance",
    )


class BatchEmLoopTests(unittest.TestCase):
    def test_alternates_wide_fixed_and_local_phases_until_converged(self):
        calls = []

        def solver(q, phase, lag, warm_start):
            calls.append((phase, lag, warm_start is not None, q.copy()))
            marginal = float(np.sum((np.log(q) - np.log(4.0)) ** 2))
            selected_lag = 0.012 if phase is not EStepPhase.FIXED_LAG else lag
            return _step(q, selected_lag, marginal)

        progress = []
        result = run_laplace_em(
            _definition(),
            np.ones(6),
            np.full(6, 1.0e-8),
            np.asarray((0.01, 0.02)),
            0.02,
            solver,
            LaplaceEmSettings(
                maximum_iterations=4,
                minimum_iterations=2,
                log_q_tolerance=1.0e-10,
                lag_tolerance=1.0e-10,
                map_objective_tolerance=1.0e-10,
                marginal_objective_tolerance=1.0e-10,
            ),
            progress=progress.append,
        )
        self.assertTrue(result.converged)
        self.assertEqual(
            result.reason,
            LaplaceEmTerminationReason.CONVERGENCE_TOLERANCES,
        )
        np.testing.assert_allclose(result.final_step.q, 4.0)
        self.assertEqual(result.final_step.lag, 0.012)
        self.assertEqual(len(result.iterations), 2)
        self.assertEqual(len(progress), 2)
        self.assertEqual(calls[0][0], EStepPhase.WIDE_LAG_PROFILE)
        self.assertIn(EStepPhase.FIXED_LAG, tuple(item[0] for item in calls))
        self.assertIn(
            EStepPhase.LOCAL_LAG_PROFILE,
            tuple(item[0] for item in calls),
        )

    def test_repeated_nonworsening_rejection_has_distinct_reason(self):
        def solver(q, phase, lag, warm_start):
            if phase is EStepPhase.WIDE_LAG_PROFILE:
                return _step(q, lag, 0.0)
            return _step(q, lag, 1.0)

        result = run_laplace_em(
            _definition(),
            np.ones(6),
            np.full(6, 1.0e-8),
            np.ones(2),
            0.01,
            solver,
            LaplaceEmSettings(
                maximum_iterations=5,
                maximum_repeated_q_rejections=2,
                q_minimum_alpha=0.5,
            ),
        )
        self.assertFalse(result.converged)
        self.assertEqual(
            result.reason,
            LaplaceEmTerminationReason.REPEATED_Q_REJECTION,
        )
        self.assertEqual(len(result.iterations), 2)
        np.testing.assert_array_equal(result.final_step.q, np.ones(6))

    def test_local_lag_failures_keep_fixed_solution_and_are_counted(self):
        def solver(q, phase, lag, warm_start):
            if phase is EStepPhase.LOCAL_LAG_PROFILE:
                raise LaplaceEStepFailure("profile_nonconverged", 4)
            marginal = float(np.sum((np.log(q) - np.log(4.0)) ** 2))
            return _step(q, lag, marginal)

        result = run_laplace_em(
            _definition(),
            np.ones(6),
            np.full(6, 1.0e-8),
            np.ones(2),
            0.01,
            solver,
            LaplaceEmSettings(
                maximum_iterations=4,
                maximum_repeated_lag_profile_failures=2,
            ),
        )
        self.assertEqual(
            result.reason,
            LaplaceEmTerminationReason.REPEATED_LAG_PROFILE_FAILURE,
        )
        self.assertTrue(result.iterations[0].lag_refinement_failed)
        self.assertEqual(
            result.iterations[0].lag_refinement_failure_reason,
            "profile_nonconverged",
        )

    def test_cancel_is_checked_at_em_iteration_boundary(self):
        checks = []

        def solver(q, phase, lag, warm_start):
            return _step(q, lag, 0.0)

        def cancelled():
            checks.append(True)
            return True

        with self.assertRaises(LaplaceEmCancelled) as context:
            run_laplace_em(
                _definition(),
                np.ones(6),
                np.full(6, 1.0e-8),
                np.ones(2),
                0.01,
                solver,
                cancellation_requested=cancelled,
            )
        self.assertEqual(context.exception.iterations, ())
        self.assertEqual(len(checks), 1)


if __name__ == "__main__":
    unittest.main()
