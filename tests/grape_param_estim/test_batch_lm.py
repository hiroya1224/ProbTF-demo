import unittest

import numpy as np

from grape_param_estim.batch.factor import FactorEvaluation, JacobianBlock
from grape_param_estim.batch.factors.prior import (
    evaluate_orientation_prior_factor,
    evaluate_vector_prior_factor,
)
from grape_param_estim.batch.layout import VariableLayout
from grape_param_estim.batch.lm import (
    BatchMapCancelled,
    LMSettings,
    LMTerminationReason,
    solve_batch_map,
    solve_conditional_batch_map,
)
from grape_param_estim.batch.problem import (
    BatchProblem,
    RecoverableModelEvaluationError,
)
from grape_param_estim.batch.state import BatchState, StateScaling
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.geometry import so3_exp, so3_log, so3_right_jacobian_inverse


KNOT_KINDS = (
    VariableKind.POSITION,
    VariableKind.ORIENTATION_TANGENT,
    VariableKind.LINEAR_VELOCITY,
    VariableKind.ANGULAR_VELOCITY,
    VariableKind.CONTROLLER_INTEGRAL,
    VariableKind.ACTUATOR_THRUST,
    VariableKind.GIMBAL_ANGLE,
)


def _layout():
    keys = [VariableKey(VariableKind.STATIC_PARAMETERS)]
    keys.extend(
        VariableKey(kind, bag_id="bag-a", knot_index=0)
        for kind in KNOT_KINDS
    )
    return VariableLayout(tuple(keys))


def _initial_state(layout):
    values = {}
    for index, key in enumerate(layout.variable_keys):
        if key.kind is VariableKind.ORIENTATION_TANGENT:
            values[key] = so3_exp((0.2, -0.12, 0.08))
        else:
            values[key] = np.full(key.dimension, 0.05 + 0.005 * index)
    return BatchState(layout, values)


def _correct_prior_factors(state):
    factors = []
    for key in state.layout.variable_keys:
        if key.kind is VariableKind.ORIENTATION_TANGENT:
            factors.append(
                evaluate_orientation_prior_factor(
                    key,
                    state.value(key),
                    np.eye(3),
                    np.eye(3),
                )
            )
        else:
            factors.append(
                evaluate_vector_prior_factor(
                    key,
                    state.value(key),
                    np.zeros(key.dimension),
                    np.eye(key.dimension),
                )
            )
    return tuple(factors)


def _wrong_sign_factors(state):
    factors = []
    for key in state.layout.variable_keys:
        if key.kind is VariableKind.ORIENTATION_TANGENT:
            residual = so3_log(state.value(key))
            jacobian = -so3_right_jacobian_inverse(residual)
        else:
            residual = state.value(key).copy()
            jacobian = -np.eye(key.dimension)
        factors.append(
            FactorEvaluation(
                residual=residual,
                jacobian_blocks=(JacobianBlock(key, jacobian),),
                squared_error=float(residual @ residual),
                active_set={},
            )
        )
    return tuple(factors)


class BatchLMTests(unittest.TestCase):
    def setUp(self):
        self.layout = _layout()
        self.initial = _initial_state(self.layout)

    def test_sparse_lm_converges_and_preserves_iteration_evidence(self):
        problem = BatchProblem(
            self.layout,
            StateScaling.unit(),
            _correct_prior_factors,
        )
        result = solve_batch_map(
            problem,
            self.initial,
            LMSettings(
                maximum_iterations=12,
                initial_damping=1.0e-3,
                gradient_tolerance=1.0e-10,
                scaled_step_tolerance=1.0e-12,
                relative_objective_tolerance=1.0e-12,
            ),
        )
        self.assertTrue(result.converged)
        self.assertIn(
            result.reason,
            (
                LMTerminationReason.GRADIENT_TOLERANCE,
                LMTerminationReason.RELATIVE_OBJECTIVE_TOLERANCE,
            ),
        )
        self.assertLess(result.objective, 1.0e-16)
        self.assertTrue(any(record.accepted for record in result.iterations))
        for key in self.layout.variable_keys:
            if key.kind is VariableKind.ORIENTATION_TANGENT:
                np.testing.assert_allclose(
                    result.state.value(key), np.eye(3), atol=1.0e-10
                )
            else:
                np.testing.assert_allclose(
                    result.state.value(key), 0.0, atol=1.0e-10
                )

    def test_conditional_lm_optimizes_locals_without_moving_static_block(self):
        problem = BatchProblem(
            self.layout,
            StateScaling.unit(),
            _correct_prior_factors,
        )
        static_key = VariableKey(VariableKind.STATIC_PARAMETERS)
        static_before = self.initial.value(static_key).copy()

        result = solve_conditional_batch_map(
            problem,
            self.initial,
            LMSettings(
                maximum_iterations=12,
                initial_damping=1.0e-3,
                gradient_tolerance=1.0e-10,
                scaled_step_tolerance=1.0e-12,
                relative_objective_tolerance=1.0e-12,
            ),
        )

        self.assertTrue(result.converged)
        np.testing.assert_array_equal(
            result.state.value(static_key), static_before
        )
        self.assertGreater(result.objective, 0.0)
        for key in self.layout.variable_keys[1:]:
            if key.kind is VariableKind.ORIENTATION_TANGENT:
                np.testing.assert_allclose(
                    result.state.value(key), np.eye(3), atol=1.0e-10
                )
            else:
                np.testing.assert_allclose(
                    result.state.value(key), 0.0, atol=1.0e-10
                )

    def test_rejected_steps_raise_damping_and_maximum_is_not_convergence(self):
        problem = BatchProblem(
            self.layout,
            StateScaling.unit(),
            _wrong_sign_factors,
        )
        result = solve_batch_map(
            problem,
            self.initial,
            LMSettings(
                maximum_iterations=3,
                initial_damping=1.0e-3,
                gradient_tolerance=0.0,
                scaled_step_tolerance=0.0,
                relative_objective_tolerance=0.0,
            ),
        )
        self.assertEqual(result.reason, LMTerminationReason.MAXIMUM_ITERATIONS)
        self.assertFalse(result.converged)
        self.assertEqual(len(result.iterations), 3)
        self.assertTrue(all(not item.accepted for item in result.iterations))
        self.assertGreater(result.final_damping, 1.0e-3)
        for key in self.layout.variable_keys:
            np.testing.assert_array_equal(
                result.state.value(key), self.initial.value(key)
            )

    def test_unregularized_singular_local_system_reports_failure(self):
        def singular_factors(state):
            factors = []
            for key in state.layout.variable_keys:
                dimension = (
                    3
                    if key.kind is VariableKind.ORIENTATION_TANGENT
                    else key.dimension
                )
                residual = np.ones(dimension)
                jacobian = (
                    np.eye(dimension)
                    if key.kind is VariableKind.STATIC_PARAMETERS
                    else np.zeros((dimension, key.dimension))
                )
                factors.append(
                    FactorEvaluation(
                        residual=residual,
                        jacobian_blocks=(
                            JacobianBlock(key, jacobian),
                        ),
                        squared_error=float(residual @ residual),
                        active_set={},
                    )
                )
            return tuple(factors)

        problem = BatchProblem(
            self.layout,
            StateScaling.unit(),
            singular_factors,
        )
        result = solve_batch_map(
            problem,
            self.initial,
            LMSettings(
                maximum_iterations=3,
                initial_damping=0.0,
                minimum_damping=0.0,
                maximum_damping=0.0,
                gradient_tolerance=0.0,
                scaled_step_tolerance=0.0,
                relative_objective_tolerance=0.0,
                maximum_factorization_retries=1,
            ),
        )
        self.assertEqual(
            result.reason,
            LMTerminationReason.NUMERICAL_FACTORIZATION_FAILURE,
        )
        self.assertFalse(result.converged)
        self.assertTrue(result.iterations[-1].factorization_failed)

    def test_repeated_recoverable_trial_failure_has_explicit_reason(self):
        initial = self.initial

        def evaluator(state):
            if state is not initial:
                raise RecoverableModelEvaluationError("outside model domain")
            return _correct_prior_factors(state)

        problem = BatchProblem(
            self.layout,
            StateScaling.unit(),
            evaluator,
        )
        result = solve_batch_map(
            problem,
            initial,
            LMSettings(
                maximum_iterations=5,
                initial_damping=1.0e-3,
                gradient_tolerance=0.0,
                scaled_step_tolerance=0.0,
                relative_objective_tolerance=0.0,
                maximum_model_evaluation_retries=2,
            ),
        )
        self.assertEqual(
            result.reason,
            LMTerminationReason.NONFINITE_MODEL_EVALUATION,
        )
        self.assertFalse(result.converged)
        self.assertEqual(len(result.iterations), 2)
        self.assertTrue(
            all(item.model_evaluation_failed for item in result.iterations)
        )

    def test_problem_and_settings_validate_contracts(self):
        with self.assertRaises(ValueError):
            LMSettings(maximum_iterations=0)
        with self.assertRaises(ValueError):
            LMSettings(
                minimum_damping=2.0,
                initial_damping=1.0,
                maximum_damping=3.0,
            )
        with self.assertRaises(TypeError):
            BatchProblem(self.layout, StateScaling.unit(), object())

    def test_progress_and_cancellation_use_nonlinear_iteration_boundaries(self):
        problem = BatchProblem(
            self.layout,
            StateScaling.unit(),
            _wrong_sign_factors,
        )
        records = []

        def cancelled():
            return len(records) == 2

        with self.assertRaises(BatchMapCancelled) as context:
            solve_batch_map(
                problem,
                self.initial,
                LMSettings(
                    maximum_iterations=10,
                    gradient_tolerance=0.0,
                    scaled_step_tolerance=0.0,
                    relative_objective_tolerance=0.0,
                ),
                cancellation_requested=cancelled,
                progress=records.append,
            )
        self.assertEqual(len(records), 2)
        self.assertEqual(context.exception.iterations, tuple(records))


if __name__ == "__main__":
    unittest.main()
