from dataclasses import replace
import unittest
from unittest import mock

import numpy as np

import grape_param_estim.batch.sparse_solver as sparse_solver_module
from grape_param_estim.batch.factor import FactorEvaluation, JacobianBlock
from grape_param_estim.batch.layout import VariableLayout
from grape_param_estim.batch.linearize import assemble_sparse_linearization
from grape_param_estim.batch.sparse_solver import (
    solve_scaled_conditional_lm_step,
    solve_scaled_lm_step,
)
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


def _complete_keys(bag_ids):
    keys = [VariableKey(VariableKind.STATIC_PARAMETERS)]
    for bag_id in bag_ids:
        keys.extend(
            (
                VariableKey(VariableKind.GYRO_BIAS, bag_id=bag_id),
                VariableKey(
                    VariableKind.ACCELEROMETER_BIAS, bag_id=bag_id
                ),
            )
        )
        keys.extend(
            VariableKey(kind, bag_id=bag_id, knot_index=0)
            for kind in KNOT_KINDS
        )
    return tuple(keys)


def _arrowhead_problem(bag_ids=("bag-a", "bag-b"), seed=441):
    layout = VariableLayout(_complete_keys(bag_ids))
    generator = np.random.RandomState(seed)
    factors = []
    for key in layout.variable_keys:
        residual = generator.normal(scale=0.2, size=key.dimension)
        weight = 0.7 + generator.uniform()
        factors.append(
            FactorEvaluation(
                residual=residual,
                jacobian_blocks=(
                    JacobianBlock(key, weight * np.eye(key.dimension)),
                ),
                squared_error=float(residual @ residual),
                active_set={},
            )
        )

    static_key = VariableKey(VariableKind.STATIC_PARAMETERS)
    for bag_id in bag_ids:
        position_key = VariableKey(
            VariableKind.POSITION,
            bag_id=bag_id,
            knot_index=0,
        )
        residual = generator.normal(scale=0.1, size=5)
        factors.append(
            FactorEvaluation(
                residual=residual,
                jacobian_blocks=(
                    JacobianBlock(
                        static_key, generator.normal(scale=0.1, size=(5, 18))
                    ),
                    JacobianBlock(
                        position_key,
                        generator.normal(scale=0.1, size=(5, 3)),
                    ),
                ),
                squared_error=float(residual @ residual),
                active_set={"near_kink": np.zeros(5, dtype=bool)},
            )
        )
    return layout, tuple(factors), assemble_sparse_linearization(layout, factors)


def _dense_scaled_step(linearization, scale, damping):
    hessian = linearization.hessian.toarray()
    scaling = np.diag(scale)
    scaled_hessian = scaling @ hessian @ scaling
    scaled_gradient = scale * linearization.gradient
    scaled_delta = np.linalg.solve(
        scaled_hessian + damping * np.eye(scale.size),
        -scaled_gradient,
    )
    return scale * scaled_delta, scaled_delta


class ScaledSchurStepTests(unittest.TestCase):
    def test_layout_exposes_exact_shared_and_contiguous_bag_slices(self):
        layout = VariableLayout(_complete_keys(("bag-z", "bag-a")))
        self.assertEqual(layout.shared_slice, slice(0, 18))
        self.assertEqual(layout.bag_ids, ("bag-a", "bag-z"))
        self.assertEqual(layout.bag_slice("bag-a").start, 18)
        self.assertEqual(
            layout.bag_slice("bag-a").stop,
            layout.bag_slice("bag-z").start,
        )
        self.assertEqual(
            layout.bag_slice("bag-z").stop, layout.total_dimension
        )
        with self.assertRaises(TypeError):
            layout.bag_slice(2)
        with self.assertRaises(KeyError):
            layout.bag_slice("unknown")

    def test_two_bag_schur_step_matches_full_dense_scaled_system(self):
        _, _, linearization = _arrowhead_problem()
        generator = np.random.RandomState(91)
        scale = np.exp(
            generator.normal(scale=0.35, size=linearization.layout.total_dimension)
        )
        damping = 0.23
        result = solve_scaled_lm_step(linearization, scale, damping)
        expected_delta, expected_scaled_delta = _dense_scaled_step(
            linearization, scale, damping
        )

        np.testing.assert_allclose(
            result.delta, expected_delta, rtol=2.0e-11, atol=2.0e-12
        )
        np.testing.assert_allclose(
            result.scaled_delta,
            expected_scaled_delta,
            rtol=2.0e-11,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(result.delta, scale * result.scaled_delta)
        self.assertAlmostEqual(
            result.gradient_inf_norm,
            float(np.max(np.abs(scale * linearization.gradient))),
        )
        self.assertAlmostEqual(
            result.scaled_step_norm,
            float(np.linalg.norm(expected_scaled_delta)),
        )
        expected_prediction = -float(
            linearization.gradient @ expected_delta
            + 0.5
            * expected_delta
            @ linearization.hessian.dot(expected_delta)
        )
        self.assertAlmostEqual(
            result.predicted_reduction, expected_prediction
        )

    def test_reduced_system_matches_dense_schur_oracle(self):
        layout, _, linearization = _arrowhead_problem()
        scale = np.linspace(
            0.7, 1.5, layout.total_dimension, dtype=float
        )
        damping = 0.11
        result = solve_scaled_lm_step(linearization, scale, damping)

        dense_hessian = linearization.hessian.toarray()
        scaled = scale[:, None] * dense_hessian * scale[None, :]
        damped = scaled + damping * np.eye(layout.total_dimension)
        scaled_gradient = scale * linearization.gradient
        shared = layout.shared_slice
        expected_hessian = damped[shared, shared].copy()
        expected_rhs = -scaled_gradient[shared].copy()
        for bag_id in layout.bag_ids:
            local = layout.bag_slice(bag_id)
            solved_cross = np.linalg.solve(
                damped[local, local], damped[local, shared]
            )
            solved_gradient = np.linalg.solve(
                damped[local, local], scaled_gradient[local]
            )
            expected_hessian -= damped[shared, local] @ solved_cross
            expected_rhs += damped[shared, local] @ solved_gradient

        np.testing.assert_allclose(
            result.reduced_hessian,
            expected_hessian,
            rtol=2.0e-12,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            result.reduced_rhs,
            expected_rhs,
            rtol=2.0e-12,
            atol=2.0e-13,
        )

    def test_each_bag_is_factorized_and_solved_once_with_19_rhs(self):
        layout, _, linearization = _arrowhead_problem()
        real_splu = sparse_solver_module.splu
        counts = {"factorization": 0, "solve": 0}

        class CountingFactorization:
            def __init__(self, factorization):
                self._factorization = factorization
                self.L = factorization.L
                self.U = factorization.U

            def solve(self, rhs):
                counts["solve"] += 1
                return self._factorization.solve(rhs)

        def counting_splu(matrix):
            counts["factorization"] += 1
            return CountingFactorization(real_splu(matrix))

        with mock.patch.object(
            sparse_solver_module, "splu", side_effect=counting_splu
        ):
            result = solve_scaled_lm_step(
                linearization,
                np.ones(layout.total_dimension),
                0.2,
            )

        self.assertEqual(counts["factorization"], len(layout.bag_ids))
        self.assertEqual(counts["solve"], len(layout.bag_ids))
        for diagnostic in result.bag_diagnostics:
            self.assertEqual(diagnostic.rhs_count, 19)
            self.assertGreater(diagnostic.local_dimension, 0)
            self.assertGreater(diagnostic.damped_hessian_nnz, 0)
            self.assertGreater(diagnostic.factor_l_nnz, 0)
            self.assertGreater(diagnostic.factor_u_nnz, 0)

    def test_conditional_step_matches_dense_local_solves_and_fixes_shared(self):
        layout, _, linearization = _arrowhead_problem()
        scale = np.linspace(0.6, 1.7, layout.total_dimension)
        damping = 0.17

        result = solve_scaled_conditional_lm_step(
            linearization, scale, damping
        )

        dense_hessian = linearization.hessian.toarray()
        scaled_hessian = scale[:, None] * dense_hessian * scale[None, :]
        scaled_gradient = scale * linearization.gradient
        expected_scaled = np.zeros(layout.total_dimension)
        for bag_id in layout.bag_ids:
            local = layout.bag_slice(bag_id)
            expected_scaled[local] = np.linalg.solve(
                scaled_hessian[local, local]
                + damping * np.eye(local.stop - local.start),
                -scaled_gradient[local],
            )

        np.testing.assert_array_equal(
            result.delta[layout.shared_slice], np.zeros(18)
        )
        np.testing.assert_allclose(
            result.scaled_delta,
            expected_scaled,
            rtol=2.0e-11,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(result.delta, scale * expected_scaled)
        self.assertAlmostEqual(
            result.gradient_inf_norm,
            float(np.max(np.abs(scaled_gradient[layout.shared_slice.stop :]))),
        )
        self.assertTrue(
            all(item.rhs_count == 1 for item in result.bag_diagnostics)
        )

    def test_invalid_scale_damping_and_nonfinite_system_are_rejected(self):
        layout, _, linearization = _arrowhead_problem(("bag-a",))
        dimension = layout.total_dimension
        invalid_scales = (
            np.ones(dimension - 1),
            np.zeros(dimension),
            -np.ones(dimension),
            np.full(dimension, np.nan),
            np.full(dimension, np.inf),
        )
        for scale in invalid_scales:
            with self.assertRaises(ValueError):
                solve_scaled_lm_step(linearization, scale, 0.1)
        for damping in (-0.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                solve_scaled_lm_step(
                    linearization, np.ones(dimension), damping
                )
        for damping in (True, "0.1"):
            with self.assertRaises(TypeError):
                solve_scaled_lm_step(
                    linearization, np.ones(dimension), damping
                )

        bad_hessian = linearization.hessian.copy()
        bad_hessian.data[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite CSC"):
            solve_scaled_lm_step(
                replace(linearization, hessian=bad_hessian),
                np.ones(dimension),
                0.1,
            )
        bad_gradient = linearization.gradient.copy()
        bad_gradient[0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite CSC"):
            solve_scaled_lm_step(
                replace(linearization, gradient=bad_gradient),
                np.ones(dimension),
                0.1,
            )

    def test_singular_local_system_is_reported(self):
        layout = VariableLayout(_complete_keys(("bag-a",)))
        factors = tuple(
            FactorEvaluation(
                residual=np.zeros(1),
                jacobian_blocks=(
                    JacobianBlock(key, np.zeros((1, key.dimension))),
                ),
                squared_error=0.0,
                active_set={},
            )
            for key in layout.variable_keys
        )
        linearization = assemble_sparse_linearization(layout, factors)
        with self.assertRaisesRegex(np.linalg.LinAlgError, "bag-local"):
            solve_scaled_lm_step(
                linearization,
                np.ones(layout.total_dimension),
                0.0,
            )

    def test_cross_bag_factors_and_hessian_entries_are_rejected(self):
        layout, factors, linearization = _arrowhead_problem()
        key_a = VariableKey(
            VariableKind.POSITION, bag_id="bag-a", knot_index=0
        )
        key_b = VariableKey(
            VariableKind.POSITION, bag_id="bag-b", knot_index=0
        )
        cross_factor = FactorEvaluation(
            residual=np.ones(1),
            jacobian_blocks=(
                JacobianBlock(key_a, np.ones((1, 3))),
                JacobianBlock(key_b, np.ones((1, 3))),
            ),
            squared_error=1.0,
            active_set={},
        )
        with self.assertRaisesRegex(ValueError, "multiple bags"):
            assemble_sparse_linearization(
                layout, factors + (cross_factor,)
            )

        bad_hessian = linearization.hessian.tolil()
        first = layout.bag_slice("bag-a").start
        second = layout.bag_slice("bag-b").start
        bad_hessian[first, second] = 0.25
        bad_hessian[second, first] = 0.25
        with self.assertRaisesRegex(ValueError, "multiple bags"):
            solve_scaled_lm_step(
                replace(linearization, hessian=bad_hessian.tocsc()),
                np.ones(layout.total_dimension),
                0.1,
            )


if __name__ == "__main__":
    unittest.main()
