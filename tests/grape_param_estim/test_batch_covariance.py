import unittest

import numpy as np

from grape_param_estim.batch.covariance import ArrowheadLaplaceFactorization
from grape_param_estim.batch.factor import FactorEvaluation, JacobianBlock
from grape_param_estim.batch.layout import VariableLayout
from grape_param_estim.batch.linearize import assemble_sparse_linearization
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


def _layout():
    keys = [VariableKey(VariableKind.STATIC_PARAMETERS)]
    for bag_id in ("bag-a", "bag-b"):
        keys.extend(
            VariableKey(kind, bag_id=bag_id, knot_index=0)
            for kind in KNOT_KINDS
        )
    return VariableLayout(tuple(keys))


def _positive_linearization(layout):
    factors = []
    for key in layout.variable_keys:
        residual = np.zeros(key.dimension)
        factors.append(
            FactorEvaluation(
                residual=residual,
                jacobian_blocks=(JacobianBlock(key, np.eye(key.dimension)),),
                squared_error=0.0,
                active_set={},
            )
        )
    generator = np.random.RandomState(7319)
    cross_factors = []
    static_key = VariableKey(VariableKind.STATIC_PARAMETERS)
    for bag_id in layout.bag_ids:
        position_key = VariableKey(
            VariableKind.POSITION,
            bag_id=bag_id,
            knot_index=0,
        )
        factor = FactorEvaluation(
            residual=np.zeros(4),
            jacobian_blocks=(
                JacobianBlock(static_key, generator.normal(size=(4, 18)) * 0.2),
                JacobianBlock(position_key, generator.normal(size=(4, 3)) * 0.2),
            ),
            squared_error=0.0,
            active_set={},
        )
        factors.append(factor)
        cross_factors.append(factor)
    return assemble_sparse_linearization(layout, tuple(factors)), cross_factors


class BatchCovarianceTests(unittest.TestCase):
    def setUp(self):
        self.layout = _layout()
        self.linearization, self.cross_factors = _positive_linearization(
            self.layout
        )
        self.factorization = ArrowheadLaplaceFactorization(
            self.linearization
        )
        self.dense_hessian = self.linearization.hessian.toarray()

    def test_multiple_rhs_solve_matches_dense_oracle(self):
        generator = np.random.RandomState(843)
        rhs = generator.normal(size=(self.layout.total_dimension, 5))
        expected = np.linalg.solve(self.dense_hessian, rhs)
        actual = self.factorization.solve(rhs)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12)
        vector = rhs[:, 0]
        np.testing.assert_allclose(
            self.factorization.solve(vector),
            expected[:, 0],
            rtol=2.0e-12,
            atol=2.0e-12,
        )

    def test_selected_covariance_and_logdet_match_dense_oracle(self):
        keys = (
            VariableKey(VariableKind.STATIC_PARAMETERS),
            VariableKey(
                VariableKind.POSITION,
                bag_id="bag-b",
                knot_index=0,
            ),
        )
        indices = []
        for key in keys:
            local_slice = self.layout.column_slice(key)
            indices.extend(range(local_slice.start, local_slice.stop))
        inverse = np.linalg.inv(self.dense_hessian)
        expected = inverse[np.ix_(indices, indices)]
        actual = self.factorization.selected_covariance(keys)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12)
        sign, logdet = np.linalg.slogdet(self.dense_hessian)
        self.assertEqual(sign, 1.0)
        self.assertAlmostEqual(
            self.factorization.diagnostics.log_determinant,
            logdet,
            places=10,
        )
        shared = self.layout.shared_slice
        local = slice(shared.stop, self.layout.total_dimension)
        expected_reduced = (
            self.dense_hessian[shared, shared]
            - self.dense_hessian[shared, local]
            @ np.linalg.solve(
                self.dense_hessian[local, local],
                self.dense_hessian[local, shared],
            )
        )
        np.testing.assert_allclose(
            self.factorization.reduced_hessian,
            expected_reduced,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        self.assertFalse(self.factorization.reduced_hessian.flags.writeable)

    def test_factor_residual_covariance_matches_dense_oracle(self):
        factor = self.cross_factors[0]
        dense_jacobian = np.zeros(
            (factor.residual.size, self.layout.total_dimension), dtype=float
        )
        for block in factor.jacobian_blocks:
            dense_jacobian[:, self.layout.column_slice(block.variable_key)] = (
                block.value
            )
        expected = (
            dense_jacobian
            @ np.linalg.solve(self.dense_hessian, dense_jacobian.T)
        )
        actual = self.factorization.residual_covariance(
            factor.jacobian_blocks
        )
        np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12)
        self.assertTrue(np.all(np.linalg.eigvalsh(actual) >= -1.0e-12))

    def test_singular_local_hessian_and_invalid_queries_are_rejected(self):
        factors = []
        for key in self.layout.variable_keys:
            residual = np.zeros(key.dimension)
            jacobian = (
                np.eye(key.dimension)
                if key.kind is VariableKind.STATIC_PARAMETERS
                else np.zeros((key.dimension, key.dimension))
            )
            factors.append(
                FactorEvaluation(
                    residual=residual,
                    jacobian_blocks=(JacobianBlock(key, jacobian),),
                    squared_error=0.0,
                    active_set={},
                )
            )
        singular = assemble_sparse_linearization(self.layout, tuple(factors))
        with self.assertRaises(np.linalg.LinAlgError):
            ArrowheadLaplaceFactorization(singular)
        with self.assertRaisesRegex(ValueError, "unique"):
            key = VariableKey(VariableKind.STATIC_PARAMETERS)
            self.factorization.selected_covariance((key, key))
        with self.assertRaisesRegex(ValueError, "right_hand_side"):
            self.factorization.solve(np.zeros(self.layout.total_dimension - 1))


if __name__ == "__main__":
    unittest.main()
