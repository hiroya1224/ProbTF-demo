import unittest

import numpy as np
from scipy.sparse import isspmatrix_csc

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


def _complete_keys(bag_knot_counts, include_biases=True):
    keys = [VariableKey(VariableKind.STATIC_PARAMETERS)]
    for bag_id, knot_count in bag_knot_counts.items():
        if include_biases:
            keys.extend(
                (
                    VariableKey(VariableKind.GYRO_BIAS, bag_id=bag_id),
                    VariableKey(
                        VariableKind.ACCELEROMETER_BIAS, bag_id=bag_id
                    ),
                )
            )
        for knot_index in range(knot_count):
            keys.extend(
                VariableKey(kind, bag_id=bag_id, knot_index=knot_index)
                for kind in KNOT_KINDS
            )
    return tuple(keys)


def _one_factor_per_key(layout, generator=None):
    factors = []
    for index, key in enumerate(layout.variable_keys):
        if generator is None:
            residual = np.asarray((1.0,))
            value = np.ones((1, key.dimension), dtype=float)
        else:
            residual = generator.normal(size=2)
            value = generator.normal(size=(2, key.dimension))
        factors.append(
            FactorEvaluation(
                residual=residual,
                jacobian_blocks=(JacobianBlock(key, value),),
                squared_error=float(residual @ residual),
                active_set={
                    "factor_{}_active".format(index): np.asarray(
                        (index % 2 == 0,), dtype=bool
                    )
                },
            )
        )
    return tuple(factors)


class VariableLayoutTests(unittest.TestCase):
    def test_offsets_follow_shared_bag_knot_canonical_order(self):
        supplied = list(_complete_keys({"bag-z": 1, "bag-a": 2}))
        supplied.reverse()
        layout = VariableLayout(tuple(supplied))

        expected_keys = _complete_keys({"bag-a": 2, "bag-z": 1})
        self.assertEqual(layout.variable_keys, expected_keys)
        offset = 0
        for key in expected_keys:
            self.assertEqual(layout.column_offset(key), offset)
            self.assertEqual(
                layout.column_slice(key), slice(offset, offset + key.dimension)
            )
            offset += key.dimension
        self.assertEqual(layout.total_dimension, offset)
        self.assertEqual(offset, 18 + 2 * 6 + 3 * 26)

    def test_optional_biases_do_not_change_complete_knot_requirement(self):
        layout = VariableLayout(_complete_keys({"bag-a": 1}, False))
        self.assertEqual(layout.total_dimension, 18 + 26)
        self.assertEqual(len(layout), 8)

    def test_duplicate_unknown_and_missing_layout_keys_are_rejected(self):
        complete = list(_complete_keys({"bag-a": 2}))
        with self.assertRaisesRegex(ValueError, "unique"):
            VariableLayout(tuple(complete + [complete[-1]]))
        with self.assertRaises(TypeError):
            VariableLayout(tuple(complete[:-1] + [object()]))
        with self.assertRaisesRegex(ValueError, "shared static"):
            VariableLayout(tuple(complete[1:]))

        missing_field = [
            key
            for key in complete
            if not (
                key.kind is VariableKind.GIMBAL_ANGLE
                and key.knot_index == 1
            )
        ]
        with self.assertRaisesRegex(ValueError, "missing canonical"):
            VariableLayout(tuple(missing_field))

        missing_knot = [
            key
            for key in complete
            if key.knot_index != 0
        ]
        with self.assertRaisesRegex(ValueError, "start at zero"):
            VariableLayout(tuple(missing_knot))

    def test_unknown_column_key_is_rejected(self):
        layout = VariableLayout(_complete_keys({"bag-a": 1}))
        with self.assertRaises(TypeError):
            layout.column_slice("position")
        with self.assertRaises(KeyError):
            layout.column_slice(
                VariableKey(
                    VariableKind.POSITION,
                    bag_id="bag-b",
                    knot_index=0,
                )
            )


class SparseLinearizationTests(unittest.TestCase):
    def test_sparse_assembly_matches_dense_oracle_and_provenance(self):
        layout = VariableLayout(_complete_keys({"bag-a": 1}))
        generator = np.random.RandomState(9127)
        factors = _one_factor_per_key(layout, generator)
        result = assemble_sparse_linearization(layout, factors)

        total_rows = sum(factor.residual.size for factor in factors)
        dense_jacobian = np.zeros(
            (total_rows, layout.total_dimension), dtype=float
        )
        dense_residual = np.empty(total_rows, dtype=float)
        row_offset = 0
        for factor in factors:
            next_row = row_offset + factor.residual.size
            dense_residual[row_offset:next_row] = factor.residual
            for block in factor.jacobian_blocks:
                dense_jacobian[
                    row_offset:next_row,
                    layout.column_slice(block.variable_key),
                ] = block.value
            row_offset = next_row

        self.assertTrue(isspmatrix_csc(result.jacobian))
        self.assertTrue(isspmatrix_csc(result.hessian))
        np.testing.assert_allclose(result.jacobian.toarray(), dense_jacobian)
        np.testing.assert_allclose(result.residual, dense_residual)
        self.assertAlmostEqual(
            result.objective, 0.5 * float(dense_residual @ dense_residual)
        )
        np.testing.assert_allclose(
            result.gradient, dense_jacobian.T @ dense_residual
        )
        np.testing.assert_allclose(
            result.hessian.toarray(), dense_jacobian.T @ dense_jacobian
        )
        self.assertEqual(
            result.factor_row_slices,
            tuple(slice(2 * index, 2 * index + 2) for index in range(len(factors))),
        )
        for index, item in enumerate(result.factor_provenance):
            self.assertEqual(item.factor_index, index)
            self.assertEqual(
                item.variable_keys,
                tuple(
                    block.variable_key for block in factors[index].jacobian_blocks
                ),
            )
            np.testing.assert_array_equal(
                item.active_set["factor_{}_active".format(index)],
                (index % 2 == 0,),
            )

    def test_unknown_missing_and_malformed_factor_inputs_are_rejected(self):
        layout = VariableLayout(_complete_keys({"bag-a": 1}))
        factors = _one_factor_per_key(layout)
        with self.assertRaises(TypeError):
            assemble_sparse_linearization(object(), factors)
        with self.assertRaises(ValueError):
            assemble_sparse_linearization(layout, ())
        with self.assertRaises(TypeError):
            assemble_sparse_linearization(layout, factors[:-1] + (object(),))
        with self.assertRaisesRegex(ValueError, "every layout key"):
            assemble_sparse_linearization(layout, factors[:-1])

        unknown_key = VariableKey(
            VariableKind.POSITION,
            bag_id="bag-b",
            knot_index=0,
        )
        unknown_factor = FactorEvaluation(
            residual=np.zeros(1),
            jacobian_blocks=(JacobianBlock(unknown_key, np.zeros((1, 3))),),
            squared_error=0.0,
            active_set={},
        )
        with self.assertRaisesRegex(ValueError, "absent from the layout"):
            assemble_sparse_linearization(
                layout,
                factors + (unknown_factor,),
            )

    def test_large_problem_remains_sparse_with_predictable_nnz(self):
        knot_count = 1200
        layout = VariableLayout(_complete_keys({"large-bag": knot_count}))
        factors = _one_factor_per_key(layout)
        result = assemble_sparse_linearization(layout, factors)

        self.assertTrue(isspmatrix_csc(result.jacobian))
        self.assertTrue(isspmatrix_csc(result.hessian))
        self.assertEqual(result.jacobian.nnz, layout.total_dimension)
        self.assertLess(
            result.jacobian.nnz,
            result.jacobian.shape[0] * result.jacobian.shape[1] // 1000,
        )
        expected_hessian_nnz = sum(
            key.dimension * key.dimension for key in layout.variable_keys
        )
        self.assertEqual(result.hessian.nnz, expected_hessian_nnz)
        self.assertEqual(len(result.factor_provenance), len(factors))


if __name__ == "__main__":
    unittest.main()
