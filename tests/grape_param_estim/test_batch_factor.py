from dataclasses import FrozenInstanceError
import unittest

import numpy as np

from grape_param_estim.batch import (
    FactorEvaluation,
    JacobianBlock,
    VariableKey,
    VariableKind,
    VariableScope,
)


class VariableKeyTests(unittest.TestCase):
    def test_every_planned_key_kind_has_canonical_scope_and_dimension(self):
        shared = VariableKey(VariableKind.STATIC_PARAMETERS)
        bag_keys = (
            VariableKey(VariableKind.GYRO_BIAS, bag_id="flight-a"),
            VariableKey(
                VariableKind.ACCELEROMETER_BIAS, bag_id="flight-a"
            ),
        )
        knot_specs = (
            (VariableKind.POSITION, 3),
            (VariableKind.ORIENTATION_TANGENT, 3),
            (VariableKind.LINEAR_VELOCITY, 3),
            (VariableKind.ANGULAR_VELOCITY, 3),
            (VariableKind.CONTROLLER_INTEGRAL, 6),
            (VariableKind.ACTUATOR_THRUST, 4),
            (VariableKind.GIMBAL_ANGLE, 4),
        )
        knot_keys = tuple(
            VariableKey(kind, bag_id="flight-a", knot_index=7)
            for kind, _ in knot_specs
        )

        self.assertEqual(shared.scope, VariableScope.SHARED)
        self.assertEqual(shared.dimension, 18)
        for key in bag_keys:
            self.assertEqual(key.scope, VariableScope.BAG)
            self.assertEqual(key.dimension, 3)
        for key, (_, dimension) in zip(knot_keys, knot_specs):
            self.assertEqual(key.scope, VariableScope.KNOT)
            self.assertEqual(key.dimension, dimension)
        self.assertEqual(sum(key.dimension for key in knot_keys), 26)
        self.assertEqual(len(set((shared,) + bag_keys + knot_keys)), 10)

    def test_key_is_immutable_hashable_and_normalizes_integral_index(self):
        key = VariableKey(
            VariableKind.POSITION,
            bag_id="flight-a",
            knot_index=np.int64(3),
        )
        self.assertEqual(key.knot_index, 3)
        self.assertIs(type(key.knot_index), int)
        self.assertEqual({key: "value"}[key], "value")
        with self.assertRaises(FrozenInstanceError):
            key.knot_index = 4

    def test_malformed_kind_and_scope_are_rejected(self):
        with self.assertRaises(TypeError):
            VariableKey("position", bag_id="flight-a", knot_index=0)
        for bag_id, knot_index in (("flight-a", None), (None, 0)):
            with self.assertRaisesRegex(ValueError, "shared"):
                VariableKey(
                    VariableKind.STATIC_PARAMETERS,
                    bag_id=bag_id,
                    knot_index=knot_index,
                )
        for kind in (VariableKind.GYRO_BIAS, VariableKind.ACCELEROMETER_BIAS):
            with self.assertRaisesRegex(ValueError, "bag_id"):
                VariableKey(kind)
            with self.assertRaisesRegex(ValueError, "knot_index"):
                VariableKey(kind, bag_id="flight-a", knot_index=0)
        for kind in (
            VariableKind.POSITION,
            VariableKind.ORIENTATION_TANGENT,
            VariableKind.LINEAR_VELOCITY,
            VariableKind.ANGULAR_VELOCITY,
            VariableKind.CONTROLLER_INTEGRAL,
            VariableKind.ACTUATOR_THRUST,
            VariableKind.GIMBAL_ANGLE,
        ):
            with self.assertRaisesRegex(ValueError, "bag_id"):
                VariableKey(kind, knot_index=0)
            for invalid_index in (None, -1, 1.5, True):
                with self.assertRaisesRegex(ValueError, "knot_index"):
                    VariableKey(
                        kind,
                        bag_id="flight-a",
                        knot_index=invalid_index,
                    )
        for invalid_bag_id in ("", " flight-a", "flight-a ", 4):
            with self.assertRaisesRegex(ValueError, "bag_id"):
                VariableKey(VariableKind.GYRO_BIAS, bag_id=invalid_bag_id)


class FactorContractTests(unittest.TestCase):
    def setUp(self):
        self.position_key = VariableKey(
            VariableKind.POSITION,
            bag_id="flight-a",
            knot_index=2,
        )
        self.static_key = VariableKey(VariableKind.STATIC_PARAMETERS)

    def test_all_key_kinds_form_canonical_factor_local_blocks(self):
        keys = (
            self.static_key,
            VariableKey(VariableKind.GYRO_BIAS, bag_id="flight-a"),
            VariableKey(
                VariableKind.ACCELEROMETER_BIAS, bag_id="flight-a"
            ),
        ) + tuple(
            VariableKey(kind, bag_id="flight-a", knot_index=2)
            for kind in (
                VariableKind.POSITION,
                VariableKind.ORIENTATION_TANGENT,
                VariableKind.LINEAR_VELOCITY,
                VariableKind.ANGULAR_VELOCITY,
                VariableKind.CONTROLLER_INTEGRAL,
                VariableKind.ACTUATOR_THRUST,
                VariableKind.GIMBAL_ANGLE,
            )
        )
        blocks = tuple(
            JacobianBlock(key, np.zeros((2, key.dimension))) for key in keys
        )
        evaluation = FactorEvaluation(
            residual=np.zeros(2),
            jacobian_blocks=blocks,
            squared_error=0.0,
            active_set={},
        )

        self.assertEqual(len(evaluation.jacobian_blocks), len(VariableKind))
        for key, block in zip(keys, evaluation.jacobian_blocks):
            self.assertEqual(block.value.shape, (2, key.dimension))

    def test_valid_evaluation_copies_and_freezes_factor_local_arrays(self):
        residual = np.asarray((3.0, 4.0))
        position_value = np.arange(6, dtype=float).reshape(2, 3)
        static_value = np.arange(36, dtype=float).reshape(2, 18)
        active_mask = np.asarray((True, False), dtype=bool)
        evaluation = FactorEvaluation(
            residual=residual,
            jacobian_blocks=(
                JacobianBlock(self.position_key, position_value),
                JacobianBlock(self.static_key, static_value),
            ),
            squared_error=25.0,
            active_set={"saturated": active_mask},
        )

        residual[:] = 0.0
        position_value[:] = -1.0
        active_mask[:] = False
        np.testing.assert_array_equal(evaluation.residual, (3.0, 4.0))
        np.testing.assert_array_equal(
            evaluation.jacobian_blocks[0].value,
            np.arange(6, dtype=float).reshape(2, 3),
        )
        np.testing.assert_array_equal(
            evaluation.active_set["saturated"], (True, False)
        )
        self.assertFalse(evaluation.residual.flags.writeable)
        self.assertFalse(evaluation.jacobian_blocks[0].value.flags.writeable)
        self.assertFalse(evaluation.active_set["saturated"].flags.writeable)
        with self.assertRaises(TypeError):
            evaluation.active_set["new"] = np.asarray((True,))

    def test_jacobian_block_rejects_bad_key_shape_and_nonfinite_values(self):
        with self.assertRaises(TypeError):
            JacobianBlock("position", np.zeros((2, 3)))
        for invalid in (
            np.zeros(3),
            np.zeros((0, 3)),
            np.zeros((2, 2)),
            np.full((2, 3), np.nan),
            np.full((2, 3), np.inf),
        ):
            with self.assertRaises(ValueError):
                JacobianBlock(self.position_key, invalid)

    def test_factor_rejects_residual_and_row_alignment_errors(self):
        valid_block = JacobianBlock(self.position_key, np.zeros((2, 3)))
        for invalid_residual in (
            np.zeros((1, 2)),
            np.zeros(0),
            np.asarray((0.0, np.nan)),
            np.asarray((0.0, np.inf)),
        ):
            with self.assertRaises(ValueError):
                FactorEvaluation(
                    residual=invalid_residual,
                    jacobian_blocks=(valid_block,),
                    squared_error=0.0,
                    active_set={},
                )
        with self.assertRaisesRegex(ValueError, "one row"):
            FactorEvaluation(
                residual=np.zeros(2),
                jacobian_blocks=(
                    JacobianBlock(self.position_key, np.zeros((3, 3))),
                ),
                squared_error=0.0,
                active_set={},
            )

    def test_factor_rejects_non_tuple_duplicate_and_malformed_blocks(self):
        block = JacobianBlock(self.position_key, np.zeros((2, 3)))
        with self.assertRaisesRegex(TypeError, "tuple"):
            FactorEvaluation(np.zeros(2), [block], 0.0, {})
        with self.assertRaisesRegex(TypeError, "tuple"):
            FactorEvaluation(np.zeros(2), (), 0.0, {})
        with self.assertRaisesRegex(TypeError, "JacobianBlock"):
            FactorEvaluation(np.zeros(2), (object(),), 0.0, {})
        with self.assertRaisesRegex(ValueError, "unique"):
            FactorEvaluation(np.zeros(2), (block, block), 0.0, {})

    def test_factor_rejects_inconsistent_or_nonfinite_squared_error(self):
        block = JacobianBlock(self.position_key, np.zeros((2, 3)))
        for invalid in (-1.0, 4.1, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "squared_error"):
                FactorEvaluation(
                    np.asarray((1.0, 2.0)),
                    (block,),
                    invalid,
                    {},
                )
        with self.assertRaises(TypeError):
            FactorEvaluation(np.zeros(2), (block,), True, {})
        with self.assertRaises(TypeError):
            FactorEvaluation(np.zeros(2), (block,), np.asarray(0.0), {})

    def test_factor_rejects_malformed_active_set(self):
        block = JacobianBlock(self.position_key, np.zeros((2, 3)))
        invalid_active_sets = (
            [],
            {"": np.asarray((True,))},
            {" near_kink": np.asarray((True,))},
            {"near_kink": [True]},
            {"near_kink": np.asarray((1,), dtype=int)},
            {"near_kink": np.asarray(True)},
            {"near_kink": np.asarray((), dtype=bool)},
        )
        for active_set in invalid_active_sets:
            with self.assertRaises((TypeError, ValueError)):
                FactorEvaluation(
                    np.zeros(2),
                    (block,),
                    0.0,
                    active_set,
                )


if __name__ == "__main__":
    unittest.main()
