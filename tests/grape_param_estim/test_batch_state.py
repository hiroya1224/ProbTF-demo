import unittest

import numpy as np

from grape_param_estim.batch.layout import VariableLayout
from grape_param_estim.batch.state import BatchState, StateScaling
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.geometry import so3_exp


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
    for bag_id in ("bag-b", "bag-a"):
        keys.extend(
            (
                VariableKey(VariableKind.GYRO_BIAS, bag_id=bag_id),
                VariableKey(
                    VariableKind.ACCELEROMETER_BIAS,
                    bag_id=bag_id,
                ),
            )
        )
        for knot_index in range(2):
            keys.extend(
                VariableKey(kind, bag_id=bag_id, knot_index=knot_index)
                for kind in KNOT_KINDS
            )
    return VariableLayout(tuple(reversed(keys)))


def _state_values(layout):
    values = {}
    for key_index, key in enumerate(layout.variable_keys):
        if key.kind is VariableKind.ORIENTATION_TANGENT:
            values[key] = so3_exp(
                (0.01 * key_index, -0.005 * key_index, 0.002 * key_index)
            )
        else:
            values[key] = np.full(key.dimension, 0.1 * key_index)
    return values


class BatchStateTests(unittest.TestCase):
    def setUp(self):
        self.layout = _layout()
        self.input_values = _state_values(self.layout)
        self.state = BatchState(self.layout, self.input_values)

    def test_complete_two_bag_state_is_copied_read_only_and_addressable(self):
        first_key = self.layout.variable_keys[0]
        original = self.state.value(first_key).copy()
        self.input_values[first_key][:] = 99.0
        np.testing.assert_array_equal(self.state.value(first_key), original)
        self.assertFalse(self.state.value(first_key).flags.writeable)
        with self.assertRaises(TypeError):
            self.state.values[first_key] = np.zeros(first_key.dimension)

        position = self.state.knot_value(
            "bag-a", 1, VariableKind.POSITION
        )
        key = VariableKey(
            VariableKind.POSITION,
            bag_id="bag-a",
            knot_index=1,
        )
        self.assertIs(position, self.state.value(key))

    def test_retraction_adds_vectors_and_right_retracts_rotations(self):
        delta = np.linspace(
            -0.02,
            0.03,
            self.layout.total_dimension,
        )
        retracted = self.state.retract(delta)
        for key in self.layout.variable_keys:
            local = delta[self.layout.column_slice(key)]
            if key.kind is VariableKind.ORIENTATION_TANGENT:
                expected = self.state.value(key) @ so3_exp(local)
            else:
                expected = self.state.value(key) + local
            np.testing.assert_allclose(
                retracted.value(key), expected, rtol=0.0, atol=2.0e-15
            )
            np.testing.assert_array_equal(
                self.state.value(key),
                BatchState(self.layout, _state_values(self.layout)).value(key),
            )

    def test_zero_retraction_preserves_every_nonlinear_value(self):
        retracted = self.state.retract(
            np.zeros(self.layout.total_dimension, dtype=float)
        )
        for key in self.layout.variable_keys:
            np.testing.assert_array_equal(
                retracted.value(key), self.state.value(key)
            )

    def test_incomplete_extra_and_malformed_values_are_rejected(self):
        missing = dict(self.input_values)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError, "every layout key"):
            BatchState(self.layout, missing)

        extra = dict(self.input_values)
        extra[
            VariableKey(VariableKind.GYRO_BIAS, bag_id="unknown")
        ] = np.zeros(3)
        with self.assertRaisesRegex(ValueError, "every layout key"):
            BatchState(self.layout, extra)

        orientation_key = next(
            key
            for key in self.layout.variable_keys
            if key.kind is VariableKind.ORIENTATION_TANGENT
        )
        malformed = dict(self.input_values)
        malformed[orientation_key] = 2.0 * np.eye(3)
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            BatchState(self.layout, malformed)

        position_key = next(
            key
            for key in self.layout.variable_keys
            if key.kind is VariableKind.POSITION
        )
        malformed = dict(self.input_values)
        malformed[position_key] = np.asarray((0.0, np.nan, 0.0))
        with self.assertRaisesRegex(ValueError, "finite"):
            BatchState(self.layout, malformed)

    def test_retraction_rejects_bad_shape_and_nonfinite_delta(self):
        with self.assertRaisesRegex(ValueError, "physical_delta"):
            self.state.retract(np.zeros(self.layout.total_dimension - 1))
        delta = np.zeros(self.layout.total_dimension)
        delta[0] = np.inf
        with self.assertRaisesRegex(ValueError, "physical_delta"):
            self.state.retract(delta)


class StateScalingTests(unittest.TestCase):
    def test_kind_scales_expand_in_layout_order(self):
        layout = _layout()
        scalar_by_kind = {
            kind: float(index + 1)
            for index, kind in enumerate(VariableKind)
        }
        scaling = StateScaling(scalar_by_kind)
        vector = scaling.vector_for(layout)
        self.assertEqual(vector.shape, (layout.total_dimension,))
        self.assertFalse(vector.flags.writeable)
        for key in layout.variable_keys:
            np.testing.assert_array_equal(
                vector[layout.column_slice(key)],
                np.full(key.dimension, scalar_by_kind[key.kind]),
            )

    def test_scaling_requires_every_kind_and_positive_finite_values(self):
        valid = {kind: 1.0 for kind in VariableKind}
        missing = dict(valid)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError, "every VariableKind"):
            StateScaling(missing)
        for invalid in (0.0, -1.0, np.inf, np.nan):
            values = dict(valid)
            values[VariableKind.POSITION] = invalid
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    StateScaling(values)
        values = dict(valid)
        values[VariableKind.POSITION] = True
        with self.assertRaises(TypeError):
            StateScaling(values)


if __name__ == "__main__":
    unittest.main()
