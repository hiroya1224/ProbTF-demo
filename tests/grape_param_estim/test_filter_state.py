import unittest

import numpy as np

from grape_param_estim.filter_state import (
    FILTER_STATE_DIMENSION,
    GRAPE_FILTER_STATE_LAYOUT,
    GrapeFilterState,
    GrapeFilterStateChart,
    orientation_anchor_from_quaternions,
)
from grape_param_estim.geometry import (
    euler_xyz_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from grape_param_estim.system import (
    ActuatorState,
    ControllerState,
    RigidBodyState,
)


def _state(index, active=True):
    value = float(index)
    return GrapeFilterState(
        rigid=RigidBodyState(
            position=np.asarray((1.0 + value, -2.0 + value, 3.0 - value)),
            orientation_xyzw=matrix_to_quaternion(
                euler_xyz_to_matrix(
                    np.asarray(
                        (0.04 * value, -0.03 * value, 0.06 * value)
                    )
                )
            ),
            linear_velocity=np.asarray((0.1, -0.2, 0.3)) + value,
            angular_velocity=np.asarray((-0.4, 0.5, -0.6)) - value,
        ),
        controller=ControllerState(
            integral_error=np.arange(6, dtype=float) / 10.0 + value,
            roll_pitch_integration_active=active,
        ),
        # Deliberately outside plant limits: this chart must not clip.
        actuator=ActuatorState(
            thrust=np.asarray((-5.0, 10.0, 30.0, 50.0)) + value,
            gimbal_angle=np.asarray((-4.0, -1.0, 2.0, 5.0)) - value,
        ),
        residual_wrench=np.asarray(
            (1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
        )
        * (value + 1.0),
    )


class GrapeFilterStateChartTest(unittest.TestCase):
    def setUp(self):
        self.states = (_state(0), _state(1), _state(2, active=False))
        self.chart = GrapeFilterStateChart.from_ensemble(self.states)

    def assertStateAlmostEqual(self, actual, expected, atol=2.0e-14):
        np.testing.assert_allclose(
            actual.rigid.position, expected.rigid.position, atol=atol
        )
        np.testing.assert_allclose(
            quaternion_to_matrix(actual.rigid.orientation_xyzw),
            quaternion_to_matrix(expected.rigid.orientation_xyzw),
            atol=atol,
        )
        np.testing.assert_allclose(
            actual.rigid.linear_velocity,
            expected.rigid.linear_velocity,
            atol=atol,
        )
        np.testing.assert_allclose(
            actual.rigid.angular_velocity,
            expected.rigid.angular_velocity,
            atol=atol,
        )
        np.testing.assert_allclose(
            actual.controller.integral_error,
            expected.controller.integral_error,
            atol=atol,
        )
        self.assertEqual(
            actual.controller.roll_pitch_integration_active,
            expected.controller.roll_pitch_integration_active,
        )
        np.testing.assert_allclose(
            actual.actuator.thrust, expected.actuator.thrust, atol=atol
        )
        np.testing.assert_allclose(
            actual.actuator.gimbal_angle,
            expected.actuator.gimbal_angle,
            atol=atol,
        )
        np.testing.assert_allclose(
            actual.residual_wrench, expected.residual_wrench, atol=atol
        )

    def test_layout_and_single_state_roundtrip(self):
        layout = GRAPE_FILTER_STATE_LAYOUT
        self.assertEqual(layout.dimension, FILTER_STATE_DIMENSION)
        self.assertEqual(layout.position_slice, slice(0, 3))
        self.assertEqual(layout.orientation_tangent_slice, slice(3, 6))
        self.assertEqual(layout.linear_velocity_slice, slice(6, 9))
        self.assertEqual(layout.angular_velocity_slice, slice(9, 12))
        self.assertEqual(layout.controller_integral_slice, slice(12, 18))
        self.assertEqual(layout.actuator_thrust_slice, slice(18, 22))
        self.assertEqual(layout.actuator_gimbal_slice, slice(22, 26))
        self.assertEqual(layout.residual_wrench_slice, slice(26, 32))

        encoded = self.chart.encode(self.states[1])
        self.assertEqual(encoded.shape, (32,))
        np.testing.assert_array_equal(
            encoded[layout.position_slice], self.states[1].rigid.position
        )
        np.testing.assert_array_equal(
            encoded[layout.actuator_thrust_slice],
            self.states[1].actuator.thrust,
        )
        np.testing.assert_array_equal(
            encoded[layout.actuator_gimbal_slice],
            self.states[1].actuator.gimbal_angle,
        )
        np.testing.assert_array_equal(
            encoded[layout.residual_wrench_slice],
            self.states[1].residual_wrench,
        )
        decoded = self.chart.decode(encoded, self.states[1])
        self.assertStateAlmostEqual(decoded, self.states[1])
        self.assertAlmostEqual(
            np.linalg.norm(decoded.rigid.orientation_xyzw), 1.0
        )

    def test_decode_preserves_only_the_template_controller_flag(self):
        encoded = self.chart.encode(self.states[0])
        false_template = _state(0, active=False)
        decoded = self.chart.decode(encoded, false_template)
        self.assertFalse(decoded.controller.roll_pitch_integration_active)
        np.testing.assert_array_equal(
            decoded.controller.integral_error,
            self.states[0].controller.integral_error,
        )

    def test_member_first_ensemble_roundtrip(self):
        encoded = self.chart.encode_ensemble(self.states)
        self.assertEqual(encoded.shape, (3, 32))
        decoded = self.chart.decode_ensemble(encoded, self.states)
        self.assertEqual(len(decoded), 3)
        for actual, expected in zip(decoded, self.states):
            self.assertStateAlmostEqual(actual, expected)

    def test_orientation_anchor_is_sign_and_permutation_invariant(self):
        quaternions = tuple(
            value.rigid.orientation_xyzw.copy() for value in self.states
        )
        signed = (
            -quaternions[0],
            quaternions[1],
            -quaternions[2],
        )
        first = orientation_anchor_from_quaternions(quaternions)
        changed_sign = orientation_anchor_from_quaternions(signed)
        permuted = orientation_anchor_from_quaternions(
            (quaternions[2], quaternions[0], quaternions[1])
        )
        np.testing.assert_array_equal(first, changed_sign)
        np.testing.assert_array_equal(first, permuted)

        permuted_chart = GrapeFilterStateChart.from_ensemble(
            (self.states[2], self.states[0], self.states[1])
        )
        np.testing.assert_array_equal(
            self.chart.orientation_anchor_xyzw,
            permuted_chart.orientation_anchor_xyzw,
        )
        np.testing.assert_array_equal(
            self.chart.orientation_anchor_matrix,
            permuted_chart.orientation_anchor_matrix,
        )

    def test_pose_predictions_and_observation_use_the_same_chart(self):
        state = self.states[1]
        predicted = self.chart.predicted_pose_coordinates(state)
        observed = self.chart.observed_pose_coordinates(
            state.rigid.position, -state.rigid.orientation_xyzw
        )
        np.testing.assert_allclose(predicted, observed, atol=2.0e-16)
        ensemble = self.chart.predicted_pose_ensemble(self.states)
        self.assertEqual(ensemble.shape, (3, 6))
        np.testing.assert_array_equal(ensemble[1], predicted)
        np.testing.assert_array_equal(
            ensemble[:, :3],
            np.asarray([value.rigid.position for value in self.states]),
        )

    def test_state_and_member_first_validation_is_strict(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            GrapeFilterStateChart.from_ensemble(())
        with self.assertRaises(TypeError):
            GrapeFilterStateChart.from_ensemble((self.states[0], object()))
        with self.assertRaises(TypeError):
            self.chart.encode(object())
        with self.assertRaisesRegex(ValueError, "32"):
            self.chart.decode(np.zeros(31), self.states[0])
        with self.assertRaisesRegex(ValueError, "member-first"):
            self.chart.decode_ensemble(np.zeros((2, 32)), self.states)
        with self.assertRaisesRegex(ValueError, "member-first"):
            self.chart.decode_ensemble(
                np.full((3, 32), np.nan), self.states
            )
        with self.assertRaisesRegex(ValueError, "residual_wrench"):
            GrapeFilterState(
                rigid=self.states[0].rigid,
                controller=self.states[0].controller,
                actuator=self.states[0].actuator,
                residual_wrench=np.zeros(5),
            )
        with self.assertRaisesRegex(ValueError, "pose position"):
            self.chart.observed_pose_coordinates(
                np.zeros(2), self.states[0].rigid.orientation_xyzw
            )
        with self.assertRaisesRegex(ValueError, "quaternion"):
            self.chart.observed_pose_coordinates(
                np.zeros(3), np.zeros(4)
            )

    def test_direct_state_copies_input_arrays(self):
        state = self.states[0]
        wrench = np.arange(6, dtype=float)
        copied = GrapeFilterState(
            rigid=state.rigid,
            controller=state.controller,
            actuator=state.actuator,
            residual_wrench=wrench,
        )
        wrench[:] = -99.0
        np.testing.assert_array_equal(
            copied.residual_wrench, np.arange(6, dtype=float)
        )


if __name__ == "__main__":
    unittest.main()
