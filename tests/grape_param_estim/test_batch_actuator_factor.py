import unittest

import numpy as np

from grape_param_estim.batch.factors.actuator import (
    evaluate_actuator_interval_factor,
    evaluate_actuator_transition_factor,
    evaluate_gimbal_position_factor,
)
from grape_param_estim.batch.variables import VariableKind
from grape_param_estim.system import (
    ActuatorCommand,
    ActuatorParameters,
)


def _command(thrust, gimbal):
    return ActuatorCommand(
        thrust=thrust,
        gimbal_angle=gimbal,
        virtual_force=np.zeros(8),
        desired_acceleration=np.zeros(6),
    )


class BatchActuatorFactorTests(unittest.TestCase):
    def setUp(self):
        self.thrust0 = np.asarray((5.0, 6.0, 7.0, 8.0))
        self.thrust1 = np.asarray((5.4, 5.8, 7.3, 7.7))
        self.gimbal0 = np.asarray((0.1, -0.2, 0.25, -0.3))
        self.gimbal1 = np.asarray((0.12, -0.21, 0.23, -0.27))
        self.command = _command(
            (7.0, 5.0, 9.0, 6.0),
            (0.15, -0.25, 0.18, -0.22),
        )
        self.parameters = ActuatorParameters(
            thrust_time_constant=0.08,
            gimbal_time_constant=0.12,
            minimum_thrust=1.5,
            maximum_thrust=20.0,
            maximum_gimbal_angle=1.2,
            maximum_gimbal_rate=10.0,
        )
        self.dt = 0.025
        self.thrust_whitening = np.diag((1.2, 0.9, 1.1, 0.8))
        self.gimbal_whitening = np.diag((2.0, 1.8, 2.2, 1.7))

    def _evaluate(
        self,
        thrust0=None,
        thrust1=None,
        gimbal0=None,
        gimbal1=None,
        command=None,
        parameters=None,
    ):
        return evaluate_actuator_transition_factor(
            "bag-a",
            3,
            self.thrust0 if thrust0 is None else thrust0,
            self.thrust1 if thrust1 is None else thrust1,
            self.gimbal0 if gimbal0 is None else gimbal0,
            self.gimbal1 if gimbal1 is None else gimbal1,
            self.command if command is None else command,
            self.parameters if parameters is None else parameters,
            self.dt,
            self.thrust_whitening,
            self.gimbal_whitening,
        )

    def test_analytic_state_blocks_match_central_difference(self):
        factor = self._evaluate()
        self.assertEqual(
            tuple(block.variable_key.kind for block in factor.jacobian_blocks),
            (
                VariableKind.ACTUATOR_THRUST,
                VariableKind.ACTUATOR_THRUST,
                VariableKind.GIMBAL_ANGLE,
                VariableKind.GIMBAL_ANGLE,
            ),
        )
        values = [self.thrust0, self.thrust1, self.gimbal0, self.gimbal1]
        names = ["thrust0", "thrust1", "gimbal0", "gimbal1"]
        step = 1.0e-7
        for block_index, block in enumerate(factor.jacobian_blocks):
            numerical = np.empty((8, 4), dtype=float)
            for coordinate in range(4):
                plus = [value.copy() for value in values]
                minus = [value.copy() for value in values]
                plus[block_index][coordinate] += step
                minus[block_index][coordinate] -= step
                plus_arguments = dict(zip(names, plus))
                minus_arguments = dict(zip(names, minus))
                numerical[:, coordinate] = (
                    self._evaluate(**plus_arguments).residual
                    - self._evaluate(**minus_arguments).residual
                ) / (2.0 * step)
            np.testing.assert_allclose(
                block.value, numerical, rtol=3.0e-8, atol=3.0e-9
            )

    def test_rate_and_command_saturation_active_sets_are_propagated(self):
        parameters = ActuatorParameters(
            thrust_time_constant=0.0,
            gimbal_time_constant=0.0,
            minimum_thrust=1.5,
            maximum_thrust=20.0,
            maximum_gimbal_angle=0.5,
            maximum_gimbal_rate=0.2,
        )
        command = _command(
            (0.0, 25.0, 8.0, 9.0),
            (2.0, -2.0, 0.4, -0.4),
        )
        factor = self._evaluate(command=command, parameters=parameters)
        self.assertTrue(factor.active_set["thrust_command_lower"][0])
        self.assertTrue(factor.active_set["thrust_command_upper"][1])
        self.assertTrue(np.any(factor.active_set["gimbal_rate_lower"]))
        self.assertTrue(np.any(factor.active_set["gimbal_rate_upper"]))

    def test_zoh_switch_segments_compose_state_jacobians(self):
        second_command = _command(
            (6.0, 7.0, 6.5, 8.5),
            (-0.1, 0.2, -0.15, 0.25),
        )

        def evaluate(thrust0, gimbal0):
            return evaluate_actuator_interval_factor(
                "bag-a",
                3,
                thrust0,
                self.thrust1,
                gimbal0,
                self.gimbal1,
                ((self.command, 0.011), (second_command, 0.014)),
                self.parameters,
                self.thrust_whitening,
                self.gimbal_whitening,
            )

        factor = evaluate(self.thrust0, self.gimbal0)
        self.assertIn(
            "segment_000/gimbal_rate_near_kink", factor.active_set
        )
        self.assertIn(
            "segment_001/gimbal_rate_near_kink", factor.active_set
        )
        step = 1.0e-7
        for block_index in (0, 2):
            block = factor.jacobian_blocks[block_index]
            numerical = np.empty((8, 4), dtype=float)
            for coordinate in range(4):
                thrust_plus = self.thrust0.copy()
                thrust_minus = self.thrust0.copy()
                gimbal_plus = self.gimbal0.copy()
                gimbal_minus = self.gimbal0.copy()
                if block_index == 0:
                    thrust_plus[coordinate] += step
                    thrust_minus[coordinate] -= step
                else:
                    gimbal_plus[coordinate] += step
                    gimbal_minus[coordinate] -= step
                numerical[:, coordinate] = (
                    evaluate(thrust_plus, gimbal_plus).residual
                    - evaluate(thrust_minus, gimbal_minus).residual
                ) / (2.0 * step)
            np.testing.assert_allclose(
                block.value, numerical, rtol=3.0e-8, atol=3.0e-9
            )

    def test_invalid_weighting_and_command_type_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "full rank"):
            evaluate_actuator_transition_factor(
                "bag-a",
                0,
                self.thrust0,
                self.thrust1,
                self.gimbal0,
                self.gimbal1,
                self.command,
                self.parameters,
                self.dt,
                np.zeros((4, 4)),
                self.gimbal_whitening,
            )
        with self.assertRaises(TypeError):
            evaluate_actuator_transition_factor(
                "bag-a",
                0,
                self.thrust0,
                self.thrust1,
                self.gimbal0,
                self.gimbal1,
                object(),
                self.parameters,
                self.dt,
                self.thrust_whitening,
                self.gimbal_whitening,
            )

    def test_actual_gimbal_observation_remains_asynchronous(self):
        alpha = 0.31
        observation = np.asarray((0.11, -0.22, 0.24, -0.28))
        factor = evaluate_gimbal_position_factor(
            "bag-a",
            3,
            alpha,
            self.gimbal0,
            self.gimbal1,
            observation,
            self.gimbal_whitening,
        )
        expected = self.gimbal_whitening @ (
            observation
            - (1.0 - alpha) * self.gimbal0
            - alpha * self.gimbal1
        )
        np.testing.assert_allclose(factor.residual, expected, atol=1.0e-15)
        np.testing.assert_allclose(
            factor.jacobian_blocks[0].value,
            -(1.0 - alpha) * self.gimbal_whitening,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            factor.jacobian_blocks[1].value,
            -alpha * self.gimbal_whitening,
            atol=1.0e-15,
        )


if __name__ == "__main__":
    unittest.main()
