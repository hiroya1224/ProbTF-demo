from dataclasses import replace
import unittest

import numpy as np

from grape_param_estim.dynamics import (
    advance_actuators,
    advance_actuators_with_jacobian,
)
from grape_param_estim.system import (
    ActuatorCommand,
    ActuatorParameters,
    ActuatorState,
)


def _command(thrust, gimbal):
    return ActuatorCommand(
        thrust=thrust,
        gimbal_angle=gimbal,
        virtual_force=np.zeros(8),
        desired_acceleration=np.zeros(6),
    )


def _state_vector(state):
    return np.concatenate((state.thrust, state.gimbal_angle))


def _reference_advance_actuators(state, command, parameters, time_step):
    """Original transition formula retained as an independent oracle."""

    def response(current, target, time_constant):
        if time_constant <= 0.0:
            return target.copy()
        fraction = 1.0 - np.exp(-time_step / time_constant)
        return current + fraction * (target - current)

    target_thrust = np.clip(
        command.thrust,
        parameters.minimum_thrust,
        parameters.maximum_thrust,
    )
    target_gimbal = np.clip(
        command.gimbal_angle,
        -parameters.maximum_gimbal_angle,
        parameters.maximum_gimbal_angle,
    )
    thrust = response(
        state.thrust, target_thrust, parameters.thrust_time_constant
    )
    unconstrained_gimbal = response(
        state.gimbal_angle,
        target_gimbal,
        parameters.gimbal_time_constant,
    )
    maximum_step = parameters.maximum_gimbal_rate * time_step
    gimbal = state.gimbal_angle + np.clip(
        unconstrained_gimbal - state.gimbal_angle,
        -maximum_step,
        maximum_step,
    )
    gimbal = np.clip(
        gimbal,
        -parameters.maximum_gimbal_angle,
        parameters.maximum_gimbal_angle,
    )
    return ActuatorState(thrust, gimbal)


class ActuatorTransitionJacobianTests(unittest.TestCase):
    def setUp(self):
        self.parameters = ActuatorParameters(
            thrust_time_constant=0.08,
            gimbal_time_constant=0.12,
            minimum_thrust=1.5,
            maximum_thrust=20.0,
            maximum_gimbal_angle=1.2,
            maximum_gimbal_rate=8.0,
        )
        self.state = ActuatorState(
            thrust=(5.0, 6.0, 7.0, 8.0),
            gimbal_angle=(0.1, -0.2, 0.25, -0.3),
        )
        self.command = _command(
            (7.0, 5.5, 8.5, 6.5),
            (0.14, -0.24, 0.20, -0.26),
        )
        self.dt = 0.025

    def test_forward_result_is_shared_with_existing_api(self):
        direct = advance_actuators(
            self.state, self.command, self.parameters, self.dt
        )
        evaluation = advance_actuators_with_jacobian(
            self.state, self.command, self.parameters, self.dt
        )
        np.testing.assert_array_equal(evaluation.next_state.thrust, direct.thrust)
        np.testing.assert_array_equal(
            evaluation.next_state.gimbal_angle, direct.gimbal_angle
        )
        for mask in evaluation.active_set.values():
            self.assertEqual(mask.shape, (4,))
            self.assertEqual(mask.dtype, np.bool_)
            self.assertFalse(mask.flags.writeable)

    def test_forward_formula_is_bitwise_unchanged_across_branches(self):
        generator = np.random.RandomState(82517)
        for case in range(20):
            parameters = replace(
                self.parameters,
                thrust_time_constant=(
                    0.0 if case % 3 == 0 else generator.uniform(0.01, 0.2)
                ),
                gimbal_time_constant=(
                    0.0 if case % 4 == 0 else generator.uniform(0.01, 0.2)
                ),
                maximum_gimbal_rate=generator.uniform(0.1, 12.0),
            )
            state = ActuatorState(
                generator.uniform(-2.0, 25.0, 4),
                generator.uniform(-1.4, 1.4, 4),
            )
            command = _command(
                generator.uniform(-5.0, 30.0, 4),
                generator.uniform(-2.0, 2.0, 4),
            )
            time_step = generator.uniform(0.001, 0.08)
            expected = _reference_advance_actuators(
                state, command, parameters, time_step
            )
            actual = advance_actuators(
                state, command, parameters, time_step
            )
            with self.subTest(case=case):
                np.testing.assert_array_equal(actual.thrust, expected.thrust)
                np.testing.assert_array_equal(
                    actual.gimbal_angle, expected.gimbal_angle
                )

    def test_unsaturated_jacobians_match_central_difference(self):
        evaluation = advance_actuators_with_jacobian(
            self.state, self.command, self.parameters, self.dt
        )
        analytic_previous = np.block(
            [
                [evaluation.jacobian.thrust_previous, np.zeros((4, 4))],
                [np.zeros((4, 4)), evaluation.jacobian.gimbal_previous],
            ]
        )
        analytic_command = np.block(
            [
                [evaluation.jacobian.thrust_command, np.zeros((4, 4))],
                [np.zeros((4, 4)), evaluation.jacobian.gimbal_command],
            ]
        )
        step = 1.0e-7
        numerical_previous = np.empty((8, 8), dtype=float)
        previous = _state_vector(self.state)
        command = _state_vector(
            ActuatorState(self.command.thrust, self.command.gimbal_angle)
        )
        for coordinate in range(8):
            direction = np.zeros(8, dtype=float)
            direction[coordinate] = step
            plus_state = ActuatorState(
                previous[:4] + direction[:4],
                previous[4:] + direction[4:],
            )
            minus_state = ActuatorState(
                previous[:4] - direction[:4],
                previous[4:] - direction[4:],
            )
            numerical_previous[:, coordinate] = (
                _state_vector(
                    advance_actuators(
                        plus_state, self.command, self.parameters, self.dt
                    )
                )
                - _state_vector(
                    advance_actuators(
                        minus_state, self.command, self.parameters, self.dt
                    )
                )
            ) / (2.0 * step)

            plus_command = _command(
                command[:4] + direction[:4],
                command[4:] + direction[4:],
            )
            minus_command = _command(
                command[:4] - direction[:4],
                command[4:] - direction[4:],
            )
            numerical_command_column = (
                _state_vector(
                    advance_actuators(
                        self.state, plus_command, self.parameters, self.dt
                    )
                )
                - _state_vector(
                    advance_actuators(
                        self.state, minus_command, self.parameters, self.dt
                    )
                )
            ) / (2.0 * step)
            numerical_command = (
                numerical_command_column
                if coordinate == 0
                else np.column_stack(
                    (numerical_command, numerical_command_column)
                )
            )
        np.testing.assert_allclose(
            analytic_previous, numerical_previous, rtol=2.0e-8, atol=2.0e-9
        )
        np.testing.assert_allclose(
            analytic_command, numerical_command, rtol=2.0e-8, atol=2.0e-9
        )

        plus = _state_vector(
            advance_actuators(
                self.state,
                self.command,
                self.parameters,
                self.dt + step,
            )
        )
        minus = _state_vector(
            advance_actuators(
                self.state,
                self.command,
                self.parameters,
                self.dt - step,
            )
        )
        numerical_time_step = (plus - minus) / (2.0 * step)
        analytic_time_step = np.concatenate(
            (
                evaluation.jacobian.thrust_time_step,
                evaluation.jacobian.gimbal_time_step,
            )
        )
        np.testing.assert_allclose(
            analytic_time_step,
            numerical_time_step,
            rtol=2.0e-8,
            atol=2.0e-9,
        )

    def test_saturated_branches_zero_input_derivatives(self):
        parameters = replace(
            self.parameters,
            maximum_gimbal_rate=0.2,
            maximum_gimbal_angle=0.5,
        )
        state = ActuatorState(
            thrust=(5.0, 6.0, 7.0, 8.0),
            gimbal_angle=(0.499, -0.499, 0.0, 0.0),
        )
        command = _command(
            (0.0, 25.0, 8.0, 9.0),
            (2.0, -2.0, 0.4, -0.4),
        )
        evaluation = advance_actuators_with_jacobian(
            state, command, parameters, 0.02
        )
        self.assertTrue(evaluation.active_set["thrust_command_lower"][0])
        self.assertTrue(evaluation.active_set["thrust_command_upper"][1])
        self.assertEqual(evaluation.jacobian.thrust_command[0, 0], 0.0)
        self.assertEqual(evaluation.jacobian.thrust_command[1, 1], 0.0)
        self.assertTrue(evaluation.active_set["gimbal_rate_lower"][3])
        self.assertTrue(evaluation.active_set["gimbal_rate_upper"][2])
        np.testing.assert_array_equal(
            np.diag(evaluation.jacobian.gimbal_command), np.zeros(4)
        )

    def test_exact_boundaries_use_saturated_convention_and_warn(self):
        parameters = replace(
            self.parameters,
            thrust_time_constant=0.0,
            gimbal_time_constant=0.0,
            maximum_gimbal_rate=10.0,
        )
        command = _command(
            (parameters.minimum_thrust, 6.0, 7.0, 8.0),
            (0.0, -0.2, 0.25, -0.3),
        )
        evaluation = advance_actuators_with_jacobian(
            self.state, command, parameters, self.dt
        )
        self.assertTrue(evaluation.active_set["thrust_command_lower"][0])
        self.assertTrue(evaluation.active_set["thrust_command_near_kink"][0])
        self.assertEqual(evaluation.jacobian.thrust_command[0, 0], 0.0)

    def test_zero_time_constants_and_invalid_inputs_are_handled(self):
        parameters = replace(
            self.parameters,
            thrust_time_constant=0.0,
            gimbal_time_constant=0.0,
        )
        evaluation = advance_actuators_with_jacobian(
            self.state, self.command, parameters, self.dt
        )
        np.testing.assert_array_equal(
            evaluation.jacobian.thrust_previous, np.zeros((4, 4))
        )
        np.testing.assert_array_equal(
            evaluation.jacobian.thrust_time_step, np.zeros(4)
        )
        with self.assertRaisesRegex(ValueError, "time step"):
            advance_actuators_with_jacobian(
                self.state, self.command, parameters, 0.0
            )


if __name__ == "__main__":
    unittest.main()
