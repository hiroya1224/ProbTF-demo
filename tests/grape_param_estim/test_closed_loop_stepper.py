import unittest

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.closed_loop_stepper import (
    ClosedLoopStepper,
    ClosedLoopStepperState,
)
from grape_param_estim.controller import (
    ControllerConfig,
    GrapeController,
    initial_controller_state,
)
from grape_param_estim.dynamics import FullSixDofPlant, simulate_closed_loop
from grape_param_estim.geometry import (
    euler_xyz_to_matrix,
    matrix_to_quaternion,
)
from grape_param_estim.synthetic import full_six_dof_reference
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    ControllerState,
    GrapeGeometry,
    ReferenceState,
    RigidBodyState,
    VehicleParameters,
)


TRAJECTORY_FIELDS = (
    "times",
    "position",
    "orientation_xyzw",
    "linear_velocity",
    "angular_velocity",
    "controller_integral",
    "commanded_thrust",
    "commanded_gimbal_angle",
    "actuator_thrust",
    "actuator_gimbal_angle",
    "body_wrench",
)


class ClosedLoopStepperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times = np.asarray((0.0, 0.04, 0.11, 0.15, 0.27))
        cls.references = full_six_dof_reference(cls.times)
        cls.parameters = VehicleParameters.nominal()
        cls.geometry = GrapeGeometry.grape()
        cls.configuration = ControllerConfig.grape()
        first = cls.references[0]
        cls.initial_rigid_body = RigidBodyState(
            first.position,
            matrix_to_quaternion(euler_xyz_to_matrix(first.rpy)),
            first.linear_velocity,
            first.angular_velocity,
        )
        cls.initial_actuator_snapshot = ActuatorState(
            thrust=np.asarray((3.2, 5.1, 7.4, 9.0)),
            gimbal_angle=np.asarray((0.18, -0.13, 0.09, -0.05)),
        )
        base = np.asarray((0.3, -0.2, 0.1, 0.02, -0.015, 0.01))
        cls.residual_cases = (
            ("none", None),
            ("zero", np.zeros((cls.times.size - 1, 6))),
            ("fixed", np.tile(base, (cls.times.size - 1, 1))),
            (
                "varying",
                np.arange(1, cls.times.size)[:, None] * base[None, :],
            ),
        )

    def _controller(self):
        return GrapeController(
            self.configuration,
            self.parameters,
            self.geometry,
            articulated_model=GrapeArticulatedModel(),
        )

    def _actuator_parameters(self, delay):
        return ActuatorParameters(
            thrust_time_constant=0.06,
            gimbal_time_constant=0.08,
            delay=delay,
        )

    def _new_stepper(self, delay, initial_actuator):
        return ClosedLoopStepper(
            controller=self._controller(),
            plant=FullSixDofPlant(self.parameters, self.geometry),
            actuator_parameters=self._actuator_parameters(delay),
            initial_state=ClosedLoopStepperState(
                time=self.times[0],
                rigid_body_state=self.initial_rigid_body,
                controller_state=initial_controller_state(
                    self.configuration, trim_hover=True
                ),
                actuator_state=initial_actuator,
            ),
        )

    def _run_stepper(self, delay, initial_actuator, residual_path):
        stepper = self._new_stepper(delay, initial_actuator)
        samples = []
        states = []
        for index in range(self.times.size - 1):
            residual = (
                None if residual_path is None else residual_path[index]
            )
            samples.append(
                stepper.advance_interval(
                    self.times[index + 1],
                    self.references[index],
                    residual,
                )
            )
            states.append(stepper.state)
        final_residual = None if residual_path is None else residual_path[-1]
        samples.append(
            stepper.terminal_sample(
                self.references[-1],
                self.times[-1] - self.times[-2],
                final_residual,
            )
        )
        arrays = {
            "times": np.asarray([sample.time for sample in samples]),
            "position": np.asarray(
                [sample.rigid_body_state.position for sample in samples]
            ),
            "orientation_xyzw": np.asarray(
                [
                    sample.rigid_body_state.orientation_xyzw
                    for sample in samples
                ]
            ),
            "linear_velocity": np.asarray(
                [
                    sample.rigid_body_state.linear_velocity
                    for sample in samples
                ]
            ),
            "angular_velocity": np.asarray(
                [
                    sample.rigid_body_state.angular_velocity
                    for sample in samples
                ]
            ),
            "controller_integral": np.asarray(
                [
                    sample.controller_state.integral_error
                    for sample in samples
                ]
            ),
            "commanded_thrust": np.asarray(
                [sample.command.thrust for sample in samples]
            ),
            "commanded_gimbal_angle": np.asarray(
                [sample.command.gimbal_angle for sample in samples]
            ),
            "actuator_thrust": np.asarray(
                [sample.actuator_state.thrust for sample in samples]
            ),
            "actuator_gimbal_angle": np.asarray(
                [sample.actuator_state.gimbal_angle for sample in samples]
            ),
            "body_wrench": np.asarray(
                [sample.body_wrench for sample in samples]
            ),
        }
        return stepper, tuple(samples), tuple(states), arrays

    def _legacy_rollout(self, delay, initial_actuator, residual_path):
        return simulate_closed_loop(
            times=self.times,
            references=self.references,
            initial_state=self.initial_rigid_body,
            initial_controller_state=initial_controller_state(
                self.configuration, trim_hover=True
            ),
            controller=self._controller(),
            plant=FullSixDofPlant(self.parameters, self.geometry),
            actuator_parameters=self._actuator_parameters(delay),
            initial_actuator_state=initial_actuator,
            interval_residual_wrench=residual_path,
        )

    def test_twenty_four_rollouts_are_bit_exact(self):
        for delay in (0.0, 0.017, 0.095):
            for initial_name, initial_actuator in (
                ("command", None),
                ("snapshot", self.initial_actuator_snapshot),
            ):
                for residual_name, residual_path in self.residual_cases:
                    with self.subTest(
                        delay=delay,
                        initial=initial_name,
                        residual=residual_name,
                    ):
                        expected = self._legacy_rollout(
                            delay, initial_actuator, residual_path
                        )
                        _stepper, _samples, _states, actual = (
                            self._run_stepper(
                                delay, initial_actuator, residual_path
                            )
                        )
                        for field in TRAJECTORY_FIELDS:
                            np.testing.assert_array_equal(
                                actual[field], getattr(expected, field)
                            )

    def test_every_interval_state_and_complete_history_match_the_rollout(self):
        delay = 0.095
        residual_path = self.residual_cases[-1][1]
        expected = self._legacy_rollout(
            delay, self.initial_actuator_snapshot, residual_path
        )
        stepper, samples, states, _actual = self._run_stepper(
            delay, self.initial_actuator_snapshot, residual_path
        )
        for index, state in enumerate(states, start=1):
            self.assertEqual(state.time, self.times[index])
            np.testing.assert_array_equal(
                state.rigid_body_state.position, expected.position[index]
            )
            np.testing.assert_array_equal(
                state.rigid_body_state.orientation_xyzw,
                expected.orientation_xyzw[index],
            )
            np.testing.assert_array_equal(
                state.rigid_body_state.linear_velocity,
                expected.linear_velocity[index],
            )
            np.testing.assert_array_equal(
                state.rigid_body_state.angular_velocity,
                expected.angular_velocity[index],
            )
            np.testing.assert_array_equal(
                state.controller_state.integral_error,
                expected.controller_integral[index],
            )
            np.testing.assert_array_equal(
                state.actuator_state.thrust,
                expected.actuator_thrust[index],
            )
            np.testing.assert_array_equal(
                state.actuator_state.gimbal_angle,
                expected.actuator_gimbal_angle[index],
            )

        np.testing.assert_array_equal(
            stepper.command_issue_times, self.times
        )
        first = stepper.delayed_command_at(self.times[0] - 1.0)
        np.testing.assert_array_equal(first.thrust, samples[0].command.thrust)
        after_third_switch = stepper.delayed_command_at(
            self.times[2] + delay + 1.0e-9
        )
        np.testing.assert_array_equal(
            after_third_switch.thrust, samples[2].command.thrust
        )
        external_times = stepper.command_issue_times
        external_times[0] = 123.0
        self.assertEqual(stepper.command_issue_times[0], self.times[0])

    def test_analysis_replacement_preserves_time_and_command_history(self):
        stepper = self._new_stepper(0.095, self.initial_actuator_snapshot)
        first = stepper.advance_interval(
            self.times[1], self.references[0]
        )
        stepper.advance_interval(self.times[2], self.references[1])
        issue_times = stepper.command_issue_times.copy()
        delayed_before = stepper.delayed_command_at(self.times[2])
        old_time = stepper.state.time
        old = stepper.state
        replacement_rigid = RigidBodyState(
            old.rigid_body_state.position + np.asarray((0.2, -0.1, 0.05)),
            old.rigid_body_state.orientation_xyzw,
            old.rigid_body_state.linear_velocity
            + np.asarray((0.1, 0.0, -0.1)),
            old.rigid_body_state.angular_velocity
            + np.asarray((0.02, -0.01, 0.03)),
        )
        replacement_controller = ControllerState(
            old.controller_state.integral_error
            + np.asarray((0.1, -0.1, 0.2, 0.01, -0.02, 0.03)),
            old.controller_state.roll_pitch_integration_active,
        )
        replacement_actuator = ActuatorState(
            old.actuator_state.thrust + np.asarray((0.1, 0.2, 0.3, 0.4)),
            old.actuator_state.gimbal_angle
            + np.asarray((0.01, -0.01, 0.02, -0.02)),
        )

        replaced = stepper.replace_dynamic_state(
            rigid_body_state=replacement_rigid,
            controller_state=replacement_controller,
            actuator_state=replacement_actuator,
        )
        self.assertIs(replaced, stepper.state)
        self.assertEqual(stepper.state.time, old_time)
        np.testing.assert_array_equal(stepper.command_issue_times, issue_times)
        delayed_after = stepper.delayed_command_at(self.times[2])
        np.testing.assert_array_equal(
            delayed_after.thrust, delayed_before.thrust
        )
        np.testing.assert_array_equal(
            delayed_after.gimbal_angle, delayed_before.gimbal_angle
        )
        np.testing.assert_array_equal(
            delayed_after.thrust, first.command.thrust
        )

        sample = stepper.advance_interval(
            self.times[3], self.references[2]
        )
        np.testing.assert_array_equal(
            sample.rigid_body_state.position, replacement_rigid.position
        )
        np.testing.assert_array_equal(
            sample.controller_state.integral_error,
            replacement_controller.integral_error,
        )
        np.testing.assert_array_equal(
            sample.actuator_state.thrust, replacement_actuator.thrust
        )
        np.testing.assert_array_equal(
            stepper.command_issue_times,
            (self.times[0], self.times[1], self.times[2]),
        )

    def test_static_model_replacement_preserves_state_and_command_history(self):
        stepper = self._new_stepper(0.095, self.initial_actuator_snapshot)
        first = stepper.advance_interval(
            self.times[1], self.references[0]
        )
        state_before = stepper.state
        issue_times = stepper.command_issue_times.copy()

        changed_parameters = VehicleParameters(
            mass=1.1 * self.parameters.mass,
            inertia=self.parameters.inertia,
            cog_offset=self.parameters.cog_offset,
            force_effectiveness=self.parameters.force_effectiveness,
            torque_effectiveness=self.parameters.torque_effectiveness,
            linear_drag=self.parameters.linear_drag,
            angular_drag=self.parameters.angular_drag,
        )
        changed_actuators = self._actuator_parameters(0.017)
        stepper.replace_static_model(
            controller=GrapeController(
                self.configuration,
                changed_parameters,
                self.geometry,
                articulated_model=GrapeArticulatedModel(),
            ),
            plant=FullSixDofPlant(changed_parameters, self.geometry),
            actuator_parameters=changed_actuators,
        )

        self.assertIs(stepper.state, state_before)
        np.testing.assert_array_equal(
            stepper.command_issue_times, issue_times
        )
        delayed = stepper.delayed_command_at(self.times[1] + 0.017)
        np.testing.assert_array_equal(delayed.thrust, first.command.thrust)
        sample = stepper.advance_interval(
            self.times[2], self.references[1]
        )
        np.testing.assert_array_equal(
            sample.rigid_body_state.position,
            state_before.rigid_body_state.position,
        )

    def test_static_model_replacement_validates_before_mutation(self):
        stepper = self._new_stepper(0.095, self.initial_actuator_snapshot)
        valid_controller = self._controller()
        valid_plant = FullSixDofPlant(self.parameters, self.geometry)
        valid_actuators = self._actuator_parameters(0.017)
        for field in ("controller", "plant", "actuator_parameters"):
            arguments = {
                "controller": valid_controller,
                "plant": valid_plant,
                "actuator_parameters": valid_actuators,
            }
            arguments[field] = object()
            with self.subTest(field=field), self.assertRaises(TypeError):
                stepper.replace_static_model(**arguments)
        self.assertEqual(stepper.command_issue_times.size, 0)

    def test_terminal_and_input_validation_block_continuation(self):
        stepper = self._new_stepper(0.017, None)
        for invalid_end in (self.times[0], -1.0, float("nan"), float("inf")):
            with self.subTest(end=invalid_end), self.assertRaises(ValueError):
                stepper.advance_interval(invalid_end, self.references[0])
        with self.assertRaises(TypeError):
            stepper.advance_interval(self.times[1], object())
        for invalid_residual in (
            np.zeros(5),
            np.zeros(7),
            np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, np.nan)),
        ):
            with self.subTest(
                residual=invalid_residual.shape
            ), self.assertRaises(ValueError):
                stepper.advance_interval(
                    self.times[1], self.references[0], invalid_residual
                )
        self.assertEqual(stepper.command_issue_times.size, 0)

        for invalid_step in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(step=invalid_step), self.assertRaises(
                ValueError
            ):
                stepper.terminal_sample(self.references[0], invalid_step)
        terminal = stepper.terminal_sample(self.references[0], 0.04)
        self.assertTrue(stepper.is_terminal)
        self.assertIsNotNone(stepper.state.actuator_state)
        np.testing.assert_array_equal(
            stepper.state.actuator_state.thrust,
            terminal.actuator_state.thrust,
        )
        with self.assertRaises(RuntimeError):
            stepper.terminal_sample(self.references[0], 0.04)
        with self.assertRaises(RuntimeError):
            stepper.advance_interval(self.times[1], self.references[0])
        with self.assertRaises(RuntimeError):
            stepper.replace_dynamic_state(
                rigid_body_state=self.initial_rigid_body,
                controller_state=initial_controller_state(
                    self.configuration, trim_hover=True
                ),
                actuator_state=self.initial_actuator_snapshot,
            )
        with self.assertRaises(RuntimeError):
            stepper.replace_static_model(
                controller=self._controller(),
                plant=FullSixDofPlant(self.parameters, self.geometry),
                actuator_parameters=self._actuator_parameters(0.0),
            )

    def test_state_and_constructor_types_are_validated(self):
        controller_state = initial_controller_state(
            self.configuration, trim_hover=True
        )
        with self.assertRaises(ValueError):
            ClosedLoopStepperState(
                float("nan"), self.initial_rigid_body, controller_state, None
            )
        for field, value in (
            ("rigid", object()),
            ("controller", object()),
            ("actuator", object()),
        ):
            arguments = {
                "time": 0.0,
                "rigid_body_state": self.initial_rigid_body,
                "controller_state": controller_state,
                "actuator_state": None,
            }
            if field == "rigid":
                arguments["rigid_body_state"] = value
            elif field == "controller":
                arguments["controller_state"] = value
            else:
                arguments["actuator_state"] = value
            with self.subTest(field=field), self.assertRaises(TypeError):
                ClosedLoopStepperState(**arguments)

        valid_state = ClosedLoopStepperState(
            0.0, self.initial_rigid_body, controller_state, None
        )
        with self.assertRaises(TypeError):
            ClosedLoopStepper(
                object(),
                FullSixDofPlant(self.parameters, self.geometry),
                self._actuator_parameters(0.0),
                valid_state,
            )
        with self.assertRaises(TypeError):
            ClosedLoopStepper(
                self._controller(),
                object(),
                self._actuator_parameters(0.0),
                valid_state,
            )
        with self.assertRaises(TypeError):
            ClosedLoopStepper(
                self._controller(),
                FullSixDofPlant(self.parameters, self.geometry),
                object(),
                valid_state,
            )
        with self.assertRaises(TypeError):
            ClosedLoopStepper(
                self._controller(),
                FullSixDofPlant(self.parameters, self.geometry),
                self._actuator_parameters(0.0),
                object(),
            )


if __name__ == "__main__":
    unittest.main()
