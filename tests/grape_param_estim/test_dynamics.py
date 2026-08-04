from dataclasses import replace
import unittest

import numpy as np

from grape_param_estim.controller import (
    ControllerConfig,
    GrapeController,
    initial_controller_state,
)
from grape_param_estim.dynamics import (
    ActuatorWrenchJacobian,
    FullSixDofPlant,
    advance_actuators,
    actuator_wrench,
    actuator_wrench_with_jacobian,
    simulate_closed_loop,
)
from grape_param_estim.geometry import (
    euler_xyz_to_matrix,
    matrix_to_quaternion,
)
from grape_param_estim.synthetic import (
    full_six_dof_reference,
    run_perfect_model_experiment,
)
from grape_param_estim.system import (
    ActuatorCommand,
    ActuatorParameters,
    ActuatorState,
    GrapeGeometry,
    RigidBodyState,
    VehicleParameters,
)


def _reference_rotate_z(vector, angle):
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.asarray(
        (
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
            vector[2],
        ),
        dtype=float,
    )


def _reference_actuator_wrench(actuator_state, parameters, geometry):
    """Original forward formula retained as an independent test oracle."""

    wrench = np.zeros(6, dtype=float)
    thrust_origins = geometry.thrust_origins(
        actuator_state.gimbal_angle
    )
    for rotor in range(4):
        angle = actuator_state.gimbal_angle[rotor]
        effective_thrust = (
            actuator_state.thrust[rotor]
            * parameters.force_effectiveness[rotor]
        )
        local_force = np.asarray(
            (
                0.0,
                -effective_thrust * np.sin(angle),
                effective_thrust * np.cos(angle),
            ),
            dtype=float,
        )
        force = _reference_rotate_z(local_force, geometry.arm_yaws[rotor])
        origin = thrust_origins[rotor] - parameters.cog_offset
        reaction_torque = (
            parameters.torque_effectiveness[rotor]
            * geometry.rotor_directions[rotor]
            * geometry.moment_force_rate
            * force
        )
        wrench[:3] += force
        wrench[3:] += np.cross(origin, force) + reaction_torque
    return wrench


def _central_difference(evaluate, values, step=1.0e-6):
    point = np.asarray(values, dtype=float)
    reference = np.asarray(evaluate(point), dtype=float)
    result = np.empty((reference.size, point.size), dtype=float)
    for coordinate in range(point.size):
        direction = np.zeros_like(point)
        direction[coordinate] = step
        result[:, coordinate] = (
            evaluate(point + direction) - evaluate(point - direction)
        ) / (2.0 * step)
    return result


class DynamicsTests(unittest.TestCase):
    def test_actuator_wrench_forward_formula_is_unchanged(self):
        base_geometry = GrapeGeometry.grape()
        parameters = VehicleParameters(
            mass=2.8,
            inertia=VehicleParameters.nominal().inertia,
            cog_offset=(0.017, -0.011, 0.024),
            force_effectiveness=(0.81, 1.13, 0.94, 1.21),
            torque_effectiveness=(1.19, 0.86, 1.08, 0.91),
            linear_drag=(0.2, 0.1, 0.3),
            angular_drag=(0.02, 0.03, 0.04),
        )
        actuator_state = ActuatorState(
            thrust=(4.7, 8.2, 6.4, 9.1),
            gimbal_angle=(0.31, -0.27, 0.18, -0.42),
        )
        for geometry in (
            replace(
                base_geometry,
                thrust_offset=0.0,
                moment_force_rate=-0.027,
            ),
            replace(
                base_geometry,
                thrust_offset=0.083,
                moment_force_rate=0.019,
            ),
        ):
            with self.subTest(
                thrust_offset=geometry.thrust_offset,
                moment_force_rate=geometry.moment_force_rate,
            ):
                expected = _reference_actuator_wrench(
                    actuator_state, parameters, geometry
                )
                direct = actuator_wrench(
                    actuator_state, parameters, geometry
                )
                linearized, _ = actuator_wrench_with_jacobian(
                    actuator_state, parameters, geometry
                )
                np.testing.assert_array_equal(direct, expected)
                np.testing.assert_array_equal(linearized, direct)

    def test_actuator_wrench_jacobian_matches_central_differences(self):
        generator = np.random.RandomState(20260804)
        nominal = VehicleParameters.nominal()
        base_geometry = GrapeGeometry.grape()
        for case in range(8):
            parameters = VehicleParameters(
                mass=generator.uniform(1.8, 3.4),
                inertia=nominal.inertia,
                cog_offset=generator.uniform(-0.04, 0.04, 3),
                force_effectiveness=generator.uniform(0.65, 1.35, 4),
                torque_effectiveness=generator.uniform(0.7, 1.3, 4),
                linear_drag=generator.uniform(0.0, 0.2, 3),
                angular_drag=generator.uniform(0.0, 0.03, 3),
            )
            actuator_state = ActuatorState(
                thrust=generator.uniform(1.8, 16.0, 4),
                gimbal_angle=generator.uniform(-0.65, 0.65, 4),
            )
            geometry = replace(
                base_geometry,
                thrust_offset=(
                    0.0 if case % 2 == 0 else generator.uniform(0.02, 0.1)
                ),
                moment_force_rate=generator.uniform(-0.04, -0.008),
            )
            _, analytic = actuator_wrench_with_jacobian(
                actuator_state, parameters, geometry
            )
            numerical = {
                "actual_thrust": _central_difference(
                    lambda value: actuator_wrench(
                        ActuatorState(value, actuator_state.gimbal_angle),
                        parameters,
                        geometry,
                    ),
                    actuator_state.thrust,
                ),
                "actual_gimbal_angle": _central_difference(
                    lambda value: actuator_wrench(
                        ActuatorState(actuator_state.thrust, value),
                        parameters,
                        geometry,
                    ),
                    actuator_state.gimbal_angle,
                ),
                "cog_offset": _central_difference(
                    lambda value: actuator_wrench(
                        actuator_state,
                        replace(parameters, cog_offset=value),
                        geometry,
                    ),
                    parameters.cog_offset,
                ),
                "force_effectiveness": _central_difference(
                    lambda value: actuator_wrench(
                        actuator_state,
                        replace(parameters, force_effectiveness=value),
                        geometry,
                    ),
                    parameters.force_effectiveness,
                ),
                "torque_effectiveness": _central_difference(
                    lambda value: actuator_wrench(
                        actuator_state,
                        replace(parameters, torque_effectiveness=value),
                        geometry,
                    ),
                    parameters.torque_effectiveness,
                ),
            }
            for name, expected in numerical.items():
                with self.subTest(case=case, block=name):
                    np.testing.assert_allclose(
                        getattr(analytic, name),
                        expected,
                        rtol=3.0e-7,
                        atol=3.0e-8,
                    )

    def test_articulated_origin_and_reaction_torque_derivatives(self):
        base_geometry = GrapeGeometry.grape()
        parameters = replace(
            VehicleParameters.nominal(),
            cog_offset=np.asarray((0.02, -0.015, 0.01)),
            force_effectiveness=np.asarray((0.9, 1.1, 0.8, 1.2)),
            torque_effectiveness=np.asarray((1.2, 0.85, 1.1, 0.95)),
        )
        actuator_state = ActuatorState(
            thrust=(7.2, 5.9, 8.1, 6.7),
            gimbal_angle=(0.42, -0.31, 0.24, -0.37),
        )
        zero_offset = replace(base_geometry, thrust_offset=0.0)
        articulated = replace(base_geometry, thrust_offset=0.091)
        _, zero_offset_jacobian = actuator_wrench_with_jacobian(
            actuator_state, parameters, zero_offset
        )
        _, articulated_jacobian = actuator_wrench_with_jacobian(
            actuator_state, parameters, articulated
        )
        self.assertGreater(
            np.linalg.norm(
                articulated_jacobian.actual_gimbal_angle[3:]
                - zero_offset_jacobian.actual_gimbal_angle[3:]
            ),
            1.0e-3,
        )

        no_reaction = replace(articulated, moment_force_rate=0.0)
        _, no_reaction_jacobian = actuator_wrench_with_jacobian(
            actuator_state, parameters, no_reaction
        )
        np.testing.assert_array_equal(
            no_reaction_jacobian.torque_effectiveness,
            np.zeros((6, 4)),
        )
        np.testing.assert_array_equal(
            articulated_jacobian.torque_effectiveness[:3],
            np.zeros((3, 4)),
        )
        self.assertGreater(
            np.linalg.norm(
                articulated_jacobian.torque_effectiveness[3:]
            ),
            1.0e-3,
        )

    def test_actuator_wrench_has_no_direct_inertia_dependency(self):
        parameters = VehicleParameters.nominal()
        changed_inertia = replace(
            parameters,
            inertia=2.75 * parameters.inertia,
        )
        actuator_state = ActuatorState(
            thrust=(5.2, 6.3, 7.1, 4.8),
            gimbal_angle=(0.17, -0.23, 0.29, -0.11),
        )
        geometry = GrapeGeometry.grape()
        baseline_wrench, baseline_jacobian = (
            actuator_wrench_with_jacobian(
                actuator_state, parameters, geometry
            )
        )
        changed_wrench, changed_jacobian = actuator_wrench_with_jacobian(
            actuator_state, changed_inertia, geometry
        )
        np.testing.assert_array_equal(changed_wrench, baseline_wrench)
        self.assertFalse(hasattr(baseline_jacobian, "inertia"))
        for name in (
            "actual_thrust",
            "actual_gimbal_angle",
            "cog_offset",
            "force_effectiveness",
            "torque_effectiveness",
        ):
            np.testing.assert_array_equal(
                getattr(changed_jacobian, name),
                getattr(baseline_jacobian, name),
            )

    def test_actuator_wrench_jacobian_shapes_and_validation(self):
        actuator_state = ActuatorState(
            thrust=(5.0, 6.0, 7.0, 8.0),
            gimbal_angle=(0.1, -0.2, 0.3, -0.4),
        )
        parameters = VehicleParameters.nominal()
        geometry = GrapeGeometry.grape()
        wrench, jacobian = actuator_wrench_with_jacobian(
            actuator_state, parameters, geometry
        )
        self.assertIsInstance(jacobian, ActuatorWrenchJacobian)
        self.assertEqual(wrench.shape, (6,))
        self.assertTrue(np.all(np.isfinite(wrench)))
        expected_shapes = {
            "actual_thrust": (6, 4),
            "actual_gimbal_angle": (6, 4),
            "cog_offset": (6, 3),
            "force_effectiveness": (6, 4),
            "torque_effectiveness": (6, 4),
        }
        for name, shape in expected_shapes.items():
            value = getattr(jacobian, name)
            self.assertEqual(value.shape, shape)
            self.assertTrue(np.all(np.isfinite(value)))

        with self.assertRaisesRegex(ValueError, "actual_thrust"):
            replace(jacobian, actual_thrust=np.zeros((6, 3)))
        invalid_block = jacobian.actual_gimbal_angle.copy()
        invalid_block[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "actual_gimbal_angle"):
            replace(jacobian, actual_gimbal_angle=invalid_block)

        invalid_state = ActuatorState(
            actuator_state.thrust, actuator_state.gimbal_angle
        )
        invalid_state.thrust[0] = np.nan
        with self.assertRaisesRegex(ValueError, "actual thrust"):
            actuator_wrench(invalid_state, parameters, geometry)
        with self.assertRaisesRegex(ValueError, "actual thrust"):
            actuator_wrench_with_jacobian(
                invalid_state, parameters, geometry
            )
        with self.assertRaises(TypeError):
            actuator_wrench(object(), parameters, geometry)
        with self.assertRaises(TypeError):
            actuator_wrench(actuator_state, object(), geometry)
        with self.assertRaises(TypeError):
            actuator_wrench(actuator_state, parameters, object())

    def test_perfect_model_closed_loops_are_identical_and_continuous(self):
        experiment = run_perfect_model_experiment(2.0, 0.02)
        np.testing.assert_array_equal(
            experiment.nominal.position, experiment.truth.position
        )
        np.testing.assert_array_equal(
            experiment.nominal.orientation_xyzw,
            experiment.truth.orientation_xyzw,
        )
        np.testing.assert_array_equal(
            experiment.nominal.commanded_thrust,
            experiment.truth.commanded_thrust,
        )
        np.testing.assert_allclose(experiment.correction_translation, 0.0)
        np.testing.assert_allclose(
            experiment.correction_rotation_vector, 0.0
        )
        expected_initial = initial_controller_state(
            ControllerConfig.grape(), trim_hover=True
        )
        np.testing.assert_array_equal(
            experiment.truth.controller_integral[0],
            expected_initial.integral_error,
        )
        self.assertFalse(
            np.array_equal(
                experiment.truth.controller_integral[0],
                experiment.truth.controller_integral[-1],
            )
        )
        # There is one initial condition and no segment/reset identifier in
        # the trajectory contract.  Every following sample is integrated.
        displacement = np.linalg.norm(
            np.diff(experiment.truth.position, axis=0), axis=1
        )
        self.assertTrue(np.all(np.isfinite(displacement)))
        self.assertGreater(np.count_nonzero(displacement), 0)

    def test_parameter_mismatch_changes_future_feedback_commands(self):
        times = np.linspace(0.0, 2.0, 101)
        references = full_six_dof_reference(times)
        nominal = VehicleParameters.nominal()
        geometry = GrapeGeometry.grape()
        configuration = ControllerConfig.grape()
        initial_reference = references[0]
        initial_state = RigidBodyState(
            initial_reference.position,
            matrix_to_quaternion(
                euler_xyz_to_matrix(initial_reference.rpy)
            ),
            initial_reference.linear_velocity,
            initial_reference.angular_velocity,
        )

        def rollout(parameters):
            return simulate_closed_loop(
                times,
                references,
                initial_state,
                initial_controller_state(
                    configuration, trim_hover=True
                ),
                GrapeController(configuration, nominal, geometry),
                FullSixDofPlant(parameters, geometry),
                ActuatorParameters(),
            )

        baseline = rollout(nominal)
        heavy = VehicleParameters(
            mass=1.20 * nominal.mass,
            inertia=nominal.inertia,
            cog_offset=np.zeros(3),
            force_effectiveness=np.ones(4),
            torque_effectiveness=np.ones(4),
            linear_drag=np.zeros(3),
            angular_drag=np.zeros(3),
        )
        candidate = rollout(heavy)
        np.testing.assert_allclose(
            candidate.commanded_thrust[0], baseline.commanded_thrust[0]
        )
        self.assertFalse(
            np.allclose(
                candidate.commanded_thrust[20:],
                baseline.commanded_thrust[20:],
            )
        )
        self.assertGreater(
            np.max(
                np.linalg.norm(
                    candidate.position - baseline.position, axis=1
                )
            ),
            0.01,
        )

    def test_actuator_lag_and_hardware_limits_are_stateful(self):
        parameters = ActuatorParameters(
            thrust_time_constant=0.1,
            gimbal_time_constant=0.2,
            minimum_thrust=1.5,
            maximum_thrust=27.6145,
            maximum_gimbal_angle=3.14,
            maximum_gimbal_rate=6.0,
        )
        state = ActuatorState(np.full(4, 2.0), np.zeros(4))
        command = ActuatorCommand(
            thrust=np.asarray((-5.0, 40.0, 8.0, 10.0)),
            gimbal_angle=np.asarray((-5.0, 5.0, 0.5, -0.5)),
            virtual_force=np.zeros(8),
            desired_acceleration=np.zeros(6),
        )
        first = advance_actuators(state, command, parameters, 0.02)
        second = advance_actuators(first, command, parameters, 0.02)
        self.assertTrue(np.all(first.thrust >= parameters.minimum_thrust))
        self.assertTrue(np.all(first.thrust <= parameters.maximum_thrust))
        self.assertTrue(
            np.all(np.abs(first.gimbal_angle) <= parameters.maximum_gimbal_angle)
        )
        self.assertFalse(np.allclose(first.thrust, state.thrust))
        self.assertFalse(np.allclose(second.thrust, first.thrust))
        self.assertLess(first.thrust[1], parameters.maximum_thrust)
        np.testing.assert_allclose(
            np.abs(first.gimbal_angle[:2]), 0.12
        )
        self.assertTrue(np.all(np.abs(first.gimbal_angle[2:]) < 0.12))

    def test_inertia_is_full_spd_and_quaternion_stays_normalised(self):
        parameters = VehicleParameters.nominal()
        self.assertGreater(np.max(np.abs(parameters.inertia - np.diag(
            np.diag(parameters.inertia)))), 0.0)
        self.assertTrue(np.all(np.linalg.eigvalsh(parameters.inertia) > 0.0))
        experiment = run_perfect_model_experiment(1.0, 0.01)
        np.testing.assert_allclose(
            np.linalg.norm(experiment.truth.orientation_xyzw, axis=1),
            1.0,
            atol=1.0e-12,
        )

    def test_fractional_delay_is_continuous_and_not_tick_quantised(self):
        times = np.linspace(0.0, 0.4, 21)
        references = full_six_dof_reference(times)
        nominal = VehicleParameters.nominal()
        geometry = GrapeGeometry.grape()
        configuration = ControllerConfig.grape()
        first_reference = references[0]
        initial_state = RigidBodyState(
            first_reference.position,
            matrix_to_quaternion(
                euler_xyz_to_matrix(first_reference.rpy)
            ),
            first_reference.linear_velocity,
            first_reference.angular_velocity,
        )

        def rollout(delay):
            return simulate_closed_loop(
                times,
                references,
                initial_state,
                initial_controller_state(configuration, trim_hover=True),
                GrapeController(configuration, nominal, geometry),
                FullSixDofPlant(nominal, geometry),
                ActuatorParameters(
                    thrust_time_constant=0.04,
                    gimbal_time_constant=0.04,
                    delay=delay,
                ),
            )

        zero = rollout(0.0)
        epsilon = rollout(1.0e-6)
        quarter_tick = rollout(0.005)
        full_tick = rollout(0.02)
        np.testing.assert_allclose(
            epsilon.position, zero.position, atol=5.0e-8
        )
        self.assertFalse(
            np.array_equal(quarter_tick.position, full_tick.position)
        )
        self.assertGreater(
            np.max(np.abs(quarter_tick.position - zero.position)),
            1.0e-8,
        )

    def test_interval_discrepancy_is_held_for_every_runge_kutta_stage(self):
        parameters = VehicleParameters.nominal()
        geometry = GrapeGeometry.grape()
        state = RigidBodyState(
            position=(0.1, -0.2, 0.8),
            orientation_xyzw=matrix_to_quaternion(
                euler_xyz_to_matrix((0.1, -0.08, 0.15))
            ),
            linear_velocity=(0.2, -0.1, 0.05),
            angular_velocity=(0.12, -0.09, 0.07),
        )
        actuators = ActuatorState(
            thrust=np.full(4, parameters.mass * 9.80665 / 4.0),
            gimbal_angle=np.asarray((0.1, -0.08, 0.06, -0.04)),
        )
        residual = np.asarray((0.3, -0.2, 0.1, 0.02, -0.015, 0.01))
        callback_plant = FullSixDofPlant(
            parameters,
            geometry,
            model_discrepancy_wrench=lambda _time, _state: residual,
        )
        interval_plant = FullSixDofPlant(parameters, geometry)
        expected = callback_plant.step(0.0, state, actuators, 0.02)
        actual = interval_plant.step(
            0.0,
            state,
            actuators,
            0.02,
            interval_model_discrepancy_wrench=residual,
        )
        np.testing.assert_allclose(actual.as_vector(), expected.as_vector())


if __name__ == "__main__":
    unittest.main()
