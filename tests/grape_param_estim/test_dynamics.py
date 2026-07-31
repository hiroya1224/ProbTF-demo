import unittest

import numpy as np

from grape_param_estim.controller import (
    ControllerConfig,
    GrapeController,
    initial_controller_state,
)
from grape_param_estim.dynamics import (
    FullSixDofPlant,
    advance_actuators,
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


class DynamicsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
