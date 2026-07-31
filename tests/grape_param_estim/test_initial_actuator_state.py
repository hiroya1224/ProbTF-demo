from dataclasses import replace
import unittest

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.controller import GrapeController
from grape_param_estim.dynamics import advance_actuators
import grape_param_estim.mode_validation as mode_validation
from grape_param_estim.model_error import GaussMarkovWrenchProcess
from grape_param_estim.strong_constraint import CONTROL_DIMENSION
from grape_param_estim.strong_constraint_experiments import (
    _problem_from_synthetic,
)
from grape_param_estim.synthetic import run_synthetic_experiment
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    PoseObservations,
)
from grape_param_estim.weak_constraint import (
    WeakConstraintProblem,
)


class InitialActuatorStateConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        synthetic = run_synthetic_experiment(
            duration=0.16,
            time_step=0.04,
            seed=91,
        )
        cls.base_problem = _problem_from_synthetic(synthetic)
        cls.snapshot = ActuatorState(
            thrust=np.asarray((3.2, 5.1, 7.4, 9.0)),
            gimbal_angle=np.asarray((0.18, -0.13, 0.09, -0.05)),
        )
        cls.actuator_parameters = ActuatorParameters(
            thrust_time_constant=0.08,
            gimbal_time_constant=0.10,
            delay=0.0,
        )
        cls.problem = replace(
            cls.base_problem,
            actuator_parameters=cls.actuator_parameters,
            initial_actuator_state=cls.snapshot,
        )
        cls.control = np.zeros(CONTROL_DIMENSION)
        cls.control[:3] = np.asarray((0.004, -0.003, 0.002))
        cls.control[3:6] = np.asarray((0.002, -0.001, 0.003))
        cls.control[6:9] = np.asarray((0.03, -0.02, 0.01))
        cls.control[9:12] = np.asarray((0.01, -0.015, 0.008))
        cls.control[12:18] = np.asarray(
            (0.01, -0.02, 0.03, 0.004, -0.005, 0.006)
        )
        cls.strong_trajectory = cls.problem.forecast(cls.control)

    def test_optional_snapshot_is_passed_without_resetting_state_or_controller(self):
        problem = self.problem
        trajectory = self.strong_trajectory
        state, controller_state, _parameters = problem.decode_control(
            self.control
        )
        self.assertIsNone(self.base_problem.initial_actuator_state)
        self.assertIsNot(problem.initial_actuator_state, self.snapshot)
        np.testing.assert_array_equal(
            trajectory.position[0], state.position
        )
        np.testing.assert_array_equal(
            trajectory.orientation_xyzw[0], state.orientation_xyzw
        )
        np.testing.assert_array_equal(
            trajectory.linear_velocity[0], state.linear_velocity
        )
        np.testing.assert_array_equal(
            trajectory.angular_velocity[0], state.angular_velocity
        )
        np.testing.assert_array_equal(
            trajectory.controller_integral[0],
            controller_state.integral_error,
        )
        np.testing.assert_array_equal(
            trajectory.actuator_thrust[0], self.snapshot.thrust
        )
        np.testing.assert_array_equal(
            trajectory.actuator_gimbal_angle[0],
            self.snapshot.gimbal_angle,
        )

        time_step = trajectory.times[1] - trajectory.times[0]
        controller = GrapeController(
            problem.controller_configuration,
            problem.controller_parameters,
            problem.geometry,
            articulated_model=GrapeArticulatedModel(),
        )
        command, next_controller_state = controller.step(
            state,
            problem.references[0],
            controller_state,
            time_step,
            self.snapshot.gimbal_angle,
        )
        np.testing.assert_allclose(
            trajectory.commanded_thrust[0], command.thrust, atol=2.0e-14
        )
        np.testing.assert_allclose(
            trajectory.commanded_gimbal_angle[0],
            command.gimbal_angle,
            atol=2.0e-14,
        )
        np.testing.assert_array_equal(
            trajectory.controller_integral[1],
            next_controller_state.integral_error,
        )

        midpoint = advance_actuators(
            self.snapshot,
            command,
            self.actuator_parameters,
            0.5 * time_step,
        )
        expected_next_actuator = advance_actuators(
            midpoint,
            command,
            self.actuator_parameters,
            0.5 * time_step,
        )
        np.testing.assert_allclose(
            trajectory.actuator_thrust[1],
            expected_next_actuator.thrust,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            trajectory.actuator_gimbal_angle[1],
            expected_next_actuator.gimbal_angle,
            atol=2.0e-14,
        )

    def test_forecast_never_resets_to_intermediate_pose_observations(self):
        observations = self.problem.observations
        displaced = PoseObservations(
            times=observations.times,
            position=observations.position
            + np.linspace(0.0, 10.0, observations.times.size)[:, None],
            orientation_xyzw=observations.orientation_xyzw,
            translation_covariance=observations.translation_covariance,
            rotation_covariance=observations.rotation_covariance,
        )
        altered_problem = replace(self.problem, observations=displaced)
        altered = altered_problem.forecast(self.control)
        for field in (
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
        ):
            np.testing.assert_array_equal(
                getattr(altered, field),
                getattr(self.strong_trajectory, field),
            )

    def test_zero_residual_weak_forecast_keeps_the_same_snapshot_and_history(self):
        process = GaussMarkovWrenchProcess(
            times=self.problem.observations.times[:-1],
            stationary_standard_deviation=np.ones(6),
            correlation_time=0.4,
        )
        weak_problem = WeakConstraintProblem(self.problem, process)
        weak_control = np.zeros(weak_problem.control_dimension)
        weak_control[:CONTROL_DIMENSION] = self.control
        weak = weak_problem.forecast(weak_control)
        for field in (
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
        ):
            np.testing.assert_array_equal(
                getattr(weak, field),
                getattr(self.strong_trajectory, field),
            )

    def test_mode_problem_keeps_controller_nominal_and_snapshot_plant_only(self):
        synthetic = mode_validation.generate_mode_synthetic_experiment(
            truth_mode_id=mode_validation.SWAPPED_MODE_ID,
            duration=0.12,
            time_step=0.04,
        )
        nominal = replace(
            mode_validation._problem_for_mode(
                synthetic,
                mode_validation.plant_wiring_mode(
                    mode_validation.NOMINAL_MODE_ID
                ),
            ),
            actuator_parameters=self.actuator_parameters,
            initial_actuator_state=self.snapshot,
        )
        swapped = replace(
            mode_validation._problem_for_mode(
                synthetic,
                mode_validation.plant_wiring_mode(
                    mode_validation.SWAPPED_MODE_ID
                ),
            ),
            actuator_parameters=self.actuator_parameters,
            initial_actuator_state=self.snapshot,
        )
        control = np.zeros(CONTROL_DIMENSION)
        nominal_trajectory = nominal.forecast(control)
        swapped_trajectory = swapped.forecast(control)
        np.testing.assert_array_equal(
            nominal.geometry.rotor_origins, swapped.geometry.rotor_origins
        )
        np.testing.assert_array_equal(
            nominal_trajectory.actuator_thrust[0], self.snapshot.thrust
        )
        np.testing.assert_array_equal(
            swapped_trajectory.actuator_thrust[0], self.snapshot.thrust
        )
        # Same x_0/controller/snapshot gives the same nominal command; only
        # the plant-side geometry permutation changes the resulting wrench.
        np.testing.assert_array_equal(
            nominal_trajectory.commanded_thrust[0],
            swapped_trajectory.commanded_thrust[0],
        )
        np.testing.assert_array_equal(
            nominal_trajectory.commanded_gimbal_angle[0],
            swapped_trajectory.commanded_gimbal_angle[0],
        )
        self.assertGreater(
            np.linalg.norm(
                nominal_trajectory.body_wrench[0]
                - swapped_trajectory.body_wrench[0]
            ),
            1.0e-3,
        )


if __name__ == "__main__":
    unittest.main()
