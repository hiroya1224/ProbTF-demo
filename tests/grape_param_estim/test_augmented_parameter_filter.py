import unittest
import numpy as np

from grape_param_estim.augmented_parameter_filter import (
    COMMAND_COORDINATE_DIMENSION,
    _analyse_auxiliary_ensemble,
    run_augmented_parameter_filter,
)
from grape_param_estim.augmented_parameter_state import (
    AUGMENTED_FILTER_DIMENSION,
    MINIMUM_PROCESS_NOISE_MEMBER_COUNT,
    SHARED_STATIC_DIMENSION,
    AugmentedInitialEnsemble,
    decode_shared_static_coordinates,
    draw_augmented_initial_ensemble,
)
from grape_param_estim.controller import (
    ControllerConfig,
    initial_controller_state,
)
from grape_param_estim.diagonal_q import (
    BodyWrenchDiagonalCovariance,
    OuTransitionFactors,
)
from grape_param_estim.filter_state import GrapeFilterStateChart
from grape_param_estim.ensemble_state_smoother import (
    deterministic_square_root_update,
)
from grape_param_estim.progress import (
    CancellationToken,
    ProgressCancelled,
)
from grape_param_estim.parameterization import VehicleParameterChart
from grape_param_estim.stochastic_closed_loop import (
    PoseObservationCovariance,
)
from grape_param_estim.synthetic import run_synthetic_experiment
from grape_param_estim.strong_constraint import StrongConstraintProblem
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    GrapeGeometry,
    RigidBodyState,
    VehicleParameters,
)


class AugmentedParameterFilterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        synthetic = run_synthetic_experiment(
            duration=0.08,
            time_step=0.04,
            truth_actuators=ActuatorParameters(
                thrust_time_constant=0.035,
                gimbal_time_constant=0.045,
                delay=0.018,
            ),
            truth_residual_wrench=lambda _time, _state: np.zeros(6),
            translation_noise=0.003,
            rotation_noise=0.002,
            seed=81,
        )
        configuration = ControllerConfig.grape()
        controller_state = initial_controller_state(
            configuration, trim_hover=True
        )
        nominal_parameters = VehicleParameters.nominal()
        geometry = GrapeGeometry.grape()
        initial_actuator = ActuatorState(
            synthetic.nominal.actuator_thrust[0],
            synthetic.nominal.actuator_gimbal_angle[0],
        )
        cls.problem = StrongConstraintProblem(
            references=synthetic.references,
            observations=synthetic.observations,
            nominal_trajectory=synthetic.nominal,
            initial_state_anchor=RigidBodyState(
                synthetic.nominal.position[0],
                synthetic.nominal.orientation_xyzw[0],
                synthetic.nominal.linear_velocity[0],
                synthetic.nominal.angular_velocity[0],
            ),
            initial_controller_anchor=controller_state,
            controller_configuration=configuration,
            controller_parameters=nominal_parameters,
            geometry=geometry,
            actuator_parameters=ActuatorParameters(
                thrust_time_constant=0.03,
                gimbal_time_constant=0.04,
                delay=0.02,
            ),
            parameter_chart=VehicleParameterChart(nominal_parameters),
            initial_actuator_state=initial_actuator,
        )
        cls.q_covariance = BodyWrenchDiagonalCovariance(
            np.asarray((0.04, 0.0625, 0.09, 0.0064, 0.0081, 0.0121))
        )
        cls.initial_ensemble = draw_augmented_initial_ensemble(
            cls.problem,
            cls.q_covariance,
            MINIMUM_PROCESS_NOISE_MEMBER_COUNT,
            seed=407,
        )
        cls.observation_covariance = PoseObservationCovariance.isotropic(
            0.02, 0.012
        )
        cls.progress_events = []
        cls.result = cls._run_once(
            seed=919, progress_callback=cls.progress_events.append
        )

    @classmethod
    def _run_once(cls, seed, **overrides):
        arguments = {
            "problem": cls.problem,
            "initial_ensemble": cls.initial_ensemble,
            "wrench_covariance": cls.q_covariance,
            "correlation_time": 0.17,
            "observation_covariance": cls.observation_covariance,
            "seed": seed,
            "progress_run_id": "synthetic-augmented",
            "bag_id": "synthetic",
        }
        arguments.update(overrides)
        return run_augmented_parameter_filter(**arguments)

    def test_dimensions_static_update_model_history_and_clipping(self):
        result = self.result
        members = MINIMUM_PROCESS_NOISE_MEMBER_COUNT
        times = self.problem.observations.times
        static_shape = (members, times.size, SHARED_STATIC_DIMENSION)
        self.assertEqual(result.prior_static_ensemble.shape, (members, 19))
        self.assertEqual(result.final_static_ensemble.shape, (members, 19))
        self.assertEqual(result.static_forecast_ensemble.shape, static_shape)
        self.assertEqual(result.static_analysis_ensemble.shape, static_shape)
        self.assertEqual(result.static_smoothed_ensemble.shape, static_shape)
        self.assertEqual(result.smoothed_wrench_ensemble.shape, (members, 3, 6))
        self.assertEqual(len(result.smoothing_gains), 2)
        for gain in result.smoothing_gains:
            self.assertEqual(
                gain.shape,
                (AUGMENTED_FILTER_DIMENSION, AUGMENTED_FILTER_DIMENSION),
            )
        np.testing.assert_array_equal(
            result.static_forecast_ensemble[:, 1:, :],
            result.static_analysis_ensemble[:, :-1, :],
        )
        self.assertGreater(
            np.linalg.norm(
                result.final_static_ensemble
                - result.prior_static_ensemble
            ),
            1.0e-10,
        )

        for member in (0, 17, members - 1):
            for time_index in range(times.size):
                parameters, delay = decode_shared_static_coordinates(
                    self.problem,
                    result.static_analysis_ensemble[member, time_index],
                )
                self.assertEqual(
                    result.applied_model_mass[member, time_index],
                    parameters.mass,
                )
                self.assertEqual(
                    result.applied_model_delay[member, time_index], delay
                )
        self.assertGreater(np.std(result.applied_model_mass[:, -1]), 0.0)
        self.assertGreater(np.std(result.applied_model_delay[:, -1]), 0.0)
        for history in result.command_issue_times:
            np.testing.assert_array_equal(history, times[:-1])
        self.assertEqual(result.maximum_delay, 0.2)
        self.assertEqual(
            result.final_command_history_ensemble.shape,
            (members, times.size - 1, COMMAND_COORDINATE_DIMENSION),
        )
        self.assertTrue(
            np.all(np.isfinite(result.final_command_history_ensemble))
        )

        limits = self.problem.actuator_parameters
        for ensemble in result.dynamic_analysis_state_ensembles:
            for state in ensemble:
                self.assertTrue(
                    np.all(state.actuator.thrust >= limits.minimum_thrust)
                )
                self.assertTrue(
                    np.all(state.actuator.thrust <= limits.maximum_thrust)
                )
                self.assertTrue(
                    np.all(
                        np.abs(state.actuator.gimbal_angle)
                        <= limits.maximum_gimbal_angle
                    )
                )
        completed = [event.completed_units for event in self.progress_events]
        self.assertEqual(completed, sorted(completed))
        self.assertEqual(self.progress_events[0].completed_units, 0)
        self.assertEqual(
            self.progress_events[-1].completed_units,
            self.progress_events[-1].total_units,
        )

    def test_process_noise_is_exact_and_orthogonal_to_51d_analysis(self):
        result = self.result
        factors = OuTransitionFactors(result.times, 0.17)
        variance = factors.innovation_variance(self.q_covariance)
        for time_index in range(result.times.size - 1):
            analysis_dynamic = (
                result.dynamic_analysis_state_ensembles[time_index]
            )
            forecast_dynamic = (
                result.dynamic_forecast_state_ensembles[time_index + 1]
            )
            current = np.asarray(
                [state.residual_wrench for state in analysis_dynamic]
            )
            following = np.asarray(
                [state.residual_wrench for state in forecast_dynamic]
            )
            noise = following - factors.rho[time_index] * current
            np.testing.assert_allclose(np.mean(noise, axis=0), 0.0, atol=2e-15)
            np.testing.assert_allclose(
                np.cov(noise, rowvar=False),
                np.diag(variance[time_index]),
                rtol=4e-14,
                atol=5e-16,
            )
            chart = GrapeFilterStateChart.from_ensemble(analysis_dynamic)
            coordinates = np.concatenate(
                (
                    result.static_analysis_ensemble[:, time_index, :],
                    chart.encode_ensemble(analysis_dynamic),
                ),
                axis=1,
            )
            anomalies = coordinates - np.mean(coordinates, axis=0)
            np.testing.assert_allclose(
                anomalies.T @ noise / (noise.shape[0] - 1.0),
                0.0,
                atol=8e-15,
            )

    def test_seeded_filter_is_reproducible(self):
        repeated = self._run_once(seed=919)
        changed = self._run_once(seed=920)
        np.testing.assert_array_equal(
            self.result.static_smoothed_ensemble,
            repeated.static_smoothed_ensemble,
        )
        np.testing.assert_array_equal(
            self.result.smoothed_wrench_ensemble,
            repeated.smoothed_wrench_ensemble,
        )
        np.testing.assert_array_equal(
            self.result.filter_log_likelihood_by_time,
            repeated.filter_log_likelihood_by_time,
        )
        self.assertFalse(
            np.array_equal(
                self.result.smoothed_wrench_ensemble,
                changed.smoothed_wrench_ensemble,
            )
        )

    def test_auxiliary_history_uses_the_exact_etkf_member_analysis(self):
        generator = np.random.RandomState(310)
        members = MINIMUM_PROCESS_NOISE_MEMBER_COUNT
        core = generator.normal(size=(members, 9))
        auxiliary = generator.normal(size=(members, 44))
        predicted = (
            core[:, :6]
            + 0.25 * auxiliary[:, :6]
            + generator.normal(scale=0.1, size=(members, 6))
        )
        covariance = np.diag(np.linspace(0.04, 0.09, 6))
        observation = np.mean(predicted, axis=0) + np.asarray(
            (0.2, -0.1, 0.05, 0.03, -0.02, 0.04)
        )

        core_update = deterministic_square_root_update(
            core, predicted, observation, covariance
        )
        combined_update = deterministic_square_root_update(
            np.concatenate((core, auxiliary), axis=1),
            predicted,
            observation,
            covariance,
        )
        actual = _analyse_auxiliary_ensemble(
            auxiliary, predicted, core_update
        )
        np.testing.assert_allclose(
            actual,
            combined_update.analysis_ensemble[:, core.shape[1]:],
            rtol=3.0e-14,
            atol=3.0e-14,
        )

        zero_update = deterministic_square_root_update(
            core, predicted, np.mean(predicted, axis=0), covariance
        )
        zero_actual = _analyse_auxiliary_ensemble(
            auxiliary, predicted, zero_update
        )
        np.testing.assert_allclose(
            np.mean(zero_actual, axis=0),
            np.mean(auxiliary, axis=0),
            rtol=0.0,
            atol=3.0e-15,
        )
        self.assertLess(
            np.linalg.norm(
                zero_actual - np.mean(zero_actual, axis=0, keepdims=True)
            ),
            np.linalg.norm(
                auxiliary - np.mean(auxiliary, axis=0, keepdims=True)
            ),
        )

    def test_cancellation_member_floor_and_types_are_validated(self):
        cancellation = CancellationToken()
        cancellation.cancel("unit_test")
        with self.assertRaises(ProgressCancelled):
            self._run_once(seed=1, cancellation_token=cancellation)

        short = AugmentedInitialEnsemble(
            self.initial_ensemble.member_id[:52],
            self.initial_ensemble.shared_coordinates[:52],
            self.initial_ensemble.local_coordinates[:52],
            self.initial_ensemble.filter_states[:52],
        )
        with self.assertRaisesRegex(ValueError, "at least 58"):
            self._run_once(seed=1, initial_ensemble=short)
        with self.assertRaises(TypeError):
            self._run_once(seed=1, problem=object())
        with self.assertRaises(TypeError):
            self._run_once(seed=1, wrench_covariance=object())
        with self.assertRaises(TypeError):
            self._run_once(seed=1, observation_covariance=object())
        with self.assertRaisesRegex(ValueError, "seed"):
            self._run_once(seed=-1)


if __name__ == "__main__":
    unittest.main()
