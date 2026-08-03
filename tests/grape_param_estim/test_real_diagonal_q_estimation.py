from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from grape_param_estim.controller import (
    ControllerConfig,
    initial_controller_state,
)
from grape_param_estim.diagonal_q import BodyWrenchDiagonalCovariance
from grape_param_estim.diagonal_q_em import (
    BACKTRACKING_ACCEPTED,
    BACKTRACKING_NUMERICAL_FAILURE,
    DiagonalQEmConfig,
)
from grape_param_estim.filter_state import FILTER_STATE_DIMENSION
from grape_param_estim.parallel_stepper import ParallelStepperError
from grape_param_estim.real_assimilation import build_real_strong_problem
from grape_param_estim.real_calibration import ModelErrorCalibration
from grape_param_estim.real_diagonal_q_estimation import (
    Q_ONLY_MINIMUM_MEMBER_COUNT,
    PreparedDiagonalQBag,
    draw_q_only_initial_ensemble,
    run_real_diagonal_q_em,
)
from grape_param_estim.stochastic_closed_loop import (
    StochasticClosedLoopEStepResult,
)
from grape_param_estim.synthetic import run_synthetic_experiment
from grape_param_estim.system import ActuatorParameters, ActuatorState


class RealDiagonalQEstimationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        synthetic = run_synthetic_experiment(
            duration=0.32,
            time_step=0.04,
            truth_actuators=ActuatorParameters(delay=0.01),
            truth_residual_wrench=lambda _time, _state: np.zeros(6),
            translation_noise=0.001,
            rotation_noise=0.001,
            seed=81,
        )
        configuration = ControllerConfig.grape()
        episode = SimpleNamespace(
            observations=synthetic.observations,
            references=synthetic.references,
            controller_configuration=configuration,
            initial_controller_state=initial_controller_state(
                configuration, trim_hover=True
            ),
            initial_actuator_state=ActuatorState(
                synthetic.nominal.actuator_thrust[0],
                synthetic.nominal.actuator_gimbal_angle[0],
            ),
        )
        problem = build_real_strong_problem(
            episode, actuator_parameters=ActuatorParameters(delay=0.01)
        )[0]
        count = problem.observations.times.size
        calibration = ModelErrorCalibration(
            stationary_standard_deviation=np.asarray(
                (0.30, 0.35, 0.40, 0.08, 0.09, 0.10)
            ),
            pilot_location=np.zeros(6),
            correlation_time=0.16,
            proxy_wrench=np.zeros((count, 6)),
            valid_mask=np.ones(count, dtype=bool),
            derivative_window_samples=5,
            method="unit-test-pilot/v1",
        )
        cls.first = PreparedDiagonalQBag(
            "bag-a", problem, calibration, "same-airframe"
        )
        cls.second = PreparedDiagonalQBag(
            "bag-b", problem, calibration, "same-airframe"
        )

    @staticmethod
    def _mock_e_step_result(arguments, log_likelihood):
        times = np.asarray(arguments["times"], dtype=float)
        states = tuple(arguments["initial_state_ensemble"])
        state_series = tuple(states for _time in times)
        smoothed_wrench = np.transpose(
            np.asarray(
                [
                    [state.residual_wrench for state in ensemble]
                    for ensemble in state_series
                ]
            ),
            (1, 0, 2),
        )
        likelihood_by_time = np.zeros(times.size, dtype=float)
        likelihood_by_time[0] = float(log_likelihood)
        return StochasticClosedLoopEStepResult(
            times=times,
            smoothed_wrench_ensemble=smoothed_wrench,
            filter_log_likelihood=float(log_likelihood),
            filter_log_likelihood_by_time=likelihood_by_time,
            filter_nis=np.zeros(times.size, dtype=float),
            forecast_state_ensembles=state_series,
            analysis_state_ensembles=state_series,
            smoothed_state_ensembles=state_series,
            smoothing_gains=tuple(
                np.zeros(
                    (FILTER_STATE_DIMENSION, FILTER_STATE_DIMENSION),
                    dtype=float,
                )
                for _index in range(max(0, times.size - 1))
            ),
            command_issue_times=tuple(
                times[:-1].copy() for _state in states
            ),
        )

    def test_initial_local_and_wrench_priors_are_exact_and_uncorrelated(self):
        covariance = BodyWrenchDiagonalCovariance(
            np.asarray((0.5, 0.6, 0.7, 0.05, 0.06, 0.07))
        )
        initial = draw_q_only_initial_ensemble(
            self.first, covariance, 40, seed=91
        )
        wrench = np.asarray(
            [value.residual_wrench for value in initial.filter_states]
        )
        np.testing.assert_allclose(wrench.mean(0), 0.0, atol=1.0e-15)
        np.testing.assert_allclose(
            np.cov(wrench, rowvar=False), covariance.matrix, atol=2.0e-15
        )
        local_anomalies = (
            initial.local_coordinates
            - initial.local_coordinates.mean(axis=0, keepdims=True)
        )
        np.testing.assert_allclose(
            local_anomalies.T @ wrench / 39.0, 0.0, atol=2.0e-15
        )

    def test_one_iteration_is_deterministic_and_retains_final_paths(self):
        config = DiagonalQEmConfig(1, 1.0e-12, 1.0e-9)
        first = run_real_diagonal_q_em(
            (self.first,), config, ensemble_size=40, seed=123
        )
        repeated = run_real_diagonal_q_em(
            (self.first,), config, ensemble_size=40, seed=123
        )
        np.testing.assert_array_equal(
            first.covariance.stationary_variance,
            repeated.covariance.stationary_variance,
        )
        np.testing.assert_array_equal(
            first.e_step("bag-a").smoothed_wrench_ensemble,
            repeated.e_step("bag-a").smoothed_wrench_ensemble,
        )
        self.assertEqual(first.em_result.bag_ids, ("bag-a",))
        self.assertEqual(
            first.e_step("bag-a").times.size,
            self.first.boundary_count,
        )
        self.assertTrue(np.isfinite(first.e_step("bag-a").log_likelihood))

    def test_input_bag_order_does_not_change_shared_q(self):
        config = DiagonalQEmConfig(1, 1.0e-12, 1.0e-9)
        forward = run_real_diagonal_q_em(
            (self.first, self.second),
            config,
            ensemble_size=40,
            seed=22,
        )
        reverse = run_real_diagonal_q_em(
            (self.second, self.first),
            config,
            ensemble_size=40,
            seed=22,
        )
        self.assertEqual(
            tuple(value.bag_id for value in forward.prepared_bags),
            ("bag-a", "bag-b"),
        )
        np.testing.assert_array_equal(
            forward.covariance.stationary_variance,
            reverse.covariance.stationary_variance,
        )
        for bag_id in ("bag-a", "bag-b"):
            np.testing.assert_array_equal(
                forward.e_step(bag_id).smoothed_wrench_ensemble,
                reverse.e_step(bag_id).smoothed_wrench_ensemble,
            )

    def test_fixed_R_correlation_time_and_member_floor_are_explicit(self):
        covariance = self.first.observation_covariance
        np.testing.assert_array_equal(
            covariance.translation,
            self.first.problem.observations.translation_covariance,
        )
        np.testing.assert_array_equal(
            covariance.rotation_tangent,
            self.first.problem.observations.rotation_covariance,
        )
        self.assertEqual(self.first.calibration.correlation_time, 0.16)
        with self.assertRaisesRegex(ValueError, "at least 39"):
            draw_q_only_initial_ensemble(
                self.first,
                BodyWrenchDiagonalCovariance(np.ones(6)),
                Q_ONLY_MINIMUM_MEMBER_COUNT - 1,
                seed=1,
            )
        incompatible = PreparedDiagonalQBag(
            "bag-c",
            self.first.problem,
            self.first.calibration,
            "different-airframe",
        )
        with self.assertRaisesRegex(ValueError, "configuration fingerprint"):
            run_real_diagonal_q_em(
                (self.first, incompatible),
                DiagonalQEmConfig(1, 1.0e-3, 1.0e-9),
                ensemble_size=40,
            )

    def test_q_dependent_numerical_failure_is_backtracked(self):
        failures = (
            ParallelStepperError(
                "ValueError: rigid-body state must contain 13 finite values"
            ),
            ParallelStepperError(
                "OverflowError: dynamics overflow during integration"
            ),
            OverflowError("intermediate overflow in fsum"),
        )
        for failure in failures:
            with self.subTest(failure=repr(failure)):
                calls = []

                def mocked_e_step(**arguments):
                    calls.append(arguments)
                    if len(calls) == 2:
                        raise failure
                    likelihood = -10.0 if len(calls) == 1 else -9.0
                    return self._mock_e_step_result(
                        arguments, likelihood
                    )

                with mock.patch(
                    "grape_param_estim.real_diagonal_q_estimation."
                    "run_stochastic_closed_loop_e_step",
                    side_effect=mocked_e_step,
                ):
                    result = run_real_diagonal_q_em(
                        (self.first,),
                        DiagonalQEmConfig(
                            1,
                            1.0e-12,
                            1.0e-9,
                            backtracking_step_fractions=(1.0, 0.5),
                        ),
                        ensemble_size=40,
                        seed=123,
                    )

                iteration = result.em_result.iterations[0]
                self.assertEqual(len(calls), 3)
                self.assertEqual(iteration.accepted_step_fraction, 0.5)
                self.assertEqual(
                    tuple(
                        trial.outcome
                        for trial in iteration.backtracking_trials
                    ),
                    (
                        BACKTRACKING_NUMERICAL_FAILURE,
                        BACKTRACKING_ACCEPTED,
                    ),
                )

    def test_q_dependent_initial_ensemble_failure_is_backtracked(self):
        draw_calls = []
        e_step_calls = []
        real_draw = draw_q_only_initial_ensemble

        def mocked_draw(*arguments, **keywords):
            draw_calls.append((arguments, keywords))
            if len(draw_calls) == 2:
                raise np.linalg.LinAlgError(
                    "exact Gaussian ensemble is not representable"
                )
            return real_draw(*arguments, **keywords)

        def mocked_e_step(**arguments):
            e_step_calls.append(arguments)
            likelihood = -10.0 if len(e_step_calls) == 1 else -9.0
            return self._mock_e_step_result(arguments, likelihood)

        with mock.patch(
            "grape_param_estim.real_diagonal_q_estimation."
            "draw_q_only_initial_ensemble",
            side_effect=mocked_draw,
        ), mock.patch(
            "grape_param_estim.real_diagonal_q_estimation."
            "run_stochastic_closed_loop_e_step",
            side_effect=mocked_e_step,
        ):
            result = run_real_diagonal_q_em(
                (self.first,),
                DiagonalQEmConfig(
                    1,
                    1.0e-12,
                    1.0e-9,
                    backtracking_step_fractions=(1.0, 0.5),
                ),
                ensemble_size=40,
                seed=123,
            )

        iteration = result.em_result.iterations[0]
        self.assertEqual(len(draw_calls), 3)
        self.assertEqual(len(e_step_calls), 2)
        self.assertEqual(iteration.accepted_step_fraction, 0.5)
        self.assertEqual(
            tuple(trial.outcome for trial in iteration.backtracking_trials),
            (BACKTRACKING_NUMERICAL_FAILURE, BACKTRACKING_ACCEPTED),
        )

    def test_unexpected_parallel_worker_failure_is_not_swallowed(self):
        calls = []

        def mocked_e_step(**arguments):
            calls.append(arguments)
            if len(calls) == 2:
                raise ParallelStepperError(
                    "parallel stepper worker exited unexpectedly"
                )
            return self._mock_e_step_result(arguments, -10.0)

        with mock.patch(
            "grape_param_estim.real_diagonal_q_estimation."
            "run_stochastic_closed_loop_e_step",
            side_effect=mocked_e_step,
        ):
            with self.assertRaisesRegex(
                ParallelStepperError, "worker exited unexpectedly"
            ):
                run_real_diagonal_q_em(
                    (self.first,),
                    DiagonalQEmConfig(
                        1,
                        1.0e-12,
                        1.0e-9,
                        backtracking_step_fractions=(1.0, 0.5),
                    ),
                    ensemble_size=40,
                    seed=123,
                )
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
