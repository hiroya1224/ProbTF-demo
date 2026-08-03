from types import SimpleNamespace
import unittest

import numpy as np

from grape_param_estim.controller import (
    ControllerConfig,
    initial_controller_state,
)
from grape_param_estim.diagonal_q import BodyWrenchDiagonalCovariance
from grape_param_estim.diagonal_q_em import DiagonalQEmConfig
from grape_param_estim.real_assimilation import build_real_strong_problem
from grape_param_estim.real_calibration import ModelErrorCalibration
from grape_param_estim.real_diagonal_q_estimation import (
    Q_ONLY_MINIMUM_MEMBER_COUNT,
    PreparedDiagonalQBag,
    draw_q_only_initial_ensemble,
    run_real_diagonal_q_em,
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


if __name__ == "__main__":
    unittest.main()
