from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares


_MINIMAL = Path(__file__).resolve().parents[1]
_PACKAGE = _MINIMAL.parent / "src"
sys.path.insert(0, str(_MINIMAL))
sys.path.insert(0, str(_PACKAGE))

import deterministic_estimator as baseline  # noqa: E402
import deterministic_multiple_shooting_estimator as strict  # noqa: E402
import deterministic_smooth_lag_multiple_shooting_estimator as estimator  # noqa: E402
import estimate_recorded_control as entrypoint  # noqa: E402
from grape_param_estim.dynamics import (  # noqa: E402
    advance_actuators_with_jacobian,
)
from grape_param_estim.real_rosbag import load_flight_data  # noqa: E402
from grape_param_estim.system import ActuatorCommand, ActuatorState  # noqa: E402
from smooth_command import QuinticSmoothZoh  # noqa: E402


def _command(thrust, gimbal):
    return ActuatorCommand(
        thrust=np.asarray(thrust, dtype=float),
        gimbal_angle=np.asarray(gimbal, dtype=float),
        virtual_force=np.zeros(8),
        desired_acceleration=np.zeros(6),
    )


class SmoothLagConfigurationTests(unittest.TestCase):
    def test_global_coordinate_is_thirteen_physical_plus_delay(self):
        self.assertEqual(strict.PHYSICAL_DIMENSION, 13)
        self.assertEqual(estimator.GLOBAL_DIMENSION, 14)
        self.assertEqual(estimator.DELAY_INDEX, 13)

    def test_parser_has_requested_continuation_and_polish_defaults(self):
        arguments = estimator.create_argument_parser().parse_args([])
        np.testing.assert_allclose(
            arguments.smoothstep_width_fractions,
            (0.50, 0.20, 0.05),
        )
        self.assertAlmostEqual(arguments.initial_delay, 0.01)
        self.assertAlmostEqual(arguments.zoh_polish_radius, 0.004)
        self.assertAlmostEqual(arguments.zoh_polish_step, 0.001)
        self.assertEqual(arguments.zoh_polish_top_k, 3)
        self.assertAlmostEqual(arguments.body_displacement_scale, 1.0)
        self.assertEqual(arguments.smooth_max_nfev, 60)
        self.assertEqual(arguments.max_nfev, 120)

    def test_entrypoint_exposes_new_method_without_changing_default(self):
        self.assertEqual(
            entrypoint.DEFAULT_METHOD,
            "deterministic_multiple_shooting",
        )
        with patch.object(estimator, "main", return_value=0) as selected_main:
            status = entrypoint.main(
                (
                    "--method",
                    "deterministic_smooth_lag_multiple_shooting",
                    "--max-nfev",
                    "7",
                )
            )
        self.assertEqual(status, 0)
        selected_main.assert_called_once_with(["--max-nfev", "7"])

    def test_local_zoh_polish_grid_is_bounded_and_unique(self):
        delays = estimator.zoh_polish_delays(0.001, 0.004, 0.001, (0.0, 0.2))
        np.testing.assert_allclose(delays, (0.0, 0.001, 0.002, 0.003, 0.004, 0.005))
        self.assertTrue(np.all(np.diff(delays) > 0.0))


class ActuatorLagSensitivityTests(unittest.TestCase):
    def setUp(self):
        parameterization = strict.FullyPhysicalInertiaParameterization(
            strict.VehicleParameters.nominal()
        )
        decoded, physical_jacobian = strict._physical_parameter_jacobian(
            parameterization,
            np.zeros(strict.PHYSICAL_DIMENSION),
            0.01,
        )
        self.decoded = decoded
        self.parameter_jacobian = strict._extend_parameter_jacobian(
            physical_jacobian,
            estimator.GLOBAL_DIMENSION,
        )
        self.state = ActuatorState(
            thrust=np.asarray((2.0, 2.1, 2.2, 2.3)),
            gimbal_angle=np.asarray((0.01, -0.01, 0.02, -0.02)),
        )

    def test_command_derivative_propagates_through_actuator_step(self):
        base = _command(
            (3.0, 3.1, 3.2, 3.3),
            (0.02, -0.02, 0.03, -0.03),
        )
        direction = np.asarray(
            (0.2, -0.3, 0.1, 0.4, 0.02, -0.01, 0.015, -0.025)
        )
        command_sensitivity = np.zeros((8, estimator.GLOBAL_DIMENSION))
        command_sensitivity[:, estimator.DELAY_INDEX] = direction
        _state, analytic = strict._actuator_step_with_sensitivity(
            self.state,
            np.zeros((8, estimator.GLOBAL_DIMENSION)),
            base,
            self.decoded,
            self.parameter_jacobian,
            0.025,
            command_sensitivity,
        )
        step = 1.0e-7

        def evaluate(offset):
            command = _command(
                base.thrust + offset * direction[:4],
                base.gimbal_angle + offset * direction[4:],
            )
            value, _derivative = strict._actuator_step_with_sensitivity(
                self.state,
                np.zeros((8, estimator.GLOBAL_DIMENSION)),
                command,
                self.decoded,
                self.parameter_jacobian,
                0.025,
            )
            return np.concatenate((value.thrust, value.gimbal_angle))

        numerical = (evaluate(step) - evaluate(-step)) / (2.0 * step)
        np.testing.assert_allclose(
            analytic[:, estimator.DELAY_INDEX],
            numerical,
            rtol=1.0e-8,
            atol=2.0e-9,
        )

    def test_active_gimbal_rate_limit_has_finite_zero_command_gradient(self):
        command = _command((3.0, 3.0, 3.0, 3.0), (0.5, -0.5, 0.5, -0.5))
        command_sensitivity = np.zeros((8, estimator.GLOBAL_DIMENSION))
        command_sensitivity[4:, estimator.DELAY_INDEX] = 1.0
        state, sensitivity = strict._actuator_step_with_sensitivity(
            self.state,
            np.zeros((8, estimator.GLOBAL_DIMENSION)),
            command,
            self.decoded,
            self.parameter_jacobian,
            0.001,
            command_sensitivity,
        )
        self.assertTrue(np.all(np.isfinite(state.gimbal_angle)))
        self.assertTrue(np.all(np.isfinite(sensitivity)))
        np.testing.assert_array_equal(
            sensitivity[4:, estimator.DELAY_INDEX],
            np.zeros(4),
        )

    def test_thrust_and_gimbal_clipping_active_sets_are_reported(self):
        command = _command(
            (-1.0, 1.0e6, 3.0, 3.0),
            (-10.0, 10.0, 0.0, 0.0),
        )
        evaluation = advance_actuators_with_jacobian(
            self.state,
            command,
            self.decoded.actuator_parameters,
            0.025,
        )
        self.assertTrue(evaluation.active_set["thrust_command_lower"][0])
        self.assertTrue(evaluation.active_set["thrust_command_upper"][1])
        self.assertTrue(evaluation.active_set["gimbal_command_lower"][0])
        self.assertTrue(evaluation.active_set["gimbal_command_upper"][1])


class LagRecoveryTests(unittest.TestCase):
    def test_smooth_command_fit_recovers_known_delay(self):
        history = QuinticSmoothZoh(
            np.arange(0.0, 2.1, 0.1),
            np.sin(np.arange(0.0, 2.1, 0.1)[:, None] * 7.0),
        )
        times = np.arange(0.25, 1.85, 0.013)
        true_delay = 0.037
        width = 0.5
        observed = np.asarray(
            [history.evaluate(time, true_delay, width).value for time in times]
        ).reshape(-1)

        def residual(value):
            return np.asarray(
                [history.evaluate(time, value[0], width).value for time in times]
            ).reshape(-1) - observed

        def jacobian(value):
            return np.asarray(
                [
                    history.evaluate(time, value[0], width).delay_derivative
                    for time in times
                ]
            ).reshape(-1, 1)

        result = least_squares(
            residual,
            np.asarray((0.01,)),
            jac=jacobian,
            bounds=((0.0,), (0.08,)),
        )
        self.assertAlmostEqual(result.x[0], true_delay, places=8)

    def test_exact_zoh_local_profile_selects_known_delay(self):
        history = QuinticSmoothZoh(
            (0.0, 0.1, 0.2, 0.3, 0.4),
            np.asarray(((0.0,), (1.0,), (-2.0,), (3.0,), (-1.0,))),
        )
        times = np.arange(0.0, 0.451, 0.0005)
        true_delay = 0.04
        observed = np.asarray(
            [history.exact_zoh(time, true_delay) for time in times]
        )
        candidates = estimator.zoh_polish_delays(0.04, 0.004, 0.001, (0.0, 0.2))
        losses = np.asarray(
            [
                sum(
                    float(
                        np.sum(
                            (history.exact_zoh(time, delay) - target) ** 2
                        )
                    )
                    for time, target in zip(times, observed)
                )
                for delay in candidates
            ]
        )
        selected = candidates[int(np.argmin(losses))]
        self.assertLessEqual(abs(selected - true_delay), 0.001)


@unittest.skipUnless(baseline.DEFAULT_BAG.is_file(), "sample rosbag unavailable")
class RecordedFlightJacobianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flight = load_flight_data(
            str(baseline.DEFAULT_BAG),
            start_local=19.0,
            end_local=19.2,
            include_fc_specific_force=True,
            compute_sha256=False,
        )
        cls.problem = estimator.SmoothLagMultipleShootingProblem(
            flight=flight,
            sample_step=0.05,
            integration_step=0.025,
            initial_delay=0.01,
            width_fraction=0.5,
            segment_duration=0.1,
            body_displacement_scale=1.0,
            prior_weight=1.0,
            node_position_bound=2.0,
            node_orientation_bound=1.5,
            node_velocity_bound=5.0,
            node_angular_velocity_bound=10.0,
        )

    def test_full_residual_delay_jacobian_matches_central_difference(self):
        coordinate = self.problem.initial_coordinate()
        evaluation = self.problem.evaluate(coordinate)
        analytic = np.concatenate(
            (
                evaluation.data_jacobian[:, estimator.DELAY_INDEX],
                evaluation.continuity_jacobian[:, estimator.DELAY_INDEX],
            )
        )
        step = 1.0e-7
        positive = coordinate.copy()
        negative = coordinate.copy()
        positive[estimator.DELAY_INDEX] += step
        negative[estimator.DELAY_INDEX] -= step
        positive_evaluation = self.problem.evaluate(positive)
        negative_evaluation = self.problem.evaluate(negative)
        numerical = (
            np.concatenate(
                (
                    positive_evaluation.data_residual,
                    positive_evaluation.continuity_residual,
                )
            )
            - np.concatenate(
                (
                    negative_evaluation.data_residual,
                    negative_evaluation.continuity_residual,
                )
            )
        ) / (2.0 * step)
        np.testing.assert_allclose(analytic, numerical, rtol=3.0e-4, atol=2.0e-5)

    def test_smooth_full_rollout_accepts_fourteen_global_coordinates(self):
        coordinate = self.problem.initial_coordinate()
        global_coordinate, _nodes = self.problem.split_coordinate(coordinate)
        position, orientation, residual = self.problem.full_rollout(
            global_coordinate
        )
        self.assertEqual(
            position.shape[0],
            self.problem.direct_problem.output_time.size,
        )
        self.assertEqual(orientation.shape, (position.shape[0], 4))
        self.assertEqual(residual.shape, (position.shape[0] * 6,))
        self.assertTrue(np.all(np.isfinite(residual)))


if __name__ == "__main__":
    unittest.main()
