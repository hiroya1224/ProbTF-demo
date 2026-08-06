from pathlib import Path
import sys
import unittest

import numpy as np


_MINIMAL = Path(__file__).resolve().parents[1]
_PACKAGE = _MINIMAL.parent / "src"
sys.path.insert(0, str(_MINIMAL))
sys.path.insert(0, str(_PACKAGE))

import deterministic_multiple_shooting_estimator as estimator  # noqa: E402
import estimate_recorded_control as entrypoint  # noqa: E402
from grape_param_estim.geometry import (  # noqa: E402
    so3_exp,
    so3_left_jacobian,
)
from grape_param_estim.system import (  # noqa: E402
    ActuatorCommand,
    ActuatorState,
    RigidBodyState,
    VehicleParameters,
)


class SegmentScheduleTests(unittest.TestCase):
    def test_schedule_covers_full_grid_without_continuation(self):
        boundaries = estimator.segment_boundaries(101, 0.05, 0.5)
        np.testing.assert_array_equal(boundaries, np.arange(0, 101, 10))

    def test_last_short_segment_is_retained(self):
        boundaries = estimator.segment_boundaries(12, 0.05, 0.2)
        np.testing.assert_array_equal(boundaries, (0, 4, 8, 11))


class Se3ResidualTests(unittest.TestCase):
    def test_identical_poses_have_zero_error(self):
        position = np.asarray((1.0, -2.0, 0.5))
        rotation = so3_exp((0.2, -0.1, 0.3))
        np.testing.assert_allclose(
            estimator.se3_log_error(position, rotation, position, rotation),
            np.zeros(6),
            atol=1.0e-12,
        )

    def test_translation_is_pushed_to_se3_coordinate(self):
        observed_position = np.asarray((0.2, -0.3, 0.4))
        observed_rotation = so3_exp((0.1, 0.2, -0.15))
        rho = np.asarray((0.4, -0.2, 0.1))
        phi = np.asarray((0.3, -0.1, 0.2))
        relative_translation = so3_left_jacobian(phi) @ rho
        simulated_rotation = observed_rotation @ so3_exp(phi)
        simulated_position = (
            observed_position + observed_rotation @ relative_translation
        )
        residual = estimator.se3_log_error(
            observed_position,
            observed_rotation,
            simulated_position,
            simulated_rotation,
        )
        np.testing.assert_allclose(residual[:3], rho, atol=1.0e-10)
        np.testing.assert_allclose(residual[3:], phi, atol=1.0e-10)


class NodeChartTests(unittest.TestCase):
    def test_node_encode_decode_round_trip(self):
        reference = estimator.NodeReference(
            position=np.asarray((1.0, 2.0, 3.0)),
            rotation=so3_exp((0.1, -0.2, 0.05)),
            linear_velocity=np.asarray((0.2, 0.3, -0.4)),
            angular_velocity=np.asarray((0.4, -0.5, 0.6)),
            thrust=np.asarray((5.0, 6.0, 7.0, 8.0)),
            gimbal=np.asarray((0.1, -0.1, 0.2, -0.2)),
        )
        correction = np.linspace(-0.05, 0.05, estimator.NODE_DIMENSION)
        rigid, actuator, _, _ = estimator._decode_node(reference, correction)
        encoded = estimator._encode_node(reference, rigid, actuator)
        np.testing.assert_allclose(encoded, correction, atol=1.0e-10)


class FullyPhysicalInertiaParameterizationTests(unittest.TestCase):
    def setUp(self):
        self.nominal = VehicleParameters.nominal()
        self.parameterization = (
            estimator.FullyPhysicalInertiaParameterization(self.nominal)
        )

    def test_zero_coordinate_reconstructs_nominal_inertia(self):
        coordinate = np.zeros(estimator.analytic.SEARCH_DIMENSION)
        decoded = self.parameterization.decode(coordinate)
        np.testing.assert_allclose(
            decoded.parameters.inertia,
            self.nominal.inertia,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        self.assertGreater(decoded.inertia_triangle_margin, 0.0)
        self.assertEqual(
            decoded.actuator_parameters.thrust_time_constant, 0.0
        )
        self.assertEqual(
            decoded.actuator_parameters.gimbal_time_constant, 0.0
        )

    def test_bounded_chart_always_has_strict_triangle_inequalities(self):
        rng = np.random.default_rng(20260806)
        for _ in range(200):
            coordinate = np.zeros(estimator.analytic.SEARCH_DIMENSION)
            coordinate[1:4] = rng.uniform(
                np.log(0.5), np.log(2.0), size=3
            )
            coordinate[4:7] = rng.uniform(-0.8, 0.8, size=3)
            decoded = self.parameterization.decode(coordinate)
            principal = np.linalg.eigvalsh(decoded.parameters.inertia)
            self.assertGreater(principal[0], 0.0)
            self.assertGreater(principal[0] + principal[1] - principal[2], 0.0)

    def test_analytic_inertia_jacobian_matches_central_difference(self):
        coordinate = np.zeros(estimator.analytic.SEARCH_DIMENSION)
        coordinate[1:7] = np.asarray((0.1, -0.2, 0.15, 0.2, -0.3, 0.1))
        _decoded, jacobian = self.parameterization.decode_with_jacobian(
            coordinate
        )
        step = 1.0e-7
        for index in range(1, 7):
            positive = coordinate.copy()
            negative = coordinate.copy()
            positive[index] += step
            negative[index] -= step
            finite_difference = (
                self.parameterization.decode(positive).parameters.inertia
                - self.parameterization.decode(negative).parameters.inertia
            ) / (2.0 * step)
            np.testing.assert_allclose(
                jacobian.inertia[:, :, index],
                finite_difference,
                rtol=2.0e-7,
                atol=2.0e-10,
            )

    def test_removed_time_constants_apply_commands_without_first_order_lag(self):
        coordinate = np.zeros(estimator.analytic.SEARCH_DIMENSION)
        decoded, jacobian = self.parameterization.decode_with_jacobian(
            coordinate
        )
        state = ActuatorState(
            thrust=np.full(4, 2.0),
            gimbal_angle=np.zeros(4),
        )
        command = ActuatorCommand(
            thrust=np.asarray((4.0, 5.0, 6.0, 7.0)),
            gimbal_angle=np.asarray((0.01, -0.02, 0.03, -0.04)),
            virtual_force=np.zeros(8),
            desired_acceleration=np.zeros(6),
        )
        next_state, sensitivity = estimator._actuator_step_with_sensitivity(
            state,
            np.zeros((8, estimator.analytic.SMOOTH_DIMENSION)),
            command,
            decoded,
            jacobian,
            0.025,
        )
        np.testing.assert_allclose(next_state.thrust, command.thrust)
        np.testing.assert_allclose(
            next_state.gimbal_angle, command.gimbal_angle
        )
        self.assertTrue(np.all(np.isfinite(sensitivity)))


class EntryPointTests(unittest.TestCase):
    def test_multiple_shooting_is_default(self):
        self.assertEqual(
            entrypoint.DEFAULT_METHOD,
            "deterministic_multiple_shooting",
        )

    def test_parser_uses_only_requested_physical_family(self):
        arguments = estimator.create_argument_parser().parse_args([])
        self.assertFalse(hasattr(arguments, "thrust_time_constant_scale_bounds"))
        self.assertFalse(hasattr(arguments, "gimbal_time_constant_scale_bounds"))
        self.assertFalse(hasattr(arguments, "mass_scale_bounds"))
        self.assertFalse(
            hasattr(arguments, "inertia_cholesky_diagonal_scale_bounds")
        )
        self.assertFalse(hasattr(arguments, "cog_bound"))
        self.assertFalse(
            hasattr(arguments, "force_effectiveness_contrast_bound")
        )
        self.assertEqual(estimator.PHYSICAL_DIMENSION, 13)
        self.assertEqual(estimator.NODE_DIMENSION, 20)

    def test_physical_coordinates_have_no_box_bounds(self):
        arguments = estimator.create_argument_parser().parse_args([])
        lower, upper, _delays = estimator._validate_arguments(arguments)
        self.assertTrue(np.all(np.isneginf(lower)))
        self.assertTrue(np.all(np.isposinf(upper)))

    def test_default_iteration_budgets_allow_continuity_refinement(self):
        arguments = estimator.create_argument_parser().parse_args([])
        self.assertEqual(arguments.max_nfev, 120)
        self.assertEqual(arguments.augmented_lagrangian_iterations, 10)

    def test_default_soft_prior_is_broad_and_enabled(self):
        arguments = estimator.create_argument_parser().parse_args([])
        self.assertEqual(arguments.prior_weight, 1.0)
        np.testing.assert_allclose(
            estimator.BROAD_SOFT_PRIOR_STANDARD_DEVIATIONS,
            (
                1.5,
                1.5,
                1.5,
                1.5,
                2.0,
                2.0,
                2.0,
                0.25,
                0.25,
                0.25,
                1.5,
                1.5,
                1.5,
            ),
        )


if __name__ == "__main__":
    unittest.main()
