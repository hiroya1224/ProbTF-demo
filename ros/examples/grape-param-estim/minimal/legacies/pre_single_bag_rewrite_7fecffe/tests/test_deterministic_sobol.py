from pathlib import Path
import sys
import unittest

import numpy as np


_MINIMAL = Path(__file__).resolve().parents[1]
_PACKAGE = _MINIMAL.parent / "src"
sys.path.insert(0, str(_MINIMAL))
sys.path.insert(0, str(_PACKAGE))

from legacies import deterministic_sobol_estimator as estimator  # noqa: E402
from grape_param_estim.system import GrapeGeometry, VehicleParameters  # noqa: E402


class PhysicalSearchParameterizationTests(unittest.TestCase):
    def test_current_coordinate_is_nominal_and_pid_neutral(self):
        nominal = VehicleParameters.nominal()
        parameterization = estimator.PhysicalSearchParameterization(nominal)
        decoded = parameterization.decode(
            parameterization.current_coordinate(0.01)
        )
        self.assertAlmostEqual(decoded.parameters.mass, nominal.mass)
        np.testing.assert_allclose(decoded.parameters.inertia, nominal.inertia)
        np.testing.assert_allclose(
            decoded.parameters.force_effectiveness,
            nominal.force_effectiveness,
        )
        self.assertGreater(decoded.inertia_triangle_margin, 0.0)
        gate = estimator.PidGainGate.from_scale_band(
            np.ones((4, 3)), 0.8, 1.2
        )
        diagnostic = gate.evaluate(
            decoded.parameters, nominal, GrapeGeometry.grape()
        )
        self.assertTrue(diagnostic["valid"])
        np.testing.assert_allclose(diagnostic["group_scales"], np.ones(4))

    def test_cholesky_and_log_coordinates_enforce_core_constraints(self):
        nominal = VehicleParameters.nominal()
        parameterization = estimator.PhysicalSearchParameterization(nominal)
        coordinate = parameterization.current_coordinate(0.03)
        coordinate[0] = -0.4
        coordinate[1:4] = (0.2, -0.1, 0.3)
        coordinate[4:7] = (0.4, -0.35, 0.2)
        coordinate[10:13] = (0.2, -0.15, 0.1)
        coordinate[13:15] = (np.log(0.6), np.log(1.7))
        decoded = parameterization.decode(coordinate)
        self.assertGreater(decoded.parameters.mass, 0.0)
        self.assertTrue(np.all(np.linalg.eigvalsh(decoded.parameters.inertia) > 0.0))
        self.assertTrue(np.all(decoded.parameters.force_effectiveness > 0.0))
        self.assertAlmostEqual(
            float(np.prod(decoded.parameters.force_effectiveness)), 1.0
        )
        self.assertGreater(decoded.actuator_parameters.thrust_time_constant, 0.0)
        self.assertGreater(decoded.actuator_parameters.gimbal_time_constant, 0.0)
        self.assertEqual(decoded.delay, 0.03)

    def test_physical_search_chart_round_trip(self):
        parameterization = estimator.PhysicalSearchParameterization(
            VehicleParameters.nominal()
        )
        coordinate = parameterization.current_coordinate(0.027)
        coordinate[: estimator.SMOOTH_DIMENSION] = np.linspace(
            -0.04, 0.04, estimator.SMOOTH_DIMENSION
        )
        decoded = parameterization.decode(coordinate)
        np.testing.assert_allclose(
            parameterization.encode(
                decoded.parameters,
                decoded.actuator_parameters,
                decoded.delay,
            ),
            coordinate,
            rtol=2.0e-12,
            atol=2.0e-12,
        )

    def test_pid_gate_rejects_out_of_band_plant_response(self):
        nominal = VehicleParameters.nominal()
        parameterization = estimator.PhysicalSearchParameterization(nominal)
        coordinate = parameterization.current_coordinate(0.01)
        coordinate[0] = np.log(1.5)
        decoded = parameterization.decode(coordinate)
        gate = estimator.PidGainGate.from_scale_band(
            np.ones((4, 3)), 0.8, 1.2
        )
        diagnostic = gate.evaluate(
            decoded.parameters, nominal, GrapeGeometry.grape()
        )
        self.assertFalse(diagnostic["valid"])
        self.assertTrue(np.any(diagnostic["gains"] > gate.upper))

    def test_physical_chart_analytic_jacobian_matches_central_difference(self):
        parameterization = estimator.PhysicalSearchParameterization(
            VehicleParameters.nominal()
        )
        coordinate = parameterization.current_coordinate(0.03)
        coordinate[: estimator.SMOOTH_DIMENSION] = np.asarray(
            (
                0.04,
                -0.02,
                0.03,
                0.01,
                0.02,
                -0.01,
                0.015,
                0.003,
                -0.002,
                0.001,
                0.02,
                -0.01,
                0.015,
                np.log(1.2),
                np.log(0.8),
            )
        )
        decoded, jacobian = parameterization.decode_with_jacobian(coordinate)
        analytic_blocks = (
            jacobian.mass[None, :],
            jacobian.inertia.reshape(9, estimator.SMOOTH_DIMENSION),
            jacobian.cog_offset,
            jacobian.force_effectiveness,
            jacobian.thrust_time_constant[None, :],
            jacobian.gimbal_time_constant[None, :],
        )

        def physical_vector(value):
            point = parameterization.decode(value)
            return np.concatenate(
                (
                    np.asarray((point.parameters.mass,)),
                    point.parameters.inertia.ravel(),
                    point.parameters.cog_offset,
                    point.parameters.force_effectiveness,
                    np.asarray(
                        (
                            point.actuator_parameters.thrust_time_constant,
                            point.actuator_parameters.gimbal_time_constant,
                        )
                    ),
                )
            )

        analytic = np.vstack(analytic_blocks)
        finite_difference = np.empty_like(analytic)
        for column in range(estimator.SMOOTH_DIMENSION):
            step = 1.0e-6
            plus = coordinate.copy()
            minus = coordinate.copy()
            plus[column] += step
            minus[column] -= step
            finite_difference[:, column] = (
                physical_vector(plus) - physical_vector(minus)
            ) / (2.0 * step)
        np.testing.assert_allclose(
            analytic,
            finite_difference,
            rtol=2.0e-8,
            atol=2.0e-10,
        )
        self.assertGreater(decoded.parameters.mass, 0.0)


class SobolSelectionTests(unittest.TestCase):
    def setUp(self):
        self.bounds = estimator.SearchBounds(
            np.zeros(estimator.SEARCH_DIMENSION),
            np.ones(estimator.SEARCH_DIMENSION),
        )

    def test_sobol_is_reproducible_and_keeps_exact_current_point(self):
        current = np.full(estimator.SEARCH_DIMENSION, 0.25)
        first, first_sources = estimator.generate_sobol_coordinates(
            self.bounds, power=3, seed=7, current_coordinate=current
        )
        second, second_sources = estimator.generate_sobol_coordinates(
            self.bounds, power=3, seed=7, current_coordinate=current
        )
        self.assertEqual(first.shape, (9, estimator.SEARCH_DIMENSION))
        np.testing.assert_array_equal(first[0], current)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_sources, second_sources)

    def test_default_design_retains_at_least_sixteen_physical_points(self):
        arguments = estimator.create_argument_parser().parse_args([])
        bounds = estimator._search_bounds(arguments)
        nominal = VehicleParameters.nominal()
        parameterization = estimator.PhysicalSearchParameterization(nominal)
        coordinates, _sources = estimator.generate_sobol_coordinates(
            bounds,
            power=arguments.sobol_power,
            seed=arguments.sobol_seed,
            current_coordinate=parameterization.current_coordinate(
                arguments.command_delay
            ),
        )
        gate = estimator.PidGainGate.disabled(np.ones((4, 3)))
        valid_count = 0
        for coordinate in coordinates:
            decoded = parameterization.decode(coordinate)
            if decoded.inertia_triangle_margin < 0.0:
                continue
            if gate.evaluate(
                decoded.parameters, nominal, GrapeGeometry.grape()
            )["valid"]:
                valid_count += 1
        self.assertGreaterEqual(valid_count, arguments.local_start_count)

    def test_disabled_pid_gate_reports_gains_without_rejection(self):
        nominal = VehicleParameters.nominal()
        parameterization = estimator.PhysicalSearchParameterization(nominal)
        coordinate = parameterization.current_coordinate(0.01)
        coordinate[0] = np.log(1.5)
        gate = estimator.PidGainGate.disabled(np.ones((4, 3)))
        diagnostic = gate.evaluate(
            parameterization.decode(coordinate).parameters,
            nominal,
            GrapeGeometry.grape(),
        )
        self.assertTrue(diagnostic["valid"])
        self.assertTrue(np.any(diagnostic["gains"] > gate.current))
        np.testing.assert_array_equal(
            diagnostic["constraint_residual"], np.zeros(24)
        )

    def test_greedy_selection_rejects_near_duplicate_basin(self):
        coordinates = np.zeros((4, estimator.SEARCH_DIMENSION))
        coordinates[1] = 0.01
        coordinates[2] = 0.50
        coordinates[3] = 0.90
        selected = estimator.select_diverse_candidate_indices(
            coordinates,
            np.asarray((1.0, 1.1, 1.2, 1.3)),
            (0, 1, 2, 3),
            self.bounds,
            count=3,
            minimum_distance=0.2,
        )
        self.assertEqual(selected, (0, 2, 3))

    def test_current_point_can_be_required_as_a_local_seed(self):
        coordinates = np.zeros((4, estimator.SEARCH_DIMENSION))
        coordinates[1] = 0.01
        coordinates[2] = 0.50
        coordinates[3] = 0.90
        selected = estimator.select_diverse_candidate_indices(
            coordinates,
            np.asarray((100.0, 1.0, 2.0, 3.0)),
            (0, 1, 2, 3),
            self.bounds,
            count=3,
            minimum_distance=0.2,
            required_indices=(0,),
        )
        self.assertEqual(selected, (2, 3, 0))

    def test_final_choice_uses_boundary_count_only_within_loss_window(self):
        records = (
            {
                "rank": 1,
                "optimizer_success": True,
                "valid": True,
                "trajectory_loss": 100.0,
                "boundary": {"near_boundary_count": 2, "proximity_score": 2.0},
            },
            {
                "rank": 2,
                "optimizer_success": True,
                "valid": True,
                "trajectory_loss": 100.5,
                "boundary": {"near_boundary_count": 0, "proximity_score": 0.0},
            },
            {
                "rank": 3,
                "optimizer_success": True,
                "valid": True,
                "trajectory_loss": 103.0,
                "boundary": {"near_boundary_count": 0, "proximity_score": 0.0},
            },
        )
        selected, diagnostic = estimator.choose_final_local_record(
            records, loss_tolerance_fraction=0.01
        )
        self.assertEqual(selected["rank"], 2)
        self.assertEqual(diagnostic["absolute_best_rank"], 1)
        self.assertEqual(diagnostic["near_minimum_count"], 2)

    def test_baseline_incumbent_is_an_unconditional_loss_guard(self):
        baseline_record = {"trajectory_loss": 10.0, "source": "baseline"}
        worse_record = {"trajectory_loss": 11.0, "source": "sobol"}
        better_record = {"trajectory_loss": 9.0, "source": "sobol"}
        selected, baseline_selected = estimator.choose_with_baseline_incumbent(
            baseline_record, worse_record
        )
        self.assertIs(selected, baseline_record)
        self.assertTrue(baseline_selected)
        selected, baseline_selected = estimator.choose_with_baseline_incumbent(
            baseline_record, better_record
        )
        self.assertIs(selected, better_record)
        self.assertFalse(baseline_selected)

    def test_boundary_diagnostic_includes_triangle_and_pid_constraints(self):
        coordinate = np.full(estimator.SEARCH_DIMENSION, 0.5)
        gate = estimator.PidGainGate.from_scale_band(
            np.ones((4, 3)), 0.8, 1.2
        )
        gains = np.ones((4, 3))
        gains[1] *= 0.801
        diagnostic = estimator._boundary_diagnostic(
            coordinate,
            self.bounds,
            0.02,
            physical={"normalized_inertia_triangle_margin": 1.0e-7},
            pid={"gains": gains},
            pid_gate=gate,
            inertia_triangle_proximity_fraction=1.0e-4,
        )
        self.assertEqual(
            diagnostic["near_boundary_names"],
            ["inertia_triangle_inequality", "pid_gain_z"],
        )
        self.assertEqual(diagnostic["near_boundary_count"], 2)


@unittest.skipUnless(estimator.baseline.DEFAULT_BAG.is_file(), "sample bag absent")
class AnalyticTrajectoryJacobianIntegrationTests(unittest.TestCase):
    def test_full_residual_jacobian_on_recorded_flight(self):
        arguments = estimator.create_argument_parser().parse_args(
            ["--start", "19.0", "--end", "19.15"]
        )
        flight = estimator.baseline.load_flight_data(
            str(arguments.bag),
            start_local=arguments.start,
            end_local=arguments.end,
            include_fc_specific_force=True,
            compute_sha256=False,
        )
        parameterization = estimator.PhysicalSearchParameterization(
            VehicleParameters.nominal()
        )
        evaluator = estimator.CandidateEvaluator(
            flight=flight,
            parameterization=parameterization,
            pid_gate=estimator.PidGainGate.disabled(
                flight.controller_snapshot.gains
            ),
            bounds=estimator._search_bounds(arguments),
            sample_step=arguments.sample_step,
            integration_step=arguments.integration_step,
            prior_weight=arguments.prior_weight,
            pid_penalty_weight=arguments.pid_constraint_penalty,
        )
        current = parameterization.current_coordinate(arguments.command_delay)
        coordinates, _sources = estimator.generate_sobol_coordinates(
            evaluator.bounds,
            power=arguments.sobol_power,
            seed=arguments.sobol_seed,
            current_coordinate=current,
        )
        for coordinate in (current, coordinates[242]):
            fixed_delay = coordinate[estimator.DELAY_INDEX]
            smooth = coordinate[: estimator.SMOOTH_DIMENSION]
            problem = evaluator.make_problem(fixed_delay)
            residual, jacobian = evaluator.optimization_residual_and_jacobian(
                problem, smooth, fixed_delay
            )
            np.testing.assert_allclose(
                residual,
                evaluator.optimization_residual(
                    problem, smooth, fixed_delay
                ),
                rtol=0.0,
                atol=2.0e-14,
            )
            finite_difference = np.empty_like(jacobian)
            for column in range(estimator.SMOOTH_DIMENSION):
                step = 1.0e-6
                plus = smooth.copy()
                minus = smooth.copy()
                plus[column] += step
                minus[column] -= step
                finite_difference[:, column] = (
                    evaluator.optimization_residual(
                        problem, plus, fixed_delay
                    )
                    - evaluator.optimization_residual(
                        problem, minus, fixed_delay
                    )
                ) / (2.0 * step)
            np.testing.assert_allclose(
                jacobian,
                finite_difference,
                rtol=1.0e-6,
                atol=2.0e-8,
            )


if __name__ == "__main__":
    unittest.main()
