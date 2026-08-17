from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from _support import MINIMAL
from grape_param_estim.controller_config import PID_GROUPS
from grape_param_estim.gimbalrotor_pid_postprocess import (
    AllocationDiagnostics,
    EstimatorResult,
    load_estimator_result,
    load_vehicle_model,
    source_compatible_pseudoinverse,
)
import gimbalrotor_pid_postprocess_sensitivity as sensitivity
from gimbalrotor_pid_postprocess_sensitivity import (
    CenteredScaleFreeSpdChart,
    SensitivityArtifacts,
    _json_condition_number,
    analyze_eigen_directions,
    centered_scale_free_spd_pushforward_jacobian,
    evaluate_static_scales,
    load_sensitivity_artifacts,
    plant_from_coordinate,
    prepare_sampling_coordinates,
    sensitivity_group_scale,
)
from single_bag_savgol_core import (
    SiParameterChart,
    common_scale_quotient_basis,
)


FAILURE2_DIRECTORY = (
    MINIMAL
    / "outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1"
    / "prior_ablation"
    / "single_rosbag_2_nominal_pseudo_conditioning_production_20260817"
    / "cases/prior_free"
)


class GimbalrotorPidCoordinateChartTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.model = load_vehicle_model(MINIMAL / "grape_vehicle_model.json")
        self.estimator_chart = SiParameterChart(self.model.parameters)
        self.basis = common_scale_quotient_basis()
        self.center_coordinate = np.asarray(
            (
                0.03,
                0.10,
                -0.05,
                0.04,
                0.025,
                -0.018,
                0.012,
                0.006,
                -0.004,
                0.008,
                0.04,
                -0.03,
                0.02,
                -0.01,
            ),
            dtype=float,
        )
        self.center_plant = plant_from_coordinate(
            self.estimator_chart, self.center_coordinate, 0.2
        )
        self.result = EstimatorResult(
            source_path=self.directory / "result.json",
            source_commit="a" * 40,
            case_name="prior_free",
            overall_case_status="completed",
            optimization_status="completed",
            prior={"active": False},
            plant=self.center_plant,
            warnings=(),
        )

    def _artifacts(self, covariance):
        return SensitivityArtifacts(
            source_path=self.directory / "arrays.npz",
            physical_coordinate=self.center_coordinate,
            coordinate_source="physical_coordinate",
            quotient_basis=self.basis,
            covariance=np.asarray(covariance, dtype=float),
            covariance_mode="conservative_fusion",
            covariance_source="synthetic_test_covariance",
        )

    def test_centered_scale_free_spd_chart_round_trip(self):
        chart = CenteredScaleFreeSpdChart.from_plant(self.center_plant)
        coordinate = np.asarray(
            (
                0.08,
                -0.05,
                0.02,
                0.03,
                -0.04,
                0.01,
                0.006,
                -0.003,
                0.004,
                0.05,
                -0.02,
                0.03,
                -0.04,
            ),
            dtype=float,
        )
        decoded = chart.decode(coordinate)
        recovered = chart.encode(decoded)
        np.testing.assert_allclose(
            recovered, coordinate, rtol=2.0e-11, atol=2.0e-11
        )

    def test_centered_chart_origin_is_fitted_scale_free_plant(self):
        chart = CenteredScaleFreeSpdChart.from_plant(self.center_plant)
        decoded = chart.decode(np.zeros(13))
        np.testing.assert_allclose(
            decoded.inertia_over_mass,
            self.center_plant.inertia_over_mass,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            decoded.cog_position_body,
            self.center_plant.cog_position_body,
            rtol=0.0,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            decoded.force_effectiveness_over_mass,
            self.center_plant.force_effectiveness_over_mass,
            rtol=2.0e-12,
            atol=2.0e-12,
        )

    def test_analytic_covariance_pushforward_jacobian_matches_finite_difference(self):
        centered = CenteredScaleFreeSpdChart.from_plant(self.center_plant)
        analytic = centered_scale_free_spd_pushforward_jacobian(
            estimator_chart=self.estimator_chart,
            center_coordinate=self.center_coordinate,
            quotient_basis=self.basis,
            centered_chart=centered,
        )
        step = 1.0e-6
        numerical = np.zeros((13, 13))
        for column in range(13):
            direction = self.basis[:, column]
            plus = plant_from_coordinate(
                self.estimator_chart,
                self.center_coordinate + step * direction,
                0.2,
            )
            minus = plant_from_coordinate(
                self.estimator_chart,
                self.center_coordinate - step * direction,
                0.2,
            )
            numerical[:, column] = (
                centered.encode(plus) - centered.encode(minus)
            ) / (2.0 * step)
        np.testing.assert_allclose(
            analytic, numerical, rtol=3.0e-6, atol=3.0e-8
        )

    def test_first_order_pid_sigma_is_coordinate_invariant(self):
        rng = np.random.default_rng(7)
        factor = rng.normal(size=(13, 13))
        covariance = 2.0e-6 * (factor @ factor.T) / 13.0
        artifacts = self._artifacts(covariance)
        estimator = analyze_eigen_directions(
            result=self.result,
            artifacts=artifacts,
            model=self.model,
            sigma_multiple=1.0,
            coordinate_mode="estimator_quotient",
            derivative_sigma_fraction=1.0e-4,
        )
        centered = analyze_eigen_directions(
            result=self.result,
            artifacts=artifacts,
            model=self.model,
            sigma_multiple=1.0,
            coordinate_mode="centered_scale_free_spd",
            derivative_sigma_fraction=1.0e-4,
        )
        for group in PID_GROUPS:
            first = estimator["group_summary"][group][
                "linearized_one_sigma"
            ]
            second = centered["group_summary"][group][
                "linearized_one_sigma"
            ]
            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertAlmostEqual(first, second, delta=3.0e-6)

    def test_centered_covariance_is_analytic_pushforward(self):
        covariance = np.diag(np.linspace(1.0e-7, 1.3e-6, 13))
        artifacts = self._artifacts(covariance)
        sampling = prepare_sampling_coordinates(
            result=self.result,
            artifacts=artifacts,
            model=self.model,
            coordinate_mode="centered_scale_free_spd",
        )
        transform = sampling.pushforward_jacobian
        expected = transform @ covariance @ transform.T
        expected = 0.5 * (expected + expected.T)
        np.testing.assert_allclose(
            sampling.covariance, expected, rtol=0.0, atol=1.0e-15
        )

    def test_rank_loss_is_diagnostic_not_an_early_rejection(self):
        nominal = sensitivity.build_nominal_controller_allocation(self.model)
        projection_direction = np.ones(6) / np.sqrt(6.0)
        projection = (
            np.eye(6)
            - np.outer(projection_direction, projection_direction)
        )
        rank_five_matrix = projection @ nominal.matrix
        singular = np.linalg.svd(rank_five_matrix, compute_uv=False)
        fake = AllocationDiagnostics(
            matrix=rank_five_matrix,
            singular_values=singular,
            source_threshold_rank=5,
            condition_number=float("inf"),
        )
        nominal_pseudoinverse = source_compatible_pseudoinverse(
            nominal.matrix
        )
        with patch.object(
            sensitivity,
            "build_real_scale_free_allocation",
            return_value=fake,
        ):
            evaluation = evaluate_static_scales(
                self.center_plant,
                self.model,
                nominal_pseudoinverse=nominal_pseudoinverse,
                characteristic_length_m=0.2,
            )
        self.assertTrue(np.isinf(evaluation.allocation_condition_number))
        self.assertEqual(evaluation.allocation_source_threshold_rank, 5)
        for group in PID_GROUPS:
            self.assertTrue(np.isfinite(evaluation.scales[group]))

        self.assertEqual(
            _json_condition_number(evaluation.allocation_condition_number),
            "infinity",
        )
        json.dumps(
            {
                "A_real_condition_number": _json_condition_number(
                    evaluation.allocation_condition_number
                )
            },
            allow_nan=False,
        )

    def test_negative_finite_group_scale_is_retained(self):
        scale = sensitivity_group_scale(-np.eye(6), (0,))
        self.assertEqual(scale, -1.0)

    def test_current_failure2_default_derivative_is_coordinate_invariant(self):
        result_path = FAILURE2_DIRECTORY / "result.json"
        arrays_path = FAILURE2_DIRECTORY / "arrays.npz"
        if not result_path.is_file() or not arrays_path.is_file():
            self.skipTest("committed failure2 artifacts are unavailable")
        result = load_estimator_result(result_path)
        artifacts = load_sensitivity_artifacts(
            arrays_path, covariance_mode="conservative_fusion"
        )
        estimator = analyze_eigen_directions(
            result=result,
            artifacts=artifacts,
            model=self.model,
            sigma_multiple=0.5,
            coordinate_mode="estimator_quotient",
        )
        centered = analyze_eigen_directions(
            result=result,
            artifacts=artifacts,
            model=self.model,
            sigma_multiple=0.5,
            coordinate_mode="centered_scale_free_spd",
        )
        for group in PID_GROUPS:
            first = estimator["group_summary"][group][
                "linearized_one_sigma"
            ]
            second = centered["group_summary"][group][
                "linearized_one_sigma"
            ]
            relative = abs(first - second) / max(abs(first), abs(second))
            self.assertLess(relative, 1.0e-4)

        sampling = centered["eigen_sampling"]
        self.assertEqual(
            sampling["valid_sample_count"]
            + sampling["invalid_sample_count"],
            sampling["expected_sample_count"],
        )
        for direction in sampling["directions"]:
            for side in ("finite_minus", "finite_plus"):
                sample = direction[side]
                if not sample["valid"]:
                    self.assertNotIn("rank deficient", sample["message"])
        json.dumps(centered, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
