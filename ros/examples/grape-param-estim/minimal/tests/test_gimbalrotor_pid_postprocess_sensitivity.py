from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from _support import MINIMAL
from grape_param_estim.controller_config import PID_GROUPS
from grape_param_estim.gimbalrotor_pid_postprocess import (
    PostprocessInputError,
    load_estimator_result,
    load_vehicle_model,
)
from gimbalrotor_pid_postprocess_sensitivity import (
    DEFAULT_COVARIANCE_MODE,
    SensitivityArtifacts,
    _psd_eigendecomposition,
    analyze_eigen_directions,
    analyze_monte_carlo,
    load_sensitivity_artifacts,
    plant_from_coordinate,
    scale_free_vector,
)
from single_bag_savgol_core import (
    COMMON_SCALE_DIRECTION,
    SiParameterChart,
    common_scale_quotient_basis,
)
from three_bag_gimbalrotor_pid_postprocess_sensitivity_summary import (
    build_summary,
)


FAILURE1_DIRECTORY = (
    MINIMAL
    / "outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1"
    / "prior_ablation"
    / "single_rosbag_1_nominal_pseudo_conditioning_production_20260817"
    / "cases/prior_free"
)
FAILURE1_RESULT = FAILURE1_DIRECTORY / "result.json"
FAILURE1_ARRAYS = FAILURE1_DIRECTORY / "arrays.npz"


class GimbalrotorPidSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.model = load_vehicle_model(MINIMAL / "grape_vehicle_model.json")
        self.chart = SiParameterChart(self.model.parameters)
        self.coordinate = np.zeros(14)
        self.basis = common_scale_quotient_basis()
        self.plant = plant_from_coordinate(
            self.chart, self.coordinate, 0.2
        )
        self.result_path = self.directory / "result.json"
        self._write_result(self.plant)

    def _write_result(self, plant):
        payload = {
            "overall_case_status": "completed",
            "optimization_status": "completed",
            "success": True,
            "case_name": "prior_free",
            "source_commit": "a" * 40,
            "prior": {"active": False},
            "parameters": {
                "scale_free": {
                    "inertia_over_mass_m2": (
                        plant.inertia_over_mass.tolist()
                    ),
                    "cog_position_body_m": (
                        plant.cog_position_body.tolist()
                    ),
                    "force_effectiveness_over_mass": (
                        plant.force_effectiveness_over_mass.tolist()
                    ),
                },
                "rotor_lag_seconds": plant.rotor_lag_seconds,
            },
        }
        self.result_path.write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def _write_arrays(
        self, covariance, coordinate=None, *, include_physical=True
    ):
        path = self.directory / "arrays.npz"
        selected_coordinate = (
            self.coordinate if coordinate is None else coordinate
        )
        payload = {
            "quotient_basis": self.basis,
            "quotient_coordinate": self.basis.T
            @ np.asarray(selected_coordinate),
            "quotient_covariance_conservative_fusion": np.asarray(
                covariance
            ),
        }
        if include_physical:
            payload["physical_coordinate"] = np.asarray(
                selected_coordinate
            )
        np.savez_compressed(path, **payload)
        return path

    def _run(self, covariance, coordinate=None):
        arrays = self._write_arrays(covariance, coordinate=coordinate)
        artifacts = load_sensitivity_artifacts(
            arrays, covariance_mode=DEFAULT_COVARIANCE_MODE
        )
        result = load_estimator_result(self.result_path)
        return analyze_eigen_directions(
            result=result,
            artifacts=artifacts,
            model=self.model,
            sigma_multiple=1.0,
        )

    def test_zero_covariance_has_27_valid_points_and_zero_scale_spread(self):
        report = self._run(np.zeros((13, 13)))
        self.assertEqual(
            report["eigen_sampling"]["expected_sample_count"], 27
        )
        self.assertEqual(report["eigen_sampling"]["valid_sample_count"], 27)
        self.assertEqual(report["eigen_sampling"]["invalid_sample_count"], 0)
        for group in PID_GROUPS:
            item = report["group_summary"][group]
            self.assertAlmostEqual(item["center_scale"], 1.0, places=12)
            self.assertAlmostEqual(
                item["linearized_one_sigma"], 0.0, places=15
            )
            self.assertAlmostEqual(
                item["sigma_point_min"], 1.0, places=12
            )
            self.assertAlmostEqual(
                item["sigma_point_max"], 1.0, places=12
            )

    def test_small_full_rank_covariance_produces_finite_local_spread(self):
        report = self._run(1.0e-6 * np.eye(13))
        self.assertEqual(report["eigen_sampling"]["valid_sample_count"], 27)
        positive_count = 0
        for group in PID_GROUPS:
            sigma = report["group_summary"][group][
                "linearized_one_sigma"
            ]
            self.assertTrue(np.isfinite(sigma))
            if sigma > 0.0:
                positive_count += 1
        self.assertGreaterEqual(positive_count, 2)

    def test_historical_arrays_reconstruct_zero_gauge_coordinate(self):
        arrays = self._write_arrays(
            np.zeros((13, 13)), include_physical=False
        )
        artifacts = load_sensitivity_artifacts(arrays)
        np.testing.assert_allclose(
            artifacts.physical_coordinate,
            self.basis @ (self.basis.T @ self.coordinate),
            rtol=0.0,
            atol=1.0e-15,
        )
        self.assertIn("zero common-scale gauge", artifacts.coordinate_source)

    def test_common_scale_gauge_shift_does_not_change_sensitivity(self):
        covariance = 2.0e-7 * np.eye(13)
        first = self._run(covariance)
        shifted_coordinate = (
            self.coordinate + 0.37 * np.asarray(COMMON_SCALE_DIRECTION)
        )
        shifted_plant = plant_from_coordinate(
            self.chart, shifted_coordinate, 0.2
        )
        first_plant = scale_free_vector(self.plant)
        second_plant = scale_free_vector(shifted_plant)
        self.assertTrue(
            np.allclose(first_plant, second_plant, rtol=2e-12, atol=2e-12)
        )
        self._write_result(shifted_plant)
        second = self._run(covariance, coordinate=shifted_coordinate)
        for group in PID_GROUPS:
            first_item = first["group_summary"][group]
            second_item = second["group_summary"][group]
            self.assertAlmostEqual(
                first_item["center_scale"],
                second_item["center_scale"],
                places=11,
            )
            self.assertAlmostEqual(
                first_item["linearized_one_sigma"],
                second_item["linearized_one_sigma"],
                places=10,
            )

    def test_materially_indefinite_covariance_is_rejected(self):
        covariance = np.eye(13)
        covariance[0, 0] = -1.0
        with self.assertRaises(PostprocessInputError):
            _psd_eigendecomposition(covariance)

    def test_tiny_negative_covariance_eigenvalue_is_clipped(self):
        covariance = np.zeros((13, 13))
        covariance[0, 0] = -1.0e-15
        eigenvalues, _vectors, _tolerance = _psd_eigendecomposition(
            covariance
        )
        self.assertTrue(np.all(eigenvalues >= 0.0))
        self.assertEqual(eigenvalues[-1], 0.0)

    def test_monte_carlo_is_reproducible_for_fixed_seed(self):
        arrays = self._write_arrays(1.0e-6 * np.eye(13))
        artifacts = load_sensitivity_artifacts(arrays)
        result = load_estimator_result(self.result_path)
        eigen = analyze_eigen_directions(
            result=result,
            artifacts=artifacts,
            model=self.model,
            sigma_multiple=1.0,
        )
        first = analyze_monte_carlo(
            result=result,
            artifacts=artifacts,
            model=self.model,
            sample_count=32,
            seed=7,
            characteristic_length_m=eigen[
                "characteristic_length_m"
            ],
        )
        second = analyze_monte_carlo(
            result=result,
            artifacts=artifacts,
            model=self.model,
            sample_count=32,
            seed=7,
            characteristic_length_m=eigen[
                "characteristic_length_m"
            ],
        )
        self.assertEqual(first, second)

    def test_three_bag_summary_separates_within_and_between_scales(self):
        def fake_report(scale, sigma):
            return {
                "schema": (
                    "grape-param-estim/"
                    "gimbalrotor-pid-postprocess-sensitivity/v1"
                ),
                "input": {
                    "covariance_mode": "conservative_fusion",
                    "estimator_result_json": "/tmp/result.json",
                },
                "center_and_eigen_sensitivity": {
                    "group_summary": {
                        group: {
                            "center_scale": scale[group],
                            "linearized_one_sigma": sigma[group],
                            "relative_linearized_one_sigma": (
                                sigma[group] / scale[group]
                            ),
                            "sigma_point_min": scale[group] - sigma[group],
                            "sigma_point_max": scale[group] + sigma[group],
                        }
                        for group in PID_GROUPS
                    }
                },
            }

        first_scale = {group: 1.0 for group in PID_GROUPS}
        second_scale = {group: 1.0 for group in PID_GROUPS}
        third_scale = {group: 1.0 for group in PID_GROUPS}
        third_scale["roll_pitch"] = 3.0
        small_sigma = {group: 0.05 for group in PID_GROUPS}
        summary = build_summary(
            (
                ("a", fake_report(first_scale, small_sigma)),
                ("b", fake_report(second_scale, small_sigma)),
                ("c", fake_report(third_scale, small_sigma)),
            )
        )
        self.assertGreater(
            summary["within_vs_between"]["roll_pitch"][
                "between_to_within_std_ratio"
            ],
            10.0,
        )
        self.assertAlmostEqual(
            summary["within_vs_between"]["xy"][
                "between_bag_center_standard_deviation"
            ],
            0.0,
        )

    def test_current_failure1_artifacts_reproduce_static_center(self):
        if not FAILURE1_ARRAYS.is_file():
            self.skipTest("committed failure1 arrays.npz is unavailable")
        result = load_estimator_result(FAILURE1_RESULT)
        artifacts = load_sensitivity_artifacts(
            FAILURE1_ARRAYS,
            covariance_mode="conservative_fusion",
        )
        report = analyze_eigen_directions(
            result=result,
            artifacts=artifacts,
            model=self.model,
            sigma_multiple=1.0,
        )
        expected = {
            "xy": 1.15204476,
            "z": 1.16976482,
            "roll_pitch": 3.52877431,
            "yaw": 3.37886790,
        }
        for group, value in expected.items():
            self.assertAlmostEqual(
                report["group_summary"][group]["center_scale"],
                value,
                places=7,
            )
        self.assertEqual(
            report["eigen_sampling"]["expected_sample_count"], 27
        )


if __name__ == "__main__":
    unittest.main()
