from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from _support import MINIMAL
from grape_param_estim.controller_config import PID_GAIN_NAMES, PID_GROUPS
from grape_param_estim.gimbalrotor_pid_postprocess import (
    PostprocessInputError,
    PostprocessNumericalError,
    load_estimator_result,
    load_vehicle_model,
)
from gimbalrotor_pid_monte_carlo_postprocess import (
    _gain_quantiles,
    _quantile_summary,
    load_static_postprocess_baseline,
    sample_pid_gain_distribution,
)
from gimbalrotor_pid_postprocess_sensitivity import (
    load_sensitivity_artifacts,
    prepare_sampling_coordinates,
)
from single_bag_savgol_core import SiParameterChart, common_scale_quotient_basis
from three_bag_gimbalrotor_pid_monte_carlo_postprocess import build_summary


class GimbalrotorPidMonteCarloPostprocessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.result = self.directory / "result.json"
        self.arrays = self.directory / "arrays.npz"
        self.static = self.directory / "pid_gain_postprocess.json"
        self.model = load_vehicle_model(MINIMAL / "grape_vehicle_model.json")
        self.chart = SiParameterChart(self.model.parameters)
        self.center_parameters = self.chart.decode(np.zeros(14))
        self._write_result()
        self._write_arrays(np.zeros((13, 13)))
        self._write_static()

    def _write_result(self):
        parameters = self.center_parameters
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
                        parameters.inertia / parameters.mass
                    ).tolist(),
                    "cog_position_body_m": parameters.cog_offset.tolist(),
                    "force_effectiveness_over_mass": (
                        parameters.force_effectiveness / parameters.mass
                    ).tolist(),
                },
                "rotor_lag_seconds": 0.2,
            },
        }
        self.result.write_text(json.dumps(payload), encoding="utf-8")

    def _write_arrays(self, covariance):
        basis = common_scale_quotient_basis()
        np.savez_compressed(
            self.arrays,
            quotient_basis=basis,
            physical_coordinate=np.zeros(14),
            quotient_coordinate=np.zeros(13),
            quotient_covariance_conservative_fusion=np.asarray(covariance),
        )

    def _write_static(self, center_scale=1.0):
        gains = {
            "xy": {"p_gain": 3.0, "i_gain": 0.1, "d_gain": 1.0},
            "z": {"p_gain": 5.0, "i_gain": 1.0, "d_gain": 2.5},
            "roll_pitch": {"p_gain": 10.0, "i_gain": 1.0, "d_gain": 8.0},
            "yaw": {"p_gain": 4.0, "i_gain": 1.0, "d_gain": 2.0},
        }
        payload = {
            "schema": "grape-param-estim/gimbalrotor-pid-postprocess/v1",
            "source_commit": "b" * 40,
            "input": {
                "estimator_source_commit": "a" * 40,
                "estimator_case_name": "prior_free",
            },
            "controller_gain_snapshot": {
                "source": "synthetic_test",
                "gains": gains,
            },
            "gain_groups": {
                group: {"scale": float(center_scale)} for group in PID_GROUPS
            },
        }
        self.static.write_text(json.dumps(payload), encoding="utf-8")

    def test_quantile_summary_retains_nonpositive_samples(self):
        item = _quantile_summary(np.asarray((-2.0, -1.0, 1.0, 4.0)))
        self.assertEqual(item["min"], -2.0)
        self.assertAlmostEqual(item["nonpositive_fraction"], 0.5)

    def test_gain_quantiles_preserve_common_group_scale(self):
        baseline = load_static_postprocess_baseline(self.static)
        scale_summary = {
            group: {
                "quantiles": {
                    "q025": 1.0,
                    "q16": 2.0,
                    "q50": 3.0,
                    "q84": 4.0,
                    "q975": 5.0,
                }
            }
            for group in PID_GROUPS
        }
        proposal = _gain_quantiles(baseline, scale_summary)
        self.assertEqual(proposal["roll_pitch"]["gains"]["p_gain"]["median"], 30.0)
        self.assertEqual(
            proposal["roll_pitch"]["gains"]["d_gain"]["range_95"],
            [8.0, 40.0],
        )

    def test_static_baseline_requires_matching_center(self):
        self._write_static(center_scale=1.5)
        with self.assertRaises(PostprocessInputError):
            sample_pid_gain_distribution(
                result_path=self.result,
                arrays_path=self.arrays,
                static_postprocess_path=self.static,
                vehicle_model_path=MINIMAL / "grape_vehicle_model.json",
                sample_count=8,
                seed=0,
            )

    def test_zero_covariance_returns_point_mass_and_exact_gain_proposal(self):
        report, samples = sample_pid_gain_distribution(
            result_path=self.result,
            arrays_path=self.arrays,
            static_postprocess_path=self.static,
            vehicle_model_path=MINIMAL / "grape_vehicle_model.json",
            sample_count=32,
            seed=7,
        )
        self.assertEqual(report["sampling"]["valid_count"], 32)
        for group in PID_GROUPS:
            q = report["gain_scale_distribution"][group]["quantiles"]
            self.assertAlmostEqual(q["q025"], 1.0, places=11)
            self.assertAlmostEqual(q["q50"], 1.0, places=11)
            self.assertAlmostEqual(q["q975"], 1.0, places=11)
            for gain in PID_GAIN_NAMES:
                item = report["pid_gain_proposal"][group]["gains"][gain]
                self.assertAlmostEqual(item["median"], item["recorded"], places=11)
        self.assertEqual(samples["gain_scale_samples"].shape, (32, 4))
        self.assertEqual(samples["scale_free_samples"].shape, (32, 13))

    def test_fixed_seed_is_reproducible(self):
        self._write_arrays(1.0e-8 * np.eye(13))
        first, first_samples = sample_pid_gain_distribution(
            result_path=self.result,
            arrays_path=self.arrays,
            static_postprocess_path=self.static,
            vehicle_model_path=MINIMAL / "grape_vehicle_model.json",
            sample_count=24,
            seed=11,
        )
        second, second_samples = sample_pid_gain_distribution(
            result_path=self.result,
            arrays_path=self.arrays,
            static_postprocess_path=self.static,
            vehicle_model_path=MINIMAL / "grape_vehicle_model.json",
            sample_count=24,
            seed=11,
        )
        np.testing.assert_allclose(
            first_samples["quotient_delta_samples"],
            second_samples["quotient_delta_samples"],
        )
        np.testing.assert_allclose(
            first_samples["gain_scale_samples"],
            second_samples["gain_scale_samples"],
            equal_nan=True,
        )
        self.assertEqual(
            first["gain_scale_distribution"],
            second["gain_scale_distribution"],
        )

    def test_known_estimator_chart_roundoff_is_numerical_not_input_error(self):
        result = load_estimator_result(self.result)
        artifacts = load_sensitivity_artifacts(self.arrays)
        sampling = prepare_sampling_coordinates(
            result=result,
            artifacts=artifacts,
            model=self.model,
            coordinate_mode="estimator_quotient",
        )
        with patch(
            "gimbalrotor_pid_postprocess_sensitivity.plant_from_coordinate",
            side_effect=ValueError(
                "inertia must be symmetric positive definite"
            ),
        ):
            with self.assertRaises(PostprocessNumericalError):
                sampling.decode(np.zeros(13))
        with patch(
            "gimbalrotor_pid_postprocess_sensitivity.plant_from_coordinate",
            side_effect=ValueError("unexpected programming failure"),
        ):
            with self.assertRaisesRegex(
                ValueError, "unexpected programming failure"
            ):
                sampling.decode(np.zeros(13))

    def test_three_bag_summary_keeps_each_case_distribution(self):
        def fake_report(center):
            return {
                "sampling": {"requested_count": 100, "valid_count": 100, "invalid_count": 0},
                "warnings": [],
                "center": {"scales": {group: center for group in PID_GROUPS}},
                "gain_scale_distribution": {
                    group: {
                        "standard_deviation": 0.1,
                        "nonpositive_fraction": 0.0,
                        "quantiles": {
                            "q025": center - 0.2,
                            "q16": center - 0.1,
                            "q50": center,
                            "q84": center + 0.1,
                            "q975": center + 0.2,
                        },
                    }
                    for group in PID_GROUPS
                },
                "pid_gain_proposal": {
                    group: {
                        "gains": {
                            gain: {
                                "recorded": 1.0,
                                "median": center,
                                "range_68": [center - 0.1, center + 0.1],
                                "range_95": [center - 0.2, center + 0.2],
                            }
                            for gain in PID_GAIN_NAMES
                        }
                    }
                    for group in PID_GROUPS
                },
                "joint_gain_scale_distribution": {
                    "group_order": list(PID_GROUPS),
                    "covariance": np.eye(4).tolist(),
                    "correlation": np.eye(4).tolist(),
                },
            }

        summary = build_summary(
            {
                "conservative_fusion": {
                    "failure1": fake_report(1.0),
                    "failure2": fake_report(2.0),
                    "success": fake_report(3.0),
                }
            }
        )
        self.assertEqual(
            summary["cases"]["conservative_fusion"]["failure2"]["groups"]["roll_pitch"]["scale_quantiles"]["q50"],
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
