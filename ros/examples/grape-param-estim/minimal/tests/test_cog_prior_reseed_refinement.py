from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from _support import MINIMAL
from single_bag_cog_prior_reseed_refinement import (
    PRODUCTION_COG_PRIOR_STD_M,
    build_refinement_arguments,
    condition_chart_gaussian_on_cog_prior,
    execute_prior_free_refinement,
    load_completed_baseline_case,
    refinement_argument_audit,
    validate_production_conditioning,
    verify_baseline_files_unchanged,
)
from single_bag_savgol_core import (
    COMMON_SCALE_DIRECTION,
    SiParameterChart,
    load_vehicle_model,
)
from three_bag_cog_prior_reseed_consensus import (
    _comparison_rows,
    _human_summary_text,
    _write_report,
)


def _random_spd(dimension: int, seed: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((dimension, dimension))
    return factor @ factor.T + 0.3 * np.eye(dimension)


def _baseline_arguments() -> dict:
    return {
        "bag": "/tmp/example.bag",
        "bag_json": "bag_jsons/example.json",
        "bag_id": "example",
        "bag_start": 1.0,
        "bag_end": 2.0,
        "skip_bag_sha256": False,
        "vehicle_model": "grape_vehicle_model.json",
        "sg_window": 1.0,
        "sg_degree": 5,
        "covariance_mode": "identity",
        "naive_so3_derivatives": False,
        "gimbal_source": "measured_sg",
        "lag_mode": "estimated",
        "initial_rotor_lag": None,
        "initial_rotor_lag_multiplier": 1.0,
        "fixed_rotor_lag": None,
        "lag_continuation_depth": 9,
        "lag_continuation_schedule": None,
        "disable_lag_continuation": False,
        "smooth_max_nfev": 2000,
        "strict_max_nfev": 2000,
        "disable_kkt": False,
        "solver_type": "custom_kkt_lm",
        "initial_coordinate": [0.0] * 14,
        "scale_initial_offset": 0.0,
        "lm_initial_damping": 1.0e-3,
        "lm_initial_trust_radius": 1.0,
        "lm_maximum_trust_radius": 8.0,
        "lm_minimum_trust_radius": 1.0e-10,
        "lm_acceptance_ratio": 1.0e-4,
        "gtol": 1.0e-8,
        "ftol": float(np.sqrt(np.finfo(float).eps)),
        "xtol": float(np.sqrt(np.finfo(float).eps)),
        "thrust_time_constant": 0.0,
        "gimbal_time_constant": 0.0,
        "minimum_thrust": 1.5,
        "maximum_thrust": 27.6145,
        "maximum_gimbal_angle": 3.14,
        "maximum_gimbal_rate": 6.0,
        "output_root": "/tmp/original-output",
        "run_id": "original",
        "base_plan_commit": "ignored-metadata",
    }


class CogPriorReseedRefinementTests(unittest.TestCase):
    def test_nominal_cog_chart_identity(self):
        model = load_vehicle_model(MINIMAL / "grape_vehicle_model.json")
        decoded = SiParameterChart(model.parameters).decode(np.zeros(14))
        expected = np.asarray(
            (-0.002024708562282, -0.000030526578941, 0.009509749599446)
        )
        self.assertTrue(
            np.allclose(decoded.cog_offset, expected, rtol=0.0, atol=5e-16)
        )
        self.assertTrue(np.array_equal(decoded.cog_offset, model.parameters.cog_offset))
        self.assertEqual(PRODUCTION_COG_PRIOR_STD_M, 0.001)

    def test_zero_cross_covariance_changes_only_cog_block(self):
        covariance = np.diag(np.linspace(0.2, 1.5, 14))
        mean = np.linspace(-0.4, 0.5, 14)
        result = condition_chart_gaussian_on_cog_prior(
            mean, covariance, cog_std_m=0.001
        )
        non_cog = np.asarray(tuple(range(7)) + tuple(range(10, 14)))
        self.assertTrue(
            np.array_equal(result.conditioned_mean[non_cog], mean[non_cog])
        )
        self.assertTrue(
            np.array_equal(
                result.conditioned_covariance[np.ix_(non_cog, non_cog)],
                covariance[np.ix_(non_cog, non_cog)],
            )
        )
        self.assertLess(
            np.linalg.norm(result.conditioned_mean[7:10]),
            np.linalg.norm(mean[7:10]),
        )

    def test_known_cross_covariance_propagates_mean_and_covariance(self):
        covariance = np.eye(14)
        covariance[0, 7] = covariance[7, 0] = 0.4
        covariance[4, 8] = covariance[8, 4] = -0.25
        mean = np.zeros(14)
        mean[7:10] = (0.3, -0.2, 0.1)
        std = 0.2
        result = condition_chart_gaussian_on_cog_prior(
            mean, covariance, cog_std_m=std
        )
        shrink = 1.0 / (1.0 + std**2)
        expected_mean = mean.copy()
        expected_mean[7:10] -= shrink * mean[7:10]
        expected_mean[0] -= 0.4 * shrink * mean[7]
        expected_mean[4] -= -0.25 * shrink * mean[8]
        selector = np.zeros((3, 14))
        selector[:, 7:10] = np.eye(3)
        innovation = selector @ covariance @ selector.T + std**2 * np.eye(3)
        expected_covariance = covariance - (
            covariance
            @ selector.T
            @ np.linalg.solve(innovation, selector @ covariance)
        )
        self.assertTrue(np.allclose(result.conditioned_mean, expected_mean))
        self.assertTrue(
            np.allclose(result.conditioned_covariance, expected_covariance)
        )

    def test_centered_prior_mean_keeps_mean_and_reduces_covariance(self):
        mean = np.linspace(-0.2, 0.3, 14)
        mean[7:10] = 0.0
        covariance = _random_spd(14)
        result = condition_chart_gaussian_on_cog_prior(
            mean, covariance, cog_std_m=0.001
        )
        self.assertTrue(np.array_equal(result.conditioned_mean, mean))
        self.assertLess(
            np.trace(result.conditioned_covariance), np.trace(covariance)
        )

    def test_weak_and_tight_prior_limits(self):
        mean = np.linspace(-0.4, 0.6, 14)
        covariance = _random_spd(14, seed=10)
        weak = condition_chart_gaussian_on_cog_prior(
            mean, covariance, cog_std_m=1.0e8
        )
        self.assertTrue(np.allclose(weak.conditioned_mean, mean, atol=2e-14))
        self.assertTrue(
            np.allclose(weak.conditioned_covariance, covariance, atol=2e-14)
        )
        tight = condition_chart_gaussian_on_cog_prior(
            mean, covariance, cog_std_m=1.0e-12
        )
        self.assertTrue(np.allclose(tight.conditioned_mean[7:10], 0.0, atol=2e-13))

    def test_conditioning_psd_symmetry_and_information_order(self):
        covariance = _random_spd(14, seed=22)
        result = condition_chart_gaussian_on_cog_prior(
            np.linspace(-0.1, 0.2, 14), covariance, cog_std_m=0.001
        )
        self.assertTrue(
            np.array_equal(
                result.conditioned_covariance,
                result.conditioned_covariance.T,
            )
        )
        difference = covariance - result.conditioned_covariance
        tolerance = 14 * np.finfo(float).eps * np.max(np.abs(difference))
        self.assertGreaterEqual(np.linalg.eigvalsh(difference)[0], -tolerance)
        self.assertGreaterEqual(
            np.linalg.eigvalsh(result.conditioned_covariance)[0], -2e-13
        )

    def test_gauge_preservation_without_projection(self):
        gauge = np.asarray(COMMON_SCALE_DIRECTION, dtype=float)
        projector = np.eye(14) - np.outer(gauge, gauge) / (gauge @ gauge)
        covariance = projector @ _random_spd(14, seed=31) @ projector
        mean = projector @ np.linspace(-0.3, 0.4, 14)
        result = condition_chart_gaussian_on_cog_prior(
            mean, covariance, cog_std_m=0.001
        )
        self.assertLess(
            abs(gauge @ (result.conditioned_mean - mean)), 2e-13
        )
        self.assertLess(
            np.linalg.norm(result.conditioned_covariance @ gauge), 3e-13
        )
        validate_production_conditioning(result)

    def test_conditioning_never_uses_full_covariance_pseudoinverse(self):
        covariance = _random_spd(14)
        with mock.patch(
            "numpy.linalg.pinv", side_effect=AssertionError("pinv forbidden")
        ):
            condition_chart_gaussian_on_cog_prior(
                np.zeros(14), covariance, cog_std_m=0.001
            )

    def test_refinement_arguments_preserve_science_and_keep_all_coordinates(self):
        baseline = _baseline_arguments()
        conditioned = np.asarray(
            (0.7, -44.0, 31.0, 2.5, -1.0, 7.0, 0.2, 0.01, -0.02, 0.03, 4.0, -3.0, 2.0, -1.0)
        )
        arguments = build_refinement_arguments(
            baseline,
            conditioned,
            0.234567,
            output_root=Path("/tmp/refined-output"),
            run_id="refined",
        )
        self.assertTrue(np.array_equal(arguments.initial_coordinate, conditioned))
        self.assertEqual(arguments.initial_rotor_lag, 0.234567)
        self.assertEqual(arguments.scale_initial_offset, 0.0)
        self.assertEqual(arguments.lag_mode, "estimated")
        self.assertIsNone(arguments.fixed_rotor_lag)
        self.assertFalse(arguments.disable_kkt)
        audit = refinement_argument_audit(baseline, arguments)
        self.assertTrue(audit["all_other_scientific_arguments_preserved"])
        self.assertEqual(
            set(audit["allowed_changed_fields"]),
            {
                "initial_coordinate",
                "initial_rotor_lag",
                "output_root",
                "run_id",
            },
        )
        self.assertFalse(any("prior" in key.lower() for key in vars(arguments)))

    def test_existing_estimator_is_called_without_prior_or_fixed_cog(self):
        baseline = _baseline_arguments()
        conditioned = np.linspace(-12.0, 13.0, 14)
        arguments = build_refinement_arguments(
            baseline,
            conditioned,
            0.19,
            output_root=Path("/tmp/refined-output"),
            run_id="refined",
        )
        observed = {}

        def fake_runner(received, *, case_name, output_directory):
            observed["arguments"] = received
            observed["case_name"] = case_name
            observed["output_directory"] = output_directory
            return Path(output_directory), {"status": "completed"}

        output, payload = execute_prior_free_refinement(
            arguments,
            Path("/tmp/refined-output/refined"),
            estimator_runner=fake_runner,
        )
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(output, Path("/tmp/refined-output/refined"))
        self.assertTrue(
            np.array_equal(observed["arguments"].initial_coordinate, conditioned)
        )
        self.assertEqual(observed["arguments"].lag_mode, "estimated")
        self.assertIsNone(observed["arguments"].fixed_rotor_lag)
        self.assertFalse(
            any("prior" in key.lower() for key in vars(observed["arguments"]))
        )
        self.assertNotIn("fixed_cog", vars(observed["arguments"]))

    def test_baseline_loader_is_read_only(self):
        source = "a" * 40
        covariance = np.eye(14)
        result = {
            "status": "completed",
            "strict_final_evaluation": True,
            "source_commit": source,
            "parameters": {"chart_coordinate": [0.0] * 14},
            "uncertainty": {
                "parameter_covariance_conservative_fusion": covariance.tolist()
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)
            (case / "result.json").write_text(json.dumps(result))
            (case / "status.json").write_text(
                json.dumps({"status": "completed", "source_commit": source})
            )
            (case / "arguments.json").write_text(
                json.dumps({"bag_id": "synthetic"})
            )
            np.savez_compressed(
                case / "arrays.npz",
                parameter_covariance_conservative_fusion=covariance,
            )
            before = {path.name: path.read_bytes() for path in case.iterdir()}
            loaded = load_completed_baseline_case(
                case,
                expected_directory=case,
                expected_source_commit=source,
            )
            verify_baseline_files_unchanged(loaded)
            after = {path.name: path.read_bytes() for path in case.iterdir()}
            self.assertEqual(before, after)


class CogPriorReseedConsensusReportingTests(unittest.TestCase):
    @staticmethod
    def _stage(value: float) -> dict:
        inertia = np.eye(3) * value
        force = np.arange(1.0, 5.0) * value
        return {
            "strict_identity_objective_sum": 10.0 * value,
            "rotor_lag_seconds": 0.1 * value,
            "cog_position_body_m": np.arange(3.0) * 0.001 * value,
            "force_effectiveness": force,
            "inertia_over_mass_m2": inertia,
            "force_effectiveness_over_mass": force / 2.0,
        }

    def test_machine_and_human_consensus_summaries_are_complete(self):
        refinements = []
        for index, bag_id in enumerate(
            ("single_rosbag_1", "single_rosbag_2", "single_rosbag_succeeded"),
            start=1,
        ):
            original = self._stage(float(index))
            conditioned = self._stage(float(index) + 0.25)
            conditioned["evaluation_rotor_lag_seconds"] = conditioned.pop(
                "rotor_lag_seconds"
            )
            refined = self._stage(float(index) + 0.5)
            refinements.append(
                {
                    "bag_id": bag_id,
                    "comparison": {
                        "original": original,
                        "conditioned": conditioned,
                        "refined": refined,
                        "delta_L_refined_minus_original": 5.0,
                        "relative_delta_L_refined_minus_original": 0.5,
                    },
                }
            )
        rows = _comparison_rows(refinements)
        self.assertEqual([row["case"] for row in rows], ["failure1", "failure2", "success"])
        self.assertIn("CoG_refined_z_m", rows[0])
        self.assertIn("force_effectiveness_conditioned_4", rows[0])
        matrix = np.arange(9.0).reshape(3, 3)
        summary = _human_summary_text(
            rows=rows,
            original_distance=matrix,
            refined_distance=matrix + 1.0,
            original_cross_cost=matrix + 2.0,
            original_cross_delta=matrix + 3.0,
            refined_cross_cost=matrix + 4.0,
            refined_cross_delta=matrix + 5.0,
            success_comparisons={"failure1_original_vs_original_success": {}},
        )
        self.assertIn("J/m conditioned", summary)
        self.assertIn("Refined absolute cross cost", summary)
        self.assertIn("failure1_original_vs_original_success", summary)

    def test_consensus_pdf_smoke(self):
        stages = {}
        for stage_index, stage_name in enumerate(
            ("original", "conditioned", "refined"), start=1
        ):
            stages[stage_name] = {
                "objective": np.arange(1.0, 4.0) * stage_index,
                "rotor_lag": np.arange(1.0, 4.0) * 0.01,
                "cog": np.ones((3, 3)) * 0.001 * stage_index,
                "force_over_mass": np.ones((3, 4)) * stage_index,
                "inertia_over_mass": np.stack(
                    [np.eye(3) * (stage_index + bag) for bag in range(3)]
                ),
            }
        matrix = np.arange(9.0).reshape(3, 3)
        comparisons = {
            "{}_{}_vs_{}_success".format(failure, stage, target): {
                "distance": float(index)
            }
            for index, (failure, stage, target) in enumerate(
                (
                    ("failure1", "original", "original"),
                    ("failure1", "refined", "original"),
                    ("failure1", "refined", "refined"),
                    ("failure2", "original", "original"),
                    ("failure2", "refined", "original"),
                    ("failure2", "refined", "refined"),
                )
            )
        }
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.pdf"
            _write_report(
                report,
                bag_ids=("single_rosbag_1", "single_rosbag_2", "single_rosbag_succeeded"),
                stages=stages,
                original_distance=matrix,
                refined_distance=matrix + 1.0,
                original_cross_cost=matrix + 2.0,
                original_cross_delta=matrix + 3.0,
                refined_cross_cost=matrix + 4.0,
                refined_cross_delta=matrix + 5.0,
                success_comparisons=comparisons,
            )
            self.assertTrue(report.is_file())
            self.assertGreater(report.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
