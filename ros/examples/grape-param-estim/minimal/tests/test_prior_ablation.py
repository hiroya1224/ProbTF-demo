from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from _support import MINIMAL
from single_bag_parameter_prior import load_parameter_prior
from single_bag_prior_ablation import (
    PRIOR_FREE_CASE,
    _case_arguments,
    build_summary,
    load_ablation_manifest,
    write_ablation_pdf,
)
from single_bag_savgol_core import (
    COMMON_SCALE_DIRECTION,
    SiParameterChart,
    load_vehicle_model,
)
from three_bag_prior_ablation_summary import (
    EXPECTED_BAG_IDS,
    build_three_bag_summary,
    write_three_bag_pdf,
)


EXPECTED_CASES = (
    "cog_x_nominal",
    "cog_y_nominal",
    "cog_z_nominal",
    "inertia_over_mass_xx_nominal",
    "inertia_over_mass_yy_nominal",
    "inertia_over_mass_zz_nominal",
    "inertia_over_mass_xy_nominal",
    "inertia_over_mass_xz_nominal",
    "inertia_over_mass_yz_nominal",
    "force_over_mass_rotor_1_nominal",
    "force_over_mass_rotor_2_nominal",
    "force_over_mass_rotor_3_nominal",
    "force_over_mass_rotor_4_nominal",
    "cog_all_nominal",
    "inertia_over_mass_all_nominal",
    "force_over_mass_all_nominal",
)


def _payload(value, prior=None, status="completed", postfit="completed"):
    vector = np.asarray(value, dtype=float)
    inertia = np.asarray(
        (
            (vector[0], vector[3], vector[4]),
            (vector[3], vector[1], vector[5]),
            (vector[4], vector[5], vector[2]),
        )
    )
    return {
        "status": status,
        "overall_case_status": status,
        "optimization_status": "completed" if status != "failed" else "failed",
        "postfit_uncertainty_status": postfit,
        "parameters": {
            "rotor_lag_seconds": 0.05,
            "scale_free": {
                "inertia_over_mass_m2": inertia,
                "cog_position_body_m": vector[6:9],
                "force_effectiveness_over_mass": vector[9:13],
            },
        },
        "optimization_objective": {
            "data_objective_sum": 10.0 + vector[0],
            "prior_objective_sum": 0.0 if prior is None else 0.25,
            "total_objective_sum": 10.0 + vector[0] + (0.0 if prior is None else 0.25),
        },
        "prior": {"active": False} if prior is None else prior,
    }


class PriorAblationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_ablation_manifest(
            MINIMAL / "config/prior_ablation/nominal_pseudo_conditioning.json"
        )
        self.model = load_vehicle_model(MINIMAL / "grape_vehicle_model.json")

    def test_manifest_has_exact_fixed_primary_matrix_and_valid_strengths(self):
        names = tuple(item["case_name"] for item in self.manifest["cases"])
        self.assertEqual(names, EXPECTED_CASES)
        self.assertEqual(len(names) + 1, 17)
        configured_paths = {
            Path(item["prior_json"]).resolve()
            for item in self.manifest["cases"]
        }
        inventory_paths = {
            path.resolve()
            for path in (
                MINIMAL / "config/priors/pseudo_conditioning"
            ).rglob("*.json")
        }
        self.assertEqual(configured_paths, inventory_paths)
        prior_names = []
        for item in self.manifest["cases"]:
            prior = load_parameter_prior(Path(item["prior_json"]), self.model)
            prior_names.append(prior.name)
            self.assertEqual(prior.role, "pseudo_conditioning_ablation")
            for factor in prior.factors:
                self.assertEqual(factor.target_source, "vehicle_model_nominal")
                expected = {
                    "cog_position_body_m": 1.0e-5,
                    "inertia_over_mass_m2": 1.0e-6,
                    "force_effectiveness_over_mass": 1.0e-5,
                }[factor.quantity]
                self.assertTrue(np.array_equal(factor.standard_deviation, np.full(len(factor.components), expected)))
            evaluation = prior.evaluate(
                SiParameterChart(self.model.parameters),
                np.linspace(-0.02, 0.02, 14),
            )
            self.assertLess(
                np.linalg.norm(
                    evaluation.jacobian @ COMMON_SCALE_DIRECTION
                ),
                2e-8,
            )
        self.assertEqual(len(set(prior_names)), len(prior_names))

    def test_every_case_clones_the_same_initialization_and_changes_only_prior(self):
        base = argparse.Namespace(
            prior_json=None,
            initial_coordinate=np.linspace(-0.2, 0.2, 14),
            initial_rotor_lag=0.123,
            run_id="outer",
            covariance_mode="identity",
        )
        before = deepcopy(vars(base))
        first = _case_arguments(base, Path("first.json"))
        second = _case_arguments(base, Path("second.json"))
        baseline = _case_arguments(base, None)
        self.assertTrue(np.array_equal(first.initial_coordinate, second.initial_coordinate))
        self.assertEqual(first.initial_rotor_lag, second.initial_rotor_lag)
        self.assertEqual(first.covariance_mode, second.covariance_mode)
        self.assertNotEqual(first.prior_json, second.prior_json)
        self.assertIsNone(baseline.prior_json)
        differing = {
            key
            for key in vars(first)
            if key != "initial_coordinate"
            and vars(first)[key] != vars(second)[key]
        }
        self.assertEqual(differing, {"prior_json"})
        self.assertEqual(set(vars(base)), set(before))
        self.assertTrue(
            np.array_equal(base.initial_coordinate, before["initial_coordinate"])
        )
        for key in set(before) - {"initial_coordinate"}:
            self.assertEqual(vars(base)[key], before[key])

    def test_per_bag_summary_keeps_point_estimate_when_postfit_failed(self):
        baseline = np.linspace(0.01, 0.13, 13)
        payloads = {PRIOR_FREE_CASE: _payload(baseline)}
        for index, item in enumerate(self.manifest["cases"]):
            vector = baseline + (index + 1) * 1.0e-4
            quotient_index = index % 13
            prior = {
                "active": True,
                "factor_evaluations": [
                    {
                        "quotient_indices": [quotient_index],
                        "physical_error": [2.0e-6],
                        "standardized_residual": [0.2],
                        "physical_target": [vector[quotient_index] - 2.0e-6],
                        "physical_value": [vector[quotient_index]],
                    }
                ],
            }
            status = "point_estimate_completed" if index == 0 else "completed"
            postfit = "failed" if index == 0 else "completed"
            payloads[item["case_name"]] = _payload(vector, prior, status=status, postfit=postfit)
        summary, arrays = build_summary(
            source_revision="abc",
            manifest=self.manifest,
            bag_id="single_rosbag_1",
            payloads=payloads,
        )
        self.assertEqual(summary["case_count"], 17)
        self.assertEqual(summary["point_estimate_count"], 17)
        self.assertEqual(summary["postfit_uncertainty_failed_count"], 1)
        self.assertTrue(np.all(np.isfinite(arrays["x_per_case"])))
        self.assertTrue(np.allclose(arrays["delta_x_per_case"][0], 0.0))
        self.assertEqual(arrays["postfit_uncertainty_status_per_case"][1], "failed")
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "prior_ablation.pdf"
            write_ablation_pdf(report, summary, arrays)
            self.assertGreater(report.stat().st_size, 0)

    def test_three_bag_spreads_and_pairwise_distances_use_physical_points(self):
        cases = np.asarray(("prior_free", "cog_x_nominal"), dtype="U")
        loaded = []
        for bag_index, bag_id in enumerate(EXPECTED_BAG_IDS):
            x = np.zeros((2, 13))
            x[:, 6] = float(bag_index)
            x[:, 0] = 2.0 * bag_index
            x[:, 9] = 3.0 * bag_index
            arrays = {
                "case_names": cases,
                "x_per_case": x,
                "data_objective_per_case": np.asarray((10.0, 11.0 + bag_index)),
                "delta_data_objective_per_case": np.asarray((0.0, 1.0 + bag_index)),
                "overall_case_status_per_case": np.asarray(("completed", "completed")),
                "optimization_status_per_case": np.asarray(("completed", "completed")),
                "postfit_uncertainty_status_per_case": np.asarray(("completed", "completed")),
                "source_directory": np.asarray("/tmp/{}".format(bag_id)),
            }
            summary = {
                "bag_id": bag_id,
                "source_commit": "abc",
                "manifest": {"source_sha256": "same"},
            }
            loaded.append((summary, arrays))
        summary, arrays = build_three_bag_summary(loaded, source_revision="abc")
        expected_rms = np.sqrt(2.0 / 3.0)
        self.assertTrue(np.allclose(arrays["cog_spread_m"], expected_rms))
        self.assertTrue(np.allclose(arrays["inertia_over_mass_spread_m2_frobenius"], 2.0 * expected_rms))
        self.assertTrue(np.allclose(arrays["force_over_mass_spread_kg_inverse"], 3.0 * expected_rms))
        self.assertTrue(np.allclose(arrays["pairwise_cog_distance_m"][0], (1.0, 2.0, 1.0)))
        self.assertEqual(summary["cross_evaluation"]["status"], "not_run")
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "three_bag.pdf"
            write_three_bag_pdf(report, summary, arrays)
            self.assertGreater(report.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
