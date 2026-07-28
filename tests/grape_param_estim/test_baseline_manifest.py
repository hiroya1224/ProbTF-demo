import importlib.util
import json
from pathlib import Path
import unittest


REPOSITORY = Path(__file__).resolve().parents[2]
PACKAGE = REPOSITORY / "ros/examples/grape-param-estim"
SCRIPT = PACKAGE / "scripts/verify_inverse_dynamics_baseline.py"
MANIFEST = PACKAGE / "config/inverse_dynamics_baseline.json"
SYNTHETIC_SCRIPT = (
    PACKAGE / "scripts/verify_synthetic_sanity_baseline.py"
)
SYNTHETIC_MANIFEST = (
    PACKAGE / "config/synthetic_sanity_baseline.json"
)

SPEC = importlib.util.spec_from_file_location("baseline_verifier", SCRIPT)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)
SYNTHETIC_SPEC = importlib.util.spec_from_file_location(
    "synthetic_baseline_verifier", SYNTHETIC_SCRIPT
)
SYNTHETIC_VERIFIER = importlib.util.module_from_spec(SYNTHETIC_SPEC)
SYNTHETIC_SPEC.loader.exec_module(SYNTHETIC_VERIFIER)


class InverseDynamicsBaselineManifestTests(unittest.TestCase):
    def test_canonical_listing_and_frozen_payloads_are_reproducible(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        listing = VERIFIER.canonical_payload_listing(manifest)
        self.assertTrue(listing.endswith(b"\n"))
        self.assertNotIn(str(REPOSITORY).encode("utf-8"), listing)
        self.assertEqual(
            VERIFIER.canonical_payload_listing_sha256(manifest),
            "76575356f084d99e5d1a2d1af9a3282d4208afb6b61c697ca06da1652773131c",
        )
        result = VERIFIER.verify_baseline_manifest(MANIFEST)
        self.assertEqual(
            result["verified_runs"],
            ["20260612-04", "20260612-07", "20260612-08"],
        )
        self.assertEqual(
            result["frozen_payload_model_ids"],
            ["low_dimensional_effective_response/v1"],
        )
        self.assertIn("family label", result["model_id_role"])

    def test_listing_hash_rejects_tampered_anchor(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["runs"]["20260612-04"]["summary_sha256"] = "0" * 64
        self.assertNotEqual(
            VERIFIER.canonical_payload_listing_sha256(manifest),
            manifest["combined_payload_listing_sha256"],
        )


class SyntheticSanityBaselineTests(unittest.TestCase):
    def test_path_canonicalization_preserves_non_path_output(self):
        generator = {
            "output_bag": "/tmp/one/input.bag",
            "samples": 301,
            "seed": 7,
        }
        evaluator = {
            "analysis_bag": "/tmp/one/analysis.bag",
            "estimate": {
                "source_bag": "/tmp/one/input.bag",
                "mean": [1.0, 2.0],
            },
        }
        normalized_generator = (
            SYNTHETIC_VERIFIER.normalize_generator_summary(generator)
        )
        normalized_evaluator = (
            SYNTHETIC_VERIFIER.normalize_evaluation_summary(evaluator)
        )
        self.assertEqual(
            normalized_generator,
            {
                "output_bag": "$INPUT_BAG",
                "samples": 301,
                "seed": 7,
            },
        )
        self.assertEqual(
            normalized_evaluator,
            {
                "analysis_bag": "$ANALYSIS_BAG",
                "estimate": {
                    "source_bag": "$INPUT_BAG",
                    "mean": [1.0, 2.0],
                },
            },
        )
        self.assertEqual(generator["output_bag"], "/tmp/one/input.bag")
        self.assertEqual(
            evaluator["estimate"]["source_bag"],
            "/tmp/one/input.bag",
        )

    def test_legacy_three_command_golden_is_reproducible(self):
        manifest = json.loads(
            SYNTHETIC_MANIFEST.read_text(encoding="utf-8")
        )
        result = SYNTHETIC_VERIFIER.reproduce_synthetic_baseline(
            SYNTHETIC_MANIFEST
        )
        self.assertEqual(
            result["summary_sha256"],
            {
                key: manifest["expected"][key]
                for key in (
                    "generator_summary_sha256",
                    "estimator_summary_sha256",
                    "evaluation_summary_sha256",
                )
            },
        )
        self.assertEqual(result["generator"]["samples"], 301)
        self.assertEqual(result["generator"]["excitation_rank"], 10)
        self.assertEqual(result["estimator"]["analysis_messages"], 180)
        self.assertEqual(
            result["evaluation"]["estimate"]["observation_count"], 30
        )
        self.assertEqual(
            result["evaluation"]["estimate"]["model"],
            "inverse_dynamics_baseline_v1:calibrated_wrench",
        )
        self.assertTrue(result["evaluation"]["report_only"])
        self.assertFalse(result["evaluation"]["passed"])


if __name__ == "__main__":
    unittest.main()
