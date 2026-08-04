from pathlib import Path
import tempfile
import unittest

from grape_param_estim.artifact_io import ArtifactValidationError
from grape_param_estim.pid.request import (
    PID_EVALUATION_REQUEST_SCHEMA,
    validate_pid_evaluation_request,
)


def _payload(root):
    run = root / "run"
    run.mkdir(exist_ok=True)
    bag = root / "flight.bag"
    bag.touch()
    return {
        "schema": PID_EVALUATION_REQUEST_SCHEMA,
        "evaluation_id": "pid-evaluation",
        "estimation_run": str(run),
        "output_directory": str(root / "result"),
        "resume": False,
        "forecast_workers": "auto",
        "baseline_bag_id": "bag-a",
        "selected_mode_id": "mode-map",
        "bags": [
            {
                "bag_id": "bag-a",
                "path": str(bag),
                "sha256": "sha256:" + "a" * 64,
                "roll_pitch_integration_active": True,
            }
        ],
        "fixed_plant_parameters": {
            "linear_drag": [0.0, 0.0, 0.0],
            "angular_drag": [0.0, 0.0, 0.0],
        },
        "model_discrepancy": {
            "policy": "sample_model_discrepancy",
            "base_seed": 20260804,
            "replicates": 3,
        },
        "plant_sample_subset": {
            "method": "explicit_equal_weight_mcmc_subset",
            "sample_ids": ["chain-a:1", "chain-b:1"],
        },
        "derived_candidate_population": {
            "method": "deterministic_k_medoids",
            "maximum_candidates": 8,
            "required_source_sample_ids": ["chain-a:1"],
        },
        "candidates": [
            {
                "candidate_id": "current",
                "source": "current",
                "source_sample_id": None,
                "gain_values": None,
            },
            {
                "candidate_id": "user-a",
                "source": "user",
                "source_sample_id": None,
                "gain_values": [[1.0, 0.1, 2.0]] * 4,
            },
        ],
        "quantile_level": 0.95,
        "cvar_level": 0.9,
        "selected_candidate_id": None,
        "maximum_reference_age_seconds": 0.05,
    }


class PidRequestTests(unittest.TestCase):
    def test_valid_request_has_no_scientific_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = _payload(Path(temporary))
            request = validate_pid_evaluation_request(payload)
            self.assertEqual(request.evaluation_id, "pid-evaluation")
            self.assertEqual(request.discrepancy_replicates, 3)
            self.assertEqual(request.forecast_workers, "auto")
            self.assertEqual(request.plant_sample_ids, ("chain-a:1", "chain-b:1"))
            self.assertEqual(request.derived_candidate_method, "deterministic_k_medoids")
            self.assertEqual(request.maximum_derived_candidates, 8)
            self.assertEqual(request.required_derived_sample_ids, ("chain-a:1",))
            self.assertEqual(request.candidates[1].gain_values.shape, (4, 3))
            self.assertTrue(request.fingerprint.startswith("sha256:"))

    def test_unknown_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = _payload(Path(temporary))
            payload["legacy_member_id"] = 4
            with self.assertRaisesRegex(ArtifactValidationError, "unknown"):
                validate_pid_evaluation_request(payload)

    def test_q_values_and_residual_replay_are_not_request_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = _payload(Path(temporary))
            payload["model_discrepancy"]["diagonal_q"] = [1.0] * 6
            with self.assertRaisesRegex(ArtifactValidationError, "unknown"):
                validate_pid_evaluation_request(payload)
            payload = _payload(Path(temporary))
            payload["model_discrepancy"]["policy"] = "posterior_replay"
            with self.assertRaisesRegex(ArtifactValidationError, "must be one"):
                validate_pid_evaluation_request(payload)

    def test_candidate_source_fields_are_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = _payload(Path(temporary))
            payload["candidates"][1]["source"] = "sample-derived"
            payload["candidates"][1]["source_sample_id"] = "chain-a:1"
            payload["candidates"][1]["gain_values"] = None
            with self.assertRaisesRegex(ArtifactValidationError, "must be one"):
                validate_pid_evaluation_request(payload)

    def test_derived_population_policy_is_strict_and_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = _payload(Path(temporary))
            payload["derived_candidate_population"] = {
                "method": "all_raw_mcmc_samples",
                "maximum_candidates": None,
                "required_source_sample_ids": [],
            }
            parsed = validate_pid_evaluation_request(payload)
            self.assertIsNone(parsed.maximum_derived_candidates)

            payload = _payload(Path(temporary))
            payload["derived_candidate_population"]["maximum_candidates"] = 0
            with self.assertRaisesRegex(
                ArtifactValidationError, r"\[1, 1000000\]"
            ):
                validate_pid_evaluation_request(payload)

            payload = _payload(Path(temporary))
            payload["derived_candidate_population"]["maximum_candidates"] = 1
            payload["derived_candidate_population"]["required_source_sample_ids"] = [
                "chain-a:1",
                "chain-b:1",
            ]
            with self.assertRaisesRegex(ArtifactValidationError, "cannot exceed"):
                validate_pid_evaluation_request(payload)

    def test_all_sample_method_requires_null_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = _payload(Path(temporary))
            payload["plant_sample_subset"]["method"] = (
                "all_equal_weight_mcmc_samples"
            )
            with self.assertRaisesRegex(ArtifactValidationError, "must be null"):
                validate_pid_evaluation_request(payload)

    def test_worker_count_is_explicit_but_non_scientific_resume_is_normalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = _payload(Path(temporary))
            first = validate_pid_evaluation_request(payload)
            payload["resume"] = True
            resumed = validate_pid_evaluation_request(payload)
            self.assertEqual(first.fingerprint, resumed.fingerprint)
            payload["forecast_workers"] = 2
            changed_runtime = validate_pid_evaluation_request(payload)
            self.assertNotEqual(first.fingerprint, changed_runtime.fingerprint)
            self.assertEqual(changed_runtime.forecast_workers, 2)

    def test_worker_count_rejects_implicit_or_unbounded_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = _payload(Path(temporary))
            del payload["forecast_workers"]
            with self.assertRaisesRegex(ArtifactValidationError, "missing"):
                validate_pid_evaluation_request(payload)
            for value in (0, 33, True, "automatic"):
                with self.subTest(value=value):
                    payload = _payload(Path(temporary))
                    payload["forecast_workers"] = value
                    with self.assertRaises(ArtifactValidationError):
                        validate_pid_evaluation_request(payload)


if __name__ == "__main__":
    unittest.main()
