import copy
import json
from pathlib import Path
import tempfile
import unittest

from grape_param_estim.artifact_io import ArtifactValidationError
from grape_param_estim.posterior_sampling_request import (
    POSTERIOR_SAMPLING_REQUEST_SCHEMA,
    load_posterior_sampling_request,
    validate_posterior_sampling_request,
)


class PosteriorSamplingRequestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.estimation_request = root / "estimate.json"
        self.estimation_request.write_text("{}", encoding="utf-8")
        digest = "sha256:" + "a" * 64
        self.payload = {
            "schema": POSTERIOR_SAMPLING_REQUEST_SCHEMA,
            "sampling_id": "run-a-mcmc",
            "resume": False,
            "estimation_run_directory": str(root / "run-a"),
            "estimation_request_path": str(self.estimation_request),
            "upstream": {
                "run_id": "run-a",
                "request_fingerprint": digest,
                "configuration_fingerprint": "sha256:" + "b" * 64,
                "controller_snapshot_fingerprint": "sha256:" + "c" * 64,
                "estimator_revision": "revision-a",
                "selected_bag_ids": ["flight-a"],
                "selected_intervals": {"flight-a": [18.0, 24.0]},
                "selected_bag_sha256": {"flight-a": "sha256:" + "d" * 64},
            },
            "mcmc_settings": {
                "chain_count": 2,
                "warmup_steps": 10,
                "retained_draws": 20,
                "thinning": 1,
                "random_seed": 17,
                "local_scale": 0.1,
                "exact_ridge_scale": 0.2,
                "near_ridge_scale": 0.1,
                "identified_scale": 0.05,
                "delay_scale_seconds": 0.001,
                "near_relative_threshold": 1.0e-6,
                "rhat_threshold": 1.01,
                "minimum_effective_sample_size": 4.0,
            },
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_strict_round_trip_and_resume_stable_identity(self):
        path = Path(self.temporary.name) / "sample.json"
        path.write_text(json.dumps(self.payload), encoding="utf-8")
        loaded = load_posterior_sampling_request(path)
        resumed_payload = copy.deepcopy(self.payload)
        resumed_payload["resume"] = True
        resumed = validate_posterior_sampling_request(resumed_payload)
        self.assertEqual(loaded.fingerprint, resumed.fingerprint)
        self.assertEqual(
            loaded.estimation_run_directory,
            Path(self.payload["estimation_run_directory"]).resolve(),
        )

    def test_rejects_unknown_fields_and_incomplete_upstream_bag_identity(self):
        changed = copy.deepcopy(self.payload)
        changed["mcmc_settings"]["implicit_default"] = 1.0
        with self.assertRaisesRegex(ArtifactValidationError, "keys"):
            validate_posterior_sampling_request(changed)
        changed = copy.deepcopy(self.payload)
        changed["upstream"]["selected_bag_sha256"] = {}
        with self.assertRaisesRegex(ArtifactValidationError, "bag maps"):
            validate_posterior_sampling_request(changed)

    def test_rejects_noncanonical_fingerprint_and_relative_paths(self):
        changed = copy.deepcopy(self.payload)
        changed["upstream"]["request_fingerprint"] = "abc"
        with self.assertRaisesRegex(ArtifactValidationError, "sha256"):
            validate_posterior_sampling_request(changed)
        changed = copy.deepcopy(self.payload)
        changed["estimation_run_directory"] = "relative/run"
        with self.assertRaisesRegex(ArtifactValidationError, "absolute"):
            validate_posterior_sampling_request(changed)


if __name__ == "__main__":
    unittest.main()
