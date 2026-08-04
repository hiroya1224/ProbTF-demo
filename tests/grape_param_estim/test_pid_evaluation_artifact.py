from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.artifact_io import (
    ArtifactValidationError,
    IncompleteArtifactError,
    UnsupportedArtifactSchema,
)
from grape_param_estim.controller_config import PidGainConfiguration
from grape_param_estim.pid.artifact import (
    PID_PROPOSAL_EVALUATION_SCHEMA,
    PidEvaluationArtifactIdentity,
    load_pid_proposal_evaluation,
    write_pid_proposal_evaluation,
)
from grape_param_estim.pid.metrics import ForecastMetrics
from grape_param_estim.pid.particle_search import (
    BODY_WRENCH_MODEL_DISCREPANCY,
    CONTINUOUS_SPECTRAL_DENSITY,
    SAMPLE_MODEL_DISCREPANCY,
    ModelDiscrepancyConfiguration,
    evaluate_pid_candidates,
)
from grape_param_estim.pid.proposal import (
    PhysicalPlantPosterior,
    user_pid_candidate,
)
from grape_param_estim.system import VehicleParameters


class PidEvaluationArtifactTests(unittest.TestCase):
    def setUp(self):
        nominal = VehicleParameters.nominal()
        self.posterior = PhysicalPlantPosterior.from_aligned_values(
            ("chain-a:000001", "chain-b:000001"),
            (nominal, replace(nominal, mass=1.1 * nominal.mass)),
            (0.011, 0.019),
            ("mode-map", "mode-map"),
        )
        self.current = PidGainConfiguration(np.ones((4, 3)))
        self.user = user_pid_candidate(
            "user-better", PidGainConfiguration(np.full((4, 3), 1.2))
        )

        def evaluator(candidate, sample, bag_id, realization):
            del sample, bag_id
            realization.interval_average_residual((0.02,))
            error = 1.0 if candidate.candidate_id == "current" else 0.5
            return ForecastMetrics(
                position_rmse=error,
                orientation_rmse=2.0 * error,
                maximum_position_error=3.0 * error,
                maximum_orientation_error=4.0 * error,
                forecast_completion=1.0,
                numerical_failure_count=0,
                actuator_saturation_duration=0.2 * error,
                actuator_saturation_rate=0.1 * error,
            )

        self.evaluation = evaluate_pid_candidates(
            (self.user,),
            self.posterior,
            ("bag-a", "bag-b"),
            evaluator,
            self.current,
            ModelDiscrepancyConfiguration(
                SAMPLE_MODEL_DISCREPANCY,
                np.arange(1.0, 7.0),
                base_seed=12345,
                residual_quantity=BODY_WRENCH_MODEL_DISCREPANCY,
                interval_model=CONTINUOUS_SPECTRAL_DENSITY,
                replicates=2,
            ),
        )
        self.identity = PidEvaluationArtifactIdentity(
            evaluation_id="pid-evaluation-20260804",
            estimation_run_id="batch-run-20260804",
            estimation_request_fingerprint="sha256:" + "a" * 64,
            request_fingerprint="sha256:" + "b" * 64,
        )

    def test_round_trip_preserves_full_cross_evaluation_and_yaml(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "pid-evaluation"
            artifact = write_pid_proposal_evaluation(
                destination,
                identity=self.identity,
                posterior=self.posterior,
                evaluation=self.evaluation,
                selected_candidate_id="user-better",
            )
            self.assertEqual(
                artifact.manifest["schema"], PID_PROPOSAL_EVALUATION_SCHEMA
            )
            self.assertEqual(
                artifact.manifest["model_discrepancy_residual_quantity"],
                BODY_WRENCH_MODEL_DISCREPANCY,
            )
            self.assertEqual(
                artifact.manifest["recommended_candidate_ids"], ["user-better"]
            )
            self.assertEqual(
                tuple(artifact.candidate_particles["candidate_id"]),
                ("current", "user-better"),
            )
            self.assertEqual(artifact.bags["bag-a"]["candidate_id"].size, 8)
            self.assertIn("xy:", artifact.proposed_yaml)
            self.assertIn("current:", artifact.proposed_diff_yaml)
            reloaded = load_pid_proposal_evaluation(destination)
            np.testing.assert_array_equal(
                reloaded.source_samples["sample_id"], self.posterior.sample_id
            )

    def test_common_random_seed_is_identical_across_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = write_pid_proposal_evaluation(
                Path(temporary) / "pid-evaluation",
                identity=self.identity,
                posterior=self.posterior,
                evaluation=self.evaluation,
            )
            bag = artifact.bags["bag-a"]
            by_identity = {}
            for candidate, sample, replicate, seed in zip(
                bag["candidate_id"],
                bag["sample_id"],
                bag["replicate_index"],
                bag["discrepancy_seed"],
            ):
                del candidate
                by_identity.setdefault((sample, int(replicate)), set()).add(
                    int(seed)
                )
            self.assertEqual(len(by_identity), 4)
            self.assertTrue(all(len(value) == 1 for value in by_identity.values()))

    def test_selected_candidate_must_be_recommended(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ArtifactValidationError, "recommended"):
                write_pid_proposal_evaluation(
                    Path(temporary) / "pid-evaluation",
                    identity=self.identity,
                    posterior=self.posterior,
                    evaluation=self.evaluation,
                    selected_candidate_id="current",
                )

    def test_tampered_payload_is_rejected_by_sha256(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "pid-evaluation"
            write_pid_proposal_evaluation(
                destination,
                identity=self.identity,
                posterior=self.posterior,
                evaluation=self.evaluation,
            )
            with (destination / "summary.npz").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ArtifactValidationError, "SHA-256"):
                load_pid_proposal_evaluation(destination)

    def test_incomplete_and_unknown_schema_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "pid-evaluation"
            write_pid_proposal_evaluation(
                destination,
                identity=self.identity,
                posterior=self.posterior,
                evaluation=self.evaluation,
            )
            manifest_path = destination / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "writing"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(IncompleteArtifactError):
                load_pid_proposal_evaluation(destination)
            manifest["status"] = "complete"
            manifest["schema"] = "grape-param-estim/legacy/v0"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(UnsupportedArtifactSchema):
                load_pid_proposal_evaluation(destination)


if __name__ == "__main__":
    unittest.main()
