import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from grape_param_estim.artifact_io import ArtifactValidationError, read_json
from grape_param_estim.held_out_validation import (
    HELD_OUT_COST_METRICS,
    HELD_OUT_VALIDATION_REQUEST_SCHEMA,
    STRICT_HOLD_OUT,
    TUNING_EVALUATION,
    HeldOutMetricRecord,
    HeldOutValidationArtifact,
    HeldOutValidationIdentity,
    load_held_out_validation,
    main,
    summarize_held_out_records,
    validate_data_split_against_source,
    validate_held_out_validation_request,
    write_held_out_validation,
)
from grape_param_estim.pid.metrics import ForecastMetrics
from grape_param_estim.pid.proposal import PhysicalPlantPosterior
from grape_param_estim.system import VehicleParameters


class HeldOutValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.estimation = self.root / "estimation"
        self.estimation.mkdir()
        self.bag = self.root / "held-out.bag"
        self.bag.write_bytes(b"held-out-flight")
        self.digest = "sha256:" + "1" * 64

    def tearDown(self):
        self.temporary.cleanup()

    def payload(self, *, role=STRICT_HOLD_OUT, estimator=False, pid=False):
        return {
            "schema": HELD_OUT_VALIDATION_REQUEST_SCHEMA,
            "validation_id": "held-out-001",
            "estimation_run": str(self.estimation),
            "output_directory": str(self.root / "validation"),
            "held_out_bag": {
                "bag_id": "success-flight",
                "path": str(self.bag),
                "sha256": self.digest,
                "interval_seconds": [45.0, 51.0],
                "roll_pitch_integration_active": True,
            },
            "data_split": {
                "role": role,
                "used_for_estimator_tuning": estimator,
                "used_for_pid_tuning": pid,
            },
            "configuration_compatibility": {
                "status": "unconfirmed",
                "evidence": ["inspection configuration provenance incomplete"],
            },
            "selected_mode_id": "recorded-mode",
            "fixed_plant_parameters": {
                "linear_drag": [0.1, 0.1, 0.2],
                "angular_drag": [0.01, 0.01, 0.02],
            },
            "model_discrepancy": {
                "policy": "zero_model_discrepancy",
                "base_seed": 17,
                "replicates": 2,
            },
            "posterior_sample_subset": {
                "method": "all_equal_weight_mcmc_samples",
                "sample_ids": None,
            },
            "forecast_settings": {
                "knot_period_seconds": 0.05,
                "pose_smoothing_window": 5,
                "allow_zero_integral_fallback": False,
                "maximum_reference_age_seconds": 0.2,
            },
            "summary_settings": {
                "quantile_level": 0.95,
                "cvar_level": 0.9,
            },
        }

    def request(self, **kwargs):
        return validate_held_out_validation_request(self.payload(**kwargs))

    def metrics(self, position, completion=1.0):
        return ForecastMetrics(
            position_rmse=position,
            orientation_rmse=position + 0.1,
            maximum_position_error=position + 0.2,
            maximum_orientation_error=position + 0.3,
            forecast_completion=completion,
            numerical_failure_count=0 if completion == 1.0 else 1,
            actuator_saturation_duration=0.4,
            actuator_saturation_rate=0.05,
        )

    def records(self):
        return (
            HeldOutMetricRecord(
                "chain-0:sample-0", 0, 10, self.metrics(1.0), self.metrics(2.0)
            ),
            HeldOutMetricRecord(
                "chain-0:sample-0", 1, 11, self.metrics(3.0), self.metrics(4.0)
            ),
        )

    def posterior(self):
        return PhysicalPlantPosterior.from_aligned_values(
            ("chain-0:sample-0",),
            (VehicleParameters.nominal(),),
            (0.01,),
            ("recorded-mode",),
        )

    def test_strict_request_has_explicit_not_tuned_boundary(self):
        request = self.request()
        self.assertEqual(request.data_split.role, STRICT_HOLD_OUT)
        self.assertEqual(
            request.data_split.semantic_label, "strict held-out validation"
        )

    def test_strict_hold_out_rejects_any_tuning_use(self):
        with self.assertRaisesRegex(
            ArtifactValidationError, "strict_hold_out forbids"
        ):
            self.request(estimator=True)

    def test_tuning_evaluation_requires_and_exposes_tuning_use(self):
        with self.assertRaisesRegex(
            ArtifactValidationError, "requires at least one"
        ):
            self.request(role=TUNING_EVALUATION)
        request = self.request(role=TUNING_EVALUATION, pid=True)
        self.assertEqual(
            request.data_split.semantic_label,
            "tuning evaluation (not held-out)",
        )

    def test_request_rejects_unknown_fields_and_relative_paths(self):
        payload = self.payload()
        payload["legacy_member_count"] = 64
        with self.assertRaisesRegex(ArtifactValidationError, "unknown"):
            validate_held_out_validation_request(payload)
        payload = self.payload()
        payload["estimation_run"] = "relative/run"
        with self.assertRaisesRegex(ArtifactValidationError, "absolute"):
            validate_held_out_validation_request(payload)

    def test_strict_hold_out_rejects_source_fit_rosbag_sha(self):
        request = self.request()
        source = {"selected_bag_sha256": {"fit": self.digest}}
        with self.assertRaisesRegex(ValueError, "already part"):
            validate_data_split_against_source(
                request.data_split, self.digest, source
            )
        tuning = self.request(role=TUNING_EVALUATION, estimator=True)
        validate_data_split_against_source(tuning.data_split, self.digest, source)

    def test_summary_keeps_observed_and_reference_metrics_separate(self):
        result = summarize_held_out_records(
            self.records(), quantile_level=0.95, cvar_level=0.5
        )
        self.assertEqual(result.metric_names, HELD_OUT_COST_METRICS)
        self.assertAlmostEqual(result.mean[0], 2.0)
        self.assertAlmostEqual(result.mean[4], 3.0)
        self.assertAlmostEqual(result.forecast_completion_mean, 1.0)

    def write_artifact(self, *, tuning=False):
        request = (
            self.request(role=TUNING_EVALUATION, estimator=True)
            if tuning
            else self.request()
        )
        result = summarize_held_out_records(
            self.records(), quantile_level=0.95, cvar_level=0.5
        )
        return write_held_out_validation(
            request.output_directory,
            request=request,
            identity=HeldOutValidationIdentity(
                source_estimation_run_id="batch-run",
                source_estimation_request_fingerprint="sha256:" + "2" * 64,
                source_estimator_revision="revision-abc",
                selected_mode_id="recorded-mode",
            ),
            selected_posterior=self.posterior(),
            result=result,
            q_quantity="body_wrench",
            q_interval_model="continuous_spectral_density",
            q_diagonal=np.arange(1.0, 7.0),
            source_actuator_model={
                "source": "test calibrated actuator model",
                "thrust_time_constant_seconds": 0.01,
                "gimbal_time_constant_seconds": 0.02,
                "minimum_thrust_newtons": 1.5,
                "maximum_thrust_newtons": 27.6,
                "maximum_gimbal_angle_radians": 3.14,
                "maximum_gimbal_rate_radians_per_second": 6.0,
            },
        )

    def test_pickle_free_artifact_round_trip(self):
        artifact = self.write_artifact()
        loaded = load_held_out_validation(artifact.root)
        self.assertEqual(loaded.manifest["status"], "complete")
        self.assertEqual(
            loaded.manifest["semantic_label"], "strict held-out validation"
        )
        self.assertEqual(loaded.arrays["metric_values"].shape, (2, 11))
        with np.load(str(artifact.root / "validation.npz"), allow_pickle=False) as data:
            self.assertFalse(any(data[name].dtype.hasobject for name in data.files))

    def test_tuning_artifact_cannot_call_itself_held_out(self):
        artifact = self.write_artifact(tuning=True)
        self.assertEqual(
            artifact.manifest["semantic_label"],
            "tuning evaluation (not held-out)",
        )
        self.assertIn("not held-out evidence", artifact.manifest["warnings"][1])

    def test_artifact_hash_tampering_is_rejected(self):
        artifact = self.write_artifact()
        with (artifact.root / "validation.npz").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ArtifactValidationError, "hash disagrees"):
            load_held_out_validation(artifact.root)

    def test_main_reports_semantic_label(self):
        request = self.request()
        fake = HeldOutValidationArtifact(
            root=self.root / "result",
            manifest={"semantic_label": "strict held-out validation"},
            arrays={},
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch(
            "grape_param_estim.held_out_validation.load_held_out_validation_request",
            return_value=request,
        ), mock.patch(
            "grape_param_estim.held_out_validation.execute_held_out_validation",
            return_value=fake,
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(main(["--request", str(self.root / "request.json")]), 0)
        self.assertIn("strict held-out validation complete", stderr.getvalue())

    def test_manifest_duplicate_or_changed_split_is_rejected(self):
        artifact = self.write_artifact()
        manifest = read_json(artifact.root / "manifest.json")
        manifest["data_split"]["used_for_pid_tuning"] = True
        from grape_param_estim.artifact_io import write_json_atomic

        write_json_atomic(artifact.root / "manifest.json", manifest)
        with self.assertRaisesRegex(ArtifactValidationError, "forbids"):
            load_held_out_validation(artifact.root)


if __name__ == "__main__":
    unittest.main()
