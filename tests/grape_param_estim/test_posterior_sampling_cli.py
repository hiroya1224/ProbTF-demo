import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from grape_param_estim.artifact_io import ArtifactValidationError, read_json
from grape_param_estim.batch.state import StateScaling
from grape_param_estim.batch_artifact import (
    load_batch_estimation_run,
    write_batch_estimation_run,
)
from grape_param_estim.batch_artifact_export import (
    ArtifactRunIdentity,
    DelayLocalGeometry,
    export_batch_estimation_artifact_payload,
)
from grape_param_estim.batch_checkpoint import (
    load_batch_estimation_checkpoint,
    mark_batch_checkpoint_published,
    write_batch_estimation_checkpoint,
)
from grape_param_estim.batch_estimation_cli import (
    configuration_fingerprint,
    controller_snapshot_fingerprint,
)
from grape_param_estim.batch_request import validate_batch_estimation_request
from grape_param_estim.posterior_sampling_cli import execute_posterior_sampling
from grape_param_estim.posterior_sampling_request import (
    POSTERIOR_SAMPLING_REQUEST_SCHEMA,
    validate_posterior_sampling_request,
)
from grape_param_estim.progress import CancellationToken
from grape_param_estim.real_estimation import RealEstimationInputs
from tests.grape_param_estim.test_batch_artifact_export import (
    BatchArtifactExportTests,
)


class PosteriorSamplingCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.helper = BatchArtifactExportTests()
        self.helper.setUp()
        estimate_payload = copy.deepcopy(self.helper.helper.payload)
        estimate_payload["output_directory"] = str(self.root / "run")
        self.estimation = validate_batch_estimation_request(estimate_payload)
        self.estimation_path = self.root / "estimate.json"
        self.estimation_path.write_text(
            json.dumps(estimate_payload), encoding="utf-8"
        )
        self.configuration = configuration_fingerprint(self.estimation)
        self.controller = controller_snapshot_fingerprint(
            SimpleNamespace(flight_data=(self.helper.helper.flight,))
        )
        self.estimator_revision = "test-estimator-revision"
        core = export_batch_estimation_artifact_payload(
            request=self.estimation,
            flight_data=(self.helper.helper.flight,),
            initializations=(self.helper.helper.initialization,),
            final_solution=self.helper.solution,
            em_result=self.helper.em_result,
            static_geometry=self.helper.solution.static_geometry(),
            final_q_lag_profile=self.helper._final_q_lag_profile(),
            delay_geometry=DelayLocalGeometry(
                0.001,
                "positive local quadratic profile curvature",
                1.0e6,
            ),
            identity=ArtifactRunIdentity(
                estimator_revision=self.estimator_revision,
                configuration_fingerprint=self.configuration,
                controller_snapshot_fingerprint=self.controller,
            ),
            performance=self.helper._performance(mcmc=False),
        )
        checkpoint = write_batch_estimation_checkpoint(
            self.estimation.output_directory,
            request=self.estimation,
            estimator_revision=self.estimator_revision,
            configuration_fingerprint=self.configuration,
            controller_snapshot_fingerprint=self.controller,
            selected_mode_id="recorded-mode",
            core=core,
            state=self.helper.solution.lm.state,
        )
        self.run = write_batch_estimation_run(
            self.estimation.output_directory, **core.writer_arguments
        )
        mark_batch_checkpoint_published(checkpoint.root)
        self.inputs = RealEstimationInputs(
            request=self.estimation,
            flight_data=(self.helper.helper.flight,),
            initializations=(self.helper.helper.initialization,),
            parameter_chart=self.helper.helper.chart,
            geometry=self.helper.helper.geometry,
            actuator_parameters=self.helper.helper.actuators,
            scaling=StateScaling.unit(),
            loading_seconds=0.0,
        )

    def tearDown(self):
        self.helper.tearDown()
        self.temporary.cleanup()

    def _payload(self, resume=False):
        manifest = self.run.manifest
        return {
            "schema": POSTERIOR_SAMPLING_REQUEST_SCHEMA,
            "sampling_id": "sample-run",
            "resume": resume,
            "estimation_run_directory": str(self.estimation.output_directory),
            "estimation_request_path": str(self.estimation_path),
            "upstream": {
                "run_id": manifest["run_id"],
                "request_fingerprint": manifest["request_fingerprint"],
                "configuration_fingerprint": manifest[
                    "configuration_fingerprint"
                ],
                "controller_snapshot_fingerprint": manifest[
                    "controller_snapshot_fingerprint"
                ],
                "estimator_revision": manifest["estimator_revision"],
                "selected_bag_ids": manifest["selected_bag_ids"],
                "selected_intervals": manifest["selected_intervals"],
                "selected_bag_sha256": manifest["selected_bag_sha256"],
            },
            "mcmc_settings": {
                "chain_count": 2,
                "warmup_steps": 0,
                "retained_draws": 4,
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

    def _completed_mcmc(self, timing_callback):
        timing_callback(0.2)
        chains, diagnostics = self.helper._chains_and_diagnostics()
        return type("McmcResult", (), {
            "chains": chains,
            "diagnostics": diagnostics,
        })()

    @patch(
        "grape_param_estim.posterior_sampling_cli.prepare_real_estimation_inputs"
    )
    @patch("grape_param_estim.posterior_sampling_cli.sample_laplace_solution")
    def test_appends_to_same_directory_and_same_request_is_idempotent(
        self, sample, prepare
    ):
        prepare.return_value = self.inputs

        def sampled(*_args, **kwargs):
            return self._completed_mcmc(kwargs["target_timing_callback"])

        sample.side_effect = sampled
        request = validate_posterior_sampling_request(self._payload())
        upgraded = execute_posterior_sampling(
            request, sampler_revision="sampler-test"
        )
        self.assertEqual(upgraded.root, self.estimation.output_directory)
        self.assertEqual(upgraded.mcmc_samples["sample_id"].size, 8)
        self.assertEqual(
            upgraded.manifest["request_fingerprint"],
            self.estimation.fingerprint,
        )
        self.assertEqual(
            upgraded.manifest["mcmc_settings"][
                "sampling_request_fingerprint"
            ],
            request.fingerprint,
        )
        again = execute_posterior_sampling(
            request, sampler_revision="sampler-test"
        )
        self.assertEqual(again.mcmc_samples["sample_id"].size, 8)
        self.assertEqual(sample.call_count, 1)

    def test_rejects_upstream_bag_or_configuration_mismatch(self):
        payload = self._payload()
        payload["upstream"]["selected_bag_sha256"]["flight-a"] = (
            "sha256:" + "f" * 64
        )
        request = validate_posterior_sampling_request(payload)
        with self.assertRaisesRegex(ArtifactValidationError, "upstream"):
            execute_posterior_sampling(
                request, sampler_revision="sampler-test"
            )

    @patch(
        "grape_param_estim.posterior_sampling_cli.prepare_real_estimation_inputs"
    )
    @patch("grape_param_estim.posterior_sampling_cli.sample_laplace_solution")
    def test_cancel_keeps_original_and_resume_finishes(
        self, sample, prepare
    ):
        prepare.return_value = self.inputs
        token = CancellationToken()
        original_manifest = (
            self.estimation.output_directory / "manifest.json"
        ).read_bytes()

        def cancel_sample(*_args, **_kwargs):
            token.cancel("user_requested")
            raise RuntimeError("cancelled at proposal boundary")

        sample.side_effect = cancel_sample
        fresh = validate_posterior_sampling_request(self._payload())
        with self.assertRaisesRegex(RuntimeError, "proposal boundary"):
            execute_posterior_sampling(
                fresh,
                sampler_revision="sampler-test",
                cancellation_token=token,
            )
        self.assertEqual(
            (self.estimation.output_directory / "manifest.json").read_bytes(),
            original_manifest,
        )
        self.assertIsNone(
            load_batch_estimation_run(
                self.estimation.output_directory
            ).mcmc_samples
        )
        checkpoint = load_batch_estimation_checkpoint(
            self.estimation.output_directory,
            request=self.estimation,
            estimator_revision=self.estimator_revision,
            configuration_fingerprint=self.configuration,
            controller_snapshot_fingerprint=self.controller,
            allow_published=True,
        )
        self.assertEqual(checkpoint.manifest["status"], "cancelled")

        resumed_payload = self._payload(resume=True)
        resumed = validate_posterior_sampling_request(resumed_payload)
        sample.side_effect = lambda *_args, **kwargs: self._completed_mcmc(
            kwargs["target_timing_callback"]
        )
        upgraded = execute_posterior_sampling(
            resumed, sampler_revision="sampler-test"
        )
        self.assertEqual(upgraded.mcmc_samples["sample_id"].size, 8)
        context = read_json(checkpoint.root / "manifest.json")[
            "sampling_context"
        ]
        self.assertEqual(
            context["sampling_request_fingerprint"], fresh.fingerprint
        )


if __name__ == "__main__":
    unittest.main()
