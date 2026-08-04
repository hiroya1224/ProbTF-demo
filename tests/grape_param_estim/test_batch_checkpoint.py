import copy
from pathlib import Path
import unittest

import numpy as np

from grape_param_estim.artifact_io import ArtifactValidationError, read_json
from grape_param_estim.batch_checkpoint import (
    batch_checkpoint_path,
    load_batch_estimation_checkpoint,
    mark_batch_checkpoint_cancelled,
    save_batch_chain_checkpoint,
    write_batch_estimation_checkpoint,
)
from grape_param_estim.batch_artifact_export import (
    ArtifactRunIdentity,
    export_batch_estimation_artifact_payload,
)
from grape_param_estim.batch_request import validate_batch_estimation_request
from grape_param_estim.batch.state import StateScaling
from grape_param_estim.posterior.mcmc import (
    McmcCancelled,
    McmcChainSettings,
    run_mcmc_chain,
)
from grape_param_estim.real_estimation import (
    RealEstimationInputs,
    restore_laplace_checkpoint,
)
from tests.grape_param_estim.test_batch_artifact_export import (
    BatchArtifactExportTests,
)
from tests.grape_param_estim.test_mcmc_checkpoint import _sampler


class BatchCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.helper = BatchArtifactExportTests()
        self.helper.setUp()
        self.request = validate_batch_estimation_request(
            self.helper._mcmc_payload_request(self.helper.helper.payload)
        )
        self.configuration = "sha256:" + "1" * 64
        self.controller = "sha256:" + "2" * 64
        self.revision = "checkpoint-test-revision"
        self.core = export_batch_estimation_artifact_payload(
            request=self.request,
            flight_data=(self.helper.helper.flight,),
            initializations=(self.helper.helper.initialization,),
            final_solution=self.helper.solution,
            em_result=self.helper.em_result,
            static_geometry=self.helper.solution.static_geometry(),
            final_q_lag_profile=self.helper._final_q_lag_profile(),
            delay_geometry=self.helper._delay_geometry(),
            identity=ArtifactRunIdentity(
                estimator_revision=self.revision,
                configuration_fingerprint=self.configuration,
                controller_snapshot_fingerprint=self.controller,
            ),
            performance=self.helper._performance(mcmc=False),
            pending_mcmc_checkpoint=True,
        )

    def tearDown(self):
        self.helper.tearDown()

    def _write(self):
        return write_batch_estimation_checkpoint(
            self.request.output_directory,
            request=self.request,
            estimator_revision=self.revision,
            configuration_fingerprint=self.configuration,
            controller_snapshot_fingerprint=self.controller,
            selected_mode_id="recorded-mode",
            core=self.core,
            state=self.helper.solution.lm.state,
        )

    def _load(self, request=None):
        return load_batch_estimation_checkpoint(
            self.request.output_directory,
            request=self.request if request is None else request,
            estimator_revision=self.revision,
            configuration_fingerprint=self.configuration,
            controller_snapshot_fingerprint=self.controller,
        )

    @staticmethod
    def _partial_chain(transition):
        sampler, initial = _sampler()
        settings = McmcChainSettings(
            "chain-000", "recorded-mode", 3, 5, thinning=2
        )
        completed = [0]

        def progress(value, _total, _step):
            completed[0] = value

        try:
            run_mcmc_chain(
                sampler,
                initial,
                settings,
                np.random.RandomState(819),
                cancellation_requested=lambda: completed[0] >= transition,
                progress=progress,
            )
        except McmcCancelled as error:
            return error.checkpoint
        raise AssertionError("synthetic chain did not cancel")

    def test_round_trip_requires_exact_request_output_and_fingerprints(self):
        checkpoint = self._write()
        self.assertEqual(checkpoint.manifest["status"], "core_complete")
        self.assertEqual(checkpoint.chain_checkpoints, {})
        self.assertFalse(self.request.output_directory.exists())
        self.assertEqual(
            set(checkpoint.state_values),
            set(self.helper.solution.lm.state.layout.variable_keys),
        )

        resumed_payload = copy.deepcopy(self.helper.helper.payload)
        resumed_payload = self.helper._mcmc_payload_request(resumed_payload)
        resumed_payload["resume"] = True
        resumed = validate_batch_estimation_request(resumed_payload)
        self.assertEqual(resumed.fingerprint, self.request.fingerprint)
        loaded = self._load(resumed)
        self.assertEqual(loaded.manifest["output_directory"], str(resumed.output_directory))

        changed = copy.deepcopy(resumed_payload)
        changed["q"]["initial_diagonal"][0] = 2.0
        with self.assertRaisesRegex(ArtifactValidationError, "request_fingerprint"):
            self._load(validate_batch_estimation_request(changed))
        with self.assertRaisesRegex(ArtifactValidationError, "estimator_revision"):
            load_batch_estimation_checkpoint(
                self.request.output_directory,
                request=resumed,
                estimator_revision="changed-revision",
                configuration_fingerprint=self.configuration,
                controller_snapshot_fingerprint=self.controller,
            )

    def test_completed_core_reuses_map_em_and_laplace_without_optimization(self):
        checkpoint = self._write()
        inputs = RealEstimationInputs(
            request=self.request,
            flight_data=(self.helper.helper.flight,),
            initializations=(self.helper.helper.initialization,),
            parameter_chart=self.helper.helper.chart,
            geometry=self.helper.helper.geometry,
            actuator_parameters=self.helper.helper.actuators,
            scaling=StateScaling.unit(),
            loading_seconds=0.0,
        )
        solution, geometry, delay_static_geometry = restore_laplace_checkpoint(
            inputs,
            "recorded-mode",
            checkpoint.state_values,
            checkpoint.core.map_static,
            checkpoint.core.q_em,
            checkpoint.core.laplace,
        )
        self.assertEqual(solution.lm.iterations, ())
        np.testing.assert_allclose(
            geometry.covariance,
            checkpoint.core.laplace["covariance"],
        )
        self.assertAlmostEqual(
            delay_static_geometry.standard_deviation_seconds, 0.001
        )

    def test_chain_updates_are_content_addressed_and_cancel_is_resumable(self):
        checkpoint = self._write()
        first = self._partial_chain(3)
        second = self._partial_chain(5)
        save_batch_chain_checkpoint(checkpoint.root, first)
        first_manifest = read_json(checkpoint.root / "manifest.json")
        first_path = first_manifest["chain_checkpoints"]["chain-000"]["path"]
        save_batch_chain_checkpoint(checkpoint.root, second)
        second_manifest = read_json(checkpoint.root / "manifest.json")
        second_path = second_manifest["chain_checkpoints"]["chain-000"]["path"]
        self.assertNotEqual(first_path, second_path)
        self.assertTrue((checkpoint.root / first_path).is_file())
        self.assertTrue((checkpoint.root / second_path).is_file())

        mark_batch_checkpoint_cancelled(checkpoint.root, "user_requested")
        loaded = self._load()
        self.assertEqual(loaded.manifest["status"], "cancelled")
        self.assertEqual(
            loaded.chain_checkpoints["chain-000"].completed_transition, 5
        )
        self.assertFalse(self.request.output_directory.exists())
        self.assertEqual(
            batch_checkpoint_path(self.request.output_directory), checkpoint.root
        )

        referenced = checkpoint.root / second_path
        referenced.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ArtifactValidationError, "SHA-256"):
            self._load()


if __name__ == "__main__":
    unittest.main()
