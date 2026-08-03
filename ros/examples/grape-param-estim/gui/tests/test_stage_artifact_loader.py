from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from grape_param_estim_gui import artifact_loader


def _stage2_bundle(root: Path):
    members = 2
    samples = 3
    member_id = np.asarray((7, 11), dtype=np.int64)
    shared = {
        "member_id": member_id,
        "final_shared_coordinates": np.arange(38, dtype=float).reshape(2, 19),
        "mass": np.asarray((2.1, 2.3)),
        "inertia": np.asarray((np.eye(3), 1.2 * np.eye(3))),
        "cog_offset": np.asarray(((0.01, 0.0, 0.02), (0.0, -0.01, 0.03))),
        "force_effectiveness": np.full((members, 4), 0.95),
        "torque_effectiveness": np.full((members, 4), 1.05),
        "constant_delay": np.asarray((0.014, 0.028)),
        "ridge_covariance": np.eye(19),
        "ridge_eigenvalues": np.ones(19),
        "ridge_eigenvectors": np.eye(19),
        "expected_physical_ridge_direction": np.eye(19)[0],
        "expected_physical_ridge_variance": np.asarray((1.0,)),
        "ensemble_rank": np.asarray((1,), dtype=np.int64),
    }
    times = np.asarray((0.0, 0.1, 0.2))
    identity = np.tile(np.asarray((0.0, 0.0, 0.0, 1.0)), (samples, 1))
    yaw_quaternion = np.tile(
        np.asarray((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5))),
        (samples, 1),
    )
    bag = {
        "times": times,
        "record_times": times + 100.0,
        "reference_position": np.zeros((samples, 3)),
        "reference_orientation_xyzw": yaw_quaternion,
        "observation_position": np.ones((samples, 3)),
        "observation_orientation_xyzw": identity,
        "nominal_position": np.full((samples, 3), 0.25),
        "nominal_orientation_xyzw": identity,
        "smoothed_position": np.zeros((members, samples, 3)),
        "smoothed_orientation_xyzw": np.tile(identity, (members, 1, 1)),
        "smoothed_correction_translation": np.full(
            (members, samples, 3), 0.1
        ),
        "smoothed_correction_rotation_vector": np.full(
            (members, samples, 3), 0.01
        ),
        "observed_correction_translation": np.full((samples, 3), 0.2),
        "observed_correction_rotation_vector": np.full((samples, 3), 0.02),
        "smoothed_residual_wrench": np.full((members, samples, 6), 0.03),
        "fixed_q_stationary_variance": np.arange(1.0, 7.0),
        "fixed_r_translation_covariance": np.diag((0.1, 0.2, 0.3)),
        "fixed_r_rotation_covariance": np.diag((0.01, 0.02, 0.03)),
        "fixed_correlation_time": np.asarray((0.4,)),
        "filter_log_likelihood_by_time": np.asarray((-2.0, -3.0, -4.0)),
    }
    metadata = {
        "source_path": "/flights/bag-a.bag",
        "source_sha256": "a" * 64,
        "source_size_bytes": 1234,
        "episode_index": 2,
        "configuration_fingerprint": "manual-group:sha256:" + "b" * 64,
        "time_basis": "episode_relative_seconds_with_rosbag_record_times",
        "requested_interval_record_seconds": [99.9, 100.3],
        "effective_interval_record_seconds": [100.0, 100.2],
        "effective_interval_local_seconds": [4.0, 4.2],
        "episode_provenance": {
            "bag_path": "/flights/bag-a.bag",
            "selected_flight_state": 5,
        },
        "episode_provenance_fingerprint": "sha256:" + "c" * 64,
        "controller_snapshot": {"groups": ["xy", "z", "roll_pitch", "yaw"]},
        "controller_snapshot_fingerprint": "sha256:" + "d" * 64,
        "controller_configuration": {"xy_control_mode": "position"},
        "controller_configuration_fingerprint": "sha256:" + "e" * 64,
        "model_provenance": {"algorithm": "closed-loop-stepper-v1"},
        "model_provenance_fingerprint": "sha256:" + "f" * 64,
    }
    manifest = {
        "run_id": "parameter-run",
        "request_fingerprint": "sha256:" + "1" * 64,
        "project_fingerprint": "sha256:" + "2" * 64,
        "path_semantics": {
            "kind": (
                "sequential_enrts_marginal_with_time_varying_static_coordinates"
            ),
            "static_coordinates_at_each_time_are_actual": True,
            "earlier_bags_recomputed_with_final_shared_posterior": False,
        },
        "bags": {"bag-a": metadata},
    }
    return SimpleNamespace(
        root=root,
        manifest=manifest,
        shared_posterior=shared,
        bags={"bag-a": bag},
        bag_ids=("bag-a",),
    )


class StageArtifactLoaderTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_diagonal_q_loader_and_fingerprint_delegate_to_strict_backend(self):
        bundle = SimpleNamespace(
            manifest={"schema": "q", "status": "complete", "value": 3}
        )
        with mock.patch.object(
            artifact_loader.diagonal_q_artifact_io,
            "load_diagonal_q_artifact",
            return_value=bundle,
        ) as strict_loader:
            self.assertIs(
                artifact_loader.load_diagonal_q_stage(self.root), bundle
            )
            expected = artifact_loader.artifact_io.request_fingerprint(
                bundle.manifest
            )
            self.assertEqual(
                artifact_loader.diagonal_q_stage_fingerprint(self.root),
                expected,
            )
        self.assertEqual(strict_loader.call_count, 2)

    def test_stage2_adapter_preserves_only_measured_and_stored_quantities(self):
        bundle = _stage2_bundle(self.root)
        original_manifest = dict(bundle.manifest)
        with mock.patch.object(
            artifact_loader.augmented_parameter_artifact_io,
            "load_augmented_parameter_artifact",
            return_value=bundle,
        ) as strict_loader:
            run = artifact_loader.load_augmented_parameter_assimilation(
                self.root
            )
        strict_loader.assert_called_once_with(self.root)
        shared = run.shared_posterior
        np.testing.assert_array_equal(
            shared.parameter_coordinate,
            bundle.shared_posterior["final_shared_coordinates"],
        )
        np.testing.assert_array_equal(
            shared.cog, bundle.shared_posterior["cog_offset"]
        )
        np.testing.assert_array_equal(
            shared.ridge["expected_direction"],
            bundle.shared_posterior["expected_physical_ridge_direction"],
        )
        self.assertEqual(shared.mode, {})
        self.assertEqual(shared.iteration_diagnostics, {})
        self.assertEqual(run.diagnostics, {})

        flight = run.bag_results["bag-a"]
        np.testing.assert_allclose(
            flight.reference_rpy[:, 2], np.pi / 2.0, atol=1.0e-15
        )
        np.testing.assert_array_equal(
            flight.member_position, bundle.bags["bag-a"]["smoothed_position"]
        )
        np.testing.assert_array_equal(
            flight.correction_translation,
            bundle.bags["bag-a"]["smoothed_correction_translation"],
        )
        np.testing.assert_array_equal(
            flight.residual_wrench,
            bundle.bags["bag-a"]["smoothed_residual_wrench"],
        )
        self.assertEqual(flight.objective_contribution, -9.0)
        self.assertIsNone(flight.flight_state)
        self.assertIsNone(flight.q_resolution_sufficient)
        self.assertEqual(
            set(flight.calibration),
            {
                "fixed_q_stationary_variance",
                "fixed_r_translation_covariance",
                "fixed_r_rotation_covariance",
                "fixed_correlation_time",
            },
        )
        self.assertEqual(flight.coverage, {})
        self.assertEqual(
            flight.provenance["effective_interval_local_seconds"],
            [4.0, 4.2],
        )
        self.assertNotIn("command", flight.provenance)
        self.assertEqual(flight.provenance["selected_flight_state"], 5)
        self.assertEqual(
            run.manifest["project_request_fingerprint"],
            bundle.manifest["project_fingerprint"],
        )
        self.assertEqual(bundle.manifest, original_manifest)
        self.assertEqual(len(run.warnings), 1)
        self.assertIn("sequential EnRTS marginals", run.warnings[0])
        self.assertIn("not recomputed", run.warnings[0])

    def test_strict_stage_backend_errors_are_wrapped_for_gui(self):
        backend_error = artifact_loader.artifact_io.ArtifactValidationError(
            "bad digest"
        )
        with mock.patch.object(
            artifact_loader.diagonal_q_artifact_io,
            "load_diagonal_q_artifact",
            side_effect=backend_error,
        ):
            with self.assertRaisesRegex(
                artifact_loader.GuiArtifactError, "bad digest"
            ):
                artifact_loader.load_diagonal_q_stage(self.root)
        with mock.patch.object(
            artifact_loader.augmented_parameter_artifact_io,
            "load_augmented_parameter_artifact",
            side_effect=backend_error,
        ):
            with self.assertRaisesRegex(
                artifact_loader.GuiArtifactError, "bad digest"
            ):
                artifact_loader.load_augmented_parameter_assimilation(
                    self.root
                )


if __name__ == "__main__":
    unittest.main()
