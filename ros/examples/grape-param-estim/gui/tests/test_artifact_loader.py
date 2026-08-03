from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from grape_param_estim_gui import artifact_loader


def _run_bundle(root: Path):
    member_id = np.array([13, 42], dtype=np.int64)
    shared = {
        "member_id": member_id,
        "parameter_coordinates": np.arange(38, dtype=float).reshape(2, 19),
        "mass": np.array([1.8, 2.1]),
        "inertia": np.stack((np.eye(3), 1.2 * np.eye(3))),
        "cog": np.array([[0.01, 0.0, -0.03], [0.02, -0.01, -0.04]]),
        "force_effectiveness": np.full((2, 4), 0.9),
        "torque_effectiveness": np.full((2, 4), 1.1),
        "constant_delay": np.array([0.017, 0.031]),
    }
    time = np.array([0.0, 0.1, 0.2])
    quaternion = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (3, 1))
    correction_rotvec = np.arange(18, dtype=float).reshape(2, 3, 3) * 0.001
    bag = {
        "times": time,
        "record_times": time + 100.0,
        "reference_position": np.zeros((3, 3)),
        "reference_rpy": np.zeros((3, 3)),
        "observed_position": np.zeros((3, 3)),
        "observed_orientation_xyzw": quaternion,
        "nominal_position": np.zeros((3, 3)),
        "nominal_orientation_xyzw": quaternion,
        "posterior_position": np.zeros((2, 3, 3)),
        "posterior_orientation_xyzw": np.tile(quaternion, (2, 1, 1)),
        "correction_translation": np.zeros((2, 3, 3)),
        "correction_rotation_vector": correction_rotvec,
        "observed_correction_translation": np.zeros((3, 3)),
        "observed_correction_rotation_vector": np.ones((3, 3)) * 0.01,
        "residual_wrench_interval": np.zeros((2, 2, 6)),
        "q_resolution_sufficient": np.array([False]),
        "pose_component_coverage": np.array([0.875]),
        "objective_contribution": np.array([2.0, 4.0]),
        "provenance_bag_sha256": np.array(["a" * 64]),
    }
    diagnostics = {
        "ridge_eigenvalues": np.arange(19, dtype=float),
        "mode_weight": np.array([1.0]),
        "objective": np.array([8.0, 5.0]),
    }
    return SimpleNamespace(
        root=root,
        manifest={"run_id": "run-a", "request_fingerprint": "sha256:abc"},
        shared_posterior=shared,
        diagnostics=diagnostics,
        bags={"bag-a": bag},
        warnings=("bag-a: Q resolution is insufficient",),
    ), correction_rotvec


class ArtifactLoaderAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_run_adapter_preserves_ids_tau_raw_paths_and_q_warning(self):
        bundle, correction_rotvec = _run_bundle(self.root)
        with mock.patch.object(
            artifact_loader.artifact_io, "load_assimilation_run", return_value=bundle
        ) as strict_loader:
            run = artifact_loader.load_assimilation(self.root)
        strict_loader.assert_called_once_with(self.root)
        np.testing.assert_array_equal(run.shared_posterior.member_id, [13, 42])
        np.testing.assert_allclose(run.shared_posterior.constant_delay, [0.017, 0.031])
        np.testing.assert_allclose(run.shared_posterior.equal_weights, [0.5, 0.5])
        self.assertFalse(hasattr(run.shared_posterior, "weight"))
        np.testing.assert_array_equal(
            run.bag_results["bag-a"].correction_rotation_vector,
            correction_rotvec,
        )
        self.assertFalse(run.bag_results["bag-a"].q_resolution_sufficient)
        self.assertEqual(run.warnings, ("bag-a: Q resolution is insufficient",))
        self.assertEqual(run.bag_results["bag-a"].objective_contribution, 3.0)
        self.assertEqual(run.bag_results["bag-a"].coverage["value"], 0.875)

    def test_no_missing_tau_fallback_exists(self):
        bundle, _correction = _run_bundle(self.root)
        del bundle.shared_posterior["constant_delay"]
        with mock.patch.object(
            artifact_loader.artifact_io, "load_assimilation_run", return_value=bundle
        ):
            with self.assertRaisesRegex(KeyError, "constant_delay"):
                artifact_loader.load_assimilation(self.root)

    def test_unknown_schema_error_from_strict_backend_is_not_hidden(self):
        error = artifact_loader.artifact_io.UnsupportedArtifactSchema("unknown")
        with mock.patch.object(
            artifact_loader.artifact_io, "load_assimilation_run", side_effect=error
        ):
            with self.assertRaisesRegex(
                artifact_loader.artifact_io.UnsupportedArtifactSchema, "unknown"
            ):
                artifact_loader.load_assimilation(self.root)

    def test_pid_yaml_is_loaded_as_read_only_source_text(self):
        proposed = self.root / "proposed.yaml"
        difference = self.root / "difference.yaml"
        proposed.write_text("xy:\n  p_gain: 1.5\n")
        difference.write_text("xy.p_gain: +0.2\n")
        bundle = SimpleNamespace(
            root=self.root,
            manifest={"evaluation_id": "pid-a"},
            proposal_ensemble={"source_member_id": np.array([13])},
            summary={"candidate_id": np.array(["current"])},
            bags={},
            proposed_yaml_path=proposed,
            proposed_diff_yaml_path=difference,
        )
        with mock.patch.object(
            artifact_loader.artifact_io,
            "load_pid_proposal_evaluation",
            return_value=bundle,
        ):
            evaluation = artifact_loader.load_pid_evaluation(self.root)
        self.assertEqual(evaluation.proposed_yaml, proposed.read_text())
        self.assertEqual(evaluation.proposed_diff_yaml, difference.read_text())


if __name__ == "__main__":
    unittest.main()
