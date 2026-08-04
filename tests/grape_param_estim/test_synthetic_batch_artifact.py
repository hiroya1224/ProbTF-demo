import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import grape_param_estim.synthetic_batch as synthetic_batch_module
from grape_param_estim.synthetic import (
    SYNTHETIC_BATCH_TRUTH_SCHEMA,
    SYNTHETIC_BATCH_TRUTH_SUMMARY_SCHEMA,
    generate_perfect_model_batch_trajectory,
    load_synthetic_batch_truth_artifact,
    save_synthetic_batch_truth_artifact,
)
from grape_param_estim.synthetic_cli import main as synthetic_cli_main


class SyntheticBatchTruthArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = 812
        cls.trajectory = generate_perfect_model_batch_trajectory(
            interval_count=20,
            seed=cls.seed,
        )

    def _save(self, directory, name="truth.npz"):
        return save_synthetic_batch_truth_artifact(
            str(Path(directory) / name),
            self.trajectory,
            generator_seed=self.seed,
        )

    def test_strict_pickle_free_round_trip_preserves_solver_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory)
            loaded = load_synthetic_batch_truth_artifact(str(path))

            self.assertEqual(loaded.schema, SYNTHETIC_BATCH_TRUTH_SCHEMA)
            self.assertEqual(loaded.generator_seed, self.seed)
            self.assertEqual(len(loaded.payload_sha256), 64)
            self.assertEqual(len(loaded.parameter_coordinate_order), 18)
            self.assertEqual(loaded.units["times"], "s")
            for name in (
                "times",
                "position",
                "rotation",
                "linear_velocity",
                "angular_velocity",
                "actuator_thrust",
                "gimbal_angle",
                "truth_parameter_coordinates",
            ):
                np.testing.assert_array_equal(
                    getattr(loaded.trajectory, name),
                    getattr(self.trajectory, name),
                )
            decoded = self.trajectory.parameter_chart.decode(
                self.trajectory.truth_parameter_coordinates
            )
            self.assertAlmostEqual(loaded.truth_parameters.mass, decoded.mass)
            np.testing.assert_array_equal(
                loaded.truth_parameters.inertia, decoded.inertia
            )
            np.testing.assert_array_equal(
                loaded.truth_parameters.force_effectiveness,
                decoded.force_effectiveness,
            )
            self.assertGreater(np.ptp(loaded.trajectory.time_step), 0.005)
            residual = np.vstack(
                tuple(
                    loaded.trajectory.dynamics_evaluation(
                        index,
                        loaded.trajectory.truth_parameter_coordinates,
                    ).residual
                    for index in range(loaded.trajectory.interval_count)
                )
            )
            self.assertLess(np.linalg.norm(residual, ord=np.inf), 3.0e-10)

            with np.load(str(path), allow_pickle=False) as archive:
                self.assertTrue(
                    all(archive[name].dtype.kind != "O" for name in archive.files)
                )
                metadata = json.loads(str(archive["metadata_json"].item()))
                self.assertEqual(metadata["schema"], SYNTHETIC_BATCH_TRUTH_SCHEMA)
                self.assertEqual(metadata["units"]["actuator_thrust"], "N; per rotor")
                self.assertEqual(
                    metadata["provenance"]["dynamics_factor"],
                    "grape_param_estim.batch.factors.dynamics."
                    "evaluate_raw_dynamics_residual",
                )
                self.assertEqual(
                    metadata["provenance"]["construction_derivatives"],
                    "analytic production factor Jacobian",
                )
                self.assertFalse(
                    metadata["provenance"]["finite_difference_derivatives"]
                )

    def test_checksum_and_exact_member_contract_reject_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self._save(directory, "original.npz")
            with np.load(str(original), allow_pickle=False) as archive:
                payload = {name: archive[name].copy() for name in archive.files}

            changed = dict(payload)
            changed["position"] = changed["position"].copy()
            changed["position"][3, 1] += 0.25
            changed_path = Path(directory) / "changed.npz"
            np.savez_compressed(str(changed_path), **changed)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_synthetic_batch_truth_artifact(str(changed_path))

            for label, transform in (
                (
                    "missing",
                    lambda values: {
                        key: value
                        for key, value in values.items()
                        if key != "angular_velocity"
                    },
                ),
                (
                    "extra",
                    lambda values: dict(values, unexpected=np.zeros(1)),
                ),
            ):
                with self.subTest(label=label):
                    malformed_path = Path(directory) / (label + ".npz")
                    np.savez_compressed(
                        str(malformed_path),
                        **transform(payload),
                    )
                    with self.assertRaisesRegex(ValueError, "archive members"):
                        load_synthetic_batch_truth_artifact(str(malformed_path))

            wrong_metadata = dict(payload)
            metadata = json.loads(str(wrong_metadata["metadata_json"].item()))
            metadata["units"]["times"] = "milliseconds"
            wrong_metadata["metadata_json"] = np.asarray(
                json.dumps(metadata, separators=(",", ":"), sort_keys=True)
            )
            digest_payload = {
                name: wrong_metadata[name]
                for name in synthetic_batch_module._SYNTHETIC_BATCH_PAYLOAD_NAMES
            }
            wrong_metadata["payload_sha256"] = np.asarray(
                synthetic_batch_module._payload_sha256(digest_payload)
            )
            wrong_metadata_path = Path(directory) / "wrong-metadata.npz"
            np.savez_compressed(str(wrong_metadata_path), **wrong_metadata)
            with self.assertRaisesRegex(ValueError, "unit contract"):
                load_synthetic_batch_truth_artifact(str(wrong_metadata_path))

    def test_save_requires_an_explicit_npz_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "npz suffix"):
                self._save(directory, "truth.data")


class SyntheticBatchTruthCliTests(unittest.TestCase):
    def test_cli_emits_verified_new_solver_truth_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "cli-truth.npz"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                return_code = synthetic_cli_main(
                    (
                        "--output",
                        str(destination),
                        "--interval-count",
                        "18",
                        "--seed",
                        "404",
                    )
                )
            self.assertEqual(return_code, 0)
            summary = json.loads(output.getvalue())
            self.assertEqual(summary["schema"], SYNTHETIC_BATCH_TRUTH_SUMMARY_SCHEMA)
            self.assertEqual(summary["artifact_schema"], SYNTHETIC_BATCH_TRUTH_SCHEMA)
            self.assertEqual(summary["intervals"], 18)
            self.assertEqual(summary["samples"], 19)
            self.assertEqual(summary["truth_parameter_dimension"], 18)
            self.assertTrue(summary["perfect_model"])
            self.assertTrue(summary["production_factor_analytic_jacobian"])
            self.assertGreater(
                summary["maximum_time_step_s"], summary["minimum_time_step_s"]
            )
            self.assertEqual(
                summary["direct_truth_channels"],
                [
                    "position",
                    "rotation_so3",
                    "linear_velocity",
                    "angular_velocity",
                    "actuator_thrust",
                    "gimbal_angle",
                ],
            )
            artifact = load_synthetic_batch_truth_artifact(str(destination))
            self.assertEqual(artifact.generator_seed, 404)
            self.assertEqual(artifact.payload_sha256, summary["payload_sha256"])


if __name__ == "__main__":
    unittest.main()
