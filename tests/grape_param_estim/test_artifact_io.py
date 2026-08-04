import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.artifact_io import (
    FLIGHT_INSPECTION_SCHEMA,
    INSPECTION_BUNDLE_SCHEMA,
    ArtifactStateError,
    ArtifactValidationError,
    IncompleteArtifactError,
    UnsupportedArtifactSchema,
    begin_bundle,
    load_inspection_bundle,
    load_npz_strict,
    mark_bundle_cancelled,
    mark_bundle_complete,
    read_json,
    read_manifest,
    request_fingerprint,
    request_fingerprint_file,
    write_json_atomic,
    write_npz_atomic,
)


class ArtifactIoTests(unittest.TestCase):
    @staticmethod
    def _manifest():
        return {
            "schema": INSPECTION_BUNDLE_SCHEMA,
            "bag_ids": ["bag-a"],
            "artifacts": {
                "bags": {
                    "bag-a": {
                        "inspection": "bags/bag-a.inspection.json",
                        "preview": "bags/bag-a.preview.npz",
                    }
                }
            },
        }

    @staticmethod
    def _inspection():
        return {
            "schema": FLIGHT_INSPECTION_SCHEMA,
            "bag_id": "bag-a",
            "bag_path": "/archive/bags/a.bag",
            "bag_size": 123,
            "bag_mtime": 1.5,
            "bag_sha256": "a" * 64,
            "record_time_start": 100.0,
            "record_time_end": 110.0,
            "topic_contract": [],
            "complete_episodes": [],
            "state5_intervals": [],
            "recommended_interval": {
                "episode_index": 0,
                "reason": "preferred_state",
                "warnings": [],
                "interval": {
                    "start_local_time": 2.0,
                    "end_local_time": 8.0,
                },
            },
            "warnings": [],
            "controller_snapshot": {},
            "controller_flags": {},
            "configuration_fingerprint": {
                "value": "complete:vehicle-a",
                "complete": True,
                "missing_components": [],
            },
            "estimated_work_units": {
                "sample_count": 151,
                "knot_count": 151,
                "lag_profile_point_units": 42,
                "nonlinear_iteration_units": 1260,
                "mcmc_proposal_units": 0,
                "estimate_kind": (
                    "upper_bound_excluding_lm_retries_and_q_backtracking"
                ),
            },
            "status": "ready",
        }

    @staticmethod
    def _preview():
        time = np.asarray((0.0, 0.1, 0.2))
        return {
            "time": time,
            "position": np.zeros((3, 3)),
            "orientation_xyzw": np.tile(
                (0.0, 0.0, 0.0, 1.0), (3, 1)
            ),
            "reference_position": np.zeros((3, 3)),
            "reference_rpy": np.zeros((3, 3)),
            "flight_state": np.asarray((3, 5, 6), dtype=np.int32),
        }

    def _prepare_inspection(self, root):
        begin_bundle(root, self._manifest())
        write_json_atomic(
            root / "bags" / "bag-a.inspection.json", self._inspection()
        )
        write_npz_atomic(
            root / "bags" / "bag-a.preview.npz", self._preview()
        )

    def test_request_fingerprint_is_canonical_and_finite(self):
        first = {
            "selected": ["bag-a", "bag-b"],
            "settings": {"iterations": 2, "seed": 4},
        }
        reordered = {
            "settings": {"seed": 4, "iterations": 2},
            "selected": ["bag-a", "bag-b"],
        }
        self.assertEqual(
            request_fingerprint(first), request_fingerprint(reordered)
        )
        self.assertNotEqual(
            request_fingerprint(first),
            request_fingerprint({**first, "selected": ["bag-b", "bag-a"]}),
        )
        with self.assertRaises(ArtifactValidationError):
            request_fingerprint({"bad": float("nan")})
        with self.assertRaises(ArtifactValidationError):
            request_fingerprint({1: "non-string key"})

    def test_strict_json_round_trip_and_file_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            value = {"beta": [2, 3], "alpha": 1.25}
            write_json_atomic(path, value)
            self.assertEqual(read_json(path), value)
            self.assertEqual(
                request_fingerprint_file(path), request_fingerprint(value)
            )

            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"x":1,"x":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                ArtifactValidationError, "duplicate key"
            ):
                read_json(duplicate)

            nonfinite = Path(directory) / "nonfinite.json"
            nonfinite.write_text('{"x":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                ArtifactValidationError, "non-finite"
            ):
                read_json(nonfinite)

    def test_strict_npz_round_trip_rejects_missing_and_object_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "arrays.npz"
            write_npz_atomic(path, {"safe": np.arange(3, dtype=float)})
            arrays = load_npz_strict(path, required_keys=("safe",))
            np.testing.assert_array_equal(arrays["safe"], (0.0, 1.0, 2.0))
            with self.assertRaisesRegex(
                ArtifactValidationError, "missing required keys"
            ):
                load_npz_strict(path, required_keys=("absent",))
            with self.assertRaisesRegex(
                ArtifactValidationError, "object dtype"
            ):
                write_npz_atomic(
                    root / "unsafe.npz",
                    {"unsafe": np.asarray(({"x": 1},), dtype=object)},
                )

    def test_inspection_completion_is_atomic_and_strictly_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "inspection"
            self._prepare_inspection(root)
            with self.assertRaises(IncompleteArtifactError):
                load_inspection_bundle(root)
            mark_bundle_complete(root)
            bundle = load_inspection_bundle(root)
            self.assertEqual(bundle.manifest["status"], "complete")
            self.assertEqual(
                bundle.inspections["bag-a"]["bag_sha256"], "a" * 64
            )
            self.assertEqual(bundle.previews["bag-a"]["position"].shape, (3, 3))
            with self.assertRaises(ArtifactStateError):
                mark_bundle_cancelled(root, "too_late")

    def test_cancelled_manifest_is_authoritative_over_partial_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cancelled"
            self._prepare_inspection(root)
            mark_bundle_cancelled(root, "user_requested")
            manifest = read_manifest(root)
            self.assertEqual(manifest["status"], "cancelled")
            self.assertEqual(manifest["cancellation_reason"], "user_requested")
            with self.assertRaises(IncompleteArtifactError):
                load_inspection_bundle(root)
            mark_bundle_cancelled(root, "user_requested")
            with self.assertRaises(ArtifactStateError):
                mark_bundle_cancelled(root, "different_reason")

    def test_unknown_schema_and_escaping_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(UnsupportedArtifactSchema):
                begin_bundle(root / "unknown", {"schema": "unknown/v1"})

            escaped = root / "escaped"
            manifest = self._manifest()
            manifest["artifacts"]["bags"]["bag-a"]["inspection"] = (
                "../outside.json"
            )
            begin_bundle(escaped, manifest)
            with self.assertRaisesRegex(
                ArtifactValidationError, "inside the bundle"
            ):
                mark_bundle_complete(escaped)
            self.assertEqual(read_manifest(escaped)["status"], "writing")


if __name__ == "__main__":
    unittest.main()
