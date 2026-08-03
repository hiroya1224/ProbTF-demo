from pathlib import Path
import tempfile
import unittest

from grape_param_estim_gui.workflow import (
    WorkflowError,
    canonical_fingerprint,
)
from grape_param_estim_gui.workflow_artifacts import (
    artifact_ref_from_validated_bundle,
)


class WorkflowArtifactBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.project = Path(self.temporary.name) / "project"
        self.artifact = self.project / "runs" / "run-a" / "diagonal_q"
        self.artifact.mkdir(parents=True)
        (self.artifact / "q.npz").write_bytes(b"payload")
        self.stage_input = canonical_fingerprint({"stage": "q"})
        self.request = canonical_fingerprint({"request": "q"})
        self.manifest = {
            "schema": "test/q/v1",
            "status": "complete",
            "run_id": "run-a",
            "stage_id": "diagonal_q",
            "stage_input_fingerprint": self.stage_input,
            "request_fingerprint": self.request,
            "artifacts": {
                "q": {
                    "path": "q.npz",
                    "sha256": "sha256:" + "1" * 64,
                    "size_bytes": 7,
                }
            },
        }

    def tearDown(self):
        self.temporary.cleanup()

    def _bind(self, **changes):
        manifest = dict(self.manifest)
        manifest.update(changes)
        return artifact_ref_from_validated_bundle(
            project_root=self.project,
            artifact_root=self.artifact,
            manifest=manifest,
            expected_stage_id="diagonal_q",
            expected_stage_input=self.stage_input,
            expected_request_fingerprint=self.request,
        )

    def test_complete_bundle_is_bound_to_exact_stage_and_request(self):
        reference = self._bind()
        self.assertEqual(reference.artifact_id, "run-a")
        self.assertEqual(
            reference.relative_path, "runs/run-a/diagonal_q"
        )
        self.assertEqual(reference.schema, "test/q/v1")
        self.assertTrue(reference.content_fingerprint.startswith("sha256:"))
        self.assertTrue(
            reference.completion_fingerprint.startswith("sha256:")
        )

    def test_incomplete_or_mismatched_bundle_is_rejected(self):
        for changes, message in (
            ({"status": "writing"}, "complete artifact"),
            ({"stage_id": "other"}, "stage_id"),
            (
                {"stage_input_fingerprint": canonical_fingerprint("other")},
                "stage input",
            ),
            (
                {"request_fingerprint": canonical_fingerprint("other")},
                "request fingerprint",
            ),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(
                WorkflowError, message
            ):
                self._bind(**changes)

    def test_duplicate_declared_payload_path_is_rejected(self):
        descriptor = self.manifest["artifacts"]["q"]
        with self.assertRaisesRegex(WorkflowError, "more than once"):
            self._bind(
                artifacts={"first": descriptor, "second": descriptor}
            )

    def test_artifact_outside_project_is_rejected(self):
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        with self.assertRaisesRegex(WorkflowError, "inside the current project"):
            artifact_ref_from_validated_bundle(
                project_root=self.project,
                artifact_root=outside,
                manifest=self.manifest,
                expected_stage_id="diagonal_q",
                expected_stage_input=self.stage_input,
                expected_request_fingerprint=self.request,
            )


if __name__ == "__main__":
    unittest.main()
