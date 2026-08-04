from pathlib import Path
import tempfile
import unittest

from grape_param_estim.batch_artifact import BATCH_ESTIMATION_RUN_SCHEMA
from grape_param_estim_gui.workflow_artifacts import (
    artifact_ref_from_validated_bundle,
)


class WorkflowArtifactContractTests(unittest.TestCase):
    def test_batch_writer_descriptors_bind_without_post_run_shape_error(self):
        request_fingerprint = "sha256:" + "1" * 64
        stage_input = "sha256:" + "2" * 64
        digest = "sha256:" + "3" * 64
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            artifact = project / "runs" / "batch-a" / "estimation_run"
            artifact.mkdir(parents=True)
            manifest = {
                "schema": BATCH_ESTIMATION_RUN_SCHEMA,
                "status": "complete",
                "run_id": "batch-a",
                "request_fingerprint": request_fingerprint,
                "artifacts": {
                    "map_static": {
                        "path": "map_static.npz",
                        "sha256": digest,
                    },
                    "bags": {
                        "bag-a": {
                            "path": "bags/bag-a.npz",
                            "sha256": "sha256:" + "4" * 64,
                        }
                    },
                },
            }
            reference = artifact_ref_from_validated_bundle(
                project_root=project,
                artifact_root=artifact,
                manifest=manifest,
                expected_stage_id="batch_estimation",
                expected_stage_input=stage_input,
                expected_request_fingerprint=request_fingerprint,
            )
        self.assertEqual(reference.schema, BATCH_ESTIMATION_RUN_SCHEMA)
        self.assertEqual(reference.artifact_id, "batch-a")
        self.assertEqual(
            reference.relative_path,
            "runs/batch-a/estimation_run",
        )


if __name__ == "__main__":
    unittest.main()
