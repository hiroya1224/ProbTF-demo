"""Optional offscreen E2E for portable projects and real artifact files."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

    from grape_param_estim_gui.main_window import MainWindow
except ImportError as error:
    raise unittest.SkipTest(
        "PySide6 or another production GUI dependency is unavailable: {}".format(
            error
        )
    )

from grape_param_estim.artifact_io import (
    begin_bundle,
    mark_bundle_complete,
)
from grape_param_estim_gui.project_io import (
    copy_bag_into_project,
    create_project_directory,
    freshness_fingerprint,
    new_project_manifest,
    save_project_archive,
    write_project_manifest,
)
from grape_param_estim_gui.state import ProjectStore


def _artifact_fixture():
    fixture_path = (
        Path(__file__).resolve().parents[5]
        / "tests"
        / "grape_param_estim"
        / "test_artifact_io.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_grape_real_artifact_fixture", fixture_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the strict artifact fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ArtifactIoTests()


class PortableProjectGuiE2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_real_manifest_npz_archive_restores_gui_state(self):
        fixture = _artifact_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = new_project_manifest("portable-artifact-e2e")
            project = create_project_directory(
                root / "source-projects", manifest
            )
            for bag_id, content in (
                ("bag-a", b"first bag"),
                ("bag-b", b"second bag"),
            ):
                source = root / (bag_id + ".bag")
                source.write_bytes(content)
                manifest["bags"].append(
                    copy_bag_into_project(project, source, bag_id)
                )
            manifest["selected_bag_ids"] = ["bag-a", "bag-b"]
            manifest["intervals"] = {
                "bag-a": {
                    "auto": [1.0, 2.0],
                    "selected": [1.0, 2.0],
                    "state": "AUTO",
                },
                "bag-b": {
                    "auto": [3.0, 5.0],
                    "selected": [3.0, 5.0],
                    "state": "LOCKED",
                },
            }
            manifest["configuration_fingerprints"] = {
                "bag-a": "complete:vehicle-a",
                "bag-b": "complete:vehicle-a",
            }
            manifest["controller_snapshots"] = {
                bag_id: {"gains": [[1.0, 0.1, 0.5]] * 4}
                for bag_id in ("bag-a", "bag-b")
            }
            manifest["estimator_settings"] = {
                "ensemble_size": 3,
                "maximum_iterations": 1,
            }
            project_fingerprint = freshness_fingerprint(manifest)

            run_root = project / "runs" / "run-a" / "assimilation_run"
            run_manifest = fixture._run_manifest()
            run_manifest["project_request_fingerprint"] = project_fingerprint
            begin_bundle(run_root, run_manifest)
            fixture._save(
                run_root / "shared_posterior.npz",
                **fixture._shared_arrays(),
            )
            fixture._save(
                run_root / "diagnostics.npz",
                **fixture._diagnostic_arrays(),
            )
            fixture._save(
                run_root / "bags" / "bag-a.npz",
                **fixture._run_bag_arrays(),
            )
            fixture._save(
                run_root / "bags" / "bag-b.npz",
                **fixture._run_bag_arrays(q_sufficient=False),
            )
            mark_bundle_complete(run_root)

            pid_root = (
                project
                / "pid_proposals"
                / "pid-a"
                / "pid_proposal_evaluation"
            )
            fixture._prepare_pid(pid_root)
            mark_bundle_complete(pid_root)

            manifest["current_assimilation_run_id"] = "run-a"
            manifest["current_pid_proposal_evaluation_id"] = "pid-a"
            manifest["run_request_fingerprint"] = project_fingerprint
            write_project_manifest(project, manifest)
            (project / "gui_state.json").write_text(
                json.dumps(
                    {
                        "schema": "grape-param-estim/gui-state/v1",
                        "current_bag_id": "bag-b",
                        "selected_member_id": 7,
                        "selected_mode_id": "nominal",
                        "selected_pid_proposal_id": "member-41",
                        "bags": {
                            "bag-b": {
                                "current_time": 0.22,
                                "view_range": [0.0, 0.4],
                            }
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            archive = save_project_archive(
                project, root / "portable-project.zip"
            )

            empty_store = ProjectStore(
                root / "empty",
                new_project_manifest("empty-project"),
            )
            with mock.patch(
                "grape_param_estim_gui.widgets.scene_3d.pv", None
            ), mock.patch(
                "grape_param_estim_gui.widgets.scene_3d.QtInteractor", None
            ):
                window = MainWindow(empty_store, root / "loaded-package")
                with mock.patch.object(
                    QFileDialog,
                    "getOpenFileName",
                    return_value=(str(archive), ""),
                ), mock.patch.object(
                    QMessageBox, "critical"
                ) as critical:
                    window.load_project()
                critical.assert_not_called()
                self.assertEqual(
                    empty_store.project_id, "portable-artifact-e2e"
                )
                self.assertEqual(empty_store.current_bag_id, "bag-b")
                self.assertEqual(empty_store.selected_member_id, 7)
                self.assertEqual(
                    empty_store.selected_pid_proposal_id, "member-41"
                )
                self.assertFalse(empty_store.results_stale)
                self.assertEqual(
                    tuple(
                        empty_store.parameter_ensemble.member_id.tolist()
                    ),
                    (41, 7, 99),
                )
                self.assertEqual(
                    empty_store.parameter_ensemble.constant_delay.tolist(),
                    [0.001, 0.007, 0.012],
                )
                self.assertEqual(
                    tuple(
                        empty_store.pid_evaluation.summary[
                            "candidate_id"
                        ].tolist()
                    ),
                    ("current", "member-41"),
                )
                self.assertIsNotNone(empty_store.get("bag-a").result)
                self.assertIsNotNone(empty_store.get("bag-b").result)
                window.close()


if __name__ == "__main__":
    unittest.main()
