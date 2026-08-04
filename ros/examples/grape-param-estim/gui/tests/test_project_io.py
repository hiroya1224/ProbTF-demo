import copy
import json
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

import numpy as np

from grape_param_estim_gui.project_io import (
    GUI_STATE_SCHEMA,
    PROJECT_ARTIFACT_LOADER_ID,
    PROJECT_ARTIFACT_LOADER_VERSION,
    PROJECT_MANIFEST_NAME,
    ProjectIoError,
    copy_bag_into_project,
    create_project_directory,
    freshness_fingerprint,
    load_project_archive,
    new_project_manifest,
    read_project_manifest,
    result_is_fresh,
    read_gui_state,
    save_project_archive,
    sha256_file,
    validate_project_manifest,
    validate_gui_state,
    write_gui_state,
    write_project_manifest,
)


def _complete_project(root: Path, project_id: str = "flight-test"):
    source = root / "source.bag"
    source.write_bytes(b"ROS bag bytes\x00" * 256)
    manifest = new_project_manifest(project_id)
    project = create_project_directory(root / "projects", manifest)
    bag = copy_bag_into_project(project, source, "bag-a")
    manifest["bags"].append(bag)
    manifest["selected_bag_ids"] = ["bag-a"]
    manifest["intervals"] = {
        "bag-a": {"auto": [1.0, 5.0], "selected": [1.2, 4.8], "state": "MODIFIED"}
    }
    manifest["configuration_fingerprints"] = {"bag-a": "complete:abc"}
    manifest["controller_snapshots"] = {"bag-a": {"gains": [[[1.0, 2.0, 3.0]]]}}
    manifest["estimator_settings"] = {"solver_settings": {"maximum_iterations": 5}}
    manifest["current_estimation_run_id"] = "run-a"
    manifest["current_pid_proposal_evaluation_id"] = "pid-a"
    write_project_manifest(project, manifest)
    write_gui_state(
        project,
        {
            "schema": GUI_STATE_SCHEMA,
            "current_bag_id": "bag-a",
            "selected_sample_id": "sample-17",
            "selected_mode_id": "mode-a",
            "selected_pid_proposal_id": "current",
            "bags": {
                "bag-a": {"current_time": 2.0, "view_range": [1.0, 5.0]}
            },
        },
        registered_bag_ids=("bag-a",),
    )
    (project / "inspection" / "payload.txt").write_text("inspection")
    run_root = project / "runs" / "run-a" / "estimation_run"
    run_root.mkdir(parents=True)
    (run_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "grape-param-estim/batch-estimation-run/v1",
                "status": "complete",
                "run_id": "run-a",
                "artifacts": {"mcmc_samples": "mcmc_samples.npz"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        str(run_root / "mcmc_samples.npz"),
        sample_id=np.asarray(("sample-17", "sample-29")),
        mass=np.asarray((2.1, 2.2)),
        constant_delay=np.asarray((0.012, 0.019)),
    )
    pid_root = (
        project
        / "pid_proposals"
        / "pid-a"
        / "pid_proposal_evaluation"
    )
    pid_root.mkdir(parents=True)
    (pid_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "grape-param-estim/pid-proposal-evaluation/v2",
                "status": "complete",
                "evaluation_id": "pid-a",
                "source_run_id": "run-a",
                "artifacts": {"summary": "summary.npz"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        str(pid_root / "summary.npz"),
        candidate_id=np.asarray(("current", "member-17-exact")),
        current_pid=np.ones((4, 3)),
        proposed_pid=np.stack(
            (np.ones((4, 3)), 1.1 * np.ones((4, 3)))
        ),
    )
    (pid_root / "proposed_GimbalrotorControl.yaml").write_text(
        "xy:\n  p_gain: 1.1\n", encoding="utf-8"
    )
    return source, project, read_project_manifest(project)


class ProjectIoTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_zip64_round_trip_is_self_contained_and_sha_verified(self):
        source, project, manifest = _complete_project(self.root)
        archive = save_project_archive(project, self.root / "portable.zip")
        source.unlink()
        restored = load_project_archive(archive, self.root / "restored-projects")
        loaded = read_project_manifest(restored, verify_bags=True)
        self.assertEqual(loaded["project_id"], manifest["project_id"])
        self.assertEqual((restored / "inspection" / "payload.txt").read_text(), "inspection")
        with np.load(
            str(
                restored
                / "runs"
                / "run-a"
                / "estimation_run"
                / "mcmc_samples.npz"
            ),
            allow_pickle=False,
        ) as posterior:
            np.testing.assert_array_equal(
                posterior["sample_id"], ("sample-17", "sample-29")
            )
            np.testing.assert_allclose(
                posterior["constant_delay"], (0.012, 0.019)
            )
        pid_root = (
            restored
            / "pid_proposals"
            / "pid-a"
            / "pid_proposal_evaluation"
        )
        with np.load(str(pid_root / "summary.npz"), allow_pickle=False) as summary:
            np.testing.assert_array_equal(
                summary["candidate_id"],
                ("current", "member-17-exact"),
            )
        self.assertEqual(
            (pid_root / "proposed_GimbalrotorControl.yaml").read_text(
                encoding="utf-8"
            ),
            "xy:\n  p_gain: 1.1\n",
        )
        bag = loaded["bags"][0]
        self.assertEqual(sha256_file(restored / bag["relative_path"]), bag["sha256"])
        self.assertFalse(Path(bag["source_path"]).exists())

    def test_loading_same_project_creates_distinct_working_project(self):
        _source, project, _manifest = _complete_project(self.root)
        archive = save_project_archive(project, self.root / "portable.zip")
        first = load_project_archive(archive, self.root / "imports")
        second = load_project_archive(archive, self.root / "imports")
        self.assertNotEqual(first, second)
        self.assertNotEqual(read_project_manifest(second)["project_id"], "flight-test")

    def test_safe_extraction_rejects_non_relative_members(self):
        for index, member in enumerate(("../escape", "/absolute", "C:/drive", "a\\b")):
            archive_path = self.root / "bad-{}.zip".format(index)
            with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
                archive.writestr(PROJECT_MANIFEST_NAME, "{}")
                archive.writestr(member, "escape")
            with self.assertRaisesRegex(ProjectIoError, "relative path|drive"):
                load_project_archive(archive_path, self.root / "import-{}".format(index))
        self.assertFalse((self.root / "escape").exists())

    def test_safe_extraction_rejects_symlink_entries(self):
        archive_path = self.root / "symlink.zip"
        link = zipfile.ZipInfo("bags/link.bag")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
            archive.writestr(PROJECT_MANIFEST_NAME, "{}")
            archive.writestr(link, "../../outside")
        with self.assertRaisesRegex(ProjectIoError, "symlink"):
            load_project_archive(archive_path, self.root / "imports")

    def test_load_rejects_modified_bag_sha(self):
        _source, project, manifest = _complete_project(self.root)
        bag_path = project / manifest["bags"][0]["relative_path"]
        bag_path.write_bytes(b"modified after manifest")
        archive = self.root / "tampered.zip"
        with zipfile.ZipFile(archive, "w", allowZip64=True) as output:
            for path in project.rglob("*"):
                if path.is_file():
                    output.write(path, path.relative_to(project).as_posix())
        with self.assertRaisesRegex(ProjectIoError, "SHA256 mismatch"):
            load_project_archive(archive, self.root / "imports")

    def test_loader_accepts_forced_zip64_members(self):
        _source, project, _manifest = _complete_project(self.root)
        archive_path = self.root / "forced-zip64.zip"
        with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
            for path in project.rglob("*"):
                if not path.is_file():
                    continue
                info = zipfile.ZipInfo(path.relative_to(project).as_posix())
                with archive.open(info, "w", force_zip64=True) as destination:
                    destination.write(path.read_bytes())
        # ``force_zip64`` writes the ZIP64 extended-information field into
        # every local header even though these tiny fixtures do not need a
        # ZIP64 end-of-central-directory record.
        self.assertIn(b"\x01\x00\x10\x00", archive_path.read_bytes())
        restored = load_project_archive(archive_path, self.root / "imports")
        manifest = read_project_manifest(restored)
        self.assertEqual(manifest["project_id"], "flight-test")
        self.assertEqual(manifest["current_estimation_run_id"], "run-a")
        self.assertEqual(
            manifest["current_pid_proposal_evaluation_id"], "pid-a"
        )
        self.assertTrue(
            (
                restored
                / "runs"
                / "run-a"
                / "estimation_run"
                / "mcmc_samples.npz"
            ).is_file()
        )
        self.assertTrue(
            (
                restored
                / "pid_proposals"
                / "pid-a"
                / "pid_proposal_evaluation"
                / "summary.npz"
            ).is_file()
        )

    def test_freshness_changes_for_every_estimation_input(self):
        _source, _project, manifest = _complete_project(self.root)
        manifest["run_request_fingerprint"] = freshness_fingerprint(manifest)
        self.assertTrue(result_is_fresh(manifest))
        mutations = (
            lambda value: value["selected_bag_ids"].clear(),
            lambda value: value["bags"][0].update(sha256="f" * 64),
            lambda value: value["intervals"]["bag-a"].update(selected=[1.3, 4.8]),
            lambda value: value["controller_snapshots"]["bag-a"].update(source="changed"),
            lambda value: value["configuration_fingerprints"].__setitem__("bag-a", "complete:def"),
            lambda value: value["estimator_settings"]["solver_settings"].update(maximum_iterations=6),
        )
        for mutate in mutations:
            candidate = copy.deepcopy(manifest)
            mutate(candidate)
            self.assertFalse(result_is_fresh(candidate))

    def test_artifact_loader_identity_and_version_are_strict(self):
        manifest = new_project_manifest("loader-contract")
        self.assertEqual(
            set(manifest["artifact_loaders"]),
            {"inspection", "estimation_run", "pid_proposal_evaluation"},
        )
        for metadata in manifest["artifact_loaders"].values():
            self.assertEqual(metadata["id"], PROJECT_ARTIFACT_LOADER_ID)
            self.assertEqual(
                metadata["version"], PROJECT_ARTIFACT_LOADER_VERSION
            )

        bad_version = copy.deepcopy(manifest)
        bad_version["artifact_loaders"]["estimation_run"]["version"] += 1
        with self.assertRaisesRegex(ProjectIoError, "artifact loader version"):
            validate_project_manifest(bad_version)

        missing_kind = copy.deepcopy(manifest)
        del missing_kind["artifact_loaders"]["inspection"]
        with self.assertRaisesRegex(ProjectIoError, "incomplete or unsupported"):
            validate_project_manifest(missing_kind)

    def test_current_artifact_ids_cannot_escape_the_project(self):
        manifest = new_project_manifest("artifact-path-contract")
        for key in (
            "current_estimation_run_id",
            "current_pid_proposal_evaluation_id",
        ):
            candidate = copy.deepcopy(manifest)
            candidate[key] = "../../outside"
            with self.assertRaisesRegex(ProjectIoError, "safe identifier"):
                validate_project_manifest(candidate)

    def test_gui_state_v2_is_strict_and_uses_sample_identity(self):
        state = {
            "schema": GUI_STATE_SCHEMA,
            "current_bag_id": "bag-a",
            "selected_sample_id": "sample-42",
            "selected_mode_id": None,
            "selected_pid_proposal_id": None,
            "bags": {
                "bag-a": {"current_time": 2.0, "view_range": [1.0, 3.0]}
            },
        }
        self.assertEqual(
            validate_gui_state(state, registered_bag_ids=("bag-a",)), state
        )
        project = self.root / "gui-state-project"
        project.mkdir()
        write_gui_state(project, state, registered_bag_ids=("bag-a",))
        self.assertEqual(
            read_gui_state(project, registered_bag_ids=("bag-a",)), state
        )

        for mutation in (
            lambda value: value.update(schema="grape-param-estim/gui-state/v1"),
            lambda value: value.update(selected_member_id=42),
            lambda value: value["bags"]["bag-a"].update(view_range=[3.0, 1.0]),
        ):
            candidate = copy.deepcopy(state)
            mutation(candidate)
            with self.assertRaises(ProjectIoError):
                validate_gui_state(candidate, registered_bag_ids=("bag-a",))


if __name__ == "__main__":
    unittest.main()
