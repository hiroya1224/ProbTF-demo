from __future__ import annotations

import unittest
import tempfile

import fcntl

import json

from pathlib import Path

import subprocess

import sys

from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import watch_pid_voxel_fast_catalog as catalog

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))

def finished_datasets(repo):
    fingerprints = {}
    for case, root in catalog.dataset_roots(repo).items():
        fingerprint = f"fingerprint-{case}"
        fingerprints[case] = fingerprint
        write_json(root / "manifest.json", {
            "case": case, "fingerprint": fingerprint, "source_commit": "abc123",
            "arguments": {"groups": ["xy"], "grid_sizes": [3], "samples": 2, "seed": 0},
            "plant_samples": 2, "covariance": "conservative_fusion", "criterion": "delta = 1 - max rho",
            "full_recorded_case_gains": [[1, 1, 1]] * 4})
        write_json(root / "status.json", {"case": case, "status": "complete", "updated_utc": "2026-10-08T00:00:00+00:00"})
        directory = root / "xy/grid_3"
        directory.mkdir(parents=True)
        for name in catalog.REQUIRED_VOLUME_FILES:
            (directory / name).write_bytes(b"sample-artifact")
        write_json(directory / "status.json", {"complete": True, "pending": 0, "evaluated": 27, "total": 27})
        write_json(directory / "metadata.json", {"fingerprint": fingerprint, "group": "xy", "shape": [3, 3, 3]})
        for name in ("index.html", "plant_samples.npz", "references.json", "reference_trims.npz", "run.log"):
            (root / name).write_bytes(b"root-artifact")
        (root / ".lock").touch()
        (root / ".mpl-cache").mkdir()
        (root / ".mpl-cache/cached-font").write_text("excluded")
        (root / "volume.tmp.npz").write_text("excluded")
    return fingerprints

def command(cwd, *args):
    return subprocess.check_output(("git", *args), cwd=cwd, text=True).strip()

def make_publication_repository(tmp_path):
    remote = tmp_path / "remote.git"
    repo = tmp_path / "repo"
    command(tmp_path, "init", "--bare", str(remote))
    repo.mkdir()
    command(repo, "init")
    command(repo, "symbolic-ref", "HEAD", "refs/heads/main")
    command(repo, "config", "user.name", "Catalog test")
    command(repo, "config", "user.email", "catalog-test@example.invalid")
    (repo / "existing.txt").write_text("base\n")
    command(repo, "add", "existing.txt")
    command(repo, "commit", "-m", "Initial fixture")
    command(repo, "remote", "add", "origin", str(remote))
    command(repo, "push", "-u", "origin", "main")
    return repo, remote

class CatalogChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='pid-fast-catalog-test-')
        self.tmp_path = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_running_case_prevents_publication_even_when_other_case_is_complete(self):
        tmp_path = self.tmp_path
        fingerprints = finished_datasets(tmp_path)
        root = catalog.dataset_roots(tmp_path)["failure2"]
        write_json(root / "status.json", {"case": "failure2", "status": "running"})
        with self.assertRaisesRegex(catalog.NotReady, "failure2: running"):
            catalog.catalog_payload(tmp_path, fingerprints, complete=True)

    def test_complete_flag_requires_every_volume_artifact(self):
        tmp_path = self.tmp_path
        fingerprints = finished_datasets(tmp_path)
        missing = catalog.dataset_roots(tmp_path)["failure2"] / "xy/grid_3/volume.npz"
        missing.unlink()
        with self.assertRaisesRegex(catalog.NotReady, "artifact missing"):
            catalog.catalog_payload(tmp_path, fingerprints, complete=True)

    def test_output_lock_prevents_reading_while_numerical_process_is_finishing(self):
        tmp_path = self.tmp_path
        fingerprints = finished_datasets(tmp_path)
        with (catalog.dataset_roots(tmp_path)["failure2"] / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(catalog.NotReady, "still holds output lock"):
                catalog.catalog_payload(tmp_path, fingerprints, complete=True)

    def test_replaced_calculation_cannot_be_mixed_into_catalog(self):
        tmp_path = self.tmp_path
        fingerprints = finished_datasets(tmp_path)
        fingerprints["failure2"] = "different-run"
        with self.assertRaisesRegex(ValueError, "dataset replaced"):
            catalog.catalog_payload(tmp_path, fingerprints, complete=True)

    def test_catalog_checksums_paths_and_counts_are_reproducible(self):
        tmp_path = self.tmp_path
        fingerprints = finished_datasets(tmp_path)
        payload = catalog.catalog_payload(tmp_path, fingerprints, complete=True)
        assert payload["complete"] is True
        assert payload["data_files_included_in_git_commit"] is False
        for dataset in payload["datasets"]:
            assert len(dataset["files"]) == dataset["file_count"]
            assert dataset["total_bytes"] == sum(row["bytes"] for row in dataset["files"])
            for row in dataset["files"]:
                assert row["sha256"] == catalog.file_digest(Path(row["absolute_path"]))
                assert tmp_path / row["repository_relative_path"] == Path(row["absolute_path"])
                assert ".mpl-cache" not in row["absolute_path"]
                assert ".tmp" not in row["absolute_path"]
        assert catalog.catalog_files(payload) == catalog.catalog_files(catalog.catalog_payload(tmp_path, fingerprints, complete=True))

    def test_markdown_lists_every_actual_filename_and_case_root(self):
        tmp_path = self.tmp_path
        fingerprints = finished_datasets(tmp_path)
        payload = catalog.catalog_payload(tmp_path, fingerprints, complete=True)
        markdown = catalog.catalog_files(payload)['README.md'].decode()
        assert 'Fast exact' in markdown
        for dataset in payload['datasets']:
            assert dataset['dataset_root'] in markdown
            assert f"## {dataset['case']} file names" in markdown
            for row in dataset['files']:
                assert row['dataset_relative_path'] in markdown
        assert 'stability_fraction.npz' in markdown

    def test_fast_catalog_waits_for_primary_stability_fraction_exports(self):
        tmp_path = self.tmp_path
        fingerprints = finished_datasets(tmp_path)
        missing = catalog.dataset_roots(tmp_path)['failure1']/'xy/grid_3/stability_fraction.npz'
        missing.unlink()
        with self.assertRaisesRegex(catalog.NotReady, 'artifact missing'):
            catalog.catalog_payload(tmp_path, fingerprints, complete=True)

    def test_fast_observer_targets_new_outputs_and_catalog_separately(self):
        tmp_path = self.tmp_path
        assert catalog.CATALOG_DIRECTORY.endswith('/catalogs/pid_gain_voxels_fast_interior_failure_context')
        assert catalog.parser().parse_args([]).control.endswith('/outputs/pid_voxel_fast_catalog_automation')
        for root in catalog.dataset_roots(tmp_path).values():
            assert root.name == 'pid_gain_voxels_fast_interior_failure_context'

    def test_publication_changes_only_catalog_and_preserves_worktree_and_index(self):
        tmp_path = self.tmp_path
        publication_repository = make_publication_repository(tmp_path)
        repo, remote = publication_repository
        base = command(repo, "rev-parse", "HEAD")
        (repo / "existing.txt").write_text("user's unstaged edit\n")
        (repo / "unrelated-staged.txt").write_text("user's staged edit\n")
        command(repo, "add", "unrelated-staged.txt")
        staged_before = command(repo, "diff", "--cached")
        files = {"catalog.json": b'{"complete": true}\n', "files.tsv": b"case\tpath\n", "README.md": b"Finished datasets\n"}
        result = catalog.publish_catalog(repo, files, remote="origin", branch="main", fetch_url=str(remote), push_url=str(remote))
        assert result["result"] == "pushed"
        assert command(repo, "rev-parse", "HEAD") == base
        assert command(repo, "symbolic-ref", "--short", "HEAD") == "main"
        assert command(repo, "diff", "--cached") == staged_before
        assert (repo / "existing.txt").read_text() == "user's unstaged edit\n"
        commit = command(remote, "rev-parse", "refs/heads/main")
        paths = set(command(remote, "diff", "--name-only", base, commit).splitlines())
        assert paths == {f"{catalog.CATALOG_DIRECTORY}/{name}" for name in catalog.CATALOG_NAMES}
        duplicate = catalog.publish_catalog(repo, files, remote="origin", branch="main", fetch_url=str(remote), push_url=str(remote))
        assert duplicate["result"] == "already_present"
        assert command(remote, "rev-parse", "refs/heads/main") == commit

    def test_preview_can_never_be_pushed(self):
        tmp_path = self.tmp_path
        publication_repository = make_publication_repository(tmp_path)
        repo, remote = publication_repository
        files = {"catalog.json": b'{"complete": false}\n', "files.tsv": b"", "README.md": b""}
        with self.assertRaisesRegex(ValueError, "incomplete preview"):
            catalog.publish_catalog(repo, files, remote="origin", branch="main", fetch_url=str(remote), push_url=str(remote))

    def test_observer_prepares_completed_inventory_without_publication_by_default(self):
        tmp_path = self.tmp_path
        from unittest.mock import patch
        repo = tmp_path / 'repo'
        finished_datasets(repo)
        control = tmp_path / 'control'
        args = SimpleNamespace(repo=str(repo), control=str(control), remote='origin',
                               branch='main', preview=False, poll_seconds=1)
        with patch.object(catalog, 'git', return_value='/tmp/local-only-test') as git_call, \
             patch.object(catalog, 'publish_catalog', side_effect=AssertionError('must not publish')) as publish:
            assert catalog.watch(args) == 0
            publish.assert_not_called()
            assert all(call.args[1:3] == ('remote', 'get-url') for call in git_call.call_args_list)
        state = json.loads((control/'status.json').read_text())
        assert state['status'] == 'prepared'
        assert (control/'prepared/catalog.json').is_file()
        assert 'publication' not in state
        assert not catalog.parser().parse_args([]).publish

if __name__ == '__main__':
    unittest.main()
