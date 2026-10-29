"""Final-artifact selection and ordinary-current-branch publication checks."""
from __future__ import annotations
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import watch_failure_publication as watcher


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def fixture_data(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "render.py").write_text("# fixture renderer")
    (repo / "settings.json").write_text("{}")
    source = tmp_path / "source"
    config = {"repository": str(repo), "input_root": str(source),
              "output_relative": "publication/failure_maps", "grid_sizes": [33, 65],
              "groups": ["xy"], "fingerprints": {}, "renderer": {
                  "script": "render.py", "settings": "settings.json"},
              "publish": {"enabled": False, "max_file_bytes": 1000000, "max_total_bytes": 1000000}}
    for case in watcher.CASES:
        root = source / case
        fingerprint = "fixed-" + case
        config["fingerprints"][case] = fingerprint
        write(root / "manifest.json", {"fingerprint": fingerprint, "case": case,
                                      "arguments": {"success_json": None}})
        write(root / "status.json", {"case": case, "status": "complete"})
        write(root / "references.json", {"recorded_case": {"gains": [[1, 1, 1]]*4}})
        write(root / "retention.json", {"case": case})
        np.savez_compressed(root / "plant_samples.npz", quotient_coordinates=np.zeros((3, 13)))
        write(root / "inputs/vehicle_model_source.json", {"frozen_model": True})
        for name in ("original_estimate.json", "runtime_estimate.json"):
            write(root / "inputs" / name, {"case": case, "input": {
                "vehicle_model": str(root / "inputs/vehicle_model_source.json")}})
        (root / ".lock").touch()
        for size in (33, 65):
            volume = root / "xy" / f"grid_{size}"
            write(volume / "metadata.json", {"fingerprint": fingerprint, "group": "xy",
                "shape": [2, 2, 2], "axes": [[1, 2]]*3, "plant_samples": 3})
            write(volume / "status.json", {"complete": True, "pending": 0,
                                          "evaluated": 8, "total": 8})
            write(volume / "stability_fraction.json", {"sample_count": 3})
            np.savez_compressed(volume / "stability_fraction.npz", p=[1, 2], i=[1, 2], d=[1, 2],
                                stable_fraction=np.full((2, 2, 2), 1/3))
            radii = np.tile([.9, 1.1, np.nan], (2, 2, 2, 1))
            np.save(volume / "spectral_radius.npy", radii)
            np.save(volume / "evaluated.npy", np.ones((2, 2, 2), dtype=bool))
            np.savez_compressed(volume / "volume.npz", spectral_radius=radii)
    return repo, source, config


def fake_render(config, output):
    paths = []
    for size in config["grid_sizes"]:
        directory = output / "figures" / f"grid_{size}"
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("figure_index.json", "figure_index.csv"):
            p = directory / name;p.write_text("final figure index");paths.append(p)
    return paths


def test_running_case_defers_without_rendering_or_git(tmp_path):
    repo, source, config = fixture_data(tmp_path)
    write(source / "failure2/status.json", {"case": "failure2", "status": "running"})
    with patch.object(watcher, "render_figures") as render:
        state = watcher.run_once(config, tmp_path / "control")
    assert state["status"] == "waiting_for_calculations"
    assert set(state["fingerprints"]) == set(watcher.CASES)
    render.assert_not_called()


def test_lock_prevents_publication_before_writer_has_finished(tmp_path):
    repo, source, config = fixture_data(tmp_path)
    with (source / "failure1/.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = watcher.run_once(config, tmp_path / "control")
    assert state["status"] == "waiting_for_calculations"
    assert "still finishing" in state["waiting_reason"]


def test_replaced_input_fingerprint_is_rejected(tmp_path):
    repo, source, config = fixture_data(tmp_path)
    config["fingerprints"]["failure2"] = "replacement"
    with pytest.raises(ValueError, match="calculation changed"):
        watcher.run_once(config, tmp_path / "control")


def test_final_bundle_has_lossless_per_sample_bits_but_no_large_floats(tmp_path):
    repo, source, config = fixture_data(tmp_path)
    with patch.object(watcher, "render_figures", side_effect=fake_render):
        state = watcher.run_once(config, tmp_path / "control")
    assert state["status"] == "prepared_only"
    output = repo / config["output_relative"]
    files = {row["path"] for row in state["prepared"]["files"]}
    assert not any(name.endswith(".npy") or name.endswith("volume.npz") for name in files)
    for case in watcher.CASES:
        for size in (33, 65):
            with np.load(output / "data" / case / "xy" / f"grid_{size}/sample_bits.npz") as archive:
                n = int(archive["sample_count"])
                stable = np.unpackbits(archive["stable_bits"], axis=-1, count=n, bitorder="little")
                valid = np.unpackbits(archive["valid_bits"], axis=-1, count=n, bitorder="little")
                assert np.array_equal(stable, np.tile([1, 0, 0], (2, 2, 2, 1)))
                assert np.array_equal(valid, np.tile([1, 1, 0], (2, 2, 2, 1)))
                assert np.all(archive["stable_count"] == 1)
                assert np.all(archive["unresolved_count"] == 1)
                assert not any("baseline" in name for name in archive.files)
    stored = json.loads((output / "publication.json").read_text())
    assert stored["within_budget"] is True
    assert stored["transfer_total_bytes"] == stored["total_bytes"] + (output / "publication.json").stat().st_size
    assert "publication_manifest_sha256" not in stored
    assert (output / "data/failure1/inputs/vehicle_model.json").read_bytes() == (source / "failure1/inputs/vehicle_model_source.json").read_bytes()
    catalog = json.loads((output / "local_data_catalog.json").read_text())
    for row in catalog["cases"]["failure1"]["large_local_files"]:
        assert row["sha256"] == watcher.digest(Path(row["path"]))


def test_oversize_result_stays_prepared_and_never_calls_git(tmp_path):
    repo, source, config = fixture_data(tmp_path)
    config["publish"].update(enabled=True, max_total_bytes=10)
    with patch.object(watcher, "render_figures", side_effect=fake_render), patch.object(watcher, "git") as git:
        state = watcher.run_once(config, tmp_path / "control")
    assert state["status"] == "prepared_over_budget"
    assert "total budget" in state["waiting_reason"]
    assert (repo / config["output_relative"] / "publication.json").exists()
    git.assert_not_called()


def test_unowned_output_directory_is_not_overwritten(tmp_path):
    repo, source, config = fixture_data(tmp_path)
    p = repo / config["output_relative"] / "README.md"
    p.parent.mkdir(parents=True);p.write_text("user draft")
    with pytest.raises(RuntimeError, match="not owned"):
        watcher.run_once(config, tmp_path / "control")
    assert p.read_text() == "user draft"


def command(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()


def publication_fixture(tmp_path):
    repo = tmp_path / "repo";repo.mkdir()
    remote = tmp_path / "remote.git"
    command(tmp_path, "init", "--bare", str(remote))
    command(repo, "init")
    command(repo, "symbolic-ref", "HEAD", "refs/heads/main")
    command(repo, "config", "user.name", "Watcher test")
    command(repo, "config", "user.email", "test@example.invalid")
    (repo / "outside.txt").write_text("base")
    command(repo, "add", "outside.txt")
    command(repo, "commit", "-m", "fixture base")
    command(repo, "remote", "add", "origin", str(remote))
    command(repo, "push", "-u", "origin", "main")
    output = repo / "publication/maps";output.mkdir(parents=True)
    (output / "README.md").write_text("final figures")
    manifest = output / "publication.json";manifest.write_text("{}")
    prepared = {"within_budget": True, "files": [{"path": "README.md", "size_bytes": 13,
                "sha256": watcher.digest(output / "README.md")}],
                "publication_manifest": "publication.json", "publication_manifest_sha256": watcher.digest(manifest)}
    config = {"repository": str(repo), "input_root": str(tmp_path / "unused"),
              "output_relative": "publication/maps", "publish": {"enabled": True,
                  "remote": "origin", "branch": "main", "fetch_url": str(remote), "push_url": str(remote)}}
    return repo, remote, config, prepared


def test_ordinary_commit_push_moves_current_main_only_for_selected_files(tmp_path):
    repo, remote, config, prepared = publication_fixture(tmp_path)
    (repo / "outside.txt").write_text("user unstaged edit")
    before = command(repo, "rev-parse", "HEAD")
    state = {}
    result = watcher.publish(config, prepared, state, lambda: None)
    assert result["result"] == "pushed"
    after = command(repo, "rev-parse", "HEAD")
    assert before != after == command(remote, "rev-parse", "refs/heads/main")
    assert command(repo, "symbolic-ref", "--short", "HEAD") == "main"
    assert (repo / "outside.txt").read_text() == "user unstaged edit"
    assert set(command(repo, "diff", "--name-only", before, after).splitlines()) == {
        "publication/maps/README.md", "publication/maps/publication.json"}
    assert not (repo / ".git/worktrees").exists()


def test_unrelated_staged_work_defers_without_committing_it(tmp_path):
    repo, remote, config, prepared = publication_fixture(tmp_path)
    (repo / "private.txt").write_text("user draft")
    command(repo, "add", "private.txt")
    before = command(repo, "rev-parse", "HEAD")
    with pytest.raises(watcher.GitDeferred, match="unrelated staged"):
        watcher.publish(config, prepared, {}, lambda: None)
    assert command(repo, "rev-parse", "HEAD") == before
    assert command(repo, "diff", "--cached", "--name-only") == "private.txt"


def test_push_retry_does_not_create_another_commit(tmp_path):
    repo, remote, config, prepared = publication_fixture(tmp_path)
    original = watcher.git
    state = {}
    def fail_push(repo, *args):
        if args[0] == "push":
            raise watcher.GitDeferred("temporary network failure")
        return original(repo, *args)
    with patch.object(watcher, "git", side_effect=fail_push):
        with pytest.raises(watcher.GitDeferred, match="network failure"):
            watcher.publish(config, prepared, state, lambda: None)
    committed = command(repo, "rev-parse", "HEAD")
    assert state["publication_commit"] == committed
    watcher.publish(config, prepared, state, lambda: None)
    assert command(repo, "rev-parse", "HEAD") == committed
    assert command(remote, "rev-parse", "refs/heads/main") == committed


def test_staging_race_cannot_add_an_unrelated_file_to_the_commit(tmp_path):
    repo, remote, config, prepared = publication_fixture(tmp_path)
    original = watcher.git
    def stage_user_work_before_commit(repo, *args):
        if args[0] == "commit":
            (repo / "concurrent.txt").write_text("user concurrent draft")
            command(repo, "add", "concurrent.txt")
        return original(repo, *args)
    with patch.object(watcher, "git", side_effect=stage_user_work_before_commit):
        watcher.publish(config, prepared, {}, lambda: None)
    assert command(repo, "diff", "--cached", "--name-only") == "concurrent.txt"
    assert "concurrent.txt" not in command(repo, "show", "--format=", "--name-only", "HEAD").splitlines()
