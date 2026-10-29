#!/usr/bin/env python3
"""Wait for the two completed failure-only runs, render, then publish selected files.

Only explicitly listed small final artifacts are staged. Numerical inputs,
large spectral-radius arrays, checkpoints, and unrelated work are never staged.
Publishing uses ordinary git commit/push in the existing main working tree.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from typing import Any, Mapping

CASES = ("failure1", "failure2")
GROUPS = ("roll_pitch", "yaw", "xy", "z")
SMALL_ROOT_FILES = ("manifest.json", "status.json", "plant_samples.npz", "references.json", "retention.json")
SMALL_VOLUME_FILES = ("metadata.json", "status.json", "stability_fraction.npz", "stability_fraction.json")
LOCAL_VOLUME_FILES = ("volume.npz", "spectral_radius.npy", "evaluated.npy")


class NotReady(RuntimeError):
    pass


class GitDeferred(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Mapping[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise NotReady(f"not available yet: {path}") from error


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path must be inside {root}: {relative}") from error
    if path == root.resolve():
        raise ValueError(f"path must identify a child of {root}: {relative}")
    return path


def git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=repo,
                            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=180)
    if result.returncode:
        raise GitDeferred((result.stderr or result.stdout).strip())
    return result.stdout.strip()


def configured_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    repo = Path(config["repository"]).expanduser().resolve()
    source = Path(config["input_root"]).expanduser().resolve()
    output = within(repo, config["output_relative"])
    return repo, source, output


def require_file(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"unexpected artifact symlink: {path}")
    if not path.is_file() or path.stat().st_size == 0:
        raise NotReady(f"artifact not available yet: {path}")


def inspect_completed(config: Mapping[str, Any], state: dict[str, Any], stack: ExitStack) -> Mapping[str, Any]:
    _repo, source, _output = configured_paths(config)
    pinned = state.setdefault("fingerprints", {})
    # Pin both identities even when the first case is still running.
    manifests = {}
    for case in CASES:
        manifest = read_json(source / case / "manifest.json")
        fingerprint = manifest["fingerprint"]
        required = config.get("fingerprints", {}).get(case, pinned.get(case))
        if required is not None and required != fingerprint:
            raise ValueError(f"input calculation changed while waiting: {case}")
        pinned[case] = fingerprint
        manifests[case] = manifest
    results = {}
    for case in CASES:
        root = source / case
        manifest = manifests[case]
        fingerprint = manifest["fingerprint"]
        status = read_json(root / "status.json")
        if status.get("status") == "failed":
            raise RuntimeError(f"{case} calculation failed: {status.get('error', 'see run.log')}")
        if status.get("status") != "complete":
            raise NotReady(f"{case}: {status.get('status', 'unknown')}")
        handle = stack.enter_context((root / ".lock").open("a"))
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise NotReady(f"{case}: calculation is still finishing") from error
        if manifest.get("case") != case or status.get("case") != case:
            raise ValueError(f"case identity differs: {root}")
        if manifest["arguments"].get("success_json") is not None:
            raise ValueError(f"unexpected successful-flight input in {case}")
        volumes = []
        for group in config.get("groups", GROUPS):
            for size in config.get("grid_sizes", [33, 65]):
                directory = root / group / f"grid_{size}"
                details = read_json(directory / "status.json")
                metadata = read_json(directory / "metadata.json")
                if metadata.get("fingerprint") != fingerprint or metadata.get("group") != group:
                    raise ValueError(f"volume identity differs: {directory}")
                if not details.get("complete") or details.get("pending") != 0 or details.get("evaluated") != details.get("total"):
                    raise NotReady(f"volume incomplete: {directory}")
                for name in (*SMALL_VOLUME_FILES, *LOCAL_VOLUME_FILES):
                    require_file(directory / name)
                volumes.append({"group": group, "grid": int(size), "shape": metadata["shape"],
                                "directory": directory, "status": details})
        for name in SMALL_ROOT_FILES:
            require_file(root / name)
        results[case] = {"root": root, "manifest": manifest, "status": status, "volumes": volumes}
    return results


def pack_sample_archive(directory: Path, destination: Path) -> None:
    """Losslessly preserve every sample's validity/stability without GB floats."""
    radii = np.load(directory / "spectral_radius.npy", mmap_mode="r")
    done = np.load(directory / "evaluated.npy", mmap_mode="r")
    metadata = read_json(directory / "metadata.json")
    n = int(metadata["plant_samples"])
    if radii.shape != tuple(metadata["shape"]) + (n,) or done.shape != tuple(metadata["shape"]):
        raise ValueError(f"per-sample source dimensions differ: {directory}")
    valid = np.isfinite(radii) & done[..., None]
    stable = valid & (radii < 1.0)
    stable_bits = np.packbits(stable, axis=-1, bitorder="little")
    valid_bits = np.packbits(valid, axis=-1, bitorder="little")
    stable_count = np.count_nonzero(stable, axis=-1).astype(np.int32)
    valid_count = np.count_nonzero(valid, axis=-1).astype(np.int32)
    temporary = destination.with_name(destination.stem + ".tmp.npz")
    np.savez_compressed(temporary,
        p=np.asarray(metadata["axes"][0]), i=np.asarray(metadata["axes"][1]),
        d=np.asarray(metadata["axes"][2]), sample_count=np.asarray(n),
        stable_bits=stable_bits, valid_bits=valid_bits, evaluated=done,
        stable_count=stable_count, valid_count=valid_count,
        unresolved_count=np.where(done, n - valid_count, 0),
        stable_fraction=np.where(done, stable_count / n, np.nan),
        bitorder=np.asarray("little"), axis_order=np.asarray(["P", "I", "D", "plant_sample"]))
    os.replace(temporary, destination)


def copy_final_data(output: Path, datasets: Mapping[str, Any]) -> tuple[list[Path], Mapping[str, Any]]:
    managed = []
    catalog = {"large_arrays_in_git": False, "cross_failure_fusion_performed": False, "cases": {}}
    for case, dataset in datasets.items():
        root = dataset["root"]
        local_files = []
        for name in SMALL_ROOT_FILES:
            target = output / "data" / case / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
            managed.append(target)
        for name in ("original_estimate.json", "runtime_estimate.json"):
            original = root / "inputs" / name
            require_file(original)
            target = output / "data" / case / "inputs" / name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(original, target)
            managed.append(target)
        runtime_input = read_json(root / "inputs/runtime_estimate.json")
        model_source = Path(runtime_input["input"]["vehicle_model"])
        require_file(model_source)
        model_copy = output / "data" / case / "inputs" / "vehicle_model.json"
        shutil.copyfile(model_source, model_copy)
        managed.append(model_copy)
        for volume in dataset["volumes"]:
            directory = volume["directory"]
            for name in SMALL_VOLUME_FILES:
                target = output / "data" / case / volume["group"] / f"grid_{volume['grid']}" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(directory / name, target)
                managed.append(target)
            archive = output / "data" / case / volume["group"] / f"grid_{volume['grid']}" / "sample_bits.npz"
            pack_sample_archive(directory, archive)
            managed.append(archive)
            for name in LOCAL_VOLUME_FILES:
                path = directory / name
                local_files.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest(path)})
        catalog["cases"][case] = {"source_root": str(root), "fingerprint": dataset["manifest"]["fingerprint"],
                                  "completed_utc": dataset["status"].get("updated_utc"),
                                  "runtime_model_path": str(model_source),
                                  "model_copy": model_copy.relative_to(output).as_posix(),
                                  "model_sha256": digest(model_copy),
                                  "large_local_files": local_files}
    return managed, catalog


def render_figures(config: Mapping[str, Any], output: Path) -> list[Path]:
    repo, source, _output = configured_paths(config)
    renderer = config["renderer"]
    script = within(repo, renderer["script"])
    settings = within(repo, renderer["settings"])
    managed = []
    for size in config.get("grid_sizes", [33, 65]):
        directory = output / "figures" / f"grid_{size}"
        directory.mkdir(parents=True, exist_ok=True)
        command = [renderer.get("python", "/usr/bin/python3"), str(script),
                   "--input-root", str(source), "--output", str(directory),
                   "--settings", str(settings), "--grid", str(size), "--view", "both"]
        subprocess.run(command, cwd=repo, check=True)
        for view in ("3D", "PD"):
            for suffix in ("pdf", "svg", "png"):
                path = directory / f"all_pid_failure_{view}_grid_{size}.{suffix}"
                require_file(path)
                managed.append(path)
        for name in ("figure_index.json", "figure_index.csv"):
            require_file(directory / name)
            managed.append(directory / name)
    return managed


def inventory(config: Mapping[str, Any], paths: list[Path], output: Path) -> Mapping[str, Any]:
    publish = config.get("publish", {})
    max_file = int(publish.get("max_file_bytes", 49_000_000))
    max_total = int(publish.get("max_total_bytes", 49_000_000))
    rows = []
    over = []
    for path in sorted(set(paths)):
        require_file(path)
        size = path.stat().st_size
        if size > max_file:
            over.append(f"selected final artifact exceeds {max_file} bytes: {path} ({size})")
        rows.append({"path": path.relative_to(output).as_posix(), "size_bytes": size, "sha256": digest(path)})
    total = sum(row["size_bytes"] for row in rows)
    if total > max_total:
        over.append(f"selected final artifacts exceed total budget {max_total}: {total}")
    return {"files": rows, "total_bytes": total, "max_file_bytes": max_file, "max_total_bytes": max_total,
            "within_budget": not over, "over_budget_reason": "; ".join(over)}


def prepare(config: Mapping[str, Any], state: dict[str, Any]) -> Mapping[str, Any]:
    repo, _source, output = configured_paths(config)
    owner = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    marker = output / ".publication_watcher_owner"
    if output.exists() and any(output.iterdir()) and not marker.exists():
        raise RuntimeError(f"publication output is not owned by this watcher: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if marker.exists() and marker.read_text().strip() != owner:
        raise RuntimeError(f"publication output belongs to another watcher configuration: {output}")
    marker.write_text(owner + "\n")
    with ExitStack() as locks:
        datasets = inspect_completed(config, state, locks)
        managed, catalog = copy_final_data(output, datasets)
        managed.extend(render_figures(config, output))
    catalog_path = output / "local_data_catalog.json"
    atomic_json(catalog_path, catalog)
    managed.append(catalog_path)
    ignore = output / ".gitignore"
    ignore.write_text("/.publication_watcher_owner\n")
    managed.append(ignore)
    settings_source = within(repo, config["renderer"]["settings"])
    settings_copy = output / "figure_settings.json"
    shutil.copyfile(settings_source, settings_copy)
    managed.append(settings_copy)
    renderer_source = within(repo, config["renderer"]["script"])
    readme = output / "README.md"
    readme.write_text("# Failure-only PID maps: 33 and 65 grids\n\n"
                      "Both failure recordings are evaluated independently, with all original plant samples in each denominator. "
                      "Other nine gains remain at that same recording's values.\n\n"
                      "The figures follow the earlier publication layout and styles; the withdrawn success-conditioned statistic is not used. "
                      "No cross-failure fusion has been performed.\n\n"
                      "`data/` contains final coordinate axes and stable/valid/unresolved counts, sample coordinates, fixed controller context and source metadata. "
                      "`sample_bits.npz` losslessly preserves every plant's stable/valid bit in original sample order (little-endian bit packing), plus evaluated masks. "
                      "Large per-plant spectral radii and checkpoints remain locally at paths and SHA-256 hashes in `local_data_catalog.json`.\n\n"
                      "Original and runtime estimate files retain their exact numerical values. The runtime input still points to its local frozen model; "
                      "a byte-identical `inputs/vehicle_model.json` is included for portability.\n\n"
                      "See `figures/grid_33/` and `figures/grid_65/` for the PDF, SVG and PNG panel figures.\n")
    managed.append(readme)
    payload = {"schema": "grape-param-estim/failure-publication/v1", "prepared_utc": now(),
               "input_fingerprints": dict(state["fingerprints"]), "cross_failure_fusion_performed": False,
               "renderer": {"script": config["renderer"]["script"], "sha256": digest(renderer_source),
                            "settings": "figure_settings.json", "settings_sha256": digest(settings_copy)},
               **inventory(config, managed, output)}
    # No self-hash is embedded in this file. Its transfer size reaches an
    # exact fixed point once the decimal length of the size/budget fields settles.
    base_total = payload["total_bytes"]
    file_errors = payload["over_budget_reason"]
    payload["transfer_total_bytes"] = 0
    while True:
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        total = base_total + len(encoded)
        errors = [file_errors] if file_errors else []
        if len(encoded) > payload["max_file_bytes"]:
            errors.append(f"publication manifest exceeds file budget: {len(encoded)}")
        if total > payload["max_total_bytes"]:
            errors.append(f"selected artifacts and manifest exceed total budget {payload['max_total_bytes']}: {total}")
        updated = {"transfer_total_bytes": total, "within_budget": not errors,
                   "over_budget_reason": "; ".join(errors)}
        if all(payload.get(key) == value for key, value in updated.items()):
            break
        payload.update(updated)
    atomic_json(output / "publication.json", payload)
    payload["publication_manifest"] = "publication.json"
    payload["publication_manifest_sha256"] = digest(output / "publication.json")
    return payload


def publish(config: Mapping[str, Any], prepared: Mapping[str, Any], state: dict[str, Any], save) -> Mapping[str, Any]:
    repo, _source, output = configured_paths(config)
    options = config.get("publish", {})
    if not prepared.get("within_budget", False):
        raise GitDeferred(prepared.get("over_budget_reason", "publication exceeds configured byte budget"))
    if not options.get("enabled", False):
        return {"result": "prepared_only"}
    if digest(output / prepared["publication_manifest"]) != prepared["publication_manifest_sha256"]:
        raise GitDeferred("publication manifest changed after preparation")
    branch = options.get("branch", "main")
    remote = options.get("remote", "origin")
    if git(repo, "symbolic-ref", "--short", "HEAD") != branch:
        raise GitDeferred(f"current branch is not {branch}; no branch switch performed")
    if git(repo, "remote", "get-url", remote) != options["fetch_url"] or git(repo, "remote", "get-url", "--push", remote) != options["push_url"]:
        raise GitDeferred("remote destination differs from the pinned configuration")
    paths = [str((output / row["path"]).relative_to(repo)) for row in prepared["files"]]
    paths.append(str((output / prepared["publication_manifest"]).relative_to(repo)))
    for row in prepared["files"]:
        path = within(output, row["path"])
        if path.stat().st_size != row["size_bytes"] or digest(path) != row["sha256"]:
            raise GitDeferred(f"prepared artifact changed after rendering: {path}")
    staged = set(filter(None, git(repo, "diff", "--cached", "--name-only").splitlines()))
    if not staged.issubset(set(paths)):
        raise GitDeferred("unrelated staged work exists; it was not changed")
    git(repo, "fetch", "--no-tags", remote)
    # Refuse divergence; do not create a worktree, switch branches, or reset user work.
    counts = git(repo, "rev-list", "--left-right", "--count", f"HEAD...{remote}/{branch}").split()
    if int(counts[1]):
        if int(counts[0]):
            raise GitDeferred("local and remote branches diverged; ordinary push will wait")
        git(repo, "merge", "--ff-only", f"{remote}/{branch}")
    git(repo, "--literal-pathspecs", "add", "--", *paths)
    staged = set(filter(None, git(repo, "diff", "--cached", "--name-only").splitlines()))
    if not staged.issubset(set(paths)):
        raise GitDeferred("unrelated staging appeared concurrently; no commit was made")
    if staged:
        git(repo, "commit", "--only", "-m", "Publish audited failure-only PID maps at 33 and 65 grids", "--", *paths)
        state["publication_commit"] = git(repo, "rev-parse", "HEAD")
        state["status"] = "committed_pending_push"
        save()
    else:
        state.setdefault("publication_commit", git(repo, "rev-parse", "HEAD"))
    git(repo, "push", remote, branch)
    return {"result": "pushed", "commit": state["publication_commit"], "remote": remote, "branch": branch}


def run_once(config: Mapping[str, Any], control: Path) -> dict[str, Any]:
    state_path = control / "status.json"
    state = dict(read_json(state_path)) if state_path.exists() else {"started_utc": now()}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if state.get("configuration_sha256", signature) != signature:
        raise ValueError("watcher configuration changed; use a separate control directory")
    state["configuration_sha256"] = signature

    def save():
        state["updated_utc"] = now()
        atomic_json(state_path, state)
    try:
        if state.get("status") in ("pushed", "prepared_only"):
            return state
        if "prepared" not in state:
            state["status"] = "checking_calculations"
            save()
            state["prepared"] = prepare(config, state)
            state["status"] = "prepared"
            save()
        if not state["prepared"]["within_budget"]:
            state["status"] = "prepared_over_budget"
            state["waiting_reason"] = state["prepared"]["over_budget_reason"]
            save()
            return state
        state["publication"] = publish(config, state["prepared"], state, save)
        state["status"] = state["publication"]["result"]
        state.pop("waiting_reason", None)
        save()
    except (NotReady, GitDeferred) as error:
        state["status"] = "waiting_for_calculations" if isinstance(error, NotReady) else "waiting_for_git"
        state["waiting_reason"] = str(error)
        save()
    except Exception as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        save()
        raise
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.control.mkdir(parents=True, exist_ok=True)
    with (args.control / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another publication watcher already holds this control directory")
        while True:
            state = run_once(config, args.control)
            print(json.dumps({key: state.get(key) for key in ("status", "waiting_reason", "publication")}, ensure_ascii=False), flush=True)
            if args.once or state["status"] in ("pushed", "prepared_only", "prepared_over_budget"):
                return 0
            time.sleep(float(config.get("poll_seconds", 60)))


if __name__ == "__main__":
    raise SystemExit(main())
