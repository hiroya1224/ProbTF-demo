#!/usr/bin/env python3
"""Prepare a local file catalog after both frozen PID voxel calculations finish.

No commit or push occurs by default. This observer runs independently of the numerical jobs. Explicit legacy publishing uses a
detached temporary worktree based on the current remote branch, so the main
worktree, its branch, and its staged changes are not used for the commit.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from typing import Any, Mapping

REPOSITORY = Path(__file__).resolve().parents[6]
FIRST_ORDER = "ros/examples/grape-param-estim/minimal/first-order-lag"
CASES = ("failure1", "failure2")
CATALOG_DIRECTORY = FIRST_ORDER + "/catalogs/pid_gain_voxels_failure_context"
CATALOG_NAMES = ("catalog.json", "files.tsv", "README.md")
REQUIRED_VOLUME_FILES = (
    "metadata.json", "status.json", "pid_3d.html", "pid_3d.png", "volume.vtr", "volume.npz",
    "delta.npy", "worst_spectral_radius.npy", "worst_sample_index.npy",
    "pole_valid_count.npy", "near_kink_count.npy", "spectral_radius.npy", "evaluated.npy",
    "stability_fraction.npz", "stability_fraction.json",
)


class NotReady(RuntimeError):
    pass


class PublishFailure(RuntimeError):
    def __init__(self, phase: str, message: str) -> None:
        super().__init__(message)
        self.phase = phase


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def git(repo: Path, *arguments: str, phase: str = "git") -> str:
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        result = subprocess.run(("git", *arguments), cwd=repo, env=environment,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, timeout=120, check=True)
    except subprocess.CalledProcessError as error:
        raise PublishFailure(phase, (error.stderr or error.stdout or str(error)).strip()) from error
    except subprocess.TimeoutExpired as error:
        raise PublishFailure(phase, f"git {arguments[0]} timed out") from error
    return result.stdout.strip()


def dataset_roots(repo: Path) -> Mapping[str, Path]:
    return {case: repo / FIRST_ORDER / "outputs" / case / "pid_gain_voxels_failure_context" for case in CASES}


def read_json(path: Path) -> Mapping[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise NotReady(f"not available yet: {path}") from error


def validate_completed(root: Path, case: str, fingerprint: str) -> tuple[Mapping[str, Any], Mapping[str, Any], list[Mapping[str, Any]]]:
    status = read_json(root / "status.json")
    manifest = read_json(root / "manifest.json")
    if status.get("case") != case or manifest.get("case") != case:
        raise ValueError(f"dataset case mismatch: {root}")
    if manifest.get("fingerprint") != fingerprint:
        raise ValueError(f"dataset replaced while waiting: {root}")
    if status.get("status") != "complete":
        raise NotReady(f"{case}: {status.get('status', 'unknown')}")
    volumes = []
    arguments = manifest["arguments"]
    for group in arguments["groups"]:
        for size in arguments["grid_sizes"]:
            directory = root / group / f"grid_{size}"
            details = read_json(directory / "status.json")
            metadata = read_json(directory / "metadata.json")
            if metadata.get("fingerprint") != fingerprint or metadata.get("group") != group:
                raise ValueError(f"volume belongs to a different calculation: {directory}")
            if not details.get("complete") or details.get("pending") != 0 or details.get("evaluated") != details.get("total"):
                raise NotReady(f"incomplete volume: {directory}")
            shape_count = 1
            for dimension in metadata["shape"]:
                shape_count *= int(dimension)
            if shape_count != details["total"]:
                raise ValueError(f"volume count mismatch: {directory}")
            for name in REQUIRED_VOLUME_FILES:
                path = directory / name
                if not path.is_file() or path.stat().st_size == 0:
                    raise NotReady(f"completed-volume artifact missing: {path}")
            volumes.append({"group": group, "regular_grid_size": size,
                            "shape": metadata["shape"], "counts": details,
                            "directory": str(directory)})
    for name in ("manifest.json", "status.json", "index.html", "plant_samples.npz", "references.json", "reference_trims.npz"):
        if not (root / name).is_file():
            raise NotReady(f"dataset artifact missing: {root / name}")
    return manifest, status, volumes


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def catalog_payload(repo: Path, fingerprints: Mapping[str, str], *, complete: bool) -> Mapping[str, Any]:
    datasets = []
    with ExitStack() as stack:
        if complete:
            # A numerical process retains its exclusive lock through shutdown.
            # Acquire both shared locks before reading finished files.
            for case, root in dataset_roots(repo).items():
                handle = stack.enter_context((root / ".lock").open("a"))
                try:
                    fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise NotReady(f"{case}: calculation process still holds output lock") from error
        for case, root in dataset_roots(repo).items():
            if complete:
                manifest, status, volumes = validate_completed(root, case, fingerprints[case])
            else:
                manifest = read_json(root / "manifest.json")
                status = read_json(root / "status.json")
                volumes = []
            files = []
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root)
                if not path.is_file() or any(part.startswith(".") for part in relative.parts) or ".tmp" in path.name:
                    continue
                # Symlinks outside the dataset must not be swept into a public catalog.
                if path.is_symlink():
                    raise ValueError(f"unexpected dataset symlink: {path}")
                try:
                    size = path.stat().st_size
                    digest = file_digest(path) if complete else None
                except FileNotFoundError:
                    if complete:
                        raise
                    continue
                files.append({"repository_relative_path": path.relative_to(repo).as_posix(),
                              "dataset_relative_path": relative.as_posix(), "absolute_path": str(path),
                              "bytes": size, "sha256": digest})
            datasets.append({"case": case, "status": status.get("status"),
                             "dataset_root": str(root), "repository_relative_root": root.relative_to(repo).as_posix(),
                             "fingerprint": manifest["fingerprint"], "source_commit": manifest["source_commit"],
                             "completed_utc": status.get("updated_utc") if complete else None,
                             "plant_samples": manifest["plant_samples"], "covariance": manifest["covariance"],
                             "arguments": manifest["arguments"], "criterion": manifest["criterion"],
                             "full_recorded_case_gains": manifest["full_recorded_case_gains"], "volumes": volumes,
                             "file_count": len(files), "total_bytes": sum(row["bytes"] for row in files), "files": files})
    return {"schema": "grape-param-estim/pid-voxel-file-catalog/v1", "complete": complete,
            "host": socket.gethostname(), "repository_root": str(repo),
            "data_files_included_in_git_commit": False,
            "datasets": datasets}


def catalog_files(payload: Mapping[str, Any]) -> Mapping[str, bytes]:
    json_bytes = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(("case", "repository_relative_path", "absolute_path", "bytes", "sha256"))
    for dataset in payload["datasets"]:
        for row in dataset["files"]:
            writer.writerow((dataset["case"], row["repository_relative_path"], row["absolute_path"], row["bytes"], row["sha256"] or ""))
    lines = ["# PID voxel data file catalog", "", f"Complete: {payload['complete']}. Host: `{payload['host']}`.", "",
             "This commit contains a catalog. The calculation data remain at the local paths listed below.", "",
             "`files.tsv` lists every artifact with its size and SHA-256. `catalog.json` also retains calculation conditions and completed grid counts.", "",
             "| Case | Saved dataset directory | Files | Bytes | Completed UTC |", "| --- | --- | ---: | ---: | --- |"]
    for dataset in payload["datasets"]:
        lines.append(f"| {dataset['case']} | `{dataset['dataset_root']}` | {dataset['file_count']} | {dataset['total_bytes']} | {dataset['completed_utc'] or 'pending'} |")
    lines += ["", "Open `<dataset>/index.html` for the 3D plots. Each `<group>/grid_<size>/` contains `pid_3d.html`, `pid_3d.png`, `volume.vtr`, `volume.npz`, checkpoint arrays, and metadata.", "",
              "P/I/D vary together for one group; the other nine gains equal that same failure recording. Every original plant sample remains in the denominator. Nonfinite evaluations count as unresolved and are not removed. Values are grid-node data, not joint-controller or cell-interior guarantees.", ""]
    return {"catalog.json": json_bytes, "files.tsv": stream.getvalue().encode(), "README.md": "\n".join(lines).encode()}


def write_files(directory: Path, files: Mapping[str, bytes]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        temporary = directory / (name + ".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, directory / name)


def publish_catalog(repo: Path, files: Mapping[str, bytes], *, remote: str, branch: str,
                    fetch_url: str, push_url: str) -> Mapping[str, str]:
    if set(files) != set(CATALOG_NAMES):
        raise ValueError("unexpected catalog file set")
    if json.loads(files["catalog.json"]).get("complete") is not True:
        raise ValueError("an incomplete preview cannot be published")
    if git(repo, "remote", "get-url", remote) != fetch_url or git(repo, "remote", "get-url", "--push", remote) != push_url:
        raise ValueError("remote URL changed while waiting; refusing to publish to a different destination")
    tracking = f"refs/remotes/{remote}/{branch}"
    git(repo, "fetch", "--no-tags", remote, f"refs/heads/{branch}:{tracking}", phase="fetch")
    base = git(repo, "rev-parse", tracking)
    temporary = Path(tempfile.mkdtemp(prefix="pid-voxel-catalog-", dir="/tmp"))
    checkout = temporary / "checkout"
    created = False
    try:
        git(repo, "worktree", "add", "--detach", str(checkout), base, phase="worktree")
        created = True
        paths = [f"{CATALOG_DIRECTORY}/{name}" for name in CATALOG_NAMES]
        write_files(checkout / CATALOG_DIRECTORY, files)
        git(checkout, "add", "-f", "--", *paths, phase="stage")
        changed = set(git(checkout, "diff", "--cached", "--name-only").splitlines())
        if not changed:
            return {"commit": base, "branch": branch, "remote": remote, "result": "already_present"}
        if not changed.issubset(set(paths)):
            raise ValueError("catalog staging included unrelated files")
        git(checkout, "commit", "-m", "Catalog completed failure1 and failure2 PID voxel datasets", phase="commit")
        commit = git(checkout, "rev-parse", "HEAD")
        actual = set(git(checkout, "diff", "--name-only", base, commit).splitlines())
        if not actual.issubset(set(paths)):
            raise ValueError("catalog commit contains unrelated files")
        # Normal fast-forward push only. If the remote advances, the watcher
        # fetches again and builds the catalog commit on the new remote tip.
        git(checkout, "push", remote, f"HEAD:refs/heads/{branch}", phase="push")
        return {"commit": commit, "branch": branch, "remote": remote, "result": "pushed"}
    finally:
        if created:
            try:
                git(repo, "worktree", "remove", str(checkout), phase="cleanup")
            except PublishFailure:
                # Preserve any uncommitted hook failure for inspection.
                print(f"Temporary publication worktree retained: {checkout}", flush=True)
        try:
            temporary.rmdir()
        except OSError:
            pass


def watch(arguments: argparse.Namespace) -> int:
    repo = Path(arguments.repo).expanduser().resolve()
    control = Path(arguments.control).expanduser().resolve()
    control.mkdir(parents=True, exist_ok=True)
    fingerprints = {case: read_json(root / "manifest.json")["fingerprint"] for case, root in dataset_roots(repo).items()}
    settings = {"repo": str(repo), "remote": arguments.remote, "branch": arguments.branch,
                "fetch_url": git(repo, "remote", "get-url", arguments.remote),
                "push_url": git(repo, "remote", "get-url", "--push", arguments.remote),
                "fingerprints": fingerprints, "catalog_directory": CATALOG_DIRECTORY}
    settings_path = control / "settings.json"
    if settings_path.exists() and read_json(settings_path) != settings:
        raise ValueError("observer settings or calculation fingerprints changed; choose a new --control directory")
    atomic_json(settings_path, settings)
    if arguments.preview:
        write_files(control / "preview", catalog_files(catalog_payload(repo, fingerprints, complete=False)))
        print(f"Unpublished preview: {control / 'preview/catalog.json'}", flush=True)
        return 0
    status_path = control / "status.json"
    if status_path.exists() and read_json(status_path).get("status") in ("prepared", "published"):
        print("This observer has already prepared the completed catalog.", flush=True)
        return 0
    state = {"status": "waiting", "pid": os.getpid(), "started_utc": utc_now(),
             "updated_utc": utc_now(), "catalog_directory": CATALOG_DIRECTORY,
             "remote": arguments.remote, "branch": arguments.branch, "fingerprints": fingerprints}
    previous_message = None
    try:
        while True:
            statuses = {case: read_json(root / "status.json").get("status") for case, root in dataset_roots(repo).items()}
            state.update(updated_utc=utc_now(), calculations=statuses)
            try:
                if any(status != "complete" for status in statuses.values()):
                    raise NotReady(str(statuses))
                payload = catalog_payload(repo, fingerprints, complete=True)
                files = catalog_files(payload)
                write_files(control / "prepared", files)
                if not getattr(arguments, "publish", False):
                    state.update(status="prepared", updated_utc=utc_now(),
                                 file_counts={d["case"]: d["file_count"] for d in payload["datasets"]})
                    atomic_json(status_path, state)
                    print(f"Completed catalog saved locally: {control / 'prepared'}; no commit or push performed.", flush=True)
                    return 0
                state.update(status="publishing", file_counts={d["case"]: d["file_count"] for d in payload["datasets"]})
                atomic_json(status_path, state)
                print("Both calculations complete. Publishing catalog files only.", flush=True)
                publication = publish_catalog(repo, files, remote=arguments.remote, branch=arguments.branch,
                                              fetch_url=settings["fetch_url"], push_url=settings["push_url"])
                for key in ("error", "git_phase", "wait_reason"):
                    state.pop(key, None)
                state.update(status="published", updated_utc=utc_now(), publication=publication)
                atomic_json(status_path, state)
                print(f"Catalog published: {publication['commit']} -> {arguments.remote}/{arguments.branch}", flush=True)
                return 0
            except NotReady as error:
                state.update(status="waiting", wait_reason=str(error))
                message = f"Waiting for completed calculations: {error}"
            except PublishFailure as error:
                if error.phase not in ("fetch", "push"):
                    raise
                state.update(status="retrying_publish", error=str(error), git_phase=error.phase)
                message = f"Publication will retry ({error.phase}): {error}"
            atomic_json(status_path, state)
            if message != previous_message:
                print(f"{utc_now()} {message}", flush=True)
                previous_message = message
            time.sleep(arguments.poll_seconds)
    except BaseException as error:
        state.update(status="failed", updated_utc=utc_now(), error=f"{type(error).__name__}: {error}")
        atomic_json(status_path, state)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", default=str(REPOSITORY))
    result.add_argument("--control", default=str(REPOSITORY / FIRST_ORDER / "outputs/pid_voxel_catalog_automation"))
    result.add_argument("--remote", default="origin")
    result.add_argument("--branch", default="main")
    result.add_argument("--poll-seconds", type=float, default=60.0)
    result.add_argument("--preview", action="store_true", help="write an unpublished partial inventory and exit")
    result.add_argument("--publish", action="store_true", help="explicit legacy publication; default only prepares local files")
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    if not 1 <= args.poll_seconds <= 60:
        raise SystemExit("--poll-seconds must be between 1 and 60")
    control_path = Path(args.control).expanduser().resolve()
    control_path.mkdir(parents=True, exist_ok=True)
    with (control_path / ".lock").open("a") as observer_lock:
        try:
            fcntl.flock(observer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A catalog observer is already running with this control directory.")
        raise SystemExit(watch(args))
