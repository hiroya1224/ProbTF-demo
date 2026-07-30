"""Run-directory and worker-process helpers for the Phase-2 GUI."""

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any, Mapping, Optional

from grape_param_estim.data import save_yaml


ACTIVE_STATES = ("queued", "running")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, content: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(destination)


def read_json(path: Path) -> dict:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def list_run_directories(run_root: str):
    root = Path(run_root).expanduser()
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            (
                path.resolve()
                for path in root.iterdir()
                if path.is_dir()
            ),
            reverse=True,
        )
    )


def latest_run_directory(run_root: str) -> Optional[Path]:
    directories = list_run_directories(run_root)
    return directories[0] if directories else None


def latest_completed_run_directory(run_root: str) -> Optional[Path]:
    for run_path in list_run_directories(run_root):
        status = read_json(run_path / "status.json")
        if (
            status.get("state") == "completed"
            and (run_path / "result.npz").is_file()
        ):
            return run_path
    return None


def _worker_process_matches(pid: int, run_path: Path) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
        command_line = Path("/proc/{}/cmdline".format(pid)).read_bytes()
    except (OSError, ValueError):
        return False
    arguments = tuple(
        item.decode("utf-8", errors="replace")
        for item in command_line.split(b"\0")
        if item
    )
    return (
        "grape_param_estim_worker.py"
        in " ".join(arguments)
        and str(run_path.resolve()) in arguments
    )


def active_run_directory(run_root: str) -> Optional[Path]:
    for run_path in list_run_directories(run_root):
        status = read_json(run_path / "status.json")
        if status.get("state") not in ACTIVE_STATES:
            continue
        pid = status.get("pid")
        if pid is None and status.get("state") == "queued":
            return run_path
        try:
            process_matches = _worker_process_matches(int(pid), run_path)
        except (TypeError, ValueError):
            process_matches = False
        if process_matches:
            return run_path
    return None


def start_run(
    run_root: str,
    configuration: Mapping[str, Any],
    worker_script: Optional[str] = None,
) -> Path:
    """Persist a configuration and start exactly one detached worker."""

    root = Path(run_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".worker-start.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if active_run_directory(run_root) is not None:
            raise RuntimeError("another estimation worker is already active")
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%d-%H%M%S-%f"
        )
        run_path = root / timestamp
        run_path.mkdir()
        save_yaml(str(run_path / "config.yaml"), configuration)
        created_at = _utc_now()
        atomic_json(
            run_path / "status.json",
            {
                "schema": "grape_param_estim/phase2-status",
                "state": "queued",
                "progress": 0.0,
                "stage": "queued",
                "message": "waiting for worker process",
                "created_at": created_at,
                "updated_at": created_at,
                "run_directory": str(run_path),
            },
        )
        script = (
            Path(worker_script).expanduser().resolve()
            if worker_script
            else Path(__file__).resolve().parents[2]
            / "scripts"
            / "grape_param_estim_worker.py"
        )
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(script),
                    "--run-dir",
                    str(run_path),
                ],
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
        except Exception:
            atomic_json(
                run_path / "status.json",
                {
                    "schema": "grape_param_estim/phase2-status",
                    "state": "failed",
                    "progress": 1.0,
                    "stage": "failed to start",
                    "message": "worker process could not be started",
                    "created_at": created_at,
                    "updated_at": _utc_now(),
                    "run_directory": str(run_path),
                },
            )
            raise

        status = read_json(run_path / "status.json")
        if status.get("state") == "queued":
            status["pid"] = process.pid
            status["updated_at"] = _utc_now()
            atomic_json(run_path / "status.json", status)
    return run_path


def request_stop(run_root: str) -> Path:
    run_path = active_run_directory(run_root)
    if run_path is None:
        raise RuntimeError("no active estimation worker was found")
    status = read_json(run_path / "status.json")
    try:
        pid = int(status["pid"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("active worker PID is not available yet") from error
    if not _worker_process_matches(pid, run_path):
        raise RuntimeError("refusing to signal an unrelated process")
    os.kill(pid, signal.SIGTERM)
    return run_path


__all__ = [
    "ACTIVE_STATES",
    "active_run_directory",
    "atomic_json",
    "latest_completed_run_directory",
    "latest_run_directory",
    "list_run_directories",
    "read_json",
    "request_stop",
    "start_run",
]
