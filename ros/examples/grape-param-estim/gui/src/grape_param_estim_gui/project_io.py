"""Portable, authenticated project directories and standard ZIP/ZIP64 I/O."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any, BinaryIO, Iterable, Mapping
import uuid
import zipfile


PROJECT_SCHEMA = "grape-param-estim/project/v1"
PROJECT_WRITER_ID = "grape-param-estim-gui-project-writer"
PROJECT_WRITER_VERSION = 1
PROJECT_LOADER_ID = "grape-param-estim-gui-project-loader"
PROJECT_LOADER_VERSION = 1
PROJECT_ARTIFACT_LOADER_ID = "grape-param-estim-gui-artifact-loader"
PROJECT_ARTIFACT_LOADER_VERSION = 1
PROJECT_MANIFEST_NAME = "project.json"
GUI_STATE_NAME = "gui_state.json"

_PROJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FRESHNESS_FINGERPRINT = re.compile(r"^sha256:[0-9a-f]{64}$")
_INTERVAL_STATES = {"AUTO", "MODIFIED", "LOCKED"}
_ARTIFACT_KINDS = {
    "inspection",
    "assimilation_run",
    "pid_proposal_evaluation",
}


class ProjectIoError(ValueError):
    """A project or archive violates the portable project contract."""


@dataclass(frozen=True)
class ArchiveLimits:
    max_entries: int = 100_000
    max_total_uncompressed_bytes: int = 2 * 1024**4
    max_single_file_bytes: int = 1024**4
    max_compression_ratio: float = 10_000.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024**2) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProjectIoError("project data is not finite JSON") from error


def freshness_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return only inputs that determine whether a loaded run is current."""

    bag_by_id = {str(item["bag_id"]): item for item in manifest.get("bags", [])}
    selected = [str(value) for value in manifest.get("selected_bag_ids", [])]
    intervals = manifest.get("intervals", {})
    fingerprints = manifest.get("configuration_fingerprints", {})
    snapshots = manifest.get("controller_snapshots", {})
    return {
        "selected_bag_ids": selected,
        "bags": [
            {"bag_id": bag_id, "sha256": str(bag_by_id[bag_id]["sha256"])}
            for bag_id in selected
        ],
        "selected_intervals": {
            bag_id: (
                None
                if bag_id not in intervals
                else list(intervals[bag_id]["selected"])
            )
            for bag_id in selected
        },
        "controller_snapshots": {
            bag_id: snapshots.get(bag_id) for bag_id in selected
        },
        "configuration_fingerprints": {
            bag_id: fingerprints.get(bag_id) for bag_id in selected
        },
        "estimator_settings": manifest.get("estimator_settings", {}),
    }


def freshness_fingerprint(manifest: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(freshness_payload(manifest))).hexdigest()
    return "sha256:" + digest


def result_is_fresh(manifest: Mapping[str, Any]) -> bool:
    current = freshness_fingerprint(manifest)
    return bool(manifest.get("run_request_fingerprint")) and (
        current == manifest.get("run_request_fingerprint")
    )


def new_project_manifest(
    project_id: str | None = None,
    *,
    gui_revision: str = "unknown",
    estimator_revision: str = "unknown",
) -> dict[str, Any]:
    identifier = project_id or "project-{}".format(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    if not _PROJECT_ID.fullmatch(identifier):
        raise ProjectIoError("project_id is not a safe directory identifier")
    created = utc_now()
    return {
        "schema": PROJECT_SCHEMA,
        "project_id": identifier,
        "created_at": created,
        "updated_at": created,
        "gui_revision": str(gui_revision),
        "estimator_revision": str(estimator_revision),
        "writer": {"id": PROJECT_WRITER_ID, "version": PROJECT_WRITER_VERSION},
        "loader": {"id": PROJECT_LOADER_ID, "version": PROJECT_LOADER_VERSION},
        "artifact_loaders": {
            "inspection": {
                "id": PROJECT_ARTIFACT_LOADER_ID,
                "version": PROJECT_ARTIFACT_LOADER_VERSION,
            },
            "assimilation_run": {
                "id": PROJECT_ARTIFACT_LOADER_ID,
                "version": PROJECT_ARTIFACT_LOADER_VERSION,
            },
            "pid_proposal_evaluation": {
                "id": PROJECT_ARTIFACT_LOADER_ID,
                "version": PROJECT_ARTIFACT_LOADER_VERSION,
            },
        },
        "bags": [],
        "selected_bag_ids": [],
        "intervals": {},
        "controller_snapshots": {},
        "configuration_fingerprints": {},
        "estimator_settings": {},
        "current_assimilation_run_id": None,
        "current_pid_proposal_evaluation_id": None,
        "run_request_fingerprint": None,
        "result_freshness": "NOT_ESTIMATED",
    }


def _safe_relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ProjectIoError("{} must be a safe POSIX relative path".format(label))
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ProjectIoError("{} must be a safe POSIX relative path".format(label))
    if path.parts and ":" in path.parts[0]:
        raise ProjectIoError("{} cannot contain a drive prefix".format(label))
    return path


def validate_project_manifest(
    value: Mapping[str, Any], *, source_root: str | Path | None = None
) -> dict[str, Any]:
    required = {
        "schema", "project_id", "created_at", "updated_at", "gui_revision",
        "estimator_revision", "writer", "loader", "artifact_loaders", "bags",
        "selected_bag_ids", "intervals", "controller_snapshots",
        "configuration_fingerprints", "estimator_settings",
        "current_assimilation_run_id", "current_pid_proposal_evaluation_id",
        "run_request_fingerprint", "result_freshness",
    }
    missing = required - set(value)
    if missing:
        raise ProjectIoError("project manifest is missing {}".format(", ".join(sorted(missing))))
    if value["schema"] != PROJECT_SCHEMA:
        raise ProjectIoError("unsupported project schema {!r}".format(value["schema"]))
    project_id = value["project_id"]
    if not isinstance(project_id, str) or not _PROJECT_ID.fullmatch(project_id):
        raise ProjectIoError("project_id is not a safe directory identifier")
    for key, expected_id, expected_version in (
        ("writer", PROJECT_WRITER_ID, PROJECT_WRITER_VERSION),
        ("loader", PROJECT_LOADER_ID, PROJECT_LOADER_VERSION),
    ):
        metadata = value[key]
        if not isinstance(metadata, dict) or metadata.get("id") != expected_id:
            raise ProjectIoError("unsupported project {}".format(key))
        if metadata.get("version") != expected_version:
            raise ProjectIoError("unsupported project {} version".format(key))
    artifact_loaders = value["artifact_loaders"]
    if (
        not isinstance(artifact_loaders, dict)
        or set(artifact_loaders) != _ARTIFACT_KINDS
    ):
        raise ProjectIoError("project artifact_loaders are incomplete or unsupported")
    for kind in sorted(_ARTIFACT_KINDS):
        metadata = artifact_loaders[kind]
        if (
            not isinstance(metadata, dict)
            or metadata.get("id") != PROJECT_ARTIFACT_LOADER_ID
        ):
            raise ProjectIoError("unsupported {} artifact loader".format(kind))
        if metadata.get("version") != PROJECT_ARTIFACT_LOADER_VERSION:
            raise ProjectIoError(
                "unsupported {} artifact loader version".format(kind)
            )
    bags = value["bags"]
    if not isinstance(bags, list):
        raise ProjectIoError("bags must be a list")
    identifiers: list[str] = []
    relative_paths: set[str] = set()
    for item in bags:
        if not isinstance(item, dict):
            raise ProjectIoError("each bag entry must be an object")
        if not {"bag_id", "source_path", "relative_path", "sha256"} <= set(item):
            raise ProjectIoError("bag entry is incomplete")
        bag_id = item["bag_id"]
        if not isinstance(bag_id, str) or not _PROJECT_ID.fullmatch(bag_id):
            raise ProjectIoError("bag_id is not a safe identifier")
        relative = str(_safe_relative(item["relative_path"], "bag relative_path"))
        if not relative.startswith("bags/") or relative in relative_paths:
            raise ProjectIoError("bag relative_path must be unique below bags/")
        digest = item["sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ProjectIoError("bag sha256 must be lowercase hexadecimal")
        identifiers.append(bag_id)
        relative_paths.add(relative)
    if len(set(identifiers)) != len(identifiers):
        raise ProjectIoError("bag IDs must be unique")
    selected = value["selected_bag_ids"]
    if (
        not isinstance(selected, list)
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(identifiers)
    ):
        raise ProjectIoError("selected_bag_ids must be unique registered bag IDs")
    intervals = value["intervals"]
    if not isinstance(intervals, dict) or not set(selected) <= set(intervals):
        raise ProjectIoError("each selected bag needs interval metadata")
    for bag_id, interval in intervals.items():
        if bag_id not in identifiers or not isinstance(interval, dict):
            raise ProjectIoError("invalid interval bag ID")
        try:
            auto = [float(x) for x in interval["auto"]]
            chosen = [float(x) for x in interval["selected"]]
            interval_state = interval["state"]
        except (KeyError, TypeError, ValueError) as error:
            raise ProjectIoError("invalid interval metadata") from error
        if (
            len(auto) != 2 or len(chosen) != 2
            or not auto[0] < auto[1] or not chosen[0] < chosen[1]
            or interval_state not in _INTERVAL_STATES
        ):
            raise ProjectIoError("invalid interval bounds or state")
    freshness = value["result_freshness"]
    if freshness not in {"NOT_ESTIMATED", "UP_TO_DATE", "STALE"}:
        raise ProjectIoError("invalid result_freshness")
    run_request_fingerprint = value["run_request_fingerprint"]
    if run_request_fingerprint is not None and (
        not isinstance(run_request_fingerprint, str)
        or not _FRESHNESS_FINGERPRINT.fullmatch(run_request_fingerprint)
    ):
        raise ProjectIoError(
            "run_request_fingerprint must be null or a SHA256 fingerprint"
        )
    for key in (
        "current_assimilation_run_id",
        "current_pid_proposal_evaluation_id",
    ):
        identifier = value[key]
        if identifier is not None and (
            not isinstance(identifier, str)
            or not _PROJECT_ID.fullmatch(identifier)
        ):
            raise ProjectIoError("{} must be null or a safe identifier".format(key))
    root = None if source_root is None else Path(source_root).resolve()
    if root is not None:
        for item in bags:
            path = root.joinpath(*PurePosixPath(item["relative_path"]).parts)
            if not path.is_file():
                raise ProjectIoError("project bag is missing: {}".format(item["relative_path"]))
            if sha256_file(path) != item["sha256"]:
                raise ProjectIoError("project bag SHA256 mismatch: {}".format(item["bag_id"]))
    return json.loads(_canonical_json(dict(value)).decode("utf-8"))


def write_project_manifest(root: str | Path, manifest: Mapping[str, Any]) -> Path:
    directory = Path(root).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    candidate = dict(manifest)
    candidate["updated_at"] = utc_now()
    if candidate.get("run_request_fingerprint"):
        candidate["result_freshness"] = (
            "UP_TO_DATE" if result_is_fresh(candidate) else "STALE"
        )
    validated = validate_project_manifest(candidate)
    destination = directory / PROJECT_MANIFEST_NAME
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(directory), prefix=".project-", suffix=".json"
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json.dumps(validated, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return destination


def read_project_manifest(root: str | Path, *, verify_bags: bool = True) -> dict[str, Any]:
    directory = Path(root).resolve()
    try:
        value = json.loads((directory / PROJECT_MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProjectIoError("cannot read project.json: {}".format(error)) from error
    if not isinstance(value, dict):
        raise ProjectIoError("project.json must contain an object")
    return validate_project_manifest(value, source_root=directory if verify_bags else None)


def create_project_directory(
    projects_root: str | Path, manifest: Mapping[str, Any]
) -> Path:
    validated = validate_project_manifest(manifest)
    base = Path(projects_root).resolve()
    base.mkdir(parents=True, exist_ok=True)
    destination = base / validated["project_id"]
    destination.mkdir(mode=0o700)
    for name in ("bags", "inspection", "runs", "pid_proposals", "logs"):
        (destination / name).mkdir()
    write_project_manifest(destination, validated)
    return destination


def copy_bag_into_project(
    project_root: str | Path,
    source: str | Path,
    bag_id: str | None = None,
) -> dict[str, str]:
    root = Path(project_root).resolve()
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise ProjectIoError("rosbag does not exist: {}".format(source_path))
    identifier = bag_id or "bag-{}".format(sha256_file(source_path)[:12])
    if not _PROJECT_ID.fullmatch(identifier):
        raise ProjectIoError("bag_id is not a safe identifier")
    suffix = source_path.suffix if source_path.suffix else ".bag"
    relative = PurePosixPath("bags") / (identifier + suffix)
    destination = root.joinpath(*relative.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source_path):
            raise ProjectIoError("a different bag already uses {}".format(identifier))
    else:
        shutil.copy2(source_path, destination)
    return {
        "bag_id": identifier,
        "source_path": str(source_path),
        "relative_path": str(relative),
        "sha256": sha256_file(destination),
    }


def _iter_project_files(root: Path) -> Iterable[tuple[Path, str]]:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ProjectIoError("project archives cannot contain symlinks")
        if path.is_file():
            yield path, path.relative_to(root).as_posix()


def save_project_archive(project_root: str | Path, archive_path: str | Path) -> Path:
    root = Path(project_root).resolve()
    read_project_manifest(root, verify_bags=True)
    destination = Path(archive_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=".grape-project-", suffix=".zip"
    )
    os.close(descriptor)
    try:
        with zipfile.ZipFile(
            temporary_name,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as archive:
            for path, relative in _iter_project_files(root):
                archive.write(path, arcname=relative)
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return destination


def _validate_zip_members(
    archive: zipfile.ZipFile, limits: ArchiveLimits
) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    infos = archive.infolist()
    if len(infos) > limits.max_entries:
        raise ProjectIoError("project archive contains too many entries")
    total = 0
    names: set[str] = set()
    result: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    for info in infos:
        relative = _safe_relative(info.filename.rstrip("/"), "archive member")
        normalised = str(relative)
        if normalised in names:
            raise ProjectIoError("project archive contains duplicate paths")
        names.add(normalised)
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            raise ProjectIoError("project archive cannot contain symlinks")
        if info.flag_bits & 0x1:
            raise ProjectIoError("encrypted ZIP entries are not supported")
        if info.file_size > limits.max_single_file_bytes:
            raise ProjectIoError("project archive member exceeds the size limit")
        total += info.file_size
        if total > limits.max_total_uncompressed_bytes:
            raise ProjectIoError("project archive exceeds the total size limit")
        if (
            info.file_size > 0
            and info.compress_size > 0
            and info.file_size / info.compress_size > limits.max_compression_ratio
        ):
            raise ProjectIoError("project archive compression ratio is unsafe")
        result.append((info, relative))
    if PROJECT_MANIFEST_NAME not in names:
        raise ProjectIoError("project archive has no project.json")
    return result


def _copy_limited(source: BinaryIO, destination: BinaryIO, expected: int) -> None:
    remaining = expected
    while remaining:
        chunk = source.read(min(4 * 1024**2, remaining))
        if not chunk:
            raise ProjectIoError("project archive entry ended early")
        destination.write(chunk)
        remaining -= len(chunk)
    if source.read(1):
        raise ProjectIoError("project archive entry exceeds its declared size")


def load_project_archive(
    archive_path: str | Path,
    projects_root: str | Path,
    *,
    limits: ArchiveLimits = ArchiveLimits(),
) -> Path:
    source = Path(archive_path).expanduser().resolve()
    base = Path(projects_root).resolve()
    base.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=str(base), prefix=".grape-load-"))
    try:
        try:
            archive = zipfile.ZipFile(source, mode="r", allowZip64=True)
        except (OSError, zipfile.BadZipFile) as error:
            raise ProjectIoError("cannot open project ZIP: {}".format(error)) from error
        with archive:
            members = _validate_zip_members(archive, limits)
            for info, relative in members:
                destination = temporary.joinpath(*relative.parts)
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, mode="r") as input_stream:
                    with destination.open("xb") as output_stream:
                        _copy_limited(input_stream, output_stream, info.file_size)
        manifest = read_project_manifest(temporary, verify_bags=True)
        destination = base / manifest["project_id"]
        if destination.exists():
            manifest["project_id"] = unique_project_id(
                "{}-import".format(manifest["project_id"])
            )
            write_project_manifest(temporary, manifest)
            destination = base / manifest["project_id"]
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def unique_project_id(prefix: str = "project") -> str:
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", prefix).strip("-.") or "project"
    return "{}-{}".format(safe_prefix[:64], uuid.uuid4().hex[:12])


__all__ = [
    "ArchiveLimits", "GUI_STATE_NAME", "PROJECT_LOADER_ID",
    "PROJECT_LOADER_VERSION", "PROJECT_MANIFEST_NAME", "PROJECT_SCHEMA",
    "PROJECT_ARTIFACT_LOADER_ID", "PROJECT_ARTIFACT_LOADER_VERSION",
    "PROJECT_WRITER_ID", "PROJECT_WRITER_VERSION", "ProjectIoError",
    "copy_bag_into_project", "create_project_directory", "freshness_fingerprint",
    "freshness_payload", "load_project_archive", "new_project_manifest",
    "read_project_manifest", "result_is_fresh", "save_project_archive",
    "sha256_file", "unique_project_id", "validate_project_manifest",
    "write_project_manifest",
]
