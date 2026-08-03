"""Safe, Qt-free persistence for the staged project workflow."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping

from .workflow import (
    WorkflowError,
    WorkflowMode,
    WorkflowStage,
    WorkflowState,
    workflow_definition_fingerprint,
)


WORKFLOW_FILE_NAME = "workflow.json"
DEFAULT_DEFINITION_ID = "diagonal-q-then-static-parameters-v1"
DIAGONAL_Q_STAGE_ID = "diagonal_q"
STATIC_PARAMETERS_STAGE_ID = "static_parameters"

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


class WorkflowIoError(WorkflowError):
    """A workflow file or project-local path violates the I/O contract."""


def default_workflow_stages() -> tuple[WorkflowStage, WorkflowStage]:
    """Return the supported two-stage definition without attempt history."""

    return (
        WorkflowStage(
            stage_id=DIAGONAL_Q_STAGE_ID,
            algorithm_version="diagonal-q-em-v1",
        ),
        WorkflowStage(
            stage_id=STATIC_PARAMETERS_STAGE_ID,
            algorithm_version="augmented-static-enkf-v1",
            depends_on=(DIAGONAL_Q_STAGE_ID,),
        ),
    )


def create_default_workflow(
    project_id: str,
    *,
    workflow_id: str | None = None,
    mode: WorkflowMode | str = WorkflowMode.STEP,
) -> WorkflowState:
    """Create a project-bound state using the supported stage definition."""

    selected_project = _safe_id(project_id, "project_id")
    selected_workflow = _safe_id(
        selected_project if workflow_id is None else workflow_id,
        "workflow_id",
    )
    return WorkflowState.create(
        workflow_id=selected_workflow,
        definition_id=DEFAULT_DEFINITION_ID,
        mode=mode,
        stages=default_workflow_stages(),
    )


def load_workflow(
    project_root: str | Path,
    project_id: str,
    *,
    workflow_id: str | None = None,
) -> WorkflowState:
    """Load ``workflow.json``, or return a new STEP state when it is absent."""

    selected_project = _safe_id(project_id, "project_id")
    expected_workflow = _safe_id(
        selected_project if workflow_id is None else workflow_id,
        "workflow_id",
    )
    root = _project_directory(project_root)
    source = root / WORKFLOW_FILE_NAME
    source_kind = _existing_file_kind(source)
    if source_kind is None:
        return create_default_workflow(
            selected_project, workflow_id=expected_workflow
        )
    if source_kind != "regular":
        raise WorkflowIoError(
            "workflow.json must be a regular file inside the project root"
        )

    value = _read_json_object(source)
    try:
        state = WorkflowState.from_dict(value)
    except WorkflowError as error:
        raise WorkflowIoError("invalid workflow.json: {}".format(error)) from error
    _validate_supported_state(state, expected_workflow)
    return state


def save_workflow(
    project_root: str | Path,
    project_id: str,
    state: WorkflowState,
    *,
    workflow_id: str | None = None,
) -> Path:
    """Atomically publish a validated workflow state inside one project."""

    selected_project = _safe_id(project_id, "project_id")
    expected_workflow = _safe_id(
        selected_project if workflow_id is None else workflow_id,
        "workflow_id",
    )
    if not isinstance(state, WorkflowState):
        raise WorkflowIoError("state must be a WorkflowState")
    _validate_supported_state(state, expected_workflow)
    root = _project_directory(project_root)
    destination = root / WORKFLOW_FILE_NAME
    destination_kind = _existing_file_kind(destination)
    if destination_kind not in {None, "regular"}:
        raise WorkflowIoError(
            "workflow.json must be a regular file inside the project root"
        )
    payload = _json_bytes(state.to_dict())

    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".workflow.json.", suffix=".tmp", dir=str(root)
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        # Recheck the fixed destination after the potentially blocking write.
        destination_kind = _existing_file_kind(destination)
        if destination_kind not in {None, "regular"}:
            raise WorkflowIoError(
                "workflow.json changed to a non-regular project entry"
            )
        os.replace(str(temporary_path), str(destination))
        temporary_path = None
        _fsync_directory(root)
    except WorkflowIoError:
        raise
    except OSError as error:
        raise WorkflowIoError(
            "cannot write workflow.json: {}".format(error)
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
    return destination


def recover_interrupted_attempt(
    state: WorkflowState,
    *,
    finished_at: str | None = None,
) -> WorkflowState:
    """Convert the one queued/running attempt left by a restart to INTERRUPTED."""

    if not isinstance(state, WorkflowState):
        raise WorkflowIoError("state must be a WorkflowState")
    timestamp = _aware_utc_timestamp(finished_at)
    active = state.active_attempt
    if active is None:
        return state
    return state.mark_interrupted(
        active.attempt_id,
        "application_restart",
        finished_at=timestamp,
    )


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise WorkflowIoError("{} must be a safe identifier".format(label))
    return value


def _project_directory(project_root: str | Path) -> Path:
    try:
        root = Path(project_root).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise WorkflowIoError(
            "cannot resolve project root: {}".format(error)
        ) from error
    if not root.is_dir():
        raise WorkflowIoError("project root must be an existing directory")
    return root


def _existing_file_kind(path: Path) -> str | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise WorkflowIoError(
            "cannot inspect project workflow entry: {}".format(error)
        ) from error
    if stat.S_ISREG(metadata.st_mode):
        return "regular"
    if stat.S_ISLNK(metadata.st_mode):
        return "symlink"
    return "other"


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WorkflowIoError(
                "workflow.json contains duplicate key {!r}".format(key)
            )
        result[key] = value
    return result


def _reject_constant(token: str) -> Any:
    raise WorkflowIoError(
        "workflow.json contains non-finite number {}".format(token)
    )


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise WorkflowIoError(
            "workflow.json contains non-finite number {}".format(token)
        )
    return value


def _read_json_object(path: Path) -> Mapping[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(str(path), flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise WorkflowIoError("workflow.json must be a regular file")
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = None
            value = json.load(
                stream,
                object_pairs_hook=_reject_duplicate_object,
                parse_constant=_reject_constant,
                parse_float=_finite_float,
            )
    except WorkflowIoError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise WorkflowIoError(
            "cannot read workflow.json: {}".format(error)
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not isinstance(value, Mapping):
        raise WorkflowIoError("workflow.json must contain one JSON object")
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise WorkflowIoError("workflow state is not finite JSON") from error


def _validate_supported_state(state: WorkflowState, workflow_id: str) -> None:
    if state.workflow_id != workflow_id:
        raise WorkflowIoError(
            "workflow ID does not match this project workflow"
        )
    expected_stages = default_workflow_stages()
    expected_fingerprint = workflow_definition_fingerprint(
        DEFAULT_DEFINITION_ID, expected_stages
    )
    if (
        state.definition_id != DEFAULT_DEFINITION_ID
        or state.definition_fingerprint != expected_fingerprint
    ):
        raise WorkflowIoError(
            "workflow definition does not match the supported two-stage definition"
        )


def _aware_utc_timestamp(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if not isinstance(value, str) or not value:
        raise WorkflowIoError("finished_at must be an aware UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise WorkflowIoError(
            "finished_at must be an aware UTC timestamp"
        ) from error
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.utcoffset() != timedelta(0)
    ):
        raise WorkflowIoError("finished_at must be an aware UTC timestamp")
    return value


def _fsync_directory(directory: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            str(directory), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        os.fsync(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)


__all__ = [
    "DEFAULT_DEFINITION_ID",
    "DIAGONAL_Q_STAGE_ID",
    "STATIC_PARAMETERS_STAGE_ID",
    "WORKFLOW_FILE_NAME",
    "WorkflowIoError",
    "create_default_workflow",
    "default_workflow_stages",
    "load_workflow",
    "recover_interrupted_attempt",
    "save_workflow",
]
