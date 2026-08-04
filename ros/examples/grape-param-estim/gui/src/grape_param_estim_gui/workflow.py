"""Qt-free immutable state for one-command batch-estimation attempts.

Artifact bundles retain their own ``writing``/``complete``/``cancelled``
contract.  This module records attempts around those bundles and derives the
user-facing READY/BLOCKED/RUNNING/COMPLETE/RETRY/STALE state without mutating
an already completed artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Any, Mapping, Sequence


WORKFLOW_SCHEMA = "grape-param-estim/batch-workflow/v1"
STAGE_INPUT_SCHEMA = "grape-param-estim/batch-run-input/v1"
ARTIFACT_CONTENT_SCHEMA = "grape-param-estim/artifact-content/v1"
COMPLETION_SCHEMA = "grape-param-estim/stage-completion/v1"

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_FINGERPRINT = re.compile(r"^sha256:[0-9a-f]{64}$")
_RAW_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class WorkflowError(ValueError):
    """A workflow value or serialized representation is invalid."""


class WorkflowTransitionError(WorkflowError):
    """A requested attempt transition is not allowed."""


class WorkflowMode(str, Enum):
    STEP = "STEP"
    ALL = "ALL"


class AttemptStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"

    @property
    def active(self) -> bool:
        return self in {AttemptStatus.QUEUED, AttemptStatus.RUNNING}

    @property
    def retryable(self) -> bool:
        return self in {
            AttemptStatus.FAILED,
            AttemptStatus.CANCELLED,
            AttemptStatus.INTERRUPTED,
        }


class StageStatus(str, Enum):
    READY = "READY"
    BLOCKED = "BLOCKED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    RETRY = "RETRY"
    STALE = "STALE"


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        details = []
        if missing:
            details.append("missing {}".format(", ".join(missing)))
        if extra:
            details.append("unexpected {}".format(", ".join(extra)))
        raise WorkflowError("{} has {}".format(label, "; ".join(details)))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkflowError("{} must be an object".format(label))
    if any(not isinstance(key, str) for key in value):
        raise WorkflowError("{} keys must be strings".format(label))
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise WorkflowError("{} must be a list".format(label))
    return value


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise WorkflowError("{} must be a safe identifier".format(label))
    return value


def _safe_relative(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\\" in value
    ):
        raise WorkflowError("{} must be a safe POSIX relative path".format(label))
    path = PurePosixPath(value)
    if path.is_absolute() or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise WorkflowError("{} must be a safe POSIX relative path".format(label))
    if path.parts and ":" in path.parts[0]:
        raise WorkflowError("{} cannot contain a drive prefix".format(label))
    return str(path)


def _fingerprint(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _FINGERPRINT.fullmatch(value):
        raise WorkflowError("{} must be a SHA256 fingerprint".format(label))
    return value


def _timestamp(value: Any, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise WorkflowError("{} must be an ISO-8601 timestamp".format(label))
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise WorkflowError(
            "{} must be an ISO-8601 timestamp".format(label)
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise WorkflowError("{} must include a UTC offset".format(label))
    return value


def _finite_json(value: Any, label: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise WorkflowError("{} contains a non-finite number".format(label))
        return value
    if isinstance(value, (list, tuple)):
        return [
            _finite_json(item, "{}[{}]".format(label, index))
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise WorkflowError("{} contains a non-string key".format(label))
        return {
            key: _finite_json(item, "{}.{}".format(label, key))
            for key, item in value.items()
        }
    raise WorkflowError(
        "{} contains unsupported {}".format(label, type(value).__name__)
    )


def canonical_fingerprint(value: Any) -> str:
    """Fingerprint finite JSON independently of mapping insertion order."""

    payload = json.dumps(
        _finite_json(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class UpstreamRef:
    stage_id: str
    attempt_id: str
    completion_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_id", _safe_id(self.stage_id, "stage_id"))
        object.__setattr__(
            self, "attempt_id", _safe_id(self.attempt_id, "attempt_id")
        )
        object.__setattr__(
            self,
            "completion_fingerprint",
            _fingerprint(
                self.completion_fingerprint, "completion_fingerprint"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "attempt_id": self.attempt_id,
            "completion_fingerprint": self.completion_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UpstreamRef":
        source = _mapping(value, "upstream reference")
        _exact_keys(
            source,
            {"stage_id", "attempt_id", "completion_fingerprint"},
            "upstream reference",
        )
        return cls(
            stage_id=source["stage_id"],
            attempt_id=source["attempt_id"],
            completion_fingerprint=source["completion_fingerprint"],
        )


def _normalise_upstream(values: Sequence[UpstreamRef]) -> tuple[UpstreamRef, ...]:
    result = tuple(values)
    if any(not isinstance(value, UpstreamRef) for value in result):
        raise WorkflowError("upstream must contain UpstreamRef values")
    if len({value.stage_id for value in result}) != len(result):
        raise WorkflowError("upstream stage IDs must be unique")
    return tuple(sorted(result, key=lambda value: value.stage_id))


def stage_input_fingerprint(
    *,
    definition_fingerprint: str,
    stage_id: str,
    algorithm_version: str,
    root_input_fingerprint: str,
    stage_settings: Mapping[str, Any],
    upstream: Sequence[UpstreamRef] = (),
) -> str:
    """Fingerprint one stage's scientific inputs and transitive parents."""

    definition = _fingerprint(
        definition_fingerprint, "definition_fingerprint"
    )
    identifier = _safe_id(stage_id, "stage_id")
    if not isinstance(algorithm_version, str) or not algorithm_version:
        raise WorkflowError("algorithm_version cannot be empty")
    root = _fingerprint(root_input_fingerprint, "root_input_fingerprint")
    settings = _mapping(stage_settings, "stage_settings")
    parents = _normalise_upstream(tuple(upstream))
    return canonical_fingerprint(
        {
            "schema": STAGE_INPUT_SCHEMA,
            "definition_fingerprint": definition,
            "stage_id": identifier,
            "algorithm_version": algorithm_version,
            "root_input_fingerprint": root,
            "stage_settings": settings,
            "upstream": [value.to_dict() for value in parents],
        }
    )


def artifact_content_fingerprint(
    manifest: Mapping[str, Any], file_fingerprints: Mapping[str, str]
) -> str:
    """Fingerprint a validated manifest and its declared file digests."""

    selected_manifest = _mapping(manifest, "artifact manifest")
    selected_files = _mapping(file_fingerprints, "file_fingerprints")
    files = []
    for raw_path, raw_digest in selected_files.items():
        path = _safe_relative(raw_path, "artifact file path")
        if not isinstance(raw_digest, str):
            raise WorkflowError("artifact file digest must be a string")
        digest = (
            raw_digest[7:]
            if raw_digest.startswith("sha256:")
            else raw_digest
        )
        if not _RAW_SHA256.fullmatch(digest):
            raise WorkflowError("artifact file digest must be SHA256")
        files.append({"path": path, "sha256": digest})
    files.sort(key=lambda value: value["path"])
    return canonical_fingerprint(
        {
            "schema": ARTIFACT_CONTENT_SCHEMA,
            "manifest": selected_manifest,
            "files": files,
        }
    )


def completion_fingerprint(
    *,
    stage_input: str,
    request_fingerprint: str,
    artifact_schema: str,
    artifact_content: str,
) -> str:
    """Bind a complete artifact to the exact stage request that produced it."""

    if not isinstance(artifact_schema, str) or not artifact_schema:
        raise WorkflowError("artifact_schema cannot be empty")
    return canonical_fingerprint(
        {
            "schema": COMPLETION_SCHEMA,
            "stage_input_fingerprint": _fingerprint(
                stage_input, "stage_input"
            ),
            "request_fingerprint": _fingerprint(
                request_fingerprint, "request_fingerprint"
            ),
            "artifact_schema": artifact_schema,
            "artifact_content_fingerprint": _fingerprint(
                artifact_content, "artifact_content"
            ),
        }
    )


@dataclass(frozen=True)
class ArtifactRef:
    schema: str
    artifact_id: str
    relative_path: str
    content_fingerprint: str
    completion_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.schema, str) or not self.schema:
            raise WorkflowError("artifact schema cannot be empty")
        object.__setattr__(
            self, "artifact_id", _safe_id(self.artifact_id, "artifact_id")
        )
        object.__setattr__(
            self,
            "relative_path",
            _safe_relative(self.relative_path, "artifact relative_path"),
        )
        object.__setattr__(
            self,
            "content_fingerprint",
            _fingerprint(self.content_fingerprint, "content_fingerprint"),
        )
        object.__setattr__(
            self,
            "completion_fingerprint",
            _fingerprint(
                self.completion_fingerprint, "completion_fingerprint"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "relative_path": self.relative_path,
            "content_fingerprint": self.content_fingerprint,
            "completion_fingerprint": self.completion_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        source = _mapping(value, "artifact reference")
        _exact_keys(
            source,
            {
                "schema",
                "artifact_id",
                "relative_path",
                "content_fingerprint",
                "completion_fingerprint",
            },
            "artifact reference",
        )
        return cls(
            schema=source["schema"],
            artifact_id=source["artifact_id"],
            relative_path=source["relative_path"],
            content_fingerprint=source["content_fingerprint"],
            completion_fingerprint=source["completion_fingerprint"],
        )


@dataclass(frozen=True)
class StageAttempt:
    attempt_id: str
    number: int
    status: AttemptStatus
    request_path: str
    output_path: str
    root_input_fingerprint: str
    stage_input_fingerprint: str
    request_fingerprint: str
    upstream: tuple[UpstreamRef, ...]
    retry_of: str | None
    resume: bool
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    artifact: ArtifactRef | None = None
    failure: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "attempt_id", _safe_id(self.attempt_id, "attempt_id")
        )
        if (
            isinstance(self.number, bool)
            or not isinstance(self.number, int)
            or self.number < 1
        ):
            raise WorkflowError("attempt number must be a positive integer")
        try:
            selected_status = AttemptStatus(self.status)
        except (TypeError, ValueError) as error:
            raise WorkflowError("unknown attempt status") from error
        object.__setattr__(self, "status", selected_status)
        request_path = _safe_relative(self.request_path, "request_path")
        output_path = _safe_relative(self.output_path, "output_path")
        if request_path == output_path:
            raise WorkflowError("request_path and output_path must differ")
        object.__setattr__(self, "request_path", request_path)
        object.__setattr__(self, "output_path", output_path)
        for name in (
            "root_input_fingerprint",
            "stage_input_fingerprint",
            "request_fingerprint",
        ):
            object.__setattr__(self, name, _fingerprint(getattr(self, name), name))
        object.__setattr__(self, "upstream", _normalise_upstream(self.upstream))
        if self.retry_of is not None:
            object.__setattr__(
                self, "retry_of", _safe_id(self.retry_of, "retry_of")
            )
            if self.retry_of == self.attempt_id:
                raise WorkflowError("an attempt cannot retry itself")
        if not isinstance(self.resume, bool):
            raise WorkflowError("resume must be boolean")
        if self.resume and self.retry_of is None:
            raise WorkflowError("a resumed attempt must identify its prior attempt")
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))
        object.__setattr__(
            self,
            "started_at",
            _timestamp(self.started_at, "started_at", optional=True),
        )
        object.__setattr__(
            self,
            "finished_at",
            _timestamp(self.finished_at, "finished_at", optional=True),
        )
        if self.artifact is not None and not isinstance(self.artifact, ArtifactRef):
            raise WorkflowError("artifact must be an ArtifactRef")
        if self.failure is not None and (
            not isinstance(self.failure, str) or not self.failure
        ):
            raise WorkflowError("failure must be null or non-empty text")

        if selected_status == AttemptStatus.QUEUED:
            if any(
                value is not None
                for value in (
                    self.started_at,
                    self.finished_at,
                    self.artifact,
                    self.failure,
                )
            ):
                raise WorkflowError("a queued attempt cannot have terminal data")
        elif selected_status == AttemptStatus.RUNNING:
            if (
                self.started_at is None
                or self.finished_at is not None
                or self.artifact is not None
                or self.failure is not None
            ):
                raise WorkflowError("a running attempt has invalid lifecycle data")
        elif selected_status == AttemptStatus.COMPLETE:
            if (
                self.finished_at is None
                or self.artifact is None
                or self.failure is not None
            ):
                raise WorkflowError("a complete attempt requires one artifact")
            if self.artifact.relative_path != self.output_path:
                raise WorkflowError("complete artifact path must equal output_path")
            expected = completion_fingerprint(
                stage_input=self.stage_input_fingerprint,
                request_fingerprint=self.request_fingerprint,
                artifact_schema=self.artifact.schema,
                artifact_content=self.artifact.content_fingerprint,
            )
            if self.artifact.completion_fingerprint != expected:
                raise WorkflowError(
                    "artifact completion fingerprint does not match the attempt"
                )
        else:
            if (
                self.finished_at is None
                or self.artifact is not None
                or self.failure is None
            ):
                raise WorkflowError(
                    "an unsuccessful attempt requires a reason and finish time"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "number": self.number,
            "status": self.status.value,
            "request_path": self.request_path,
            "output_path": self.output_path,
            "root_input_fingerprint": self.root_input_fingerprint,
            "stage_input_fingerprint": self.stage_input_fingerprint,
            "request_fingerprint": self.request_fingerprint,
            "upstream": [value.to_dict() for value in self.upstream],
            "retry_of": self.retry_of,
            "resume": self.resume,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "artifact": None if self.artifact is None else self.artifact.to_dict(),
            "failure": self.failure,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageAttempt":
        source = _mapping(value, "stage attempt")
        _exact_keys(
            source,
            {
                "attempt_id",
                "number",
                "status",
                "request_path",
                "output_path",
                "root_input_fingerprint",
                "stage_input_fingerprint",
                "request_fingerprint",
                "upstream",
                "retry_of",
                "resume",
                "created_at",
                "started_at",
                "finished_at",
                "artifact",
                "failure",
            },
            "stage attempt",
        )
        artifact = source["artifact"]
        return cls(
            attempt_id=source["attempt_id"],
            number=source["number"],
            status=source["status"],
            request_path=source["request_path"],
            output_path=source["output_path"],
            root_input_fingerprint=source["root_input_fingerprint"],
            stage_input_fingerprint=source["stage_input_fingerprint"],
            request_fingerprint=source["request_fingerprint"],
            upstream=tuple(
                UpstreamRef.from_dict(item)
                for item in _list(source["upstream"], "attempt upstream")
            ),
            retry_of=source["retry_of"],
            resume=source["resume"],
            created_at=source["created_at"],
            started_at=source["started_at"],
            finished_at=source["finished_at"],
            artifact=(
                None if artifact is None else ArtifactRef.from_dict(artifact)
            ),
            failure=source["failure"],
        )


@dataclass(frozen=True)
class WorkflowStage:
    stage_id: str
    algorithm_version: str
    depends_on: tuple[str, ...] = ()
    attempts: tuple[StageAttempt, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_id", _safe_id(self.stage_id, "stage_id"))
        if not isinstance(self.algorithm_version, str) or not self.algorithm_version:
            raise WorkflowError("algorithm_version cannot be empty")
        dependencies = tuple(
            _safe_id(value, "dependency stage_id") for value in self.depends_on
        )
        if len(set(dependencies)) != len(dependencies):
            raise WorkflowError("stage dependencies must be unique")
        if self.stage_id in dependencies:
            raise WorkflowError("a stage cannot depend on itself")
        object.__setattr__(self, "depends_on", dependencies)
        attempts = tuple(self.attempts)
        if any(not isinstance(value, StageAttempt) for value in attempts):
            raise WorkflowError("attempts must contain StageAttempt values")
        if any(value.number != index for index, value in enumerate(attempts, 1)):
            raise WorkflowError("attempt numbers must be contiguous")
        object.__setattr__(self, "attempts", attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "algorithm_version": self.algorithm_version,
            "depends_on": list(self.depends_on),
            "attempts": [value.to_dict() for value in self.attempts],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkflowStage":
        source = _mapping(value, "workflow stage")
        _exact_keys(
            source,
            {"stage_id", "algorithm_version", "depends_on", "attempts"},
            "workflow stage",
        )
        return cls(
            stage_id=source["stage_id"],
            algorithm_version=source["algorithm_version"],
            depends_on=tuple(
                _list(source["depends_on"], "stage dependencies")
            ),
            attempts=tuple(
                StageAttempt.from_dict(item)
                for item in _list(source["attempts"], "stage attempts")
            ),
        )


def workflow_definition_fingerprint(
    definition_id: str, stages: Sequence[WorkflowStage]
) -> str:
    identifier = _safe_id(definition_id, "definition_id")
    selected = tuple(stages)
    if not selected or any(
        not isinstance(stage, WorkflowStage) for stage in selected
    ):
        raise WorkflowError(
            "workflow definition requires at least one valid stage"
        )
    return canonical_fingerprint(
        {
            "schema": WORKFLOW_SCHEMA + "/definition",
            "definition_id": identifier,
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "algorithm_version": stage.algorithm_version,
                    "depends_on": list(stage.depends_on),
                }
                for stage in selected
            ],
        }
    )


@dataclass(frozen=True)
class WorkflowState:
    workflow_id: str
    definition_id: str
    definition_fingerprint: str
    mode: WorkflowMode
    stages: tuple[WorkflowStage, ...]
    schema: str = WORKFLOW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != WORKFLOW_SCHEMA:
            raise WorkflowError("unsupported workflow schema")
        object.__setattr__(
            self, "workflow_id", _safe_id(self.workflow_id, "workflow_id")
        )
        object.__setattr__(
            self, "definition_id", _safe_id(self.definition_id, "definition_id")
        )
        object.__setattr__(
            self,
            "definition_fingerprint",
            _fingerprint(self.definition_fingerprint, "definition_fingerprint"),
        )
        try:
            selected_mode = WorkflowMode(self.mode)
        except (TypeError, ValueError) as error:
            raise WorkflowError("workflow mode must be STEP or ALL") from error
        object.__setattr__(self, "mode", selected_mode)
        stages = tuple(self.stages)
        if not stages or any(not isinstance(value, WorkflowStage) for value in stages):
            raise WorkflowError("workflow requires at least one valid stage")
        stage_ids = [value.stage_id for value in stages]
        if len(set(stage_ids)) != len(stage_ids):
            raise WorkflowError("workflow stage IDs must be unique")
        seen: set[str] = set()
        attempts_by_id: dict[str, tuple[str, StageAttempt]] = {}
        active = []
        for stage in stages:
            if any(dependency not in seen for dependency in stage.depends_on):
                raise WorkflowError(
                    "stage dependencies must precede their dependent stage"
                )
            seen.add(stage.stage_id)
            for attempt in stage.attempts:
                if attempt.attempt_id in attempts_by_id:
                    raise WorkflowError("attempt IDs must be globally unique")
                attempts_by_id[attempt.attempt_id] = (stage.stage_id, attempt)
                if attempt.status.active:
                    active.append(attempt.attempt_id)
        if len(active) > 1:
            raise WorkflowError("a workflow can contain only one active attempt")
        for stage in stages:
            dependencies = set(stage.depends_on)
            prior_attempt_ids: set[str] = set()
            for attempt in stage.attempts:
                if {value.stage_id for value in attempt.upstream} != dependencies:
                    raise WorkflowError(
                        "attempt upstream does not match stage dependencies"
                    )
                for parent in attempt.upstream:
                    located = attempts_by_id.get(parent.attempt_id)
                    if located is None or located[0] != parent.stage_id:
                        raise WorkflowError("upstream attempt does not exist")
                    parent_attempt = located[1]
                    if (
                        parent_attempt.status != AttemptStatus.COMPLETE
                        or parent_attempt.artifact is None
                        or parent_attempt.artifact.completion_fingerprint
                        != parent.completion_fingerprint
                    ):
                        raise WorkflowError("upstream attempt is not complete")
                if attempt.retry_of is not None and attempt.retry_of not in prior_attempt_ids:
                    raise WorkflowError(
                        "retry_of must identify an earlier attempt in the stage"
                    )
                if attempt.resume:
                    previous = next(
                        value
                        for value in stage.attempts
                        if value.attempt_id == attempt.retry_of
                    )
                    if previous.status.active:
                        raise WorkflowError(
                            "a resumed attempt must follow a terminal attempt"
                        )
                    if previous.output_path != attempt.output_path:
                        raise WorkflowError(
                            "a resumed attempt must reuse its prior output path"
                        )
                prior_attempt_ids.add(attempt.attempt_id)
        object.__setattr__(self, "stages", stages)
        expected_definition = workflow_definition_fingerprint(
            self.definition_id, stages
        )
        if self.definition_fingerprint != expected_definition:
            raise WorkflowError(
                "definition fingerprint does not match the stage definition"
            )

    @classmethod
    def create(
        cls,
        workflow_id: str,
        definition_id: str,
        mode: WorkflowMode | str,
        stages: Sequence[WorkflowStage],
    ) -> "WorkflowState":
        selected_stages = tuple(stages)
        try:
            selected_mode = WorkflowMode(mode)
        except (TypeError, ValueError) as error:
            raise WorkflowError("workflow mode must be STEP or ALL") from error
        return cls(
            workflow_id=workflow_id,
            definition_id=definition_id,
            definition_fingerprint=workflow_definition_fingerprint(
                definition_id, selected_stages
            ),
            mode=selected_mode,
            stages=selected_stages,
        )

    def with_mode(self, mode: WorkflowMode | str) -> "WorkflowState":
        try:
            selected = WorkflowMode(mode)
        except (TypeError, ValueError) as error:
            raise WorkflowError("workflow mode must be STEP or ALL") from error
        return replace(self, mode=selected)

    @property
    def active_attempt(self) -> StageAttempt | None:
        for stage in self.stages:
            for attempt in stage.attempts:
                if attempt.status.active:
                    return attempt
        return None

    def stage(self, stage_id: str) -> WorkflowStage:
        identifier = _safe_id(stage_id, "stage_id")
        for stage in self.stages:
            if stage.stage_id == identifier:
                return stage
        raise WorkflowError("unknown workflow stage {}".format(identifier))

    def attempt(self, attempt_id: str) -> StageAttempt:
        identifier = _safe_id(attempt_id, "attempt_id")
        for stage in self.stages:
            for attempt in stage.attempts:
                if attempt.attempt_id == identifier:
                    return attempt
        raise WorkflowError("unknown workflow attempt {}".format(identifier))

    def _upstream_is_valid(
        self, stage: WorkflowStage, upstream: tuple[UpstreamRef, ...]
    ) -> bool:
        if {value.stage_id for value in upstream} != set(stage.depends_on):
            return False
        for parent in upstream:
            try:
                attempt = self.attempt(parent.attempt_id)
            except WorkflowError:
                return False
            if (
                attempt.status != AttemptStatus.COMPLETE
                or attempt.artifact is None
                or attempt.artifact.completion_fingerprint
                != parent.completion_fingerprint
            ):
                return False
            parent_stage = next(
                value
                for value in self.stages
                if attempt in value.attempts
            )
            if parent_stage.stage_id != parent.stage_id:
                return False
        return True

    def stage_status(
        self,
        stage_id: str,
        expected_input_fingerprint: str,
        upstream: Sequence[UpstreamRef] = (),
    ) -> StageStatus:
        stage = self.stage(stage_id)
        expected = _fingerprint(
            expected_input_fingerprint, "expected_input_fingerprint"
        )
        selected_upstream = _normalise_upstream(tuple(upstream))
        active = self.active_attempt
        if active is not None and active in stage.attempts:
            return StageStatus.RUNNING
        if not self._upstream_is_valid(stage, selected_upstream):
            return StageStatus.BLOCKED
        matching = [
            value
            for value in stage.attempts
            if value.stage_input_fingerprint == expected
            and value.upstream == selected_upstream
        ]
        if matching:
            latest = matching[-1]
            if latest.status == AttemptStatus.COMPLETE:
                return StageStatus.COMPLETE
            if latest.status.retryable:
                return StageStatus.RETRY
        if any(
            value.status == AttemptStatus.COMPLETE for value in stage.attempts
        ):
            return StageStatus.STALE
        return StageStatus.READY

    def completion_ref(
        self,
        stage_id: str,
        expected_input_fingerprint: str,
        upstream: Sequence[UpstreamRef] = (),
    ) -> UpstreamRef | None:
        if (
            self.stage_status(stage_id, expected_input_fingerprint, upstream)
            != StageStatus.COMPLETE
        ):
            return None
        selected_upstream = _normalise_upstream(tuple(upstream))
        stage = self.stage(stage_id)
        attempt = next(
            value
            for value in reversed(stage.attempts)
            if value.stage_input_fingerprint == expected_input_fingerprint
            and value.upstream == selected_upstream
            and value.status == AttemptStatus.COMPLETE
        )
        if attempt.artifact is None:  # protected by StageAttempt invariants
            raise AssertionError("complete attempt has no artifact")
        return UpstreamRef(
            stage_id=stage.stage_id,
            attempt_id=attempt.attempt_id,
            completion_fingerprint=attempt.artifact.completion_fingerprint,
        )

    def begin_attempt(
        self,
        *,
        stage_id: str,
        attempt_id: str,
        request_path: str,
        output_path: str,
        root_input_fingerprint: str,
        stage_input: str,
        request_fingerprint: str,
        upstream: Sequence[UpstreamRef] = (),
        resume: bool = False,
        created_at: str,
    ) -> "WorkflowState":
        stage = self.stage(stage_id)
        selected_upstream = _normalise_upstream(tuple(upstream))
        status = self.stage_status(stage.stage_id, stage_input, selected_upstream)
        if self.active_attempt is not None:
            raise WorkflowTransitionError(
                "another workflow attempt is already active"
            )
        if status not in {StageStatus.READY, StageStatus.RETRY, StageStatus.STALE}:
            raise WorkflowTransitionError(
                "cannot begin {} while stage is {}".format(
                    stage.stage_id, status.value
                )
            )
        identifier = _safe_id(attempt_id, "attempt_id")
        if any(
            identifier == value.attempt_id
            for selected_stage in self.stages
            for value in selected_stage.attempts
        ):
            raise WorkflowTransitionError("attempt ID already exists")
        retry_of = (
            stage.attempts[-1].attempt_id
            if stage.attempts
            and status in {StageStatus.RETRY, StageStatus.STALE}
            else None
        )
        if resume and retry_of is None:
            raise WorkflowTransitionError(
                "resume requires an earlier terminal attempt"
            )
        attempt = StageAttempt(
            attempt_id=identifier,
            number=len(stage.attempts) + 1,
            status=AttemptStatus.QUEUED,
            request_path=request_path,
            output_path=output_path,
            root_input_fingerprint=root_input_fingerprint,
            stage_input_fingerprint=stage_input,
            request_fingerprint=request_fingerprint,
            upstream=selected_upstream,
            retry_of=retry_of,
            resume=resume,
            created_at=created_at,
        )
        return self._replace_stage(
            replace(stage, attempts=stage.attempts + (attempt,))
        )

    def retry_attempt(
        self,
        *,
        stage_id: str,
        attempt_id: str,
        request_path: str,
        output_path: str,
        root_input_fingerprint: str,
        stage_input: str,
        request_fingerprint: str,
        upstream: Sequence[UpstreamRef] = (),
        resume: bool = False,
        created_at: str,
    ) -> "WorkflowState":
        status = self.stage_status(stage_id, stage_input, upstream)
        if status not in {StageStatus.RETRY, StageStatus.STALE}:
            raise WorkflowTransitionError(
                "stage {} is not retryable".format(stage_id)
            )
        return self.begin_attempt(
            stage_id=stage_id,
            attempt_id=attempt_id,
            request_path=request_path,
            output_path=output_path,
            root_input_fingerprint=root_input_fingerprint,
            stage_input=stage_input,
            request_fingerprint=request_fingerprint,
            upstream=upstream,
            resume=resume,
            created_at=created_at,
        )

    def resume_attempt(
        self,
        *,
        stage_id: str,
        attempt_id: str,
        request_path: str,
        output_path: str,
        root_input_fingerprint: str,
        stage_input: str,
        request_fingerprint: str,
        upstream: Sequence[UpstreamRef] = (),
        created_at: str,
    ) -> "WorkflowState":
        """Resume the latest terminal attempt in its existing run directory."""

        stage = self.stage(stage_id)
        if not stage.attempts:
            raise WorkflowTransitionError(
                "resume requires an earlier terminal attempt"
            )
        previous = stage.attempts[-1]
        if previous.status.active:
            raise WorkflowTransitionError(
                "resume requires an earlier terminal attempt"
            )
        selected_output = _safe_relative(output_path, "output_path")
        if selected_output != previous.output_path:
            raise WorkflowTransitionError(
                "resume must reuse the prior output path"
            )
        return self.begin_attempt(
            stage_id=stage_id,
            attempt_id=attempt_id,
            request_path=request_path,
            output_path=selected_output,
            root_input_fingerprint=root_input_fingerprint,
            stage_input=stage_input,
            request_fingerprint=request_fingerprint,
            upstream=upstream,
            resume=True,
            created_at=created_at,
        )

    def mark_running(
        self, attempt_id: str, *, started_at: str
    ) -> "WorkflowState":
        attempt = self.attempt(attempt_id)
        if attempt.status != AttemptStatus.QUEUED:
            raise WorkflowTransitionError("only a queued attempt can start")
        return self._replace_attempt(
            replace(
                attempt,
                status=AttemptStatus.RUNNING,
                started_at=started_at,
            )
        )

    def mark_complete(
        self,
        attempt_id: str,
        artifact: ArtifactRef,
        *,
        finished_at: str,
    ) -> "WorkflowState":
        attempt = self.attempt(attempt_id)
        if attempt.status not in {AttemptStatus.QUEUED, AttemptStatus.RUNNING}:
            raise WorkflowTransitionError(
                "only an active attempt can become complete"
            )
        return self._replace_attempt(
            replace(
                attempt,
                status=AttemptStatus.COMPLETE,
                finished_at=finished_at,
                artifact=artifact,
            )
        )

    def replace_completed_artifact(
        self,
        attempt_id: str,
        artifact: ArtifactRef,
        *,
        expected_completion_fingerprint: str,
    ) -> "WorkflowState":
        """Atomically rebind one complete attempt to upgraded bundle content.

        Posterior sampling appends to an estimate-only bundle without changing
        the estimation request or stage input.  The expected fingerprint is a
        compare-and-swap guard against silently replacing a different archived
        result.  A referenced upstream attempt cannot be changed in place.
        """

        attempt = self.attempt(attempt_id)
        expected = _fingerprint(
            expected_completion_fingerprint,
            "expected_completion_fingerprint",
        )
        if attempt.status != AttemptStatus.COMPLETE or attempt.artifact is None:
            raise WorkflowTransitionError(
                "only a complete attempt artifact can be replaced"
            )
        if attempt.artifact.completion_fingerprint != expected:
            raise WorkflowTransitionError(
                "completed artifact changed since posterior sampling started"
            )
        if not isinstance(artifact, ArtifactRef):
            raise WorkflowTransitionError("artifact must be an ArtifactRef")
        if artifact.relative_path != attempt.output_path:
            raise WorkflowTransitionError(
                "replacement artifact must reuse the completed output path"
            )
        if any(
            parent.attempt_id == attempt.attempt_id
            for stage in self.stages
            for child in stage.attempts
            for parent in child.upstream
        ):
            raise WorkflowTransitionError(
                "a completed artifact referenced by a downstream attempt "
                "cannot be replaced"
            )
        return self._replace_attempt(replace(attempt, artifact=artifact))

    def recover_validated_worker_output(
        self, attempt_id: str, artifact: ArtifactRef
    ) -> "WorkflowState":
        """Adopt output that was valid but failed GUI post-processing."""

        attempt = self.attempt(attempt_id)
        if (
            attempt.status is not AttemptStatus.FAILED
            or attempt.failure is None
            or not attempt.failure.startswith("invalid_worker_output:")
        ):
            raise WorkflowTransitionError(
                "only an invalid-worker-output failure can be recovered"
            )
        if not isinstance(artifact, ArtifactRef):
            raise WorkflowTransitionError("artifact must be an ArtifactRef")
        if artifact.relative_path != attempt.output_path:
            raise WorkflowTransitionError(
                "recovered artifact must reuse the failed output path"
            )
        return self._replace_attempt(
            replace(
                attempt,
                status=AttemptStatus.COMPLETE,
                artifact=artifact,
                failure=None,
            )
        )

    def mark_failed(
        self, attempt_id: str, reason: str, *, finished_at: str
    ) -> "WorkflowState":
        return self._mark_unsuccessful(
            attempt_id, AttemptStatus.FAILED, reason, finished_at
        )

    def mark_cancelled(
        self, attempt_id: str, reason: str, *, finished_at: str
    ) -> "WorkflowState":
        return self._mark_unsuccessful(
            attempt_id, AttemptStatus.CANCELLED, reason, finished_at
        )

    def mark_interrupted(
        self, attempt_id: str, reason: str, *, finished_at: str
    ) -> "WorkflowState":
        return self._mark_unsuccessful(
            attempt_id, AttemptStatus.INTERRUPTED, reason, finished_at
        )

    def _mark_unsuccessful(
        self,
        attempt_id: str,
        status: AttemptStatus,
        reason: str,
        finished_at: str,
    ) -> "WorkflowState":
        attempt = self.attempt(attempt_id)
        if attempt.status not in {AttemptStatus.QUEUED, AttemptStatus.RUNNING}:
            raise WorkflowTransitionError(
                "only an active attempt can finish unsuccessfully"
            )
        if not isinstance(reason, str) or not reason:
            raise WorkflowTransitionError("failure reason cannot be empty")
        return self._replace_attempt(
            replace(
                attempt,
                status=status,
                finished_at=finished_at,
                failure=reason,
            )
        )

    def _replace_attempt(self, replacement: StageAttempt) -> "WorkflowState":
        for stage in self.stages:
            if any(
                value.attempt_id == replacement.attempt_id
                for value in stage.attempts
            ):
                attempts = tuple(
                    replacement
                    if value.attempt_id == replacement.attempt_id
                    else value
                    for value in stage.attempts
                )
                return self._replace_stage(replace(stage, attempts=attempts))
        raise WorkflowError("unknown workflow attempt")

    def _replace_stage(self, replacement: WorkflowStage) -> "WorkflowState":
        return replace(
            self,
            stages=tuple(
                replacement if value.stage_id == replacement.stage_id else value
                for value in self.stages
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "workflow_id": self.workflow_id,
            "definition_id": self.definition_id,
            "definition_fingerprint": self.definition_fingerprint,
            "mode": self.mode.value,
            "stages": [value.to_dict() for value in self.stages],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkflowState":
        source = _mapping(value, "workflow")
        _exact_keys(
            source,
            {
                "schema",
                "workflow_id",
                "definition_id",
                "definition_fingerprint",
                "mode",
                "stages",
            },
            "workflow",
        )
        return cls(
            schema=source["schema"],
            workflow_id=source["workflow_id"],
            definition_id=source["definition_id"],
            definition_fingerprint=source["definition_fingerprint"],
            mode=source["mode"],
            stages=tuple(
                WorkflowStage.from_dict(item)
                for item in _list(source["stages"], "workflow stages")
            ),
        )


__all__ = [
    "ARTIFACT_CONTENT_SCHEMA",
    "COMPLETION_SCHEMA",
    "STAGE_INPUT_SCHEMA",
    "WORKFLOW_SCHEMA",
    "ArtifactRef",
    "AttemptStatus",
    "StageAttempt",
    "StageStatus",
    "UpstreamRef",
    "WorkflowError",
    "WorkflowMode",
    "WorkflowStage",
    "WorkflowState",
    "WorkflowTransitionError",
    "artifact_content_fingerprint",
    "canonical_fingerprint",
    "completion_fingerprint",
    "stage_input_fingerprint",
    "workflow_definition_fingerprint",
]
