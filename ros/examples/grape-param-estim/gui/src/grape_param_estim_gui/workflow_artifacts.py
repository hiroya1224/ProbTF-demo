"""Bind validated estimator bundles to immutable workflow attempts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

try:
    from grape_param_estim.batch_artifact import (
        ARTIFACT_DESCRIPTOR_KEYS,
        BATCH_ESTIMATION_RUN_SCHEMA,
    )
    from grape_param_estim.batch_request import (
        validate_batch_estimation_request,
    )
except ImportError as error:  # pragma: no cover - GUI startup reports this
    ARTIFACT_DESCRIPTOR_KEYS = ()
    BATCH_ESTIMATION_RUN_SCHEMA = None
    validate_batch_estimation_request = None
    _BACKEND_IMPORT_ERROR = error
else:
    _BACKEND_IMPORT_ERROR = None

from .workflow import (
    ArtifactRef,
    WorkflowError,
    artifact_content_fingerprint,
    completion_fingerprint,
)


def _declared_file_fingerprints(
    value: Any,
    *,
    location: str = "manifest.artifacts",
) -> dict[str, str]:
    """Collect exact payload descriptors from a validated manifest tree."""

    if not isinstance(value, Mapping):
        raise WorkflowError("{} must be an object".format(location))
    if set(value) == set(ARTIFACT_DESCRIPTOR_KEYS):
        path = value["path"]
        digest = value["sha256"]
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
        ):
            raise WorkflowError(
                "{} is not a complete payload descriptor".format(location)
            )
        return {path: digest}

    result: dict[str, str] = {}
    for key, child in value.items():
        if not isinstance(key, str):
            raise WorkflowError("{} has a non-string key".format(location))
        nested = _declared_file_fingerprints(
            child,
            location="{}.{}".format(location, key),
        )
        overlap = set(result).intersection(nested)
        if overlap:
            raise WorkflowError(
                "artifact payload path is declared more than once: {}".format(
                    sorted(overlap)[0]
                )
            )
        result.update(nested)
    if not result:
        raise WorkflowError("{} declares no payload files".format(location))
    return result


def preflight_batch_estimation_launch(
    request: Mapping[str, Any], *, source_path: str | Path
) -> str:
    """Validate the request and GUI/backend artifact handshake before work."""

    if (
        validate_batch_estimation_request is None
        or BATCH_ESTIMATION_RUN_SCHEMA is None
    ):
        raise WorkflowError(
            "batch-estimation backend is unavailable during launch preflight"
        ) from _BACKEND_IMPORT_ERROR
    if BATCH_ESTIMATION_RUN_SCHEMA != (
        "grape-param-estim/batch-estimation-run/v1"
    ):
        raise WorkflowError(
            "GUI does not support backend artifact schema {!r}".format(
                BATCH_ESTIMATION_RUN_SCHEMA
            )
        )
    probe_digest = "sha256:" + "0" * 64
    probe = {
        "probe": {"path": "probe.npz", "sha256": probe_digest}
    }
    if _declared_file_fingerprints(probe) != {"probe.npz": probe_digest}:
        raise WorkflowError(
            "GUI and backend artifact descriptor contracts disagree"
        )
    validated = validate_batch_estimation_request(
        request, source_path=source_path
    )
    return validated.fingerprint


def artifact_ref_from_validated_bundle(
    *,
    project_root: str | Path,
    artifact_root: str | Path,
    manifest: Mapping[str, Any],
    expected_stage_id: str,
    expected_stage_input: str,
    expected_request_fingerprint: str,
) -> ArtifactRef:
    """Create a workflow reference only after stage/request binding checks."""

    if not isinstance(manifest, Mapping):
        raise WorkflowError("artifact manifest must be an object")
    if manifest.get("status") != "complete":
        raise WorkflowError("workflow can reference only a complete artifact")
    schema = manifest.get("schema")
    run_id = manifest.get("run_id")
    if not isinstance(schema, str) or not schema:
        raise WorkflowError("artifact schema cannot be empty")
    if not isinstance(run_id, str) or not run_id:
        raise WorkflowError("artifact run_id cannot be empty")
    if expected_stage_id != "batch_estimation":
        raise WorkflowError("only the batch_estimation stage is supported")
    if manifest.get("request_fingerprint") != expected_request_fingerprint:
        raise WorkflowError(
            "artifact request fingerprint does not match the attempt"
        )
    artifacts = manifest.get("artifacts")
    files = _declared_file_fingerprints(artifacts)

    try:
        project = Path(project_root).resolve(strict=True)
        root = Path(artifact_root).resolve(strict=True)
        relative = root.relative_to(project).as_posix()
    except (OSError, RuntimeError, ValueError) as error:
        raise WorkflowError(
            "artifact root must be inside the current project"
        ) from error
    if not root.is_dir():
        raise WorkflowError("artifact root must be a directory")

    content = artifact_content_fingerprint(manifest, files)
    completion = completion_fingerprint(
        stage_input=expected_stage_input,
        request_fingerprint=expected_request_fingerprint,
        artifact_schema=schema,
        artifact_content=content,
    )
    return ArtifactRef(
        schema=schema,
        artifact_id=run_id,
        relative_path=relative,
        content_fingerprint=content,
        completion_fingerprint=completion,
    )


__all__ = [
    "artifact_ref_from_validated_bundle",
    "preflight_batch_estimation_launch",
]
