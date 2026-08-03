"""Bind validated estimator bundles to immutable workflow attempts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

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
    if set(value) == {"path", "sha256", "size_bytes"}:
        path = value["path"]
        digest = value["sha256"]
        size = value["size_bytes"]
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
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
    if manifest.get("stage_id") != expected_stage_id:
        raise WorkflowError("artifact stage_id does not match the attempt")
    if manifest.get("stage_input_fingerprint") != expected_stage_input:
        raise WorkflowError(
            "artifact stage input fingerprint does not match the attempt"
        )
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


__all__ = ["artifact_ref_from_validated_bundle"]
