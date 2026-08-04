"""Strict atomic checkpoints for one sparse batch estimation request."""

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from grape_param_estim.artifact_io import (
    ArtifactValidationError,
    load_npz_strict,
    read_json,
    write_json_atomic,
    write_npz_atomic,
)
from grape_param_estim.batch.state import BatchState
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.batch_artifact import (
    _validate_bag,
    _validate_diagnostics,
    _validate_laplace,
    _validate_map_static,
    _validate_q_em,
    file_sha256,
)
from grape_param_estim.batch_artifact_export import BatchArtifactPayload
from grape_param_estim.batch_request import BatchEstimationRequest
from grape_param_estim.posterior.checkpoint import (
    McmcChainCheckpoint,
    load_mcmc_checkpoint,
    save_mcmc_checkpoint,
)


BATCH_ESTIMATION_CHECKPOINT_SCHEMA = (
    "grape-param-estim/batch-estimation-checkpoint/v1"
)
_STATUSES = {"core_complete", "sampling", "cancelled", "published"}
_MANIFEST_KEYS = {
    "schema",
    "status",
    "run_id",
    "request_fingerprint",
    "configuration_fingerprint",
    "controller_snapshot_fingerprint",
    "estimator_revision",
    "output_directory",
    "selected_mode_id",
    "core_artifacts",
    "chain_checkpoints",
    "cancellation_reason",
}
_DESCRIPTOR_KEYS = {"path", "sha256"}


def batch_checkpoint_path(output_directory: Union[str, Path]) -> Path:
    """Return the deterministic sibling checkpoint for one exact output."""

    output = Path(output_directory).expanduser().resolve()
    if not output.name:
        raise ValueError("output_directory must name a directory")
    return output.parent / ".{}-batch-checkpoint".format(output.name)


def _canonical(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ArtifactValidationError("{} must be canonical text".format(name))
    return value


def _descriptor(root: Path, relative: str) -> Dict[str, str]:
    return {"path": relative, "sha256": file_sha256(root / relative)}


def _descriptor_path(
    root: Path, descriptor: Any, expected: str, location: str
) -> Path:
    if not isinstance(descriptor, Mapping) or set(descriptor) != _DESCRIPTOR_KEYS:
        raise ArtifactValidationError("{} must be an exact descriptor".format(location))
    if descriptor["path"] != expected:
        raise ArtifactValidationError("{}.path mismatch".format(location))
    digest = descriptor["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 71
        or not digest.startswith("sha256:")
        or any(value not in "0123456789abcdef" for value in digest[7:])
    ):
        raise ArtifactValidationError("{}.sha256 is invalid".format(location))
    relative = Path(expected)
    if relative.is_absolute() or ".." in relative.parts:
        raise ArtifactValidationError("{} escapes checkpoint root".format(location))
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ArtifactValidationError("{} is missing".format(location))
    if file_sha256(path) != digest:
        raise ArtifactValidationError("{} SHA-256 mismatch".format(location))
    return path


def _state_arrays(state: BatchState) -> Mapping[str, np.ndarray]:
    if not isinstance(state, BatchState):
        raise TypeError("state must be BatchState")
    keys = state.layout.variable_keys
    arrays: Dict[str, np.ndarray] = {
        "kind": np.asarray(tuple(value.kind.value for value in keys)),
        "bag_id": np.asarray(tuple(value.bag_id or "" for value in keys)),
        "knot_index": np.asarray(
            tuple(-1 if value.knot_index is None else value.knot_index for value in keys),
            dtype=np.int64,
        ),
    }
    for index, key in enumerate(keys):
        arrays["value_{:06d}".format(index)] = np.asarray(state.value(key))
    return arrays


def _load_state_values(path: Path) -> Mapping[VariableKey, np.ndarray]:
    arrays = load_npz_strict(path)
    required = {"kind", "bag_id", "knot_index"}
    if not required.issubset(arrays):
        raise ArtifactValidationError("checkpoint state index is incomplete")
    kinds = arrays["kind"]
    bags = arrays["bag_id"]
    knots = arrays["knot_index"]
    if (
        kinds.ndim != 1
        or kinds.dtype.kind not in {"U", "S"}
        or bags.shape != kinds.shape
        or bags.dtype.kind not in {"U", "S"}
        or knots.shape != kinds.shape
        or knots.dtype.kind not in {"i", "u"}
        or kinds.size == 0
    ):
        raise ArtifactValidationError("checkpoint state index has invalid arrays")
    expected = required.union(
        "value_{:06d}".format(index) for index in range(kinds.size)
    )
    if set(arrays) != expected:
        raise ArtifactValidationError("checkpoint state value keys are not exact")
    result = {}
    for index in range(kinds.size):
        kind_text = str(kinds[index])
        try:
            kind = VariableKind(kind_text)
        except ValueError as error:
            raise ArtifactValidationError(
                "checkpoint state has an unknown variable kind"
            ) from error
        bag_text = str(bags[index])
        knot = int(knots[index])
        key = VariableKey(
            kind,
            bag_id=(bag_text or None),
            knot_index=(None if knot < 0 else knot),
        )
        if key in result:
            raise ArtifactValidationError("checkpoint state has duplicate keys")
        result[key] = np.asarray(arrays["value_{:06d}".format(index)]).copy()
    return result


@dataclass(frozen=True)
class BatchEstimationCheckpoint:
    """A validated core payload and all durable chain proposal boundaries."""

    root: Path
    manifest: Mapping[str, Any]
    core: BatchArtifactPayload
    state_values: Mapping[VariableKey, np.ndarray]
    chain_checkpoints: Mapping[str, McmcChainCheckpoint]


def _core_descriptors(root: Path, bag_ids: Tuple[str, ...]) -> Mapping[str, Any]:
    return {
        "manifest_metadata": _descriptor(root, "core/manifest_metadata.json"),
        "map_static": _descriptor(root, "core/map_static.npz"),
        "q_em": _descriptor(root, "core/q_em.npz"),
        "laplace": _descriptor(root, "core/laplace.npz"),
        "diagnostics": _descriptor(root, "core/diagnostics.npz"),
        "state": _descriptor(root, "core/state.npz"),
        "bags": {
            bag_id: _descriptor(root, "core/bags/{}.npz".format(bag_id))
            for bag_id in bag_ids
        },
    }


def write_batch_estimation_checkpoint(
    output_directory: Union[str, Path],
    *,
    request: BatchEstimationRequest,
    estimator_revision: str,
    configuration_fingerprint: str,
    controller_snapshot_fingerprint: str,
    selected_mode_id: str,
    core: BatchArtifactPayload,
    state: BatchState,
) -> BatchEstimationCheckpoint:
    """Publish a completed inference core atomically before MCMC starts."""

    if not isinstance(request, BatchEstimationRequest):
        raise TypeError("request must be BatchEstimationRequest")
    output = Path(output_directory).expanduser().resolve()
    if output != request.output_directory:
        raise ValueError("checkpoint output must equal request output_directory")
    if not bool(request.payload["mcmc_settings"]["enabled"]):
        raise ValueError("batch checkpoints are required only before enabled MCMC")
    if core.mcmc_samples is not None or core.trajectories:
        raise ValueError("checkpoint core cannot contain posterior samples")
    if core.manifest_metadata.get("request_fingerprint") != request.fingerprint:
        raise ValueError("checkpoint core request fingerprint mismatch")
    root = batch_checkpoint_path(output)
    if root.exists() or root.is_symlink():
        raise ArtifactValidationError("checkpoint already exists: {}".format(root))
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".{}-".format(root.name), suffix=".writing", dir=str(root.parent)
        )
    )
    published = False
    try:
        write_json_atomic(staging / "core" / "manifest_metadata.json", core.manifest_metadata)
        write_npz_atomic(staging / "core" / "map_static.npz", core.map_static)
        write_npz_atomic(staging / "core" / "q_em.npz", core.q_em)
        write_npz_atomic(staging / "core" / "laplace.npz", core.laplace)
        write_npz_atomic(staging / "core" / "diagnostics.npz", core.diagnostics)
        write_npz_atomic(staging / "core" / "state.npz", _state_arrays(state))
        for bag_id in request.bag_ids:
            write_npz_atomic(
                staging / "core" / "bags" / "{}.npz".format(bag_id),
                core.bags[bag_id],
            )
        manifest = {
            "schema": BATCH_ESTIMATION_CHECKPOINT_SCHEMA,
            "status": "core_complete",
            "run_id": str(request.payload["run_id"]),
            "request_fingerprint": request.fingerprint,
            "configuration_fingerprint": _canonical(
                configuration_fingerprint, "configuration_fingerprint"
            ),
            "controller_snapshot_fingerprint": _canonical(
                controller_snapshot_fingerprint,
                "controller_snapshot_fingerprint",
            ),
            "estimator_revision": _canonical(estimator_revision, "estimator_revision"),
            "output_directory": str(output),
            "selected_mode_id": _canonical(selected_mode_id, "selected_mode_id"),
            "core_artifacts": _core_descriptors(staging, request.bag_ids),
            "chain_checkpoints": {},
            "cancellation_reason": "",
        }
        write_json_atomic(staging / "manifest.json", manifest)
        os.replace(str(staging), str(root))
        published = True
    finally:
        if not published and staging.exists():
            # Files are intentionally left only in a private staging path if
            # cleanup itself is unavailable; they can never be mistaken for
            # the deterministic checkpoint root.
            pass
    return load_batch_estimation_checkpoint(
        output,
        request=request,
        estimator_revision=estimator_revision,
        configuration_fingerprint=configuration_fingerprint,
        controller_snapshot_fingerprint=controller_snapshot_fingerprint,
    )


def _load_core(root: Path, manifest: Mapping[str, Any], bag_ids: Tuple[str, ...]):
    descriptors = manifest["core_artifacts"]
    expected_keys = {
        "manifest_metadata",
        "map_static",
        "q_em",
        "laplace",
        "diagnostics",
        "state",
        "bags",
    }
    if not isinstance(descriptors, Mapping) or set(descriptors) != expected_keys:
        raise ArtifactValidationError("core_artifacts keys are not exact")
    metadata_path = _descriptor_path(
        root,
        descriptors["manifest_metadata"],
        "core/manifest_metadata.json",
        "core_artifacts.manifest_metadata",
    )
    paths = {}
    for name in ("map_static", "q_em", "laplace", "diagnostics", "state"):
        paths[name] = _descriptor_path(
            root,
            descriptors[name],
            "core/{}.npz".format(name),
            "core_artifacts.{}".format(name),
        )
    bag_descriptors = descriptors["bags"]
    if not isinstance(bag_descriptors, Mapping) or set(bag_descriptors) != set(bag_ids):
        raise ArtifactValidationError("core bag descriptors do not match request")
    bag_paths = {
        bag_id: _descriptor_path(
            root,
            bag_descriptors[bag_id],
            "core/bags/{}.npz".format(bag_id),
            "core_artifacts.bags.{}".format(bag_id),
        )
        for bag_id in bag_ids
    }
    metadata = read_json(metadata_path)
    map_static = load_npz_strict(paths["map_static"])
    q_em = load_npz_strict(paths["q_em"])
    laplace = load_npz_strict(paths["laplace"])
    diagnostics = load_npz_strict(paths["diagnostics"])
    bags = {bag_id: load_npz_strict(bag_paths[bag_id]) for bag_id in bag_ids}
    _validate_map_static(map_static, bag_ids, str(paths["map_static"]))
    _validate_q_em(q_em, str(paths["q_em"]))
    _validate_laplace(laplace, str(paths["laplace"]))
    _validate_diagnostics(diagnostics, bag_ids, False, str(paths["diagnostics"]))
    for bag_id in bag_ids:
        _validate_bag(bags[bag_id], bag_id, str(bag_paths[bag_id]))
    if not np.array_equal(map_static["q_diagonal"], q_em["accepted_q"][-1]):
        raise ArtifactValidationError("checkpoint final Q arrays disagree")
    if not np.isclose(map_static["delay"][0], q_em["lag"][-1]):
        raise ArtifactValidationError("checkpoint final delay arrays disagree")
    return (
        BatchArtifactPayload(
            manifest_metadata=metadata,
            map_static=map_static,
            q_em=q_em,
            laplace=laplace,
            diagnostics=diagnostics,
            bags=bags,
            mcmc_samples=None,
            trajectories={},
        ),
        _load_state_values(paths["state"]),
    )


def load_batch_estimation_checkpoint(
    output_directory: Union[str, Path],
    *,
    request: BatchEstimationRequest,
    estimator_revision: str,
    configuration_fingerprint: str,
    controller_snapshot_fingerprint: str,
) -> BatchEstimationCheckpoint:
    """Load a resumable checkpoint only for its exact request and output."""

    if not isinstance(request, BatchEstimationRequest):
        raise TypeError("request must be BatchEstimationRequest")
    output = Path(output_directory).expanduser().resolve()
    root = batch_checkpoint_path(output)
    manifest = read_json(root / "manifest.json")
    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_KEYS:
        raise ArtifactValidationError("checkpoint manifest keys are not exact")
    if manifest["schema"] != BATCH_ESTIMATION_CHECKPOINT_SCHEMA:
        raise ArtifactValidationError("unsupported batch checkpoint schema")
    if manifest["status"] not in _STATUSES or manifest["status"] == "published":
        raise ArtifactValidationError("checkpoint is not resumable")
    expected = {
        "run_id": str(request.payload["run_id"]),
        "request_fingerprint": request.fingerprint,
        "configuration_fingerprint": configuration_fingerprint,
        "controller_snapshot_fingerprint": controller_snapshot_fingerprint,
        "estimator_revision": estimator_revision,
        "output_directory": str(output),
    }
    for key, value in expected.items():
        if manifest[key] != value:
            raise ArtifactValidationError(
                "checkpoint {} mismatch".format(key)
            )
    selected_mode_id = _canonical(manifest["selected_mode_id"], "selected_mode_id")
    requested_modes = {
        str(value["mode_id"]) for value in request.payload["mode_hypotheses"]
    }
    if selected_mode_id not in requested_modes:
        raise ArtifactValidationError("checkpoint selected mode is not requested")
    reason = manifest["cancellation_reason"]
    if manifest["status"] == "cancelled":
        _canonical(reason, "cancellation_reason")
    elif reason != "":
        raise ArtifactValidationError("non-cancelled checkpoint has a reason")
    core, state_values = _load_core(root, manifest, request.bag_ids)
    if core.manifest_metadata.get("request_fingerprint") != request.fingerprint:
        raise ArtifactValidationError("core request fingerprint mismatch")
    if core.manifest_metadata.get("configuration_fingerprint") != configuration_fingerprint:
        raise ArtifactValidationError("core configuration fingerprint mismatch")
    if core.manifest_metadata.get("controller_snapshot_fingerprint") != controller_snapshot_fingerprint:
        raise ArtifactValidationError("core controller fingerprint mismatch")

    raw_chains = manifest["chain_checkpoints"]
    if not isinstance(raw_chains, Mapping):
        raise ArtifactValidationError("chain_checkpoints must be an object")
    chains = {}
    for chain_id, descriptor in raw_chains.items():
        _canonical(chain_id, "chain_id")
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "path",
            "sha256",
            "completed_transition",
        }:
            raise ArtifactValidationError("chain checkpoint descriptor is invalid")
        path = _descriptor_path(
            root,
            {"path": descriptor["path"], "sha256": descriptor["sha256"]},
            descriptor["path"],
            "chain_checkpoints.{}".format(chain_id),
        )
        checkpoint = load_mcmc_checkpoint(str(path))
        if (
            checkpoint.chain_id != chain_id
            or checkpoint.mode_id != selected_mode_id
            or checkpoint.completed_transition != descriptor["completed_transition"]
        ):
            raise ArtifactValidationError("chain checkpoint identity mismatch")
        chains[chain_id] = checkpoint
    return BatchEstimationCheckpoint(root, manifest, core, state_values, chains)


def save_batch_chain_checkpoint(
    checkpoint_root: Union[str, Path], value: McmcChainCheckpoint
) -> None:
    """Atomically advance one content-addressed chain and then its manifest."""

    if not isinstance(value, McmcChainCheckpoint):
        raise TypeError("value must be McmcChainCheckpoint")
    root = Path(checkpoint_root).expanduser().resolve()
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema") != BATCH_ESTIMATION_CHECKPOINT_SCHEMA:
        raise ArtifactValidationError("unsupported batch checkpoint schema")
    temporary_path = root / "mcmc" / ".{}.next.npz".format(value.chain_id)
    save_mcmc_checkpoint(str(temporary_path), value)
    digest = file_sha256(temporary_path)
    relative = "mcmc/{}-{}.npz".format(value.chain_id, digest[7:])
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        temporary_path.unlink()
    else:
        os.replace(str(temporary_path), str(destination))
    chains = dict(manifest["chain_checkpoints"])
    chains[value.chain_id] = {
        "path": relative,
        "sha256": digest,
        "completed_transition": value.completed_transition,
    }
    updated = dict(manifest)
    updated["status"] = "sampling"
    updated["chain_checkpoints"] = chains
    updated["cancellation_reason"] = ""
    write_json_atomic(root / "manifest.json", updated)


def mark_batch_checkpoint_cancelled(
    checkpoint_root: Union[str, Path], reason: str
) -> None:
    root = Path(checkpoint_root).expanduser().resolve()
    manifest = read_json(root / "manifest.json")
    updated = dict(manifest)
    updated["status"] = "cancelled"
    updated["cancellation_reason"] = _canonical(reason, "reason")
    write_json_atomic(root / "manifest.json", updated)


def mark_batch_checkpoint_published(checkpoint_root: Union[str, Path]) -> None:
    root = Path(checkpoint_root).expanduser().resolve()
    manifest = read_json(root / "manifest.json")
    updated = dict(manifest)
    updated["status"] = "published"
    updated["cancellation_reason"] = ""
    write_json_atomic(root / "manifest.json", updated)


__all__ = [
    "BATCH_ESTIMATION_CHECKPOINT_SCHEMA",
    "BatchEstimationCheckpoint",
    "batch_checkpoint_path",
    "load_batch_estimation_checkpoint",
    "mark_batch_checkpoint_cancelled",
    "mark_batch_checkpoint_published",
    "save_batch_chain_checkpoint",
    "write_batch_estimation_checkpoint",
]
