"""Artifact-manifest compatibility and integrity verification."""

from dataclasses import fields
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np

from grape_param_estim.episode import stable_hash
from grape_param_estim.output.artifacts import (
    ARTIFACT_SCHEMA,
    PlantRunProvenance,
    REQUIRED_ARTIFACTS,
    plain_data,
)


_RUN_MANIFEST_NAME = "run_manifest.json"
_REQUIRED_DISK_FILES = frozenset(REQUIRED_ARTIFACTS) | {
    _RUN_MANIFEST_NAME
}
_REQUIRED_MANIFEST_FILES = _REQUIRED_DISK_FILES - {
    _RUN_MANIFEST_NAME
}
_REQUIRED_MANIFEST_KEYS = frozenset(
    (
        "schema",
        "artifact_provenance",
        "provenance",
        "posterior_particles_sha256",
        "posterior_content_sha256",
        "files",
        "manifest_sha256",
    )
)
_REQUIRED_PROVENANCE_KEYS = frozenset(
    item.name for item in fields(PlantRunProvenance)
)


def _set_mismatch(
    label: str, expected: frozenset, actual: frozenset
) -> ValueError:
    details = []
    missing = tuple(sorted(expected - actual))
    extra = tuple(sorted(actual - expected))
    if missing:
        details.append("missing {}".format(", ".join(missing)))
    if extra:
        details.append("extra {}".format(", ".join(extra)))
    return ValueError(
        "{} mismatch ({})".format(label, "; ".join(details))
    )


def _sha256(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("{} must be a lowercase SHA-256".format(name))
    return value


def _validated_provenance(
    value: Any,
) -> Tuple[PlantRunProvenance, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("run manifest provenance must be an object")
    keys = frozenset(str(name) for name in value)
    if keys != _REQUIRED_PROVENANCE_KEYS:
        raise _set_mismatch(
            "run manifest provenance fields",
            _REQUIRED_PROVENANCE_KEYS,
            keys,
        )
    for name in (
        "source_commit",
        "plant_backend_id",
        "plant_geometry_profile_id",
        "prior_id",
        "likelihood_id",
    ):
        if type(value.get(name)) is not str or not value[name]:
            raise ValueError(
                "run manifest provenance {} must be a non-empty string".format(
                    name
                )
            )
    if type(value.get("seed")) is not int:
        raise ValueError("run manifest provenance seed must be an integer")
    try:
        provenance = PlantRunProvenance(**dict(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid run manifest provenance") from exc
    canonical = plain_data(provenance)
    if dict(value) != canonical:
        raise ValueError("run manifest provenance is not canonical")
    return provenance, canonical


def _validated_artifact_provenance(
    value: Any,
    canonical_provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("run manifest artifact_provenance must be an object")
    model_id = value.get("model_id")
    if type(model_id) is not str or not model_id:
        raise ValueError(
            "run manifest artifact_provenance model_id must be a "
            "non-empty string"
        )
    expected = dict(canonical_provenance)
    expected["model_id"] = model_id
    if dict(value) != expected:
        raise ValueError(
            "run manifest artifact_provenance does not match provenance"
        )
    return expected


def _npz_scalar_text(archive: Any, name: str) -> str:
    if name not in archive.files:
        raise ValueError(
            "posterior_particles.npz lacks {}".format(name)
        )
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(
            "posterior_particles.npz {} must be a scalar".format(name)
        )
    scalar = value.item()
    if type(scalar) is not str:
        raise ValueError(
            "posterior_particles.npz {} must be text".format(name)
        )
    return scalar


def verify_run_manifest(run_directory: Any) -> Mapping[str, Any]:
    directory = Path(run_directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("run directory must be an existing directory")
    entries = tuple(directory.iterdir())
    disk_files = frozenset(item.name for item in entries)
    if disk_files != _REQUIRED_DISK_FILES:
        raise _set_mismatch(
            "run artifact file set",
            _REQUIRED_DISK_FILES,
            disk_files,
        )
    non_regular = tuple(
        sorted(
            item.name
            for item in entries
            if not item.is_file() or item.is_symlink()
        )
    )
    if non_regular:
        raise ValueError(
            "run artifacts must be regular files: {}".format(
                ", ".join(non_regular)
            )
        )

    manifest_path = directory / _RUN_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("run manifest must be a JSON object")
    manifest_keys = frozenset(str(name) for name in manifest)
    if manifest_keys != _REQUIRED_MANIFEST_KEYS:
        raise _set_mismatch(
            "run manifest top-level fields",
            _REQUIRED_MANIFEST_KEYS,
            manifest_keys,
        )
    if manifest.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("unsupported run manifest schema")
    content = dict(manifest)
    expected = _sha256(
        content.pop("manifest_sha256"), "manifest_sha256"
    )
    if stable_hash(content) != expected:
        raise ValueError("run manifest content hash mismatch")

    _, canonical_provenance = _validated_provenance(
        manifest["provenance"]
    )
    artifact_provenance = _validated_artifact_provenance(
        manifest["artifact_provenance"],
        canonical_provenance,
    )
    posterior_particles_sha256 = _sha256(
        manifest["posterior_particles_sha256"],
        "posterior_particles_sha256",
    )
    posterior_content_sha256 = _sha256(
        manifest["posterior_content_sha256"],
        "posterior_content_sha256",
    )

    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("run manifest files must be an object")
    manifest_files = frozenset(str(name) for name in files)
    if manifest_files != _REQUIRED_MANIFEST_FILES:
        raise _set_mismatch(
            "run manifest files",
            _REQUIRED_MANIFEST_FILES,
            manifest_files,
        )
    for name in sorted(_REQUIRED_MANIFEST_FILES):
        entry = files[name]
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"sha256", "bytes"}
            or type(entry.get("bytes")) is not int
            or entry["bytes"] < 0
        ):
            raise ValueError(
                "invalid run manifest file entry: {}".format(name)
            )
        entry_digest = _sha256(
            entry.get("sha256"),
            "{} file sha256".format(name),
        )
        path = directory / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            digest != entry_digest
            or path.stat().st_size != entry["bytes"]
        ):
            raise ValueError("artifact hash/size mismatch: {}".format(name))
    particle_file_digest = files["posterior_particles.npz"]["sha256"]
    if posterior_particles_sha256 != particle_file_digest:
        raise ValueError(
            "posterior_particles_sha256 does not match its files entry"
        )

    try:
        with np.load(
            str(directory / "posterior_particles.npz"),
            allow_pickle=False,
        ) as archive:
            embedded_content_sha256 = _sha256(
                _npz_scalar_text(
                    archive, "posterior_content_sha256"
                ),
                "embedded posterior_content_sha256",
            )
            embedded_provenance_text = _npz_scalar_text(
                archive, "artifact_provenance_json"
            )
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(
            "posterior_particles.npz"
        ):
            raise
        raise ValueError(
            "posterior_particles.npz identity could not be verified"
        ) from exc
    if embedded_content_sha256 != posterior_content_sha256:
        raise ValueError(
            "posterior_content_sha256 does not match posterior_particles.npz"
        )
    try:
        embedded_provenance = json.loads(embedded_provenance_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "posterior_particles.npz artifact_provenance_json is invalid"
        ) from exc
    if embedded_provenance != artifact_provenance:
        raise ValueError(
            "artifact_provenance does not match posterior_particles.npz"
        )
    return manifest


__all__ = ["verify_run_manifest"]
