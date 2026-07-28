"""Complete, hashed, non-overwriting plant posterior artifact bundles."""

from dataclasses import dataclass, fields, is_dataclass
import ctypes
import csv
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.episode import stable_hash
from grape_param_estim.inference.posterior import PlantPosterior


ARTIFACT_SCHEMA = "grape_plant_assimilation_artifacts/v2"
ARTIFACT_PROVENANCE_KEY = "artifact_provenance"
REQUIRED_ARTIFACTS = (
    "run_manifest.json",
    "controller_snapshot.json",
    "controller_replay_audit.json",
    "factual_replay_report.json",
    "posterior_particles.npz",
    "posterior_summary.json",
    "posterior_hpd95.csv",
    "identifiability_report.json",
    "likelihood_components.csv",
    "posterior_predictive.npz",
    "failure_validation.json",
    "success_validation.json",
    "REPORT.md",
)

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish ``source`` without replacing a concurrent writer.

    Plain POSIX ``rename`` may replace an empty destination directory.  That
    leaves a TOCTOU window between the early existence check and publication.
    Linux ``renameat2(..., RENAME_NOREPLACE)`` closes that window in one
    filesystem operation.  Fail closed when the primitive is unavailable;
    falling back to a check followed by ``rename`` would recreate the race.
    """

    if os.name == "nt":
        # Windows rename already fails when the destination exists.
        os.rename(str(source), str(destination))
        return

    library = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = library.renameat2
    except AttributeError as exc:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unavailable",
            str(destination),
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            errno.EEXIST,
            os.strerror(errno.EEXIST),
            str(destination),
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


def _digest(value: Any, name: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(
        item not in "0123456789abcdef" for item in text
    ):
        raise ValueError("{} must be a lowercase SHA-256".format(name))
    return text


def plain_data(value: Any) -> Any:
    """Return a JSON-safe tree without deep-copying immutable containers.

    ``dataclasses.asdict`` uses ``copy.deepcopy`` for leaf values.  Several
    public contracts intentionally freeze metadata with ``MappingProxyType``,
    which is not pickleable and therefore cannot be processed by ``asdict``.
    Walking the declared fields also lets us normalize non-finite Python and
    NumPy floats before ``json.dumps(..., allow_nan=False)`` sees them.
    """

    if is_dataclass(value):
        return {
            item.name: plain_data(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): plain_data(item) for key, item in value.items()
        }
    if isinstance(value, np.ndarray):
        return plain_data(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if np.isfinite(result):
            return result
        if np.isnan(result):
            return "NaN"
        return "Infinity" if result > 0.0 else "-Infinity"
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (tuple, list)):
        return [plain_data(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


# Backward-compatible private spelling for callers within this module.
_plain = plain_data


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            _plain(value),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _artifact_provenance(
    posterior: PlantPosterior,
    provenance: "PlantRunProvenance",
) -> Mapping[str, Any]:
    """Return the provenance block embedded literally in every artifact."""

    result = dict(_plain(provenance))
    result["model_id"] = str(posterior.model_id)
    return result


def _with_artifact_provenance(
    value: Any,
    artifact_provenance: Mapping[str, Any],
    artifact_name: str,
) -> Mapping[str, Any]:
    payload = _plain(value)
    if isinstance(payload, Mapping):
        result = dict(payload)
    else:
        # Keep permissive ``Any`` inputs serializable while giving the
        # provenance block a stable location.
        result = {"payload": payload}
    if (
        ARTIFACT_PROVENANCE_KEY in result
        and result[ARTIFACT_PROVENANCE_KEY] != artifact_provenance
    ):
        raise ValueError(
            "{} contains conflicting artifact provenance".format(
                artifact_name
            )
        )
    result[ARTIFACT_PROVENANCE_KEY] = dict(artifact_provenance)
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _npz_provenance_arrays(
    artifact_provenance: Mapping[str, Any],
) -> Mapping[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {
        "{}_json".format(ARTIFACT_PROVENANCE_KEY): np.asarray(
            _canonical_json(artifact_provenance)
        )
    }
    for key, value in artifact_provenance.items():
        name = "{}_{}".format(ARTIFACT_PROVENANCE_KEY, key)
        if isinstance(value, (tuple, list)):
            arrays[name] = np.asarray(value, dtype=np.str_)
        elif value is None:
            arrays[name] = np.asarray("null")
        elif isinstance(value, (int, np.integer)):
            arrays[name] = np.asarray(int(value), dtype=np.int64)
        else:
            arrays[name] = np.asarray(str(value))
    return arrays


def _csv_provenance(
    artifact_provenance: Mapping[str, Any],
) -> Tuple[Tuple[str, ...], Mapping[str, Any]]:
    fields = tuple(
        "{}_{}".format(ARTIFACT_PROVENANCE_KEY, key)
        for key in artifact_provenance
    )
    values = {}
    for key, value in artifact_provenance.items():
        field = "{}_{}".format(ARTIFACT_PROVENANCE_KEY, key)
        if isinstance(value, (tuple, list, Mapping)) or value is None:
            values[field] = _canonical_json(value)
        else:
            values[field] = value
    return fields, values


@dataclass(frozen=True)
class PlantRunProvenance:
    source_commit: str
    source_bag_sha256: Tuple[str, ...]
    normalized_episode_sha256: Tuple[str, ...]
    controller_snapshot_sha256: str
    controller_artifact_sha256: str
    plant_backend_id: str
    plant_backend_sha256: str
    plant_geometry_profile_id: str
    plant_geometry_sha256: str
    prior_id: str
    likelihood_id: str
    seed: int
    config_sha256: str
    fixture_sha256: str
    actuator_calibration_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if not str(self.source_commit):
            raise ValueError("source_commit is required")
        bags = tuple(
            _digest(item, "source_bag_sha256")
            for item in self.source_bag_sha256
        )
        episodes = tuple(
            _digest(item, "normalized_episode_sha256")
            for item in self.normalized_episode_sha256
        )
        if not bags or len(bags) != len(episodes):
            raise ValueError("bag and normalized episode hashes must align")
        for name in (
            "controller_snapshot_sha256",
            "controller_artifact_sha256",
            "plant_backend_sha256",
            "plant_geometry_sha256",
            "config_sha256",
            "fixture_sha256",
        ):
            object.__setattr__(
                self, name, _digest(getattr(self, name), name)
            )
        if self.actuator_calibration_sha256 is not None:
            object.__setattr__(
                self,
                "actuator_calibration_sha256",
                _digest(
                    self.actuator_calibration_sha256,
                    "actuator_calibration_sha256",
                ),
            )
        for name in (
            "plant_backend_id",
            "plant_geometry_profile_id",
            "prior_id",
            "likelihood_id",
        ):
            if not str(getattr(self, name)):
                raise ValueError("{} is required".format(name))
        object.__setattr__(self, "source_bag_sha256", bags)
        object.__setattr__(self, "normalized_episode_sha256", episodes)


class PlantAssimilationArtifactWriter:
    """Write the Definition-of-Done artifact set as one atomic directory."""

    def __init__(self, output_root: Any) -> None:
        self.output_root = Path(output_root).expanduser().resolve()

    def write(
        self,
        *,
        run_id: str,
        posterior: PlantPosterior,
        provenance: PlantRunProvenance,
        controller_snapshot: Any,
        controller_replay_audit: Any,
        factual_replay_report: Any,
        identifiability_report: Any,
        likelihood_components: Sequence[Any],
        posterior_predictive: Mapping[str, Any],
        failure_validation: Any,
        success_validation: Any,
        interpretation: str,
    ) -> Path:
        if not isinstance(posterior, PlantPosterior):
            raise TypeError("posterior must be PlantPosterior")
        if not isinstance(provenance, PlantRunProvenance):
            raise TypeError("provenance must be PlantRunProvenance")
        identifier = str(run_id)
        if (
            not identifier
            or identifier in (".", "..")
            or "/" in identifier
            or os.sep in identifier
        ):
            raise ValueError("run_id must be one safe path component")
        if posterior.prior_id != provenance.prior_id:
            raise ValueError("posterior/prior provenance mismatch")
        if posterior.likelihood_id != provenance.likelihood_id:
            raise ValueError("posterior/likelihood provenance mismatch")
        if posterior.controller_snapshot_id != provenance.controller_snapshot_sha256:
            raise ValueError("posterior/controller snapshot provenance mismatch")
        expected_interpretation = (
            "calibrated_physical_plant_posterior"
            if provenance.actuator_calibration_sha256 is not None
            else "effective_plant_posterior"
        )
        if str(interpretation) != expected_interpretation:
            raise ValueError(
                "posterior interpretation does not match actuator calibration gate"
            )

        destination = self.output_root / identifier
        if destination.exists():
            raise FileExistsError(str(destination))
        self.output_root.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=".{}.staging.".format(identifier),
                dir=str(self.output_root),
            )
        )
        try:
            self._write_payloads(
                staging=staging,
                posterior=posterior,
                provenance=provenance,
                controller_snapshot=controller_snapshot,
                controller_replay_audit=controller_replay_audit,
                factual_replay_report=factual_replay_report,
                identifiability_report=identifiability_report,
                likelihood_components=likelihood_components,
                posterior_predictive=posterior_predictive,
                failure_validation=failure_validation,
                success_validation=success_validation,
                interpretation=expected_interpretation,
            )
            _publish_directory_no_replace(staging, destination)
        except Exception:
            shutil.rmtree(str(staging), ignore_errors=True)
            raise
        return destination

    @staticmethod
    def _write_payloads(
        *,
        staging: Path,
        posterior: PlantPosterior,
        provenance: PlantRunProvenance,
        controller_snapshot: Any,
        controller_replay_audit: Any,
        factual_replay_report: Any,
        identifiability_report: Any,
        likelihood_components: Sequence[Any],
        posterior_predictive: Mapping[str, Any],
        failure_validation: Any,
        success_validation: Any,
        interpretation: str,
    ) -> None:
        artifact_provenance = _artifact_provenance(posterior, provenance)
        _write_json(
            staging / "controller_snapshot.json",
            _with_artifact_provenance(
                controller_snapshot,
                artifact_provenance,
                "controller_snapshot.json",
            ),
        )
        _write_json(
            staging / "controller_replay_audit.json",
            _with_artifact_provenance(
                controller_replay_audit,
                artifact_provenance,
                "controller_replay_audit.json",
            ),
        )
        _write_json(
            staging / "factual_replay_report.json",
            _with_artifact_provenance(
                factual_replay_report,
                artifact_provenance,
                "factual_replay_report.json",
            ),
        )
        _write_json(
            staging / "identifiability_report.json",
            _with_artifact_provenance(
                identifiability_report,
                artifact_provenance,
                "identifiability_report.json",
            ),
        )
        _write_json(
            staging / "failure_validation.json",
            _with_artifact_provenance(
                failure_validation,
                artifact_provenance,
                "failure_validation.json",
            ),
        )
        _write_json(
            staging / "success_validation.json",
            _with_artifact_provenance(
                success_validation,
                artifact_provenance,
                "success_validation.json",
            ),
        )

        particle_model_ids = np.asarray(
            [item.model_id for item in posterior.particles], dtype=np.str_
        )
        provenance_arrays = _npz_provenance_arrays(artifact_provenance)
        np.savez_compressed(
            str(staging / "posterior_particles.npz"),
            schema=np.asarray(ARTIFACT_SCHEMA),
            weights=posterior.weights,
            log_likelihood=posterior.log_likelihood,
            raw_parameters=posterior.raw_parameters,
            derived_parameters=posterior.derived_parameters,
            raw_parameter_names=np.asarray(
                posterior.raw_parameter_names, dtype=np.str_
            ),
            derived_parameter_names=np.asarray(
                posterior.derived_parameter_names, dtype=np.str_
            ),
            particle_model_id=particle_model_ids,
            posterior_model_id=np.asarray(posterior.model_id),
            posterior_content_sha256=np.asarray(
                posterior.content_sha256
            ),
            **provenance_arrays
        )
        summary = {
            "schema": ARTIFACT_SCHEMA,
            "interpretation": interpretation,
            "model_id": posterior.model_id,
            "prior_id": posterior.prior_id,
            "likelihood_id": posterior.likelihood_id,
            "controller_snapshot_id": posterior.controller_snapshot_id,
            "particle_count": len(posterior.particles),
            "effective_sample_size": posterior.effective_sample_size,
            "credible_probability": posterior.credible_probability,
            "hpd_particle_count": int(posterior.hpd_indices.size),
            "hpd_weight": posterior.hpd_weight,
            "parameter_names": posterior.raw_parameter_names,
            "mean": posterior.mean,
            "covariance": posterior.covariance,
            "correlation": posterior.correlation,
            "multimodality": posterior.multimodality_diagnostic(),
            "posterior_content_sha256": posterior.content_sha256,
            "credible_set_statement": (
                "Credible under the recorded prior, model, likelihood, "
                "controller snapshot, and source bags; not an objective "
                "frequentist coverage claim."
            ),
            ARTIFACT_PROVENANCE_KEY: artifact_provenance,
        }
        _write_json(staging / "posterior_summary.json", summary)

        provenance_fields, provenance_row = _csv_provenance(
            artifact_provenance
        )
        hpd_fields = (
            ("particle_index", "normalized_weight", "log_likelihood")
            + posterior.raw_parameter_names
            + posterior.derived_parameter_names
            + provenance_fields
        )
        with (staging / "posterior_hpd95.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=hpd_fields)
            writer.writeheader()
            for index in posterior.hpd_indices:
                row = {
                    "particle_index": int(index),
                    "normalized_weight": float(posterior.weights[index]),
                    "log_likelihood": float(
                        posterior.log_likelihood[index]
                    ),
                }
                row.update(
                    zip(
                        posterior.raw_parameter_names,
                        posterior.raw_parameters[index].tolist(),
                    )
                )
                row.update(
                    zip(
                        posterior.derived_parameter_names,
                        posterior.derived_parameters[index].tolist(),
                    )
                )
                row.update(provenance_row)
                writer.writerow(
                    row
                )

        component_rows = [_plain(item) for item in likelihood_components]
        fieldnames = (
            sorted(
                {
                    key
                    for row in component_rows
                    if isinstance(row, Mapping)
                    for key in row.keys()
                    if key != "diagnostics"
                }
            )
            if component_rows
            else ["episode_id", "total"]
        )
        reserved_csv_fields = set(fieldnames).intersection(
            provenance_fields
        )
        if reserved_csv_fields:
            raise ValueError(
                "likelihood component rows use reserved provenance fields: "
                + ", ".join(sorted(reserved_csv_fields))
            )
        likelihood_fields = tuple(fieldnames) + provenance_fields
        with (staging / "likelihood_components.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=likelihood_fields)
            writer.writeheader()
            for row in component_rows:
                output_row = {
                    key: row.get(key, "") for key in fieldnames
                }
                output_row.update(provenance_row)
                writer.writerow(output_row)
            if not component_rows:
                # A header alone cannot carry literal provenance values.
                # Preserve the empty semantic fields and emit one explicit
                # provenance-only row.
                writer.writerow(provenance_row)

        predictive_arrays: Dict[str, np.ndarray] = {
            "schema": np.asarray(ARTIFACT_SCHEMA)
        }
        for key, value in posterior_predictive.items():
            array = np.asarray(value)
            if array.dtype == object:
                raise ValueError(
                    "posterior predictive arrays may not require pickle"
                )
            predictive_arrays[str(key)] = array
        reserved_predictive = set(predictive_arrays).intersection(
            provenance_arrays
        )
        if reserved_predictive:
            raise ValueError(
                "posterior predictive uses reserved provenance arrays: "
                + ", ".join(sorted(reserved_predictive))
            )
        predictive_arrays.update(provenance_arrays)
        np.savez_compressed(
            str(staging / "posterior_predictive.npz"),
            **predictive_arrays
        )

        report = [
            "# Grape plant assimilation report",
            "",
            "## Artifact provenance",
            "",
            "```json",
            json.dumps(
                artifact_provenance,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ),
            "```",
            "",
            "## Posterior summary",
            "",
            "- Interpretation: `{}`".format(interpretation),
            "- Plant posterior model: `{}`".format(posterior.model_id),
            "- Plant backend: `{}`".format(provenance.plant_backend_id),
            "- Plant backend SHA-256: `{}`".format(
                provenance.plant_backend_sha256
            ),
            "- Plant geometry profile: `{}` (`{}`)".format(
                provenance.plant_geometry_profile_id,
                provenance.plant_geometry_sha256,
            ),
            "- Controller snapshot: `{}`".format(
                posterior.controller_snapshot_id
            ),
            "- Weighted particles: `{}`".format(len(posterior.particles)),
            "- Joint {:.1f}% HPD particle subset: `{}` particles, weight `{:.6f}`".format(
                100.0 * posterior.credible_probability,
                posterior.hpd_indices.size,
                posterior.hpd_weight,
            ),
            "- Effective sample size: `{:.3f}`".format(
                posterior.effective_sample_size
            ),
            "",
            "The complete weighted empirical law in `posterior_particles.npz` "
            "is the authoritative posterior. Marginal intervals are not used "
            "as a replacement for joint particles, correlations, or modes.",
            "",
            "Controller recommendations remain unavailable unless exactness, "
            "actuator calibration, support, probability calibration, and "
            "held-out failure and success gates all pass.",
            "",
        ]
        (staging / "REPORT.md").write_text(
            "\n".join(report), encoding="utf-8"
        )

        payload_names = tuple(
            name for name in REQUIRED_ARTIFACTS if name != "run_manifest.json"
        )
        files = {
            name: {
                "sha256": _sha256_file(staging / name),
                "bytes": int((staging / name).stat().st_size),
            }
            for name in payload_names
        }
        manifest_without_hash = {
            "schema": ARTIFACT_SCHEMA,
            ARTIFACT_PROVENANCE_KEY: artifact_provenance,
            "provenance": _plain(provenance),
            "posterior_particles_sha256": files[
                "posterior_particles.npz"
            ]["sha256"],
            "posterior_content_sha256": posterior.content_sha256,
            "files": files,
        }
        manifest = dict(manifest_without_hash)
        manifest["manifest_sha256"] = stable_hash(manifest_without_hash)
        _write_json(staging / "run_manifest.json", manifest)


__all__ = [
    "ARTIFACT_SCHEMA",
    "PlantAssimilationArtifactWriter",
    "PlantRunProvenance",
    "REQUIRED_ARTIFACTS",
    "plain_data",
]
