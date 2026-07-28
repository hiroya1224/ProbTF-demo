"""Atomic, content-addressed artifacts for posterior controller evaluation."""

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

from grape_param_estim.episode import stable_hash
from grape_param_estim.inference.posterior import PlantPosterior
from grape_param_estim.output.artifacts import (
    _publish_directory_no_replace,
    plain_data,
)
from grape_param_estim.validation.controller_design import (
    ControllerCandidate,
    ControllerDesignEvaluation,
    ControllerRecommendationBinding,
    ControllerRecommendationEvidence,
    ControllerRecommendationGates,
    VerifiedPlantArtifactIdentity,
    evaluate_controller_candidate,
)


CONTROLLER_EVALUATION_ARTIFACT_SCHEMA = (
    "grape_controller_evaluation_artifacts/v1"
)
CONTROLLER_EVALUATION_ARTIFACTS = (
    "controller_evaluation.json",
    "particle_outcomes.json",
    "artifact_manifest.json",
)


def _sha256(value: Any, name: str) -> str:
    digest = str(value).lower()
    if (
        len(digest) != 64
        or any(item not in "0123456789abcdef" for item in digest)
    ):
        raise ValueError("{} must be a lowercase SHA-256".format(name))
    return digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            plain_data(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class ControllerEvaluationProvenance:
    """Identity binding a controller evaluation to code, plant, and backend.

    For an allowed recommendation, the writer independently checks these
    fields against the v3 recommendation binding embedded in the evaluation.
    """

    source_commit: str
    plant_posterior_sha256: str
    controller_backend_id: str
    controller_backend_sha256: str
    config_sha256: str
    plant_artifact_manifest_sha256: Optional[str] = None
    plant_artifact_identity: Optional[
        VerifiedPlantArtifactIdentity
    ] = field(default=None, repr=False, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "grape_controller_evaluation_provenance/v1"

    def __post_init__(self) -> None:
        source_commit = str(self.source_commit).strip()
        backend_id = str(self.controller_backend_id).strip()
        schema = str(self.schema).strip()
        if not source_commit or not backend_id:
            raise ValueError(
                "source_commit and controller_backend_id are required"
            )
        if schema != "grape_controller_evaluation_provenance/v1":
            raise ValueError(
                "unsupported controller evaluation provenance schema"
            )
        object.__setattr__(
            self,
            "plant_posterior_sha256",
            _sha256(
                self.plant_posterior_sha256,
                "plant_posterior_sha256",
            ),
        )
        object.__setattr__(
            self,
            "controller_backend_sha256",
            _sha256(
                self.controller_backend_sha256,
                "controller_backend_sha256",
            ),
        )
        object.__setattr__(
            self,
            "config_sha256",
            _sha256(self.config_sha256, "config_sha256"),
        )
        plant_artifact_identity = self.plant_artifact_identity
        if plant_artifact_identity is not None:
            if (
                not isinstance(
                    plant_artifact_identity,
                    VerifiedPlantArtifactIdentity,
                )
                or not plant_artifact_identity.content_is_valid()
            ):
                raise TypeError(
                    "plant_artifact_identity must be a verified plant run"
                )
            plant_artifact_identity = VerifiedPlantArtifactIdentity(
                plant_artifact_identity.run_directory
            )
            if (
                plant_artifact_identity.posterior_content_sha256
                != self.plant_posterior_sha256
            ):
                raise ValueError(
                    "plant artifact/posterior provenance mismatch"
                )
            if (
                self.plant_artifact_manifest_sha256 is not None
                and self.plant_artifact_manifest_sha256
                != plant_artifact_identity.manifest_sha256
            ):
                raise ValueError(
                    "plant artifact manifest provenance mismatch"
                )
            object.__setattr__(
                self,
                "plant_artifact_manifest_sha256",
                plant_artifact_identity.manifest_sha256,
            )
            object.__setattr__(
                self,
                "plant_artifact_identity",
                plant_artifact_identity,
            )
        elif self.plant_artifact_manifest_sha256 is not None:
            object.__setattr__(
                self,
                "plant_artifact_manifest_sha256",
                _sha256(
                    self.plant_artifact_manifest_sha256,
                    "plant_artifact_manifest_sha256",
                ),
            )
        object.__setattr__(self, "source_commit", source_commit)
        object.__setattr__(self, "controller_backend_id", backend_id)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "source_commit": self.source_commit,
            "plant_posterior_sha256": self.plant_posterior_sha256,
            "controller_backend_id": self.controller_backend_id,
            "controller_backend_sha256": (
                self.controller_backend_sha256
            ),
            "config_sha256": self.config_sha256,
            "plant_artifact_manifest_sha256": (
                self.plant_artifact_manifest_sha256
            ),
            "plant_artifact_identity": (
                None
                if self.plant_artifact_identity is None
                else self.plant_artifact_identity.to_mapping()
            ),
            "metadata": plain_data(self.metadata),
        }

    @property
    def content_sha256(self) -> str:
        return stable_hash(self.to_mapping())


@dataclass(frozen=True)
class ControllerEvaluationRun:
    evaluation: ControllerDesignEvaluation
    artifact_directory: Path

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation, ControllerDesignEvaluation):
            raise TypeError("evaluation must be ControllerDesignEvaluation")
        object.__setattr__(
            self,
            "artifact_directory",
            Path(self.artifact_directory).resolve(),
        )


class ControllerEvaluationArtifactWriter:
    """Write one non-overwriting controller-evaluation bundle atomically."""

    def __init__(self, output_root: Any) -> None:
        self.output_root = Path(output_root).expanduser().resolve()

    def write(
        self,
        *,
        run_id: str,
        evaluation: ControllerDesignEvaluation,
        provenance: ControllerEvaluationProvenance,
    ) -> Path:
        if not isinstance(evaluation, ControllerDesignEvaluation):
            raise TypeError("evaluation must be ControllerDesignEvaluation")
        if not isinstance(provenance, ControllerEvaluationProvenance):
            raise TypeError(
                "provenance must be ControllerEvaluationProvenance"
            )
        if (
            provenance.plant_posterior_sha256
            != evaluation.plant_posterior_sha256
        ):
            raise ValueError(
                "controller evaluation/plant posterior provenance mismatch"
            )
        if evaluation.recommendation_allowed and (
            not evaluation.gates.passed
            or not evaluation.gates.evidence_bound
        ):
            raise ValueError(
                "recommendation cannot bypass evidence-bound gates"
            )
        if evaluation.recommendation_allowed:
            self._validate_recommendation_provenance(
                evaluation,
                provenance,
            )
        identifier = str(run_id)
        if (
            not identifier
            or identifier in (".", "..")
            or "/" in identifier
            or os.sep in identifier
        ):
            raise ValueError("run_id must be one safe path component")

        self.output_root.mkdir(parents=True, exist_ok=True)
        destination = self.output_root / identifier
        if destination.exists():
            raise FileExistsError(str(destination))
        staging = Path(
            tempfile.mkdtemp(
                prefix=".{}.staging.".format(identifier),
                dir=str(self.output_root),
            )
        )
        try:
            self._write_payloads(
                staging,
                identifier,
                evaluation,
                provenance,
            )
            _publish_directory_no_replace(staging, destination)
        except Exception:
            shutil.rmtree(str(staging), ignore_errors=True)
            raise
        return destination

    @staticmethod
    def _validate_recommendation_provenance(
        evaluation: ControllerDesignEvaluation,
        provenance: ControllerEvaluationProvenance,
    ) -> None:
        evidence = evaluation.gates.evidence
        binding = (
            None
            if evidence is None
            else evidence.binding
        )
        if (
            not isinstance(binding, ControllerRecommendationBinding)
            or not binding.content_is_valid()
        ):
            raise ValueError(
                "allowed recommendation lacks a valid v3 context binding"
            )
        if (
            binding.candidate_sha256
            != evaluation.candidate.content_sha256
            or binding.plant_posterior_sha256
            != evaluation.plant_posterior_sha256
            or evaluation.gates.evaluation_context_sha256
            != binding.evaluation_context_sha256
        ):
            raise ValueError(
                "allowed recommendation evaluation context mismatch"
            )
        if (
            provenance.plant_artifact_identity is None
            or not provenance.plant_artifact_identity.content_is_valid()
            or provenance.plant_artifact_identity.to_mapping()
            != binding.plant_artifact_identity.to_mapping()
        ):
            raise ValueError(
                "allowed recommendation plant artifact provenance mismatch"
            )
        identity = binding.exact_controller_identity
        evaluator_identity = binding.evaluator_identity
        if (
            provenance.controller_backend_id != identity.backend_id
            or provenance.controller_backend_sha256
            != identity.artifact_sha256
        ):
            raise ValueError(
                "allowed recommendation controller provenance mismatch"
            )
        if (
            provenance.config_sha256
            != evaluator_identity.evaluation_config_sha256
        ):
            raise ValueError(
                "allowed recommendation evaluator config provenance mismatch"
            )

    @staticmethod
    def _write_payloads(
        directory: Path,
        run_id: str,
        evaluation: ControllerDesignEvaluation,
        provenance: ControllerEvaluationProvenance,
    ) -> None:
        evaluation_mapping = dict(evaluation.to_mapping())
        particle_outcomes = evaluation_mapping.pop("particle_outcomes")
        summary = {
            "schema": CONTROLLER_EVALUATION_ARTIFACT_SCHEMA,
            "run_id": run_id,
            "evaluation_sha256": evaluation.content_sha256,
            "provenance": provenance.to_mapping(),
            "provenance_sha256": provenance.content_sha256,
            "evaluation": evaluation_mapping,
        }
        particle_payload = {
            "schema": CONTROLLER_EVALUATION_ARTIFACT_SCHEMA,
            "run_id": run_id,
            "evaluation_sha256": evaluation.content_sha256,
            "plant_posterior_sha256": (
                evaluation.plant_posterior_sha256
            ),
            "candidate_sha256": evaluation.candidate.content_sha256,
            "particle_outcomes": particle_outcomes,
        }
        _write_json(directory / "controller_evaluation.json", summary)
        _write_json(directory / "particle_outcomes.json", particle_payload)

        files = {
            name: {
                "sha256": _sha256_file(directory / name),
                "bytes": int((directory / name).stat().st_size),
            }
            for name in CONTROLLER_EVALUATION_ARTIFACTS
            if name != "artifact_manifest.json"
        }
        manifest_without_hash = {
            "schema": CONTROLLER_EVALUATION_ARTIFACT_SCHEMA,
            "run_id": run_id,
            "evaluation_sha256": evaluation.content_sha256,
            "provenance_sha256": provenance.content_sha256,
            "files": files,
        }
        manifest = dict(manifest_without_hash)
        manifest["manifest_sha256"] = stable_hash(
            manifest_without_hash
        )
        _write_json(directory / "artifact_manifest.json", manifest)


def evaluate_and_write_controller_candidate(
    *,
    output_root: Any,
    run_id: str,
    candidate: ControllerCandidate,
    plant_posterior: PlantPosterior,
    particle_evaluator: Callable[[ControllerCandidate, Any], Any],
    evidence: ControllerRecommendationEvidence,
    recommendation_threshold: float,
    provenance: ControllerEvaluationProvenance,
) -> ControllerEvaluationRun:
    """Production boundary for an evidence-gated Phase 8 candidate run."""

    if not isinstance(evidence, ControllerRecommendationEvidence):
        raise TypeError(
            "evidence must be ControllerRecommendationEvidence"
        )
    gates = ControllerRecommendationGates.from_evidence(evidence)
    evaluation = evaluate_controller_candidate(
        candidate,
        plant_posterior,
        particle_evaluator,
        gates,
        recommendation_threshold=recommendation_threshold,
    )
    destination = ControllerEvaluationArtifactWriter(output_root).write(
        run_id=run_id,
        evaluation=evaluation,
        provenance=provenance,
    )
    return ControllerEvaluationRun(
        evaluation=evaluation,
        artifact_directory=destination,
    )


__all__ = [
    "CONTROLLER_EVALUATION_ARTIFACT_SCHEMA",
    "CONTROLLER_EVALUATION_ARTIFACTS",
    "ControllerEvaluationArtifactWriter",
    "ControllerEvaluationProvenance",
    "ControllerEvaluationRun",
    "evaluate_and_write_controller_candidate",
]
