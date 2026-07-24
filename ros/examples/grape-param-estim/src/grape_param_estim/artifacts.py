"""Immutable, human-review-only analysis artifact bundles.

The writer intentionally has no API for updating vehicle parameters.  It
creates a new run directory atomically and refuses to overwrite an existing
run, keeping source-bag and normalized-dataset provenance beside every
candidate summary and trajectory particle archive.
"""

from dataclasses import asdict, dataclass, is_dataclass
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .counterfactual import CounterfactualResult, SUPPORTED
from .episode import stable_hash


ARTIFACT_SCHEMA = "grape_counterfactual_artifacts/v1"
MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
EXPERIMENTAL = "EXPERIMENTAL"
_AXES = ("x", "y", "z", "roll", "pitch", "yaw")
CONTROLLER_PARAMETER_NAMES = tuple(
    "{}.{}".format(term, axis)
    for term in ("p_gain", "i_gain", "d_gain")
    for axis in _AXES
) + (
    "controller_mass",
    "controller_inertia.x",
    "controller_inertia.y",
    "controller_inertia.z",
) + tuple("allocation_scale.{}".format(axis) for axis in _AXES) + (
    "thrust_scale",
    "delay_compensation_s",
)


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validate_digest(value: str, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("{} must be a lowercase SHA-256 digest".format(name))
    return digest


@dataclass(frozen=True)
class ArtifactProvenance:
    source_bag_sha256: Tuple[str, ...]
    normalized_dataset_sha256: Tuple[str, ...]
    source_topics: Tuple[str, ...]
    interval_start_s: float
    interval_end_s: float
    config_sha256: str
    source_commit: str
    model_version: str
    seed: int
    analysis_mode: str
    prefix_cutoff_s: Optional[float] = None

    def __post_init__(self) -> None:
        source_hashes = tuple(
            _validate_digest(item, "source_bag_sha256")
            for item in self.source_bag_sha256
        )
        dataset_hashes = tuple(
            _validate_digest(item, "normalized_dataset_sha256")
            for item in self.normalized_dataset_sha256
        )
        if not source_hashes or not dataset_hashes:
            raise ValueError("source and normalized dataset hashes must not be empty")
        start = float(self.interval_start_s)
        end = float(self.interval_end_s)
        if not np.isfinite(start) or not np.isfinite(end) or end < start:
            raise ValueError("artifact interval must have finite start <= end")
        if not self.source_topics or any(not str(item) for item in self.source_topics):
            raise ValueError("source_topics must not be empty")
        if not self.source_commit or not self.model_version:
            raise ValueError("source_commit and model_version must not be empty")
        if self.analysis_mode not in ("retrospective", "online_prefix"):
            raise ValueError("analysis_mode must be retrospective or online_prefix")
        cutoff = self.prefix_cutoff_s
        if self.analysis_mode == "online_prefix":
            if cutoff is None or not np.isfinite(float(cutoff)):
                raise ValueError("online-prefix artifacts require prefix_cutoff_s")
            if end > float(cutoff):
                raise ValueError(
                    "online-prefix source interval must end at or before "
                    "prefix_cutoff_s"
                )
        object.__setattr__(self, "source_bag_sha256", source_hashes)
        object.__setattr__(self, "normalized_dataset_sha256", dataset_hashes)
        object.__setattr__(
            self, "source_topics", tuple(str(item) for item in self.source_topics)
        )
        object.__setattr__(self, "interval_start_s", start)
        object.__setattr__(self, "interval_end_s", end)
        object.__setattr__(
            self,
            "config_sha256",
            _validate_digest(self.config_sha256, "config_sha256"),
        )
        object.__setattr__(self, "seed", int(self.seed))
        if cutoff is not None:
            object.__setattr__(self, "prefix_cutoff_s", float(cutoff))


@dataclass(frozen=True)
class AnalysisBagRecord:
    """One application message to merge into a derived analysis bag."""

    topic: str
    message: Any
    record_time_ns: int

    def __post_init__(self) -> None:
        topic = str(self.topic)
        stamp = int(self.record_time_ns)
        if not topic.startswith("/analysis/") or not topic.strip("/"):
            raise ValueError("analysis topics must be below /analysis/")
        if self.message is None or stamp < 0:
            raise ValueError("analysis record needs a message and non-negative time")
        object.__setattr__(self, "topic", topic)
        object.__setattr__(self, "record_time_ns", stamp)


def merge_analysis_bag(
    source_bag: Any,
    output_bag: Any,
    records: Sequence[AnalysisBagRecord],
    expected_source_sha256: str,
) -> Mapping[str, Any]:
    """Merge analysis records into a new bag without changing the source bag.

    Both source and application records are written in exact integer
    record-time order.  The source hash is verified before and after copying,
    and an existing destination is never overwritten.
    """

    try:
        import genpy
        import rosbag
    except ImportError as exc:  # pragma: no cover - ROS integration only.
        raise RuntimeError("analysis bag writing requires ROS 1 rosbag/genpy") from exc

    source = Path(source_bag).expanduser().resolve()
    destination = Path(output_bag).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(str(source))
    if source == destination:
        raise ValueError("analysis bag output must differ from the source bag")
    if destination.exists():
        raise FileExistsError(str(destination))
    expected = _validate_digest(expected_source_sha256, "expected_source_sha256")
    before = _hash_file(source)
    if before != expected:
        raise ValueError("source bag hash does not match expected_source_sha256")
    normalized = tuple(records)
    if not normalized or any(
        not isinstance(item, AnalysisBagRecord) for item in normalized
    ):
        raise ValueError("records must contain at least one AnalysisBagRecord")
    ordered = tuple(
        item
        for _, item in sorted(
            enumerate(normalized),
            key=lambda pair: (pair[1].record_time_ns, pair[0]),
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(destination.name),
        suffix=".tmp.bag",
        dir=str(destination.parent),
    )
    os.close(descriptor)
    os.unlink(temporary_name)
    temporary = Path(temporary_name)
    analysis_index = 0

    def ros_time(nanoseconds: int):
        seconds, remainder = divmod(int(nanoseconds), 1_000_000_000)
        return genpy.Time(seconds, remainder)

    try:
        with rosbag.Bag(str(source), "r") as input_stream, rosbag.Bag(
            str(temporary), "w"
        ) as output_stream:
            for topic, message, source_stamp in input_stream.read_messages():
                source_ns = int(source_stamp.to_nsec())
                while (
                    analysis_index < len(ordered)
                    and ordered[analysis_index].record_time_ns <= source_ns
                ):
                    item = ordered[analysis_index]
                    output_stream.write(
                        item.topic,
                        item.message,
                        t=ros_time(item.record_time_ns),
                    )
                    analysis_index += 1
                output_stream.write(topic, message, t=source_stamp)
            while analysis_index < len(ordered):
                item = ordered[analysis_index]
                output_stream.write(
                    item.topic,
                    item.message,
                    t=ros_time(item.record_time_ns),
                )
                analysis_index += 1
        after = _hash_file(source)
        if after != before:
            raise RuntimeError("source bag changed while analysis bag was written")
        if destination.exists():
            raise FileExistsError(str(destination))
        os.rename(str(temporary), str(destination))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "source_bag": str(source),
        "source_bag_sha256": before,
        "analysis_bag": str(destination),
        "analysis_bag_sha256": _hash_file(destination),
        "analysis_record_count": len(ordered),
        "source_bag_unchanged": True,
        "record_time_order": "ascending_integer_nanoseconds",
    }


def _candidate_summary(
    result: CounterfactualResult,
    recommendation_threshold: float,
    exact_controller_gate_passed: bool,
) -> Dict[str, Any]:
    if type(exact_controller_gate_passed) is not bool:
        raise TypeError(
            "exact_controller_gate_passed must be a built-in bool"
        )
    vector = result.candidate.vector()
    if vector.shape != (len(CONTROLLER_PARAMETER_NAMES),):
        raise ValueError("counterfactual candidate has an unexpected parameter vector")
    threshold = float(recommendation_threshold)
    if abs(threshold - float(result.recommendation_threshold)) > 1.0e-12:
        raise ValueError(
            "artifact recommendation threshold must match counterfactual result"
        )
    eligible = (
        exact_controller_gate_passed
        and result.recommendable
    )
    if (
        not exact_controller_gate_passed
        or not result.exact_controller_gate_passed
    ):
        status = EXPERIMENTAL
        reason = "verified_exact_controller_replay_gate_not_passed"
    elif not result.probability_calibration_gate_passed:
        status = EXPERIMENTAL
        reason = "probability_calibration_gate_not_passed"
    elif not result.integrator_state_gate_passed:
        status = EXPERIMENTAL
        reason = "controller_integrator_state_not_restored_or_inferred"
    elif result.dependence_handling != "JOINT_POSTERIOR_SAMPLES":
        status = EXPERIMENTAL
        reason = "joint_posterior_dependence_not_available"
    elif result.support.label != SUPPORTED:
        status = MANUAL_REVIEW_REQUIRED
        reason = "candidate_is_not_supported"
    elif result.lower_credible_bound < threshold:
        status = MANUAL_REVIEW_REQUIRED
        reason = "lower_credible_bound_below_threshold"
    else:
        status = MANUAL_REVIEW_REQUIRED
        reason = "human_review_required_before_any_flight_use"
    return {
        "candidate_id": result.candidate.candidate_id,
        "candidate_parameters": dict(zip(CONTROLLER_PARAMETER_NAMES, vector)),
        "success_probability": result.success_probability,
        "credible_lower": result.credible_lower,
        "credible_upper": result.credible_upper,
        "lower_credible_bound": result.lower_credible_bound,
        "recommendation_threshold": threshold,
        "support": _plain(result.support),
        "violation_probability": _plain(result.violation_probability),
        "effective_rollout_sample_size": result.effective_rollout_sample_size,
        "proposal_eligible_after_statistical_gates": bool(eligible),
        "proposal_status": status,
        "proposal_reason": reason,
        "exact_controller_gate_passed": result.exact_controller_gate_passed,
        "probability_calibration_gate_passed": (
            result.probability_calibration_gate_passed
        ),
        "integrator_state_gate_passed": result.integrator_state_gate_passed,
        "dependence_handling": result.dependence_handling,
        "manual_review_required": True,
        "counterfactual_run_id": result.run_id,
        "counterfactual_provenance": _plain(result.provenance),
    }


class AnalysisArtifactWriter:
    """Write a single non-overwriting counterfactual run bundle."""

    def __init__(self, output_root: Any):
        self.output_root = Path(output_root).expanduser().resolve()

    def write(
        self,
        results: Sequence[CounterfactualResult],
        trajectory_timestamps: Sequence[float],
        provenance: ArtifactProvenance,
        config: Mapping[str, Any],
        recommendation_threshold: float,
        exact_controller_gate_passed: bool = False,
        notes: Sequence[str] = (),
        run_id: Optional[str] = None,
    ) -> Path:
        candidates = tuple(results)
        if not candidates or any(
            not isinstance(item, CounterfactualResult) for item in candidates
        ):
            raise ValueError("results must contain CounterfactualResult values")
        if not isinstance(provenance, ArtifactProvenance):
            raise TypeError("provenance must be ArtifactProvenance")
        if type(exact_controller_gate_passed) is not bool:
            raise TypeError(
                "exact_controller_gate_passed must be a built-in bool"
            )
        timestamps = np.asarray(trajectory_timestamps, dtype=float).reshape(-1)
        if (
            timestamps.size < 2
            or not np.all(np.isfinite(timestamps))
            or np.any(np.diff(timestamps) <= 0.0)
        ):
            raise ValueError("trajectory_timestamps must be finite and increasing")
        if provenance.analysis_mode == "online_prefix":
            if abs(timestamps[0] - float(provenance.prefix_cutoff_s)) > 1.0e-9:
                raise ValueError(
                    "online-prefix rollout must start at prefix_cutoff_s"
                )
        elif (
            timestamps[0] < provenance.interval_start_s - 1.0e-9
            or timestamps[-1] > provenance.interval_end_s + 1.0e-9
        ):
            raise ValueError(
                "retrospective trajectory must lie inside the source interval"
            )
        threshold = float(recommendation_threshold)
        if not 0.0 < threshold < 1.0:
            raise ValueError("recommendation_threshold must lie in (0, 1)")
        config_payload = _plain(dict(config))
        if stable_hash(config_payload) != provenance.config_sha256:
            raise ValueError("config content does not match provenance config_sha256")
        summaries = tuple(
            _candidate_summary(item, threshold, exact_controller_gate_passed)
            for item in candidates
        )
        derived_run_id = stable_hash(
            {
                "schema": ARTIFACT_SCHEMA,
                "provenance": provenance,
                "config": config_payload,
                "candidate_summaries": summaries,
            }
        )[:20]
        identifier = str(run_id or derived_run_id)
        if not identifier or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in identifier
        ):
            raise ValueError("run_id may contain only letters, digits, '-' and '_'")
        self.output_root.mkdir(parents=True, exist_ok=True)
        destination = self.output_root / identifier
        if destination.exists():
            raise FileExistsError(str(destination))
        staging = Path(
            tempfile.mkdtemp(prefix=".{}.".format(identifier), dir=str(self.output_root))
        )
        try:
            self._write_bundle(
                staging,
                identifier,
                candidates,
                summaries,
                timestamps,
                provenance,
                config_payload,
                threshold,
                exact_controller_gate_passed,
                tuple(str(item) for item in notes),
            )
            if destination.exists():
                raise FileExistsError(str(destination))
            os.rename(str(staging), str(destination))
        except Exception:
            if staging.exists():
                shutil.rmtree(str(staging))
            raise
        return destination

    @staticmethod
    def _write_bundle(
        directory: Path,
        run_id: str,
        results: Tuple[CounterfactualResult, ...],
        summaries: Tuple[Mapping[str, Any], ...],
        timestamps: np.ndarray,
        provenance: ArtifactProvenance,
        config: Mapping[str, Any],
        recommendation_threshold: float,
        exact_controller_gate_passed: bool,
        notes: Tuple[str, ...],
    ) -> None:
        any_statistically_eligible = any(
            item["proposal_eligible_after_statistical_gates"]
            for item in summaries
        )
        provenance_payload = {
            "schema": ARTIFACT_SCHEMA,
            "run_id": run_id,
            "manual_review_required": True,
            "workflow_status": (
                MANUAL_REVIEW_REQUIRED
                if any_statistically_eligible
                else EXPERIMENTAL
            ),
            "caller_exact_controller_replay_gate_passed": (
                exact_controller_gate_passed
            ),
            "provenance": _plain(provenance),
            "config": config,
        }
        with (directory / "provenance.json").open("w", encoding="utf-8") as stream:
            json.dump(
                provenance_payload,
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
        candidate_payload = {
            "schema": ARTIFACT_SCHEMA,
            "run_id": run_id,
            "manual_review_required": True,
            "candidates": summaries,
        }
        with (directory / "counterfactual_candidates.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(
                candidate_payload,
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
        csv_fields = (
            "candidate_id",
            "success_probability",
            "credible_lower",
            "credible_upper",
            "lower_credible_bound",
            "recommendation_threshold",
            "support_label",
            "candidate_support_distance",
            "state_action_support_distance_p95",
            "importance_weight_ess",
            "maximum_predictive_std",
            "effective_rollout_sample_size",
            "proposal_eligible_after_statistical_gates",
            "proposal_status",
            "proposal_reason",
            "manual_review_required",
        )
        with (directory / "counterfactual_candidates.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=csv_fields, lineterminator="\n"
            )
            writer.writeheader()
            for result, summary in zip(results, summaries):
                writer.writerow(
                    {
                        "candidate_id": result.candidate.candidate_id,
                        "success_probability": result.success_probability,
                        "credible_lower": result.credible_lower,
                        "credible_upper": result.credible_upper,
                        "lower_credible_bound": result.lower_credible_bound,
                        "recommendation_threshold": recommendation_threshold,
                        "support_label": result.support.label,
                        "candidate_support_distance": result.support.candidate_distance,
                        "state_action_support_distance_p95": (
                            result.support.state_action_distance_p95
                        ),
                        "importance_weight_ess": result.support.importance_weight_ess,
                        "maximum_predictive_std": result.support.maximum_predictive_std,
                        "effective_rollout_sample_size": (
                            result.effective_rollout_sample_size
                        ),
                        "proposal_eligible_after_statistical_gates": summary[
                            "proposal_eligible_after_statistical_gates"
                        ],
                        "proposal_status": summary["proposal_status"],
                        "proposal_reason": summary["proposal_reason"],
                        "manual_review_required": True,
                    }
                )
        AnalysisArtifactWriter._write_trajectory_archive(
            directory / "trajectory_particles.npz", results, timestamps
        )
        AnalysisArtifactWriter._write_report(
            directory / "report.md",
            run_id,
            summaries,
            provenance,
            exact_controller_gate_passed,
            notes,
        )
        files = {}
        for path in sorted(directory.iterdir()):
            if path.is_file():
                files[path.name] = {
                    "sha256": _hash_file(path),
                    "size_bytes": path.stat().st_size,
                }
        manifest = {
            "schema": ARTIFACT_SCHEMA,
            "run_id": run_id,
            "files": files,
        }
        with (directory / "artifact_manifest.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")

    @staticmethod
    def _write_trajectory_archive(
        path: Path,
        results: Tuple[CounterfactualResult, ...],
        timestamps: np.ndarray,
    ) -> None:
        candidate_index = []
        rollout_id = []
        initial_sample_id = []
        response_sample_id = []
        noise_sample_id = []
        joint_sample_id = []
        weight = []
        position = []
        velocity = []
        command = []
        saturation = []
        tube_success = []
        for candidate, result in enumerate(results):
            for rollout in result.rollouts:
                if rollout.position.shape[0] != timestamps.size:
                    raise ValueError(
                        "every rollout must align with trajectory_timestamps"
                    )
                candidate_index.append(candidate)
                rollout_id.append(rollout.rollout_id)
                initial_sample_id.append(rollout.initial_sample_id)
                response_sample_id.append(rollout.response_sample_id)
                noise_sample_id.append(rollout.noise_sample_id)
                joint_sample_id.append(rollout.joint_sample_id)
                weight.append(rollout.weight)
                position.append(rollout.position)
                velocity.append(rollout.velocity)
                command.append(rollout.command)
                saturation.append(rollout.saturation)
                tube_success.append(rollout.tube.success)
        if not position:
            raise ValueError("results contain no trajectory rollouts")
        np.savez_compressed(
            str(path),
            schema=np.asarray(ARTIFACT_SCHEMA),
            timestamps=np.asarray(timestamps, dtype=np.float64),
            candidate_ids=np.asarray(
                [item.candidate.candidate_id for item in results], dtype=np.str_
            ),
            candidate_index=np.asarray(candidate_index, dtype=np.int64),
            rollout_id=np.asarray(rollout_id, dtype=np.int64),
            initial_sample_id=np.asarray(initial_sample_id, dtype=np.int64),
            response_sample_id=np.asarray(response_sample_id, dtype=np.int64),
            noise_sample_id=np.asarray(noise_sample_id, dtype=np.int64),
            joint_sample_id=np.asarray(joint_sample_id, dtype=np.int64),
            weight=np.asarray(weight, dtype=np.float64),
            position=np.asarray(position, dtype=np.float64),
            velocity=np.asarray(velocity, dtype=np.float64),
            command=np.asarray(command, dtype=np.float64),
            saturation=np.asarray(saturation, dtype=np.bool_),
            tube_success=np.asarray(tube_success, dtype=np.bool_),
        )

    @staticmethod
    def _write_report(
        path: Path,
        run_id: str,
        summaries: Tuple[Mapping[str, Any], ...],
        provenance: ArtifactProvenance,
        exact_controller_gate_passed: bool,
        notes: Tuple[str, ...],
    ) -> None:
        any_statistically_eligible = any(
            item["proposal_eligible_after_statistical_gates"]
            for item in summaries
        )
        lines = [
            "# Grape counterfactual analysis {}".format(run_id),
            "",
            "**MANUAL REVIEW REQUIRED. This bundle is not an automatic flight command.**",
            "",
            "Workflow status: `{}`.".format(
                MANUAL_REVIEW_REQUIRED
                if any_statistically_eligible
                else EXPERIMENTAL
            ),
        ]
        if not any_statistically_eligible:
            lines.extend(
                [
                    "",
                    "At least one required exact-controller, probability-"
                    "calibration, joint-dependence, support, or lower-bound "
                    "gate has not passed. Candidate probabilities are "
                    "experimental and must not be presented as flight "
                    "recommendations.",
                ]
            )
        lines.extend(
            [
                "",
                "Source bags: `{}`.".format(
                    "`, `".join(provenance.source_bag_sha256)
                ),
                "",
                "Config SHA-256: `{}`; source commit: `{}`; seed: `{}`.".format(
                    provenance.config_sha256,
                    provenance.source_commit,
                    provenance.seed,
                ),
                "",
                "## Candidate summary",
                "",
                "| candidate | q | credible interval | support | lower-bound gate | status |",
                "|---|---:|---:|---|---:|---|",
            ]
        )
        for summary in summaries:
            support = summary["support"]
            lines.append(
                "| {candidate_id} | {success_probability:.6g} | "
                "[{credible_lower:.6g}, {credible_upper:.6g}] | {support_label} | "
                "{lower_credible_bound:.6g} / {recommendation_threshold:.6g} | "
                "{proposal_status} |".format(
                    support_label=support["label"],
                    candidate_id=summary["candidate_id"],
                    success_probability=summary["success_probability"],
                    credible_lower=summary["credible_lower"],
                    credible_upper=summary["credible_upper"],
                    lower_credible_bound=summary["lower_credible_bound"],
                    recommendation_threshold=summary[
                        "recommendation_threshold"
                    ],
                    proposal_status=summary["proposal_status"],
                )
            )
        if notes:
            lines.extend(["", "## Notes", ""])
            lines.extend("- {}".format(item) for item in notes)
        lines.extend(
            [
                "",
                "Machine-readable values and coherent trajectory samples are "
                "stored beside this report; `artifact_manifest.json` binds them "
                "to their content hashes.",
                "",
            ]
        )
        with path.open("w", encoding="utf-8") as stream:
            stream.write("\n".join(lines))


__all__ = [
    "ARTIFACT_SCHEMA",
    "CONTROLLER_PARAMETER_NAMES",
    "EXPERIMENTAL",
    "MANUAL_REVIEW_REQUIRED",
    "AnalysisBagRecord",
    "AnalysisArtifactWriter",
    "ArtifactProvenance",
    "merge_analysis_bag",
]
