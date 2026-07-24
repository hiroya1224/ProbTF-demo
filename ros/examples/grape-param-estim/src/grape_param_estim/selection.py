"""Frozen leave-one-bag-out selection protocol and decision runner."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .episode import stable_hash


SELECTION_SCHEMA = "grape_backend_selection/v1"
DEFAULT_CANDIDATE = "DEFAULT_CANDIDATE"
OPTIONAL_CANDIDATE = "OPTIONAL_CANDIDATE"
PRUNE = "PRUNE"
EXPERIMENTAL = "EXPERIMENTAL"
_STATUSES = (
    DEFAULT_CANDIDATE,
    OPTIONAL_CANDIDATE,
    PRUNE,
    EXPERIMENTAL,
)


def _digest(value: Any, name: str) -> str:
    result = str(value).lower()
    if len(result) != 64 or any(item not in "0123456789abcdef" for item in result):
        raise ValueError("{} must be a lowercase SHA-256 digest".format(name))
    return result


def load_selection_protocol(path: Any) -> Dict[str, Any]:
    import yaml

    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        protocol = yaml.safe_load(stream) or {}
    validate_selection_protocol(protocol)
    return protocol


def validate_selection_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema") != SELECTION_SCHEMA:
        raise ValueError("unsupported selection protocol schema")
    _digest(protocol.get("manifest_hash", ""), "manifest_hash")
    episodes = protocol.get("episodes")
    folds = protocol.get("outer_folds")
    groups = protocol.get("candidate_groups")
    if not isinstance(episodes, Mapping) or len(episodes) < 2:
        raise ValueError("selection protocol needs at least two episodes")
    episode_ids = set(str(item) for item in episodes)
    for episode_id, item in episodes.items():
        _digest(item.get("bag_sha256", ""), "{} bag hash".format(episode_id))
        if not item.get("stratum"):
            raise ValueError("{} is missing stratum".format(episode_id))
    if not isinstance(folds, list) or len(folds) != len(episodes):
        raise ValueError("outer_folds must contain one fold per episode")
    held_out = []
    fold_ids = set()
    for fold in folds:
        fold_id = str(fold.get("fold_id", ""))
        test_episode = str(fold.get("held_out_episode", ""))
        validation = tuple(str(item) for item in fold.get("inner_validation", ()))
        training = tuple(str(item) for item in fold.get("train_episodes", ()))
        if not fold_id or fold_id in fold_ids:
            raise ValueError("outer fold IDs must be non-empty and unique")
        fold_ids.add(fold_id)
        if test_episode not in episode_ids or len(validation) != 1:
            raise ValueError("{} has invalid held-out/validation episodes".format(fold_id))
        partitions = [set(training), set(validation), {test_episode}]
        if any(partitions[i] & partitions[j] for i in range(3) for j in range(i)):
            raise ValueError("{} leaks an episode across partitions".format(fold_id))
        if set.union(*partitions) != episode_ids:
            raise ValueError("{} does not partition every episode".format(fold_id))
        if fold.get("held_out_bag_sha256") != episodes[test_episode]["bag_sha256"]:
            raise ValueError("{} held-out hash does not match episode".format(fold_id))
        held_out.append(test_episode)
    if set(held_out) != episode_ids or len(set(held_out)) != len(episodes):
        raise ValueError("each episode must be held out exactly once")
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("candidate_groups must not be empty")
    candidate_ids = set()
    for group_id, group in groups.items():
        if group.get("direction") not in ("maximize", "minimize"):
            raise ValueError("{} has invalid metric direction".format(group_id))
        if not group.get("primary_metric"):
            raise ValueError("{} is missing primary_metric".format(group_id))
        candidates = group.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("{} has no candidates".format(group_id))
        group_ids = set()
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id", ""))
            if (
                not candidate_id
                or candidate_id in candidate_ids
                or candidate_id in group_ids
            ):
                raise ValueError("candidate IDs must be globally unique")
            group_ids.add(candidate_id)
            candidate_ids.add(candidate_id)
            if int(candidate.get("complexity_rank", 0)) < 1:
                raise ValueError("{} needs a positive complexity_rank".format(candidate_id))
        blocker = group.get("blocking_candidate_id")
        if blocker is not None and blocker not in group_ids:
            raise ValueError("{} blocker is not in its candidate group".format(group_id))
    evaluation = protocol.get("evaluation", {})
    required_gates = evaluation.get("required_hard_gates")
    seeds = evaluation.get("seeds")
    if not isinstance(required_gates, list) or not required_gates:
        raise ValueError("required_hard_gates must be frozen and non-empty")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("evaluation seeds must be frozen and non-empty")
    bootstrap = evaluation.get("bootstrap", {})
    if int(bootstrap.get("draws", 0)) < 100:
        raise ValueError("episode bootstrap requires at least 100 draws")
    probability = float(bootstrap.get("credible_probability", 0.0))
    if not 0.0 < probability < 1.0:
        raise ValueError("bootstrap credible probability is invalid")


@dataclass(frozen=True)
class SelectionObservation:
    candidate_id: str
    fold_id: str
    held_out_episode: str
    stratum: str
    metrics: Mapping[str, float]
    hard_gates: Mapping[str, bool]
    trajectory_sample_bundle_sha256: str
    candidate_grid_sha256: str
    random_stream_sha256: str
    run_sha256: str
    model_version: str = ""
    controller_backend_identity_sha256: str = ""
    exact_conformance_report_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.fold_id or not self.held_out_episode:
            raise ValueError("selection observation IDs must not be empty")
        metrics = {str(key): float(value) for key, value in self.metrics.items()}
        if not metrics or not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError("selection metrics must be finite and non-empty")
        if any(type(value) is not bool for value in self.hard_gates.values()):
            raise ValueError(
                "selection hard-gate values must be JSON booleans"
            )
        gates = {
            str(key): value for key, value in self.hard_gates.items()
        }
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "hard_gates", gates)
        for name in (
            "trajectory_sample_bundle_sha256",
            "candidate_grid_sha256",
            "random_stream_sha256",
            "run_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "model_version", str(self.model_version))
        for name in (
            "controller_backend_identity_sha256",
            "exact_conformance_report_sha256",
        ):
            value = str(getattr(self, name))
            if value:
                value = _digest(value, name)
            object.__setattr__(self, name, value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SelectionObservation":
        return cls(
            candidate_id=str(value["candidate_id"]),
            fold_id=str(value["fold_id"]),
            held_out_episode=str(value["held_out_episode"]),
            stratum=str(value["stratum"]),
            metrics=dict(value["metrics"]),
            hard_gates=dict(value["hard_gates"]),
            trajectory_sample_bundle_sha256=str(
                value["trajectory_sample_bundle_sha256"]
            ),
            candidate_grid_sha256=str(value["candidate_grid_sha256"]),
            random_stream_sha256=str(value["random_stream_sha256"]),
            run_sha256=str(value["run_sha256"]),
            model_version=str(value.get("model_version", "")),
            controller_backend_identity_sha256=str(
                value.get("controller_backend_identity_sha256", "")
            ),
            exact_conformance_report_sha256=str(
                value.get("exact_conformance_report_sha256", "")
            ),
        )


def load_selection_observations(path: Any) -> Tuple[SelectionObservation, ...]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    values = payload.get("observations", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(values, list):
        raise ValueError("observation JSON must contain a list")
    return tuple(SelectionObservation.from_mapping(item) for item in values)


def episode_bootstrap_mean(
    values: Sequence[float],
    seed: int,
    draws: int = 4000,
    credible_probability: float = 0.95,
) -> Mapping[str, float]:
    """Bootstrap whole held-out episodes, never individual timestamps."""

    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("episode bootstrap needs finite episode-level values")
    count = int(draws)
    probability = float(credible_probability)
    if count < 100 or not 0.0 < probability < 1.0:
        raise ValueError("invalid episode-bootstrap configuration")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(count, array.size))
    means = np.mean(array[indices], axis=1)
    tail = 0.5 * (1.0 - probability)
    standard_error = (
        float(np.std(array, ddof=1) / np.sqrt(array.size))
        if array.size > 1
        else 0.0
    )
    return {
        "episode_count": int(array.size),
        "mean": float(np.mean(array)),
        "standard_error": standard_error,
        "bootstrap_lower": float(np.quantile(means, tail)),
        "bootstrap_upper": float(np.quantile(means, 1.0 - tail)),
        "bootstrap_draws": count,
        "credible_probability": probability,
    }


def _protocol_indexes(protocol: Mapping[str, Any]):
    fold_by_id = {
        str(item["fold_id"]): item for item in protocol["outer_folds"]
    }
    candidates = {}
    for group_id, group in protocol["candidate_groups"].items():
        for candidate in group["candidates"]:
            candidates[candidate["candidate_id"]] = (str(group_id), group, candidate)
    return fold_by_id, candidates


def _validate_comparison_inputs(
    observations: Tuple[SelectionObservation, ...],
    fold_by_id: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Any],
) -> None:
    seen = set()
    common_by_fold = {}
    for item in observations:
        if item.candidate_id not in candidates:
            raise ValueError("unknown candidate '{}'".format(item.candidate_id))
        if item.fold_id not in fold_by_id:
            raise ValueError("unknown fold '{}'".format(item.fold_id))
        fold = fold_by_id[item.fold_id]
        if (
            item.held_out_episode != fold["held_out_episode"]
            or item.stratum
            != fold.get("stratum", "")
        ):
            raise ValueError("{} observation does not match frozen fold".format(item.fold_id))
        key = (item.candidate_id, item.fold_id)
        if key in seen:
            raise ValueError("candidate/fold observation is duplicated")
        seen.add(key)
        comparison = (
            item.trajectory_sample_bundle_sha256,
            item.candidate_grid_sha256,
            item.random_stream_sha256,
        )
        group_id = candidates[item.candidate_id][0]
        comparison_key = (group_id, item.fold_id)
        previous = common_by_fold.setdefault(comparison_key, comparison)
        if previous != comparison:
            raise ValueError(
                "{} candidates in {} did not use common "
                "samples/grid/random stream".format(
                    item.fold_id, group_id
                )
            )


def run_selection(
    protocol: Mapping[str, Any],
    observations: Sequence[SelectionObservation],
    source_commit: str,
) -> Dict[str, Any]:
    """Apply hard gates, episode bootstrap, and the one-standard-error rule."""

    validate_selection_protocol(protocol)
    values = tuple(observations)
    if any(not isinstance(item, SelectionObservation) for item in values):
        raise TypeError("observations must be SelectionObservation values")
    fold_by_id, candidate_index = _protocol_indexes(protocol)
    _validate_comparison_inputs(values, fold_by_id, candidate_index)
    evaluation = protocol["evaluation"]
    required_gates = tuple(str(item) for item in evaluation["required_hard_gates"])
    bootstrap = evaluation["bootstrap"]
    all_fold_ids = set(fold_by_id)
    by_candidate: Dict[str, list] = {item: [] for item in candidate_index}
    for item in values:
        by_candidate[item.candidate_id].append(item)
    candidate_results = {}
    for candidate_id, (group_id, group, candidate) in candidate_index.items():
        candidate_observations = tuple(by_candidate[candidate_id])
        present_folds = {item.fold_id for item in candidate_observations}
        missing_folds = tuple(sorted(all_fold_ids - present_folds))
        failed_gates = sorted(
            {
                gate
                for item in candidate_observations
                for gate in required_gates
                if item.hard_gates.get(gate) is not True
            }
        )
        candidate_required = tuple(candidate.get("required_hard_gates", ()))
        failed_gates.extend(
            sorted(
                {
                    gate
                    for item in candidate_observations
                    for gate in candidate_required
                    if item.hard_gates.get(gate) is not True
                }
            )
        )
        failed_gates = tuple(dict.fromkeys(failed_gates))
        metric = str(group["primary_metric"])
        missing_metric_folds = tuple(
            sorted(
                item.fold_id
                for item in candidate_observations
                if metric not in item.metrics
            )
        )
        statistics = None
        if candidate_observations and not missing_metric_folds:
            metric_values = [item.metrics[metric] for item in candidate_observations]
            seed_material = stable_hash(
                {
                    "protocol_hash": stable_hash(protocol),
                    "candidate_id": candidate_id,
                    "bootstrap_seed": bootstrap["seed"],
                }
            )
            seed = int(seed_material[:16], 16)
            statistics = episode_bootstrap_mean(
                metric_values,
                seed,
                int(bootstrap["draws"]),
                float(bootstrap["credible_probability"]),
            )
        complete = not missing_folds and not missing_metric_folds
        if not candidate_observations or not complete:
            status = EXPERIMENTAL
            reasons = ["incomplete_outer_fold_evaluation"]
        elif failed_gates:
            status = PRUNE
            reasons = ["hard_gate_failed"]
        else:
            status = EXPERIMENTAL
            reasons = ["awaiting_group_comparison"]
        candidate_results[candidate_id] = {
            "candidate_id": candidate_id,
            "group_id": group_id,
            "description": candidate.get("description", ""),
            "complexity_rank": int(candidate["complexity_rank"]),
            "status": status,
            "reasons": reasons,
            "observation_count": len(candidate_observations),
            "missing_folds": missing_folds,
            "missing_metric_folds": missing_metric_folds,
            "failed_hard_gates": failed_gates,
            "primary_metric": metric,
            "direction": group["direction"],
            "statistics": statistics,
            "run_hashes": tuple(sorted(item.run_sha256 for item in candidate_observations)),
            "trajectory_sample_bundle_hashes": tuple(
                sorted(
                    item.trajectory_sample_bundle_sha256
                    for item in candidate_observations
                )
            ),
            "model_versions": tuple(
                sorted(
                    {
                        item.model_version
                        for item in candidate_observations
                        if item.model_version
                    }
                )
            ),
            "controller_backend_identity_hashes": tuple(
                sorted(
                    {
                        item.controller_backend_identity_sha256
                        for item in candidate_observations
                        if item.controller_backend_identity_sha256
                    }
                )
            ),
            "exact_conformance_report_hashes": tuple(
                sorted(
                    {
                        item.exact_conformance_report_sha256
                        for item in candidate_observations
                        if item.exact_conformance_report_sha256
                    }
                )
            ),
        }
    group_results = {}
    for group_id, group in protocol["candidate_groups"].items():
        identifiers = [item["candidate_id"] for item in group["candidates"]]
        eligible = [
            candidate_results[item]
            for item in identifiers
            if candidate_results[item]["statistics"] is not None
            and not candidate_results[item]["missing_folds"]
            and not candidate_results[item]["missing_metric_folds"]
            and not candidate_results[item]["failed_hard_gates"]
        ]
        blocker = group.get("blocking_candidate_id")
        blocker_ready = blocker is None or any(
            item["candidate_id"] == blocker for item in eligible
        )
        if eligible and blocker_ready:
            reverse = group["direction"] == "maximize"
            best = sorted(
                eligible,
                key=lambda item: item["statistics"]["mean"],
                reverse=reverse,
            )[0]
            tolerance = best["statistics"]["standard_error"]
            if reverse:
                one_se = [
                    item
                    for item in eligible
                    if item["statistics"]["mean"]
                    >= best["statistics"]["mean"] - tolerance
                ]
            else:
                one_se = [
                    item
                    for item in eligible
                    if item["statistics"]["mean"]
                    <= best["statistics"]["mean"] + tolerance
                ]
            chosen = min(
                one_se,
                key=lambda item: (
                    item["complexity_rank"],
                    item["candidate_id"],
                ),
            )
            for item in eligible:
                if item["candidate_id"] == chosen["candidate_id"]:
                    item["status"] = DEFAULT_CANDIDATE
                    item["reasons"] = ["one_standard_error_rule_simplest"]
                elif item in one_se:
                    item["status"] = OPTIONAL_CANDIDATE
                    item["reasons"] = ["within_one_standard_error"]
                else:
                    item["status"] = PRUNE
                    item["reasons"] = ["outside_one_standard_error"]
        elif eligible and not blocker_ready:
            for item in eligible:
                item["status"] = EXPERIMENTAL
                item["reasons"] = [
                    "blocking_candidate_{}_not_validated".format(blocker)
                ]
        group_results[group_id] = {
            "group_id": group_id,
            "primary_metric": group["primary_metric"],
            "direction": group["direction"],
            "blocking_candidate_id": blocker,
            "candidate_ids": tuple(identifiers),
            "selected_default": next(
                (
                    item
                    for item in identifiers
                    if candidate_results[item]["status"] == DEFAULT_CANDIDATE
                ),
                None,
            ),
        }
    result = {
        "schema": SELECTION_SCHEMA,
        "protocol_hash": stable_hash(protocol),
        "manifest_hash": protocol["manifest_hash"],
        "source_commit": str(source_commit),
        "outer_fold_count": len(fold_by_id),
        "observation_count": len(values),
        "episode_level_resampling": True,
        "groups": group_results,
        "candidates": candidate_results,
    }
    result["selection_complete"] = bool(
        all(item["selected_default"] is not None for item in group_results.values())
        and all(
            item["observation_count"] == len(fold_by_id)
            and not item["missing_folds"]
            and not item["missing_metric_folds"]
            for item in candidate_results.values()
        )
    )
    result["result_hash"] = stable_hash(result)
    return result


def render_selection_markdown(result: Mapping[str, Any]) -> str:
    status = "COMPLETE" if result["selection_complete"] else EXPERIMENTAL
    lines = [
        "# Grape backend selection results",
        "",
        "Selection status: `{}`.".format(status),
        "",
        "This file records frozen held-out decisions. A missing comparison is "
        "reported as `EXPERIMENTAL`; it is not treated as evidence for a default.",
        "",
        "- Source commit: `{}`".format(result["source_commit"]),
        "- Bag manifest hash: `{}`".format(result["manifest_hash"]),
        "- Selection protocol hash: `{}`".format(result["protocol_hash"]),
        "- Result hash: `{}`".format(result["result_hash"]),
        "- Outer held-out folds: `{}`".format(result["outer_fold_count"]),
        "- Submitted observations: `{}`".format(result["observation_count"]),
        "- Resampling unit: whole episode/bag",
        "",
        "## Decisions",
        "",
        "| component | candidate | status | held-out bags | metric mean (95% bootstrap CI) | hard-gate failures | reason |",
        "|---|---|---|---:|---|---|---|",
    ]
    for candidate_id, item in sorted(
        result["candidates"].items(),
        key=lambda pair: (pair[1]["group_id"], pair[0]),
    ):
        statistics = item["statistics"]
        if statistics is None:
            metric = "not measured"
        else:
            metric = "{:.6g} [{:.6g}, {:.6g}]".format(
                statistics["mean"],
                statistics["bootstrap_lower"],
                statistics["bootstrap_upper"],
            )
        lines.append(
            "| {} | {} | `{}` | {} | {} | {} | {} |".format(
                item["group_id"],
                candidate_id,
                item["status"],
                item["observation_count"],
                metric,
                ", ".join(item["failed_hard_gates"]) or "not evaluated",
                ", ".join(item["reasons"]),
            )
        )
    lines.extend(
        [
            "",
            "## Current limitation",
            "",
            "No candidate may be promoted from this file until all frozen outer "
            "folds and hard gates are present. In particular, an unavailable or "
            "unverified exact PC/MCU controller oracle blocks counterfactual "
            "recommendation; Python replay remains an approximation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_selection_outputs(
    result: Mapping[str, Any],
    markdown_path: Any,
    json_path: Optional[Any] = None,
    overwrite: bool = False,
) -> None:
    outputs = [(Path(markdown_path).expanduser().resolve(), render_selection_markdown(result))]
    if json_path is not None:
        outputs.append(
            (
                Path(json_path).expanduser().resolve(),
                json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            )
        )
    for destination, _ in outputs:
        if destination.exists() and not overwrite:
            raise FileExistsError(str(destination))
    for destination, content in outputs:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=destination.name + ".",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, str(destination))
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise


__all__ = [
    "DEFAULT_CANDIDATE",
    "EXPERIMENTAL",
    "OPTIONAL_CANDIDATE",
    "PRUNE",
    "SELECTION_SCHEMA",
    "SelectionObservation",
    "episode_bootstrap_mean",
    "load_selection_observations",
    "load_selection_protocol",
    "render_selection_markdown",
    "run_selection",
    "validate_selection_protocol",
    "write_selection_outputs",
]
