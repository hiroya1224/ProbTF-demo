#!/usr/bin/env python3
"""Aggregate three completed single-bag parameter-prior ablations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/grape-prior-three-bag-matplotlib")

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from single_bag_parameter_prior import (  # noqa: E402
    QUOTIENT_COMPONENT_LABELS,
    QUOTIENT_COMPONENT_UNITS,
)
from single_bag_savgol_core import BASE_PLAN_COMMIT  # noqa: E402
from single_bag_savgol_reports import (  # noqa: E402
    output_run_directory,
    source_commit,
    write_json,
)


EXPECTED_BAG_IDS = (
    "single_rosbag_1",
    "single_rosbag_2",
    "single_rosbag_succeeded",
)
PAIR_INDICES = ((0, 1), (0, 2), (1, 2))


def _load_completed_ablation(directory: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = Path(directory).expanduser().resolve()
    try:
        status = json.loads((path / "status.json").read_text(encoding="utf-8"))
        summary = json.loads((path / "prior_ablation.json").read_text(encoding="utf-8"))
        archive = np.load(path / "prior_ablation.npz", allow_pickle=False)
        arrays = {name: archive[name] for name in archive.files}
        archive.close()
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("completed prior ablation cannot be read: {}".format(path)) from error
    if status.get("status") != "completed" or summary.get("status") != "completed":
        raise ValueError("prior ablation is not terminal-completed: {}".format(path))
    if summary.get("schema") != "grape-param-estim/prior-ablation-summary/v1":
        raise ValueError("unsupported per-bag prior-ablation summary schema")
    arrays["source_directory"] = np.asarray(str(path))
    return summary, arrays


def _inertia_frobenius_norm(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    return np.sqrt(
        np.sum(array[..., :3] ** 2, axis=-1)
        + 2.0 * np.sum(array[..., 3:6] ** 2, axis=-1)
    )


def _spread(values: np.ndarray, family: str) -> np.ndarray:
    centered = values - np.mean(values, axis=0, keepdims=True)
    if family == "inertia":
        squared = _inertia_frobenius_norm(centered) ** 2
    else:
        squared = np.sum(centered**2, axis=-1)
    return np.sqrt(np.mean(squared, axis=0))


def _pairwise(values: np.ndarray, family: str) -> np.ndarray:
    distances = []
    for first, second in PAIR_INDICES:
        difference = values[first] - values[second]
        if family == "inertia":
            distances.append(_inertia_frobenius_norm(difference))
        else:
            distances.append(np.linalg.norm(difference, axis=-1))
    return np.asarray(distances).T


def build_three_bag_summary(
    loaded: Sequence[tuple[Mapping[str, Any], Mapping[str, np.ndarray]]],
    *,
    source_revision: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if len(loaded) != 3:
        raise ValueError("exactly three per-bag ablations are required")
    by_id = {str(summary["bag_id"]): (summary, arrays) for summary, arrays in loaded}
    if set(by_id) != set(EXPECTED_BAG_IDS) or len(by_id) != 3:
        raise ValueError("inputs must be the three fixed production bag IDs")
    ordered = [by_id[bag_id] for bag_id in EXPECTED_BAG_IDS]
    summaries = [item[0] for item in ordered]
    per_bag = [item[1] for item in ordered]
    source_commits = {str(item["source_commit"]) for item in summaries}
    if source_commits != {source_revision}:
        raise ValueError("all per-bag outputs must match the current source commit")
    manifest_hashes = {str(item["manifest"]["source_sha256"]) for item in summaries}
    if len(manifest_hashes) != 1:
        raise ValueError("all three bags must use the same prior-ablation manifest")
    case_names = np.asarray(per_bag[0]["case_names"]).astype("U")
    for arrays in per_bag[1:]:
        if not np.array_equal(case_names, np.asarray(arrays["case_names"]).astype("U")):
            raise ValueError("case order differs among per-bag ablations")
    x = np.stack([np.asarray(item["x_per_case"], dtype=float) for item in per_bag])
    data_cost = np.stack([np.asarray(item["data_objective_per_case"], dtype=float) for item in per_bag])
    delta_data = np.stack([np.asarray(item["delta_data_objective_per_case"], dtype=float) for item in per_bag])
    overall_status = np.stack([np.asarray(item["overall_case_status_per_case"]).astype("U") for item in per_bag])
    optimization_status = np.stack([np.asarray(item["optimization_status_per_case"]).astype("U") for item in per_bag])
    postfit_status = np.stack([np.asarray(item["postfit_uncertainty_status_per_case"]).astype("U") for item in per_bag])
    if x.shape != (3, case_names.size, 13):
        raise ValueError("per-bag quotient arrays have inconsistent dimensions")

    inertia = x[:, :, :6]
    cog = x[:, :, 6:9]
    force = x[:, :, 9:13]
    spread_cog = _spread(cog, "vector")
    spread_inertia = _spread(inertia, "inertia")
    spread_force = _spread(force, "vector")
    pairwise_cog = _pairwise(cog, "vector")
    pairwise_inertia = _pairwise(inertia, "inertia")
    pairwise_force = _pairwise(force, "vector")
    pair_labels = tuple(
        "{}__{}".format(EXPECTED_BAG_IDS[first], EXPECTED_BAG_IDS[second])
        for first, second in PAIR_INDICES
    )
    records = []
    for index, case_name in enumerate(case_names):
        records.append(
            {
                "case_name": str(case_name),
                "x_per_bag": {
                    bag_id: x[bag_index, index]
                    for bag_index, bag_id in enumerate(EXPECTED_BAG_IDS)
                },
                "delta_data_objective_per_bag": {
                    bag_id: delta_data[bag_index, index]
                    for bag_index, bag_id in enumerate(EXPECTED_BAG_IDS)
                },
                "data_objective_per_bag": {
                    bag_id: data_cost[bag_index, index]
                    for bag_index, bag_id in enumerate(EXPECTED_BAG_IDS)
                },
                "cog_spread_m": spread_cog[index],
                "inertia_over_mass_spread_m2_frobenius": spread_inertia[index],
                "force_over_mass_spread_kg_inverse": spread_force[index],
                "pairwise_cog_distance_m": dict(zip(pair_labels, pairwise_cog[index])),
                "pairwise_inertia_over_mass_distance_m2_frobenius": dict(zip(pair_labels, pairwise_inertia[index])),
                "pairwise_force_over_mass_distance_kg_inverse": dict(zip(pair_labels, pairwise_force[index])),
                "overall_case_status_per_bag": dict(zip(EXPECTED_BAG_IDS, overall_status[:, index])),
                "optimization_status_per_bag": dict(zip(EXPECTED_BAG_IDS, optimization_status[:, index])),
                "postfit_uncertainty_status_per_bag": dict(zip(EXPECTED_BAG_IDS, postfit_status[:, index])),
            }
        )
    arrays = {
        "bag_ids": np.asarray(EXPECTED_BAG_IDS, dtype="U"),
        "case_names": case_names,
        "quotient_component_labels": np.asarray(QUOTIENT_COMPONENT_LABELS, dtype="U"),
        "quotient_component_units": np.asarray(QUOTIENT_COMPONENT_UNITS, dtype="U"),
        "x_per_bag_per_case": x,
        "data_objective_per_bag_per_case": data_cost,
        "delta_data_objective_per_bag_per_case": delta_data,
        "cog_spread_m": spread_cog,
        "inertia_over_mass_spread_m2_frobenius": spread_inertia,
        "force_over_mass_spread_kg_inverse": spread_force,
        "pairwise_labels": np.asarray(pair_labels, dtype="U"),
        "pairwise_cog_distance_m": pairwise_cog,
        "pairwise_inertia_over_mass_distance_m2_frobenius": pairwise_inertia,
        "pairwise_force_over_mass_distance_kg_inverse": pairwise_force,
        "overall_case_status_per_bag_per_case": overall_status,
        "optimization_status_per_bag_per_case": optimization_status,
        "postfit_uncertainty_status_per_bag_per_case": postfit_status,
    }
    summary = {
        "schema": "grape-param-estim/prior-ablation-three-bag/v1",
        "status": "completed",
        "source_commit": source_revision,
        "base_plan_commit": BASE_PLAN_COMMIT,
        "manifest_sha256": next(iter(manifest_hashes)),
        "bag_ids": EXPECTED_BAG_IDS,
        "source_directories": [str(item["source_directory"]) for item in per_bag],
        "case_names": case_names,
        "quotient_component_labels": QUOTIENT_COMPONENT_LABELS,
        "quotient_component_units": QUOTIENT_COMPONENT_UNITS,
        "cross_evaluation": {
            "status": "not_run",
            "phase": "secondary_explicit_phase",
            "reason": "Primary study uses covariance-independent point spreads and all 17 predefined cases.",
        },
        "cases": records,
    }
    return summary, arrays


def _family_page(
    pdf: PdfPages,
    *,
    x: np.ndarray,
    case_names: Sequence[str],
    component_slice: slice,
    title: str,
    unit: str,
) -> None:
    values = x[:, :, component_slice]
    finite = values[np.isfinite(values)]
    lower = float(np.min(finite)) if finite.size else 0.0
    upper = float(np.max(finite)) if finite.size else 1.0
    if lower == upper:
        upper = lower + 1.0
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 8.5), sharey=True)
    labels = QUOTIENT_COMPONENT_LABELS[component_slice]
    for bag_index, axis in enumerate(axes):
        image = axis.imshow(values[bag_index], aspect="auto", cmap="viridis", vmin=lower, vmax=upper)
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        axis.set_title(EXPECTED_BAG_IDS[bag_index])
        axis.set_yticks(np.arange(len(case_names)))
        axis.set_yticklabels(case_names, fontsize=6)
    figure.colorbar(image, ax=axes.ravel().tolist(), label=unit, shrink=0.8)
    figure.suptitle(title + " (actual point estimates)")
    figure.subplots_adjust(left=0.22, right=0.9, bottom=0.12, top=0.9, wspace=0.08)
    pdf.savefig(figure)
    plt.close(figure)


def write_three_bag_pdf(path: Path, summary: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    names = [str(item) for item in arrays["case_names"]]
    x = np.asarray(arrays["x_per_bag_per_case"], dtype=float)
    with PdfPages(path) as pdf:
        figure = plt.figure(figsize=(11.0, 8.5))
        axis = figure.add_subplot(111)
        axis.axis("off")
        lines = [
            "Three-bag nominal pseudo-conditioning ablation",
            "source commit: {}".format(summary["source_commit"]),
            "manifest SHA256: {}".format(summary["manifest_sha256"]),
            "bags: {}".format(", ".join(summary["bag_ids"])),
            "cases: {}".format(len(names)),
            "",
            "Spreads are RMS distances about the three-bag mean. J/m uses the",
            "Frobenius matrix norm; CoG and f/m use Euclidean vector norms.",
            "Cross-evaluation is intentionally a separate secondary phase.",
        ]
        axis.text(0.04, 0.96, "\n".join(lines), va="top", family="monospace", fontsize=10)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
        index = np.arange(len(names))
        for bag_index, bag_id in enumerate(EXPECTED_BAG_IDS):
            axes[0, 0].plot(index, arrays["delta_data_objective_per_bag_per_case"][bag_index], "o-", ms=3, label=bag_id)
        axes[0, 0].set_ylabel("delta data objective")
        axes[0, 0].legend(fontsize=7)
        axes[0, 1].plot(index, arrays["cog_spread_m"], "o-")
        axes[0, 1].set_ylabel("CoG spread [m]")
        axes[1, 0].plot(index, arrays["inertia_over_mass_spread_m2_frobenius"], "o-")
        axes[1, 0].set_ylabel("J/m spread [m^2], Frobenius")
        axes[1, 1].plot(index, arrays["force_over_mass_spread_kg_inverse"], "o-")
        axes[1, 1].set_ylabel("f/m spread [kg^-1]")
        for axis in axes.flat:
            axis.set_xticks(index)
            axis.set_xticklabels(names, rotation=75, ha="right", fontsize=5.5)
            axis.grid(True, alpha=0.3)
        figure.suptitle("Data-fit price and cross-bag point spreads")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        _family_page(pdf, x=x, case_names=names, component_slice=slice(6, 9), title="CoG", unit="m")
        _family_page(pdf, x=x, case_names=names, component_slice=slice(0, 6), title="Inertia over mass", unit="m^2")
        _family_page(pdf, x=x, case_names=names, component_slice=slice(9, 13), title="Force effectiveness over mass", unit="kg^-1")

        figure, axes = plt.subplots(3, 1, figsize=(12.0, 9.0), sharex=True)
        fields = (
            ("pairwise_cog_distance_m", "CoG pairwise distance [m]"),
            ("pairwise_inertia_over_mass_distance_m2_frobenius", "J/m pairwise distance [m^2], Frobenius"),
            ("pairwise_force_over_mass_distance_kg_inverse", "f/m pairwise distance [kg^-1]"),
        )
        for axis, (field, ylabel) in zip(axes, fields):
            for pair_index, pair_label in enumerate(arrays["pairwise_labels"]):
                axis.plot(np.arange(len(names)), arrays[field][:, pair_index], "o-", ms=3, label=str(pair_label))
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
        axes[0].legend(fontsize=7)
        axes[-1].set_xticks(np.arange(len(names)))
        axes[-1].set_xticklabels(names, rotation=75, ha="right", fontsize=6)
        figure.suptitle("All pairwise cross-bag physical distances")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-directory", type=Path, nargs=3, required=True)
    parser.add_argument("--output-root", type=Path, default=_HERE / "outputs")
    parser.add_argument("--aggregate-run-id", default=None)
    return parser


def run_summary(arguments: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    revision = source_commit(_HERE.parent)
    loaded = [_load_completed_ablation(path) for path in arguments.ablation_directory]
    summary, arrays = build_three_bag_summary(loaded, source_revision=revision)
    directory = output_run_directory(
        arguments.output_root,
        "prior_ablation",
        arguments.aggregate_run_id,
        commit=revision,
    )
    write_json(directory / "prior_ablation_three_bag.json", summary)
    np.savez_compressed(directory / "prior_ablation_three_bag.npz", **arrays)
    write_three_bag_pdf(directory / "prior_ablation_three_bag.pdf", summary, arrays)
    write_json(
        directory / "status.json",
        {
            "status": "completed",
            "source_commit": revision,
            "bag_ids": EXPECTED_BAG_IDS,
            "case_count": len(summary["case_names"]),
        },
    )
    write_json(directory / "arguments.json", vars(arguments))
    return directory, summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    directory, _summary = run_summary(arguments)
    print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
