#!/usr/bin/env python3
"""Monte Carlo PID gain postprocessing from the estimator's native Gaussian.

This is the production uncertainty propagation layer downstream of the physical
parameter estimator.  The estimator's Gaussian approximation is sampled in its
native 13-D common-scale quotient coordinate.  Every sample is decoded through
the estimator chart, propagated through the static Gimbalrotor effectiveness
map, and converted into four PID gain-group scale samples.

The output distribution is not forced back to a Gaussian.  Quantile ranges,
joint covariance/correlation, and the raw Monte Carlo samples are retained.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import yaml


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from grape_param_estim.controller_config import (  # noqa: E402
    PID_GAIN_NAMES,
    PID_GROUPS,
)
from grape_param_estim.gimbalrotor_pid_postprocess import (  # noqa: E402
    PostprocessInputError,
    PostprocessNumericalError,
    build_nominal_controller_allocation,
    characteristic_length,
    load_estimator_result,
    load_vehicle_model,
    source_compatible_pseudoinverse,
)
from gimbalrotor_pid_postprocess_sensitivity import (  # noqa: E402
    COVARIANCE_MODES,
    DEFAULT_COVARIANCE_MODE,
    _json_condition_number,
    _psd_eigendecomposition,
    evaluate_static_scales,
    load_sensitivity_artifacts,
    prepare_sampling_coordinates,
    scale_free_vector,
    source_commit,
    write_json,
)


MONTE_CARLO_POSTPROCESS_SCHEMA = (
    "grape-param-estim/gimbalrotor-pid-monte-carlo-postprocess/v1"
)
DEFAULT_SAMPLE_COUNT = 10000
DEFAULT_SEED = 0
QUANTILE_LEVELS = (0.025, 0.16, 0.5, 0.84, 0.975)
QUANTILE_NAMES = ("q025", "q16", "q50", "q84", "q975")
SCALE_FREE_LABELS = (
    "Jxx_over_m",
    "Jyy_over_m",
    "Jzz_over_m",
    "Jxy_over_m",
    "Jxz_over_m",
    "Jyz_over_m",
    "CoG_x",
    "CoG_y",
    "CoG_z",
    "f1_over_m",
    "f2_over_m",
    "f3_over_m",
    "f4_over_m",
)


@dataclass(frozen=True)
class StaticPostprocessBaseline:
    source_path: Path
    source_commit: str
    estimator_source_commit: str
    estimator_case_name: str
    recorded_gain_source: str
    gains: Mapping[str, Mapping[str, float]]
    center_scales: Mapping[str, float]

    def __post_init__(self) -> None:
        source = Path(self.source_path).expanduser().resolve()
        if not self.source_commit:
            raise PostprocessInputError(
                "static postprocess source_commit must be non-empty"
            )
        if not self.estimator_source_commit:
            raise PostprocessInputError(
                "static postprocess estimator_source_commit must be non-empty"
            )
        if not self.estimator_case_name:
            raise PostprocessInputError(
                "static postprocess estimator_case_name must be non-empty"
            )
        if tuple(self.gains) != tuple(PID_GROUPS):
            raise PostprocessInputError(
                "static postprocess gains must use canonical PID groups"
            )
        if tuple(self.center_scales) != tuple(PID_GROUPS):
            raise PostprocessInputError(
                "static postprocess center scales must use canonical PID groups"
            )
        copied_gains: dict[str, Mapping[str, float]] = {}
        for group in PID_GROUPS:
            group_values = self.gains[group]
            if tuple(group_values) != tuple(PID_GAIN_NAMES):
                raise PostprocessInputError(
                    "static postprocess gain group {!r} is incomplete".format(
                        group
                    )
                )
            selected = {
                gain: float(group_values[gain]) for gain in PID_GAIN_NAMES
            }
            values = np.asarray(list(selected.values()), dtype=float)
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise PostprocessInputError(
                    "recorded controller gains must be finite and non-negative"
                )
            copied_gains[group] = selected
        copied_scales = {
            group: float(self.center_scales[group]) for group in PID_GROUPS
        }
        if np.any(~np.isfinite(np.asarray(list(copied_scales.values())))):
            raise PostprocessInputError(
                "static postprocess center scales must be finite"
            )
        object.__setattr__(self, "source_path", source)
        object.__setattr__(self, "gains", copied_gains)
        object.__setattr__(self, "center_scales", copied_scales)


def _read_json(path: Path, label: str) -> tuple[Path, Mapping[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PostprocessInputError(
            "{} cannot be read: {}".format(label, source)
        ) from error
    if not isinstance(payload, Mapping):
        raise PostprocessInputError("{} must contain a JSON object".format(label))
    return source, payload


def load_static_postprocess_baseline(path: Path) -> StaticPostprocessBaseline:
    source, payload = _read_json(path, "static PID postprocess JSON")
    if payload.get("schema") != "grape-param-estim/gimbalrotor-pid-postprocess/v1":
        raise PostprocessInputError(
            "static PID postprocess JSON has an unsupported schema"
        )
    input_payload = payload.get("input")
    snapshot = payload.get("controller_gain_snapshot")
    groups = payload.get("gain_groups")
    if not isinstance(input_payload, Mapping):
        raise PostprocessInputError("static PID postprocess input is missing")
    if not isinstance(snapshot, Mapping):
        raise PostprocessInputError(
            "static PID postprocess controller_gain_snapshot is missing"
        )
    if not isinstance(groups, Mapping):
        raise PostprocessInputError("static PID postprocess gain_groups is missing")
    gains_payload = snapshot.get("gains")
    if not isinstance(gains_payload, Mapping):
        raise PostprocessInputError(
            "static PID postprocess recorded gain snapshot is missing"
        )
    gains: dict[str, Mapping[str, float]] = {}
    center_scales: dict[str, float] = {}
    for group in PID_GROUPS:
        group_gain = gains_payload.get(group)
        group_report = groups.get(group)
        if not isinstance(group_gain, Mapping) or not isinstance(
            group_report, Mapping
        ):
            raise PostprocessInputError(
                "static PID postprocess is missing group {!r}".format(group)
            )
        gains[group] = {
            gain: float(group_gain[gain]) for gain in PID_GAIN_NAMES
        }
        center_scales[group] = float(group_report["scale"])
    return StaticPostprocessBaseline(
        source_path=source,
        source_commit=str(payload.get("source_commit", "")),
        estimator_source_commit=str(
            input_payload.get("estimator_source_commit", "")
        ),
        estimator_case_name=str(input_payload.get("estimator_case_name", "")),
        recorded_gain_source=str(snapshot.get("source", "unknown")),
        gains=gains,
        center_scales=center_scales,
    )


def _strict_json_matrix(matrix: np.ndarray) -> list[list[Any]]:
    selected = np.asarray(matrix, dtype=float)
    rows: list[list[Any]] = []
    for row in selected:
        output_row: list[Any] = []
        for value in row:
            number = float(value)
            output_row.append(number if np.isfinite(number) else None)
        rows.append(output_row)
    return rows


def _quantile_summary(values: np.ndarray) -> Mapping[str, Any]:
    selected = np.asarray(values, dtype=float)
    if selected.ndim != 1 or selected.size == 0 or np.any(~np.isfinite(selected)):
        raise PostprocessNumericalError(
            "quantile summary requires a non-empty finite vector"
        )
    q = np.quantile(selected, QUANTILE_LEVELS)
    return {
        "mean": float(np.mean(selected)),
        "standard_deviation": float(np.std(selected, ddof=0)),
        "min": float(np.min(selected)),
        "max": float(np.max(selected)),
        "nonpositive_fraction": float(np.mean(selected <= 0.0)),
        "quantiles": {
            name: float(value) for name, value in zip(QUANTILE_NAMES, q)
        },
    }


def _gain_quantiles(
    baseline: StaticPostprocessBaseline,
    scale_summary: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for group in PID_GROUPS:
        scale_quantiles = scale_summary[group]["quantiles"]
        gain_group: dict[str, Any] = {
            "scale": {
                "median": float(scale_quantiles["q50"]),
                "range_68": [
                    float(scale_quantiles["q16"]),
                    float(scale_quantiles["q84"]),
                ],
                "range_95": [
                    float(scale_quantiles["q025"]),
                    float(scale_quantiles["q975"]),
                ],
            },
            "gains": {},
        }
        for gain in PID_GAIN_NAMES:
            old = float(baseline.gains[group][gain])
            gain_group["gains"][gain] = {
                "recorded": old,
                "median": old * float(scale_quantiles["q50"]),
                "range_68": [
                    old * float(scale_quantiles["q16"]),
                    old * float(scale_quantiles["q84"]),
                ],
                "range_95": [
                    old * float(scale_quantiles["q025"]),
                    old * float(scale_quantiles["q975"]),
                ],
            }
        result[group] = gain_group
    return result


def _validate_static_center(
    *,
    baseline: StaticPostprocessBaseline,
    center_scales: Mapping[str, float],
    estimator_source_commit: str,
    estimator_case_name: str,
) -> None:
    if baseline.estimator_source_commit != estimator_source_commit:
        raise PostprocessInputError(
            "static PID postprocess and estimator result use different estimator commits"
        )
    if baseline.estimator_case_name != estimator_case_name:
        raise PostprocessInputError(
            "static PID postprocess and estimator result use different cases"
        )
    for group in PID_GROUPS:
        if not np.isclose(
            float(center_scales[group]),
            float(baseline.center_scales[group]),
            rtol=2.0e-9,
            atol=2.0e-11,
        ):
            raise PostprocessInputError(
                "static PID postprocess center scale does not match current "
                "plant/model for group {!r}".format(group)
            )


def sample_pid_gain_distribution(
    *,
    result_path: Path,
    arrays_path: Path,
    static_postprocess_path: Path,
    vehicle_model_path: Path,
    covariance_mode: str = DEFAULT_COVARIANCE_MODE,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    seed: int = DEFAULT_SEED,
    characteristic_length_override: Optional[float] = None,
) -> tuple[Mapping[str, Any], Mapping[str, np.ndarray]]:
    count = int(sample_count)
    if count <= 0:
        raise PostprocessInputError("sample_count must be positive")
    baseline = load_static_postprocess_baseline(static_postprocess_path)
    result = load_estimator_result(result_path)
    artifacts = load_sensitivity_artifacts(
        arrays_path, covariance_mode=covariance_mode
    )
    model = load_vehicle_model(vehicle_model_path)

    # Production sampling deliberately stays in the estimator's native quotient
    # Gaussian.  No Gaussian is re-fitted after a nonlinear coordinate change.
    sampling = prepare_sampling_coordinates(
        result=result,
        artifacts=artifacts,
        model=model,
        coordinate_mode="estimator_quotient",
    )
    eigenvalues, eigenvectors, psd_tolerance = _psd_eigendecomposition(
        sampling.covariance
    )
    factor = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    nominal = build_nominal_controller_allocation(model)
    nominal_pseudoinverse = source_compatible_pseudoinverse(nominal.matrix)
    length = characteristic_length(model, characteristic_length_override)
    center = evaluate_static_scales(
        sampling.center_plant,
        model,
        nominal_pseudoinverse=nominal_pseudoinverse,
        characteristic_length_m=length,
    )
    _validate_static_center(
        baseline=baseline,
        center_scales=center.scales,
        estimator_source_commit=result.source_commit,
        estimator_case_name=result.case_name,
    )

    rng = np.random.default_rng(int(seed))
    quotient_delta_samples = rng.standard_normal((count, 13)) @ factor.T
    gain_scale_samples = np.full((count, len(PID_GROUPS)), np.nan, dtype=float)
    scale_free_samples = np.full((count, len(SCALE_FREE_LABELS)), np.nan, dtype=float)
    allocation_rank = np.full(count, -1, dtype=int)
    allocation_condition = np.full(count, np.nan, dtype=float)
    valid_mask = np.zeros(count, dtype=bool)
    invalid_examples: list[Mapping[str, str]] = []

    for sample_index in range(count):
        delta = quotient_delta_samples[sample_index]
        try:
            with np.errstate(
                over="raise",
                invalid="raise",
                divide="raise",
                under="raise",
            ):
                plant = sampling.decode(delta)
                evaluation = evaluate_static_scales(
                    plant,
                    model,
                    nominal_pseudoinverse=nominal_pseudoinverse,
                    characteristic_length_m=length,
                )
            gain_scale_samples[sample_index] = np.asarray(
                [evaluation.scales[group] for group in PID_GROUPS],
                dtype=float,
            )
            scale_free_samples[sample_index] = scale_free_vector(plant)
            allocation_rank[sample_index] = (
                evaluation.allocation_source_threshold_rank
            )
            allocation_condition[sample_index] = (
                evaluation.allocation_condition_number
            )
            valid_mask[sample_index] = True
        except (
            PostprocessNumericalError,
            OverflowError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as error:
            if len(invalid_examples) < 50:
                invalid_examples.append(
                    {
                        "sample_index": sample_index,
                        "exception_type": type(error).__name__,
                        "message": str(error),
                    }
                )

    valid_scales = gain_scale_samples[valid_mask]
    if valid_scales.shape[0] == 0:
        raise PostprocessNumericalError(
            "all Monte Carlo PID postprocess samples were numerically undefined"
        )

    scale_summary = {
        group: _quantile_summary(valid_scales[:, index])
        for index, group in enumerate(PID_GROUPS)
    }
    scale_covariance = np.cov(valid_scales, rowvar=False, ddof=0)
    scale_correlation = np.corrcoef(valid_scales, rowvar=False)
    gain_proposal = _gain_quantiles(baseline, scale_summary)

    valid_rank = allocation_rank[valid_mask]
    valid_condition = allocation_condition[valid_mask]
    finite_condition = valid_condition[np.isfinite(valid_condition)]
    rank_histogram = {
        str(rank): int(np.count_nonzero(valid_rank == rank))
        for rank in sorted(set(int(value) for value in valid_rank))
    }
    warnings: list[str] = []
    if np.any(valid_scales <= 0.0):
        warnings.append("nonpositive_gain_scale_samples_present")
    if np.count_nonzero(~valid_mask):
        warnings.append("numerically_undefined_samples_present")
    if np.any(valid_rank < 6):
        warnings.append("source_threshold_rank_loss_present")
    if np.any(np.isposinf(valid_condition)):
        warnings.append("infinite_allocation_condition_present")

    report = {
        "schema": MONTE_CARLO_POSTPROCESS_SCHEMA,
        "source_commit": source_commit(),
        "method": "native_quotient_gaussian_monte_carlo_static_pid_pushforward",
        "input": {
            "estimator_result_json": str(Path(result_path).expanduser().resolve()),
            "arrays_npz": str(Path(arrays_path).expanduser().resolve()),
            "static_pid_postprocess_json": str(baseline.source_path),
            "vehicle_model_json": str(Path(vehicle_model_path).expanduser().resolve()),
            "estimator_source_commit": result.source_commit,
            "estimator_case_name": result.case_name,
            "covariance_mode": artifacts.covariance_mode,
            "covariance_source": artifacts.covariance_source,
            "coordinate_mode": "estimator_quotient",
            "sample_count": count,
            "seed": int(seed),
        },
        "distribution_contract": {
            "input_distribution": (
                "Gaussian approximation in the estimator's native 13-D "
                "common-scale quotient coordinate"
            ),
            "output_distribution": (
                "empirical nonlinear Monte Carlo push-forward; no Gaussian "
                "refit is applied to the PID gain scales"
            ),
            "quantile_ranges": {
                "range_68": [0.16, 0.84],
                "range_95": [0.025, 0.975],
            },
            "rotor_lag_treatment": (
                "held fixed at the fitted value because the static PID map "
                "does not model delay"
            ),
        },
        "center": {
            "scales": {
                group: float(center.scales[group]) for group in PID_GROUPS
            },
            "recorded_gains": {
                group: dict(baseline.gains[group]) for group in PID_GROUPS
            },
            "static_postprocess_source_commit": baseline.source_commit,
            "recorded_gain_source": baseline.recorded_gain_source,
        },
        "estimator_covariance": {
            "eigenvalues_descending": eigenvalues.tolist(),
            "psd_negative_tolerance": psd_tolerance,
        },
        "sampling": {
            "requested_count": count,
            "valid_count": int(np.count_nonzero(valid_mask)),
            "invalid_count": int(np.count_nonzero(~valid_mask)),
            "valid_fraction": float(np.mean(valid_mask)),
            "invalid_examples": invalid_examples,
        },
        "gain_scale_distribution": scale_summary,
        "joint_gain_scale_distribution": {
            "group_order": list(PID_GROUPS),
            "covariance": _strict_json_matrix(scale_covariance),
            "correlation": _strict_json_matrix(scale_correlation),
        },
        "pid_gain_proposal": gain_proposal,
        "allocation_diagnostics": {
            "source_threshold_rank_histogram": rank_histogram,
            "infinite_condition_fraction": float(
                np.mean(np.isposinf(valid_condition))
            ),
            "max_finite_condition_number": (
                float(np.max(finite_condition))
                if finite_condition.size
                else None
            ),
        },
        "warnings": warnings,
    }
    samples = {
        "quotient_delta_samples": quotient_delta_samples,
        "gain_scale_samples": gain_scale_samples,
        "scale_free_samples": scale_free_samples,
        "valid_mask": valid_mask,
        "A_real_source_threshold_rank": allocation_rank,
        "A_real_condition_number": allocation_condition,
        "pid_group_order": np.asarray(PID_GROUPS),
        "scale_free_labels": np.asarray(SCALE_FREE_LABELS),
    }
    return report, samples


def _proposal_overlay(report: Mapping[str, Any]) -> Mapping[str, Any]:
    controller: dict[str, Any] = {}
    proposal = report["pid_gain_proposal"]
    for group in PID_GROUPS:
        controller[group] = {
            gain: float(proposal[group]["gains"][gain]["median"])
            for gain in PID_GAIN_NAMES
        }
    return {"controller": controller}


def _range_yaml(report: Mapping[str, Any]) -> Mapping[str, Any]:
    groups: dict[str, Any] = {}
    proposal = report["pid_gain_proposal"]
    for group in PID_GROUPS:
        groups[group] = {
            "scale": dict(proposal[group]["scale"]),
            "gains": {
                gain: dict(proposal[group]["gains"][gain])
                for gain in PID_GAIN_NAMES
            },
        }
    return {
        "schema": MONTE_CARLO_POSTPROCESS_SCHEMA + "/proposal-ranges/v1",
        "covariance_mode": report["input"]["covariance_mode"],
        "sample_count": report["sampling"]["requested_count"],
        "valid_sample_count": report["sampling"]["valid_count"],
        "seed": report["input"]["seed"],
        "controller": groups,
        "warnings": list(report["warnings"]),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Gimbalrotor PID Monte Carlo proposal",
        "",
        "The estimator Gaussian is sampled in its native common-scale quotient",
        "coordinate and propagated through the nonlinear static PID map. The PID",
        "output is retained as an empirical distribution rather than re-Gaussianized.",
        "",
        "- covariance mode: `{}`".format(report["input"]["covariance_mode"]),
        "- coordinate mode: `estimator_quotient`",
        "- samples: `{}/{}` valid".format(
            report["sampling"]["valid_count"],
            report["sampling"]["requested_count"],
        ),
        "- seed: `{}`".format(report["input"]["seed"]),
        "",
        "## Gain-scale distribution",
        "",
        "| group | point | median | 16–84% | 2.5–97.5% | std | nonpositive |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in PID_GROUPS:
        center = report["center"]["scales"][group]
        item = report["gain_scale_distribution"][group]
        q = item["quantiles"]
        lines.append(
            "| {} | {:.8g} | {:.8g} | [{:.8g}, {:.8g}] | [{:.8g}, {:.8g}] | {:.8g} | {:.3%} |".format(
                group,
                center,
                q["q50"],
                q["q16"],
                q["q84"],
                q["q025"],
                q["q975"],
                item["standard_deviation"],
                item["nonpositive_fraction"],
            )
        )
    lines.extend(("", "## Proposed PID gains", ""))
    for group in PID_GROUPS:
        lines.extend(
            (
                "### {}".format(group),
                "",
                "| gain | recorded | median proposal | 16–84% | 2.5–97.5% |",
                "|---|---:|---:|---:|---:|",
            )
        )
        for gain in PID_GAIN_NAMES:
            item = report["pid_gain_proposal"][group]["gains"][gain]
            lines.append(
                "| {} | {:.8g} | {:.8g} | [{:.8g}, {:.8g}] | [{:.8g}, {:.8g}] |".format(
                    gain,
                    item["recorded"],
                    item["median"],
                    item["range_68"][0],
                    item["range_68"][1],
                    item["range_95"][0],
                    item["range_95"][1],
                )
            )
        lines.append("")
    lines.extend(
        (
            "## Joint gain-scale correlation",
            "",
            "| | {} |".format(" | ".join(PID_GROUPS)),
            "|---|{}|".format("|".join("---:" for _ in PID_GROUPS)),
        )
    )
    correlation = report["joint_gain_scale_distribution"]["correlation"]
    for row_index, group in enumerate(PID_GROUPS):
        cells = []
        for value in correlation[row_index]:
            cells.append("undefined" if value is None else "{:.5g}".format(value))
        lines.append("| {} | {} |".format(group, " | ".join(cells)))
    diagnostics = report["allocation_diagnostics"]
    lines.extend(
        (
            "",
            "## Numerical diagnostics",
            "",
            "- source-threshold rank histogram: `{}`".format(
                diagnostics["source_threshold_rank_histogram"]
            ),
            "- infinite allocation-condition fraction: `{:.6g}`".format(
                diagnostics["infinite_condition_fraction"]
            ),
            "- max finite allocation condition number: `{}`".format(
                diagnostics["max_finite_condition_number"]
            ),
            "- warnings: `{}`".format(report["warnings"]),
            "",
            "The ranges above are empirical quantiles of the selected estimator",
            "Gaussian approximation after nonlinear PID postprocessing. No sample is",
            "discarded merely because `A_real` loses the source SVD-threshold rank or",
            "has a large condition number; only a genuinely undefined/non-finite",
            "floating-point evaluation is marked invalid.",
            "",
        )
    )
    return "\n".join(lines)


def write_outputs(
    *,
    output_dir: Path,
    report: Mapping[str, Any],
    samples: Mapping[str, np.ndarray],
) -> None:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "pid_gain_monte_carlo_postprocess.json", report)
    (directory / "pid_gain_monte_carlo_postprocess.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    np.savez_compressed(
        directory / "pid_gain_monte_carlo_samples.npz", **samples
    )
    (directory / "pid_gain_median_overlay.yaml").write_text(
        yaml.safe_dump(
            _proposal_overlay(report), sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8",
    )
    (directory / "pid_gain_proposal_ranges.yaml").write_text(
        yaml.safe_dump(
            _range_yaml(report), sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8",
    )
    write_json(
        directory / "status.json",
        {
            "schema": MONTE_CARLO_POSTPROCESS_SCHEMA + "/status/v1",
            "status": "completed",
            "source_commit": report["source_commit"],
            "covariance_mode": report["input"]["covariance_mode"],
            "sample_count": report["sampling"]["requested_count"],
            "valid_sample_count": report["sampling"]["valid_count"],
            "invalid_sample_count": report["sampling"]["invalid_count"],
            "warnings": list(report["warnings"]),
        },
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample the estimator's native quotient Gaussian and produce "
            "quantile-based PID gain proposals."
        )
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--arrays",
        type=Path,
        help="Estimator arrays.npz; defaults to arrays.npz next to --result.",
    )
    parser.add_argument("--static-postprocess", type=Path, required=True)
    parser.add_argument("--vehicle-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--covariance-mode",
        choices=COVARIANCE_MODES,
        default=DEFAULT_COVARIANCE_MODE,
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--characteristic-length", type=float)
    return parser


def execute(arguments: argparse.Namespace) -> Mapping[str, Any]:
    result_path = Path(arguments.result).expanduser().resolve()
    arrays_path = (
        Path(arguments.arrays).expanduser().resolve()
        if arguments.arrays is not None
        else result_path.parent / "arrays.npz"
    )
    report, samples = sample_pid_gain_distribution(
        result_path=result_path,
        arrays_path=arrays_path,
        static_postprocess_path=arguments.static_postprocess,
        vehicle_model_path=arguments.vehicle_model,
        covariance_mode=arguments.covariance_mode,
        sample_count=arguments.samples,
        seed=arguments.seed,
        characteristic_length_override=arguments.characteristic_length,
    )
    write_outputs(output_dir=arguments.output_dir, report=report, samples=samples)
    print(render_markdown(report), end="")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        execute(arguments)
    except PostprocessInputError as error:
        print("input error: {}".format(error), file=sys.stderr)
        return 2
    except PostprocessNumericalError as error:
        print("numerical error: {}".format(error), file=sys.stderr)
        return 3
    except OSError as error:
        print("output error: {}".format(error), file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
