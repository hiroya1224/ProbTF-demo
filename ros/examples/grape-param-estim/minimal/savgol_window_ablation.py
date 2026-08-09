#!/usr/bin/env python3
"""Window-width ablation for deterministic geometric Savitzky-Golay dynamics.

Each W is a fully independent estimator run with its own JSON/PDF/NPZ output.
The default study is the data-supported minimum W plus 0.5, 1.0, 1.5, 2.0 s.
Requested widths below the data-supported minimum are rejected before any
parameter optimization and recorded as skipped cases.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

import deterministic_savgol_dynamics_estimator as estimator
import savgol_trajectory as sg
import savgol_dynamics_confidence as confidence


SCHEMA = "grape-param-estim/savgol-window-ablation/v1"
DEFAULT_WINDOWS_SECONDS = (0.5, 1.0, 1.5, 2.0)


def _sanitize(value: Any) -> Any:
    return estimator._json_sanitize(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _sanitize(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _window_label(window_seconds: float) -> str:
    text = "{:.9f}".format(float(window_seconds)).rstrip("0").rstrip(".")
    return "W_{}s".format(text.replace(".", "p"))


def _deduplicated(values: Sequence[float]) -> list[float]:
    result: list[float] = []
    for value in sorted(float(v) for v in values):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("ablation windows must be finite and positive")
        if not result or not np.isclose(
            value,
            result[-1],
            atol=1.0e-9,
            rtol=1.0e-9,
        ):
            result.append(value)
    return result


def _minimum_windows(
    config_path: Path,
) -> tuple[float, list[dict[str, Any]]]:
    config = estimator.base.multi.load_multi_bag_config(
        config_path.expanduser().resolve()
    )
    per_bag: list[dict[str, Any]] = []
    for specification in config.bags:
        flight = estimator.load_flight_data(
            str(specification.path),
            start_local=specification.start,
            end_local=specification.end,
            include_fc_specific_force=True,
            compute_sha256=False,
        )
        pose_time = np.asarray(flight.pose.times, dtype=float)
        minimum = sg.minimum_feasible_window_seconds(pose_time)
        intervals = np.diff(pose_time)
        per_bag.append(
            {
                "id": specification.bag_id,
                "minimum_feasible_window_seconds": minimum,
                "raw_pose_sample_count": int(pose_time.size),
                "raw_pose_interval_seconds": {
                    "minimum": float(np.min(intervals)),
                    "median": float(np.median(intervals)),
                    "maximum": float(np.max(intervals)),
                },
            }
        )
    if not per_bag:
        raise ValueError("ablation config contains no bags")
    global_minimum = max(
        float(item["minimum_feasible_window_seconds"])
        for item in per_bag
    )
    return global_minimum, per_bag


def _weighted_bag_metric(
    diagnostics: Sequence[Mapping[str, Any]],
    path: Sequence[str],
) -> float:
    values = []
    weights = []
    for bag in diagnostics:
        current: Any = bag
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                current = None
                break
            current = current[key]
        if current is None:
            continue
        value = float(current)
        weight = float(bag.get("normalized_weight", 1.0))
        if np.isfinite(value) and np.isfinite(weight) and weight > 0.0:
            values.append(value)
            weights.append(weight)
    if not values:
        return math.nan
    weights_array = np.asarray(weights, dtype=float)
    weights_array /= np.sum(weights_array)
    return float(weights_array @ np.asarray(values, dtype=float))


def _weighted_wrench_rms(
    diagnostics: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    values = []
    weights = []
    for bag in diagnostics:
        statistics = bag.get("residual_wrench_statistics")
        if not isinstance(statistics, Mapping):
            continue
        value = np.asarray(statistics.get("rmse"), dtype=float)
        weight = float(bag.get("normalized_weight", 1.0))
        if (
            value.shape == (6,)
            and np.all(np.isfinite(value))
            and np.isfinite(weight)
            and weight > 0.0
        ):
            values.append(value)
            weights.append(weight)
    if not values:
        return np.full(6, np.nan, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    weights_array /= np.sum(weights_array)
    return np.einsum(
        "n,nj->j",
        weights_array,
        np.asarray(values, dtype=float),
    )


def _weighted_raw_residual_wrench_rms(
    diagnostics: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    values = []
    weights = []
    for bag in diagnostics:
        statistics = bag.get(
            "raw_inverse_dynamics_residual_wrench_statistics"
        )
        if not isinstance(statistics, Mapping):
            continue
        value = np.asarray(statistics.get("rms"), dtype=float)
        weight = float(bag.get("normalized_weight", 1.0))
        if (
            value.shape == (6,)
            and np.all(np.isfinite(value))
            and np.isfinite(weight)
            and weight > 0.0
        ):
            values.append(value)
            weights.append(weight)
    if not values:
        return np.full(6, np.nan, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    weights_array /= np.sum(weights_array)
    return np.einsum(
        "n,nj->j",
        weights_array,
        np.asarray(values, dtype=float),
    )


def _summary_from_result(
    window_seconds: float,
    result_path: Path,
) -> dict[str, Any]:
    root = json.loads(result_path.read_text(encoding="utf-8"))
    selection = root["selection"]
    parameters = selection["parameters"]
    inertia = np.asarray(parameters["inertia_kg_m2"], dtype=float)
    cog_raw = parameters.get(
        "cog_position_body_m",
        parameters.get("cog_offset_m"),
    )
    diagnostics = tuple(root.get("bag_diagnostics", ()))
    rotor_median_interval = math.nan
    gimbal_median_interval = math.nan
    if diagnostics:
        timing = diagnostics[0].get("command_timestamp_intervals")
        if isinstance(timing, Mapping):
            rotor = timing.get("rotor")
            gimbal = timing.get("gimbal")
            if isinstance(rotor, Mapping) and rotor.get("median_seconds") is not None:
                rotor_median_interval = float(rotor["median_seconds"])
            if isinstance(gimbal, Mapping) and gimbal.get("median_seconds") is not None:
                gimbal_median_interval = float(gimbal["median_seconds"])
    return {
        "status": "completed",
        "window_seconds": float(window_seconds),
        "result_json": str(result_path),
        "elapsed_seconds": float(root.get("elapsed_seconds", math.nan)),
        "selected_delay_seconds": float(selection["delay_seconds"]),
        "rotor_command_median_interval_seconds": rotor_median_interval,
        "gimbal_command_median_interval_seconds": gimbal_median_interval,
        "joint_dynamics_loss": float(selection["joint_dynamics_loss"]),
        "gaussian_prior_cost": float(selection["gaussian_prior_cost"]),
        "joint_objective_cost": float(selection["joint_objective_cost"]),
        "mass_kg": float(parameters["mass_kg"]),
        "cog_position_body_m": np.asarray(cog_raw, dtype=float),
        "inertia_kg_m2": inertia,
        "inertia_principal_moments_kg_m2": np.linalg.eigvalsh(inertia),
        "force_effectiveness": np.asarray(
            parameters["force_effectiveness"],
            dtype=float,
        ),
        "weighted_external_wrench_rms": _weighted_wrench_rms(diagnostics),
        "weighted_raw_residual_wrench_rms": (
            _weighted_raw_residual_wrench_rms(diagnostics)
        ),
        "weighted_free_rollout_position_rmse_m": _weighted_bag_metric(
            diagnostics,
            ("estimated_forward_metrics", "position_rmse_m"),
        ),
        "weighted_free_rollout_orientation_rmse_deg": _weighted_bag_metric(
            diagnostics,
            ("estimated_forward_metrics", "orientation_angle_rmse_deg"),
        ),
        "weighted_replay_position_rmse_m": _weighted_bag_metric(
            diagnostics,
            (
                "estimated_with_external_wrench_forward_metrics",
                "position_rmse_m",
            ),
        ),
        "weighted_replay_orientation_rmse_deg": _weighted_bag_metric(
            diagnostics,
            (
                "estimated_with_external_wrench_forward_metrics",
                "orientation_angle_rmse_deg",
            ),
        ),
    }


def _write_ablation_pdf(path: Path, completed: Sequence[Mapping[str, Any]]) -> None:
    if not completed:
        return
    plt = estimator.base.plt
    PdfPages = estimator.base.PdfPages
    ordered = sorted(completed, key=lambda item: float(item["window_seconds"]))
    windows = np.asarray([item["window_seconds"] for item in ordered], dtype=float)

    def vector(key: str) -> np.ndarray:
        return np.asarray([item[key] for item in ordered], dtype=float)

    with PdfPages(path) as pdf:
        figure, axes = plt.subplots(2, 2, figsize=(11.7, 8.3), constrained_layout=True)
        axes[0, 0].plot(windows, vector("joint_dynamics_loss"), marker="o")
        axes[0, 0].set_ylabel("data dynamics loss")
        axes[0, 1].plot(windows, vector("gaussian_prior_cost"), marker="o")
        axes[0, 1].set_ylabel("Gaussian prior cost")
        axes[1, 0].plot(windows, vector("joint_objective_cost"), marker="o")
        axes[1, 0].set_ylabel("joint objective cost")
        axes[1, 1].plot(windows, 1000.0 * vector("selected_delay_seconds"), marker="o", label="selected lag")
        rotor_period = vector("rotor_command_median_interval_seconds")
        gimbal_period = vector("gimbal_command_median_interval_seconds")
        if np.any(np.isfinite(rotor_period)):
            axes[1, 1].axhline(
                1000.0 * float(rotor_period[np.flatnonzero(np.isfinite(rotor_period))[0]]),
                linestyle="--",
                label="rotor median publish dt",
            )
        if np.any(np.isfinite(gimbal_period)):
            axes[1, 1].axhline(
                1000.0 * float(gimbal_period[np.flatnonzero(np.isfinite(gimbal_period))[0]]),
                linestyle=":",
                label="gimbal median publish dt",
            )
        axes[1, 1].set_xlabel("W [s]")
        axes[1, 1].set_ylabel("time [ms]")
        axes[1, 1].legend(loc="best")
        for axis in axes.ravel():
            axis.grid(True, alpha=0.25)
            if axis is not axes[1, 1]:
                axis.set_xlabel("W [s]")
        figure.suptitle("Savitzky-Golay window ablation: objective and lag")
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(2, 2, figsize=(11.7, 8.3), constrained_layout=True)
        axes[0, 0].plot(windows, vector("mass_kg"), marker="o")
        axes[0, 0].set_ylabel("mass [kg]")
        cog = np.asarray([item["cog_position_body_m"] for item in ordered], dtype=float)
        for index, label in enumerate(("x", "y", "z")):
            axes[0, 1].plot(windows, cog[:, index], marker="o", label=label)
        axes[0, 1].set_ylabel("CoG body position [m]")
        axes[0, 1].legend(loc="best")
        principal = np.asarray(
            [item["inertia_principal_moments_kg_m2"] for item in ordered],
            dtype=float,
        )
        for index in range(3):
            axes[1, 0].plot(
                windows,
                principal[:, index],
                marker="o",
                label="J{}".format(index + 1),
            )
        axes[1, 0].set_ylabel("principal inertia [kg m2]")
        axes[1, 0].legend(loc="best")
        effectiveness = np.asarray(
            [item["force_effectiveness"] for item in ordered],
            dtype=float,
        )
        for index in range(4):
            axes[1, 1].plot(
                windows,
                effectiveness[:, index],
                marker="o",
                label="rotor {}".format(index + 1),
            )
        axes[1, 1].set_ylabel("force effectiveness")
        axes[1, 1].legend(loc="best")
        for axis in axes.ravel():
            axis.set_xlabel("W [s]")
            axis.grid(True, alpha=0.25)
        figure.suptitle("Savitzky-Golay window ablation: physical parameters")
        pdf.savefig(figure)
        plt.close(figure)

        for key, title in (
            (
                "weighted_raw_residual_wrench_rms",
                "Raw inverse-dynamics residual wrench (no replay grid)",
            ),
            (
                "weighted_external_wrench_rms",
                "Trajectory-fitted replay external wrench",
            ),
        ):
            wrench = np.asarray(
                [item[key] for item in ordered],
                dtype=float,
            )
            figure, axes = plt.subplots(
                2, 1, figsize=(11.7, 8.3), sharex=True, constrained_layout=True
            )
            for index, label in enumerate(("Fx", "Fy", "Fz")):
                axes[0].plot(windows, wrench[:, index], marker="o", label=label)
            for index, label in enumerate(("Mx", "My", "Mz"), start=3):
                axes[1].plot(windows, wrench[:, index], marker="o", label=label)
            axes[0].set_ylabel("force RMS [N]")
            axes[1].set_ylabel("torque RMS [N m]")
            axes[1].set_xlabel("W [s]")
            for axis in axes:
                axis.grid(True, alpha=0.25)
                axis.legend(loc="best")
            figure.suptitle(
                "Savitzky-Golay window ablation: {}".format(title)
            )
            pdf.savefig(figure)
            plt.close(figure)

        figure, axes = plt.subplots(2, 2, figsize=(11.7, 8.3), constrained_layout=True)
        fields = (
            ("weighted_free_rollout_position_rmse_m", "free position RMSE [m]"),
            ("weighted_free_rollout_orientation_rmse_deg", "free orientation RMSE [deg]"),
            ("weighted_replay_position_rmse_m", "replay position RMSE [m]"),
            ("weighted_replay_orientation_rmse_deg", "replay orientation RMSE [deg]"),
        )
        for axis, (field, label) in zip(axes.ravel(), fields):
            axis.plot(windows, vector(field), marker="o")
            axis.set_xlabel("W [s]")
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.25)
        figure.suptitle("Savitzky-Golay window ablation: rollout validation")
        pdf.savefig(figure)
        plt.close(figure)

        confidence_cases = [
            item for item in ordered
            if item.get("confidence_status") == "completed"
        ]
        if confidence_cases:
            confidence_windows = np.asarray(
                [item["window_seconds"] for item in confidence_cases],
                dtype=float,
            )
            weakest = np.asarray(
                [
                    item["weakest_relative_information_strength"]
                    for item in confidence_cases
                ],
                dtype=float,
            )
            rank = np.asarray(
                [item["data_information_numerical_rank"] for item in confidence_cases],
                dtype=float,
            )
            counts = np.asarray(
                [item["confidence_disjoint_window_count"] for item in confidence_cases],
                dtype=float,
            )
            figure, axes = plt.subplots(3, 1, figsize=(11.7, 8.3), sharex=True, constrained_layout=True)
            axes[0].semilogy(confidence_windows, weakest, marker="o")
            axes[0].set_ylabel("weakest relative\ninformation")
            axes[1].plot(confidence_windows, rank, marker="o")
            axes[1].set_ylabel("data rank")
            axes[2].plot(confidence_windows, counts, marker="o")
            axes[2].set_ylabel("disjoint SG\nconfidence windows")
            axes[2].set_xlabel("W [s]")
            for axis in axes:
                axis.grid(True, alpha=0.25)
            figure.suptitle("Savitzky-Golay window ablation: confidence/ridge diagnostics")
            pdf.savefig(figure)
            plt.close(figure)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent degree-5 geometric Savitzky-Golay parameter "
            "estimates over a data-derived minimum W and selected window widths."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vehicle-model-json", type=Path, required=True)
    parser.add_argument("--prior-json", type=Path, required=True)
    parser.add_argument(
        "--windows-seconds",
        type=float,
        nargs="*",
        default=DEFAULT_WINDOWS_SECONDS,
        help=(
            "Additional W values. The data-supported minimum W is always "
            "included automatically. Default: 0.5 1.0 1.5 2.0."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "savgol_window_ablation",
    )
    parser.add_argument(
        "--skip-confidence",
        action="store_true",
        help=(
            "Skip the per-W confidence/ridge report. By default the runner "
            "reuses each deterministic result and also writes confidence.pdf, "
            "confidence.json, parameter_likelihood.json and parameter_posterior.json."
        ),
    )
    return parser


def run(arguments: argparse.Namespace, passthrough: Sequence[str] = ()) -> int:
    started = time.perf_counter()
    output_directory = arguments.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    global_minimum, per_bag_minimum = _minimum_windows(arguments.config)
    requested = _deduplicated(
        (global_minimum,) + tuple(float(v) for v in arguments.windows_seconds)
    )
    tolerance = max(1.0e-10, 1.0e-8 * global_minimum)

    print(
        "data-supported minimum degree-5 SG window: {:.12g}s".format(
            global_minimum
        ),
        flush=True,
    )

    cases: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    for window in requested:
        label = _window_label(window)
        run_root = output_directory / label
        if window < global_minimum - tolerance:
            case = {
                "status": "skipped_below_minimum",
                "window_seconds": float(window),
                "minimum_feasible_window_seconds": float(global_minimum),
            }
            cases.append(case)
            print(
                "skip W={:.12g}s (< W_min={:.12g}s)".format(
                    window,
                    global_minimum,
                ),
                flush=True,
            )
            continue

        print("running W={:.12g}s".format(window), flush=True)
        core_argv = [
            "--config",
            str(arguments.config),
            "--vehicle-model-json",
            str(arguments.vehicle_model_json),
            "--prior-json",
            str(arguments.prior_json),
            "--window-seconds",
            "{:.17g}".format(window),
            "--output-dir",
            str(run_root),
        ] + list(passthrough)
        estimator.main(core_argv)
        result_path = (
            run_root
            / estimator.OUTPUT_SUBDIRECTORY
            / "result.json"
        )
        if not result_path.is_file():
            raise RuntimeError(
                "W={} estimator did not produce {}".format(
                    window,
                    result_path,
                )
            )
        case = _summary_from_result(window, result_path)

        if not arguments.skip_confidence:
            if len(per_bag_minimum) != 1:
                case["confidence_status"] = (
                    "skipped_requires_single_bag_config"
                )
            else:
                bag_id = str(per_bag_minimum[0]["id"])
                deterministic_npz = (
                    run_root
                    / estimator.OUTPUT_SUBDIRECTORY
                    / "bags"
                    / bag_id
                    / "savgol_dynamics.npz"
                )
                with np.load(deterministic_npz, allow_pickle=False) as archive:
                    deterministic_centers = np.asarray(
                        archive["collocation_time"], dtype=float
                    )
                try:
                    disjoint_indices = confidence._nonoverlapping_window_indices(
                        deterministic_centers, window
                    )
                except ValueError as error:
                    case["confidence_status"] = (
                        "skipped_insufficient_disjoint_windows"
                    )
                    case["confidence_skip_reason"] = str(error)
                else:
                    confidence_argv = [
                        "--config",
                        str(arguments.config),
                        "--vehicle-model-json",
                        str(arguments.vehicle_model_json),
                        "--prior-json",
                        str(arguments.prior_json),
                        "--window-seconds",
                        "{:.17g}".format(window),
                        "--output-dir",
                        str(run_root),
                        "--deterministic-result",
                        str(result_path),
                    ] + list(passthrough)
                    confidence.main(confidence_argv)
                    confidence_directory = (
                        run_root
                        / confidence.OUTPUT_SUBDIRECTORY
                        / bag_id
                    )
                    confidence_json = confidence_directory / "confidence.json"
                    if not confidence_json.is_file():
                        raise RuntimeError(
                            "W={} confidence run did not produce {}".format(
                                window, confidence_json
                            )
                        )
                    confidence_payload = json.loads(
                        confidence_json.read_text(encoding="utf-8")
                    )
                    svd_payload = confidence_payload["data_information"]["svd"]
                    posterior_physical = confidence_payload[
                        "prior_and_local_posterior"
                    ]["posterior_physical"]
                    case.update(
                        {
                            "confidence_status": "completed",
                            "confidence_json": str(confidence_json),
                            "confidence_pdf": str(confidence_directory / "confidence.pdf"),
                            "parameter_likelihood_json": str(
                                confidence_directory / "parameter_likelihood.json"
                            ),
                            "parameter_posterior_json": str(
                                confidence_directory / "parameter_posterior.json"
                            ),
                            "confidence_disjoint_window_count": int(
                                confidence_payload["bag"][
                                    "confidence_disjoint_window_count"
                                ]
                            ),
                            "preflight_disjoint_window_count": int(
                                disjoint_indices.size
                            ),
                            "data_information_numerical_rank": int(
                                svd_payload["numerical_rank"]
                            ),
                            "weakest_relative_information_strength": float(
                                svd_payload[
                                    "weakest_relative_information_strength"
                                ]
                            ),
                            "posterior_physical_std": np.asarray(
                                posterior_physical["std"], dtype=float
                            ),
                        }
                    )

        cases.append(case)
        completed.append(case)

    payload = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(arguments.config.expanduser().resolve()),
        "vehicle_model_path": str(
            arguments.vehicle_model_json.expanduser().resolve()
        ),
        "parameter_prior_path": str(
            arguments.prior_json.expanduser().resolve()
        ),
        "polynomial_degree": sg.POLYNOMIAL_DEGREE,
        "minimum_required_pose_samples_per_window": sg.MINIMUM_WINDOW_POINTS,
        "global_minimum_feasible_window_seconds": global_minimum,
        "per_bag_window_support": per_bag_minimum,
        "requested_windows_seconds": requested,
        "cases": cases,
        "elapsed_seconds": float(time.perf_counter() - started),
        "notes": {
            "minimum_window_covariance": (
                "A degree-5 fit needs six samples. With exactly six samples "
                "the derivatives are defined but OLS residual covariance has "
                "zero residual degrees of freedom and is reported unavailable."
            ),
            "independence": (
                "Each completed W is a separate full estimator run with its "
                "own JSON/PDF/NPZ outputs."
            ),
            "confidence": (
                "Unless --skip-confidence is given, each deterministic W result "
                "is reused to generate the SG confidence/ridge PDF and "
                "likelihood/posterior JSON without re-running the physical "
                "parameter optimizer."
            ),
        },
    }
    _write_json(output_directory / "ablation.json", payload)
    _write_ablation_pdf(output_directory / "ablation.pdf", completed)
    print(
        "wrote {}".format(output_directory / "ablation.json"),
        flush=True,
    )
    if completed:
        print(
            "wrote {}".format(output_directory / "ablation.pdf"),
            flush=True,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_argument_parser()
    arguments, passthrough = parser.parse_known_args(argv)
    forbidden = {"--window-seconds", "--output-dir"}
    if any(token.split("=", 1)[0] in forbidden for token in passthrough):
        raise SystemExit(
            "--window-seconds/--output-dir are controlled by the ablation runner"
        )
    try:
        return run(arguments, passthrough)
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    sys.exit(main())
