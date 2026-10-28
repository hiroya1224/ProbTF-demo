#!/usr/bin/env python3
"""Bag-wise PID gain-region exploration from one first-order estimate JSON."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_MINIMAL = _HERE.parent
_PROJECT_ROOT = _MINIMAL.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
for _path in (_HERE, _MINIMAL, _SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from core import (  # noqa: E402
    COVARIANCE_NAMES,
    GAIN_REGION_SCHEMA,
    PID_GROUPS,
    actuator_parameters_from_estimate,
    draw_quotient_coordinates,
    load_estimate_json,
    quotient_to_scale_free_plants,
)
from grape_param_estim.controller import ControllerConfig  # noqa: E402
from grape_param_estim.controller_config import (  # noqa: E402
    PidGainConfiguration,
    apply_pid_gain_configuration,
)
from grape_param_estim.gimbalrotor_pid_postprocess import (  # noqa: E402
    ScaleFreePlant,
    load_vehicle_model,
)
from gimbalrotor_pid_local_pole_validation import (  # noqa: E402
    NUMERICAL_SAMPLE_EXCEPTIONS,
    _analyze_plant,
    decompose_thrust_delay,
)
from single_bag_savgol_reports import write_json  # noqa: E402


DEFAULT_ALPHA = 0.95
DEFAULT_SAMPLE_COUNT = 512
DEFAULT_SEED = 0
DEFAULT_FD_RATIO = 1.12
DEFAULT_GRID_SIZE = 21
DEFAULT_SCALE_MIN = 0.35
DEFAULT_SCALE_MAX = 3.0
PROJECTION_SPECS = (
    ("pi", 0, 1, 2, "P", "I"),
    ("id", 1, 2, 0, "I", "D"),
    ("dp", 2, 0, 1, "D", "P"),
)


@dataclass(frozen=True)
class GainTriple:
    p: float
    i: float
    d: float

    def array(self) -> np.ndarray:
        result = np.asarray((self.p, self.i, self.d), dtype=float)
        if np.any(~np.isfinite(result)) or np.any(result < 0.0):
            raise ValueError("PID gains must be finite and non-negative")
        return result


def _recorded_gains(estimate: Mapping[str, Any]) -> Mapping[str, GainTriple]:
    payload = estimate["controller"]["gains"]
    return {
        group: GainTriple(
            float(payload[group]["p_gain"]),
            float(payload[group]["i_gain"]),
            float(payload[group]["d_gain"]),
        )
        for group in PID_GROUPS
    }


def _configuration(gains: Mapping[str, GainTriple]) -> Any:
    values = np.asarray([gains[group].array() for group in PID_GROUPS])
    return apply_pid_gain_configuration(
        ControllerConfig.grape(), PidGainConfiguration(values)
    )


def _replace_group(
    gains: Mapping[str, GainTriple],
    group: str,
    triple: GainTriple,
) -> Mapping[str, GainTriple]:
    result = dict(gains)
    result[group] = triple
    return result


def _inputs(
    *,
    vehicle_model: Any,
    actuator_parameters: Any,
    gains: Mapping[str, GainTriple],
) -> Any:
    return SimpleNamespace(
        vehicle_model=vehicle_model,
        actuator_parameters=actuator_parameters,
        controller_configuration=_configuration(gains),
    )


def _spectral_radii(
    *,
    plants: Sequence[ScaleFreePlant],
    vehicle_model: Any,
    actuator_parameters: Any,
    controller_dt: float,
    gains: Mapping[str, GainTriple],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full(len(plants), np.nan, dtype=float)
    valid = np.zeros(len(plants), dtype=bool)
    selected_inputs = _inputs(
        vehicle_model=vehicle_model,
        actuator_parameters=actuator_parameters,
        gains=gains,
    )
    delay = decompose_thrust_delay(0.0, controller_dt)
    for index, plant in enumerate(plants):
        try:
            result = _analyze_plant(
                scale_free=plant,
                inputs=selected_inputs,
                controller_dt=controller_dt,
                delay=delay,
                fd_check=False,
            )
            classification = result["classification"]
            if classification is None:
                continue
            values[index] = float(classification["spectral_radius"])
            valid[index] = True
        except NUMERICAL_SAMPLE_EXCEPTIONS:
            continue
    return values, valid


def _local_radius_surrogate(
    *,
    group: str,
    baseline_gains: Mapping[str, GainTriple],
    plants: Sequence[ScaleFreePlant],
    vehicle_model: Any,
    actuator_parameters: Any,
    controller_dt: float,
    fd_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline, baseline_valid = _spectral_radii(
        plants=plants,
        vehicle_model=vehicle_model,
        actuator_parameters=actuator_parameters,
        controller_dt=controller_dt,
        gains=baseline_gains,
    )
    h = float(math.log(float(fd_ratio)))
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("fd-ratio must exceed one")
    center = baseline_gains[group].array()
    gradient = np.full((len(plants), 3), np.nan, dtype=float)
    valid = baseline_valid.copy()
    for axis in range(3):
        direction = np.zeros(3)
        direction[axis] = h
        plus = GainTriple(*(center * np.exp(direction)))
        minus = GainTriple(*(center * np.exp(-direction)))
        plus_radius, plus_valid = _spectral_radii(
            plants=plants,
            vehicle_model=vehicle_model,
            actuator_parameters=actuator_parameters,
            controller_dt=controller_dt,
            gains=_replace_group(baseline_gains, group, plus),
        )
        minus_radius, minus_valid = _spectral_radii(
            plants=plants,
            vehicle_model=vehicle_model,
            actuator_parameters=actuator_parameters,
            controller_dt=controller_dt,
            gains=_replace_group(baseline_gains, group, minus),
        )
        valid &= plus_valid & minus_valid
        gradient[:, axis] = (plus_radius - minus_radius) / (2.0 * h)
    valid &= np.all(np.isfinite(gradient), axis=1)
    return baseline[valid], gradient[valid], valid


def _stable_fraction_tensor(
    baseline: np.ndarray,
    gradient: np.ndarray,
    log_ratio: np.ndarray,
) -> np.ndarray:
    size = int(log_ratio.size)
    result = np.empty((size, size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            delta_prefix = np.asarray((log_ratio[i], log_ratio[j]), dtype=float)
            for k in range(size):
                delta = np.asarray((delta_prefix[0], delta_prefix[1], log_ratio[k]))
                radius = baseline + gradient @ delta
                result[i, j, k] = float(np.mean(radius < 1.0))
    return result


def _project(stable_fraction: np.ndarray) -> Mapping[str, np.ndarray]:
    return {
        name: np.max(stable_fraction, axis=hidden)
        for name, _a, _b, hidden, _la, _lb in PROJECTION_SPECS
    }


def _range_with_overlay(
    baseline: GainTriple,
    overlay: Optional[GainTriple],
    scale_min: float,
    scale_max: float,
) -> tuple[float, float]:
    lower = float(scale_min)
    upper = float(scale_max)
    if overlay is None:
        return lower, upper
    base = baseline.array()
    other = overlay.array()
    positive = base > 0.0
    ratios = other[positive] / base[positive]
    if ratios.size:
        lower = min(lower, float(np.min(ratios)))
        upper = max(upper, float(np.max(ratios)))
    return lower, upper


def _plot(
    *,
    path: Path,
    case_name: str,
    group: str,
    alpha: float,
    ratio: np.ndarray,
    projected: Mapping[str, np.ndarray],
    baseline: GainTriple,
    success: Optional[GainTriple],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    base = baseline.array()
    overlay = None if success is None else success.array()
    gain_grid = base[None, :] * ratio[:, None]
    color = None
    for axis, (name, first, second, _hidden, first_name, second_name) in zip(
        axes, PROJECTION_SPECS
    ):
        field = projected[name]
        x, y = np.meshgrid(
            gain_grid[:, first], gain_grid[:, second], indexing="ij"
        )
        color = axis.pcolormesh(x, y, field, shading="auto", vmin=0.0, vmax=1.0)
        if float(np.min(field)) <= alpha <= float(np.max(field)):
            axis.contour(x, y, field, levels=[alpha], linewidths=2.0)
            axis.contourf(
                x,
                y,
                field,
                levels=[alpha, 1.0],
                colors=[(0.0, 0.0, 0.0, 0.10)],
            )
        axis.plot(
            base[first], base[second], marker="x", markersize=9,
            markeredgewidth=2.0, label="recorded"
        )
        if overlay is not None:
            axis.plot(
                overlay[first], overlay[second], marker="o", markersize=6,
                label="success-recorded"
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(f"{group} {first_name}")
        axis.set_ylabel(f"{group} {second_name}")
        axis.set_title(f"{first_name}{second_name} projection")
        axis.grid(True)
        axis.legend(loc="best")
    assert color is not None
    figure.colorbar(color, ax=axes, shrink=0.92).set_label(
        "max stable fraction over hidden gain"
    )
    figure.suptitle(f"{case_name} / {group} / alpha={alpha:.2f}")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def analyze(arguments: argparse.Namespace) -> Mapping[str, Any]:
    estimate_path = Path(arguments.estimate_json).expanduser().resolve()
    estimate = load_estimate_json(estimate_path)
    success_estimate = (
        None
        if arguments.success_json is None
        else load_estimate_json(Path(arguments.success_json))
    )
    case_name = str(estimate["case_name"])
    output = (
        Path(arguments.output_dir).expanduser().resolve()
        if arguments.output_dir is not None
        else estimate_path.parent / "pid_gain_region"
    )
    output.mkdir(parents=True, exist_ok=True)

    model_path = Path(estimate["input"]["vehicle_model"])
    vehicle_model = load_vehicle_model(model_path)
    actuator_parameters = actuator_parameters_from_estimate(estimate)
    controller_dt = float(estimate["controller_timing"]["median_seconds"])
    quotient = draw_quotient_coordinates(
        estimate,
        str(arguments.covariance),
        int(arguments.samples),
        int(arguments.seed),
    )
    plants = quotient_to_scale_free_plants(
        estimate,
        quotient,
        vehicle_model.parameters,
        ScaleFreePlant,
    )
    baseline_gains = _recorded_gains(estimate)
    success_gains = (
        None if success_estimate is None else _recorded_gains(success_estimate)
    )

    summary_rows = []
    for group in arguments.group or PID_GROUPS:
        baseline_radius, gradient, valid_mask = _local_radius_surrogate(
            group=group,
            baseline_gains=baseline_gains,
            plants=plants,
            vehicle_model=vehicle_model,
            actuator_parameters=actuator_parameters,
            controller_dt=controller_dt,
            fd_ratio=float(arguments.fd_ratio),
        )
        if baseline_radius.size == 0:
            raise RuntimeError(f"no pole-valid samples for group {group}")
        overlay = None if success_gains is None else success_gains[group]
        scale_min, scale_max = _range_with_overlay(
            baseline_gains[group],
            overlay,
            float(arguments.scale_min),
            float(arguments.scale_max),
        )
        ratio = np.geomspace(scale_min, scale_max, int(arguments.grid_size))
        stable_fraction = _stable_fraction_tensor(
            baseline_radius, gradient, np.log(ratio)
        )
        projected = _project(stable_fraction)

        exact_success_fraction = None
        exact_success_valid = None
        if success_gains is not None:
            success_radius, success_valid = _spectral_radii(
                plants=plants,
                vehicle_model=vehicle_model,
                actuator_parameters=actuator_parameters,
                controller_dt=controller_dt,
                gains=success_gains,
            )
            exact_success_valid = int(np.count_nonzero(success_valid))
            if exact_success_valid:
                exact_success_fraction = float(
                    np.mean(success_radius[success_valid] < 1.0)
                )

        group_dir = output / group
        group_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            group_dir / "gain_region.npz",
            ratio_grid=ratio,
            stable_fraction=stable_fraction,
            projected_pi=projected["pi"],
            projected_id=projected["id"],
            projected_dp=projected["dp"],
            surrogate_valid_mask=valid_mask,
        )
        group_payload = {
            "schema": GAIN_REGION_SCHEMA,
            "case_name": case_name,
            "estimate_json": str(estimate_path),
            "group": group,
            "alpha": float(arguments.alpha),
            "covariance": str(arguments.covariance),
            "requested_samples": int(arguments.samples),
            "surrogate_valid_samples": int(baseline_radius.size),
            "recorded_gain": {
                "p_gain": baseline_gains[group].p,
                "i_gain": baseline_gains[group].i,
                "d_gain": baseline_gains[group].d,
            },
            "recorded_exact_stable_fraction": float(
                np.mean(baseline_radius < 1.0)
            ),
            "surrogate_stable_fraction_min": float(np.min(stable_fraction)),
            "surrogate_stable_fraction_max": float(np.max(stable_fraction)),
            "ratio_grid": ratio.tolist(),
            "fd_ratio": float(arguments.fd_ratio),
            "success_overlay": (
                None
                if overlay is None
                else {
                    "gain": {
                        "p_gain": overlay.p,
                        "i_gain": overlay.i,
                        "d_gain": overlay.d,
                    },
                    "exact_stable_fraction_on_this_plant_distribution": (
                        exact_success_fraction
                    ),
                    "exact_pole_valid_samples": exact_success_valid,
                }
            ),
            "interpretation": (
                "The plotted field is a local log-gain spectral-radius surrogate. "
                "Recorded and optional success-overlay fractions are evaluated "
                "with the exact 26-state first-order closed-loop pole model."
            ),
        }
        write_json(group_dir / "gain_region.json", group_payload)
        _plot(
            path=group_dir / "gain_region.png",
            case_name=case_name,
            group=group,
            alpha=float(arguments.alpha),
            ratio=ratio,
            projected=projected,
            baseline=baseline_gains[group],
            success=overlay,
        )
        summary_rows.append(group_payload)

    summary = {
        "schema": GAIN_REGION_SCHEMA + "-summary",
        "case_name": case_name,
        "estimate_json": str(estimate_path),
        "first_order_time_constant_seconds": float(
            estimate["actuator_model"]["thrust_time_constant_seconds"]
        ),
        "alpha": float(arguments.alpha),
        "covariance": str(arguments.covariance),
        "rows": summary_rows,
    }
    write_json(output / "summary.json", summary)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate-json", type=Path, required=True)
    parser.add_argument(
        "--success-json",
        type=Path,
        default=None,
        help="optional success estimate; its recorded PID is overlaid and exactly evaluated",
    )
    parser.add_argument(
        "--covariance",
        choices=COVARIANCE_NAMES,
        default="conservative_fusion",
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--group", action="append", choices=PID_GROUPS)
    parser.add_argument("--fd-ratio", type=float, default=DEFAULT_FD_RATIO)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--scale-min", type=float, default=DEFAULT_SCALE_MIN)
    parser.add_argument("--scale-max", type=float, default=DEFAULT_SCALE_MAX)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    summary = analyze(arguments)
    print(Path(arguments.output_dir) if arguments.output_dir else Path(arguments.estimate_json).parent / "pid_gain_region")
    return 0 if summary["rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
