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
UNCONDITIONAL_GAIN_REGION_SCHEMA = (
    "grape-param-estim/first-order-lag-pid-gain-region/v2-unconditional"
)
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
            valid[index] = np.isfinite(values[index]) and values[index] >= 0.0
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
    # Keep sample identity and N even when a finite-difference fit is unresolved.
    return baseline, gradient, valid


def _stable_fraction_tensor(
    baseline: np.ndarray,
    gradient: np.ndarray,
    log_ratio: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    baseline = np.asarray(baseline, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    if baseline.ndim != 1 or not baseline.size or gradient.shape != (baseline.size, 3):
        raise ValueError("surrogate arrays must retain all original plant samples")
    valid = np.ones(baseline.size, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != baseline.shape:
        raise ValueError("surrogate validity must have one entry per original sample")
    size = int(log_ratio.size)
    result = np.empty((size, size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            delta_prefix = np.asarray((log_ratio[i], log_ratio[j]), dtype=float)
            for k in range(size):
                delta = np.asarray((delta_prefix[0], delta_prefix[1], log_ratio[k]))
                radius = baseline + gradient @ delta
                stable = valid & np.isfinite(radius) & (radius < 1.0)
                result[i, j, k] = float(np.count_nonzero(stable) / baseline.size)
    return result


def _project(stable_fraction: np.ndarray) -> Mapping[str, np.ndarray]:
    return {
        name: np.max(stable_fraction, axis=hidden)
        for name, _a, _b, hidden, _la, _lb in PROJECTION_SPECS
    }


def _range_from_failure(
    scale_min: float,
    scale_max: float,
) -> tuple[float, float]:
    lower = float(scale_min)
    upper = float(scale_max)
    if not np.isfinite(lower) or not np.isfinite(upper) or not 0.0 < lower < upper:
        raise ValueError("gain scales must satisfy finite 0 < min < max")
    return lower, upper


def _exact_stability_summary(radius: np.ndarray, valid_mask: np.ndarray) -> Mapping[str, Any]:
    radius = np.asarray(radius, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(radius) & (radius >= 0.0)
    stable = valid & (radius < 1.0)
    return {
        "sample_count": int(radius.size),
        "stable_count": int(np.count_nonzero(stable)),
        "stable_fraction": float(np.count_nonzero(stable) / radius.size),
        "pole_valid_count": int(np.count_nonzero(valid)),
        "unresolved_count": int(np.count_nonzero(~valid)),
    }


def _plot(
    *,
    path: Path,
    case_name: str,
    group: str,
    alpha: float,
    ratio: np.ndarray,
    projected: Mapping[str, np.ndarray],
    baseline: GainTriple,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    base = baseline.array()
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
            markeredgewidth=2.0, label="recorded visible coordinates"
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
        "max surrogate stable samples / all original samples over hidden gain"
    )
    figure.suptitle(f"{case_name} / {group}: local surrogate; other 9 gains at recorded values")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def analyze(arguments: argparse.Namespace) -> Mapping[str, Any]:
    estimate_path = Path(arguments.estimate_json).expanduser().resolve()
    estimate = load_estimate_json(estimate_path)
    case_name = str(estimate["case_name"])
    output = (
        Path(arguments.output_dir).expanduser().resolve()
        if arguments.output_dir is not None
        else estimate_path.parent / "pid_gain_region_unconditional"
    )
    output.mkdir(parents=True, exist_ok=True)
    vehicle_model = load_vehicle_model(Path(estimate["input"]["vehicle_model"]))
    actuator_parameters = actuator_parameters_from_estimate(estimate)
    controller_dt = float(estimate["controller_timing"]["median_seconds"])
    quotient = draw_quotient_coordinates(
        estimate, str(arguments.covariance), int(arguments.samples), int(arguments.seed),
    )
    plants = quotient_to_scale_free_plants(
        estimate, quotient, vehicle_model.parameters, ScaleFreePlant,
    )
    if not plants:
        raise ValueError("the original plant sample population must be nonempty")
    baseline_gains = _recorded_gains(estimate)
    scale_min, scale_max = _range_from_failure(
        float(arguments.scale_min), float(arguments.scale_max),
    )
    ratio = np.geomspace(scale_min, scale_max, int(arguments.grid_size))
    # The exact controller comparison uses all original samples independently
    # of which finite-difference surrogate fits can be evaluated.
    recorded_radius, recorded_valid = _spectral_radii(
        plants=plants, vehicle_model=vehicle_model,
        actuator_parameters=actuator_parameters, controller_dt=controller_dt,
        gains=baseline_gains,
    )
    recorded_summary = _exact_stability_summary(recorded_radius, recorded_valid)
    full_references = {"recorded_case": {
        "gains": {name: triple.array().tolist() for name, triple in baseline_gains.items()},
        **recorded_summary,
    }}
    summary_rows = []
    for group in arguments.group or PID_GROUPS:
        baseline_radius, gradient, valid_mask = _local_radius_surrogate(
            group=group, baseline_gains=baseline_gains, plants=plants,
            vehicle_model=vehicle_model, actuator_parameters=actuator_parameters,
            controller_dt=controller_dt, fd_ratio=float(arguments.fd_ratio),
        )
        stable_fraction = _stable_fraction_tensor(
            baseline_radius, gradient, np.log(ratio), valid_mask,
        )
        projected = _project(stable_fraction)
        group_dir = output / group
        group_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            group_dir / "gain_region.npz", quotient_coordinates=quotient,
            ratio_grid=ratio, stable_fraction=stable_fraction,
            projected_pi=projected["pi"], projected_id=projected["id"],
            projected_dp=projected["dp"], surrogate_valid_mask=valid_mask,
            surrogate_baseline_spectral_radius=baseline_radius,
            surrogate_radius_gradient=gradient,
            recorded_exact_spectral_radius=recorded_radius,
            recorded_exact_valid_mask=recorded_valid,
        )
        group_payload = {
            "schema": UNCONDITIONAL_GAIN_REGION_SCHEMA,
            "case_name": case_name, "estimate_json": str(estimate_path),
            "group": group, "alpha": float(arguments.alpha),
            "covariance": str(arguments.covariance),
            "requested_samples": int(arguments.samples),
            "sample_count": len(plants),
            "sample_population_conditioned_on_success": False,
            "success_used_in_map_generation": False,
            "surrogate_valid_samples": int(np.count_nonzero(valid_mask)),
            "surrogate_unresolved_samples": int(np.count_nonzero(~valid_mask)),
            "recorded_gain": {
                "p_gain": baseline_gains[group].p,
                "i_gain": baseline_gains[group].i,
                "d_gain": baseline_gains[group].d,
            },
            "fixed_controller_gains": {
                name: triple.array().tolist() for name, triple in baseline_gains.items()
            },
            "recorded_exact_stable_fraction": recorded_summary["stable_fraction"],
            "recorded_exact_stable_count": recorded_summary["stable_count"],
            "recorded_exact_pole_valid_count": recorded_summary["pole_valid_count"],
            "recorded_exact_unresolved_count": recorded_summary["unresolved_count"],
            "surrogate_stable_fraction_min": float(np.min(stable_fraction)),
            "surrogate_stable_fraction_max": float(np.max(stable_fraction)),
            "ratio_grid": ratio.tolist(), "fd_ratio": float(arguments.fd_ratio),
            "success_overlay": None,
            "gain_domain_source": "declared CLI scale bounds only, relative to recorded gains",
            "definition": {
                "population": "all original plant samples at every grid point",
                "surrogate_stable_fraction": "count(valid surrogate and finite predicted radius < 1) / all original samples",
                "unresolved": "not counted stable, retained in the denominator and counted separately",
                "fixed_gains": "other nine gains fixed at this record; projection maximizes over the hidden gain in the selected group",
                "success_reference": "optional independent full-12-controller comparison after all group maps; no slice marker",
            },
            "interpretation": (
                "The field is the original local linear log-gain spectral-radius surrogate, "
                "not exact pole evaluations away from the recorded controller. "
                "Maximization over the hidden selected-group gain is only a projection. "
                "It cannot certify arbitrary combinations from different panels, joint "
                "12D stability, or real flight success probability. Exact full-controller "
                "references use all original samples independently of surrogate validity."
            ),
        }
        _plot(
            path=group_dir / "gain_region.png", case_name=case_name, group=group,
            alpha=float(arguments.alpha), ratio=ratio, projected=projected,
            baseline=baseline_gains[group],
        )
        summary_rows.append(group_payload)

    # No success data enter surrogate construction, gain bounds, or any map query.
    success_path = getattr(arguments, "success_json", None)
    reference_arrays = {
        "recorded_case_spectral_radius": recorded_radius,
        "recorded_case_valid_mask": recorded_valid,
    }
    if success_path is not None:
        success_path = Path(success_path).expanduser().resolve()
        success_gains = _recorded_gains(load_estimate_json(success_path))
        success_radius, success_valid = _spectral_radii(
            plants=plants, vehicle_model=vehicle_model,
            actuator_parameters=actuator_parameters, controller_dt=controller_dt,
            gains=success_gains,
        )
        full_references["recorded_success"] = {
            "gains": {name: triple.array().tolist() for name, triple in success_gains.items()},
            "role": "independent full-12-controller comparison after all group maps",
            **_exact_stability_summary(success_radius, success_valid),
        }
        reference_arrays.update(
            recorded_success_spectral_radius=success_radius,
            recorded_success_valid_mask=success_valid,
        )
    np.savez_compressed(output / "full_controller_references.npz", **reference_arrays)
    for group_payload in summary_rows:
        group_payload["full_controller_references"] = full_references
        write_json(output / group_payload["group"] / "gain_region.json", group_payload)
    summary = {
        "schema": UNCONDITIONAL_GAIN_REGION_SCHEMA + "-summary",
        "case_name": case_name, "estimate_json": str(estimate_path),
        "success_json": None if success_path is None else str(success_path),
        "first_order_time_constant_seconds": float(
            estimate["actuator_model"]["thrust_time_constant_seconds"]
        ),
        "alpha": float(arguments.alpha), "covariance": str(arguments.covariance),
        "sample_count": len(plants), "sample_population_conditioned_on_success": False,
        "success_used_in_map_generation": False,
        "full_controller_references": full_references, "rows": summary_rows,
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
        help="optional independent full-controller comparison after all group maps",
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
    print(Path(arguments.output_dir) if arguments.output_dir else Path(arguments.estimate_json).parent / "pid_gain_region_unconditional")
    return 0 if summary["rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
