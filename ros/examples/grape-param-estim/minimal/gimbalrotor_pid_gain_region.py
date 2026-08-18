#!/usr/bin/env python3
"""Project bag-wise PID gain regions from local pole stability samples.

This is a bag-wise diagnostic/proposal tool built on top of the existing
Gimbalrotor local-pole validation.  For each bag and each PID group
(xy / z / roll_pitch / yaw), it constructs a local log-gain surrogate of the
spectral radius for a fixed set of sampled plants, then projects the resulting
stable-fraction field onto the PI / ID / DP planes.

The outputs are intended to visualize regions whose local-stability fraction is
at least alpha (default 0.95).  Each bag is handled independently.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
for _path in (_HERE, _SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gimbalrotor_pid_local_pole_validation as local_poles  # noqa: E402
import three_bag_gimbalrotor_pid_local_pole_validation as three_bag  # noqa: E402
from grape_param_estim.controller_config import (  # noqa: E402
    PidGainConfiguration,
    apply_pid_gain_configuration,
)


SCHEMA = "grape-param-estim/gimbalrotor-pid-gain-region/v1"
DEFAULT_CASES = ("failure1", "failure2", "success")
DEFAULT_GROUPS = ("xy", "z", "roll_pitch", "yaw")
DEFAULT_COVARIANCE_MODE = "conservative_fusion"
DEFAULT_DELAY_MODE = "fitted_thrust_delay"
DEFAULT_ALPHA = 0.95
DEFAULT_GRID_SIZE = 17
DEFAULT_SCALE_MIN = 0.5
DEFAULT_SCALE_MAX = 2.0
DEFAULT_FD_RATIO = 1.12
DEFAULT_MAX_SAMPLE_COUNT = 64
DEFAULT_WORKERS = min(12, os.cpu_count() or 1)
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

    def as_array(self) -> np.ndarray:
        value = np.asarray((self.p, self.i, self.d), dtype=float)
        if np.any(~np.isfinite(value)) or np.any(value <= 0.0):
            raise ValueError("gain triple must be positive and finite")
        return value


@dataclass(frozen=True)
class CaseArtifacts:
    case_name: str
    inputs: Any
    controller_dt: float
    delay: Any
    recorded_gains: Mapping[str, GainTriple]
    success_gains: Mapping[str, GainTriple]
    scale_free_samples: np.ndarray
    local_pole_report_path: Path
    local_pole_samples_path: Path


@dataclass(frozen=True)
class GroupResult:
    case_name: str
    group_name: str
    alpha: float
    ratio_grid: np.ndarray
    gain_grid: np.ndarray
    stable_fraction: np.ndarray
    projected_fields: Mapping[str, np.ndarray]
    baseline_gain: GainTriple
    success_gain: GainTriple
    valid_sample_count: int
    requested_sample_count: int
    fd_log_step: float
    covariance_mode: str
    delay_mode: str


class GainOverrideContext:
    def __init__(self, holder: Any, values: Mapping[str, GainTriple]):
        self._holder = holder
        self._values = values
        self._original = None
        self._original_controller_configuration = None

    def __enter__(self):
        configuration = _maybe_get(self._holder, "controller_configuration")
        if configuration is not None:
            self._original_controller_configuration = configuration
            gain_values = np.asarray(
                [self._values[group].as_array() for group in DEFAULT_GROUPS],
                dtype=float,
            )
            replacement = apply_pid_gain_configuration(
                configuration, PidGainConfiguration(gain_values)
            )
            object.__setattr__(
                self._holder, "controller_configuration", replacement
            )
            return self
        self._original = _extract_group_gains_from_object(self._holder)
        _apply_group_gains_to_object(self._holder, self._values)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._original_controller_configuration is not None:
            object.__setattr__(
                self._holder,
                "controller_configuration",
                self._original_controller_configuration,
            )
        if self._original is not None:
            _apply_group_gains_to_object(self._holder, self._original)
        return False


class GainRegionNumericalError(RuntimeError):
    """A narrowly classified numerical sample failure."""


# -----------------------------------------------------------------------------
# Generic JSON / object helpers
# -----------------------------------------------------------------------------

def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _iter_json_scalars(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_json_scalars(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_json_scalars(item)
    else:
        yield value


def _find_first_string_ending_with(value: Any, suffix: str) -> Optional[str]:
    for item in _iter_json_scalars(value):
        if isinstance(item, str) and item.endswith(suffix):
            return item
    return None


def _contains_scalar(value: Any, target: str) -> bool:
    for item in _iter_json_scalars(value):
        if item == target:
            return True
    return False


def _maybe_get(container: Any, key: str) -> Any:
    if isinstance(container, Mapping):
        return container.get(key)
    return getattr(container, key, None)


def _set_mapping_or_attr(container: Any, key: str, value: Any) -> bool:
    if isinstance(container, MutableMapping):
        container[key] = value
        return True
    if hasattr(container, key):
        setattr(container, key, value)
        return True
    return False


def _to_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("value must be finite")
    return result


# -----------------------------------------------------------------------------
# Gain parsing / gain injection
# -----------------------------------------------------------------------------

def _gain_key_candidates(name: str) -> tuple[str, ...]:
    lowered = name.lower()
    if lowered == "p":
        return ("p_gain", "p", "P", "kp", "k_p", "Kp", "K_P")
    if lowered == "i":
        return ("i_gain", "i", "I", "ki", "k_i", "Ki", "K_I")
    if lowered == "d":
        return ("d_gain", "d", "D", "kd", "k_d", "Kd", "K_D")
    raise KeyError(name)


def _group_key_candidates(name: str) -> tuple[str, ...]:
    if name == "xy":
        return ("xy", "XY")
    if name == "z":
        return ("z", "Z")
    if name == "roll_pitch":
        return ("roll_pitch", "rollPitch", "rollpitch", "roll-pitch", "rp")
    if name == "yaw":
        return ("yaw", "Yaw")
    raise KeyError(name)


def _read_gain_component(container: Any, component: str) -> float:
    for key in _gain_key_candidates(component):
        value = _maybe_get(container, key)
        if value is not None:
            return _to_float(value)
    raise KeyError(f"missing gain component {component}")


def _extract_group_entry(container: Any, group_name: str) -> Any:
    for key in _group_key_candidates(group_name):
        value = _maybe_get(container, key)
        if value is not None:
            return value
    raise KeyError(f"missing gain group {group_name}")


def _extract_group_gains_from_json(path: Path) -> Mapping[str, GainTriple]:
    data = _read_json(path)
    snapshot = data.get("controller_gain_snapshot", data)
    groups_container = snapshot.get("gains", snapshot)
    result: dict[str, GainTriple] = {}
    for group in DEFAULT_GROUPS:
        entry = _extract_group_entry(groups_container, group)
        result[group] = GainTriple(
            p=_read_gain_component(entry, "p"),
            i=_read_gain_component(entry, "i"),
            d=_read_gain_component(entry, "d"),
        )
    return result


def _extract_group_gains_from_object(root: Any) -> Mapping[str, GainTriple]:
    candidates = [
        root,
        _maybe_get(root, "static_postprocess"),
        _maybe_get(_maybe_get(root, "static_postprocess"), "controller_gain_snapshot"),
        _maybe_get(root, "controller_gain_snapshot"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        groups_container = _maybe_get(candidate, "gains")
        if groups_container is None:
            groups_container = candidate
        try:
            result: dict[str, GainTriple] = {}
            for group in DEFAULT_GROUPS:
                entry = _extract_group_entry(groups_container, group)
                result[group] = GainTriple(
                    p=_read_gain_component(entry, "p"),
                    i=_read_gain_component(entry, "i"),
                    d=_read_gain_component(entry, "d"),
                )
            return result
        except Exception:
            continue
    raise RuntimeError("could not extract recorded group gains from inputs object")


def _set_gain_component(container: Any, component: str, value: float) -> bool:
    for key in _gain_key_candidates(component):
        if _set_mapping_or_attr(container, key, float(value)):
            return True
    return False


def _apply_group_gains_to_object(root: Any, values: Mapping[str, GainTriple]) -> None:
    candidates = [
        _maybe_get(root, "static_postprocess"),
        _maybe_get(_maybe_get(root, "static_postprocess"), "controller_gain_snapshot"),
        _maybe_get(root, "controller_gain_snapshot"),
        root,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        groups_container = _maybe_get(candidate, "gains")
        if groups_container is None:
            groups_container = candidate
        success = True
        try:
            for group_name, triple in values.items():
                entry = _extract_group_entry(groups_container, group_name)
                success &= _set_gain_component(entry, "p", triple.p)
                success &= _set_gain_component(entry, "i", triple.i)
                success &= _set_gain_component(entry, "d", triple.d)
            if success:
                return
        except Exception:
            pass
    raise RuntimeError("could not apply gain overrides to inputs object")


# -----------------------------------------------------------------------------
# Local-pole sample discovery
# -----------------------------------------------------------------------------

def _bag_path_from_case_definition(case_name: str) -> str:
    definition = three_bag.CASE_DEFINITIONS[case_name]
    bag_json = _read_json(Path(definition["bag_json"]))
    candidate = _find_first_string_ending_with(bag_json, ".bag")
    if candidate is None:
        raise RuntimeError(f"could not determine bag path for {case_name}")
    return candidate


def _find_local_pole_artifacts(
    *,
    outputs_root: Path,
    case_name: str,
    covariance_mode: str,
    delay_mode: str,
) -> tuple[Path, Path]:
    desired_bag_path = _bag_path_from_case_definition(case_name)
    matches: list[tuple[Path, Path]] = []
    for report_path in outputs_root.rglob("local_pole_validation.json"):
        try:
            report = _read_json(report_path)
        except Exception:
            continue
        if not _contains_scalar(report, covariance_mode):
            continue
        if not _contains_scalar(report, delay_mode):
            continue
        if desired_bag_path not in list(_iter_json_scalars(report)):
            continue
        samples_path = report_path.with_name("local_pole_samples.npz")
        if samples_path.exists():
            matches.append((report_path, samples_path))
    if not matches:
        raise RuntimeError(
            f"no local_pole_validation artifact found for {case_name}, "
            f"covariance={covariance_mode}, delay={delay_mode} under {outputs_root}"
        )
    matches.sort(key=lambda pair: len(str(pair[0])))
    return matches[0]


def _load_scale_free_samples(path: Path, max_sample_count: int) -> np.ndarray:
    arrays = np.load(path, allow_pickle=False)
    if "scale_free_samples" not in arrays:
        raise RuntimeError(f"{path} lacks scale_free_samples")
    sample_array = np.asarray(arrays["scale_free_samples"])
    if sample_array.ndim == 0:
        raise RuntimeError("scale_free_samples must be an array of samples")
    count = min(int(max_sample_count), int(sample_array.shape[0]))
    if count <= 0:
        raise RuntimeError("requested sample count is zero")
    return sample_array[:count].copy()


# -----------------------------------------------------------------------------
# Pole evaluation and surrogate construction
# -----------------------------------------------------------------------------

def _spectral_radius_from_result(result: Mapping[str, Any]) -> float:
    eigenvalues = result.get("eigenvalues")
    if eigenvalues is None:
        raise RuntimeError("analyzed plant result has no eigenvalues")
    values = np.asarray(eigenvalues, dtype=complex)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise RuntimeError("eigenvalues are invalid")
    return float(np.max(np.abs(values)))


def _scale_free_plant_from_vector(sample: Any, rotor_lag_seconds: float) -> Any:
    if hasattr(sample, "inertia_over_mass"):
        return sample
    value = np.asarray(sample, dtype=float)
    if value.shape != (13,) or np.any(~np.isfinite(value)):
        raise ValueError("scale-free sample must be a finite 13-vector")
    inertia = np.asarray(
        (
            (value[0], value[3], value[4]),
            (value[3], value[1], value[5]),
            (value[4], value[5], value[2]),
        ),
        dtype=float,
    )
    return local_poles.ScaleFreePlant(
        inertia_over_mass=inertia,
        cog_position_body=value[6:9],
        force_effectiveness_over_mass=value[9:13],
        rotor_lag_seconds=float(rotor_lag_seconds),
    )


def _analyze_spectral_radius(
    *,
    inputs: Any,
    scale_free: Any,
    controller_dt: float,
    delay: Any,
) -> float:
    analyzed = local_poles._analyze_plant(  # pylint: disable=protected-access
        scale_free=scale_free,
        inputs=inputs,
        controller_dt=controller_dt,
        delay=delay,
        fd_check=False,
    )
    trim = analyzed.get("trim")
    if trim is None or not bool(trim.equilibrium_valid):
        raise GainRegionNumericalError(
            "equilibrium is invalid under the requested gains"
        )
    return _spectral_radius_from_result(analyzed)


def _evaluate_group_gains_for_samples(
    *,
    inputs: Any,
    group_values: Mapping[str, GainTriple],
    scale_free_samples: np.ndarray,
    controller_dt: float,
    delay: Any,
) -> np.ndarray:
    result = np.full(int(scale_free_samples.shape[0]), np.nan, dtype=float)
    with GainOverrideContext(inputs, group_values):
        for index, sample in enumerate(scale_free_samples):
            try:
                result[index] = _analyze_spectral_radius(
                    inputs=inputs,
                    scale_free=_scale_free_plant_from_vector(
                        sample, inputs.result.plant.rotor_lag_seconds
                    ),
                    controller_dt=controller_dt,
                    delay=delay,
                )
            except (
                GainRegionNumericalError,
                *local_poles.NUMERICAL_SAMPLE_EXCEPTIONS,
            ):
                result[index] = np.nan
    return result


def _clone_group_values(values: Mapping[str, GainTriple]) -> dict[str, GainTriple]:
    return {key: GainTriple(triple.p, triple.i, triple.d) for key, triple in values.items()}


def _make_group_triple_from_array(array: np.ndarray) -> GainTriple:
    value = np.asarray(array, dtype=float)
    if value.shape != (3,) or np.any(~np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError("gain array must be positive and length-3")
    return GainTriple(p=float(value[0]), i=float(value[1]), d=float(value[2]))


def _build_local_surrogate(
    *,
    inputs: Any,
    controller_dt: float,
    delay: Any,
    group_name: str,
    recorded_gains: Mapping[str, GainTriple],
    scale_free_samples: np.ndarray,
    fd_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_values = _clone_group_values(recorded_gains)
    baseline = _evaluate_group_gains_for_samples(
        inputs=inputs,
        group_values=base_values,
        scale_free_samples=scale_free_samples,
        controller_dt=controller_dt,
        delay=delay,
    )
    h = float(math.log(float(fd_ratio)))
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("fd_ratio must be > 1")
    baseline_array = recorded_gains[group_name].as_array()
    gradients = np.full((int(scale_free_samples.shape[0]), 3), np.nan, dtype=float)
    for axis in range(3):
        step_vector = np.zeros(3, dtype=float)
        step_vector[axis] = h

        plus = _clone_group_values(recorded_gains)
        minus = _clone_group_values(recorded_gains)
        plus[group_name] = _make_group_triple_from_array(
            baseline_array * np.exp(step_vector)
        )
        minus[group_name] = _make_group_triple_from_array(
            baseline_array * np.exp(-step_vector)
        )

        plus_values = _evaluate_group_gains_for_samples(
            inputs=inputs,
            group_values=plus,
            scale_free_samples=scale_free_samples,
            controller_dt=controller_dt,
            delay=delay,
        )
        minus_values = _evaluate_group_gains_for_samples(
            inputs=inputs,
            group_values=minus,
            scale_free_samples=scale_free_samples,
            controller_dt=controller_dt,
            delay=delay,
        )
        gradients[:, axis] = (plus_values - minus_values) / (2.0 * h)

    valid = np.isfinite(baseline) & np.all(np.isfinite(gradients), axis=1)
    return baseline[valid], gradients[valid], valid


def _evaluate_stable_fraction_tensor(
    *,
    baseline_radius: np.ndarray,
    gradient: np.ndarray,
    log_ratio_grid: np.ndarray,
) -> np.ndarray:
    grid_size = int(log_ratio_grid.shape[0])
    result = np.empty((grid_size, grid_size, grid_size), dtype=float)
    for index_p in range(grid_size):
        for index_i in range(grid_size):
            for index_d in range(grid_size):
                delta = np.asarray(
                    (
                        log_ratio_grid[index_p],
                        log_ratio_grid[index_i],
                        log_ratio_grid[index_d],
                    ),
                    dtype=float,
                )
                radius = baseline_radius + gradient @ delta
                result[index_p, index_i, index_d] = float(np.mean(radius < 1.0))
    return result


def _project_fields(stable_fraction: np.ndarray) -> Mapping[str, np.ndarray]:
    projected: dict[str, np.ndarray] = {}
    for name, axis_a, axis_b, axis_hidden, _label_a, _label_b in PROJECTION_SPECS:
        projected[name] = np.max(stable_fraction, axis=axis_hidden)
    return projected


# -----------------------------------------------------------------------------
# Plotting / reporting
# -----------------------------------------------------------------------------

def _render_group_figure(
    *,
    output_path: Path,
    title: str,
    group_name: str,
    alpha: float,
    ratio_grid: np.ndarray,
    projected_fields: Mapping[str, np.ndarray],
    baseline_gain: GainTriple,
    success_gain: Optional[GainTriple],
    include_success_overlay: bool,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    gain_values = baseline_gain.as_array()[None, :] * ratio_grid[:, None]
    baseline_array = baseline_gain.as_array()
    success_array = None if success_gain is None else success_gain.as_array()

    for axis_plot, (projection_name, axis_a, axis_b, _axis_hidden, label_a, label_b) in zip(axes, PROJECTION_SPECS):
        field = projected_fields[projection_name]
        x_values = gain_values[:, axis_a]
        y_values = gain_values[:, axis_b]
        mesh_x, mesh_y = np.meshgrid(x_values, y_values, indexing="ij")
        color = axis_plot.pcolormesh(mesh_x, mesh_y, field, shading="auto", vmin=0.0, vmax=1.0)
        axis_plot.contour(
            mesh_x,
            mesh_y,
            field,
            levels=[alpha],
            colors=["black"],
            linewidths=2.0,
        )
        axis_plot.contourf(
            mesh_x,
            mesh_y,
            field,
            levels=[alpha, 1.0],
            colors=[(0.0, 0.0, 0.0, 0.10)],
        )
        axis_plot.plot(
            [baseline_array[axis_a]],
            [baseline_array[axis_b]],
            marker="x",
            markersize=8,
            color="white",
            markeredgewidth=2.0,
            label="recorded",
        )
        if include_success_overlay and success_array is not None:
            axis_plot.plot(
                [success_array[axis_a]],
                [success_array[axis_b]],
                marker="o",
                markersize=6,
                color="red",
                label="success-recorded",
            )
        axis_plot.set_xscale("log")
        axis_plot.set_yscale("log")
        axis_plot.set_xlabel(f"{group_name} {label_a}")
        axis_plot.set_ylabel(f"{group_name} {label_b}")
        axis_plot.set_title(f"{label_a}{label_b} plane")
        axis_plot.grid(True)
        axis_plot.legend(loc="best")

    colorbar = figure.colorbar(color, ax=axes, shrink=0.92)
    colorbar.set_label("max stable fraction over hidden gain")
    figure.suptitle(title)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_group_json(path: Path, result: GroupResult) -> None:
    payload = {
        "schema": SCHEMA,
        "source_commit": local_poles.source_commit(),
        "case": result.case_name,
        "group": result.group_name,
        "alpha": result.alpha,
        "covariance_mode": result.covariance_mode,
        "delay_mode": result.delay_mode,
        "baseline_gain": {
            "p": result.baseline_gain.p,
            "i": result.baseline_gain.i,
            "d": result.baseline_gain.d,
        },
        "success_gain": {
            "p": result.success_gain.p,
            "i": result.success_gain.i,
            "d": result.success_gain.d,
        },
        "ratio_grid": result.ratio_grid.tolist(),
        "gain_grid": result.gain_grid.tolist(),
        "stable_fraction_shape": list(result.stable_fraction.shape),
        "stable_fraction_min": float(np.min(result.stable_fraction)),
        "stable_fraction_max": float(np.max(result.stable_fraction)),
        "stable_fraction_at_recorded_gain": float(
            result.stable_fraction[
                result.stable_fraction.shape[0] // 2,
                result.stable_fraction.shape[1] // 2,
                result.stable_fraction.shape[2] // 2,
            ]
        ),
        "valid_sample_count": int(result.valid_sample_count),
        "requested_sample_count": int(result.requested_sample_count),
        "fd_log_step": float(result.fd_log_step),
    }
    local_poles.write_json(path, payload)


def _write_group_md(path: Path, result: GroupResult, with_success_overlay: bool) -> None:
    center_index = result.stable_fraction.shape[0] // 2
    center_fraction = float(
        result.stable_fraction[center_index, center_index, center_index]
    )
    lines = [
        f"# PID gain region: {result.case_name} / {result.group_name}",
        "",
        f"- alpha threshold: `{result.alpha}`",
        f"- covariance mode: `{result.covariance_mode}`",
        f"- delay mode: `{result.delay_mode}`",
        f"- valid samples: `{result.valid_sample_count}` / `{result.requested_sample_count}`",
        f"- recorded-gain stable fraction (surrogate center): `{center_fraction:.6g}`",
        "",
        "## Recorded gain",
        "",
        f"- P = `{result.baseline_gain.p:.6g}`",
        f"- I = `{result.baseline_gain.i:.6g}`",
        f"- D = `{result.baseline_gain.d:.6g}`",
    ]
    if with_success_overlay:
        lines.extend(
            [
                "",
                "## Success overlay",
                "",
                f"- success P = `{result.success_gain.p:.6g}`",
                f"- success I = `{result.success_gain.i:.6g}`",
                f"- success D = `{result.success_gain.d:.6g}`",
            ]
        )
    lines.extend(
        [
            "",
            "The shading on each projection shows the maximum stable fraction over the hidden gain coordinate.  The black contour is the alpha-level boundary, and the hatched/filled region corresponds to stable fraction >= alpha.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------

def _load_case_artifacts(
    *,
    case_name: str,
    controller_yaml: Path,
    vehicle_model: Path,
    outputs_root: Path,
    covariance_mode: str,
    delay_mode: str,
    max_sample_count: int,
    success_gains: Mapping[str, GainTriple],
) -> CaseArtifacts:
    definition = three_bag.CASE_DEFINITIONS[case_name]
    estimator = Path(definition["estimator"])
    static_postprocess = Path(definition["static"])
    bag_json = Path(definition["bag_json"])
    inputs = local_poles.load_case_inputs(
        result_path=estimator / "result.json",
        arrays_path=estimator / "arrays.npz",
        static_postprocess_path=static_postprocess,
        arguments_path=estimator / "arguments.json",
        bag_json_path=bag_json,
        controller_yaml_path=controller_yaml,
        vehicle_model_path=vehicle_model,
        covariance_mode=covariance_mode,
    )
    timing = local_poles.controller_timing_from_bag(inputs.bag)
    controller_dt = float(timing["median_seconds"])
    delay_seconds = 0.0
    if delay_mode == "fitted_thrust_delay":
        delay_seconds = float(inputs.result.plant.rotor_lag_seconds)
    elif delay_mode == "zero_thrust_delay":
        delay_seconds = 0.0
    else:
        raise ValueError(f"unsupported delay mode {delay_mode}")
    delay = local_poles.decompose_thrust_delay(delay_seconds, controller_dt)

    report_path, samples_path = _find_local_pole_artifacts(
        outputs_root=outputs_root,
        case_name=case_name,
        covariance_mode=covariance_mode,
        delay_mode=delay_mode,
    )
    scale_free_samples = _load_scale_free_samples(samples_path, max_sample_count)
    return CaseArtifacts(
        case_name=case_name,
        inputs=inputs,
        controller_dt=controller_dt,
        delay=delay,
        recorded_gains=_extract_group_gains_from_json(static_postprocess),
        success_gains=success_gains,
        scale_free_samples=scale_free_samples,
        local_pole_report_path=report_path,
        local_pole_samples_path=samples_path,
    )


def _analyze_case_group(
    *,
    artifacts: CaseArtifacts,
    group_name: str,
    alpha: float,
    grid_size: int,
    scale_min: float,
    scale_max: float,
    fd_ratio: float,
    covariance_mode: str,
    delay_mode: str,
) -> GroupResult:
    baseline_radius, gradient, valid = _build_local_surrogate(
        inputs=artifacts.inputs,
        controller_dt=artifacts.controller_dt,
        delay=artifacts.delay,
        group_name=group_name,
        recorded_gains=artifacts.recorded_gains,
        scale_free_samples=artifacts.scale_free_samples,
        fd_ratio=fd_ratio,
    )
    if baseline_radius.size == 0:
        raise RuntimeError(
            f"no valid surrogate samples for {artifacts.case_name}/{group_name}"
        )
    ratio_grid = np.geomspace(float(scale_min), float(scale_max), int(grid_size))
    log_ratio_grid = np.log(ratio_grid)
    stable_fraction = _evaluate_stable_fraction_tensor(
        baseline_radius=baseline_radius,
        gradient=gradient,
        log_ratio_grid=log_ratio_grid,
    )
    projected = _project_fields(stable_fraction)
    baseline_gain = artifacts.recorded_gains[group_name]
    gain_grid = baseline_gain.as_array()[None, :] * ratio_grid[:, None]
    return GroupResult(
        case_name=artifacts.case_name,
        group_name=group_name,
        alpha=float(alpha),
        ratio_grid=ratio_grid,
        gain_grid=gain_grid,
        stable_fraction=stable_fraction,
        projected_fields=projected,
        baseline_gain=baseline_gain,
        success_gain=artifacts.success_gains[group_name],
        valid_sample_count=int(baseline_radius.size),
        requested_sample_count=int(artifacts.scale_free_samples.shape[0]),
        fd_log_step=float(math.log(float(fd_ratio))),
        covariance_mode=covariance_mode,
        delay_mode=delay_mode,
    )


def _run_group_task(task: Mapping[str, Any]) -> Mapping[str, Any]:
    case_name = str(task["case_name"])
    group_name = str(task["group_name"])
    success_definition = three_bag.CASE_DEFINITIONS["success"]
    success_gains = _extract_group_gains_from_json(
        Path(success_definition["static"])
    )
    artifacts = _load_case_artifacts(
        case_name=case_name,
        controller_yaml=Path(task["controller_yaml"]),
        vehicle_model=Path(task["vehicle_model"]),
        outputs_root=Path(task["outputs_root"]),
        covariance_mode=str(task["covariance_mode"]),
        delay_mode=str(task["delay_mode"]),
        max_sample_count=int(task["max_sample_count"]),
        success_gains=success_gains,
    )
    result = _analyze_case_group(
        artifacts=artifacts,
        group_name=group_name,
        alpha=float(task["alpha"]),
        grid_size=int(task["grid_size"]),
        scale_min=float(task["scale_min"]),
        scale_max=float(task["scale_max"]),
        fd_ratio=float(task["fd_ratio"]),
        covariance_mode=str(task["covariance_mode"]),
        delay_mode=str(task["delay_mode"]),
    )
    group_dir = Path(task["output_dir"]) / case_name / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        group_dir / "gain_region_rollout.npz",
        ratio_grid=result.ratio_grid,
        gain_grid=result.gain_grid,
        stable_fraction=result.stable_fraction,
        projected_pi=result.projected_fields["pi"],
        projected_id=result.projected_fields["id"],
        projected_dp=result.projected_fields["dp"],
    )
    _write_group_json(group_dir / "gain_region_rollout.json", result)
    _write_group_md(
        group_dir / "gain_region_rollout.md",
        result,
        with_success_overlay=False,
    )
    _render_group_figure(
        output_path=group_dir / "gain_region_rollout.png",
        title=(
            f"{case_name} / {group_name} / alpha={result.alpha:.2f} / "
            f"cov={task['covariance_mode']} / delay={task['delay_mode']}"
        ),
        group_name=group_name,
        alpha=result.alpha,
        ratio_grid=result.ratio_grid,
        projected_fields=result.projected_fields,
        baseline_gain=result.baseline_gain,
        success_gain=None,
        include_success_overlay=False,
    )
    if case_name != "success":
        _write_group_md(
            group_dir / "gain_region_rollout_with_success_overlay.md",
            result,
            with_success_overlay=True,
        )
        _render_group_figure(
            output_path=(
                group_dir / "gain_region_rollout_with_success_overlay.png"
            ),
            title=(
                f"{case_name} / {group_name} / alpha={result.alpha:.2f} / "
                "success overlay"
            ),
            group_name=group_name,
            alpha=result.alpha,
            ratio_grid=result.ratio_grid,
            projected_fields=result.projected_fields,
            baseline_gain=result.baseline_gain,
            success_gain=result.success_gain,
            include_success_overlay=True,
        )
    center_index = result.stable_fraction.shape[0] // 2
    return {
        "case": case_name,
        "group": group_name,
        "alpha": result.alpha,
        "covariance_mode": str(task["covariance_mode"]),
        "delay_mode": str(task["delay_mode"]),
        "stable_fraction_at_recorded": float(
            result.stable_fraction[center_index, center_index, center_index]
        ),
        "max_projected_pi": float(np.max(result.projected_fields["pi"])),
        "max_projected_id": float(np.max(result.projected_fields["id"])),
        "max_projected_dp": float(np.max(result.projected_fields["dp"])),
        "valid_sample_count": result.valid_sample_count,
        "requested_sample_count": result.requested_sample_count,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller-yaml",
        type=Path,
        default=Path(
            "/home/leus/catkin_ws/src/jsk_aerial_robot/robots/gimbalrotor/config/grape/GimbalrotorControl.yaml"
        ),
    )
    parser.add_argument(
        "--vehicle-model",
        type=Path,
        default=_HERE / "grape_vehicle_model.json",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=_HERE / "outputs",
        help="search root containing prior local_pole_validation results",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(three_bag.CASE_DEFINITIONS.keys()),
        help="repeat to select cases; default: failure1, failure2, success",
    )
    parser.add_argument(
        "--group",
        action="append",
        choices=DEFAULT_GROUPS,
        help="repeat to select PID groups; default: xy, z, roll_pitch, yaw",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--covariance-mode",
        type=str,
        default=DEFAULT_COVARIANCE_MODE,
        choices=("conservative_fusion", "overlap_corrected"),
    )
    parser.add_argument(
        "--delay-mode",
        type=str,
        default=DEFAULT_DELAY_MODE,
        choices=("fitted_thrust_delay", "zero_thrust_delay"),
    )
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--scale-min", type=float, default=DEFAULT_SCALE_MIN)
    parser.add_argument("--scale-max", type=float, default=DEFAULT_SCALE_MAX)
    parser.add_argument("--fd-ratio", type=float, default=DEFAULT_FD_RATIO)
    parser.add_argument(
        "--max-sample-count",
        type=int,
        default=DEFAULT_MAX_SAMPLE_COUNT,
        help="use the first N fixed plant samples from prior local-pole outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_HERE / "outputs" / "pid_gain_region",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="independent bag/group workers; sample order within each is fixed",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    cases = tuple(arguments.case or DEFAULT_CASES)
    groups = tuple(arguments.group or DEFAULT_GROUPS)
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    workers = int(arguments.workers)
    if workers <= 0:
        raise ValueError("workers must be positive")
    tasks = [
        {
            "case_name": case_name,
            "group_name": group_name,
            "controller_yaml": arguments.controller_yaml,
            "vehicle_model": arguments.vehicle_model,
            "outputs_root": arguments.outputs_root,
            "covariance_mode": arguments.covariance_mode,
            "delay_mode": arguments.delay_mode,
            "max_sample_count": arguments.max_sample_count,
            "alpha": arguments.alpha,
            "grid_size": arguments.grid_size,
            "scale_min": arguments.scale_min,
            "scale_max": arguments.scale_max,
            "fd_ratio": arguments.fd_ratio,
            "output_dir": output_dir,
        }
        for case_name in cases
        for group_name in groups
    ]
    if workers == 1:
        summary_rows = [_run_group_task(task) for task in tasks]
    else:
        summary_rows = []
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            futures = [pool.submit(_run_group_task, task) for task in tasks]
            for future in as_completed(futures):
                summary_rows.append(future.result())
    order = {
        (case_name, group_name): index
        for index, (case_name, group_name) in enumerate(
            (case_name, group_name)
            for case_name in cases
            for group_name in groups
        )
    }
    summary_rows.sort(key=lambda row: order[(row["case"], row["group"])])

    summary = {
        "schema": SCHEMA + "-summary",
        "source_commit": local_poles.source_commit(),
        "alpha": float(arguments.alpha),
        "covariance_mode": arguments.covariance_mode,
        "delay_mode": arguments.delay_mode,
        "grid_size": int(arguments.grid_size),
        "scale_min": float(arguments.scale_min),
        "scale_max": float(arguments.scale_max),
        "fd_ratio": float(arguments.fd_ratio),
        "max_sample_count": int(arguments.max_sample_count),
        "rows": summary_rows,
    }
    local_poles.write_json(output_dir / "pid_gain_region_summary.json", summary)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
