#!/usr/bin/env python3
"""Local quotient-space sensitivity analysis for Gimbalrotor PID postprocessing.

This tool is deliberately downstream of the physical estimator.  It does not
re-open a ROS bag, refit the plant, or turn the estimator covariance into a
claim of a calibrated posterior.  It uses the estimator's existing local
common-scale quotient covariance as a deterministic sensitivity envelope.

The primary calculation evaluates the center plus +/- k-sigma along all 13
eigen-directions of the selected quotient covariance.  Optional Gaussian Monte
Carlo sampling is provided as a nonlinear stress test.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from grape_param_estim.controller_config import PID_GROUPS  # noqa: E402
from grape_param_estim.gimbalrotor_pid_postprocess import (  # noqa: E402
    GAIN_GROUP_AXES,
    EstimatorResult,
    PostprocessInputError,
    PostprocessNumericalError,
    ScaleFreePlant,
    VehicleModel,
    build_nominal_controller_allocation,
    build_real_scale_free_allocation,
    characteristic_length,
    dimensionless_effectiveness,
    group_scale,
    load_estimator_result,
    load_vehicle_model,
    source_compatible_pseudoinverse,
)
from single_bag_savgol_core import (  # noqa: E402
    COMMON_SCALE_DIRECTION,
    PHYSICAL_CHART_LABELS,
    SiParameterChart,
)


SENSITIVITY_SCHEMA = (
    "grape-param-estim/gimbalrotor-pid-postprocess-sensitivity/v1"
)
COVARIANCE_MODES = (
    "naive",
    "overlap_corrected",
    "wrench_corrected",
    "conservative_fusion",
)
DEFAULT_COVARIANCE_MODE = "conservative_fusion"
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
SCALE_FREE_UNITS = (
    "m^2",
    "m^2",
    "m^2",
    "m^2",
    "m^2",
    "m^2",
    "m",
    "m",
    "m",
    "kg^-1",
    "kg^-1",
    "kg^-1",
    "kg^-1",
)


@dataclass(frozen=True)
class SensitivityArtifacts:
    source_path: Path
    physical_coordinate: np.ndarray
    coordinate_source: str
    quotient_basis: np.ndarray
    covariance: np.ndarray
    covariance_mode: str
    covariance_source: str

    def __post_init__(self) -> None:
        source = Path(self.source_path).expanduser().resolve()
        coordinate = np.asarray(self.physical_coordinate, dtype=float)
        basis = np.asarray(self.quotient_basis, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        if coordinate.shape != (14,) or np.any(~np.isfinite(coordinate)):
            raise PostprocessInputError(
                "arrays physical_coordinate must be finite and 14-D"
            )
        if basis.shape != (14, 13) or np.any(~np.isfinite(basis)):
            raise PostprocessInputError(
                "arrays quotient_basis must be finite with shape (14, 13)"
            )
        if covariance.shape != (13, 13) or np.any(~np.isfinite(covariance)):
            raise PostprocessInputError(
                "selected quotient covariance must be finite and 13x13"
            )
        gram = basis.T @ basis
        if not np.allclose(gram, np.eye(13), rtol=0.0, atol=5.0e-11):
            raise PostprocessInputError(
                "quotient_basis must be orthonormal"
            )
        scale_direction = np.asarray(COMMON_SCALE_DIRECTION, dtype=float)
        if not np.allclose(
            basis.T @ scale_direction,
            np.zeros(13),
            rtol=0.0,
            atol=5.0e-11,
        ):
            raise PostprocessInputError(
                "quotient_basis is not orthogonal to the common-scale gauge"
            )
        object.__setattr__(self, "source_path", source)
        object.__setattr__(self, "physical_coordinate", coordinate.copy())
        object.__setattr__(self, "quotient_basis", basis.copy())
        object.__setattr__(self, "covariance", covariance.copy())


@dataclass(frozen=True)
class StaticScaleEvaluation:
    scales: Mapping[str, float]
    effectiveness_dimensionless: np.ndarray
    effectiveness_diagonal: np.ndarray
    coupling_ratio: float
    allocation_condition_number: float

    def __post_init__(self) -> None:
        matrix = np.asarray(self.effectiveness_dimensionless, dtype=float)
        diagonal = np.asarray(self.effectiveness_diagonal, dtype=float)
        if (
            tuple(self.scales) != tuple(PID_GROUPS)
            or any(
                not np.isfinite(float(self.scales[group]))
                or float(self.scales[group]) <= 0.0
                for group in PID_GROUPS
            )
            or matrix.shape != (6, 6)
            or np.any(~np.isfinite(matrix))
            or diagonal.shape != (6,)
            or np.any(~np.isfinite(diagonal))
            or not np.isfinite(float(self.coupling_ratio))
            or not np.isfinite(float(self.allocation_condition_number))
        ):
            raise PostprocessNumericalError(
                "static sensitivity evaluation is invalid"
            )
        object.__setattr__(
            self, "effectiveness_dimensionless", matrix.copy()
        )
        object.__setattr__(self, "effectiveness_diagonal", diagonal.copy())


def source_commit(repository_root: Path = _PROJECT_ROOT) -> str:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=repository_root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def scale_free_vector(plant: ScaleFreePlant) -> np.ndarray:
    inertia = np.asarray(plant.inertia_over_mass, dtype=float)
    return np.concatenate(
        (
            inertia[[0, 1, 2], [0, 1, 2]],
            inertia[[0, 0, 1], [1, 2, 2]],
            np.asarray(plant.cog_position_body, dtype=float),
            np.asarray(plant.force_effectiveness_over_mass, dtype=float),
        )
    )


def plant_from_coordinate(
    chart: SiParameterChart,
    coordinate: Sequence[float],
    rotor_lag_seconds: float,
) -> ScaleFreePlant:
    decoded = chart.decode(coordinate)
    return ScaleFreePlant(
        inertia_over_mass=decoded.inertia / decoded.mass,
        cog_position_body=decoded.cog_offset,
        force_effectiveness_over_mass=(
            decoded.force_effectiveness / decoded.mass
        ),
        rotor_lag_seconds=rotor_lag_seconds,
    )


def _selected_covariance(
    arrays: Mapping[str, np.ndarray],
    basis: np.ndarray,
    covariance_mode: str,
) -> tuple[np.ndarray, str]:
    quotient_key = "quotient_covariance_{}".format(covariance_mode)
    if quotient_key in arrays:
        return np.asarray(arrays[quotient_key], dtype=float), quotient_key
    parameter_key = "parameter_covariance_{}".format(covariance_mode)
    if parameter_key in arrays:
        parameter = np.asarray(arrays[parameter_key], dtype=float)
        if parameter.shape != (14, 14):
            raise PostprocessInputError(
                "{} must be 14x14".format(parameter_key)
            )
        return basis.T @ parameter @ basis, (
            "{} projected through quotient_basis".format(parameter_key)
        )
    raise PostprocessInputError(
        "arrays.npz has no covariance for mode {!r}".format(covariance_mode)
    )


def load_sensitivity_artifacts(
    arrays_path: Path,
    *,
    covariance_mode: str = DEFAULT_COVARIANCE_MODE,
) -> SensitivityArtifacts:
    mode = str(covariance_mode)
    if mode not in COVARIANCE_MODES:
        raise PostprocessInputError(
            "covariance_mode must be one of {}".format(
                ", ".join(COVARIANCE_MODES)
            )
        )
    source = Path(arrays_path).expanduser().resolve()
    try:
        loaded = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise PostprocessInputError(
            "arrays.npz cannot be read: {}".format(source)
        ) from error
    try:
        with loaded as arrays:
            if "quotient_basis" not in arrays:
                raise PostprocessInputError(
                    "arrays.npz is missing quotient_basis"
                )
            basis = np.asarray(arrays["quotient_basis"], dtype=float)
            if "physical_coordinate" in arrays:
                coordinate = np.asarray(
                    arrays["physical_coordinate"], dtype=float
                )
                coordinate_source = "physical_coordinate"
            elif "quotient_coordinate" in arrays:
                quotient_coordinate = np.asarray(
                    arrays["quotient_coordinate"], dtype=float
                )
                if basis.shape != (14, 13):
                    raise PostprocessInputError(
                        "arrays quotient_basis must have shape (14, 13)"
                    )
                if quotient_coordinate.shape != (13,):
                    raise PostprocessInputError(
                        "arrays quotient_coordinate must have shape (13,)"
                    )
                coordinate = basis @ quotient_coordinate
                coordinate_source = (
                    "quotient_basis @ quotient_coordinate "
                    "(zero common-scale gauge representative)"
                )
            else:
                raise PostprocessInputError(
                    "arrays.npz is missing both physical_coordinate and "
                    "quotient_coordinate"
                )
            covariance, covariance_source = _selected_covariance(
                arrays, basis, mode
            )
    except PostprocessInputError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise PostprocessInputError(
            "arrays.npz cannot be read: {}".format(source)
        ) from error
    return SensitivityArtifacts(
        source_path=source,
        physical_coordinate=coordinate,
        coordinate_source=coordinate_source,
        quotient_basis=basis,
        covariance=covariance,
        covariance_mode=mode,
        covariance_source=covariance_source,
    )


def _psd_eigendecomposition(
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    matrix = np.asarray(covariance, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = (
        100.0 * matrix.shape[0] * np.finfo(float).eps * scale
    )
    minimum = float(np.min(eigenvalues))
    if minimum < -tolerance:
        raise PostprocessInputError(
            "selected quotient covariance is materially indefinite: "
            "minimum eigenvalue {:.12g}, tolerance {:.12g}".format(
                minimum, tolerance
            )
        )
    clipped = np.maximum(eigenvalues, 0.0)
    order = np.argsort(clipped)[::-1]
    return clipped[order], eigenvectors[:, order], tolerance


def validate_center_matches_result(
    result: EstimatorResult,
    chart: SiParameterChart,
    coordinate: np.ndarray,
) -> ScaleFreePlant:
    decoded = plant_from_coordinate(
        chart, coordinate, result.plant.rotor_lag_seconds
    )
    checks = (
        (
            decoded.inertia_over_mass,
            result.plant.inertia_over_mass,
            "inertia_over_mass",
        ),
        (
            decoded.cog_position_body,
            result.plant.cog_position_body,
            "cog_position_body",
        ),
        (
            decoded.force_effectiveness_over_mass,
            result.plant.force_effectiveness_over_mass,
            "force_effectiveness_over_mass",
        ),
    )
    for actual, expected, name in checks:
        if not np.allclose(
            actual, expected, rtol=2.0e-9, atol=2.0e-11
        ):
            raise PostprocessInputError(
                "arrays physical_coordinate does not reproduce result.json "
                "scale-free {}".format(name)
            )
    return decoded


def evaluate_static_scales(
    plant: ScaleFreePlant,
    model: VehicleModel,
    *,
    nominal_pseudoinverse: np.ndarray,
    characteristic_length_m: float,
) -> StaticScaleEvaluation:
    real = build_real_scale_free_allocation(plant, model)
    if real.source_threshold_rank < 6:
        raise PostprocessNumericalError(
            "sampled real allocation is rank deficient under source threshold"
        )
    effectiveness = real.matrix @ nominal_pseudoinverse
    normalized = dimensionless_effectiveness(
        effectiveness, characteristic_length_m
    )
    scales = {
        group: group_scale(normalized, GAIN_GROUP_AXES[group])
        for group in PID_GROUPS
    }
    norm = float(np.linalg.norm(normalized, ord="fro"))
    coupling = (
        float(
            np.linalg.norm(
                normalized - np.diag(np.diag(normalized)), ord="fro"
            )
            / norm
        )
        if norm > 0.0
        else 0.0
    )
    return StaticScaleEvaluation(
        scales=scales,
        effectiveness_dimensionless=normalized,
        effectiveness_diagonal=np.diag(normalized),
        coupling_ratio=coupling,
        allocation_condition_number=real.condition_number,
    )


def _sample_record(
    *,
    sample_name: str,
    coordinate: np.ndarray,
    center_vector: np.ndarray,
    chart: SiParameterChart,
    result: EstimatorResult,
    model: VehicleModel,
    nominal_pseudoinverse: np.ndarray,
    characteristic_length_m: float,
) -> Mapping[str, Any]:
    try:
        plant = plant_from_coordinate(
            chart, coordinate, result.plant.rotor_lag_seconds
        )
        evaluation = evaluate_static_scales(
            plant,
            model,
            nominal_pseudoinverse=nominal_pseudoinverse,
            characteristic_length_m=characteristic_length_m,
        )
        vector = scale_free_vector(plant)
        return {
            "name": sample_name,
            "valid": True,
            "scale_free_vector": vector.tolist(),
            "scale_free_delta_from_center": (
                vector - center_vector
            ).tolist(),
            "scales": {
                group: float(evaluation.scales[group])
                for group in PID_GROUPS
            },
            "H_dimensionless_diagonal": (
                evaluation.effectiveness_diagonal.tolist()
            ),
            "coupling_ratio": evaluation.coupling_ratio,
            "A_real_condition_number": (
                evaluation.allocation_condition_number
            ),
        }
    except (
        PostprocessInputError,
        PostprocessNumericalError,
        OverflowError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        return {
            "name": sample_name,
            "valid": False,
            "exception_type": type(error).__name__,
            "message": str(error),
        }


def analyze_eigen_directions(
    *,
    result: EstimatorResult,
    artifacts: SensitivityArtifacts,
    model: VehicleModel,
    sigma_multiple: float,
    characteristic_length_override: Optional[float] = None,
) -> Mapping[str, Any]:
    multiplier = float(sigma_multiple)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise PostprocessInputError(
            "sigma_multiple must be finite and positive"
        )
    chart = SiParameterChart(model.parameters)
    center_plant = validate_center_matches_result(
        result, chart, artifacts.physical_coordinate
    )
    center_vector = scale_free_vector(center_plant)
    nominal = build_nominal_controller_allocation(model)
    nominal_pseudoinverse = source_compatible_pseudoinverse(
        nominal.matrix
    )
    length = characteristic_length(
        model, characteristic_length_override
    )
    center = evaluate_static_scales(
        center_plant,
        model,
        nominal_pseudoinverse=nominal_pseudoinverse,
        characteristic_length_m=length,
    )
    eigenvalues, eigenvectors, psd_tolerance = _psd_eigendecomposition(
        artifacts.covariance
    )
    directions = []
    samples = [
        {
            "name": "center",
            "valid": True,
            "scale_free_vector": center_vector.tolist(),
            "scale_free_delta_from_center": np.zeros(13).tolist(),
            "scales": {
                group: float(center.scales[group])
                for group in PID_GROUPS
            },
            "H_dimensionless_diagonal": (
                center.effectiveness_diagonal.tolist()
            ),
            "coupling_ratio": center.coupling_ratio,
            "A_real_condition_number": center.allocation_condition_number,
        }
    ]
    one_sigma_effects = {
        group: np.zeros(13, dtype=float) for group in PID_GROUPS
    }
    for index in range(13):
        eigenvalue = float(eigenvalues[index])
        sigma = float(np.sqrt(eigenvalue))
        quotient_direction = eigenvectors[:, index]
        chart_direction = artifacts.quotient_basis @ quotient_direction
        offset = multiplier * sigma * chart_direction
        minus = _sample_record(
            sample_name="direction_{:02d}_minus".format(index),
            coordinate=artifacts.physical_coordinate - offset,
            center_vector=center_vector,
            chart=chart,
            result=result,
            model=model,
            nominal_pseudoinverse=nominal_pseudoinverse,
            characteristic_length_m=length,
        )
        plus = _sample_record(
            sample_name="direction_{:02d}_plus".format(index),
            coordinate=artifacts.physical_coordinate + offset,
            center_vector=center_vector,
            chart=chart,
            result=result,
            model=model,
            nominal_pseudoinverse=nominal_pseudoinverse,
            characteristic_length_m=length,
        )
        samples.extend((minus, plus))
        direction_report: dict[str, Any] = {
            "direction_index": index,
            "covariance_eigenvalue": eigenvalue,
            "one_sigma_quotient_coordinate": sigma,
            "quotient_eigenvector": quotient_direction.tolist(),
            "physical_chart_direction": chart_direction.tolist(),
            "minus": minus,
            "plus": plus,
            "valid_pair": bool(minus["valid"] and plus["valid"]),
        }
        if minus["valid"] and plus["valid"]:
            physical_one_sigma_effect = (
                np.asarray(
                    plus["scale_free_delta_from_center"], dtype=float
                )
                - np.asarray(
                    minus["scale_free_delta_from_center"], dtype=float
                )
            ) / (2.0 * multiplier)
            direction_report["scale_free_one_sigma_effect"] = (
                physical_one_sigma_effect.tolist()
            )
            scale_effects = {}
            nonlinearities = {}
            for group in PID_GROUPS:
                minus_scale = float(minus["scales"][group])
                plus_scale = float(plus["scales"][group])
                center_scale = float(center.scales[group])
                effect = (
                    plus_scale - minus_scale
                ) / (2.0 * multiplier)
                curvature = (
                    plus_scale + minus_scale - 2.0 * center_scale
                ) / (multiplier * multiplier)
                one_sigma_effects[group][index] = effect
                scale_effects[group] = effect
                nonlinearities[group] = curvature
            direction_report["scale_one_sigma_effect"] = scale_effects
            direction_report["scale_second_difference_per_sigma2"] = (
                nonlinearities
            )
        directions.append(direction_report)

    group_summary = {}
    valid_samples = [sample for sample in samples if sample["valid"]]
    for group in PID_GROUPS:
        center_scale = float(center.scales[group])
        effects = one_sigma_effects[group]
        sigma_linear = float(np.sqrt(np.sum(effects**2)))
        values = np.asarray(
            [sample["scales"][group] for sample in valid_samples],
            dtype=float,
        )
        contribution = effects**2
        contribution_sum = float(np.sum(contribution))
        order = np.argsort(contribution)[::-1]
        top = []
        for direction_index in order[:5]:
            top.append(
                {
                    "direction_index": int(direction_index),
                    "one_sigma_scale_effect": float(
                        effects[direction_index]
                    ),
                    "variance_contribution_fraction": (
                        float(
                            contribution[direction_index]
                            / contribution_sum
                        )
                        if contribution_sum > 0.0
                        else 0.0
                    ),
                }
            )
        group_summary[group] = {
            "center_scale": center_scale,
            "linearized_one_sigma": sigma_linear,
            "relative_linearized_one_sigma": (
                sigma_linear / abs(center_scale)
                if center_scale != 0.0
                else None
            ),
            "sigma_point_min": float(np.min(values)),
            "sigma_point_max": float(np.max(values)),
            "top_eigen_directions": top,
        }

    retained = eigenvalues[
        eigenvalues
        > max(
            psd_tolerance,
            np.finfo(float).eps
            * max(1.0, float(eigenvalues[0])),
        )
    ]
    covariance_condition = (
        float(retained[0] / retained[-1])
        if retained.size > 0
        else None
    )
    return {
        "characteristic_length_m": length,
        "center": {
            "scale_free_vector": center_vector.tolist(),
            "scales": {
                group: float(center.scales[group])
                for group in PID_GROUPS
            },
            "H_dimensionless": (
                center.effectiveness_dimensionless.tolist()
            ),
            "H_dimensionless_diagonal": (
                center.effectiveness_diagonal.tolist()
            ),
            "coupling_ratio": center.coupling_ratio,
            "A_real_condition_number": center.allocation_condition_number,
        },
        "covariance": {
            "mode": artifacts.covariance_mode,
            "source": artifacts.covariance_source,
            "eigenvalues_descending": eigenvalues.tolist(),
            "numerical_rank": int(retained.size),
            "retained_condition_number": covariance_condition,
            "psd_negative_tolerance": psd_tolerance,
            "interpretation": (
                "local quotient-space sensitivity covariance; "
                "not asserted to be a calibrated posterior"
            ),
        },
        "eigen_sampling": {
            "sigma_multiple": multiplier,
            "expected_sample_count": 27,
            "valid_sample_count": len(valid_samples),
            "invalid_sample_count": len(samples) - len(valid_samples),
            "directions": directions,
        },
        "group_summary": group_summary,
    }


def analyze_monte_carlo(
    *,
    result: EstimatorResult,
    artifacts: SensitivityArtifacts,
    model: VehicleModel,
    sample_count: int,
    seed: int,
    characteristic_length_m: float,
) -> Mapping[str, Any]:
    count = int(sample_count)
    if count < 0:
        raise PostprocessInputError(
            "monte_carlo_samples cannot be negative"
        )
    if count == 0:
        return {
            "enabled": False,
            "sample_count_requested": 0,
            "interpretation": (
                "Monte Carlo disabled; deterministic eigen-direction "
                "sensitivity remains available"
            ),
        }
    eigenvalues, eigenvectors, _tolerance = _psd_eigendecomposition(
        artifacts.covariance
    )
    factor = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    chart = SiParameterChart(model.parameters)
    center_plant = validate_center_matches_result(
        result, chart, artifacts.physical_coordinate
    )
    center_vector = scale_free_vector(center_plant)
    nominal = build_nominal_controller_allocation(model)
    nominal_pseudoinverse = source_compatible_pseudoinverse(
        nominal.matrix
    )
    rng = np.random.default_rng(int(seed))
    values = {group: [] for group in PID_GROUPS}
    invalid_reasons: list[Mapping[str, str]] = []
    for sample_index in range(count):
        delta_quotient = factor @ rng.standard_normal(13)
        coordinate = (
            artifacts.physical_coordinate
            + artifacts.quotient_basis @ delta_quotient
        )
        sample = _sample_record(
            sample_name="monte_carlo_{:06d}".format(sample_index),
            coordinate=coordinate,
            center_vector=center_vector,
            chart=chart,
            result=result,
            model=model,
            nominal_pseudoinverse=nominal_pseudoinverse,
            characteristic_length_m=characteristic_length_m,
        )
        if sample["valid"]:
            for group in PID_GROUPS:
                values[group].append(float(sample["scales"][group]))
        elif len(invalid_reasons) < 20:
            invalid_reasons.append(
                {
                    "name": str(sample["name"]),
                    "exception_type": str(sample["exception_type"]),
                    "message": str(sample["message"]),
                }
            )
    valid_count = len(values[PID_GROUPS[0]])
    summaries = {}
    quantiles = (0.025, 0.16, 0.5, 0.84, 0.975)
    for group in PID_GROUPS:
        selected = np.asarray(values[group], dtype=float)
        if selected.size == 0:
            summaries[group] = None
            continue
        q = np.quantile(selected, quantiles)
        summaries[group] = {
            "mean": float(np.mean(selected)),
            "standard_deviation": float(np.std(selected, ddof=0)),
            "min": float(np.min(selected)),
            "max": float(np.max(selected)),
            "quantiles": {
                "q025": float(q[0]),
                "q16": float(q[1]),
                "q50": float(q[2]),
                "q84": float(q[3]),
                "q975": float(q[4]),
            },
        }
    return {
        "enabled": True,
        "sampling_model": (
            "Gaussian draws in the selected local quotient covariance; "
            "used only as a nonlinear sensitivity stress test"
        ),
        "seed": int(seed),
        "sample_count_requested": count,
        "valid_sample_count": valid_count,
        "invalid_sample_count": count - valid_count,
        "invalid_examples": invalid_reasons,
        "group_summary": summaries,
    }


def build_report(
    *,
    revision: str,
    result: EstimatorResult,
    artifacts: SensitivityArtifacts,
    model: VehicleModel,
    eigen_analysis: Mapping[str, Any],
    monte_carlo: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "schema": SENSITIVITY_SCHEMA,
        "source_commit": str(revision),
        "method": "quotient_chart_eigen_direction_sensitivity",
        "input": {
            "estimator_result_json": str(result.source_path),
            "estimator_source_commit": result.source_commit,
            "estimator_case_name": result.case_name,
            "arrays_npz": str(artifacts.source_path),
            "vehicle_model_json": str(model.source_path),
            "covariance_mode": artifacts.covariance_mode,
            "covariance_source": artifacts.covariance_source,
            "coordinate_source": artifacts.coordinate_source,
        },
        "coordinate_contract": {
            "sampling_space": (
                "13-D common-scale quotient of the estimator's 14-D SI chart"
            ),
            "physical_chart_labels": list(PHYSICAL_CHART_LABELS),
            "scale_free_output_labels": list(SCALE_FREE_LABELS),
            "scale_free_output_units": list(SCALE_FREE_UNITS),
            "common_scale_direction": (
                np.asarray(COMMON_SCALE_DIRECTION, dtype=float).tolist()
            ),
            "physical_validity": (
                "samples are decoded through SiParameterChart, preserving "
                "positive mass, positive force effectiveness, and the "
                "inertia second-moment parameterization"
            ),
        },
        "interpretation": {
            "role": "local_sensitivity_analysis",
            "posterior_claim": False,
            "note": (
                "The covariance is used to define local perturbation scales. "
                "In particular, conservative_fusion is a conservative "
                "sensitivity distribution rather than a calibrated "
                "generative posterior."
            ),
            "rotor_lag_treatment": (
                "held fixed at the fitted point because the v1 static H "
                "does not model delay"
            ),
        },
        "center_and_eigen_sensitivity": eigen_analysis,
        "monte_carlo": monte_carlo,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    analysis = report["center_and_eigen_sensitivity"]
    groups = analysis["group_summary"]
    mc = report["monte_carlo"]
    lines = [
        "# Gimbalrotor PID postprocess sensitivity",
        "",
        "This is a local sensitivity analysis of the static PID gain correction.",
        "The selected estimator covariance defines perturbation size; it is not",
        "reported as a calibrated posterior probability.",
        "",
        "- estimator result: `{}`".format(
            report["input"]["estimator_result_json"]
        ),
        "- arrays: `{}`".format(report["input"]["arrays_npz"]),
        "- covariance mode: `{}`".format(
            report["input"]["covariance_mode"]
        ),
        "- center coordinate source: `{}`".format(
            report["input"]["coordinate_source"]
        ),
        "- characteristic length: `{:.9g} m`".format(
            analysis["characteristic_length_m"]
        ),
        "- valid eigen samples: `{}/{}`".format(
            analysis["eigen_sampling"]["valid_sample_count"],
            analysis["eigen_sampling"]["expected_sample_count"],
        ),
        "",
        "## Gain-scale sensitivity",
        "",
        "| group | center | local linear 1-sigma | relative | eigen-point min | eigen-point max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in PID_GROUPS:
        item = groups[group]
        relative = item["relative_linearized_one_sigma"]
        lines.append(
            "| {} | {:.8g} | {:.8g} | {:.3%} | {:.8g} | {:.8g} |".format(
                group,
                item["center_scale"],
                item["linearized_one_sigma"],
                0.0 if relative is None else relative,
                item["sigma_point_min"],
                item["sigma_point_max"],
            )
        )
    lines.extend(
        (
            "",
            "The `local linear 1-sigma` column is reconstructed from the",
            "centered +/- eigen-direction evaluations. A large value means that",
            "the corresponding static PID correction is sensitive to the",
            "identified-plant ridge even when the point estimate itself is fixed.",
            "",
            "## Dominant covariance directions",
            "",
        )
    )
    for group in PID_GROUPS:
        lines.extend(
            (
                "### {}".format(group),
                "",
                "| direction | 1-sigma scale effect | variance contribution |",
                "|---:|---:|---:|",
            )
        )
        for item in groups[group]["top_eigen_directions"]:
            lines.append(
                "| {} | {:+.8g} | {:.2%} |".format(
                    item["direction_index"],
                    item["one_sigma_scale_effect"],
                    item["variance_contribution_fraction"],
                )
            )
        lines.append("")
    if mc.get("enabled", False):
        lines.extend(
            (
                "## Optional Monte Carlo stress test",
                "",
                "| group | mean | std | q16 | median | q84 | q2.5 | q97.5 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for group in PID_GROUPS:
            item = mc["group_summary"][group]
            if item is None:
                lines.append(
                    "| {} | invalid | invalid | invalid | invalid | invalid | invalid | invalid |".format(
                        group
                    )
                )
                continue
            quantile = item["quantiles"]
            lines.append(
                "| {} | {:.8g} | {:.8g} | {:.8g} | {:.8g} | {:.8g} | {:.8g} | {:.8g} |".format(
                    group,
                    item["mean"],
                    item["standard_deviation"],
                    quantile["q16"],
                    quantile["q50"],
                    quantile["q84"],
                    quantile["q025"],
                    quantile["q975"],
                )
            )
        lines.extend(
            (
                "",
                "Monte Carlo here is a nonlinear stress test of the selected local",
                "covariance model. Do not relabel these quantiles as posterior",
                "credible intervals without an independent probabilistic",
                "calibration argument.",
                "",
            )
        )
    else:
        lines.extend(
            (
                "## Monte Carlo",
                "",
                "Disabled for this run. The deterministic 27-point eigen-direction",
                "analysis is the primary result.",
                "",
            )
        )
    return "\n".join(lines) + "\n"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Propagate the estimator's local common-scale quotient "
            "covariance through the static Gimbalrotor PID correction."
        )
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--arrays",
        type=Path,
        help=(
            "Estimator arrays.npz. Defaults to the arrays.npz next to "
            "--result."
        ),
    )
    parser.add_argument("--vehicle-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--covariance-mode",
        choices=COVARIANCE_MODES,
        default=DEFAULT_COVARIANCE_MODE,
    )
    parser.add_argument(
        "--sigma-multiple",
        type=float,
        default=1.0,
        help="Evaluate +/- this many local standard deviations.",
    )
    parser.add_argument(
        "--monte-carlo-samples",
        type=int,
        default=0,
        help=(
            "Optional Gaussian quotient-space stress-test sample count. "
            "Zero disables it."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--characteristic-length", type=float)
    parser.add_argument("--allow-non-prior-free-result", action="store_true")
    parser.add_argument("--allow-point-estimate-only", action="store_true")
    return parser


def execute(arguments: argparse.Namespace) -> Mapping[str, Any]:
    result = load_estimator_result(
        arguments.result,
        allow_non_prior_free_result=(
            arguments.allow_non_prior_free_result
        ),
        allow_point_estimate_only=arguments.allow_point_estimate_only,
    )
    arrays_path = (
        arguments.arrays
        if arguments.arrays is not None
        else Path(arguments.result).expanduser().resolve().parent
        / "arrays.npz"
    )
    artifacts = load_sensitivity_artifacts(
        arrays_path, covariance_mode=arguments.covariance_mode
    )
    model = load_vehicle_model(arguments.vehicle_model)
    eigen_analysis = analyze_eigen_directions(
        result=result,
        artifacts=artifacts,
        model=model,
        sigma_multiple=arguments.sigma_multiple,
        characteristic_length_override=arguments.characteristic_length,
    )
    monte_carlo = analyze_monte_carlo(
        result=result,
        artifacts=artifacts,
        model=model,
        sample_count=arguments.monte_carlo_samples,
        seed=arguments.seed,
        characteristic_length_m=eigen_analysis[
            "characteristic_length_m"
        ],
    )
    report = build_report(
        revision=source_commit(),
        result=result,
        artifacts=artifacts,
        model=model,
        eigen_analysis=eigen_analysis,
        monte_carlo=monte_carlo,
    )
    directory = arguments.output_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "pid_gain_sensitivity.json", report)
    markdown = render_markdown(report)
    (directory / "pid_gain_sensitivity.md").write_text(
        markdown, encoding="utf-8"
    )
    write_json(
        directory / "status.json",
        {
            "schema": SENSITIVITY_SCHEMA + "/status/v1",
            "status": "completed",
            "source_commit": report["source_commit"],
            "covariance_mode": report["input"]["covariance_mode"],
            "valid_eigen_sample_count": report[
                "center_and_eigen_sensitivity"
            ]["eigen_sampling"]["valid_sample_count"],
            "invalid_eigen_sample_count": report[
                "center_and_eigen_sensitivity"
            ]["eigen_sampling"]["invalid_sample_count"],
            "monte_carlo_enabled": bool(
                report["monte_carlo"].get("enabled", False)
            ),
        },
    )
    print(markdown, end="")
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
