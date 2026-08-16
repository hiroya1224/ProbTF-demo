#!/usr/bin/env python3
"""CoG-conditioned reseeding followed by the unchanged prior-free estimator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
for _path in (_HERE, _SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/grape-cog-reseed-matplotlib")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from grape_param_estim.real_rosbag import load_flight_data  # noqa: E402
from single_bag_savgol_core import (  # noqa: E402
    COMMON_SCALE_DIRECTION,
    SiParameterChart,
    SingleBagDynamicsProblem,
    load_vehicle_model,
    metric_cross_evaluation,
    prepare_single_bag_dataset,
)
from single_bag_savgol_estimator import (  # noqa: E402
    actuator_parameters_from_arguments,
    estimator_config_from_arguments,
    run_estimator,
)
from single_bag_savgol_reports import (  # noqa: E402
    output_run_directory,
    source_commit,
    write_json,
)


BASELINE_REPOSITORY_COMMIT = "b2daa3066ed1838b213d88f5a5f10abba1d3ab35"
BASELINE_CASE_SOURCE_COMMIT = "d575adf7062789188e10e3356b0d1a1f9dfb725a"
CONDITIONING_SOURCE_COVARIANCE_NAME = (
    "parameter_covariance_conservative_fusion"
)
PRODUCTION_COG_PRIOR_STD_M = 0.001
REFINEMENT_INTERPRETATION = (
    "The CoG Gaussian prior is used only to condition the saved joint "
    "distribution and generate a nonlinear refinement initialization. The "
    "refinement objective is the unchanged prior-free pose-derived data "
    "objective. CoG, inertia, force effectiveness, and rotor lag remain free "
    "during refinement."
)
BASELINE_RUN_BY_BAG_ID = {
    "single_rosbag_1": (
        "single_rosbag_1_conservative_fusion_production_20260816"
    ),
    "single_rosbag_2": (
        "single_rosbag_2_conservative_fusion_production_20260816"
    ),
    "single_rosbag_succeeded": (
        "single_rosbag_succeeded_conservative_fusion_production_20260816"
    ),
}
_BASELINE_REQUIRED_FILES = (
    "result.json",
    "arguments.json",
    "arrays.npz",
    "status.json",
)
_ALLOWED_REFINEMENT_ARGUMENT_CHANGES = {
    "initial_coordinate",
    "initial_rotor_lag",
    "scale_initial_offset",
    "output_root",
    "run_id",
}


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CogConditioningResult:
    original_mean: np.ndarray
    original_covariance: np.ndarray
    cog_selector: np.ndarray
    prior_mean_chart: np.ndarray
    prior_std_m: np.ndarray
    prior_covariance_m2: np.ndarray
    innovation_covariance: np.ndarray
    conditioning_gain: np.ndarray
    conditioned_mean: np.ndarray
    conditioned_covariance: np.ndarray
    covariance_symmetry_error: float
    covariance_min_eigenvalue: float
    update_scale_gauge_dot: float
    covariance_scale_gauge_norm: float
    innovation_condition_number: float


@dataclass(frozen=True)
class BaselineCase:
    directory: Path
    bag_id: str
    result: Mapping[str, Any]
    arguments: Mapping[str, Any]
    chart_coordinate: np.ndarray
    conservative_covariance: np.ndarray
    immutable_file_sha256: Mapping[str, str]


def condition_chart_gaussian_on_cog_prior(
    mean: Sequence[float],
    covariance: np.ndarray,
    *,
    cog_std_m: Sequence[float] | float,
) -> CogConditioningResult:
    """Condition a 14-D chart Gaussian on zero CoG displacement.

    Only the strictly positive-definite 3-D innovation system is solved.  The
    singular full chart covariance is never inverted or pseudoinverted.
    """

    original_mean = np.asarray(mean, dtype=float)
    original_covariance = np.asarray(covariance, dtype=float)
    prior_std = np.asarray(cog_std_m, dtype=float)
    if prior_std.ndim == 0:
        prior_std = np.repeat(prior_std, 3)
    if (
        original_mean.shape != (14,)
        or original_covariance.shape != (14, 14)
        or prior_std.shape != (3,)
        or np.any(~np.isfinite(original_mean))
        or np.any(~np.isfinite(original_covariance))
        or np.any(~np.isfinite(prior_std))
        or np.any(prior_std <= 0.0)
    ):
        raise ValueError("CoG conditioning inputs must be finite with valid shapes")
    covariance_scale = max(
        float(np.max(np.abs(original_covariance))), np.finfo(float).tiny
    )
    covariance_tolerance = 14.0 * np.finfo(float).eps * covariance_scale
    if not np.allclose(
        original_covariance,
        original_covariance.T,
        rtol=0.0,
        atol=covariance_tolerance,
    ):
        raise ValueError("input chart covariance must be symmetric")

    selector = np.zeros((3, 14), dtype=float)
    selector[:, 7:10] = np.eye(3)
    prior_mean = np.zeros(3, dtype=float)
    prior_covariance = np.diag(prior_std**2)
    innovation = (
        selector @ original_covariance @ selector.T + prior_covariance
    )
    innovation_eigenvalues = np.linalg.eigvalsh(innovation)
    if np.any(innovation_eigenvalues <= 0.0):
        raise ValueError("CoG innovation covariance must be positive definite")

    cross_covariance = original_covariance @ selector.T
    gain = np.linalg.solve(innovation, cross_covariance.T).T
    innovation_mean = selector @ original_mean - prior_mean
    conditioned_mean = original_mean - gain @ innovation_mean
    conditioned_covariance_raw = (
        original_covariance - gain @ selector @ original_covariance
    )
    symmetry_error = float(
        np.max(
            np.abs(
                conditioned_covariance_raw - conditioned_covariance_raw.T
            )
        )
    )
    conditioned_covariance = 0.5 * (
        conditioned_covariance_raw + conditioned_covariance_raw.T
    )
    gauge = np.asarray(COMMON_SCALE_DIRECTION, dtype=float)
    return CogConditioningResult(
        original_mean=_readonly(original_mean),
        original_covariance=_readonly(original_covariance),
        cog_selector=_readonly(selector),
        prior_mean_chart=_readonly(prior_mean),
        prior_std_m=_readonly(prior_std),
        prior_covariance_m2=_readonly(prior_covariance),
        innovation_covariance=_readonly(innovation),
        conditioning_gain=_readonly(gain),
        conditioned_mean=_readonly(conditioned_mean),
        conditioned_covariance=_readonly(conditioned_covariance),
        covariance_symmetry_error=symmetry_error,
        covariance_min_eigenvalue=float(
            np.linalg.eigvalsh(conditioned_covariance)[0]
        ),
        update_scale_gauge_dot=float(
            gauge @ (conditioned_mean - original_mean)
        ),
        covariance_scale_gauge_norm=float(
            np.linalg.norm(conditioned_covariance @ gauge)
        ),
        innovation_condition_number=float(np.linalg.cond(innovation)),
    )


def validate_production_conditioning(result: CogConditioningResult) -> None:
    """Reject material violations without projecting or clipping the result."""

    gauge = np.asarray(COMMON_SCALE_DIRECTION, dtype=float)
    update = result.conditioned_mean - result.original_mean
    update_tolerance = (
        200.0
        * np.finfo(float).eps
        * np.linalg.norm(gauge)
        * max(float(np.linalg.norm(update)), 1.0)
    )
    if abs(result.update_scale_gauge_dot) > update_tolerance:
        raise RuntimeError("CoG conditioning injected a common-scale displacement")
    covariance_scale = max(
        float(np.linalg.norm(result.conditioned_covariance, ord=2)),
        np.finfo(float).tiny,
    )
    gauge_tolerance = (
        200.0
        * np.finfo(float).eps
        * covariance_scale
        * np.linalg.norm(gauge)
    )
    if result.covariance_scale_gauge_norm > gauge_tolerance:
        raise RuntimeError("conditioned covariance lost the common-scale null")
    eigenvalues = np.linalg.eigvalsh(result.conditioned_covariance)
    psd_tolerance = (
        result.conditioned_covariance.shape[0]
        * np.finfo(float).eps
        * max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    )
    if eigenvalues[0] < -psd_tolerance:
        raise RuntimeError("conditioned covariance is materially indefinite")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(label))
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_baseline_directory(bag_id: str) -> Path:
    if bag_id not in BASELINE_RUN_BY_BAG_ID:
        raise ValueError("unknown production baseline bag id: {}".format(bag_id))
    return (
        _HERE
        / "outputs"
        / BASELINE_CASE_SOURCE_COMMIT
        / "default"
        / BASELINE_RUN_BY_BAG_ID[bag_id]
    )


def load_completed_baseline_case(
    directory: Path,
    *,
    expected_directory: Optional[Path] = None,
    expected_source_commit: str = BASELINE_CASE_SOURCE_COMMIT,
) -> BaselineCase:
    """Load a completed case read-only and verify its saved distribution."""

    case = Path(directory).expanduser().resolve()
    if expected_directory is not None and case != Path(expected_directory).resolve():
        raise ValueError("baseline case directory does not match the exact input")
    for name in _BASELINE_REQUIRED_FILES:
        path = case / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError("baseline case is missing {}".format(name))
    hashes = {
        name: _file_sha256(case / name) for name in _BASELINE_REQUIRED_FILES
    }
    result = _read_json_object(case / "result.json", "baseline result")
    status = _read_json_object(case / "status.json", "baseline status")
    arguments = _read_json_object(case / "arguments.json", "baseline arguments")
    if (
        result.get("status") != "completed"
        or result.get("strict_final_evaluation") is not True
        or status.get("status") != "completed"
    ):
        raise ValueError("baseline case is not a completed strict result")
    if (
        result.get("source_commit") != expected_source_commit
        or status.get("source_commit") != expected_source_commit
    ):
        raise ValueError("baseline case source commit is not the required revision")
    bag_id = str(arguments.get("bag_id"))
    coordinate = np.asarray(
        result["parameters"]["chart_coordinate"], dtype=float
    )
    with np.load(case / "arrays.npz") as arrays:
        covariance = np.asarray(
            arrays[CONDITIONING_SOURCE_COVARIANCE_NAME], dtype=float
        ).copy()
    json_covariance = np.asarray(
        result["uncertainty"][CONDITIONING_SOURCE_COVARIANCE_NAME],
        dtype=float,
    )
    if coordinate.shape != (14,) or covariance.shape != (14, 14):
        raise ValueError("baseline chart distribution has invalid dimensions")
    if not np.allclose(
        covariance, json_covariance, rtol=2.0e-14, atol=2.0e-14
    ):
        raise ValueError("baseline JSON and NPZ covariances disagree")
    return BaselineCase(
        directory=case,
        bag_id=bag_id,
        result=result,
        arguments=arguments,
        chart_coordinate=_readonly(coordinate),
        conservative_covariance=_readonly(covariance),
        immutable_file_sha256=hashes,
    )


def verify_baseline_files_unchanged(baseline: BaselineCase) -> None:
    for name, expected in baseline.immutable_file_sha256.items():
        if _file_sha256(baseline.directory / name) != expected:
            raise RuntimeError("immutable baseline file changed: {}".format(name))


def _physical_payload(parameters: Any) -> dict[str, Any]:
    inertia = np.asarray(parameters.inertia, dtype=float)
    mass = float(parameters.mass)
    effectiveness = np.asarray(parameters.force_effectiveness, dtype=float)
    return {
        "mass_kg": mass,
        "inertia_kg_m2": inertia,
        "principal_inertia_moments_kg_m2": np.linalg.eigvalsh(inertia),
        "cog_position_body_m": np.asarray(parameters.cog_offset, dtype=float),
        "force_effectiveness": effectiveness,
        "inertia_over_mass_m2": inertia / mass,
        "force_effectiveness_over_mass": effectiveness / mass,
    }


def _result_stage_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    estimated = result["parameters"]["estimated"]
    inertia = np.asarray(estimated["inertia_kg_m2"], dtype=float)
    mass = float(estimated["mass_kg"])
    effectiveness = np.asarray(estimated["force_effectiveness"], dtype=float)
    common = result["common_evaluation"]
    return {
        "chart_coordinate": np.asarray(
            result["parameters"]["chart_coordinate"], dtype=float
        ),
        "rotor_lag_seconds": float(
            result["parameters"]["rotor_lag_seconds"]
        ),
        "strict_identity_objective_sum": float(
            common["identity_objective_sum"]
        ),
        "specific_acceleration_rmse_m_per_s2": float(
            common["specific_acceleration_rmse_m_per_s2"]
        ),
        "angular_acceleration_rmse_rad_per_s2": float(
            common["angular_acceleration_rmse_rad_per_s2"]
        ),
        "mass_kg": mass,
        "inertia_kg_m2": inertia,
        "principal_inertia_moments_kg_m2": np.linalg.eigvalsh(inertia),
        "cog_position_body_m": np.asarray(
            estimated["cog_position_body_m"], dtype=float
        ),
        "force_effectiveness": effectiveness,
        "inertia_over_mass_m2": inertia / mass,
        "force_effectiveness_over_mass": effectiveness / mass,
    }


def _nominal_mass_gauge_parameters(
    chart: SiParameterChart, coordinate: np.ndarray
) -> dict[str, Any]:
    fixed_coordinate = (
        np.asarray(coordinate)
        - float(coordinate[0]) * np.asarray(COMMON_SCALE_DIRECTION)
    )
    payload = _physical_payload(chart.decode(fixed_coordinate))
    payload["chart_coordinate"] = fixed_coordinate
    return payload


def build_refinement_arguments(
    baseline_arguments: Mapping[str, Any],
    conditioned_chart_coordinate: np.ndarray,
    original_rotor_lag_seconds: float,
    *,
    output_root: Path,
    run_id: str,
) -> argparse.Namespace:
    """Clone the baseline arguments and alter initialization/output only."""

    values = {
        key: value
        for key, value in baseline_arguments.items()
        if key != "base_plan_commit"
    }
    coordinate = np.asarray(conditioned_chart_coordinate, dtype=float)
    if coordinate.shape != (14,) or np.any(~np.isfinite(coordinate)):
        raise ValueError("conditioned initializer must be finite and 14-D")
    values.update(
        {
            "initial_coordinate": coordinate.copy(),
            "initial_rotor_lag": float(original_rotor_lag_seconds),
            "scale_initial_offset": 0.0,
            "output_root": Path(output_root),
            "run_id": str(run_id),
            "_bag_json_resolved": True,
        }
    )
    for key in ("bag", "bag_json", "vehicle_model"):
        if values.get(key) is not None:
            values[key] = Path(values[key])
    if values.get("lag_mode") == "estimated" and values.get("fixed_rotor_lag") is not None:
        raise ValueError("estimated-lag baseline unexpectedly has a fixed lag")
    forbidden = [key for key in values if "prior" in key.lower()]
    if forbidden:
        raise ValueError("prior fields cannot enter estimator arguments")
    return argparse.Namespace(**values)


def refinement_argument_audit(
    baseline_arguments: Mapping[str, Any],
    refinement_arguments: argparse.Namespace,
) -> dict[str, Any]:
    def normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    changed = {}
    refinement_values = vars(refinement_arguments)
    for key, original in baseline_arguments.items():
        if key == "base_plan_commit":
            continue
        refined = refinement_values.get(key)
        if normalize(original) != normalize(refined):
            changed[key] = {"original": normalize(original), "refined": normalize(refined)}
    unexpected = set(changed) - _ALLOWED_REFINEMENT_ARGUMENT_CHANGES
    if unexpected:
        raise RuntimeError(
            "refinement changed scientific arguments: {}".format(
                ", ".join(sorted(unexpected))
            )
        )
    return {
        "allowed_changed_fields": sorted(changed),
        "changes": changed,
        "all_other_scientific_arguments_preserved": True,
    }


def execute_prior_free_refinement(
    arguments: argparse.Namespace,
    output_directory: Path,
    *,
    estimator_runner: Callable[..., tuple[Path, Mapping[str, Any]]] = run_estimator,
) -> tuple[Path, Mapping[str, Any]]:
    """Call the existing estimator without adding any prior information."""

    return estimator_runner(
        arguments,
        case_name="cog_prior_reseed_refined",
        output_directory=output_directory,
    )


def _prepare_diagnostic_problem(
    arguments: argparse.Namespace,
) -> tuple[SingleBagDynamicsProblem, Any]:
    config = estimator_config_from_arguments(arguments)
    actuator = actuator_parameters_from_arguments(arguments)
    model = load_vehicle_model(arguments.vehicle_model)
    flight = load_flight_data(
        path=str(arguments.bag),
        start_local=float(arguments.bag_start),
        end_local=float(arguments.bag_end),
        include_fc_specific_force=True,
        compute_sha256=not arguments.skip_bag_sha256,
        bag_id=arguments.bag_id,
    )
    dataset = prepare_single_bag_dataset(
        flight=flight,
        window_seconds=float(arguments.sg_window),
        degree=int(arguments.sg_degree),
        covariance_mode=config.covariance_mode,
        geometric_correction=config.geometric_correction,
    )
    return (
        SingleBagDynamicsProblem(
            dataset,
            model,
            actuator,
            gimbal_source=config.gimbal_source,
        ),
        model,
    )


def _metadata(
    *, baseline: BaselineCase, source_revision: str
) -> dict[str, Any]:
    return {
        "source_commit": source_revision,
        "baseline_repository_commit": BASELINE_REPOSITORY_COMMIT,
        "baseline_case_source_commit": BASELINE_CASE_SOURCE_COMMIT,
        "baseline_case_directory": baseline.directory,
        "conditioning_source_covariance_name": (
            CONDITIONING_SOURCE_COVARIANCE_NAME
        ),
        "conditioning_prior_role": "initialization_only",
        "refinement_prior_role": "none",
        "refinement_all_parameters_free": True,
        "refinement_lag_reestimated": True,
        "refinement_existing_estimator_reused": True,
    }


def _conditioning_payload(
    baseline: BaselineCase,
    conditioning: CogConditioningResult,
    chart: SiParameterChart,
    source_revision: str,
) -> dict[str, Any]:
    original = baseline.result
    conditioned_physical = _physical_payload(
        chart.decode(conditioning.conditioned_mean)
    )
    return {
        **_metadata(baseline=baseline, source_revision=source_revision),
        "status": "conditioned",
        "original_chart_coordinate": conditioning.original_mean,
        "original_parameter_covariance_conservative_fusion": (
            conditioning.original_covariance
        ),
        "original_rotor_lag_seconds": float(
            original["parameters"]["rotor_lag_seconds"]
        ),
        "original_strict_identity_objective": float(
            original["common_evaluation"]["identity_objective_sum"]
        ),
        "original_nominal_mass_gauge_parameters": (
            _nominal_mass_gauge_parameters(chart, conditioning.original_mean)
        ),
        "cog_prior_mean_physical_m": np.asarray(
            chart.reference.cog_offset, dtype=float
        ),
        "cog_prior_mean_chart": conditioning.prior_mean_chart,
        "cog_prior_std_m": conditioning.prior_std_m,
        "cog_prior_covariance_m2": conditioning.prior_covariance_m2,
        "cog_selector_H": conditioning.cog_selector,
        "innovation_covariance": conditioning.innovation_covariance,
        "conditioning_gain": conditioning.conditioning_gain,
        "conditioned_chart_coordinate": conditioning.conditioned_mean,
        "conditioned_parameter_covariance": (
            conditioning.conditioned_covariance
        ),
        "conditioned_mass_kg": conditioned_physical["mass_kg"],
        "conditioned_inertia_kg_m2": conditioned_physical["inertia_kg_m2"],
        "conditioned_principal_inertia_moments_kg_m2": (
            conditioned_physical["principal_inertia_moments_kg_m2"]
        ),
        "conditioned_cog_position_body_m": (
            conditioned_physical["cog_position_body_m"]
        ),
        "conditioned_force_effectiveness": (
            conditioned_physical["force_effectiveness"]
        ),
        "conditioned_scale_free_inertia_over_mass_m2": (
            conditioned_physical["inertia_over_mass_m2"]
        ),
        "conditioned_scale_free_force_effectiveness_over_mass": (
            conditioned_physical["force_effectiveness_over_mass"]
        ),
        "conditioning_covariance_symmetry_error": (
            conditioning.covariance_symmetry_error
        ),
        "conditioning_covariance_min_eigenvalue": (
            conditioning.covariance_min_eigenvalue
        ),
        "conditioning_update_scale_gauge_dot": (
            conditioning.update_scale_gauge_dot
        ),
        "conditioning_covariance_scale_gauge_norm": (
            conditioning.covariance_scale_gauge_norm
        ),
        "conditioning_innovation_condition_number": (
            conditioning.innovation_condition_number
        ),
    }


def _write_conditioning_arrays(
    path: Path, conditioning: CogConditioningResult
) -> None:
    np.savez_compressed(
        path,
        original_chart_coordinate=conditioning.original_mean,
        original_parameter_covariance_conservative_fusion=(
            conditioning.original_covariance
        ),
        cog_prior_mean_chart=conditioning.prior_mean_chart,
        cog_prior_std_m=conditioning.prior_std_m,
        cog_prior_covariance_m2=conditioning.prior_covariance_m2,
        cog_selector_H=conditioning.cog_selector,
        innovation_covariance=conditioning.innovation_covariance,
        conditioning_gain=conditioning.conditioning_gain,
        conditioned_chart_coordinate=conditioning.conditioned_mean,
        conditioned_parameter_covariance=conditioning.conditioned_covariance,
    )


def _comparison_payload(
    *,
    baseline: BaselineCase,
    conditioning_payload: Mapping[str, Any],
    refined_result: Mapping[str, Any],
    refined_arrays_path: Path,
    source_revision: str,
) -> dict[str, Any]:
    original = _result_stage_payload(baseline.result)
    refined = _result_stage_payload(refined_result)
    conditioned = {
        "chart_coordinate": conditioning_payload[
            "conditioned_chart_coordinate"
        ],
        "evaluation_rotor_lag_seconds": conditioning_payload[
            "original_rotor_lag_seconds"
        ],
        "strict_identity_objective_sum": conditioning_payload[
            "conditioned_seed_identity_objective_at_original_lag"
        ],
        "specific_acceleration_rmse_m_per_s2": conditioning_payload[
            "conditioned_seed_specific_acceleration_rmse"
        ],
        "angular_acceleration_rmse_rad_per_s2": conditioning_payload[
            "conditioned_seed_angular_acceleration_rmse"
        ],
        "mass_kg": conditioning_payload["conditioned_mass_kg"],
        "inertia_kg_m2": conditioning_payload["conditioned_inertia_kg_m2"],
        "principal_inertia_moments_kg_m2": conditioning_payload[
            "conditioned_principal_inertia_moments_kg_m2"
        ],
        "cog_position_body_m": conditioning_payload[
            "conditioned_cog_position_body_m"
        ],
        "force_effectiveness": conditioning_payload[
            "conditioned_force_effectiveness"
        ],
        "inertia_over_mass_m2": conditioning_payload[
            "conditioned_scale_free_inertia_over_mass_m2"
        ],
        "force_effectiveness_over_mass": conditioning_payload[
            "conditioned_scale_free_force_effectiveness_over_mass"
        ],
    }
    delta_objective = (
        refined["strict_identity_objective_sum"]
        - original["strict_identity_objective_sum"]
    )
    with np.load(baseline.directory / "arrays.npz") as arrays:
        original_quotient_covariance = np.asarray(
            arrays["quotient_covariance_conservative_fusion"]
        ).copy()
    with np.load(refined_arrays_path) as arrays:
        refined_quotient_covariance = np.asarray(
            arrays["quotient_covariance_conservative_fusion"]
        ).copy()
    return {
        **_metadata(baseline=baseline, source_revision=source_revision),
        "status": "completed",
        "interpretation": REFINEMENT_INTERPRETATION,
        "original": original,
        "conditioned": conditioned,
        "refined": refined,
        "delta_L_refined_minus_original": delta_objective,
        "relative_delta_L_refined_minus_original": (
            delta_objective
            / max(
                original["strict_identity_objective_sum"],
                np.finfo(float).eps,
            )
        ),
        "refined_minus_original_chart": (
            refined["chart_coordinate"] - original["chart_coordinate"]
        ),
        "conditioned_minus_original_chart": (
            np.asarray(conditioned["chart_coordinate"])
            - original["chart_coordinate"]
        ),
        "refined_minus_original_cog_m": (
            refined["cog_position_body_m"]
            - original["cog_position_body_m"]
        ),
        "refined_minus_original_force_effectiveness": (
            refined["force_effectiveness"]
            - original["force_effectiveness"]
        ),
        "refined_minus_original_inertia_over_mass_m2": (
            refined["inertia_over_mass_m2"]
            - original["inertia_over_mass_m2"]
        ),
        "distribution_comparison": {
            "conditioned_covariance_role": "initialization_diagnostic_only",
            "refined_covariance_role": "recomputed_at_refined_solution",
            "original_quotient_covariance_conservative_fusion": (
                original_quotient_covariance
            ),
            "refined_quotient_covariance_conservative_fusion": (
                refined_quotient_covariance
            ),
            "original_quotient_covariance_eigenvalues": np.linalg.eigvalsh(
                original_quotient_covariance
            ),
            "refined_quotient_covariance_eigenvalues": np.linalg.eigvalsh(
                refined_quotient_covariance
            ),
            "original_quotient_covariance_trace": float(
                np.trace(original_quotient_covariance)
            ),
            "refined_quotient_covariance_trace": float(
                np.trace(refined_quotient_covariance)
            ),
        },
    }


def _write_comparison_pdf(
    path: Path,
    *,
    case_name: str,
    comparison: Mapping[str, Any],
    nominal_cog: np.ndarray,
    original_arrays_path: Path,
    refined_arrays_path: Path,
) -> None:
    original = comparison["original"]
    conditioned = comparison["conditioned"]
    refined = comparison["refined"]
    stages = (original, conditioned, refined)
    labels = ("original", "conditioned seed", "refined")
    colors = ("tab:blue", "tab:orange", "tab:green")
    with np.load(original_arrays_path) as arrays:
        original_time = np.asarray(arrays["sg_time"])
        original_residual = np.asarray(arrays["residual_acceleration"])
    with np.load(refined_arrays_path) as arrays:
        refined_time = np.asarray(arrays["sg_time"])
        refined_residual = np.asarray(arrays["residual_acceleration"])

    with PdfPages(path) as pdf:
        figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
        index = np.arange(3)
        width = 0.24
        for stage_index, (label, color, stage) in enumerate(
            zip(labels, colors, stages)
        ):
            axes[0, 0].bar(
                index + (stage_index - 1) * width,
                np.asarray(stage["cog_position_body_m"]) * 1000.0,
                width,
                label=label,
                color=color,
            )
        axes[0, 0].errorbar(
            index,
            np.asarray(nominal_cog) * 1000.0,
            yerr=np.ones(3),
            fmt="kx",
            capsize=4,
            label="nominal CoG +/- 1 mm",
        )
        axes[0, 0].set_xticks(index, ("x", "y", "z"))
        axes[0, 0].set_ylabel("CoG [mm]")
        axes[0, 0].legend(fontsize=7)
        axes[0, 0].grid(True, alpha=0.3)

        rotor = np.arange(4)
        for stage_index, (label, color, stage) in enumerate(
            zip(labels, colors, stages)
        ):
            axes[0, 1].bar(
                rotor + (stage_index - 1) * width,
                stage["force_effectiveness"],
                width,
                label=label,
                color=color,
            )
        axes[0, 1].set_xticks(rotor, ("f1", "f2", "f3", "f4"))
        axes[0, 1].set_ylabel("force effectiveness")
        axes[0, 1].grid(True, alpha=0.3)

        moment = np.arange(3)
        for stage_index, (label, color, stage) in enumerate(
            zip(labels, colors, stages)
        ):
            axes[1, 0].bar(
                moment + (stage_index - 1) * width,
                stage["principal_inertia_moments_kg_m2"],
                width,
                label=label,
                color=color,
            )
        axes[1, 0].set_xticks(moment, ("J1", "J2", "J3"))
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_ylabel("principal inertia [kg m^2]")
        axes[1, 0].grid(True, alpha=0.3)

        objective = [stage["strict_identity_objective_sum"] for stage in stages]
        axes[1, 1].bar(labels, objective, color=colors)
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_ylabel("identity objective")
        axes[1, 1].tick_params(axis="x", rotation=15)
        axes[1, 1].grid(True, alpha=0.3)
        figure.suptitle("{}: original / conditioned / refined".format(case_name))
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(2, 1, figsize=(11.0, 8.5))
        axes[0].plot(
            original_time - original_time[0],
            np.linalg.norm(original_residual, axis=1),
            label="original",
        )
        axes[0].plot(
            refined_time - refined_time[0],
            np.linalg.norm(refined_residual, axis=1),
            label="refined",
        )
        axes[0].set_ylabel("acceleration residual norm")
        axes[0].set_xlabel("time from SG start [s]")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        component = np.arange(6)
        original_rmse = np.sqrt(np.mean(original_residual**2, axis=0))
        refined_rmse = np.sqrt(np.mean(refined_residual**2, axis=0))
        axes[1].bar(component - 0.18, original_rmse, 0.36, label="original")
        axes[1].bar(component + 0.18, refined_rmse, 0.36, label="refined")
        axes[1].set_xticks(
            component, ("sx", "sy", "sz", "ax", "ay", "az")
        )
        axes[1].set_ylabel("component RMSE")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        figure.suptitle(
            "{}: prior-free fit and rotor lag ({:.7g} -> {:.7g} s)".format(
                case_name,
                original["rotor_lag_seconds"],
                refined["rotor_lag_seconds"],
            )
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        pdf.savefig(figure)
        plt.close(figure)

        figure = plt.figure(figsize=(11.0, 8.5))
        axis = figure.add_subplot(111)
        axis.axis("off")
        lines = [
            "CoG-conditioned reseed and prior-free full refinement",
            "",
            REFINEMENT_INTERPRETATION,
            "",
            "No MAP term, CoG fixing, inertia fixing, force fixing, or lag fixing is used.",
            "The conditioned covariance is not passed to the optimizer.",
            "",
            "objective original / seed / refined:",
            "  {:.12g} / {:.12g} / {:.12g}".format(
                original["strict_identity_objective_sum"],
                conditioned["strict_identity_objective_sum"],
                refined["strict_identity_objective_sum"],
            ),
            "delta refined-original: {:.12g}".format(
                comparison["delta_L_refined_minus_original"]
            ),
            "",
            "J/m original:",
            np.array2string(np.asarray(original["inertia_over_mass_m2"]), precision=8),
            "J/m conditioned:",
            np.array2string(np.asarray(conditioned["inertia_over_mass_m2"]), precision=8),
            "J/m refined:",
            np.array2string(np.asarray(refined["inertia_over_mass_m2"]), precision=8),
        ]
        axis.text(
            0.03,
            0.97,
            "\n".join(lines),
            va="top",
            family="monospace",
            fontsize=8,
            wrap=True,
        )
        figure.suptitle("{}: experiment definition and result".format(case_name))
        pdf.savefig(figure)
        plt.close(figure)


def run_cog_prior_reseed_refinement(
    arguments: argparse.Namespace,
) -> tuple[Path, Mapping[str, Any]]:
    revision = source_commit(_PROJECT_ROOT)
    directory = output_run_directory(
        arguments.output_root,
        "cog_prior_reseed_refinement",
        arguments.run_id,
        commit=revision,
    )
    started = time.perf_counter()
    stage = "baseline_validation"
    baseline: Optional[BaselineCase] = None
    conditioning_payload: Optional[dict[str, Any]] = None
    try:
        candidate = Path(arguments.baseline_case_directory)
        provisional_arguments = _read_json_object(
            candidate / "arguments.json", "baseline arguments"
        )
        bag_id = str(provisional_arguments.get("bag_id"))
        baseline = load_completed_baseline_case(
            candidate,
            expected_directory=expected_baseline_directory(bag_id),
        )
        metadata = _metadata(baseline=baseline, source_revision=revision)
        write_json(
            directory / "arguments.json",
            {
                **metadata,
                "baseline_case_directory": baseline.directory,
                "cog_prior_std_m": np.repeat(
                    float(arguments.cog_prior_std_m), 3
                ),
                "output_root": arguments.output_root,
                "run_id": arguments.run_id,
            },
        )

        stage = "gaussian_conditioning"
        conditioning = condition_chart_gaussian_on_cog_prior(
            baseline.chart_coordinate,
            baseline.conservative_covariance,
            cog_std_m=float(arguments.cog_prior_std_m),
        )
        validate_production_conditioning(conditioning)
        baseline_vehicle_model = Path(baseline.arguments["vehicle_model"])
        model = load_vehicle_model(baseline_vehicle_model)
        chart = SiParameterChart(model.parameters)
        conditioning_payload = _conditioning_payload(
            baseline, conditioning, chart, revision
        )
        write_json(directory / "conditioning.json", conditioning_payload)
        _write_conditioning_arrays(
            directory / "conditioning_arrays.npz", conditioning
        )

        original_lag = float(
            baseline.result["parameters"]["rotor_lag_seconds"]
        )
        refinement_arguments = build_refinement_arguments(
            baseline.arguments,
            conditioning.conditioned_mean,
            original_lag,
            output_root=arguments.output_root,
            run_id="{}_refined".format(arguments.run_id),
        )
        argument_audit = refinement_argument_audit(
            baseline.arguments, refinement_arguments
        )
        if refinement_arguments.lag_mode != "estimated":
            raise RuntimeError("production refinement must re-estimate rotor lag")

        stage = "conditioned_seed_evaluation"
        problem, _diagnostic_model = _prepare_diagnostic_problem(
            refinement_arguments
        )
        conditioned_evaluation = problem.evaluate_physical(
            conditioning.conditioned_mean,
            original_lag,
            command_mode="strict",
        )
        conditioned_metrics = metric_cross_evaluation(conditioned_evaluation)
        conditioned_finite = bool(
            np.isfinite(conditioned_evaluation.cost)
            and np.all(np.isfinite(conditioned_evaluation.acceleration_residual))
        )
        if not conditioned_finite:
            raise RuntimeError("conditioned seed nonlinear evaluation is non-finite")
        conditioning_payload.update(
            {
                "status": "conditioned_and_evaluated",
                "conditioned_seed_identity_objective_at_original_lag": float(
                    conditioned_metrics["identity_objective_sum"]
                ),
                "conditioned_seed_specific_acceleration_rmse": float(
                    conditioned_metrics[
                        "specific_acceleration_rmse_m_per_s2"
                    ]
                ),
                "conditioned_seed_angular_acceleration_rmse": float(
                    conditioned_metrics[
                        "angular_acceleration_rmse_rad_per_s2"
                    ]
                ),
                "conditioned_seed_is_finite": conditioned_finite,
                "refinement_argument_audit": argument_audit,
                "baseline_immutable_file_sha256": (
                    baseline.immutable_file_sha256
                ),
            }
        )
        write_json(directory / "conditioning.json", conditioning_payload)

        stage = "prior_free_full_refinement"
        refined_directory, refined_payload = execute_prior_free_refinement(
            refinement_arguments, directory / "refined"
        )
        if refined_payload.get("status") != "completed":
            raise RuntimeError(
                "prior-free refinement failed: {}".format(
                    refined_payload.get("message", "unknown estimator failure")
                )
            )

        stage = "comparison"
        refined_result = _read_json_object(
            refined_directory / "result.json", "refined result"
        )
        comparison = _comparison_payload(
            baseline=baseline,
            conditioning_payload=conditioning_payload,
            refined_result=refined_result,
            refined_arrays_path=refined_directory / "arrays.npz",
            source_revision=revision,
        )
        write_json(directory / "comparison.json", comparison)
        _write_comparison_pdf(
            directory / "comparison.pdf",
            case_name=baseline.bag_id,
            comparison=comparison,
            nominal_cog=np.asarray(model.parameters.cog_offset),
            original_arrays_path=baseline.directory / "arrays.npz",
            refined_arrays_path=refined_directory / "arrays.npz",
        )
        verify_baseline_files_unchanged(baseline)
        elapsed = time.perf_counter() - started
        status = {
            **metadata,
            "status": "completed",
            "bag_id": baseline.bag_id,
            "refined_directory": refined_directory,
            "elapsed_seconds": elapsed,
        }
        write_json(directory / "status.json", status)
        write_json(directory / "timing.json", {"elapsed_seconds": elapsed})
        return directory, status
    except Exception as error:
        elapsed = time.perf_counter() - started
        failure = {
            "status": "failed",
            "source_commit": revision,
            "baseline_repository_commit": BASELINE_REPOSITORY_COMMIT,
            "baseline_case_source_commit": BASELINE_CASE_SOURCE_COMMIT,
            "conditioning_source_covariance_name": (
                CONDITIONING_SOURCE_COVARIANCE_NAME
            ),
            "conditioning_prior_role": "initialization_only",
            "refinement_prior_role": "none",
            "refinement_all_parameters_free": True,
            "refinement_lag_reestimated": True,
            "refinement_existing_estimator_reused": True,
            "failure_stage": stage,
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": elapsed,
        }
        if baseline is not None:
            failure.update(
                _metadata(baseline=baseline, source_revision=revision)
            )
        write_json(directory / "status.json", failure)
        write_json(directory / "timing.json", {"elapsed_seconds": elapsed})
        return directory, failure


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-case-directory", type=Path, required=True)
    parser.add_argument(
        "--cog-prior-std-m",
        type=float,
        default=PRODUCTION_COG_PRIOR_STD_M,
        help="conditioning-only scalar CoG prior standard deviation",
    )
    parser.add_argument("--output-root", type=Path, default=_HERE / "outputs")
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    directory, payload = run_cog_prior_reseed_refinement(
        build_argument_parser().parse_args(argv)
    )
    print(directory)
    return 0 if payload.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
