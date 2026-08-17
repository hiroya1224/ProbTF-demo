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
    load_estimator_result,
    load_vehicle_model,
    source_compatible_pseudoinverse,
)
from single_bag_savgol_core import (  # noqa: E402
    COMMON_SCALE_DIRECTION,
    PHYSICAL_CHART_LABELS,
    SYMMETRIC_BASIS,
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
COORDINATE_MODES = (
    "estimator_quotient",
    "centered_scale_free_spd",
)
DEFAULT_COORDINATE_MODE = "estimator_quotient"
DEFAULT_DERIVATIVE_SIGMA_FRACTION = 1.0e-5
CENTERED_SCALE_FREE_SPD_LABELS = (
    "log_second_moment_11",
    "log_second_moment_22",
    "log_second_moment_33",
    "log_second_moment_12_sqrt2",
    "log_second_moment_13_sqrt2",
    "log_second_moment_23_sqrt2",
    "delta_cog_x",
    "delta_cog_y",
    "delta_cog_z",
    "log_f1_over_m_ratio",
    "log_f2_over_m_ratio",
    "log_f3_over_m_ratio",
    "log_f4_over_m_ratio",
)
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
    allocation_source_threshold_rank: int
    allocation_condition_number: float

    def __post_init__(self) -> None:
        matrix = np.asarray(self.effectiveness_dimensionless, dtype=float)
        diagonal = np.asarray(self.effectiveness_diagonal, dtype=float)
        if (
            tuple(self.scales) != tuple(PID_GROUPS)
            or any(
                not np.isfinite(float(self.scales[group]))
                for group in PID_GROUPS
            )
            or matrix.shape != (6, 6)
            or np.any(~np.isfinite(matrix))
            or diagonal.shape != (6,)
            or np.any(~np.isfinite(diagonal))
            or not np.isfinite(float(self.coupling_ratio))
            or int(self.allocation_source_threshold_rank) < 0
            or int(self.allocation_source_threshold_rank) > 6
            or np.isnan(float(self.allocation_condition_number))
            or float(self.allocation_condition_number) < 0.0
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


def _json_condition_number(value: float) -> Any:
    """Return a strict-JSON value without discarding positive infinity."""

    selected = float(value)
    if np.isposinf(selected):
        return "infinity"
    if not np.isfinite(selected) or selected < 0.0:
        raise PostprocessNumericalError(
            "condition number must be non-negative or positive infinity"
        )
    return selected


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


def _second_moment_over_mass(inertia_over_mass: np.ndarray) -> np.ndarray:
    inertia = np.asarray(inertia_over_mass, dtype=float)
    if inertia.shape != (3, 3) or np.any(~np.isfinite(inertia)):
        raise PostprocessNumericalError(
            "inertia_over_mass must be a finite 3x3 matrix"
        )
    inertia = 0.5 * (inertia + inertia.T)
    result = 0.5 * float(np.trace(inertia)) * np.eye(3) - inertia
    return 0.5 * (result + result.T)


def _symmetric_coordinates(matrix: np.ndarray) -> np.ndarray:
    selected = np.asarray(matrix, dtype=float)
    if selected.shape != (3, 3) or np.any(~np.isfinite(selected)):
        raise PostprocessNumericalError(
            "symmetric-coordinate input must be a finite 3x3 matrix"
        )
    selected = 0.5 * (selected + selected.T)
    return np.asarray(
        [float(np.sum(selected * basis)) for basis in SYMMETRIC_BASIS],
        dtype=float,
    )


def _symmetric_matrix(coordinate: Sequence[float]) -> np.ndarray:
    selected = np.asarray(coordinate, dtype=float)
    if selected.shape != (6,) or np.any(~np.isfinite(selected)):
        raise PostprocessInputError(
            "symmetric matrix coordinate must be finite and 6-D"
        )
    return sum(
        (
            float(coefficient) * basis
            for coefficient, basis in zip(selected, SYMMETRIC_BASIS)
        ),
        start=np.zeros((3, 3), dtype=float),
    )


def _symmetric_spd_power(matrix: np.ndarray, exponent: float) -> np.ndarray:
    selected = np.asarray(matrix, dtype=float)
    selected = 0.5 * (selected + selected.T)
    eigenvalues, eigenvectors = np.linalg.eigh(selected)
    if np.any(eigenvalues <= 0.0):
        raise PostprocessNumericalError(
            "SPD matrix function received a non-positive eigenvalue"
        )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        powered = eigenvalues ** float(exponent)
    result = eigenvectors @ np.diag(powered) @ eigenvectors.T
    return 0.5 * (result + result.T)


def _symmetric_spd_log(matrix: np.ndarray) -> np.ndarray:
    selected = np.asarray(matrix, dtype=float)
    selected = 0.5 * (selected + selected.T)
    eigenvalues, eigenvectors = np.linalg.eigh(selected)
    if np.any(eigenvalues <= 0.0):
        raise PostprocessNumericalError(
            "matrix logarithm received a non-positive SPD eigenvalue"
        )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        logged = np.log(eigenvalues)
    result = eigenvectors @ np.diag(logged) @ eigenvectors.T
    return 0.5 * (result + result.T)


def _symmetric_exp(matrix: np.ndarray) -> np.ndarray:
    selected = np.asarray(matrix, dtype=float)
    selected = 0.5 * (selected + selected.T)
    eigenvalues, eigenvectors = np.linalg.eigh(selected)
    with np.errstate(over="raise", invalid="raise"):
        exponentiated = np.exp(eigenvalues)
    result = eigenvectors @ np.diag(exponentiated) @ eigenvectors.T
    return 0.5 * (result + result.T)


def _inertia_over_mass_from_second_moment(
    second_moment: np.ndarray,
) -> np.ndarray:
    """Compute J/m = tr(K)I-K without subtractive cancellation."""

    selected = np.asarray(second_moment, dtype=float)
    if selected.shape != (3, 3) or np.any(~np.isfinite(selected)):
        raise PostprocessNumericalError(
            "scale-free second moment must be a finite 3x3 matrix"
        )
    selected = 0.5 * (selected + selected.T)
    eigenvalues, eigenvectors = np.linalg.eigh(selected)
    if np.any(~np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0.0):
        raise PostprocessNumericalError(
            "scale-free second moment lost numerical positive definiteness"
        )

    # In K's eigenbasis, each inertia eigenvalue is the sum of the other
    # two positive K eigenvalues.  Direct pairwise sums avoid cancellation
    # in trace(K) - lambda_max(K) for highly anisotropic finite samples.
    inertia_eigenvalues = np.asarray(
        (
            eigenvalues[1] + eigenvalues[2],
            eigenvalues[0] + eigenvalues[2],
            eigenvalues[0] + eigenvalues[1],
        ),
        dtype=float,
    )
    if (
        np.any(~np.isfinite(inertia_eigenvalues))
        or np.any(inertia_eigenvalues <= 0.0)
    ):
        raise PostprocessNumericalError(
            "scale-free inertia is not representable as a positive finite matrix"
        )
    inertia = eigenvectors @ np.diag(inertia_eigenvalues) @ eigenvectors.T
    inertia = 0.5 * (inertia + inertia.T)
    represented_eigenvalues = np.linalg.eigvalsh(inertia)
    if (
        np.any(~np.isfinite(represented_eigenvalues))
        or np.any(represented_eigenvalues <= 0.0)
    ):
        raise PostprocessNumericalError(
            "scale-free inertia lost numerical positive definiteness"
        )
    return inertia


@dataclass(frozen=True)
class CenteredScaleFreeSpdChart:
    """Estimate-centered chart on SPD(3) x R^3 x R_+^4.

    The SPD variable is the scale-free second moment K = Sigma / m, where
    J/m = tr(K) I - K.  The fitted plant is the chart origin.
    """

    center_plant: ScaleFreePlant
    second_moment_over_mass: np.ndarray
    second_moment_sqrt: np.ndarray
    second_moment_inverse_sqrt: np.ndarray

    @classmethod
    def from_plant(
        cls, center_plant: ScaleFreePlant
    ) -> "CenteredScaleFreeSpdChart":
        second_moment = _second_moment_over_mass(
            center_plant.inertia_over_mass
        )
        square_root = _symmetric_spd_power(second_moment, 0.5)
        inverse_square_root = _symmetric_spd_power(second_moment, -0.5)
        return cls(
            center_plant=center_plant,
            second_moment_over_mass=second_moment,
            second_moment_sqrt=square_root,
            second_moment_inverse_sqrt=inverse_square_root,
        )

    def encode(self, plant: ScaleFreePlant) -> np.ndarray:
        second_moment = _second_moment_over_mass(plant.inertia_over_mass)
        normalized = (
            self.second_moment_inverse_sqrt
            @ second_moment
            @ self.second_moment_inverse_sqrt
        )
        log_second_moment = _symmetric_spd_log(normalized)
        force = np.asarray(plant.force_effectiveness_over_mass, dtype=float)
        center_force = np.asarray(
            self.center_plant.force_effectiveness_over_mass, dtype=float
        )
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            log_force_ratio = np.log(force / center_force)
        return np.concatenate(
            (
                _symmetric_coordinates(log_second_moment),
                np.asarray(plant.cog_position_body)
                - np.asarray(self.center_plant.cog_position_body),
                log_force_ratio,
            )
        )

    def decode(self, coordinate: Sequence[float]) -> ScaleFreePlant:
        selected = np.asarray(coordinate, dtype=float)
        if selected.shape != (13,) or np.any(~np.isfinite(selected)):
            raise PostprocessInputError(
                "centered scale-free SPD coordinate must be finite and 13-D"
            )
        log_second_moment = _symmetric_matrix(selected[:6])
        normalized = _symmetric_exp(log_second_moment)
        second_moment = (
            self.second_moment_sqrt
            @ normalized
            @ self.second_moment_sqrt
        )
        second_moment = 0.5 * (second_moment + second_moment.T)
        inertia_over_mass = _inertia_over_mass_from_second_moment(
            second_moment
        )
        with np.errstate(
            over="raise", invalid="raise", under="raise"
        ):
            force = (
                np.asarray(
                    self.center_plant.force_effectiveness_over_mass,
                    dtype=float,
                )
                * np.exp(selected[9:13])
            )
        return ScaleFreePlant(
            inertia_over_mass=inertia_over_mass,
            cog_position_body=(
                np.asarray(self.center_plant.cog_position_body, dtype=float)
                + selected[6:9]
            ),
            force_effectiveness_over_mass=force,
            rotor_lag_seconds=self.center_plant.rotor_lag_seconds,
        )


def centered_scale_free_spd_pushforward_jacobian(
    *,
    estimator_chart: SiParameterChart,
    center_coordinate: np.ndarray,
    quotient_basis: np.ndarray,
    centered_chart: CenteredScaleFreeSpdChart,
) -> np.ndarray:
    """Return dy/dz at the fitted point for the centered 13-D chart."""

    parameters, jacobian = estimator_chart.decode_with_jacobian(
        center_coordinate
    )
    mass = float(parameters.mass)
    inertia_over_mass = np.asarray(parameters.inertia, dtype=float) / mass
    force_over_mass = (
        np.asarray(parameters.force_effectiveness, dtype=float) / mass
    )
    result = np.zeros((13, 13), dtype=float)
    for column in range(13):
        direction = np.asarray(quotient_basis[:, column], dtype=float)
        dmass = float(np.dot(np.asarray(jacobian.mass), direction))
        dinertia = np.tensordot(
            np.asarray(jacobian.inertia), direction, axes=(2, 0)
        )
        dcog = np.asarray(jacobian.cog_offset) @ direction
        dforce = np.asarray(jacobian.force_effectiveness) @ direction
        dinertia_over_mass = (
            dinertia / mass
            - inertia_over_mass * (dmass / mass)
        )
        dsecond_moment = (
            0.5 * float(np.trace(dinertia_over_mass)) * np.eye(3)
            - dinertia_over_mass
        )
        dlog_second_moment = (
            centered_chart.second_moment_inverse_sqrt
            @ dsecond_moment
            @ centered_chart.second_moment_inverse_sqrt
        )
        dforce_over_mass = (
            dforce / mass
            - force_over_mass * (dmass / mass)
        )
        result[:6, column] = _symmetric_coordinates(
            dlog_second_moment
        )
        result[6:9, column] = dcog
        result[9:13, column] = dforce_over_mass / force_over_mass
    if np.any(~np.isfinite(result)):
        raise PostprocessNumericalError(
            "centered scale-free SPD push-forward Jacobian is non-finite"
        )
    return result


@dataclass(frozen=True)
class SamplingCoordinates:
    mode: str
    covariance: np.ndarray
    covariance_source: str
    coordinate_labels: tuple[str, ...]
    center_plant: ScaleFreePlant
    estimator_chart: SiParameterChart
    estimator_center_coordinate: np.ndarray
    quotient_basis: np.ndarray
    centered_spd_chart: Optional[CenteredScaleFreeSpdChart]
    pushforward_jacobian: np.ndarray

    def decode(self, delta: Sequence[float]) -> ScaleFreePlant:
        selected = np.asarray(delta, dtype=float)
        if selected.shape != (13,) or np.any(~np.isfinite(selected)):
            raise PostprocessInputError(
                "sampling coordinate must be finite and 13-D"
            )
        if self.mode == "estimator_quotient":
            try:
                return plant_from_coordinate(
                    self.estimator_chart,
                    self.estimator_center_coordinate
                    + self.quotient_basis @ selected,
                    self.center_plant.rotor_lag_seconds,
                )
            except ValueError as error:
                message = str(error)
                numerical_messages = (
                    "mass must be finite and positive",
                    "inertia must be symmetric positive definite",
                    "force_effectiveness must contain 4 finite values",
                    "force_effectiveness must be positive",
                    "inertia_over_mass must be positive definite",
                    "force_effectiveness_over_mass must be positive",
                )
                if message in numerical_messages:
                    raise PostprocessNumericalError(
                        "estimator-chart sample lost floating-point physical "
                        "validity: {}".format(message)
                    ) from error
                raise
        if self.mode == "centered_scale_free_spd":
            assert self.centered_spd_chart is not None
            return self.centered_spd_chart.decode(selected)
        raise RuntimeError("unknown sampling coordinate mode")


def prepare_sampling_coordinates(
    *,
    result: EstimatorResult,
    artifacts: SensitivityArtifacts,
    model: VehicleModel,
    coordinate_mode: str,
) -> SamplingCoordinates:
    mode = str(coordinate_mode)
    if mode not in COORDINATE_MODES:
        raise PostprocessInputError(
            "coordinate_mode must be one of {}".format(
                ", ".join(COORDINATE_MODES)
            )
        )
    estimator_chart = SiParameterChart(model.parameters)
    center_plant = validate_center_matches_result(
        result, estimator_chart, artifacts.physical_coordinate
    )
    if mode == "estimator_quotient":
        return SamplingCoordinates(
            mode=mode,
            covariance=np.asarray(artifacts.covariance, dtype=float),
            covariance_source=artifacts.covariance_source,
            coordinate_labels=tuple(
                "estimator_quotient_{:02d}".format(index)
                for index in range(13)
            ),
            center_plant=center_plant,
            estimator_chart=estimator_chart,
            estimator_center_coordinate=np.asarray(
                artifacts.physical_coordinate, dtype=float
            ),
            quotient_basis=np.asarray(
                artifacts.quotient_basis, dtype=float
            ),
            centered_spd_chart=None,
            pushforward_jacobian=np.eye(13),
        )
    centered_chart = CenteredScaleFreeSpdChart.from_plant(center_plant)
    pushforward = centered_scale_free_spd_pushforward_jacobian(
        estimator_chart=estimator_chart,
        center_coordinate=np.asarray(
            artifacts.physical_coordinate, dtype=float
        ),
        quotient_basis=np.asarray(artifacts.quotient_basis, dtype=float),
        centered_chart=centered_chart,
    )
    covariance = pushforward @ artifacts.covariance @ pushforward.T
    covariance = 0.5 * (covariance + covariance.T)
    return SamplingCoordinates(
        mode=mode,
        covariance=covariance,
        covariance_source=(
            "{} pushed forward to centered_scale_free_spd".format(
                artifacts.covariance_source
            )
        ),
        coordinate_labels=CENTERED_SCALE_FREE_SPD_LABELS,
        center_plant=center_plant,
        estimator_chart=estimator_chart,
        estimator_center_coordinate=np.asarray(
            artifacts.physical_coordinate, dtype=float
        ),
        quotient_basis=np.asarray(artifacts.quotient_basis, dtype=float),
        centered_spd_chart=centered_chart,
        pushforward_jacobian=pushforward,
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



def sensitivity_group_scale(
    matrix: np.ndarray, axes: Sequence[int]
) -> float:
    """Least-squares group scale with no plausibility rejection.

    A negative finite scale is retained as a diagnostic result.  The only
    rejected denominator is exact machine zero, where the requested quotient
    is mathematically undefined in the current floating-point calculation.
    """

    selected = np.asarray(matrix, dtype=float)
    indices = tuple(int(value) for value in axes)
    if (
        selected.shape != (6, 6)
        or np.any(~np.isfinite(selected))
        or not indices
        or any(value < 0 or value >= 6 for value in indices)
        or len(set(indices)) != len(indices)
    ):
        raise PostprocessNumericalError(
            "sensitivity group-scale inputs are invalid"
        )
    denominator = float(np.sum(selected[:, indices] ** 2))
    numerator = float(sum(selected[index, index] for index in indices))
    if denominator == 0.0:
        raise PostprocessNumericalError(
            "sensitivity gain-group denominator is exactly zero"
        )
    scale = numerator / denominator
    if not np.isfinite(scale):
        raise PostprocessNumericalError(
            "sensitivity gain-group scale is non-finite"
        )
    return float(scale)


def evaluate_static_scales(
    plant: ScaleFreePlant,
    model: VehicleModel,
    *,
    nominal_pseudoinverse: np.ndarray,
    characteristic_length_m: float,
) -> StaticScaleEvaluation:
    real = build_real_scale_free_allocation(plant, model)
    effectiveness = real.matrix @ nominal_pseudoinverse
    normalized = dimensionless_effectiveness(
        effectiveness, characteristic_length_m
    )
    scales = {
        group: sensitivity_group_scale(
            normalized, GAIN_GROUP_AXES[group]
        )
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
        allocation_source_threshold_rank=real.source_threshold_rank,
        allocation_condition_number=real.condition_number,
    )



def _sample_record(
    *,
    sample_name: str,
    delta_coordinate: np.ndarray,
    center_vector: np.ndarray,
    sampling: SamplingCoordinates,
    model: VehicleModel,
    nominal_pseudoinverse: np.ndarray,
    characteristic_length_m: float,
) -> Mapping[str, Any]:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            plant = sampling.decode(delta_coordinate)
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
            "sampling_coordinate": np.asarray(
                delta_coordinate, dtype=float
            ).tolist(),
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
            "A_real_source_threshold_rank": (
                evaluation.allocation_source_threshold_rank
            ),
            "A_real_condition_number": (
                _json_condition_number(
                    evaluation.allocation_condition_number
                )
            ),
        }
    except (
        PostprocessNumericalError,
        OverflowError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        return {
            "name": sample_name,
            "valid": False,
            "sampling_coordinate": np.asarray(
                delta_coordinate, dtype=float
            ).tolist(),
            "exception_type": type(error).__name__,
            "message": str(error),
        }



def analyze_eigen_directions(
    *,
    result: EstimatorResult,
    artifacts: SensitivityArtifacts,
    model: VehicleModel,
    sigma_multiple: float,
    coordinate_mode: str = DEFAULT_COORDINATE_MODE,
    derivative_sigma_fraction: float = DEFAULT_DERIVATIVE_SIGMA_FRACTION,
    characteristic_length_override: Optional[float] = None,
) -> Mapping[str, Any]:
    multiplier = float(sigma_multiple)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise PostprocessInputError(
            "sigma_multiple must be finite and positive"
        )
    derivative_fraction = float(derivative_sigma_fraction)
    if not np.isfinite(derivative_fraction) or derivative_fraction <= 0.0:
        raise PostprocessInputError(
            "derivative_sigma_fraction must be finite and positive"
        )
    sampling = prepare_sampling_coordinates(
        result=result,
        artifacts=artifacts,
        model=model,
        coordinate_mode=coordinate_mode,
    )
    center_plant = sampling.center_plant
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
        sampling.covariance
    )
    directions = []
    finite_samples = [
        {
            "name": "center",
            "valid": True,
            "sampling_coordinate": np.zeros(13).tolist(),
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
            "A_real_source_threshold_rank": (
                center.allocation_source_threshold_rank
            ),
            "A_real_condition_number": _json_condition_number(
                center.allocation_condition_number
            ),
        }
    ]
    local_effects = {
        group: np.full(13, np.nan, dtype=float)
        for group in PID_GROUPS
    }
    finite_effects = {
        group: np.full(13, np.nan, dtype=float)
        for group in PID_GROUPS
    }
    for index in range(13):
        eigenvalue = float(eigenvalues[index])
        sigma = float(np.sqrt(eigenvalue))
        sampling_direction = eigenvectors[:, index]

        finite_offset = multiplier * sigma * sampling_direction
        minus = _sample_record(
            sample_name="direction_{:02d}_minus".format(index),
            delta_coordinate=-finite_offset,
            center_vector=center_vector,
            sampling=sampling,
            model=model,
            nominal_pseudoinverse=nominal_pseudoinverse,
            characteristic_length_m=length,
        )
        plus = _sample_record(
            sample_name="direction_{:02d}_plus".format(index),
            delta_coordinate=finite_offset,
            center_vector=center_vector,
            sampling=sampling,
            model=model,
            nominal_pseudoinverse=nominal_pseudoinverse,
            characteristic_length_m=length,
        )
        finite_samples.extend((minus, plus))

        derivative_offset = (
            derivative_fraction * sigma * sampling_direction
        )
        derivative_minus = _sample_record(
            sample_name="direction_{:02d}_derivative_minus".format(index),
            delta_coordinate=-derivative_offset,
            center_vector=center_vector,
            sampling=sampling,
            model=model,
            nominal_pseudoinverse=nominal_pseudoinverse,
            characteristic_length_m=length,
        )
        derivative_plus = _sample_record(
            sample_name="direction_{:02d}_derivative_plus".format(index),
            delta_coordinate=derivative_offset,
            center_vector=center_vector,
            sampling=sampling,
            model=model,
            nominal_pseudoinverse=nominal_pseudoinverse,
            characteristic_length_m=length,
        )
        direction_report: dict[str, Any] = {
            "direction_index": index,
            "covariance_eigenvalue": eigenvalue,
            "one_sigma_sampling_coordinate": sigma,
            "sampling_eigenvector": sampling_direction.tolist(),
            "finite_minus": minus,
            "finite_plus": plus,
            "finite_pair_valid": bool(minus["valid"] and plus["valid"]),
            "derivative_sigma_fraction": derivative_fraction,
            "derivative_minus": derivative_minus,
            "derivative_plus": derivative_plus,
            "derivative_pair_valid": bool(
                derivative_minus["valid"] and derivative_plus["valid"]
            ),
        }
        if sampling.mode == "estimator_quotient":
            direction_report["one_sigma_quotient_coordinate"] = sigma
            direction_report["quotient_eigenvector"] = (
                sampling_direction.tolist()
            )
            direction_report["physical_chart_direction"] = (
                sampling.quotient_basis @ sampling_direction
            ).tolist()

        if derivative_minus["valid"] and derivative_plus["valid"]:
            physical_one_sigma_effect = (
                np.asarray(
                    derivative_plus[
                        "scale_free_delta_from_center"
                    ],
                    dtype=float,
                )
                - np.asarray(
                    derivative_minus[
                        "scale_free_delta_from_center"
                    ],
                    dtype=float,
                )
            ) / (2.0 * derivative_fraction)
            direction_report["scale_free_local_one_sigma_effect"] = (
                physical_one_sigma_effect.tolist()
            )
            local_scale_effects = {}
            for group in PID_GROUPS:
                effect = (
                    float(derivative_plus["scales"][group])
                    - float(derivative_minus["scales"][group])
                ) / (2.0 * derivative_fraction)
                local_effects[group][index] = effect
                local_scale_effects[group] = effect
            direction_report["local_one_sigma_scale_effect"] = (
                local_scale_effects
            )

        if minus["valid"] and plus["valid"]:
            finite_scale_effects = {}
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
                finite_effects[group][index] = effect
                finite_scale_effects[group] = effect
                nonlinearities[group] = curvature
            direction_report[
                "finite_secant_one_sigma_scale_effect"
            ] = finite_scale_effects
            direction_report[
                "scale_second_difference_per_sigma2"
            ] = nonlinearities
        directions.append(direction_report)

    group_summary = {}
    valid_finite_samples = [
        sample for sample in finite_samples if sample["valid"]
    ]
    derivative_valid_direction_count = sum(
        bool(direction["derivative_pair_valid"])
        for direction in directions
    )
    finite_valid_direction_count = sum(
        bool(direction["finite_pair_valid"])
        for direction in directions
    )
    for group in PID_GROUPS:
        center_scale = float(center.scales[group])
        local = local_effects[group]
        finite = finite_effects[group]
        local_complete = bool(np.all(np.isfinite(local)))
        finite_complete = bool(np.all(np.isfinite(finite)))
        sigma_linear = (
            float(np.sqrt(np.sum(local**2)))
            if local_complete
            else None
        )
        sigma_finite = (
            float(np.sqrt(np.sum(finite**2)))
            if finite_complete
            else None
        )
        values = np.asarray(
            [
                sample["scales"][group]
                for sample in valid_finite_samples
            ],
            dtype=float,
        )
        contribution = np.where(np.isfinite(local), local**2, 0.0)
        contribution_sum = float(np.sum(contribution))
        order = np.argsort(contribution)[::-1]
        top = []
        for direction_index in order[:5]:
            top.append(
                {
                    "direction_index": int(direction_index),
                    "one_sigma_scale_effect": (
                        float(local[direction_index])
                        if np.isfinite(local[direction_index])
                        else None
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
                if sigma_linear is not None and center_scale != 0.0
                else None
            ),
            "linearized_complete": local_complete,
            "finite_secant_one_sigma": sigma_finite,
            "finite_to_local_sigma_ratio": (
                sigma_finite / sigma_linear
                if sigma_finite is not None
                and sigma_linear is not None
                and sigma_linear != 0.0
                else None
            ),
            "finite_secant_complete": finite_complete,
            "sigma_point_min": (
                float(np.min(values)) if values.size else None
            ),
            "sigma_point_max": (
                float(np.max(values)) if values.size else None
            ),
            "sigma_point_envelope_complete": (
                len(valid_finite_samples) == len(finite_samples)
            ),
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
    transform_singular = np.linalg.svd(
        sampling.pushforward_jacobian, compute_uv=False
    )
    transform_condition = (
        float(transform_singular[0] / transform_singular[-1])
        if transform_singular[-1] > 0.0
        else float("inf")
    )
    return {
        "coordinate_mode": sampling.mode,
        "coordinate_labels": list(sampling.coordinate_labels),
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
            "A_real_condition_number": _json_condition_number(
                center.allocation_condition_number
            ),
            "A_real_source_threshold_rank": (
                center.allocation_source_threshold_rank
            ),
        },
        "coordinate_transform": {
            "source": "estimator_quotient",
            "target": sampling.mode,
            "pushforward_jacobian": (
                sampling.pushforward_jacobian.tolist()
            ),
            "singular_values": transform_singular.tolist(),
            "condition_number": _json_condition_number(
                transform_condition
            ),
        },
        "covariance": {
            "mode": artifacts.covariance_mode,
            "source": sampling.covariance_source,
            "eigenvalues_descending": eigenvalues.tolist(),
            "numerical_rank": int(retained.size),
            "retained_condition_number": covariance_condition,
            "psd_negative_tolerance": psd_tolerance,
            "interpretation": (
                "local sensitivity covariance in the selected coordinate "
                "chart; not asserted to be a calibrated posterior"
            ),
        },
        "eigen_sampling": {
            "sigma_multiple": multiplier,
            "expected_sample_count": 27,
            "valid_sample_count": len(valid_finite_samples),
            "invalid_sample_count": (
                len(finite_samples) - len(valid_finite_samples)
            ),
            "valid_direction_pair_count": finite_valid_direction_count,
            "derivative_sigma_fraction": derivative_fraction,
            "derivative_valid_direction_count": (
                derivative_valid_direction_count
            ),
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
    coordinate_mode: str = DEFAULT_COORDINATE_MODE,
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
    sampling = prepare_sampling_coordinates(
        result=result,
        artifacts=artifacts,
        model=model,
        coordinate_mode=coordinate_mode,
    )
    eigenvalues, eigenvectors, _tolerance = _psd_eigendecomposition(
        sampling.covariance
    )
    factor = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    center_vector = scale_free_vector(sampling.center_plant)
    nominal = build_nominal_controller_allocation(model)
    nominal_pseudoinverse = source_compatible_pseudoinverse(
        nominal.matrix
    )
    rng = np.random.default_rng(int(seed))
    values = {group: [] for group in PID_GROUPS}
    invalid_reasons: list[Mapping[str, str]] = []
    for sample_index in range(count):
        delta = factor @ rng.standard_normal(13)
        sample = _sample_record(
            sample_name="monte_carlo_{:06d}".format(sample_index),
            delta_coordinate=delta,
            center_vector=center_vector,
            sampling=sampling,
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
        "coordinate_mode": sampling.mode,
        "sampling_model": (
            "Gaussian draws in the selected local covariance expressed in "
            "{} coordinates; used only as a nonlinear sensitivity stress "
            "test".format(sampling.mode)
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
    coordinate_mode = str(eigen_analysis["coordinate_mode"])
    if coordinate_mode == "estimator_quotient":
        sampling_space = (
            "13-D common-scale quotient of the estimator's 14-D SI chart"
        )
        physical_validity = (
            "samples are decoded through SiParameterChart, preserving "
            "positive mass, positive force effectiveness, and the "
            "inertia second-moment parameterization"
        )
    else:
        sampling_space = (
            "13-D estimate-centered scale-free chart on "
            "SPD(3) x R^3 x R_+^4"
        )
        physical_validity = (
            "the scale-free second moment is decoded by an SPD matrix "
            "exponential, force-over-mass by exponentials, and CoG "
            "additively; no common-scale gauge remains"
        )
    return {
        "schema": SENSITIVITY_SCHEMA,
        "source_commit": str(revision),
        "method": "coordinate_chart_eigen_direction_sensitivity",
        "input": {
            "estimator_result_json": str(result.source_path),
            "estimator_source_commit": result.source_commit,
            "estimator_case_name": result.case_name,
            "arrays_npz": str(artifacts.source_path),
            "vehicle_model_json": str(model.source_path),
            "covariance_mode": artifacts.covariance_mode,
            "covariance_source": artifacts.covariance_source,
            "coordinate_source": artifacts.coordinate_source,
            "coordinate_mode": coordinate_mode,
        },
        "coordinate_contract": {
            "sampling_space": sampling_space,
            "sampling_coordinate_labels": list(
                eigen_analysis["coordinate_labels"]
            ),
            "physical_chart_labels": list(PHYSICAL_CHART_LABELS),
            "scale_free_output_labels": list(SCALE_FREE_LABELS),
            "scale_free_output_units": list(SCALE_FREE_UNITS),
            "common_scale_direction": (
                np.asarray(COMMON_SCALE_DIRECTION, dtype=float).tolist()
            ),
            "physical_validity": physical_validity,
            "covariance_pushforward": (
                eigen_analysis["coordinate_transform"]
            ),
        },
        "interpretation": {
            "role": "local_sensitivity_analysis",
            "posterior_claim": False,
            "note": (
                "The covariance is used to define local perturbation scales. "
                "The infinitesimal linear sensitivity and the finite "
                "sigma-envelope are reported separately.  In particular, "
                "conservative_fusion is a conservative sensitivity "
                "distribution rather than a calibrated generative posterior."
            ),
            "rotor_lag_treatment": (
                "held fixed at the fitted point because the v1 static H "
                "does not model delay"
            ),
            "rank_treatment": (
                "sampled A_real source-threshold rank and condition number "
                "are diagnostics only; a finite calculation is not rejected "
                "for rank loss or large condition number"
            ),
        },
        "center_and_eigen_sensitivity": eigen_analysis,
        "monte_carlo": monte_carlo,
    }



def _format_optional(value: Any, fmt: str = ".8g") -> str:
    if value is None:
        return "incomplete"
    return format(float(value), fmt)


def render_markdown(report: Mapping[str, Any]) -> str:
    analysis = report["center_and_eigen_sensitivity"]
    groups = analysis["group_summary"]
    mc = report["monte_carlo"]
    sampling = analysis["eigen_sampling"]
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
        "- coordinate mode: `{}`".format(
            report["input"]["coordinate_mode"]
        ),
        "- center coordinate source: `{}`".format(
            report["input"]["coordinate_source"]
        ),
        "- characteristic length: `{:.9g} m`".format(
            analysis["characteristic_length_m"]
        ),
        "- finite sigma multiple: `{:.9g}`".format(
            sampling["sigma_multiple"]
        ),
        "- local derivative sigma fraction: `{:.9g}`".format(
            sampling["derivative_sigma_fraction"]
        ),
        "- valid finite eigen samples: `{}/{}`".format(
            sampling["valid_sample_count"],
            sampling["expected_sample_count"],
        ),
        "- valid local-derivative directions: `{}/13`".format(
            sampling["derivative_valid_direction_count"]
        ),
        "",
        "## Gain-scale sensitivity",
        "",
        (
            "| group | center | infinitesimal 1-sigma | relative | "
            "finite secant 1-sigma | finite/local | eigen-point min | "
            "eigen-point max |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in PID_GROUPS:
        item = groups[group]
        relative = item["relative_linearized_one_sigma"]
        ratio = item["finite_to_local_sigma_ratio"]
        lines.append(
            "| {} | {:.8g} | {} | {} | {} | {} | {} | {} |".format(
                group,
                item["center_scale"],
                _format_optional(item["linearized_one_sigma"]),
                (
                    "incomplete"
                    if relative is None
                    else "{:.3%}".format(relative)
                ),
                _format_optional(item["finite_secant_one_sigma"]),
                _format_optional(ratio),
                _format_optional(item["sigma_point_min"]),
                _format_optional(item["sigma_point_max"]),
            )
        )
    lines.extend(
        (
            "",
            "The infinitesimal value uses a dedicated small centered finite",
            "difference and is independent of the requested finite sigma",
            "excursion up to numerical differentiation error. The finite",
            "secant value uses the requested +/- k-sigma points. A",
            "`finite/local` ratio far from one therefore measures chart/output",
            "nonlinearity over that finite excursion rather than changing the",
            "definition of the local covariance.",
            "",
            "Sampled `A_real` rank loss or a large condition number is retained",
            "as a diagnostic. A sample is only marked invalid after the",
            "floating-point calculation itself becomes non-finite or otherwise",
            "mathematically undefined.",
            "",
            "## Dominant local covariance directions",
            "",
        )
    )
    for group in PID_GROUPS:
        lines.extend(
            (
                "### {}".format(group),
                "",
                "| direction | local 1-sigma scale effect | variance contribution |",
                "|---:|---:|---:|",
            )
        )
        for item in groups[group]["top_eigen_directions"]:
            effect = item["one_sigma_scale_effect"]
            lines.append(
                "| {} | {} | {:.2%} |".format(
                    item["direction_index"],
                    (
                        "incomplete"
                        if effect is None
                        else "{:+.8g}".format(effect)
                    ),
                    item["variance_contribution_fraction"],
                )
            )
        lines.append("")
    if mc.get("enabled", False):
        lines.extend(
            (
                "## Optional Monte Carlo stress test",
                "",
                "- valid samples: `{}/{}`".format(
                    mc["valid_sample_count"],
                    mc["sample_count_requested"],
                ),
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
                "covariance model in the selected coordinate chart. Do not",
                "relabel these quantiles as posterior credible intervals without",
                "an independent probabilistic calibration argument.",
                "",
            )
        )
    else:
        lines.extend(
            (
                "## Monte Carlo",
                "",
                "Disabled for this run. The deterministic eigen-direction",
                "analysis remains available.",
                "",
            )
        )
    return "\n".join(lines) + "\n"



def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Propagate the estimator's local common-scale quotient "
            "covariance through the static Gimbalrotor PID correction in "
            "either the estimator quotient chart or an estimate-centered "
            "scale-free SPD chart."
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
        "--coordinate-mode",
        choices=COORDINATE_MODES,
        default=DEFAULT_COORDINATE_MODE,
    )
    parser.add_argument(
        "--sigma-multiple",
        type=float,
        default=1.0,
        help="Evaluate the finite envelope at +/- this many sigma.",
    )
    parser.add_argument(
        "--derivative-sigma-fraction",
        type=float,
        default=DEFAULT_DERIVATIVE_SIGMA_FRACTION,
        help=(
            "Small fraction of one covariance sigma used only for the "
            "infinitesimal centered derivative."
        ),
    )
    parser.add_argument(
        "--monte-carlo-samples",
        type=int,
        default=0,
        help=(
            "Optional Gaussian stress-test sample count in the selected "
            "coordinate chart. Zero disables it."
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
        coordinate_mode=arguments.coordinate_mode,
        derivative_sigma_fraction=arguments.derivative_sigma_fraction,
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
        coordinate_mode=arguments.coordinate_mode,
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
            "coordinate_mode": report["input"]["coordinate_mode"],
            "valid_eigen_sample_count": report[
                "center_and_eigen_sensitivity"
            ]["eigen_sampling"]["valid_sample_count"],
            "invalid_eigen_sample_count": report[
                "center_and_eigen_sensitivity"
            ]["eigen_sampling"]["invalid_sample_count"],
            "valid_local_derivative_direction_count": report[
                "center_and_eigen_sensitivity"
            ]["eigen_sampling"]["derivative_valid_direction_count"],
            "monte_carlo_enabled": bool(
                report["monte_carlo"].get("enabled", False)
            ),
            "monte_carlo_valid_sample_count": (
                report["monte_carlo"].get("valid_sample_count")
            ),
            "monte_carlo_invalid_sample_count": (
                report["monte_carlo"].get("invalid_sample_count")
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
