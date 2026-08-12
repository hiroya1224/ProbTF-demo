#!/usr/bin/env python3
"""Flat experimental dimensionless SG rigid-body identification.

This script is intentionally self-contained at the experiment layer.  It reuses
only the established raw-pose geometric Savitzky--Golay front end, recorded-data
loaders, actuator primitives, and rigid-body wrench primitives from the project.

The deterministic fit is prior-free.  The external Gaussian physical prior is
used only by the optional local posterior calculation.

Fixed reference nondimensionalization
--------------------------------------
For the vehicle-model reference (M_*, J_*), define

    L_* = sqrt(trace(J_*) / (3 M_*))
    T_* = sqrt(L_* / g)
    F_* = M_* L_* / T_*^2
    N_* = M_* L_*^2 / T_*^2

The optimized rigid-body residual is written entirely with these fixed scales.

Parameter chart
---------------
The 14 physical coordinates are all dimensionless:

    0       log mass scale
    1:7     symmetric log-Euclidean chart of the dimensionless
            mass-distribution second moment
    7:10    CoG displacement divided by L_*
    10:14   four independent log rotor-force-effectiveness scales

The four force scales remain independent.  The common mass/inertia/thrust scale
ridge is not normalized away.  The log-Euclidean second-moment chart makes that
exact ridge a globally straight coordinate direction, so a dense SVD-based
least-squares solve can leave it in the nullspace without runaway along a
curved Cholesky orbit.

Code is deliberately flat and experimental.  It is not a refactoring target.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
import shutil
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/grape-minimal-matplotlib")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.linalg import expm, expm_frechet  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
import deterministic_spline_dynamics_estimator as spline_reports  # noqa: E402

import savgol_trajectory as sg  # noqa: E402
from smooth_command import QuinticSmoothZoh  # noqa: E402
from legacies import deterministic_estimator as baseline  # noqa: E402
from legacies import (  # noqa: E402
    deterministic_multi_bag_multiple_shooting_estimator as multi,
)
from legacies import deterministic_multiple_shooting_estimator as strict  # noqa: E402
from legacies import (  # noqa: E402
    deterministic_smooth_lag_multiple_shooting_estimator as smooth,
)
from grape_param_estim.dynamics import (  # noqa: E402
    FullSixDofPlant,
    advance_actuators,
)
from grape_param_estim.geometry import (  # noqa: E402
    matrix_to_quaternion,
    quaternion_to_matrix,
    skew,
    so3_log,
)
from grape_param_estim.real_rosbag import load_flight_data  # noqa: E402
from grape_param_estim.system import (  # noqa: E402
    GRAVITY,
    ActuatorParameters,
    ActuatorState,
    GrapeGeometry,
    RigidBodyState,
    VehicleParameters,
)


SCHEMA = "grape-param-estim/dimensionless-savgol-experiment/v1"
OUTPUT_SUBDIRECTORY = "dimensionless_savgol_experiment"
PHYSICAL_DIMENSION = 14
GLOBAL_DIMENSION = 16
ROTOR_DELAY_INDEX = 14
GIMBAL_DELAY_INDEX = 15

PHYSICAL_PARAMETER_NAMES = (
    "log_mass_scale",
    "log_second_moment_mode_xx",
    "log_second_moment_mode_yy",
    "log_second_moment_mode_zz",
    "log_second_moment_mode_xy",
    "log_second_moment_mode_xz",
    "log_second_moment_mode_yz",
    "dimensionless_cog_displacement_x",
    "dimensionless_cog_displacement_y",
    "dimensionless_cog_displacement_z",
    "log_force_effectiveness_1",
    "log_force_effectiveness_2",
    "log_force_effectiveness_3",
    "log_force_effectiveness_4",
)

PHYSICAL_VALUE_NAMES = (
    "mass_kg",
    "inertia_xx_kg_m2",
    "inertia_yy_kg_m2",
    "inertia_zz_kg_m2",
    "inertia_xy_kg_m2",
    "inertia_xz_kg_m2",
    "inertia_yz_kg_m2",
    "cog_position_body_x_m",
    "cog_position_body_y_m",
    "cog_position_body_z_m",
    "force_effectiveness_1",
    "force_effectiveness_2",
    "force_effectiveness_3",
    "force_effectiveness_4",
)

INERTIA_COMPONENTS = (
    (0, 0),
    (1, 1),
    (2, 2),
    (0, 1),
    (0, 2),
    (1, 2),
)

SQRT_TWO = math.sqrt(2.0)
SYMMETRIC_BASIS = (
    np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
    np.asarray(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0))),
    np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
    np.asarray(
        (
            (0.0, 1.0 / SQRT_TWO, 0.0),
            (1.0 / SQRT_TWO, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
    ),
    np.asarray(
        (
            (0.0, 0.0, 1.0 / SQRT_TWO),
            (0.0, 0.0, 0.0),
            (1.0 / SQRT_TWO, 0.0, 0.0),
        )
    ),
    np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0 / SQRT_TWO),
            (0.0, 1.0 / SQRT_TWO, 0.0),
        )
    ),
)

COMMON_SCALE_DIRECTION = np.asarray(
    (
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ),
    dtype=float,
)


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_sanitize(value.tolist())
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_sanitize(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _safe_label(value: float) -> str:
    text = "{:.9f}".format(float(value)).rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or np.any(~np.isfinite(result)):
        raise ValueError(
            "{} must contain {} finite values".format(name, size)
        )
    return result


@dataclass(frozen=True)
class VehicleModelInput:
    source_path: Path
    parameters: VehicleParameters
    geometry: GrapeGeometry
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class GaussianPhysicalPrior:
    source_path: Path
    mean: np.ndarray
    std: np.ndarray
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class ReferenceScales:
    mass_kg: float
    length_m: float
    time_s: float
    force_n: float
    torque_nm: float

    @classmethod
    def from_reference(
        cls,
        parameters: VehicleParameters,
    ) -> "ReferenceScales":
        mass = float(parameters.mass)
        length = math.sqrt(
            float(np.trace(parameters.inertia))
            / (3.0 * mass)
        )
        time_scale = math.sqrt(length / float(GRAVITY))
        force = mass * length / time_scale**2
        torque = force * length
        values = np.asarray(
            (mass, length, time_scale, force, torque),
            dtype=float,
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(
                "fixed-reference nondimensionalization scales are invalid"
            )
        return cls(
            mass_kg=mass,
            length_m=length,
            time_s=time_scale,
            force_n=force,
            torque_nm=torque,
        )

    @property
    def acceleration_m_per_s2(self) -> float:
        return self.length_m / self.time_s**2

    @property
    def angular_velocity_rad_per_s(self) -> float:
        return 1.0 / self.time_s

    @property
    def angular_acceleration_rad_per_s2(self) -> float:
        return 1.0 / self.time_s**2

    def payload(self) -> dict[str, float]:
        return {
            "mass_kg": self.mass_kg,
            "length_m": self.length_m,
            "time_s": self.time_s,
            "force_n": self.force_n,
            "torque_nm": self.torque_nm,
            "dimensionless_gravity": (
                float(GRAVITY)
                * self.time_s**2
                / self.length_m
            ),
        }


def _read_json_object(
    path: Path,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    source = path.expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "{} JSON cannot be read: {}".format(label, source)
        ) from error
    if not isinstance(value, dict):
        raise ValueError("{} JSON root must be an object".format(label))
    return source, value


def load_vehicle_model(path: Path) -> VehicleModelInput:
    source, raw = _read_json_object(path, "vehicle model")
    required = (
        "mass_kg",
        "inertia_kg_m2",
        "cog_position_body_m",
        "force_effectiveness",
        "torque_effectiveness",
        "linear_drag",
        "angular_drag",
        "geometry",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(
            "vehicle model is missing: {}".format(", ".join(missing))
        )

    inertia = np.asarray(raw["inertia_kg_m2"], dtype=float)
    if (
        inertia.shape != (3, 3)
        or np.any(~np.isfinite(inertia))
        or not np.allclose(
            inertia,
            inertia.T,
            atol=1.0e-12,
            rtol=0.0,
        )
    ):
        raise ValueError(
            "inertia_kg_m2 must be a finite symmetric 3x3 matrix"
        )

    parameters = VehicleParameters(
        mass=float(raw["mass_kg"]),
        inertia=inertia,
        cog_offset=_finite_vector(
            raw["cog_position_body_m"],
            3,
            "cog_position_body_m",
        ),
        force_effectiveness=_finite_vector(
            raw["force_effectiveness"],
            4,
            "force_effectiveness",
        ),
        torque_effectiveness=_finite_vector(
            raw["torque_effectiveness"],
            4,
            "torque_effectiveness",
        ),
        linear_drag=_finite_vector(
            raw["linear_drag"],
            3,
            "linear_drag",
        ),
        angular_drag=_finite_vector(
            raw["angular_drag"],
            3,
            "angular_drag",
        ),
    )

    geometry_raw = raw["geometry"]
    if not isinstance(geometry_raw, dict):
        raise ValueError("vehicle-model geometry must be an object")
    geometry = GrapeGeometry(
        rotor_origins=np.asarray(
            geometry_raw["rotor_origins_body_m"],
            dtype=float,
        ),
        arm_yaws=_finite_vector(
            geometry_raw["arm_yaws_rad"],
            4,
            "arm_yaws_rad",
        ),
        rotor_directions=_finite_vector(
            geometry_raw["rotor_directions"],
            4,
            "rotor_directions",
        ),
        moment_force_rate=float(
            geometry_raw["moment_force_rate_m"]
        ),
        thrust_offset=float(geometry_raw["thrust_offset_m"]),
    )
    return VehicleModelInput(
        source_path=source,
        parameters=parameters,
        geometry=geometry,
        raw=raw,
    )


def load_parameter_prior(
    path: Optional[Path],
) -> Optional[GaussianPhysicalPrior]:
    if path is None:
        return None
    source, raw = _read_json_object(path, "parameter prior")

    def block(name: str) -> Mapping[str, Any]:
        value = raw.get(name)
        if (
            not isinstance(value, dict)
            or "mean" not in value
            or "std" not in value
        ):
            raise ValueError(
                "{} prior requires mean and std".format(name)
            )
        return value

    mass = block("mass_kg")
    inertia = block("inertia_kg_m2")
    cog = block("cog_position_body_m")
    effectiveness = block("force_effectiveness")

    mass_mean = float(mass["mean"])
    mass_std = float(mass["std"])
    inertia_mean = np.asarray(inertia["mean"], dtype=float)
    inertia_std = float(inertia["std"])
    cog_mean = _finite_vector(
        cog["mean"],
        3,
        "cog_position_body_m.mean",
    )
    cog_std = float(cog["std"])
    effectiveness_mean = _finite_vector(
        effectiveness["mean"],
        4,
        "force_effectiveness.mean",
    )
    effectiveness_std = float(effectiveness["std"])

    if (
        not np.isfinite(mass_mean)
        or mass_mean <= 0.0
        or not np.isfinite(mass_std)
        or mass_std <= 0.0
        or inertia_mean.shape != (3, 3)
        or np.any(~np.isfinite(inertia_mean))
        or not np.allclose(
            inertia_mean,
            inertia_mean.T,
            atol=1.0e-12,
            rtol=0.0,
        )
        or not np.isfinite(inertia_std)
        or inertia_std <= 0.0
        or not np.isfinite(cog_std)
        or cog_std <= 0.0
        or not np.isfinite(effectiveness_std)
        or effectiveness_std <= 0.0
        or np.any(effectiveness_mean <= 0.0)
    ):
        raise ValueError("parameter-prior values are invalid")

    mean = np.concatenate(
        (
            np.asarray((mass_mean,), dtype=float),
            np.asarray(
                [
                    inertia_mean[row, column]
                    for row, column in INERTIA_COMPONENTS
                ],
                dtype=float,
            ),
            cog_mean,
            effectiveness_mean,
        )
    )
    std = np.concatenate(
        (
            np.asarray((mass_std,), dtype=float),
            np.full(6, inertia_std, dtype=float),
            np.full(3, cog_std, dtype=float),
            np.full(4, effectiveness_std, dtype=float),
        )
    )
    return GaussianPhysicalPrior(
        source_path=source,
        mean=mean,
        std=std,
        raw=raw,
    )


def physical_parameter_vector(
    parameters: VehicleParameters,
) -> np.ndarray:
    inertia = np.asarray(parameters.inertia, dtype=float)
    return np.concatenate(
        (
            np.asarray((parameters.mass,), dtype=float),
            np.asarray(
                [
                    inertia[row, column]
                    for row, column in INERTIA_COMPONENTS
                ],
                dtype=float,
            ),
            np.asarray(parameters.cog_offset, dtype=float),
            np.asarray(
                parameters.force_effectiveness,
                dtype=float,
            ),
        )
    )


def physical_parameter_jacobian(
    parameter_jacobian: Any,
) -> np.ndarray:
    mass = np.asarray(parameter_jacobian.mass, dtype=float)
    dimension = mass.size
    inertia = np.asarray(parameter_jacobian.inertia, dtype=float)
    cog = np.asarray(parameter_jacobian.cog_offset, dtype=float)
    effectiveness = np.asarray(
        parameter_jacobian.force_effectiveness,
        dtype=float,
    )
    if (
        inertia.shape != (3, 3, dimension)
        or cog.shape != (3, dimension)
        or effectiveness.shape != (4, dimension)
    ):
        raise ValueError(
            "physical parameter Jacobian has inconsistent shape"
        )
    rows = [mass]
    rows.extend(
        inertia[row, column]
        for row, column in INERTIA_COMPONENTS
    )
    rows.extend(cog[index] for index in range(3))
    rows.extend(effectiveness[index] for index in range(4))
    return np.vstack(rows)


class DimensionlessParameterization:
    """Fixed-reference dimensionless physical chart.

    The second-moment chart is

        Sigma_bar(q) = B0 exp(S(q)) B0,

    where B0 is the symmetric square root of the reference dimensionless
    second moment and S(q) is symmetric.  Common physical scaling corresponds
    exactly to adding the same scalar to the mass log coordinate, all three
    diagonal S coordinates, and all four effectiveness log coordinates.
    """

    def __init__(
        self,
        reference: VehicleParameters,
        scales: ReferenceScales,
    ) -> None:
        self.reference = reference
        self.scales = scales

        reference_inertia_bar = (
            np.asarray(reference.inertia, dtype=float)
            / (
                scales.mass_kg
                * scales.length_m**2
            )
        )
        reference_second_moment_bar = (
            0.5
            * float(np.trace(reference_inertia_bar))
            * np.eye(3)
            - reference_inertia_bar
        )
        reference_second_moment_bar = 0.5 * (
            reference_second_moment_bar
            + reference_second_moment_bar.T
        )
        eigenvalues, eigenvectors = np.linalg.eigh(
            reference_second_moment_bar
        )
        if (
            np.any(~np.isfinite(eigenvalues))
            or np.any(eigenvalues <= 0.0)
        ):
            raise ValueError(
                "reference inertia does not define a positive "
                "dimensionless second moment"
            )
        self.reference_second_moment_bar = (
            reference_second_moment_bar
        )
        self.reference_second_moment_sqrt = (
            eigenvectors
            @ np.diag(np.sqrt(eigenvalues))
            @ eigenvectors.T
        )

    @staticmethod
    def common_scale_direction() -> np.ndarray:
        return COMMON_SCALE_DIRECTION.copy()

    @staticmethod
    def _symmetric_log_matrix(
        coordinate: np.ndarray,
    ) -> np.ndarray:
        value = np.asarray(coordinate, dtype=float)
        if value.shape != (6,):
            raise ValueError(
                "second-moment log coordinate must be 6-D"
            )
        result = np.zeros((3, 3), dtype=float)
        for coefficient, basis in zip(
            value,
            SYMMETRIC_BASIS,
        ):
            result += float(coefficient) * basis
        return result

    def decode_with_jacobian(
        self,
        coordinate: Sequence[float],
        delay_seconds: float,
    ) -> tuple[Any, Any]:
        value = np.asarray(coordinate, dtype=float)
        if (
            value.shape != (PHYSICAL_DIMENSION,)
            or np.any(~np.isfinite(value))
            or not np.isfinite(delay_seconds)
            or delay_seconds < 0.0
        ):
            raise ValueError(
                "dimensionless physical coordinate is invalid"
            )

        mass = self.scales.mass_kg * math.exp(float(value[0]))
        log_matrix = self._symmetric_log_matrix(value[1:7])
        exponential = expm(log_matrix)
        base = self.reference_second_moment_sqrt
        second_moment_bar = base @ exponential @ base
        second_moment_bar = 0.5 * (
            second_moment_bar + second_moment_bar.T
        )
        inertia_bar = (
            np.trace(second_moment_bar) * np.eye(3)
            - second_moment_bar
        )
        inertia_bar = 0.5 * (inertia_bar + inertia_bar.T)
        inertia = (
            self.scales.mass_kg
            * self.scales.length_m**2
            * inertia_bar
        )

        cog = (
            np.asarray(self.reference.cog_offset, dtype=float)
            + self.scales.length_m * value[7:10]
        )
        effectiveness = (
            np.asarray(
                self.reference.force_effectiveness,
                dtype=float,
            )
            * np.exp(value[10:14])
        )

        mandatory = (
            np.asarray((mass,), dtype=float),
            second_moment_bar,
            inertia_bar,
            inertia,
            cog,
            effectiveness,
        )
        if any(np.any(~np.isfinite(item)) for item in mandatory):
            raise FloatingPointError(
                "dimensionless physical decoding became non-finite"
            )

        principal_bar = np.linalg.eigvalsh(inertia_bar)
        if np.any(principal_bar <= 0.0):
            raise FloatingPointError(
                "dimensionless inertia lost positive definiteness"
            )
        principal = (
            self.scales.mass_kg
            * self.scales.length_m**2
            * principal_bar
        )
        triangle_margin = float(
            principal[0] + principal[1] - principal[2]
        )

        parameters = VehicleParameters(
            mass=mass,
            inertia=inertia,
            cog_offset=cog,
            force_effectiveness=effectiveness,
            torque_effectiveness=(
                self.reference.torque_effectiveness
            ),
            linear_drag=self.reference.linear_drag,
            angular_drag=self.reference.angular_drag,
        )
        actuator_parameters = ActuatorParameters(
            thrust_time_constant=0.0,
            gimbal_time_constant=0.0,
            delay=0.0,
        )
        decoded = strict.analytic.DecodedSearchPoint(
            parameters=parameters,
            actuator_parameters=actuator_parameters,
            delay=float(delay_seconds),
            inertia_principal_moments=principal,
            inertia_triangle_margin=triangle_margin,
        )

        dimension = PHYSICAL_DIMENSION
        mass_jacobian = np.zeros(dimension, dtype=float)
        mass_jacobian[0] = mass

        inertia_jacobian = np.zeros(
            (3, 3, dimension),
            dtype=float,
        )
        inertia_scale = (
            self.scales.mass_kg
            * self.scales.length_m**2
        )
        for local_index, basis in enumerate(SYMMETRIC_BASIS):
            exponential_derivative = expm_frechet(
                log_matrix,
                basis,
                compute_expm=False,
            )
            second_derivative_bar = (
                base @ exponential_derivative @ base
            )
            second_derivative_bar = 0.5 * (
                second_derivative_bar
                + second_derivative_bar.T
            )
            inertia_derivative_bar = (
                np.trace(second_derivative_bar)
                * np.eye(3)
                - second_derivative_bar
            )
            inertia_jacobian[
                :,
                :,
                1 + local_index,
            ] = (
                inertia_scale
                * inertia_derivative_bar
            )

        cog_jacobian = np.zeros(
            (3, dimension),
            dtype=float,
        )
        cog_jacobian[:, 7:10] = (
            self.scales.length_m * np.eye(3)
        )

        effectiveness_jacobian = np.zeros(
            (4, dimension),
            dtype=float,
        )
        effectiveness_jacobian[:, 10:14] = np.diag(
            effectiveness
        )

        zero = np.zeros(dimension, dtype=float)
        jacobian = strict.analytic.DecodedSearchJacobian(
            mass=mass_jacobian,
            inertia=inertia_jacobian,
            cog_offset=cog_jacobian,
            force_effectiveness=effectiveness_jacobian,
            thrust_time_constant=zero.copy(),
            gimbal_time_constant=zero.copy(),
        )
        return decoded, jacobian

    def decode(
        self,
        coordinate: Sequence[float],
        delay_seconds: float = 0.0,
    ) -> Any:
        decoded, _jacobian = self.decode_with_jacobian(
            coordinate,
            delay_seconds,
        )
        return decoded

    def self_test(self) -> dict[str, float]:
        coordinate = np.asarray(
            (
                0.1,
                0.05,
                -0.03,
                0.02,
                0.01,
                -0.015,
                0.008,
                0.02,
                -0.01,
                0.03,
                0.04,
                -0.02,
                0.03,
                -0.01,
            ),
            dtype=float,
        )
        shift = 0.37
        direction = self.common_scale_direction()
        first, first_jacobian = self.decode_with_jacobian(
            coordinate,
            0.01,
        )
        second, _ = self.decode_with_jacobian(
            coordinate + shift * direction,
            0.01,
        )
        factor = math.exp(shift)
        mass_error = abs(
            second.parameters.mass
            / first.parameters.mass
            - factor
        )
        inertia_error = float(
            np.linalg.norm(
                second.parameters.inertia
                - factor * first.parameters.inertia
            )
            / max(
                np.linalg.norm(first.parameters.inertia),
                np.finfo(float).tiny,
            )
        )
        effectiveness_error = float(
            np.linalg.norm(
                second.parameters.force_effectiveness
                - factor
                * first.parameters.force_effectiveness
            )
            / max(
                np.linalg.norm(
                    first.parameters.force_effectiveness
                ),
                np.finfo(float).tiny,
            )
        )
        cog_error = float(
            np.linalg.norm(
                second.parameters.cog_offset
                - first.parameters.cog_offset
            )
        )
        test_direction = np.asarray(
            (
                0.2,
                -0.1,
                0.15,
                0.08,
                -0.04,
                0.06,
                -0.03,
                0.05,
                -0.02,
                0.01,
                0.07,
                -0.05,
                0.03,
                -0.02,
            ),
            dtype=float,
        )
        test_direction /= np.linalg.norm(test_direction)
        finite_step = 1.0e-6
        plus = self.decode(
            coordinate + finite_step * test_direction,
            0.01,
        )
        minus = self.decode(
            coordinate - finite_step * test_direction,
            0.01,
        )
        finite_physical = (
            physical_parameter_vector(plus.parameters)
            - physical_parameter_vector(minus.parameters)
        ) / (2.0 * finite_step)
        analytic_physical = (
            physical_parameter_jacobian(first_jacobian)
            @ test_direction
        )
        jacobian_relative_error = float(
            np.linalg.norm(
                finite_physical - analytic_physical
            )
            / max(
                np.linalg.norm(finite_physical),
                np.linalg.norm(analytic_physical),
                np.finfo(float).tiny,
            )
        )

        maximum = max(
            mass_error,
            inertia_error,
            effectiveness_error,
            cog_error,
            jacobian_relative_error,
        )
        if maximum > 2.0e-7:
            raise RuntimeError(
                "dimensionless chart self-test failed"
            )
        return {
            "mass_relative_error": mass_error,
            "inertia_relative_error": inertia_error,
            "effectiveness_relative_error": effectiveness_error,
            "cog_absolute_error_m": cog_error,
            "physical_jacobian_relative_error": (
                jacobian_relative_error
            ),
        }


def _extend_parameter_jacobian(
    source: Any,
    dimension: int,
) -> Any:
    source_dimension = int(np.asarray(source.mass).size)
    if (
        source_dimension != PHYSICAL_DIMENSION
        or dimension < source_dimension
    ):
        raise ValueError(
            "extended parameter-Jacobian dimension is invalid"
        )

    def extend_vector(vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=float)
        result = np.zeros(dimension, dtype=float)
        result[:source_dimension] = value
        return result

    def extend_matrix(matrix: np.ndarray) -> np.ndarray:
        value = np.asarray(matrix, dtype=float)
        result = np.zeros(
            (value.shape[0], dimension),
            dtype=float,
        )
        result[:, :source_dimension] = value
        return result

    inertia = np.zeros(
        (3, 3, dimension),
        dtype=float,
    )
    inertia[:, :, :source_dimension] = np.asarray(
        source.inertia,
        dtype=float,
    )
    return strict.analytic.DecodedSearchJacobian(
        mass=extend_vector(source.mass),
        inertia=inertia,
        cog_offset=extend_matrix(source.cog_offset),
        force_effectiveness=extend_matrix(
            source.force_effectiveness
        ),
        thrust_time_constant=extend_vector(
            source.thrust_time_constant
        ),
        gimbal_time_constant=extend_vector(
            source.gimbal_time_constant
        ),
    )


@dataclass(frozen=True)
class BagData:
    specification: Any
    normalized_weight: float
    flight: Any
    direct_problem: Any
    selection: Any
    kinematics: Any
    rotor_history: QuinticSmoothZoh
    gimbal_history: QuinticSmoothZoh
    initial_gimbal: np.ndarray

    @property
    def time(self) -> np.ndarray:
        return np.asarray(self.kinematics.time, dtype=float)

    @property
    def spline_selection(self) -> Any:
        # Compatibility alias used only by report helpers.
        return self.selection


@dataclass(frozen=True)
class BagEvaluation:
    acceleration_residual: np.ndarray
    acceleration_jacobian: np.ndarray
    required_body_wrench: np.ndarray
    modeled_body_wrench: np.ndarray
    residual_body_wrench: np.ndarray
    residual_body_wrench_jacobian: np.ndarray
    actuator_thrust: np.ndarray
    actuator_gimbal: np.ndarray
    cog_position_world: np.ndarray
    cog_velocity_world: np.ndarray
    cog_acceleration_world: np.ndarray

    @property
    def data_loss(self) -> float:
        residual = np.asarray(
            self.acceleration_residual,
            dtype=float,
        )
        return 0.5 * float(
            np.mean(
                np.sum(
                    residual * residual,
                    axis=1,
                )
            )
        )


@dataclass(frozen=True)
class JointEvaluation:
    residual: np.ndarray
    jacobian: np.ndarray
    bag_evaluations: tuple[BagEvaluation, ...]
    decoded: Any
    physical_coordinate: np.ndarray
    rotor_delay_seconds: float
    gimbal_delay_seconds: float
    data_loss: float


@dataclass(frozen=True)
class Solution:
    physical_coordinate: np.ndarray
    rotor_delay_seconds: float
    gimbal_delay_seconds: float
    evaluation: JointEvaluation
    optimizer: Mapping[str, Any]


class CachedObjective:
    def __init__(self, evaluator: Any) -> None:
        self.evaluator = evaluator
        self.coordinate: Optional[np.ndarray] = None
        self.evaluation: Optional[JointEvaluation] = None

    def get(self, coordinate: np.ndarray) -> JointEvaluation:
        value = np.asarray(coordinate, dtype=float)
        if (
            self.coordinate is None
            or not np.array_equal(value, self.coordinate)
        ):
            self.coordinate = value.copy()
            self.evaluation = self.evaluator(value)
        if self.evaluation is None:
            raise RuntimeError("cached objective has no evaluation")
        return self.evaluation

    def residual(self, coordinate: np.ndarray) -> np.ndarray:
        return self.get(coordinate).residual

    def jacobian(self, coordinate: np.ndarray) -> np.ndarray:
        return self.get(coordinate).jacobian


def _build_bag(
    specification: Any,
    normalized_weight: float,
    flight: Any,
    window_seconds: float,
    arguments: argparse.Namespace,
    model: VehicleModelInput,
    scales: ReferenceScales,
) -> BagData:
    direct = baseline.DirectShootingProblem(
        flight=flight,
        sample_step=arguments.sample_step,
        integration_step=arguments.integration_step,
        command_delay=0.0,
        prior_weight=0.0,
        reference_parameters=model.parameters,
        geometry=model.geometry,
    )

    raw_time = np.asarray(flight.pose.times, dtype=float)
    if not sg.window_is_feasible(
        raw_time,
        window_seconds,
        degree=sg.POLYNOMIAL_DEGREE,
    ):
        minimum = sg.minimum_feasible_window_seconds(raw_time)
        raise ValueError(
            "W={:.12g}s is infeasible for bag {}; "
            "minimum is {:.12g}s".format(
                window_seconds,
                specification.bag_id,
                minimum,
            )
        )

    selection = sg.select_pose_spline(
        time_axis=raw_time,
        sensor_position=np.asarray(
            flight.pose.positions,
            dtype=float,
        ),
        sensor_orientation_xyzw=np.asarray(
            flight.pose.orientations_xyzw,
            dtype=float,
        ),
        body_to_pose_sensor_rotation=(
            direct.pose_body_to_sensor_rotation
        ),
        knot_spacing_candidates_seconds=(
            float(window_seconds),
        ),
        rotational_metric=(
            np.asarray(model.parameters.inertia, dtype=float)
            / float(model.parameters.mass)
        ),
    )

    maximum_delay = float(arguments.delay_bounds[1])
    support_start = max(
        float(direct.output_time[0]),
        float(flight.rotor_command.all_times[0])
        + maximum_delay,
        float(flight.gimbal_command.all_times[0])
        + maximum_delay,
        float(flight.gimbal_position.times[0]),
    )
    support_end = min(
        float(direct.output_time[-1]),
        float(raw_time[-1]),
    )
    time_axis = selection.spline.centered_raw_times(
        support_start=support_start,
        support_end=support_end,
    )
    kinematics = selection.spline.evaluate(time_axis)
    initial_gimbal = baseline._linear_interpolate(
        flight.gimbal_position.times,
        flight.gimbal_position.values,
        np.asarray((time_axis[0],), dtype=float),
    )[0]

    return BagData(
        specification=specification,
        normalized_weight=float(normalized_weight),
        flight=flight,
        direct_problem=direct,
        selection=selection,
        kinematics=kinematics,
        rotor_history=QuinticSmoothZoh(
            flight.rotor_command.all_times,
            flight.rotor_command.all_values,
        ),
        gimbal_history=QuinticSmoothZoh(
            flight.gimbal_command.all_times,
            flight.gimbal_command.all_values,
        ),
        initial_gimbal=np.asarray(
            initial_gimbal,
            dtype=float,
        ),
    )


class DimensionlessDynamicsProblem:
    def __init__(
        self,
        bags: Sequence[BagData],
        model: VehicleModelInput,
        scales: ReferenceScales,
    ) -> None:
        self.bags = tuple(bags)
        if not self.bags:
            raise ValueError("at least one bag is required")
        weights = np.asarray(
            [
                bag.normalized_weight
                for bag in self.bags
            ],
            dtype=float,
        )
        if not np.isclose(
            np.sum(weights),
            1.0,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "normalized bag weights must sum to one"
            )
        self.model = model
        self.scales = scales
        self.parameterization = DimensionlessParameterization(
            model.parameters,
            scales,
        )
        self.exact_common_scale_symmetry = bool(
            np.allclose(
                model.parameters.linear_drag,
                0.0,
                atol=0.0,
                rtol=0.0,
            )
            and np.allclose(
                model.parameters.angular_drag,
                0.0,
                atol=0.0,
                rtol=0.0,
            )
        )

    def _decode(
        self,
        physical_coordinate: np.ndarray,
        rotor_delay_seconds: float,
        dimension: int,
    ) -> tuple[Any, Any]:
        decoded, jacobian = (
            self.parameterization.decode_with_jacobian(
                physical_coordinate,
                rotor_delay_seconds,
            )
        )
        return decoded, _extend_parameter_jacobian(
            jacobian,
            dimension,
        )

    def _actuator_series(
        self,
        bag: BagData,
        decoded: Any,
        parameter_jacobian: Any,
        dimension: int,
        rotor_delay_seconds: float,
        gimbal_delay_seconds: float,
        smooth_mode: bool,
        width_fraction: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        time_axis = bag.time
        count = time_axis.size
        thrust = np.empty((count, 4), dtype=float)
        gimbal = np.empty((count, 4), dtype=float)
        sensitivity = np.zeros(
            (count, 8, dimension),
            dtype=float,
        )

        actuator_parameters = decoded.actuator_parameters
        if (
            actuator_parameters.thrust_time_constant != 0.0
            or actuator_parameters.gimbal_time_constant != 0.0
        ):
            raise RuntimeError(
                "dimensionless SG experiment assumes zero "
                "actuator time constants"
            )

        for index, sample_time in enumerate(time_axis):
            if smooth_mode:
                rotor = bag.rotor_history.evaluate(
                    float(sample_time),
                    rotor_delay_seconds,
                    width_fraction,
                )
                raw = np.asarray(rotor.value, dtype=float)
                free = (
                    (raw > actuator_parameters.minimum_thrust)
                    & (raw < actuator_parameters.maximum_thrust)
                )
                thrust[index] = np.clip(
                    raw,
                    actuator_parameters.minimum_thrust,
                    actuator_parameters.maximum_thrust,
                )
                sensitivity[
                    index,
                    :4,
                    ROTOR_DELAY_INDEX,
                ] = (
                    np.where(
                        free,
                        rotor.delay_derivative,
                        0.0,
                    )
                    * self.scales.time_s
                )
            else:
                thrust[index] = np.clip(
                    bag.rotor_history.exact_zoh(
                        float(sample_time),
                        rotor_delay_seconds,
                    ),
                    actuator_parameters.minimum_thrust,
                    actuator_parameters.maximum_thrust,
                )

        current_state = ActuatorState(
            thrust=thrust[0].copy(),
            gimbal_angle=np.asarray(
                bag.initial_gimbal,
                dtype=float,
            ).copy(),
        )
        current_sensitivity = np.zeros(
            (8, dimension),
            dtype=float,
        )
        gimbal[0] = current_state.gimbal_angle

        shifted_switches = (
            np.asarray(
                bag.gimbal_history.times[1:],
                dtype=float,
            )
            + gimbal_delay_seconds
        )

        for index in range(count - 1):
            left = float(time_axis[index])
            right = float(time_axis[index + 1])

            if smooth_mode:
                middle = 0.5 * (left + right)
                intervals = (
                    (left, middle),
                    (middle, right),
                )
                for sub_left, sub_right in intervals:
                    step = float(sub_right - sub_left)
                    if step <= 0.0:
                        continue
                    query = 0.5 * (
                        sub_left + sub_right
                    )
                    rotor = bag.rotor_history.evaluate(
                        query,
                        rotor_delay_seconds,
                        width_fraction,
                    )
                    gimbal_command = (
                        bag.gimbal_history.evaluate(
                            query,
                            gimbal_delay_seconds,
                            width_fraction,
                        )
                    )
                    command = smooth._command(
                        rotor.value,
                        gimbal_command.value,
                    )
                    command_sensitivity = np.zeros(
                        (8, dimension),
                        dtype=float,
                    )
                    command_sensitivity[
                        :4,
                        ROTOR_DELAY_INDEX,
                    ] = (
                        rotor.delay_derivative
                        * self.scales.time_s
                    )
                    command_sensitivity[
                        4:,
                        GIMBAL_DELAY_INDEX,
                    ] = (
                        gimbal_command.delay_derivative
                        * self.scales.time_s
                    )
                    (
                        current_state,
                        current_sensitivity,
                    ) = strict._actuator_step_with_sensitivity(
                        current_state,
                        current_sensitivity,
                        command,
                        decoded,
                        parameter_jacobian,
                        step,
                        command_sensitivity,
                    )
            else:
                first = int(
                    np.searchsorted(
                        shifted_switches,
                        left,
                        side="right",
                    )
                )
                last = int(
                    np.searchsorted(
                        shifted_switches,
                        right,
                        side="left",
                    )
                )
                boundaries = np.concatenate(
                    (
                        np.asarray((left,), dtype=float),
                        shifted_switches[first:last],
                        np.asarray((right,), dtype=float),
                    )
                )
                for sub_left, sub_right in zip(
                    boundaries[:-1],
                    boundaries[1:],
                ):
                    step = float(sub_right - sub_left)
                    if step <= 0.0:
                        continue
                    query = 0.5 * (
                        float(sub_left)
                        + float(sub_right)
                    )
                    command = smooth._command(
                        bag.rotor_history.exact_zoh(
                            query,
                            rotor_delay_seconds,
                        ),
                        bag.gimbal_history.exact_zoh(
                            query,
                            gimbal_delay_seconds,
                        ),
                    )
                    current_state = advance_actuators(
                        current_state,
                        command,
                        actuator_parameters,
                        step,
                    )

            gimbal[index + 1] = (
                current_state.gimbal_angle
            )
            if smooth_mode:
                sensitivity[
                    index + 1,
                    4:,
                ] = current_sensitivity[4:]

        arrays = (thrust, gimbal, sensitivity)
        if any(np.any(~np.isfinite(item)) for item in arrays):
            raise FloatingPointError(
                "actuator evaluation became non-finite"
            )
        return thrust, gimbal, sensitivity

    def _evaluate_bag(
        self,
        bag: BagData,
        decoded: Any,
        parameter_jacobian: Any,
        dimension: int,
        rotor_delay_seconds: float,
        gimbal_delay_seconds: float,
        smooth_mode: bool,
        width_fraction: float,
    ) -> BagEvaluation:
        kinematics = bag.kinematics
        count = bag.time.size
        (
            thrust,
            gimbal,
            actuator_jacobian,
        ) = self._actuator_series(
            bag,
            decoded,
            parameter_jacobian,
            dimension,
            rotor_delay_seconds,
            gimbal_delay_seconds,
            smooth_mode,
            width_fraction,
        )

        acceleration_residual = np.empty(
            (count, 6),
            dtype=float,
        )
        acceleration_jacobian = np.empty(
            (count, 6, dimension),
            dtype=float,
        )
        required_wrench = np.empty(
            (count, 6),
            dtype=float,
        )
        modeled_wrench = np.empty(
            (count, 6),
            dtype=float,
        )
        residual_wrench = np.empty(
            (count, 6),
            dtype=float,
        )
        residual_wrench_jacobian = np.empty(
            (count, 6, dimension),
            dtype=float,
        )
        cog_position = np.empty(
            (count, 3),
            dtype=float,
        )
        cog_velocity = np.empty(
            (count, 3),
            dtype=float,
        )
        cog_acceleration = np.empty(
            (count, 3),
            dtype=float,
        )

        parameters = decoded.parameters
        mass_bar = (
            float(parameters.mass)
            / self.scales.mass_kg
        )
        inertia_bar = (
            np.asarray(parameters.inertia, dtype=float)
            / (
                self.scales.mass_kg
                * self.scales.length_m**2
            )
        )
        mass_jacobian_bar = (
            np.asarray(
                parameter_jacobian.mass,
                dtype=float,
            )
            / self.scales.mass_kg
        )
        inertia_jacobian_bar = (
            np.asarray(
                parameter_jacobian.inertia,
                dtype=float,
            )
            / (
                self.scales.mass_kg
                * self.scales.length_m**2
            )
        )

        zero_rotation_sensitivity = np.zeros(
            (3, dimension),
            dtype=float,
        )
        zero_omega_sensitivity = np.zeros(
            (3, dimension),
            dtype=float,
        )
        gravity_world = np.asarray(
            (0.0, 0.0, -GRAVITY),
            dtype=float,
        )

        lever = (
            np.asarray(
                bag.direct_problem.pose_sensor_position,
                dtype=float,
            )
            - np.asarray(
                parameters.cog_offset,
                dtype=float,
            )
        )
        cog_position[:] = (
            np.asarray(
                kinematics.sensor_position,
                dtype=float,
            )
            - np.einsum(
                "nij,j->ni",
                kinematics.body_rotation,
                lever,
            )
        )
        cog_velocity[:] = (
            np.asarray(
                kinematics.sensor_velocity_world,
                dtype=float,
            )
            - np.einsum(
                "nij,nj->ni",
                kinematics.body_rotation,
                np.cross(
                    kinematics.body_angular_velocity,
                    lever,
                ),
            )
        )
        rotational_acceleration = (
            np.cross(
                kinematics.body_angular_acceleration,
                lever,
            )
            + np.cross(
                kinematics.body_angular_velocity,
                np.cross(
                    kinematics.body_angular_velocity,
                    lever,
                ),
            )
        )
        cog_acceleration[:] = (
            np.asarray(
                kinematics.sensor_acceleration_world,
                dtype=float,
            )
            - np.einsum(
                "nij,nj->ni",
                kinematics.body_rotation,
                rotational_acceleration,
            )
        )

        for index in range(count):
            rotation = np.asarray(
                kinematics.body_rotation[index],
                dtype=float,
            )
            omega = np.asarray(
                kinematics.body_angular_velocity[index],
                dtype=float,
            )
            alpha = np.asarray(
                kinematics.body_angular_acceleration[index],
                dtype=float,
            )
            angular_kinematics = (
                skew(alpha)
                + skew(omega) @ skew(omega)
            )

            velocity_sensitivity = (
                rotation
                @ skew(omega)
                @ parameter_jacobian.cog_offset
            )
            acceleration_body = (
                rotation.T
                @ (
                    cog_acceleration[index]
                    - gravity_world
                )
            )
            acceleration_body_sensitivity = (
                angular_kinematics
                @ parameter_jacobian.cog_offset
            )

            actuator_state = ActuatorState(
                thrust=thrust[index],
                gimbal_angle=gimbal[index],
            )
            (
                wrench,
                wrench_jacobian,
            ) = strict._body_wrench_with_sensitivity(
                bag.direct_problem,
                decoded,
                parameter_jacobian,
                rotation,
                zero_rotation_sensitivity,
                cog_velocity[index],
                velocity_sensitivity,
                omega,
                zero_omega_sensitivity,
                actuator_state,
                actuator_jacobian[index],
            )

            force_bar = (
                wrench[:3]
                / self.scales.force_n
            )
            torque_bar = (
                wrench[3:]
                / self.scales.torque_nm
            )
            force_jacobian_bar = (
                wrench_jacobian[:3]
                / self.scales.force_n
            )
            torque_jacobian_bar = (
                wrench_jacobian[3:]
                / self.scales.torque_nm
            )

            observed_acceleration_bar = (
                acceleration_body
                * self.scales.time_s**2
                / self.scales.length_m
            )
            observed_acceleration_jacobian_bar = (
                acceleration_body_sensitivity
                * self.scales.time_s**2
                / self.scales.length_m
            )
            omega_bar = (
                omega * self.scales.time_s
            )
            observed_alpha_bar = (
                alpha * self.scales.time_s**2
            )

            predicted_acceleration_bar = (
                force_bar / mass_bar
            )
            predicted_acceleration_jacobian_bar = (
                force_jacobian_bar / mass_bar
                - np.outer(
                    force_bar,
                    mass_jacobian_bar,
                )
                / mass_bar**2
            )

            inertia_omega_bar = (
                inertia_bar @ omega_bar
            )
            angular_rhs_bar = (
                torque_bar
                - np.cross(
                    omega_bar,
                    inertia_omega_bar,
                )
            )
            predicted_alpha_bar = np.linalg.solve(
                inertia_bar,
                angular_rhs_bar,
            )
            inertia_omega_jacobian_bar = np.einsum(
                "ijk,j->ik",
                inertia_jacobian_bar,
                omega_bar,
            )
            angular_rhs_jacobian_bar = (
                torque_jacobian_bar
                - skew(omega_bar)
                @ inertia_omega_jacobian_bar
            )
            inertia_alpha_jacobian_bar = np.einsum(
                "ijk,j->ik",
                inertia_jacobian_bar,
                predicted_alpha_bar,
            )
            predicted_alpha_jacobian_bar = (
                np.linalg.solve(
                    inertia_bar,
                    angular_rhs_jacobian_bar
                    - inertia_alpha_jacobian_bar,
                )
            )

            acceleration_residual[index, :3] = (
                observed_acceleration_bar
                - predicted_acceleration_bar
            )
            acceleration_residual[index, 3:] = (
                observed_alpha_bar
                - predicted_alpha_bar
            )
            acceleration_jacobian[index, :3] = (
                observed_acceleration_jacobian_bar
                - predicted_acceleration_jacobian_bar
            )
            acceleration_jacobian[index, 3:] = (
                -predicted_alpha_jacobian_bar
            )

            required_force = (
                parameters.mass
                * acceleration_body
            )
            required_force_jacobian = (
                np.outer(
                    acceleration_body,
                    parameter_jacobian.mass,
                )
                + parameters.mass
                * acceleration_body_sensitivity
            )
            inertia_omega = (
                parameters.inertia @ omega
            )
            required_torque = (
                parameters.inertia @ alpha
                + np.cross(
                    omega,
                    inertia_omega,
                )
            )
            inertia_omega_jacobian = np.einsum(
                "ijk,j->ik",
                parameter_jacobian.inertia,
                omega,
            )
            required_torque_jacobian = (
                np.einsum(
                    "ijk,j->ik",
                    parameter_jacobian.inertia,
                    alpha,
                )
                + skew(omega)
                @ inertia_omega_jacobian
            )

            required_wrench[index, :3] = (
                required_force
            )
            required_wrench[index, 3:] = (
                required_torque
            )
            modeled_wrench[index] = wrench
            residual_wrench[index] = (
                required_wrench[index] - wrench
            )
            residual_wrench_jacobian[
                index,
                :3,
            ] = (
                required_force_jacobian
                - wrench_jacobian[:3]
            )
            residual_wrench_jacobian[
                index,
                3:,
            ] = (
                required_torque_jacobian
                - wrench_jacobian[3:]
            )

        arrays = (
            acceleration_residual,
            acceleration_jacobian,
            required_wrench,
            modeled_wrench,
            residual_wrench,
            residual_wrench_jacobian,
            thrust,
            gimbal,
            cog_position,
            cog_velocity,
            cog_acceleration,
        )
        if any(np.any(~np.isfinite(item)) for item in arrays):
            raise FloatingPointError(
                "dimensionless SG bag evaluation became non-finite"
            )
        return BagEvaluation(
            acceleration_residual=acceleration_residual,
            acceleration_jacobian=acceleration_jacobian,
            required_body_wrench=required_wrench,
            modeled_body_wrench=modeled_wrench,
            residual_body_wrench=residual_wrench,
            residual_body_wrench_jacobian=(
                residual_wrench_jacobian
            ),
            actuator_thrust=thrust,
            actuator_gimbal=gimbal,
            cog_position_world=cog_position,
            cog_velocity_world=cog_velocity,
            cog_acceleration_world=cog_acceleration,
        )

    def _evaluate_joint(
        self,
        physical_coordinate: np.ndarray,
        rotor_delay_seconds: float,
        gimbal_delay_seconds: float,
        dimension: int,
        smooth_mode: bool,
        width_fraction: float,
    ) -> JointEvaluation:
        decoded, parameter_jacobian = self._decode(
            physical_coordinate,
            rotor_delay_seconds,
            dimension,
        )
        residual_blocks = []
        jacobian_blocks = []
        bag_evaluations = []
        data_loss = 0.0
        for bag in self.bags:
            evaluation = self._evaluate_bag(
                bag,
                decoded,
                parameter_jacobian,
                dimension,
                rotor_delay_seconds,
                gimbal_delay_seconds,
                smooth_mode,
                width_fraction,
            )
            bag_evaluations.append(evaluation)
            root_scale = math.sqrt(
                bag.normalized_weight
                / bag.time.size
            )
            residual_blocks.append(
                root_scale
                * evaluation.acceleration_residual.ravel()
            )
            jacobian_blocks.append(
                root_scale
                * evaluation.acceleration_jacobian.reshape(
                    -1,
                    dimension,
                )
            )
            data_loss += (
                bag.normalized_weight
                * evaluation.data_loss
            )

        residual = np.concatenate(residual_blocks)
        jacobian = np.vstack(jacobian_blocks)
        objective_cost = 0.5 * float(
            residual @ residual
        )
        if not np.isclose(
            objective_cost,
            data_loss,
            rtol=2.0e-11,
            atol=1.0e-13,
        ):
            raise RuntimeError(
                "dimensionless joint residual scaling is inconsistent"
            )
        return JointEvaluation(
            residual=residual,
            jacobian=jacobian,
            bag_evaluations=tuple(bag_evaluations),
            decoded=decoded,
            physical_coordinate=np.asarray(
                physical_coordinate,
                dtype=float,
            ).copy(),
            rotor_delay_seconds=float(
                rotor_delay_seconds
            ),
            gimbal_delay_seconds=float(
                gimbal_delay_seconds
            ),
            data_loss=float(data_loss),
        )

    def evaluate_smooth(
        self,
        coordinate: Sequence[float],
        width_fraction: float,
    ) -> JointEvaluation:
        value = np.asarray(coordinate, dtype=float)
        if (
            value.shape != (GLOBAL_DIMENSION,)
            or np.any(~np.isfinite(value))
        ):
            raise ValueError(
                "smooth dimensionless coordinate must be 16-D"
            )
        rotor_delay = (
            float(value[ROTOR_DELAY_INDEX])
            * self.scales.time_s
        )
        gimbal_delay = (
            float(value[GIMBAL_DELAY_INDEX])
            * self.scales.time_s
        )
        return self._evaluate_joint(
            value[:PHYSICAL_DIMENSION],
            rotor_delay,
            gimbal_delay,
            GLOBAL_DIMENSION,
            True,
            float(width_fraction),
        )

    def evaluate_strict(
        self,
        physical_coordinate: Sequence[float],
        rotor_delay_seconds: float,
        gimbal_delay_seconds: float,
    ) -> JointEvaluation:
        value = np.asarray(
            physical_coordinate,
            dtype=float,
        )
        if (
            value.shape != (PHYSICAL_DIMENSION,)
            or np.any(~np.isfinite(value))
        ):
            raise ValueError(
                "strict dimensionless coordinate must be 14-D"
            )
        return self._evaluate_joint(
            value,
            float(rotor_delay_seconds),
            float(gimbal_delay_seconds),
            PHYSICAL_DIMENSION,
            False,
            1.0,
        )


def _physical_bounds(
    arguments: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    del arguments
    return (
        np.full(PHYSICAL_DIMENSION, -np.inf, dtype=float),
        np.full(PHYSICAL_DIMENSION, np.inf, dtype=float),
    )


def _jacobian_spectrum(
    jacobian: np.ndarray,
) -> dict[str, Any]:
    value = np.asarray(jacobian, dtype=float)
    _u, singular, vt = np.linalg.svd(
        value,
        full_matrices=False,
    )
    tolerance = (
        0.0
        if singular.size == 0
        else (
            max(value.shape)
            * np.finfo(float).eps
            * float(singular[0])
        )
    )
    rank = int(np.count_nonzero(singular > tolerance))
    condition = (
        None
        if rank == 0
        else float(singular[0] / singular[rank - 1])
    )
    return {
        "singular_values": singular,
        "right_singular_vectors": vt,
        "numerical_tolerance": tolerance,
        "numerical_rank": rank,
        "nullity": int(value.shape[1] - rank),
        "condition_number_nonzero_subspace": condition,
        "rank_threshold_interpretation": (
            "machine-precision numerical rank only; "
            "not a scientific ridge cutoff"
        ),
    }


def _normalized_scale_gauge(
    dimension: int,
) -> np.ndarray:
    if dimension < PHYSICAL_DIMENSION:
        raise ValueError(
            "gauge dimension is smaller than physical dimension"
        )
    direction = np.zeros(dimension, dtype=float)
    direction[:PHYSICAL_DIMENSION] = COMMON_SCALE_DIRECTION
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(
            "common-scale gauge direction is invalid"
        )
    return direction / norm


def _gauge_diagnostics(
    evaluator: Any,
    coordinate: np.ndarray,
    evaluation: JointEvaluation,
    exact_expected: bool,
) -> dict[str, Any]:
    value = np.asarray(coordinate, dtype=float)
    direction = _normalized_scale_gauge(value.size)

    jacobian = np.asarray(evaluation.jacobian, dtype=float)
    image = jacobian @ direction
    denominator = max(
        float(np.linalg.norm(jacobian)),
        np.finfo(float).tiny,
    )
    analytic_relative = float(
        np.linalg.norm(image) / denominator
    )

    step = 1.0e-5
    plus = evaluator(value + step * direction).residual
    minus = evaluator(value - step * direction).residual
    finite = (plus - minus) / (2.0 * step)
    residual_scale = max(
        float(np.linalg.norm(evaluation.residual)),
        1.0,
    )
    finite_relative = float(
        np.linalg.norm(finite) / residual_scale
    )

    spectrum = _jacobian_spectrum(jacobian)
    alignment = None
    if spectrum["right_singular_vectors"].size:
        weakest = np.asarray(
            spectrum["right_singular_vectors"][-1],
            dtype=float,
        )
        alignment = float(abs(weakest @ direction))

    payload = {
        "exact_symmetry_expected": exact_expected,
        "coordinate_component_along_normalized_gauge": float(
            value @ direction
        ),
        "analytic_jacobian_relative_null_norm": analytic_relative,
        "finite_difference_relative_null_norm": finite_relative,
        "weakest_right_singular_vector_alignment": alignment,
        "normalized_common_scale_direction": direction,
    }
    if (
        exact_expected
        and (
            analytic_relative > 2.0e-7
            or finite_relative > 2.0e-7
        )
    ):
        raise RuntimeError(
            "common mass/inertia/thrust scale is not a numerical "
            "null direction; dimensionless implementation is inconsistent"
        )
    return payload


def _project_away_exact_gauge(
    vector: np.ndarray,
    direction: Optional[np.ndarray],
) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    if direction is None:
        return value.copy()
    return value - direction * float(direction @ value)


def _trial_coordinate_is_safe(
    coordinate: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    arguments: argparse.Namespace,
) -> tuple[bool, Optional[str]]:
    value = np.asarray(coordinate, dtype=float)
    if np.any(~np.isfinite(value)):
        return False, "nonfinite_coordinate"
    if np.any(value < lower) or np.any(value > upper):
        return False, "explicit_delay_bound"
    guard = float(arguments.numeric_coordinate_guard)
    physical = value[:PHYSICAL_DIMENSION]
    if np.max(np.abs(physical)) > guard:
        return False, "broad_numeric_coordinate_guard"
    return True, None


def _solve_lm_kkt_step(
    jacobian: np.ndarray,
    residual: np.ndarray,
    damping: float,
    gauge_direction: Optional[np.ndarray],
) -> tuple[np.ndarray, float, float, float]:
    value = np.asarray(jacobian, dtype=float)
    r = np.asarray(residual, dtype=float)
    hessian = value.T @ value
    gradient = value.T @ r
    dimension = hessian.shape[0]
    regularized = (
        hessian + float(damping) * np.eye(dimension)
    )

    if gauge_direction is None:
        step = np.linalg.solve(regularized, -gradient)
        multiplier = 0.0
        constraint_violation = 0.0
    else:
        direction = np.asarray(
            gauge_direction,
            dtype=float,
        )
        kkt = np.zeros(
            (dimension + 1, dimension + 1),
            dtype=float,
        )
        kkt[:dimension, :dimension] = regularized
        kkt[:dimension, dimension] = direction
        kkt[dimension, :dimension] = direction
        rhs = np.concatenate(
            (-gradient, np.asarray((0.0,), dtype=float))
        )
        solution = np.linalg.solve(kkt, rhs)
        step = solution[:dimension]
        multiplier = float(solution[dimension])
        constraint_violation = abs(
            float(direction @ step)
        )

    predicted_reduction = -float(
        gradient @ step
        + 0.5 * step @ (hessian @ step)
    )
    return (
        step,
        predicted_reduction,
        multiplier,
        constraint_violation,
    )


def _adaptive_lm(
    evaluator: Any,
    initial: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_nfev: int,
    exact_gauge: bool,
    arguments: argparse.Namespace,
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    started = time.perf_counter()
    initial_value = np.asarray(initial, dtype=float).copy()
    lower_value = np.asarray(lower, dtype=float)
    upper_value = np.asarray(upper, dtype=float)
    if (
        initial_value.ndim != 1
        or lower_value.shape != initial_value.shape
        or upper_value.shape != initial_value.shape
    ):
        raise ValueError(
            "LM initial/bound dimensions are inconsistent"
        )

    gauge_direction = (
        _normalized_scale_gauge(initial_value.size)
        if exact_gauge
        else None
    )
    coordinate = _project_away_exact_gauge(
        initial_value,
        gauge_direction,
    )
    allowed, reason = _trial_coordinate_is_safe(
        coordinate,
        lower_value,
        upper_value,
        arguments,
    )
    if not allowed:
        raise ValueError(
            "initial LM coordinate violates numerical domain: "
            + str(reason)
        )

    evaluation_seconds = 0.0
    linear_solve_seconds = 0.0
    diagnostics = []
    nfev = 0
    linear_solve_count = 0
    rejected_steps = 0
    domain_guard_rejections = 0
    small_reduction_count = 0

    tick = time.perf_counter()
    current = evaluator(coordinate)
    evaluation_seconds += time.perf_counter() - tick
    nfev += 1

    residual = np.asarray(current.residual, dtype=float)
    jacobian = np.asarray(current.jacobian, dtype=float)
    cost = 0.5 * float(residual @ residual)
    hessian = jacobian.T @ jacobian
    diagonal_scale = max(
        float(np.max(np.diag(hessian))),
        1.0,
    )
    damping = (
        float(arguments.lm_initial_damping_relative)
        * diagonal_scale
    )
    trust_radius = float(
        arguments.lm_initial_trust_radius
    )
    maximum_trust_radius = float(
        arguments.lm_maximum_trust_radius
    )
    minimum_trust_radius = max(
        float(arguments.lm_minimum_trust_radius),
        100.0 * np.finfo(float).eps,
    )
    acceptance_ratio = float(
        arguments.lm_acceptance_ratio
    )
    ftol = (
        math.sqrt(np.finfo(float).eps)
        if arguments.ftol is None
        else float(arguments.ftol)
    )
    xtol = (
        math.sqrt(np.finfo(float).eps)
        if arguments.xtol is None
        else float(arguments.xtol)
    )

    status = 0
    success = False
    message = (
        "maximum number of objective evaluations exceeded"
    )

    while nfev < int(max_nfev):
        residual = np.asarray(current.residual, dtype=float)
        jacobian = np.asarray(current.jacobian, dtype=float)
        gradient = jacobian.T @ residual
        projected_gradient = _project_away_exact_gauge(
            gradient,
            gauge_direction,
        )
        gradient_inf = float(
            np.linalg.norm(
                projected_gradient,
                ord=np.inf,
            )
        )
        if gradient_inf <= float(arguments.gtol):
            status = 1
            success = True
            message = (
                "projected gradient tolerance satisfied"
            )
            break

        old_cost = cost
        accepted = False
        inner_attempt = 0
        while (
            inner_attempt < 48
            and nfev < int(max_nfev)
        ):
            inner_attempt += 1
            tick = time.perf_counter()
            try:
                (
                    step,
                    predicted_reduction,
                    kkt_multiplier,
                    constraint_violation,
                ) = _solve_lm_kkt_step(
                    jacobian,
                    residual,
                    damping,
                    gauge_direction,
                )
            except np.linalg.LinAlgError:
                linear_solve_seconds += (
                    time.perf_counter() - tick
                )
                linear_solve_count += 1
                damping *= 10.0
                trust_radius = max(
                    minimum_trust_radius,
                    0.5 * trust_radius,
                )
                continue
            linear_solve_seconds += (
                time.perf_counter() - tick
            )
            linear_solve_count += 1

            step_norm = float(np.linalg.norm(step))
            if (
                not np.isfinite(step_norm)
                or step_norm > trust_radius
                or predicted_reduction <= 0.0
                or not np.isfinite(
                    predicted_reduction
                )
            ):
                damping *= 4.0
                continue

            trial_coordinate = coordinate + step
            if gauge_direction is not None:
                trial_coordinate = (
                    _project_away_exact_gauge(
                        trial_coordinate,
                        gauge_direction,
                    )
                )
            allowed, guard_reason = (
                _trial_coordinate_is_safe(
                    trial_coordinate,
                    lower_value,
                    upper_value,
                    arguments,
                )
            )
            if not allowed:
                domain_guard_rejections += 1
                diagnostics.append(
                    {
                        "accepted": False,
                        "reason": guard_reason,
                        "cost_before": old_cost,
                        "damping": damping,
                        "trust_radius": trust_radius,
                        "step_l2": step_norm,
                    }
                )
                damping *= 4.0
                trust_radius = max(
                    minimum_trust_radius,
                    0.5 * trust_radius,
                )
                continue

            tick = time.perf_counter()
            try:
                trial = evaluator(trial_coordinate)
                trial_residual = np.asarray(
                    trial.residual,
                    dtype=float,
                )
                trial_cost = 0.5 * float(
                    trial_residual @ trial_residual
                )
                if not np.isfinite(trial_cost):
                    raise FloatingPointError(
                        "trial objective is non-finite"
                    )
            except (
                FloatingPointError,
                OverflowError,
                ValueError,
                np.linalg.LinAlgError,
            ) as error:
                evaluation_seconds += (
                    time.perf_counter() - tick
                )
                nfev += 1
                rejected_steps += 1
                diagnostics.append(
                    {
                        "accepted": False,
                        "reason": (
                            "trial_evaluation_failure"
                        ),
                        "error_type": (
                            type(error).__name__
                        ),
                        "cost_before": old_cost,
                        "damping": damping,
                        "trust_radius": trust_radius,
                        "step_l2": step_norm,
                    }
                )
                damping *= 4.0
                trust_radius = max(
                    minimum_trust_radius,
                    0.5 * trust_radius,
                )
                continue
            evaluation_seconds += (
                time.perf_counter() - tick
            )
            nfev += 1

            actual_reduction = old_cost - trial_cost
            ratio = (
                actual_reduction
                / predicted_reduction
                if predicted_reduction > 0.0
                else -np.inf
            )
            accepted = bool(
                actual_reduction > 0.0
                and ratio > acceptance_ratio
            )
            diagnostics.append(
                {
                    "accepted": accepted,
                    "cost_before": old_cost,
                    "cost_after": trial_cost,
                    "actual_reduction": (
                        actual_reduction
                    ),
                    "predicted_reduction": (
                        predicted_reduction
                    ),
                    "reduction_ratio": ratio,
                    "damping": damping,
                    "trust_radius": trust_radius,
                    "step_l2": step_norm,
                    "kkt_multiplier": (
                        kkt_multiplier
                    ),
                    "gauge_step_constraint_abs": (
                        constraint_violation
                    ),
                }
            )

            if not accepted:
                rejected_steps += 1
                damping *= 4.0
                trust_radius = max(
                    minimum_trust_radius,
                    0.5 * trust_radius,
                )
                continue

            coordinate = trial_coordinate
            current = trial
            cost = trial_cost

            if ratio > 0.75:
                damping = max(
                    damping / 3.0,
                    np.finfo(float).tiny,
                )
                trust_radius = min(
                    maximum_trust_radius,
                    max(
                        trust_radius,
                        2.0
                        * max(
                            step_norm,
                            minimum_trust_radius,
                        ),
                    ),
                )
            elif ratio < 0.25:
                damping *= 4.0
                trust_radius = max(
                    minimum_trust_radius,
                    0.5 * trust_radius,
                )

            relative_reduction = (
                actual_reduction
                / max(old_cost, 1.0)
            )
            if relative_reduction <= ftol:
                small_reduction_count += 1
            else:
                small_reduction_count = 0

            if step_norm <= xtol * (
                xtol
                + float(np.linalg.norm(coordinate))
            ):
                status = 3
                success = True
                message = "step tolerance satisfied"
            elif small_reduction_count >= 3:
                status = 2
                success = True
                message = (
                    "three consecutive accepted steps "
                    "satisfied relative objective-"
                    "reduction tolerance"
                )
            break

        if success:
            break
        if not accepted:
            if trust_radius <= minimum_trust_radius:
                status = -2
                message = (
                    "trust region collapsed before an "
                    "acceptable step could be found"
                )
                break
            if nfev >= int(max_nfev):
                break

    final_residual = np.asarray(
        current.residual,
        dtype=float,
    )
    final_jacobian = np.asarray(
        current.jacobian,
        dtype=float,
    )
    final_gradient = (
        final_jacobian.T @ final_residual
    )
    projected_final_gradient = (
        _project_away_exact_gauge(
            final_gradient,
            gauge_direction,
        )
    )
    gauge_coordinate = (
        None
        if gauge_direction is None
        else float(gauge_direction @ coordinate)
    )

    payload = {
        "method": (
            "adaptive Levenberg-Marquardt / "
            "trust region with KKT hard constraint "
            "for analytically known common-scale gauge"
        ),
        "cost": 0.5
        * float(final_residual @ final_residual),
        "optimality": float(
            np.linalg.norm(
                projected_final_gradient,
                ord=np.inf,
            )
        ),
        "nfev": int(nfev),
        "njev": int(nfev),
        "status": int(status),
        "success": bool(success),
        "message": message,
        "coordinate_step_l2": float(
            np.linalg.norm(
                coordinate - initial_value
            )
        ),
        "gradient_l2": float(
            np.linalg.norm(
                projected_final_gradient
            )
        ),
        "gradient_inf": float(
            np.linalg.norm(
                projected_final_gradient,
                ord=np.inf,
            )
        ),
        "jacobian_spectrum": _jacobian_spectrum(
            final_jacobian
        ),
        "known_exact_gauge_handling": {
            "hard_constraint_applied": bool(
                gauge_direction is not None
            ),
            "constraint": (
                "v_scale^T step = 0"
                if gauge_direction is not None
                else None
            ),
            "normalized_direction": (
                gauge_direction
            ),
            "final_coordinate_component": (
                gauge_coordinate
            ),
            "maximum_recorded_step_constraint_abs": (
                max(
                    (
                        float(
                            item.get(
                                "gauge_step_constraint_abs",
                                0.0,
                            )
                        )
                        for item in diagnostics
                    ),
                    default=0.0,
                )
            ),
        },
        "near_ridge_handling": {
            "hard_singular_value_threshold": None,
            "unknown_weak_modes_removed": False,
            "mechanism": (
                "continuous LM damping plus trust "
                "radius; scientific ridge interpretation "
                "is deferred to raw-Jacobian output"
            ),
        },
        "numerical_safety": {
            "physical_box_bounds_active": False,
            "broad_trial_coordinate_guard_abs": (
                float(
                    arguments.numeric_coordinate_guard
                )
            ),
            "domain_guard_rejections": int(
                domain_guard_rejections
            ),
            "rejected_steps": int(
                rejected_steps
            ),
        },
        "lm": {
            "initial_damping_relative": float(
                arguments.lm_initial_damping_relative
            ),
            "final_damping": float(damping),
            "initial_trust_radius": float(
                arguments.lm_initial_trust_radius
            ),
            "final_trust_radius": float(
                trust_radius
            ),
            "maximum_trust_radius": (
                maximum_trust_radius
            ),
            "minimum_trust_radius": (
                minimum_trust_radius
            ),
            "acceptance_ratio": acceptance_ratio,
            "effective_ftol": ftol,
            "effective_xtol": xtol,
            "gtol": float(arguments.gtol),
            "trial_history": diagnostics,
        },
        "timing": {
            "elapsed_seconds": float(
                time.perf_counter() - started
            ),
            "objective_evaluation_seconds": float(
                evaluation_seconds
            ),
            "linear_kkt_solve_seconds": float(
                linear_solve_seconds
            ),
            "linear_kkt_solve_count": int(
                linear_solve_count
            ),
        },
        "ridge_threshold_used": None,
    }
    return coordinate.copy(), current, payload


def _solve_smooth(
    problem: DimensionlessDynamicsProblem,
    initial: np.ndarray,
    width_fraction: float,
    lower: np.ndarray,
    upper: np.ndarray,
    arguments: argparse.Namespace,
) -> tuple[
    np.ndarray,
    JointEvaluation,
    Mapping[str, Any],
]:
    evaluator = lambda value: problem.evaluate_smooth(
        value,
        width_fraction,
    )
    coordinate, evaluation, optimizer = (
        _adaptive_lm(
            evaluator,
            np.asarray(initial, dtype=float),
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
            int(arguments.smooth_max_nfev),
            problem.exact_common_scale_symmetry,
            arguments,
        )
    )
    tick = time.perf_counter()
    gauge = _gauge_diagnostics(
        evaluator,
        coordinate,
        evaluation,
        problem.exact_common_scale_symmetry,
    )
    optimizer = dict(optimizer)
    optimizer["common_scale_gauge"] = gauge
    optimizer["timing"] = dict(
        optimizer["timing"]
    )
    optimizer["timing"][
        "gauge_diagnostics_seconds"
    ] = float(time.perf_counter() - tick)
    return coordinate, evaluation, optimizer


def _solve_strict(
    problem: DimensionlessDynamicsProblem,
    initial: np.ndarray,
    rotor_delay_seconds: float,
    gimbal_delay_seconds: float,
    lower: np.ndarray,
    upper: np.ndarray,
    arguments: argparse.Namespace,
) -> Solution:
    evaluator = lambda value: problem.evaluate_strict(
        value,
        rotor_delay_seconds,
        gimbal_delay_seconds,
    )
    coordinate, evaluation, optimizer = (
        _adaptive_lm(
            evaluator,
            np.asarray(initial, dtype=float),
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
            int(arguments.strict_max_nfev),
            problem.exact_common_scale_symmetry,
            arguments,
        )
    )
    tick = time.perf_counter()
    gauge = _gauge_diagnostics(
        evaluator,
        coordinate,
        evaluation,
        problem.exact_common_scale_symmetry,
    )
    optimizer = dict(optimizer)
    optimizer["common_scale_gauge"] = gauge
    optimizer["timing"] = dict(
        optimizer["timing"]
    )
    optimizer["timing"][
        "gauge_diagnostics_seconds"
    ] = float(time.perf_counter() - tick)
    return Solution(
        physical_coordinate=coordinate,
        rotor_delay_seconds=float(
            rotor_delay_seconds
        ),
        gimbal_delay_seconds=float(
            gimbal_delay_seconds
        ),
        evaluation=evaluation,
        optimizer=optimizer,
    )

def _median_command_period(
    bags: Sequence[BagData],
    channel: str,
) -> float:
    values = []
    for bag in bags:
        history = (
            bag.rotor_history
            if channel == "rotor"
            else bag.gimbal_history
        )
        if history.times.size > 1:
            differences = np.diff(history.times)
            positive = differences[differences > 0.0]
            if positive.size:
                values.append(float(np.median(positive)))
    if not values:
        raise ValueError(
            "{} command has no positive timestamp interval".format(
                channel
            )
        )
    return float(np.median(values))


def _strict_screen(
    problem: DimensionlessDynamicsProblem,
    physical: np.ndarray,
    center_rotor: float,
    center_gimbal: float,
    rotor_period: float,
    gimbal_period: float,
    delay_bounds: Sequence[float],
) -> dict[str, Any]:
    started = time.perf_counter()
    lower, upper = map(float, delay_bounds)
    rotor_values = {
        round(
            float(
                np.clip(
                    center_rotor + offset,
                    lower,
                    upper,
                )
            ),
            12,
        )
        for offset in (
            -rotor_period,
            0.0,
            rotor_period,
        )
    }
    gimbal_values = {
        round(
            float(
                np.clip(
                    center_gimbal + offset,
                    lower,
                    upper,
                )
            ),
            12,
        )
        for offset in (
            -gimbal_period,
            0.0,
            gimbal_period,
        )
    }
    costs: dict[tuple[float, float], float] = {}
    expansions = []

    def evaluate_new() -> None:
        for rotor in sorted(rotor_values):
            for gimbal in sorted(gimbal_values):
                key = (rotor, gimbal)
                if key in costs:
                    continue
                evaluation = problem.evaluate_strict(
                    physical,
                    rotor,
                    gimbal,
                )
                costs[key] = 0.5 * float(
                    evaluation.residual
                    @ evaluation.residual
                )

    for _iteration in range(256):
        evaluate_new()
        best = min(
            costs,
            key=lambda key: (
                costs[key],
                key[0],
                key[1],
            ),
        )
        rotor_order = sorted(rotor_values)
        gimbal_order = sorted(gimbal_values)
        additions = []

        if (
            best[0] == rotor_order[0]
            and best[0] > lower
        ):
            value = round(
                max(
                    lower,
                    best[0] - rotor_period,
                ),
                12,
            )
            if value not in rotor_values:
                rotor_values.add(value)
                additions.append(
                    {
                        "axis": "rotor",
                        "delay_seconds": value,
                    }
                )
        elif (
            best[0] == rotor_order[-1]
            and best[0] < upper
        ):
            value = round(
                min(
                    upper,
                    best[0] + rotor_period,
                ),
                12,
            )
            if value not in rotor_values:
                rotor_values.add(value)
                additions.append(
                    {
                        "axis": "rotor",
                        "delay_seconds": value,
                    }
                )

        if (
            best[1] == gimbal_order[0]
            and best[1] > lower
        ):
            value = round(
                max(
                    lower,
                    best[1] - gimbal_period,
                ),
                12,
            )
            if value not in gimbal_values:
                gimbal_values.add(value)
                additions.append(
                    {
                        "axis": "gimbal",
                        "delay_seconds": value,
                    }
                )
        elif (
            best[1] == gimbal_order[-1]
            and best[1] < upper
        ):
            value = round(
                min(
                    upper,
                    best[1] + gimbal_period,
                ),
                12,
            )
            if value not in gimbal_values:
                gimbal_values.add(value)
                additions.append(
                    {
                        "axis": "gimbal",
                        "delay_seconds": value,
                    }
                )

        if not additions:
            break
        expansions.append(
            {
                "best_before_expansion": {
                    "rotor_delay_seconds": best[0],
                    "gimbal_delay_seconds": best[1],
                    "cost": costs[best],
                },
                "added": additions,
            }
        )
    else:
        raise RuntimeError(
            "strict lag screen did not terminate"
        )

    evaluate_new()
    best = min(
        costs,
        key=lambda key: (
            costs[key],
            key[0],
            key[1],
        ),
    )
    return {
        "elapsed_seconds": float(
            time.perf_counter() - started
        ),
        "best_pair": best,
        "best_cost": costs[best],
        "rotor_range_seconds": (
            min(rotor_values),
            max(rotor_values),
        ),
        "gimbal_range_seconds": (
            min(gimbal_values),
            max(gimbal_values),
        ),
        "expansions": expansions,
        "candidates": [
            {
                "rotor_delay_seconds": rotor,
                "gimbal_delay_seconds": gimbal,
                "cost": costs[(rotor, gimbal)],
                "selected": (rotor, gimbal) == best,
            }
            for rotor in sorted(rotor_values)
            for gimbal in sorted(gimbal_values)
        ],
    }


def _lag_search_impl(
    problem: DimensionlessDynamicsProblem,
    arguments: argparse.Namespace,
) -> tuple[Solution, dict[str, Any]]:
    physical_lower, physical_upper = (
        _physical_bounds(arguments)
    )
    rotor_period = _median_command_period(
        problem.bags,
        "rotor",
    )
    gimbal_period = _median_command_period(
        problem.bags,
        "gimbal",
    )
    common_initial = (
        None
        if arguments.initial_delay is None
        else float(arguments.initial_delay)
    )
    rotor_initial = (
        (
            rotor_period
            if common_initial is None
            else common_initial
        )
        if arguments.initial_rotor_delay is None
        else float(arguments.initial_rotor_delay)
    )
    gimbal_initial = (
        (
            gimbal_period
            if common_initial is None
            else common_initial
        )
        if arguments.initial_gimbal_delay is None
        else float(arguments.initial_gimbal_delay)
    )
    delay_lower, delay_upper = map(
        float,
        arguments.delay_bounds,
    )

    physical = np.zeros(
        PHYSICAL_DIMENSION,
        dtype=float,
    )

    if arguments.skip_lag_search:
        solution = _solve_strict(
            problem,
            physical,
            rotor_initial,
            gimbal_initial,
            physical_lower,
            physical_upper,
            arguments,
        )
        return solution, {
            "mode": "fixed_data_derived_initial_lags",
            "rotor_period_seconds": rotor_period,
            "gimbal_period_seconds": gimbal_period,
            "selected_rotor_delay_seconds": rotor_initial,
            "selected_gimbal_delay_seconds": gimbal_initial,
        }

    coordinate = np.concatenate(
        (
            physical,
            np.asarray(
                (
                    rotor_initial
                    / problem.scales.time_s,
                    gimbal_initial
                    / problem.scales.time_s,
                ),
                dtype=float,
            ),
        )
    )
    lower = np.concatenate(
        (
            physical_lower,
            np.asarray(
                (
                    delay_lower
                    / problem.scales.time_s,
                    delay_lower
                    / problem.scales.time_s,
                )
            ),
        )
    )
    upper = np.concatenate(
        (
            physical_upper,
            np.asarray(
                (
                    delay_upper
                    / problem.scales.time_s,
                    delay_upper
                    / problem.scales.time_s,
                )
            ),
        )
    )

    smooth_stages = []
    for width in arguments.smoothstep_width_fractions:
        stage_initial = coordinate.copy()
        (
            coordinate,
            evaluation,
            optimizer,
        ) = _solve_smooth(
            problem,
            coordinate,
            float(width),
            lower,
            upper,
            arguments,
        )
        smooth_stages.append(
            {
                "half_width_period_multiplier": float(width),
                "initial_coordinate": stage_initial,
                "final_coordinate": coordinate,
                "objective_cost": 0.5 * float(
                    evaluation.residual
                    @ evaluation.residual
                ),
                "rotor_delay_seconds": float(
                    coordinate[ROTOR_DELAY_INDEX]
                    * problem.scales.time_s
                ),
                "gimbal_delay_seconds": float(
                    coordinate[GIMBAL_DELAY_INDEX]
                    * problem.scales.time_s
                ),
                "optimizer": optimizer,
            }
        )

    physical = coordinate[:PHYSICAL_DIMENSION].copy()
    rotor = float(
        coordinate[ROTOR_DELAY_INDEX]
        * problem.scales.time_s
    )
    gimbal = float(
        coordinate[GIMBAL_DELAY_INDEX]
        * problem.scales.time_s
    )

    iterations = []
    visited: set[tuple[float, float]] = set()
    solutions = []
    selected: Optional[Solution] = None
    termination = None

    for iteration in range(
        int(arguments.strict_alternations)
    ):
        screen = _strict_screen(
            problem,
            physical,
            rotor,
            gimbal,
            rotor_period,
            gimbal_period,
            arguments.delay_bounds,
        )
        best_rotor, best_gimbal = screen["best_pair"]
        pair = (
            round(float(best_rotor), 12),
            round(float(best_gimbal), 12),
        )
        if pair in visited:
            selected = min(
                solutions,
                key=lambda item: 0.5
                * float(
                    item.evaluation.residual
                    @ item.evaluation.residual
                ),
            )
            termination = "cycle_guard"
            break
        visited.add(pair)

        solution = _solve_strict(
            problem,
            physical,
            best_rotor,
            best_gimbal,
            physical_lower,
            physical_upper,
            arguments,
        )
        solutions.append(solution)
        verify = _strict_screen(
            problem,
            solution.physical_coordinate,
            best_rotor,
            best_gimbal,
            rotor_period,
            gimbal_period,
            arguments.delay_bounds,
        )
        iterations.append(
            {
                "iteration": iteration + 1,
                "screening": screen,
                "solution": {
                    "physical_coordinate": (
                        solution.physical_coordinate
                    ),
                    "rotor_delay_seconds": (
                        solution.rotor_delay_seconds
                    ),
                    "gimbal_delay_seconds": (
                        solution.gimbal_delay_seconds
                    ),
                    "objective_cost": 0.5
                    * float(
                        solution.evaluation.residual
                        @ solution.evaluation.residual
                    ),
                    "optimizer": solution.optimizer,
                },
                "post_refinement_screening": verify,
            }
        )
        next_rotor, next_gimbal = (
            verify["best_pair"]
        )
        if (
            np.isclose(
                next_rotor,
                best_rotor,
                atol=5.0e-13,
                rtol=0.0,
            )
            and np.isclose(
                next_gimbal,
                best_gimbal,
                atol=5.0e-13,
                rtol=0.0,
            )
        ):
            selected = solution
            termination = "fixed_point"
            break

        physical = solution.physical_coordinate.copy()
        rotor = float(next_rotor)
        gimbal = float(next_gimbal)

    if selected is None:
        if not solutions:
            raise RuntimeError(
                "strict lag refinement produced no solution"
            )
        selected = min(
            solutions,
            key=lambda item: 0.5
            * float(
                item.evaluation.residual
                @ item.evaluation.residual
            ),
        )
        termination = "iteration_limit_best_solution"

    return selected, {
        "mode": "smooth_then_strict_split_lag",
        "rotor_period_seconds": rotor_period,
        "gimbal_period_seconds": gimbal_period,
        "initial_rotor_delay_seconds": rotor_initial,
        "initial_gimbal_delay_seconds": gimbal_initial,
        "smooth_stages": smooth_stages,
        "strict_iterations": iterations,
        "termination": termination,
        "selected_rotor_delay_seconds": (
            selected.rotor_delay_seconds
        ),
        "selected_gimbal_delay_seconds": (
            selected.gimbal_delay_seconds
        ),
    }


def _lag_search(
    problem: DimensionlessDynamicsProblem,
    arguments: argparse.Namespace,
) -> tuple[Solution, dict[str, Any]]:
    started = time.perf_counter()
    solution, payload = _lag_search_impl(
        problem,
        arguments,
    )
    result = dict(payload)
    result["elapsed_seconds"] = float(
        time.perf_counter() - started
    )
    smooth_seconds = [
        float(
            item.get("optimizer", {})
            .get("timing", {})
            .get("elapsed_seconds", 0.0)
        )
        for item in result.get("smooth_stages", ())
    ]
    strict_iterations = result.get(
        "strict_iterations",
        (),
    )
    result["timing"] = {
        "elapsed_seconds": result["elapsed_seconds"],
        "smooth_optimizer_seconds": float(
            sum(smooth_seconds)
        ),
        "strict_screen_seconds": float(
            sum(
                float(
                    item.get("screening", {}).get(
                        "elapsed_seconds",
                        0.0,
                    )
                )
                + float(
                    item.get(
                        "post_refinement_screening",
                        {},
                    ).get(
                        "elapsed_seconds",
                        0.0,
                    )
                )
                for item in strict_iterations
            )
        ),
        "strict_optimizer_seconds": float(
            sum(
                float(
                    item.get("solution", {})
                    .get("optimizer", {})
                    .get("timing", {})
                    .get("elapsed_seconds", 0.0)
                )
                for item in strict_iterations
            )
        ),
    }
    return solution, result

def _pseudo_whitener(
    covariance: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    value = np.asarray(covariance, dtype=float)
    value = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    largest = max(
        float(np.max(eigenvalues)),
        0.0,
    )
    tolerance = (
        value.shape[0]
        * np.finfo(float).eps
        * max(1.0, largest)
    )
    positive = eigenvalues > tolerance
    inverse_sqrt = np.zeros_like(eigenvalues)
    inverse_sqrt[positive] = (
        1.0 / np.sqrt(eigenvalues[positive])
    )
    whitener = (
        eigenvectors
        @ np.diag(inverse_sqrt)
        @ eigenvectors.T
    )
    return whitener, {
        "eigenvalues": eigenvalues,
        "numerical_tolerance": tolerance,
        "numerical_rank": int(
            np.count_nonzero(positive)
        ),
        "uses_pseudoinverse": True,
    }


def _confidence_payload(
    problem: DimensionlessDynamicsProblem,
    solution: Solution,
    prior: Optional[GaussianPhysicalPrior],
) -> dict[str, Any]:
    jacobian_blocks = []
    residual_blocks = []
    bag_models = []

    for bag, evaluation in zip(
        problem.bags,
        solution.evaluation.bag_evaluations,
    ):
        residual = np.asarray(
            evaluation.acceleration_residual,
            dtype=float,
        )
        jacobian = np.asarray(
            evaluation.acceleration_jacobian,
            dtype=float,
        )
        mean = np.mean(residual, axis=0)
        covariance = (
            np.cov(
                residual,
                rowvar=False,
                ddof=1,
            )
            if residual.shape[0] > 1
            else np.eye(6)
        )
        whitener, covariance_diagnostics = (
            _pseudo_whitener(covariance)
        )
        centered = residual - mean[None, :]
        whitened_residual = np.einsum(
            "ij,nj->ni",
            whitener,
            centered,
        )
        whitened_jacobian = np.einsum(
            "ij,njk->nik",
            whitener,
            jacobian,
        )
        weight_scale = math.sqrt(
            bag.normalized_weight
        )
        residual_blocks.append(
            weight_scale
            * whitened_residual.reshape(-1)
        )
        jacobian_blocks.append(
            weight_scale
            * whitened_jacobian.reshape(
                -1,
                PHYSICAL_DIMENSION,
            )
        )
        bag_models.append(
            {
                "id": bag.specification.bag_id,
                "sample_count": int(residual.shape[0]),
                "mean_dimensionless_generalized_acceleration": mean,
                "covariance_dimensionless_generalized_acceleration": (
                    covariance
                ),
                "covariance_diagnostics": (
                    covariance_diagnostics
                ),
            }
        )

    stacked_residual = np.concatenate(
        residual_blocks
    )
    stacked_jacobian = np.vstack(
        jacobian_blocks
    )
    data_information = (
        stacked_jacobian.T @ stacked_jacobian
    )
    data_information_vector = -(
        stacked_jacobian.T @ stacked_residual
    )
    spectrum = _jacobian_spectrum(
        stacked_jacobian
    )

    direction = COMMON_SCALE_DIRECTION.copy()
    direction /= np.linalg.norm(direction)
    data_gauge_relative = float(
        np.linalg.norm(
            stacked_jacobian @ direction
        )
        / max(
            np.linalg.norm(stacked_jacobian),
            np.finfo(float).tiny,
        )
    )
    weakest_alignment = None
    if spectrum["right_singular_vectors"].size:
        weakest_alignment = float(
            abs(
                spectrum["right_singular_vectors"][-1]
                @ direction
            )
        )

    payload: dict[str, Any] = {
        "schema": SCHEMA + "/confidence",
        "likelihood_residual": (
            "dimensionless generalized acceleration residual; "
            "not absolute wrench"
        ),
        "bag_models": bag_models,
        "data_information_matrix": data_information,
        "data_information_vector": (
            data_information_vector
        ),
        "data_spectrum": spectrum,
        "common_scale_gauge": {
            "normalized_direction": direction,
            "relative_jacobian_image_norm": (
                data_gauge_relative
            ),
            "weakest_mode_alignment": (
                weakest_alignment
            ),
        },
        "prior_used_in_deterministic_fit": False,
        "posterior_available": prior is not None,
    }

    if prior is not None:
        decoded, parameter_jacobian = (
            problem.parameterization.decode_with_jacobian(
                solution.physical_coordinate,
                solution.rotor_delay_seconds,
            )
        )
        physical_reference = physical_parameter_vector(
            decoded.parameters
        )
        physical_jacobian = (
            physical_parameter_jacobian(
                parameter_jacobian
            )
        )
        inverse_variance = 1.0 / (
            prior.std**2
        )
        prior_precision = (
            physical_jacobian.T
            @ (
                inverse_variance[:, None]
                * physical_jacobian
            )
        )
        prior_information_vector = (
            physical_jacobian.T
            @ (
                inverse_variance
                * (
                    prior.mean
                    - physical_reference
                )
            )
        )
        posterior_precision = (
            data_information
            + prior_precision
        )
        posterior_precision = 0.5 * (
            posterior_precision
            + posterior_precision.T
        )
        posterior_information_vector = (
            data_information_vector
            + prior_information_vector
        )
        posterior_covariance = np.linalg.pinv(
            posterior_precision
        )
        posterior_covariance = 0.5 * (
            posterior_covariance
            + posterior_covariance.T
        )
        posterior_delta = (
            posterior_covariance
            @ posterior_information_vector
        )
        posterior_coordinate_mean = (
            solution.physical_coordinate
            + posterior_delta
        )
        posterior_physical_mean_linearized = (
            physical_reference
            + physical_jacobian
            @ posterior_delta
        )
        posterior_physical_covariance = (
            physical_jacobian
            @ posterior_covariance
            @ physical_jacobian.T
        )
        posterior_physical_covariance = 0.5 * (
            posterior_physical_covariance
            + posterior_physical_covariance.T
        )
        payload["posterior"] = {
            "prior_source_path": prior.source_path,
            "prior_vector_order": (
                PHYSICAL_VALUE_NAMES
            ),
            "prior_mean": prior.mean,
            "prior_std": prior.std,
            "prior_precision_in_dimensionless_chart": (
                prior_precision
            ),
            "posterior_precision": (
                posterior_precision
            ),
            "posterior_covariance": (
                posterior_covariance
            ),
            "posterior_delta_mean": (
                posterior_delta
            ),
            "posterior_coordinate_mean": (
                posterior_coordinate_mean
            ),
            "selected_physical_vector": (
                physical_reference
            ),
            "posterior_physical_mean_linearized": (
                posterior_physical_mean_linearized
            ),
            "posterior_physical_covariance_linearized": (
                posterior_physical_covariance
            ),
            "posterior_physical_std_linearized": (
                np.sqrt(
                    np.maximum(
                        np.diag(
                            posterior_physical_covariance
                        ),
                        0.0,
                    )
                )
            ),
        }

    return payload


def _residual_wrench_statistics(
    evaluation: BagEvaluation,
    scales: ReferenceScales,
) -> dict[str, Any]:
    raw = np.asarray(
        evaluation.residual_body_wrench,
        dtype=float,
    )
    scale = np.asarray(
        (
            1.0 / scales.force_n,
            1.0 / scales.force_n,
            1.0 / scales.force_n,
            1.0 / scales.torque_nm,
            1.0 / scales.torque_nm,
            1.0 / scales.torque_nm,
        ),
        dtype=float,
    )
    dimensionless = raw * scale[None, :]
    return {
        "mean_physical": np.mean(raw, axis=0),
        "std_physical": np.std(
            raw,
            axis=0,
            ddof=1,
        ),
        "rms_physical": np.sqrt(
            np.mean(raw * raw, axis=0)
        ),
        "mean_dimensionless": np.mean(
            dimensionless,
            axis=0,
        ),
        "std_dimensionless": np.std(
            dimensionless,
            axis=0,
            ddof=1,
        ),
        "rms_dimensionless": np.sqrt(
            np.mean(
                dimensionless * dimensionless,
                axis=0,
            )
        ),
    }


def _parameter_payload(
    parameters: VehicleParameters,
) -> dict[str, Any]:
    inertia = np.asarray(parameters.inertia, dtype=float)
    return {
        "mass_kg": float(parameters.mass),
        "inertia_kg_m2": inertia,
        "inertia_principal_moments_kg_m2": (
            np.linalg.eigvalsh(inertia)
        ),
        "cog_position_body_m": np.asarray(
            parameters.cog_offset,
            dtype=float,
        ),
        "force_effectiveness": np.asarray(
            parameters.force_effectiveness,
            dtype=float,
        ),
        "torque_effectiveness": np.asarray(
            parameters.torque_effectiveness,
            dtype=float,
        ),
        "linear_drag": np.asarray(
            parameters.linear_drag,
            dtype=float,
        ),
        "angular_drag": np.asarray(
            parameters.angular_drag,
            dtype=float,
        ),
    }


def _experiment_namespace(
    arguments: argparse.Namespace,
) -> str:
    config = arguments.config.expanduser().resolve()
    model = (
        arguments.vehicle_model_json
        .expanduser()
        .resolve()
    )
    prior = (
        None
        if arguments.prior_json is None
        else arguments.prior_json.expanduser().resolve()
    )
    digest = hashlib.sha256()
    for label, path in (
        ("config", config),
        ("vehicle", model),
        ("prior", prior),
    ):
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        if path is None:
            digest.update(b"<none>")
        else:
            digest.update(
                str(path).encode("utf-8")
            )
            digest.update(b"\0")
            digest.update(path.read_bytes())
        digest.update(b"\0")
    stem = "".join(
        character
        if (
            character.isalnum()
            or character in ("-", "_")
        )
        else "_"
        for character in config.stem
    ).strip("_")
    return "{}_{}".format(
        stem or "experiment",
        digest.hexdigest()[:12],
    )


def _write_text_pdf(
    path: Path,
    title: str,
    text: str,
) -> None:
    lines = text.splitlines()
    page_size = 56
    with PdfPages(path) as pdf:
        for start in range(
            0,
            max(len(lines), 1),
            page_size,
        ):
            figure = plt.figure(
                figsize=(8.3, 11.7),
                constrained_layout=True,
            )
            figure.text(
                0.04,
                0.98,
                title
                + "\n\n"
                + "\n".join(
                    lines[start : start + page_size]
                ),
                va="top",
                ha="left",
                family="monospace",
                fontsize=7.4,
            )
            pdf.savefig(figure)
            plt.close(figure)


def _write_savgol_fit_pdf(
    path: Path,
    bag: BagData,
) -> None:
    raw_time = np.asarray(
        bag.flight.pose.times,
        dtype=float,
    )
    raw_position = np.asarray(
        bag.flight.pose.positions,
        dtype=float,
    )
    raw_orientation = np.asarray(
        bag.flight.pose.orientations_xyzw,
        dtype=float,
    )
    fitted = bag.selection.spline.evaluate(raw_time)
    fitted_sensor_rotation = np.einsum(
        "nij,jk->nik",
        np.asarray(
            fitted.body_rotation,
            dtype=float,
        ),
        np.asarray(
            bag.direct_problem
            .pose_body_to_sensor_rotation,
            dtype=float,
        ),
    )
    fitted_orientation = np.asarray(
        [
            matrix_to_quaternion(rotation)
            for rotation in fitted_sensor_rotation
        ],
        dtype=float,
    )
    raw_rpy = baseline._rpy_series(
        raw_orientation
    )
    fitted_rpy = baseline._rpy_series(
        fitted_orientation
    )
    relative_time = raw_time - raw_time[0]

    with PdfPages(path) as pdf:
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        for component, axis in enumerate(axes):
            axis.plot(
                relative_time,
                raw_position[:, component],
                label="raw mocap",
            )
            axis.plot(
                relative_time,
                fitted.sensor_position[
                    :, component
                ],
                linestyle="--",
                label="geometric SG",
            )
            axis.set_ylabel(
                ("x [m]", "y [m]", "z [m]")[
                    component
                ]
            )
            axis.grid(True, alpha=0.25)
        axes[0].legend(loc="best")
        axes[-1].set_xlabel(
            "time from raw-pose start [s]"
        )
        figure.suptitle(
            "Raw mocap and geometric-SG position"
        )
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(
            3,
            1,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        for component, axis in enumerate(axes):
            axis.plot(
                relative_time,
                raw_rpy[:, component],
                label="raw mocap",
            )
            axis.plot(
                relative_time,
                fitted_rpy[:, component],
                linestyle="--",
                label="geometric SG",
            )
            axis.set_ylabel(
                (
                    "roll [rad]",
                    "pitch [rad]",
                    "yaw [rad]",
                )[component]
            )
            axis.grid(True, alpha=0.25)
        axes[0].legend(loc="best")
        axes[-1].set_xlabel(
            "time from raw-pose start [s]"
        )
        figure.suptitle(
            "Raw mocap and geometric-SG orientation"
        )
        pdf.savefig(figure)
        plt.close(figure)

        center_time = bag.time - raw_time[0]
        figure, axes = plt.subplots(
            3,
            2,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        for component in range(3):
            axes[component, 0].plot(
                center_time,
                bag.kinematics
                .sensor_velocity_world[
                    :, component
                ],
                label="velocity",
            )
            axes[component, 0].plot(
                center_time,
                bag.kinematics
                .sensor_acceleration_world[
                    :, component
                ],
                label="acceleration",
            )
            axes[component, 1].plot(
                center_time,
                bag.kinematics
                .body_angular_velocity[
                    :, component
                ],
                label="omega",
            )
            axes[component, 1].plot(
                center_time,
                bag.kinematics
                .body_angular_acceleration[
                    :, component
                ],
                label="alpha",
            )
            axes[component, 0].grid(
                True,
                alpha=0.25,
            )
            axes[component, 1].grid(
                True,
                alpha=0.25,
            )
        axes[0, 0].legend(loc="best")
        axes[0, 1].legend(loc="best")
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        figure.suptitle(
            "Geometric-SG derivatives"
        )
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(
            2,
            1,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        axes[0].plot(
            center_time,
            bag.kinematics.window_sample_count,
        )
        axes[0].set_ylabel(
            "samples / SG window"
        )
        axes[1].semilogy(
            center_time,
            bag.kinematics
            .position_fit_condition_number,
            label="position",
        )
        axes[1].semilogy(
            center_time,
            bag.kinematics
            .rotation_fit_condition_number,
            label="rotation",
        )
        axes[1].set_ylabel(
            "local fit condition"
        )
        axes[1].set_xlabel("time [s]")
        axes[1].legend(loc="best")
        for axis in axes:
            axis.grid(True, alpha=0.25)
        figure.suptitle(
            "SG support / conditioning"
        )
        pdf.savefig(figure)
        plt.close(figure)


def _diagnostic_forward_rollout(
    bag: BagData,
    parameters: VehicleParameters,
    rotor_delay_seconds: float,
    gimbal_delay_seconds: float,
    external_body_wrench: Optional[Any] = None,
) -> Any:
    time_axis = np.asarray(
        bag.time,
        dtype=float,
    )
    if time_axis.size < 2:
        raise ValueError(
            "diagnostic rollout requires "
            "at least two SG times"
        )
    rotation = np.asarray(
        bag.kinematics.body_rotation[0],
        dtype=float,
    )
    omega = np.asarray(
        bag.kinematics.body_angular_velocity[0],
        dtype=float,
    )
    pose_lever = (
        np.asarray(
            bag.direct_problem
            .pose_sensor_position,
            dtype=float,
        )
        - np.asarray(
            parameters.cog_offset,
            dtype=float,
        )
    )
    rigid = RigidBodyState(
        position=np.asarray(
            bag.kinematics.sensor_position[0],
            dtype=float,
        )
        - rotation @ pose_lever,
        orientation_xyzw=matrix_to_quaternion(
            rotation
        ),
        linear_velocity=np.asarray(
            bag.kinematics
            .sensor_velocity_world[0],
            dtype=float,
        )
        - rotation
        @ np.cross(
            omega,
            pose_lever,
        ),
        angular_velocity=omega,
    )
    plant = FullSixDofPlant(
        parameters,
        bag.direct_problem.geometry,
        model_discrepancy_wrench=(
            external_body_wrench
        ),
    )
    actuator_parameters = ActuatorParameters(
        thrust_time_constant=0.0,
        gimbal_time_constant=0.0,
        delay=0.0,
    )

    count = time_axis.size
    arrays = {
        "sensor_position": np.empty(
            (count, 3)
        ),
        "sensor_orientation_xyzw": np.empty(
            (count, 4)
        ),
        "sensor_velocity_world": np.empty(
            (count, 3)
        ),
        "angular_velocity_sensor": np.empty(
            (count, 3)
        ),
        "specific_force_sensor": np.empty(
            (count, 3)
        ),
        "cog_position": np.empty(
            (count, 3)
        ),
        "cog_velocity_world": np.empty(
            (count, 3)
        ),
        "actuator_thrust": np.empty(
            (count, 4)
        ),
        "actuator_gimbal": np.empty(
            (count, 4)
        ),
    }

    def actuator_at(
        query_time: float,
    ) -> ActuatorState:
        thrust = np.clip(
            bag.rotor_history.exact_zoh(
                query_time,
                rotor_delay_seconds,
            ),
            actuator_parameters.minimum_thrust,
            actuator_parameters.maximum_thrust,
        )
        if np.isclose(
            query_time,
            float(time_axis[0]),
            atol=1.0e-12,
            rtol=0.0,
        ):
            gimbal = np.asarray(
                bag.initial_gimbal,
                dtype=float,
            )
        else:
            gimbal = (
                bag.gimbal_history.exact_zoh(
                    query_time,
                    gimbal_delay_seconds,
                )
            )
        return ActuatorState(
            thrust=np.asarray(
                thrust,
                dtype=float,
            ),
            gimbal_angle=np.asarray(
                gimbal,
                dtype=float,
            ),
        )

    def store(index: int) -> None:
        body_rotation = quaternion_to_matrix(
            rigid.orientation_xyzw
        )
        actuator = actuator_at(
            float(time_axis[index])
        )
        velocity_lever = (
            np.asarray(
                bag.direct_problem
                .velocity_sensor_position,
                dtype=float,
            )
            - np.asarray(
                parameters.cog_offset,
                dtype=float,
            )
        )
        imu_lever = (
            np.asarray(
                bag.direct_problem
                .imu_sensor_position,
                dtype=float,
            )
            - np.asarray(
                parameters.cog_offset,
                dtype=float,
            )
        )
        wrench = plant.total_body_wrench(
            float(time_axis[index]),
            rigid,
            actuator,
        )
        angular_acceleration = np.linalg.solve(
            parameters.inertia,
            wrench[3:]
            - np.cross(
                rigid.angular_velocity,
                parameters.inertia
                @ rigid.angular_velocity,
            ),
        )
        specific_force_body = (
            wrench[:3] / parameters.mass
            + np.cross(
                angular_acceleration,
                imu_lever,
            )
            + np.cross(
                rigid.angular_velocity,
                np.cross(
                    rigid.angular_velocity,
                    imu_lever,
                ),
            )
        )
        arrays["sensor_position"][index] = (
            rigid.position
            + body_rotation @ pose_lever
        )
        arrays[
            "sensor_orientation_xyzw"
        ][index] = matrix_to_quaternion(
            body_rotation
            @ bag.direct_problem
            .pose_body_to_sensor_rotation
        )
        arrays[
            "sensor_velocity_world"
        ][index] = (
            rigid.linear_velocity
            + body_rotation
            @ np.cross(
                rigid.angular_velocity,
                velocity_lever,
            )
        )
        arrays[
            "angular_velocity_sensor"
        ][index] = (
            bag.direct_problem
            .body_to_imu_rotation
            @ rigid.angular_velocity
            + bag.direct_problem.gyro_bias
        )
        arrays[
            "specific_force_sensor"
        ][index] = (
            bag.direct_problem
            .body_to_imu_rotation
            @ specific_force_body
            + bag.direct_problem
            .accelerometer_bias
        )
        arrays["cog_position"][index] = (
            rigid.position
        )
        arrays[
            "cog_velocity_world"
        ][index] = rigid.linear_velocity
        arrays[
            "actuator_thrust"
        ][index] = actuator.thrust
        arrays[
            "actuator_gimbal"
        ][index] = actuator.gimbal_angle

    store(0)
    rotor_switches = (
        np.asarray(
            bag.rotor_history.times[1:],
            dtype=float,
        )
        + float(rotor_delay_seconds)
    )
    gimbal_switches = (
        np.asarray(
            bag.gimbal_history.times[1:],
            dtype=float,
        )
        + float(gimbal_delay_seconds)
    )
    maximum_step = float(
        bag.direct_problem.integration_step
    )

    for index in range(count - 1):
        left = float(time_axis[index])
        right = float(time_axis[index + 1])
        switches = np.concatenate(
            (
                rotor_switches[
                    (rotor_switches > left)
                    & (rotor_switches < right)
                ],
                gimbal_switches[
                    (gimbal_switches > left)
                    & (gimbal_switches < right)
                ],
            )
        )
        boundaries = np.unique(
            np.concatenate(
                (
                    np.asarray(
                        (left,),
                        dtype=float,
                    ),
                    switches,
                    np.asarray(
                        (right,),
                        dtype=float,
                    ),
                )
            )
        )
        for (
            segment_left,
            segment_right,
        ) in zip(
            boundaries[:-1],
            boundaries[1:],
        ):
            segment_span = float(
                segment_right - segment_left
            )
            sub_count = max(
                1,
                int(
                    math.ceil(
                        segment_span
                        / maximum_step
                    )
                ),
            )
            sub_boundaries = np.linspace(
                segment_left,
                segment_right,
                sub_count + 1,
            )
            for (
                sub_left,
                sub_right,
            ) in zip(
                sub_boundaries[:-1],
                sub_boundaries[1:],
            ):
                dt = float(
                    sub_right - sub_left
                )
                midpoint = 0.5 * float(
                    sub_left + sub_right
                )
                actuator = actuator_at(
                    midpoint
                )
                rigid = plant.step(
                    float(sub_left),
                    rigid,
                    actuator,
                    dt,
                )
        store(index + 1)

    simulation = baseline.Simulation(
        time=time_axis.copy(),
        **arrays,
    )
    for value in arrays.values():
        if np.any(~np.isfinite(value)):
            raise FloatingPointError(
                "diagnostic forward rollout "
                "became non-finite"
            )
    return simulation


def _trajectory_rmse(
    bag: BagData,
    simulation: Any,
) -> dict[str, float]:
    observed = (
        spline_reports
        ._observations_at_times(
            bag.direct_problem.observations,
            simulation.time,
        )
    )
    position_error = (
        simulation.sensor_position
        - observed.sensor_position
    )
    orientation_error = np.empty(
        simulation.time.size,
        dtype=float,
    )
    for index in range(
        simulation.time.size
    ):
        observed_rotation = (
            quaternion_to_matrix(
                observed
                .sensor_orientation_xyzw[
                    index
                ]
            )
        )
        simulated_rotation = (
            quaternion_to_matrix(
                simulation
                .sensor_orientation_xyzw[
                    index
                ]
            )
        )
        orientation_error[index] = (
            math.degrees(
                float(
                    np.linalg.norm(
                        so3_log(
                            observed_rotation.T
                            @ simulated_rotation
                        )
                    )
                )
            )
        )
    return {
        "position_rmse_m": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        position_error**2,
                        axis=1,
                    )
                )
            )
        ),
        "orientation_angle_rmse_deg": (
            float(
                np.sqrt(
                    np.mean(
                        orientation_error**2
                    )
                )
            )
        ),
    }


def _write_free_trajectory_pdf(
    path: Path,
    bag: BagData,
    free_rollout: Any,
) -> None:
    observed = spline_reports._observations_at_times(
        bag.direct_problem.observations,
        free_rollout.time,
    )
    relative_time = free_rollout.time - free_rollout.time[0]
    observed_rpy = baseline._rpy_series(
        observed.sensor_orientation_xyzw
    )
    free_rpy = baseline._rpy_series(
        free_rollout.sensor_orientation_xyzw
    )
    lower, upper = spline_reports._common_3d_limits(
        observed.sensor_position,
        free_rollout.sensor_position,
    )

    with PdfPages(path) as pdf:
        figure = plt.figure(
            figsize=(11.7, 8.3),
            constrained_layout=True,
        )
        axis = figure.add_subplot(111, projection="3d")
        axis.plot(
            observed.sensor_position[:, 0],
            observed.sensor_position[:, 1],
            observed.sensor_position[:, 2],
            linewidth=2.5,
            label="observed",
        )
        axis.plot(
            free_rollout.sensor_position[:, 0],
            free_rollout.sensor_position[:, 1],
            free_rollout.sensor_position[:, 2],
            linewidth=2.0,
            linestyle="--",
            label="free rollout with estimated parameters",
        )
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.set_zlim(lower[2], upper[2])
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("z [m]")
        axis.set_title("Observed and free-rollout 3D trajectory")
        axis.legend(loc="best")
        pdf.savefig(figure)
        plt.close(figure)

        for title, reference, prediction, labels in (
            (
                "Observed and free-rollout sensor position",
                observed.sensor_position,
                free_rollout.sensor_position,
                ("x [m]", "y [m]", "z [m]"),
            ),
            (
                "Observed and free-rollout sensor orientation",
                observed_rpy,
                free_rpy,
                ("roll [rad]", "pitch [rad]", "yaw [rad]"),
            ),
        ):
            figure, axes = plt.subplots(
                3,
                1,
                figsize=(11.7, 8.3),
                sharex=True,
                constrained_layout=True,
            )
            for component, axis in enumerate(axes):
                axis.plot(
                    relative_time,
                    reference[:, component],
                    linewidth=2.2,
                    label="observed",
                )
                axis.plot(
                    relative_time,
                    prediction[:, component],
                    linewidth=1.8,
                    linestyle="--",
                    label="free rollout",
                )
                axis.set_ylabel(labels[component])
                axis.grid(True, alpha=0.25)
            axes[0].set_title(title)
            axes[0].legend(loc="best")
            axes[-1].set_xlabel("time from rollout-support start [s]")
            pdf.savefig(figure)
            plt.close(figure)


def _write_trajectory_3d_pdf(
    path: Path,
    bag: BagData,
    free_rollout: Any,
    wrench_rollout: Any,
) -> None:
    reference = np.asarray(
        bag.kinematics.sensor_position,
        dtype=float,
    )
    lower, upper = (
        spline_reports._common_3d_limits(
            reference,
            free_rollout.sensor_position,
            wrench_rollout.sensor_position,
        )
    )
    with PdfPages(path) as pdf:
        figure = plt.figure(
            figsize=(11.7, 8.3),
            constrained_layout=True,
        )
        axis = figure.add_subplot(
            111,
            projection="3d",
        )
        axis.plot(
            reference[:, 0],
            reference[:, 1],
            reference[:, 2],
            label="SG trajectory",
            linewidth=2.4,
        )
        axis.plot(
            free_rollout
            .sensor_position[:, 0],
            free_rollout
            .sensor_position[:, 1],
            free_rollout
            .sensor_position[:, 2],
            linestyle="--",
            label="free rollout",
        )
        axis.plot(
            wrench_rollout
            .sensor_position[:, 0],
            wrench_rollout
            .sensor_position[:, 1],
            wrench_rollout
            .sensor_position[:, 2],
            linestyle=":",
            label=(
                "raw-residual-wrench rollout"
            ),
        )
        axis.set_xlim(
            lower[0],
            upper[0],
        )
        axis.set_ylim(
            lower[1],
            upper[1],
        )
        axis.set_zlim(
            lower[2],
            upper[2],
        )
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("z [m]")
        axis.legend(loc="best")
        axis.set_title(
            "Trajectory comparison"
        )
        pdf.savefig(figure)
        plt.close(figure)


def _write_raw_wrench_pdf(
    path: Path,
    time_axis: np.ndarray,
    wrench: np.ndarray,
    title: str,
) -> None:
    relative_time = (
        np.asarray(
            time_axis,
            dtype=float,
        )
        - float(time_axis[0])
    )
    values = np.asarray(
        wrench,
        dtype=float,
    )
    labels = (
        "F_x [N]",
        "F_y [N]",
        "F_z [N]",
        "M_x [N m]",
        "M_y [N m]",
        "M_z [N m]",
    )
    with PdfPages(path) as pdf:
        figure, axes = plt.subplots(
            3,
            2,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        for component, axis in enumerate(
            axes.ravel()
        ):
            axis.plot(
                relative_time,
                values[:, component],
            )
            axis.axhline(
                0.0,
                linewidth=0.8,
            )
            axis.set_ylabel(
                labels[component]
            )
            axis.grid(True, alpha=0.25)
        axes[-1, 0].set_xlabel(
            "time [s]"
        )
        axes[-1, 1].set_xlabel(
            "time [s]"
        )
        figure.suptitle(title)
        pdf.savefig(figure)
        plt.close(figure)


def _ridge_payload(
    evaluation: JointEvaluation,
    exact_common_scale_expected: bool,
) -> dict[str, Any]:
    jacobian = np.asarray(
        evaluation.jacobian,
        dtype=float,
    )
    spectrum = _jacobian_spectrum(
        jacobian
    )
    information = (
        jacobian.T @ jacobian
    )
    direction = (
        _normalized_scale_gauge(
            PHYSICAL_DIMENSION
        )
    )
    image_relative = float(
        np.linalg.norm(
            jacobian @ direction
        )
        / max(
            np.linalg.norm(jacobian),
            np.finfo(float).tiny,
        )
    )
    weakest_alignment = None
    if spectrum[
        "right_singular_vectors"
    ].size:
        weakest_alignment = float(
            abs(
                np.asarray(
                    spectrum[
                        "right_singular_vectors"
                    ][-1],
                    dtype=float,
                )
                @ direction
            )
        )
    return {
        "schema": (
            SCHEMA
            + "/raw-ridge-analysis/v1"
        ),
        "source": (
            "raw final deterministic data "
            "Jacobian; solver LM damping "
            "and KKT stabilization are "
            "not included"
        ),
        "jacobian_shape": (
            jacobian.shape
        ),
        "data_information_matrix": (
            information
        ),
        "singular_values": (
            spectrum["singular_values"]
        ),
        "right_singular_vectors": (
            spectrum[
                "right_singular_vectors"
            ]
        ),
        "machine_rank": (
            spectrum["numerical_rank"]
        ),
        "machine_nullity": (
            spectrum["nullity"]
        ),
        "machine_rank_tolerance": (
            spectrum[
                "numerical_tolerance"
            ]
        ),
        "condition_number_nonzero_subspace": (
            spectrum[
                "condition_number_nonzero_subspace"
            ]
        ),
        "scientific_ridge_threshold": None,
        "known_common_scale_gauge": {
            "exact_expected": bool(
                exact_common_scale_expected
            ),
            "normalized_direction": direction,
            "relative_jacobian_image_norm": (
                image_relative
            ),
            "weakest_mode_alignment": (
                weakest_alignment
            ),
        },
    }


def _write_ridge_pdf(
    path: Path,
    ridge: Mapping[str, Any],
) -> None:
    singular = np.asarray(
        ridge["singular_values"],
        dtype=float,
    )
    vectors = np.asarray(
        ridge["right_singular_vectors"],
        dtype=float,
    )
    with PdfPages(path) as pdf:
        figure = plt.figure(
            figsize=(11.7, 8.3),
            constrained_layout=True,
        )
        axis = figure.add_subplot(111)
        axis.semilogy(
            np.arange(
                1,
                singular.size + 1,
            ),
            singular,
            marker="o",
        )
        axis.set_xlabel(
            "singular index"
        )
        axis.set_ylabel(
            "raw data-Jacobian "
            "singular value"
        )
        axis.grid(True, alpha=0.25)
        axis.set_title(
            "Raw deterministic Jacobian "
            "spectrum (no LM/KKT "
            "stabilization included)"
        )
        pdf.savefig(figure)
        plt.close(figure)

        weak_count = min(
            4,
            singular.size,
        )
        for offset in range(
            weak_count
        ):
            index = (
                singular.size
                - weak_count
                + offset
            )
            figure = plt.figure(
                figsize=(11.7, 8.3),
                constrained_layout=True,
            )
            axis = figure.add_subplot(111)
            axis.bar(
                np.arange(
                    PHYSICAL_DIMENSION
                ),
                vectors[index],
            )
            axis.set_xticks(
                np.arange(
                    PHYSICAL_DIMENSION
                )
            )
            axis.set_xticklabels(
                PHYSICAL_PARAMETER_NAMES,
                rotation=75,
                ha="right",
            )
            axis.set_ylabel(
                "right-singular-vector "
                "component"
            )
            axis.grid(
                True,
                axis="y",
                alpha=0.25,
            )
            axis.set_title(
                "weak mode {} / "
                "sigma={:.6g}".format(
                    index + 1,
                    singular[index],
                )
            )
            pdf.savefig(figure)
            plt.close(figure)


def _write_confidence_pdf(
    path: Path,
    confidence: Mapping[str, Any],
) -> None:
    spectrum = confidence.get(
        "data_spectrum",
        {},
    )
    singular = np.asarray(
        spectrum.get(
            "singular_values",
            (),
        ),
        dtype=float,
    )
    with PdfPages(path) as pdf:
        if singular.size:
            figure = plt.figure(
                figsize=(11.7, 8.3),
                constrained_layout=True,
            )
            axis = figure.add_subplot(111)
            axis.semilogy(
                np.arange(
                    1,
                    singular.size + 1,
                ),
                singular,
                marker="o",
            )
            axis.set_xlabel(
                "singular index"
            )
            axis.set_ylabel(
                "whitened likelihood-"
                "Jacobian singular value"
            )
            axis.grid(
                True,
                alpha=0.25,
            )
            axis.set_title(
                "Local data-likelihood "
                "spectrum"
            )
            pdf.savefig(figure)
            plt.close(figure)

        posterior = confidence.get(
            "posterior"
        )
        if isinstance(
            posterior,
            Mapping,
        ):
            mean = np.asarray(
                posterior.get(
                    "posterior_physical_"
                    "mean_linearized",
                    (),
                ),
                dtype=float,
            )
            std = np.asarray(
                posterior.get(
                    "posterior_physical_"
                    "std_linearized",
                    (),
                ),
                dtype=float,
            )
            if (
                mean.shape
                == (
                    len(
                        PHYSICAL_VALUE_NAMES
                    ),
                )
                and std.shape
                == mean.shape
            ):
                figure = plt.figure(
                    figsize=(11.7, 8.3),
                    constrained_layout=True,
                )
                axis = (
                    figure.add_subplot(111)
                )
                positions = np.arange(
                    mean.size
                )
                axis.errorbar(
                    positions,
                    mean,
                    yerr=std,
                    fmt="o",
                )
                axis.set_xticks(
                    positions
                )
                axis.set_xticklabels(
                    PHYSICAL_VALUE_NAMES,
                    rotation=75,
                    ha="right",
                )
                axis.grid(
                    True,
                    axis="y",
                    alpha=0.25,
                )
                axis.set_title(
                    "Linearized posterior "
                    "mean +/- 1 std"
                )
                pdf.savefig(figure)
                plt.close(figure)


def _write_delay_profile_pdf(
    path: Path,
    lag_search: Mapping[str, Any],
) -> None:
    with PdfPages(path) as pdf:
        figure = plt.figure(
            figsize=(11.7, 8.3),
            constrained_layout=True,
        )
        axis = figure.add_subplot(111)
        axis.axis("off")
        text = "\n".join(
            (
                "Command-lag search",
                "",
                "mode: {}".format(
                    lag_search.get("mode")
                ),
                (
                    "rotor period [s]: {}"
                ).format(
                    lag_search.get(
                        "rotor_period_seconds"
                    )
                ),
                (
                    "gimbal period [s]: {}"
                ).format(
                    lag_search.get(
                        "gimbal_period_seconds"
                    )
                ),
                (
                    "selected rotor lag [s]: {}"
                ).format(
                    lag_search.get(
                        "selected_rotor_delay_seconds"
                    )
                ),
                (
                    "selected gimbal lag [s]: {}"
                ).format(
                    lag_search.get(
                        "selected_gimbal_delay_seconds"
                    )
                ),
                "termination: {}".format(
                    lag_search.get(
                        "termination"
                    )
                ),
                "elapsed [s]: {}".format(
                    lag_search.get(
                        "elapsed_seconds"
                    )
                ),
            )
        )
        axis.text(
            0.02,
            0.98,
            text,
            va="top",
            family="monospace",
        )
        pdf.savefig(figure)
        plt.close(figure)

        stages = lag_search.get(
            "smooth_stages",
            (),
        )
        if stages:
            stage_index = np.arange(
                1,
                len(stages) + 1,
            )
            rotor = np.asarray(
                [
                    item[
                        "rotor_delay_seconds"
                    ]
                    for item in stages
                ],
                dtype=float,
            )
            gimbal = np.asarray(
                [
                    item[
                        "gimbal_delay_seconds"
                    ]
                    for item in stages
                ],
                dtype=float,
            )
            cost = np.asarray(
                [
                    item[
                        "objective_cost"
                    ]
                    for item in stages
                ],
                dtype=float,
            )
            figure, axes = plt.subplots(
                2,
                1,
                figsize=(11.7, 8.3),
                constrained_layout=True,
            )
            axes[0].plot(
                stage_index,
                rotor,
                marker="o",
                label="rotor",
            )
            axes[0].plot(
                stage_index,
                gimbal,
                marker="o",
                label="gimbal",
            )
            axes[0].set_ylabel(
                "lag [s]"
            )
            axes[0].legend(loc="best")
            axes[1].semilogy(
                stage_index,
                cost,
                marker="o",
            )
            axes[1].set_ylabel(
                "objective"
            )
            axes[1].set_xlabel(
                "smooth stage"
            )
            for axis in axes:
                axis.grid(
                    True,
                    alpha=0.25,
                )
            pdf.savefig(figure)
            plt.close(figure)


def _write_per_bag_outputs(
    bag_directory: Path,
    bag: BagData,
    evaluation: BagEvaluation,
    solution: Solution,
    scales: ReferenceScales,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    started = time.perf_counter()
    bag_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    timing: dict[str, Any] = {}

    tick = time.perf_counter()
    _write_bag_npz(
        bag_directory
        / "savgol_dynamics.npz",
        bag,
        evaluation,
        scales,
    )
    timing[
        "write_savgol_npz_seconds"
    ] = float(
        time.perf_counter() - tick
    )

    tick = time.perf_counter()
    _write_savgol_fit_pdf(
        bag_directory
        / "savgol_fit.pdf",
        bag,
    )
    timing[
        "write_savgol_fit_pdf_seconds"
    ] = float(
        time.perf_counter() - tick
    )

    tick = time.perf_counter()
    free_rollout = (
        _diagnostic_forward_rollout(
            bag,
            solution.evaluation
            .decoded.parameters,
            solution.rotor_delay_seconds,
            solution.gimbal_delay_seconds,
            external_body_wrench=None,
        )
    )
    timing[
        "free_rollout_seconds"
    ] = float(
        time.perf_counter() - tick
    )

    raw_wrench_history = (
        spline_reports.BodyWrenchHistory(
            bag.time,
            evaluation.residual_body_wrench,
        )
    )
    tick = time.perf_counter()
    wrench_rollout = (
        _diagnostic_forward_rollout(
            bag,
            solution.evaluation
            .decoded.parameters,
            solution.rotor_delay_seconds,
            solution.gimbal_delay_seconds,
            external_body_wrench=(
                raw_wrench_history
            ),
        )
    )
    timing[
        "raw_residual_wrench_"
        "rollout_seconds"
    ] = float(
        time.perf_counter() - tick
    )

    tick = time.perf_counter()
    spline_reports._write_trajectory_pdf(
        bag_directory / "trajectory.pdf",
        bag,
        wrench_rollout,
    )
    _write_free_trajectory_pdf(
        bag_directory
        / "trajectory_free.pdf",
        bag,
        free_rollout,
    )
    _write_trajectory_3d_pdf(
        bag_directory
        / "trajectory_3d.pdf",
        bag,
        free_rollout,
        wrench_rollout,
    )
    timing[
        "write_trajectory_pdfs_seconds"
    ] = float(
        time.perf_counter() - tick
    )

    observed = (
        spline_reports
        ._observations_at_times(
            bag.direct_problem.observations,
            wrench_rollout.time,
        )
    )
    sg_gyro, sg_force = (
        spline_reports
        ._pose_spline_implied_sensor_series(
            bag,
            solution.evaluation
            .decoded.parameters,
            wrench_rollout.time,
        )
    )

    tick = time.perf_counter()
    wrench_sensor_metrics = (
        spline_reports._sensor_metrics(
            observed,
            wrench_rollout,
        )
    )
    free_observed = (
        spline_reports
        ._observations_at_times(
            bag.direct_problem.observations,
            free_rollout.time,
        )
    )
    free_sensor_metrics = (
        spline_reports._sensor_metrics(
            free_observed,
            free_rollout,
        )
    )
    spline_reports._write_sensor_validation_pdf(
        bag_directory
        / "sensor_consistency.pdf",
        bag,
        wrench_rollout,
        wrench_sensor_metrics,
    )
    spline_reports._write_sensor_validation_pdf(
        bag_directory
        / "sensor_consistency_free.pdf",
        bag,
        free_rollout,
        free_sensor_metrics,
    )
    spline_reports._write_diagnostic_pdf(
        bag_directory / "diagnostic.pdf",
        observed,
        sg_gyro,
        sg_force,
        wrench_rollout,
    )
    timing[
        "write_sensor_diagnostic_"
        "pdfs_seconds"
    ] = float(
        time.perf_counter() - tick
    )

    wrench_statistics = (
        _residual_wrench_statistics(
            evaluation,
            scales,
        )
    )
    tick = time.perf_counter()
    _write_raw_wrench_pdf(
        bag_directory
        / "raw_residual_wrench.pdf",
        bag.time,
        evaluation.residual_body_wrench,
        (
            "Raw inverse-dynamics "
            "residual body wrench"
        ),
    )
    _write_raw_wrench_pdf(
        bag_directory
        / "external_wrench.pdf",
        bag.time,
        evaluation.residual_body_wrench,
        (
            "External-wrench diagnostic: "
            "raw inverse-dynamics residual "
            "(no external-wrench minimization)"
        ),
    )
    timing[
        "write_wrench_pdfs_seconds"
    ] = float(
        time.perf_counter() - tick
    )

    tick = time.perf_counter()
    np.savez_compressed(
        bag_directory
        / "rollout_diagnostics.npz",
        time=wrench_rollout.time,
        free_sensor_position=(
            free_rollout.sensor_position
        ),
        free_sensor_orientation_xyzw=(
            free_rollout
            .sensor_orientation_xyzw
        ),
        free_sensor_velocity_world=(
            free_rollout
            .sensor_velocity_world
        ),
        free_angular_velocity_sensor=(
            free_rollout
            .angular_velocity_sensor
        ),
        free_specific_force_sensor=(
            free_rollout
            .specific_force_sensor
        ),
        wrench_sensor_position=(
            wrench_rollout.sensor_position
        ),
        wrench_sensor_orientation_xyzw=(
            wrench_rollout
            .sensor_orientation_xyzw
        ),
        wrench_sensor_velocity_world=(
            wrench_rollout
            .sensor_velocity_world
        ),
        wrench_angular_velocity_sensor=(
            wrench_rollout
            .angular_velocity_sensor
        ),
        wrench_specific_force_sensor=(
            wrench_rollout
            .specific_force_sensor
        ),
    )
    timing[
        "write_rollout_npz_seconds"
    ] = float(
        time.perf_counter() - tick
    )

    diagnostic = {
        "schema": (
            SCHEMA
            + "/bag-diagnostic/v1"
        ),
        "free_rollout": {
            "trajectory": _trajectory_rmse(
                bag,
                free_rollout,
            ),
            "sensor": (
                free_sensor_metrics
            ),
        },
        "raw_residual_wrench_rollout": {
            "trajectory": _trajectory_rmse(
                bag,
                wrench_rollout,
            ),
            "sensor": (
                wrench_sensor_metrics
            ),
        },
        "pose_sg_implied_vs_measured": (
            spline_reports
            ._sensor_pair_metrics(
                observed.time,
                observed
                .angular_velocity_sensor,
                observed
                .specific_force_sensor,
                sg_gyro,
                sg_force,
            )
        ),
        "raw_residual_wrench_statistics": (
            wrench_statistics
        ),
        "external_wrench_interpretation": (
            "diagnostic residual wrench "
            "only; its magnitude is not "
            "part of the deterministic "
            "parameter objective"
        ),
        "timing": timing,
    }
    _write_json(
        bag_directory / "diagnostic.json",
        diagnostic,
    )
    _write_json(
        bag_directory
        / "external_wrench.json",
        {
            "time_seconds": bag.time,
            "body_wrench_physical": (
                evaluation
                .residual_body_wrench
            ),
            "statistics": (
                wrench_statistics
            ),
        },
    )

    outputs = {
        "result_json": "result.json",
        "diagnostic_json": (
            "diagnostic.json"
        ),
        "diagnostic_pdf": (
            "diagnostic.pdf"
        ),
        "savgol_fit_pdf": (
            "savgol_fit.pdf"
        ),
        "trajectory_pdf": (
            "trajectory.pdf"
        ),
        "trajectory_free_pdf": (
            "trajectory_free.pdf"
        ),
        "trajectory_3d_pdf": (
            "trajectory_3d.pdf"
        ),
        "sensor_consistency_pdf": (
            "sensor_consistency.pdf"
        ),
        "sensor_consistency_free_pdf": (
            "sensor_consistency_free.pdf"
        ),
        "raw_residual_wrench_pdf": (
            "raw_residual_wrench.pdf"
        ),
        "external_wrench_pdf": (
            "external_wrench.pdf"
        ),
        "external_wrench_json": (
            "external_wrench.json"
        ),
        "savgol_dynamics_npz": (
            "savgol_dynamics.npz"
        ),
        "rollout_diagnostics_npz": (
            "rollout_diagnostics.npz"
        ),
    }
    timing["elapsed_seconds"] = float(
        time.perf_counter() - started
    )
    return outputs, diagnostic


def _parameters_pdf(
    path: Path,
    text: str,
) -> None:
    _write_text_pdf(
        path,
        "Dimensionless SG parameter result",
        text,
    )

def _write_bag_npz(
    path: Path,
    bag: BagData,
    evaluation: BagEvaluation,
    scales: ReferenceScales,
) -> None:
    wrench_scale = np.asarray(
        (
            1.0 / scales.force_n,
            1.0 / scales.force_n,
            1.0 / scales.force_n,
            1.0 / scales.torque_nm,
            1.0 / scales.torque_nm,
            1.0 / scales.torque_nm,
        ),
        dtype=float,
    )
    np.savez_compressed(
        path,
        time=bag.time,
        sensor_position=bag.kinematics.sensor_position,
        sensor_velocity_world=(
            bag.kinematics.sensor_velocity_world
        ),
        sensor_acceleration_world=(
            bag.kinematics.sensor_acceleration_world
        ),
        body_rotation=bag.kinematics.body_rotation,
        body_angular_velocity=(
            bag.kinematics.body_angular_velocity
        ),
        body_angular_acceleration=(
            bag.kinematics.body_angular_acceleration
        ),
        acceleration_residual_dimensionless=(
            evaluation.acceleration_residual
        ),
        acceleration_jacobian_dimensionless=(
            evaluation.acceleration_jacobian
        ),
        required_body_wrench_physical=(
            evaluation.required_body_wrench
        ),
        modeled_body_wrench_physical=(
            evaluation.modeled_body_wrench
        ),
        residual_body_wrench_physical=(
            evaluation.residual_body_wrench
        ),
        residual_body_wrench_dimensionless=(
            evaluation.residual_body_wrench
            * wrench_scale[None, :]
        ),
        residual_body_wrench_jacobian_physical=(
            evaluation.residual_body_wrench_jacobian
        ),
        actuator_thrust=evaluation.actuator_thrust,
        actuator_gimbal=evaluation.actuator_gimbal,
        cog_position_world=evaluation.cog_position_world,
        cog_velocity_world=evaluation.cog_velocity_world,
        cog_acceleration_world=(
            evaluation.cog_acceleration_world
        ),
        reference_mass_kg=scales.mass_kg,
        reference_length_m=scales.length_m,
        reference_time_s=scales.time_s,
        reference_force_n=scales.force_n,
        reference_torque_nm=scales.torque_nm,
    )


def _write_bag_pdf(
    path: Path,
    bag: BagData,
    evaluation: BagEvaluation,
) -> None:
    relative_time = bag.time - bag.time[0]
    acceleration_labels = (
        "linear x",
        "linear y",
        "linear z",
        "angular x",
        "angular y",
        "angular z",
    )
    wrench_labels = (
        "F_x [N]",
        "F_y [N]",
        "F_z [N]",
        "M_x [N m]",
        "M_y [N m]",
        "M_z [N m]",
    )
    with PdfPages(path) as pdf:
        figure, axes = plt.subplots(
            3,
            2,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        for index, axis in enumerate(
            axes.ravel()
        ):
            axis.plot(
                relative_time,
                evaluation.acceleration_residual[:, index],
            )
            axis.set_ylabel(acceleration_labels[index])
            axis.grid(True, alpha=0.25)
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        figure.suptitle(
            "Dimensionless generalized-acceleration residual"
        )
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(
            3,
            2,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        for index, axis in enumerate(
            axes.ravel()
        ):
            axis.plot(
                relative_time,
                evaluation.residual_body_wrench[:, index],
            )
            axis.set_ylabel(wrench_labels[index])
            axis.grid(True, alpha=0.25)
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        figure.suptitle(
            "Raw inverse-dynamics residual body wrench"
        )
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(
            2,
            2,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        for rotor in range(4):
            axes[0, 0].plot(
                relative_time,
                evaluation.actuator_thrust[:, rotor],
                label="rotor {}".format(rotor + 1),
            )
            axes[0, 1].plot(
                relative_time,
                evaluation.actuator_gimbal[:, rotor],
                label="gimbal {}".format(rotor + 1),
            )
        axes[0, 0].set_ylabel("thrust command [N]")
        axes[0, 1].set_ylabel("gimbal [rad]")
        axes[0, 0].legend(loc="best")
        axes[0, 1].legend(loc="best")
        axes[1, 0].plot(
            relative_time,
            bag.kinematics.window_sample_count,
        )
        axes[1, 0].set_ylabel("SG samples/window")
        axes[1, 1].semilogy(
            relative_time,
            bag.kinematics.position_fit_condition_number,
            label="position",
        )
        axes[1, 1].semilogy(
            relative_time,
            bag.kinematics.rotation_fit_condition_number,
            label="rotation",
        )
        axes[1, 1].set_ylabel("local condition")
        axes[1, 1].legend(loc="best")
        for axis in axes.ravel():
            axis.grid(True, alpha=0.25)
            axis.set_xlabel("time [s]")
        figure.suptitle(
            "Actuator and SG diagnostics"
        )
        pdf.savefig(figure)
        plt.close(figure)


def _write_summary_pdf(
    path: Path,
    result: Mapping[str, Any],
) -> None:
    singular = np.asarray(
        result["optimizer"]["jacobian_spectrum"][
            "singular_values"
        ],
        dtype=float,
    )
    coordinate = np.asarray(
        result["selection"]["physical_coordinate"],
        dtype=float,
    )
    with PdfPages(path) as pdf:
        figure = plt.figure(
            figsize=(11.7, 8.3),
            constrained_layout=True,
        )
        axis = figure.add_subplot(1, 1, 1)
        axis.semilogy(
            np.arange(1, singular.size + 1),
            singular,
            marker="o",
        )
        axis.set_xlabel("singular index")
        axis.set_ylabel("objective Jacobian singular value")
        axis.grid(True, alpha=0.25)
        axis.set_title(
            "Dimensionless deterministic information spectrum"
        )
        pdf.savefig(figure)
        plt.close(figure)

        figure = plt.figure(
            figsize=(11.7, 8.3),
            constrained_layout=True,
        )
        axis = figure.add_subplot(1, 1, 1)
        axis.bar(
            np.arange(PHYSICAL_DIMENSION),
            coordinate,
        )
        axis.set_xticks(
            np.arange(PHYSICAL_DIMENSION)
        )
        axis.set_xticklabels(
            PHYSICAL_PARAMETER_NAMES,
            rotation=75,
            ha="right",
        )
        axis.set_ylabel("dimensionless coordinate")
        axis.grid(True, axis="y", alpha=0.25)
        axis.set_title(
            "Selected dimensionless physical coordinate"
        )
        pdf.savefig(figure)
        plt.close(figure)


def _parameters_text(
    result: Mapping[str, Any],
) -> str:
    parameters = result["selection"]["parameters"]
    lines = [
        "Dimensionless geometric Savitzky-Golay experiment",
        "",
        "Fixed reference scales",
    ]
    for key, value in result[
        "nondimensionalization"
    ].items():
        lines.append("  {}: {}".format(key, value))

    lines.extend(
        [
            "",
            "Selected deterministic point (prior-free)",
            "  objective cost: {}".format(
                result["selection"]["objective_cost"]
            ),
            "  rotor delay [s]: {}".format(
                result["selection"][
                    "rotor_delay_seconds"
                ]
            ),
            "  gimbal delay [s]: {}".format(
                result["selection"][
                    "gimbal_delay_seconds"
                ]
            ),
            "  optimizer success: {}".format(
                result["optimizer"]["success"]
            ),
            "  optimizer message: {}".format(
                result["optimizer"]["message"]
            ),
            "  exact scale gauge: KKT hard constraint",
            "  unknown weak-mode cutoff: none",
            "  Jacobian rank: {}".format(
                result["optimizer"][
                    "jacobian_spectrum"
                ]["numerical_rank"]
            ),
            "  Jacobian nullity: {}".format(
                result["optimizer"][
                    "jacobian_spectrum"
                ]["nullity"]
            ),
            "  common-scale analytic null norm: {}".format(
                result["optimizer"][
                    "common_scale_gauge"
                ][
                    "analytic_jacobian_relative_null_norm"
                ]
            ),
            "  common-scale FD null norm: {}".format(
                result["optimizer"][
                    "common_scale_gauge"
                ][
                    "finite_difference_relative_null_norm"
                ]
            ),
            "",
            "Dimensionless coordinate",
        ]
    )
    for name, value in zip(
        PHYSICAL_PARAMETER_NAMES,
        result["selection"]["physical_coordinate"],
    ):
        lines.append(
            "  {:48s} {: .12g}".format(
                name,
                float(value),
            )
        )

    lines.extend(
        [
            "",
            "Physical representative",
            "  mass [kg]: {}".format(
                parameters["mass_kg"]
            ),
            "  CoG body [m]: {}".format(
                parameters["cog_position_body_m"]
            ),
            "  force effectiveness: {}".format(
                parameters["force_effectiveness"]
            ),
            "  inertia [kg m^2]:",
        ]
    )
    for row in parameters["inertia_kg_m2"]:
        lines.append(
            "    [{: .10g}, {: .10g}, {: .10g}]".format(
                *row
            )
        )
    lines.append(
        "  principal moments [kg m^2]: {}".format(
            parameters[
                "inertia_principal_moments_kg_m2"
            ]
        )
    )

    lines.extend(
        [
            "",
            "Per-bag raw residual-wrench RMS",
        ]
    )
    for bag in result["bags"]:
        lines.append(
            "  {}: {}".format(
                bag["id"],
                bag[
                    "residual_wrench_statistics"
                ]["rms_physical"],
            )
        )
    return "\n".join(lines) + "\n"


def _load_problem(
    arguments: argparse.Namespace,
    window_seconds: float,
) -> tuple[
    DimensionlessDynamicsProblem,
    VehicleModelInput,
    Optional[GaussianPhysicalPrior],
    dict[str, Any],
]:
    total_started = time.perf_counter()
    timing: dict[str, Any] = {
        "per_bag": {},
    }

    tick = time.perf_counter()
    config = multi.load_multi_bag_config(
        arguments.config
    )
    model = load_vehicle_model(
        arguments.vehicle_model_json
    )
    prior = load_parameter_prior(
        arguments.prior_json
    )
    scales = ReferenceScales.from_reference(
        model.parameters
    )
    parameterization = DimensionlessParameterization(
        model.parameters,
        scales,
    )
    parameterization.self_test()
    timing[
        "config_model_parameterization_seconds"
    ] = float(
        time.perf_counter() - tick
    )

    raw_weights = np.asarray(
        [
            specification.weight
            for specification in config.bags
        ],
        dtype=float,
    )
    normalized_weights = (
        raw_weights / np.sum(raw_weights)
    )
    bags = []
    rosbag_total = 0.0
    sg_total = 0.0
    for specification, weight in zip(
        config.bags,
        normalized_weights,
    ):
        bag_timing: dict[str, float] = {}

        tick = time.perf_counter()
        flight = load_flight_data(
            str(specification.path),
            start_local=specification.start,
            end_local=specification.end,
            include_fc_specific_force=True,
            compute_sha256=False,
        )
        elapsed = time.perf_counter() - tick
        rosbag_total += elapsed
        bag_timing[
            "rosbag_loading_seconds"
        ] = float(elapsed)

        tick = time.perf_counter()
        bag = _build_bag(
            specification,
            float(weight),
            flight,
            window_seconds,
            arguments,
            model,
            scales,
        )
        elapsed = time.perf_counter() - tick
        sg_total += elapsed
        bag_timing[
            "savgol_and_problem_"
            "construction_seconds"
        ] = float(elapsed)

        timing["per_bag"][
            str(specification.bag_id)
        ] = bag_timing
        bags.append(bag)

    timing[
        "rosbag_loading_seconds"
    ] = float(rosbag_total)
    timing[
        "savgol_and_problem_"
        "construction_seconds"
    ] = float(sg_total)

    tick = time.perf_counter()
    problem = DimensionlessDynamicsProblem(
        bags,
        model,
        scales,
    )
    timing[
        "joint_problem_construction_seconds"
    ] = float(
        time.perf_counter() - tick
    )
    timing["elapsed_seconds"] = float(
        time.perf_counter() - total_started
    )
    return problem, model, prior, timing

def run_window(
    arguments: argparse.Namespace,
    window_seconds: float,
    output_directory: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    timing: dict[str, Any] = {
        "per_bag": {},
    }

    (
        problem,
        model,
        prior,
        setup_timing,
    ) = _load_problem(
        arguments,
        window_seconds,
    )
    timing["problem_setup"] = setup_timing

    tick = time.perf_counter()
    solution, lag_search = _lag_search(
        problem,
        arguments,
    )
    timing[
        "lag_and_physical_optimization_seconds"
    ] = float(
        time.perf_counter() - tick
    )
    timing["lag_search"] = (
        lag_search.get(
            "timing",
            {
                "elapsed_seconds": (
                    lag_search.get(
                        "elapsed_seconds"
                    )
                )
            },
        )
    )

    tick = time.perf_counter()
    ridge = _ridge_payload(
        solution.evaluation,
        problem.exact_common_scale_symmetry,
    )
    timing[
        "final_raw_jacobian_ridge_seconds"
    ] = float(
        time.perf_counter() - tick
    )

    tick = time.perf_counter()
    confidence = (
        None
        if arguments.skip_confidence
        else _confidence_payload(
            problem,
            solution,
            prior,
        )
    )
    timing["confidence_seconds"] = float(
        time.perf_counter() - tick
    )

    selection_parameters = _parameter_payload(
        solution.evaluation
        .decoded.parameters
    )
    bags_directory = (
        output_directory / "bags"
    )
    bags_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    bag_payloads = []

    tick_reports = time.perf_counter()
    for bag, evaluation in zip(
        problem.bags,
        solution.evaluation.bag_evaluations,
    ):
        bag_id = str(
            bag.specification.bag_id
        )
        bag_directory = (
            bags_directory / bag_id
        )
        outputs, diagnostic = (
            _write_per_bag_outputs(
                bag_directory,
                bag,
                evaluation,
                solution,
                problem.scales,
            )
        )
        timing["per_bag"][bag_id] = dict(
            diagnostic.get(
                "timing",
                {},
            )
        )
        payload = {
            "id": bag_id,
            "path": bag.specification.path,
            "start_seconds": (
                bag.specification.start
            ),
            "end_seconds": (
                bag.specification.end
            ),
            "normalized_weight": (
                bag.normalized_weight
            ),
            "valid_sg_center_count": int(
                bag.time.size
            ),
            "sg_window_seconds": float(
                window_seconds
            ),
            "minimum_window_sample_count": int(
                np.min(
                    bag.kinematics
                    .window_sample_count
                )
            ),
            "maximum_window_sample_count": int(
                np.max(
                    bag.kinematics
                    .window_sample_count
                )
            ),
            "data_loss": (
                evaluation.data_loss
            ),
            "residual_wrench_statistics": (
                _residual_wrench_statistics(
                    evaluation,
                    problem.scales,
                )
            ),
            "outputs": {
                key: (
                    "bags/{}/{}".format(
                        bag_id,
                        value,
                    )
                )
                for key, value in outputs.items()
            },
            "diagnostic": diagnostic,
        }
        _write_json(
            bag_directory / "result.json",
            payload,
        )
        bag_payloads.append(payload)

    timing[
        "per_bag_reporting_seconds"
    ] = float(
        time.perf_counter() - tick_reports
    )

    objective_cost = 0.5 * float(
        solution.evaluation.residual
        @ solution.evaluation.residual
    )
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "window_seconds": float(
            window_seconds
        ),
        "vehicle_model_path": (
            model.source_path
        ),
        "parameter_prior_path": (
            None
            if prior is None
            else prior.source_path
        ),
        "deterministic_prior": None,
        "nondimensionalization": (
            problem.scales.payload()
        ),
        "parameter_chart": {
            "coordinate_names": (
                PHYSICAL_PARAMETER_NAMES
            ),
            "dimension": (
                PHYSICAL_DIMENSION
            ),
            "second_moment_chart": (
                "Sigma_bar = B0 exp(S) B0"
            ),
            "cog_coordinate": (
                "(c - c_reference) / L_*"
            ),
            "lag_coordinate": (
                "delay_seconds / T_* "
                "during smooth solve"
            ),
            "common_scale_ridge_retained": True,
            "common_scale_direction": (
                COMMON_SCALE_DIRECTION
            ),
            "physical_box_bounds_active": False,
            "broad_numeric_trial_guard_abs": (
                float(
                    arguments
                    .numeric_coordinate_guard
                )
            ),
            "old_box_bound_arguments": {
                "log_scale_bound": (
                    arguments.log_scale_bound
                ),
                "matrix_log_bound": (
                    arguments
                    .matrix_log_bound
                ),
                "cog_bound": (
                    arguments.cog_bound
                ),
                "interpretation": (
                    "deprecated compatibility "
                    "values; not used as "
                    "optimization box constraints"
                ),
            },
        },
        "selection": {
            "physical_coordinate": (
                solution.physical_coordinate
            ),
            "rotor_delay_seconds": (
                solution.rotor_delay_seconds
            ),
            "gimbal_delay_seconds": (
                solution.gimbal_delay_seconds
            ),
            "objective_cost": objective_cost,
            "data_loss": (
                solution.evaluation.data_loss
            ),
            "parameters": selection_parameters,
            "interpretation": (
                "one gauge-fixed numerical "
                "representative; scientific "
                "ridge information is in "
                "ridge.json"
            ),
        },
        "optimizer": solution.optimizer,
        "lag_search": lag_search,
        "ridge_path": "ridge.json",
        "bags": bag_payloads,
        "confidence_path": (
            None
            if confidence is None
            else "confidence.json"
        ),
        "outputs": {
            "result_json": "result.json",
            "arguments_json": (
                "arguments.json"
            ),
            "timing_json": "timing.json",
            "parameters_txt": (
                "parameters.txt"
            ),
            "parameters_pdf": (
                "parameters.pdf"
            ),
            "summary_pdf": "summary.pdf",
            "ridge_json": "ridge.json",
            "ridge_pdf": "ridge.pdf",
            "delay_profile_json": (
                "delay_profile.json"
            ),
            "delay_profile_pdf": (
                "delay_profile.pdf"
            ),
            "confidence_json": (
                None
                if confidence is None
                else "confidence.json"
            ),
            "confidence_pdf": (
                None
                if confidence is None
                else "confidence.pdf"
            ),
            "data_dictionary": (
                "DATA_DICTIONARY.md"
            ),
            "bags_directory": "bags",
        },
    }

    tick = time.perf_counter()
    _write_json(
        output_directory / "ridge.json",
        ridge,
    )
    _write_ridge_pdf(
        output_directory / "ridge.pdf",
        ridge,
    )
    _write_json(
        output_directory
        / "delay_profile.json",
        lag_search,
    )
    _write_delay_profile_pdf(
        output_directory
        / "delay_profile.pdf",
        lag_search,
    )
    if confidence is not None:
        _write_json(
            output_directory
            / "confidence.json",
            confidence,
        )
        _write_confidence_pdf(
            output_directory
            / "confidence.pdf",
            confidence,
        )
    parameters_text = _parameters_text(
        result
    )
    (
        output_directory / "parameters.txt"
    ).write_text(
        parameters_text,
        encoding="utf-8",
    )
    _parameters_pdf(
        output_directory / "parameters.pdf",
        parameters_text,
    )
    _write_summary_pdf(
        output_directory / "summary.pdf",
        result,
    )
    _write_json(
        output_directory / "arguments.json",
        vars(arguments),
    )
    dictionary_source = (
        Path(__file__)
        .resolve()
        .with_name(
            "dimensionless_savgol_"
            "experiment_data_dictionary.md"
        )
    )
    if dictionary_source.is_file():
        shutil.copyfile(
            dictionary_source,
            output_directory
            / "DATA_DICTIONARY.md",
        )
    timing[
        "root_reporting_seconds"
    ] = float(
        time.perf_counter() - tick
    )

    timing["elapsed_seconds"] = float(
        time.perf_counter() - started
    )
    result["timing"] = timing
    result["elapsed_seconds"] = (
        timing["elapsed_seconds"]
    )
    _write_json(
        output_directory / "timing.json",
        timing,
    )
    _write_json(
        output_directory / "result.json",
        result,
    )
    return result

def _minimum_window(
    arguments: argparse.Namespace,
) -> float:
    config = multi.load_multi_bag_config(
        arguments.config
    )
    minimums = []
    for specification in config.bags:
        flight = load_flight_data(
            str(specification.path),
            start_local=specification.start,
            end_local=specification.end,
            include_fc_specific_force=True,
            compute_sha256=False,
        )
        minimums.append(
            sg.minimum_feasible_window_seconds(
                flight.pose.times
            )
        )
    return float(max(minimums))


def _write_ablation_pdf(
    path: Path,
    cases: Sequence[Mapping[str, Any]],
) -> None:
    completed = [
        case
        for case in cases
        if case.get("status") == "completed"
    ]
    if not completed:
        return
    windows = np.asarray(
        [
            float(case["window_seconds"])
            for case in completed
        ],
        dtype=float,
    )
    objective = np.asarray(
        [
            float(
                case["selection"]["objective_cost"]
            )
            for case in completed
        ],
        dtype=float,
    )
    mass = np.asarray(
        [
            float(
                case["selection"]["parameters"][
                    "mass_kg"
                ]
            )
            for case in completed
        ],
        dtype=float,
    )
    wrench_rms = np.asarray(
        [
            case["bags"][0][
                "residual_wrench_statistics"
            ]["rms_physical"]
            for case in completed
        ],
        dtype=float,
    )
    null_norm = np.asarray(
        [
            float(
                case["optimizer"][
                    "common_scale_gauge"
                ][
                    "analytic_jacobian_relative_null_norm"
                ]
            )
            for case in completed
        ],
        dtype=float,
    )

    with PdfPages(path) as pdf:
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(11.7, 8.3),
            constrained_layout=True,
        )
        axes[0, 0].plot(
            windows,
            objective,
            marker="o",
        )
        axes[0, 0].set_ylabel(
            "dimensionless objective"
        )
        axes[0, 1].plot(
            windows,
            mass,
            marker="o",
        )
        axes[0, 1].set_ylabel("mass representative [kg]")
        axes[1, 0].semilogy(
            windows,
            null_norm,
            marker="o",
        )
        axes[1, 0].set_ylabel(
            "common-scale null error"
        )
        for component in range(6):
            axes[1, 1].plot(
                windows,
                wrench_rms[:, component],
                marker="o",
                label=(
                    ("F", "F", "F", "M", "M", "M")[
                        component
                    ]
                    + str(component % 3 + 1)
                ),
            )
        axes[1, 1].set_ylabel(
            "raw residual wrench RMS"
        )
        axes[1, 1].legend(loc="best")
        for axis in axes.ravel():
            axis.set_xlabel("W [s]")
            axis.grid(True, alpha=0.25)
        figure.suptitle(
            "Dimensionless SG window ablation"
        )
        pdf.savefig(figure)
        plt.close(figure)


def run_ablation(
    arguments: argparse.Namespace,
) -> int:
    output_root = (
        arguments.output_dir.expanduser().resolve()
        / OUTPUT_SUBDIRECTORY
        / _experiment_namespace(arguments)
    )
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    windows = list(
        float(value)
        for value in arguments.windows_seconds
    )
    if arguments.include_minimum_window:
        windows.append(_minimum_window(arguments))
    if arguments.window_seconds is not None:
        windows.append(
            float(arguments.window_seconds)
        )
    windows = sorted(set(windows))
    if not windows:
        raise SystemExit(
            "ablation requires at least one window"
        )

    cases = []
    for window in windows:
        case_directory = (
            output_root
            / "W_{}s".format(
                _safe_label(window)
            )
        )
        print(
            "running dimensionless SG W={:.12g}s".format(
                window
            ),
            flush=True,
        )
        try:
            cases.append(
                run_window(
                    arguments,
                    window,
                    case_directory,
                )
            )
        except Exception as error:
            case_directory.mkdir(
                parents=True,
                exist_ok=True,
            )
            failure = {
                "schema": SCHEMA + "/failed-window",
                "status": "failed",
                "window_seconds": float(window),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _write_json(
                case_directory / "failure.json",
                failure,
            )
            cases.append(failure)
            print(
                "W={:.12g}s failed: {}: {}".format(
                    window,
                    type(error).__name__,
                    error,
                ),
                flush=True,
            )

    payload = {
        "schema": SCHEMA + "/ablation",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "windows_seconds": windows,
        "cases": cases,
    }
    _write_json(
        output_root / "ablation.json",
        payload,
    )
    _write_ablation_pdf(
        output_root / "ablation.pdf",
        cases,
    )
    print(
        "wrote {}".format(
            output_root / "ablation.json"
        ),
        flush=True,
    )
    return 0


def run_fit(arguments: argparse.Namespace) -> int:
    if arguments.window_seconds is None:
        raise SystemExit(
            "--window-seconds is required in fit mode"
        )
    output_directory = (
        arguments.output_dir.expanduser().resolve()
        / OUTPUT_SUBDIRECTORY
        / _experiment_namespace(arguments)
        / "W_{}s".format(
            _safe_label(
                arguments.window_seconds
            )
        )
    )
    result = run_window(
        arguments,
        float(arguments.window_seconds),
        output_directory,
    )
    print(
        "wrote {}".format(
            output_directory / "result.json"
        ),
        flush=True,
    )
    print(
        "objective {:.12g}; mass representative {:.9g} kg".format(
            result["selection"]["objective_cost"],
            result["selection"]["parameters"]["mass_kg"],
        ),
        flush=True,
    )
    return 0


def run_confidence_only(
    arguments: argparse.Namespace,
) -> int:
    if arguments.deterministic_result is None:
        raise SystemExit(
            "--deterministic-result is required in confidence mode"
        )
    result_path = (
        arguments.deterministic_result
        .expanduser()
        .resolve()
    )
    try:
        result = json.loads(
            result_path.read_text(
                encoding="utf-8"
            )
        )
        if result.get("schema") != SCHEMA:
            raise ValueError(
                "deterministic result was not produced by "
                "dimensionless_savgol_experiment.py"
            )
        window = float(
            result["window_seconds"]
        )
        coordinate = np.asarray(
            result["selection"][
                "physical_coordinate"
            ],
            dtype=float,
        )
        rotor_delay = float(
            result["selection"][
                "rotor_delay_seconds"
            ]
        )
        gimbal_delay = float(
            result["selection"][
                "gimbal_delay_seconds"
            ]
        )
        if coordinate.shape != (PHYSICAL_DIMENSION,):
            raise ValueError(
                "deterministic result coordinate is not 14-D"
            )
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(
            "deterministic result cannot be read"
        ) from error

    problem, _model, prior, setup_timing = _load_problem(
        arguments,
        window,
    )
    evaluation = problem.evaluate_strict(
        coordinate,
        rotor_delay,
        gimbal_delay,
    )
    solution = Solution(
        physical_coordinate=coordinate,
        rotor_delay_seconds=rotor_delay,
        gimbal_delay_seconds=gimbal_delay,
        evaluation=evaluation,
        optimizer={
            "source": "deterministic_result",
            "path": result_path,
        },
    )
    confidence = _confidence_payload(
        problem,
        solution,
        prior,
    )
    output_path = (
        result_path.parent / "confidence.json"
    )
    _write_json(output_path, confidence)
    _write_confidence_pdf(
        result_path.parent / "confidence.pdf",
        confidence,
    )
    _write_json(
        result_path.parent / "confidence_timing.json",
        {"problem_setup": setup_timing},
    )
    print("wrote {}".format(output_path), flush=True)
    return 0


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prior-free fixed-reference dimensionless "
            "geometric-SG rigid-body experiment"
        )
    )
    parser.add_argument(
        "--mode",
        choices=("fit", "ablation", "confidence"),
        default="fit",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=False,
    )
    parser.add_argument(
        "--vehicle-model-json",
        type=Path,
        required=False,
    )
    parser.add_argument(
        "--prior-json",
        type=Path,
        default=None,
        help=(
            "Used only for confidence/posterior output; "
            "never enters the deterministic objective."
        ),
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--windows-seconds",
        type=float,
        nargs="*",
        default=(0.5, 1.0, 1.5, 2.0),
    )
    parser.add_argument(
        "--include-minimum-window",
        action="store_true",
    )
    parser.add_argument(
        "--deterministic-result",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--sample-step",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--integration-step",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--smooth-max-nfev",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--strict-max-nfev",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--strict-alternations",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--ftol",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--xtol",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--gtol",
        type=float,
        default=1.0e-8,
    )
    parser.add_argument(
        "--log-scale-bound",
        type=float,
        default=8.0,
        help=(
            "Deprecated compatibility option. It no longer "
            "defines an active optimization box bound."
        ),
    )
    parser.add_argument(
        "--optimizer-svd-rcond",
        type=float,
        default=None,
        help=(
            "Deprecated no-op. Unknown weak modes are not removed "
            "by an SVD cutoff in the gauge-constrained LM solver."
        ),
    )
    parser.add_argument(
        "--matrix-log-bound",
        type=float,
        default=6.0,
        help=(
            "Deprecated compatibility option. It no longer "
            "defines an active matrix-log box bound."
        ),
    )
    parser.add_argument(
        "--cog-bound",
        type=float,
        default=10.0,
        help=(
            "Deprecated compatibility option. It no longer "
            "defines an active CoG box bound."
        ),
    )
    parser.add_argument(
        "--numeric-coordinate-guard",
        type=float,
        default=50.0,
        help=(
            "Broad floating-point safety guard for trial physical "
            "coordinates. Guarded trials are rejected and LM damping "
            "is increased; this is not an active box constraint."
        ),
    )
    parser.add_argument(
        "--lm-initial-damping-relative",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--lm-initial-trust-radius",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lm-maximum-trust-radius",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--lm-minimum-trust-radius",
        type=float,
        default=1.0e-10,
    )
    parser.add_argument(
        "--lm-acceptance-ratio",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument(
        "--delay-bounds",
        type=float,
        nargs=2,
        default=(0.0, 0.20),
    )
    parser.add_argument(
        "--initial-delay",
        type=float,
        default=None,
        help=(
            "Compatibility alias: used for both rotor and gimbal "
            "initial lag unless the channel-specific option is supplied."
        ),
    )
    parser.add_argument(
        "--initial-rotor-delay",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--initial-gimbal-delay",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--smoothstep-width-fractions",
        type=float,
        nargs="+",
        default=(4.0, 2.0, 1.0, 0.5),
    )
    parser.add_argument(
        "--search-lags",
        action="store_true",
        help=(
            "Deprecated no-op compatibility flag. Lag search is on "
            "by default; use --skip-lag-search to disable it."
        ),
    )
    parser.add_argument(
        "--skip-lag-search",
        action="store_true",
        help=(
            "Skip smooth/strict lag search and keep the data-derived "
            "initial rotor/gimbal lags fixed."
        ),
    )
    parser.add_argument(
        "--skip-confidence",
        action="store_true",
        help="Do not write confidence/posterior output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "output"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
    )
    return parser


def _validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    if arguments.self_test:
        if arguments.vehicle_model_json is None:
            raise SystemExit(
                "--self-test requires --vehicle-model-json"
            )
        return
    if arguments.config is None:
        raise SystemExit("--config is required")
    if arguments.vehicle_model_json is None:
        raise SystemExit(
            "--vehicle-model-json is required"
        )
    positive = (
        arguments.sample_step,
        arguments.integration_step,
        arguments.smooth_max_nfev,
        arguments.strict_max_nfev,
        arguments.strict_alternations,
        arguments.gtol,
        arguments.log_scale_bound,
        arguments.matrix_log_bound,
        arguments.cog_bound,
        arguments.numeric_coordinate_guard,
        arguments.lm_initial_damping_relative,
        arguments.lm_initial_trust_radius,
        arguments.lm_maximum_trust_radius,
        arguments.lm_minimum_trust_radius,
    )
    if any(
        not np.isfinite(value)
        or value <= 0.0
        for value in positive
    ):
        raise SystemExit(
            "positive numerical settings are invalid"
        )
    ratio = (
        arguments.sample_step
        / arguments.integration_step
    )
    if not np.isclose(
        ratio,
        round(ratio),
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise SystemExit(
            "sample-step must be an integer multiple "
            "of integration-step"
        )
    bounds = np.asarray(
        arguments.delay_bounds,
        dtype=float,
    )
    widths = np.asarray(
        arguments.smoothstep_width_fractions,
        dtype=float,
    )
    if (
        not (0.0 < arguments.lm_acceptance_ratio < 1.0)
        or arguments.lm_maximum_trust_radius
        < arguments.lm_initial_trust_radius
        or arguments.lm_initial_trust_radius
        < arguments.lm_minimum_trust_radius
        or bounds.shape != (2,)
        or np.any(~np.isfinite(bounds))
        or bounds[0] < 0.0
        or bounds[1] <= bounds[0]
        or widths.ndim != 1
        or widths.size == 0
        or np.any(~np.isfinite(widths))
        or np.any(widths <= 0.0)
    ):
        raise SystemExit(
            "delay/smoothstep settings are invalid"
        )


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:
    arguments = create_argument_parser().parse_args(
        argv
    )
    _validate_arguments(arguments)

    if arguments.self_test:
        model = load_vehicle_model(
            arguments.vehicle_model_json
        )
        scales = ReferenceScales.from_reference(
            model.parameters
        )
        parameterization = (
            DimensionlessParameterization(
                model.parameters,
                scales,
            )
        )
        payload = {
            "schema": SCHEMA + "/self-test",
            "nondimensionalization": scales.payload(),
            "parameterization": (
                parameterization.self_test()
            ),
        }
        print(
            json.dumps(
                _json_sanitize(payload),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if arguments.mode == "fit":
        return run_fit(arguments)
    if arguments.mode == "ablation":
        return run_ablation(arguments)
    return run_confidence_only(arguments)


if __name__ == "__main__":
    sys.exit(main())
