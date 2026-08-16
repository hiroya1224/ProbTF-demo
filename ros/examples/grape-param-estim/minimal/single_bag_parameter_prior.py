#!/usr/bin/env python3
"""Optional Gaussian factors on the identifiable physical parameter quotient."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from scipy.linalg import solve_triangular

from single_bag_savgol_core import (
    COMMON_SCALE_DIRECTION,
    PHYSICAL_DIMENSION,
    SiParameterChart,
    VehicleModelInput,
)


PARAMETER_PRIOR_SCHEMA = "grape-param-estim/parameter-prior/v1"
QUOTIENT_COMPONENT_LABELS = (
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
QUOTIENT_COMPONENT_UNITS = (
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
QUANTITY_COMPONENTS = {
    "inertia_over_mass_m2": ("xx", "yy", "zz", "xy", "xz", "yz"),
    "cog_position_body_m": ("x", "y", "z"),
    "force_effectiveness_over_mass": (
        "rotor_1",
        "rotor_2",
        "rotor_3",
        "rotor_4",
    ),
}
_QUANTITY_OFFSETS = {
    "inertia_over_mass_m2": 0,
    "cog_position_body_m": 6,
    "force_effectiveness_over_mass": 9,
}
_INERTIA_COMPONENTS = (
    (0, 0),
    (1, 1),
    (2, 2),
    (0, 1),
    (0, 2),
    (1, 2),
)


def _readonly(value: Any) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class GaussianPriorFactorSpec:
    name: str
    quantity: str
    components: tuple[str, ...]
    target_source: Optional[str]
    target_value: Optional[np.ndarray]
    standard_deviation: Optional[np.ndarray]
    covariance: np.ndarray


@dataclass(frozen=True)
class ResolvedGaussianPriorFactor:
    name: str
    quantity: str
    components: tuple[str, ...]
    quotient_indices: tuple[int, ...]
    target_source: Optional[str]
    target: np.ndarray
    covariance: np.ndarray
    standard_deviation: np.ndarray
    cholesky: np.ndarray


@dataclass(frozen=True)
class PriorFactorEvaluation:
    name: str
    quantity: str
    components: tuple[str, ...]
    quotient_indices: tuple[int, ...]
    value: np.ndarray
    target: np.ndarray
    error: np.ndarray
    standardized_residual: np.ndarray
    jacobian: np.ndarray
    objective: float
    row_start: int
    row_stop: int


@dataclass(frozen=True)
class PriorEvaluation:
    residual: np.ndarray
    jacobian: np.ndarray
    quotient_value: np.ndarray
    quotient_jacobian: np.ndarray
    factors: tuple[PriorFactorEvaluation, ...]

    @property
    def objective(self) -> float:
        return 0.5 * float(self.residual @ self.residual)


@dataclass(frozen=True)
class ParameterPrior:
    schema: str
    name: str
    description: str
    role: str
    source_path: Path
    source_sha256: str
    factors: tuple[ResolvedGaussianPriorFactor, ...]

    def evaluate(
        self, chart: SiParameterChart, coordinate: Sequence[float]
    ) -> PriorEvaluation:
        quotient, quotient_jacobian = quotient_value_and_jacobian(
            chart, coordinate
        )
        residual_rows: list[np.ndarray] = []
        jacobian_rows: list[np.ndarray] = []
        evaluations: list[PriorFactorEvaluation] = []
        start = 0
        for factor in self.factors:
            indices = np.asarray(factor.quotient_indices, dtype=int)
            value = quotient[indices]
            raw_jacobian = quotient_jacobian[indices]
            error = value - factor.target
            residual = solve_triangular(
                factor.cholesky, error, lower=True, check_finite=False
            )
            jacobian = solve_triangular(
                factor.cholesky,
                raw_jacobian,
                lower=True,
                check_finite=False,
            )
            stop = start + residual.size
            evaluations.append(
                PriorFactorEvaluation(
                    name=factor.name,
                    quantity=factor.quantity,
                    components=factor.components,
                    quotient_indices=factor.quotient_indices,
                    value=_readonly(value),
                    target=factor.target,
                    error=_readonly(error),
                    standardized_residual=_readonly(residual),
                    jacobian=_readonly(jacobian),
                    objective=0.5 * float(residual @ residual),
                    row_start=start,
                    row_stop=stop,
                )
            )
            residual_rows.append(residual)
            jacobian_rows.append(jacobian)
            start = stop
        combined_residual = np.concatenate(residual_rows)
        combined_jacobian = np.vstack(jacobian_rows)
        gauge_response = combined_jacobian @ COMMON_SCALE_DIRECTION
        tolerance = (
            500.0
            * np.finfo(float).eps
            * max(
                float(np.linalg.norm(combined_jacobian))
                * float(np.linalg.norm(COMMON_SCALE_DIRECTION)),
                1.0,
            )
        )
        if np.linalg.norm(gauge_response) > tolerance:
            raise RuntimeError(
                "parameter prior materially responds to common-scale gauge"
            )
        return PriorEvaluation(
            residual=_readonly(combined_residual),
            jacobian=_readonly(combined_jacobian),
            quotient_value=quotient,
            quotient_jacobian=quotient_jacobian,
            factors=tuple(evaluations),
        )

    def metadata(self, evaluation: PriorEvaluation) -> dict[str, Any]:
        return {
            "active": True,
            "schema": self.schema,
            "name": self.name,
            "description": self.description,
            "role": self.role,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "quotient_component_labels": QUOTIENT_COMPONENT_LABELS,
            "quotient_component_units": QUOTIENT_COMPONENT_UNITS,
            "resolved_factors": [
                {
                    "name": factor.name,
                    "quantity": factor.quantity,
                    "components": factor.components,
                    "quotient_indices": factor.quotient_indices,
                    "target_source": factor.target_source,
                    "target": factor.target,
                    "covariance": factor.covariance,
                    "std": factor.standard_deviation,
                }
                for factor in self.factors
            ],
            "factor_offsets": [
                {
                    "factor_name": item.name,
                    "row_start": item.row_start,
                    "row_stop": item.row_stop,
                }
                for item in evaluation.factors
            ],
            "factor_evaluations": [
                {
                    "factor_name": item.name,
                    "quantity": item.quantity,
                    "components": item.components,
                    "quotient_indices": item.quotient_indices,
                    "physical_value": item.value,
                    "physical_target": item.target,
                    "physical_error": item.error,
                    "standardized_residual": item.standardized_residual,
                    "factor_objective": item.objective,
                }
                for item in evaluation.factors
            ],
        }


def quotient_value_and_jacobian(
    chart: SiParameterChart, coordinate: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return `[J/m, CoG, f/m]` and its exact analytic chart Jacobian."""

    parameters, derivative = chart.decode_with_jacobian(coordinate)
    mass = float(parameters.mass)
    inertia = np.asarray(parameters.inertia, dtype=float)
    force = np.asarray(parameters.force_effectiveness, dtype=float)
    mass_jacobian = np.asarray(derivative.mass, dtype=float)
    inertia_jacobian = np.asarray(derivative.inertia, dtype=float)
    force_jacobian = np.asarray(derivative.force_effectiveness, dtype=float)
    inertia_over_mass = inertia / mass
    force_over_mass = force / mass
    inertia_rows = []
    inertia_values = []
    for row, column in _INERTIA_COMPONENTS:
        inertia_values.append(inertia_over_mass[row, column])
        inertia_rows.append(
            inertia_jacobian[row, column] / mass
            - inertia[row, column] * mass_jacobian / mass**2
        )
    force_rows = (
        force_jacobian / mass
        - np.outer(force, mass_jacobian) / mass**2
    )
    value = np.concatenate(
        (
            np.asarray(inertia_values),
            np.asarray(parameters.cog_offset, dtype=float),
            force_over_mass,
        )
    )
    jacobian = np.vstack(
        (
            np.asarray(inertia_rows),
            np.asarray(derivative.cog_offset, dtype=float),
            force_rows,
        )
    )
    if (
        value.shape != (13,)
        or jacobian.shape != (13, PHYSICAL_DIMENSION)
        or np.any(~np.isfinite(value))
        or np.any(~np.isfinite(jacobian))
    ):
        raise FloatingPointError("physical quotient evaluation is non-finite")
    return _readonly(value), _readonly(jacobian)


def quotient_value(chart: SiParameterChart, coordinate: Sequence[float]) -> np.ndarray:
    return quotient_value_and_jacobian(chart, coordinate)[0]


def _read_prior_object(path: Path) -> tuple[Path, dict[str, Any], str]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("parameter-prior JSON cannot be read: {}".format(source)) from error
    if not isinstance(value, dict):
        raise ValueError("parameter-prior JSON root must be an object")
    return source, value, _source_sha256(source)


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(label))
    return value.strip()


def _parse_factor(raw: Any, index: int) -> GaussianPriorFactorSpec:
    if not isinstance(raw, dict):
        raise ValueError("prior factor {} must be an object".format(index))
    name = _nonempty_string(raw.get("name"), "factor name")
    quantity = _nonempty_string(raw.get("quantity"), "factor quantity")
    if quantity not in QUANTITY_COMPONENTS:
        raise ValueError("unknown prior quantity: {}".format(quantity))
    components_raw = raw.get("components")
    if not isinstance(components_raw, list) or not components_raw:
        raise ValueError("factor components must be a non-empty list")
    if any(not isinstance(item, str) for item in components_raw):
        raise ValueError("factor component names must be strings")
    components = tuple(components_raw)
    if len(set(components)) != len(components):
        raise ValueError("factor components must not contain duplicates")
    unknown = [
        component
        for component in components
        if component not in QUANTITY_COMPONENTS[quantity]
    ]
    if unknown:
        raise ValueError(
            "unknown component(s) for {}: {}".format(
                quantity, ", ".join(unknown)
            )
        )
    target_raw = raw.get("target")
    if not isinstance(target_raw, dict):
        raise ValueError("factor target must be an object")
    has_source = "source" in target_raw
    has_value = "value" in target_raw
    if has_source == has_value:
        raise ValueError("factor target requires exactly one of source or value")
    target_source: Optional[str] = None
    target_value: Optional[np.ndarray] = None
    if has_source:
        target_source = _nonempty_string(target_raw["source"], "target source")
        if target_source != "vehicle_model_nominal":
            raise ValueError("unknown prior target source: {}".format(target_source))
    else:
        target_value = np.asarray(target_raw["value"], dtype=float)
        if target_value.shape != (len(components),) or np.any(
            ~np.isfinite(target_value)
        ):
            raise ValueError("explicit prior target has invalid dimensions")
        target_value = _readonly(target_value)
    has_std = "std" in raw
    has_covariance = "covariance" in raw
    if has_std == has_covariance:
        raise ValueError("factor requires exactly one of std or covariance")
    standard_deviation: Optional[np.ndarray] = None
    if has_std:
        standard_deviation = np.asarray(raw["std"], dtype=float)
        if standard_deviation.shape != (len(components),) or np.any(
            ~np.isfinite(standard_deviation)
        ) or np.any(standard_deviation <= 0.0):
            raise ValueError("prior std must contain positive finite values")
        covariance = np.diag(standard_deviation**2)
        standard_deviation = _readonly(standard_deviation)
    else:
        covariance = np.asarray(raw["covariance"], dtype=float)
        dimension = len(components)
        if covariance.shape != (dimension, dimension) or np.any(
            ~np.isfinite(covariance)
        ):
            raise ValueError("prior covariance has invalid dimensions")
        scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
        tolerance = dimension * np.finfo(float).eps * scale
        if not np.allclose(
            covariance, covariance.T, rtol=0.0, atol=tolerance
        ):
            raise ValueError("prior covariance must be symmetric")
        covariance = 0.5 * (covariance + covariance.T)
        standard_deviation = _readonly(np.sqrt(np.diag(covariance)))
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("prior covariance must be strictly positive definite") from error
    return GaussianPriorFactorSpec(
        name=name,
        quantity=quantity,
        components=components,
        target_source=target_source,
        target_value=target_value,
        standard_deviation=standard_deviation,
        covariance=_readonly(covariance),
    )


def load_parameter_prior(path: Path, model: VehicleModelInput) -> ParameterPrior:
    """Parse and resolve one v1 prior against the loaded vehicle model."""

    source, raw, digest = _read_prior_object(path)
    if raw.get("schema") != PARAMETER_PRIOR_SCHEMA:
        raise ValueError("parameter-prior schema must be {}".format(PARAMETER_PRIOR_SCHEMA))
    name = _nonempty_string(raw.get("name"), "prior name")
    description = _nonempty_string(raw.get("description"), "prior description")
    role = _nonempty_string(raw.get("role"), "prior role")
    factors_raw = raw.get("factors")
    if not isinstance(factors_raw, list) or not factors_raw:
        raise ValueError("parameter prior must contain at least one factor")
    specs = tuple(_parse_factor(item, index) for index, item in enumerate(factors_raw))
    factor_names = [factor.name for factor in specs]
    if len(set(factor_names)) != len(factor_names):
        raise ValueError("prior factor names must be unique")
    chart = SiParameterChart(model.parameters)
    nominal = quotient_value(chart, np.zeros(PHYSICAL_DIMENSION))
    resolved: list[ResolvedGaussianPriorFactor] = []
    for spec in specs:
        component_order = QUANTITY_COMPONENTS[spec.quantity]
        offset = _QUANTITY_OFFSETS[spec.quantity]
        indices = tuple(
            offset + component_order.index(component)
            for component in spec.components
        )
        target = (
            nominal[np.asarray(indices, dtype=int)]
            if spec.target_source == "vehicle_model_nominal"
            else np.asarray(spec.target_value, dtype=float)
        )
        cholesky = np.linalg.cholesky(spec.covariance)
        resolved.append(
            ResolvedGaussianPriorFactor(
                name=spec.name,
                quantity=spec.quantity,
                components=spec.components,
                quotient_indices=indices,
                target_source=spec.target_source,
                target=_readonly(target),
                covariance=spec.covariance,
                standard_deviation=_readonly(spec.standard_deviation),
                cholesky=_readonly(cholesky),
            )
        )
    prior = ParameterPrior(
        schema=PARAMETER_PRIOR_SCHEMA,
        name=name,
        description=description,
        role=role,
        source_path=source,
        source_sha256=digest,
        factors=tuple(resolved),
    )
    # Resolve-time gauge audit at the vehicle-model chart origin.
    prior.evaluate(chart, np.zeros(PHYSICAL_DIMENSION))
    return prior


def prior_target_vectors(
    prior: ParameterPrior, evaluation: PriorEvaluation
) -> tuple[np.ndarray, np.ndarray]:
    """Map factor errors and standardized residuals into the stable 13-D order."""

    error = np.full(13, np.nan)
    standardized = np.full(13, np.nan)
    for item in evaluation.factors:
        indices = np.asarray(item.quotient_indices, dtype=int)
        error[indices] = item.error
        standardized[indices] = item.standardized_residual
    return _readonly(error), _readonly(standardized)
