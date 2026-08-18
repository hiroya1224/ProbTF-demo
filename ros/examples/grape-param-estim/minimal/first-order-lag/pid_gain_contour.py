#!/usr/bin/env python3
"""Newton/continuation PID stability contours for the first-order Gimbalrotor model.

The closed-loop model is the full 26-state sampled-data system used by the
first-order-lag pole validation.  For one selected PID gain group, all other
PID groups remain fixed at the recorded gains of the analyzed bag.

The expensive nonlinear one-step Jacobian is first audited for affine
dependence on the selected raw P/I/D gains.  Samples that pass the audit are
represented as

    F(P,I,D) = F_base + (P-P0) F_P + (I-I0) F_I + (D-D0) F_D.

This representation is then used for exact 26x26 eigenvalue calculations.
Any sample that fails the affine audit falls back to the original full
26-state Jacobian evaluator at every queried gain point; it is never silently
discarded.

For N plant samples and alpha in (0,1], define mu_n = log rho(F_n) and let
q_alpha be the ceil(alpha*N)-th order statistic of mu_n.  The empirical robust
region is q_alpha < 0.  Each 2-D PI/ID/DP projection minimizes q_alpha over the
hidden gain, and its boundary Phi=0 is traced by predictor-corrector Newton
continuation.  A small coarse grid is used only to discover connected
components and to provide light interior shading; it is not the boundary
approximation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.optimize import minimize_scalar

_HERE = Path(__file__).resolve().parent
_MINIMAL = _HERE.parent
_PROJECT_ROOT = _MINIMAL.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
for _path in (_HERE, _MINIMAL, _SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from core import (  # noqa: E402
    COVARIANCE_NAMES,
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
from single_bag_savgol_reports import source_commit, write_json  # noqa: E402


CONTOUR_SCHEMA = "grape-param-estim/first-order-lag-pid-gain-contour/v1"
DEFAULT_ALPHA = 0.95
DEFAULT_SAMPLE_COUNT = 512
DEFAULT_SEED = 0
DEFAULT_GROUP = "roll_pitch"
DEFAULT_SCALE_MIN = 0.35
DEFAULT_SCALE_MAX = 3.0
DEFAULT_AFFINE_BASIS_RATIO = 1.12
DEFAULT_SEED_GRID_SIZE = 9
DEFAULT_HIDDEN_SEED_COUNT = 9
DEFAULT_MAX_CONTOUR_POINTS = 600
DEFAULT_WORKERS = min(12, os.cpu_count() or 1)
PROJECTION_SPECS = (
    ("pi", 0, 1, 2, "P", "I", "D"),
    ("id", 1, 2, 0, "I", "D", "P"),
    ("dp", 2, 0, 1, "D", "P", "I"),
)

_EPS = np.finfo(float).eps
_AFFINE_AUDIT_TOLERANCE = float(4096.0 * _EPS ** (2.0 / 3.0))
_BOUNDARY_VALUE_TOLERANCE = float(512.0 * _EPS ** (2.0 / 3.0))
_COORDINATE_TOLERANCE = float(np.sqrt(_EPS))
_ORDER_SWITCH_TOLERANCE = float(1024.0 * _EPS ** (2.0 / 3.0))


@dataclass(frozen=True)
class GainTriple:
    p: float
    i: float
    d: float

    def array(self) -> np.ndarray:
        result = np.asarray((self.p, self.i, self.d), dtype=float)
        if result.shape != (3,) or np.any(~np.isfinite(result)) or np.any(result <= 0.0):
            raise ValueError("contour PID gains must be finite and strictly positive")
        return result


@dataclass(frozen=True)
class AffineSampleResult:
    index: int
    base_matrix: np.ndarray
    gain_derivatives: np.ndarray
    audit_relative_errors: np.ndarray
    affine_valid: bool
    piecewise_near_kink: bool


@dataclass(frozen=True)
class QuantileEvaluation:
    value: float
    stable_fraction: float
    active_sample: int
    log_spectral_radius: np.ndarray
    gradient: Optional[np.ndarray]
    analytic_gradient: bool


@dataclass(frozen=True)
class ProjectionPoint:
    coordinate: np.ndarray
    value: float
    hidden_coordinate: float
    hidden_gain: float
    active_sample: int
    projected_gradient: Optional[np.ndarray]
    analytic_gradient: bool


@dataclass(frozen=True)
class ProjectionResult:
    name: str
    first_axis: int
    second_axis: int
    hidden_axis: int
    coarse_axis: np.ndarray
    coarse_value: np.ndarray
    components: tuple[np.ndarray, ...]
    hidden_components: tuple[np.ndarray, ...]
    active_sample_components: tuple[np.ndarray, ...]


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
    values = np.asarray([gains[group].array() for group in PID_GROUPS], dtype=float)
    return apply_pid_gain_configuration(
        ControllerConfig.grape(), PidGainConfiguration(values)
    )


def _replace_group(
    gains: Mapping[str, GainTriple], group: str, triple: GainTriple
) -> Mapping[str, GainTriple]:
    result = dict(gains)
    result[group] = triple
    return result


def _inputs(vehicle_model: Any, actuator_parameters: Any, gains: Mapping[str, GainTriple]) -> Any:
    return SimpleNamespace(
        vehicle_model=vehicle_model,
        actuator_parameters=actuator_parameters,
        controller_configuration=_configuration(gains),
    )


def _exact_matrix(
    *,
    plant: ScaleFreePlant,
    vehicle_model: Any,
    actuator_parameters: Any,
    controller_dt: float,
    gains: Mapping[str, GainTriple],
) -> tuple[np.ndarray, bool]:
    result = _analyze_plant(
        scale_free=plant,
        inputs=_inputs(vehicle_model, actuator_parameters, gains),
        controller_dt=float(controller_dt),
        delay=decompose_thrust_delay(0.0, float(controller_dt)),
        fd_check=False,
    )
    matrix = result.get("jacobian")
    trim = result.get("trim")
    if matrix is None or trim is None or not bool(trim.equilibrium_valid):
        raise RuntimeError("full 26-state pole Jacobian is unavailable at the requested gain")
    selected = np.asarray(matrix, dtype=float)
    if selected.shape != (26, 26) or np.any(~np.isfinite(selected)):
        raise RuntimeError("first-order closed-loop Jacobian must be finite 26x26")
    return selected, bool(trim.piecewise_linearization_near_kink)


def _audit_triples(
    base: GainTriple,
    overlay: Optional[GainTriple],
    scale_min: float,
    scale_max: float,
) -> tuple[GainTriple, GainTriple]:
    center = base.array()
    midpoint = math.sqrt(float(scale_min) * float(scale_max))
    first = (
        overlay
        if overlay is not None
        else GainTriple(*(center * np.asarray((scale_min, midpoint, scale_max))))
    )
    second = GainTriple(*(center * np.asarray((scale_max, midpoint, scale_min))))
    return first, second


def _relative_matrix_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    numerator = float(np.linalg.norm(actual - predicted, ord="fro"))
    denominator = float(np.linalg.norm(actual, ord="fro"))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _build_affine_sample(task: tuple[Any, ...]) -> AffineSampleResult:
    (
        index,
        plant,
        vehicle_model,
        actuator_parameters,
        controller_dt,
        baseline_gains,
        group,
        overlay,
        scale_min,
        scale_max,
        basis_ratio,
    ) = task
    baseline = baseline_gains[group]
    base_values = baseline.array()
    base_matrix, piecewise = _exact_matrix(
        plant=plant,
        vehicle_model=vehicle_model,
        actuator_parameters=actuator_parameters,
        controller_dt=controller_dt,
        gains=baseline_gains,
    )
    derivatives = np.empty((3, 26, 26), dtype=float)
    for axis in range(3):
        perturbed = base_values.copy()
        perturbed[axis] *= float(basis_ratio)
        delta = float(perturbed[axis] - base_values[axis])
        if delta == 0.0:
            raise RuntimeError("affine basis perturbation vanished")
        matrix, near_kink = _exact_matrix(
            plant=plant,
            vehicle_model=vehicle_model,
            actuator_parameters=actuator_parameters,
            controller_dt=controller_dt,
            gains=_replace_group(baseline_gains, group, GainTriple(*perturbed)),
        )
        derivatives[axis] = (matrix - base_matrix) / delta
        piecewise = piecewise or near_kink

    errors = []
    for triple in _audit_triples(baseline, overlay, scale_min, scale_max):
        values = triple.array()
        actual, near_kink = _exact_matrix(
            plant=plant,
            vehicle_model=vehicle_model,
            actuator_parameters=actuator_parameters,
            controller_dt=controller_dt,
            gains=_replace_group(baseline_gains, group, triple),
        )
        predicted = base_matrix + np.einsum(
            "j,jkl->kl", values - base_values, derivatives
        )
        errors.append(_relative_matrix_error(actual, predicted))
        piecewise = piecewise or near_kink
    audit = np.asarray(errors, dtype=float)
    valid = bool(
        np.all(np.isfinite(audit))
        and float(np.max(audit)) <= _AFFINE_AUDIT_TOLERANCE
    )
    return AffineSampleResult(
        index=int(index),
        base_matrix=base_matrix,
        gain_derivatives=derivatives,
        audit_relative_errors=audit,
        affine_valid=valid,
        piecewise_near_kink=piecewise,
    )


class RobustGainEvaluator:
    def __init__(
        self,
        *,
        plants: Sequence[ScaleFreePlant],
        sample_results: Sequence[AffineSampleResult],
        baseline_gains: Mapping[str, GainTriple],
        group: str,
        vehicle_model: Any,
        actuator_parameters: Any,
        controller_dt: float,
        alpha: float,
    ) -> None:
        self.plants = tuple(plants)
        ordered = sorted(sample_results, key=lambda item: item.index)
        if len(ordered) != len(self.plants) or any(
            item.index != index for index, item in enumerate(ordered)
        ):
            raise ValueError("affine sample results do not align with plant samples")
        self.sample_results = tuple(ordered)
        self.base_matrix = np.asarray([item.base_matrix for item in ordered], dtype=float)
        self.derivative = np.asarray(
            [item.gain_derivatives for item in ordered], dtype=float
        )
        self.affine_valid = np.asarray([item.affine_valid for item in ordered], dtype=bool)
        self.baseline_gains = dict(baseline_gains)
        self.group = str(group)
        self.base_gain = self.baseline_gains[self.group].array()
        self.vehicle_model = vehicle_model
        self.actuator_parameters = actuator_parameters
        self.controller_dt = float(controller_dt)
        self.alpha = float(alpha)
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must lie in (0,1]")
        self.order = int(math.ceil(self.alpha * len(self.plants))) - 1
        self._value_cache: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}

    @property
    def fallback_count(self) -> int:
        return int(np.count_nonzero(~self.affine_valid))

    def gain_from_log_ratio(self, log_ratio: Sequence[float]) -> np.ndarray:
        selected = np.asarray(log_ratio, dtype=float)
        if selected.shape != (3,) or np.any(~np.isfinite(selected)):
            raise ValueError("log gain ratio must be finite length three")
        return self.base_gain * np.exp(selected)

    def _matrices(self, log_ratio: Sequence[float]) -> np.ndarray:
        gain = self.gain_from_log_ratio(log_ratio)
        delta = gain - self.base_gain
        matrices = self.base_matrix + np.einsum("j,njkl->nkl", delta, self.derivative)
        fallback = np.flatnonzero(~self.affine_valid)
        if fallback.size:
            triple = GainTriple(*gain)
            gains = _replace_group(self.baseline_gains, self.group, triple)
            for index in fallback:
                matrices[index], _near_kink = _exact_matrix(
                    plant=self.plants[int(index)],
                    vehicle_model=self.vehicle_model,
                    actuator_parameters=self.actuator_parameters,
                    controller_dt=self.controller_dt,
                    gains=gains,
                )
        return matrices

    def _log_radii(self, log_ratio: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        key = tuple(float(value) for value in np.asarray(log_ratio, dtype=float))
        cached = self._value_cache.get(key)
        if cached is not None:
            return cached[0].copy(), cached[1].copy()
        matrices = self._matrices(key)
        eigenvalues = np.linalg.eigvals(matrices)
        radii = np.max(np.abs(eigenvalues), axis=1)
        if np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
            raise FloatingPointError("closed-loop spectral radius became invalid")
        log_radius = np.log(radii)
        self._value_cache[key] = (log_radius.copy(), matrices.copy())
        return log_radius, matrices

    def stable_fraction(self, log_ratio: Sequence[float]) -> float:
        value, _matrix = self._log_radii(log_ratio)
        return float(np.mean(value < 0.0))

    def _finite_difference_gradient(self, log_ratio: np.ndarray) -> np.ndarray:
        step = float(_EPS ** (1.0 / 3.0))
        result = np.empty(3, dtype=float)
        for axis in range(3):
            plus = log_ratio.copy()
            minus = log_ratio.copy()
            plus[axis] += step
            minus[axis] -= step
            result[axis] = (
                self.quantile(plus, want_gradient=False).value
                - self.quantile(minus, want_gradient=False).value
            ) / (2.0 * step)
        return result

    def quantile(
        self, log_ratio: Sequence[float], *, want_gradient: bool = True
    ) -> QuantileEvaluation:
        selected = np.asarray(log_ratio, dtype=float)
        log_radius, matrices = self._log_radii(selected)
        partition = np.argpartition(log_radius, self.order)
        active = int(partition[self.order])
        q = float(log_radius[active])
        stable_fraction = float(np.mean(log_radius < 0.0))
        if not want_gradient:
            return QuantileEvaluation(
                q, stable_fraction, active, log_radius, None, False
            )

        ordered = np.sort(log_radius)
        neighbor_gap = float("inf")
        if self.order > 0:
            neighbor_gap = min(neighbor_gap, abs(ordered[self.order] - ordered[self.order - 1]))
        if self.order + 1 < ordered.size:
            neighbor_gap = min(neighbor_gap, abs(ordered[self.order + 1] - ordered[self.order]))
        if neighbor_gap <= _ORDER_SWITCH_TOLERANCE * max(1.0, abs(q)):
            gradient = self._finite_difference_gradient(selected)
            return QuantileEvaluation(
                q, stable_fraction, active, log_radius, gradient, False
            )

        if not self.affine_valid[active]:
            gradient = self._finite_difference_gradient(selected)
            return QuantileEvaluation(
                q, stable_fraction, active, log_radius, gradient, False
            )

        matrix = matrices[active]
        values, left, right = linalg.eig(matrix, left=True, right=True)
        magnitude = np.abs(values)
        maximum = float(np.max(magnitude))
        candidates = np.flatnonzero(
            np.abs(magnitude - maximum)
            <= _ORDER_SWITCH_TOLERANCE * max(1.0, maximum)
        )
        if candidates.size > 2:
            gradient = self._finite_difference_gradient(selected)
            return QuantileEvaluation(
                q, stable_fraction, active, log_radius, gradient, False
            )
        if candidates.size == 2:
            first, second = values[candidates[0]], values[candidates[1]]
            if not np.isclose(
                first,
                np.conjugate(second),
                rtol=0.0,
                atol=_ORDER_SWITCH_TOLERANCE * max(1.0, maximum),
            ):
                gradient = self._finite_difference_gradient(selected)
                return QuantileEvaluation(
                    q, stable_fraction, active, log_radius, gradient, False
                )
        eigen_index = int(candidates[np.argmax(values[candidates].imag)])
        eigenvalue = values[eigen_index]
        left_vector = left[:, eigen_index]
        right_vector = right[:, eigen_index]
        denominator = np.vdot(left_vector, right_vector)
        denominator_scale = float(
            np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
        )
        if (
            eigenvalue == 0.0
            or abs(denominator)
            <= np.sqrt(_EPS) * max(1.0, denominator_scale)
        ):
            gradient = self._finite_difference_gradient(selected)
            return QuantileEvaluation(
                q, stable_fraction, active, log_radius, gradient, False
            )

        gain = self.gain_from_log_ratio(selected)
        gradient = np.empty(3, dtype=float)
        for axis in range(3):
            d_matrix = gain[axis] * self.derivative[active, axis]
            d_eigenvalue = (
                np.vdot(left_vector, d_matrix @ right_vector) / denominator
            )
            gradient[axis] = float(np.real(d_eigenvalue / eigenvalue))
        if np.any(~np.isfinite(gradient)):
            gradient = self._finite_difference_gradient(selected)
            analytic = False
        else:
            analytic = True
        return QuantileEvaluation(
            q, stable_fraction, active, log_radius, gradient, analytic
        )


class ProjectionEvaluator:
    def __init__(
        self,
        robust: RobustGainEvaluator,
        *,
        first_axis: int,
        second_axis: int,
        hidden_axis: int,
        lower: float,
        upper: float,
        hidden_seed_count: int,
    ) -> None:
        self.robust = robust
        self.first_axis = int(first_axis)
        self.second_axis = int(second_axis)
        self.hidden_axis = int(hidden_axis)
        self.lower = float(lower)
        self.upper = float(upper)
        self.hidden_seed_count = int(hidden_seed_count)
        self._cache: dict[tuple[float, float], ProjectionPoint] = {}

    def _full_coordinate(self, visible: Sequence[float], hidden: float) -> np.ndarray:
        point = np.zeros(3, dtype=float)
        point[self.first_axis] = float(visible[0])
        point[self.second_axis] = float(visible[1])
        point[self.hidden_axis] = float(hidden)
        return point

    def _hidden_minimum(self, visible: np.ndarray) -> tuple[float, QuantileEvaluation]:
        grid = np.linspace(self.lower, self.upper, self.hidden_seed_count)
        values = np.asarray(
            [
                self.robust.quantile(
                    self._full_coordinate(visible, hidden), want_gradient=False
                ).value
                for hidden in grid
            ],
            dtype=float,
        )
        candidates: list[tuple[float, float]] = [
            (float(values[0]), float(grid[0])),
            (float(values[-1]), float(grid[-1])),
        ]
        xatol = _COORDINATE_TOLERANCE * max(1.0, self.upper - self.lower)
        for index in range(1, grid.size - 1):
            if values[index] <= values[index - 1] and values[index] <= values[index + 1]:
                solved = minimize_scalar(
                    lambda hidden: self.robust.quantile(
                        self._full_coordinate(visible, hidden),
                        want_gradient=False,
                    ).value,
                    bounds=(float(grid[index - 1]), float(grid[index + 1])),
                    method="bounded",
                    options={"xatol": xatol},
                )
                if solved.success and np.isfinite(solved.fun):
                    candidates.append((float(solved.fun), float(solved.x)))
                else:
                    candidates.append((float(values[index]), float(grid[index])))
        best_value, best_hidden = min(candidates, key=lambda item: item[0])
        evaluation = self.robust.quantile(
            self._full_coordinate(visible, best_hidden), want_gradient=True
        )
        if not np.isclose(
            best_value,
            evaluation.value,
            rtol=0.0,
            atol=_BOUNDARY_VALUE_TOLERANCE * max(1.0, abs(best_value)),
        ):
            best_value = evaluation.value
        return best_hidden, evaluation

    def _finite_difference_gradient(self, visible: np.ndarray) -> np.ndarray:
        step = float(_EPS ** (1.0 / 3.0))
        result = np.empty(2, dtype=float)
        for axis in range(2):
            plus = visible.copy()
            minus = visible.copy()
            plus[axis] += step
            minus[axis] -= step
            result[axis] = (
                self.evaluate(plus, want_gradient=False).value
                - self.evaluate(minus, want_gradient=False).value
            ) / (2.0 * step)
        return result

    def evaluate(
        self, visible: Sequence[float], *, want_gradient: bool = True
    ) -> ProjectionPoint:
        point = np.asarray(visible, dtype=float)
        if point.shape != (2,) or np.any(~np.isfinite(point)):
            raise ValueError("projection coordinate must be finite length two")
        key = (float(point[0]), float(point[1]))
        cached = self._cache.get(key)
        if cached is not None and (not want_gradient or cached.projected_gradient is not None):
            return cached
        hidden, evaluation = self._hidden_minimum(point)
        gradient = None
        analytic = False
        if want_gradient:
            assert evaluation.gradient is not None
            projected = np.asarray(
                (
                    evaluation.gradient[self.first_axis],
                    evaluation.gradient[self.second_axis],
                ),
                dtype=float,
            )
            hidden_gradient = float(evaluation.gradient[self.hidden_axis])
            interior = (
                hidden > self.lower + _COORDINATE_TOLERANCE
                and hidden < self.upper - _COORDINATE_TOLERANCE
            )
            if (
                interior
                and abs(hidden_gradient) > _EPS ** (1.0 / 3.0)
            ) or np.any(~np.isfinite(projected)):
                projected = self._finite_difference_gradient(point)
                analytic = False
            else:
                analytic = bool(evaluation.analytic_gradient)
            gradient = projected
        result = ProjectionPoint(
            coordinate=point.copy(),
            value=float(evaluation.value),
            hidden_coordinate=float(hidden),
            hidden_gain=float(
                self.robust.base_gain[self.hidden_axis] * math.exp(hidden)
            ),
            active_sample=int(evaluation.active_sample),
            projected_gradient=None if gradient is None else gradient.copy(),
            analytic_gradient=analytic,
        )
        self._cache[key] = result
        return result


def _normal_newton_projection(
    evaluator: ProjectionEvaluator,
    initial: Sequence[float],
    lower: float,
    upper: float,
    maximum_iterations: int = 16,
) -> ProjectionPoint:
    point = np.asarray(initial, dtype=float).copy()
    for _iteration in range(maximum_iterations):
        evaluated = evaluator.evaluate(point, want_gradient=True)
        gradient = np.asarray(evaluated.projected_gradient, dtype=float)
        norm_squared = float(gradient @ gradient)
        if not np.isfinite(norm_squared) or norm_squared == 0.0:
            raise RuntimeError("projected robust boundary gradient vanished")
        if abs(evaluated.value) <= _BOUNDARY_VALUE_TOLERANCE:
            return evaluated
        point -= evaluated.value * gradient / norm_squared
        if np.any(point < lower) or np.any(point > upper):
            raise RuntimeError("Newton boundary projection left the search domain")
    final = evaluator.evaluate(point, want_gradient=True)
    if abs(final.value) > _BOUNDARY_VALUE_TOLERANCE:
        raise RuntimeError("Newton boundary projection did not converge")
    return final


def _corrector(
    evaluator: ProjectionEvaluator,
    predictor: np.ndarray,
    tangent: np.ndarray,
    lower: float,
    upper: float,
    maximum_iterations: int = 16,
) -> tuple[ProjectionPoint, int]:
    point = predictor.copy()
    for iteration in range(maximum_iterations):
        evaluated = evaluator.evaluate(point, want_gradient=True)
        gradient = np.asarray(evaluated.projected_gradient, dtype=float)
        constraint = float((point - predictor) @ tangent)
        residual = np.asarray((evaluated.value, constraint), dtype=float)
        if (
            abs(residual[0]) <= _BOUNDARY_VALUE_TOLERANCE
            and abs(residual[1]) <= _COORDINATE_TOLERANCE
        ):
            return evaluated, iteration + 1
        jacobian = np.vstack((gradient, tangent))
        determinant = float(np.linalg.det(jacobian))
        if not np.isfinite(determinant) or abs(determinant) <= np.sqrt(_EPS):
            raise RuntimeError("pseudo-arclength Newton system became singular")
        delta = np.linalg.solve(jacobian, residual)
        point -= delta
        if np.any(point < lower) or np.any(point > upper):
            raise RuntimeError("pseudo-arclength Newton corrector left the search domain")
    raise RuntimeError("pseudo-arclength Newton corrector did not converge")


def _unit_tangent(gradient: np.ndarray, preferred: Optional[np.ndarray] = None) -> np.ndarray:
    tangent = np.asarray((-gradient[1], gradient[0]), dtype=float)
    norm = float(np.linalg.norm(tangent))
    if not np.isfinite(norm) or norm == 0.0:
        raise RuntimeError("boundary tangent is undefined")
    tangent /= norm
    if preferred is not None and float(tangent @ preferred) < 0.0:
        tangent = -tangent
    return tangent


def _trace_direction(
    evaluator: ProjectionEvaluator,
    start: ProjectionPoint,
    initial_tangent: np.ndarray,
    *,
    lower: float,
    upper: float,
    initial_step: float,
    maximum_points: int,
) -> list[ProjectionPoint]:
    result = [start]
    point = start
    tangent = initial_tangent.copy()
    span = upper - lower
    step = float(initial_step)
    minimum_step = span / 5000.0
    maximum_step = span / 20.0
    cumulative = 0.0
    for _index in range(maximum_points - 1):
        accepted = False
        trial_step = step
        for _attempt in range(8):
            predictor = point.coordinate + trial_step * tangent
            if np.any(predictor < lower) or np.any(predictor > upper):
                trial_step *= 0.5
                if trial_step < minimum_step:
                    return result
                continue
            try:
                corrected, iterations = _corrector(
                    evaluator,
                    predictor,
                    tangent,
                    lower,
                    upper,
                )
            except RuntimeError:
                trial_step *= 0.5
                if trial_step < minimum_step:
                    return result
                continue
            accepted = True
            break
        if not accepted:
            return result
        displacement = corrected.coordinate - point.coordinate
        distance = float(np.linalg.norm(displacement))
        if distance <= _COORDINATE_TOLERANCE:
            return result
        cumulative += distance
        gradient = np.asarray(corrected.projected_gradient, dtype=float)
        next_tangent = _unit_tangent(gradient, tangent)
        turn_cosine = float(np.clip(tangent @ next_tangent, -1.0, 1.0))
        turn_angle = float(math.acos(turn_cosine))
        result.append(corrected)
        point = corrected
        tangent = next_tangent
        if (
            len(result) > 20
            and cumulative > 8.0 * initial_step
            and np.linalg.norm(point.coordinate - start.coordinate) < 1.5 * trial_step
        ):
            return result
        if iterations <= 3 and turn_angle < 0.08:
            step = min(trial_step * 1.25, maximum_step)
        elif iterations >= 7 or turn_angle > 0.30:
            step = max(trial_step * 0.5, minimum_step)
        else:
            step = trial_step
    return result


def _coarse_contour_segments(axis: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, ...]:
    figure, plot_axis = plt.subplots()
    try:
        contour = plot_axis.contour(axis, axis, value.T, levels=[0.0])
        segments = tuple(
            np.asarray(segment, dtype=float)
            for segment in contour.allsegs[0]
            if np.asarray(segment).shape[0] >= 2
        )
    finally:
        plt.close(figure)
    return segments


def _trace_projection(
    robust: RobustGainEvaluator,
    *,
    name: str,
    first_axis: int,
    second_axis: int,
    hidden_axis: int,
    lower: float,
    upper: float,
    seed_grid_size: int,
    hidden_seed_count: int,
    maximum_points: int,
) -> ProjectionResult:
    evaluator = ProjectionEvaluator(
        robust,
        first_axis=first_axis,
        second_axis=second_axis,
        hidden_axis=hidden_axis,
        lower=lower,
        upper=upper,
        hidden_seed_count=hidden_seed_count,
    )
    coarse_axis = np.linspace(lower, upper, int(seed_grid_size))
    coarse_value = np.empty((coarse_axis.size, coarse_axis.size), dtype=float)
    for first_index, first in enumerate(coarse_axis):
        for second_index, second in enumerate(coarse_axis):
            coarse_value[first_index, second_index] = evaluator.evaluate(
                (first, second), want_gradient=False
            ).value

    coarse_segments = _coarse_contour_segments(coarse_axis, coarse_value)
    components: list[np.ndarray] = []
    hidden_components: list[np.ndarray] = []
    active_components: list[np.ndarray] = []
    span = upper - lower
    initial_step = span / 80.0
    duplicate_tolerance = span / max(4.0 * seed_grid_size, 1.0)

    for segment in coarse_segments:
        coarse_seed = segment[0]
        if any(
            np.min(np.linalg.norm(component - coarse_seed[None, :], axis=1))
            < duplicate_tolerance
            for component in components
        ):
            continue
        try:
            start = _normal_newton_projection(
                evaluator, coarse_seed, lower, upper
            )
        except RuntimeError:
            continue
        preferred = segment[1] - segment[0]
        preferred_norm = float(np.linalg.norm(preferred))
        preferred = None if preferred_norm == 0.0 else preferred / preferred_norm
        tangent = _unit_tangent(
            np.asarray(start.projected_gradient, dtype=float), preferred
        )
        forward = _trace_direction(
            evaluator,
            start,
            tangent,
            lower=lower,
            upper=upper,
            initial_step=initial_step,
            maximum_points=maximum_points,
        )
        backward = _trace_direction(
            evaluator,
            start,
            -tangent,
            lower=lower,
            upper=upper,
            initial_step=initial_step,
            maximum_points=maximum_points,
        )
        combined = list(reversed(backward[1:])) + forward
        coordinates = np.asarray([item.coordinate for item in combined], dtype=float)
        hidden = np.asarray([item.hidden_coordinate for item in combined], dtype=float)
        active = np.asarray([item.active_sample for item in combined], dtype=int)
        if coordinates.shape[0] >= 2:
            components.append(coordinates)
            hidden_components.append(hidden)
            active_components.append(active)

    return ProjectionResult(
        name=name,
        first_axis=first_axis,
        second_axis=second_axis,
        hidden_axis=hidden_axis,
        coarse_axis=coarse_axis,
        coarse_value=coarse_value,
        components=tuple(components),
        hidden_components=tuple(hidden_components),
        active_sample_components=tuple(active_components),
    )


def _range_with_overlay(
    base: GainTriple,
    overlay: Optional[GainTriple],
    scale_min: float,
    scale_max: float,
) -> tuple[float, float]:
    lower = float(scale_min)
    upper = float(scale_max)
    if overlay is not None:
        ratio = overlay.array() / base.array()
        lower = min(lower, float(np.min(ratio)))
        upper = max(upper, float(np.max(ratio)))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or upper <= lower:
        raise ValueError("gain-ratio search range is invalid")
    return math.log(lower), math.log(upper)


def _plot_projection(
    axis: Any,
    *,
    result: ProjectionResult,
    robust: RobustGainEvaluator,
    labels: tuple[str, str, str],
    recorded_fraction: float,
    success: Optional[GainTriple],
    success_fraction: Optional[float],
) -> None:
    first = result.first_axis
    second = result.second_axis
    base = robust.base_gain
    grid_first = base[first] * np.exp(result.coarse_axis)
    grid_second = base[second] * np.exp(result.coarse_axis)
    x, y = np.meshgrid(grid_first, grid_second, indexing="ij")
    field = result.coarse_value
    if np.any(field < 0.0):
        axis.contourf(
            x,
            y,
            field,
            levels=[float(np.min(field)) - 1.0, 0.0],
            colors="none",
            hatches=["///"],
            alpha=0.0,
        )
    for component in result.components:
        axis.plot(
            base[first] * np.exp(component[:, 0]),
            base[second] * np.exp(component[:, 1]),
            linewidth=2.0,
        )
    axis.plot(
        base[first],
        base[second],
        marker="x",
        markersize=9,
        markeredgewidth=2.0,
        linestyle="none",
        label=f"recorded ({recorded_fraction:.3f})",
    )
    if success is not None:
        success_gain = success.array()
        label = "success group gain"
        if success_fraction is not None:
            label += f" ({success_fraction:.3f})"
        axis.plot(
            success_gain[first],
            success_gain[second],
            marker="o",
            markersize=6,
            linestyle="none",
            label=label,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(labels[first])
    axis.set_ylabel(labels[second])
    axis.set_title(f"{labels[first]}{labels[second]} projection")
    axis.grid(True)
    axis.legend(loc="best", fontsize=8)


def _flatten_components(
    projections: Sequence[ProjectionResult],
) -> Mapping[str, np.ndarray]:
    rows = []
    hidden = []
    active = []
    projection_id = []
    component_id = []
    for p_index, projection in enumerate(projections):
        for c_index, component in enumerate(projection.components):
            rows.append(component)
            hidden.append(projection.hidden_components[c_index])
            active.append(projection.active_sample_components[c_index])
            projection_id.append(np.full(component.shape[0], p_index, dtype=int))
            component_id.append(np.full(component.shape[0], c_index, dtype=int))
    if not rows:
        return {
            "boundary_visible_log_ratio": np.empty((0, 2), dtype=float),
            "boundary_hidden_log_ratio": np.empty(0, dtype=float),
            "boundary_active_sample": np.empty(0, dtype=int),
            "boundary_projection_id": np.empty(0, dtype=int),
            "boundary_component_id": np.empty(0, dtype=int),
        }
    return {
        "boundary_visible_log_ratio": np.concatenate(rows, axis=0),
        "boundary_hidden_log_ratio": np.concatenate(hidden, axis=0),
        "boundary_active_sample": np.concatenate(active, axis=0),
        "boundary_projection_id": np.concatenate(projection_id, axis=0),
        "boundary_component_id": np.concatenate(component_id, axis=0),
    }


def analyze(arguments: argparse.Namespace) -> Mapping[str, Any]:
    estimate_path = Path(arguments.estimate_json).expanduser().resolve()
    estimate = load_estimate_json(estimate_path)
    success_estimate = (
        None
        if arguments.success_json is None
        else load_estimate_json(Path(arguments.success_json).expanduser().resolve())
    )
    group = str(arguments.group)
    if group not in PID_GROUPS:
        raise ValueError("unknown PID group")
    baseline_gains = _recorded_gains(estimate)
    success_gains = None if success_estimate is None else _recorded_gains(success_estimate)
    overlay = None if success_gains is None else success_gains[group]
    lower, upper = _range_with_overlay(
        baseline_gains[group],
        overlay,
        float(arguments.scale_min),
        float(arguments.scale_max),
    )

    vehicle_model = load_vehicle_model(Path(estimate["input"]["vehicle_model"]))
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

    tasks = [
        (
            index,
            plant,
            vehicle_model,
            actuator_parameters,
            controller_dt,
            baseline_gains,
            group,
            overlay,
            math.exp(lower),
            math.exp(upper),
            float(arguments.affine_basis_ratio),
        )
        for index, plant in enumerate(plants)
    ]
    workers = min(int(arguments.workers), len(tasks))
    if workers <= 1:
        sample_results = [_build_affine_sample(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            sample_results = list(executor.map(_build_affine_sample, tasks, chunksize=1))

    robust = RobustGainEvaluator(
        plants=plants,
        sample_results=sample_results,
        baseline_gains=baseline_gains,
        group=group,
        vehicle_model=vehicle_model,
        actuator_parameters=actuator_parameters,
        controller_dt=controller_dt,
        alpha=float(arguments.alpha),
    )
    recorded_fraction = robust.stable_fraction(np.zeros(3))
    success_fraction = None
    success_log_ratio = None
    if overlay is not None:
        success_log_ratio = np.log(overlay.array() / robust.base_gain)
        success_fraction = robust.stable_fraction(success_log_ratio)

    projections = []
    for name, first, second, hidden, _first_label, _second_label, _hidden_label in PROJECTION_SPECS:
        projections.append(
            _trace_projection(
                robust,
                name=name,
                first_axis=first,
                second_axis=second,
                hidden_axis=hidden,
                lower=lower,
                upper=upper,
                seed_grid_size=int(arguments.seed_grid_size),
                hidden_seed_count=int(arguments.hidden_seed_count),
                maximum_points=int(arguments.max_contour_points),
            )
        )

    output = (
        Path(arguments.output_dir).expanduser().resolve()
        if arguments.output_dir is not None
        else estimate_path.parent / "pid_gain_contour" / group
    )
    output.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    labels = (f"{group} P", f"{group} I", f"{group} D")
    for axis, projection in zip(axes, projections):
        _plot_projection(
            axis,
            result=projection,
            robust=robust,
            labels=labels,
            recorded_fraction=recorded_fraction,
            success=overlay,
            success_fraction=success_fraction,
        )
    figure.suptitle(
        f"{estimate['case_name']} / {group} / exact 26-state alpha={float(arguments.alpha):.2f} boundary"
    )
    figure.savefig(output / "gain_contour.png", dpi=180)
    plt.close(figure)

    flattened = _flatten_components(projections)
    np.savez_compressed(
        output / "gain_contour.npz",
        base_gain=robust.base_gain,
        affine_valid=robust.affine_valid,
        affine_audit_relative_error=np.asarray(
            [item.audit_relative_errors for item in sorted(sample_results, key=lambda value: value.index)]
        ),
        piecewise_near_kink=np.asarray(
            [item.piecewise_near_kink for item in sorted(sample_results, key=lambda value: value.index)],
            dtype=bool,
        ),
        projection_names=np.asarray([item.name for item in projections]),
        coarse_axis_log_ratio=np.asarray([item.coarse_axis for item in projections]),
        coarse_projection_value=np.asarray([item.coarse_value for item in projections]),
        success_group_log_ratio=(
            np.full(3, np.nan) if success_log_ratio is None else success_log_ratio
        ),
        **flattened,
    )

    audit_errors = np.asarray(
        [item.audit_relative_errors for item in sample_results], dtype=float
    )
    projection_payload = []
    for projection in projections:
        projection_payload.append(
            {
                "name": projection.name,
                "visible_axes": [projection.first_axis, projection.second_axis],
                "hidden_axis": projection.hidden_axis,
                "component_count": len(projection.components),
                "boundary_point_count": int(
                    sum(component.shape[0] for component in projection.components)
                ),
                "coarse_seed_grid_size": int(projection.coarse_axis.size),
                "coarse_seed_min": float(np.min(projection.coarse_value)),
                "coarse_seed_max": float(np.max(projection.coarse_value)),
            }
        )
    payload = {
        "schema": CONTOUR_SCHEMA,
        "source_commit": source_commit(_PROJECT_ROOT),
        "case_name": str(estimate["case_name"]),
        "group": group,
        "estimate_json": str(estimate_path),
        "success_json": None if arguments.success_json is None else str(Path(arguments.success_json).expanduser().resolve()),
        "first_order_time_constant_seconds": float(
            estimate["actuator_model"]["thrust_time_constant_seconds"]
        ),
        "controller_dt_seconds": controller_dt,
        "alpha": float(arguments.alpha),
        "sample_count": int(arguments.samples),
        "covariance": str(arguments.covariance),
        "seed": int(arguments.seed),
        "method": {
            "closed_loop_model": "full_26_state_sampled_data_first_order_actuator",
            "robust_scalar": "ceil(alpha*N)-th order statistic of log spectral radius",
            "projection": "minimize robust scalar over hidden gain",
            "boundary": "Phi=0 predictor-corrector pseudo-arclength Newton continuation",
            "eigenvalue_gradient": "left/right eigenvector sensitivity when differentiable",
            "nonsmooth_gradient_fallback": "central finite difference in log-gain coordinates",
            "coarse_grid_role": "connected-component seeding and interior shading only",
            "topology_limitation": "components smaller than the coarse seed cells are not certified absent",
        },
        "gain_domain": {
            "ratio_min": float(math.exp(lower)),
            "ratio_max": float(math.exp(upper)),
            "coordinate": "log(gain / recorded_gain)",
        },
        "recorded_gain": {
            "p_gain": float(robust.base_gain[0]),
            "i_gain": float(robust.base_gain[1]),
            "d_gain": float(robust.base_gain[2]),
            "exact_stable_fraction": recorded_fraction,
        },
        "success_group_only": (
            None
            if overlay is None
            else {
                "p_gain": overlay.p,
                "i_gain": overlay.i,
                "d_gain": overlay.d,
                "exact_stable_fraction_with_other_groups_kept_at_this_bag_recorded_values": success_fraction,
            }
        ),
        "affine_matrix_audit": {
            "basis_ratio": float(arguments.affine_basis_ratio),
            "tolerance": _AFFINE_AUDIT_TOLERANCE,
            "valid_samples": int(np.count_nonzero(robust.affine_valid)),
            "fallback_samples": robust.fallback_count,
            "maximum_relative_frobenius_error": float(np.max(audit_errors)),
            "median_relative_frobenius_error": float(np.median(audit_errors)),
            "piecewise_near_kink_samples": int(
                np.count_nonzero([item.piecewise_near_kink for item in sample_results])
            ),
        },
        "projections": projection_payload,
        "files": {
            "npz": str(output / "gain_contour.npz"),
            "png": str(output / "gain_contour.png"),
        },
    }
    write_json(output / "gain_contour.json", payload)
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate-json", type=Path, required=True)
    parser.add_argument("--success-json", type=Path, default=None)
    parser.add_argument("--group", choices=PID_GROUPS, default=DEFAULT_GROUP)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--covariance", choices=COVARIANCE_NAMES, default="conservative_fusion")
    parser.add_argument("--scale-min", type=float, default=DEFAULT_SCALE_MIN)
    parser.add_argument("--scale-max", type=float, default=DEFAULT_SCALE_MAX)
    parser.add_argument("--affine-basis-ratio", type=float, default=DEFAULT_AFFINE_BASIS_RATIO)
    parser.add_argument("--seed-grid-size", type=int, default=DEFAULT_SEED_GRID_SIZE)
    parser.add_argument("--hidden-seed-count", type=int, default=DEFAULT_HIDDEN_SEED_COUNT)
    parser.add_argument("--max-contour-points", type=int, default=DEFAULT_MAX_CONTOUR_POINTS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    if not (0.0 < arguments.alpha <= 1.0):
        raise SystemExit("--alpha must lie in (0,1]")
    if arguments.samples <= 0:
        raise SystemExit("--samples must be positive")
    if arguments.scale_min <= 0.0 or arguments.scale_max <= arguments.scale_min:
        raise SystemExit("gain scale bounds must satisfy 0 < min < max")
    if arguments.affine_basis_ratio <= 1.0:
        raise SystemExit("--affine-basis-ratio must exceed one")
    if arguments.seed_grid_size < 3:
        raise SystemExit("--seed-grid-size must be at least three")
    if arguments.hidden_seed_count < 3:
        raise SystemExit("--hidden-seed-count must be at least three")
    if arguments.max_contour_points < 2:
        raise SystemExit("--max-contour-points must be at least two")
    if arguments.workers <= 0:
        raise SystemExit("--workers must be positive")
    payload = analyze(arguments)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
