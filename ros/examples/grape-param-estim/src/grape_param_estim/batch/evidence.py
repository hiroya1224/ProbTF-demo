"""Laplace evidence and reduced static-parameter geometry.

The nonlinear graph objective omits Gaussian normalization terms that are
constant for a fixed model.  They are not constant while diagonal ``Q`` is
being updated, so the Laplace-EM acceptance objective must restore them.  This
module also separates the explicit static prior from the reduced posterior
information; a likelihood ridge is never hidden by the proper prior used to
select a MAP point.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.batch.covariance import (
    ArrowheadLaplaceFactorization,
)
from grape_param_estim.batch.laplace_em import (
    DiagonalQDefinition,
)
from grape_param_estim.batch.lag_profile import LagProfileResult
from grape_param_estim.batch.ridge import (
    ReducedInformationAnalysis,
    analyze_reduced_information,
)
from grape_param_estim.parameterization import PARAMETER_DIMENSION


DELAY_GEOMETRY_METHOD = "nonuniform_three_point_map_profile_v1"
JOINT_SURROGATE_METHOD = "joint_profile_information_v1"
FALLBACK_SURROGATE_METHOD = "proposal_only_block_diagonal_fallback_v1"


def _canonical_text(value: object, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError("{} must be canonical non-empty text".format(name))
    return value


def _immutable_array(value: object) -> np.ndarray:
    result = np.asarray(value, dtype=float).copy()
    result.setflags(write=False)
    return result


def _nonuniform_three_point_weights(
    lag: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return first/second derivative weights at the middle abscissa."""

    x = np.asarray(lag, dtype=float)
    if (
        x.shape != (3,)
        or not np.all(np.isfinite(x))
        or not (x[0] < x[1] < x[2])
    ):
        raise ValueError("lag support must be three increasing finite values")
    left = x[1] - x[0]
    right = x[2] - x[1]
    first = np.asarray(
        (
            -right / (left * (left + right)),
            (right - left) / (left * right),
            left / (right * (left + right)),
        ),
        dtype=float,
    )
    second = np.asarray(
        (
            2.0 / (left * (left + right)),
            -2.0 / (left * right),
            2.0 / (right * (left + right)),
        ),
        dtype=float,
    )
    return first, second


@dataclass(frozen=True)
class DelayStaticLaplaceGeometry:
    """Audited local joint geometry of 18-D static coordinates and delay."""

    valid: bool
    method: str
    reason: str
    standard_deviation_seconds: float
    source: str
    curvature: Optional[float]
    profile_gradient: Optional[float]
    static_sensitivity: np.ndarray
    support_lag: np.ndarray
    support_map_objective: np.ndarray
    support_static_coordinate: np.ndarray
    joint_information: np.ndarray
    joint_covariance: np.ndarray
    parameter_delay_cross_covariance: np.ndarray
    mcmc_quadratic_surrogate_method: str

    def __post_init__(self) -> None:
        if not isinstance(self.valid, (bool, np.bool_)):
            raise TypeError("valid must be boolean")
        valid = bool(self.valid)
        for name in (
            "method",
            "reason",
            "source",
            "mcmc_quadratic_surrogate_method",
        ):
            object.__setattr__(
                self, name, _canonical_text(getattr(self, name), name)
            )
        standard_deviation = float(self.standard_deviation_seconds)
        if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
            raise ValueError("standard_deviation_seconds must be positive")
        object.__setattr__(
            self, "standard_deviation_seconds", standard_deviation
        )
        curvature = None if self.curvature is None else float(self.curvature)
        gradient = (
            None if self.profile_gradient is None else float(self.profile_gradient)
        )
        arrays = {
            name: _immutable_array(getattr(self, name))
            for name in (
                "static_sensitivity",
                "support_lag",
                "support_map_objective",
                "support_static_coordinate",
                "joint_information",
                "joint_covariance",
                "parameter_delay_cross_covariance",
            )
        }
        for name, value in arrays.items():
            if not np.all(np.isfinite(value)):
                raise ValueError("{} must be finite".format(name))
            object.__setattr__(self, name, value)
        if valid:
            if self.method != DELAY_GEOMETRY_METHOD:
                raise ValueError("valid geometry uses an unknown method")
            if self.reason != "valid":
                raise ValueError("valid geometry reason must be canonical 'valid'")
            if self.mcmc_quadratic_surrogate_method != JOINT_SURROGATE_METHOD:
                raise ValueError("valid geometry must use joint MCMC information")
            if curvature is None or not np.isfinite(curvature) or curvature <= 0.0:
                raise ValueError("valid geometry requires positive curvature")
            if gradient is None or not np.isfinite(gradient):
                raise ValueError("valid geometry requires a finite profile gradient")
            expected_shapes = {
                "static_sensitivity": (PARAMETER_DIMENSION,),
                "support_lag": (3,),
                "support_map_objective": (3,),
                "support_static_coordinate": (3, PARAMETER_DIMENSION),
                "joint_information": (
                    PARAMETER_DIMENSION + 1,
                    PARAMETER_DIMENSION + 1,
                ),
                "joint_covariance": (
                    PARAMETER_DIMENSION + 1,
                    PARAMETER_DIMENSION + 1,
                ),
                "parameter_delay_cross_covariance": (PARAMETER_DIMENSION,),
            }
            for name, shape in expected_shapes.items():
                if arrays[name].shape != shape:
                    raise ValueError("{} has invalid shape".format(name))
            first, second = _nonuniform_three_point_weights(
                arrays["support_lag"]
            )
            derived_gradient = float(first @ arrays["support_map_objective"])
            derived_curvature = float(second @ arrays["support_map_objective"])
            derived_sensitivity = first @ arrays["support_static_coordinate"]
            if not np.isclose(gradient, derived_gradient, rtol=2.0e-11, atol=1.0e-12):
                raise ValueError("profile_gradient disagrees with support points")
            if not np.isclose(
                curvature, derived_curvature, rtol=2.0e-11, atol=1.0e-10
            ):
                raise ValueError("curvature disagrees with support points")
            if not np.allclose(
                arrays["static_sensitivity"],
                derived_sensitivity,
                rtol=2.0e-11,
                atol=1.0e-10,
            ):
                raise ValueError("static_sensitivity disagrees with support points")
            information = arrays["joint_information"]
            covariance = arrays["joint_covariance"]
            if not np.allclose(information, information.T, rtol=1.0e-11, atol=1.0e-10):
                raise ValueError("joint_information must be symmetric")
            if not np.allclose(covariance, covariance.T, rtol=1.0e-11, atol=1.0e-10):
                raise ValueError("joint_covariance must be symmetric")
            identity = np.eye(PARAMETER_DIMENSION + 1)
            if not np.allclose(
                information @ covariance,
                identity,
                rtol=2.0e-9,
                atol=2.0e-9,
            ):
                raise ValueError("joint covariance is not the information inverse")
            delay_variance = 1.0 / curvature
            if not np.isclose(
                covariance[-1, -1], delay_variance, rtol=2.0e-10, atol=1.0e-14
            ):
                raise ValueError("joint delay variance must equal 1/curvature")
            expected_cross = arrays["static_sensitivity"] / curvature
            if not np.allclose(
                arrays["parameter_delay_cross_covariance"],
                expected_cross,
                rtol=2.0e-10,
                atol=1.0e-12,
            ) or not np.allclose(
                covariance[:-1, -1],
                expected_cross,
                rtol=2.0e-10,
                atol=1.0e-12,
            ):
                raise ValueError("parameter-delay cross covariance is inconsistent")
            if not np.isclose(
                standard_deviation**2,
                delay_variance,
                rtol=2.0e-10,
                atol=1.0e-14,
            ):
                raise ValueError("delay standard deviation disagrees with curvature")
        else:
            if curvature is not None or gradient is not None:
                raise ValueError("invalid geometry cannot retain derivatives")
            empty_shapes = {
                "static_sensitivity": (0,),
                "support_lag": (0,),
                "support_map_objective": (0,),
                "support_static_coordinate": (0, PARAMETER_DIMENSION),
                "joint_information": (0, 0),
                "joint_covariance": (0, 0),
                "parameter_delay_cross_covariance": (0,),
            }
            for name, shape in empty_shapes.items():
                if arrays[name].shape != shape:
                    raise ValueError(
                        "invalid geometry requires empty {}".format(name)
                    )
            if self.mcmc_quadratic_surrogate_method != FALLBACK_SURROGATE_METHOD:
                raise ValueError("invalid geometry must use explicit MCMC fallback")
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "curvature", curvature)
        object.__setattr__(self, "profile_gradient", gradient)


def _invalid_delay_geometry(
    bounds: Tuple[float, float], reason: str
) -> DelayStaticLaplaceGeometry:
    lower, upper = (float(value) for value in bounds)
    return DelayStaticLaplaceGeometry(
        valid=False,
        method=DELAY_GEOMETRY_METHOD,
        reason=reason,
        standard_deviation_seconds=(upper - lower) / np.sqrt(12.0),
        source="uniform_delay_prior_fallback",
        curvature=None,
        profile_gradient=None,
        static_sensitivity=np.empty((0,), dtype=float),
        support_lag=np.empty((0,), dtype=float),
        support_map_objective=np.empty((0,), dtype=float),
        support_static_coordinate=np.empty((0, PARAMETER_DIMENSION), dtype=float),
        joint_information=np.empty((0, 0), dtype=float),
        joint_covariance=np.empty((0, 0), dtype=float),
        parameter_delay_cross_covariance=np.empty((0,), dtype=float),
        mcmc_quadratic_surrogate_method=FALLBACK_SURROGATE_METHOD,
    )


def compute_delay_static_laplace_geometry(
    profiles: Sequence[LagProfileResult],
    bounds: Tuple[float, float],
    fixed_delay_static_information: np.ndarray,
    expected_map_delay: float,
    expected_map_static_coordinate: np.ndarray,
    refinement_tolerance_seconds: float,
) -> DelayStaticLaplaceGeometry:
    """Build local 19-D information from one final-Q MAP lag profile."""

    lower_bound, upper_bound = (float(value) for value in bounds)
    if (
        not np.all(np.isfinite((lower_bound, upper_bound)))
        or lower_bound < 0.0
        or lower_bound >= upper_bound
    ):
        raise ValueError("delay bounds must be finite, non-negative, and increasing")
    selected_profiles = tuple(profiles)
    if any(not isinstance(value, LagProfileResult) for value in selected_profiles):
        raise TypeError("profiles must contain LagProfileResult values")
    information = np.asarray(fixed_delay_static_information, dtype=float)
    if (
        information.shape != (PARAMETER_DIMENSION, PARAMETER_DIMENSION)
        or not np.all(np.isfinite(information))
        or not np.allclose(information, information.T, rtol=1.0e-10, atol=1.0e-11)
    ):
        raise ValueError("fixed_delay_static_information must be finite symmetric 18 by 18")
    try:
        np.linalg.cholesky(information)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "fixed_delay_static_information must be positive definite"
        ) from error
    expected_delay = float(expected_map_delay)
    expected_static = np.asarray(expected_map_static_coordinate, dtype=float)
    refinement_tolerance = float(refinement_tolerance_seconds)
    if (
        not np.isfinite(expected_delay)
        or expected_delay < lower_bound
        or expected_delay > upper_bound
    ):
        raise ValueError("expected_map_delay must lie within delay bounds")
    if (
        expected_static.shape != (PARAMETER_DIMENSION,)
        or not np.all(np.isfinite(expected_static))
    ):
        raise ValueError("expected_map_static_coordinate must contain 18 finite values")
    if not np.isfinite(refinement_tolerance) or refinement_tolerance <= 0.0:
        raise ValueError("refinement_tolerance_seconds must be positive")
    if not selected_profiles:
        return _invalid_delay_geometry(bounds, "final_q_profile_unavailable")
    profile = selected_profiles[-1]
    center = float(profile.best_lag)
    if not np.isclose(
        center,
        expected_delay,
        rtol=1.0e-12,
        atol=max(1.0e-14, 0.01 * refinement_tolerance),
    ):
        raise ValueError("final-Q profile center delay disagrees with final MAP")
    boundary_tolerance = 16.0 * np.finfo(float).eps * max(
        1.0, upper_bound - lower_bound
    )
    if center <= lower_bound + boundary_tolerance or center >= upper_bound - boundary_tolerance:
        return _invalid_delay_geometry(bounds, "profile_optimum_at_boundary")
    points = {}
    for point in profile.points:
        if not point.converged or point.objective is None:
            continue
        previous = points.get(float(point.lag))
        if previous is None or float(point.objective) < float(previous.objective):
            points[float(point.lag)] = point
    center_candidates = [
        point
        for lag, point in points.items()
        if np.isclose(lag, center, rtol=0.0, atol=boundary_tolerance)
    ]
    lower = [point for lag, point in points.items() if lag < center]
    upper = [point for lag, point in points.items() if lag > center]
    if not center_candidates:
        raise ValueError("final-Q profile has no point at its reported MAP center")
    center_point = min(center_candidates, key=lambda point: point.objective)
    if center_point.static_coordinate is None or not np.allclose(
        center_point.static_coordinate,
        expected_static,
        rtol=1.0e-10,
        atol=1.0e-11,
    ):
        raise ValueError(
            "final-Q profile center static coordinate disagrees with final MAP"
        )
    if not lower or not upper:
        return _invalid_delay_geometry(bounds, "missing_bilateral_profile_support")
    selected = (
        max(lower, key=lambda point: point.lag),
        center_point,
        min(upper, key=lambda point: point.lag),
    )
    if any(point.static_coordinate is None for point in selected):
        return _invalid_delay_geometry(bounds, "missing_static_coordinate_support")
    support_lag = np.asarray(tuple(point.lag for point in selected), dtype=float)
    support_objective = np.asarray(
        tuple(float(point.objective) for point in selected), dtype=float
    )
    support_coordinate = np.asarray(
        tuple(point.static_coordinate for point in selected), dtype=float
    )
    first, second = _nonuniform_three_point_weights(support_lag)
    gradient = float(first @ support_objective)
    curvature = float(second @ support_objective)
    sensitivity = np.asarray(first @ support_coordinate, dtype=float)
    if not np.all(np.isfinite((gradient, curvature))) or not np.all(
        np.isfinite(sensitivity)
    ):
        return _invalid_delay_geometry(bounds, "nonfinite_local_profile_geometry")
    if curvature <= 0.0:
        return _invalid_delay_geometry(bounds, "nonpositive_profile_curvature")
    local_minimum_shift = abs(gradient / curvature)
    support_scale = min(
        support_lag[1] - support_lag[0],
        support_lag[2] - support_lag[1],
    )
    stationarity_tolerance = max(
        2.0 * refinement_tolerance,
        0.25 * support_scale,
    )
    if (
        not np.isfinite(local_minimum_shift)
        or local_minimum_shift > stationarity_tolerance
        or local_minimum_shift >= support_scale
    ):
        return _invalid_delay_geometry(bounds, "unstable_profile_stationarity")
    conditional_covariance = np.linalg.solve(
        information, np.eye(PARAMETER_DIMENSION)
    )
    information_times_sensitivity = information @ sensitivity
    joint_information = np.empty(
        (PARAMETER_DIMENSION + 1, PARAMETER_DIMENSION + 1), dtype=float
    )
    joint_information[:-1, :-1] = information
    joint_information[:-1, -1] = -information_times_sensitivity
    joint_information[-1, :-1] = -information_times_sensitivity
    joint_information[-1, -1] = curvature + float(
        sensitivity @ information_times_sensitivity
    )
    joint_covariance = np.empty_like(joint_information)
    joint_covariance[:-1, :-1] = (
        conditional_covariance + np.outer(sensitivity, sensitivity) / curvature
    )
    cross = sensitivity / curvature
    joint_covariance[:-1, -1] = cross
    joint_covariance[-1, :-1] = cross
    joint_covariance[-1, -1] = 1.0 / curvature
    condition = float(np.linalg.cond(joint_information))
    identity_error = float(
        np.linalg.norm(
            joint_information @ joint_covariance
            - np.eye(PARAMETER_DIMENSION + 1),
            ord=np.inf,
        )
    )
    if (
        not np.all(np.isfinite(joint_information))
        or not np.all(np.isfinite(joint_covariance))
        or not np.isfinite(condition)
        or condition > 1.0 / np.finfo(float).eps
        or not np.isfinite(identity_error)
        or identity_error > 2.0e-7
    ):
        return _invalid_delay_geometry(bounds, "unstable_local_joint_geometry")
    return DelayStaticLaplaceGeometry(
        valid=True,
        method=DELAY_GEOMETRY_METHOD,
        reason="valid",
        standard_deviation_seconds=float(np.sqrt(1.0 / curvature)),
        source="three_point_final_q_map_profile_curvature",
        curvature=curvature,
        profile_gradient=gradient,
        static_sensitivity=sensitivity,
        support_lag=support_lag,
        support_map_objective=support_objective,
        support_static_coordinate=support_coordinate,
        joint_information=joint_information,
        joint_covariance=joint_covariance,
        parameter_delay_cross_covariance=cross,
        mcmc_quadratic_surrogate_method=JOINT_SURROGATE_METHOD,
    )


def mcmc_quadratic_surrogate_information(
    fixed_delay_static_information: np.ndarray,
    geometry: DelayStaticLaplaceGeometry,
    fallback_delay_scale_seconds: Optional[float] = None,
) -> np.ndarray:
    """Return joint information or the explicit proposal-only fallback."""

    information = np.asarray(fixed_delay_static_information, dtype=float)
    if information.shape != (PARAMETER_DIMENSION, PARAMETER_DIMENSION):
        raise ValueError("fixed_delay_static_information must be 18 by 18")
    if not isinstance(geometry, DelayStaticLaplaceGeometry):
        raise TypeError("geometry must be DelayStaticLaplaceGeometry")
    if geometry.valid:
        if not np.allclose(
            geometry.joint_information[:-1, :-1],
            information,
            rtol=2.0e-10,
            atol=2.0e-10,
        ):
            raise ValueError("joint information static block changed")
        result = geometry.joint_information.copy()
    else:
        if not np.all(np.isfinite(information)):
            raise ValueError("fixed_delay_static_information must be finite")
        if fallback_delay_scale_seconds is None:
            raise ValueError(
                "invalid geometry requires a positive fallback delay scale"
            )
        fallback_scale = float(fallback_delay_scale_seconds)
        if not np.isfinite(fallback_scale) or fallback_scale <= 0.0:
            raise ValueError(
                "invalid geometry requires a positive fallback delay scale"
            )
        result = np.zeros(
            (PARAMETER_DIMENSION + 1, PARAMETER_DIMENSION + 1), dtype=float
        )
        result[:-1, :-1] = information
        result[-1, -1] = 1.0 / fallback_scale**2
    result.setflags(write=False)
    return result


def mcmc_parameter_delay_initialization_covariance(
    fixed_delay_conditional_static_covariance: np.ndarray,
    geometry: DelayStaticLaplaceGeometry,
    fallback_delay_scale_seconds: Optional[float] = None,
) -> np.ndarray:
    """Return joint Laplace covariance or the audited block fallback."""

    conditional = np.asarray(
        fixed_delay_conditional_static_covariance, dtype=float
    )
    if (
        conditional.shape != (PARAMETER_DIMENSION, PARAMETER_DIMENSION)
        or not np.all(np.isfinite(conditional))
        or not np.allclose(
            conditional, conditional.T, rtol=1.0e-10, atol=1.0e-11
        )
    ):
        raise ValueError(
            "fixed_delay_conditional_static_covariance must be finite "
            "symmetric 18 by 18"
        )
    if not isinstance(geometry, DelayStaticLaplaceGeometry):
        raise TypeError("geometry must be DelayStaticLaplaceGeometry")
    if geometry.valid:
        # The static block is the delay-marginal covariance, so it must not
        # equal the fixed-delay conditional block when sensitivity is nonzero.
        expected_static = conditional + np.outer(
            geometry.static_sensitivity, geometry.static_sensitivity
        ) / float(geometry.curvature)
        if not np.allclose(
            geometry.joint_covariance[:-1, :-1],
            expected_static,
            rtol=2.0e-10,
            atol=2.0e-10,
        ):
            raise ValueError("joint covariance static marginal is inconsistent")
        result = geometry.joint_covariance.copy()
    else:
        if fallback_delay_scale_seconds is None:
            raise ValueError(
                "invalid geometry requires a positive fallback delay scale"
            )
        fallback_scale = float(fallback_delay_scale_seconds)
        if not np.isfinite(fallback_scale) or fallback_scale <= 0.0:
            raise ValueError(
                "invalid geometry requires a positive fallback delay scale"
            )
        result = np.zeros(
            (PARAMETER_DIMENSION + 1, PARAMETER_DIMENSION + 1), dtype=float
        )
        result[:-1, :-1] = conditional
        result[-1, -1] = fallback_scale**2
    try:
        np.linalg.cholesky(result)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "parameter-delay initialization covariance must be positive definite"
        ) from error
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MarginalObjectiveBreakdown:
    """Q-dependent approximate negative log marginal objective."""

    graph_objective: float
    q_log_normalization: float
    hessian_log_determinant_term: float
    value: float

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.graph_objective,
                self.q_log_normalization,
                self.hessian_log_determinant_term,
                self.value,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("marginal objective terms must be finite")
        expected = float(np.sum(values[:3]))
        if not np.isclose(
            values[3], expected, rtol=1.0e-12, atol=1.0e-14
        ):
            raise ValueError("marginal objective terms do not sum to value")
        object.__setattr__(self, "graph_objective", float(values[0]))
        object.__setattr__(self, "q_log_normalization", float(values[1]))
        object.__setattr__(
            self, "hessian_log_determinant_term", float(values[2])
        )
        object.__setattr__(self, "value", float(values[3]))


@dataclass(frozen=True)
class StaticLaplaceGeometry:
    """Prior-separated 18-D information and proper posterior covariance."""

    information: ReducedInformationAnalysis
    covariance: np.ndarray
    exact_ridge_direction: np.ndarray
    ridge_alignment: float

    def __post_init__(self) -> None:
        if not isinstance(self.information, ReducedInformationAnalysis):
            raise TypeError("information must be ReducedInformationAnalysis")
        covariance = np.asarray(self.covariance, dtype=float)
        direction = np.asarray(self.exact_ridge_direction, dtype=float)
        alignment = float(self.ridge_alignment)
        if (
            covariance.shape
            != (PARAMETER_DIMENSION, PARAMETER_DIMENSION)
            or not np.all(np.isfinite(covariance))
        ):
            raise ValueError("covariance must be a finite 18 by 18 matrix")
        if (
            direction.shape != (PARAMETER_DIMENSION,)
            or not np.all(np.isfinite(direction))
            or not np.isclose(np.linalg.norm(direction), 1.0)
        ):
            raise ValueError("exact_ridge_direction must be a unit 18-vector")
        if not np.isfinite(alignment) or alignment < 0.0 or alignment > 1.0:
            raise ValueError("ridge_alignment must be in [0, 1]")
        covariance = 0.5 * (covariance + covariance.T)
        covariance.setflags(write=False)
        direction = direction.copy()
        direction.setflags(write=False)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "exact_ridge_direction", direction)
        object.__setattr__(self, "ridge_alignment", alignment)


def dynamics_q_log_normalization(
    definition: DiagonalQDefinition,
    q_diagonal: np.ndarray,
    interval_time_steps: np.ndarray,
) -> float:
    """Return ``0.5 sum_k log det(Sigma_k)`` for valid intervals.

    The graph uses ``Sigma_k = Q / dt_k`` for the sole body-wrench continuous
    spectral-density contract.  Constants involving ``2*pi`` are omitted
    because the residual count is fixed across candidate Q values.
    """

    if not isinstance(definition, DiagonalQDefinition):
        raise TypeError("definition must be DiagonalQDefinition")
    q = np.asarray(q_diagonal, dtype=float)
    time_steps = np.asarray(interval_time_steps, dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)) or np.any(q <= 0.0):
        raise ValueError("q_diagonal must contain six positive finite values")
    definition.interval_weights(time_steps)
    result = 0.5 * time_steps.size * float(np.sum(np.log(q)))
    result -= 3.0 * float(np.sum(np.log(time_steps)))
    return float(result)


def approximate_marginal_objective(
    graph_objective: float,
    factorization: ArrowheadLaplaceFactorization,
    definition: DiagonalQDefinition,
    q_diagonal: np.ndarray,
    interval_time_steps: np.ndarray,
) -> MarginalObjectiveBreakdown:
    """Combine MAP, Q normalization, and undamped Laplace volume terms."""

    if isinstance(graph_objective, (bool, np.bool_)) or not isinstance(
        graph_objective, Real
    ):
        raise TypeError("graph_objective must be a real scalar")
    objective = float(graph_objective)
    if not np.isfinite(objective):
        raise ValueError("graph_objective must be finite")
    if not isinstance(factorization, ArrowheadLaplaceFactorization):
        raise TypeError(
            "factorization must be ArrowheadLaplaceFactorization"
        )
    normalization = dynamics_q_log_normalization(
        definition, q_diagonal, interval_time_steps
    )
    hessian_term = 0.5 * factorization.diagnostics.log_determinant
    return MarginalObjectiveBreakdown(
        graph_objective=objective,
        q_log_normalization=normalization,
        hessian_log_determinant_term=hessian_term,
        value=objective + normalization + hessian_term,
    )


def compute_static_laplace_geometry(
    factorization: ArrowheadLaplaceFactorization,
    static_prior_square_root_information: np.ndarray,
    exact_ridge_direction: np.ndarray,
    *,
    relative_rank_tolerance: float = 1.0e-10,
) -> StaticLaplaceGeometry:
    """Separate static prior precision from the Schur-complement Hessian."""

    if not isinstance(factorization, ArrowheadLaplaceFactorization):
        raise TypeError(
            "factorization must be ArrowheadLaplaceFactorization"
        )
    whitening = np.asarray(
        static_prior_square_root_information, dtype=float
    )
    if (
        whitening.shape
        != (PARAMETER_DIMENSION, PARAMETER_DIMENSION)
        or not np.all(np.isfinite(whitening))
    ):
        raise ValueError(
            "static prior square-root information must be finite 18 by 18"
        )
    direction = np.asarray(exact_ridge_direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if (
        direction.shape != (PARAMETER_DIMENSION,)
        or not np.all(np.isfinite(direction))
        or not np.isfinite(norm)
        or norm <= 0.0
    ):
        raise ValueError("exact_ridge_direction must be a finite nonzero vector")
    direction = direction / norm

    posterior = factorization.reduced_hessian
    prior = whitening.T @ whitening
    likelihood = posterior - prior
    likelihood = 0.5 * (likelihood + likelihood.T)
    information = analyze_reduced_information(
        likelihood,
        posterior,
        relative_rank_tolerance=relative_rank_tolerance,
    )
    identity = np.eye(PARAMETER_DIMENSION)
    covariance = np.linalg.solve(posterior, identity)
    covariance = 0.5 * (covariance + covariance.T)
    numerical_direction = information.likelihood.eigenvectors[:, 0]
    alignment = min(
        1.0, abs(float(direction @ numerical_direction))
    )
    return StaticLaplaceGeometry(
        information=information,
        covariance=covariance,
        exact_ridge_direction=direction,
        ridge_alignment=alignment,
    )


__all__ = [
    "DELAY_GEOMETRY_METHOD",
    "DelayStaticLaplaceGeometry",
    "FALLBACK_SURROGATE_METHOD",
    "JOINT_SURROGATE_METHOD",
    "MarginalObjectiveBreakdown",
    "StaticLaplaceGeometry",
    "approximate_marginal_objective",
    "compute_delay_static_laplace_geometry",
    "compute_static_laplace_geometry",
    "dynamics_q_log_normalization",
    "mcmc_quadratic_surrogate_information",
    "mcmc_parameter_delay_initialization_covariance",
]
