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

import numpy as np

from grape_param_estim.batch.covariance import (
    ArrowheadLaplaceFactorization,
)
from grape_param_estim.batch.laplace_em import (
    DiagonalQDefinition,
    QIntervalModel,
)
from grape_param_estim.batch.ridge import (
    ReducedInformationAnalysis,
    analyze_reduced_information,
)
from grape_param_estim.parameterization import PARAMETER_DIMENSION


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

    For continuous spectral density, the graph uses
    ``Sigma_k = Q / dt_k``.  For fixed-interval covariance it uses
    ``Sigma_k = Q``.  Constants involving ``2*pi`` are omitted because the
    residual count is fixed across candidate Q values.
    """

    if not isinstance(definition, DiagonalQDefinition):
        raise TypeError("definition must be DiagonalQDefinition")
    q = np.asarray(q_diagonal, dtype=float)
    time_steps = np.asarray(interval_time_steps, dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)) or np.any(q <= 0.0):
        raise ValueError("q_diagonal must contain six positive finite values")
    definition.interval_weights(time_steps)
    result = 0.5 * time_steps.size * float(np.sum(np.log(q)))
    if definition.interval_model is QIntervalModel.CONTINUOUS_SPECTRAL_DENSITY:
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
    "MarginalObjectiveBreakdown",
    "StaticLaplaceGeometry",
    "approximate_marginal_objective",
    "compute_static_laplace_geometry",
    "dynamics_q_log_normalization",
]
