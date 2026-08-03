"""Deterministic ensemble filter and fixed-interval smoother primitives.

The routines in this module are deliberately Euclidean and model agnostic.
The Grape state chart, SO(3) handling, and closed-loop propagation live at a
higher layer.  Member order is never changed because lag-one smoothing and
the downstream trajectory artifacts rely on member alignment.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


def _finite_matrix(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.ndim != 2
        or min(result.shape) < 1
        or np.any(~np.isfinite(result))
    ):
        raise ValueError("{} must be a non-empty finite matrix".format(name))
    return result.copy()


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or np.any(~np.isfinite(result)):
        raise ValueError(
            "{} must contain {} finite values".format(name, size)
        )
    return result.copy()


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be a positive integer".format(name))
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "{} must be a positive integer".format(name)
        ) from error
    if result != value or result <= 0:
        raise ValueError("{} must be a positive integer".format(name))
    return result


def _symmetric_positive_definite(value, size: int, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (size, size)
        or np.any(~np.isfinite(matrix))
        or not np.allclose(matrix, matrix.T, rtol=1.0e-12, atol=1.0e-14)
    ):
        raise ValueError(
            "{} must be a finite symmetric {} by {} matrix".format(
                name, size, size
            )
        )
    symmetric = 0.5 * (matrix + matrix.T)
    if np.any(np.linalg.eigvalsh(symmetric) <= 0.0):
        raise ValueError("{} must be positive definite".format(name))
    return symmetric


def _low_rank_ensemble_transform(
    observation_sample_factor: np.ndarray,
    observation_covariance: np.ndarray,
) -> np.ndarray:
    """Evaluate the ETKF inverse square root in observation rank.

    ``I + Y.T @ R^-1 @ Y`` differs from identity in at most observation
    dimension directions.  An SVD of the usually 6-by-M whitened anomaly
    matrix therefore avoids an M-by-M eigendecomposition at every sample.
    """

    factor = np.asarray(observation_sample_factor, dtype=float)
    covariance = np.asarray(observation_covariance, dtype=float)
    if (
        factor.ndim != 2
        or covariance.ndim != 2
        or factor.shape[0] != covariance.shape[0]
        or covariance.shape[0] != covariance.shape[1]
        or np.any(~np.isfinite(factor))
    ):
        raise ValueError("observation sample factor is misaligned")
    cholesky = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(cholesky, factor)
    _left, singular_values, right_transpose = np.linalg.svd(
        whitened, full_matrices=False
    )
    correction = 1.0 / np.sqrt(1.0 + singular_values**2) - 1.0
    transform = np.eye(factor.shape[1]) + (
        right_transpose.T * correction[None, :]
    ) @ right_transpose
    return 0.5 * (transform + transform.T)


def exact_gaussian_ensemble(
    mean,
    covariance,
    member_count: int,
    seed: int,
    orthogonal_to: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a seeded ensemble with exact sample mean and covariance.

    If ``orthogonal_to`` is supplied, the new anomaly columns are also
    sample-orthogonal to every non-degenerate centered column in that array.
    This is used to prevent finite-ensemble process-noise/forecast-state cross
    covariance from being introduced merely by the random draw.
    """

    selected_mean = np.asarray(mean, dtype=float)
    if selected_mean.ndim != 1 or np.any(~np.isfinite(selected_mean)):
        raise ValueError("mean must be a non-empty finite vector")
    dimension = int(selected_mean.size)
    if dimension < 1:
        raise ValueError("mean must be a non-empty finite vector")
    selected_covariance = _symmetric_positive_definite(
        covariance, dimension, "covariance"
    )
    count = _positive_integer(member_count, "member_count")

    constraints = np.ones((count, 1), dtype=float)
    if orthogonal_to is not None:
        reference = np.asarray(orthogonal_to, dtype=float)
        if (
            reference.ndim != 2
            or reference.shape[0] != count
            or np.any(~np.isfinite(reference))
        ):
            raise ValueError(
                "orthogonal_to must be a finite member-first matrix"
            )
        constraints = np.column_stack(
            (
                constraints,
                reference - np.mean(reference, axis=0, keepdims=True),
            )
        )

    left, singular_values, _right = np.linalg.svd(
        constraints, full_matrices=True
    )
    tolerance = 0.0
    if singular_values.size:
        tolerance = (
            max(constraints.shape)
            * np.finfo(float).eps
            * singular_values[0]
        )
    rank = int(np.count_nonzero(singular_values > tolerance))
    available = count - rank
    if available < dimension:
        raise ValueError(
            "member_count leaves {} orthogonal anomaly directions; {} are "
            "required".format(available, dimension)
        )
    null_basis = left[:, rank:]
    generator = np.random.RandomState(int(seed))
    random_coordinates = generator.normal(size=(available, dimension))
    orthonormal, triangular = np.linalg.qr(
        random_coordinates, mode="reduced"
    )
    diagonal = np.diag(triangular)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    orthonormal *= signs[None, :]
    standard = null_basis @ orthonormal * np.sqrt(count - 1.0)
    factor = np.linalg.cholesky(selected_covariance)
    result = selected_mean[None, :] + standard @ factor.T
    if np.any(~np.isfinite(result)):
        raise ValueError("exact Gaussian ensemble is not representable")
    return result


@dataclass(frozen=True)
class DeterministicEnsembleUpdate:
    """One square-root ensemble Kalman analysis with audit diagnostics."""

    analysis_ensemble: np.ndarray
    forecast_state_mean: np.ndarray
    forecast_observation_mean: np.ndarray
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    kalman_gain: np.ndarray
    member_transform: np.ndarray
    approximate_log_likelihood: float

    def __post_init__(self) -> None:
        analysis = _finite_matrix(
            self.analysis_ensemble, "analysis_ensemble"
        )
        members, state_dimension = analysis.shape
        state_mean = _finite_vector(
            self.forecast_state_mean,
            state_dimension,
            "forecast_state_mean",
        )
        observation_mean = np.asarray(
            self.forecast_observation_mean, dtype=float
        )
        if observation_mean.ndim != 1 or np.any(
            ~np.isfinite(observation_mean)
        ):
            raise ValueError(
                "forecast_observation_mean must be a finite vector"
            )
        observation_dimension = observation_mean.size
        innovation = _finite_vector(
            self.innovation, observation_dimension, "innovation"
        )
        innovation_covariance = _symmetric_positive_definite(
            self.innovation_covariance,
            observation_dimension,
            "innovation_covariance",
        )
        gain = np.asarray(self.kalman_gain, dtype=float)
        transform = np.asarray(self.member_transform, dtype=float)
        log_likelihood = float(self.approximate_log_likelihood)
        if (
            gain.shape != (state_dimension, observation_dimension)
            or np.any(~np.isfinite(gain))
            or transform.shape != (members, members)
            or np.any(~np.isfinite(transform))
            or not np.allclose(transform, transform.T, atol=1.0e-12)
            or not np.isfinite(log_likelihood)
        ):
            raise ValueError("ensemble update diagnostics are misaligned")
        for name, value in (
            ("analysis_ensemble", analysis),
            ("forecast_state_mean", state_mean),
            ("forecast_observation_mean", observation_mean),
            ("innovation", innovation),
            ("innovation_covariance", innovation_covariance),
            ("kalman_gain", gain),
            ("member_transform", transform),
        ):
            object.__setattr__(self, name, np.asarray(value).copy())
        object.__setattr__(
            self, "approximate_log_likelihood", log_likelihood
        )


def deterministic_square_root_update(
    forecast_ensemble: np.ndarray,
    predicted_observation_ensemble: np.ndarray,
    observation,
    observation_covariance: np.ndarray,
) -> DeterministicEnsembleUpdate:
    """Apply one deterministic ensemble Kalman measurement update."""

    forecast = _finite_matrix(forecast_ensemble, "forecast_ensemble")
    predicted = _finite_matrix(
        predicted_observation_ensemble,
        "predicted_observation_ensemble",
    )
    members, state_dimension = forecast.shape
    if members < 2 or predicted.shape[0] != members:
        raise ValueError(
            "forecast and predicted observation need at least two aligned "
            "members"
        )
    observation_dimension = predicted.shape[1]
    selected_observation = _finite_vector(
        observation, observation_dimension, "observation"
    )
    selected_covariance = _symmetric_positive_definite(
        observation_covariance,
        observation_dimension,
        "observation_covariance",
    )

    state_mean = np.mean(forecast, axis=0)
    observation_mean = np.mean(predicted, axis=0)
    state_anomalies = (forecast - state_mean[None, :]).T
    observation_anomalies = (predicted - observation_mean[None, :]).T
    scale = np.sqrt(members - 1.0)
    state_sample_factor = state_anomalies / scale
    observation_sample_factor = observation_anomalies / scale
    innovation_covariance = (
        observation_sample_factor @ observation_sample_factor.T
        + selected_covariance
    )
    state_observation_covariance = (
        state_sample_factor @ observation_sample_factor.T
    )
    kalman_gain = np.linalg.solve(
        innovation_covariance,
        state_observation_covariance.T,
    ).T
    innovation = selected_observation - observation_mean
    analysis_mean = state_mean + kalman_gain @ innovation

    transform = _low_rank_ensemble_transform(
        observation_sample_factor, selected_covariance
    )
    analysis_anomalies = state_anomalies @ transform
    # Suppress roundoff drift along the exact mean-null direction.
    analysis_anomalies -= np.mean(
        analysis_anomalies, axis=1, keepdims=True
    )
    analysis = (analysis_mean[:, None] + analysis_anomalies).T

    sign, log_determinant = np.linalg.slogdet(innovation_covariance)
    if sign <= 0.0 or not np.isfinite(log_determinant):
        raise ValueError("innovation covariance determinant is invalid")
    quadratic = float(
        innovation
        @ np.linalg.solve(innovation_covariance, innovation)
    )
    approximate_log_likelihood = -0.5 * (
        observation_dimension * np.log(2.0 * np.pi)
        + log_determinant
        + quadratic
    )
    return DeterministicEnsembleUpdate(
        analysis_ensemble=analysis,
        forecast_state_mean=state_mean,
        forecast_observation_mean=observation_mean,
        innovation=innovation,
        innovation_covariance=innovation_covariance,
        kalman_gain=kalman_gain,
        member_transform=transform,
        approximate_log_likelihood=approximate_log_likelihood,
    )


def ensemble_rts_smoothing_step(
    analysis_ensemble: np.ndarray,
    next_forecast_ensemble: np.ndarray,
    next_smoothed_ensemble: np.ndarray,
    covariance_rcond: float = 1.0e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply one member-aligned ensemble Rauch--Tung--Striebel step."""

    analysis = _finite_matrix(analysis_ensemble, "analysis_ensemble")
    next_forecast = _finite_matrix(
        next_forecast_ensemble, "next_forecast_ensemble"
    )
    next_smoothed = _finite_matrix(
        next_smoothed_ensemble, "next_smoothed_ensemble"
    )
    if (
        analysis.shape[0] < 2
        or next_forecast.shape != next_smoothed.shape
        or next_forecast.shape[0] != analysis.shape[0]
    ):
        raise ValueError("RTS ensembles must contain aligned members")
    selected_rcond = float(covariance_rcond)
    if not np.isfinite(selected_rcond) or selected_rcond <= 0.0:
        raise ValueError("covariance_rcond must be finite and positive")
    members = analysis.shape[0]
    analysis_anomalies = analysis - np.mean(analysis, axis=0, keepdims=True)
    forecast_anomalies = next_forecast - np.mean(
        next_forecast, axis=0, keepdims=True
    )
    cross_covariance = (
        analysis_anomalies.T @ forecast_anomalies / (members - 1.0)
    )
    forecast_covariance = (
        forecast_anomalies.T @ forecast_anomalies / (members - 1.0)
    )
    gain = cross_covariance @ np.linalg.pinv(
        forecast_covariance, rcond=selected_rcond
    )
    smoothed = analysis + (next_smoothed - next_forecast) @ gain.T
    if np.any(~np.isfinite(gain)) or np.any(~np.isfinite(smoothed)):
        raise ValueError("RTS smoothing result is not representable")
    return smoothed, gain


@dataclass(frozen=True)
class EnsembleRtsResult:
    smoothed_ensembles: Tuple[np.ndarray, ...]
    smoothing_gains: Tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        ensembles = tuple(
            _finite_matrix(value, "smoothed ensemble")
            for value in self.smoothed_ensembles
        )
        if not ensembles:
            raise ValueError("at least one smoothed ensemble is required")
        reference_shape = ensembles[0].shape
        if any(value.shape != reference_shape for value in ensembles):
            raise ValueError("smoothed ensembles must have one common shape")
        gains = tuple(
            _finite_matrix(value, "smoothing gain")
            for value in self.smoothing_gains
        )
        if len(gains) != len(ensembles) - 1 or any(
            value.shape != (reference_shape[1], reference_shape[1])
            for value in gains
        ):
            raise ValueError("smoothing gains are misaligned")
        object.__setattr__(
            self,
            "smoothed_ensembles",
            tuple(value.copy() for value in ensembles),
        )
        object.__setattr__(
            self,
            "smoothing_gains",
            tuple(value.copy() for value in gains),
        )


def ensemble_rts_smoother(
    analysis_ensembles: Sequence[np.ndarray],
    forecast_ensembles: Sequence[np.ndarray],
    covariance_rcond: float = 1.0e-12,
) -> EnsembleRtsResult:
    """Smooth a Euclidean ensemble time series with aligned members."""

    analysis = tuple(
        _finite_matrix(value, "analysis ensemble")
        for value in analysis_ensembles
    )
    forecast = tuple(
        _finite_matrix(value, "forecast ensemble")
        for value in forecast_ensembles
    )
    if not analysis or len(analysis) != len(forecast):
        raise ValueError(
            "analysis_ensembles and forecast_ensembles must have equal "
            "non-zero length"
        )
    shape = analysis[0].shape
    if shape[0] < 2 or any(
        value.shape != shape for value in analysis + forecast
    ):
        raise ValueError("all smoother ensembles must have one common shape")
    smoothed = [None] * len(analysis)
    gains = [None] * max(0, len(analysis) - 1)
    smoothed[-1] = analysis[-1].copy()
    for index in range(len(analysis) - 2, -1, -1):
        smoothed[index], gains[index] = ensemble_rts_smoothing_step(
            analysis[index],
            forecast[index + 1],
            smoothed[index + 1],
            covariance_rcond=covariance_rcond,
        )
    return EnsembleRtsResult(tuple(smoothed), tuple(gains))


__all__ = [
    "DeterministicEnsembleUpdate",
    "EnsembleRtsResult",
    "deterministic_square_root_update",
    "ensemble_rts_smoother",
    "ensemble_rts_smoothing_step",
    "exact_gaussian_ensemble",
]
