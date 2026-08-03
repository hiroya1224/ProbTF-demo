"""Diagonal stationary covariance and EM statistics for OU body wrench.

This module deliberately contains no filtering or smoothing algorithm.  It
defines the physical meaning of ``Q`` and the closed-form M-step that consumes
member-aligned, fixed-interval smoothed wrench paths produced elsewhere.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence, Tuple

import numpy as np


BODY_WRENCH_DIMENSION = 6
BODY_WRENCH_FRAME = "body"
BODY_WRENCH_COMPONENT_ORDER = (
    "force_x",
    "force_y",
    "force_z",
    "torque_x",
    "torque_y",
    "torque_z",
)
BODY_WRENCH_COMPONENT_UNITS = (
    "N",
    "N",
    "N",
    "N*m",
    "N*m",
    "N*m",
)
BODY_WRENCH_VARIANCE_UNITS = (
    "N^2",
    "N^2",
    "N^2",
    "(N*m)^2",
    "(N*m)^2",
    "(N*m)^2",
)


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or np.any(~np.isfinite(result)):
        raise ValueError(
            "{} must contain {} finite values".format(name, size)
        )
    return result.copy()


def _positive_scalar(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "{} must be finite and positive".format(name)
        ) from error
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("{} must be finite and positive".format(name))
    return result


def _validated_times(value) -> np.ndarray:
    times = np.asarray(value, dtype=float)
    if (
        times.ndim != 1
        or times.size < 1
        or np.any(~np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("times must be a finite increasing vector")
    return times.copy()


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


def _variance_floor(value) -> np.ndarray:
    raw = np.asarray(value, dtype=float)
    if raw.ndim == 0:
        raw = np.full(BODY_WRENCH_DIMENSION, float(raw), dtype=float)
    if (
        raw.shape != (BODY_WRENCH_DIMENSION,)
        or np.any(~np.isfinite(raw))
        or np.any(raw <= 0.0)
    ):
        raise ValueError(
            "variance_floor must be one or six positive finite values"
        )
    return raw.copy()


@dataclass(frozen=True)
class BodyWrenchDiagonalCovariance:
    """Stationary diagonal covariance of a body-frame residual wrench.

    The first three variances have units ``N^2`` and the final three have
    units ``(N*m)^2``.  A full matrix is exposed only as a derived value so
    off-diagonal terms cannot silently enter this model family.
    """

    stationary_variance: np.ndarray

    def __post_init__(self) -> None:
        variance = _finite_vector(
            self.stationary_variance,
            BODY_WRENCH_DIMENSION,
            "stationary_variance",
        )
        if np.any(variance <= 0.0):
            raise ValueError(
                "stationary_variance must contain six positive values"
            )
        object.__setattr__(self, "stationary_variance", variance)

    @property
    def frame(self) -> str:
        return BODY_WRENCH_FRAME

    @property
    def component_order(self) -> Tuple[str, ...]:
        return BODY_WRENCH_COMPONENT_ORDER

    @property
    def component_units(self) -> Tuple[str, ...]:
        return BODY_WRENCH_COMPONENT_UNITS

    @property
    def variance_units(self) -> Tuple[str, ...]:
        return BODY_WRENCH_VARIANCE_UNITS

    @property
    def stationary_standard_deviation(self) -> np.ndarray:
        return np.sqrt(self.stationary_variance)

    @property
    def matrix(self) -> np.ndarray:
        return np.diag(self.stationary_variance)


@dataclass(frozen=True)
class OuTransitionFactors:
    """Irregular-grid OU transition factors independent of wrench units."""

    times: np.ndarray
    correlation_time: float

    def __post_init__(self) -> None:
        times = _validated_times(self.times)
        correlation_time = _positive_scalar(
            self.correlation_time, "correlation_time"
        )
        ratio = np.diff(times) / correlation_time
        rho = np.exp(-ratio)
        # Stable for small positive time steps, unlike ``1 - rho**2``.
        innovation_fraction = -np.expm1(-2.0 * ratio)
        if (
            np.any(~np.isfinite(rho))
            or np.any(~np.isfinite(innovation_fraction))
            or np.any(rho < 0.0)
            or np.any(rho >= 1.0)
            or np.any(innovation_fraction <= 0.0)
            or np.any(innovation_fraction > 1.0)
        ):
            raise ValueError("OU transition factors are not representable")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "correlation_time", correlation_time)
        object.__setattr__(self, "_time_step", np.diff(times))
        object.__setattr__(self, "_rho", rho)
        object.__setattr__(
            self, "_innovation_variance_fraction", innovation_fraction
        )

    @property
    def time_step(self) -> np.ndarray:
        return self._time_step.copy()

    @property
    def rho(self) -> np.ndarray:
        return self._rho.copy()

    @property
    def innovation_variance_fraction(self) -> np.ndarray:
        """Return ``1-rho**2`` for every transition."""

        return self._innovation_variance_fraction.copy()

    def innovation_variance(
        self, covariance: BodyWrenchDiagonalCovariance
    ) -> np.ndarray:
        """Return the six innovation variances for each transition."""

        if not isinstance(covariance, BodyWrenchDiagonalCovariance):
            raise TypeError(
                "covariance must be a BodyWrenchDiagonalCovariance"
            )
        return (
            self._innovation_variance_fraction[:, None]
            * covariance.stationary_variance[None, :]
        )

    def innovation_covariance(
        self, covariance: BodyWrenchDiagonalCovariance
    ) -> np.ndarray:
        """Return one exact diagonal 6-by-6 covariance per transition."""

        variance = self.innovation_variance(covariance)
        result = np.zeros(
            (variance.shape[0], BODY_WRENCH_DIMENSION, BODY_WRENCH_DIMENSION),
            dtype=float,
        )
        indices = np.arange(BODY_WRENCH_DIMENSION)
        result[:, indices, indices] = variance
        return result


def ou_transition_factors(
    times: Sequence[float], correlation_time: float
) -> OuTransitionFactors:
    """Build exact irregular-grid OU correlation and variance factors."""

    return OuTransitionFactors(np.asarray(times, dtype=float), correlation_time)


@dataclass(frozen=True)
class DiagonalQEmSufficientStatistics:
    """Per-bag sufficient statistics from a smoothed wrench ensemble."""

    bag_id: str
    member_count: int
    times: np.ndarray
    correlation_time: float
    initial_second_moment: np.ndarray
    transition_second_moment: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.bag_id, str) or not self.bag_id:
            raise ValueError("bag_id must be a non-empty string")
        identifier = self.bag_id
        member_count = _positive_integer(self.member_count, "member_count")
        factors = OuTransitionFactors(self.times, self.correlation_time)
        initial = _finite_vector(
            self.initial_second_moment,
            BODY_WRENCH_DIMENSION,
            "initial_second_moment",
        )
        transition = np.asarray(self.transition_second_moment, dtype=float)
        expected_shape = (factors.times.size - 1, BODY_WRENCH_DIMENSION)
        if (
            transition.shape != expected_shape
            or np.any(~np.isfinite(transition))
            or np.any(transition < 0.0)
            or np.any(initial < 0.0)
        ):
            raise ValueError(
                "transition_second_moment must be a finite non-negative {} "
                "array".format(expected_shape)
            )
        object.__setattr__(self, "bag_id", identifier)
        object.__setattr__(self, "member_count", member_count)
        object.__setattr__(self, "times", factors.times.copy())
        object.__setattr__(
            self, "correlation_time", factors.correlation_time
        )
        object.__setattr__(self, "initial_second_moment", initial)
        object.__setattr__(
            self, "transition_second_moment", transition.copy()
        )

    @property
    def boundary_count(self) -> int:
        return int(self.times.size)

    @property
    def transition_count(self) -> int:
        return max(0, self.boundary_count - 1)

    @property
    def transition_factors(self) -> OuTransitionFactors:
        return OuTransitionFactors(self.times, self.correlation_time)

    @property
    def scaled_transition_second_moment_sum(self) -> np.ndarray:
        fraction = self.transition_factors.innovation_variance_fraction
        if fraction.size == 0:
            return np.zeros(BODY_WRENCH_DIMENSION, dtype=float)
        scaled = self.transition_second_moment / fraction[:, None]
        result = np.asarray(
            [
                math.fsum(float(value) for value in scaled[:, component])
                for component in range(BODY_WRENCH_DIMENSION)
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(result)):
            raise ValueError(
                "scaled transition second moment is not representable"
            )
        return result

    @property
    def m_step_numerator(self) -> np.ndarray:
        result = (
            self.initial_second_moment
            + self.scaled_transition_second_moment_sum
        )
        if np.any(~np.isfinite(result)):
            raise ValueError("M-step numerator is not representable")
        return result


def diagonal_q_em_sufficient_statistics(
    bag_id: str,
    times: Sequence[float],
    correlation_time: float,
    smoothed_wrench: np.ndarray,
) -> DiagonalQEmSufficientStatistics:
    """Compute OU EM moments from a member-first smoothed wrench path.

    ``smoothed_wrench`` must have shape ``(M, N, 6)``.  Member-alignment
    across adjacent boundaries is essential because the lag-one moment is
    evaluated through each member's transition residual.
    """

    if not isinstance(bag_id, str) or not bag_id:
        raise ValueError("bag_id must be a non-empty string")
    factors = OuTransitionFactors(
        np.asarray(times, dtype=float), correlation_time
    )
    wrench = np.asarray(smoothed_wrench, dtype=float)
    expected_shape = (None, factors.times.size, BODY_WRENCH_DIMENSION)
    if (
        wrench.ndim != 3
        or wrench.shape[0] < 1
        or wrench.shape[1:] != expected_shape[1:]
        or np.any(~np.isfinite(wrench))
    ):
        raise ValueError(
            "smoothed_wrench must have finite member-first shape "
            "(M, {}, 6)".format(factors.times.size)
        )
    with np.errstate(over="raise", invalid="raise"):
        try:
            initial = np.mean(wrench[:, 0, :] ** 2, axis=0)
            if factors.rho.size:
                residual = (
                    wrench[:, 1:, :]
                    - factors.rho[None, :, None] * wrench[:, :-1, :]
                )
                transition = np.mean(residual**2, axis=0)
            else:
                transition = np.empty(
                    (0, BODY_WRENCH_DIMENSION), dtype=float
                )
        except FloatingPointError as error:
            raise ValueError(
                "smoothed wrench moments are not representable"
            ) from error
    if np.any(~np.isfinite(initial)) or np.any(~np.isfinite(transition)):
        raise ValueError("smoothed wrench moments are not representable")
    return DiagonalQEmSufficientStatistics(
        bag_id=bag_id,
        member_count=int(wrench.shape[0]),
        times=factors.times,
        correlation_time=factors.correlation_time,
        initial_second_moment=initial,
        transition_second_moment=transition,
    )


@dataclass(frozen=True)
class DiagonalQEmUpdate:
    """One shared diagonal-Q M-step with explicit floor provenance."""

    covariance: BodyWrenchDiagonalCovariance
    raw_stationary_variance: np.ndarray
    variance_floor: np.ndarray
    floor_applied: np.ndarray
    bag_ids: Tuple[str, ...]
    total_boundary_count: int
    total_transition_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.covariance, BodyWrenchDiagonalCovariance):
            raise TypeError(
                "covariance must be a BodyWrenchDiagonalCovariance"
            )
        raw = _finite_vector(
            self.raw_stationary_variance,
            BODY_WRENCH_DIMENSION,
            "raw_stationary_variance",
        )
        if np.any(raw < 0.0):
            raise ValueError("raw_stationary_variance cannot be negative")
        floor = _variance_floor(self.variance_floor)
        applied = np.asarray(self.floor_applied)
        if applied.shape != (BODY_WRENCH_DIMENSION,) or not np.issubdtype(
            applied.dtype, np.bool_
        ):
            raise ValueError("floor_applied must contain six booleans")
        identifiers = tuple(str(value) for value in self.bag_ids)
        if (
            not identifiers
            or identifiers != tuple(sorted(identifiers))
            or len(set(identifiers)) != len(identifiers)
            or any(not value for value in identifiers)
        ):
            raise ValueError("bag_ids must be sorted, unique, and non-empty")
        boundaries = _positive_integer(
            self.total_boundary_count, "total_boundary_count"
        )
        transitions = int(self.total_transition_count)
        if (
            isinstance(self.total_transition_count, (bool, np.bool_))
            or transitions != self.total_transition_count
            or transitions < 0
            or boundaries != transitions + len(identifiers)
        ):
            raise ValueError(
                "total counts must contain one initial term per bag"
            )
        expected_applied = raw < floor
        expected_variance = np.maximum(raw, floor)
        if not np.array_equal(applied, expected_applied):
            raise ValueError("floor_applied does not match raw variance")
        if not np.array_equal(
            self.covariance.stationary_variance, expected_variance
        ):
            raise ValueError(
                "covariance does not match the floored stationary variance"
            )
        object.__setattr__(self, "raw_stationary_variance", raw)
        object.__setattr__(self, "variance_floor", floor)
        object.__setattr__(self, "floor_applied", applied.astype(bool, copy=True))
        object.__setattr__(self, "bag_ids", identifiers)
        object.__setattr__(self, "total_boundary_count", boundaries)
        object.__setattr__(self, "total_transition_count", transitions)


def shared_diagonal_q_m_step(
    statistics: Iterable[DiagonalQEmSufficientStatistics],
    variance_floor,
) -> DiagonalQEmUpdate:
    """Pool independent bags into the shared stationary diagonal-Q M-step.

    Each bag contributes one stationary initial term and ``N_b - 1`` OU
    transition terms, hence the common ML denominator is ``sum_b N_b``.
    Member counts do not weight bags because each sufficient statistic already
    represents an expectation under that bag's smoothing distribution.
    """

    values = tuple(statistics)
    if not values:
        raise ValueError("at least one bag statistic is required")
    if any(
        not isinstance(value, DiagonalQEmSufficientStatistics)
        for value in values
    ):
        raise TypeError(
            "statistics must contain DiagonalQEmSufficientStatistics"
        )
    identifiers = tuple(sorted(value.bag_id for value in values))
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("bag statistics must have unique bag IDs")
    ordered = tuple(sorted(values, key=lambda value: value.bag_id))
    total_boundaries = sum(value.boundary_count for value in ordered)
    total_transitions = sum(value.transition_count for value in ordered)
    numerator = np.asarray(
        [
            math.fsum(
                float(value.m_step_numerator[component])
                for value in ordered
            )
            for component in range(BODY_WRENCH_DIMENSION)
        ],
        dtype=float,
    )
    raw_variance = numerator / float(total_boundaries)
    if np.any(~np.isfinite(raw_variance)) or np.any(raw_variance < 0.0):
        raise ValueError("shared M-step variance is not representable")
    floor = _variance_floor(variance_floor)
    applied = raw_variance < floor
    covariance = BodyWrenchDiagonalCovariance(
        np.maximum(raw_variance, floor)
    )
    return DiagonalQEmUpdate(
        covariance=covariance,
        raw_stationary_variance=raw_variance,
        variance_floor=floor,
        floor_applied=applied,
        bag_ids=identifiers,
        total_boundary_count=total_boundaries,
        total_transition_count=total_transitions,
    )


__all__ = [
    "BODY_WRENCH_COMPONENT_ORDER",
    "BODY_WRENCH_COMPONENT_UNITS",
    "BODY_WRENCH_DIMENSION",
    "BODY_WRENCH_FRAME",
    "BODY_WRENCH_VARIANCE_UNITS",
    "BodyWrenchDiagonalCovariance",
    "DiagonalQEmSufficientStatistics",
    "DiagonalQEmUpdate",
    "OuTransitionFactors",
    "diagonal_q_em_sufficient_statistics",
    "ou_transition_factors",
    "shared_diagonal_q_m_step",
]
