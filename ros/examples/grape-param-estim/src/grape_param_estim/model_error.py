"""Time-correlated residual-wrench coordinates for weak constraints."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


WRENCH_DIMENSION = 6


def _validated_process_parameters(
    times: Sequence[float],
    stationary_standard_deviation: Sequence[float],
    correlation_time: float,
    time_name: str,
    minimum_time_count: int,
):
    selected_times = np.asarray(times, dtype=float)
    standard_deviation = np.asarray(
        stationary_standard_deviation, dtype=float
    )
    selected_correlation_time = float(correlation_time)
    if (
        selected_times.ndim != 1
        or selected_times.size < minimum_time_count
        or not np.all(np.isfinite(selected_times))
        or (
            selected_times.size > 1
            and np.any(np.diff(selected_times) <= 0.0)
        )
    ):
        raise ValueError(
            "{} must be a finite increasing vector".format(time_name)
        )
    if (
        standard_deviation.shape != (WRENCH_DIMENSION,)
        or not np.all(np.isfinite(standard_deviation))
        or np.any(standard_deviation <= 0.0)
    ):
        raise ValueError(
            "stationary_standard_deviation must contain six positive "
            "finite values"
        )
    if (
        not np.isfinite(selected_correlation_time)
        or selected_correlation_time <= 0.0
    ):
        raise ValueError("correlation_time must be finite and positive")
    return (
        selected_times.copy(),
        standard_deviation.copy(),
        selected_correlation_time,
    )


def _decode_ou_knots(
    times: np.ndarray,
    stationary_standard_deviation: np.ndarray,
    correlation_time: float,
    standard_innovations: Sequence[float],
) -> np.ndarray:
    dimension = WRENCH_DIMENSION * int(times.size)
    values = np.asarray(standard_innovations, dtype=float)
    if values.shape != (dimension,) or not np.all(np.isfinite(values)):
        raise ValueError(
            "standard_innovations must contain {} finite values".format(
                dimension
            )
        )
    innovations = values.reshape(times.size, WRENCH_DIMENSION)
    process = np.empty_like(innovations)
    with np.errstate(over="raise", invalid="raise"):
        try:
            process[0] = stationary_standard_deviation * innovations[0]
            for index in range(1, times.size):
                time_step = times[index] - times[index - 1]
                ratio = time_step / correlation_time
                rho = float(np.exp(-ratio))
                # ``-expm1(-2 ratio)`` is the stable evaluation of
                # ``1 - rho**2`` for both very small and large time steps.
                innovation_variance = float(-np.expm1(-2.0 * ratio))
                innovation_scale = np.sqrt(max(0.0, innovation_variance))
                process[index] = (
                    rho * process[index - 1]
                    + stationary_standard_deviation
                    * innovation_scale
                    * innovations[index]
                )
        except FloatingPointError as error:
            raise ValueError("decoded wrench is not representable") from error
    if not np.all(np.isfinite(process)):
        raise ValueError("decoded wrench is not representable")
    return process


def _sample_standard_innovations(
    member_count: int, dimension: int, seed: int
) -> np.ndarray:
    if isinstance(member_count, (bool, np.bool_)):
        raise ValueError("member_count must be a positive integer")
    count = int(member_count)
    if count != member_count or count <= 0:
        raise ValueError("member_count must be a positive integer")
    generator = np.random.RandomState(int(seed))
    result = generator.normal(size=(count, dimension))
    result -= np.mean(result, axis=0, keepdims=True)
    return result


@dataclass(frozen=True)
class GaussMarkovWrenchProcess:
    """A stationary six-dimensional Ornstein--Uhlenbeck wrench process.

    Args:
        times: Strictly increasing sample times.  One residual wrench is
            represented at every supplied time.
        stationary_standard_deviation: Six positive stationary standard
            deviations, ordered as body force followed by body torque.
        correlation_time: Positive OU correlation time in the same unit as
            ``times``.

    The unconstrained coordinates are independent standard-normal
    innovations ``xi`` with shape ``(N, 6)``.  ``decode`` maps their flattened
    representation to the stationary process

    ``eta[0] = sigma * xi[0]`` and
    ``eta[k] = rho * eta[k-1] + sigma * sqrt(1-rho**2) * xi[k]``,

    where ``rho = exp(-(times[k] - times[k-1]) / correlation_time)``.
    """

    times: np.ndarray
    stationary_standard_deviation: np.ndarray
    correlation_time: float

    def __post_init__(self) -> None:
        times, standard_deviation, correlation_time = (
            _validated_process_parameters(
                self.times,
                self.stationary_standard_deviation,
                self.correlation_time,
                "times",
                minimum_time_count=1,
            )
        )
        object.__setattr__(self, "times", times.copy())
        object.__setattr__(
            self,
            "stationary_standard_deviation",
            standard_deviation.copy(),
        )
        object.__setattr__(self, "correlation_time", correlation_time)

    @property
    def innovation_dimension(self) -> int:
        """Number of scalar standard innovations in one process member."""

        return WRENCH_DIMENSION * int(self.times.size)

    @property
    def compatibility_signature(self):
        """Immutable signature for matching a problem and its Q prior."""

        return (
            "dense-ou-body-wrench-v1",
            tuple(float(value) for value in self.times),
            tuple(
                float(value)
                for value in self.stationary_standard_deviation
            ),
            float(self.correlation_time),
        )

    @classmethod
    def from_knots(
        cls,
        integration_times: Sequence[float],
        knot_indices: Sequence[int],
        stationary_standard_deviation: Sequence[float],
        correlation_time: float,
    ):
        """Construct the sparse interval-average implementation."""

        return KnotGaussMarkovWrenchProcess(
            integration_times=integration_times,
            knot_indices=knot_indices,
            stationary_standard_deviation=(
                stationary_standard_deviation
            ),
            correlation_time=correlation_time,
        )

    def decode(self, standard_innovations: Sequence[float]) -> np.ndarray:
        """Decode one flattened standard-innovation vector to ``(N, 6)``.

        Raises:
            ValueError: If the input does not have shape
                ``(innovation_dimension,)``, contains non-finite values, or
                produces values outside floating-point representation.
        """

        return _decode_ou_knots(
            self.times,
            self.stationary_standard_deviation,
            self.correlation_time,
            standard_innovations,
        )

    def sample_innovations(self, member_count: int, seed: int) -> np.ndarray:
        """Draw reproducible recentered standard innovations for an ensemble.

        The returned array has shape ``(member_count, innovation_dimension)``.
        Each coordinate is sampled independently before its sample mean across
        members is subtracted.  No covariance shaping, localization, or
        inflation is applied here.
        """

        return _sample_standard_innovations(
            member_count, self.innovation_dimension, seed
        )


@dataclass(frozen=True)
class KnotGaussMarkovWrenchProcess:
    """Sparse OU body wrench evaluated through interval-average knots.

    ``integration_times`` are the state/integration boundaries and therefore
    contain one more entry than the returned interval wrench path.
    ``knot_indices`` select boundaries at which an OU wrench is represented;
    the first and final integration boundaries must both be knots.  Between
    knots the wrench is piecewise linear.  ``decode`` returns its exact mean
    over every integration interval so it can replace the dense process in a
    :class:`~grape_param_estim.weak_constraint.WeakConstraintProblem`.
    """

    integration_times: np.ndarray
    knot_indices: np.ndarray
    stationary_standard_deviation: np.ndarray
    correlation_time: float

    def __post_init__(self) -> None:
        integration_times, standard_deviation, correlation_time = (
            _validated_process_parameters(
                self.integration_times,
                self.stationary_standard_deviation,
                self.correlation_time,
                "integration_times",
                minimum_time_count=2,
            )
        )
        raw_indices = np.asarray(self.knot_indices)
        if (
            raw_indices.ndim != 1
            or raw_indices.size < 2
            or np.issubdtype(raw_indices.dtype, np.bool_)
            or not np.issubdtype(raw_indices.dtype, np.integer)
        ):
            raise ValueError(
                "knot_indices must be an integer vector with at least two "
                "entries"
            )
        indices = raw_indices.astype(np.int64, copy=True)
        if (
            indices[0] != 0
            or indices[-1] != integration_times.size - 1
            or np.any(np.diff(indices) <= 0)
        ):
            raise ValueError(
                "knot_indices must increase from the first through the "
                "final integration boundary"
            )
        matrix = np.zeros(
            (integration_times.size - 1, indices.size), dtype=float
        )
        for knot in range(indices.size - 1):
            first_interval = int(indices[knot])
            final_interval = int(indices[knot + 1])
            left_time = integration_times[indices[knot]]
            right_time = integration_times[indices[knot + 1]]
            midpoint = 0.5 * (
                integration_times[first_interval:final_interval]
                + integration_times[first_interval + 1:final_interval + 1]
            )
            duration = right_time - left_time
            matrix[first_interval:final_interval, knot] = (
                right_time - midpoint
            ) / duration
            matrix[first_interval:final_interval, knot + 1] = (
                midpoint - left_time
            ) / duration
        object.__setattr__(
            self, "integration_times", integration_times.copy()
        )
        object.__setattr__(self, "knot_indices", indices)
        object.__setattr__(
            self,
            "stationary_standard_deviation",
            standard_deviation.copy(),
        )
        object.__setattr__(self, "correlation_time", correlation_time)
        object.__setattr__(self, "_interpolation_matrix", matrix)

    @property
    def times(self) -> np.ndarray:
        """Dense-compatible interval start times."""

        return self.integration_times[:-1].copy()

    @property
    def knot_times(self) -> np.ndarray:
        """Times at which independent OU innovations are represented."""

        return self.integration_times[self.knot_indices].copy()

    @property
    def innovation_dimension(self) -> int:
        """Six independent standard innovations per knot."""

        return WRENCH_DIMENSION * int(self.knot_indices.size)

    @property
    def interpolation_matrix(self) -> np.ndarray:
        """Exact piecewise-linear knot-to-interval-average operator."""

        return self._interpolation_matrix.copy()

    @property
    def interval_average_interpolation_matrix(self) -> np.ndarray:
        """Explicit alias documenting the interpolation convention."""

        return self.interpolation_matrix

    @property
    def compatibility_signature(self):
        """Immutable signature for matching a problem and its Q prior."""

        return (
            "knot-ou-body-wrench-v1",
            tuple(float(value) for value in self.integration_times),
            tuple(int(value) for value in self.knot_indices),
            tuple(
                float(value)
                for value in self.stationary_standard_deviation
            ),
            float(self.correlation_time),
        )

    def decode_knots(
        self, standard_innovations: Sequence[float]
    ) -> np.ndarray:
        """Decode the standard innovations to the stationary OU knots."""

        return _decode_ou_knots(
            self.integration_times[self.knot_indices],
            self.stationary_standard_deviation,
            self.correlation_time,
            standard_innovations,
        )

    def decode(self, standard_innovations: Sequence[float]) -> np.ndarray:
        """Decode to one exact average body wrench per integration interval."""

        knots = self.decode_knots(standard_innovations)
        result = self._interpolation_matrix @ knots
        if not np.all(np.isfinite(result)):
            raise ValueError("decoded wrench is not representable")
        return result

    def sample_innovations(self, member_count: int, seed: int) -> np.ndarray:
        """Draw reproducible recentered standard innovations at the knots."""

        return _sample_standard_innovations(
            member_count, self.innovation_dimension, seed
        )


__all__ = [
    "GaussMarkovWrenchProcess",
    "KnotGaussMarkovWrenchProcess",
    "WRENCH_DIMENSION",
]
