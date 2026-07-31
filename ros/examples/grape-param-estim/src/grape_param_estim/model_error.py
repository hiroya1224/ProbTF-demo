"""Time-correlated residual-wrench coordinates for weak constraints."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


WRENCH_DIMENSION = 6


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
        times = np.asarray(self.times, dtype=float)
        standard_deviation = np.asarray(
            self.stationary_standard_deviation, dtype=float
        )
        correlation_time = float(self.correlation_time)
        if (
            times.ndim != 1
            or times.size < 1
            or not np.all(np.isfinite(times))
            or (times.size > 1 and np.any(np.diff(times) <= 0.0))
        ):
            raise ValueError("times must be a non-empty finite increasing vector")
        if (
            standard_deviation.shape != (WRENCH_DIMENSION,)
            or not np.all(np.isfinite(standard_deviation))
            or np.any(standard_deviation <= 0.0)
        ):
            raise ValueError(
                "stationary_standard_deviation must contain six positive "
                "finite values"
            )
        if not np.isfinite(correlation_time) or correlation_time <= 0.0:
            raise ValueError("correlation_time must be finite and positive")
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

    def decode(self, standard_innovations: Sequence[float]) -> np.ndarray:
        """Decode one flattened standard-innovation vector to ``(N, 6)``.

        Raises:
            ValueError: If the input does not have shape
                ``(innovation_dimension,)``, contains non-finite values, or
                produces values outside floating-point representation.
        """

        values = np.asarray(standard_innovations, dtype=float)
        if values.shape != (self.innovation_dimension,) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "standard_innovations must contain {} finite values".format(
                    self.innovation_dimension
                )
            )
        innovations = values.reshape(self.times.size, WRENCH_DIMENSION)
        process = np.empty_like(innovations)
        with np.errstate(over="raise", invalid="raise"):
            try:
                process[0] = (
                    self.stationary_standard_deviation * innovations[0]
                )
                for index in range(1, self.times.size):
                    time_step = self.times[index] - self.times[index - 1]
                    ratio = time_step / self.correlation_time
                    rho = float(np.exp(-ratio))
                    # ``-expm1(-2 ratio)`` is the stable evaluation of
                    # ``1 - rho**2`` for both very small and large time steps.
                    innovation_variance = float(-np.expm1(-2.0 * ratio))
                    innovation_scale = np.sqrt(
                        max(0.0, innovation_variance)
                    )
                    process[index] = (
                        rho * process[index - 1]
                        + self.stationary_standard_deviation
                        * innovation_scale
                        * innovations[index]
                    )
            except FloatingPointError as error:
                raise ValueError(
                    "decoded wrench is not representable"
                ) from error
        if not np.all(np.isfinite(process)):
            raise ValueError("decoded wrench is not representable")
        return process

    def sample_innovations(self, member_count: int, seed: int) -> np.ndarray:
        """Draw reproducible recentered standard innovations for an ensemble.

        The returned array has shape ``(member_count, innovation_dimension)``.
        Each coordinate is sampled independently before its sample mean across
        members is subtracted.  No covariance shaping, localization, or
        inflation is applied here.
        """

        if isinstance(member_count, (bool, np.bool_)):
            raise ValueError("member_count must be a positive integer")
        count = int(member_count)
        if count != member_count or count <= 0:
            raise ValueError("member_count must be a positive integer")
        generator = np.random.RandomState(int(seed))
        result = generator.normal(
            size=(count, self.innovation_dimension)
        )
        result -= np.mean(result, axis=0, keepdims=True)
        return result


__all__ = ["GaussMarkovWrenchProcess", "WRENCH_DIMENSION"]
