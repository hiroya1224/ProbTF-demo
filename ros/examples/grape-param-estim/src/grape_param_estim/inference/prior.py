"""Proper bounded priors used by plant assimilation."""

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import numpy as np

from grape_param_estim._legacy_inference import BoxUniformPrior


@dataclass(frozen=True)
class PriorDimension:
    name: str
    kind: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        name = str(self.name)
        kind = str(self.kind)
        lower = float(self.lower)
        upper = float(self.upper)
        if not name:
            raise ValueError("prior dimension name is required")
        if kind not in ("bounded_uniform", "bounded_log_uniform"):
            raise ValueError("unsupported prior kind: {}".format(kind))
        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or upper <= lower
            or (kind == "bounded_log_uniform" and lower <= 0.0)
        ):
            raise ValueError("prior bounds are invalid")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


class BoundedLogUniformPrior:
    """Independent log-uniform law on a finite positive box."""

    def __init__(self, lower: Any, upper: Any) -> None:
        self.lower = np.asarray(lower, dtype=float).reshape(-1)
        self.upper = np.asarray(upper, dtype=float).reshape(-1)
        if (
            self.lower.size == 0
            or self.upper.shape != self.lower.shape
            or np.any(self.lower <= 0.0)
            or np.any(self.upper <= self.lower)
            or not np.all(np.isfinite(self.lower))
            or not np.all(np.isfinite(self.upper))
        ):
            raise ValueError("log-uniform prior needs finite 0 < lower < upper")
        self.dimension = int(self.lower.size)
        self._log_normalizer = np.log(
            np.log(self.upper / self.lower)
        )

    def sample(self, count: int, rng: Any) -> np.ndarray:
        log_values = rng.uniform(
            np.log(self.lower),
            np.log(self.upper),
            size=(int(count), self.dimension),
        )
        return np.exp(log_values)

    def log_prob(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape[-1] != self.dimension:
            raise ValueError("prior values have invalid shape")
        inside = np.all(
            (array >= self.lower) & (array <= self.upper), axis=-1
        )
        density = -np.sum(np.log(array) + self._log_normalizer, axis=-1)
        return np.where(inside, density, -np.inf)


class IndependentBoundedPrior:
    """Mixed bounded-uniform/log-uniform product prior."""

    def __init__(self, dimensions: Sequence[PriorDimension]) -> None:
        self.dimensions_spec: Tuple[PriorDimension, ...] = tuple(dimensions)
        if (
            not self.dimensions_spec
            or len({item.name for item in self.dimensions_spec})
            != len(self.dimensions_spec)
        ):
            raise ValueError("prior dimensions must be non-empty and unique")
        self.dimension = len(self.dimensions_spec)
        self.lower = np.asarray([item.lower for item in self.dimensions_spec])
        self.upper = np.asarray([item.upper for item in self.dimensions_spec])

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(item.name for item in self.dimensions_spec)

    def sample(self, count: int, rng: Any) -> np.ndarray:
        count_value = int(count)
        if count_value < 1:
            raise ValueError("sample count must be positive")
        result = np.empty((count_value, self.dimension))
        for index, item in enumerate(self.dimensions_spec):
            if item.kind == "bounded_uniform":
                result[:, index] = rng.uniform(item.lower, item.upper, count_value)
            else:
                result[:, index] = np.exp(
                    rng.uniform(
                        np.log(item.lower), np.log(item.upper), count_value
                    )
                )
        epsilon = np.sqrt(np.finfo(float).eps)
        return np.clip(
            result,
            self.lower + epsilon * (self.upper - self.lower),
            self.upper - epsilon * (self.upper - self.lower),
        )

    def log_prob(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape[-1] != self.dimension:
            raise ValueError("prior values have invalid shape")
        result = np.zeros(array.shape[:-1])
        inside = np.ones(array.shape[:-1], dtype=bool)
        for index, item in enumerate(self.dimensions_spec):
            value = array[..., index]
            inside &= (value >= item.lower) & (value <= item.upper)
            if item.kind == "bounded_uniform":
                result -= np.log(item.upper - item.lower)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    result -= np.log(value) + np.log(
                        np.log(item.upper / item.lower)
                    )
        return np.where(inside, result, -np.inf)


__all__ = [
    "BoundedLogUniformPrior",
    "BoxUniformPrior",
    "IndependentBoundedPrior",
    "PriorDimension",
]
