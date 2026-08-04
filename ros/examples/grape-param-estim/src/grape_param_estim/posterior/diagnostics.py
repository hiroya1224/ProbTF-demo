"""Mode-local convergence diagnostics for equal-weight posterior chains."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence, Tuple

import numpy as np

from grape_param_estim.posterior.delayed_acceptance import (
    POSTERIOR_DIMENSION,
    STATIC_PARAMETER_DIMENSION,
)
from grape_param_estim.posterior.mcmc import (
    KernelAcceptanceSummary,
    McmcChainResult,
)


def _chain_array(value: np.ndarray) -> np.ndarray:
    chains = np.asarray(value, dtype=float)
    if (
        chains.ndim != 3
        or chains.shape[0] < 2
        or chains.shape[1] < 4
        or chains.shape[2] == 0
        or not np.all(np.isfinite(chains))
    ):
        raise ValueError(
            "chains must be finite with shape (at least 2, at least 4, d)"
        )
    return chains


def split_rhat(value: np.ndarray) -> np.ndarray:
    """Compute split Gelman-Rubin R-hat independently per coordinate."""

    chains = _chain_array(value)
    chain_count, draw_count, dimension = chains.shape
    half = draw_count // 2
    split = np.concatenate(
        (chains[:, :half, :], chains[:, -half:, :]), axis=0
    )
    means = np.mean(split, axis=1)
    variances = np.var(split, axis=1, ddof=1)
    within = np.mean(variances, axis=0)
    between = half * np.var(means, axis=0, ddof=1)
    variance = (half - 1.0) / half * within + between / half
    result = np.empty(dimension, dtype=float)
    positive = within > 0.0
    result[positive] = np.sqrt(
        np.maximum(0.0, variance[positive] / within[positive])
    )
    zero_between = (~positive) & (between == 0.0)
    result[zero_between] = 1.0
    result[(~positive) & (~zero_between)] = float("inf")
    result.setflags(write=False)
    return result


def _mean_autocovariance_path(chains: np.ndarray) -> np.ndarray:
    """Return biased per-lag autocovariance using a zero-padded FFT."""

    centered = chains - np.mean(chains, axis=1, keepdims=True)
    draw_count = chains.shape[1]
    transform_length = 1 << (2 * draw_count - 1).bit_length()
    transformed = np.fft.rfft(centered, n=transform_length, axis=1)
    autocovariance = np.fft.irfft(
        transformed * np.conjugate(transformed),
        n=transform_length,
        axis=1,
    )[:, :draw_count, :]
    return np.mean(autocovariance, axis=0) / draw_count


def effective_sample_size(value: np.ndarray) -> np.ndarray:
    """Estimate bulk ESS with Geyer's initial positive pair sequence."""

    chains = _chain_array(value)
    chain_count, draw_count, dimension = chains.shape
    within = np.mean(np.var(chains, axis=1, ddof=1), axis=0)
    chain_means = np.mean(chains, axis=1)
    between = draw_count * np.var(chain_means, axis=0, ddof=1)
    variance = (
        (draw_count - 1.0) / draw_count * within
        + between / draw_count
    )
    autocovariance = _mean_autocovariance_path(chains)
    total = float(chain_count * draw_count)
    result = np.empty(dimension, dtype=float)
    for coordinate in range(dimension):
        if variance[coordinate] <= 0.0:
            result[coordinate] = total
            continue
        rho = [1.0]
        for lag in range(1, draw_count):
            covariance = autocovariance[lag, coordinate]
            estimate = 1.0 - (
                within[coordinate] - covariance
            ) / variance[coordinate]
            rho.append(float(estimate))
        positive_pair_sum = 0.0
        previous_pair = float("inf")
        for first_lag in range(1, draw_count - 1, 2):
            pair = rho[first_lag] + rho[first_lag + 1]
            if pair <= 0.0:
                break
            pair = min(pair, previous_pair)
            positive_pair_sum += pair
            previous_pair = pair
        autocorrelation_time = max(1.0, 1.0 + 2.0 * positive_pair_sum)
        result[coordinate] = min(total, total / autocorrelation_time)
    result.setflags(write=False)
    return result


def integrated_autocorrelation_time(value: np.ndarray) -> np.ndarray:
    """Return total draws divided by the coordinate-wise ESS estimate."""

    chains = _chain_array(value)
    total = float(chains.shape[0] * chains.shape[1])
    result = total / effective_sample_size(chains)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class McmcDiagnostics:
    """Completed-chain diagnostics with an independent convergence decision."""

    chain_ids: Tuple[str, ...]
    mode_id: str
    draws_per_chain: int
    split_rhat: np.ndarray
    effective_sample_size: np.ndarray
    integrated_autocorrelation_time: np.ndarray
    ridge_coordinate_trace: np.ndarray
    delay_trace: np.ndarray
    log_density_trace: np.ndarray
    kernel_summaries: Mapping[str, KernelAcceptanceSummary]
    completed: bool
    converged: bool
    rhat_threshold: float
    minimum_effective_sample_size: float

    def __post_init__(self) -> None:
        if (
            type(self.chain_ids) is not tuple
            or len(self.chain_ids) < 2
            or len(set(self.chain_ids)) != len(self.chain_ids)
            or any(not isinstance(value, str) or not value for value in self.chain_ids)
        ):
            raise ValueError("chain_ids must contain at least two unique strings")
        if not isinstance(self.mode_id, str) or not self.mode_id:
            raise ValueError("mode_id must be a non-empty string")
        if (
            isinstance(self.draws_per_chain, (bool, np.bool_))
            or not isinstance(self.draws_per_chain, (int, np.integer))
            or self.draws_per_chain < 4
        ):
            raise ValueError("draws_per_chain must be an integer >= 4")
        chain_shape = (len(self.chain_ids), int(self.draws_per_chain))
        specifications = (
            ("split_rhat", (POSTERIOR_DIMENSION,)),
            ("effective_sample_size", (POSTERIOR_DIMENSION,)),
            ("integrated_autocorrelation_time", (POSTERIOR_DIMENSION,)),
            ("ridge_coordinate_trace", chain_shape),
            ("delay_trace", chain_shape),
            ("log_density_trace", chain_shape),
        )
        for name, shape in specifications:
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or np.any(np.isnan(value)):
                raise ValueError("{} must have shape {} without NaN".format(name, shape))
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        if np.any(self.effective_sample_size <= 0.0):
            raise ValueError("effective_sample_size must be positive")
        if np.any(self.integrated_autocorrelation_time < 1.0):
            raise ValueError("integrated autocorrelation time must be >= 1")
        for name in ("completed", "converged"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError("{} must be boolean".format(name))
            object.__setattr__(self, name, bool(getattr(self, name)))
        if self.converged and not self.completed:
            raise ValueError("an incomplete MCMC run cannot be converged")
        rhat_threshold = float(self.rhat_threshold)
        minimum_ess = float(self.minimum_effective_sample_size)
        if not np.isfinite(rhat_threshold) or rhat_threshold <= 1.0:
            raise ValueError("rhat_threshold must be finite and greater than one")
        if not np.isfinite(minimum_ess) or minimum_ess <= 0.0:
            raise ValueError("minimum_effective_sample_size must be positive")
        object.__setattr__(self, "rhat_threshold", rhat_threshold)
        object.__setattr__(self, "minimum_effective_sample_size", minimum_ess)
        summaries = {}
        for name, summary in self.kernel_summaries.items():
            if not isinstance(name, str) or not name:
                raise ValueError("kernel names must be non-empty strings")
            if not isinstance(summary, KernelAcceptanceSummary):
                raise TypeError("kernel summary values are invalid")
            summaries[name] = summary
        object.__setattr__(self, "kernel_summaries", MappingProxyType(summaries))

    @property
    def maximum_rhat(self) -> float:
        return float(np.max(self.split_rhat))

    @property
    def minimum_ess(self) -> float:
        return float(np.min(self.effective_sample_size))

    @property
    def inner_solve_failure_count(self) -> int:
        return sum(
            value.inner_solve_failures
            for value in self.kernel_summaries.values()
        )


def _aggregate_kernel_summaries(
    chains: Sequence[McmcChainResult],
) -> Mapping[str, KernelAcceptanceSummary]:
    names = sorted(
        set(
            name
            for chain in chains
            for name in chain.kernel_summaries
        )
    )
    fields = (
        "attempts",
        "stage_one_accepted",
        "stage_two_attempted",
        "stage_two_accepted",
        "full_target_cache_hits",
        "inner_solve_failures",
        "inner_iterations",
    )
    return {
        name: KernelAcceptanceSummary(
            **{
                field: sum(
                    getattr(chain.kernel_summaries[name], field)
                    for chain in chains
                    if name in chain.kernel_summaries
                )
                for field in fields
            }
        )
        for name in names
    }


def diagnose_mcmc_chains(
    chains: Sequence[McmcChainResult],
    exact_ridge_direction: np.ndarray,
    *,
    rhat_threshold: float = 1.01,
    minimum_effective_sample_size: float = 100.0,
) -> McmcDiagnostics:
    """Diagnose equal-length chains from one posterior mode only."""

    if not isinstance(chains, (tuple, list)) or len(chains) < 2:
        raise ValueError("at least two completed chains are required")
    if any(not isinstance(chain, McmcChainResult) for chain in chains):
        raise TypeError("chains must contain McmcChainResult values")
    mode_ids = {chain.mode_id for chain in chains}
    if len(mode_ids) != 1:
        raise ValueError("MCMC diagnostics cannot average across mode IDs")
    draw_counts = {chain.static_coordinate.shape[0] for chain in chains}
    if len(draw_counts) != 1 or next(iter(draw_counts)) < 4:
        raise ValueError("chains must have equal retained draw counts >= 4")
    ridge = np.asarray(exact_ridge_direction, dtype=float)
    if ridge.shape != (STATIC_PARAMETER_DIMENSION,) or not np.all(
        np.isfinite(ridge)
    ):
        raise ValueError("exact_ridge_direction must contain 18 finite values")
    norm = float(np.linalg.norm(ridge))
    if norm == 0.0:
        raise ValueError("exact_ridge_direction cannot be zero")
    ridge = ridge / norm

    posterior = np.stack(
        tuple(
            np.column_stack((chain.static_coordinate, chain.delay))
            for chain in chains
        ),
        axis=0,
    )
    rhat = split_rhat(posterior)
    ess = effective_sample_size(posterior)
    autocorrelation_time = integrated_autocorrelation_time(posterior)
    ridge_trace = np.stack(
        tuple(chain.static_coordinate @ ridge for chain in chains), axis=0
    )
    delay_trace = np.stack(tuple(chain.delay for chain in chains), axis=0)
    log_density_trace = np.stack(
        tuple(chain.log_density for chain in chains), axis=0
    )
    rhat_limit = float(rhat_threshold)
    ess_limit = float(minimum_effective_sample_size)
    converged = bool(
        np.all(np.isfinite(rhat))
        and np.max(rhat) <= rhat_limit
        and np.min(ess) >= ess_limit
    )
    return McmcDiagnostics(
        chain_ids=tuple(chain.chain_id for chain in chains),
        mode_id=next(iter(mode_ids)),
        draws_per_chain=posterior.shape[1],
        split_rhat=rhat,
        effective_sample_size=ess,
        integrated_autocorrelation_time=autocorrelation_time,
        ridge_coordinate_trace=ridge_trace,
        delay_trace=delay_trace,
        log_density_trace=log_density_trace,
        kernel_summaries=_aggregate_kernel_summaries(chains),
        completed=True,
        converged=converged,
        rhat_threshold=rhat_limit,
        minimum_effective_sample_size=ess_limit,
    )


__all__ = [
    "McmcDiagnostics",
    "diagnose_mcmc_chains",
    "effective_sample_size",
    "integrated_autocorrelation_time",
    "split_rhat",
]
