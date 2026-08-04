"""Multi-chain orchestration for equal-weight static-parameter MCMC draws."""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.posterior.delayed_acceptance import (
    DelayedAcceptanceSampler,
    DelayedAcceptanceStep,
    PosteriorPoint,
    STATIC_PARAMETER_DIMENSION,
    TargetEvaluation,
)
from grape_param_estim.posterior.diagnostics import (
    McmcDiagnostics,
    diagnose_mcmc_chains,
)
from grape_param_estim.posterior.mcmc import (
    McmcChainResult,
    McmcChainSettings,
    run_mcmc_chain,
)


SamplerFactory = Callable[[str], DelayedAcceptanceSampler]
CancellationCheck = Callable[[], bool]
McmcRunProgress = Callable[
    [int, int, int, int, DelayedAcceptanceStep], None
]
CompletedChainCheckpoint = Callable[[McmcChainResult], None]


def _integer(value: object, minimum: int, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) < minimum
    ):
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return int(value)


def _positive(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("{} must be a real scalar".format(name))
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("{} must be finite and positive".format(name))
    return result


@dataclass(frozen=True)
class McmcRunSettings:
    """Common settings for independent chains in one posterior mode."""

    mode_id: str
    chain_count: int
    warmup_steps: int
    retained_draws: int
    thinning: int
    random_seed: int
    rhat_threshold: float
    minimum_effective_sample_size: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.mode_id, str)
            or not self.mode_id
            or self.mode_id.strip() != self.mode_id
        ):
            raise ValueError("mode_id must be a canonical non-empty string")
        object.__setattr__(
            self, "chain_count", _integer(self.chain_count, 2, "chain_count")
        )
        object.__setattr__(
            self, "warmup_steps", _integer(self.warmup_steps, 0, "warmup_steps")
        )
        object.__setattr__(
            self,
            "retained_draws",
            _integer(self.retained_draws, 4, "retained_draws"),
        )
        object.__setattr__(
            self, "thinning", _integer(self.thinning, 1, "thinning")
        )
        object.__setattr__(
            self, "random_seed", _integer(self.random_seed, 0, "random_seed")
        )
        rhat = _positive(self.rhat_threshold, "rhat_threshold")
        if rhat <= 1.0:
            raise ValueError("rhat_threshold must exceed one")
        object.__setattr__(self, "rhat_threshold", rhat)
        object.__setattr__(
            self,
            "minimum_effective_sample_size",
            _positive(
                self.minimum_effective_sample_size,
                "minimum_effective_sample_size",
            ),
        )

    def chain_settings(self, chain_index: int) -> McmcChainSettings:
        index = _integer(chain_index, 0, "chain_index")
        if index >= self.chain_count:
            raise ValueError("chain_index is outside this run")
        return McmcChainSettings(
            chain_id="chain-{:03d}".format(index),
            mode_id=self.mode_id,
            warmup_steps=self.warmup_steps,
            retained_draws=self.retained_draws,
            thinning=self.thinning,
        )


@dataclass(frozen=True)
class ChainInitialization:
    """One audited initial point before its exact target evaluation."""

    chain_id: str
    source: str
    point: PosteriorPoint

    def __post_init__(self) -> None:
        for name in ("chain_id", "source"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or value.strip() != value
            ):
                raise ValueError("{} must be canonical text".format(name))
        if not isinstance(self.point, PosteriorPoint):
            raise TypeError("point must be PosteriorPoint")


@dataclass(frozen=True)
class McmcRunResult:
    """Completed chains, convergence diagnostics, and initialization audit."""

    settings: McmcRunSettings
    initializations: Tuple[ChainInitialization, ...]
    chains: Tuple[McmcChainResult, ...]
    diagnostics: McmcDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.settings, McmcRunSettings):
            raise TypeError("settings must be McmcRunSettings")
        if (
            type(self.initializations) is not tuple
            or len(self.initializations) != self.settings.chain_count
            or any(
                not isinstance(value, ChainInitialization)
                for value in self.initializations
            )
        ):
            raise ValueError("initializations must contain one item per chain")
        if (
            type(self.chains) is not tuple
            or len(self.chains) != self.settings.chain_count
            or any(not isinstance(value, McmcChainResult) for value in self.chains)
        ):
            raise ValueError("chains must contain every completed chain")
        initial_ids = tuple(value.chain_id for value in self.initializations)
        chain_ids = tuple(value.chain_id for value in self.chains)
        expected = tuple(
            self.settings.chain_settings(index).chain_id
            for index in range(self.settings.chain_count)
        )
        if initial_ids != expected or chain_ids != expected:
            raise ValueError("chain IDs must be canonical and ordered")
        sample_ids = [
            str(sample_id)
            for chain in self.chains
            for sample_id in chain.sample_id.tolist()
        ]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample IDs must be globally unique")
        if not isinstance(self.diagnostics, McmcDiagnostics):
            raise TypeError("diagnostics must be McmcDiagnostics")
        if self.diagnostics.chain_ids != expected:
            raise ValueError("diagnostic chain IDs disagree with chains")


def _bounded_delay(value: float, bounds: Tuple[float, float]) -> float:
    lower, upper = bounds
    selected = float(value)
    # Reflect rather than pile unconstrained Gaussian draws onto a boundary.
    width = upper - lower
    folded = (selected - lower) % (2.0 * width)
    if folded > width:
        folded = 2.0 * width - folded
    return float(lower + folded)


def initialize_mcmc_chains(
    map_point: PosteriorPoint,
    static_covariance: np.ndarray,
    delay_standard_deviation: float,
    exact_ridge_direction: np.ndarray,
    delay_bounds: Tuple[float, float],
    chain_count: int,
    random_seed: int,
) -> Tuple[ChainInitialization, ...]:
    """Create MAP, Laplace, and ridge-dispersed chain initial points."""

    if not isinstance(map_point, PosteriorPoint):
        raise TypeError("map_point must be PosteriorPoint")
    covariance = np.asarray(static_covariance, dtype=float)
    if (
        covariance.shape
        != (STATIC_PARAMETER_DIMENSION, STATIC_PARAMETER_DIMENSION)
        or not np.all(np.isfinite(covariance))
        or not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-10)
    ):
        raise ValueError("static_covariance must be finite symmetric 18 by 18")
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("static_covariance must be positive definite") from error
    delay_scale = _positive(
        delay_standard_deviation, "delay_standard_deviation"
    )
    ridge = np.asarray(exact_ridge_direction, dtype=float)
    if ridge.shape != (STATIC_PARAMETER_DIMENSION,) or not np.all(
        np.isfinite(ridge)
    ):
        raise ValueError("exact_ridge_direction must contain 18 finite values")
    norm = float(np.linalg.norm(ridge))
    if norm <= 0.0:
        raise ValueError("exact_ridge_direction cannot be zero")
    ridge = ridge / norm
    if type(delay_bounds) is not tuple or len(delay_bounds) != 2:
        raise TypeError("delay_bounds must be a (lower, upper) tuple")
    lower, upper = (float(item) for item in delay_bounds)
    if not np.all(np.isfinite((lower, upper))) or lower >= upper:
        raise ValueError("delay_bounds must be finite and increasing")
    count = _integer(chain_count, 2, "chain_count")
    seed = _integer(random_seed, 0, "random_seed")
    random = np.random.RandomState(seed)
    ridge_deviation = float(np.sqrt(ridge @ covariance @ ridge))

    result = [
        ChainInitialization("chain-000", "map", map_point)
    ]
    for index in range(1, count):
        if index % 2:
            coordinate = (
                map_point.static_coordinate
                + 0.5 * cholesky @ random.normal(size=18)
            )
            source = "laplace_dispersion"
        else:
            sign = -1.0 if (index // 2) % 2 else 1.0
            coordinate = (
                map_point.static_coordinate
                + sign * max(ridge_deviation, 0.1) * ridge
                + 0.1 * cholesky @ random.normal(size=18)
            )
            source = "exact_ridge_dispersion"
        delay = _bounded_delay(
            map_point.delay + delay_scale * random.normal(),
            (lower, upper),
        )
        result.append(
            ChainInitialization(
                "chain-{:03d}".format(index),
                source,
                PosteriorPoint(coordinate, delay),
            )
        )
    return tuple(result)


def run_mcmc_chains(
    sampler_factory: SamplerFactory,
    settings: McmcRunSettings,
    initializations: Sequence[ChainInitialization],
    exact_ridge_direction: np.ndarray,
    *,
    completed_chains: Optional[Mapping[str, McmcChainResult]] = None,
    cancellation_requested: Optional[CancellationCheck] = None,
    progress: Optional[McmcRunProgress] = None,
    checkpoint_completed_chain: Optional[CompletedChainCheckpoint] = None,
) -> McmcRunResult:
    """Run or resume independent chains and compute diagnostics.

    ``completed_chains`` is the immutable chain-level resume boundary.  A
    completed chain is never rerun, while the single-chain runner still checks
    cancellation before every proposal.
    """

    if not callable(sampler_factory):
        raise TypeError("sampler_factory must be callable")
    if not isinstance(settings, McmcRunSettings):
        raise TypeError("settings must be McmcRunSettings")
    selected_initializations = tuple(initializations)
    expected_ids = tuple(
        settings.chain_settings(index).chain_id
        for index in range(settings.chain_count)
    )
    if (
        len(selected_initializations) != settings.chain_count
        or any(
            not isinstance(value, ChainInitialization)
            for value in selected_initializations
        )
        or tuple(value.chain_id for value in selected_initializations)
        != expected_ids
    ):
        raise ValueError("initializations must match canonical chain IDs")
    resumed = {} if completed_chains is None else dict(completed_chains)
    if not set(resumed).issubset(set(expected_ids)):
        raise ValueError("completed_chains contains an unknown chain ID")
    for chain_id, chain in resumed.items():
        if (
            not isinstance(chain, McmcChainResult)
            or chain.chain_id != chain_id
            or chain.mode_id != settings.mode_id
            or chain.static_coordinate.shape[0] != settings.retained_draws
            or chain.warmup_steps != settings.warmup_steps
            or chain.thinning != settings.thinning
        ):
            raise ValueError("completed chain does not match run settings")
    for callback, name in (
        (cancellation_requested, "cancellation_requested"),
        (progress, "progress"),
        (checkpoint_completed_chain, "checkpoint_completed_chain"),
    ):
        if callback is not None and not callable(callback):
            raise TypeError("{} must be callable".format(name))

    chains = []
    for chain_index, initialization in enumerate(selected_initializations):
        chain_settings = settings.chain_settings(chain_index)
        if chain_settings.chain_id in resumed:
            chains.append(resumed[chain_settings.chain_id])
            continue
        if cancellation_requested is not None and cancellation_requested():
            from grape_param_estim.posterior.mcmc import McmcCancelled

            raise McmcCancelled(0)
        sampler = sampler_factory(chain_settings.chain_id)
        if not isinstance(sampler, DelayedAcceptanceSampler):
            raise TypeError(
                "sampler_factory must return DelayedAcceptanceSampler"
            )
        initial = sampler.target_evaluator(initialization.point, None)
        if not isinstance(initial, TargetEvaluation):
            raise TypeError("target evaluator must return TargetEvaluation")
        if not initial.successful:
            raise ValueError(
                "initial target failed for {}: {}".format(
                    chain_settings.chain_id, initial.failure_reason
                )
            )
        random = np.random.RandomState(
            (settings.random_seed + 104729 * (chain_index + 1))
            % (2**32)
        )

        def chain_progress(completed, total, step):
            if progress is not None:
                progress(
                    chain_index,
                    settings.chain_count,
                    completed,
                    total,
                    step,
                )

        chain = run_mcmc_chain(
            sampler,
            initial,
            chain_settings,
            random,
            cancellation_requested=cancellation_requested,
            progress=chain_progress,
        )
        chains.append(chain)
        if checkpoint_completed_chain is not None:
            checkpoint_completed_chain(chain)

    diagnostics = diagnose_mcmc_chains(
        chains,
        exact_ridge_direction,
        rhat_threshold=settings.rhat_threshold,
        minimum_effective_sample_size=(
            settings.minimum_effective_sample_size
        ),
    )
    return McmcRunResult(
        settings=settings,
        initializations=selected_initializations,
        chains=tuple(chains),
        diagnostics=diagnostics,
    )


__all__ = [
    "ChainInitialization",
    "CompletedChainCheckpoint",
    "McmcRunProgress",
    "McmcRunResult",
    "McmcRunSettings",
    "SamplerFactory",
    "initialize_mcmc_chains",
    "run_mcmc_chains",
]
