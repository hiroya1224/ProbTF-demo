"""Reusable transformed-prior tempered resample-move SMC primitives."""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


def _logsumexp(values, axis=None):
    array = np.asarray(values, dtype=float)
    maximum = np.max(array, axis=axis, keepdims=True)
    finite_maximum = np.isfinite(maximum)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        shifted = np.where(finite_maximum, array - maximum, -np.inf)
        result = maximum + np.log(
            np.sum(np.exp(shifted), axis=axis, keepdims=True)
        )
    result = np.where(finite_maximum, result, maximum)
    if axis is None:
        return float(result.reshape(-1)[0])
    return np.squeeze(result, axis=axis)


def _normalized(log_weights):
    normalizer = _logsumexp(log_weights)
    values = np.exp(np.asarray(log_weights, dtype=float) - normalizer)
    values /= np.sum(values)
    return values, normalizer


def effective_sample_size(weights):
    values = np.asarray(weights, dtype=float).reshape(-1)
    if values.size == 0 or np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("weights must be finite and non-negative")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("weights must have positive mass")
    normalized = values / total
    return float(1.0 / np.dot(normalized, normalized))


def systematic_resample(weights, rng):
    values = np.asarray(weights, dtype=float).reshape(-1)
    total = float(np.sum(values))
    if (
        values.size == 0
        or np.any(values < 0.0)
        or not np.all(np.isfinite(values))
        or total <= 0.0
    ):
        raise ValueError("weights must be a finite positive measure")
    values = values / total
    cumulative = np.cumsum(values)
    cumulative[-1] = 1.0
    locations = (float(rng.random()) + np.arange(values.size)) / values.size
    return np.searchsorted(cumulative, locations, side="right")


class IdentityTransform:
    def __init__(self, dimension):
        self.dimension = int(dimension)
        if self.dimension < 1:
            raise ValueError("dimension must be positive")

    def to_unconstrained(self, values):
        array = np.asarray(values, dtype=float)
        if array.shape[-1] != self.dimension or not np.all(np.isfinite(array)):
            raise ValueError("identity-transform values have invalid shape")
        return np.array(array, copy=True)

    def from_unconstrained(self, values):
        return self.to_unconstrained(values)

    def log_abs_det_jacobian(self, unconstrained):
        values = np.asarray(unconstrained, dtype=float)
        if values.shape[-1] != self.dimension:
            raise ValueError("unconstrained values have invalid shape")
        return np.zeros(values.shape[:-1])


class BoundedLogitTransform:
    """Elementwise map between a finite box and unconstrained space."""

    def __init__(self, lower, upper):
        self.lower = np.asarray(lower, dtype=float).reshape(-1)
        self.upper = np.asarray(upper, dtype=float).reshape(-1)
        if (
            self.lower.size == 0
            or self.upper.shape != self.lower.shape
            or not np.all(np.isfinite(self.lower))
            or not np.all(np.isfinite(self.upper))
            or np.any(self.upper <= self.lower)
        ):
            raise ValueError("bounded transform needs finite lower < upper")
        self.dimension = self.lower.size

    def to_unconstrained(self, values):
        array = np.asarray(values, dtype=float)
        if array.shape[-1] != self.dimension or np.any(array <= self.lower) or np.any(array >= self.upper):
            raise ValueError("values must lie strictly inside transform bounds")
        probability = (array - self.lower) / (self.upper - self.lower)
        return np.log(probability) - np.log1p(-probability)

    def from_unconstrained(self, values):
        array = np.asarray(values, dtype=float)
        if array.shape[-1] != self.dimension or not np.all(np.isfinite(array)):
            raise ValueError("unconstrained values have invalid shape")
        probability = np.empty_like(array)
        positive = array >= 0.0
        probability[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
        exponential = np.exp(array[~positive])
        probability[~positive] = exponential / (1.0 + exponential)
        # Preserve a strict interior after floating-point saturation.
        epsilon = np.finfo(float).eps
        probability = np.clip(probability, epsilon, 1.0 - epsilon)
        return self.lower + (self.upper - self.lower) * probability

    def log_abs_det_jacobian(self, unconstrained):
        values = np.asarray(unconstrained, dtype=float)
        transformed = self.from_unconstrained(values)
        probability = (transformed - self.lower) / (self.upper - self.lower)
        return np.sum(
            np.log(self.upper - self.lower)
            + np.log(probability)
            + np.log1p(-probability),
            axis=-1,
        )


class BoxUniformPrior:
    def __init__(self, lower, upper):
        self.lower = np.asarray(lower, dtype=float).reshape(-1)
        self.upper = np.asarray(upper, dtype=float).reshape(-1)
        if (
            self.lower.size == 0
            or self.upper.shape != self.lower.shape
            or not np.all(np.isfinite(self.lower))
            or not np.all(np.isfinite(self.upper))
            or np.any(self.upper <= self.lower)
        ):
            raise ValueError("box prior needs finite lower < upper")
        self.dimension = self.lower.size
        self._log_density = -float(np.sum(np.log(self.upper - self.lower)))

    def sample(self, count, rng):
        values = rng.uniform(
            self.lower, self.upper, size=(int(count), self.dimension)
        )
        # A logit transform has open support.  NumPy's uniform is half-open and
        # can return the lower endpoint when the generator emits exactly zero.
        width = self.upper - self.lower
        epsilon = np.sqrt(np.finfo(float).eps)
        interior_lower = self.lower + epsilon * width
        interior_upper = self.upper - epsilon * width
        return np.clip(values, interior_lower, interior_upper)

    def log_prob(self, values):
        array = np.asarray(values, dtype=float)
        if array.shape[-1] != self.dimension:
            raise ValueError("prior values have invalid shape")
        inside = np.all((array >= self.lower) & (array <= self.upper), axis=-1)
        return np.where(inside, self._log_density, -np.inf)


@dataclass(frozen=True)
class TemperedSmcConfig:
    particle_count: int = 1024
    target_ess_fraction: float = 0.7
    resample_ess_fraction: float = 0.5
    mcmc_steps: int = 2
    proposal_scale: float = 0.7
    max_tempering_steps: int = 128
    seed: int = 7

    def __post_init__(self):
        if int(self.particle_count) < 32:
            raise ValueError("particle_count must be at least 32")
        if not 0.0 < float(self.resample_ess_fraction) <= float(self.target_ess_fraction) <= 1.0:
            raise ValueError("ESS fractions require 0 < resample <= target <= 1")
        if int(self.mcmc_steps) < 0 or float(self.proposal_scale) <= 0.0:
            raise ValueError("MCMC settings are invalid")
        if int(self.max_tempering_steps) < 1:
            raise ValueError("max_tempering_steps must be positive")


@dataclass(frozen=True)
class SmcStage:
    inverse_temperature: float
    increment: float
    ess_before: float
    ess_after_weighting: float
    ess_after: float
    resampled: bool
    mcmc_accepted: int
    mcmc_proposed: int
    log_evidence_increment: float


@dataclass(frozen=True)
class SmcPosterior:
    particles: np.ndarray
    weights: np.ndarray
    log_likelihood: np.ndarray
    log_evidence: float
    stages: Tuple[SmcStage, ...]
    seed: int

    def mean(self):
        return np.average(self.particles, axis=0, weights=self.weights)

    def covariance(self):
        mean = self.mean()
        centered = self.particles - mean
        return (centered * self.weights[:, None]).T @ centered


class TemperedResampleMoveSmc:
    def __init__(self, prior, transform, config=TemperedSmcConfig()):
        self.prior = prior
        self.transform = transform
        self.config = config
        if prior.dimension != transform.dimension:
            raise ValueError("prior and transform dimensions must match")

    def _choose_increment(self, log_weights, log_likelihood, remaining):
        target = self.config.target_ess_fraction * self.config.particle_count

        def candidate_ess(value):
            weights, _ = _normalized(log_weights + value * log_likelihood)
            return effective_sample_size(weights)

        if candidate_ess(remaining) >= target:
            return remaining
        low, high = 0.0, remaining
        for _ in range(48):
            midpoint = 0.5 * (low + high)
            if candidate_ess(midpoint) < target:
                high = midpoint
            else:
                low = midpoint
        return max(low, min(remaining, 1.0e-10))

    def _move(self, particles, likelihood, beta, rng, log_likelihood_function):
        if self.config.mcmc_steps == 0:
            return particles, likelihood, 0, 0
        unconstrained = self.transform.to_unconstrained(particles)
        covariance = np.cov(unconstrained, rowvar=False)
        covariance = np.atleast_2d(covariance)
        covariance += np.eye(covariance.shape[0]) * 1.0e-8
        covariance *= (
            self.config.proposal_scale * 2.38 / np.sqrt(self.prior.dimension)
        ) ** 2

        def target(values, known_likelihood=None):
            constrained = self.transform.from_unconstrained(values)
            evaluated = (
                np.asarray(log_likelihood_function(constrained), dtype=float)
                if known_likelihood is None
                else known_likelihood
            )
            density = (
                self.prior.log_prob(constrained)
                + beta * evaluated
                + self.transform.log_abs_det_jacobian(values)
            )
            return constrained, evaluated, density

        _, _, current_target = target(unconstrained, likelihood)
        accepted = 0
        proposed = 0
        for _ in range(self.config.mcmc_steps):
            proposal_unconstrained = unconstrained + rng.multivariate_normal(
                np.zeros(self.prior.dimension),
                covariance,
                size=self.config.particle_count,
            )
            proposal, proposal_likelihood, proposal_target = target(
                proposal_unconstrained
            )
            accept = np.log(rng.random(self.config.particle_count)) < (
                proposal_target - current_target
            )
            unconstrained[accept] = proposal_unconstrained[accept]
            particles[accept] = proposal[accept]
            likelihood[accept] = proposal_likelihood[accept]
            current_target[accept] = proposal_target[accept]
            accepted += int(np.count_nonzero(accept))
            proposed += self.config.particle_count
        return particles, likelihood, accepted, proposed

    def run(self, log_likelihood_function):
        rng = np.random.default_rng(int(self.config.seed))
        particles = np.asarray(
            self.prior.sample(self.config.particle_count, rng), dtype=float
        )
        likelihood = np.asarray(log_likelihood_function(particles), dtype=float)
        if likelihood.shape != (self.config.particle_count,) or not np.all(
            np.isfinite(likelihood)
        ):
            raise ValueError("log-likelihood function must return one finite value per particle")
        log_weights = np.full(
            self.config.particle_count, -np.log(self.config.particle_count)
        )
        weights = np.full(
            self.config.particle_count, 1.0 / self.config.particle_count
        )
        beta = 0.0
        log_evidence = 0.0
        stages = []
        while beta < 1.0 - 1.0e-12:
            ess_before = effective_sample_size(weights)
            increment = self._choose_increment(
                log_weights, likelihood, 1.0 - beta
            )
            unnormalized = log_weights + increment * likelihood
            weights, normalizer = _normalized(unnormalized)
            log_evidence += normalizer
            log_weights = np.log(weights)
            beta = min(1.0, beta + increment)
            ess_weighted = effective_sample_size(weights)
            resampled = (
                beta < 1.0 - 1.0e-12
                or ess_weighted
                < self.config.resample_ess_fraction
                * self.config.particle_count
            )
            accepted = proposed = 0
            if resampled:
                ancestors = systematic_resample(weights, rng)
                particles = particles[ancestors].copy()
                likelihood = likelihood[ancestors].copy()
                weights.fill(1.0 / self.config.particle_count)
                log_weights.fill(-np.log(self.config.particle_count))
                particles, likelihood, accepted, proposed = self._move(
                    particles,
                    likelihood,
                    beta,
                    rng,
                    log_likelihood_function,
                )
            stages.append(
                SmcStage(
                    inverse_temperature=beta,
                    increment=increment,
                    ess_before=ess_before,
                    ess_after_weighting=ess_weighted,
                    ess_after=effective_sample_size(weights),
                    resampled=resampled,
                    mcmc_accepted=accepted,
                    mcmc_proposed=proposed,
                    log_evidence_increment=normalizer,
                )
            )
            if len(stages) > self.config.max_tempering_steps:
                raise RuntimeError("tempering did not reach unit inverse temperature")
        return SmcPosterior(
            particles=particles,
            weights=weights,
            log_likelihood=likelihood,
            log_evidence=log_evidence,
            stages=tuple(stages),
            seed=int(self.config.seed),
        )


def marginalize_trajectory_log_likelihood(
    conditional_log_likelihood: np.ndarray,
    trajectory_weights: np.ndarray,
) -> np.ndarray:
    """Marginalize trajectory samples without replacing them by a point state."""

    values = np.asarray(conditional_log_likelihood, dtype=float)
    weights = np.asarray(trajectory_weights, dtype=float).reshape(-1)
    if (
        values.ndim != 2
        or values.shape[1] != weights.size
        or np.any(weights < 0.0)
        or not np.isclose(np.sum(weights), 1.0)
    ):
        raise ValueError("likelihood matrix and trajectory weights are incompatible")
    log_weights = np.full(weights.shape, -np.inf)
    positive = weights > 0.0
    log_weights[positive] = np.log(weights[positive])
    return _logsumexp(values + log_weights[None, :], axis=1)


@dataclass(frozen=True)
class ChainDiagnostics:
    r_hat: np.ndarray
    maximum_r_hat: float
    posterior_mean_spread: np.ndarray
    converged: bool


def chain_diagnostics(posteriors: Sequence[SmcPosterior], draw_count=512, seed=123):
    """Between-seed mixing diagnostic analogous to an R-hat check."""

    chains = tuple(posteriors)
    if len(chains) < 2:
        raise ValueError("at least two SMC chains are required")
    dimension = chains[0].particles.shape[1]
    if any(item.particles.shape[1] != dimension for item in chains):
        raise ValueError("SMC chains have different dimensions")
    count = int(draw_count)
    rng = np.random.default_rng(int(seed))
    draws = np.empty((len(chains), count, dimension))
    for index, item in enumerate(chains):
        ancestors = rng.choice(
            item.particles.shape[0], size=count, replace=True, p=item.weights
        )
        draws[index] = item.particles[ancestors]
    chain_means = np.mean(draws, axis=1)
    within = np.mean(np.var(draws, axis=1, ddof=1), axis=0)
    between = count * np.var(chain_means, axis=0, ddof=1)
    variance = (count - 1.0) / count * within + between / count
    r_hat = np.sqrt(
        np.divide(
            variance,
            within,
            out=np.full_like(variance, np.inf),
            where=within > 0.0,
        )
    )
    spread = np.ptp(chain_means, axis=0)
    maximum = float(np.max(r_hat))
    return ChainDiagnostics(
        r_hat=r_hat,
        maximum_r_hat=maximum,
        posterior_mean_spread=spread,
        converged=bool(maximum <= 1.1),
    )


def predictive_interval_coverage(
    observations: np.ndarray,
    predictive_samples: np.ndarray,
    probabilities=(0.5, 0.8, 0.95),
):
    observed = np.asarray(observations, dtype=float)
    samples = np.asarray(predictive_samples, dtype=float)
    if samples.ndim != observed.ndim + 1 or samples.shape[1:] != observed.shape:
        raise ValueError("predictive samples must have shape (S, ...) over observations")
    output = {}
    previous = -np.inf
    for probability in probabilities:
        level = float(probability)
        if not 0.0 < level < 1.0 or level <= previous:
            raise ValueError("probabilities must be increasing values in (0, 1)")
        tail = 0.5 * (1.0 - level)
        lower = np.quantile(samples, tail, axis=0)
        upper = np.quantile(samples, 1.0 - tail, axis=0)
        output[level] = float(np.mean((observed >= lower) & (observed <= upper)))
        previous = level
    return output


__all__ = [
    "BoundedLogitTransform",
    "BoxUniformPrior",
    "ChainDiagnostics",
    "IdentityTransform",
    "SmcPosterior",
    "SmcStage",
    "TemperedResampleMoveSmc",
    "TemperedSmcConfig",
    "chain_diagnostics",
    "effective_sample_size",
    "marginalize_trajectory_log_likelihood",
    "predictive_interval_coverage",
    "systematic_resample",
]
