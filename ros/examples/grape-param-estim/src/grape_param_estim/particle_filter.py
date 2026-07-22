"""Sequential Monte Carlo for static Grape inertial parameters.

The filter deliberately receives only bounded support, never a nominal URDF
mean.  A batch likelihood is introduced through adaptive tempering, followed
by systematic resampling and Metropolis resample-move rejuvenation.  This is a
genuine weighted particle method (unlike a deterministic particle scan) and
keeps the complete particle state available for rosbag diagnostics.
"""

from dataclasses import dataclass
from math import lgamma, pi
from typing import Iterable, Sequence, Tuple

import numpy as np

from .dynamics import PARAMETER_NAMES, physical_parameter_mask, predict_wrench


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return maximum
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def _normalize_log_weights(log_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normalizer = _logsumexp(log_weights)
    if not np.isfinite(normalizer):
        raise FloatingPointError("All particle weights became non-finite.")
    normalized_log = log_weights - normalizer
    weights = np.exp(normalized_log)
    weights /= np.sum(weights)
    return normalized_log, weights


def effective_sample_size(weights: np.ndarray) -> float:
    """Return the usual importance-weight effective sample size."""

    values = np.asarray(weights, dtype=float).reshape(-1)
    total = float(np.sum(values))
    if values.size == 0 or not np.isfinite(total) or total <= 0.0:
        raise ValueError("weights must be a non-empty finite positive measure.")
    values = values / total
    return float(1.0 / np.dot(values, values))


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw low-variance systematic-resampling ancestor indices."""

    values = np.asarray(weights, dtype=float).reshape(-1)
    if values.size == 0 or np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("weights must be finite and non-negative.")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("weights must have positive total mass.")
    values = values / total
    cumulative = np.cumsum(values)
    cumulative[-1] = 1.0
    positions = (float(rng.random()) + np.arange(values.size)) / values.size
    return np.searchsorted(cumulative, positions, side="right")


@dataclass(frozen=True)
class ParameterBounds:
    """Finite support for the non-nominal-centred uniform initial law."""

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float).reshape(-1)
        upper = np.asarray(self.upper, dtype=float).reshape(-1)
        if lower.shape != (len(PARAMETER_NAMES),) or upper.shape != lower.shape:
            raise ValueError("parameter bounds must contain one value per parameter.")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("parameter bounds must be finite.")
        if np.any(upper <= lower):
            raise ValueError("every upper parameter bound must exceed its lower bound.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower

    def reflect(self, values: np.ndarray) -> np.ndarray:
        """Reflect proposals at finite bounds without creating edge atoms."""

        array = np.asarray(values, dtype=float)
        phase = np.mod(array - self.lower, 2.0 * self.width)
        return np.where(
            phase <= self.width,
            self.lower + phase,
            self.upper - (phase - self.width),
        )


@dataclass(frozen=True)
class ObservationBatch:
    """Synchronized body kinematics and calibrated actuator wrench samples."""

    specific_acceleration: np.ndarray
    angular_velocity: np.ndarray
    angular_acceleration: np.ndarray
    actuator_wrench: np.ndarray
    force_sigma: np.ndarray
    torque_sigma: np.ndarray

    def __post_init__(self) -> None:
        specific = np.asarray(self.specific_acceleration, dtype=float)
        omega = np.asarray(self.angular_velocity, dtype=float)
        alpha = np.asarray(self.angular_acceleration, dtype=float)
        wrench = np.asarray(self.actuator_wrench, dtype=float)
        if specific.ndim != 2 or specific.shape[1] != 3:
            raise ValueError("specific_acceleration must have shape (K, 3).")
        count = specific.shape[0]
        if count == 0:
            raise ValueError("an observation batch must not be empty.")
        for name, value, columns in (
            ("angular_velocity", omega, 3),
            ("angular_acceleration", alpha, 3),
            ("actuator_wrench", wrench, 6),
        ):
            if value.shape != (count, columns):
                raise ValueError("{} has an incompatible shape.".format(name))
        force_sigma = np.broadcast_to(np.asarray(self.force_sigma, dtype=float), (count, 3)).copy()
        torque_sigma = np.broadcast_to(np.asarray(self.torque_sigma, dtype=float), (count, 3)).copy()
        arrays = (specific, omega, alpha, wrench, force_sigma, torque_sigma)
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("observation values and noise scales must be finite.")
        if np.any(force_sigma <= 0.0) or np.any(torque_sigma <= 0.0):
            raise ValueError("observation noise scales must be positive.")
        object.__setattr__(self, "specific_acceleration", specific)
        object.__setattr__(self, "angular_velocity", omega)
        object.__setattr__(self, "angular_acceleration", alpha)
        object.__setattr__(self, "actuator_wrench", wrench)
        object.__setattr__(self, "force_sigma", force_sigma)
        object.__setattr__(self, "torque_sigma", torque_sigma)

    @property
    def count(self) -> int:
        return int(self.specific_acceleration.shape[0])


@dataclass(frozen=True)
class ParticleFilterConfig:
    particle_count: int = 4096
    resample_ess_fraction: float = 0.5
    tempering_ess_fraction: float = 0.7
    mcmc_steps: int = 2
    local_move_scale: float = 0.35
    prior_move_probability: float = 0.03
    student_t_degrees_of_freedom: float = 5.0
    seed: int = 7

    def __post_init__(self) -> None:
        if self.particle_count < 32:
            raise ValueError("particle_count must be at least 32.")
        for name in ("resample_ess_fraction", "tempering_ess_fraction"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError("{} must lie in (0, 1].".format(name))
        if self.tempering_ess_fraction < self.resample_ess_fraction:
            raise ValueError("tempering ESS fraction must not be below the resampling fraction.")
        if self.mcmc_steps < 0 or self.local_move_scale <= 0.0:
            raise ValueError("MCMC settings must be non-negative and finite.")
        if not 0.0 <= self.prior_move_probability <= 1.0:
            raise ValueError("prior_move_probability must lie in [0, 1].")
        if self.student_t_degrees_of_freedom <= 0.0:
            raise ValueError("Student-t degrees of freedom must be positive.")


@dataclass(frozen=True)
class ParticleFilterUpdate:
    update_index: int
    observation_count: int
    ess_before: float
    ess_after: float
    resampled: bool
    tempering_steps: int
    mcmc_accepted: int
    mcmc_proposed: int
    log_evidence_increment: float


@dataclass(frozen=True)
class PosteriorSummary:
    mean: np.ndarray
    map: np.ndarray
    standard_deviation: np.ndarray
    lower_95: np.ndarray
    upper_95: np.ndarray
    covariance: np.ndarray
    effective_sample_size: float
    log_evidence: float


class StaticParameterParticleFilter:
    """Tempered resample-move particle filter for a static 10-D parameter."""

    def __init__(self, bounds: ParameterBounds, config: ParticleFilterConfig) -> None:
        self.bounds = bounds
        self.config = config
        self.rng = np.random.default_rng(int(config.seed))
        self.particles = self._sample_uniform_physical(config.particle_count)
        self.log_weights = np.full(config.particle_count, -np.log(config.particle_count))
        self.weights = np.full(config.particle_count, 1.0 / config.particle_count)
        self._history = []
        self.update_index = 0
        self.observation_count = 0
        self.log_evidence = 0.0

    def _sample_uniform_physical(self, count: int) -> np.ndarray:
        accepted = []
        remaining = int(count)
        attempts = 0
        while remaining > 0:
            proposal_count = max(256, 4 * remaining)
            values = self.rng.uniform(self.bounds.lower, self.bounds.upper, (proposal_count, 10))
            valid = values[physical_parameter_mask(values)]
            if valid.size:
                take = min(remaining, valid.shape[0])
                accepted.append(valid[:take])
                remaining -= take
            attempts += proposal_count
            if attempts > 2000 * count:
                raise ValueError("parameter bounds contain too little physically valid inertia support.")
        return np.concatenate(accepted, axis=0)

    def _batch_log_likelihood(self, particles: np.ndarray, batch: ObservationBatch) -> np.ndarray:
        particles = np.asarray(particles, dtype=float)
        result = np.zeros(particles.shape[0], dtype=float)
        dof = float(self.config.student_t_degrees_of_freedom)
        normalizer = lgamma(0.5 * (dof + 1.0)) - lgamma(0.5 * dof) - 0.5 * np.log(dof * pi)
        for index in range(batch.count):
            predicted = predict_wrench(
                particles,
                batch.specific_acceleration[index],
                batch.angular_velocity[index],
                batch.angular_acceleration[index],
            )
            sigma = np.concatenate((batch.force_sigma[index], batch.torque_sigma[index]))
            scaled = (batch.actuator_wrench[index] - predicted) / sigma
            result += np.sum(
                normalizer - np.log(sigma) - 0.5 * (dof + 1.0) * np.log1p(scaled * scaled / dof),
                axis=-1,
            )
        return result

    def _target_log_density(
        self,
        particles: np.ndarray,
        pending_batch: ObservationBatch = None,
        pending_power: float = 0.0,
    ) -> np.ndarray:
        result = np.zeros(np.asarray(particles).shape[0], dtype=float)
        for batch in self._history:
            result += self._batch_log_likelihood(particles, batch)
        if pending_batch is not None and pending_power > 0.0:
            result += float(pending_power) * self._batch_log_likelihood(particles, pending_batch)
        return result

    def _choose_tempering_increment(self, log_likelihood: np.ndarray, remaining: float) -> float:
        target = self.config.tempering_ess_fraction * self.config.particle_count

        def candidate_ess(increment: float) -> float:
            _, candidate_weights = _normalize_log_weights(self.log_weights + increment * log_likelihood)
            return effective_sample_size(candidate_weights)

        if candidate_ess(remaining) >= target:
            return remaining
        low, high = 0.0, remaining
        for _ in range(32):
            midpoint = 0.5 * (low + high)
            if candidate_ess(midpoint) < target:
                high = midpoint
            else:
                low = midpoint
        return max(low, min(remaining, 1.0e-8))

    def _resample(self) -> None:
        ancestors = systematic_resample(self.weights, self.rng)
        self.particles = self.particles[ancestors].copy()
        self.log_weights.fill(-np.log(self.config.particle_count))
        self.weights.fill(1.0 / self.config.particle_count)

    def _mcmc_move(
        self, pending_batch: ObservationBatch, pending_power: float
    ) -> Tuple[int, int]:
        if self.config.mcmc_steps == 0:
            return 0, 0
        dimension = self.particles.shape[1]
        covariance = np.cov(self.particles, rowvar=False)
        minimum_scale = 1.0e-3 * self.bounds.width
        covariance = covariance + np.diag(minimum_scale * minimum_scale)
        multiplier = self.config.local_move_scale * 2.38 / np.sqrt(dimension)
        covariance *= multiplier * multiplier
        current_density = self._target_log_density(
            self.particles, pending_batch=pending_batch, pending_power=pending_power
        )
        accepted_total = 0
        proposed_total = 0
        for _ in range(self.config.mcmc_steps):
            proposal = self.bounds.reflect(
                self.particles
                + self.rng.multivariate_normal(np.zeros(dimension), covariance, self.config.particle_count)
            )
            global_mask = self.rng.random(self.config.particle_count) < self.config.prior_move_probability
            if np.any(global_mask):
                proposal[global_mask] = self._sample_uniform_physical(int(np.sum(global_mask)))
            valid = physical_parameter_mask(proposal)
            proposal_density = np.full(self.config.particle_count, -np.inf)
            if np.any(valid):
                proposal_density[valid] = self._target_log_density(
                    proposal[valid], pending_batch=pending_batch, pending_power=pending_power
                )
            log_acceptance = proposal_density - current_density
            accept = valid & (np.log(self.rng.random(self.config.particle_count)) < log_acceptance)
            self.particles[accept] = proposal[accept]
            current_density[accept] = proposal_density[accept]
            accepted_total += int(np.sum(accept))
            proposed_total += self.config.particle_count
        return accepted_total, proposed_total

    def update(self, batch: ObservationBatch) -> ParticleFilterUpdate:
        """Assimilate one chronological observation batch."""

        ess_before = effective_sample_size(self.weights)
        resampled = False
        accepted = 0
        proposed = 0
        tempering_steps = 0
        power = 0.0
        log_evidence_before = self.log_evidence
        while power < 1.0 - 1.0e-10:
            likelihood = self._batch_log_likelihood(self.particles, batch)
            increment = self._choose_tempering_increment(likelihood, 1.0 - power)
            unnormalized = self.log_weights + increment * likelihood
            evidence_increment = _logsumexp(unnormalized)
            self.log_evidence += evidence_increment
            self.log_weights, self.weights = _normalize_log_weights(unnormalized)
            power = min(1.0, power + increment)
            tempering_steps += 1
            must_resample = (
                power < 1.0 - 1.0e-10
                or effective_sample_size(self.weights)
                < self.config.resample_ess_fraction * self.config.particle_count
            )
            if must_resample:
                self._resample()
                resampled = True
                moved, attempted = self._mcmc_move(batch, power)
                accepted += moved
                proposed += attempted
            if tempering_steps > 128:
                raise RuntimeError("adaptive tempering did not reach the full batch likelihood.")

        self._history.append(batch)
        self.update_index += 1
        self.observation_count += batch.count
        return ParticleFilterUpdate(
            update_index=self.update_index,
            observation_count=self.observation_count,
            ess_before=ess_before,
            ess_after=effective_sample_size(self.weights),
            resampled=resampled,
            tempering_steps=tempering_steps,
            mcmc_accepted=accepted,
            mcmc_proposed=proposed,
            log_evidence_increment=self.log_evidence - log_evidence_before,
        )

    def posterior_summary(self) -> PosteriorSummary:
        """Return weighted moments, quantiles, and an in-support MAP particle."""

        mean = np.average(self.particles, axis=0, weights=self.weights)
        centered = self.particles - mean
        covariance = (centered * self.weights[:, None]).T @ centered
        covariance = 0.5 * (covariance + covariance.T)
        standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        lower = np.empty(self.particles.shape[1])
        upper = np.empty(self.particles.shape[1])
        for column in range(self.particles.shape[1]):
            order = np.argsort(self.particles[:, column])
            cumulative = np.cumsum(self.weights[order])
            lower[column] = np.interp(0.025, cumulative, self.particles[order, column])
            upper[column] = np.interp(0.975, cumulative, self.particles[order, column])
        if self._history:
            map_index = int(np.argmax(self._target_log_density(self.particles)))
        else:
            map_index = int(np.argmax(self.weights))
        return PosteriorSummary(
            mean=mean,
            map=self.particles[map_index].copy(),
            standard_deviation=standard_deviation,
            lower_95=lower,
            upper_95=upper,
            covariance=covariance,
            effective_sample_size=effective_sample_size(self.weights),
            log_evidence=float(self.log_evidence),
        )

    def excitation_metrics(
        self,
        parameters: np.ndarray,
        pending_batch: ObservationBatch = None,
    ) -> Tuple[int, float]:
        """Return finite-difference whitened Jacobian rank and condition number.

        ``pending_batch`` lets the caller apply an excitation gate before the
        evidence is admitted to filter history.
        """

        point = np.asarray(parameters, dtype=float).reshape(10)
        rows = []
        batches = list(self._history)
        if pending_batch is not None:
            batches.append(pending_batch)
        for batch in batches:
            for index in range(batch.count):
                sigma = np.concatenate((batch.force_sigma[index], batch.torque_sigma[index]))
                jacobian = np.empty((6, 10))
                for column in range(10):
                    step = max(abs(point[column]), self.bounds.width[column]) * 1.0e-5
                    plus = point.copy()
                    minus = point.copy()
                    plus[column] += step
                    minus[column] -= step
                    predicted_plus = predict_wrench(
                        plus,
                        batch.specific_acceleration[index],
                        batch.angular_velocity[index],
                        batch.angular_acceleration[index],
                    )
                    predicted_minus = predict_wrench(
                        minus,
                        batch.specific_acceleration[index],
                        batch.angular_velocity[index],
                        batch.angular_acceleration[index],
                    )
                    jacobian[:, column] = (predicted_plus - predicted_minus) / (2.0 * step)
                rows.append(jacobian / sigma[:, None])
        if not rows:
            return 0, float("inf")
        singular_values = np.linalg.svd(np.concatenate(rows, axis=0), compute_uv=False)
        tolerance = singular_values[0] * max(1.0e-8, np.finfo(float).eps * len(rows) * 6)
        rank = int(np.sum(singular_values > tolerance))
        condition = (
            float(singular_values[0] / singular_values[-1])
            if rank == 10 and singular_values[-1] > 0.0
            else float("inf")
        )
        return rank, condition


def concatenate_batches(batches: Sequence[ObservationBatch]) -> ObservationBatch:
    """Concatenate compatible batches for tests and held-out evaluation."""

    if not batches:
        raise ValueError("at least one observation batch is required.")
    return ObservationBatch(
        specific_acceleration=np.concatenate([item.specific_acceleration for item in batches]),
        angular_velocity=np.concatenate([item.angular_velocity for item in batches]),
        angular_acceleration=np.concatenate([item.angular_acceleration for item in batches]),
        actuator_wrench=np.concatenate([item.actuator_wrench for item in batches]),
        force_sigma=np.concatenate([item.force_sigma for item in batches]),
        torque_sigma=np.concatenate([item.torque_sigma for item in batches]),
    )
