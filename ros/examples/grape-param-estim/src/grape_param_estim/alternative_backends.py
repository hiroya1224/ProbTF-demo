"""Fail-closed optional backends for the grape counterfactual analysis.

The default implementations live in :mod:`state_smoother`,
:mod:`effective_response`, and :mod:`inference`.  This module contains small,
independently selectable vertical slices for the candidates that need an
explicit comparison before they can be promoted:

* a batch translation factor graph with IMU preintegration factors,
* particle-marginal Metropolis--Hastings (PMMH),
* a structured rigid-body inverse-dynamics response with an explicit
  actuator/inertia gauge, and
* a strict external exact-controller oracle boundary and conformance gate.

None of these candidates silently becomes a default.  In particular, a
Python controller surrogate cannot satisfy the exact-oracle identity contract
and conditional GP/BayesSim candidates remain ``PRUNE`` until all of their
preconditions are recorded as passing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
import ctypes
import hashlib
import json
import os
import selectors
import subprocess
import threading
import time
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from grape_param_estim.controller.contracts import (
    FIDELITY_PC_EXACT,
    FIDELITY_PC_MCU_EXACT,
    FrozenMapping,
    PC_EXACT_REQUIRED_CAPABILITIES,
    deep_freeze,
    expand_capabilities,
    normalize_fidelity,
)
from grape_param_estim.controller_replay import ReplayMetrics, replay_metrics
from grape_param_estim.episode import stable_hash
from grape_param_estim.dynamics import (
    PARAMETER_COUNT,
    predict_wrench,
    validate_physical_parameters,
)
from grape_param_estim.inference import systematic_resample
from grape_param_estim.state_smoother import (
    ERROR_STATE_SIZE,
    SmootherConfig,
    TrajectoryObservations,
    TrajectoryPosterior,
    smooth_trajectory,
)


# ---------------------------------------------------------------------------
# Optional batch factor graph


@dataclass(frozen=True)
class FactorGraphSmootherConfig:
    """Configuration for the dense, dependency-light factor-graph slice.

    The graph estimates translation and world velocity.  Orientation and IMU
    biases are initialized by the common error-state estimator, so this is a
    useful comparison slice rather than a claim of a complete production
    preintegration implementation.
    """

    bootstrap_config: SmootherConfig = field(default_factory=SmootherConfig)
    prior_position_sigma: float = 0.02
    prior_velocity_sigma: float = 0.5
    preintegration_acceleration_sigma: float = 0.35
    preintegration_velocity_floor_sigma: float = 1.0e-3
    preintegration_position_floor_sigma: float = 1.0e-4
    mocap_position_sigma: float = 0.01
    numerical_jitter: float = 1.0e-10
    max_dense_event_count: int = 512

    def __post_init__(self) -> None:
        if not isinstance(self.bootstrap_config, SmootherConfig):
            raise TypeError("bootstrap_config must be SmootherConfig")
        for name in (
            "prior_position_sigma",
            "prior_velocity_sigma",
            "preintegration_acceleration_sigma",
            "preintegration_velocity_floor_sigma",
            "preintegration_position_floor_sigma",
            "mocap_position_sigma",
            "numerical_jitter",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("{} must be finite and positive".format(name))
            object.__setattr__(self, name, value)
        maximum = int(self.max_dense_event_count)
        if maximum < 2:
            raise ValueError("max_dense_event_count must be at least two")
        object.__setattr__(self, "max_dense_event_count", maximum)


class BatchImuPreintegrationSmoother:
    """Translation/velocity factor-graph vertical slice.

    Offline mode solves a joint Gaussian graph containing a pose prior,
    mocap-position factors, and IMU-preintegrated position/velocity factors.
    The returned object is the same :class:`TrajectoryPosterior` used by the
    EKF+RTS backend, including trajectory-wide sample IDs.

    ``online_prefix=True`` deliberately disables the batch graph and delegates
    to the causal filter.  Messages after ``cutoff`` are ignored by that
    filter, and the returned posterior contains no timestamp after the cutoff.
    This makes it impossible to accidentally use a future batch factor in an
    online-prefix score.
    """

    backend_id = "batch_imu_preintegration_translation_factor_graph/v1"
    candidate_status = "OPTIONAL_CANDIDATE"

    def __init__(
        self, config: FactorGraphSmootherConfig = FactorGraphSmootherConfig()
    ) -> None:
        if not isinstance(config, FactorGraphSmootherConfig):
            raise TypeError("config must be FactorGraphSmootherConfig")
        self.config = config

    @staticmethod
    def _variable_slice(index: int) -> slice:
        return slice(6 * index, 6 * (index + 1))

    @staticmethod
    def _latest_imu_indices(
        observations: TrajectoryObservations, timestamps: np.ndarray
    ) -> np.ndarray:
        indices = np.searchsorted(
            observations.imu_times, timestamps, side="right"
        ) - 1
        if np.any(indices < 0):
            raise ValueError("factor graph has no causal IMU value at its start")
        # A masked IMU sample is never promoted into a preintegration factor.
        valid_indices = np.flatnonzero(observations.imu_valid_mask)
        if not valid_indices.size:
            raise ValueError("factor graph requires a valid IMU sample")
        output = np.empty(indices.shape, dtype=int)
        for item, candidate in enumerate(indices):
            location = np.searchsorted(valid_indices, candidate, side="right") - 1
            if location < 0:
                raise ValueError(
                    "factor graph has no valid causal IMU value at timestamp"
                )
            output[item] = valid_indices[location]
        return output

    def _build_graph(
        self,
        observations: TrajectoryObservations,
        bootstrap: TrajectoryPosterior,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = bootstrap.timestamps
        count = times.size
        if count > self.config.max_dense_event_count:
            raise ValueError(
                "dense factor-graph slice supports at most {} events; "
                "downsample the comparison interval or use a sparse backend".format(
                    self.config.max_dense_event_count
                )
            )
        dimension = 6 * count
        rows = []
        targets = []

        def add_factor(
            blocks: Sequence[Tuple[int, np.ndarray]],
            target: np.ndarray,
            sigma: float,
        ) -> None:
            value = np.asarray(target, dtype=float).reshape(3)
            row = np.zeros((3, dimension))
            for state_index, block in blocks:
                matrix = np.asarray(block, dtype=float)
                if matrix.shape != (3, 6):
                    raise ValueError("factor block must have shape (3, 6)")
                row[:, self._variable_slice(state_index)] += matrix
            rows.append(row / float(sigma))
            targets.append(value / float(sigma))

        position_selector = np.hstack((np.eye(3), np.zeros((3, 3))))
        velocity_selector = np.hstack((np.zeros((3, 3)), np.eye(3)))
        add_factor(
            ((0, position_selector),),
            bootstrap.position_world[0],
            self.config.prior_position_sigma,
        )
        add_factor(
            ((0, velocity_selector),),
            bootstrap.velocity_world[0],
            self.config.prior_velocity_sigma,
        )

        imu_indices = self._latest_imu_indices(observations, times[:-1])
        gravity = np.asarray(
            self.config.bootstrap_config.gravity_world, dtype=float
        )
        for index, delta in enumerate(np.diff(times)):
            dt = float(delta)
            imu_index = int(imu_indices[index])
            body_acceleration = (
                observations.accelerometer_body[imu_index]
                - bootstrap.accelerometer_bias_body[index]
            )
            world_acceleration = (
                Rotation.from_quat(bootstrap.quaternion_xyzw[index]).apply(
                    body_acceleration
                )
                + gravity
            )
            next_index = index + 1
            position_sigma = max(
                self.config.preintegration_position_floor_sigma,
                self.config.preintegration_acceleration_sigma
                * dt
                * dt
                / np.sqrt(3.0),
            )
            velocity_sigma = max(
                self.config.preintegration_velocity_floor_sigma,
                self.config.preintegration_acceleration_sigma * dt,
            )
            position_from_current = np.hstack((-np.eye(3), -dt * np.eye(3)))
            add_factor(
                (
                    (index, position_from_current),
                    (next_index, position_selector),
                ),
                0.5 * world_acceleration * dt * dt,
                position_sigma,
            )
            add_factor(
                (
                    (index, -velocity_selector),
                    (next_index, velocity_selector),
                ),
                world_acceleration * dt,
                velocity_sigma,
            )

        event_index = {float(stamp): index for index, stamp in enumerate(times)}
        for mocap_index, stamp in enumerate(observations.mocap_times):
            if not observations.mocap_valid_mask[mocap_index]:
                continue
            state_index = event_index.get(float(stamp))
            if state_index is None:
                continue
            # Do not resurrect an outlier rejected by the bootstrap pose gate.
            # When several measurements share a timestamp, the common
            # posterior only exposes aggregate accepted/rejected flags, so the
            # conservative choice is to omit that timestamp if either flag
            # says a measurement was rejected.
            if (
                not bootstrap.mocap_used[state_index]
                or bootstrap.mocap_rejected[state_index]
            ):
                continue
            add_factor(
                ((state_index, position_selector),),
                observations.mocap_positions_world[mocap_index],
                self.config.mocap_position_sigma,
            )

        design = np.vstack(rows)
        target = np.concatenate(targets)
        information = design.T @ design
        information += np.eye(dimension) * self.config.numerical_jitter
        try:
            estimate = np.linalg.solve(information, design.T @ target)
            covariance = np.linalg.inv(information)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("factor-graph normal equations are singular") from exc
        covariance = 0.5 * (covariance + covariance.T)
        if (
            not np.all(np.isfinite(estimate))
            or not np.all(np.isfinite(covariance))
            or np.min(np.linalg.eigvalsh(covariance)) < -1.0e-8
        ):
            raise RuntimeError("factor graph produced an invalid Gaussian")
        return estimate.reshape(count, 6), covariance, information

    def smooth(
        self,
        observations: TrajectoryObservations,
        online_prefix: bool = False,
        cutoff: Optional[float] = None,
    ) -> TrajectoryPosterior:
        if not isinstance(observations, TrajectoryObservations):
            raise TypeError("observations must be TrajectoryObservations")
        if online_prefix:
            causal = smooth_trajectory(
                observations,
                config=self.config.bootstrap_config,
                online_prefix=True,
                cutoff=cutoff,
            )
            return replace(
                causal,
                sampling_approximation=(
                    "causal_error_state_filter;batch_factor_graph_disabled"
                ),
            )

        bootstrap = smooth_trajectory(
            observations,
            config=self.config.bootstrap_config,
            online_prefix=False,
            cutoff=cutoff,
        )
        estimate, joint_covariance, _ = self._build_graph(
            observations, bootstrap
        )
        count = bootstrap.timestamps.size
        position = estimate[:, :3]
        velocity = estimate[:, 3:]
        velocity_body = Rotation.from_quat(
            bootstrap.quaternion_xyzw
        ).inv().apply(velocity)

        covariance = np.zeros(
            (count, ERROR_STATE_SIZE, ERROR_STATE_SIZE), dtype=float
        )
        for index in range(count):
            block = self._variable_slice(index)
            covariance[index, :6, :6] = joint_covariance[block, block]
            covariance[index, 6:, 6:] = bootstrap.covariance[index, 6:, 6:]

        sample_count = bootstrap.sample_count
        if sample_count:
            rng = np.random.default_rng(int(self.config.bootstrap_config.seed))
            eigenvalues, eigenvectors = np.linalg.eigh(joint_covariance)
            square_root = eigenvectors @ np.diag(
                np.sqrt(np.maximum(eigenvalues, 0.0))
            )
            standard = rng.normal(size=(sample_count, 6 * count))
            errors = (standard @ square_root.T).reshape(sample_count, count, 6)
            sample_position = position[None, :, :] + errors[:, :, :3]
            sample_velocity = velocity[None, :, :] + errors[:, :, 3:]
        else:
            sample_position = np.empty((0, count, 3))
            sample_velocity = np.empty((0, count, 3))

        return TrajectoryPosterior(
            timestamps=bootstrap.timestamps,
            position_world=position,
            velocity_world=velocity,
            velocity_body=velocity_body,
            quaternion_xyzw=bootstrap.quaternion_xyzw,
            angular_velocity_body=bootstrap.angular_velocity_body,
            accelerometer_bias_body=bootstrap.accelerometer_bias_body,
            gyro_bias_body=bootstrap.gyro_bias_body,
            covariance=covariance,
            sample_ids=bootstrap.sample_ids,
            sample_weights=bootstrap.sample_weights,
            sample_position_world=sample_position,
            sample_velocity_world=sample_velocity,
            sample_quaternion_xyzw=bootstrap.sample_quaternion_xyzw,
            sample_angular_velocity_body=bootstrap.sample_angular_velocity_body,
            sample_accelerometer_bias_body=(
                bootstrap.sample_accelerometer_bias_body
            ),
            sample_gyro_bias_body=bootstrap.sample_gyro_bias_body,
            mocap_used=bootstrap.mocap_used,
            mocap_rejected=bootstrap.mocap_rejected,
            imu_used=bootstrap.imu_used,
            is_smoothed=True,
            causal_cutoff=bootstrap.causal_cutoff,
            sampling_approximation=(
                "joint_gaussian_translation_imu_preintegration_factor_graph;"
                "orientation_bias_from_error_state_rts"
            ),
        )


# ---------------------------------------------------------------------------
# Generic PMMH vertical slice and a shared synthetic model


class ParticleStateSpaceModel(ABC):
    """Minimal bootstrap-particle likelihood contract used by PMMH."""

    parameter_dimension: int

    @abstractmethod
    def sample_initial(
        self, parameters: np.ndarray, count: int, rng: np.random.Generator
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def sample_transition(
        self,
        states: np.ndarray,
        parameters: np.ndarray,
        time_index: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def observation_log_likelihood(
        self,
        observation: np.ndarray,
        states: np.ndarray,
        parameters: np.ndarray,
        time_index: int,
    ) -> np.ndarray:
        raise NotImplementedError


class ParticleLikelihoodDegeneracy(RuntimeError):
    """Raised when no particle supports an observation."""


def _finite_particle_states(
    values: np.ndarray, count: int, name: str
) -> np.ndarray:
    states = np.asarray(values, dtype=float)
    if states.ndim != 2 or states.shape[0] != count or not np.all(
        np.isfinite(states)
    ):
        raise ValueError(
            "{} must return a finite matrix with one row per particle".format(
                name
            )
        )
    return states


def _log_mean_exp(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    maximum = float(np.max(array))
    if not np.isfinite(maximum):
        raise ParticleLikelihoodDegeneracy(
            "all particle observation weights are zero"
        )
    return maximum + float(np.log(np.mean(np.exp(array - maximum))))


def bootstrap_particle_log_likelihood(
    model: ParticleStateSpaceModel,
    observations: np.ndarray,
    parameters: np.ndarray,
    particle_count: int,
    rng: np.random.Generator,
) -> float:
    """Return an unbiased bootstrap-particle estimate of the likelihood."""

    if not isinstance(model, ParticleStateSpaceModel):
        raise TypeError("model must implement ParticleStateSpaceModel")
    count = int(particle_count)
    if count < 16:
        raise ValueError("particle_count must be at least 16")
    parameter = np.asarray(parameters, dtype=float).reshape(-1)
    if (
        parameter.shape != (int(model.parameter_dimension),)
        or not np.all(np.isfinite(parameter))
    ):
        raise ValueError("parameters have the wrong dimension or are non-finite")
    observed = np.asarray(observations, dtype=float)
    if observed.ndim == 0 or observed.shape[0] < 1 or not np.all(
        np.isfinite(observed)
    ):
        raise ValueError("observations must have a finite leading time axis")

    states = _finite_particle_states(
        model.sample_initial(parameter, count, rng),
        count,
        "sample_initial",
    )
    log_likelihood = 0.0
    for time_index in range(observed.shape[0]):
        if time_index:
            states = _finite_particle_states(
                model.sample_transition(
                    states, parameter, time_index, rng
                ),
                count,
                "sample_transition",
            )
        log_weights = np.asarray(
            model.observation_log_likelihood(
                observed[time_index],
                states,
                parameter,
                time_index,
            ),
            dtype=float,
        )
        if log_weights.shape != (count,) or np.any(np.isnan(log_weights)):
            raise ValueError(
                "observation_log_likelihood must return one non-NaN value "
                "per particle"
            )
        increment = _log_mean_exp(log_weights)
        log_likelihood += increment
        if time_index + 1 < observed.shape[0]:
            normalized = np.exp(log_weights - increment)
            normalized /= np.sum(normalized)
            ancestors = systematic_resample(normalized, rng)
            states = states[ancestors].copy()
    return float(log_likelihood)


@dataclass(frozen=True)
class ParticleMarginalMhConfig:
    iteration_count: int = 2000
    burn_in: int = 500
    thin: int = 2
    particle_count: int = 256
    proposal_scale: float = 0.15
    seed: int = 7

    def __post_init__(self) -> None:
        iterations = int(self.iteration_count)
        burn = int(self.burn_in)
        thin = int(self.thin)
        particles = int(self.particle_count)
        scale = float(self.proposal_scale)
        if iterations < 2 or burn < 0 or burn >= iterations:
            raise ValueError("PMMH requires 0 <= burn_in < iteration_count")
        if thin < 1 or particles < 16:
            raise ValueError("thin must be positive and particle_count >= 16")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("proposal_scale must be finite and positive")
        object.__setattr__(self, "iteration_count", iterations)
        object.__setattr__(self, "burn_in", burn)
        object.__setattr__(self, "thin", thin)
        object.__setattr__(self, "particle_count", particles)
        object.__setattr__(self, "proposal_scale", scale)


@dataclass(frozen=True)
class ParticleMarginalMhPosterior:
    chain: np.ndarray
    log_likelihood: np.ndarray
    accepted: np.ndarray
    burn_in: int
    thin: int
    particle_count: int
    seed: int

    def __post_init__(self) -> None:
        chain = np.asarray(self.chain, dtype=float)
        likelihood = np.asarray(self.log_likelihood, dtype=float)
        accepted = np.asarray(self.accepted, dtype=bool)
        if (
            chain.ndim != 2
            or likelihood.shape != (chain.shape[0],)
            or accepted.shape != (chain.shape[0],)
            or not np.all(np.isfinite(chain))
            or not np.all(np.isfinite(likelihood))
        ):
            raise ValueError("PMMH chain fields are invalid")
        object.__setattr__(self, "chain", chain)
        object.__setattr__(self, "log_likelihood", likelihood)
        object.__setattr__(self, "accepted", accepted)

    @property
    def samples(self) -> np.ndarray:
        return self.chain[self.burn_in :: self.thin]

    @property
    def acceptance_rate(self) -> float:
        return float(np.mean(self.accepted[1:]))

    def mean(self) -> np.ndarray:
        return np.mean(self.samples, axis=0)

    def covariance(self) -> np.ndarray:
        return np.atleast_2d(np.cov(self.samples, rowvar=False))


class ParticleMarginalMetropolisHastings:
    """Random-walk PMMH over the same prior/transform contract as modular SMC."""

    backend_id = "particle_marginal_metropolis_hastings/v1"
    candidate_status = "OPTIONAL_CANDIDATE"

    def __init__(
        self,
        model: ParticleStateSpaceModel,
        prior: Any,
        transform: Any,
        config: ParticleMarginalMhConfig = ParticleMarginalMhConfig(),
    ) -> None:
        if not isinstance(model, ParticleStateSpaceModel):
            raise TypeError("model must implement ParticleStateSpaceModel")
        if (
            int(getattr(prior, "dimension", -1)) != model.parameter_dimension
            or int(getattr(transform, "dimension", -1))
            != model.parameter_dimension
        ):
            raise ValueError(
                "model, prior, and transform dimensions must match"
            )
        if not isinstance(config, ParticleMarginalMhConfig):
            raise TypeError("config must be ParticleMarginalMhConfig")
        self.model = model
        self.prior = prior
        self.transform = transform
        self.config = config

    @staticmethod
    def _scalar_log_probability(value: Any, name: str) -> float:
        array = np.asarray(value, dtype=float)
        if array.size != 1 or np.isnan(array.reshape(-1)[0]):
            raise ValueError("{} must return a scalar non-NaN value".format(name))
        return float(array.reshape(-1)[0])

    def run(
        self,
        observations: np.ndarray,
        initial_parameters: Optional[np.ndarray] = None,
    ) -> ParticleMarginalMhPosterior:
        rng = np.random.default_rng(int(self.config.seed))
        if initial_parameters is None:
            current = np.asarray(self.prior.sample(1, rng), dtype=float)[0]
        else:
            current = np.asarray(initial_parameters, dtype=float).reshape(-1)
        if current.shape != (self.model.parameter_dimension,):
            raise ValueError("initial_parameters has the wrong dimension")
        current_unconstrained = np.asarray(
            self.transform.to_unconstrained(current), dtype=float
        )

        def evaluate(parameter: np.ndarray, unconstrained: np.ndarray):
            log_prior = self._scalar_log_probability(
                self.prior.log_prob(parameter), "prior.log_prob"
            )
            jacobian = self._scalar_log_probability(
                self.transform.log_abs_det_jacobian(unconstrained),
                "transform.log_abs_det_jacobian",
            )
            if not np.isfinite(log_prior) or not np.isfinite(jacobian):
                return -np.inf, -np.inf
            likelihood = bootstrap_particle_log_likelihood(
                self.model,
                observations,
                parameter,
                self.config.particle_count,
                rng,
            )
            return likelihood, log_prior + jacobian + likelihood

        current_likelihood, current_target = evaluate(
            current, current_unconstrained
        )
        if not np.isfinite(current_target):
            raise ValueError("initial PMMH state has zero posterior density")

        chain = np.empty(
            (self.config.iteration_count, self.model.parameter_dimension)
        )
        likelihoods = np.empty(self.config.iteration_count)
        accepted = np.zeros(self.config.iteration_count, dtype=bool)
        chain[0] = current
        likelihoods[0] = current_likelihood
        for iteration in range(1, self.config.iteration_count):
            proposal_unconstrained = (
                current_unconstrained
                + rng.normal(
                    0.0,
                    self.config.proposal_scale,
                    size=self.model.parameter_dimension,
                )
            )
            proposal = np.asarray(
                self.transform.from_unconstrained(proposal_unconstrained),
                dtype=float,
            )
            try:
                proposal_likelihood, proposal_target = evaluate(
                    proposal, proposal_unconstrained
                )
            except ParticleLikelihoodDegeneracy:
                proposal_likelihood, proposal_target = -np.inf, -np.inf
            if (
                np.log(max(float(rng.random()), np.finfo(float).tiny))
                < proposal_target - current_target
            ):
                current = proposal
                current_unconstrained = proposal_unconstrained
                current_likelihood = proposal_likelihood
                current_target = proposal_target
                accepted[iteration] = True
            chain[iteration] = current
            likelihoods[iteration] = current_likelihood
        return ParticleMarginalMhPosterior(
            chain=chain,
            log_likelihood=likelihoods,
            accepted=accepted,
            burn_in=self.config.burn_in,
            thin=self.config.thin,
            particle_count=self.config.particle_count,
            seed=int(self.config.seed),
        )


@dataclass(frozen=True)
class LinearGaussianRandomWalkModel(ParticleStateSpaceModel):
    """One-dimensional synthetic model shared by PMMH and modular SMC tests."""

    initial_mean: float = 0.0
    initial_sigma: float = 0.5
    process_sigma: float = 0.2
    observation_sigma: float = 0.3
    parameter_dimension: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        for name in (
            "initial_mean",
            "initial_sigma",
            "process_sigma",
            "observation_sigma",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError("{} must be finite".format(name))
            if name != "initial_mean" and value <= 0.0:
                raise ValueError("{} must be positive".format(name))
            object.__setattr__(self, name, value)

    def sample_initial(
        self, parameters: np.ndarray, count: int, rng: np.random.Generator
    ) -> np.ndarray:
        del parameters
        return rng.normal(
            self.initial_mean, self.initial_sigma, size=(int(count), 1)
        )

    def sample_transition(
        self,
        states: np.ndarray,
        parameters: np.ndarray,
        time_index: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del time_index
        return (
            np.asarray(states, dtype=float)
            + float(parameters[0])
            + rng.normal(0.0, self.process_sigma, size=states.shape)
        )

    def observation_log_likelihood(
        self,
        observation: np.ndarray,
        states: np.ndarray,
        parameters: np.ndarray,
        time_index: int,
    ) -> np.ndarray:
        del parameters, time_index
        value = float(np.asarray(observation, dtype=float).reshape(-1)[0])
        residual = (value - states[:, 0]) / self.observation_sigma
        return (
            -0.5 * residual * residual
            - np.log(self.observation_sigma)
            - 0.5 * np.log(2.0 * np.pi)
        )

    def simulate(
        self, drift: float, count: int, seed: int = 7
    ) -> Tuple[np.ndarray, np.ndarray]:
        length = int(count)
        if length < 1 or not np.isfinite(float(drift)):
            raise ValueError("simulation count/drift is invalid")
        rng = np.random.default_rng(int(seed))
        states = np.empty(length)
        observations = np.empty(length)
        states[0] = rng.normal(self.initial_mean, self.initial_sigma)
        observations[0] = rng.normal(states[0], self.observation_sigma)
        for index in range(1, length):
            states[index] = rng.normal(
                states[index - 1] + float(drift), self.process_sigma
            )
            observations[index] = rng.normal(
                states[index], self.observation_sigma
            )
        return states, observations

    def exact_log_likelihood(
        self, parameter_particles: np.ndarray, observations: np.ndarray
    ) -> np.ndarray:
        """Vectorized Kalman likelihood usable by TemperedResampleMoveSmc."""

        parameters = np.asarray(parameter_particles, dtype=float)
        if (
            parameters.ndim != 2
            or parameters.shape[1] != 1
            or not np.all(np.isfinite(parameters))
        ):
            raise ValueError("parameter_particles must have shape (N, 1)")
        observed = np.asarray(observations, dtype=float).reshape(-1)
        if not observed.size or not np.all(np.isfinite(observed)):
            raise ValueError("observations must be finite and non-empty")
        mean = np.full(parameters.shape[0], self.initial_mean)
        variance = np.full(
            parameters.shape[0], self.initial_sigma * self.initial_sigma
        )
        output = np.zeros(parameters.shape[0])
        observation_variance = self.observation_sigma**2
        for index, observation in enumerate(observed):
            innovation_variance = variance + observation_variance
            residual = observation - mean
            output += (
                -0.5
                * (
                    residual * residual / innovation_variance
                    + np.log(2.0 * np.pi * innovation_variance)
                )
            )
            gain = variance / innovation_variance
            mean = mean + gain * residual
            variance = (1.0 - gain) * variance
            if index + 1 < observed.size:
                mean = mean + parameters[:, 0]
                variance = variance + self.process_sigma**2
        return output


@dataclass(frozen=True)
class BayesianBackendComparison:
    pmmh_mean: np.ndarray
    modular_smc_mean: np.ndarray
    maximum_absolute_difference: float
    tolerance: float
    passed: bool


def compare_pmmh_with_modular_smc(
    pmmh: ParticleMarginalMhPosterior,
    modular_smc: Any,
    tolerance: float,
) -> BayesianBackendComparison:
    """Compare posterior means without claiming either backend is selected."""

    if not isinstance(pmmh, ParticleMarginalMhPosterior):
        raise TypeError("pmmh must be ParticleMarginalMhPosterior")
    smc_particles = np.asarray(modular_smc.particles, dtype=float)
    smc_weights = np.asarray(modular_smc.weights, dtype=float)
    if (
        smc_particles.ndim != 2
        or smc_weights.shape != (smc_particles.shape[0],)
        or np.any(smc_weights < 0.0)
        or not np.isclose(np.sum(smc_weights), 1.0)
    ):
        raise ValueError("modular_smc has invalid particles or weights")
    pmmh_mean = pmmh.mean()
    smc_mean = np.average(smc_particles, axis=0, weights=smc_weights)
    if pmmh_mean.shape != smc_mean.shape:
        raise ValueError("PMMH and SMC posterior dimensions differ")
    threshold = float(tolerance)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    difference = float(np.max(np.abs(pmmh_mean - smc_mean)))
    return BayesianBackendComparison(
        pmmh_mean=pmmh_mean,
        modular_smc_mean=smc_mean,
        maximum_absolute_difference=difference,
        tolerance=threshold,
        passed=bool(difference <= threshold),
    )


# ---------------------------------------------------------------------------
# Structured 6-DoF inverse-dynamics response and its gauge


def _positive_six_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(6, float(array))
    if array.shape != (6,) or not np.all(np.isfinite(array)) or np.any(
        array <= 0.0
    ):
        raise ValueError("{} must be a finite positive six-vector".format(name))
    output = np.array(array, copy=True)
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class StructuredMechanicsParameters:
    inertial_parameters: np.ndarray
    actuator_wrench_scale: Optional[np.ndarray] = None
    calibrated_wrench: bool = True

    def __post_init__(self) -> None:
        inertial = np.asarray(
            validate_physical_parameters(self.inertial_parameters), dtype=float
        )
        if inertial.shape != (PARAMETER_COUNT,):
            raise ValueError("inertial_parameters must be a single 10-vector")
        inertial = np.array(inertial, copy=True)
        inertial.setflags(write=False)
        object.__setattr__(self, "inertial_parameters", inertial)
        calibrated = bool(self.calibrated_wrench)
        if calibrated:
            if self.actuator_wrench_scale is not None:
                raise ValueError(
                    "calibrated-wrench mode must not fit actuator_wrench_scale"
                )
        else:
            if self.actuator_wrench_scale is None:
                raise ValueError(
                    "uncalibrated command mode requires actuator_wrench_scale"
                )
            object.__setattr__(
                self,
                "actuator_wrench_scale",
                _positive_six_vector(
                    self.actuator_wrench_scale, "actuator_wrench_scale"
                ),
            )
        object.__setattr__(self, "calibrated_wrench", calibrated)


@dataclass(frozen=True)
class MechanicsGaugeReport:
    calibrated_wrench: bool
    gauge_dimension: int
    calibration_status: str
    gauge_action: str
    reportable_quantities: Tuple[str, ...]
    forbidden_claims: Tuple[str, ...]
    local_null_direction: np.ndarray


@dataclass(frozen=True)
class MechanicsIdentifiabilityReport:
    parameter_count: int
    jacobian_rank: int
    structural_gauge_dimension: int
    excitation_nullity: int
    singular_values: np.ndarray
    null_directions: np.ndarray
    condition_number_on_reportable_subspace: float
    identifiable_up_to_declared_gauge: bool


class StructuredSixDofMechanicsResponse:
    """Inverse-dynamics adapter around :func:`dynamics.predict_wrench`."""

    model_id = "structured_six_dof_inverse_dynamics/v1"
    candidate_status = "OPTIONAL_CANDIDATE"

    def predict_observation(
        self,
        parameters: StructuredMechanicsParameters,
        specific_acceleration: np.ndarray,
        angular_velocity: np.ndarray,
        angular_acceleration: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(parameters, StructuredMechanicsParameters):
            raise TypeError("parameters must be StructuredMechanicsParameters")
        required_wrench = predict_wrench(
            parameters.inertial_parameters,
            specific_acceleration,
            angular_velocity,
            angular_acceleration,
        )
        if parameters.calibrated_wrench:
            return required_wrench
        return required_wrench / parameters.actuator_wrench_scale

    def gaussian_log_likelihood(
        self,
        parameters: StructuredMechanicsParameters,
        specific_acceleration: np.ndarray,
        angular_velocity: np.ndarray,
        angular_acceleration: np.ndarray,
        observed_wrench_or_command: np.ndarray,
        sigma: np.ndarray,
    ) -> float:
        predicted = np.asarray(
            self.predict_observation(
                parameters,
                specific_acceleration,
                angular_velocity,
                angular_acceleration,
            ),
            dtype=float,
        )
        observed = np.asarray(observed_wrench_or_command, dtype=float)
        if observed.shape != predicted.shape or not np.all(np.isfinite(observed)):
            raise ValueError("observed input must match the predicted shape")
        scale = np.asarray(sigma, dtype=float)
        try:
            scale = np.broadcast_to(scale, predicted.shape)
        except ValueError as exc:
            raise ValueError("sigma is not broadcast-compatible") from exc
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("sigma must be finite and positive")
        residual = (observed - predicted) / scale
        return float(
            np.sum(
                -0.5 * residual * residual
                - np.log(scale)
                - 0.5 * np.log(2.0 * np.pi)
            )
        )

    def gauge_report(
        self, parameters: StructuredMechanicsParameters
    ) -> MechanicsGaugeReport:
        if not isinstance(parameters, StructuredMechanicsParameters):
            raise TypeError("parameters must be StructuredMechanicsParameters")
        if parameters.calibrated_wrench:
            return MechanicsGaugeReport(
                calibrated_wrench=True,
                gauge_dimension=0,
                calibration_status="CALIBRATED_WRENCH",
                gauge_action="none after fixing frame, units, and wrench calibration",
                reportable_quantities=(
                    "mass/inertia joint posterior conditional on excitation",
                    "center of mass",
                ),
                forbidden_claims=(
                    "well-identified individual values without a rank check",
                ),
                local_null_direction=np.zeros(PARAMETER_COUNT),
            )
        inertial = parameters.inertial_parameters
        null = np.zeros(PARAMETER_COUNT + 6)
        null[0] = inertial[0]
        null[4:10] = inertial[4:10]
        null[PARAMETER_COUNT:] = parameters.actuator_wrench_scale
        null.setflags(write=False)
        return MechanicsGaugeReport(
            calibrated_wrench=False,
            gauge_dimension=1,
            calibration_status="UNIDENTIFIED_GLOBAL_ACTUATOR_INERTIA_SCALE",
            gauge_action=(
                "(mass, inertia, actuator_wrench_scale) -> "
                "(lambda*mass, lambda*inertia, lambda*scale); center is fixed"
            ),
            reportable_quantities=(
                "actuator force scale / mass",
                "joint inertia-to-torque-scale region",
                "center of mass conditional on excitation",
            ),
            forbidden_claims=(
                "absolute mass",
                "absolute inertia",
                "absolute thrust or torque scale",
            ),
            local_null_direction=null,
        )

    def apply_global_gauge(
        self, parameters: StructuredMechanicsParameters, factor: float
    ) -> StructuredMechanicsParameters:
        """Move along the exact uncalibrated global scale gauge."""

        if parameters.calibrated_wrench:
            raise ValueError("calibrated-wrench parameters have no scale gauge")
        scale = float(factor)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("gauge factor must be finite and positive")
        inertial = np.array(parameters.inertial_parameters, copy=True)
        inertial[0] *= scale
        inertial[4:10] *= scale
        return StructuredMechanicsParameters(
            inertial_parameters=inertial,
            actuator_wrench_scale=parameters.actuator_wrench_scale * scale,
            calibrated_wrench=False,
        )

    def local_identifiability(
        self,
        parameters: StructuredMechanicsParameters,
        specific_acceleration: np.ndarray,
        angular_velocity: np.ndarray,
        angular_acceleration: np.ndarray,
        relative_step: float = 1.0e-5,
        rank_relative_tolerance: float = 1.0e-7,
    ) -> MechanicsIdentifiabilityReport:
        """Numerically rank the local response map after declaring its gauge.

        This is an excitation/rank diagnostic, not proof that the physical
        parameters are globally identifiable.  In uncalibrated-command mode
        the free vector is ``[10 inertial values, 6 actuator scales]`` and one
        null direction is structurally expected.
        """

        if not isinstance(parameters, StructuredMechanicsParameters):
            raise TypeError("parameters must be StructuredMechanicsParameters")
        step_scale = float(relative_step)
        tolerance = float(rank_relative_tolerance)
        if (
            not np.isfinite(step_scale)
            or step_scale <= 0.0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("finite positive finite-difference settings required")
        specific = np.asarray(specific_acceleration, dtype=float)
        omega = np.asarray(angular_velocity, dtype=float)
        alpha = np.asarray(angular_acceleration, dtype=float)
        if (
            specific.ndim != 2
            or specific.shape[1] != 3
            or omega.shape != specific.shape
            or alpha.shape != specific.shape
            or not np.all(np.isfinite(specific))
            or not np.all(np.isfinite(omega))
            or not np.all(np.isfinite(alpha))
        ):
            raise ValueError("identifiability inputs must be aligned finite Nx3 arrays")

        if parameters.calibrated_wrench:
            vector = np.array(parameters.inertial_parameters, copy=True)
        else:
            vector = np.concatenate(
                (
                    parameters.inertial_parameters,
                    parameters.actuator_wrench_scale,
                )
            )

        def from_vector(values: np.ndarray) -> StructuredMechanicsParameters:
            if parameters.calibrated_wrench:
                return StructuredMechanicsParameters(values[:PARAMETER_COUNT])
            return StructuredMechanicsParameters(
                values[:PARAMETER_COUNT],
                actuator_wrench_scale=values[PARAMETER_COUNT:],
                calibrated_wrench=False,
            )

        columns = []
        for index, value in enumerate(vector):
            step = step_scale * max(1.0, abs(float(value)))
            derivative = None
            for _ in range(8):
                lower = vector.copy()
                upper = vector.copy()
                lower[index] -= step
                upper[index] += step
                try:
                    lower_prediction = self.predict_observation(
                        from_vector(lower), specific, omega, alpha
                    )
                    upper_prediction = self.predict_observation(
                        from_vector(upper), specific, omega, alpha
                    )
                except ValueError:
                    step *= 0.1
                    continue
                derivative = (
                    np.asarray(upper_prediction) - np.asarray(lower_prediction)
                ) / (2.0 * step)
                break
            if derivative is None:
                raise RuntimeError(
                    "could not form a physical local mechanics perturbation"
                )
            columns.append(derivative.reshape(-1))
        jacobian = np.column_stack(columns)
        _, singular_values, right_vectors = np.linalg.svd(
            jacobian, full_matrices=True
        )
        largest = float(singular_values[0]) if singular_values.size else 0.0
        threshold = tolerance * max(largest, 1.0)
        rank = int(np.count_nonzero(singular_values > threshold))
        structural_gauge = 0 if parameters.calibrated_wrench else 1
        reportable_dimension = vector.size - structural_gauge
        excitation_nullity = max(0, reportable_dimension - rank)
        condition = (
            float(singular_values[0] / singular_values[reportable_dimension - 1])
            if (
                reportable_dimension > 0
                and singular_values.size >= reportable_dimension
                and singular_values[reportable_dimension - 1] > threshold
            )
            else float("inf")
        )
        null_directions = right_vectors[rank:]
        return MechanicsIdentifiabilityReport(
            parameter_count=int(vector.size),
            jacobian_rank=rank,
            structural_gauge_dimension=structural_gauge,
            excitation_nullity=excitation_nullity,
            singular_values=singular_values,
            null_directions=null_directions,
            condition_number_on_reportable_subspace=condition,
            identifiable_up_to_declared_gauge=bool(
                rank >= reportable_dimension
            ),
        )


# ---------------------------------------------------------------------------
# Exact external controller oracle boundary


EXACT_ORACLE_PROTOCOL = "grape.exact-controller-oracle/v1"
# Legacy names remain the PC+MCU defaults so existing callers and frozen
# reports retain their meaning.
REQUIRED_ORACLE_CAPABILITIES = (
    "pc_mcu_closed_loop_replay",
    "command_timestamp",
    "pid_terms",
    "four_axis_command",
    "vectoring_force",
    "allocation_internal",
    "torque_allocation_matrix_inverse",
    "pwm",
    "mode_and_saturation_events",
)
PC_EXACT_ORACLE_CAPABILITIES = PC_EXACT_REQUIRED_CAPABILITIES
PC_MCU_EXACT_ORACLE_CAPABILITIES = REQUIRED_ORACLE_CAPABILITIES
REQUIRED_CONFORMANCE_CHANNELS = (
    "command_timestamp",
    "pid_terms",
    "four_axis_command",
    "vectoring_force",
    "allocation_internal",
    "torque_allocation_matrix_inverse",
    "pwm",
)
PC_EXACT_CONFORMANCE_CHANNELS = (
    "command_timestamp",
    "pid_terms",
    "four_axis_command",
    "vectoring_force",
    "gimbal_command",
    "allocation_internal",
    "torque_allocation_matrix_inverse",
)
PC_MCU_EXACT_CONFORMANCE_CHANNELS = REQUIRED_CONFORMANCE_CHANNELS
FROZEN_REPLAY_RMSE_THRESHOLD = 0.01
FROZEN_REPLAY_MAXIMUM_ERROR_THRESHOLD = 0.03
FROZEN_REPLAY_EVENT_AGREEMENT_THRESHOLD = 1.0
COMMAND_TIMESTAMP_TOLERANCE_S = 0.0


def required_oracle_capabilities(fidelity: str) -> Tuple[str, ...]:
    normalized = normalize_fidelity(fidelity)
    if normalized == FIDELITY_PC_EXACT:
        return PC_EXACT_REQUIRED_CAPABILITIES
    if normalized == FIDELITY_PC_MCU_EXACT:
        return PC_MCU_EXACT_ORACLE_CAPABILITIES
    raise ValueError(
        "exact controller oracle fidelity must be pc_exact or pc_mcu_exact"
    )


def required_conformance_channels(fidelity: str) -> Tuple[str, ...]:
    normalized = normalize_fidelity(fidelity)
    if normalized == FIDELITY_PC_EXACT:
        return PC_EXACT_CONFORMANCE_CHANNELS
    if normalized == FIDELITY_PC_MCU_EXACT:
        return PC_MCU_EXACT_CONFORMANCE_CHANNELS
    raise ValueError(
        "controller conformance fidelity must be pc_exact or pc_mcu_exact"
    )


class ExactOracleError(RuntimeError):
    pass


class ExactOracleUnavailable(ExactOracleError):
    pass


class ExactOracleProtocolError(ExactOracleError):
    pass


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_CONTROLLER_CORE_SHARED_OBJECTS = (
    "libgimbalrotor_allocation_core.so",
    "libpose_linear_controller_core.so",
)


def _sanitized_loader_environment(
    source: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Return an environment that cannot redirect or inject dynamic code."""

    values = os.environ if source is None else source
    return {
        str(key): str(value)
        for key, value in values.items()
        if not str(key).startswith(("LD_", "DYLD_"))
    }


def _controller_core_dynamic_dependencies(executable: str) -> Tuple[str, ...]:
    """Read the ELF dependency table without executing the candidate."""

    readelf = "/usr/bin/readelf"
    if not os.path.isfile(readelf) or not os.access(readelf, os.X_OK):
        raise ExactOracleUnavailable(
            "readelf is required to verify exact-controller static linkage"
        )
    try:
        completed = subprocess.run(
            (readelf, "-d", executable),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10.0,
            check=False,
            env=_sanitized_loader_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExactOracleUnavailable(
            "exact-controller ELF dependencies could not be inspected"
        ) from exc
    if completed.returncode != 0 or completed.stderr:
        raise ExactOracleProtocolError(
            "exact controller oracle must be an inspectable ELF executable"
        )
    return tuple(
        name
        for name in _CONTROLLER_CORE_SHARED_OBJECTS
        if "[{}]".format(name) in completed.stdout
    )


def _require_static_controller_cores(executable: str) -> None:
    dynamic = _controller_core_dynamic_dependencies(executable)
    if dynamic:
        raise ExactOracleProtocolError(
            "exact controller core must be statically linked; dynamic "
            "dependencies found: {}".format(", ".join(dynamic))
        )


def _mapped_controller_core_dsos(process_id: int) -> Tuple[str, ...]:
    """Return controller-core DSOs actually mapped into one live process."""

    maps_path = "/proc/{}/maps".format(int(process_id))
    try:
        with open(maps_path, "r", encoding="utf-8") as stream:
            lines = tuple(stream)
    except (OSError, UnicodeDecodeError) as exc:
        raise ExactOracleUnavailable(
            "live exact-controller process mappings are unavailable"
        ) from exc
    mapped = set()
    for line in lines:
        fields = line.rstrip().split(None, 5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = fields[5]
        if path.endswith(" (deleted)"):
            path = path[: -len(" (deleted)")]
        if os.path.basename(path) in _CONTROLLER_CORE_SHARED_OBJECTS:
            mapped.add(os.path.realpath(path))
    return tuple(sorted(mapped))


def _verify_live_static_artifact(
    process_id: int,
    expected_executable_sha256: str,
) -> str:
    """Verify the executed inode and reject controller-core DSO substitution."""

    live_executable = "/proc/{}/exe".format(int(process_id))
    try:
        measured = _sha256_file(live_executable)
    except OSError as exc:
        raise ExactOracleUnavailable(
            "live exact-controller executable is unavailable"
        ) from exc
    if measured != expected_executable_sha256:
        raise ExactOracleProtocolError(
            "live exact-controller executable hash mismatch"
        )
    mapped = _mapped_controller_core_dsos(process_id)
    if mapped:
        raise ExactOracleProtocolError(
            "exact controller core DSO substitution detected: {}".format(
                ", ".join(mapped)
            )
        )
    return measured


@dataclass(frozen=True)
class ExactOracleIdentity:
    protocol: str
    backend_id: str
    implementation_language: str
    source_commit: str
    artifact_sha256: str
    capabilities: Tuple[str, ...]
    fidelity: str = FIDELITY_PC_MCU_EXACT

    def __post_init__(self) -> None:
        capabilities = tuple(str(item) for item in self.capabilities)
        fidelity = normalize_fidelity(self.fidelity)
        required = required_oracle_capabilities(fidelity)
        expanded = set(expand_capabilities(capabilities))
        missing = sorted(set(required) - expanded)
        digest = str(self.artifact_sha256).lower()
        if self.protocol != EXACT_ORACLE_PROTOCOL:
            raise ValueError("exact oracle protocol mismatch")
        if (
            not self.backend_id
            or "surrogate" in self.backend_id.lower()
            or "python" in self.backend_id.lower()
        ):
            raise ValueError("exact oracle backend_id cannot identify a surrogate")
        if self.implementation_language.strip().lower() not in (
            "c++",
            "cpp",
        ):
            raise ValueError("exact oracle must identify a C++ implementation")
        if (
            not self.source_commit.strip()
            or self.source_commit.strip().lower() == "unknown"
        ):
            raise ValueError(
                "exact oracle source_commit must identify a known revision"
            )
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("artifact_sha256 must be a lowercase SHA-256")
        if missing:
            raise ValueError(
                "exact oracle is missing capabilities: {}".format(
                    ", ".join(missing)
                )
            )
        object.__setattr__(self, "artifact_sha256", digest)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "fidelity", fidelity)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ExactOracleIdentity":
        try:
            return cls(
                protocol=str(values["protocol"]),
                backend_id=str(values["backend_id"]),
                implementation_language=str(
                    values["implementation_language"]
                ),
                source_commit=str(values["source_commit"]),
                artifact_sha256=str(values["artifact_sha256"]),
                capabilities=tuple(values["capabilities"]),
                fidelity=str(
                    values.get("fidelity", FIDELITY_PC_MCU_EXACT)
                ),
            )
        except (KeyError, TypeError) as exc:
            raise ExactOracleProtocolError(
                "oracle identity is missing required fields"
            ) from exc


@dataclass(frozen=True)
class ExactOracleReplayOutput:
    identity: ExactOracleIdentity
    continuous: Mapping[str, np.ndarray]
    events: np.ndarray
    final_states: Tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ExactOracleIdentity):
            raise TypeError("identity must be ExactOracleIdentity")
        continuous: Dict[str, np.ndarray] = {}
        row_count = None
        for name, values in self.continuous.items():
            array = np.asarray(values, dtype=float)
            if array.ndim != 2 or not np.all(np.isfinite(array)):
                raise ValueError(
                    "oracle continuous channel {} must be a finite matrix".format(
                        name
                    )
                )
            if row_count is None:
                row_count = array.shape[0]
            elif array.shape[0] != row_count:
                raise ValueError("oracle continuous channels are not aligned")
            copy = np.array(array, copy=True)
            copy.setflags(write=False)
            continuous[str(name)] = copy
        events = np.asarray(self.events)
        if events.ndim < 1:
            raise ValueError("oracle events must have a time dimension")
        if row_count is not None and events.shape[0] != row_count:
            raise ValueError("oracle events are not aligned with channels")
        event_copy = np.array(events, copy=True)
        event_copy.setflags(write=False)
        final_states = []
        for index, value in enumerate(self.final_states):
            frozen = deep_freeze(value)
            if not isinstance(frozen, FrozenMapping):
                raise TypeError(
                    "oracle final state {} must be a mapping".format(index)
                )
            final_states.append(frozen)
        object.__setattr__(
            self, "continuous", MappingProxyType(continuous)
        )
        object.__setattr__(self, "events", event_copy)
        object.__setattr__(self, "final_states", tuple(final_states))


def _decode_oracle_reply(
    raw: Any, expected_identity: ExactOracleIdentity
) -> Mapping[str, Any]:
    try:
        text = (
            bytes(raw).decode("utf-8")
            if isinstance(raw, (bytes, bytearray))
            else str(raw)
        )
        parsed = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExactOracleProtocolError(
            "oracle returned invalid UTF-8 JSON"
        ) from exc
    if not isinstance(parsed, dict) or parsed.get("ok") is not True:
        raise ExactOracleProtocolError("oracle reply is not a successful object")
    identity = ExactOracleIdentity.from_mapping(parsed.get("identity", {}))
    if identity != expected_identity:
        raise ExactOracleProtocolError(
            "runtime oracle identity differs from the expected build identity"
        )
    return parsed


def _output_from_reply(
    parsed: Mapping[str, Any], identity: ExactOracleIdentity
) -> ExactOracleReplayOutput:
    if not isinstance(parsed.get("continuous"), dict) or "events" not in parsed:
        raise ExactOracleProtocolError(
            "oracle replay reply lacks continuous channels or events"
        )
    try:
        return ExactOracleReplayOutput(
            identity=identity,
            continuous={
                str(name): np.asarray(values, dtype=float)
                for name, values in parsed["continuous"].items()
            },
            events=np.asarray(parsed["events"]),
            final_states=tuple(parsed.get("final_states", ())),
        )
    except (TypeError, ValueError) as exc:
        raise ExactOracleProtocolError(
            "oracle replay payload has invalid numeric fields"
        ) from exc


class SubprocessExactControllerOracle:
    """JSON/stdin adapter for a separately built C++ replay executable."""

    is_exact = True

    def __init__(
        self,
        command: Sequence[str],
        expected_identity: ExactOracleIdentity,
        timeout_s: float = 30.0,
    ) -> None:
        command_tuple = tuple(str(item) for item in command)
        if not command_tuple:
            raise ValueError("oracle command must not be empty")
        executable = os.path.abspath(command_tuple[0])
        if (
            not os.path.isfile(executable)
            or not os.access(executable, os.X_OK)
        ):
            raise ExactOracleUnavailable(
                "exact controller oracle executable is unavailable"
            )
        if _sha256_file(executable) != expected_identity.artifact_sha256:
            raise ExactOracleProtocolError(
                "exact controller oracle artifact hash mismatch"
            )
        _require_static_controller_cores(executable)
        timeout = float(timeout_s)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        self.command = (executable,) + command_tuple[1:]
        self.identity = expected_identity
        self.timeout_s = timeout
        handshake = self._invoke("handshake", {})
        _decode_oracle_reply(handshake, self.identity)

    def _invoke(self, operation: str, payload: Mapping[str, Any]) -> str:
        request = json.dumps(
            {
                "protocol": EXACT_ORACLE_PROTOCOL,
                "operation": str(operation),
                "payload": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        try:
            completed = subprocess.run(
                self.command,
                input=request + "\n",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_s,
                check=False,
                env=_sanitized_loader_environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ExactOracleUnavailable(
                "exact controller oracle could not be executed"
            ) from exc
        if completed.returncode != 0:
            raise ExactOracleUnavailable(
                "exact controller oracle exited with status {}".format(
                    completed.returncode
                )
            )
        if completed.stderr:
            raise ExactOracleProtocolError(
                "exact oracle wrote unexpected stderr output"
            )
        lines = completed.stdout.strip().splitlines()
        if len(lines) != 1:
            raise ExactOracleProtocolError(
                "exact oracle must return exactly one JSON object"
            )
        return lines[0]

    def replay(self, payload: Mapping[str, Any]) -> ExactOracleReplayOutput:
        parsed = _decode_oracle_reply(
            self._invoke("replay", payload), self.identity
        )
        return _output_from_reply(parsed, self.identity)


class PersistentSubprocessExactControllerOracle:
    """Shared JSON-lines transport for stateful closed-loop replay.

    Unlike :class:`SubprocessExactControllerOracle`, this transport starts the
    C++ executable once and serializes handshake/replay requests through one
    lock.  The executable must implement ``--server`` and flush exactly one
    JSON reply line for each JSON request line.
    """

    is_exact = True
    transport_is_persistent = True

    def __init__(
        self,
        command: Sequence[str],
        expected_identity: ExactOracleIdentity,
        timeout_s: float = 30.0,
    ) -> None:
        command_tuple = tuple(str(item) for item in command)
        if not command_tuple:
            raise ValueError("oracle command must not be empty")
        executable = os.path.abspath(command_tuple[0])
        if (
            not os.path.isfile(executable)
            or not os.access(executable, os.X_OK)
        ):
            raise ExactOracleUnavailable(
                "exact controller oracle executable is unavailable"
            )
        if _sha256_file(executable) != expected_identity.artifact_sha256:
            raise ExactOracleProtocolError(
                "exact controller oracle artifact hash mismatch"
            )
        _require_static_controller_cores(executable)
        timeout = float(timeout_s)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        arguments = (executable,) + command_tuple[1:]
        if "--server" not in arguments[1:]:
            arguments = arguments + ("--server",)
        self.command = arguments
        self.identity = expected_identity
        self.timeout_s = timeout
        self._lock = threading.RLock()
        self._closed = False
        try:
            self._process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=_sanitized_loader_environment(),
            )
        except OSError as exc:
            raise ExactOracleUnavailable(
                "persistent exact controller oracle could not be started"
            ) from exc
        try:
            self.runtime_executable_sha256 = _verify_live_static_artifact(
                self._process.pid,
                self.identity.artifact_sha256,
            )
            handshake = self._invoke("handshake", {})
            _decode_oracle_reply(handshake, self.identity)
        except Exception:
            self.close()
            raise

    def _terminate_locked(self) -> None:
        process = self._process
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=min(self.timeout_s, 2.0))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=min(self.timeout_s, 2.0))

    def _invoke(
        self, operation: str, payload: Mapping[str, Any]
    ) -> str:
        request = json.dumps(
            {
                "protocol": EXACT_ORACLE_PROTOCOL,
                "operation": str(operation),
                "payload": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock:
            if self._closed:
                raise ExactOracleUnavailable(
                    "persistent exact controller oracle is closed"
                )
            process = self._process
            if process.poll() is not None:
                raise ExactOracleUnavailable(
                    "persistent exact controller oracle exited with status "
                    "{}".format(process.returncode)
                )
            try:
                process.stdin.write(request + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._terminate_locked()
                raise ExactOracleUnavailable(
                    "persistent exact controller oracle request failed"
                ) from exc
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            deadline = time.monotonic() + self.timeout_s
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        self._terminate_locked()
                        raise ExactOracleUnavailable(
                            "persistent exact controller oracle timed out"
                        )
                    ready = selector.select(remaining)
                    if not ready:
                        continue
                    for key, _ in ready:
                        line = key.fileobj.readline()
                        if key.data == "stderr" and line:
                            self._terminate_locked()
                            raise ExactOracleProtocolError(
                                "exact oracle wrote unexpected stderr output"
                            )
                        if key.data == "stdout" and line:
                            result = line.rstrip("\r\n")
                            if not result:
                                self._terminate_locked()
                                raise ExactOracleProtocolError(
                                    "exact oracle returned an empty reply"
                                )
                            return result
                        if line == "" and process.poll() is not None:
                            raise ExactOracleUnavailable(
                                "persistent exact controller oracle exited "
                                "before replying"
                            )
            finally:
                selector.close()

    def replay(self, payload: Mapping[str, Any]) -> ExactOracleReplayOutput:
        parsed = _decode_oracle_reply(
            self._invoke("replay", payload), self.identity
        )
        return _output_from_reply(parsed, self.identity)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
            if process.poll() is None:
                try:
                    process.stdin.close()
                    process.wait(timeout=min(self.timeout_s, 2.0))
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    self._terminate_locked()
            for stream in (
                process.stdout,
                process.stderr,
            ):
                try:
                    stream.close()
                except OSError:
                    pass

    def __enter__(self) -> "PersistentSubprocessExactControllerOracle":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class CtypesExactControllerOracle:
    """Strict C ABI adapter for a separately built exact-controller library.

    The library must expose two functions returning library-owned, immutable
    UTF-8 JSON strings:

    ``grape_controller_oracle_handshake_json(void)`` and
    ``grape_controller_oracle_replay_json(const char*)``.
    """

    is_exact = True

    def __init__(
        self, library_path: str, expected_identity: ExactOracleIdentity
    ) -> None:
        path = os.path.abspath(str(library_path))
        if not os.path.isfile(path):
            raise ExactOracleUnavailable(
                "exact controller oracle library is unavailable"
            )
        if _sha256_file(path) != expected_identity.artifact_sha256:
            raise ExactOracleProtocolError(
                "exact controller oracle library hash mismatch"
            )
        try:
            library = ctypes.CDLL(path)
            handshake = library.grape_controller_oracle_handshake_json
            replay = library.grape_controller_oracle_replay_json
        except (OSError, AttributeError) as exc:
            raise ExactOracleUnavailable(
                "exact controller oracle library ABI is unavailable"
            ) from exc
        handshake.argtypes = []
        handshake.restype = ctypes.c_char_p
        replay.argtypes = [ctypes.c_char_p]
        replay.restype = ctypes.c_char_p
        self._library = library
        self._replay = replay
        self.identity = expected_identity
        raw = handshake()
        if raw is None:
            raise ExactOracleProtocolError("oracle handshake returned null")
        _decode_oracle_reply(raw, self.identity)

    def replay(self, payload: Mapping[str, Any]) -> ExactOracleReplayOutput:
        request = json.dumps(
            {
                "protocol": EXACT_ORACLE_PROTOCOL,
                "operation": "replay",
                "payload": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        raw = self._replay(request)
        if raw is None:
            raise ExactOracleProtocolError("oracle replay returned null")
        parsed = _decode_oracle_reply(raw, self.identity)
        return _output_from_reply(parsed, self.identity)


def _validated_sha256(value: Any, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("{} must be a lowercase SHA-256".format(name))
    return digest


@dataclass(frozen=True)
class ExactOracleFixtureProvenance:
    source_bag_sha256: str
    source_topics: Tuple[str, ...]
    interval_start_time_ns: int
    interval_end_time_ns: int
    frame_conventions: Mapping[str, str]
    unit_conventions: Mapping[str, str]
    motor_order: Tuple[str, ...]
    fixture_input_payload_sha256: str
    fixture_data_sha256: str
    extraction_config_sha256: str
    source_commit: str
    content_sha256: str
    schema: str = "grape_exact_oracle_fixture_provenance/v1"

    @staticmethod
    def _payload(
        source_bag_sha256,
        source_topics,
        interval_start_time_ns,
        interval_end_time_ns,
        frame_conventions,
        unit_conventions,
        motor_order,
        fixture_input_payload_sha256,
        fixture_data_sha256,
        extraction_config_sha256,
        source_commit,
        schema,
    ) -> Mapping[str, Any]:
        return {
            "schema": str(schema),
            "source_bag_sha256": str(source_bag_sha256),
            "source_topics": tuple(source_topics),
            "interval_start_time_ns": int(interval_start_time_ns),
            "interval_end_time_ns": int(interval_end_time_ns),
            "frame_conventions": dict(frame_conventions),
            "unit_conventions": dict(unit_conventions),
            "motor_order": tuple(motor_order),
            "fixture_input_payload_sha256": str(
                fixture_input_payload_sha256
            ),
            "fixture_data_sha256": str(fixture_data_sha256),
            "extraction_config_sha256": str(extraction_config_sha256),
            "source_commit": str(source_commit),
        }

    @classmethod
    def create(
        cls,
        *,
        source_bag_sha256: str,
        source_topics: Sequence[str],
        interval_start_time_ns: int,
        interval_end_time_ns: int,
        frame_conventions: Mapping[str, str],
        unit_conventions: Mapping[str, str],
        motor_order: Sequence[str],
        request_payload: Mapping[str, Any],
        continuous: Mapping[str, np.ndarray],
        events: np.ndarray,
        extraction_config_sha256: str,
        source_commit: str,
    ) -> "ExactOracleFixtureProvenance":
        schema = "grape_exact_oracle_fixture_provenance/v1"
        payload = cls._payload(
            source_bag_sha256,
            source_topics,
            interval_start_time_ns,
            interval_end_time_ns,
            frame_conventions,
            unit_conventions,
            motor_order,
            stable_hash(request_payload),
            stable_hash({"continuous": continuous, "events": events}),
            extraction_config_sha256,
            source_commit,
            schema,
        )
        return cls(content_sha256=stable_hash(payload), **payload)

    def __post_init__(self) -> None:
        if self.schema != "grape_exact_oracle_fixture_provenance/v1":
            raise ValueError("unsupported exact-oracle fixture provenance schema")
        source_topics = tuple(str(item) for item in self.source_topics)
        motor_order = tuple(str(item) for item in self.motor_order)
        frames = {
            str(key): str(value)
            for key, value in self.frame_conventions.items()
        }
        units = {
            str(key): str(value)
            for key, value in self.unit_conventions.items()
        }
        start = int(self.interval_start_time_ns)
        end = int(self.interval_end_time_ns)
        if (
            not source_topics
            or len(set(source_topics)) != len(source_topics)
            or not frames
            or not units
            or not motor_order
            or len(set(motor_order)) != len(motor_order)
            or start < 0
            or end <= start
            or not self.source_commit.strip()
            or self.source_commit.strip().lower() == "unknown"
        ):
            raise ValueError(
                "exact-oracle fixture requires complete source/topic/time/"
                "frame/unit/motor-order provenance"
            )
        source_hash = _validated_sha256(
            self.source_bag_sha256, "source_bag_sha256"
        )
        input_hash = _validated_sha256(
            self.fixture_input_payload_sha256,
            "fixture_input_payload_sha256",
        )
        data_hash = _validated_sha256(
            self.fixture_data_sha256, "fixture_data_sha256"
        )
        config_hash = _validated_sha256(
            self.extraction_config_sha256, "extraction_config_sha256"
        )
        content_hash = _validated_sha256(
            self.content_sha256, "content_sha256"
        )
        payload = self._payload(
            source_hash,
            source_topics,
            start,
            end,
            frames,
            units,
            motor_order,
            input_hash,
            data_hash,
            config_hash,
            self.source_commit,
            self.schema,
        )
        if stable_hash(payload) != content_hash:
            raise ValueError("exact-oracle fixture provenance hash mismatch")
        object.__setattr__(self, "source_bag_sha256", source_hash)
        object.__setattr__(self, "source_topics", source_topics)
        object.__setattr__(self, "interval_start_time_ns", start)
        object.__setattr__(self, "interval_end_time_ns", end)
        object.__setattr__(
            self, "frame_conventions", MappingProxyType(frames)
        )
        object.__setattr__(
            self, "unit_conventions", MappingProxyType(units)
        )
        object.__setattr__(self, "motor_order", motor_order)
        object.__setattr__(self, "fixture_input_payload_sha256", input_hash)
        object.__setattr__(self, "fixture_data_sha256", data_hash)
        object.__setattr__(self, "extraction_config_sha256", config_hash)
        object.__setattr__(self, "content_sha256", content_hash)

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            **self._payload(
                self.source_bag_sha256,
                self.source_topics,
                self.interval_start_time_ns,
                self.interval_end_time_ns,
                self.frame_conventions,
                self.unit_conventions,
                self.motor_order,
                self.fixture_input_payload_sha256,
                self.fixture_data_sha256,
                self.extraction_config_sha256,
                self.source_commit,
                self.schema,
            ),
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "ExactOracleFixtureProvenance":
        if not isinstance(values, Mapping):
            raise TypeError(
                "exact-oracle fixture provenance must be a mapping"
            )
        try:
            return cls(
                source_bag_sha256=values["source_bag_sha256"],
                source_topics=tuple(values["source_topics"]),
                interval_start_time_ns=values["interval_start_time_ns"],
                interval_end_time_ns=values["interval_end_time_ns"],
                frame_conventions=values["frame_conventions"],
                unit_conventions=values["unit_conventions"],
                motor_order=tuple(values["motor_order"]),
                fixture_input_payload_sha256=(
                    values["fixture_input_payload_sha256"]
                ),
                fixture_data_sha256=values["fixture_data_sha256"],
                extraction_config_sha256=(
                    values["extraction_config_sha256"]
                ),
                source_commit=values["source_commit"],
                content_sha256=values["content_sha256"],
                schema=values.get(
                    "schema",
                    "grape_exact_oracle_fixture_provenance/v1",
                ),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "exact-oracle fixture provenance is incomplete"
            ) from exc

    def content_is_valid(self) -> bool:
        try:
            payload = self._payload(
                _validated_sha256(
                    self.source_bag_sha256, "source_bag_sha256"
                ),
                tuple(str(item) for item in self.source_topics),
                int(self.interval_start_time_ns),
                int(self.interval_end_time_ns),
                {
                    str(key): str(value)
                    for key, value in self.frame_conventions.items()
                },
                {
                    str(key): str(value)
                    for key, value in self.unit_conventions.items()
                },
                tuple(str(item) for item in self.motor_order),
                _validated_sha256(
                    self.fixture_input_payload_sha256,
                    "fixture_input_payload_sha256",
                ),
                _validated_sha256(
                    self.fixture_data_sha256, "fixture_data_sha256"
                ),
                _validated_sha256(
                    self.extraction_config_sha256,
                    "extraction_config_sha256",
                ),
                self.source_commit,
                self.schema,
            )
            return stable_hash(payload) == self.content_sha256
        except (AttributeError, TypeError, ValueError):
            return False


@dataclass(frozen=True)
class ExactOracleConformanceFixture:
    continuous: Mapping[str, np.ndarray]
    events: np.ndarray
    provenance: ExactOracleFixtureProvenance
    fidelity: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, ExactOracleFixtureProvenance):
            raise TypeError(
                "exact-oracle fixture requires bag-derived provenance"
            )
        output = ExactOracleReplayOutput(
            identity=ExactOracleIdentity(
                protocol=EXACT_ORACLE_PROTOCOL,
                backend_id="fixture_cpp_exact_controller",
                implementation_language="C++",
                source_commit="fixture",
                artifact_sha256="0" * 64,
                capabilities=(
                    PC_MCU_EXACT_ORACLE_CAPABILITIES
                    if (
                        self.fidelity is None
                        and "pwm" in self.continuous
                    )
                    or self.fidelity == FIDELITY_PC_MCU_EXACT
                    else PC_EXACT_ORACLE_CAPABILITIES
                ),
                fidelity=(
                    FIDELITY_PC_MCU_EXACT
                    if self.fidelity is None and "pwm" in self.continuous
                    else (
                        FIDELITY_PC_EXACT
                        if self.fidelity is None
                        else self.fidelity
                    )
                ),
            ),
            continuous=self.continuous,
            events=self.events,
        )
        fidelity = (
            FIDELITY_PC_MCU_EXACT
            if self.fidelity is None and "pwm" in output.continuous
            else (
                FIDELITY_PC_EXACT
                if self.fidelity is None
                else normalize_fidelity(self.fidelity)
            )
        )
        required_channels = required_conformance_channels(fidelity)
        missing = sorted(
            set(required_channels) - set(output.continuous)
        )
        if missing:
            raise ValueError(
                "conformance fixture lacks channels: {}".format(
                    ", ".join(missing)
                )
            )
        if output.continuous["command_timestamp"].shape[1] != 1:
            raise ValueError(
                "command_timestamp must be a one-column matrix"
            )
        if (
            stable_hash(
                {"continuous": output.continuous, "events": output.events}
            )
            != self.provenance.fixture_data_sha256
        ):
            raise ValueError(
                "fixture arrays do not match bag-derived provenance hash"
            )
        object.__setattr__(self, "continuous", output.continuous)
        object.__setattr__(self, "events", output.events)
        object.__setattr__(self, "fidelity", fidelity)

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "continuous": {
                name: values.tolist()
                for name, values in sorted(self.continuous.items())
            },
            "events": self.events.tolist(),
            "provenance": self.provenance.to_mapping(),
            "fidelity": self.fidelity,
        }

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "ExactOracleConformanceFixture":
        if not isinstance(values, Mapping):
            raise TypeError("exact-oracle fixture must be a mapping")
        try:
            continuous = values["continuous"]
            if not isinstance(continuous, Mapping):
                raise TypeError("continuous must be a mapping")
            return cls(
                continuous={
                    str(name): np.asarray(items, dtype=float)
                    for name, items in continuous.items()
                },
                events=np.asarray(values["events"]),
                provenance=ExactOracleFixtureProvenance.from_mapping(
                    values["provenance"]
                ),
                fidelity=values.get("fidelity"),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "exact-oracle conformance fixture is incomplete"
            ) from exc


def _identity_mapping(
    identity: Optional[ExactOracleIdentity],
) -> Optional[Mapping[str, Any]]:
    if identity is None:
        return None
    return {
        "protocol": identity.protocol,
        "backend_id": identity.backend_id,
        "implementation_language": identity.implementation_language,
        "source_commit": identity.source_commit,
        "artifact_sha256": identity.artifact_sha256,
        "capabilities": list(identity.capabilities),
        "fidelity": identity.fidelity,
    }


def _metric_mapping(metric: ReplayMetrics) -> Mapping[str, Any]:
    return {
        "normalized_rmse": metric.normalized_rmse.tolist(),
        "normalized_maximum_error": (
            metric.normalized_maximum_error.tolist()
        ),
        "event_agreement": metric.event_agreement,
        "passed": metric.passed,
        "rmse_threshold": metric.rmse_threshold,
        "maximum_error_threshold": metric.maximum_error_threshold,
        "event_agreement_threshold": metric.event_agreement_threshold,
    }


def _metric_from_mapping(values: Mapping[str, Any]) -> ReplayMetrics:
    if not isinstance(values, Mapping):
        raise TypeError("replay channel metric must be a mapping")
    try:
        return ReplayMetrics(
            normalized_rmse=np.asarray(
                values["normalized_rmse"], dtype=float
            ),
            normalized_maximum_error=np.asarray(
                values["normalized_maximum_error"], dtype=float
            ),
            event_agreement=values["event_agreement"],
            passed=values["passed"],
            rmse_threshold=values["rmse_threshold"],
            maximum_error_threshold=values[
                "maximum_error_threshold"
            ],
            event_agreement_threshold=values[
                "event_agreement_threshold"
            ],
        )
    except KeyError as exc:
        raise ValueError("replay channel metric is incomplete") from exc


@dataclass(frozen=True)
class ExactOracleConformanceReport:
    """Immutable, content-addressed evidence from one factual replay.

    ``evidence_sha256`` covers the executable/source identity, fidelity,
    fixture provenance and data hash, exact request hash, all frozen channel
    metrics, and the decision.  A deserialized report therefore cannot be
    rebound to another oracle, fixture, or request without invalidating the
    digest.
    """

    passed: bool
    status: str
    reasons: Tuple[str, ...]
    channel_metrics: Mapping[str, ReplayMetrics]
    identity: Optional[ExactOracleIdentity]
    fixture_provenance: ExactOracleFixtureProvenance
    fixture_content_sha256: str
    request_payload_sha256: str
    fidelity: str = FIDELITY_PC_MCU_EXACT
    evidence_sha256: str = ""
    schema: str = "grape_exact_oracle_conformance_report/v2"

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise ValueError("conformance passed must be a built-in bool")
        status = str(self.status)
        reasons = tuple(str(item) for item in self.reasons)
        if not status:
            raise ValueError("conformance status is required")
        if not isinstance(
            self.fixture_provenance, ExactOracleFixtureProvenance
        ):
            raise TypeError("conformance report requires fixture provenance")
        if not self.fixture_provenance.content_is_valid():
            raise ValueError("conformance report fixture provenance was mutated")
        if (
            self.identity is not None
            and not isinstance(self.identity, ExactOracleIdentity)
        ):
            raise TypeError(
                "conformance report identity must be ExactOracleIdentity"
            )
        if self.schema != "grape_exact_oracle_conformance_report/v2":
            raise ValueError("unsupported exact conformance report schema")
        fidelity = normalize_fidelity(self.fidelity)
        if fidelity not in (FIDELITY_PC_EXACT, FIDELITY_PC_MCU_EXACT):
            raise ValueError(
                "exact conformance fidelity must be pc_exact or pc_mcu_exact"
            )
        if (
            self.passed
            and self.identity is not None
            and self.identity.fidelity != fidelity
        ):
            raise ValueError(
                "conformance report fidelity does not match oracle identity"
            )
        fixture_hash = _validated_sha256(
            self.fixture_content_sha256, "fixture_content_sha256"
        )
        request_hash = _validated_sha256(
            self.request_payload_sha256, "request_payload_sha256"
        )
        if fixture_hash != self.fixture_provenance.content_sha256:
            raise ValueError("conformance report fixture hash mismatch")

        metrics: Dict[str, ReplayMetrics] = {}
        for raw_name, raw_metric in self.channel_metrics.items():
            name = str(raw_name)
            if not isinstance(raw_metric, ReplayMetrics):
                raise TypeError(
                    "conformance channel metrics must be ReplayMetrics"
                )
            metric = ReplayMetrics(
                normalized_rmse=raw_metric.normalized_rmse,
                normalized_maximum_error=(
                    raw_metric.normalized_maximum_error
                ),
                event_agreement=raw_metric.event_agreement,
                passed=raw_metric.passed,
                rmse_threshold=raw_metric.rmse_threshold,
                maximum_error_threshold=(
                    raw_metric.maximum_error_threshold
                ),
                event_agreement_threshold=(
                    raw_metric.event_agreement_threshold
                ),
            )
            timestamp = name == "command_timestamp"
            expected_rmse = (
                COMMAND_TIMESTAMP_TOLERANCE_S
                if timestamp
                else FROZEN_REPLAY_RMSE_THRESHOLD
            )
            expected_maximum = (
                COMMAND_TIMESTAMP_TOLERANCE_S
                if timestamp
                else FROZEN_REPLAY_MAXIMUM_ERROR_THRESHOLD
            )
            if (
                metric.rmse_threshold != expected_rmse
                or metric.maximum_error_threshold != expected_maximum
                or metric.event_agreement_threshold
                != FROZEN_REPLAY_EVENT_AGREEMENT_THRESHOLD
            ):
                raise ValueError(
                    "{} does not use frozen conformance thresholds".format(
                        name
                    )
                )
            metrics[name] = metric

        required = set(required_conformance_channels(fidelity))
        if self.passed:
            if (
                status != "PASS"
                or reasons
                or self.identity is None
                or request_hash
                != self.fixture_provenance.fixture_input_payload_sha256
                or set(metrics) != required
                or not all(metric.passed for metric in metrics.values())
            ):
                raise ValueError(
                    "passing conformance report lacks complete bound evidence"
                )
        elif status == "PASS":
            raise ValueError("a failed conformance report cannot say PASS")

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self,
            "channel_metrics",
            MappingProxyType(dict(sorted(metrics.items()))),
        )
        object.__setattr__(self, "fixture_content_sha256", fixture_hash)
        object.__setattr__(self, "request_payload_sha256", request_hash)
        object.__setattr__(self, "fidelity", fidelity)
        payload = self._evidence_payload()
        computed = stable_hash(payload)
        supplied = str(self.evidence_sha256).lower()
        if supplied and _validated_sha256(
            supplied, "evidence_sha256"
        ) != computed:
            raise ValueError("conformance report evidence hash mismatch")
        object.__setattr__(self, "evidence_sha256", computed)

    def _evidence_payload(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "passed": self.passed,
            "status": self.status,
            "reasons": list(self.reasons),
            "channel_metrics": {
                name: _metric_mapping(metric)
                for name, metric in sorted(
                    self.channel_metrics.items()
                )
            },
            "identity": _identity_mapping(self.identity),
            "fixture_provenance": (
                self.fixture_provenance.to_mapping()
            ),
            "fixture_content_sha256": self.fixture_content_sha256,
            "request_payload_sha256": self.request_payload_sha256,
            "fidelity": self.fidelity,
        }

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            **self._evidence_payload(),
            "evidence_sha256": self.evidence_sha256,
        }

    @property
    def content_sha256(self) -> str:
        return self.evidence_sha256

    def content_is_valid(self) -> bool:
        try:
            return bool(
                self.fixture_provenance.content_is_valid()
                and stable_hash(self._evidence_payload())
                == self.evidence_sha256
            )
        except (AttributeError, TypeError, ValueError):
            return False

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "ExactOracleConformanceReport":
        if not isinstance(values, Mapping):
            raise TypeError("exact conformance report must be a mapping")
        try:
            raw_metrics = values["channel_metrics"]
            if not isinstance(raw_metrics, Mapping):
                raise TypeError("channel_metrics must be a mapping")
            raw_identity = values.get("identity")
            return cls(
                passed=values["passed"],
                status=values["status"],
                reasons=tuple(values["reasons"]),
                channel_metrics={
                    str(name): _metric_from_mapping(metric)
                    for name, metric in raw_metrics.items()
                },
                identity=(
                    None
                    if raw_identity is None
                    else ExactOracleIdentity.from_mapping(raw_identity)
                ),
                fixture_provenance=(
                    ExactOracleFixtureProvenance.from_mapping(
                        values["fixture_provenance"]
                    )
                ),
                fixture_content_sha256=values[
                    "fixture_content_sha256"
                ],
                request_payload_sha256=values[
                    "request_payload_sha256"
                ],
                fidelity=values["fidelity"],
                evidence_sha256=values["evidence_sha256"],
                schema=values["schema"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "exact conformance report is incomplete"
            ) from exc


def evaluate_exact_oracle_conformance(
    oracle: Optional[Any],
    payload: Mapping[str, Any],
    fixture: ExactOracleConformanceFixture,
    rmse_threshold: float = FROZEN_REPLAY_RMSE_THRESHOLD,
    maximum_error_threshold: float = (
        FROZEN_REPLAY_MAXIMUM_ERROR_THRESHOLD
    ),
    event_agreement_threshold: float = (
        FROZEN_REPLAY_EVENT_AGREEMENT_THRESHOLD
    ),
) -> ExactOracleConformanceReport:
    """Apply the factual replay gate; absence and surrogates fail closed."""

    if not isinstance(fixture, ExactOracleConformanceFixture):
        raise TypeError("fixture must be ExactOracleConformanceFixture")
    if not isinstance(payload, Mapping):
        raise TypeError("exact-oracle conformance payload must be a mapping")
    if (
        float(rmse_threshold) != FROZEN_REPLAY_RMSE_THRESHOLD
        or float(maximum_error_threshold)
        != FROZEN_REPLAY_MAXIMUM_ERROR_THRESHOLD
        or float(event_agreement_threshold)
        != FROZEN_REPLAY_EVENT_AGREEMENT_THRESHOLD
    ):
        raise ValueError(
            "exact-oracle conformance thresholds are frozen"
        )
    if not fixture.provenance.content_is_valid():
        raise ValueError("exact-oracle fixture provenance was mutated")
    if (
        stable_hash(
            {"continuous": fixture.continuous, "events": fixture.events}
        )
        != fixture.provenance.fixture_data_sha256
    ):
        raise ValueError("exact-oracle fixture data was mutated")
    request_hash = stable_hash(payload)

    def report(
        *,
        passed: bool,
        status: str,
        reasons: Sequence[str],
        channel_metrics: Mapping[str, ReplayMetrics],
        identity: Optional[ExactOracleIdentity],
    ) -> ExactOracleConformanceReport:
        return ExactOracleConformanceReport(
            passed=passed,
            status=status,
            reasons=tuple(reasons),
            channel_metrics=dict(channel_metrics),
            identity=identity,
            fixture_provenance=fixture.provenance,
            fixture_content_sha256=fixture.provenance.content_sha256,
            request_payload_sha256=request_hash,
            fidelity=fixture.fidelity,
        )

    if oracle is None:
        return report(
            passed=False,
            status="ORACLE_UNAVAILABLE",
            reasons=("no external exact-controller oracle is connected",),
            channel_metrics={},
            identity=None,
        )
    identity = getattr(oracle, "identity", None)
    if (
        getattr(oracle, "is_exact", False) is not True
        or not isinstance(identity, ExactOracleIdentity)
        or not callable(getattr(oracle, "replay", None))
    ):
        return report(
            passed=False,
            status="IDENTITY_REJECTED",
            reasons=(
                "backend lacks a verified C++ exact-oracle identity; "
                "surrogates cannot pass this gate",
            ),
            channel_metrics={},
            identity=None,
        )
    if identity.fidelity != fixture.fidelity:
        return report(
            passed=False,
            status="FIDELITY_REJECTED",
            reasons=(
                "oracle fidelity {} does not match fixture fidelity {}".format(
                    identity.fidelity, fixture.fidelity
                ),
            ),
            channel_metrics={},
            identity=identity,
        )
    if request_hash != fixture.provenance.fixture_input_payload_sha256:
        return report(
            passed=False,
            status="FIXTURE_BINDING_REJECTED",
            reasons=(
                "oracle request payload does not match the bag-derived fixture",
            ),
            channel_metrics={},
            identity=identity,
        )
    try:
        output = oracle.replay(payload)
    except Exception as exc:  # fail closed across an external process boundary
        return report(
            passed=False,
            status="EXECUTION_FAILED",
            reasons=("oracle replay failed: {}".format(type(exc).__name__),),
            channel_metrics={},
            identity=identity,
        )
    if not isinstance(output, ExactOracleReplayOutput) or output.identity != identity:
        return report(
            passed=False,
            status="PROTOCOL_REJECTED",
            reasons=("oracle replay output identity/type mismatch",),
            channel_metrics={},
            identity=identity,
        )
    required_channels = required_conformance_channels(identity.fidelity)
    missing = sorted(set(required_channels) - set(output.continuous))
    if missing:
        return report(
            passed=False,
            status="CONFORMANCE_FAILED",
            reasons=(
                "oracle output lacks channels: {}".format(", ".join(missing)),
            ),
            channel_metrics={},
            identity=identity,
        )
    metrics: Dict[str, ReplayMetrics] = {}
    reasons = []
    for channel in required_channels:
        timestamp = channel == "command_timestamp"
        try:
            metrics[channel] = replay_metrics(
                output.continuous[channel],
                fixture.continuous[channel],
                output.events,
                fixture.events,
                rmse_threshold=(
                    COMMAND_TIMESTAMP_TOLERANCE_S
                    if timestamp
                    else rmse_threshold
                ),
                maximum_error_threshold=(
                    COMMAND_TIMESTAMP_TOLERANCE_S
                    if timestamp
                    else maximum_error_threshold
                ),
                event_agreement_threshold=event_agreement_threshold,
            )
        except ValueError as exc:
            reasons.append("{}: {}".format(channel, exc))
            continue
        if timestamp and not np.array_equal(
            output.continuous[channel],
            fixture.continuous[channel],
        ):
            reasons.append(
                "command_timestamp differs from the frozen fixture"
            )
            continue
        if not metrics[channel].passed:
            reasons.append("{} exceeded the frozen replay thresholds".format(channel))
    passed = not reasons and len(metrics) == len(required_channels)
    return report(
        passed=passed,
        status="PASS" if passed else "CONFORMANCE_FAILED",
        reasons=tuple(reasons),
        channel_metrics=metrics,
        identity=identity,
    )


# ---------------------------------------------------------------------------
# Explicit gates for conditional candidates


_CONDITIONAL_PREREQUISITES = {
    "sparse_gp_residual": (
        "heldout_parametric_residual_structure",
        "support_variance_growth_validated",
        "no_free_extrapolation",
    ),
    "likelihood_free_bayessim": (
        "likelihood_model_invalidated",
        "black_box_simulator_heldout_validated",
        "exact_controller_oracle_passed",
    ),
}


@dataclass(frozen=True)
class ConditionalCandidateGate:
    candidate: str
    decision: str
    missing_prerequisites: Tuple[str, ...]
    evidence: Mapping[str, bool]


def evaluate_conditional_candidate(
    candidate: str, evidence: Mapping[str, bool]
) -> ConditionalCandidateGate:
    """Return ``START`` only when every predeclared condition is true."""

    name = str(candidate)
    if name not in _CONDITIONAL_PREREQUISITES:
        raise ValueError("unknown conditional candidate: {}".format(name))
    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be a mapping")
    required = _CONDITIONAL_PREREQUISITES[name]
    normalized = {item: evidence.get(item) is True for item in required}
    missing = tuple(item for item in required if not normalized[item])
    return ConditionalCandidateGate(
        candidate=name,
        decision="START" if not missing else "PRUNE",
        missing_prerequisites=missing,
        evidence=normalized,
    )


__all__ = [
    "BatchImuPreintegrationSmoother",
    "BayesianBackendComparison",
    "CtypesExactControllerOracle",
    "COMMAND_TIMESTAMP_TOLERANCE_S",
    "ConditionalCandidateGate",
    "EXACT_ORACLE_PROTOCOL",
    "ExactOracleConformanceFixture",
    "ExactOracleConformanceReport",
    "ExactOracleFixtureProvenance",
    "ExactOracleError",
    "ExactOracleIdentity",
    "ExactOracleProtocolError",
    "ExactOracleReplayOutput",
    "ExactOracleUnavailable",
    "FactorGraphSmootherConfig",
    "FROZEN_REPLAY_EVENT_AGREEMENT_THRESHOLD",
    "FROZEN_REPLAY_MAXIMUM_ERROR_THRESHOLD",
    "FROZEN_REPLAY_RMSE_THRESHOLD",
    "LinearGaussianRandomWalkModel",
    "MechanicsGaugeReport",
    "MechanicsIdentifiabilityReport",
    "ParticleLikelihoodDegeneracy",
    "ParticleMarginalMetropolisHastings",
    "ParticleMarginalMhConfig",
    "ParticleMarginalMhPosterior",
    "ParticleStateSpaceModel",
    "PC_EXACT_CONFORMANCE_CHANNELS",
    "PC_EXACT_ORACLE_CAPABILITIES",
    "PC_MCU_EXACT_CONFORMANCE_CHANNELS",
    "PC_MCU_EXACT_ORACLE_CAPABILITIES",
    "PersistentSubprocessExactControllerOracle",
    "REQUIRED_CONFORMANCE_CHANNELS",
    "REQUIRED_ORACLE_CAPABILITIES",
    "StructuredMechanicsParameters",
    "StructuredSixDofMechanicsResponse",
    "SubprocessExactControllerOracle",
    "bootstrap_particle_log_likelihood",
    "compare_pmmh_with_modular_smc",
    "evaluate_conditional_candidate",
    "evaluate_exact_oracle_conformance",
    "required_conformance_channels",
    "required_oracle_capabilities",
]
