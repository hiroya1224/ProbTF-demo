"""Static weighted-particle inference for the Grape rigid-body replay."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.data import AnalysisData
from grape_param_estim.model import (
    GrapeRigidBodyModel,
    ReplayResult,
    RigidBodyParameters,
    quaternion_to_matrix,
    replay_segments,
    rotation_vector_from_matrix,
)


PARAMETER_NAMES = (
    "mass_scale",
    "force_scale",
    "inertia_scale",
    "torque_scale",
)
ProgressCallback = Callable[[float, str, str], None]


@dataclass(frozen=True)
class LikelihoodWeights:
    """Diagonal SE(3) likelihood weights."""

    translation: float
    rotation: float

    def __post_init__(self) -> None:
        values = np.asarray((self.translation, self.rotation), dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("likelihood weights must be finite and positive")


@dataclass(frozen=True)
class ParticlePosterior:
    """The canonical weighted static-parameter posterior."""

    particles: np.ndarray
    weights: np.ndarray
    log_likelihood: np.ndarray
    resampled: bool

    def __post_init__(self) -> None:
        particles = np.asarray(self.particles, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        likelihood = np.asarray(self.log_likelihood, dtype=float)
        if (
            particles.ndim != 2
            or particles.shape[1] != len(PARAMETER_NAMES)
            or particles.shape[0] < 2
            or np.any(~np.isfinite(particles))
            or np.any(particles <= 0.0)
        ):
            raise ValueError("particles must be a positive finite N by 4 array")
        if (
            weights.shape != (particles.shape[0],)
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.isclose(np.sum(weights), 1.0, atol=1.0e-10)
        ):
            raise ValueError("particle weights must be finite and sum to one")
        if likelihood.shape != weights.shape or np.any(
            np.isnan(likelihood)
        ):
            raise ValueError("log_likelihood must match particle weights")

    @property
    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(np.asarray(self.weights) ** 2))


@dataclass(frozen=True)
class BagPushforward:
    """Posterior trajectories and uncertain transforms for one bag."""

    bag_path: str
    analysis: AnalysisData
    nominal: ReplayResult
    posterior_position: np.ndarray
    posterior_orientation_xyzw: np.ndarray
    delta_translation: np.ndarray
    delta_rotation_vector: np.ndarray

    def __post_init__(self) -> None:
        particle_count = self.posterior_position.shape[0]
        sample_count = self.analysis.times.size
        expected_vector = (particle_count, sample_count, 3)
        if self.posterior_position.shape != expected_vector:
            raise ValueError("posterior_position must have shape N by T by 3")
        if self.posterior_orientation_xyzw.shape != (
            particle_count,
            sample_count,
            4,
        ):
            raise ValueError(
                "posterior_orientation_xyzw must have shape N by T by 4"
            )
        if (
            self.delta_translation.shape != expected_vector
            or self.delta_rotation_vector.shape != expected_vector
        ):
            raise ValueError("uncertain transforms must have shape N by T by 3")


@dataclass(frozen=True)
class EstimationResult:
    posterior: ParticlePosterior
    bags: Tuple[BagPushforward, ...]
    seed: int


def _prior_array(
    prior_bounds: Mapping[str, Sequence[float]],
) -> np.ndarray:
    if set(prior_bounds) != set(PARAMETER_NAMES):
        raise ValueError(
            "prior_bounds must define exactly: {}".format(
                ", ".join(PARAMETER_NAMES)
            )
        )
    bounds = np.asarray(
        [prior_bounds[name] for name in PARAMETER_NAMES], dtype=float
    )
    if (
        bounds.shape != (len(PARAMETER_NAMES), 2)
        or np.any(~np.isfinite(bounds))
        or np.any(bounds <= 0.0)
        or np.any(bounds[:, 1] <= bounds[:, 0])
    ):
        raise ValueError("every prior must contain positive increasing bounds")
    return bounds


def sample_prior(
    prior_bounds: Mapping[str, Sequence[float]],
    particle_count: int,
    seed: int,
) -> np.ndarray:
    """Draw independent uniform scale particles."""

    count = int(particle_count)
    if count < 2:
        raise ValueError("particle_count must be at least 2")
    bounds = _prior_array(prior_bounds)
    generator = np.random.default_rng(int(seed))
    return generator.uniform(
        bounds[:, 0],
        bounds[:, 1],
        size=(count, len(PARAMETER_NAMES)),
    )


def parameters_from_particle(
    nominal: RigidBodyParameters,
    particle: Sequence[float],
) -> RigidBodyParameters:
    values = np.asarray(particle, dtype=float)
    if values.shape != (len(PARAMETER_NAMES),):
        raise ValueError("particle must contain four parameter scales")
    return RigidBodyParameters(
        mass=nominal.mass,
        inertia=np.asarray(nominal.inertia, dtype=float),
        mass_scale=float(values[0]),
        force_scale=float(values[1]),
        inertia_scale=float(values[2]),
        torque_scale=float(values[3]),
    )


def trajectory_log_likelihood(
    replay: ReplayResult,
    likelihood_weights: LikelihoodWeights,
) -> float:
    """Return the diagonal weighted relative-SE(3) log likelihood."""

    residual = np.asarray(replay.residual_se3, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        loss = (
            float(likelihood_weights.translation)
            * np.sum(residual[:, :3] ** 2)
            + float(likelihood_weights.rotation)
            * np.sum(residual[:, 3:] ** 2)
        )
    return float(-0.5 * loss) if np.isfinite(loss) else float("-inf")


def normalise_log_weights(log_likelihood: np.ndarray) -> np.ndarray:
    values = np.asarray(log_likelihood, dtype=float)
    finite = np.isfinite(values)
    if values.ndim != 1 or not np.any(finite):
        raise FloatingPointError("all particle likelihoods are non-finite")
    shifted = np.full_like(values, float("-inf"))
    shifted[finite] = values[finite] - np.max(values[finite])
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("particle weights could not be normalised")
    return weights / total


def _systematic_resample(
    particles: np.ndarray,
    weights: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    count = particles.shape[0]
    positions = (generator.random() + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    indices = np.searchsorted(cumulative, positions, side="right")
    return particles[indices].copy()


def _reflect_into_bounds(
    particles: np.ndarray, bounds: np.ndarray
) -> np.ndarray:
    lower = bounds[:, 0]
    span = bounds[:, 1] - lower
    phase = np.mod(particles - lower, 2.0 * span)
    return lower + np.where(phase <= span, phase, 2.0 * span - phase)


def _evaluate_particles(
    analyses: Sequence[AnalysisData],
    particles: np.ndarray,
    nominal: RigidBodyParameters,
    model: GrapeRigidBodyModel,
    likelihood_weights: LikelihoodWeights,
    progress_callback: Optional[ProgressCallback],
    progress_start: float,
    progress_end: float,
    stage: str,
) -> np.ndarray:
    log_likelihood = np.empty(particles.shape[0], dtype=float)
    for index, particle in enumerate(particles):
        parameters = parameters_from_particle(nominal, particle)
        value = 0.0
        for analysis in analyses:
            try:
                replay = replay_segments(analysis, model, parameters)
                value += trajectory_log_likelihood(
                    replay, likelihood_weights
                )
            except (FloatingPointError, OverflowError):
                value = float("-inf")
                break
        log_likelihood[index] = value
        if progress_callback is not None:
            fraction = (index + 1) / particles.shape[0]
            progress_callback(
                progress_start
                + fraction * (progress_end - progress_start),
                stage,
                "particle {}/{}".format(index + 1, particles.shape[0]),
            )
    return log_likelihood


def relative_transform_from_nominal(
    nominal: ReplayResult,
    particle: ReplayResult,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``T_nominal^-1 T_particle`` along two replay trajectories."""

    if nominal.times.shape != particle.times.shape or not np.allclose(
        nominal.times, particle.times, atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("nominal and particle replay times must match")
    sample_count = nominal.times.size
    translation = np.empty((sample_count, 3), dtype=float)
    rotation_vector = np.empty((sample_count, 3), dtype=float)
    for index in range(sample_count):
        nominal_rotation = quaternion_to_matrix(
            nominal.orientation_xyzw[index]
        )
        particle_rotation = quaternion_to_matrix(
            particle.orientation_xyzw[index]
        )
        translation[index] = nominal_rotation.T @ (
            particle.position[index] - nominal.position[index]
        )
        rotation_vector[index] = rotation_vector_from_matrix(
            nominal_rotation.T @ particle_rotation
        )
    return translation, rotation_vector


def _pushforward_bag(
    analysis: AnalysisData,
    particles: np.ndarray,
    nominal_parameters: RigidBodyParameters,
    model: GrapeRigidBodyModel,
    progress_callback: Optional[ProgressCallback],
    progress_start: float,
    progress_end: float,
) -> BagPushforward:
    nominal = replay_segments(analysis, model, nominal_parameters)
    particle_count = particles.shape[0]
    sample_count = analysis.times.size
    position = np.empty((particle_count, sample_count, 3), dtype=float)
    orientation = np.empty((particle_count, sample_count, 4), dtype=float)
    delta_translation = np.empty_like(position)
    delta_rotation = np.empty_like(position)
    for index, particle in enumerate(particles):
        parameters = parameters_from_particle(
            nominal_parameters, particle
        )
        replay = replay_segments(analysis, model, parameters)
        position[index] = replay.position
        orientation[index] = replay.orientation_xyzw
        (
            delta_translation[index],
            delta_rotation[index],
        ) = relative_transform_from_nominal(nominal, replay)
        if progress_callback is not None:
            fraction = (index + 1) / particle_count
            progress_callback(
                progress_start
                + fraction * (progress_end - progress_start),
                "posterior pushforward",
                "{}: particle {}/{}".format(
                    Path(analysis.bag_path).name, index + 1, particle_count
                ),
            )
    return BagPushforward(
        bag_path=analysis.bag_path,
        analysis=analysis,
        nominal=nominal,
        posterior_position=position,
        posterior_orientation_xyzw=orientation,
        delta_translation=delta_translation,
        delta_rotation_vector=delta_rotation,
    )


def estimate_parameters(
    analyses: Sequence[AnalysisData],
    nominal_parameters: RigidBodyParameters,
    prior_bounds: Mapping[str, Sequence[float]],
    particle_count: int,
    likelihood_weights: LikelihoodWeights,
    seed: int = 42,
    resample_ess_fraction: float = 0.10,
    jitter_fraction: float = 0.03,
    model: Optional[GrapeRigidBodyModel] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> EstimationResult:
    """Infer shared static scales and push them through every bag trajectory."""

    selected_analyses = tuple(analyses)
    if not selected_analyses:
        raise ValueError("at least one analysis interval is required")
    count = int(particle_count)
    if count < 2:
        raise ValueError("particle_count must be at least 2")
    ess_fraction = float(resample_ess_fraction)
    jitter = float(jitter_fraction)
    if (
        not np.isfinite(ess_fraction)
        or not 0.0 <= ess_fraction <= 1.0
        or not np.isfinite(jitter)
        or jitter < 0.0
    ):
        raise ValueError("resample and jitter fractions are invalid")
    bounds = _prior_array(prior_bounds)
    generator = np.random.default_rng(int(seed))
    particles = generator.uniform(
        bounds[:, 0],
        bounds[:, 1],
        size=(count, len(PARAMETER_NAMES)),
    )
    motion_model = model or GrapeRigidBodyModel()
    log_likelihood = _evaluate_particles(
        selected_analyses,
        particles,
        nominal_parameters,
        motion_model,
        likelihood_weights,
        progress_callback,
        0.0,
        0.45,
        "initial likelihood",
    )
    weights = normalise_log_weights(log_likelihood)
    effective_sample_size = float(1.0 / np.sum(weights**2))
    resampled = effective_sample_size < ess_fraction * count
    if resampled:
        particles = _systematic_resample(particles, weights, generator)
        if jitter > 0.0:
            particles += generator.normal(
                0.0,
                jitter * (bounds[:, 1] - bounds[:, 0]),
                size=particles.shape,
            )
            particles = _reflect_into_bounds(particles, bounds)
        log_likelihood = _evaluate_particles(
            selected_analyses,
            particles,
            nominal_parameters,
            motion_model,
            likelihood_weights,
            progress_callback,
            0.45,
            0.70,
            "resampled likelihood",
        )
        weights = normalise_log_weights(log_likelihood)
    elif progress_callback is not None:
        progress_callback(
            0.70,
            "likelihood complete",
            "ESS {:.1f}/{}".format(effective_sample_size, count),
        )

    bags = []
    bag_fraction = 0.30 / len(selected_analyses)
    for index, analysis in enumerate(selected_analyses):
        start = 0.70 + bag_fraction * index
        end = 0.70 + bag_fraction * (index + 1)
        bags.append(
            _pushforward_bag(
                analysis,
                particles,
                nominal_parameters,
                motion_model,
                progress_callback,
                start,
                end,
            )
        )
    posterior = ParticlePosterior(
        particles=particles,
        weights=weights,
        log_likelihood=log_likelihood,
        resampled=resampled,
    )
    if progress_callback is not None:
        progress_callback(1.0, "complete", "posterior ready")
    return EstimationResult(
        posterior=posterior,
        bags=tuple(bags),
        seed=int(seed),
    )


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Sequence[float],
) -> np.ndarray:
    """Weighted quantiles over particle axis zero for arbitrary value shape."""

    samples = np.asarray(values, dtype=float)
    particle_weights = np.asarray(weights, dtype=float)
    requested = np.asarray(quantiles, dtype=float)
    if (
        samples.ndim < 1
        or particle_weights.shape != (samples.shape[0],)
        or np.any(~np.isfinite(samples))
        or np.any(~np.isfinite(particle_weights))
        or np.any(particle_weights < 0.0)
        or np.sum(particle_weights) <= 0.0
        or requested.ndim != 1
        or np.any(~np.isfinite(requested))
        or np.any((requested < 0.0) | (requested > 1.0))
    ):
        raise ValueError("weighted_quantile received invalid values")
    flat = samples.reshape(samples.shape[0], -1)
    result = np.empty((requested.size, flat.shape[1]), dtype=float)
    for column in range(flat.shape[1]):
        order = np.argsort(flat[:, column], kind="stable")
        sorted_values = flat[order, column]
        sorted_weights = particle_weights[order]
        cumulative = np.cumsum(sorted_weights)
        cumulative /= cumulative[-1]
        result[:, column] = np.interp(
            requested,
            np.concatenate(([0.0], cumulative)),
            np.concatenate(([sorted_values[0]], sorted_values)),
        )
    return result.reshape((requested.size,) + samples.shape[1:])


def parameter_summary(posterior: ParticlePosterior) -> dict:
    quantiles = weighted_quantile(
        posterior.particles,
        posterior.weights,
        (0.025, 0.25, 0.5, 0.75, 0.975),
    )
    summary = {}
    for index, name in enumerate(PARAMETER_NAMES):
        summary[name] = {
            "mean": float(
                np.sum(
                    posterior.weights * posterior.particles[:, index]
                )
            ),
            "q02_5": float(quantiles[0, index]),
            "q25": float(quantiles[1, index]),
            "median": float(quantiles[2, index]),
            "q75": float(quantiles[3, index]),
            "q97_5": float(quantiles[4, index]),
        }
    summary["force_mass_ratio"] = {
        "median": float(
            weighted_quantile(
                posterior.particles[:, 1] / posterior.particles[:, 0],
                posterior.weights,
                (0.5,),
            )[0]
        )
    }
    summary["torque_inertia_ratio"] = {
        "median": float(
            weighted_quantile(
                posterior.particles[:, 3] / posterior.particles[:, 2],
                posterior.weights,
                (0.5,),
            )[0]
        )
    }
    return summary


def result_arrays(result: EstimationResult) -> dict:
    """Convert an estimation result to an allow_pickle=False NPZ payload."""

    arrays = {
        "schema_version": np.asarray((1,), dtype=np.int64),
        "parameter_names": np.asarray(PARAMETER_NAMES, dtype="U32"),
        "particles": np.asarray(result.posterior.particles, dtype=float),
        "weights": np.asarray(result.posterior.weights, dtype=float),
        "log_likelihood": np.asarray(
            result.posterior.log_likelihood, dtype=float
        ),
        "resampled": np.asarray(
            (int(result.posterior.resampled),), dtype=np.int8
        ),
        "seed": np.asarray((result.seed,), dtype=np.int64),
        "bag_paths": np.asarray(
            [bag.bag_path for bag in result.bags], dtype="U1024"
        ),
    }
    for index, bag in enumerate(result.bags):
        prefix = "bag_{}_".format(index)
        arrays[prefix + "times"] = bag.analysis.times
        arrays[prefix + "segment_id"] = bag.analysis.segment_id
        arrays[prefix + "observed_position"] = bag.analysis.position
        arrays[prefix + "observed_orientation_xyzw"] = (
            bag.analysis.orientation_xyzw
        )
        arrays[prefix + "nominal_position"] = bag.nominal.position
        arrays[prefix + "nominal_orientation_xyzw"] = (
            bag.nominal.orientation_xyzw
        )
        arrays[prefix + "posterior_position"] = bag.posterior_position
        arrays[prefix + "posterior_orientation_xyzw"] = (
            bag.posterior_orientation_xyzw
        )
        arrays[prefix + "delta_translation"] = bag.delta_translation
        arrays[prefix + "delta_rotation_vector"] = (
            bag.delta_rotation_vector
        )
    return arrays


def save_result(path: str, result: EstimationResult) -> str:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **result_arrays(result))
    temporary.replace(destination)
    return str(destination.resolve())


def load_result(path: str) -> dict:
    source = Path(path).expanduser()
    with np.load(str(source), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


__all__ = [
    "BagPushforward",
    "EstimationResult",
    "LikelihoodWeights",
    "PARAMETER_NAMES",
    "ParticlePosterior",
    "estimate_parameters",
    "load_result",
    "normalise_log_weights",
    "parameter_summary",
    "parameters_from_particle",
    "result_arrays",
    "relative_transform_from_nominal",
    "sample_prior",
    "save_result",
    "trajectory_log_likelihood",
    "weighted_quantile",
]
