"""Initial-state posterior extraction from coherent trajectory samples."""

from dataclasses import dataclass, replace
from typing import Any, Tuple

import numpy as np

from grape_param_estim.episode import stable_hash
from grape_param_estim.plant.parameters import EpisodeNuisance
from grape_param_estim.state_smoother import TrajectoryPosterior


@dataclass(frozen=True)
class InitialStatePosterior:
    episode_id: str
    stamp: float
    samples: Tuple[EpisodeNuisance, ...]
    source_trajectory_sha256: str

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if not samples or any(
            not isinstance(item, EpisodeNuisance) for item in samples
        ):
            raise ValueError("initial-state posterior requires nuisance samples")
        weights = np.asarray([item.weight for item in samples], dtype=float)
        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError("initial-state posterior weights must sum to one")
        object.__setattr__(self, "samples", samples)

    @property
    def content_sha256(self) -> str:
        return stable_hash(
            {
                "episode_id": self.episode_id,
                "stamp": self.stamp,
                "source_trajectory_sha256": self.source_trajectory_sha256,
                "samples": [
                    {
                        "id": item.state_sample_id,
                        "weight": item.weight,
                        "plant": item.initial_plant_state,
                        "actuator": item.initial_actuator_state,
                        "disturbance": item.disturbance_parameters,
                        "disturbance_model_id": (
                            item.disturbance_model_id
                        ),
                        "sensor_bias": item.sensor_bias,
                    }
                    for item in self.samples
                ],
            }
        )


def initial_state_posterior(
    episode_id: str,
    trajectory: TrajectoryPosterior,
    stamp: float,
    maximum_samples: int = 8,
) -> InitialStatePosterior:
    if not isinstance(trajectory, TrajectoryPosterior):
        raise TypeError("trajectory must be TrajectoryPosterior")
    query = float(stamp)
    index = int(np.argmin(np.abs(trajectory.timestamps - query)))
    count = min(int(maximum_samples), max(1, trajectory.sample_count))
    if count < 1:
        raise ValueError("maximum_samples must be positive")
    samples = []
    if trajectory.sample_count:
        selected = np.arange(count)
        weights = np.asarray(trajectory.sample_weights[selected], dtype=float)
        weights /= np.sum(weights)
        for output_index, sample_index in enumerate(selected):
            state = np.concatenate(
                (
                    trajectory.sample_position_world[sample_index, index],
                    trajectory.sample_velocity_world[sample_index, index],
                    trajectory.sample_quaternion_xyzw[sample_index, index],
                    trajectory.sample_angular_velocity_body[sample_index, index],
                )
            )
            bias = np.concatenate(
                (
                    trajectory.sample_accelerometer_bias_body[
                        sample_index, index
                    ],
                    trajectory.sample_gyro_bias_body[sample_index, index],
                )
            )
            samples.append(
                EpisodeNuisance(
                    initial_plant_state=state,
                    initial_actuator_state=np.empty(0),
                    disturbance_parameters=np.zeros(6),
                    sensor_bias=bias,
                    controller_state=None,
                    state_sample_id="{}_{}".format(
                        episode_id,
                        int(trajectory.sample_ids[sample_index]),
                    ),
                    weight=float(weights[output_index]),
                )
            )
    else:
        state = np.concatenate(
            (
                trajectory.position_world[index],
                trajectory.velocity_world[index],
                trajectory.quaternion_xyzw[index],
                trajectory.angular_velocity_body[index],
            )
        )
        bias = np.concatenate(
            (
                trajectory.accelerometer_bias_body[index],
                trajectory.gyro_bias_body[index],
            )
        )
        samples.append(
            EpisodeNuisance(
                initial_plant_state=state,
                initial_actuator_state=np.empty(0),
                disturbance_parameters=np.zeros(6),
                sensor_bias=bias,
                controller_state=None,
                state_sample_id="{}_mean".format(episode_id),
                weight=1.0,
            )
        )
    trajectory_hash = stable_hash(
        {
            "timestamps": trajectory.timestamps,
            "position": trajectory.position_world,
            "velocity": trajectory.velocity_world,
            "orientation": trajectory.quaternion_xyzw,
            "angular_velocity": trajectory.angular_velocity_body,
            "sample_ids": trajectory.sample_ids,
            "sample_weights": trajectory.sample_weights,
        }
    )
    return InitialStatePosterior(
        episode_id=str(episode_id),
        stamp=float(trajectory.timestamps[index]),
        samples=tuple(samples),
        source_trajectory_sha256=trajectory_hash,
    )


def with_disturbance_samples(
    posterior: InitialStatePosterior,
    *,
    model_id: str,
    lower: Any,
    upper: Any,
    sample_count: int,
    seed: int,
) -> InitialStatePosterior:
    """Cross an initial-state posterior with a bounded disturbance prior."""

    if not isinstance(posterior, InitialStatePosterior):
        raise TypeError("posterior must be InitialStatePosterior")
    disturbance_model_id = str(model_id).strip()
    if disturbance_model_id not in (
        "effective_constant_acceleration_disturbance_v1",
        "constant_wrench_disturbance_v1",
    ):
        raise ValueError("unsupported episode disturbance model")
    minimum = np.asarray(lower, dtype=float).reshape(-1)
    maximum = np.asarray(upper, dtype=float).reshape(-1)
    count = int(sample_count)
    if (
        minimum.shape != (6,)
        or maximum.shape != (6,)
        or not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(maximum))
        or np.any(minimum > 0.0)
        or np.any(maximum < 0.0)
        or np.any(maximum <= minimum)
        or count < 1
    ):
        raise ValueError(
            "disturbance prior requires six finite bounds containing zero "
            "and a positive sample count"
        )
    rng = np.random.default_rng(int(seed))
    disturbance = np.zeros((count, 6), dtype=float)
    if count > 1:
        disturbance[1:] = rng.uniform(
            minimum, maximum, size=(count - 1, 6)
        )
    samples = []
    for initial in posterior.samples:
        for index, vector in enumerate(disturbance):
            vector_hash = stable_hash(vector)[:12]
            samples.append(
                replace(
                    initial,
                    disturbance_parameters=vector,
                    disturbance_model_id=disturbance_model_id,
                    state_sample_id=(
                        "{}_disturbance_{:03d}_{}".format(
                            initial.state_sample_id,
                            index,
                            vector_hash,
                        )
                    ),
                    weight=float(initial.weight) / count,
                )
            )
    return InitialStatePosterior(
        episode_id=posterior.episode_id,
        stamp=posterior.stamp,
        samples=tuple(samples),
        source_trajectory_sha256=posterior.source_trajectory_sha256,
    )


__all__ = [
    "InitialStatePosterior",
    "initial_state_posterior",
    "with_disturbance_samples",
]
