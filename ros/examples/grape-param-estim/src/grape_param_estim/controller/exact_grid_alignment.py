"""Verified replacement of command-derived ticks by exact replay-frame ticks."""

from dataclasses import replace
from types import MappingProxyType
from typing import Any, Dict, Mapping

import numpy as np

from grape_param_estim.data.controller_fixture import (
    ControllerReplayFixture,
)
from grape_param_estim.data.event_scheduler import EventGrid
from grape_param_estim.data.initial_state import initial_state_posterior
from grape_param_estim.episode import stable_hash


def _same_grid(left: EventGrid, right: EventGrid) -> bool:
    return bool(
        isinstance(left, EventGrid)
        and isinstance(right, EventGrid)
        and left.name == right.name
        and left.timestamps == right.timestamps
    )


def _initial_state_group_key(value: Any) -> str:
    return stable_hash(
        {
            "initial_plant_state": value.initial_plant_state,
            "initial_actuator_state": value.initial_actuator_state,
            "sensor_bias": value.sensor_bias,
        }
    )


def _rederive_nuisance_samples(
    *,
    episode_id: str,
    trajectory: Any,
    start: float,
    previous: tuple,
) -> tuple:
    if any(
        not all(
            hasattr(nuisance, name)
            for name in (
                "initial_plant_state",
                "initial_actuator_state",
                "sensor_bias",
                "disturbance_parameters",
                "disturbance_model_id",
                "state_sample_id",
                "weight",
            )
        )
        for nuisance in previous
    ):
        return tuple(
            initial_state_posterior(
                episode_id,
                trajectory,
                start,
                maximum_samples=len(previous),
            ).samples
        )
    groups = []
    indexes = {}
    for nuisance in previous:
        key = _initial_state_group_key(nuisance)
        if key not in indexes:
            indexes[key] = len(groups)
            groups.append([])
        groups[indexes[key]].append(nuisance)
    base = initial_state_posterior(
        episode_id,
        trajectory,
        start,
        maximum_samples=len(groups),
    ).samples
    if len(base) != len(groups):
        raise ValueError(
            "narrowed exact support cannot preserve the prepared "
            "initial-state/disturbance posterior"
        )
    result = []
    for new_initial, old_group in zip(base, groups):
        for old in old_group:
            bound_id = stable_hash(
                {
                    "schema": (
                        "grape_narrowed_exact_nuisance_sample/v1"
                    ),
                    "old_state_sample_id": old.state_sample_id,
                    "new_start": float(start),
                    "new_initial_plant_state": (
                        new_initial.initial_plant_state
                    ),
                    "disturbance_parameters": (
                        old.disturbance_parameters
                    ),
                    "disturbance_model_id": (
                        old.disturbance_model_id
                    ),
                }
            )
            result.append(
                replace(
                    new_initial,
                    initial_actuator_state=(
                        old.initial_actuator_state
                    ),
                    disturbance_parameters=(
                        old.disturbance_parameters
                    ),
                    disturbance_model_id=old.disturbance_model_id,
                    controller_state=old.controller_state,
                    state_sample_id=bound_id,
                    weight=old.weight,
                )
            )
    return tuple(result)


def align_prepared_exact_grids(
    prepared: Mapping[str, Any],
    fixtures: Mapping[str, ControllerReplayFixture],
) -> Mapping[str, Any]:
    """Adopt ReplayFrame controller ticks after strict prepared-data checks.

    ``prepare_episodes`` obtains its provisional controller events from the
    recorded command topic.  Exact closed-loop replay must instead use the
    controller's ReplayFrame events.  The fixture may preserve factual
    ReplayFrame ticks just outside the configured window, but inference is
    clipped to the prepared plant support.  This helper preserves the
    independently prepared observation, likelihood, and report grids, and
    permits only the deterministic union of the supported prepared integration
    suffix with those in-support controller ticks.
    """

    episode_ids = set(str(key) for key in prepared)
    if not episode_ids or set(fixtures) != episode_ids:
        raise ValueError(
            "exact fixtures must cover exactly the prepared episodes"
        )
    result: Dict[str, Any] = {}
    for episode_id, episode in prepared.items():
        fixture = fixtures[episode_id]
        if not isinstance(fixture, ControllerReplayFixture):
            raise TypeError(
                "exact fixture {} has the wrong type".format(episode_id)
            )
        config = episode.config
        expected_offsets = (
            float(config["replay_start_offset_s"]),
            float(config["score_start_offset_s"]),
            float(config["score_end_offset_s"]),
        )
        actual_offsets = (
            fixture.replay_start_offset_s,
            fixture.score_start_offset_s,
            fixture.score_end_offset_s,
        )
        normalized_hash = str(
            fixture.metadata.get("normalized_episode_sha256", "")
        )
        if (
            fixture.episode_id != episode_id
            or fixture.source_bag_sha256
            != episode.observations.source_bag_sha256
            or actual_offsets != expected_offsets
            or normalized_hash
            != episode.observations.normalized_episode_sha256
        ):
            raise ValueError(
                "exact fixture {} is not bound to the prepared bag/window".format(
                    episode_id
                )
            )
        for field in (
            "observation_grid",
            "likelihood_grid",
            "report_grid",
        ):
            if not _same_grid(
                getattr(fixture.grids, field),
                getattr(episode.grids, field),
            ):
                raise ValueError(
                    "exact fixture {} changes the prepared {}".format(
                        episode_id, field
                    )
                )
        controller = np.asarray(
            fixture.grids.controller_tick_grid.timestamps, dtype=float
        )
        factual_controller = np.asarray(
            fixture.factual_controller_tick_grid.timestamps, dtype=float
        )
        prepared_integration = np.asarray(
            episode.grids.plant_integration_grid.timestamps, dtype=float
        )
        if controller.size == 0:
            raise ValueError(
                "exact fixture {} has no inference ReplayFrame ticks".format(
                    episode_id
                )
            )
        supported_integration = prepared_integration[
            prepared_integration >= controller[0]
        ]
        expected_integration = tuple(
            float(item)
            for item in np.unique(
                np.concatenate((supported_integration, controller))
            )
        )
        if (
            controller[0] < prepared_integration[0]
            or controller[-1] > prepared_integration[-1]
            or supported_integration.size == 0
            or expected_integration[0] != controller[0]
            or fixture.grids.plant_integration_grid.timestamps
            != expected_integration
            or np.any(
                ~np.isin(controller, factual_controller)
            )
        ):
            raise ValueError(
                "exact fixture {} plant grid is not the supported prepared "
                "grid suffix union its inference ReplayFrame ticks".format(
                    episode_id
                )
            )
        replacements = {"grids": fixture.grids}
        if controller[0] != prepared_integration[0]:
            trajectory = getattr(episode, "trajectory_posterior", None)
            nuisance_samples = tuple(
                getattr(episode, "nuisance_samples", ())
            )
            if nuisance_samples and trajectory is None:
                raise ValueError(
                    "exact fixture {} narrows plant support but the prepared "
                    "trajectory needed to re-derive initial state is "
                    "unavailable".format(episode_id)
                )
            if nuisance_samples:
                trajectory_times = np.asarray(
                    trajectory.timestamps, dtype=float
                )
                if (
                    trajectory_times.size == 0
                    or controller[0] < trajectory_times[0]
                    or controller[0] > trajectory_times[-1]
                ):
                    raise ValueError(
                        "exact fixture {} inference start exceeds prepared "
                        "trajectory support".format(episode_id)
                    )
                replacements["nuisance_samples"] = (
                    _rederive_nuisance_samples(
                        episode_id=episode_id,
                        trajectory=trajectory,
                        start=float(controller[0]),
                        previous=nuisance_samples,
                    )
                )
        result[episode_id] = replace(episode, **replacements)
    return MappingProxyType(dict(sorted(result.items())))


__all__ = ["align_prepared_exact_grids"]
