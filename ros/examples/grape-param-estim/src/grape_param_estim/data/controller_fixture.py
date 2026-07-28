"""Hash-bound controller replay fixture and separated time-grid contract."""

from dataclasses import asdict, dataclass, field, is_dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping

import numpy as np

from .event_scheduler import EventGrid, EventScheduler
from .provenance import stable_hash, validated_sha256


CONTROLLER_REPLAY_FIXTURE_SCHEMA = "grape_controller_replay_fixture/v2"
_GRID_NAMES = (
    "controller_tick",
    "plant_integration",
    "observation",
    "likelihood",
    "report",
)


def _serialized_contract(value: Any) -> Any:
    converter = getattr(value, "to_mapping", None)
    if callable(converter):
        return converter()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(
        "fixture controller inputs must be mappings or serializable contracts"
    )


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_metadata(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        )
    if isinstance(value, np.ndarray):
        return tuple(_freeze_metadata(item) for item in value.tolist())
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_metadata(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("fixture metadata floats must be finite")
        return float(value)
    raise TypeError("fixture metadata must contain only JSON-compatible values")


def _thaw_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_metadata(item) for item in value]
    return value


@dataclass(frozen=True)
class EpisodeTimeGrids:
    """The five grids that must not be conflated during replay."""

    controller_tick_grid: EventGrid
    plant_integration_grid: EventGrid
    observation_grid: EventGrid
    likelihood_grid: EventGrid
    report_grid: EventGrid

    def __post_init__(self) -> None:
        grids = self.as_tuple()
        actual = tuple(grid.name for grid in grids)
        if actual != _GRID_NAMES:
            raise ValueError(
                "replay grid names must be {}".format(", ".join(_GRID_NAMES))
            )
        if any(not grid.times for grid in grids):
            raise ValueError("all replay grids must contain at least one event")
        controller_times = set(self.controller_tick_grid.times)
        integration_times = set(self.plant_integration_grid.times)
        missing_integration_events = tuple(
            sorted(controller_times - integration_times)
        )
        if missing_integration_events:
            raise ValueError(
                "every controller tick must be an exact plant-integration "
                "event; missing {}".format(
                    ", ".join(
                        "{:.17g}".format(value)
                        for value in missing_integration_events
                    )
                )
            )

    def as_tuple(self):
        return (
            self.controller_tick_grid,
            self.plant_integration_grid,
            self.observation_grid,
            self.likelihood_grid,
            self.report_grid,
        )

    def scheduler(self) -> EventScheduler:
        return EventScheduler(self.as_tuple(), priority=_GRID_NAMES)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "controller_tick_grid": self.controller_tick_grid.to_dict(),
            "plant_integration_grid": self.plant_integration_grid.to_dict(),
            "observation_grid": self.observation_grid.to_dict(),
            "likelihood_grid": self.likelihood_grid.to_dict(),
            "report_grid": self.report_grid.to_dict(),
        }


@dataclass(frozen=True)
class ControllerReplayFixture:
    """Replay evidence plus the controller ticks safe for plant inference.

    ``controller_inputs`` is the full factual replay stream and may contain
    the immediate ticks bracketing the configured replay/score interval.
    ``grids.controller_tick_grid`` is the clipped inference clock and must
    remain inside the prepared plant support.  Keeping the two domains
    explicit prevents an irregular hardware tick from forcing plant-state
    extrapolation merely to prove factual boundary coverage.
    """

    episode_id: str
    source_bag_sha256: str
    topic_inventory_sha256: str
    replay_start_offset_s: float
    score_start_offset_s: float
    score_end_offset_s: float
    grids: EpisodeTimeGrids
    controller_inputs: tuple = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = CONTROLLER_REPLAY_FIXTURE_SCHEMA

    def __post_init__(self) -> None:
        episode = str(self.episode_id).strip()
        if not episode:
            raise ValueError("fixture episode_id must not be empty")
        source_hash = validated_sha256(
            self.source_bag_sha256, "source_bag_sha256"
        )
        inventory_hash = validated_sha256(
            self.topic_inventory_sha256, "topic_inventory_sha256"
        )
        replay_start = float(self.replay_start_offset_s)
        score_start = float(self.score_start_offset_s)
        score_end = float(self.score_end_offset_s)
        if (
            not np.all(np.isfinite([replay_start, score_start, score_end]))
            or replay_start > score_start
            or score_start >= score_end
        ):
            raise ValueError(
                "fixture requires replay_start <= score_start < score_end"
            )
        if not isinstance(self.grids, EpisodeTimeGrids):
            raise TypeError("fixture grids must be EpisodeTimeGrids")
        controller_inputs = tuple(self.controller_inputs)
        if controller_inputs:
            input_times = tuple(
                float(
                    item.get("stamp")
                    if isinstance(item, Mapping)
                    else getattr(item, "stamp")
                )
                for item in controller_inputs
            )
            if (
                not np.all(np.isfinite(input_times))
                or len(set(input_times)) != len(input_times)
                or any(
                    right <= left
                    for left, right in zip(
                        input_times[:-1], input_times[1:]
                    )
                )
            ):
                raise ValueError(
                    "factual controller inputs must have unique increasing "
                    "timestamps"
                )
            inference_times = (
                self.grids.controller_tick_grid.timestamps
            )
            if any(stamp not in input_times for stamp in inference_times):
                raise ValueError(
                    "every inference controller tick must be present in the "
                    "factual controller input stream"
                )
            if (
                input_times[0] > replay_start
                or input_times[-1] < score_end
            ):
                raise ValueError(
                    "factual controller inputs must bracket the configured "
                    "replay and score interval"
                )
        controller = self.grids.controller_tick_grid.times
        plant = self.grids.plant_integration_grid.times
        if controller[0] != plant[0]:
            raise ValueError(
                "plant integration must begin at the first inference "
                "controller tick"
            )
        if plant[0] > score_start or plant[-1] < score_end:
            raise ValueError(
                "plant integration grid must cover the scored interval"
            )
        if not any(
            replay_start <= stamp < score_start for stamp in controller
        ):
            raise ValueError("controller tick grid must include unscored pre-roll")
        for label, grid in (
            ("likelihood", self.grids.likelihood_grid),
            ("report", self.grids.report_grid),
        ):
            if grid.times[0] < score_start or grid.times[-1] > score_end:
                raise ValueError(
                    "{} grid must stay inside the scored interval".format(label)
                )
        object.__setattr__(self, "episode_id", episode)
        object.__setattr__(self, "source_bag_sha256", source_hash)
        object.__setattr__(self, "topic_inventory_sha256", inventory_hash)
        object.__setattr__(self, "replay_start_offset_s", replay_start)
        object.__setattr__(self, "score_start_offset_s", score_start)
        object.__setattr__(self, "score_end_offset_s", score_end)
        object.__setattr__(self, "controller_inputs", controller_inputs)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )
        if self.schema != CONTROLLER_REPLAY_FIXTURE_SCHEMA:
            raise ValueError("unsupported controller replay fixture schema")

    @property
    def fixture_sha256(self) -> str:
        return stable_hash(self.to_dict(include_fixture_hash=False))

    @property
    def factual_controller_tick_grid(self) -> EventGrid:
        timestamps = tuple(
            float(
                item.get("stamp")
                if isinstance(item, Mapping)
                else getattr(item, "stamp")
            )
            for item in self.controller_inputs
        )
        if not timestamps:
            timestamps = self.grids.controller_tick_grid.timestamps
        return EventGrid("factual_controller_tick", timestamps)

    def scheduler(self) -> EventScheduler:
        return self.grids.scheduler()

    def to_dict(self, include_fixture_hash: bool = True) -> Dict[str, Any]:
        result = {
            "schema": self.schema,
            "episode_id": self.episode_id,
            "source_bag_sha256": self.source_bag_sha256,
            "topic_inventory_sha256": self.topic_inventory_sha256,
            "replay_start_offset_s": self.replay_start_offset_s,
            "score_start_offset_s": self.score_start_offset_s,
            "score_end_offset_s": self.score_end_offset_s,
            "factual_controller_tick_grid": (
                self.factual_controller_tick_grid.to_dict()
            ),
            "grids": self.grids.to_dict(),
            "controller_inputs": [
                _serialized_contract(item) for item in self.controller_inputs
            ],
            "metadata": _thaw_metadata(self.metadata),
        }
        if include_fixture_hash:
            result["fixture_sha256"] = self.fixture_sha256
        return result


# Compatibility alias retained for the first Phase-2 callers.
ReplayGrids = EpisodeTimeGrids


__all__ = [
    "CONTROLLER_REPLAY_FIXTURE_SCHEMA",
    "ControllerReplayFixture",
    "EpisodeTimeGrids",
    "ReplayGrids",
]
