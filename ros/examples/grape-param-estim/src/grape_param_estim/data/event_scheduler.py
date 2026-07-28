"""Deterministic scheduling across controller, plant, and observation grids."""

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np

from .provenance import stable_hash


@dataclass(frozen=True)
class EventGrid:
    """A named, strictly increasing event-time grid."""

    name: str
    timestamps: Tuple[float, ...]

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        values = tuple(float(value) for value in self.timestamps)
        array = np.asarray(values, dtype=float)
        if not name:
            raise ValueError("event grid name must not be empty")
        if array.ndim != 1 or not np.all(np.isfinite(array)):
            raise ValueError("event grid times must be a finite one-dimensional sequence")
        if array.size > 1 and np.any(np.diff(array) <= 0.0):
            raise ValueError("event grid times must be strictly increasing")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "timestamps", values)

    @property
    def times(self) -> Tuple[float, ...]:
        """Compatibility spelling used by the initial fixture implementation."""

        return self.timestamps

    @classmethod
    def from_times(
        cls,
        name: str,
        times: Iterable[float],
        *,
        sort_and_deduplicate: bool = False,
    ) -> "EventGrid":
        values = tuple(float(value) for value in times)
        if sort_and_deduplicate:
            values = tuple(sorted(set(values)))
        return cls(name=name, timestamps=values)

    @property
    def content_sha256(self) -> str:
        return stable_hash(self.to_dict())

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "timestamps": list(self.timestamps)}


@dataclass(frozen=True, order=True)
class ScheduledEvent:
    """One merged event; ordering fields make ties deterministic."""

    time: float
    priority: int
    grid_index: int
    grid_name: str


class EventScheduler:
    """Stable merge of event grids with explicit same-time priority."""

    def __init__(
        self,
        grids: Sequence[EventGrid],
        *,
        priority: Optional[Sequence[str]] = None,
    ) -> None:
        normalized = tuple(grids)
        if not normalized:
            raise ValueError("event scheduler requires at least one grid")
        by_name: Dict[str, EventGrid] = {}
        for grid in normalized:
            if not isinstance(grid, EventGrid):
                raise TypeError("event scheduler inputs must be EventGrid objects")
            if grid.name in by_name:
                raise ValueError("event grid names must be unique")
            by_name[grid.name] = grid
        order = tuple(priority) if priority is not None else tuple(by_name)
        if len(set(order)) != len(order) or set(order) != set(by_name):
            raise ValueError("scheduler priority must list every grid exactly once")
        ranks = {name: index for index, name in enumerate(order)}
        events = [
            ScheduledEvent(
                time=stamp,
                priority=ranks[grid.name],
                grid_index=index,
                grid_name=grid.name,
            )
            for grid in normalized
            for index, stamp in enumerate(grid.times)
        ]
        self._grids: Mapping[str, EventGrid] = dict(
            (name, by_name[name]) for name in order
        )
        self._priority = order
        self._events = tuple(sorted(events))

    @property
    def priority(self) -> Tuple[str, ...]:
        return self._priority

    @property
    def content_sha256(self) -> str:
        return stable_hash(
            {
                "priority": list(self._priority),
                "grids": [
                    self._grids[name].to_dict() for name in self._priority
                ],
            }
        )

    def __iter__(self) -> Iterator[ScheduledEvent]:
        return iter(self._events)

    def merged_events(self) -> Tuple[ScheduledEvent, ...]:
        """Return the immutable chronological merge used by rollout loops."""

        return self._events

    def events_between(
        self,
        start: float,
        end: float,
        *,
        include_end: bool = True,
    ) -> Tuple[ScheduledEvent, ...]:
        lower = float(start)
        upper = float(end)
        if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
            raise ValueError("scheduler interval must be finite and ordered")
        if include_end:
            return tuple(event for event in self._events if lower <= event.time <= upper)
        return tuple(event for event in self._events if lower <= event.time < upper)


__all__ = ["EventGrid", "EventScheduler", "ScheduledEvent"]
