"""Minimal, episode-local detection of controlled and airborne intervals."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from grape_param_estim.failure_bag import FailureBagRecording


@dataclass(frozen=True)
class EpisodeDetectionSettings:
    """Small set of controls for transparent automatic segmentation."""

    active_flight_states: Tuple[int, ...]
    diagnostic_flight_states: Tuple[int, ...]
    baseline_window_s: float
    minimum_active_duration_s: float
    minimum_liftoff_height_m: float
    minimum_airborne_duration_s: float
    persistence_s: float
    standardized_threshold: float

    def __post_init__(self) -> None:
        if not self.active_flight_states:
            raise ValueError("at least one active flight state is required")
        if not set(self.diagnostic_flight_states).issubset(
            self.active_flight_states
        ):
            raise ValueError(
                "diagnostic flight states must also be active states"
            )
        positive = (
            self.baseline_window_s,
            self.minimum_active_duration_s,
            self.minimum_liftoff_height_m,
            self.minimum_airborne_duration_s,
            self.persistence_s,
            self.standardized_threshold,
        )
        if not all(
            np.isfinite(value) and float(value) > 0.0
            for value in positive
        ):
            raise ValueError("episode detection settings must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        return cls(
            active_flight_states=tuple(
                int(value) for value in values["active_flight_states"]
            ),
            diagnostic_flight_states=tuple(
                int(value)
                for value in values["diagnostic_flight_states"]
            ),
            baseline_window_s=float(values["baseline_window_s"]),
            minimum_active_duration_s=float(
                values["minimum_active_duration_s"]
            ),
            minimum_liftoff_height_m=float(
                values["minimum_liftoff_height_m"]
            ),
            minimum_airborne_duration_s=float(
                values["minimum_airborne_duration_s"]
            ),
            persistence_s=float(values["persistence_s"]),
            standardized_threshold=float(
                values["standardized_threshold"]
            ),
        )


@dataclass(frozen=True)
class DetectedEpisode:
    """One contiguous controller-active interval and its local support plane."""

    index: int
    start_s: float
    end_s: float
    flight_states: Tuple[int, ...]
    support_height_m: float
    support_height_sigma_m: float
    support_vertical_velocity_sigma_m_s: float
    support_sample_count: int
    liftoff_s: float
    status: str
    reason: str

    @property
    def identifiable(self) -> bool:
        return self.status == "candidate"


def _robust_sigma(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    return 1.4826 * float(np.median(np.abs(array - median)))


def _zoh_values(
    source_times: np.ndarray,
    source_values: np.ndarray,
    query_times: np.ndarray,
    missing_value: int,
) -> np.ndarray:
    indices = np.searchsorted(
        source_times, query_times, side="right"
    ) - 1
    result = np.full(query_times.shape, missing_value, dtype=int)
    valid = indices >= 0
    result[valid] = source_values[indices[valid]].astype(int)
    return result


def _boolean_runs(mask: np.ndarray):
    values = np.asarray(mask, dtype=bool)
    edges = np.diff(
        np.concatenate(([False], values, [False])).astype(int)
    )
    return zip(
        np.flatnonzero(edges == 1),
        np.flatnonzero(edges == -1),
    )


def _close_short_gaps(
    timestamps: np.ndarray,
    mask: np.ndarray,
    maximum_gap_s: float,
) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for start, stop in _boolean_runs(~result):
        if start == 0 or stop == result.size:
            continue
        gap = float(timestamps[stop - 1] - timestamps[start])
        if timestamps.size > 1:
            gap += float(np.median(np.diff(timestamps)))
        if gap <= maximum_gap_s:
            result[start:stop] = True
    return result


def command_valid_mask(
    recording: FailureBagRecording,
    timestamps: Sequence[float],
    delay_s: float = 0.0,
) -> np.ndarray:
    """Return command-heartbeat validity using its observed sample period."""

    query = np.asarray(timestamps, dtype=float) - float(delay_s)
    command_times = recording.command_times
    positive_steps = np.diff(command_times)
    positive_steps = positive_steps[positive_steps > 0.0]
    if positive_steps.size == 0:
        return np.zeros(query.shape, dtype=bool)
    maximum_age = 5.0 * float(np.median(positive_steps))
    indices = np.searchsorted(
        command_times, query, side="right"
    ) - 1
    valid = (indices >= 0) & (query <= command_times[-1])
    valid_indices = np.flatnonzero(valid)
    valid[valid_indices] &= (
        query[valid_indices] - command_times[indices[valid_indices]]
        <= maximum_age
    )
    return valid


def flight_state_at(
    recording: FailureBagRecording,
    timestamps: Sequence[float],
) -> np.ndarray:
    """Sample the controller-published flight state with zero-order hold."""

    return _zoh_values(
        recording.flight_state_times,
        recording.flight_state,
        np.asarray(timestamps, dtype=float),
        missing_value=-1,
    )


def mask_to_intervals(
    timestamps: Sequence[float],
    mask: Sequence[bool],
) -> list:
    """Convert a sampled boolean mask into half-open time intervals."""

    times = np.asarray(timestamps, dtype=float)
    values = np.asarray(mask, dtype=bool)
    if times.shape != values.shape:
        raise ValueError("timestamps and mask must have the same shape")
    if times.size == 0:
        return []
    step = (
        float(np.median(np.diff(times)))
        if times.size > 1
        else 0.0
    )
    return [
        [float(times[start]), float(times[stop - 1] + step)]
        for start, stop in _boolean_runs(values)
    ]


def _episode_states(
    recording: FailureBagRecording,
    start_s: float,
    end_s: float,
) -> Tuple[int, ...]:
    selected = (
        (recording.flight_state_times >= start_s)
        & (recording.flight_state_times <= end_s)
    )
    return tuple(
        sorted(
            set(
                int(value)
                for value in recording.flight_state[selected]
            )
        )
    )


def _detect_liftoff(
    recording: FailureBagRecording,
    start_index: int,
    stop_index: int,
    support_height: float,
    support_height_sigma: float,
    support_vertical_velocity: float,
    support_vertical_velocity_sigma: float,
    settings: EpisodeDetectionSettings,
) -> float:
    times = recording.state_times
    position_z = recording.position[:, 2]
    velocity_z = recording.linear_velocity[:, 2]
    step = float(np.median(np.diff(times)))
    count = max(2, int(np.ceil(settings.persistence_s / step)))
    height_threshold = max(
        settings.minimum_liftoff_height_m,
        settings.standardized_threshold * support_height_sigma,
    )
    velocity_threshold = (
        support_vertical_velocity
        + settings.standardized_threshold
        * support_vertical_velocity_sigma
    )
    for index in range(
        start_index, max(start_index, stop_index - count + 1)
    ):
        selected = slice(index, index + count)
        relative_height = position_z[selected] - support_height
        if (
            np.all(relative_height >= height_threshold)
            and float(np.median(velocity_z[selected]))
            > max(0.0, velocity_threshold)
        ):
            return float(times[index])
    return float("nan")


def detect_control_episodes(
    recording: FailureBagRecording,
    settings: EpisodeDetectionSettings,
) -> list:
    """Detect controller-active episodes and episode-relative liftoff."""

    times = recording.state_times
    flight_state = _zoh_values(
        recording.flight_state_times,
        recording.flight_state,
        times,
        missing_value=-1,
    )
    active = np.isin(flight_state, settings.active_flight_states)
    active = _close_short_gaps(
        times, active, settings.persistence_s
    )
    step = float(np.median(np.diff(times)))
    episodes = []
    for start_index, stop_index in _boolean_runs(active):
        start_s = float(times[start_index])
        end_s = min(
            recording.bag_duration_s,
            float(times[stop_index - 1] + step),
        )
        if end_s - start_s < settings.minimum_active_duration_s:
            continue

        baseline = (
            (times >= start_s - settings.baseline_window_s)
            & (times < start_s)
            & ~active
        )
        baseline_indices = np.flatnonzero(baseline)
        support_height = float("nan")
        height_sigma = float("nan")
        velocity_sigma = float("nan")
        liftoff = float("nan")
        status = "not_identifiable"
        reason = "no_stationary_baseline"
        support_count = 0

        if baseline_indices.size >= 10:
            baseline_speed = np.linalg.norm(
                recording.linear_velocity[baseline_indices], axis=1
            )
            speed_median = float(np.median(baseline_speed))
            speed_sigma = _robust_sigma(baseline_speed)
            stationary = baseline_indices[
                baseline_speed
                <= speed_median
                + settings.standardized_threshold
                * max(speed_sigma, np.finfo(float).eps)
            ]
            if stationary.size >= 10:
                support_height = float(
                    np.median(recording.position[stationary, 2])
                )
                height_sigma = _robust_sigma(
                    recording.position[stationary, 2]
                )
                support_velocity = float(
                    np.median(recording.linear_velocity[stationary, 2])
                )
                velocity_sigma = _robust_sigma(
                    recording.linear_velocity[stationary, 2]
                )
                support_count = int(stationary.size)
                liftoff = _detect_liftoff(
                    recording,
                    start_index,
                    stop_index,
                    support_height,
                    height_sigma,
                    support_velocity,
                    velocity_sigma,
                    settings,
                )
                if not np.isfinite(liftoff):
                    reason = "no_persistent_liftoff"
                elif (
                    end_s - liftoff
                    < settings.minimum_airborne_duration_s
                ):
                    reason = "airborne_interval_too_short"
                else:
                    status = "candidate"
                    reason = "airborne_candidate"

        episodes.append(
            DetectedEpisode(
                index=len(episodes),
                start_s=start_s,
                end_s=end_s,
                flight_states=_episode_states(
                    recording, start_s, end_s
                ),
                support_height_m=support_height,
                support_height_sigma_m=height_sigma,
                support_vertical_velocity_sigma_m_s=velocity_sigma,
                support_sample_count=support_count,
                liftoff_s=liftoff,
                status=status,
                reason=reason,
            )
        )
    return episodes


__all__ = [
    "DetectedEpisode",
    "EpisodeDetectionSettings",
    "command_valid_mask",
    "detect_control_episodes",
    "flight_state_at",
    "mask_to_intervals",
]
