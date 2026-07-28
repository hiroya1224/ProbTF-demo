"""Failure-event detection and post-failure observation censoring.

The validation layer deliberately represents a failure as more than a binary
label.  A :class:`FailureEvent` records both its type and occurrence time, so
posterior-predictive validation can distinguish a particle that fails in the
observed way and at the observed time from one that merely fails eventually.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np


def _timestamps(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if (
        result.size == 0
        or not np.all(np.isfinite(result))
        or np.any(np.diff(result) <= 0.0)
    ):
        raise ValueError("timestamps must be finite and strictly increasing")
    output = np.array(result, copy=True)
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class FailureEvent:
    """One detected failure with an auditable type, time, and source."""

    failure_type: str
    stamp: float
    detector_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        failure_type = str(self.failure_type).strip()
        detector_id = str(self.detector_id).strip()
        stamp = float(self.stamp)
        if not failure_type or not detector_id:
            raise ValueError("failure_type and detector_id must not be empty")
        if not np.isfinite(stamp):
            raise ValueError("failure stamp must be finite")
        object.__setattr__(self, "failure_type", failure_type)
        object.__setattr__(self, "detector_id", detector_id)
        object.__setattr__(self, "stamp", stamp)
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata))
        )

    @property
    def failure_time(self) -> float:
        """Compatibility spelling for consumers that use ``failure_time``."""

        return self.stamp


class FailureDetector(Protocol):
    """Structural contract implemented by trajectory failure detectors."""

    detector_id: str

    def detect(
        self, timestamps: np.ndarray, **observations: Any
    ) -> Optional[FailureEvent]:
        ...


@dataclass(frozen=True)
class FirstMaskFailureDetector:
    """Return the first true sample in a caller-supplied failure mask."""

    failure_type: str
    mask_key: str = "failure_mask"
    detector_id: str = "first_mask_failure/v1"

    def __post_init__(self) -> None:
        if not str(self.failure_type).strip() or not str(self.mask_key).strip():
            raise ValueError("failure_type and mask_key must not be empty")
        if not str(self.detector_id).strip():
            raise ValueError("detector_id must not be empty")

    def detect(
        self, timestamps: np.ndarray, **observations: Any
    ) -> Optional[FailureEvent]:
        times = _timestamps(timestamps)
        if self.mask_key not in observations:
            raise KeyError(
                "failure detector requires observation '{}'".format(
                    self.mask_key
                )
            )
        mask = np.asarray(observations[self.mask_key], dtype=bool).reshape(-1)
        if mask.shape != times.shape:
            raise ValueError("failure mask must align with timestamps")
        indexes = np.flatnonzero(mask)
        if not indexes.size:
            return None
        index = int(indexes[0])
        return FailureEvent(
            failure_type=self.failure_type,
            stamp=float(times[index]),
            detector_id=self.detector_id,
            metadata={"sample_index": index, "mask_key": self.mask_key},
        )


@dataclass(frozen=True)
class ThresholdFailureDetector:
    """Detect the first lower or upper threshold crossing of a scalar signal."""

    failure_type: str
    signal_key: str
    lower: Optional[float] = None
    upper: Optional[float] = None
    detector_id: str = "threshold_failure/v1"

    def __post_init__(self) -> None:
        if not str(self.failure_type).strip() or not str(self.signal_key).strip():
            raise ValueError("failure_type and signal_key must not be empty")
        if self.lower is None and self.upper is None:
            raise ValueError("at least one threshold is required")
        lower = None if self.lower is None else float(self.lower)
        upper = None if self.upper is None else float(self.upper)
        if (
            (lower is not None and not np.isfinite(lower))
            or (upper is not None and not np.isfinite(upper))
            or (
                lower is not None
                and upper is not None
                and lower >= upper
            )
        ):
            raise ValueError("thresholds must be finite with lower < upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def detect(
        self, timestamps: np.ndarray, **observations: Any
    ) -> Optional[FailureEvent]:
        times = _timestamps(timestamps)
        if self.signal_key not in observations:
            raise KeyError(
                "failure detector requires observation '{}'".format(
                    self.signal_key
                )
            )
        values = np.asarray(observations[self.signal_key], dtype=float).reshape(
            -1
        )
        if values.shape != times.shape or not np.all(np.isfinite(values)):
            raise ValueError("threshold signal must be finite and time-aligned")
        crossed = np.zeros(times.shape, dtype=bool)
        if self.lower is not None:
            crossed |= values <= self.lower
        if self.upper is not None:
            crossed |= values >= self.upper
        indexes = np.flatnonzero(crossed)
        if not indexes.size:
            return None
        index = int(indexes[0])
        return FailureEvent(
            failure_type=self.failure_type,
            stamp=float(times[index]),
            detector_id=self.detector_id,
            metadata={
                "sample_index": index,
                "signal_key": self.signal_key,
                "signal_value": float(values[index]),
                "lower": self.lower,
                "upper": self.upper,
            },
        )


@dataclass(frozen=True)
class CompositeFailureDetector:
    """Select the earliest event emitted by several independent detectors."""

    detectors: Tuple[FailureDetector, ...]
    detector_id: str = "composite_failure/v1"

    def __post_init__(self) -> None:
        detectors = tuple(self.detectors)
        if not detectors or any(
            not callable(getattr(item, "detect", None)) for item in detectors
        ):
            raise ValueError("detectors must contain FailureDetector values")
        object.__setattr__(self, "detectors", detectors)

    def detect(
        self, timestamps: np.ndarray, **observations: Any
    ) -> Optional[FailureEvent]:
        events = tuple(
            event
            for event in (
                detector.detect(timestamps, **observations)
                for detector in self.detectors
            )
            if event is not None
        )
        if not events:
            return None
        return min(
            events,
            key=lambda item: (
                item.stamp,
                item.failure_type,
                item.detector_id,
            ),
        )


@dataclass(frozen=True)
class RolloutSafetyFailureDetector:
    """Detect ground, excessive tilt, or sustained actuator saturation."""

    maximum_tilt_rad: float = float(np.deg2rad(60.0))
    minimum_height_m: float = -0.05
    detector_id: str = "grape_rollout_safety_failure/v1"

    def __post_init__(self) -> None:
        tilt = float(self.maximum_tilt_rad)
        height = float(self.minimum_height_m)
        if (
            not np.isfinite(tilt)
            or not 0.0 < tilt < np.pi
            or not np.isfinite(height)
        ):
            raise ValueError("rollout failure thresholds are invalid")
        object.__setattr__(self, "maximum_tilt_rad", tilt)
        object.__setattr__(self, "minimum_height_m", height)

    def detect(self, rollout: Any) -> Optional[FailureEvent]:
        from scipy.spatial.transform import Rotation

        times = _timestamps(getattr(rollout, "integration_timestamps"))
        positions = np.asarray(getattr(rollout, "positions"), dtype=float)
        orientations = np.asarray(
            getattr(rollout, "orientations_xyzw"), dtype=float
        )
        if (
            positions.shape != (times.size, 3)
            or orientations.shape != (times.size, 4)
        ):
            raise ValueError("rollout pose arrays are not time aligned")
        body_z = Rotation.from_quat(orientations).apply(
            np.tile(np.array([0.0, 0.0, 1.0]), (times.size, 1))
        )
        tilt = np.arccos(np.clip(body_z[:, 2], -1.0, 1.0))
        candidates = []
        ground = np.flatnonzero(positions[:, 2] <= self.minimum_height_m)
        if ground.size:
            index = int(ground[0])
            candidates.append(
                FailureEvent(
                    failure_type="ground_contact",
                    stamp=float(times[index]),
                    detector_id=self.detector_id,
                    metadata={
                        "sample_index": index,
                        "height_m": float(positions[index, 2]),
                    },
                )
            )
        attitude = np.flatnonzero(tilt >= self.maximum_tilt_rad)
        if attitude.size:
            index = int(attitude[0])
            candidates.append(
                FailureEvent(
                    failure_type="attitude_failure",
                    stamp=float(times[index]),
                    detector_id=self.detector_id,
                    metadata={
                        "sample_index": index,
                        "tilt_rad": float(tilt[index]),
                    },
                )
            )
        saturation = tuple(
            item
            for item in getattr(rollout, "events", ())
            if item.get("type") == "actuator_saturation"
        )
        if saturation:
            candidates.append(
                FailureEvent(
                    failure_type="actuator_saturation",
                    stamp=float(saturation[0]["stamp"]),
                    detector_id=self.detector_id,
                    metadata={"event": dict(saturation[0])},
                )
            )
        return (
            None
            if not candidates
            else min(candidates, key=lambda item: (item.stamp, item.failure_type))
        )


@dataclass(frozen=True)
class FailureCensoring:
    """Likelihood mask that excludes observations after a failure."""

    timestamps: np.ndarray
    score_mask: np.ndarray
    failure_event: Optional[FailureEvent]
    include_failure_sample: bool

    def __post_init__(self) -> None:
        times = _timestamps(self.timestamps)
        mask = np.asarray(self.score_mask, dtype=bool).reshape(-1)
        if mask.shape != times.shape:
            raise ValueError("score_mask must align with timestamps")
        frozen_mask = np.array(mask, copy=True)
        frozen_mask.setflags(write=False)
        object.__setattr__(self, "timestamps", times)
        object.__setattr__(self, "score_mask", frozen_mask)
        if self.failure_event is not None and not isinstance(
            self.failure_event, FailureEvent
        ):
            raise TypeError("failure_event must be FailureEvent or None")
        if type(self.include_failure_sample) is not bool:
            raise TypeError("include_failure_sample must be a built-in bool")

    @property
    def censored_count(self) -> int:
        return int(self.score_mask.size - np.count_nonzero(self.score_mask))


def censor_after_failure(
    timestamps: Sequence[float],
    failure_event: Optional[FailureEvent],
    base_mask: Optional[Sequence[bool]] = None,
    include_failure_sample: bool = True,
) -> FailureCensoring:
    """Create a mask that retains pre-failure evidence and censors the rest."""

    times = _timestamps(timestamps)
    if type(include_failure_sample) is not bool:
        raise TypeError("include_failure_sample must be a built-in bool")
    mask = (
        np.ones(times.shape, dtype=bool)
        if base_mask is None
        else np.asarray(base_mask, dtype=bool).reshape(-1).copy()
    )
    if mask.shape != times.shape:
        raise ValueError("base_mask must align with timestamps")
    if failure_event is not None:
        if not isinstance(failure_event, FailureEvent):
            raise TypeError("failure_event must be FailureEvent or None")
        if include_failure_sample:
            mask &= times <= failure_event.stamp
        else:
            mask &= times < failure_event.stamp
    return FailureCensoring(
        timestamps=times,
        score_mask=mask,
        failure_event=failure_event,
        include_failure_sample=include_failure_sample,
    )


__all__ = [
    "CompositeFailureDetector",
    "FailureCensoring",
    "FailureDetector",
    "FailureEvent",
    "FirstMaskFailureDetector",
    "RolloutSafetyFailureDetector",
    "ThresholdFailureDetector",
    "censor_after_failure",
]
