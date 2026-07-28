"""Component-wise heavy-tailed likelihoods for forward rollouts."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from numbers import Integral
from threading import Condition, Lock
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.special import gammaln
from scipy.spatial.transform import Rotation, Slerp

from grape_param_estim._legacy_inference import (
    marginalize_trajectory_log_likelihood,
)
from grape_param_estim.forward.rollout import RolloutResult
from grape_param_estim.plant.parameters import EpisodeNuisance


CONTROLLER_EVENT_OBSERVATIONS_SCHEMA = (
    "grape_controller_event_observations/v1"
)
CONTROLLER_MODE_EVENT_MASK = (1 << 2) | (1 << 3) | (1 << 4)
CONTROLLER_SATURATION_EVENT_MASK = 1 << 5
_UINT32_MAX = (1 << 32) - 1


def _readonly(values: Any, shape_suffix: Tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (
        array.ndim < len(shape_suffix)
        or array.shape[-len(shape_suffix) :] != shape_suffix
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(
            "{} must be finite with trailing shape {}".format(name, shape_suffix)
        )
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy


def _event_bitmasks(values: Any) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in ("i", "u"):
        raise ValueError(
            "controller event bitmasks must be a non-empty integer vector"
        )
    converted = raw.astype(np.int64, copy=False)
    if np.any(converted < 0) or np.any(converted > _UINT32_MAX):
        raise ValueError("controller event bitmasks must be uint32 values")
    copy = np.array(converted, dtype=np.uint32, copy=True)
    copy.setflags(write=False)
    return copy


def _boolean_vector(values: Any, count: int, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (count,) or raw.dtype.kind != "b":
        raise ValueError(
            "{} must be a boolean vector aligned with event timestamps".format(
                name
            )
        )
    copy = np.array(raw, dtype=bool, copy=True)
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True)
class ControllerEventObservations:
    """Complete factual controller-event frames on a sparse scoring grid.

    ``event_bitmasks`` uses the exact replay-core uint32 event contract.
    Mode-transition bits and the saturation bit are scored as separate
    channels so they are not double counted in the remaining event mask.
    A zero bitmask is meaningful negative evidence, not missing data.
    """

    timestamps: np.ndarray
    event_bitmasks: np.ndarray
    saturated: Optional[np.ndarray] = None
    mode_event_mask: int = CONTROLLER_MODE_EVENT_MASK
    saturation_event_mask: int = CONTROLLER_SATURATION_EVENT_MASK
    schema: str = CONTROLLER_EVENT_OBSERVATIONS_SCHEMA

    def __post_init__(self) -> None:
        times = np.asarray(self.timestamps, dtype=float).reshape(-1)
        if (
            times.size == 0
            or not np.all(np.isfinite(times))
            or (times.size > 1 and np.any(np.diff(times) <= 0.0))
        ):
            raise ValueError(
                "controller event timestamps must be finite and increasing"
            )
        time_copy = np.array(times, copy=True)
        time_copy.setflags(write=False)
        bitmasks = _event_bitmasks(self.event_bitmasks)
        if bitmasks.shape != time_copy.shape:
            raise ValueError(
                "controller event bitmasks must align with timestamps"
            )
        if (
            isinstance(self.mode_event_mask, bool)
            or not isinstance(self.mode_event_mask, Integral)
            or isinstance(self.saturation_event_mask, bool)
            or not isinstance(self.saturation_event_mask, Integral)
        ):
            raise ValueError(
                "mode and saturation event masks must be uint32 integers"
            )
        mode_mask = int(self.mode_event_mask)
        saturation_mask = int(self.saturation_event_mask)
        if (
            mode_mask <= 0
            or saturation_mask <= 0
            or mode_mask > _UINT32_MAX
            or saturation_mask > _UINT32_MAX
            or mode_mask & saturation_mask
            or saturation_mask & (saturation_mask - 1)
        ):
            raise ValueError(
                "mode and saturation event masks must be disjoint uint32 "
                "masks and saturation must select one bit"
            )
        derived_saturated = np.asarray(
            (bitmasks & np.uint32(saturation_mask)) != 0,
            dtype=bool,
        )
        saturated = self.saturated
        if saturated is None:
            saturated_values = np.array(derived_saturated, copy=True)
            saturated_values.setflags(write=False)
        else:
            saturated_values = _boolean_vector(
                saturated, times.size, "saturated"
            )
            if not np.array_equal(saturated_values, derived_saturated):
                raise ValueError(
                    "saturation observations disagree with event bitmasks"
                )
        schema = str(self.schema).strip()
        if schema != CONTROLLER_EVENT_OBSERVATIONS_SCHEMA:
            raise ValueError("unsupported controller event observation schema")
        object.__setattr__(self, "timestamps", time_copy)
        object.__setattr__(self, "event_bitmasks", bitmasks)
        object.__setattr__(self, "saturated", saturated_values)
        object.__setattr__(self, "mode_event_mask", mode_mask)
        object.__setattr__(
            self, "saturation_event_mask", saturation_mask
        )
        object.__setattr__(self, "schema", schema)

    @property
    def mode_event_bitmasks(self) -> np.ndarray:
        values = self.event_bitmasks & np.uint32(self.mode_event_mask)
        values.setflags(write=False)
        return values

    @property
    def other_event_bitmasks(self) -> np.ndarray:
        excluded = self.mode_event_mask | self.saturation_event_mask
        values = self.event_bitmasks & np.uint32(_UINT32_MAX ^ excluded)
        values.setflags(write=False)
        return values


@dataclass(frozen=True)
class ObservationDataset:
    episode_id: str
    role: str
    timestamps: np.ndarray
    position_world: np.ndarray
    orientation_xyzw: np.ndarray
    velocity_world: Optional[np.ndarray] = None
    specific_force_body: Optional[np.ndarray] = None
    angular_velocity_body: Optional[np.ndarray] = None
    failure_time: Optional[float] = None
    failure_type: Optional[str] = None
    source_bag_sha256: str = ""
    normalized_episode_sha256: str = ""
    event_observations: Optional[ControllerEventObservations] = None

    def __post_init__(self) -> None:
        if self.role not in (
            "inference_failure",
            "validation_failure",
            "validation_success",
        ):
            raise ValueError("unsupported episode role")
        times = np.asarray(self.timestamps, dtype=float).reshape(-1)
        if (
            times.size < 2
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("observation timestamps must be increasing")
        count = times.size
        position = _readonly(self.position_world, (3,), "position_world")
        orientation = _readonly(
            self.orientation_xyzw, (4,), "orientation_xyzw"
        )
        if position.shape != (count, 3) or orientation.shape != (count, 4):
            raise ValueError("pose observations must align with timestamps")
        object.__setattr__(self, "timestamps", times)
        object.__setattr__(self, "position_world", position)
        object.__setattr__(self, "orientation_xyzw", orientation)
        for name in (
            "velocity_world",
            "specific_force_body",
            "angular_velocity_body",
        ):
            value = getattr(self, name)
            if value is not None:
                matrix = _readonly(value, (3,), name)
                if matrix.shape != (count, 3):
                    raise ValueError("{} must align with timestamps".format(name))
                object.__setattr__(self, name, matrix)
        failure_time = self.failure_time
        if failure_time is not None:
            failure_time = float(failure_time)
            if not np.isfinite(failure_time):
                raise ValueError("failure_time must be finite")
            object.__setattr__(self, "failure_time", failure_time)
        event_observations = self.event_observations
        if (
            event_observations is not None
            and not isinstance(
                event_observations, ControllerEventObservations
            )
        ):
            raise TypeError(
                "event_observations must be ControllerEventObservations"
            )


@dataclass(frozen=True)
class LikelihoodConfig:
    position_sigma: float = 0.03
    orientation_sigma_rad: float = np.deg2rad(2.0)
    velocity_sigma: float = 0.10
    imu_sigma: float = 0.30
    angular_velocity_sigma: float = 0.05
    student_t_degrees_of_freedom: float = 5.0
    failure_time_sigma: float = 0.25
    missing_failure_log_penalty: float = -20.0
    unexpected_failure_log_penalty: float = -20.0
    saturation_event_error_probability: float = 0.01
    mode_event_error_probability: float = 0.01
    other_event_error_probability: float = 0.01
    require_controller_event_evidence: bool = False
    censor_after_failure: bool = True
    likelihood_id: str = "forward_student_t_components_v2"

    def __post_init__(self) -> None:
        values = (
            self.position_sigma,
            self.orientation_sigma_rad,
            self.velocity_sigma,
            self.imu_sigma,
            self.angular_velocity_sigma,
            self.student_t_degrees_of_freedom,
            self.failure_time_sigma,
        )
        if any(not np.isfinite(item) or item <= 0.0 for item in values):
            raise ValueError(
                "likelihood scales and degrees of freedom must be positive"
            )
        event_probabilities = (
            self.saturation_event_error_probability,
            self.mode_event_error_probability,
            self.other_event_error_probability,
        )
        if any(
            not np.isfinite(item) or not 0.0 < item < 0.5
            for item in event_probabilities
        ):
            raise ValueError(
                "event error probabilities must lie strictly between 0 and 0.5"
            )
        if (
            type(self.censor_after_failure) is not bool
            or type(self.require_controller_event_evidence) is not bool
        ):
            raise TypeError("likelihood policy flags must be built-in bools")


@dataclass(frozen=True)
class LikelihoodComponents:
    episode_id: str
    pose: float
    orientation: float
    velocity: float
    imu: float
    angular_velocity: float
    command: float
    failure_event: float
    saturation_mode_event: float
    scored_sample_count: int
    censored_sample_count: int
    total: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    trajectory_total: Optional[float] = None
    event_total: Optional[float] = None
    scored_event_sample_count: int = 0
    censored_event_sample_count: int = 0
    controller_event_evidence_status: str = "legacy_unspecified"
    controller_event_evidence_schema: str = (
        CONTROLLER_EVENT_OBSERVATIONS_SCHEMA
    )

    def __post_init__(self) -> None:
        component_values = (
            self.pose,
            self.orientation,
            self.velocity,
            self.imu,
            self.angular_velocity,
            self.command,
            self.failure_event,
            self.saturation_mode_event,
            self.total,
        )
        if any(not np.isfinite(item) for item in component_values):
            raise ValueError("likelihood components must be finite")
        for name in (
            "scored_sample_count",
            "censored_sample_count",
            "scored_event_sample_count",
            "censored_event_sample_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError("{} must be an integer".format(name))
            if int(value) < 0:
                raise ValueError("{} must be non-negative".format(name))
            object.__setattr__(self, name, int(value))
        trajectory_total = float(
            self.pose
            + self.orientation
            + self.velocity
            + self.imu
            + self.angular_velocity
            + self.command
        )
        event_total = float(
            self.failure_event + self.saturation_mode_event
        )
        supplied_trajectory = self.trajectory_total
        supplied_event = self.event_total
        if supplied_trajectory is not None and not np.isclose(
            float(supplied_trajectory),
            trajectory_total,
            rtol=1.0e-10,
            atol=1.0e-10,
        ):
            raise ValueError("trajectory likelihood total is inconsistent")
        if supplied_event is not None and not np.isclose(
            float(supplied_event),
            event_total,
            rtol=1.0e-10,
            atol=1.0e-10,
        ):
            raise ValueError("event likelihood total is inconsistent")
        expected = trajectory_total + event_total
        if not np.isclose(expected, self.total, rtol=1.0e-10, atol=1.0e-10):
            raise ValueError("likelihood component total is inconsistent")
        object.__setattr__(self, "trajectory_total", trajectory_total)
        object.__setattr__(self, "event_total", event_total)
        status = str(self.controller_event_evidence_status)
        if status not in (
            "legacy_unspecified",
            "not_scored_no_evidence",
            "scored",
        ):
            raise ValueError(
                "unsupported controller event evidence status"
            )
        if (
            str(self.controller_event_evidence_schema)
            != CONTROLLER_EVENT_OBSERVATIONS_SCHEMA
        ):
            raise ValueError(
                "unsupported controller event evidence schema"
            )
        object.__setattr__(
            self, "controller_event_evidence_status", status
        )
        object.__setattr__(
            self,
            "controller_event_evidence_schema",
            CONTROLLER_EVENT_OBSERVATIONS_SCHEMA,
        )
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )


def _student_t_sum(residual: np.ndarray, sigma: float, degrees: float) -> float:
    values = np.asarray(residual, dtype=float) / float(sigma)
    constant = (
        gammaln(0.5 * (degrees + 1.0))
        - gammaln(0.5 * degrees)
        - 0.5 * np.log(degrees * np.pi)
        - np.log(float(sigma))
    )
    return float(
        np.sum(constant - 0.5 * (degrees + 1.0) * np.log1p(values * values / degrees))
    )


def _predictions_at(
    rollout: RolloutResult, query: np.ndarray
) -> Mapping[str, np.ndarray]:
    times = rollout.integration_timestamps
    positions = rollout.positions
    velocities = rollout.velocities
    orientations = rollout.orientations_xyzw
    specific = np.stack(
        [item.specific_force_body for item in rollout.predicted_observations]
    )
    omega = np.stack(
        [item.angular_velocity_body for item in rollout.predicted_observations]
    )
    columns = lambda values: np.column_stack(
        [np.interp(query, times, values[:, index]) for index in range(values.shape[1])]
    )
    if query[0] < times[0] - 1.0e-12 or query[-1] > times[-1] + 1.0e-12:
        raise ValueError("likelihood query lies outside rollout support")
    return {
        "position": columns(positions),
        "velocity": columns(velocities),
        "orientation": Slerp(
            times, Rotation.from_quat(orientations)
        )(query).as_quat(),
        "specific_force": columns(specific),
        "angular_velocity": columns(omega),
    }


def _event_frame_log_likelihood(
    observed: np.ndarray,
    predicted: np.ndarray,
    error_probability: float,
) -> Tuple[float, int, int]:
    matches = np.asarray(observed) == np.asarray(predicted)
    match_count = int(np.count_nonzero(matches))
    mismatch_count = int(matches.size - match_count)
    score = (
        match_count * np.log1p(-float(error_probability))
        + mismatch_count * np.log(float(error_probability))
    )
    return float(score), match_count, mismatch_count


def _command_event_bitmask(
    command: Any, saturation_event_mask: int
) -> int:
    raw_events = getattr(command, "events", None)
    if raw_events is None:
        raise ValueError(
            "controller command lacks event evidence"
        )
    try:
        event_codes = tuple(raw_events)
    except TypeError as exc:
        raise ValueError(
            "controller command events must be an iterable of event bits"
        ) from exc
    bitmask = 0
    for raw_code in event_codes:
        if (
            isinstance(raw_code, bool)
            or not isinstance(raw_code, Integral)
        ):
            raise ValueError(
                "controller command event codes must be integers"
            )
        code = int(raw_code)
        if (
            code <= 0
            or code > _UINT32_MAX
            or code & (code - 1)
        ):
            raise ValueError(
                "controller command events must be individual uint32 bits"
            )
        bitmask |= code
    saturated = getattr(command, "saturated", None)
    if type(saturated) is not bool:
        raise ValueError(
            "controller command lacks a factual saturation flag"
        )
    if saturated != bool(bitmask & saturation_event_mask):
        raise ValueError(
            "predicted saturation flag and controller event bitmask disagree"
        )
    return bitmask


def _predicted_controller_event_bitmasks(
    rollout: RolloutResult,
    query: np.ndarray,
    saturation_event_mask: int,
) -> np.ndarray:
    commands = tuple(rollout.commands)
    if not commands:
        raise ValueError(
            "controller event observations require controller commands"
        )
    stamps = np.asarray(
        [float(getattr(item, "stamp")) for item in commands],
        dtype=float,
    )
    if (
        not np.all(np.isfinite(stamps))
        or np.any(np.diff(stamps) <= 0.0)
    ):
        raise ValueError(
            "controller command event timestamps must be finite, unique, "
            "and increasing"
        )
    indexes = np.searchsorted(stamps, query, side="left")
    if (
        np.any(indexes >= stamps.size)
        or np.any(
            np.abs(stamps[np.minimum(indexes, stamps.size - 1)] - query)
            > 1.0e-9
        )
    ):
        raise ValueError(
            "controller event evidence lacks exact predicted tick coverage"
        )
    predicted = np.asarray(
        [
            _command_event_bitmask(
                commands[int(index)], saturation_event_mask
            )
            for index in indexes
        ],
        dtype=np.uint32,
    )
    predicted.setflags(write=False)
    return predicted


def _controller_event_likelihood(
    rollout: RolloutResult,
    observations: ObservationDataset,
    config: LikelihoodConfig,
) -> Tuple[float, int, int, Mapping[str, Any]]:
    evidence = observations.event_observations
    if evidence is None:
        if config.require_controller_event_evidence:
            raise ValueError(
                "controller event likelihood requires factual event evidence"
            )
        return (
            0.0,
            0,
            0,
            MappingProxyType(
                {
                    "status": "not_scored_no_evidence",
                    "schema": CONTROLLER_EVENT_OBSERVATIONS_SCHEMA,
                }
            ),
        )
    event_mask = np.ones(evidence.timestamps.size, dtype=bool)
    if (
        config.censor_after_failure
        and observations.failure_time is not None
    ):
        event_mask &= (
            evidence.timestamps <= observations.failure_time + 1.0e-12
        )
    query = evidence.timestamps[event_mask]
    if query.size == 0:
        raise ValueError(
            "failure censoring removed every controller event observation"
        )
    predicted = _predicted_controller_event_bitmasks(
        rollout,
        query,
        evidence.saturation_event_mask,
    )
    observed = evidence.event_bitmasks[event_mask]
    observed_saturated = evidence.saturated[event_mask]
    predicted_saturated = (
        predicted & np.uint32(evidence.saturation_event_mask)
    ) != 0
    observed_mode = observed & np.uint32(evidence.mode_event_mask)
    predicted_mode = predicted & np.uint32(evidence.mode_event_mask)
    excluded = evidence.mode_event_mask | evidence.saturation_event_mask
    other_mask = np.uint32(_UINT32_MAX ^ excluded)
    observed_other = observed & other_mask
    predicted_other = predicted & other_mask

    saturation_score, saturation_matches, saturation_mismatches = (
        _event_frame_log_likelihood(
            observed_saturated,
            predicted_saturated,
            config.saturation_event_error_probability,
        )
    )
    mode_score, mode_matches, mode_mismatches = (
        _event_frame_log_likelihood(
            observed_mode,
            predicted_mode,
            config.mode_event_error_probability,
        )
    )
    other_score, other_matches, other_mismatches = (
        _event_frame_log_likelihood(
            observed_other,
            predicted_other,
            config.other_event_error_probability,
        )
    )
    score = float(saturation_score + mode_score + other_score)
    scored = int(np.count_nonzero(event_mask))
    censored = int(event_mask.size - scored)
    diagnostics = MappingProxyType(
        {
            "status": "scored",
            "schema": evidence.schema,
            "scored_frame_count": scored,
            "censored_frame_count": censored,
            "saturation": {
                "log_likelihood": saturation_score,
                "match_count": saturation_matches,
                "mismatch_count": saturation_mismatches,
            },
            "mode_event": {
                "log_likelihood": mode_score,
                "match_count": mode_matches,
                "mismatch_count": mode_mismatches,
            },
            "other_event": {
                "log_likelihood": other_score,
                "match_count": other_matches,
                "mismatch_count": other_mismatches,
            },
        }
    )
    return score, scored, censored, diagnostics


class EpisodeLikelihood:
    def __init__(
        self,
        config: LikelihoodConfig = LikelihoodConfig(),
        failure_detector: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.failure_detector = failure_detector

    def evaluate(
        self, rollout: RolloutResult, observations: ObservationDataset
    ) -> LikelihoodComponents:
        if not isinstance(rollout, RolloutResult):
            raise TypeError("rollout must be RolloutResult")
        if not isinstance(observations, ObservationDataset):
            raise TypeError("observations must be ObservationDataset")
        times = observations.timestamps
        mask = np.ones(times.size, dtype=bool)
        if (
            self.config.censor_after_failure
            and observations.failure_time is not None
        ):
            mask &= times <= observations.failure_time + 1.0e-12
        query = times[mask]
        if query.size == 0:
            raise ValueError("failure censoring removed every observation")
        predicted = _predictions_at(rollout, query)
        degrees = self.config.student_t_degrees_of_freedom
        pose = _student_t_sum(
            observations.position_world[mask] - predicted["position"],
            self.config.position_sigma,
            degrees,
        )
        relative = (
            Rotation.from_quat(predicted["orientation"]).inv()
            * Rotation.from_quat(observations.orientation_xyzw[mask])
        )
        orientation = _student_t_sum(
            relative.as_rotvec(),
            self.config.orientation_sigma_rad,
            degrees,
        )
        velocity = 0.0
        if observations.velocity_world is not None:
            velocity = _student_t_sum(
                observations.velocity_world[mask] - predicted["velocity"],
                self.config.velocity_sigma,
                degrees,
            )
        imu = 0.0
        if observations.specific_force_body is not None:
            imu = _student_t_sum(
                observations.specific_force_body[mask]
                - predicted["specific_force"],
                self.config.imu_sigma,
                degrees,
            )
        angular_velocity = 0.0
        if observations.angular_velocity_body is not None:
            angular_velocity = _student_t_sum(
                observations.angular_velocity_body[mask]
                - predicted["angular_velocity"],
                self.config.angular_velocity_sigma,
                degrees,
            )

        # Command residual belongs only to factual controller replay, never a
        # plant-identification rollout.
        command = 0.0
        event_score = 0.0
        predicted_failure = None
        if self.failure_detector is not None:
            predicted_failure = self.failure_detector.detect(rollout)
        if observations.failure_time is not None:
            if predicted_failure is None:
                event_score = float(self.config.missing_failure_log_penalty)
            else:
                event_score = _student_t_sum(
                    np.asarray(
                        [predicted_failure.failure_time - observations.failure_time]
                    ),
                    self.config.failure_time_sigma,
                    degrees,
                )
                if (
                    observations.failure_type
                    and predicted_failure.failure_type
                    != observations.failure_type
                ):
                    event_score += float(
                        self.config.missing_failure_log_penalty
                    )
        elif predicted_failure is not None:
            event_score = float(self.config.unexpected_failure_log_penalty)
        (
            saturation_mode,
            scored_event_count,
            censored_event_count,
            controller_event_diagnostics,
        ) = _controller_event_likelihood(
            rollout, observations, self.config
        )
        trajectory_total = float(
            pose
            + orientation
            + velocity
            + imu
            + angular_velocity
            + command
        )
        event_total = float(event_score + saturation_mode)
        total = (
            trajectory_total
            + event_total
        )
        return LikelihoodComponents(
            episode_id=observations.episode_id,
            pose=pose,
            orientation=orientation,
            velocity=velocity,
            imu=imu,
            angular_velocity=angular_velocity,
            command=command,
            failure_event=event_score,
            saturation_mode_event=saturation_mode,
            scored_sample_count=int(np.count_nonzero(mask)),
            censored_sample_count=int(mask.size - np.count_nonzero(mask)),
            total=total,
            diagnostics={
                "likelihood_id": self.config.likelihood_id,
                "trajectory_total": trajectory_total,
                "event_total": event_total,
                "controller_event_likelihood": (
                    controller_event_diagnostics
                ),
                "predicted_failure": (
                    None
                    if predicted_failure is None
                    else {
                        "type": predicted_failure.failure_type,
                        "time": predicted_failure.failure_time,
                    }
                ),
            },
            trajectory_total=trajectory_total,
            event_total=event_total,
            scored_event_sample_count=scored_event_count,
            censored_event_sample_count=censored_event_count,
            controller_event_evidence_status=(
                controller_event_diagnostics["status"]
            ),
            controller_event_evidence_schema=(
                controller_event_diagnostics["schema"]
            ),
        )


class MultipleEpisodeLikelihood:
    """Shared static parameters with nuisance-state marginalization per episode."""

    def __init__(
        self,
        observations: Sequence[ObservationDataset],
        nuisance_samples: Mapping[str, Sequence[EpisodeNuisance]],
        rollout_function: Callable[
            [np.ndarray, ObservationDataset, EpisodeNuisance], RolloutResult
        ],
        episode_likelihood: EpisodeLikelihood,
        worker_count: int = 1,
    ) -> None:
        self.observations = tuple(observations)
        if not self.observations:
            raise ValueError("at least one inference episode is required")
        if any(item.role != "inference_failure" for item in self.observations):
            raise ValueError(
                "success and held-out failure episodes may not update the posterior"
            )
        self.nuisance_samples = {
            str(key): tuple(value) for key, value in nuisance_samples.items()
        }
        self.rollout_function = rollout_function
        self.episode_likelihood = episode_likelihood
        workers = int(worker_count)
        if workers < 1:
            raise ValueError("worker_count must be positive")
        self.worker_count = workers
        self._components_lock = Lock()
        self._cache_condition = Condition(Lock())
        self._active_rollout_keys = set()
        self.last_components: Dict[Tuple[int, str, int], LikelihoodComponents] = {}

    def __call__(self, particles: np.ndarray) -> np.ndarray:
        values = np.asarray(particles, dtype=float)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError("particles must be a finite matrix")
        result = np.zeros(values.shape[0])
        with self._components_lock:
            self.last_components = {}
        if values.shape[0] == 0:
            return result

        jobs = []
        conditionals = {}
        episode_weights = {}
        for particle_index in range(values.shape[0]):
            for episode_index, episode in enumerate(
                self.observations
            ):
                samples = self.nuisance_samples.get(episode.episode_id, ())
                if not samples:
                    raise ValueError(
                        "episode {} lacks initial-state samples".format(
                            episode.episode_id
                        )
                    )
                conditional = np.empty((1, len(samples)))
                conditionals[(particle_index, episode_index)] = (
                    conditional
                )
                weights = np.asarray(
                    [item.weight for item in samples], dtype=float
                )
                weights /= np.sum(weights)
                episode_weights[episode_index] = weights
                jobs.extend(
                    (
                        particle_index,
                        episode_index,
                        sample_index,
                    )
                    for sample_index in range(len(samples))
                )

        def cache_equivalence(job):
            particle_index, episode_index, sample_index = job
            episode = self.observations[episode_index]
            nuisance = self.nuisance_samples[
                episode.episode_id
            ][sample_index]
            return (
                values[particle_index].tobytes(),
                str(episode.episode_id),
                str(
                    getattr(episode, "source_bag_sha256", "")
                ),
                str(
                    getattr(
                        episode,
                        "normalized_episode_sha256",
                        "",
                    )
                ),
                str(
                    getattr(
                        nuisance,
                        "state_sample_id",
                        sample_index,
                    )
                ),
            )

        def evaluate(job):
            particle_index, episode_index, sample_index = job
            episode = self.observations[episode_index]
            nuisance = self.nuisance_samples[
                episode.episode_id
            ][sample_index]
            rollout_key = cache_equivalence(job)
            with self._cache_condition:
                while rollout_key in self._active_rollout_keys:
                    self._cache_condition.wait()
                self._active_rollout_keys.add(rollout_key)
            try:
                rollout = self.rollout_function(
                    values[particle_index], episode, nuisance
                )
            finally:
                with self._cache_condition:
                    self._active_rollout_keys.remove(rollout_key)
                    self._cache_condition.notify_all()
            components = self.episode_likelihood.evaluate(
                rollout, episode
            )
            return components

        if self.worker_count == 1:
            components_in_order = tuple(map(evaluate, jobs))
        else:
            # Jobs that resolve to the same rollout-cache key stay on one
            # worker and execute in their original order.  This prevents
            # duplicate resampled particles from racing the cache miss/put
            # sequence while unrelated episode/particle jobs run in parallel.
            grouped_jobs = {}
            for job in jobs:
                grouped_jobs.setdefault(
                    cache_equivalence(job), []
                ).append(job)

            def evaluate_group(group):
                return tuple(
                    (job, evaluate(job)) for job in group
                )

            worker_count = min(
                self.worker_count, len(grouped_jobs)
            )
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="grape-rollout",
            ) as executor:
                grouped_results = tuple(
                    executor.map(
                        evaluate_group,
                        grouped_jobs.values(),
                    )
                )
            components_by_descriptor = {
                job: components
                for group in grouped_results
                for job, components in group
            }
            components_in_order = tuple(
                components_by_descriptor[job] for job in jobs
            )

        components_by_job = {}
        for job, components in zip(jobs, components_in_order):
            particle_index, episode_index, sample_index = job
            episode = self.observations[episode_index]
            components_by_job[
                (
                    particle_index,
                    episode.episode_id,
                    sample_index,
                )
            ] = components
            conditionals[
                (particle_index, episode_index)
            ][0, sample_index] = components.total

        for particle_index in range(values.shape[0]):
            for episode_index, _ in enumerate(
                self.observations
            ):
                result[particle_index] += float(
                    marginalize_trajectory_log_likelihood(
                        conditionals[
                            (particle_index, episode_index)
                        ],
                        episode_weights[episode_index],
                    )[0]
                )
        with self._components_lock:
            self.last_components = components_by_job
        return result


__all__ = [
    "CONTROLLER_EVENT_OBSERVATIONS_SCHEMA",
    "CONTROLLER_MODE_EVENT_MASK",
    "CONTROLLER_SATURATION_EVENT_MASK",
    "ControllerEventObservations",
    "EpisodeLikelihood",
    "LikelihoodComponents",
    "LikelihoodConfig",
    "MultipleEpisodeLikelihood",
    "ObservationDataset",
]
