"""Closed-loop posterior counterfactual evaluation and support labelling."""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta as beta_distribution

from .controller_replay import (
    AXIS_COUNT,
    ControllerParameters,
    ControllerReplay,
    ReplayRequest,
)
from .effective_response import (
    EffectiveResponseParameters,
    EffectiveResponsePosterior,
    LowDimensionalEffectiveResponse,
    ResponseState,
)
from .episode import stable_hash


SUPPORTED = "SUPPORTED"
EXTRAPOLATIVE = "EXTRAPOLATIVE"
UNSUPPORTED = "UNSUPPORTED"
_SUPPORT_LABELS = (SUPPORTED, EXTRAPOLATIVE, UNSUPPORTED)


def _vector(values, name, positive=False):
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(AXIS_COUNT, float(array))
    if array.shape != (AXIS_COUNT,) or not np.all(np.isfinite(array)):
        raise ValueError("{} must be a finite scalar or six-vector".format(name))
    if positive and np.any(array <= 0.0):
        raise ValueError("{} must be strictly positive".format(name))
    output = np.array(array, copy=True)
    output.setflags(write=False)
    return output


def _matrix(values, rows, name):
    array = np.asarray(values, dtype=float)
    if array.shape != (rows, AXIS_COUNT) or not np.all(np.isfinite(array)):
        raise ValueError("{} must have finite shape ({}, 6)".format(name, rows))
    output = np.array(array, copy=True)
    output.setflags(write=False)
    return output


def _weighted_mean(values, weights):
    return np.tensordot(weights, values, axes=(0, 0))


@dataclass(frozen=True)
class TargetTrajectory:
    timestamps: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    def __post_init__(self):
        times = np.asarray(self.timestamps, dtype=float).reshape(-1)
        if (
            times.size < 2
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("target timestamps must be finite and increasing")
        count = times.size
        times_copy = np.array(times, copy=True)
        times_copy.setflags(write=False)
        object.__setattr__(self, "timestamps", times_copy)
        for name in ("position", "velocity", "acceleration"):
            object.__setattr__(
                self, name, _matrix(getattr(self, name), count, name)
            )


@dataclass(frozen=True)
class TargetTube:
    position_tolerance: np.ndarray
    velocity_tolerance: np.ndarray
    evaluation_start_offset_s: float = 0.0
    maximum_continuous_saturation_s: float = float("inf")
    minimum_height_m: Optional[float] = None
    maximum_tilt_rad: Optional[float] = None
    absolute_velocity_limit: Optional[np.ndarray] = None
    allowed_outside_duration_s: float = 0.0

    def __post_init__(self):
        object.__setattr__(
            self,
            "position_tolerance",
            _vector(self.position_tolerance, "position_tolerance", positive=True),
        )
        object.__setattr__(
            self,
            "velocity_tolerance",
            _vector(self.velocity_tolerance, "velocity_tolerance", positive=True),
        )
        start = float(self.evaluation_start_offset_s)
        saturation = float(self.maximum_continuous_saturation_s)
        allowed = float(self.allowed_outside_duration_s)
        if not np.isfinite(start) or start < 0.0:
            raise ValueError("evaluation_start_offset_s must be finite and non-negative")
        if saturation <= 0.0 or np.isnan(saturation):
            raise ValueError("maximum_continuous_saturation_s must be positive")
        if not np.isfinite(allowed) or allowed < 0.0:
            raise ValueError("allowed_outside_duration_s must be finite and non-negative")
        if self.minimum_height_m is not None and not np.isfinite(
            float(self.minimum_height_m)
        ):
            raise ValueError("minimum_height_m must be finite")
        if self.maximum_tilt_rad is not None and (
            not np.isfinite(float(self.maximum_tilt_rad))
            or float(self.maximum_tilt_rad) <= 0.0
        ):
            raise ValueError("maximum_tilt_rad must be finite and positive")
        if self.absolute_velocity_limit is not None:
            object.__setattr__(
                self,
                "absolute_velocity_limit",
                _vector(
                    self.absolute_velocity_limit,
                    "absolute_velocity_limit",
                    positive=True,
                ),
            )
        object.__setattr__(self, "evaluation_start_offset_s", start)
        object.__setattr__(self, "maximum_continuous_saturation_s", saturation)
        object.__setattr__(self, "allowed_outside_duration_s", allowed)


@dataclass(frozen=True)
class TubeEvaluation:
    success: bool
    violations: Tuple[str, ...]
    diagnostic_exceedances: Tuple[str, ...]
    outside_duration_s: float
    maximum_continuous_saturation_s: float
    maximum_position_ratio: float
    maximum_velocity_ratio: float


def _maximum_true_duration(timestamps, mask):
    times = np.asarray(timestamps, dtype=float)
    active = np.asarray(mask, dtype=bool)
    if times.size != active.size:
        raise ValueError("duration timestamps and mask must align")
    maximum = current = 0.0
    for index in range(times.size - 1):
        if active[index]:
            current += float(times[index + 1] - times[index])
            maximum = max(maximum, current)
        else:
            current = 0.0
    return maximum


def evaluate_target_tube(
    target: TargetTrajectory,
    tube: TargetTube,
    position: np.ndarray,
    velocity: np.ndarray,
    saturation: np.ndarray,
) -> TubeEvaluation:
    count = target.timestamps.size
    positions = _matrix(position, count, "position")
    velocities = _matrix(velocity, count, "velocity")
    saturated = np.asarray(saturation, dtype=bool)
    if saturated.shape != (count, AXIS_COUNT):
        raise ValueError("saturation must have shape (N, 6)")
    evaluation = (
        target.timestamps
        >= target.timestamps[0] + tube.evaluation_start_offset_s
    )
    if not np.any(evaluation):
        raise ValueError("target tube starts after the trajectory horizon")
    position_error = positions - target.position
    # Generalized attitude coordinates are local rotation-vector-like values;
    # wrap them to avoid a representational 2*pi error.
    position_error[:, 3:] = (
        position_error[:, 3:] + np.pi
    ) % (2.0 * np.pi) - np.pi
    velocity_error = velocities - target.velocity
    position_ratio = np.max(
        np.abs(position_error[evaluation]) / tube.position_tolerance
    )
    velocity_ratio = np.max(
        np.abs(velocity_error[evaluation]) / tube.velocity_tolerance
    )
    outside = np.zeros(count, dtype=bool)
    outside[evaluation] = (
        np.any(
            np.abs(position_error[evaluation])
            > tube.position_tolerance,
            axis=1,
        )
        | np.any(
            np.abs(velocity_error[evaluation])
            > tube.velocity_tolerance,
            axis=1,
        )
    )
    outside_duration = float(
        np.sum(
            np.diff(target.timestamps)
            * outside[:-1]
        )
    )
    saturation_any = np.any(saturated, axis=1) & evaluation
    maximum_saturation = _maximum_true_duration(
        target.timestamps, saturation_any
    )
    diagnostics = []
    if position_ratio > 1.0:
        diagnostics.append("position_tube")
    if velocity_ratio > 1.0:
        diagnostics.append("velocity_tube")
    violations = []
    # With zero allowance the pointwise tube is strict.  With a non-zero
    # allowance, pointwise excursions remain diagnostics and only their total
    # duration determines success.
    if tube.allowed_outside_duration_s == 0.0:
        violations.extend(diagnostics)
    if outside_duration > tube.allowed_outside_duration_s:
        violations.append("outside_duration")
    if maximum_saturation > tube.maximum_continuous_saturation_s:
        violations.append("saturation_duration")
    if tube.minimum_height_m is not None and np.any(
        positions[evaluation, 2] < float(tube.minimum_height_m)
    ):
        violations.append("ground_or_contact")
    if tube.maximum_tilt_rad is not None and np.any(
        np.linalg.norm(positions[evaluation, 3:5], axis=1)
        > float(tube.maximum_tilt_rad)
    ):
        violations.append("tilt_safety")
    if tube.absolute_velocity_limit is not None and np.any(
        np.abs(velocities[evaluation]) > tube.absolute_velocity_limit
    ):
        violations.append("velocity_safety")
    violations = tuple(dict.fromkeys(violations))
    return TubeEvaluation(
        success=not violations,
        violations=violations,
        diagnostic_exceedances=tuple(diagnostics),
        outside_duration_s=outside_duration,
        maximum_continuous_saturation_s=maximum_saturation,
        maximum_position_ratio=float(position_ratio),
        maximum_velocity_ratio=float(velocity_ratio),
    )


@dataclass(frozen=True)
class CounterfactualCandidate:
    candidate_id: str
    controller: ControllerParameters
    metadata: Mapping[str, str] = None

    def __post_init__(self):
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if not isinstance(self.controller, ControllerParameters):
            raise TypeError("controller must be ControllerParameters")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def vector(self):
        controller = self.controller
        result = np.concatenate(
            (
                controller.p_gain,
                controller.i_gain,
                controller.d_gain,
                np.array([controller.controller_mass]),
                controller.controller_inertia_diagonal,
                controller.allocation_scale,
                np.array(
                    [
                        controller.thrust_scale,
                        controller.delay_compensation_s,
                    ]
                ),
            )
        )
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class InitialStateSample:
    sample_id: int
    state: ResponseState
    weight: float
    stamp: Optional[float] = None

    def __post_init__(self):
        if not isinstance(self.state, ResponseState):
            raise TypeError("state must be ResponseState")
        weight = float(self.weight)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("initial-state weight must be finite and positive")
        if self.stamp is not None and not np.isfinite(float(self.stamp)):
            raise ValueError("initial-state stamp must be finite when provided")
        object.__setattr__(self, "weight", weight)
        if self.stamp is not None:
            object.__setattr__(self, "stamp", float(self.stamp))


@dataclass(frozen=True)
class SupportReference:
    observed_candidate_vectors: np.ndarray
    observed_state_action_points: np.ndarray
    candidate_scale: np.ndarray
    state_action_scale: np.ndarray
    supported_distance: float = 2.0
    unsupported_distance: float = 5.0
    minimum_importance_ess: float = 2.0
    maximum_predictive_std: float = float("inf")

    def __post_init__(self):
        candidates = np.asarray(self.observed_candidate_vectors, dtype=float)
        points = np.asarray(self.observed_state_action_points, dtype=float)
        if (
            candidates.ndim != 2
            or candidates.shape[0] == 0
            or points.ndim != 2
            or points.shape[0] == 0
            or not np.all(np.isfinite(candidates))
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("support reference needs finite non-empty matrices")
        candidate_scale = np.asarray(self.candidate_scale, dtype=float)
        point_scale = np.asarray(self.state_action_scale, dtype=float)
        if candidate_scale.shape != (candidates.shape[1],) or np.any(
            candidate_scale <= 0.0
        ):
            raise ValueError("candidate_scale does not match support vectors")
        if point_scale.shape != (points.shape[1],) or np.any(point_scale <= 0.0):
            raise ValueError("state_action_scale does not match support points")
        supported = float(self.supported_distance)
        unsupported = float(self.unsupported_distance)
        minimum_ess = float(self.minimum_importance_ess)
        maximum_std = float(self.maximum_predictive_std)
        if (
            not 0.0 < supported < unsupported
            or minimum_ess <= 0.0
            or maximum_std <= 0.0
            or np.isnan(maximum_std)
        ):
            raise ValueError("support thresholds are invalid")
        object.__setattr__(self, "observed_candidate_vectors", candidates)
        object.__setattr__(self, "observed_state_action_points", points)
        object.__setattr__(self, "candidate_scale", candidate_scale)
        object.__setattr__(self, "state_action_scale", point_scale)
        object.__setattr__(self, "supported_distance", supported)
        object.__setattr__(self, "unsupported_distance", unsupported)
        object.__setattr__(self, "minimum_importance_ess", minimum_ess)
        object.__setattr__(self, "maximum_predictive_std", maximum_std)


@dataclass(frozen=True)
class SupportDiagnostics:
    label: str
    candidate_distance: float
    state_action_distance_p95: float
    importance_weight_ess: float
    maximum_predictive_std: float
    reasons: Tuple[str, ...]

    def __post_init__(self):
        if self.label not in _SUPPORT_LABELS:
            raise ValueError("invalid support label")


def classify_support(
    candidate_vector: np.ndarray,
    rollout_state_action_points: np.ndarray,
    predictive_std: float,
    reference: SupportReference,
) -> SupportDiagnostics:
    candidate = np.asarray(candidate_vector, dtype=float).reshape(-1)
    rollout_points = np.asarray(rollout_state_action_points, dtype=float)
    if candidate.shape != (reference.observed_candidate_vectors.shape[1],):
        raise ValueError("candidate vector dimension does not match support reference")
    if (
        rollout_points.ndim != 2
        or rollout_points.shape[1]
        != reference.observed_state_action_points.shape[1]
        or rollout_points.shape[0] == 0
    ):
        raise ValueError("rollout state/action points do not match support reference")
    candidate_distances = np.linalg.norm(
        (reference.observed_candidate_vectors - candidate)
        / reference.candidate_scale,
        axis=1,
    )
    candidate_distance = float(np.min(candidate_distances))
    log_importance = -0.5 * candidate_distances * candidate_distances
    log_importance -= float(np.max(log_importance))
    importance = np.exp(log_importance)
    importance /= np.sum(importance)
    importance_ess = float(1.0 / np.dot(importance, importance))
    # Chunking avoids a T x N x D allocation for long offline rollouts.
    nearest = []
    for start in range(0, rollout_points.shape[0], 256):
        chunk = rollout_points[start : start + 256]
        distance = np.linalg.norm(
            (
                chunk[:, None, :]
                - reference.observed_state_action_points[None, :, :]
            )
            / reference.state_action_scale,
            axis=2,
        )
        nearest.extend(np.min(distance, axis=1))
    state_distance = float(np.quantile(nearest, 0.95))
    uncertainty = float(predictive_std)
    reasons = []
    if candidate_distance > reference.unsupported_distance:
        reasons.append("candidate_parameter_distance")
    if state_distance > reference.unsupported_distance:
        reasons.append("state_action_distance")
    if importance_ess < reference.minimum_importance_ess:
        reasons.append("importance_weight_ess")
    if not np.isfinite(uncertainty) or uncertainty > reference.maximum_predictive_std:
        reasons.append("posterior_predictive_uncertainty")
    if reasons:
        label = UNSUPPORTED
    elif (
        candidate_distance > reference.supported_distance
        or state_distance > reference.supported_distance
    ):
        label = EXTRAPOLATIVE
        reasons.append("near_support_boundary")
    else:
        label = SUPPORTED
    return SupportDiagnostics(
        label=label,
        candidate_distance=candidate_distance,
        state_action_distance_p95=state_distance,
        importance_weight_ess=importance_ess,
        maximum_predictive_std=uncertainty,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class CounterfactualConfig:
    process_noise_sigma: np.ndarray
    process_noise_replicates: int = 1
    credible_probability: float = 0.95
    seed: int = 7
    analysis_mode: str = "retrospective"
    prefix_cutoff: Optional[float] = None
    inference_data_end_time: Optional[float] = None
    source_bag_hashes: Tuple[str, ...] = ()
    normalized_dataset_hashes: Tuple[str, ...] = ()
    source_commit: str = "UNKNOWN"

    def __post_init__(self):
        sigma = _vector(self.process_noise_sigma, "process_noise_sigma")
        if np.any(sigma < 0.0):
            raise ValueError("process_noise_sigma must be non-negative")
        replicates = int(self.process_noise_replicates)
        probability = float(self.credible_probability)
        if replicates < 1 or not 0.0 < probability < 1.0:
            raise ValueError("replicate count or credible probability is invalid")
        if self.analysis_mode not in ("retrospective", "online_prefix"):
            raise ValueError("analysis_mode must be retrospective or online_prefix")
        if self.analysis_mode == "online_prefix":
            if self.prefix_cutoff is None or not np.isfinite(float(self.prefix_cutoff)):
                raise ValueError("online_prefix analysis requires a finite prefix_cutoff")
            if self.inference_data_end_time is None or not np.isfinite(
                float(self.inference_data_end_time)
            ):
                raise ValueError(
                    "online_prefix analysis requires a finite inference_data_end_time"
                )
            if float(self.inference_data_end_time) > float(self.prefix_cutoff):
                raise ValueError(
                    "inference_data_end_time cannot exceed prefix_cutoff"
                )
        object.__setattr__(self, "process_noise_sigma", sigma)
        object.__setattr__(self, "process_noise_replicates", replicates)
        object.__setattr__(self, "credible_probability", probability)
        object.__setattr__(self, "source_bag_hashes", tuple(self.source_bag_hashes))
        object.__setattr__(
            self, "normalized_dataset_hashes", tuple(self.normalized_dataset_hashes)
        )


@dataclass(frozen=True)
class TrajectoryRollout:
    rollout_id: int
    initial_sample_id: int
    response_sample_id: int
    noise_sample_id: int
    weight: float
    position: np.ndarray
    velocity: np.ndarray
    command: np.ndarray
    saturation: np.ndarray
    tube: TubeEvaluation

    def __post_init__(self):
        weight = float(self.weight)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("rollout weight must be finite and positive")
        position = np.asarray(self.position, dtype=float)
        count = position.shape[0] if position.ndim == 2 else -1
        arrays = (
            ("position", position, float),
            ("velocity", self.velocity, float),
            ("command", self.command, float),
            ("saturation", self.saturation, bool),
        )
        for name, values, dtype in arrays:
            array = np.asarray(values, dtype=dtype)
            if array.shape != (count, AXIS_COUNT):
                raise ValueError(
                    "{} must have aligned shape (N, 6)".format(name)
                )
            if dtype is float and not np.all(np.isfinite(array)):
                raise ValueError("{} must be finite".format(name))
            copy = np.array(array, copy=True)
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)
        if not isinstance(self.tube, TubeEvaluation):
            raise TypeError("tube must be TubeEvaluation")
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True)
class CounterfactualResult:
    candidate: CounterfactualCandidate
    success_probability: float
    credible_lower: float
    credible_upper: float
    lower_credible_bound: float
    support: SupportDiagnostics
    violation_probability: Mapping[str, float]
    effective_rollout_sample_size: float
    rollouts: Tuple[TrajectoryRollout, ...]
    run_id: str
    provenance: Mapping[str, object]

    @property
    def recommendable(self):
        return self.support.label == SUPPORTED


class ClosedLoopCounterfactualEvaluator:
    def __init__(
        self,
        support_reference: SupportReference,
        response_model: Optional[LowDimensionalEffectiveResponse] = None,
    ):
        if not isinstance(support_reference, SupportReference):
            raise TypeError("support_reference must be SupportReference")
        self.support_reference = support_reference
        self.response_model = response_model or LowDimensionalEffectiveResponse()

    def evaluate(
        self,
        candidate: CounterfactualCandidate,
        target: TargetTrajectory,
        tube: TargetTube,
        response_posterior: EffectiveResponsePosterior,
        initial_state_samples: Sequence[InitialStateSample],
        config: CounterfactualConfig,
    ) -> CounterfactualResult:
        if not isinstance(candidate, CounterfactualCandidate):
            raise TypeError("candidate must be CounterfactualCandidate")
        initials = tuple(initial_state_samples)
        if not initials:
            raise ValueError("at least one initial-state sample is required")
        if config.analysis_mode == "online_prefix":
            cutoff = float(config.prefix_cutoff)
            tolerance = 1.0e-9
            if abs(float(target.timestamps[0]) - cutoff) > tolerance:
                raise ValueError(
                    "online-prefix counterfactual target must start at prefix_cutoff"
                )
            if any(
                item.stamp is None or abs(float(item.stamp) - cutoff) > tolerance
                for item in initials
            ):
                raise ValueError(
                    "online-prefix initial states must be stamped at prefix_cutoff"
                )
        initial_weights = np.asarray([item.weight for item in initials], dtype=float)
        initial_weights /= np.sum(initial_weights)
        request = ReplayRequest(
            timestamps=target.timestamps,
            reference_position=target.position,
            reference_velocity=target.velocity,
            reference_acceleration=target.acceleration,
        )
        rollouts = []
        rollout_id = 0
        for initial_index, initial in enumerate(initials):
            for response_index, response_parameters in enumerate(
                response_posterior.samples
            ):
                response_weight = float(response_posterior.weights[response_index])
                if response_weight <= 0.0:
                    continue
                for noise_index in range(config.process_noise_replicates):
                    seed_sequence = np.random.SeedSequence(
                        [
                            int(config.seed),
                            int(initial.sample_id),
                            int(response_index),
                            int(noise_index),
                        ]
                    )
                    rng = np.random.default_rng(seed_sequence)
                    process_noise = rng.normal(
                        0.0,
                        config.process_noise_sigma,
                        size=(target.timestamps.size - 1, AXIS_COUNT),
                    )
                    actuator_state = initial.state.actuator_state.copy()
                    command_times = []
                    commands = []

                    def plant_step(position, velocity, command, delta, index):
                        nonlocal actuator_state
                        command_times.append(float(target.timestamps[index]))
                        commands.append(np.array(command, copy=True))
                        current = ResponseState(
                            position, velocity, actuator_state
                        )
                        transition = self.response_model.transition(
                            current,
                            np.asarray(command_times),
                            np.asarray(commands),
                            target.timestamps[index],
                            delta,
                            response_parameters,
                            process_noise=process_noise[index],
                        )
                        actuator_state = transition.state.actuator_state.copy()
                        return (
                            transition.state.generalized_position,
                            transition.state.generalized_velocity,
                        )

                    replay = ControllerReplay().run(
                        request,
                        candidate.controller,
                        replay_mode="free_run",
                        initial_position=initial.state.generalized_position,
                        initial_velocity=initial.state.generalized_velocity,
                        initial_integral_state=None,
                        plant_step=plant_step,
                        plant_input="generalized_wrench_command",
                        apply_delay_compensation=True,
                    )
                    saturation = replay.term_saturated | replay.output_saturated
                    tube_result = evaluate_target_tube(
                        target,
                        tube,
                        replay.feedback_position,
                        replay.feedback_velocity,
                        saturation,
                    )
                    weight = (
                        initial_weights[initial_index]
                        * response_weight
                        / config.process_noise_replicates
                    )
                    rollouts.append(
                        TrajectoryRollout(
                            rollout_id=rollout_id,
                            initial_sample_id=int(initial.sample_id),
                            response_sample_id=int(response_index),
                            noise_sample_id=int(noise_index),
                            weight=float(weight),
                            position=replay.feedback_position,
                            velocity=replay.feedback_velocity,
                            command=replay.generalized_wrench_command,
                            saturation=saturation,
                            tube=tube_result,
                        )
                    )
                    rollout_id += 1
        if not rollouts:
            raise ValueError(
                "response posterior and initial samples produced no rollout mass"
            )
        weights = np.asarray([item.weight for item in rollouts], dtype=float)
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights <= 0.0)
            or float(np.sum(weights)) <= 0.0
        ):
            raise ValueError("counterfactual rollout weights are invalid")
        weights /= np.sum(weights)
        # Normalize the immutable rollout weights at the result boundary.
        normalized_rollouts = []
        for item, weight in zip(rollouts, weights):
            normalized_rollouts.append(
                TrajectoryRollout(
                    rollout_id=item.rollout_id,
                    initial_sample_id=item.initial_sample_id,
                    response_sample_id=item.response_sample_id,
                    noise_sample_id=item.noise_sample_id,
                    weight=float(weight),
                    position=item.position,
                    velocity=item.velocity,
                    command=item.command,
                    saturation=item.saturation,
                    tube=item.tube,
                )
            )
        rollouts = tuple(normalized_rollouts)
        successes = np.asarray([item.tube.success for item in rollouts], dtype=float)
        success_probability = float(
            np.clip(np.dot(weights, successes), 0.0, 1.0)
        )
        effective_count = float(1.0 / np.dot(weights, weights))
        successful_equivalent = success_probability * effective_count
        alpha = 0.5 + successful_equivalent
        beta = 0.5 + effective_count - successful_equivalent
        tail = 0.5 * (1.0 - config.credible_probability)
        lower = float(beta_distribution.ppf(tail, alpha, beta))
        upper = float(beta_distribution.ppf(1.0 - tail, alpha, beta))
        all_violations = sorted(
            set(
                violation
                for item in rollouts
                for violation in item.tube.violations
            )
        )
        violation_probability = {
            violation: float(
                np.dot(
                    weights,
                    np.asarray(
                        [
                            violation in item.tube.violations
                            for item in rollouts
                        ],
                        dtype=float,
                    ),
                )
            )
            for violation in all_violations
        }
        positions = np.asarray([item.position for item in rollouts])
        mean_position = _weighted_mean(positions, weights)
        variance = _weighted_mean(
            (positions - mean_position) ** 2, weights
        )
        predictive_std = float(np.max(np.sqrt(np.maximum(variance, 0.0))))
        state_action_points = np.concatenate(
            [
                np.column_stack(
                    (item.position, item.velocity, item.command)
                )
                for item in rollouts
            ],
            axis=0,
        )
        support = classify_support(
            candidate.vector(),
            state_action_points,
            predictive_std,
            self.support_reference,
        )
        provenance = {
            "source_bag_hashes": config.source_bag_hashes,
            "normalized_dataset_hashes": config.normalized_dataset_hashes,
            "source_commit": config.source_commit,
            "model_version": self.response_model.model_id,
            "controller_backend": "python_vector_pid_surrogate/v1",
            "seed": int(config.seed),
            "analysis_mode": config.analysis_mode,
            "prefix_cutoff": config.prefix_cutoff,
            "inference_data_end_time": config.inference_data_end_time,
            "trajectory_sample_ids": tuple(
                int(item.sample_id) for item in initials
            ),
            "response_posterior_approximation": response_posterior.approximation,
        }
        run_id = stable_hash(
            {
                "candidate_id": candidate.candidate_id,
                "candidate": candidate.vector(),
                "target_times": target.timestamps,
                "tube": tube,
                "provenance": provenance,
            }
        )[:20]
        return CounterfactualResult(
            candidate=candidate,
            success_probability=success_probability,
            credible_lower=lower,
            credible_upper=upper,
            lower_credible_bound=lower,
            support=support,
            violation_probability=violation_probability,
            effective_rollout_sample_size=effective_count,
            rollouts=rollouts,
            run_id=run_id,
            provenance=provenance,
        )


def connected_candidate_regions(
    results: Sequence[CounterfactualResult],
    gamma: float,
    neighbor_radius: float = 1.05,
) -> Tuple[Tuple[str, ...], ...]:
    """Return connected, supported regions with lower bound at least gamma."""

    threshold = float(gamma)
    radius = float(neighbor_radius)
    if not 0.0 <= threshold <= 1.0 or radius <= 0.0:
        raise ValueError("gamma or neighbor_radius is invalid")
    eligible = [
        item
        for item in results
        if item.support.label == SUPPORTED
        and item.lower_credible_bound >= threshold
    ]
    if not eligible:
        return ()
    vectors = np.asarray([item.candidate.vector() for item in eligible])
    scale = np.ptp(vectors, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    distances = np.linalg.norm(
        (vectors[:, None, :] - vectors[None, :, :]) / scale,
        axis=2,
    )
    unseen = set(range(len(eligible)))
    components = []
    while unseen:
        seed = unseen.pop()
        frontier = [seed]
        component = [seed]
        while frontier:
            current = frontier.pop()
            neighbours = [
                index
                for index in tuple(unseen)
                if distances[current, index] <= radius
            ]
            for index in neighbours:
                unseen.remove(index)
                frontier.append(index)
                component.append(index)
        components.append(
            tuple(sorted(eligible[index].candidate.candidate_id for index in component))
        )
    return tuple(sorted(components))


__all__ = [
    "EXTRAPOLATIVE",
    "SUPPORTED",
    "UNSUPPORTED",
    "ClosedLoopCounterfactualEvaluator",
    "CounterfactualCandidate",
    "CounterfactualConfig",
    "CounterfactualResult",
    "InitialStateSample",
    "SupportDiagnostics",
    "SupportReference",
    "TargetTrajectory",
    "TargetTube",
    "TrajectoryRollout",
    "TubeEvaluation",
    "classify_support",
    "connected_candidate_regions",
    "evaluate_target_tube",
]
