"""Posterior-predictive controller decisions over raw physical members.

This module deliberately contains no Jacobian or local plant approximation.
Every explicit controller candidate is evaluated by running the complete
closed loop once for every member of one selected-mode posterior ensemble.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple
import warnings

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.controller import (
    ControllerConfig,
    GrapeController,
    PIDConfig,
)
from grape_param_estim.dynamics import FullSixDofPlant, simulate_closed_loop
from grape_param_estim.geometry import (
    correction_transform_path,
    euler_xyz_to_matrix,
    matrix_to_quaternion,
)
from grape_param_estim.parameterization import VehicleParameterChart
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    ClosedLoopTrajectory,
    ControllerState,
    GrapeGeometry,
    ReferenceState,
    RigidBodyState,
    VehicleParameters,
)


PHASE5_ARTIFACT_SCHEMA = (
    "grape-weak-constraint/phase5-real-assimilation"
)


class _ControllerMassArticulatedModel:
    """Apply the declared controller mass to articulated allocation.

    ``GrapeController`` obtains the current aggregate mass/inertia from its
    articulated model whenever that model is present.  Merely replacing the
    controller's nominal ``VehicleParameters`` therefore does not affect the
    allocation.  This wrapper makes the Phase-6 mass proposal effective while
    retaining the angle-dependent inertia, CoG and thrust origins.
    """

    def __init__(self, controller_mass: float):
        self._base = GrapeArticulatedModel()
        self._controller_mass = float(controller_mass)

    def at(self, gimbal_angles):
        parameters, geometry = self._base.at(gimbal_angles)
        return (
            replace(
                parameters,
                mass=self._controller_mass,
            ),
            geometry,
        )


@dataclass(frozen=True)
class ControllerParameterCandidate:
    """One explicit multiplicative controller-parameter proposal."""

    candidate_id: str
    controller_mass_scale: float = 1.0
    roll_pid_scale: float = 1.0
    pitch_pid_scale: float = 1.0
    yaw_pid_scale: float = 1.0

    def __post_init__(self) -> None:
        identifier = str(self.candidate_id)
        scales = np.asarray(
            (
                self.controller_mass_scale,
                self.roll_pid_scale,
                self.pitch_pid_scale,
                self.yaw_pid_scale,
            ),
            dtype=float,
        )
        if not identifier:
            raise ValueError("candidate_id cannot be empty")
        if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("controller candidate scales must be positive")
        object.__setattr__(self, "candidate_id", identifier)
        for name, value in zip(
            (
                "controller_mass_scale",
                "roll_pid_scale",
                "pitch_pid_scale",
                "yaw_pid_scale",
            ),
            scales,
        ):
            object.__setattr__(self, name, float(value))

    @property
    def scales(self) -> np.ndarray:
        return np.asarray(
            (
                self.controller_mass_scale,
                self.roll_pid_scale,
                self.pitch_pid_scale,
                self.yaw_pid_scale,
            )
        )


def default_controller_parameter_candidates(
) -> Tuple[ControllerParameterCandidate, ...]:
    """Return the first explicit, auditable Phase-6 proposal set."""

    return (
        ControllerParameterCandidate("baseline"),
        ControllerParameterCandidate(
            "controller_mass_0p90", controller_mass_scale=0.90
        ),
        ControllerParameterCandidate(
            "controller_mass_1p10", controller_mass_scale=1.10
        ),
        ControllerParameterCandidate(
            "roll_pid_0p85",
            roll_pid_scale=0.85,
        ),
        ControllerParameterCandidate(
            "roll_pid_1p15",
            roll_pid_scale=1.15,
        ),
        ControllerParameterCandidate(
            "pitch_pid_0p85",
            pitch_pid_scale=0.85,
        ),
        ControllerParameterCandidate(
            "pitch_pid_1p15",
            pitch_pid_scale=1.15,
        ),
        ControllerParameterCandidate(
            "yaw_pid_0p85", yaw_pid_scale=0.85
        ),
        ControllerParameterCandidate(
            "yaw_pid_1p15", yaw_pid_scale=1.15
        ),
    )


@dataclass(frozen=True)
class TrackingLossDefinition:
    """Dimensionless pose-path loss scales."""

    translation_scale: float = 0.10
    rotation_scale: float = np.deg2rad(10.0)

    def __post_init__(self) -> None:
        values = np.asarray(
            (self.translation_scale, self.rotation_scale), dtype=float
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("tracking loss scales must be positive")
        object.__setattr__(self, "translation_scale", float(values[0]))
        object.__setattr__(self, "rotation_scale", float(values[1]))


@dataclass(frozen=True)
class PosteriorPredictiveWeights:
    """Weights applied after member-level tracking losses are retained."""

    mean_tracking_loss: float = 1.0
    cvar_tracking_loss: float = 0.5
    failure_probability: float = 5.0
    parameter_change: float = 0.05

    def __post_init__(self) -> None:
        values = np.asarray(tuple(self.__dict__.values()), dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("posterior-predictive weights must be non-negative")
        if not np.any(values > 0.0):
            raise ValueError("at least one decision weight must be positive")
        for name, value in zip(self.__dict__, values):
            object.__setattr__(self, name, float(value))


def _normalised_provenance(
    provenance: Optional[Mapping[str, str]],
) -> Tuple[Tuple[str, str], ...]:
    if provenance is None:
        return tuple()
    result = tuple(
        sorted((str(key), str(value)) for key, value in provenance.items())
    )
    if any(not key for key, _value in result):
        raise ValueError("provenance keys cannot be empty")
    if len({key for key, _value in result}) != len(result):
        raise ValueError("provenance keys must be unique")
    return result


@dataclass(frozen=True)
class PosteriorPredictiveInput:
    """Selected-mode raw posterior members and a future flight scenario."""

    selected_mode_id: str
    member_ids: np.ndarray
    physical_parameter_members: Tuple[VehicleParameters, ...]
    times: np.ndarray
    references: Tuple[ReferenceState, ...]
    initial_states: Tuple[RigidBodyState, ...]
    initial_controller_states: Tuple[ControllerState, ...]
    initial_actuator_states: Tuple[Optional[ActuatorState], ...]
    interval_residual_wrench: np.ndarray
    controller_configuration: ControllerConfig
    controller_parameters: VehicleParameters
    controller_geometry: GrapeGeometry
    plant_geometry: GrapeGeometry
    actuator_parameters: ActuatorParameters
    source_mode_ids: Tuple[str, ...] = tuple()
    source_mode_weights: Optional[np.ndarray] = None
    mode_conditioning_source: str = "selected_mode_posterior"
    scenario_assumption: str = (
        "repeat_reference_member_initial_state_and_residual_path"
    )
    provenance: Tuple[Tuple[str, str], ...] = tuple()

    def __post_init__(self) -> None:
        mode_id = str(self.selected_mode_id)
        member_ids = np.asarray(self.member_ids, dtype=np.int64)
        times = np.asarray(self.times, dtype=float)
        residual_wrench = np.asarray(
            self.interval_residual_wrench, dtype=float
        )
        member_count = len(self.physical_parameter_members)
        if not mode_id:
            raise ValueError("selected_mode_id cannot be empty")
        if (
            member_count < 2
            or member_ids.shape != (member_count,)
            or np.unique(member_ids).size != member_count
        ):
            raise ValueError(
                "at least two uniquely identified raw posterior members are "
                "required"
            )
        if (
            times.ndim != 1
            or times.size < 2
            or np.any(~np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
            or len(self.references) != times.size
        ):
            raise ValueError("future times and references must align")
        aligned = (
            self.initial_states,
            self.initial_controller_states,
            self.initial_actuator_states,
        )
        if any(len(value) != member_count for value in aligned):
            raise ValueError("member initial conditions must stay aligned")
        if (
            residual_wrench.shape != (member_count, times.size - 1, 6)
            or np.any(~np.isfinite(residual_wrench))
        ):
            raise ValueError(
                "interval_residual_wrench must explicitly contain one "
                "finite (N - 1, 6) path per member"
            )
        if any(
            not isinstance(value, VehicleParameters)
            for value in self.physical_parameter_members
        ):
            raise TypeError("physical members must be VehicleParameters")
        if any(
            not isinstance(value, RigidBodyState)
            for value in self.initial_states
        ):
            raise TypeError("initial states must be RigidBodyState values")
        if any(
            not isinstance(value, ControllerState)
            for value in self.initial_controller_states
        ):
            raise TypeError(
                "initial controller states must be ControllerState values"
            )
        if any(
            value is not None and not isinstance(value, ActuatorState)
            for value in self.initial_actuator_states
        ):
            raise TypeError(
                "initial actuator states must be ActuatorState values or None"
            )
        if any(
            not isinstance(value, ReferenceState) for value in self.references
        ):
            raise TypeError("references must be ReferenceState values")
        if not isinstance(self.controller_configuration, ControllerConfig):
            raise TypeError("controller_configuration has the wrong type")
        if not isinstance(self.controller_parameters, VehicleParameters):
            raise TypeError("controller_parameters has the wrong type")
        if not isinstance(self.controller_geometry, GrapeGeometry):
            raise TypeError("controller_geometry has the wrong type")
        if not isinstance(self.plant_geometry, GrapeGeometry):
            raise TypeError("plant_geometry has the wrong type")
        if not isinstance(self.actuator_parameters, ActuatorParameters):
            raise TypeError("actuator_parameters has the wrong type")
        source_mode_ids = tuple(str(value) for value in self.source_mode_ids)
        if not source_mode_ids:
            source_mode_ids = (mode_id,)
        source_mode_weights = (
            np.ones(1, dtype=float)
            if self.source_mode_weights is None
            else np.asarray(self.source_mode_weights, dtype=float)
        )
        if (
            len(set(source_mode_ids)) != len(source_mode_ids)
            or mode_id not in source_mode_ids
            or source_mode_weights.shape != (len(source_mode_ids),)
            or np.any(~np.isfinite(source_mode_weights))
            or np.any(source_mode_weights < 0.0)
            or not np.isclose(np.sum(source_mode_weights), 1.0)
        ):
            raise ValueError("source mode law must be finite and normalized")
        conditioning_source = str(self.mode_conditioning_source)
        scenario_assumption = str(self.scenario_assumption)
        if not conditioning_source or not scenario_assumption:
            raise ValueError(
                "mode conditioning and scenario assumption cannot be empty"
            )
        provenance = tuple(
            (str(key), str(value)) for key, value in self.provenance
        )
        if (
            any(not key for key, _value in provenance)
            or len({key for key, _value in provenance}) != len(provenance)
        ):
            raise ValueError("provenance keys must be non-empty and unique")
        object.__setattr__(self, "selected_mode_id", mode_id)
        object.__setattr__(self, "member_ids", member_ids.copy())
        object.__setattr__(self, "times", times.copy())
        object.__setattr__(
            self, "physical_parameter_members",
            tuple(self.physical_parameter_members),
        )
        object.__setattr__(self, "references", tuple(self.references))
        object.__setattr__(self, "initial_states", tuple(self.initial_states))
        object.__setattr__(
            self,
            "initial_controller_states",
            tuple(self.initial_controller_states),
        )
        object.__setattr__(
            self, "initial_actuator_states", tuple(self.initial_actuator_states)
        )
        object.__setattr__(
            self, "interval_residual_wrench", residual_wrench.copy()
        )
        object.__setattr__(self, "source_mode_ids", source_mode_ids)
        object.__setattr__(
            self, "source_mode_weights", source_mode_weights.copy()
        )
        object.__setattr__(
            self, "mode_conditioning_source", conditioning_source
        )
        object.__setattr__(self, "scenario_assumption", scenario_assumption)
        object.__setattr__(self, "provenance", provenance)


@dataclass(frozen=True)
class CandidateEvaluation:
    """Raw member-aligned prediction law and scalar decision summaries."""

    candidate: ControllerParameterCandidate
    member_ids: np.ndarray
    trajectories: Tuple[Optional[ClosedLoopTrajectory], ...]
    correction_translation: np.ndarray
    correction_rotation_vector: np.ndarray
    forecast_success: np.ndarray
    forecast_failure_reason: Tuple[str, ...]
    tracking_loss: np.ndarray
    mean_tracking_loss: float
    cvar_tracking_loss: float
    failure_probability: float
    parameter_change_magnitude: float
    change_penalty: float
    decision_score: float

    def __post_init__(self) -> None:
        member_ids = np.asarray(self.member_ids, dtype=np.int64)
        losses = np.asarray(self.tracking_loss, dtype=float)
        translation = np.asarray(self.correction_translation, dtype=float)
        rotation = np.asarray(self.correction_rotation_vector, dtype=float)
        success = np.asarray(self.forecast_success, dtype=bool)
        reasons = tuple(str(value) for value in self.forecast_failure_reason)
        member_count = member_ids.size
        if (
            losses.shape != (member_count,)
            or success.shape != (member_count,)
            or len(reasons) != member_count
            or translation.ndim != 3
            or translation.shape[0] != member_count
            or translation.shape[2] != 3
            or rotation.shape != translation.shape
            or len(self.trajectories) != member_count
            or np.any(~np.isfinite(losses))
        ):
            raise ValueError("candidate member laws must be aligned")
        finite_path = np.all(np.isfinite(translation), axis=(1, 2)) & np.all(
            np.isfinite(rotation), axis=(1, 2)
        )
        missing_path = np.all(np.isnan(translation), axis=(1, 2)) & np.all(
            np.isnan(rotation), axis=(1, 2)
        )
        if (
            not np.array_equal(finite_path, success)
            or np.any((~success) & (~missing_path))
            or any(
                (trajectory is None) != (not success[index])
                for index, trajectory in enumerate(self.trajectories)
            )
            or any(
                success[index] and not isinstance(trajectory, ClosedLoopTrajectory)
                for index, trajectory in enumerate(self.trajectories)
            )
            or any(
                (success[index] and reasons[index])
                or ((not success[index]) and not reasons[index])
                for index in range(member_count)
            )
        ):
            raise ValueError(
                "forecast success, trajectories, paths and reasons must agree"
            )
        finite_scalars = np.asarray(
            (
                self.mean_tracking_loss,
                self.cvar_tracking_loss,
                self.failure_probability,
                self.parameter_change_magnitude,
                self.change_penalty,
            ),
            dtype=float,
        )
        score = float(self.decision_score)
        if (
            np.any(~np.isfinite(finite_scalars))
            or np.isnan(score)
            or np.isneginf(score)
        ):
            raise ValueError(
                "candidate summaries must be finite except for a positive "
                "infinite rejection score"
            )
        if not 0.0 <= self.failure_probability <= 1.0:
            raise ValueError("failure_probability must be in [0, 1]")
        object.__setattr__(self, "member_ids", member_ids.copy())
        object.__setattr__(self, "tracking_loss", losses.copy())
        object.__setattr__(self, "forecast_success", success.copy())
        object.__setattr__(self, "forecast_failure_reason", reasons)
        object.__setattr__(
            self, "correction_translation", translation.copy()
        )
        object.__setattr__(
            self, "correction_rotation_vector", rotation.copy()
        )
        object.__setattr__(self, "trajectories", tuple(self.trajectories))
        for name, value in zip(
            (
                "mean_tracking_loss",
                "cvar_tracking_loss",
                "failure_probability",
                "parameter_change_magnitude",
                "change_penalty",
            ),
            finite_scalars,
        ):
            object.__setattr__(self, name, float(value))
        object.__setattr__(self, "decision_score", score)


@dataclass(frozen=True)
class PosteriorPredictiveDecision:
    """Complete candidate laws and the minimum posterior-predictive score."""

    predictive_input: PosteriorPredictiveInput
    evaluations: Tuple[CandidateEvaluation, ...]
    selected_candidate_index: int
    failure_threshold: float
    cvar_level: float
    loss_definition: TrackingLossDefinition
    weights: PosteriorPredictiveWeights

    def __post_init__(self) -> None:
        evaluations = tuple(self.evaluations)
        index = int(self.selected_candidate_index)
        threshold = float(self.failure_threshold)
        level = float(self.cvar_level)
        if not isinstance(self.predictive_input, PosteriorPredictiveInput):
            raise TypeError("predictive_input has the wrong type")
        if (
            not evaluations
            or any(
                not isinstance(value, CandidateEvaluation)
                for value in evaluations
            )
            or len({value.candidate.candidate_id for value in evaluations})
            != len(evaluations)
        ):
            raise ValueError("evaluations must contain unique candidates")
        if not -1 <= index < len(evaluations):
            raise ValueError("selected_candidate_index is out of range")
        for value in evaluations:
            if not np.array_equal(
                value.member_ids, self.predictive_input.member_ids
            ):
                raise ValueError("candidate member identities must align")
        scores = np.asarray(
            [value.decision_score for value in evaluations], dtype=float
        )
        finite = np.flatnonzero(np.isfinite(scores))
        expected_index = (
            -1
            if finite.size == 0
            else int(finite[np.argmin(scores[finite])])
        )
        if index != expected_index:
            raise ValueError("selected candidate must minimize decision score")
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("failure_threshold must be finite and positive")
        if not np.isfinite(level) or not 0.0 <= level < 1.0:
            raise ValueError("cvar_level must be in [0, 1)")
        if not isinstance(self.loss_definition, TrackingLossDefinition):
            raise TypeError("loss_definition has the wrong type")
        if not isinstance(self.weights, PosteriorPredictiveWeights):
            raise TypeError("weights has the wrong type")
        object.__setattr__(self, "evaluations", evaluations)
        object.__setattr__(self, "selected_candidate_index", index)
        object.__setattr__(self, "failure_threshold", threshold)
        object.__setattr__(self, "cvar_level", level)

    @property
    def selected_evaluation(self) -> CandidateEvaluation:
        if self.selected_candidate_index < 0:
            raise RuntimeError(
                "no candidate has a complete finite forecast law"
            )
        return self.evaluations[self.selected_candidate_index]

    @property
    def selected_candidate(self) -> ControllerParameterCandidate:
        return self.selected_evaluation.candidate

    @property
    def recommendation_available(self) -> bool:
        return self.selected_candidate_index >= 0


def apply_controller_candidate(
    configuration: ControllerConfig,
    controller_parameters: VehicleParameters,
    candidate: ControllerParameterCandidate,
) -> Tuple[ControllerConfig, VehicleParameters]:
    """Apply only the four declared multiplicative proposal coordinates."""

    if not isinstance(configuration, ControllerConfig):
        raise TypeError("configuration must be a ControllerConfig")
    if not isinstance(controller_parameters, VehicleParameters):
        raise TypeError("controller_parameters must be VehicleParameters")
    if not isinstance(candidate, ControllerParameterCandidate):
        raise TypeError("candidate must be a ControllerParameterCandidate")
    pid = list(configuration.pid)
    for axis, scale in (
        (3, candidate.roll_pid_scale),
        (4, candidate.pitch_pid_scale),
        (5, candidate.yaw_pid_scale),
    ):
        value = pid[axis]
        pid[axis] = replace(
            value,
            p_gain=value.p_gain * scale,
            i_gain=value.i_gain * scale,
            d_gain=value.d_gain * scale,
        )
    return (
        replace(configuration, pid=tuple(pid)),
        replace(
            controller_parameters,
            mass=(
                controller_parameters.mass
                * candidate.controller_mass_scale
            ),
        ),
    )


def empirical_upper_cvar(losses: Sequence[float], level: float) -> float:
    """Exact equal-weight empirical upper-tail CVaR at ``level``."""

    values = np.asarray(losses, dtype=float)
    selected_level = float(level)
    if values.ndim != 1 or values.size < 1 or np.any(~np.isfinite(values)):
        raise ValueError("losses must be a non-empty finite vector")
    if not np.isfinite(selected_level) or not 0.0 <= selected_level < 1.0:
        raise ValueError("CVaR level must be in [0, 1)")
    ordered = np.sort(values)
    count = ordered.size
    left = np.arange(count, dtype=float) / count
    right = np.arange(1, count + 1, dtype=float) / count
    mass = np.maximum(0.0, right - np.maximum(left, selected_level))
    return float(np.dot(mass, ordered) / (1.0 - selected_level))


def _desired_pose(
    references: Sequence[ReferenceState],
) -> Tuple[np.ndarray, np.ndarray]:
    position = np.asarray([value.position for value in references])
    orientation = np.asarray(
        [
            matrix_to_quaternion(euler_xyz_to_matrix(value.rpy))
            for value in references
        ]
    )
    return position, orientation


def _path_tracking_loss(
    times: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    definition: TrackingLossDefinition,
) -> float:
    pointwise = np.sum(
        (translation / definition.translation_scale) ** 2, axis=1
    ) + np.sum((rotation / definition.rotation_scale) ** 2, axis=1)
    return float(np.trapz(pointwise, times) / (times[-1] - times[0]))


def evaluate_posterior_predictive(
    predictive_input: PosteriorPredictiveInput,
    candidates: Optional[Sequence[ControllerParameterCandidate]] = None,
    failure_threshold: float = 1.0,
    cvar_level: float = 0.90,
    loss_definition: Optional[TrackingLossDefinition] = None,
    weights: Optional[PosteriorPredictiveWeights] = None,
) -> PosteriorPredictiveDecision:
    """Evaluate every candidate against every selected-mode raw member."""

    if not isinstance(predictive_input, PosteriorPredictiveInput):
        raise TypeError("predictive_input has the wrong type")
    selected_candidates = tuple(
        default_controller_parameter_candidates()
        if candidates is None
        else candidates
    )
    if (
        not selected_candidates
        or any(
            not isinstance(value, ControllerParameterCandidate)
            for value in selected_candidates
        )
        or len({value.candidate_id for value in selected_candidates})
        != len(selected_candidates)
    ):
        raise ValueError("controller candidates must be non-empty and unique")
    selected_threshold = float(failure_threshold)
    selected_level = float(cvar_level)
    if not np.isfinite(selected_threshold) or selected_threshold <= 0.0:
        raise ValueError("failure_threshold must be finite and positive")
    if not np.isfinite(selected_level) or not 0.0 <= selected_level < 1.0:
        raise ValueError("cvar_level must be in [0, 1)")
    selected_loss = loss_definition or TrackingLossDefinition()
    selected_weights = weights or PosteriorPredictiveWeights()
    if not isinstance(selected_loss, TrackingLossDefinition):
        raise TypeError("loss_definition has the wrong type")
    if not isinstance(selected_weights, PosteriorPredictiveWeights):
        raise TypeError("weights has the wrong type")

    desired_position, desired_orientation = _desired_pose(
        predictive_input.references
    )
    evaluations = []
    for candidate in selected_candidates:
        configuration, controller_parameters = apply_controller_candidate(
            predictive_input.controller_configuration,
            predictive_input.controller_parameters,
            candidate,
        )
        controller = GrapeController(
            configuration,
            controller_parameters,
            predictive_input.controller_geometry,
            articulated_model=_ControllerMassArticulatedModel(
                controller_parameters.mass
            ),
        )
        member_count = len(predictive_input.physical_parameter_members)
        sample_count = predictive_input.times.size
        trajectories = [None] * member_count
        translations = np.full(
            (member_count, sample_count, 3), np.nan, dtype=float
        )
        rotations = np.full_like(translations, np.nan)
        # A numerical/physical forecast failure is explicitly assigned the
        # decision boundary loss and counted as a failure.  Its missing path
        # remains NaN in the raw artifact together with a reason string.
        losses = np.full(member_count, selected_threshold, dtype=float)
        forecast_success = np.zeros(member_count, dtype=bool)
        failure_reasons = [""] * member_count
        for member, parameters in enumerate(
            predictive_input.physical_parameter_members
        ):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    trajectory = simulate_closed_loop(
                        times=predictive_input.times,
                        references=predictive_input.references,
                        initial_state=predictive_input.initial_states[member],
                        initial_controller_state=(
                            predictive_input.initial_controller_states[member]
                        ),
                        controller=controller,
                        plant=FullSixDofPlant(
                            parameters, predictive_input.plant_geometry
                        ),
                        actuator_parameters=(
                            predictive_input.actuator_parameters
                        ),
                        initial_actuator_state=(
                            predictive_input.initial_actuator_states[member]
                        ),
                        interval_residual_wrench=(
                            predictive_input.interval_residual_wrench[member]
                        ),
                    )
                    translation, rotation = correction_transform_path(
                        desired_position,
                        desired_orientation,
                        trajectory.position,
                        trajectory.orientation_xyzw,
                    )
                    loss = _path_tracking_loss(
                        predictive_input.times,
                        translation,
                        rotation,
                        selected_loss,
                    )
            except (
                ValueError,
                FloatingPointError,
                RuntimeWarning,
                OverflowError,
                np.linalg.LinAlgError,
            ) as error:
                failure_reasons[member] = "{}: {}".format(
                    type(error).__name__, str(error).replace("\n", " ")[:240]
                )
                continue
            trajectories[member] = trajectory
            translations[member] = translation
            rotations[member] = rotation
            losses[member] = loss
            forecast_success[member] = True
        loss_array = np.asarray(losses)
        mean_loss = float(np.mean(loss_array))
        cvar_loss = empirical_upper_cvar(loss_array, selected_level)
        failure_probability = float(
            np.mean((loss_array >= selected_threshold) | (~forecast_success))
        )
        parameter_change_magnitude = float(
            np.mean(np.log(candidate.scales) ** 2)
        )
        change_penalty = float(
            selected_weights.parameter_change
            * parameter_change_magnitude
        )
        score = float(
            selected_weights.mean_tracking_loss * mean_loss
            + selected_weights.cvar_tracking_loss * cvar_loss
            + selected_weights.failure_probability * failure_probability
            + change_penalty
        )
        if not np.all(forecast_success):
            # A simulator exception is epistemically different from a finite
            # path that crosses the declared failure threshold.  Preserve the
            # failed rows, but never recommend a candidate whose raw posterior
            # law could not be evaluated completely.
            score = float("inf")
        evaluations.append(
            CandidateEvaluation(
                candidate=candidate,
                member_ids=predictive_input.member_ids,
                trajectories=tuple(trajectories),
                correction_translation=translations,
                correction_rotation_vector=rotations,
                forecast_success=forecast_success,
                forecast_failure_reason=tuple(failure_reasons),
                tracking_loss=loss_array,
                mean_tracking_loss=mean_loss,
                cvar_tracking_loss=cvar_loss,
                failure_probability=failure_probability,
                parameter_change_magnitude=parameter_change_magnitude,
                change_penalty=change_penalty,
                decision_score=score,
            )
        )
    score = np.asarray([value.decision_score for value in evaluations])
    finite = np.flatnonzero(np.isfinite(score))
    return PosteriorPredictiveDecision(
        predictive_input=predictive_input,
        evaluations=tuple(evaluations),
        selected_candidate_index=(
            -1 if finite.size == 0 else int(finite[np.argmin(score[finite])])
        ),
        failure_threshold=selected_threshold,
        cvar_level=selected_level,
        loss_definition=selected_loss,
        weights=selected_weights,
    )


def input_from_mode_posterior(
    selected_posterior,
    provenance: Optional[Mapping[str, str]] = None,
) -> PosteriorPredictiveInput:
    """Adapt one selected :class:`ModePosterior` without averaging members."""

    problem = selected_posterior.problem
    base_problem = getattr(problem, "strong_problem", problem)
    posterior = selected_posterior.posterior
    controls = np.asarray(posterior.control_ensemble, dtype=float)
    decoded = tuple(problem.decode_control(value) for value in controls)
    supplied = {} if provenance is None else dict(provenance)
    supplied.setdefault("source_kind", "selected_mode_posterior")
    residual_wrench = getattr(
        posterior,
        "residual_wrench_ensemble",
        np.zeros((controls.shape[0], base_problem.observations.times.size - 1, 6)),
    )
    return PosteriorPredictiveInput(
        selected_mode_id=selected_posterior.mode.mode_id,
        member_ids=selected_posterior.member_ids,
        physical_parameter_members=tuple(value[2] for value in decoded),
        times=base_problem.observations.times,
        references=base_problem.references,
        initial_states=tuple(value[0] for value in decoded),
        initial_controller_states=tuple(value[1] for value in decoded),
        initial_actuator_states=tuple(
            base_problem.initial_actuator_state for _value in decoded
        ),
        interval_residual_wrench=residual_wrench,
        controller_configuration=base_problem.controller_configuration,
        controller_parameters=base_problem.controller_parameters,
        controller_geometry=base_problem.geometry,
        plant_geometry=selected_posterior.mode.plant_geometry(
            base_problem.geometry
        ),
        actuator_parameters=base_problem.actuator_parameters,
        scenario_assumption=(
            "repeat_reference_member_initial_state_and_residual_path"
        ),
        provenance=_normalised_provenance(supplied),
    )


def input_from_real_assimilation(
    result,
    provenance: Optional[Mapping[str, str]] = None,
) -> PosteriorPredictiveInput:
    """Connect one Phase-5 real raw law to the same future flight scenario.

    Static parameters, residual-wrench paths and the first state/controller/
    actuator sample are taken from the same posterior row.  No posterior mean
    or center trajectory is substituted for an individual member.
    """

    posterior = result.posterior
    episode = result.episode
    trajectories = tuple(posterior.trajectory_ensemble)
    member_count = len(trajectories)
    parameter_coordinates = np.asarray(
        posterior.parameter_ensemble.coordinates, dtype=float
    )
    if parameter_coordinates.shape[0] != member_count:
        raise ValueError("real posterior parameter members are misaligned")
    chart = VehicleParameterChart(result.nominal_parameters)
    physical_members = tuple(
        chart.decode(value) for value in parameter_coordinates
    )
    initial_states = tuple(
        RigidBodyState(
            position=value.position[0],
            orientation_xyzw=value.orientation_xyzw[0],
            linear_velocity=value.linear_velocity[0],
            angular_velocity=value.angular_velocity[0],
        )
        for value in trajectories
    )
    integration_active = (
        episode.initial_controller_state.roll_pitch_integration_active
    )
    initial_controller_states = tuple(
        ControllerState(
            integral_error=value.controller_integral[0],
            roll_pitch_integration_active=integration_active,
        )
        for value in trajectories
    )
    initial_actuator_states = tuple(
        ActuatorState(
            thrust=value.actuator_thrust[0],
            gimbal_angle=value.actuator_gimbal_angle[0],
        )
        for value in trajectories
    )
    supplied = {} if provenance is None else dict(provenance)
    supplied.setdefault("source_kind", "real_assimilation_result")
    episode_provenance = getattr(episode, "provenance", None)
    for key in ("bag_path", "bag_sha256", "time_basis"):
        if episode_provenance is not None and hasattr(
            episode_provenance, key
        ):
            supplied.setdefault(key, getattr(episode_provenance, key))
    mode_id = result.mode_diagnostic.selected_mode_id
    # The current Phase-5 result registers the same audited wiring modes as
    # Phase 4.  Import locally so the core predictive API remains mode-agnostic.
    from grape_param_estim.mode_validation import plant_wiring_mode

    controller_geometry = GrapeGeometry.grape()
    plant_geometry = plant_wiring_mode(mode_id).plant_geometry(
        controller_geometry
    )
    return PosteriorPredictiveInput(
        selected_mode_id=mode_id,
        member_ids=np.arange(member_count, dtype=np.int64),
        physical_parameter_members=physical_members,
        times=episode.observations.times,
        references=episode.references,
        initial_states=initial_states,
        initial_controller_states=initial_controller_states,
        initial_actuator_states=initial_actuator_states,
        interval_residual_wrench=posterior.residual_wrench_ensemble,
        controller_configuration=episode.controller_configuration,
        controller_parameters=result.nominal_parameters,
        controller_geometry=controller_geometry,
        plant_geometry=plant_geometry,
        actuator_parameters=result.actuator_parameters,
        source_mode_ids=tuple(
            getattr(result.mode_diagnostic, "mode_ids", (mode_id,))
        ),
        source_mode_weights=getattr(
            result.mode_diagnostic, "weights", np.ones(1)
        ),
        mode_conditioning_source=(
            getattr(
                result.mode_diagnostic,
                "conditioning_source",
                "selected_mode_posterior",
            )
        ),
        scenario_assumption=(
            "repeat_phase5_reference_member_initial_state_and_residual_path"
        ),
        provenance=_normalised_provenance(supplied),
    )


def _artifact_scalar(arrays, key):
    value = np.asarray(arrays[key])
    if value.size != 1:
        raise ValueError("{} must contain exactly one value".format(key))
    return value.reshape(-1)[0]


def input_from_phase5_artifact(
    path: str,
    residual_policy: str = "posterior_replay",
) -> PosteriorPredictiveInput:
    """Load the raw selected-mode law from a Phase-5 NPZ artifact.

    The default scenario is an explicit counterfactual repeat of the Phase-5
    reference, member initial conditions and member residual-wrench paths.  It
    is suitable for selecting parameters for a repeat of the same experiment;
    it is not presented as a newly sampled future wind realization.
    """

    selected_policy = str(residual_policy)
    if selected_policy not in ("posterior_replay", "zero"):
        raise ValueError(
            "residual_policy must be 'posterior_replay' or 'zero'"
        )
    source_path = Path(path).expanduser().resolve()
    required = (
        "schema",
        "member_id",
        "times",
        "reference_position",
        "reference_linear_velocity",
        "reference_linear_acceleration",
        "reference_rpy",
        "reference_angular_velocity",
        "reference_angular_acceleration",
        "posterior_position",
        "posterior_orientation_xyzw",
        "posterior_linear_velocity",
        "posterior_angular_velocity",
        "posterior_controller_integral",
        "posterior_actuator_thrust",
        "posterior_actuator_gimbal_angle",
        "residual_wrench_interval",
        "parameter_mass",
        "parameter_inertia",
        "parameter_cog_offset",
        "parameter_force_effectiveness",
        "parameter_torque_effectiveness",
        "selected_mode_id",
        "mode_ids",
        "mode_weights",
        "mode_conditioning_source",
        "initial_controller_roll_pitch_active",
        "controller_pid_axis_names",
        "controller_pid_field_names",
        "controller_pid_configuration",
        "controller_xy_control_mode",
        "controller_need_yaw_d_control",
        "controller_start_roll_pitch_integration_height",
        "controller_initial_height",
        "controller_source_compatible_gyro_term",
        "nominal_parameter_mass",
        "nominal_parameter_inertia",
        "nominal_parameter_cog_offset",
        "nominal_parameter_force_effectiveness",
        "nominal_parameter_torque_effectiveness",
        "nominal_parameter_linear_drag",
        "nominal_parameter_angular_drag",
        "actuator_parameter_names",
        "actuator_parameter_values",
        "controller_geometry_rotor_origins",
        "controller_geometry_arm_yaws",
        "controller_geometry_rotor_directions",
        "controller_geometry_moment_force_rate",
        "controller_geometry_thrust_offset",
        "controller_articulated_model_id",
    )
    with np.load(str(source_path), allow_pickle=False) as artifact:
        missing = [key for key in required if key not in artifact.files]
        if missing:
            raise ValueError(
                "Phase-5 artifact is missing: {}".format(", ".join(missing))
            )
        arrays = {key: np.asarray(artifact[key]).copy() for key in required}
        for optional_key in (
            "bag_path",
            "bag_sha256",
            "time_basis",
        ):
            if optional_key in artifact.files:
                arrays[optional_key] = np.asarray(
                    artifact[optional_key]
                ).copy()
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("Phase-5 artifact must not contain object arrays")
    schema = str(_artifact_scalar(arrays, "schema"))
    if schema != PHASE5_ARTIFACT_SCHEMA:
        raise ValueError("unsupported Phase-5 artifact schema: {}".format(schema))
    articulated_model_id = str(
        _artifact_scalar(arrays, "controller_articulated_model_id")
    )
    if articulated_model_id != "grape_articulated_model/v1":
        raise ValueError(
            "unsupported controller articulated model: {}".format(
                articulated_model_id
            )
        )

    member_ids = np.asarray(arrays["member_id"], dtype=np.int64)
    times = np.asarray(arrays["times"], dtype=float)
    member_count = member_ids.size
    sample_count = times.size
    if (
        member_ids.shape != (member_count,)
        or np.unique(member_ids).size != member_count
        or times.shape != (sample_count,)
        or np.any(~np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
        or member_count < 2
        or sample_count < 2
    ):
        raise ValueError("Phase-5 artifact member/time coordinates are invalid")
    for key in (
        "reference_position",
        "reference_linear_velocity",
        "reference_linear_acceleration",
        "reference_rpy",
        "reference_angular_velocity",
        "reference_angular_acceleration",
    ):
        if (
            arrays[key].shape != (sample_count, 3)
            or np.any(~np.isfinite(arrays[key]))
        ):
            raise ValueError("{} has an invalid time-aligned shape".format(key))
    references = tuple(
        ReferenceState(
            arrays["reference_position"][index],
            arrays["reference_linear_velocity"][index],
            arrays["reference_linear_acceleration"][index],
            arrays["reference_rpy"][index],
            arrays["reference_angular_velocity"][index],
            arrays["reference_angular_acceleration"][index],
        )
        for index in range(sample_count)
    )
    trajectory_shapes = {
        "posterior_position": (member_count, sample_count, 3),
        "posterior_orientation_xyzw": (member_count, sample_count, 4),
        "posterior_linear_velocity": (member_count, sample_count, 3),
        "posterior_angular_velocity": (member_count, sample_count, 3),
        "posterior_controller_integral": (member_count, sample_count, 6),
        "posterior_actuator_thrust": (member_count, sample_count, 4),
        "posterior_actuator_gimbal_angle": (member_count, sample_count, 4),
        "residual_wrench_interval": (member_count, sample_count - 1, 6),
    }
    for key, shape in trajectory_shapes.items():
        if arrays[key].shape != shape or np.any(~np.isfinite(arrays[key])):
            raise ValueError("{} has an invalid member-aligned shape".format(key))
    initial_states = tuple(
        RigidBodyState(
            arrays["posterior_position"][member, 0],
            arrays["posterior_orientation_xyzw"][member, 0],
            arrays["posterior_linear_velocity"][member, 0],
            arrays["posterior_angular_velocity"][member, 0],
        )
        for member in range(member_count)
    )
    integration_active = bool(
        _artifact_scalar(arrays, "initial_controller_roll_pitch_active")
    )
    initial_controller_states = tuple(
        ControllerState(
            arrays["posterior_controller_integral"][member, 0],
            integration_active,
        )
        for member in range(member_count)
    )
    initial_actuator_states = tuple(
        ActuatorState(
            arrays["posterior_actuator_thrust"][member, 0],
            arrays["posterior_actuator_gimbal_angle"][member, 0],
        )
        for member in range(member_count)
    )

    nominal_parameters = VehicleParameters(
        mass=float(_artifact_scalar(arrays, "nominal_parameter_mass")),
        inertia=arrays["nominal_parameter_inertia"],
        cog_offset=arrays["nominal_parameter_cog_offset"],
        force_effectiveness=arrays["nominal_parameter_force_effectiveness"],
        torque_effectiveness=(
            arrays["nominal_parameter_torque_effectiveness"]
        ),
        linear_drag=arrays["nominal_parameter_linear_drag"],
        angular_drag=arrays["nominal_parameter_angular_drag"],
    )
    parameter_shapes = {
        "parameter_mass": (member_count,),
        "parameter_inertia": (member_count, 3, 3),
        "parameter_cog_offset": (member_count, 3),
        "parameter_force_effectiveness": (member_count, 4),
        "parameter_torque_effectiveness": (member_count, 4),
    }
    for key, shape in parameter_shapes.items():
        if arrays[key].shape != shape:
            raise ValueError("{} has an invalid raw-member shape".format(key))
    physical_members = tuple(
        VehicleParameters(
            mass=arrays["parameter_mass"][member],
            inertia=arrays["parameter_inertia"][member],
            cog_offset=arrays["parameter_cog_offset"][member],
            force_effectiveness=(
                arrays["parameter_force_effectiveness"][member]
            ),
            torque_effectiveness=(
                arrays["parameter_torque_effectiveness"][member]
            ),
            linear_drag=nominal_parameters.linear_drag,
            angular_drag=nominal_parameters.angular_drag,
        )
        for member in range(member_count)
    )

    pid_fields = tuple(
        str(value) for value in arrays["controller_pid_field_names"]
    )
    pid_axes = tuple(
        str(value) for value in arrays["controller_pid_axis_names"]
    )
    expected_pid_fields = tuple(PIDConfig.__dataclass_fields__)
    if (
        pid_axes != ("x", "y", "z", "roll", "pitch", "yaw")
        or len(pid_fields) != len(set(pid_fields))
        or set(pid_fields) != set(expected_pid_fields)
        or arrays["controller_pid_configuration"].shape
        != (6, len(pid_fields))
    ):
        raise ValueError("controller PID artifact contract is invalid")
    pid = tuple(
        PIDConfig(
            **{
                name: float(arrays["controller_pid_configuration"][axis, index])
                for index, name in enumerate(pid_fields)
            }
        )
        for axis in range(6)
    )
    controller_configuration = ControllerConfig(
        pid=pid,
        xy_control_mode=str(
            _artifact_scalar(arrays, "controller_xy_control_mode")
        ),
        need_yaw_d_control=bool(
            _artifact_scalar(arrays, "controller_need_yaw_d_control")
        ),
        start_roll_pitch_integration_height=float(
            _artifact_scalar(
                arrays, "controller_start_roll_pitch_integration_height"
            )
        ),
        initial_height=float(
            _artifact_scalar(arrays, "controller_initial_height")
        ),
        source_compatible_gyro_term=bool(
            _artifact_scalar(
                arrays, "controller_source_compatible_gyro_term"
            )
        ),
    )
    actuator_names = tuple(
        str(value) for value in arrays["actuator_parameter_names"]
    )
    actuator_values = np.asarray(
        arrays["actuator_parameter_values"], dtype=float
    )
    expected_actuator_names = tuple(ActuatorParameters.__dataclass_fields__)
    if (
        actuator_names != expected_actuator_names
        or actuator_values.shape != (len(actuator_names),)
    ):
        raise ValueError("actuator parameter artifact contract is invalid")
    actuator_parameters = ActuatorParameters(
        **{
            name: float(actuator_values[index])
            for index, name in enumerate(actuator_names)
        }
    )
    controller_geometry = GrapeGeometry(
        rotor_origins=arrays["controller_geometry_rotor_origins"],
        arm_yaws=arrays["controller_geometry_arm_yaws"],
        rotor_directions=arrays["controller_geometry_rotor_directions"],
        moment_force_rate=float(
            _artifact_scalar(
                arrays, "controller_geometry_moment_force_rate"
            )
        ),
        thrust_offset=float(
            _artifact_scalar(arrays, "controller_geometry_thrust_offset")
        ),
    )
    selected_mode_id = str(_artifact_scalar(arrays, "selected_mode_id"))
    from grape_param_estim.mode_validation import plant_wiring_mode

    plant_geometry = plant_wiring_mode(selected_mode_id).plant_geometry(
        controller_geometry
    )
    mode_ids = tuple(str(value) for value in arrays["mode_ids"])
    mode_weights = np.asarray(arrays["mode_weights"], dtype=float)
    conditioning_source = str(
        _artifact_scalar(arrays, "mode_conditioning_source")
    )
    residual_wrench = np.asarray(
        arrays["residual_wrench_interval"], dtype=float
    )
    if selected_policy == "zero":
        residual_wrench = np.zeros_like(residual_wrench)
    scenario_assumption = (
        "repeat_phase5_reference_member_initial_state_and_"
        + (
            "posterior_residual_path"
            if selected_policy == "posterior_replay"
            else "zero_residual_wrench"
        )
    )
    provenance = {
        "source_kind": "phase5_real_assimilation_artifact",
        "source_artifact": str(source_path),
        "source_schema": schema,
        "controller_articulated_model_id": articulated_model_id,
        "residual_policy": selected_policy,
    }
    for source_key in ("bag_path", "bag_sha256", "time_basis"):
        if source_key in arrays:
            provenance[source_key] = str(
                _artifact_scalar(arrays, source_key)
            )
    return PosteriorPredictiveInput(
        selected_mode_id=selected_mode_id,
        member_ids=member_ids,
        physical_parameter_members=physical_members,
        times=times,
        references=references,
        initial_states=initial_states,
        initial_controller_states=initial_controller_states,
        initial_actuator_states=initial_actuator_states,
        interval_residual_wrench=residual_wrench,
        controller_configuration=controller_configuration,
        controller_parameters=nominal_parameters,
        controller_geometry=controller_geometry,
        plant_geometry=plant_geometry,
        actuator_parameters=actuator_parameters,
        source_mode_ids=mode_ids,
        source_mode_weights=mode_weights,
        mode_conditioning_source=conditioning_source,
        scenario_assumption=scenario_assumption,
        provenance=_normalised_provenance(provenance),
    )


def _physical_member_payload(
    values: Sequence[VehicleParameters],
) -> Mapping[str, np.ndarray]:
    return {
        "posterior_member_mass": np.asarray([value.mass for value in values]),
        "posterior_member_inertia": np.asarray(
            [value.inertia for value in values]
        ),
        "posterior_member_cog_offset": np.asarray(
            [value.cog_offset for value in values]
        ),
        "posterior_member_force_effectiveness": np.asarray(
            [value.force_effectiveness for value in values]
        ),
        "posterior_member_torque_effectiveness": np.asarray(
            [value.torque_effectiveness for value in values]
        ),
        "posterior_member_linear_drag": np.asarray(
            [value.linear_drag for value in values]
        ),
        "posterior_member_angular_drag": np.asarray(
            [value.angular_drag for value in values]
        ),
    }


def save_posterior_predictive_decision(
    path: str, decision: PosteriorPredictiveDecision
) -> Path:
    """Save proposals, provenance and every raw prediction without pickle."""

    if not isinstance(decision, PosteriorPredictiveDecision):
        raise TypeError("decision has the wrong type")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = decision.predictive_input
    evaluations = decision.evaluations
    candidates = tuple(value.candidate for value in evaluations)
    provenance_keys = np.asarray([key for key, _value in source.provenance])
    provenance_values = np.asarray(
        [value for _key, value in source.provenance]
    )

    trajectory_width = {
        "position": 3,
        "orientation_xyzw": 4,
        "linear_velocity": 3,
        "angular_velocity": 3,
        "controller_integral": 6,
        "commanded_thrust": 4,
        "commanded_gimbal_angle": 4,
        "actuator_thrust": 4,
        "actuator_gimbal_angle": 4,
        "body_wrench": 6,
    }

    def trajectory_field(name):
        result = np.full(
            (
                len(evaluations),
                source.member_ids.size,
                source.times.size,
                trajectory_width[name],
            ),
            np.nan,
            dtype=float,
        )
        for candidate_index, evaluation in enumerate(evaluations):
            for member, trajectory in enumerate(evaluation.trajectories):
                if trajectory is not None:
                    result[candidate_index, member] = getattr(
                        trajectory, name
                    )
        return result

    pid_fields = tuple(PIDConfig.__dataclass_fields__)
    base_pid = np.asarray(
        [
            [getattr(value, field) for field in pid_fields]
            for value in source.controller_configuration.pid
        ]
    )
    applied = tuple(
        apply_controller_candidate(
            source.controller_configuration,
            source.controller_parameters,
            candidate,
        )
        for candidate in candidates
    )
    candidate_controller_mass = np.asarray(
        [parameters.mass for _configuration, parameters in applied]
    )
    candidate_pid = np.asarray(
        [
            [
                [getattr(value, field) for field in pid_fields]
                for value in configuration.pid
            ]
            for configuration, _parameters in applied
        ]
    )
    initial_actuator_present = np.asarray(
        [value is not None for value in source.initial_actuator_states],
        dtype=bool,
    )
    initial_actuator_thrust = np.asarray(
        [
            np.zeros(4) if value is None else value.thrust
            for value in source.initial_actuator_states
        ]
    )
    initial_actuator_gimbal = np.asarray(
        [
            np.zeros(4) if value is None else value.gimbal_angle
            for value in source.initial_actuator_states
        ]
    )
    actuator_fields = tuple(ActuatorParameters.__dataclass_fields__)
    actuator_values = np.asarray(
        [getattr(source.actuator_parameters, name) for name in actuator_fields]
    )
    payload = {
        "schema": np.asarray(
            ("grape-weak-constraint/phase6-posterior-predictive",)
        ),
        "selected_mode_id": np.asarray((source.selected_mode_id,)),
        "source_mode_id": np.asarray(source.source_mode_ids),
        "source_mode_weight": source.source_mode_weights,
        "mode_conditioning_source": np.asarray(
            (source.mode_conditioning_source,)
        ),
        "scenario_assumption": np.asarray((source.scenario_assumption,)),
        "posterior_member_id": source.member_ids,
        "posterior_residual_wrench_interval": (
            source.interval_residual_wrench
        ),
        "provenance_key": provenance_keys,
        "provenance_value": provenance_values,
        "candidate_id": np.asarray(
            [value.candidate_id for value in candidates]
        ),
        "candidate_scale_name": np.asarray(
            ("controller_mass", "roll_pid", "pitch_pid", "yaw_pid")
        ),
        "candidate_scale": np.asarray([value.scales for value in candidates]),
        "candidate_controller_mass": candidate_controller_mass,
        "candidate_pid_field": np.asarray(pid_fields),
        "candidate_pid": candidate_pid,
        "selected_candidate_index": np.asarray(
            (decision.selected_candidate_index,), dtype=np.int64
        ),
        "recommendation_available": np.asarray(
            (decision.recommendation_available,), dtype=bool
        ),
        "selected_candidate_id": np.asarray(
            (
                decision.selected_candidate.candidate_id
                if decision.recommendation_available
                else "",
            )
        ),
        "failure_threshold": np.asarray((decision.failure_threshold,)),
        "cvar_level": np.asarray((decision.cvar_level,)),
        "tracking_translation_scale": np.asarray(
            (decision.loss_definition.translation_scale,)
        ),
        "tracking_rotation_scale": np.asarray(
            (decision.loss_definition.rotation_scale,)
        ),
        "decision_weight_name": np.asarray(
            (
                "mean_tracking_loss",
                "cvar_tracking_loss",
                "failure_probability",
                "parameter_change",
            )
        ),
        "decision_weight": np.asarray(
            (
                decision.weights.mean_tracking_loss,
                decision.weights.cvar_tracking_loss,
                decision.weights.failure_probability,
                decision.weights.parameter_change,
            )
        ),
        "member_tracking_loss": np.asarray(
            [value.tracking_loss for value in evaluations]
        ),
        "forecast_success": np.asarray(
            [value.forecast_success for value in evaluations], dtype=bool
        ),
        "forecast_failure_reason": np.asarray(
            [value.forecast_failure_reason for value in evaluations]
        ),
        "numerical_failure_tracking_loss_convention": np.asarray(
            (
                "assign_failure_threshold_retain_nan_path_"
                "and_disqualify_candidate",
            )
        ),
        "mean_tracking_loss": np.asarray(
            [value.mean_tracking_loss for value in evaluations]
        ),
        "cvar_tracking_loss": np.asarray(
            [value.cvar_tracking_loss for value in evaluations]
        ),
        "failure_probability": np.asarray(
            [value.failure_probability for value in evaluations]
        ),
        "parameter_change_magnitude": np.asarray(
            [value.parameter_change_magnitude for value in evaluations]
        ),
        "change_penalty": np.asarray(
            [value.change_penalty for value in evaluations]
        ),
        "decision_score": np.asarray(
            [value.decision_score for value in evaluations]
        ),
        "correction_translation": np.asarray(
            [value.correction_translation for value in evaluations]
        ),
        "correction_rotation_vector": np.asarray(
            [value.correction_rotation_vector for value in evaluations]
        ),
        "prediction_position": trajectory_field("position"),
        "prediction_orientation_xyzw": trajectory_field(
            "orientation_xyzw"
        ),
        "prediction_linear_velocity": trajectory_field("linear_velocity"),
        "prediction_angular_velocity": trajectory_field("angular_velocity"),
        "prediction_controller_integral": trajectory_field(
            "controller_integral"
        ),
        "prediction_commanded_thrust": trajectory_field(
            "commanded_thrust"
        ),
        "prediction_commanded_gimbal_angle": trajectory_field(
            "commanded_gimbal_angle"
        ),
        "prediction_actuator_thrust": trajectory_field("actuator_thrust"),
        "prediction_actuator_gimbal_angle": trajectory_field(
            "actuator_gimbal_angle"
        ),
        "prediction_body_wrench": trajectory_field("body_wrench"),
        "times": source.times,
        "reference_position": np.asarray(
            [value.position for value in source.references]
        ),
        "reference_linear_velocity": np.asarray(
            [value.linear_velocity for value in source.references]
        ),
        "reference_linear_acceleration": np.asarray(
            [value.linear_acceleration for value in source.references]
        ),
        "reference_rpy": np.asarray(
            [value.rpy for value in source.references]
        ),
        "reference_angular_velocity": np.asarray(
            [value.angular_velocity for value in source.references]
        ),
        "reference_angular_acceleration": np.asarray(
            [value.angular_acceleration for value in source.references]
        ),
        "member_initial_position": np.asarray(
            [value.position for value in source.initial_states]
        ),
        "member_initial_orientation_xyzw": np.asarray(
            [value.orientation_xyzw for value in source.initial_states]
        ),
        "member_initial_linear_velocity": np.asarray(
            [value.linear_velocity for value in source.initial_states]
        ),
        "member_initial_angular_velocity": np.asarray(
            [value.angular_velocity for value in source.initial_states]
        ),
        "member_initial_controller_integral": np.asarray(
            [
                value.integral_error
                for value in source.initial_controller_states
            ]
        ),
        "member_initial_roll_pitch_integration_active": np.asarray(
            [
                value.roll_pitch_integration_active
                for value in source.initial_controller_states
            ],
            dtype=bool,
        ),
        "member_initial_actuator_present": initial_actuator_present,
        "member_initial_actuator_thrust": initial_actuator_thrust,
        "member_initial_actuator_gimbal_angle": initial_actuator_gimbal,
        "base_controller_mass": np.asarray(
            (source.controller_parameters.mass,)
        ),
        "base_controller_inertia": source.controller_parameters.inertia,
        "base_controller_cog_offset": source.controller_parameters.cog_offset,
        "base_controller_force_effectiveness": (
            source.controller_parameters.force_effectiveness
        ),
        "base_controller_torque_effectiveness": (
            source.controller_parameters.torque_effectiveness
        ),
        "base_controller_linear_drag": (
            source.controller_parameters.linear_drag
        ),
        "base_controller_angular_drag": (
            source.controller_parameters.angular_drag
        ),
        "base_pid_field": np.asarray(pid_fields),
        "base_pid": base_pid,
        "controller_xy_control_mode": np.asarray(
            (source.controller_configuration.xy_control_mode,)
        ),
        "controller_need_yaw_d_control": np.asarray(
            (source.controller_configuration.need_yaw_d_control,), dtype=bool
        ),
        "controller_start_roll_pitch_integration_height": np.asarray(
            (
                source.controller_configuration
                .start_roll_pitch_integration_height,
            )
        ),
        "controller_initial_height": np.asarray(
            (source.controller_configuration.initial_height,)
        ),
        "controller_source_compatible_gyro_term": np.asarray(
            (source.controller_configuration.source_compatible_gyro_term,),
            dtype=bool,
        ),
        "actuator_parameter_name": np.asarray(actuator_fields),
        "actuator_parameter": actuator_values,
        "controller_geometry_rotor_origins": (
            source.controller_geometry.rotor_origins
        ),
        "controller_geometry_arm_yaws": source.controller_geometry.arm_yaws,
        "controller_geometry_rotor_directions": (
            source.controller_geometry.rotor_directions
        ),
        "controller_geometry_moment_force_rate": np.asarray(
            (source.controller_geometry.moment_force_rate,)
        ),
        "controller_geometry_thrust_offset": np.asarray(
            (source.controller_geometry.thrust_offset,)
        ),
        "plant_geometry_rotor_origins": source.plant_geometry.rotor_origins,
        "plant_geometry_arm_yaws": source.plant_geometry.arm_yaws,
        "plant_geometry_rotor_directions": (
            source.plant_geometry.rotor_directions
        ),
        "plant_geometry_moment_force_rate": np.asarray(
            (source.plant_geometry.moment_force_rate,)
        ),
        "plant_geometry_thrust_offset": np.asarray(
            (source.plant_geometry.thrust_offset,)
        ),
    }
    payload.update(_physical_member_payload(source.physical_parameter_members))
    np.savez_compressed(str(destination), **payload)
    return destination


__all__ = [
    "CandidateEvaluation",
    "ControllerParameterCandidate",
    "PosteriorPredictiveDecision",
    "PosteriorPredictiveInput",
    "PosteriorPredictiveWeights",
    "TrackingLossDefinition",
    "apply_controller_candidate",
    "default_controller_parameter_candidates",
    "empirical_upper_cvar",
    "evaluate_posterior_predictive",
    "input_from_mode_posterior",
    "input_from_phase5_artifact",
    "input_from_real_assimilation",
    "save_posterior_predictive_decision",
]
