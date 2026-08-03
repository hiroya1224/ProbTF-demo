"""Leakage-resistant held-out posterior validation on an unseen flight.

The forecast and scoring APIs are deliberately separate.  A
``HeldOutForecastScenario`` contains no held-out pose observations: those are
available only while deriving the leading-sample pose/velocity anchor and
later while scoring full closed-loop forecasts with no observation resets.
No residual wrench is calibrated or replayed; every forecast receives an
explicitly zero interval wrench.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.artifact_io import (
    ArtifactValidationError,
    AssimilationRunBundle,
    load_assimilation_run,
    read_json,
    request_fingerprint,
    write_json_atomic,
)
from grape_param_estim.controller import ControllerConfig, GrapeController
from grape_param_estim.dynamics import FullSixDofPlant, simulate_closed_loop
from grape_param_estim.geometry import (
    euler_xyz_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_vector_from_matrix,
)
from grape_param_estim.posterior_predictive import (
    PHYSICAL_METRICS,
    time_integrated_error_metrics,
)
from grape_param_estim.progress import CancellationToken
from grape_param_estim.real_calibration import pose_derived_initial_state
from grape_param_estim.real_rosbag import (
    PID_AXIS_NAMES,
    PID_CONFIG_FIELD_NAMES,
    ControllerGainSnapshot,
    EpisodeProvenance,
    RealFlightEpisode,
)
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


HELD_OUT_VALIDATION_REQUEST_SCHEMA = (
    "grape-param-estim/held-out-validation-request/v1"
)
HELD_OUT_VALIDATION_SCHEMA = "grape-param-estim/held-out-validation/v1"
HELD_OUT_VALIDATION_ARRAY_SCHEMA = (
    "grape-param-estim/held-out-validation-arrays/v1"
)
WRITING_STATUS = "writing"
COMPLETE_STATUS = "complete"
MANIFEST_NAME = "manifest.json"
ARRAY_NAME = "validation.npz"
RESIDUAL_POLICY = "zero"

METRIC_NAMES = tuple(
    "observed_{}".format(value) for value in PHYSICAL_METRICS
) + tuple("reference_{}".format(value) for value in PHYSICAL_METRICS)

TRAJECTORY_FIELDS = (
    "position",
    "orientation_xyzw",
    "linear_velocity",
    "angular_velocity",
    "controller_integral",
    "commanded_thrust",
    "commanded_gimbal_angle",
    "actuator_thrust",
    "actuator_gimbal_angle",
    "body_wrench",
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _strict_keys(value, required, optional, location):
    if not isinstance(value, dict):
        raise ValueError("{} must be an object".format(location))
    missing = set(required) - set(value)
    unknown = set(value) - set(required) - set(optional)
    if missing:
        raise ValueError(
            "{} is missing: {}".format(location, ", ".join(sorted(missing)))
        )
    if unknown:
        raise ValueError(
            "{} has unknown fields: {}".format(
                location, ", ".join(sorted(unknown))
            )
        )


def _safe_identifier(value: Any, location: str) -> str:
    result = str(value)
    if not _IDENTIFIER.match(result):
        raise ValueError("{} is not a safe identifier".format(location))
    return result


def _finite_interval(value: Any, location: str) -> Tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("{} must be [start, end]".format(location))
    if any(isinstance(item, bool) for item in value):
        raise ValueError("{} bounds must be numeric".format(location))
    try:
        result = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as error:
        raise ValueError("{} bounds must be numeric".format(location)) from error
    if not np.all(np.isfinite(result)) or result[1] <= result[0]:
        raise ValueError("{} must contain finite increasing bounds".format(location))
    return result


@dataclass(frozen=True)
class HeldOutBagRequest:
    bag_id: str
    path: str
    sha256: str
    episode_index: int
    selected_interval: Tuple[float, float]
    window_state: Optional[int]
    configuration_fingerprint: str
    configuration_provenance: Tuple[Tuple[str, str], ...] = tuple()

    def __post_init__(self) -> None:
        bag_id = _safe_identifier(self.bag_id, "held_out_bag.bag_id")
        path = str(self.path)
        sha256 = str(self.sha256)
        fingerprint = str(self.configuration_fingerprint)
        if not path:
            raise ValueError("held-out bag path cannot be empty")
        if not _SHA256.match(sha256):
            raise ValueError("held-out bag SHA-256 must be lowercase hexadecimal")
        if isinstance(self.episode_index, bool):
            raise ValueError("episode_index must be an integer")
        episode_index = int(self.episode_index)
        if episode_index != self.episode_index or episode_index < 0:
            raise ValueError("episode_index must be a non-negative integer")
        interval = tuple(float(value) for value in self.selected_interval)
        if (
            len(interval) != 2
            or not np.all(np.isfinite(interval))
            or interval[1] <= interval[0]
        ):
            raise ValueError("selected_interval must be finite and increasing")
        state = self.window_state
        if state is not None:
            if isinstance(state, bool) or int(state) != state:
                raise ValueError("window_state must be an integer or null")
            state = int(state)
        if not fingerprint:
            raise ValueError("configuration_fingerprint cannot be empty")
        provenance = tuple(
            (str(key), str(value))
            for key, value in self.configuration_provenance
        )
        if any(not key or not value for key, value in provenance):
            raise ValueError("configuration provenance entries cannot be empty")
        if len({key for key, _value in provenance}) != len(provenance):
            raise ValueError("configuration provenance keys must be unique")
        object.__setattr__(self, "bag_id", bag_id)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "sha256", sha256)
        object.__setattr__(self, "episode_index", episode_index)
        object.__setattr__(self, "selected_interval", interval)
        object.__setattr__(self, "window_state", state)
        object.__setattr__(self, "configuration_fingerprint", fingerprint)
        object.__setattr__(self, "configuration_provenance", provenance)


@dataclass(frozen=True)
class HeldOutValidationRequest:
    validation_id: str
    assimilation_run: str
    held_out_bag: HeldOutBagRequest
    sample_period: float

    def __post_init__(self) -> None:
        identifier = _safe_identifier(self.validation_id, "validation_id")
        run = str(self.assimilation_run)
        period = float(self.sample_period)
        if not run:
            raise ValueError("assimilation_run cannot be empty")
        if not isinstance(self.held_out_bag, HeldOutBagRequest):
            raise TypeError("held_out_bag has the wrong type")
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("sample_period must be finite and positive")
        object.__setattr__(self, "validation_id", identifier)
        object.__setattr__(self, "assimilation_run", run)
        object.__setattr__(self, "sample_period", period)


def load_held_out_validation_request(
    path: str,
) -> Tuple[HeldOutValidationRequest, Mapping[str, Any]]:
    """Strictly parse and resolve one held-out validation request."""

    source = Path(path).expanduser().resolve()
    value = read_json(source)
    _strict_keys(
        value,
        (
            "schema",
            "validation_id",
            "assimilation_run",
            "held_out_bag",
            "settings",
        ),
        tuple(),
        "held-out validation request",
    )
    if value["schema"] != HELD_OUT_VALIDATION_REQUEST_SCHEMA:
        raise ValueError("unsupported held-out validation request schema")
    raw_bag = value["held_out_bag"]
    _strict_keys(
        raw_bag,
        (
            "bag_id",
            "path",
            "sha256",
            "episode_index",
            "selected_interval",
            "window_state",
            "configuration_fingerprint",
        ),
        ("configuration_provenance",),
        "held_out_bag",
    )
    raw_provenance = raw_bag.get("configuration_provenance", {})
    if not isinstance(raw_provenance, dict):
        raise ValueError("configuration_provenance must be an object")
    settings = value["settings"]
    _strict_keys(settings, ("sample_period",), tuple(), "settings")
    run = Path(str(value["assimilation_run"])).expanduser()
    if not run.is_absolute():
        run = source.parent / run
    bag_path = Path(str(raw_bag["path"])).expanduser()
    if not bag_path.is_absolute():
        bag_path = source.parent / bag_path
    request = HeldOutValidationRequest(
        validation_id=value["validation_id"],
        assimilation_run=str(run.resolve()),
        held_out_bag=HeldOutBagRequest(
            bag_id=raw_bag["bag_id"],
            path=str(bag_path.resolve()),
            sha256=raw_bag["sha256"],
            episode_index=raw_bag["episode_index"],
            selected_interval=_finite_interval(
                raw_bag["selected_interval"],
                "held_out_bag.selected_interval",
            ),
            window_state=raw_bag["window_state"],
            configuration_fingerprint=raw_bag[
                "configuration_fingerprint"
            ],
            configuration_provenance=tuple(raw_provenance.items()),
        ),
        sample_period=settings["sample_period"],
    )
    return request, value


@dataclass(frozen=True)
class RawPhysicalPosterior:
    """Only the cross-flight physical law permitted to leave the source run."""

    member_id: np.ndarray
    mass: np.ndarray
    inertia: np.ndarray
    cog: np.ndarray
    force_effectiveness: np.ndarray
    torque_effectiveness: np.ndarray
    constant_delay: np.ndarray
    source_run_id: str
    source_configuration_fingerprint: str
    source_root: str

    def __post_init__(self) -> None:
        member_id = np.asarray(self.member_id, dtype=np.int64)
        count = member_id.size
        expected = {
            "mass": (count,),
            "inertia": (count, 3, 3),
            "cog": (count, 3),
            "force_effectiveness": (count, 4),
            "torque_effectiveness": (count, 4),
            "constant_delay": (count,),
        }
        if (
            member_id.ndim != 1
            or count < 1
            or np.unique(member_id).size != count
            or np.any(member_id < 0)
        ):
            raise ValueError("member_id must be unique and non-negative")
        object.__setattr__(self, "member_id", member_id.copy())
        for name, shape in expected.items():
            selected = np.asarray(getattr(self, name), dtype=float)
            if selected.shape != shape or np.any(~np.isfinite(selected)):
                raise ValueError("{} must be finite and aligned".format(name))
            object.__setattr__(self, name, selected.copy())
        if (
            np.any(self.mass <= 0.0)
            or np.any(self.force_effectiveness <= 0.0)
            or np.any(self.torque_effectiveness <= 0.0)
            or np.any(self.constant_delay < 0.0)
        ):
            raise ValueError("physical posterior contains invalid members")
        for matrix in self.inertia:
            if (
                not np.allclose(matrix, matrix.T, atol=1.0e-10)
                or np.any(np.linalg.eigvalsh(matrix) <= 0.0)
            ):
                raise ValueError("posterior inertia must be positive definite")
        for name in (
            "source_run_id",
            "source_configuration_fingerprint",
            "source_root",
        ):
            if not str(getattr(self, name)):
                raise ValueError("{} cannot be empty".format(name))

    @property
    def member_count(self) -> int:
        return int(self.member_id.size)

    def vehicle(self, index: int) -> VehicleParameters:
        nominal = VehicleParameters.nominal()
        return VehicleParameters(
            mass=self.mass[index],
            inertia=self.inertia[index],
            cog_offset=self.cog[index],
            force_effectiveness=self.force_effectiveness[index],
            torque_effectiveness=self.torque_effectiveness[index],
            linear_drag=nominal.linear_drag,
            angular_drag=nominal.angular_drag,
        )


def raw_physical_posterior_from_run(
    bundle: AssimilationRunBundle,
) -> RawPhysicalPosterior:
    """Extract the shared raw members and continuous delay, and nothing local."""

    if not isinstance(bundle, AssimilationRunBundle):
        raise TypeError("bundle must be an AssimilationRunBundle")
    arrays = bundle.shared_posterior
    return RawPhysicalPosterior(
        member_id=arrays["member_id"],
        mass=arrays["mass"],
        inertia=arrays["inertia"],
        cog=arrays["cog"],
        force_effectiveness=arrays["force_effectiveness"],
        torque_effectiveness=arrays["torque_effectiveness"],
        constant_delay=arrays["constant_delay"],
        source_run_id=str(bundle.manifest["run_id"]),
        source_configuration_fingerprint=str(
            bundle.manifest["configuration_fingerprint"]
        ),
        source_root=str(bundle.root),
    )


@dataclass(frozen=True)
class HeldOutForecastScenario:
    """Forecast-only successful-flight inputs; deliberately no observations."""

    bag_id: str
    times: np.ndarray
    record_times: np.ndarray
    references: Tuple[ReferenceState, ...]
    initial_state: RigidBodyState
    initial_controller_state: ControllerState
    initial_actuator_state: ActuatorState
    controller_configuration: ControllerConfig
    controller_snapshot: ControllerGainSnapshot
    provenance: EpisodeProvenance
    base_actuator_parameters: ActuatorParameters = ActuatorParameters()
    controller_nominal_parameters: VehicleParameters = VehicleParameters.nominal()
    geometry: GrapeGeometry = GrapeGeometry.grape()

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        record_times = np.asarray(self.record_times, dtype=float)
        references = tuple(self.references)
        if (
            times.ndim != 1
            or times.size < 2
            or record_times.shape != times.shape
            or np.any(~np.isfinite(times))
            or np.any(~np.isfinite(record_times))
            or np.any(np.diff(times) <= 0.0)
            or np.any(np.diff(record_times) <= 0.0)
            or len(references) != times.size
            or any(not isinstance(value, ReferenceState) for value in references)
        ):
            raise ValueError("scenario time and reference paths must align")
        if not str(self.bag_id):
            raise ValueError("bag_id cannot be empty")
        typed = (
            (self.initial_state, RigidBodyState),
            (self.initial_controller_state, ControllerState),
            (self.initial_actuator_state, ActuatorState),
            (self.controller_configuration, ControllerConfig),
            (self.controller_snapshot, ControllerGainSnapshot),
            (self.provenance, EpisodeProvenance),
            (self.base_actuator_parameters, ActuatorParameters),
            (self.controller_nominal_parameters, VehicleParameters),
            (self.geometry, GrapeGeometry),
        )
        if any(not isinstance(value, expected) for value, expected in typed):
            raise TypeError("scenario contains an invalid typed input")
        if self.base_actuator_parameters.delay != 0.0:
            raise ValueError("base actuator delay must be zero; tau is member-local")
        object.__setattr__(self, "bag_id", str(self.bag_id))
        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "record_times", record_times.copy())
        object.__setattr__(self, "references", references)

    @property
    def reference_position(self) -> np.ndarray:
        return np.asarray([value.position for value in self.references])

    @property
    def reference_orientation_xyzw(self) -> np.ndarray:
        return np.asarray(
            [
                matrix_to_quaternion(euler_xyz_to_matrix(value.rpy))
                for value in self.references
            ]
        )


@dataclass(frozen=True)
class HeldOutEvaluationTarget:
    """Pose observations available to scoring, never to the forecast operator."""

    times: np.ndarray
    observed_position: np.ndarray
    observed_orientation_xyzw: np.ndarray

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        position = np.asarray(self.observed_position, dtype=float)
        orientation = np.asarray(self.observed_orientation_xyzw, dtype=float)
        if (
            times.ndim != 1
            or times.size < 2
            or np.any(~np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
            or position.shape != (times.size, 3)
            or orientation.shape != (times.size, 4)
            or np.any(~np.isfinite(position))
            or np.any(~np.isfinite(orientation))
        ):
            raise ValueError("evaluation observations must be finite and aligned")
        for quaternion in orientation:
            quaternion_to_matrix(quaternion)
        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "observed_position", position.copy())
        object.__setattr__(
            self, "observed_orientation_xyzw", orientation.copy()
        )


def prepare_held_out_episode(
    bag_id: str, episode: RealFlightEpisode
) -> Tuple[HeldOutForecastScenario, HeldOutEvaluationTarget]:
    """Use held-out pose once for its initial-state anchor and later scoring."""

    if not isinstance(episode, RealFlightEpisode):
        raise TypeError("episode must be a RealFlightEpisode")
    observations = episode.observations
    initial_state = pose_derived_initial_state(
        observations.times,
        observations.position,
        observations.orientation_xyzw,
    )
    scenario = HeldOutForecastScenario(
        bag_id=str(bag_id),
        times=observations.times,
        record_times=episode.record_times,
        references=episode.references,
        initial_state=initial_state,
        initial_controller_state=episode.initial_controller_state,
        initial_actuator_state=episode.initial_actuator_state,
        controller_configuration=episode.controller_configuration,
        controller_snapshot=episode.controller_snapshot,
        provenance=episode.provenance,
    )
    target = HeldOutEvaluationTarget(
        observations.times,
        observations.position,
        observations.orientation_xyzw,
    )
    return scenario, target


def _empty_trajectory_arrays(member_count: int, samples: int):
    widths = {
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
    return {
        name: np.full((member_count, samples, width), np.nan)
        for name, width in widths.items()
    }


def _copy_trajectory(target, index: int, trajectory: ClosedLoopTrajectory):
    for name in TRAJECTORY_FIELDS:
        target[name][index] = getattr(trajectory, name)


@dataclass(frozen=True)
class HeldOutForecastEnsemble:
    member_id: np.ndarray
    completed: np.ndarray
    failure_reason: np.ndarray
    trajectories: Mapping[str, np.ndarray]
    nominal_completed: bool
    nominal_failure_reason: str
    nominal_trajectory: Mapping[str, np.ndarray]
    explicit_zero_residual_wrench: np.ndarray


def forecast_held_out_posterior(
    posterior: RawPhysicalPosterior,
    scenario: HeldOutForecastScenario,
    cancellation_token: Optional[CancellationToken] = None,
) -> HeldOutForecastEnsemble:
    """Forecast all raw members and a nominal baseline with Q identically zero."""

    if not isinstance(posterior, RawPhysicalPosterior):
        raise TypeError("posterior must be RawPhysicalPosterior")
    if not isinstance(scenario, HeldOutForecastScenario):
        raise TypeError("scenario must be HeldOutForecastScenario")
    cancellation = CancellationToken() if cancellation_token is None else cancellation_token
    if not isinstance(cancellation, CancellationToken):
        raise TypeError("cancellation_token must be a CancellationToken")
    member_count = posterior.member_count
    samples = scenario.times.size
    trajectories = _empty_trajectory_arrays(member_count, samples)
    completed = np.zeros(member_count, dtype=bool)
    reasons = [""] * member_count
    zero_residual = np.zeros((samples - 1, 6), dtype=float)
    articulated = GrapeArticulatedModel()

    def simulate(parameters, delay):
        return simulate_closed_loop(
            times=scenario.times,
            references=scenario.references,
            initial_state=scenario.initial_state,
            initial_controller_state=scenario.initial_controller_state,
            controller=GrapeController(
                scenario.controller_configuration,
                scenario.controller_nominal_parameters,
                scenario.geometry,
                articulated_model=articulated,
            ),
            plant=FullSixDofPlant(parameters, scenario.geometry),
            actuator_parameters=replace(
                scenario.base_actuator_parameters, delay=float(delay)
            ),
            initial_actuator_state=scenario.initial_actuator_state,
            interval_residual_wrench=zero_residual,
        )

    for index in range(member_count):
        cancellation.raise_if_cancelled()
        try:
            trajectory = simulate(
                posterior.vehicle(index), posterior.constant_delay[index]
            )
            _copy_trajectory(trajectories, index, trajectory)
            completed[index] = True
        except Exception as error:  # numerical failures are data, not aborts
            reasons[index] = "{}: {}".format(type(error).__name__, error)

    cancellation.raise_if_cancelled()
    nominal_arrays = _empty_trajectory_arrays(1, samples)
    nominal_completed = False
    nominal_reason = ""
    try:
        nominal = simulate(VehicleParameters.nominal(), 0.0)
        _copy_trajectory(nominal_arrays, 0, nominal)
        nominal_completed = True
    except Exception as error:
        nominal_reason = "{}: {}".format(type(error).__name__, error)
    return HeldOutForecastEnsemble(
        member_id=posterior.member_id.copy(),
        completed=completed,
        failure_reason=np.asarray(reasons, dtype=str),
        trajectories=trajectories,
        nominal_completed=nominal_completed,
        nominal_failure_reason=nominal_reason,
        nominal_trajectory={
            name: value[0] for name, value in nominal_arrays.items()
        },
        explicit_zero_residual_wrench=zero_residual,
    )


def _pose_error_paths(position, orientation, target_position, target_orientation):
    position_error = np.asarray(position) - np.asarray(target_position)
    orientation_error = np.empty_like(position_error)
    for index in range(position_error.shape[0]):
        target_rotation = quaternion_to_matrix(target_orientation[index])
        predicted_rotation = quaternion_to_matrix(orientation[index])
        orientation_error[index] = rotation_vector_from_matrix(
            target_rotation.T @ predicted_rotation
        )
    return position_error, orientation_error


def _metric_row(times, observed_errors, reference_errors):
    observed = time_integrated_error_metrics(
        times, observed_errors[0], observed_errors[1]
    )
    reference = time_integrated_error_metrics(
        times, reference_errors[0], reference_errors[1]
    )
    return np.asarray(
        [getattr(observed, name) for name in PHYSICAL_METRICS]
        + [getattr(reference, name) for name in PHYSICAL_METRICS]
    )


@dataclass(frozen=True)
class HeldOutValidationResult:
    forecasts: HeldOutForecastEnsemble
    observed_position_error: np.ndarray
    observed_orientation_error: np.ndarray
    reference_position_error: np.ndarray
    reference_orientation_error: np.ndarray
    metrics: np.ndarray
    nominal_observed_position_error: np.ndarray
    nominal_observed_orientation_error: np.ndarray
    nominal_reference_position_error: np.ndarray
    nominal_reference_orientation_error: np.ndarray
    nominal_metrics: np.ndarray


def score_held_out_forecasts(
    forecasts: HeldOutForecastEnsemble,
    scenario: HeldOutForecastScenario,
    target: HeldOutEvaluationTarget,
) -> HeldOutValidationResult:
    """Score forecasts against observations and references as separate targets."""

    if not isinstance(forecasts, HeldOutForecastEnsemble):
        raise TypeError("forecasts has the wrong type")
    if not isinstance(scenario, HeldOutForecastScenario):
        raise TypeError("scenario has the wrong type")
    if not isinstance(target, HeldOutEvaluationTarget):
        raise TypeError("target has the wrong type")
    if not np.array_equal(target.times, scenario.times):
        raise ValueError("evaluation and forecast time grids must be identical")
    members = forecasts.member_id.size
    samples = scenario.times.size
    observed_position = np.full((members, samples, 3), np.nan)
    observed_orientation = np.full((members, samples, 3), np.nan)
    reference_position = np.full((members, samples, 3), np.nan)
    reference_orientation = np.full((members, samples, 3), np.nan)
    metrics = np.full((members, len(METRIC_NAMES)), np.nan)
    reference_position_path = scenario.reference_position
    reference_orientation_path = scenario.reference_orientation_xyzw
    for index in np.flatnonzero(forecasts.completed):
        prediction_position = forecasts.trajectories["position"][index]
        prediction_orientation = forecasts.trajectories[
            "orientation_xyzw"
        ][index]
        observed_errors = _pose_error_paths(
            prediction_position,
            prediction_orientation,
            target.observed_position,
            target.observed_orientation_xyzw,
        )
        reference_errors = _pose_error_paths(
            prediction_position,
            prediction_orientation,
            reference_position_path,
            reference_orientation_path,
        )
        observed_position[index], observed_orientation[index] = observed_errors
        reference_position[index], reference_orientation[index] = reference_errors
        metrics[index] = _metric_row(
            scenario.times, observed_errors, reference_errors
        )

    nominal_observed_position = np.full((samples, 3), np.nan)
    nominal_observed_orientation = np.full((samples, 3), np.nan)
    nominal_reference_position = np.full((samples, 3), np.nan)
    nominal_reference_orientation = np.full((samples, 3), np.nan)
    nominal_metrics = np.full((len(METRIC_NAMES),), np.nan)
    if forecasts.nominal_completed:
        nominal_observed = _pose_error_paths(
            forecasts.nominal_trajectory["position"],
            forecasts.nominal_trajectory["orientation_xyzw"],
            target.observed_position,
            target.observed_orientation_xyzw,
        )
        nominal_reference = _pose_error_paths(
            forecasts.nominal_trajectory["position"],
            forecasts.nominal_trajectory["orientation_xyzw"],
            reference_position_path,
            reference_orientation_path,
        )
        nominal_observed_position, nominal_observed_orientation = nominal_observed
        nominal_reference_position, nominal_reference_orientation = nominal_reference
        nominal_metrics = _metric_row(
            scenario.times, nominal_observed, nominal_reference
        )
    return HeldOutValidationResult(
        forecasts=forecasts,
        observed_position_error=observed_position,
        observed_orientation_error=observed_orientation,
        reference_position_error=reference_position,
        reference_orientation_error=reference_orientation,
        metrics=metrics,
        nominal_observed_position_error=nominal_observed_position,
        nominal_observed_orientation_error=nominal_observed_orientation,
        nominal_reference_position_error=nominal_reference_position,
        nominal_reference_orientation_error=nominal_reference_orientation,
        nominal_metrics=nominal_metrics,
    )


def _reference_arrays(references: Sequence[ReferenceState]):
    return {
        "reference_position": np.asarray([value.position for value in references]),
        "reference_linear_velocity": np.asarray(
            [value.linear_velocity for value in references]
        ),
        "reference_linear_acceleration": np.asarray(
            [value.linear_acceleration for value in references]
        ),
        "reference_rpy": np.asarray([value.rpy for value in references]),
        "reference_angular_velocity": np.asarray(
            [value.angular_velocity for value in references]
        ),
        "reference_angular_acceleration": np.asarray(
            [value.angular_acceleration for value in references]
        ),
    }


def _controller_arrays(configuration: ControllerConfig):
    return {
        "controller_pid_axis_names": np.asarray(PID_AXIS_NAMES),
        "controller_pid_field_names": np.asarray(PID_CONFIG_FIELD_NAMES),
        "controller_pid_configuration": np.asarray(
            [
                [getattr(pid, field) for field in PID_CONFIG_FIELD_NAMES]
                for pid in configuration.pid
            ]
        ),
        "controller_xy_control_mode": np.asarray(
            (configuration.xy_control_mode,)
        ),
        "controller_need_yaw_d_control": np.asarray(
            (configuration.need_yaw_d_control,), dtype=bool
        ),
        "controller_start_roll_pitch_integration_height": np.asarray(
            (configuration.start_roll_pitch_integration_height,)
        ),
        "controller_initial_height": np.asarray(
            (configuration.initial_height,)
        ),
        "controller_source_compatible_gyro_term": np.asarray(
            (configuration.source_compatible_gyro_term,), dtype=bool
        ),
    }


def _episode_provenance_arrays(provenance: EpisodeProvenance):
    scalar_fields = (
        "bag_path",
        "bag_sha256",
        "bag_size_bytes",
        "bag_record_start",
        "bag_record_end",
        "time_basis",
        "requested_window_start",
        "requested_window_end",
        "source_available_start",
        "source_available_end",
        "resample_period",
        "selected_flight_state",
        "static_window_start",
        "static_window_end",
        "static_position_samples",
        "static_position_inliers",
        "static_orientation_samples",
        "static_orientation_inliers",
        "covariance_outlier_threshold",
        "covariance_eigenvalue_floor",
        "controller_state_anchor_record_time",
        "joint_anchor_record_time",
        "thrust_anchor_record_time",
        "thrust_anchor_kind",
        "reference_acceleration_kind",
        "controller_static_source",
        "controller_source_revision",
    )
    result = {
        "provenance_{}".format(name): np.asarray((getattr(provenance, name),))
        for name in scalar_fields
    }
    for name in (
        "flight_transition_record_times",
        "flight_transition_states",
        "static_position_center",
        "static_orientation_xyzw",
        "topic_names",
        "topic_types",
    ):
        result["provenance_{}".format(name)] = np.asarray(
            getattr(provenance, name)
        )
    return result


def validation_arrays(
    posterior: RawPhysicalPosterior,
    scenario: HeldOutForecastScenario,
    target: HeldOutEvaluationTarget,
    result: HeldOutValidationResult,
) -> Dict[str, np.ndarray]:
    """Build the complete pickle-free raw result payload."""

    forecast = result.forecasts
    arrays: Dict[str, np.ndarray] = {
        "schema": np.asarray((HELD_OUT_VALIDATION_ARRAY_SCHEMA,)),
        "member_id": posterior.member_id,
        "mass": posterior.mass,
        "inertia": posterior.inertia,
        "cog": posterior.cog,
        "force_effectiveness": posterior.force_effectiveness,
        "torque_effectiveness": posterior.torque_effectiveness,
        "constant_delay": posterior.constant_delay,
        "times": scenario.times,
        "record_times": scenario.record_times,
        "observed_position": target.observed_position,
        "observed_orientation_xyzw": target.observed_orientation_xyzw,
        "initial_state": scenario.initial_state.as_vector(),
        "initial_controller_integral": (
            scenario.initial_controller_state.integral_error
        ),
        "initial_controller_roll_pitch_integration_active": np.asarray(
            (
                scenario.initial_controller_state
                .roll_pitch_integration_active,
            ),
            dtype=bool,
        ),
        "initial_actuator_thrust": scenario.initial_actuator_state.thrust,
        "initial_actuator_gimbal_angle": (
            scenario.initial_actuator_state.gimbal_angle
        ),
        "actuator_thrust_time_constant": np.asarray(
            (scenario.base_actuator_parameters.thrust_time_constant,)
        ),
        "actuator_gimbal_time_constant": np.asarray(
            (scenario.base_actuator_parameters.gimbal_time_constant,)
        ),
        "actuator_minimum_thrust": np.asarray(
            (scenario.base_actuator_parameters.minimum_thrust,)
        ),
        "actuator_maximum_thrust": np.asarray(
            (scenario.base_actuator_parameters.maximum_thrust,)
        ),
        "actuator_maximum_gimbal_angle": np.asarray(
            (scenario.base_actuator_parameters.maximum_gimbal_angle,)
        ),
        "actuator_maximum_gimbal_rate": np.asarray(
            (scenario.base_actuator_parameters.maximum_gimbal_rate,)
        ),
        "interval_residual_wrench": forecast.explicit_zero_residual_wrench,
        "forecast_completed": forecast.completed,
        "failure_reason": forecast.failure_reason,
        "metric_names": np.asarray(METRIC_NAMES),
        "metrics": result.metrics,
        "observed_position_error": result.observed_position_error,
        "observed_orientation_error": result.observed_orientation_error,
        "reference_position_error": result.reference_position_error,
        "reference_orientation_error": result.reference_orientation_error,
        "nominal_completed": np.asarray(
            (forecast.nominal_completed,), dtype=bool
        ),
        "nominal_failure_reason": np.asarray(
            (forecast.nominal_failure_reason,)
        ),
        "nominal_metrics": result.nominal_metrics,
        "nominal_observed_position_error": (
            result.nominal_observed_position_error
        ),
        "nominal_observed_orientation_error": (
            result.nominal_observed_orientation_error
        ),
        "nominal_reference_position_error": (
            result.nominal_reference_position_error
        ),
        "nominal_reference_orientation_error": (
            result.nominal_reference_orientation_error
        ),
        "snapshot_group": np.asarray(scenario.controller_snapshot.groups),
        "snapshot_record_time": scenario.controller_snapshot.record_times,
        "snapshot_gain": scenario.controller_snapshot.gains,
        "snapshot_pid_control_flag": (
            scenario.controller_snapshot.pid_control_flags
        ),
        "snapshot_source_kind": np.asarray(
            scenario.controller_snapshot.source_kinds
        ),
    }
    arrays.update(_reference_arrays(scenario.references))
    arrays.update(_controller_arrays(scenario.controller_configuration))
    arrays.update(_episode_provenance_arrays(scenario.provenance))
    for name in TRAJECTORY_FIELDS:
        arrays["posterior_{}".format(name)] = forecast.trajectories[name]
        arrays["nominal_{}".format(name)] = forecast.nominal_trajectory[name]
    return arrays


def _save_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=".{}.".format(path.name),
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as stream:
            temporary_name = stream.name
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(path))
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def save_held_out_validation(
    output: str,
    request: HeldOutValidationRequest,
    request_payload: Mapping[str, Any],
    posterior: RawPhysicalPosterior,
    scenario: HeldOutForecastScenario,
    target: HeldOutEvaluationTarget,
    result: HeldOutValidationResult,
) -> Path:
    """Atomically publish a complete, pickle-free held-out bundle."""

    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / MANIFEST_NAME
    if manifest_path.exists():
        raise FileExistsError("output already has a manifest: {}".format(manifest_path))
    common = {
        "schema": HELD_OUT_VALIDATION_SCHEMA,
        "validation_id": request.validation_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request_fingerprint": request_fingerprint(request_payload),
        "source_assimilation_run": {
            "path": posterior.source_root,
            "run_id": posterior.source_run_id,
            "configuration_fingerprint": (
                posterior.source_configuration_fingerprint
            ),
            "transferred_fields": [
                "member_id",
                "mass",
                "inertia",
                "cog",
                "force_effectiveness",
                "torque_effectiveness",
                "constant_delay",
            ],
        },
        "held_out_bag": {
            "bag_id": request.held_out_bag.bag_id,
            "path": scenario.provenance.bag_path,
            "sha256": scenario.provenance.bag_sha256,
            "episode_index": request.held_out_bag.episode_index,
            "requested_interval": list(request.held_out_bag.selected_interval),
            "actual_local_interval": [
                float(scenario.record_times[0] - scenario.provenance.bag_record_start),
                float(scenario.record_times[-1] - scenario.provenance.bag_record_start),
            ],
            "actual_record_interval": [
                float(scenario.record_times[0]),
                float(scenario.record_times[-1]),
            ],
            "window_state": request.held_out_bag.window_state,
            "configuration_fingerprint": (
                request.held_out_bag.configuration_fingerprint
            ),
        },
        "sample_period": request.sample_period,
        "member_count": posterior.member_count,
        "sample_count": int(scenario.times.size),
        "residual_policy": RESIDUAL_POLICY,
        "observation_usage": [
            "initial_pose_velocity_anchor_from_leading_pose_samples",
            "evaluation_only",
        ],
        "nominal_baseline": {
            "physical_parameters": "VehicleParameters.nominal()",
            "constant_delay_seconds": 0.0,
            "interpretation": (
                "fixed zero-delay model baseline; not a source-run posterior "
                "member and not a source-run nominal-delay estimate"
            ),
        },
        "artifacts": {"arrays": ARRAY_NAME},
    }
    writing = dict(common)
    writing["status"] = WRITING_STATUS
    write_json_atomic(manifest_path, writing)
    _save_npz_atomic(
        destination / ARRAY_NAME,
        validation_arrays(posterior, scenario, target, result),
    )
    complete = dict(common)
    complete["status"] = COMPLETE_STATUS
    complete["summary"] = {
        "forecast_completion": float(np.mean(result.forecasts.completed)),
        "numerical_failure_count": int(
            np.count_nonzero(~result.forecasts.completed)
        ),
        "nominal_completed": bool(result.forecasts.nominal_completed),
    }
    write_json_atomic(manifest_path, complete)
    return destination


@dataclass(frozen=True)
class HeldOutValidationBundle:
    root: Path
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]


_REFERENCE_ARRAY_NAMES = (
    "reference_position",
    "reference_linear_velocity",
    "reference_linear_acceleration",
    "reference_rpy",
    "reference_angular_velocity",
    "reference_angular_acceleration",
)

_CONTROLLER_ARRAY_NAMES = (
    "controller_pid_axis_names",
    "controller_pid_field_names",
    "controller_pid_configuration",
    "controller_xy_control_mode",
    "controller_need_yaw_d_control",
    "controller_start_roll_pitch_integration_height",
    "controller_initial_height",
    "controller_source_compatible_gyro_term",
)

_PROVENANCE_SCALAR_NAMES = (
    "bag_path",
    "bag_sha256",
    "bag_size_bytes",
    "bag_record_start",
    "bag_record_end",
    "time_basis",
    "requested_window_start",
    "requested_window_end",
    "source_available_start",
    "source_available_end",
    "resample_period",
    "selected_flight_state",
    "static_window_start",
    "static_window_end",
    "static_position_samples",
    "static_position_inliers",
    "static_orientation_samples",
    "static_orientation_inliers",
    "covariance_outlier_threshold",
    "covariance_eigenvalue_floor",
    "controller_state_anchor_record_time",
    "joint_anchor_record_time",
    "thrust_anchor_record_time",
    "thrust_anchor_kind",
    "reference_acceleration_kind",
    "controller_static_source",
    "controller_source_revision",
)

_PROVENANCE_PATH_NAMES = (
    "flight_transition_record_times",
    "flight_transition_states",
    "static_position_center",
    "static_orientation_xyzw",
    "topic_names",
    "topic_types",
)

_BASE_ARRAY_NAMES = {
    "schema",
    "member_id",
    "mass",
    "inertia",
    "cog",
    "force_effectiveness",
    "torque_effectiveness",
    "constant_delay",
    "times",
    "record_times",
    "observed_position",
    "observed_orientation_xyzw",
    "initial_state",
    "initial_controller_integral",
    "initial_controller_roll_pitch_integration_active",
    "initial_actuator_thrust",
    "initial_actuator_gimbal_angle",
    "actuator_thrust_time_constant",
    "actuator_gimbal_time_constant",
    "actuator_minimum_thrust",
    "actuator_maximum_thrust",
    "actuator_maximum_gimbal_angle",
    "actuator_maximum_gimbal_rate",
    "interval_residual_wrench",
    "forecast_completed",
    "failure_reason",
    "metric_names",
    "metrics",
    "observed_position_error",
    "observed_orientation_error",
    "reference_position_error",
    "reference_orientation_error",
    "nominal_completed",
    "nominal_failure_reason",
    "nominal_metrics",
    "nominal_observed_position_error",
    "nominal_observed_orientation_error",
    "nominal_reference_position_error",
    "nominal_reference_orientation_error",
    "snapshot_group",
    "snapshot_record_time",
    "snapshot_gain",
    "snapshot_pid_control_flag",
    "snapshot_source_kind",
}


def _artifact_error(message: str) -> None:
    raise ArtifactValidationError(message)


def _finite_artifact_array(arrays, name, shape):
    value = arrays[name]
    if value.shape != shape or value.dtype.kind not in "fiu" or np.any(
        ~np.isfinite(value)
    ):
        _artifact_error("{} must be a finite {} array".format(name, shape))
    return np.asarray(value, dtype=float)


def _string_artifact_array(arrays, name, shape):
    value = arrays[name]
    if value.shape != shape or value.dtype.kind not in "US":
        _artifact_error("{} must be a string {} array".format(name, shape))
    result = value.astype(str)
    if np.any(result == ""):
        _artifact_error("{} cannot contain empty strings".format(name))
    return result


def _boolean_artifact_array(arrays, name, shape):
    value = arrays[name]
    if value.shape != shape or value.dtype.kind != "b":
        _artifact_error("{} must be a boolean {} array".format(name, shape))
    return value.astype(bool)


def _validate_quaternions(value, name):
    quaternions = np.asarray(value, dtype=float)
    if (
        quaternions.shape[-1:] != (4,)
        or np.any(~np.isfinite(quaternions))
        or not np.allclose(
            np.linalg.norm(quaternions, axis=-1), 1.0, atol=1.0e-8, rtol=0.0
        )
    ):
        _artifact_error("{} contains invalid quaternions".format(name))


def _manifest_mapping(value, keys, location):
    if not isinstance(value, dict) or set(value) != set(keys):
        _artifact_error("{} keys differ from schema".format(location))
    return value


def _validate_member_score_consistency(
    arrays, member_index, times, observed_position, observed_orientation,
    reference_position, reference_orientation,
):
    prediction_position = arrays["posterior_position"][member_index]
    prediction_orientation = arrays["posterior_orientation_xyzw"][member_index]
    observed_errors = _pose_error_paths(
        prediction_position,
        prediction_orientation,
        observed_position,
        observed_orientation,
    )
    reference_errors = _pose_error_paths(
        prediction_position,
        prediction_orientation,
        reference_position,
        reference_orientation,
    )
    expected_paths = (
        ("observed_position_error", observed_errors[0]),
        ("observed_orientation_error", observed_errors[1]),
        ("reference_position_error", reference_errors[0]),
        ("reference_orientation_error", reference_errors[1]),
    )
    for name, expected in expected_paths:
        if not np.allclose(
            arrays[name][member_index], expected, atol=1.0e-10, rtol=1.0e-9
        ):
            _artifact_error("{} disagrees with posterior pose".format(name))
    expected_metrics = _metric_row(times, observed_errors, reference_errors)
    if not np.allclose(
        arrays["metrics"][member_index],
        expected_metrics,
        atol=1.0e-10,
        rtol=1.0e-9,
    ):
        _artifact_error("member metrics disagree with raw error paths")


def _validate_nominal_score_consistency(
    arrays, times, observed_position, observed_orientation,
    reference_position, reference_orientation,
):
    observed_errors = _pose_error_paths(
        arrays["nominal_position"],
        arrays["nominal_orientation_xyzw"],
        observed_position,
        observed_orientation,
    )
    reference_errors = _pose_error_paths(
        arrays["nominal_position"],
        arrays["nominal_orientation_xyzw"],
        reference_position,
        reference_orientation,
    )
    expected_paths = (
        ("nominal_observed_position_error", observed_errors[0]),
        ("nominal_observed_orientation_error", observed_errors[1]),
        ("nominal_reference_position_error", reference_errors[0]),
        ("nominal_reference_orientation_error", reference_errors[1]),
    )
    for name, expected in expected_paths:
        if not np.allclose(arrays[name], expected, atol=1.0e-10, rtol=1.0e-9):
            _artifact_error("{} disagrees with nominal pose".format(name))
    expected_metrics = _metric_row(times, observed_errors, reference_errors)
    if not np.allclose(
        arrays["nominal_metrics"],
        expected_metrics,
        atol=1.0e-10,
        rtol=1.0e-9,
    ):
        _artifact_error("nominal metrics disagree with raw error paths")


def load_held_out_validation(path: str) -> HeldOutValidationBundle:
    """Strictly load the independently registered held-out artifact schema."""

    root = Path(path).expanduser().resolve()
    manifest = read_json(root / MANIFEST_NAME)
    manifest_keys = {
        "schema",
        "validation_id",
        "created_at",
        "request_fingerprint",
        "source_assimilation_run",
        "held_out_bag",
        "sample_period",
        "member_count",
        "sample_count",
        "residual_policy",
        "observation_usage",
        "nominal_baseline",
        "artifacts",
        "status",
        "summary",
    }
    if set(manifest) != manifest_keys:
        _artifact_error("held-out manifest keys differ from schema")
    if manifest["schema"] != HELD_OUT_VALIDATION_SCHEMA:
        _artifact_error("unsupported held-out validation schema")
    if manifest["status"] != COMPLETE_STATUS:
        _artifact_error("held-out validation is not complete")
    for name in ("validation_id", "created_at", "request_fingerprint"):
        if not isinstance(manifest[name], str) or not manifest[name]:
            _artifact_error("manifest.{} must be a non-empty string".format(name))
    if manifest["artifacts"] != {"arrays": ARRAY_NAME}:
        _artifact_error("held-out artifact mapping is invalid")
    if manifest["residual_policy"] != RESIDUAL_POLICY:
        _artifact_error("held-out residual policy must be zero")
    if manifest["observation_usage"] != [
        "initial_pose_velocity_anchor_from_leading_pose_samples",
        "evaluation_only",
    ]:
        _artifact_error("held-out observation usage is invalid")
    nominal_definition = _manifest_mapping(
        manifest["nominal_baseline"],
        ("physical_parameters", "constant_delay_seconds", "interpretation"),
        "nominal_baseline",
    )
    if (
        nominal_definition["physical_parameters"]
        != "VehicleParameters.nominal()"
        or nominal_definition["constant_delay_seconds"] != 0.0
        or not isinstance(nominal_definition["interpretation"], str)
        or not nominal_definition["interpretation"]
    ):
        _artifact_error("nominal baseline definition is invalid")
    source_manifest = _manifest_mapping(
        manifest["source_assimilation_run"],
        ("path", "run_id", "configuration_fingerprint", "transferred_fields"),
        "source_assimilation_run",
    )
    transferred_fields = [
        "member_id",
        "mass",
        "inertia",
        "cog",
        "force_effectiveness",
        "torque_effectiveness",
        "constant_delay",
    ]
    if source_manifest["transferred_fields"] != transferred_fields or any(
        not isinstance(source_manifest[name], str) or not source_manifest[name]
        for name in ("path", "run_id", "configuration_fingerprint")
    ):
        _artifact_error("source assimilation provenance is invalid")
    held_out_manifest = _manifest_mapping(
        manifest["held_out_bag"],
        (
            "bag_id",
            "path",
            "sha256",
            "episode_index",
            "requested_interval",
            "actual_local_interval",
            "actual_record_interval",
            "window_state",
            "configuration_fingerprint",
        ),
        "held_out_bag",
    )
    if (
        not isinstance(held_out_manifest["bag_id"], str)
        or not held_out_manifest["bag_id"]
        or not isinstance(held_out_manifest["path"], str)
        or not held_out_manifest["path"]
        or not isinstance(held_out_manifest["sha256"], str)
        or not _SHA256.match(held_out_manifest["sha256"])
        or held_out_manifest["configuration_fingerprint"]
        != source_manifest["configuration_fingerprint"]
    ):
        _artifact_error("held-out provenance or fingerprint is invalid")
    requested_interval = _finite_interval(
        held_out_manifest["requested_interval"],
        "manifest held-out requested_interval",
    )
    actual_local_interval = _finite_interval(
        held_out_manifest["actual_local_interval"],
        "manifest held-out actual_local_interval",
    )
    actual_record_interval = _finite_interval(
        held_out_manifest["actual_record_interval"],
        "manifest held-out actual_record_interval",
    )
    for name in ("member_count", "sample_count"):
        value = manifest[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            _artifact_error("manifest.{} must be a positive integer".format(name))
    members = manifest["member_count"]
    samples = manifest["sample_count"]
    if samples < 7:
        _artifact_error("held-out artifact needs at least seven pose samples")
    sample_period = manifest["sample_period"]
    if (
        isinstance(sample_period, bool)
        or not isinstance(sample_period, (int, float))
        or not np.isfinite(sample_period)
        or sample_period <= 0.0
    ):
        _artifact_error("manifest.sample_period must be finite and positive")
    summary = _manifest_mapping(
        manifest["summary"],
        ("forecast_completion", "numerical_failure_count", "nominal_completed"),
        "summary",
    )
    try:
        with np.load(str(root / ARRAY_NAME), allow_pickle=False) as source:
            arrays = {name: source[name].copy() for name in source.files}
    except (OSError, ValueError) as error:
        raise ArtifactValidationError("cannot load validation arrays") from error
    if any(value.dtype.kind == "O" for value in arrays.values()):
        _artifact_error("validation arrays cannot use object dtype")
    expected_array_names = set(_BASE_ARRAY_NAMES)
    expected_array_names.update(_REFERENCE_ARRAY_NAMES)
    expected_array_names.update(_CONTROLLER_ARRAY_NAMES)
    expected_array_names.update(
        "provenance_{}".format(name)
        for name in _PROVENANCE_SCALAR_NAMES + _PROVENANCE_PATH_NAMES
    )
    expected_array_names.update(
        "{}_{}".format(prefix, name)
        for prefix in ("posterior", "nominal")
        for name in TRAJECTORY_FIELDS
    )
    if set(arrays) != expected_array_names:
        _artifact_error(
            "validation array keys differ from schema; missing={}, extra={}".format(
                sorted(expected_array_names - set(arrays)),
                sorted(set(arrays) - expected_array_names),
            )
        )
    schema = _string_artifact_array(arrays, "schema", (1,))
    if schema[0] != HELD_OUT_VALIDATION_ARRAY_SCHEMA:
        _artifact_error("validation array schema is invalid")
    member_id = arrays["member_id"]
    if (
        member_id.shape != (members,)
        or member_id.dtype.kind not in "iu"
        or np.any(member_id < 0)
        or np.unique(member_id).size != members
    ):
        _artifact_error("member_id must be a unique non-negative integer vector")
    physical_shapes = {
        "mass": (members,),
        "inertia": (members, 3, 3),
        "cog": (members, 3),
        "force_effectiveness": (members, 4),
        "torque_effectiveness": (members, 4),
        "constant_delay": (members,),
    }
    physical = {
        name: _finite_artifact_array(arrays, name, shape)
        for name, shape in physical_shapes.items()
    }
    try:
        RawPhysicalPosterior(
            member_id=member_id,
            source_run_id=source_manifest["run_id"],
            source_configuration_fingerprint=source_manifest[
                "configuration_fingerprint"
            ],
            source_root=source_manifest["path"],
            **physical
        )
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError("physical posterior is invalid") from error
    times = _finite_artifact_array(arrays, "times", (samples,))
    record_times = _finite_artifact_array(arrays, "record_times", (samples,))
    if (
        np.any(np.diff(times) <= 0.0)
        or np.any(np.diff(record_times) <= 0.0)
        or not np.isclose(times[0], 0.0, atol=2.0e-7, rtol=0.0)
        or not np.allclose(
            times, record_times - record_times[0], atol=2.0e-7, rtol=0.0
        )
    ):
        _artifact_error("validation time bases are inconsistent")
    observed_position = _finite_artifact_array(
        arrays, "observed_position", (samples, 3)
    )
    observed_orientation = _finite_artifact_array(
        arrays, "observed_orientation_xyzw", (samples, 4)
    )
    _validate_quaternions(observed_orientation, "observed_orientation_xyzw")
    initial_state = _finite_artifact_array(arrays, "initial_state", (13,))
    _validate_quaternions(initial_state[3:7], "initial_state orientation")
    try:
        expected_initial = pose_derived_initial_state(
            times, observed_position, observed_orientation
        ).as_vector()
    except ValueError as error:
        raise ArtifactValidationError("cannot reconstruct initial pose anchor") from error
    if not np.allclose(
        initial_state, expected_initial, atol=1.0e-9, rtol=1.0e-8
    ):
        _artifact_error("initial state is not the leading-pose-derived anchor")
    initial_integral = _finite_artifact_array(
        arrays, "initial_controller_integral", (6,)
    )
    _boolean_artifact_array(
        arrays, "initial_controller_roll_pitch_integration_active", (1,)
    )
    initial_thrust = _finite_artifact_array(
        arrays, "initial_actuator_thrust", (4,)
    )
    initial_gimbal = _finite_artifact_array(
        arrays, "initial_actuator_gimbal_angle", (4,)
    )
    actuator_values = {
        name: _finite_artifact_array(arrays, name, (1,))[0]
        for name in (
            "actuator_thrust_time_constant",
            "actuator_gimbal_time_constant",
            "actuator_minimum_thrust",
            "actuator_maximum_thrust",
            "actuator_maximum_gimbal_angle",
            "actuator_maximum_gimbal_rate",
        )
    }
    try:
        ActuatorParameters(
            thrust_time_constant=actuator_values[
                "actuator_thrust_time_constant"
            ],
            gimbal_time_constant=actuator_values[
                "actuator_gimbal_time_constant"
            ],
            delay=0.0,
            minimum_thrust=actuator_values["actuator_minimum_thrust"],
            maximum_thrust=actuator_values["actuator_maximum_thrust"],
            maximum_gimbal_angle=actuator_values[
                "actuator_maximum_gimbal_angle"
            ],
            maximum_gimbal_rate=actuator_values[
                "actuator_maximum_gimbal_rate"
            ],
        )
    except ValueError as error:
        raise ArtifactValidationError("actuator configuration is invalid") from error
    residual = _finite_artifact_array(
        arrays, "interval_residual_wrench", (samples - 1, 6)
    )
    if np.any(residual != 0.0):
        _artifact_error("held-out residual wrench must be identically zero")

    reference_arrays = {
        name: _finite_artifact_array(arrays, name, (samples, 3))
        for name in _REFERENCE_ARRAY_NAMES
    }
    reference_orientation = np.asarray(
        [
            matrix_to_quaternion(euler_xyz_to_matrix(value))
            for value in reference_arrays["reference_rpy"]
        ]
    )
    axis_names = _string_artifact_array(
        arrays, "controller_pid_axis_names", (len(PID_AXIS_NAMES),)
    )
    field_names = _string_artifact_array(
        arrays,
        "controller_pid_field_names",
        (len(PID_CONFIG_FIELD_NAMES),),
    )
    if tuple(axis_names) != PID_AXIS_NAMES or tuple(field_names) != PID_CONFIG_FIELD_NAMES:
        _artifact_error("controller PID ordering is non-canonical")
    pid_configuration = _finite_artifact_array(
        arrays,
        "controller_pid_configuration",
        (len(PID_AXIS_NAMES), len(PID_CONFIG_FIELD_NAMES)),
    )
    if np.any(pid_configuration < 0.0):
        _artifact_error("controller PID values cannot be negative")
    _string_artifact_array(arrays, "controller_xy_control_mode", (1,))
    _boolean_artifact_array(
        arrays, "controller_need_yaw_d_control", (1,)
    )
    _finite_artifact_array(
        arrays, "controller_start_roll_pitch_integration_height", (1,)
    )
    _finite_artifact_array(arrays, "controller_initial_height", (1,))
    _boolean_artifact_array(
        arrays, "controller_source_compatible_gyro_term", (1,)
    )
    snapshot_group = _string_artifact_array(arrays, "snapshot_group", (4,))
    if tuple(snapshot_group) != ("xy", "z", "roll_pitch", "yaw"):
        _artifact_error("controller snapshot group ordering is invalid")
    _finite_artifact_array(arrays, "snapshot_record_time", (4,))
    snapshot_gain = _finite_artifact_array(arrays, "snapshot_gain", (4, 3))
    if np.any(snapshot_gain < 0.0):
        _artifact_error("controller snapshot gains cannot be negative")
    _boolean_artifact_array(arrays, "snapshot_pid_control_flag", (4,))
    snapshot_source = _string_artifact_array(
        arrays, "snapshot_source_kind", (4,)
    )
    if any(
        value not in {
            "recorded_startup_parameter_update",
            "dynamic_reconfigure_applied",
        }
        for value in snapshot_source
    ):
        _artifact_error("controller snapshot source is invalid")

    for name in _PROVENANCE_SCALAR_NAMES:
        value = arrays["provenance_{}".format(name)]
        if value.shape != (1,) or value.dtype.kind == "O":
            _artifact_error("provenance_{} must be scalar".format(name))
    provenance_numeric = (
        "bag_size_bytes",
        "bag_record_start",
        "bag_record_end",
        "requested_window_start",
        "requested_window_end",
        "source_available_start",
        "source_available_end",
        "resample_period",
        "selected_flight_state",
        "static_window_start",
        "static_window_end",
        "static_position_samples",
        "static_position_inliers",
        "static_orientation_samples",
        "static_orientation_inliers",
        "covariance_outlier_threshold",
        "covariance_eigenvalue_floor",
        "controller_state_anchor_record_time",
        "joint_anchor_record_time",
        "thrust_anchor_record_time",
    )
    for name in provenance_numeric:
        _finite_artifact_array(arrays, "provenance_{}".format(name), (1,))
    for name in (
        "bag_path",
        "bag_sha256",
        "time_basis",
        "thrust_anchor_kind",
        "reference_acceleration_kind",
        "controller_static_source",
    ):
        _string_artifact_array(arrays, "provenance_{}".format(name), (1,))
    revision = arrays["provenance_controller_source_revision"]
    if revision.shape != (1,) or revision.dtype.kind not in "US":
        _artifact_error("controller source revision must be a string scalar")
    transition_times = arrays["provenance_flight_transition_record_times"]
    transition_states = arrays["provenance_flight_transition_states"]
    if (
        transition_times.ndim != 1
        or transition_times.size < 1
        or transition_states.shape != transition_times.shape
        or transition_states.dtype.kind not in "iu"
        or np.any(~np.isfinite(transition_times))
        or (transition_times.size > 1 and np.any(np.diff(transition_times) <= 0.0))
    ):
        _artifact_error("flight transition provenance is invalid")
    static_position = _finite_artifact_array(
        arrays, "provenance_static_position_center", (3,)
    )
    static_orientation = _finite_artifact_array(
        arrays, "provenance_static_orientation_xyzw", (4,)
    )
    _validate_quaternions(static_orientation, "provenance static orientation")
    del static_position
    topic_names = arrays["provenance_topic_names"]
    topic_types = arrays["provenance_topic_types"]
    if (
        topic_names.ndim != 1
        or topic_names.size < 1
        or topic_types.shape != topic_names.shape
        or topic_names.dtype.kind not in "US"
        or topic_types.dtype.kind not in "US"
        or np.any(topic_names.astype(str) == "")
        or np.any(topic_types.astype(str) == "")
    ):
        _artifact_error("topic provenance is invalid")
    bag_path = str(arrays["provenance_bag_path"][0])
    bag_sha256 = str(arrays["provenance_bag_sha256"][0])
    bag_record_start = float(arrays["provenance_bag_record_start"][0])
    if (
        bag_path != held_out_manifest["path"]
        or bag_sha256 != held_out_manifest["sha256"]
        or not _SHA256.match(bag_sha256)
        or not np.isclose(
            float(arrays["provenance_resample_period"][0]),
            sample_period,
            atol=1.0e-12,
            rtol=0.0,
        )
        or not np.allclose(
            actual_record_interval,
            (record_times[0], record_times[-1]),
            atol=2.0e-7,
            rtol=0.0,
        )
        or not np.allclose(
            actual_local_interval,
            (record_times[0] - bag_record_start, record_times[-1] - bag_record_start),
            atol=2.0e-7,
            rtol=0.0,
        )
        or not np.allclose(
            requested_interval,
            (
                float(arrays["provenance_requested_window_start"][0])
                - bag_record_start,
                float(arrays["provenance_requested_window_end"][0])
                - bag_record_start,
            ),
            atol=2.0e-7,
            rtol=0.0,
        )
    ):
        _artifact_error("manifest and episode provenance disagree")

    completed = _boolean_artifact_array(
        arrays, "forecast_completed", (members,)
    )
    failure_reason = arrays["failure_reason"]
    if failure_reason.shape != (members,) or failure_reason.dtype.kind not in "US":
        _artifact_error("failure_reason must be a string member vector")
    reasons = failure_reason.astype(str)
    metric_names = _string_artifact_array(
        arrays, "metric_names", (len(METRIC_NAMES),)
    )
    if tuple(metric_names) != METRIC_NAMES:
        _artifact_error("metric ordering is non-canonical")
    if arrays["metrics"].shape != (members, len(METRIC_NAMES)):
        _artifact_error("metrics member axes are invalid")
    error_names = (
        "observed_position_error",
        "observed_orientation_error",
        "reference_position_error",
        "reference_orientation_error",
    )
    for name in error_names:
        if arrays[name].shape != (members, samples, 3) or arrays[name].dtype.kind not in "fiu":
            _artifact_error("{} member axes are invalid".format(name))
    trajectory_widths = {
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
    for name, width in trajectory_widths.items():
        posterior_path = arrays["posterior_{}".format(name)]
        nominal_path = arrays["nominal_{}".format(name)]
        if (
            posterior_path.shape != (members, samples, width)
            or nominal_path.shape != (samples, width)
            or posterior_path.dtype.kind not in "fiu"
            or nominal_path.dtype.kind not in "fiu"
        ):
            _artifact_error("{} trajectory axes are invalid".format(name))
    for index in range(members):
        member_arrays = [arrays["metrics"][index]]
        member_arrays.extend(arrays[name][index] for name in error_names)
        member_arrays.extend(
            arrays["posterior_{}".format(name)][index]
            for name in TRAJECTORY_FIELDS
        )
        if completed[index]:
            if reasons[index] or any(np.any(~np.isfinite(value)) for value in member_arrays):
                _artifact_error("completed member contains failure data")
            _validate_quaternions(
                arrays["posterior_orientation_xyzw"][index],
                "posterior orientation",
            )
            if (
                not np.allclose(
                    arrays["posterior_position"][index, 0],
                    initial_state[:3],
                    atol=1.0e-10,
                    rtol=0.0,
                )
                or not np.allclose(
                    arrays["posterior_orientation_xyzw"][index, 0],
                    initial_state[3:7],
                    atol=1.0e-10,
                    rtol=0.0,
                )
                or not np.allclose(
                    arrays["posterior_controller_integral"][index, 0],
                    initial_integral,
                    atol=1.0e-10,
                    rtol=0.0,
                )
                or not np.allclose(
                    arrays["posterior_actuator_thrust"][index, 0],
                    initial_thrust,
                    atol=1.0e-10,
                    rtol=0.0,
                )
                or not np.allclose(
                    arrays["posterior_actuator_gimbal_angle"][index, 0],
                    initial_gimbal,
                    atol=1.0e-10,
                    rtol=0.0,
                )
            ):
                _artifact_error("posterior trajectory initial state is misaligned")
            _validate_member_score_consistency(
                arrays,
                index,
                times,
                observed_position,
                observed_orientation,
                reference_arrays["reference_position"],
                reference_orientation,
            )
        elif not reasons[index] or any(
            not np.all(np.isnan(value)) for value in member_arrays
        ):
            _artifact_error("failed member must contain only NaN paths and a reason")

    nominal_completed = _boolean_artifact_array(
        arrays, "nominal_completed", (1,)
    )[0]
    nominal_reason_array = arrays["nominal_failure_reason"]
    if nominal_reason_array.shape != (1,) or nominal_reason_array.dtype.kind not in "US":
        _artifact_error("nominal_failure_reason must be a string scalar")
    nominal_reason = str(nominal_reason_array[0])
    nominal_error_names = (
        "nominal_observed_position_error",
        "nominal_observed_orientation_error",
        "nominal_reference_position_error",
        "nominal_reference_orientation_error",
    )
    if arrays["nominal_metrics"].shape != (len(METRIC_NAMES),):
        _artifact_error("nominal metric axes are invalid")
    for name in nominal_error_names:
        if arrays[name].shape != (samples, 3) or arrays[name].dtype.kind not in "fiu":
            _artifact_error("{} axes are invalid".format(name))
    nominal_arrays = [arrays["nominal_metrics"]]
    nominal_arrays.extend(arrays[name] for name in nominal_error_names)
    nominal_arrays.extend(
        arrays["nominal_{}".format(name)] for name in TRAJECTORY_FIELDS
    )
    if nominal_completed:
        if nominal_reason or any(np.any(~np.isfinite(value)) for value in nominal_arrays):
            _artifact_error("completed nominal baseline contains failure data")
        _validate_quaternions(
            arrays["nominal_orientation_xyzw"], "nominal orientation"
        )
        if (
            not np.allclose(arrays["nominal_position"][0], initial_state[:3], atol=1.0e-10, rtol=0.0)
            or not np.allclose(arrays["nominal_orientation_xyzw"][0], initial_state[3:7], atol=1.0e-10, rtol=0.0)
            or not np.allclose(arrays["nominal_controller_integral"][0], initial_integral, atol=1.0e-10, rtol=0.0)
            or not np.allclose(arrays["nominal_actuator_thrust"][0], initial_thrust, atol=1.0e-10, rtol=0.0)
            or not np.allclose(arrays["nominal_actuator_gimbal_angle"][0], initial_gimbal, atol=1.0e-10, rtol=0.0)
        ):
            _artifact_error("nominal trajectory initial state is misaligned")
        _validate_nominal_score_consistency(
            arrays,
            times,
            observed_position,
            observed_orientation,
            reference_arrays["reference_position"],
            reference_orientation,
        )
    elif not nominal_reason or any(
        not np.all(np.isnan(value)) for value in nominal_arrays
    ):
        _artifact_error("failed nominal baseline must contain NaN paths and a reason")

    expected_completion = float(np.mean(completed))
    expected_failures = int(np.count_nonzero(~completed))
    if (
        not isinstance(summary["forecast_completion"], (int, float))
        or not np.isclose(
            summary["forecast_completion"], expected_completion, atol=1.0e-12
        )
        or isinstance(summary["numerical_failure_count"], bool)
        or summary["numerical_failure_count"] != expected_failures
        or not isinstance(summary["nominal_completed"], bool)
        or summary["nominal_completed"] != bool(nominal_completed)
    ):
        _artifact_error("manifest summary disagrees with raw member status")
    return HeldOutValidationBundle(root, manifest, arrays)


__all__ = [
    "COMPLETE_STATUS",
    "HELD_OUT_VALIDATION_ARRAY_SCHEMA",
    "HELD_OUT_VALIDATION_REQUEST_SCHEMA",
    "HELD_OUT_VALIDATION_SCHEMA",
    "HeldOutBagRequest",
    "HeldOutEvaluationTarget",
    "HeldOutForecastEnsemble",
    "HeldOutForecastScenario",
    "HeldOutValidationBundle",
    "HeldOutValidationRequest",
    "HeldOutValidationResult",
    "METRIC_NAMES",
    "RESIDUAL_POLICY",
    "RawPhysicalPosterior",
    "forecast_held_out_posterior",
    "load_held_out_validation",
    "load_held_out_validation_request",
    "prepare_held_out_episode",
    "raw_physical_posterior_from_run",
    "save_held_out_validation",
    "score_held_out_forecasts",
    "validation_arrays",
]
