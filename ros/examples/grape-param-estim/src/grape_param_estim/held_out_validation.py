"""Strict posterior validation on a flight outside the sparse batch fit.

The data-split declaration is part of both the request and the result.  A
flight used for estimator or PID tuning is called a ``tuning_evaluation`` and
can never be emitted with the ``strict_hold_out`` label.  Strict hold-out
execution additionally rejects a rosbag SHA-256 already present in the source
batch run.

Only retained MCMC draws of the static physical parameters and constant delay
leave the source estimation artifact.  The held-out flight contributes one
leading-state anchor, its recorded reference/controller streams, and pose
observations used for scoring.  Forecasts then run continuously without state
replacement.  Future model discrepancy is explicitly either zero or freshly
sampled from the estimated diagonal Q; historical residuals are not accepted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from numbers import Integral, Real
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from grape_param_estim.artifact_io import (
    ArtifactValidationError,
    read_json,
    request_fingerprint,
    write_json_atomic,
    write_npz_atomic,
)
from grape_param_estim.batch.variables import VariableKind
from grape_param_estim.batch_artifact import (
    BatchEstimationRun,
    file_sha256,
    load_batch_estimation_run,
)
from grape_param_estim.controller_config import (
    configuration_from_controller_snapshot,
)
from grape_param_estim.geometry import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    so3_geodesic_interpolation,
    so3_log,
)
from grape_param_estim.initialization import build_flight_initialization
from grape_param_estim.pid.input import physical_posterior_from_batch_run
from grape_param_estim.pid.metrics import (
    ForecastMetrics,
    empirical_lower_cvar,
    empirical_upper_cvar,
)
from grape_param_estim.pid.particle_search import (
    MODEL_DISCREPANCY_INTERVAL_MODELS,
    MODEL_DISCREPANCY_POLICIES,
    MODEL_DISCREPANCY_QUANTITIES,
    ModelDiscrepancyConfiguration,
)
from grape_param_estim.pid.predictive import (
    PidForecastInitialCondition,
    PidForecastOutcome,
    PidForecastScenario,
    run_pid_forecast,
)
from grape_param_estim.pid.proposal import (
    PhysicalPlantPosterior,
    PhysicalPlantSample,
    current_pid_candidate,
)
from grape_param_estim.progress import (
    CancellationToken,
    JsonlProgressWriter,
    ProgressCallback,
    ProgressTracker,
    STAGE_OPTIMIZING_FULL_TRAJECTORY,
    STAGE_PREPARING_TRAJECTORY,
    STAGE_WRITING_ARTIFACTS,
)
from grape_param_estim.real_rosbag import load_flight_data
from grape_param_estim.sensor_models import FlightData
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    ControllerState,
    GrapeGeometry,
    ReferenceState,
    RigidBodyState,
    VehicleParameters,
)


HELD_OUT_VALIDATION_REQUEST_SCHEMA = (
    "grape-param-estim/held-out-validation-request/v2"
)
HELD_OUT_VALIDATION_SCHEMA = "grape-param-estim/held-out-validation/v2"
HELD_OUT_VALIDATION_ARRAY_SCHEMA = (
    "grape-param-estim/held-out-validation-arrays/v2"
)
STRICT_HOLD_OUT = "strict_hold_out"
TUNING_EVALUATION = "tuning_evaluation"
DATA_SPLIT_ROLES = (STRICT_HOLD_OUT, TUNING_EVALUATION)
CONFIGURATION_COMPATIBILITY_STATUSES = (
    "manually_confirmed_same_hardware",
    "unconfirmed",
)
POSTERIOR_SUBSET_METHODS = (
    "all_equal_weight_mcmc_samples",
    "explicit_equal_weight_mcmc_subset",
)

OBSERVED_PHYSICAL_METRICS = (
    "observed_position_rmse",
    "observed_orientation_rmse",
    "observed_maximum_position_error",
    "observed_maximum_orientation_error",
)
REFERENCE_PHYSICAL_METRICS = (
    "reference_position_rmse",
    "reference_orientation_rmse",
    "reference_maximum_position_error",
    "reference_maximum_orientation_error",
)
HELD_OUT_COST_METRICS = OBSERVED_PHYSICAL_METRICS + REFERENCE_PHYSICAL_METRICS + (
    "numerical_failure_count",
    "actuator_saturation_duration",
    "actuator_saturation_rate",
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_SAFE_BAG_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMPLETE_STATUS = "complete"
_WRITING_STATUS = "writing"
_ARRAY_NAME = "validation.npz"


def _error(location: str, message: str) -> None:
    raise ArtifactValidationError("{} {}".format(location, message))


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error(location, "must be an object")
    return value


def _keys(value: Mapping[str, Any], expected: Sequence[str], location: str) -> None:
    missing = sorted(set(expected) - set(value))
    unknown = sorted(set(value) - set(expected))
    if missing or unknown:
        _error(
            location,
            "keys disagree; missing={}, unknown={}".format(missing, unknown),
        )


def _string(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
    ):
        _error(location, "must be a canonical non-empty string")
    return value


def _identifier(value: Any, location: str, *, bag: bool = False) -> str:
    selected = _string(value, location)
    pattern = _SAFE_BAG_ID if bag else _SAFE_ID
    if pattern.fullmatch(selected) is None:
        _error(location, "must be a safe identifier")
    return selected


def _choice(value: Any, choices: Sequence[str], location: str) -> str:
    selected = _string(value, location)
    if selected not in choices:
        _error(location, "must be one of {}".format(tuple(choices)))
    return selected


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        _error(location, "must be boolean")
    return value


def _number(
    value: Any,
    location: str,
    *,
    lower: float,
    upper: Optional[float] = None,
    strict_lower: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _error(location, "must be a finite number")
    selected = float(value)
    if (
        not np.isfinite(selected)
        or (selected <= lower if strict_lower else selected < lower)
        or (upper is not None and selected > upper)
    ):
        _error(location, "is outside its allowed range")
    return selected


def _integer(
    value: Any, location: str, *, minimum: int, maximum: int
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) < minimum
        or int(value) > maximum
    ):
        _error(
            location,
            "must be an integer in [{}, {}]".format(minimum, maximum),
        )
    return int(value)


def _absolute_path(value: Any, location: str, *, must_exist: bool) -> Path:
    selected = Path(_string(value, location)).expanduser()
    if not selected.is_absolute():
        _error(location, "must be an absolute path")
    selected = selected.resolve()
    if must_exist and not selected.exists():
        _error(location, "does not exist")
    return selected


def _vector3(value: Any, location: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 3:
        _error(location, "must contain three non-negative numbers")
    try:
        selected = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError(
            "{} must contain three non-negative numbers".format(location)
        ) from error
    if np.any(~np.isfinite(selected)) or np.any(selected < 0.0):
        _error(location, "must contain three non-negative numbers")
    selected.setflags(write=False)
    return selected


def _sha256(value: Any, location: str) -> str:
    selected = _string(value, location)
    if _SHA256.fullmatch(selected) is None:
        _error(location, "must have form sha256:<64 lowercase hex>")
    return selected


@dataclass(frozen=True)
class HeldOutBagRequest:
    bag_id: str
    path: Path
    sha256: str
    interval_seconds: Tuple[float, float]
    roll_pitch_integration_active: bool


@dataclass(frozen=True)
class DataSplitDeclaration:
    role: str
    used_for_estimator_tuning: bool
    used_for_pid_tuning: bool

    @property
    def semantic_label(self) -> str:
        if self.role == STRICT_HOLD_OUT:
            return "strict held-out validation"
        return "tuning evaluation (not held-out)"


@dataclass(frozen=True)
class ConfigurationCompatibility:
    status: str
    evidence: Tuple[str, ...]


@dataclass(frozen=True)
class HeldOutValidationRequest:
    source_path: Path
    payload: Mapping[str, Any]
    fingerprint: str
    validation_id: str
    estimation_run: Path
    output_directory: Path
    held_out_bag: HeldOutBagRequest
    data_split: DataSplitDeclaration
    configuration_compatibility: ConfigurationCompatibility
    selected_mode_id: Optional[str]
    fixed_linear_drag: np.ndarray
    fixed_angular_drag: np.ndarray
    discrepancy_policy: str
    discrepancy_base_seed: int
    discrepancy_replicates: int
    posterior_subset_method: str
    posterior_sample_ids: Optional[Tuple[str, ...]]
    knot_period_seconds: float
    pose_smoothing_window: int
    allow_zero_integral_fallback: bool
    maximum_reference_age_seconds: float
    quantile_level: float
    cvar_level: float


def _held_out_bag(value: Any) -> HeldOutBagRequest:
    location = "request.held_out_bag"
    item = _mapping(value, location)
    _keys(
        item,
        (
            "bag_id",
            "path",
            "sha256",
            "interval_seconds",
            "roll_pitch_integration_active",
        ),
        location,
    )
    interval = item["interval_seconds"]
    if not isinstance(interval, list) or len(interval) != 2:
        _error(location + ".interval_seconds", "must be [start, end]")
    start = _number(
        interval[0], location + ".interval_seconds[0]", lower=0.0
    )
    end = _number(
        interval[1], location + ".interval_seconds[1]", lower=start,
        strict_lower=True,
    )
    return HeldOutBagRequest(
        bag_id=_identifier(item["bag_id"], location + ".bag_id", bag=True),
        path=_absolute_path(item["path"], location + ".path", must_exist=True),
        sha256=_sha256(item["sha256"], location + ".sha256"),
        interval_seconds=(start, end),
        roll_pitch_integration_active=_boolean(
            item["roll_pitch_integration_active"],
            location + ".roll_pitch_integration_active",
        ),
    )


def _data_split(value: Any) -> DataSplitDeclaration:
    location = "request.data_split"
    item = _mapping(value, location)
    _keys(
        item,
        ("role", "used_for_estimator_tuning", "used_for_pid_tuning"),
        location,
    )
    role = _choice(item["role"], DATA_SPLIT_ROLES, location + ".role")
    estimator = _boolean(
        item["used_for_estimator_tuning"],
        location + ".used_for_estimator_tuning",
    )
    pid = _boolean(
        item["used_for_pid_tuning"], location + ".used_for_pid_tuning"
    )
    if role == STRICT_HOLD_OUT and (estimator or pid):
        _error(
            location,
            "strict_hold_out forbids estimator and PID tuning use",
        )
    if role == TUNING_EVALUATION and not (estimator or pid):
        _error(
            location,
            "tuning_evaluation requires at least one explicit tuning use",
        )
    return DataSplitDeclaration(role, estimator, pid)


def _configuration_compatibility(value: Any) -> ConfigurationCompatibility:
    location = "request.configuration_compatibility"
    item = _mapping(value, location)
    _keys(item, ("status", "evidence"), location)
    status = _choice(
        item["status"],
        CONFIGURATION_COMPATIBILITY_STATUSES,
        location + ".status",
    )
    evidence = item["evidence"]
    if not isinstance(evidence, list) or not evidence:
        _error(location + ".evidence", "must be a non-empty list")
    selected = tuple(
        _string(entry, location + ".evidence") for entry in evidence
    )
    return ConfigurationCompatibility(status, selected)


def validate_held_out_validation_request(
    payload: Mapping[str, Any],
    source_path: Union[str, Path] = "<memory>",
) -> HeldOutValidationRequest:
    """Strictly validate one v2 held-out/tuning evaluation request."""

    request = _mapping(payload, "request")
    _keys(
        request,
        (
            "schema",
            "validation_id",
            "estimation_run",
            "output_directory",
            "held_out_bag",
            "data_split",
            "configuration_compatibility",
            "selected_mode_id",
            "fixed_plant_parameters",
            "model_discrepancy",
            "posterior_sample_subset",
            "forecast_settings",
            "summary_settings",
        ),
        "request",
    )
    _choice(
        request["schema"],
        (HELD_OUT_VALIDATION_REQUEST_SCHEMA,),
        "request.schema",
    )
    validation_id = _identifier(
        request["validation_id"], "request.validation_id"
    )
    estimation_run = _absolute_path(
        request["estimation_run"], "request.estimation_run", must_exist=True
    )
    output_directory = _absolute_path(
        request["output_directory"],
        "request.output_directory",
        must_exist=False,
    )
    held_out_bag = _held_out_bag(request["held_out_bag"])
    data_split = _data_split(request["data_split"])
    compatibility = _configuration_compatibility(
        request["configuration_compatibility"]
    )
    raw_mode = request["selected_mode_id"]
    selected_mode = (
        None
        if raw_mode is None
        else _string(raw_mode, "request.selected_mode_id")
    )

    fixed = _mapping(
        request["fixed_plant_parameters"], "request.fixed_plant_parameters"
    )
    _keys(fixed, ("linear_drag", "angular_drag"), "request.fixed_plant_parameters")
    linear_drag = _vector3(
        fixed["linear_drag"], "request.fixed_plant_parameters.linear_drag"
    )
    angular_drag = _vector3(
        fixed["angular_drag"], "request.fixed_plant_parameters.angular_drag"
    )

    discrepancy = _mapping(
        request["model_discrepancy"], "request.model_discrepancy"
    )
    _keys(
        discrepancy,
        ("policy", "base_seed", "replicates"),
        "request.model_discrepancy",
    )
    discrepancy_policy = _choice(
        discrepancy["policy"],
        MODEL_DISCREPANCY_POLICIES,
        "request.model_discrepancy.policy",
    )
    discrepancy_base_seed = _integer(
        discrepancy["base_seed"],
        "request.model_discrepancy.base_seed",
        minimum=0,
        maximum=2 ** 64 - 1,
    )
    discrepancy_replicates = _integer(
        discrepancy["replicates"],
        "request.model_discrepancy.replicates",
        minimum=1,
        maximum=10 ** 6,
    )

    subset = _mapping(
        request["posterior_sample_subset"],
        "request.posterior_sample_subset",
    )
    _keys(subset, ("method", "sample_ids"), "request.posterior_sample_subset")
    subset_method = _choice(
        subset["method"],
        POSTERIOR_SUBSET_METHODS,
        "request.posterior_sample_subset.method",
    )
    raw_ids = subset["sample_ids"]
    if subset_method == "all_equal_weight_mcmc_samples":
        if raw_ids is not None:
            _error(
                "request.posterior_sample_subset.sample_ids",
                "must be null when all samples are selected",
            )
        sample_ids = None
    else:
        if not isinstance(raw_ids, list) or not raw_ids:
            _error(
                "request.posterior_sample_subset.sample_ids",
                "must be a non-empty list for an explicit subset",
            )
        sample_ids = tuple(
            _string(value, "request.posterior_sample_subset.sample_ids")
            for value in raw_ids
        )
        if len(set(sample_ids)) != len(sample_ids):
            _error(
                "request.posterior_sample_subset.sample_ids",
                "contains duplicate sample IDs",
            )

    forecast = _mapping(request["forecast_settings"], "request.forecast_settings")
    _keys(
        forecast,
        (
            "knot_period_seconds",
            "pose_smoothing_window",
            "allow_zero_integral_fallback",
            "maximum_reference_age_seconds",
        ),
        "request.forecast_settings",
    )
    knot_period = _number(
        forecast["knot_period_seconds"],
        "request.forecast_settings.knot_period_seconds",
        lower=0.0,
        strict_lower=True,
    )
    smoothing = _integer(
        forecast["pose_smoothing_window"],
        "request.forecast_settings.pose_smoothing_window",
        minimum=1,
        maximum=9999,
    )
    if smoothing % 2 == 0:
        _error(
            "request.forecast_settings.pose_smoothing_window",
            "must be odd",
        )
    zero_integral = _boolean(
        forecast["allow_zero_integral_fallback"],
        "request.forecast_settings.allow_zero_integral_fallback",
    )
    maximum_reference_age = _number(
        forecast["maximum_reference_age_seconds"],
        "request.forecast_settings.maximum_reference_age_seconds",
        lower=0.0,
        strict_lower=True,
    )

    summary = _mapping(request["summary_settings"], "request.summary_settings")
    _keys(summary, ("quantile_level", "cvar_level"), "request.summary_settings")
    quantile = _number(
        summary["quantile_level"],
        "request.summary_settings.quantile_level",
        lower=0.0,
        upper=1.0,
        strict_lower=True,
    )
    if quantile >= 1.0:
        _error("request.summary_settings.quantile_level", "must be less than one")
    cvar = _number(
        summary["cvar_level"],
        "request.summary_settings.cvar_level",
        lower=0.0,
        upper=1.0,
    )
    if cvar >= 1.0:
        _error("request.summary_settings.cvar_level", "must be less than one")

    return HeldOutValidationRequest(
        source_path=Path(source_path),
        payload=MappingProxyType(dict(request)),
        fingerprint=request_fingerprint(request),
        validation_id=validation_id,
        estimation_run=estimation_run,
        output_directory=output_directory,
        held_out_bag=held_out_bag,
        data_split=data_split,
        configuration_compatibility=compatibility,
        selected_mode_id=selected_mode,
        fixed_linear_drag=linear_drag,
        fixed_angular_drag=angular_drag,
        discrepancy_policy=discrepancy_policy,
        discrepancy_base_seed=discrepancy_base_seed,
        discrepancy_replicates=discrepancy_replicates,
        posterior_subset_method=subset_method,
        posterior_sample_ids=sample_ids,
        knot_period_seconds=knot_period,
        pose_smoothing_window=smoothing,
        allow_zero_integral_fallback=zero_integral,
        maximum_reference_age_seconds=maximum_reference_age,
        quantile_level=quantile,
        cvar_level=cvar,
    )


def load_held_out_validation_request(
    path: Union[str, Path]
) -> HeldOutValidationRequest:
    source = Path(path).expanduser().resolve()
    return validate_held_out_validation_request(read_json(source), source)


def validate_data_split_against_source(
    declaration: DataSplitDeclaration,
    held_out_sha256: str,
    source_manifest: Mapping[str, Any],
) -> None:
    """Reject a fitted bag mislabeled as strict hold-out data."""

    if not isinstance(declaration, DataSplitDeclaration):
        raise TypeError("declaration must be DataSplitDeclaration")
    digest = _sha256(held_out_sha256, "held_out_sha256")
    hashes = source_manifest.get("selected_bag_sha256")
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError("source estimation manifest has no selected bag hashes")
    source_hashes = tuple(str(value) for value in hashes.values())
    if declaration.role == STRICT_HOLD_OUT and digest in source_hashes:
        raise ValueError(
            "strict_hold_out rosbag is already part of the source estimation; "
            "declare tuning_evaluation instead"
        )


def _actuator_parameters(manifest: Mapping[str, Any]) -> ActuatorParameters:
    model = manifest.get("actuator_model")
    if not isinstance(model, Mapping):
        raise ValueError("source estimation manifest has no actuator_model")
    expected = {
        "source",
        "thrust_time_constant_seconds",
        "gimbal_time_constant_seconds",
        "minimum_thrust_newtons",
        "maximum_thrust_newtons",
        "maximum_gimbal_angle_radians",
        "maximum_gimbal_rate_radians_per_second",
    }
    if set(model) != expected:
        raise ValueError("source estimation actuator_model fields disagree")
    _string(model["source"], "source actuator_model.source")
    return ActuatorParameters(
        thrust_time_constant=float(model["thrust_time_constant_seconds"]),
        gimbal_time_constant=float(model["gimbal_time_constant_seconds"]),
        delay=0.0,
        minimum_thrust=float(model["minimum_thrust_newtons"]),
        maximum_thrust=float(model["maximum_thrust_newtons"]),
        maximum_gimbal_angle=float(model["maximum_gimbal_angle_radians"]),
        maximum_gimbal_rate=float(
            model["maximum_gimbal_rate_radians_per_second"]
        ),
    )


def _q_contract(run: BatchEstimationRun) -> Tuple[str, str]:
    definition = str(run.manifest["q_definition"]["definition"])
    parts = definition.split("/", 1)
    if len(parts) != 2:
        raise ValueError("source estimation Q definition is incomplete")
    quantity, interval_model = parts
    if quantity not in MODEL_DISCREPANCY_QUANTITIES:
        raise ValueError("source estimation Q residual quantity is unsupported")
    if interval_model not in MODEL_DISCREPANCY_INTERVAL_MODELS:
        raise ValueError("source estimation Q interval model is unsupported")
    return quantity, interval_model


def _selected_posterior(
    posterior: PhysicalPlantPosterior,
    sample_ids: Optional[Sequence[str]],
) -> PhysicalPlantPosterior:
    if not isinstance(posterior, PhysicalPlantPosterior):
        raise TypeError("posterior must be PhysicalPlantPosterior")
    if sample_ids is None:
        return posterior
    selected = tuple(posterior.sample(value) for value in sample_ids)
    return PhysicalPlantPosterior(selected)


def _reference_states(
    flight: FlightData,
    times: np.ndarray,
    maximum_age_seconds: float,
) -> Tuple[ReferenceState, ...]:
    source = flight.reference
    indices = np.searchsorted(source.times, times, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("held-out reference has no causal value at first knot")
    ages = times - source.times[indices]
    if np.any(ages < -2.0e-7) or np.any(ages > maximum_age_seconds + 2.0e-7):
        raise ValueError("held-out reference exceeds its maximum causal age")
    return tuple(
        ReferenceState(
            position=source.position[index],
            linear_velocity=source.linear_velocity[index],
            linear_acceleration=source.linear_acceleration[index],
            rpy=source.rpy[index],
            angular_velocity=source.angular_velocity[index],
            angular_acceleration=source.angular_acceleration[index],
        )
        for index in indices
    )


def _initial_condition(
    initialization,
    flight: FlightData,
    sample: PhysicalPlantSample,
    roll_pitch_integration_active: bool,
) -> PidForecastInitialCondition:
    state = initialization.state
    bag_id = flight.bag_id
    sensor_position = state.knot_value(bag_id, 0, VariableKind.POSITION)
    sensor_rotation = state.knot_value(
        bag_id, 0, VariableKind.ORIENTATION_TANGENT
    )
    extrinsics = flight.sensor_extrinsics
    body_rotation = sensor_rotation @ extrinsics.pose_sensor_to_body_rotation.T
    cog_to_pose_sensor = (
        extrinsics.pose_sensor_position_in_body
        - sample.parameters.cog_offset
    )
    cog_position = sensor_position - body_rotation @ cog_to_pose_sensor
    angular_velocity = state.knot_value(
        bag_id, 0, VariableKind.ANGULAR_VELOCITY
    )
    measured_velocity = state.knot_value(
        bag_id, 0, VariableKind.LINEAR_VELOCITY
    )
    cog_to_velocity_sensor = (
        extrinsics.velocity_sensor_position_in_body
        - sample.parameters.cog_offset
    )
    cog_velocity = measured_velocity - body_rotation @ np.cross(
        angular_velocity, cog_to_velocity_sensor
    )
    return PidForecastInitialCondition(
        sample_id=sample.sample_id,
        rigid_body_state=RigidBodyState(
            cog_position,
            matrix_to_quaternion(body_rotation),
            cog_velocity,
            angular_velocity,
        ),
        controller_state=ControllerState(
            state.knot_value(
                bag_id, 0, VariableKind.CONTROLLER_INTEGRAL
            ),
            roll_pitch_integration_active=roll_pitch_integration_active,
        ),
        actuator_state=ActuatorState(
            state.knot_value(bag_id, 0, VariableKind.ACTUATOR_THRUST),
            state.knot_value(bag_id, 0, VariableKind.GIMBAL_ANGLE),
        ),
        source="held_out_leading_observation_anchor",
    )


@dataclass(frozen=True)
class PreparedHeldOutForecast:
    """Held-out forecast scenario plus immutable observation scoring input."""

    scenario: PidForecastScenario
    flight_data: FlightData

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, PidForecastScenario):
            raise TypeError("scenario must be PidForecastScenario")
        if not isinstance(self.flight_data, FlightData):
            raise TypeError("flight_data must be FlightData")
        if self.scenario.bag_id != self.flight_data.bag_id:
            raise ValueError("scenario and held-out flight bag IDs disagree")


def prepare_held_out_forecast(
    request: HeldOutValidationRequest,
    flight: FlightData,
    posterior: PhysicalPlantPosterior,
    actuator_parameters: ActuatorParameters,
) -> PreparedHeldOutForecast:
    """Create a sample-aligned scenario with one leading observation anchor."""

    if not isinstance(request, HeldOutValidationRequest):
        raise TypeError("request must be HeldOutValidationRequest")
    if not isinstance(flight, FlightData):
        raise TypeError("flight must be FlightData")
    if flight.bag_id != request.held_out_bag.bag_id:
        raise ValueError("loaded held-out bag ID differs from request")
    initialization = build_flight_initialization(
        flight,
        request.knot_period_seconds,
        pose_smoothing_window=request.pose_smoothing_window,
        allow_zero_integral_fallback=request.allow_zero_integral_fallback,
    )
    nominal = VehicleParameters.nominal()
    geometry = GrapeGeometry.grape()
    scenario = PidForecastScenario(
        bag_id=flight.bag_id,
        times=initialization.grid.times,
        references=_reference_states(
            flight,
            initialization.grid.times,
            request.maximum_reference_age_seconds,
        ),
        initial_conditions=tuple(
            _initial_condition(
                initialization,
                flight,
                sample,
                request.held_out_bag.roll_pitch_integration_active,
            )
            for sample in posterior.samples
        ),
        controller_configuration=flight.controller_configuration,
        controller_nominal_parameters=nominal,
        controller_geometry=geometry,
        plant_geometry=geometry,
        actuator_parameters=actuator_parameters,
        provenance=(
            ("data_split_role", request.data_split.role),
            (
                "initialization_policy",
                "one leading observed state anchor; no forecast state resets",
            ),
            ("reference_policy", "causal_zero_order_hold"),
        ),
    )
    return PreparedHeldOutForecast(scenario, flight)


def _pose_at_times(
    flight: FlightData, times: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    source = flight.pose
    selected_times = np.asarray(times, dtype=float)
    if (
        selected_times.ndim != 1
        or selected_times.size < 1
        or selected_times[0] < source.times[0] - 2.0e-7
        or selected_times[-1] > source.times[-1] + 2.0e-7
    ):
        raise ValueError("forecast trace lies outside held-out pose support")
    right = np.searchsorted(source.times, selected_times, side="right")
    right = np.clip(right, 1, source.times.size - 1)
    position = np.empty((selected_times.size, 3), dtype=float)
    orientation = np.empty((selected_times.size, 3, 3), dtype=float)
    source_rotation = tuple(
        quaternion_to_matrix(value) for value in source.orientations_xyzw
    )
    for row, (time_value, right_index) in enumerate(zip(selected_times, right)):
        left_index = right_index - 1
        span = source.times[right_index] - source.times[left_index]
        fraction = float((time_value - source.times[left_index]) / span)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        position[row] = (
            (1.0 - fraction) * source.positions[left_index]
            + fraction * source.positions[right_index]
        )
        orientation[row] = so3_geodesic_interpolation(
            source_rotation[left_index],
            source_rotation[right_index],
            fraction,
        )
    return position, orientation


def _observed_metrics(
    outcome: PidForecastOutcome,
    sample: PhysicalPlantSample,
    prepared: PreparedHeldOutForecast,
) -> ForecastMetrics:
    trace = outcome.trace
    observed_position, observed_rotation = _pose_at_times(
        prepared.flight_data, trace.times
    )
    extrinsics = prepared.flight_data.sensor_extrinsics
    cog_to_sensor = (
        extrinsics.pose_sensor_position_in_body
        - sample.parameters.cog_offset
    )
    sensor_to_body = extrinsics.pose_sensor_to_body_rotation
    predicted_position = np.empty_like(trace.position)
    orientation_error = np.empty((trace.times.size, 3), dtype=float)
    for index, quaternion in enumerate(trace.orientation_xyzw):
        body_rotation = quaternion_to_matrix(quaternion)
        predicted_position[index] = (
            trace.position[index] + body_rotation @ cog_to_sensor
        )
        predicted_sensor_rotation = body_rotation @ sensor_to_body
        orientation_error[index] = so3_log(
            observed_rotation[index].T @ predicted_sensor_rotation
        )
    position_norm = np.linalg.norm(
        predicted_position - observed_position, axis=1
    )
    orientation_norm = np.linalg.norm(orientation_error, axis=1)
    reference = outcome.metrics
    return ForecastMetrics(
        position_rmse=float(np.sqrt(np.mean(position_norm * position_norm))),
        orientation_rmse=float(
            np.sqrt(np.mean(orientation_norm * orientation_norm))
        ),
        maximum_position_error=float(np.max(position_norm)),
        maximum_orientation_error=float(np.max(orientation_norm)),
        forecast_completion=reference.forecast_completion,
        numerical_failure_count=reference.numerical_failure_count,
        actuator_saturation_duration=reference.actuator_saturation_duration,
        actuator_saturation_rate=reference.actuator_saturation_rate,
    )


@dataclass(frozen=True)
class HeldOutMetricRecord:
    """Separated observed/reference metrics for one posterior forecast."""

    sample_id: str
    replicate_index: int
    discrepancy_seed: int
    observed: ForecastMetrics
    reference: ForecastMetrics

    def __post_init__(self) -> None:
        sample_id = _string(self.sample_id, "sample_id")
        replicate = _integer(
            self.replicate_index,
            "replicate_index",
            minimum=0,
            maximum=10 ** 9,
        )
        seed = _integer(
            self.discrepancy_seed,
            "discrepancy_seed",
            minimum=0,
            maximum=2 ** 64 - 1,
        )
        if not isinstance(self.observed, ForecastMetrics) or not isinstance(
            self.reference, ForecastMetrics
        ):
            raise TypeError("observed and reference must be ForecastMetrics")
        shared = (
            "forecast_completion",
            "numerical_failure_count",
            "actuator_saturation_duration",
            "actuator_saturation_rate",
        )
        if any(
            getattr(self.observed, name) != getattr(self.reference, name)
            for name in shared
        ):
            raise ValueError("observed/reference forecast status must agree")
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "replicate_index", replicate)
        object.__setattr__(self, "discrepancy_seed", seed)

    def cost_values(self) -> np.ndarray:
        observed = self.observed
        reference = self.reference
        values = np.asarray(
            (
                observed.position_rmse,
                observed.orientation_rmse,
                observed.maximum_position_error,
                observed.maximum_orientation_error,
                reference.position_rmse,
                reference.orientation_rmse,
                reference.maximum_position_error,
                reference.maximum_orientation_error,
                reference.numerical_failure_count,
                reference.actuator_saturation_duration,
                reference.actuator_saturation_rate,
            ),
            dtype=float,
        )
        values.setflags(write=False)
        return values


def metric_record_from_outcome(
    outcome: PidForecastOutcome,
    sample: PhysicalPlantSample,
    prepared: PreparedHeldOutForecast,
    replicate_index: int,
) -> HeldOutMetricRecord:
    if outcome.sample_id != sample.sample_id:
        raise ValueError("forecast outcome and posterior sample IDs disagree")
    if outcome.bag_id != prepared.scenario.bag_id:
        raise ValueError("forecast outcome and held-out bag IDs disagree")
    return HeldOutMetricRecord(
        sample_id=sample.sample_id,
        replicate_index=replicate_index,
        discrepancy_seed=outcome.discrepancy_seed,
        observed=_observed_metrics(outcome, sample, prepared),
        reference=outcome.metrics,
    )


@dataclass(frozen=True)
class HeldOutValidationResult:
    records: Tuple[HeldOutMetricRecord, ...]
    metric_names: Tuple[str, ...]
    mean: np.ndarray
    quantile: np.ndarray
    upper_cvar: np.ndarray
    forecast_completion_mean: float
    forecast_completion_lower_quantile: float
    forecast_completion_lower_cvar: float
    quantile_level: float
    cvar_level: float

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not records or any(
            not isinstance(value, HeldOutMetricRecord) for value in records
        ):
            raise ValueError("records must contain held-out metric records")
        if tuple(self.metric_names) != HELD_OUT_COST_METRICS:
            raise ValueError("metric_names must preserve held-out physical units")
        for name in ("mean", "quantile", "upper_cvar"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (len(HELD_OUT_COST_METRICS),) or np.any(
                ~np.isfinite(value)
            ):
                raise ValueError("{} must align with metric_names".format(name))
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        for name in (
            "forecast_completion_mean",
            "forecast_completion_lower_quantile",
            "forecast_completion_lower_cvar",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("{} must be in [0, 1]".format(name))
            object.__setattr__(self, name, value)
        quantile = float(self.quantile_level)
        cvar = float(self.cvar_level)
        if not 0.0 < quantile < 1.0 or not 0.0 <= cvar < 1.0:
            raise ValueError("summary levels are invalid")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "metric_names", HELD_OUT_COST_METRICS)
        object.__setattr__(self, "quantile_level", quantile)
        object.__setattr__(self, "cvar_level", cvar)


def summarize_held_out_records(
    records: Sequence[HeldOutMetricRecord],
    *,
    quantile_level: float,
    cvar_level: float,
) -> HeldOutValidationResult:
    selected = tuple(records)
    if not selected:
        raise ValueError("held-out records cannot be empty")
    values = np.vstack(tuple(value.cost_values() for value in selected))
    completion = np.asarray(
        tuple(value.reference.forecast_completion for value in selected)
    )
    quantile = float(quantile_level)
    cvar = float(cvar_level)
    return HeldOutValidationResult(
        records=selected,
        metric_names=HELD_OUT_COST_METRICS,
        mean=np.mean(values, axis=0),
        quantile=np.quantile(values, quantile, axis=0),
        upper_cvar=np.asarray(
            tuple(
                empirical_upper_cvar(values[:, index], cvar)
                for index in range(values.shape[1])
            )
        ),
        forecast_completion_mean=float(np.mean(completion)),
        forecast_completion_lower_quantile=float(
            np.quantile(completion, 1.0 - quantile)
        ),
        forecast_completion_lower_cvar=empirical_lower_cvar(completion, cvar),
        quantile_level=quantile,
        cvar_level=cvar,
    )


@dataclass(frozen=True)
class HeldOutValidationIdentity:
    source_estimation_run_id: str
    source_estimation_request_fingerprint: str
    source_estimator_revision: str
    selected_mode_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_estimation_run_id",
            _string(
                self.source_estimation_run_id,
                "source_estimation_run_id",
            ),
        )
        object.__setattr__(
            self,
            "source_estimation_request_fingerprint",
            _sha256(
                self.source_estimation_request_fingerprint,
                "source_estimation_request_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "source_estimator_revision",
            _string(self.source_estimator_revision, "source_estimator_revision"),
        )
        object.__setattr__(
            self,
            "selected_mode_id",
            _string(self.selected_mode_id, "selected_mode_id"),
        )


@dataclass(frozen=True)
class HeldOutValidationArtifact:
    root: Path
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]


def _data_split_payload(value: DataSplitDeclaration) -> Mapping[str, Any]:
    return {
        "role": value.role,
        "used_for_estimator_tuning": value.used_for_estimator_tuning,
        "used_for_pid_tuning": value.used_for_pid_tuning,
    }


def _compatibility_payload(
    value: ConfigurationCompatibility,
) -> Mapping[str, Any]:
    return {"status": value.status, "evidence": list(value.evidence)}


def _result_arrays(result: HeldOutValidationResult) -> Mapping[str, np.ndarray]:
    records = result.records
    return {
        "schema": np.asarray(HELD_OUT_VALIDATION_ARRAY_SCHEMA),
        "sample_id": np.asarray(
            tuple(value.sample_id for value in records), dtype=str
        ),
        "replicate_index": np.asarray(
            tuple(value.replicate_index for value in records), dtype=np.int64
        ),
        "discrepancy_seed": np.asarray(
            tuple(value.discrepancy_seed for value in records), dtype=np.uint64
        ),
        "metric_names": np.asarray(result.metric_names, dtype=str),
        "metric_values": np.vstack(
            tuple(value.cost_values() for value in records)
        ),
        "forecast_completion": np.asarray(
            tuple(
                value.reference.forecast_completion for value in records
            ),
            dtype=float,
        ),
        "mean": result.mean,
        "quantile": result.quantile,
        "upper_cvar": result.upper_cvar,
        "forecast_completion_mean": np.asarray(
            result.forecast_completion_mean
        ),
        "forecast_completion_lower_quantile": np.asarray(
            result.forecast_completion_lower_quantile
        ),
        "forecast_completion_lower_cvar": np.asarray(
            result.forecast_completion_lower_cvar
        ),
    }


_ARRAY_KEYS = (
    "schema",
    "sample_id",
    "replicate_index",
    "discrepancy_seed",
    "metric_names",
    "metric_values",
    "forecast_completion",
    "mean",
    "quantile",
    "upper_cvar",
    "forecast_completion_mean",
    "forecast_completion_lower_quantile",
    "forecast_completion_lower_cvar",
)


def _load_npz(path: Path) -> Mapping[str, np.ndarray]:
    try:
        with np.load(str(path), allow_pickle=False) as source:
            if set(source.files) != set(_ARRAY_KEYS):
                raise ArtifactValidationError(
                    "held-out validation NPZ keys disagree"
                )
            arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    except ArtifactValidationError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise ArtifactValidationError(
            "cannot load held-out validation arrays: {}".format(error)
        ) from error
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ArtifactValidationError("held-out arrays contain object dtype")
    return MappingProxyType(arrays)


def _validate_arrays(
    arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any]
) -> None:
    schema = np.asarray(arrays["schema"])
    if schema.shape != () or str(schema.item()) != HELD_OUT_VALIDATION_ARRAY_SCHEMA:
        raise ArtifactValidationError("held-out array schema is invalid")
    record_count = int(manifest["record_count"])
    metric_count = len(HELD_OUT_COST_METRICS)
    expected_shapes = {
        "sample_id": (record_count,),
        "replicate_index": (record_count,),
        "discrepancy_seed": (record_count,),
        "metric_names": (metric_count,),
        "metric_values": (record_count, metric_count),
        "forecast_completion": (record_count,),
        "mean": (metric_count,),
        "quantile": (metric_count,),
        "upper_cvar": (metric_count,),
        "forecast_completion_mean": (),
        "forecast_completion_lower_quantile": (),
        "forecast_completion_lower_cvar": (),
    }
    for name, shape in expected_shapes.items():
        if np.asarray(arrays[name]).shape != shape:
            raise ArtifactValidationError(
                "held-out array {} shape disagrees".format(name)
            )
    names = tuple(str(value) for value in arrays["metric_names"].tolist())
    if names != HELD_OUT_COST_METRICS or names != tuple(manifest["metric_names"]):
        raise ArtifactValidationError("held-out metric names disagree")
    sample_ids = tuple(str(value) for value in arrays["sample_id"].tolist())
    selected_ids = tuple(str(value) for value in manifest["plant_sample_ids"])
    if set(sample_ids) != set(selected_ids):
        raise ArtifactValidationError("held-out sample IDs disagree")
    replicate_count = int(manifest["model_discrepancy_replicates"])
    expected_records = len(selected_ids) * replicate_count
    if record_count != expected_records:
        raise ArtifactValidationError("held-out Cartesian record count disagrees")
    replicate_indices = tuple(
        int(value) for value in arrays["replicate_index"].tolist()
    )
    actual_pairs = tuple(zip(sample_ids, replicate_indices))
    expected_pairs = tuple(
        (sample_id, replicate)
        for sample_id in selected_ids
        for replicate in range(replicate_count)
    )
    if actual_pairs != expected_pairs:
        raise ArtifactValidationError(
            "held-out sample/replicate Cartesian order disagrees"
        )
    numeric = (
        "metric_values",
        "forecast_completion",
        "mean",
        "quantile",
        "upper_cvar",
        "forecast_completion_mean",
        "forecast_completion_lower_quantile",
        "forecast_completion_lower_cvar",
    )
    if any(np.any(~np.isfinite(arrays[name])) for name in numeric):
        raise ArtifactValidationError("held-out metric arrays must be finite")
    completion = np.asarray(arrays["forecast_completion"], dtype=float)
    if np.any(completion < 0.0) or np.any(completion > 1.0):
        raise ArtifactValidationError("held-out forecast completion is invalid")
    if np.asarray(arrays["replicate_index"]).dtype.kind not in "iu":
        raise ArtifactValidationError("held-out replicate index is invalid")
    if np.asarray(arrays["discrepancy_seed"]).dtype.kind not in "iu":
        raise ArtifactValidationError("held-out discrepancy seed is invalid")


_MANIFEST_KEYS = (
    "schema",
    "status",
    "validation_id",
    "semantic_label",
    "data_split",
    "configuration_compatibility",
    "source_estimation_run_id",
    "source_estimation_request_fingerprint",
    "source_estimator_revision",
    "source_estimation_run_path",
    "request_fingerprint",
    "held_out_bag_id",
    "held_out_bag_sha256",
    "held_out_interval_seconds",
    "selected_mode_id",
    "model_discrepancy_policy",
    "model_discrepancy_residual_quantity",
    "model_discrepancy_interval_model",
    "model_discrepancy_q_diagonal",
    "model_discrepancy_base_seed",
    "model_discrepancy_replicates",
    "plant_sample_subset_method",
    "plant_sample_ids",
    "fixed_plant_parameters",
    "source_actuator_model",
    "forecast_settings",
    "record_count",
    "metric_names",
    "quantile_level",
    "cvar_level",
    "forecast_initialization_policy",
    "warnings",
    "artifacts",
)


def _manifest_payload(
    request: HeldOutValidationRequest,
    identity: HeldOutValidationIdentity,
    selected_posterior: PhysicalPlantPosterior,
    result: HeldOutValidationResult,
    q_quantity: str,
    q_interval_model: str,
    q_diagonal: np.ndarray,
    source_actuator_model: Mapping[str, Any],
    status: str,
    artifacts: Mapping[str, Any],
) -> Mapping[str, Any]:
    warnings = []
    if request.configuration_compatibility.status == "unconfirmed":
        warnings.append(
            "held-out hardware compatibility is unconfirmed; do not interpret "
            "forecast mismatch as parameter error alone"
        )
    if request.data_split.role == TUNING_EVALUATION:
        warnings.append(
            "this flight was used for tuning and is not held-out evidence"
        )
    return {
        "schema": HELD_OUT_VALIDATION_SCHEMA,
        "status": status,
        "validation_id": request.validation_id,
        "semantic_label": request.data_split.semantic_label,
        "data_split": _data_split_payload(request.data_split),
        "configuration_compatibility": _compatibility_payload(
            request.configuration_compatibility
        ),
        "source_estimation_run_id": identity.source_estimation_run_id,
        "source_estimation_request_fingerprint": (
            identity.source_estimation_request_fingerprint
        ),
        "source_estimator_revision": identity.source_estimator_revision,
        "source_estimation_run_path": str(request.estimation_run),
        "request_fingerprint": request.fingerprint,
        "held_out_bag_id": request.held_out_bag.bag_id,
        "held_out_bag_sha256": request.held_out_bag.sha256,
        "held_out_interval_seconds": list(
            request.held_out_bag.interval_seconds
        ),
        "selected_mode_id": identity.selected_mode_id,
        "model_discrepancy_policy": request.discrepancy_policy,
        "model_discrepancy_residual_quantity": q_quantity,
        "model_discrepancy_interval_model": q_interval_model,
        "model_discrepancy_q_diagonal": np.asarray(
            q_diagonal, dtype=float
        ).tolist(),
        "model_discrepancy_base_seed": request.discrepancy_base_seed,
        "model_discrepancy_replicates": request.discrepancy_replicates,
        "plant_sample_subset_method": request.posterior_subset_method,
        "plant_sample_ids": [
            value.sample_id for value in selected_posterior.samples
        ],
        "fixed_plant_parameters": {
            "linear_drag": request.fixed_linear_drag.tolist(),
            "angular_drag": request.fixed_angular_drag.tolist(),
        },
        "source_actuator_model": dict(source_actuator_model),
        "forecast_settings": {
            "knot_period_seconds": request.knot_period_seconds,
            "pose_smoothing_window": request.pose_smoothing_window,
            "allow_zero_integral_fallback": (
                request.allow_zero_integral_fallback
            ),
            "maximum_reference_age_seconds": (
                request.maximum_reference_age_seconds
            ),
            "roll_pitch_integration_active": (
                request.held_out_bag.roll_pitch_integration_active
            ),
        },
        "record_count": len(result.records),
        "metric_names": list(result.metric_names),
        "quantile_level": result.quantile_level,
        "cvar_level": result.cvar_level,
        "forecast_initialization_policy": (
            "sample-specific CoG state from one leading pose/velocity anchor; "
            "continuous closed-loop forecast without observation resets"
        ),
        "warnings": warnings,
        "artifacts": dict(artifacts),
    }


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    _keys(manifest, _MANIFEST_KEYS, "manifest")
    if manifest["schema"] != HELD_OUT_VALIDATION_SCHEMA:
        raise ArtifactValidationError("unsupported held-out validation schema")
    if manifest["status"] != _COMPLETE_STATUS:
        raise ArtifactValidationError("held-out validation is incomplete")
    _identifier(manifest["validation_id"], "manifest.validation_id")
    split = _data_split(manifest["data_split"])
    if manifest["semantic_label"] != split.semantic_label:
        raise ArtifactValidationError("held-out semantic label disagrees")
    _configuration_compatibility(manifest["configuration_compatibility"])
    _string(manifest["source_estimation_run_id"], "manifest.source_estimation_run_id")
    _string(manifest["source_estimator_revision"], "manifest.source_estimator_revision")
    _string(manifest["source_estimation_run_path"], "manifest.source_estimation_run_path")
    _string(manifest["selected_mode_id"], "manifest.selected_mode_id")
    _sha256(
        manifest["source_estimation_request_fingerprint"],
        "manifest.source_estimation_request_fingerprint",
    )
    _sha256(manifest["request_fingerprint"], "manifest.request_fingerprint")
    _sha256(manifest["held_out_bag_sha256"], "manifest.held_out_bag_sha256")
    if manifest["model_discrepancy_policy"] not in MODEL_DISCREPANCY_POLICIES:
        raise ArtifactValidationError("held-out discrepancy policy is invalid")
    if (
        manifest["model_discrepancy_residual_quantity"]
        not in MODEL_DISCREPANCY_QUANTITIES
        or manifest["model_discrepancy_interval_model"]
        not in MODEL_DISCREPANCY_INTERVAL_MODELS
    ):
        raise ArtifactValidationError("held-out Q definition is invalid")
    q = np.asarray(manifest["model_discrepancy_q_diagonal"], dtype=float)
    if q.shape != (6,) or np.any(~np.isfinite(q)) or np.any(q < 0.0):
        raise ArtifactValidationError("held-out Q diagonal is invalid")
    _integer(
        manifest["model_discrepancy_base_seed"],
        "manifest.model_discrepancy_base_seed",
        minimum=0,
        maximum=2 ** 64 - 1,
    )
    replicates = _integer(
        manifest["model_discrepancy_replicates"],
        "manifest.model_discrepancy_replicates",
        minimum=1,
        maximum=10 ** 6,
    )
    if manifest["plant_sample_subset_method"] not in POSTERIOR_SUBSET_METHODS:
        raise ArtifactValidationError("held-out sample subset method is invalid")
    sample_ids = manifest["plant_sample_ids"]
    if (
        not isinstance(sample_ids, list)
        or not sample_ids
        or len(set(sample_ids)) != len(sample_ids)
        or any(not isinstance(value, str) or not value for value in sample_ids)
    ):
        raise ArtifactValidationError("held-out sample IDs are invalid")
    fixed = _mapping(
        manifest["fixed_plant_parameters"],
        "manifest.fixed_plant_parameters",
    )
    _keys(
        fixed,
        ("linear_drag", "angular_drag"),
        "manifest.fixed_plant_parameters",
    )
    _vector3(fixed["linear_drag"], "manifest.fixed_plant_parameters.linear_drag")
    _vector3(fixed["angular_drag"], "manifest.fixed_plant_parameters.angular_drag")
    _actuator_parameters({"actuator_model": manifest["source_actuator_model"]})
    forecast = _mapping(manifest["forecast_settings"], "manifest.forecast_settings")
    _keys(
        forecast,
        (
            "knot_period_seconds",
            "pose_smoothing_window",
            "allow_zero_integral_fallback",
            "maximum_reference_age_seconds",
            "roll_pitch_integration_active",
        ),
        "manifest.forecast_settings",
    )
    _number(
        forecast["knot_period_seconds"],
        "manifest.forecast_settings.knot_period_seconds",
        lower=0.0,
        strict_lower=True,
    )
    smoothing = _integer(
        forecast["pose_smoothing_window"],
        "manifest.forecast_settings.pose_smoothing_window",
        minimum=1,
        maximum=9999,
    )
    if smoothing % 2 == 0:
        raise ArtifactValidationError("held-out pose smoothing window must be odd")
    _boolean(
        forecast["allow_zero_integral_fallback"],
        "manifest.forecast_settings.allow_zero_integral_fallback",
    )
    _boolean(
        forecast["roll_pitch_integration_active"],
        "manifest.forecast_settings.roll_pitch_integration_active",
    )
    _number(
        forecast["maximum_reference_age_seconds"],
        "manifest.forecast_settings.maximum_reference_age_seconds",
        lower=0.0,
        strict_lower=True,
    )
    record_count = _integer(
        manifest["record_count"],
        "manifest.record_count",
        minimum=1,
        maximum=10 ** 9,
    )
    if record_count != len(sample_ids) * replicates:
        raise ArtifactValidationError("held-out manifest record count disagrees")
    if tuple(manifest["metric_names"]) != HELD_OUT_COST_METRICS:
        raise ArtifactValidationError("held-out manifest metrics disagree")
    quantile = _number(
        manifest["quantile_level"],
        "manifest.quantile_level",
        lower=0.0,
        upper=1.0,
        strict_lower=True,
    )
    cvar = _number(
        manifest["cvar_level"],
        "manifest.cvar_level",
        lower=0.0,
        upper=1.0,
    )
    if quantile >= 1.0 or cvar >= 1.0:
        raise ArtifactValidationError("held-out summary levels are invalid")
    interval = manifest["held_out_interval_seconds"]
    if not isinstance(interval, list) or len(interval) != 2:
        raise ArtifactValidationError("held-out interval is invalid")
    start = _number(interval[0], "manifest.held_out_interval_seconds[0]", lower=0.0)
    _number(
        interval[1],
        "manifest.held_out_interval_seconds[1]",
        lower=start,
        strict_lower=True,
    )
    if not isinstance(manifest["warnings"], list) or any(
        not isinstance(value, str) or not value for value in manifest["warnings"]
    ):
        raise ArtifactValidationError("held-out warnings are invalid")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"validation"}:
        raise ArtifactValidationError("held-out artifact descriptor is invalid")
    descriptor = artifacts["validation"]
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "path",
        "sha256",
        "size_bytes",
        "schema",
    }:
        raise ArtifactValidationError("held-out NPZ descriptor is invalid")
    if descriptor["path"] != _ARRAY_NAME:
        raise ArtifactValidationError("held-out NPZ path is invalid")
    _sha256(descriptor["sha256"], "manifest.artifacts.validation.sha256")
    if descriptor["schema"] != HELD_OUT_VALIDATION_ARRAY_SCHEMA:
        raise ArtifactValidationError("held-out NPZ descriptor schema is invalid")
    _integer(
        descriptor["size_bytes"],
        "manifest.artifacts.validation.size_bytes",
        minimum=1,
        maximum=2 ** 63 - 1,
    )


def load_held_out_validation(
    path: Union[str, Path]
) -> HeldOutValidationArtifact:
    root = Path(path).expanduser().resolve()
    manifest = read_json(root / "manifest.json")
    _validate_manifest(manifest)
    array_path = root / _ARRAY_NAME
    descriptor = manifest["artifacts"]["validation"]
    if not array_path.is_file():
        raise ArtifactValidationError("held-out validation NPZ is missing")
    if file_sha256(array_path) != descriptor["sha256"]:
        raise ArtifactValidationError("held-out validation NPZ hash disagrees")
    if array_path.stat().st_size != descriptor["size_bytes"]:
        raise ArtifactValidationError("held-out validation NPZ size disagrees")
    arrays = _load_npz(array_path)
    _validate_arrays(arrays, manifest)
    return HeldOutValidationArtifact(
        root=root,
        manifest=MappingProxyType(dict(manifest)),
        arrays=arrays,
    )


def write_held_out_validation(
    path: Union[str, Path],
    *,
    request: HeldOutValidationRequest,
    identity: HeldOutValidationIdentity,
    selected_posterior: PhysicalPlantPosterior,
    result: HeldOutValidationResult,
    q_quantity: str,
    q_interval_model: str,
    q_diagonal: Sequence[float],
    source_actuator_model: Mapping[str, Any],
) -> HeldOutValidationArtifact:
    """Atomically publish a strict JSON/NPZ held-out result directory."""

    if not isinstance(request, HeldOutValidationRequest):
        raise TypeError("request must be HeldOutValidationRequest")
    if not isinstance(identity, HeldOutValidationIdentity):
        raise TypeError("identity must be HeldOutValidationIdentity")
    if not isinstance(selected_posterior, PhysicalPlantPosterior):
        raise TypeError("selected_posterior must be PhysicalPlantPosterior")
    if not isinstance(result, HeldOutValidationResult):
        raise TypeError("result must be HeldOutValidationResult")
    q = np.asarray(q_diagonal, dtype=float)
    if q.shape != (6,) or np.any(~np.isfinite(q)) or np.any(q < 0.0):
        raise ValueError("q_diagonal must contain six non-negative values")
    _actuator_parameters({"actuator_model": source_actuator_model})
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            "held-out validation destination already exists: {}".format(
                destination
            )
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=".{}.".format(destination.name), dir=str(destination.parent)
        )
    )
    published = False
    try:
        writing_manifest = _manifest_payload(
            request,
            identity,
            selected_posterior,
            result,
            q_quantity,
            q_interval_model,
            q,
            source_actuator_model,
            _WRITING_STATUS,
            {},
        )
        write_json_atomic(staging / "manifest.json", writing_manifest)
        array_path = write_npz_atomic(staging / _ARRAY_NAME, _result_arrays(result))
        descriptor = {
            "path": _ARRAY_NAME,
            "sha256": file_sha256(array_path),
            "size_bytes": array_path.stat().st_size,
            "schema": HELD_OUT_VALIDATION_ARRAY_SCHEMA,
        }
        complete_manifest = _manifest_payload(
            request,
            identity,
            selected_posterior,
            result,
            q_quantity,
            q_interval_model,
            q,
            source_actuator_model,
            _COMPLETE_STATUS,
            {"validation": descriptor},
        )
        _validate_manifest(complete_manifest)
        _validate_arrays(_load_npz(array_path), complete_manifest)
        write_json_atomic(staging / "manifest.json", complete_manifest)
        os.rename(str(staging), str(destination))
        published = True
    finally:
        if not published:
            shutil.rmtree(str(staging), ignore_errors=True)
    return load_held_out_validation(destination)


FlightLoader = Callable[..., FlightData]


def execute_held_out_validation(
    request: HeldOutValidationRequest,
    *,
    cancellation_token: Optional[CancellationToken] = None,
    progress_callback: Optional[ProgressCallback] = None,
    flight_loader: FlightLoader = load_flight_data,
) -> HeldOutValidationArtifact:
    """Run all selected MCMC plants on one declared held-out/tuning flight."""

    if not isinstance(request, HeldOutValidationRequest):
        raise TypeError("request must be HeldOutValidationRequest")
    if not callable(flight_loader):
        raise TypeError("flight_loader must be callable")
    cancellation = (
        CancellationToken() if cancellation_token is None else cancellation_token
    )
    cancellation.raise_if_cancelled()
    run = load_batch_estimation_run(request.estimation_run)
    validate_data_split_against_source(
        request.data_split,
        request.held_out_bag.sha256,
        run.manifest,
    )
    posterior = physical_posterior_from_batch_run(
        run,
        fixed_linear_drag=request.fixed_linear_drag,
        fixed_angular_drag=request.fixed_angular_drag,
        selected_mode_id=request.selected_mode_id,
    )
    posterior = _selected_posterior(posterior, request.posterior_sample_ids)
    forecast_count = len(posterior.samples) * request.discrepancy_replicates
    tracker = ProgressTracker(
        request.validation_id,
        overall_total_units=2 + forecast_count + 1,
        callback=progress_callback,
        cancellation_token=cancellation,
    )
    preparation = tracker.begin_stage(STAGE_PREPARING_TRAJECTORY, 2)
    preparation.emit(1, message="Loaded authenticated sparse batch/MCMC artifact")
    bag = request.held_out_bag
    tracker.checkpoint()
    if file_sha256(bag.path) != bag.sha256:
        raise ValueError("held-out rosbag SHA-256 disagrees with request")
    flight = flight_loader(
        path=str(bag.path),
        start_local=bag.interval_seconds[0],
        end_local=bag.interval_seconds[1],
        compute_sha256=False,
        checkpoint=tracker.checkpoint,
        bag_id=bag.bag_id,
    )
    if not isinstance(flight, FlightData):
        raise TypeError("flight_loader must return FlightData")
    preparation.emit(
        2,
        bag_id=bag.bag_id,
        message="Loaded held-out observations, references, and controller snapshot",
    )
    actuator_parameters = _actuator_parameters(run.manifest)
    prepared = prepare_held_out_forecast(
        request,
        flight,
        posterior,
        actuator_parameters,
    )
    current = configuration_from_controller_snapshot(
        flight.controller_snapshot, bag.bag_id
    )
    candidate = current_pid_candidate(current)
    q_quantity, q_interval_model = _q_contract(run)
    discrepancy = ModelDiscrepancyConfiguration(
        policy=request.discrepancy_policy,
        diagonal_q=run.map_static["q_diagonal"],
        base_seed=request.discrepancy_base_seed,
        residual_quantity=q_quantity,
        interval_model=q_interval_model,
        replicates=request.discrepancy_replicates,
    )
    evaluation = tracker.begin_stage(
        STAGE_OPTIMIZING_FULL_TRAJECTORY, forecast_count
    )
    records = []
    completed = 0
    for sample in posterior.samples:
        for replicate in range(request.discrepancy_replicates):
            tracker.checkpoint()
            realization = discrepancy.realization(
                sample.sample_id, bag.bag_id, replicate
            )
            outcome = run_pid_forecast(
                candidate, sample, prepared.scenario, realization
            )
            records.append(
                metric_record_from_outcome(
                    outcome, sample, prepared, replicate
                )
            )
            completed += 1
            evaluation.emit(
                completed,
                bag_id=bag.bag_id,
                sample_id=sample.sample_id,
                message="replicate={}".format(replicate),
            )
    result = summarize_held_out_records(
        records,
        quantile_level=request.quantile_level,
        cvar_level=request.cvar_level,
    )
    source_modes = tuple(
        dict.fromkeys(value.source_mode_id for value in posterior.samples)
    )
    if len(source_modes) != 1:
        raise ValueError("selected held-out posterior contains multiple modes")
    writing = tracker.begin_stage(STAGE_WRITING_ARTIFACTS, 1)
    artifact = write_held_out_validation(
        request.output_directory,
        request=request,
        identity=HeldOutValidationIdentity(
            source_estimation_run_id=str(run.manifest["run_id"]),
            source_estimation_request_fingerprint=str(
                run.manifest["request_fingerprint"]
            ),
            source_estimator_revision=str(run.manifest["estimator_revision"]),
            selected_mode_id=source_modes[0],
        ),
        selected_posterior=posterior,
        result=result,
        q_quantity=q_quantity,
        q_interval_model=q_interval_model,
        q_diagonal=run.map_static["q_diagonal"],
        source_actuator_model=run.manifest["actuator_model"],
    )
    writing.complete(message="Held-out validation artifact complete")
    return artifact


def _signal_reason(signum: int) -> str:
    try:
        return "signal_{}".format(signal.Signals(signum).name)
    except ValueError:
        return "signal_{}".format(signum)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate retained sparse-batch MCMC plants on a declared "
            "held-out or tuning flight."
        )
    )
    parser.add_argument("--request", required=True)
    arguments = parser.parse_args(argv)
    cancellation = CancellationToken()

    def request_cancel(signum, _frame):
        cancellation.cancel(_signal_reason(signum))

    previous = {
        signum: signal.signal(signum, request_cancel)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        request = load_held_out_validation_request(arguments.request)
        artifact = execute_held_out_validation(
            request,
            cancellation_token=cancellation,
            progress_callback=JsonlProgressWriter(sys.stdout),
        )
        print(
            "{} complete: {}".format(
                artifact.manifest["semantic_label"], artifact.root
            ),
            file=sys.stderr,
        )
        return 0
    except Exception as error:  # pylint: disable=broad-except
        print("held-out validation failed: {}".format(error), file=sys.stderr)
        return 2 if cancellation.cancelled else 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


__all__ = [
    "CONFIGURATION_COMPATIBILITY_STATUSES",
    "DATA_SPLIT_ROLES",
    "HELD_OUT_COST_METRICS",
    "HELD_OUT_VALIDATION_ARRAY_SCHEMA",
    "HELD_OUT_VALIDATION_REQUEST_SCHEMA",
    "HELD_OUT_VALIDATION_SCHEMA",
    "POSTERIOR_SUBSET_METHODS",
    "STRICT_HOLD_OUT",
    "TUNING_EVALUATION",
    "ConfigurationCompatibility",
    "DataSplitDeclaration",
    "HeldOutBagRequest",
    "HeldOutMetricRecord",
    "HeldOutValidationArtifact",
    "HeldOutValidationIdentity",
    "HeldOutValidationRequest",
    "HeldOutValidationResult",
    "execute_held_out_validation",
    "load_held_out_validation",
    "load_held_out_validation_request",
    "main",
    "metric_record_from_outcome",
    "prepare_held_out_forecast",
    "summarize_held_out_records",
    "validate_data_split_against_source",
    "validate_held_out_validation_request",
    "write_held_out_validation",
]


if __name__ == "__main__":
    raise SystemExit(main())
