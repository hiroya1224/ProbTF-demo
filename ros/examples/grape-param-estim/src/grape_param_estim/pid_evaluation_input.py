"""Strict completed-run adapter for production PID proposal evaluation."""

from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from grape_param_estim.artifact_io import (
    ASSIMILATION_RUN_SCHEMA,
    ArtifactValidationError,
    AssimilationRunBundle,
    load_assimilation_run,
    read_json,
)
from grape_param_estim.controller import ControllerConfig, PIDConfig
from grape_param_estim.controller_config import (
    PidGainConfiguration,
    configuration_from_controller_snapshot,
    select_baseline_pid_configuration,
)
from grape_param_estim.pid_proposal import (
    PidGainCandidate,
    current_pid_candidate,
    derive_pid_proposal_ensemble,
    member_pid_candidate,
    user_pid_candidate,
)
from grape_param_estim.posterior_predictive import (
    CounterfactualBagScenario,
    ErrorThresholds,
    PosteriorPredictiveInput,
    RESIDUAL_POLICIES,
)
from grape_param_estim.progress import CancellationToken
from grape_param_estim.real_rosbag import (
    ControllerGainSnapshot,
    PID_AXIS_NAMES,
    PID_CONFIG_FIELD_NAMES,
)
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    ControllerState,
    GrapeGeometry,
    ReferenceState,
    RigidBodyState,
    VehicleParameters,
)


PID_EVALUATION_REQUEST_SCHEMA = (
    "grape-param-estim/pid-evaluation-request/v1"
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _strict_keys(value, required, optional, location):
    if not isinstance(value, dict):
        raise ValueError("{} must be an object".format(location))
    missing = set(required) - set(value)
    unknown = set(value) - set(required) - set(optional)
    if missing:
        raise ValueError(
            "{} is missing: {}".format(
                location, ", ".join(sorted(missing))
            )
        )
    if unknown:
        raise ValueError(
            "{} has unknown fields: {}".format(
                location, ", ".join(sorted(unknown))
            )
        )


def _identifier(value: Any, name: str) -> str:
    result = str(value)
    if not _IDENTIFIER.match(result):
        raise ValueError("{} is not a safe identifier".format(name))
    return result


@dataclass(frozen=True)
class PidEvaluationCandidateRequest:
    candidate_id: str
    source: str
    source_member_id: Optional[int] = None
    configuration: Optional[PidGainConfiguration] = None

    def __post_init__(self) -> None:
        identifier = _identifier(self.candidate_id, "candidate_id")
        source = str(self.source)
        if source not in {"current", "member-derived", "user"}:
            raise ValueError("unknown PID candidate source")
        member = self.source_member_id
        configuration = self.configuration
        if source == "current":
            if identifier != "current" or member is not None or configuration is not None:
                raise ValueError(
                    "current must use candidate_id=current and no supplied gains"
                )
        elif source == "member-derived":
            if (
                isinstance(member, bool)
                or not isinstance(member, (int, np.integer))
                or int(member) < 0
                or configuration is not None
            ):
                raise ValueError(
                    "member-derived candidate needs one source_member_id"
                )
            member = int(member)
        elif member is not None or not isinstance(
            configuration, PidGainConfiguration
        ):
            raise ValueError("user candidate needs exact 4x3 gain values")
        if source != "current" and identifier == "current":
            raise ValueError("candidate_id=current is reserved")
        object.__setattr__(self, "candidate_id", identifier)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "source_member_id", member)


@dataclass(frozen=True)
class PidEvaluationRequest:
    evaluation_id: str
    assimilation_run: str
    baseline_bag_id: str
    residual_policy: Any
    cvar_level: float
    thresholds: ErrorThresholds
    candidates: Tuple[PidEvaluationCandidateRequest, ...]
    selected_candidate_id: Optional[str]

    def __post_init__(self) -> None:
        evaluation_id = _identifier(self.evaluation_id, "evaluation_id")
        run = str(self.assimilation_run)
        baseline = str(self.baseline_bag_id)
        candidates = tuple(self.candidates)
        level = float(self.cvar_level)
        if not run:
            raise ValueError("assimilation_run cannot be empty")
        if not baseline:
            raise ValueError("baseline_bag_id cannot be empty")
        if not np.isfinite(level) or not 0.0 <= level < 1.0:
            raise ValueError("cvar_level must be in [0, 1)")
        if not isinstance(self.thresholds, ErrorThresholds):
            raise TypeError("thresholds must be ErrorThresholds")
        if (
            not candidates
            or any(
                not isinstance(value, PidEvaluationCandidateRequest)
                for value in candidates
            )
            or len({value.candidate_id for value in candidates})
            != len(candidates)
            or sum(value.source == "current" for value in candidates) != 1
        ):
            raise ValueError(
                "candidate requests must be unique and contain current once"
            )
        policy = self.residual_policy
        if isinstance(policy, str):
            if policy not in RESIDUAL_POLICIES:
                raise ValueError("unknown residual policy")
        else:
            try:
                policy_items = tuple(
                    sorted(
                        (str(key), str(value))
                        for key, value in dict(policy).items()
                    )
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "residual_policy must be a policy or bag mapping"
                ) from error
            if not policy_items or any(
                not key or value not in RESIDUAL_POLICIES
                for key, value in policy_items
            ):
                raise ValueError("residual policy mapping is invalid")
            policy = policy_items
        selected = (
            None
            if self.selected_candidate_id is None
            else str(self.selected_candidate_id)
        )
        if selected is not None and selected not in {
            value.candidate_id for value in candidates
        }:
            raise ValueError("selected_candidate_id is not a candidate")
        object.__setattr__(self, "evaluation_id", evaluation_id)
        object.__setattr__(self, "assimilation_run", run)
        object.__setattr__(self, "baseline_bag_id", baseline)
        object.__setattr__(self, "residual_policy", policy)
        object.__setattr__(self, "cvar_level", level)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "selected_candidate_id", selected)

    def residual_policies(
        self, bag_ids: Tuple[str, ...]
    ) -> Mapping[str, str]:
        identifiers = tuple(str(value) for value in bag_ids)
        if isinstance(self.residual_policy, str):
            return {value: self.residual_policy for value in identifiers}
        policies = dict(self.residual_policy)
        if set(policies) != set(identifiers):
            raise ValueError(
                "residual_policy mapping must cover exactly the selected bags"
            )
        return policies


def _candidate_request(value: Mapping[str, Any], index: int):
    location = "candidate {}".format(index)
    if not isinstance(value, dict):
        raise ValueError("{} must be an object".format(location))
    source = value.get("source")
    if source == "current":
        _strict_keys(
            value, ("candidate_id", "source"), tuple(), location
        )
        return PidEvaluationCandidateRequest(
            value["candidate_id"], source
        )
    if source == "member-derived":
        _strict_keys(
            value,
            ("candidate_id", "source", "source_member_id"),
            tuple(),
            location,
        )
        return PidEvaluationCandidateRequest(
            value["candidate_id"],
            source,
            source_member_id=value["source_member_id"],
        )
    if source == "user":
        _strict_keys(
            value,
            ("candidate_id", "source", "values"),
            tuple(),
            location,
        )
        return PidEvaluationCandidateRequest(
            value["candidate_id"],
            source,
            configuration=PidGainConfiguration(value["values"]),
        )
    raise ValueError("{} has unknown source".format(location))


def load_pid_evaluation_request(path: str) -> PidEvaluationRequest:
    """Strictly parse one v1 worker request and resolve its source run."""

    source = Path(path).expanduser().resolve()
    value = read_json(source)
    _strict_keys(
        value,
        (
            "schema",
            "evaluation_id",
            "assimilation_run",
            "baseline_bag_id",
            "residual_policy",
            "cvar_level",
            "thresholds",
            "candidates",
            "selected_candidate_id",
        ),
        tuple(),
        "PID evaluation request",
    )
    if value["schema"] != PID_EVALUATION_REQUEST_SCHEMA:
        raise ValueError("unsupported PID evaluation request schema")
    thresholds = value["thresholds"]
    _strict_keys(
        thresholds,
        (
            "position",
            "orientation",
            "position_metric",
            "orientation_metric",
        ),
        tuple(),
        "thresholds",
    )
    raw_candidates = value["candidates"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidates must be a non-empty list")
    run = Path(str(value["assimilation_run"])).expanduser()
    if not run.is_absolute():
        run = source.parent / run
    return PidEvaluationRequest(
        evaluation_id=value["evaluation_id"],
        assimilation_run=str(run.resolve()),
        baseline_bag_id=value["baseline_bag_id"],
        residual_policy=value["residual_policy"],
        cvar_level=value["cvar_level"],
        thresholds=ErrorThresholds(**thresholds),
        candidates=tuple(
            _candidate_request(candidate, index)
            for index, candidate in enumerate(raw_candidates)
        ),
        selected_candidate_id=value["selected_candidate_id"],
    )


def _required(arrays, keys, location):
    missing = [key for key in keys if key not in arrays]
    if missing:
        raise ArtifactValidationError(
            "{} is missing PID evaluation fields: {}".format(
                location, ", ".join(missing)
            )
        )


def _finite_array(arrays, key, shape, location):
    value = np.asarray(arrays[key])
    if (
        value.shape != shape
        or not np.issubdtype(value.dtype, np.number)
        or np.any(~np.isfinite(value))
    ):
        raise ArtifactValidationError(
            "{}:{} must have finite shape {}".format(location, key, shape)
        )
    return value.astype(float, copy=True)


def _boolean_array(arrays, key, shape, location):
    value = np.asarray(arrays[key])
    if value.shape != shape or not np.issubdtype(value.dtype, np.bool_):
        raise ArtifactValidationError(
            "{}:{} must have boolean shape {}".format(location, key, shape)
        )
    return value.astype(bool, copy=True)


def _strings(arrays, key, shape, location):
    value = np.asarray(arrays[key])
    if value.shape != shape or value.dtype.kind not in {"U", "S"}:
        raise ArtifactValidationError(
            "{}:{} must have string shape {}".format(location, key, shape)
        )
    result = value.astype(str)
    if np.any(result == ""):
        raise ArtifactValidationError(
            "{}:{} cannot contain empty strings".format(location, key)
        )
    return result


def _scalar(arrays, key, location):
    return float(_finite_array(arrays, key, (1,), location)[0])


def _scalar_string(arrays, key, location):
    return str(_strings(arrays, key, (1,), location)[0])


def _controller_snapshot(arrays, bag_id, location):
    required = (
        "controller_snapshot_groups",
        "controller_snapshot_record_times",
        "controller_snapshot_gains",
        "controller_snapshot_pid_control_flags",
        "controller_snapshot_source_kinds",
    )
    _required(arrays, required, location)
    snapshot = ControllerGainSnapshot(
        groups=tuple(
            _strings(
                arrays, "controller_snapshot_groups", (4,), location
            )
        ),
        record_times=_finite_array(
            arrays, "controller_snapshot_record_times", (4,), location
        ),
        gains=_finite_array(
            arrays, "controller_snapshot_gains", (4, 3), location
        ),
        pid_control_flags=_boolean_array(
            arrays,
            "controller_snapshot_pid_control_flags",
            (4,),
            location,
        ),
        source_kinds=tuple(
            _strings(
                arrays,
                "controller_snapshot_source_kinds",
                (4,),
                location,
            )
        ),
    )
    return snapshot, configuration_from_controller_snapshot(snapshot, bag_id)


def _controller_configuration(arrays, location):
    required = (
        "controller_pid_axis_names",
        "controller_pid_field_names",
        "controller_pid_configuration",
        "controller_xy_control_mode",
        "controller_need_yaw_d_control",
        "controller_start_roll_pitch_integration_height",
        "controller_initial_height",
        "controller_source_compatible_gyro_term",
    )
    _required(arrays, required, location)
    axes = tuple(
        _strings(
            arrays,
            "controller_pid_axis_names",
            (len(PID_AXIS_NAMES),),
            location,
        )
    )
    fields = tuple(
        _strings(
            arrays,
            "controller_pid_field_names",
            (len(PID_CONFIG_FIELD_NAMES),),
            location,
        )
    )
    if axes != PID_AXIS_NAMES or fields != PID_CONFIG_FIELD_NAMES:
        raise ArtifactValidationError(
            "{} controller PID ordering is not canonical".format(location)
        )
    values = _finite_array(
        arrays,
        "controller_pid_configuration",
        (len(PID_AXIS_NAMES), len(PID_CONFIG_FIELD_NAMES)),
        location,
    )
    return ControllerConfig(
        pid=tuple(
            PIDConfig(**dict(zip(PID_CONFIG_FIELD_NAMES, row)))
            for row in values
        ),
        xy_control_mode=_scalar_string(
            arrays, "controller_xy_control_mode", location
        ),
        need_yaw_d_control=bool(
            _boolean_array(
                arrays,
                "controller_need_yaw_d_control",
                (1,),
                location,
            )[0]
        ),
        start_roll_pitch_integration_height=_scalar(
            arrays,
            "controller_start_roll_pitch_integration_height",
            location,
        ),
        initial_height=_scalar(
            arrays, "controller_initial_height", location
        ),
        source_compatible_gyro_term=bool(
            _boolean_array(
                arrays,
                "controller_source_compatible_gyro_term",
                (1,),
                location,
            )[0]
        ),
    )


def _actuator_parameters(arrays, location):
    keys = (
        "actuator_thrust_time_constant",
        "actuator_gimbal_time_constant",
        "actuator_minimum_thrust",
        "actuator_maximum_thrust",
        "actuator_maximum_gimbal_angle",
        "actuator_maximum_gimbal_rate",
    )
    _required(arrays, keys, location)
    return ActuatorParameters(
        thrust_time_constant=_scalar(
            arrays, "actuator_thrust_time_constant", location
        ),
        gimbal_time_constant=_scalar(
            arrays, "actuator_gimbal_time_constant", location
        ),
        delay=0.0,
        minimum_thrust=_scalar(
            arrays, "actuator_minimum_thrust", location
        ),
        maximum_thrust=_scalar(
            arrays, "actuator_maximum_thrust", location
        ),
        maximum_gimbal_angle=_scalar(
            arrays, "actuator_maximum_gimbal_angle", location
        ),
        maximum_gimbal_rate=_scalar(
            arrays, "actuator_maximum_gimbal_rate", location
        ),
    )


def _selected_mode(shared, location):
    _required(shared, ("mode_id", "mode_weight", "selected_mode_id"), location)
    mode_ids = _strings(
        shared, "mode_id", np.asarray(shared["mode_id"]).shape, location
    )
    if mode_ids.ndim != 1 or mode_ids.size < 1:
        raise ArtifactValidationError("{}:mode_id must be a vector".format(location))
    weights = _finite_array(
        shared, "mode_weight", (mode_ids.size,), location
    )
    selected = _scalar_string(shared, "selected_mode_id", location)
    if (
        selected not in set(mode_ids.tolist())
        or np.any(weights < 0.0)
        or float(np.sum(weights)) <= 0.0
    ):
        raise ArtifactValidationError("{} mode law is invalid".format(location))
    return selected


def _physical_members(bundle: AssimilationRunBundle):
    shared = bundle.shared_posterior
    members = np.asarray(shared["member_id"], dtype=np.int64)
    count = members.size
    location = "shared_posterior"
    values = {
        "mass": _finite_array(shared, "mass", (count,), location),
        "inertia": _finite_array(shared, "inertia", (count, 3, 3), location),
        "cog": _finite_array(shared, "cog", (count, 3), location),
        "force": _finite_array(
            shared, "force_effectiveness", (count, 4), location
        ),
        "torque": _finite_array(
            shared, "torque_effectiveness", (count, 4), location
        ),
        "delay": _finite_array(
            shared, "constant_delay", (count,), location
        ),
    }
    nominal = VehicleParameters.nominal()
    physical = tuple(
        VehicleParameters(
            mass=values["mass"][index],
            inertia=values["inertia"][index],
            cog_offset=values["cog"][index],
            force_effectiveness=values["force"][index],
            torque_effectiveness=values["torque"][index],
            linear_drag=nominal.linear_drag,
            angular_drag=nominal.angular_drag,
        )
        for index in range(count)
    )
    return members, physical, values["delay"], _selected_mode(shared, location)


def _bag_scenario(
    bag_id,
    arrays,
    member_count,
    residual_policy,
    nominal_parameters,
    geometry,
):
    location = "bag {}".format(bag_id)
    times = _finite_array(
        arrays, "times", (np.asarray(arrays["times"]).size,), location
    )
    samples = times.size
    reference_keys = (
        "reference_position",
        "reference_linear_velocity",
        "reference_linear_acceleration",
        "reference_rpy",
        "reference_angular_velocity",
        "reference_angular_acceleration",
    )
    state_keys = (
        "initial_position",
        "initial_orientation_xyzw",
        "initial_linear_velocity",
        "initial_angular_velocity",
        "initial_controller_integral",
        "initial_controller_roll_pitch_integration_active",
        "initial_actuator_thrust",
        "initial_actuator_gimbal_angle",
        "residual_wrench_interval",
    )
    _required(arrays, reference_keys + state_keys, location)
    references = {
        key: _finite_array(arrays, key, (samples, 3), location)
        for key in reference_keys
    }
    reference_states = tuple(
        ReferenceState(
            position=references["reference_position"][index],
            linear_velocity=references["reference_linear_velocity"][index],
            linear_acceleration=references[
                "reference_linear_acceleration"
            ][index],
            rpy=references["reference_rpy"][index],
            angular_velocity=references[
                "reference_angular_velocity"
            ][index],
            angular_acceleration=references[
                "reference_angular_acceleration"
            ][index],
        )
        for index in range(samples)
    )
    position = _finite_array(
        arrays, "initial_position", (member_count, 3), location
    )
    orientation = _finite_array(
        arrays,
        "initial_orientation_xyzw",
        (member_count, 4),
        location,
    )
    linear_velocity = _finite_array(
        arrays,
        "initial_linear_velocity",
        (member_count, 3),
        location,
    )
    angular_velocity = _finite_array(
        arrays,
        "initial_angular_velocity",
        (member_count, 3),
        location,
    )
    initial_states = tuple(
        RigidBodyState(
            position[index],
            orientation[index],
            linear_velocity[index],
            angular_velocity[index],
        )
        for index in range(member_count)
    )
    integral = _finite_array(
        arrays,
        "initial_controller_integral",
        (member_count, 6),
        location,
    )
    integration_active = _boolean_array(
        arrays,
        "initial_controller_roll_pitch_integration_active",
        (member_count,),
        location,
    )
    controller_states = tuple(
        ControllerState(integral[index], bool(integration_active[index]))
        for index in range(member_count)
    )
    thrust = _finite_array(
        arrays, "initial_actuator_thrust", (member_count, 4), location
    )
    gimbal = _finite_array(
        arrays,
        "initial_actuator_gimbal_angle",
        (member_count, 4),
        location,
    )
    actuator_states = tuple(
        ActuatorState(thrust[index], gimbal[index])
        for index in range(member_count)
    )
    residual = _finite_array(
        arrays,
        "residual_wrench_interval",
        (member_count, samples - 1, 6),
        location,
    )
    controller_configuration = _controller_configuration(arrays, location)
    provenance = (("bag_id", str(bag_id)),)
    if "provenance_bag_sha256" in arrays:
        provenance += (
            (
                "bag_sha256",
                _scalar_string(arrays, "provenance_bag_sha256", location),
            ),
        )
    return CounterfactualBagScenario(
        bag_id=str(bag_id),
        times=times,
        references=reference_states,
        initial_states=initial_states,
        initial_controller_states=controller_states,
        initial_actuator_states=actuator_states,
        posterior_residual_wrench=residual,
        controller_configuration=controller_configuration,
        controller_nominal_parameters=nominal_parameters,
        controller_geometry=geometry,
        plant_geometry=geometry,
        actuator_parameters=_actuator_parameters(arrays, location),
        residual_policy=str(residual_policy),
        provenance=provenance,
    )


def input_from_assimilation_run(
    run_directory: str,
    baseline_bag_id: str,
    residual_policy: Any = "posterior_replay",
    cancellation_token: Optional[CancellationToken] = None,
) -> PosteriorPredictiveInput:
    """Restore every raw member and bag-local state from a complete run."""

    cancellation = (
        CancellationToken()
        if cancellation_token is None
        else cancellation_token
    )
    if not isinstance(cancellation, CancellationToken):
        raise TypeError("cancellation_token must be a CancellationToken")
    cancellation.raise_if_cancelled()
    bundle = load_assimilation_run(run_directory)
    cancellation.raise_if_cancelled()
    if bundle.manifest.get("schema") != ASSIMILATION_RUN_SCHEMA:
        raise ArtifactValidationError("source is not an assimilation run")
    bag_ids = tuple(bundle.manifest["selected_bag_ids"])
    baseline = str(baseline_bag_id)
    if baseline not in bag_ids:
        raise ValueError("baseline_bag_id is not a selected bag")
    if isinstance(residual_policy, str):
        if residual_policy not in RESIDUAL_POLICIES:
            raise ValueError("unknown residual policy")
        policies = {value: residual_policy for value in bag_ids}
    else:
        policies = {
            str(key): str(value)
            for key, value in dict(residual_policy).items()
        }
        if set(policies) != set(bag_ids) or any(
            value not in RESIDUAL_POLICIES for value in policies.values()
        ):
            raise ValueError(
                "residual policy mapping must cover selected bags"
            )
    member_ids, physical, delay, selected_mode = _physical_members(bundle)
    snapshots = {}
    for bag_id in bag_ids:
        cancellation.raise_if_cancelled()
        _snapshot, snapshots[bag_id] = _controller_snapshot(
            bundle.bags[bag_id], bag_id, "bag {}".format(bag_id)
        )
    current = select_baseline_pid_configuration(snapshots, baseline)
    nominal_parameters = VehicleParameters.nominal()
    geometry = GrapeGeometry.grape()
    proposals = derive_pid_proposal_ensemble(
        member_id=member_ids,
        physical_parameter_members=physical,
        constant_delay=delay,
        source_mode_id=tuple(selected_mode for _value in member_ids),
        controller_nominal_parameters=nominal_parameters,
        geometry=geometry,
        current=current,
    )
    scenarios = []
    for bag_id in bag_ids:
        cancellation.raise_if_cancelled()
        scenarios.append(
            _bag_scenario(
                bag_id,
                bundle.bags[bag_id],
                member_ids.size,
                policies[bag_id],
                nominal_parameters,
                geometry,
            )
        )
    cancellation.raise_if_cancelled()
    provenance = (
        ("source_run_id", str(bundle.manifest["run_id"])),
        ("source_run_path", str(bundle.root)),
        ("baseline_bag_id", baseline),
    )
    if "request_fingerprint" in bundle.manifest:
        provenance += (
            (
                "source_request_fingerprint",
                str(bundle.manifest["request_fingerprint"]),
            ),
        )
    return PosteriorPredictiveInput(
        selected_mode_id=selected_mode,
        physical_parameter_members=physical,
        proposal_ensemble=proposals,
        bags=tuple(scenarios),
        provenance=provenance,
    )


def candidates_from_request(
    request: PidEvaluationRequest,
    predictive_input: PosteriorPredictiveInput,
) -> Tuple[PidGainCandidate, ...]:
    """Resolve exact current/member/user candidates without averaging."""

    if not isinstance(request, PidEvaluationRequest):
        raise TypeError("request must be PidEvaluationRequest")
    if not isinstance(predictive_input, PosteriorPredictiveInput):
        raise TypeError("predictive_input has the wrong type")
    candidates = []
    for value in request.candidates:
        if value.source == "current":
            candidate = current_pid_candidate(predictive_input.current)
        elif value.source == "member-derived":
            candidate = member_pid_candidate(
                predictive_input.proposal_ensemble,
                int(value.source_member_id),
            )
            if candidate.candidate_id != value.candidate_id:
                candidate = replace(
                    candidate, candidate_id=value.candidate_id
                )
        else:
            candidate = user_pid_candidate(
                value.candidate_id, value.configuration
            )
        candidates.append(candidate)
    return tuple(candidates)


__all__ = [
    "PID_EVALUATION_REQUEST_SCHEMA",
    "PidEvaluationCandidateRequest",
    "PidEvaluationRequest",
    "candidates_from_request",
    "input_from_assimilation_run",
    "load_pid_evaluation_request",
]
