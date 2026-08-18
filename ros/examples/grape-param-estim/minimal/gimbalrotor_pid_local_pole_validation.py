#!/usr/bin/env python3
"""Local sampled-data pole validation for one recorded Gimbalrotor flight.

The controller always uses the PID snapshot recorded in the selected bag and
the nominal allocation model.  The real plant is sampled in the estimator's
native 13-D common-scale quotient coordinate.  A fitted rotor lag is modelled
as an exact, thrust-only ZOH queue; gimbal commands are never delayed by it.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from grape_param_estim.controller import (  # noqa: E402
    ControllerConfig,
    GrapeController,
    initial_controller_state,
)
from grape_param_estim.controller_config import (  # noqa: E402
    PID_GAIN_NAMES,
    PID_GROUPS,
    PidGainConfiguration,
    apply_pid_gain_configuration,
)
from grape_param_estim.dynamics import (  # noqa: E402
    FullSixDofPlant,
    advance_actuators,
)
from grape_param_estim.geometry import (  # noqa: E402
    matrix_to_quaternion,
    quaternion_to_matrix,
    so3_exp,
    so3_log,
)
from grape_param_estim.gimbalrotor_pid_postprocess import (  # noqa: E402
    PostprocessInputError,
    PostprocessNumericalError,
    ScaleFreePlant,
    build_controller_snapshot_geometry,
    load_bag_provenance,
    load_controller_yaml,
    load_estimator_result,
    load_vehicle_model,
)
from grape_param_estim.system import (  # noqa: E402
    ActuatorCommand,
    ActuatorParameters,
    ActuatorState,
    ControllerState,
    ReferenceState,
    RigidBodyState,
    VehicleParameters,
)
from gimbalrotor_pid_monte_carlo_postprocess import (  # noqa: E402
    SCALE_FREE_LABELS,
    load_static_postprocess_baseline,
)
from gimbalrotor_pid_postprocess_sensitivity import (  # noqa: E402
    _psd_eigendecomposition,
    load_sensitivity_artifacts,
    prepare_sampling_coordinates,
    source_commit,
    write_json,
)


LOCAL_POLE_SCHEMA = "grape-param-estim/gimbalrotor-pid-local-poles/v1"
STATUS_SCHEMA = "grape-param-estim/gimbalrotor-pid-local-poles-status/v1"
COVARIANCE_MODES = ("conservative_fusion", "overlap_corrected")
DELAY_MODES = ("fitted_thrust_delay", "zero_thrust_delay")
DEFAULT_SAMPLE_COUNT = 512
DEFAULT_SEED = 0
DEFAULT_FD_CHECK_SAMPLES = 4
PREFIX_COUNTS = (128, 256, 512)
QUANTILE_LEVELS = (0.025, 0.16, 0.5, 0.84, 0.975)
QUANTILE_NAMES = ("q025", "q16", "q50", "q84", "q975")
PLAN_BASE_COMMIT = "aba27b2e51efab80271aa6cd94cd8e521a3a2efd"


class LocalPoleNumericalError(RuntimeError):
    """A narrowly identified non-finite numerical sample failure."""


NUMERICAL_SAMPLE_EXCEPTIONS = (
    LocalPoleNumericalError,
    PostprocessNumericalError,
    np.linalg.LinAlgError,
    FloatingPointError,
)


@dataclass(frozen=True)
class DelayDecomposition:
    delay_seconds: float
    controller_dt: float
    whole_steps: int
    remainder_seconds: float
    depth: int

    def __post_init__(self) -> None:
        values = np.asarray(
            (self.delay_seconds, self.controller_dt, self.remainder_seconds),
            dtype=float,
        )
        if (
            np.any(~np.isfinite(values))
            or self.delay_seconds < 0.0
            or self.controller_dt <= 0.0
            or self.whole_steps < 0
            or self.depth < 0
            or self.remainder_seconds < 0.0
            or self.remainder_seconds >= self.controller_dt
        ):
            raise ValueError("invalid thrust-delay decomposition")


@dataclass(frozen=True)
class AugmentedState:
    rigid_body: RigidBodyState
    controller: ControllerState
    actuators: ActuatorState
    thrust_queue: np.ndarray

    def __post_init__(self) -> None:
        queue = np.asarray(self.thrust_queue, dtype=float)
        if queue.ndim != 2 or queue.shape[1] != 4 or np.any(~np.isfinite(queue)):
            raise ValueError("thrust delay queue must be a finite depth-by-4 array")
        object.__setattr__(self, "thrust_queue", queue.copy())


@dataclass(frozen=True)
class TrimResult:
    state: AugmentedState
    issued_command: ActuatorCommand
    root_status: int
    root_message: str
    residual: np.ndarray
    residual_norm: float
    one_step_defect: np.ndarray
    one_step_defect_norm: float
    controller_integral_defect: np.ndarray
    gimbal_fixed_point_defect: np.ndarray
    equilibrium_tolerance: float
    equilibrium_valid: bool
    piecewise_linearization_near_kink: bool


@dataclass(frozen=True)
class PoleContext:
    controller: GrapeController
    plant: FullSixDofPlant
    actuator_parameters: ActuatorParameters
    reference: ReferenceState
    controller_dt: float
    delay: DelayDecomposition
    trim: AugmentedState

    @property
    def local_dimension(self) -> int:
        return 26 + 4 * self.delay.depth


@dataclass(frozen=True)
class CaseInputs:
    result_path: Path
    arrays_path: Path
    static_path: Path
    arguments_path: Path
    bag_json_path: Path
    controller_yaml_path: Path
    vehicle_model_path: Path
    result: Any
    sampling_coordinates: Any
    static_baseline: Any
    static_payload: Mapping[str, Any]
    arguments: Mapping[str, Any]
    bag: Any
    controller_yaml: Any
    vehicle_model: Any
    controller_configuration: ControllerConfig
    actuator_parameters: ActuatorParameters
    recorded_gains: PidGainConfiguration


_TIMING_CACHE: dict[tuple[str, float, float], Mapping[str, Any]] = {}


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _read_json(path: Path, label: str) -> tuple[Path, Mapping[str, Any]]:
    source = _resolved(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PostprocessInputError("{} cannot be read: {}".format(label, source)) from error
    if not isinstance(payload, Mapping):
        raise PostprocessInputError("{} must contain a JSON object".format(label))
    return source, payload


def _same_path(actual: Any, expected: Path, label: str) -> None:
    if _resolved(Path(str(actual))) != _resolved(expected):
        raise PostprocessInputError("{} provenance does not match".format(label))


def _nested_recorded_gains(payload: Mapping[str, Any]) -> PidGainConfiguration:
    snapshot = payload.get("controller_gain_snapshot")
    if not isinstance(snapshot, Mapping):
        raise PostprocessInputError("static PID snapshot is missing")
    gains = snapshot.get("gains")
    if not isinstance(gains, Mapping):
        raise PostprocessInputError("static recorded PID gains are missing")
    try:
        values = np.asarray(
            [
                [float(gains[group][gain]) for gain in PID_GAIN_NAMES]
                for group in PID_GROUPS
            ],
            dtype=float,
        )
    except (KeyError, TypeError) as error:
        raise PostprocessInputError("static recorded PID gains are incomplete") from error
    return PidGainConfiguration(values)


def _actuator_parameters(arguments: Mapping[str, Any]) -> ActuatorParameters:
    fields = (
        "thrust_time_constant",
        "gimbal_time_constant",
        "minimum_thrust",
        "maximum_thrust",
        "maximum_gimbal_angle",
        "maximum_gimbal_rate",
    )
    missing = [name for name in fields if name not in arguments]
    if missing:
        raise PostprocessInputError(
            "arguments JSON is missing actuator fields: {}".format(", ".join(missing))
        )
    try:
        return ActuatorParameters(
            thrust_time_constant=float(arguments["thrust_time_constant"]),
            gimbal_time_constant=float(arguments["gimbal_time_constant"]),
            delay=0.0,
            minimum_thrust=float(arguments["minimum_thrust"]),
            maximum_thrust=float(arguments["maximum_thrust"]),
            maximum_gimbal_angle=float(arguments["maximum_gimbal_angle"]),
            maximum_gimbal_rate=float(arguments["maximum_gimbal_rate"]),
        )
    except (TypeError, ValueError) as error:
        raise PostprocessInputError("arguments actuator model is invalid") from error


def load_case_inputs(
    *,
    result_path: Path,
    arrays_path: Path,
    static_postprocess_path: Path,
    arguments_path: Path,
    bag_json_path: Path,
    controller_yaml_path: Path,
    vehicle_model_path: Path,
    covariance_mode: str,
) -> CaseInputs:
    """Load and cross-check the complete scientific input contract."""

    result_path = _resolved(result_path)
    arrays_path = _resolved(arrays_path)
    static_path = _resolved(static_postprocess_path)
    arguments_path = _resolved(arguments_path)
    bag_json_path = _resolved(bag_json_path)
    controller_yaml_path = _resolved(controller_yaml_path)
    vehicle_model_path = _resolved(vehicle_model_path)
    if arrays_path.parent != result_path.parent or arguments_path.parent != result_path.parent:
        raise PostprocessInputError(
            "result, arrays, and arguments must be siblings in one estimator case"
        )

    result = load_estimator_result(result_path)
    artifacts = load_sensitivity_artifacts(
        arrays_path, covariance_mode=covariance_mode
    )
    model = load_vehicle_model(vehicle_model_path)
    coordinates = prepare_sampling_coordinates(
        result=result,
        artifacts=artifacts,
        model=model,
        coordinate_mode="estimator_quotient",
    )
    baseline = load_static_postprocess_baseline(static_path)
    _, static_payload = _read_json(static_path, "static PID postprocess JSON")
    _, arguments = _read_json(arguments_path, "estimator arguments JSON")
    bag = load_bag_provenance(bag_json_path)
    controller_yaml = load_controller_yaml(controller_yaml_path)

    if baseline.estimator_source_commit != result.source_commit:
        raise PostprocessInputError("static PID estimator source commit differs")
    if baseline.estimator_case_name != result.case_name:
        raise PostprocessInputError("static PID estimator case differs")
    static_input = static_payload.get("input")
    snapshot = static_payload.get("controller_gain_snapshot")
    if not isinstance(static_input, Mapping) or not isinstance(snapshot, Mapping):
        raise PostprocessInputError("static PID provenance is incomplete")
    _same_path(static_input.get("estimator_result_json", ""), result_path, "estimator result")
    _same_path(static_input.get("bag_json", ""), bag_json_path, "bag JSON")
    _same_path(static_input.get("controller_yaml", ""), controller_yaml_path, "controller YAML")
    _same_path(static_input.get("vehicle_model_json", ""), vehicle_model_path, "vehicle model")
    if controller_yaml.sha256 != str(static_input.get("controller_yaml_sha256", "")):
        raise PostprocessInputError("controller YAML SHA-256 differs from audited artifact")
    template_gains = snapshot.get("controller_yaml_template_gains")
    if not isinstance(template_gains, Mapping):
        raise PostprocessInputError("static PID YAML-template gain provenance is missing")
    for group in PID_GROUPS:
        if group not in template_gains:
            raise PostprocessInputError("static PID YAML-template gains are incomplete")
        actual = controller_yaml.gains[group].as_dict()
        expected = {
            gain: float(template_gains[group][gain]) for gain in PID_GAIN_NAMES
        }
        if actual != expected:
            raise PostprocessInputError("controller YAML values differ from audited static artifact")
    if controller_yaml.reference_values_differ:
        raise PostprocessInputError("ControllerConfig.grape() no longer matches controller YAML structure")
    expected_bag = str(_resolved(Path(bag.bag_path)))
    if str(_resolved(Path(str(static_input.get("bag_path", ""))))) != expected_bag:
        raise PostprocessInputError("static PID bag path differs")
    interval = np.asarray(static_input.get("bag_interval_seconds", ()), dtype=float)
    if interval.shape != (2,) or not np.array_equal(
        interval, np.asarray((bag.start_seconds, bag.end_seconds), dtype=float)
    ):
        raise PostprocessInputError("static PID bag interval differs")
    if str(arguments.get("bag", "")) != bag.bag_path:
        raise PostprocessInputError("estimator arguments bag path differs")
    if float(arguments.get("bag_start", np.nan)) != bag.start_seconds or float(
        arguments.get("bag_end", np.nan)
    ) != bag.end_seconds:
        raise PostprocessInputError("estimator arguments bag interval differs")
    gain_source = str(snapshot.get("source", ""))
    if not gain_source or gain_source != str(static_input.get("controller_gain_source", "")):
        raise PostprocessInputError("recorded controller gain source differs")
    if not gain_source.startswith("rosbag_recorded_"):
        raise PostprocessInputError("controller gains are not a rosbag-recorded snapshot")

    recorded_gains = _nested_recorded_gains(static_payload)
    configuration = apply_pid_gain_configuration(
        ControllerConfig.grape(), recorded_gains
    )
    return CaseInputs(
        result_path=result_path,
        arrays_path=arrays_path,
        static_path=static_path,
        arguments_path=arguments_path,
        bag_json_path=bag_json_path,
        controller_yaml_path=controller_yaml_path,
        vehicle_model_path=vehicle_model_path,
        result=result,
        sampling_coordinates=coordinates,
        static_baseline=baseline,
        static_payload=static_payload,
        arguments=arguments,
        bag=bag,
        controller_yaml=controller_yaml,
        vehicle_model=model,
        controller_configuration=configuration,
        actuator_parameters=_actuator_parameters(arguments),
        recorded_gains=recorded_gains,
    )


def controller_timing_from_bag(bag: Any) -> Mapping[str, Any]:
    """Read representative issued-thrust timing through the audited adapter."""

    key = (str(_resolved(Path(bag.bag_path))), bag.start_seconds, bag.end_seconds)
    if key in _TIMING_CACHE:
        return dict(_TIMING_CACHE[key])
    try:
        from grape_param_estim.real_rosbag import load_flight_data

        flight = load_flight_data(
            path=bag.bag_path,
            start_local=bag.start_seconds,
            end_local=bag.end_seconds,
            include_fc_specific_force=False,
            compute_sha256=False,
            bag_id=bag.source_path.stem,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise PostprocessInputError(
            "ROS bag command timing cannot be reconstructed: {}".format(error)
        ) from error
    stream = getattr(flight, "commanded_thrust", None)
    if stream is None:
        stream = flight.rotor_command
    if stream is None:
        raise PostprocessInputError("selected bag interval has no issued thrust stream")
    times = np.asarray(stream.record_times, dtype=float)
    differences = np.diff(times)
    if differences.size == 0 or np.any(~np.isfinite(differences)) or np.any(differences <= 0.0):
        raise PostprocessInputError("issued thrust record times are not strictly increasing")
    result = {
        "sample_count": int(times.size),
        "interval_count": int(differences.size),
        "median_seconds": float(np.median(differences)),
        "mean_seconds": float(np.mean(differences)),
        "standard_deviation_seconds": float(np.std(differences, ddof=0)),
        "min_seconds": float(np.min(differences)),
        "max_seconds": float(np.max(differences)),
    }
    _TIMING_CACHE[key] = result
    return dict(result)


def decompose_thrust_delay(delay_seconds: float, controller_dt: float) -> DelayDecomposition:
    tau = float(delay_seconds)
    dt = float(controller_dt)
    if not np.isfinite(tau) or tau < 0.0 or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("delay and controller_dt must be finite and non-negative/positive")
    if tau == 0.0:
        return DelayDecomposition(0.0, dt, 0, 0.0, 0)
    ratio = tau / dt
    nearest = int(round(ratio))
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, abs(ratio))
    if abs(ratio - nearest) <= tolerance:
        whole = nearest
        remainder = 0.0
    else:
        whole = int(np.floor(ratio))
        remainder = float(tau - whole * dt)
        if remainder / dt <= tolerance:
            remainder = 0.0
        elif 1.0 - remainder / dt <= tolerance:
            whole += 1
            remainder = 0.0
    depth = whole + (1 if remainder > 0.0 else 0)
    return DelayDecomposition(tau, dt, whole, remainder, depth)


def delayed_thrust_segments(
    delay: DelayDecomposition,
    queue: np.ndarray,
    current_issued_thrust: Sequence[float],
) -> tuple[tuple[float, np.ndarray], ...]:
    """Return exact ZOH segment durations and thrust-only targets."""

    history = np.asarray(queue, dtype=float)
    current = np.asarray(current_issued_thrust, dtype=float)
    if history.shape != (delay.depth, 4) or current.shape != (4,):
        raise ValueError("delay queue/current thrust shape mismatch")
    if delay.depth == 0:
        return ((delay.controller_dt, current.copy()),)
    if delay.remainder_seconds == 0.0:
        return ((delay.controller_dt, history[-delay.whole_steps].copy()),)
    old = history[-(delay.whole_steps + 1)].copy()
    recent = current.copy() if delay.whole_steps == 0 else history[-delay.whole_steps].copy()
    return (
        (delay.remainder_seconds, old),
        (delay.controller_dt - delay.remainder_seconds, recent),
    )


def shift_thrust_queue(queue: np.ndarray, current_issued_thrust: Sequence[float]) -> np.ndarray:
    history = np.asarray(queue, dtype=float)
    current = np.asarray(current_issued_thrust, dtype=float)
    if history.ndim != 2 or history.shape[1] != 4 or current.shape != (4,):
        raise ValueError("delay queue/current thrust shape mismatch")
    if history.shape[0] == 0:
        return history.copy()
    return np.concatenate((history[1:], current[None, :]), axis=0)


def hover_reference() -> ReferenceState:
    zero = np.zeros(3, dtype=float)
    return ReferenceState(zero, zero, zero, zero, zero, zero)


def hover_rigid_body() -> RigidBodyState:
    zero = np.zeros(3, dtype=float)
    return RigidBodyState(zero, np.asarray((0.0, 0.0, 0.0, 1.0)), zero, zero)


def physical_plant_from_scale_free(scale_free: ScaleFreePlant, model: Any) -> VehicleParameters:
    """Decode the common-scale quotient in the explicit nominal-mass gauge."""

    mass = float(model.parameters.mass)
    inertia = mass * np.asarray(scale_free.inertia_over_mass, dtype=float)
    force = mass * np.asarray(scale_free.force_effectiveness_over_mass, dtype=float)
    if np.any(~np.isfinite(inertia)) or np.any(~np.isfinite(force)):
        raise LocalPoleNumericalError("scale-free sample cannot be represented in nominal-mass gauge")
    return VehicleParameters(
        mass=mass,
        inertia=inertia,
        cog_offset=np.asarray(scale_free.cog_position_body, dtype=float),
        force_effectiveness=force,
        torque_effectiveness=model.parameters.torque_effectiveness,
        linear_drag=model.parameters.linear_drag,
        angular_drag=model.parameters.angular_drag,
    )


def scale_free_vector(plant: ScaleFreePlant) -> np.ndarray:
    inertia = np.asarray(plant.inertia_over_mass, dtype=float)
    return np.asarray(
        (
            inertia[0, 0], inertia[1, 1], inertia[2, 2],
            inertia[0, 1], inertia[0, 2], inertia[1, 2],
            *np.asarray(plant.cog_position_body, dtype=float),
            *np.asarray(plant.force_effectiveness_over_mass, dtype=float),
        ),
        dtype=float,
    )


def _segment_command(command: ActuatorCommand, thrust: np.ndarray) -> ActuatorCommand:
    return ActuatorCommand(
        thrust,
        command.gimbal_angle,
        command.virtual_force,
        command.desired_acceleration,
    )


def _propagate_segment(
    rigid: RigidBodyState,
    actuators: ActuatorState,
    command: ActuatorCommand,
    duration: float,
    plant: FullSixDofPlant,
    actuator_parameters: ActuatorParameters,
    start_time: float,
) -> tuple[RigidBodyState, ActuatorState]:
    midpoint = advance_actuators(actuators, command, actuator_parameters, 0.5 * duration)
    next_rigid = plant.step(start_time, rigid, midpoint, duration)
    next_actuators = advance_actuators(midpoint, command, actuator_parameters, 0.5 * duration)
    return next_rigid, next_actuators


def advance_augmented_state(
    state: AugmentedState,
    *,
    controller: GrapeController,
    plant: FullSixDofPlant,
    actuator_parameters: ActuatorParameters,
    reference: ReferenceState,
    controller_dt: float,
    delay: DelayDecomposition,
) -> tuple[AugmentedState, ActuatorCommand]:
    command, next_controller = controller.step(
        state.rigid_body,
        reference,
        state.controller,
        controller_dt,
        state.actuators.gimbal_angle,
    )
    rigid = state.rigid_body
    actuators = state.actuators
    elapsed = 0.0
    for duration, thrust in delayed_thrust_segments(delay, state.thrust_queue, command.thrust):
        rigid, actuators = _propagate_segment(
            rigid,
            actuators,
            _segment_command(command, thrust),
            duration,
            plant,
            actuator_parameters,
            elapsed,
        )
        elapsed += duration
    return (
        AugmentedState(
            rigid,
            next_controller,
            actuators,
            shift_thrust_queue(state.thrust_queue, command.thrust),
        ),
        command,
    )


def _trim_residual(
    candidate: Sequence[float],
    controller: GrapeController,
    plant: FullSixDofPlant,
    actuator_parameters: ActuatorParameters,
    reference: ReferenceState,
    controller_dt: float,
) -> np.ndarray:
    selected = np.asarray(candidate, dtype=float)
    rigid = hover_rigid_body()
    controller_state = ControllerState(selected[:6], True)
    gimbal = selected[6:]
    command, _ = controller.step(
        rigid, reference, controller_state, controller_dt, gimbal
    )
    clipped_thrust = np.clip(
        command.thrust,
        actuator_parameters.minimum_thrust,
        actuator_parameters.maximum_thrust,
    )
    actuator_state = ActuatorState(clipped_thrust, gimbal)
    target = advance_actuators(
        actuator_state, command, actuator_parameters, controller_dt
    )
    derivative = plant.derivative(0.0, rigid.as_vector(), actuator_state)
    result = np.concatenate((derivative[7:10], derivative[10:13], target.gimbal_angle - gimbal))
    if np.any(~np.isfinite(result)):
        raise LocalPoleNumericalError("trim residual became non-finite")
    return result


def _near_actuator_kink(command: ActuatorCommand, state: ActuatorState, parameters: ActuatorParameters) -> bool:
    tolerance = 64.0 * np.sqrt(np.finfo(float).eps)
    checks = (
        np.abs(command.thrust - parameters.minimum_thrust),
        np.abs(command.thrust - parameters.maximum_thrust),
        np.abs(np.abs(command.gimbal_angle) - parameters.maximum_gimbal_angle),
        np.abs(np.abs(state.gimbal_angle) - parameters.maximum_gimbal_angle),
    )
    return any(np.any(value <= tolerance * np.maximum(1.0, np.abs(value))) for value in checks)


def _near_piecewise_kink(
    command: ActuatorCommand,
    state: ActuatorState,
    controller_state: ControllerState,
    configuration: ControllerConfig,
    parameters: ActuatorParameters,
) -> bool:
    if _near_actuator_kink(command, state, parameters):
        return True
    tolerance = 64.0 * np.sqrt(np.finfo(float).eps)

    def near(value: float, boundary: float) -> bool:
        return abs(abs(value) - boundary) <= tolerance * max(
            1.0, abs(value), abs(boundary)
        )

    for axis, pid in enumerate(configuration.pid):
        integral = float(controller_state.integral_error[axis])
        i_term = integral * pid.i_gain
        if near(integral, pid.limit_error_i) or near(i_term, pid.limit_i):
            return True
        if near(i_term, pid.limit_sum):
            return True
        if axis == 2 and abs(integral) <= tolerance:
            return True
    return False


def solve_hover_trim(
    *,
    controller: GrapeController,
    plant: FullSixDofPlant,
    actuator_parameters: ActuatorParameters,
    reference: ReferenceState,
    controller_dt: float,
    delay: DelayDecomposition,
) -> TrimResult:
    initial = np.concatenate(
        (
            initial_controller_state(controller.configuration, trim_hover=True).integral_error,
            np.zeros(4, dtype=float),
        )
    )
    objective: Callable[[np.ndarray], np.ndarray] = lambda value: _trim_residual(
        value, controller, plant, actuator_parameters, reference, controller_dt
    )
    solved = least_squares(
        objective,
        initial,
        method="trf",
        ftol=np.sqrt(np.finfo(float).eps),
        xtol=np.sqrt(np.finfo(float).eps),
        gtol=np.sqrt(np.finfo(float).eps),
        max_nfev=2000,
    )
    candidate = np.asarray(solved.x, dtype=float)
    residual = objective(candidate)
    if np.any(~np.isfinite(candidate)):
        raise LocalPoleNumericalError("trim solver returned a non-finite point")
    rigid = hover_rigid_body()
    controller_state = ControllerState(candidate[:6], True)
    gimbal = candidate[6:]
    command, _ = controller.step(
        rigid, reference, controller_state, controller_dt, gimbal
    )
    clipped_thrust = np.clip(
        command.thrust,
        actuator_parameters.minimum_thrust,
        actuator_parameters.maximum_thrust,
    )
    actuator_state = ActuatorState(clipped_thrust, gimbal)
    queue = np.repeat(command.thrust[None, :], delay.depth, axis=0)
    state = AugmentedState(rigid, controller_state, actuator_state, queue)
    next_state, _ = advance_augmented_state(
        state,
        controller=controller,
        plant=plant,
        actuator_parameters=actuator_parameters,
        reference=reference,
        controller_dt=controller_dt,
        delay=delay,
    )
    temporary = PoleContext(
        controller, plant, actuator_parameters, reference, controller_dt, delay, state
    )
    defect = encode_local_state(next_state, temporary)
    scale = max(
        1.0,
        float(np.max(np.abs(candidate))),
        float(np.max(np.abs(command.thrust))),
    )
    tolerance = float(32.0 * np.finfo(float).eps ** (2.0 / 3.0) * scale)
    defect_norm = float(np.linalg.norm(defect, ord=np.inf))
    integral_defect = next_state.controller.integral_error - controller_state.integral_error
    gimbal_defect = next_state.actuators.gimbal_angle - actuator_state.gimbal_angle
    valid = bool(
        np.all(np.isfinite(defect))
        and np.all(np.isfinite(residual))
        and defect_norm <= tolerance
    )
    return TrimResult(
        state=state,
        issued_command=command,
        root_status=int(solved.status),
        root_message=str(solved.message),
        residual=residual.copy(),
        residual_norm=float(np.linalg.norm(residual)),
        one_step_defect=defect,
        one_step_defect_norm=defect_norm,
        controller_integral_defect=integral_defect,
        gimbal_fixed_point_defect=gimbal_defect,
        equilibrium_tolerance=tolerance,
        equilibrium_valid=valid,
        piecewise_linearization_near_kink=_near_piecewise_kink(
            command,
            actuator_state,
            controller_state,
            controller.configuration,
            actuator_parameters,
        ),
    )


def decode_local_state(delta: Sequence[float], context: PoleContext) -> AugmentedState:
    selected = np.asarray(delta, dtype=float)
    if selected.shape != (context.local_dimension,) or np.any(~np.isfinite(selected)):
        raise ValueError("local state must have the context dimension and be finite")
    trim = context.trim
    rotation = quaternion_to_matrix(trim.rigid_body.orientation_xyzw) @ so3_exp(selected[3:6])
    rigid = RigidBodyState(
        trim.rigid_body.position + selected[0:3],
        matrix_to_quaternion(rotation),
        trim.rigid_body.linear_velocity + selected[6:9],
        trim.rigid_body.angular_velocity + selected[9:12],
    )
    controller_state = ControllerState(
        trim.controller.integral_error + selected[12:18],
        trim.controller.roll_pitch_integration_active,
    )
    actuators = ActuatorState(
        trim.actuators.thrust + selected[18:22],
        trim.actuators.gimbal_angle + selected[22:26],
    )
    queue = trim.thrust_queue + selected[26:].reshape(context.delay.depth, 4)
    return AugmentedState(rigid, controller_state, actuators, queue)


def encode_local_state(state: AugmentedState, context: PoleContext) -> np.ndarray:
    trim = context.trim
    rotation = quaternion_to_matrix(state.rigid_body.orientation_xyzw)
    trim_rotation = quaternion_to_matrix(trim.rigid_body.orientation_xyzw)
    return np.concatenate(
        (
            state.rigid_body.position - trim.rigid_body.position,
            so3_log(trim_rotation.T @ rotation),
            state.rigid_body.linear_velocity - trim.rigid_body.linear_velocity,
            state.rigid_body.angular_velocity - trim.rigid_body.angular_velocity,
            state.controller.integral_error - trim.controller.integral_error,
            state.actuators.thrust - trim.actuators.thrust,
            state.actuators.gimbal_angle - trim.actuators.gimbal_angle,
            (state.thrust_queue - trim.thrust_queue).reshape(-1),
        )
    )


def local_closed_loop_step(delta: Sequence[float], context: PoleContext) -> np.ndarray:
    state = decode_local_state(delta, context)
    next_state, _ = advance_augmented_state(
        state,
        controller=context.controller,
        plant=context.plant,
        actuator_parameters=context.actuator_parameters,
        reference=context.reference,
        controller_dt=context.controller_dt,
        delay=context.delay,
    )
    result = encode_local_state(next_state, context)
    if np.any(~np.isfinite(result)):
        raise LocalPoleNumericalError("local closed-loop map became non-finite")
    return result


def finite_difference_steps(context: PoleContext, divisor: float = 1.0) -> np.ndarray:
    trim = context.trim
    scale = np.concatenate(
        (
            np.maximum(1.0, np.abs(trim.rigid_body.position)),
            np.ones(3),
            np.maximum(1.0, np.abs(trim.rigid_body.linear_velocity)),
            np.maximum(1.0, np.abs(trim.rigid_body.angular_velocity)),
            np.maximum(1.0, np.abs(trim.controller.integral_error)),
            np.maximum(1.0, np.abs(trim.actuators.thrust)),
            np.maximum(1.0, np.abs(trim.actuators.gimbal_angle)),
            np.maximum(1.0, np.abs(trim.thrust_queue.reshape(-1))),
        )
    )
    return np.finfo(float).eps ** (1.0 / 3.0) * scale / float(divisor)


def central_difference_jacobian(context: PoleContext, divisor: float = 1.0) -> np.ndarray:
    dimension = context.local_dimension
    steps = finite_difference_steps(context, divisor)
    jacobian = np.empty((dimension, dimension), dtype=float)
    for index, step in enumerate(steps):
        delta = np.zeros(dimension, dtype=float)
        delta[index] = step
        jacobian[:, index] = (
            local_closed_loop_step(delta, context)
            - local_closed_loop_step(-delta, context)
        ) / (2.0 * step)
    if np.any(~np.isfinite(jacobian)):
        raise LocalPoleNumericalError("finite-difference Jacobian became non-finite")
    return jacobian


def classify_eigenvalues(eigenvalues: Sequence[complex]) -> Mapping[str, Any]:
    values = np.asarray(eigenvalues, dtype=complex)
    magnitudes = np.abs(values)
    radius = float(np.max(magnitudes))
    return {
        "spectral_radius": radius,
        "spectral_margin": 1.0 - radius,
        "stable": bool(np.all(magnitudes < 1.0)),
        "unstable_pole_count": int(np.count_nonzero(magnitudes > 1.0)),
        "marginal_pole_count": int(np.count_nonzero(magnitudes == 1.0)),
    }


def finite_difference_diagnostic(primary: np.ndarray, half: np.ndarray) -> Mapping[str, float]:
    difference = np.asarray(primary) - np.asarray(half)
    denominator = max(float(np.linalg.norm(half, ord="fro")), np.finfo(float).tiny)
    primary_radius = classify_eigenvalues(np.linalg.eigvals(primary))["spectral_radius"]
    half_radius = classify_eigenvalues(np.linalg.eigvals(half))["spectral_radius"]
    return {
        "relative_frobenius_difference": float(np.linalg.norm(difference, ord="fro") / denominator),
        "maximum_absolute_entry_difference": float(np.max(np.abs(difference))),
        "spectral_radius_difference": float(abs(primary_radius - half_radius)),
    }


def draw_quotient_samples(
    covariance: np.ndarray,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Draw one reproducible ordered realization from a PSD covariance."""

    if int(sample_count) <= 0:
        raise ValueError("sample_count must be positive")
    eigenvalues, eigenvectors, tolerance = _psd_eigendecomposition(covariance)
    standard = np.random.default_rng(int(seed)).standard_normal(
        (int(sample_count), covariance.shape[0])
    )
    transform = eigenvectors * np.sqrt(eigenvalues)[None, :]
    return standard @ transform.T, eigenvalues, eigenvectors, tolerance


def _analyze_plant(
    *,
    scale_free: ScaleFreePlant,
    inputs: CaseInputs,
    controller_dt: float,
    delay: DelayDecomposition,
    fd_check: bool,
) -> Mapping[str, Any]:
    parameters = physical_plant_from_scale_free(scale_free, inputs.vehicle_model)
    controller = GrapeController(
        inputs.controller_configuration,
        inputs.vehicle_model.parameters,
        build_controller_snapshot_geometry(inputs.vehicle_model),
    )
    plant = FullSixDofPlant(parameters, inputs.vehicle_model.body_geometry)
    reference = hover_reference()
    trim = solve_hover_trim(
        controller=controller,
        plant=plant,
        actuator_parameters=inputs.actuator_parameters,
        reference=reference,
        controller_dt=controller_dt,
        delay=delay,
    )
    context = PoleContext(
        controller,
        plant,
        inputs.actuator_parameters,
        reference,
        controller_dt,
        delay,
        trim.state,
    )
    result: dict[str, Any] = {
        "trim": trim,
        "context": context,
        "jacobian": None,
        "eigenvalues": None,
        "classification": None,
        "finite_difference_diagnostic": None,
    }
    if not trim.equilibrium_valid:
        return result
    jacobian = central_difference_jacobian(context)
    eigenvalues = np.linalg.eigvals(jacobian)
    if np.any(~np.isfinite(eigenvalues)):
        raise LocalPoleNumericalError("eigenvalue calculation became non-finite")
    result["jacobian"] = jacobian
    result["eigenvalues"] = eigenvalues
    result["classification"] = classify_eigenvalues(eigenvalues)
    if fd_check:
        half = central_difference_jacobian(context, divisor=2.0)
        result["finite_difference_diagnostic"] = finite_difference_diagnostic(jacobian, half)
    return result


def _trim_json(trim: TrimResult) -> Mapping[str, Any]:
    return {
        "trim_root_status": trim.root_status,
        "trim_root_message": trim.root_message,
        "trim_residual_vector": trim.residual.tolist(),
        "trim_residual_norm": trim.residual_norm,
        "full_one_step_trim_defect": trim.one_step_defect.tolist(),
        "full_one_step_trim_defect_norm": trim.one_step_defect_norm,
        "controller_integral_defect": trim.controller_integral_defect.tolist(),
        "gimbal_fixed_point_defect": trim.gimbal_fixed_point_defect.tolist(),
        "equilibrium_tolerance": trim.equilibrium_tolerance,
        "equilibrium_valid": trim.equilibrium_valid,
        "piecewise_linearization_near_kink": trim.piecewise_linearization_near_kink,
        "integral_error": trim.state.controller.integral_error.tolist(),
        "steady_gimbal_angle": trim.state.actuators.gimbal_angle.tolist(),
        "issued_thrust": trim.issued_command.thrust.tolist(),
        "issued_gimbal_angle": trim.issued_command.gimbal_angle.tolist(),
        "actual_thrust": trim.state.actuators.thrust.tolist(),
        "actual_gimbal_angle": trim.state.actuators.gimbal_angle.tolist(),
    }


def _eigenvalue_json(values: Optional[np.ndarray]) -> Optional[list[Mapping[str, float]]]:
    if values is None:
        return None
    return [
        {"real": float(value.real), "imag": float(value.imag), "magnitude": float(abs(value))}
        for value in values
    ]


def _metric_summary(values: np.ndarray) -> Optional[Mapping[str, float]]:
    selected = np.asarray(values, dtype=float)
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return None
    quantiles = np.quantile(selected, QUANTILE_LEVELS)
    result: dict[str, float] = {
        "mean": float(np.mean(selected)),
        "standard_deviation": float(np.std(selected, ddof=0)),
        "min": float(np.min(selected)),
        "max": float(np.max(selected)),
    }
    result.update({name: float(value) for name, value in zip(QUANTILE_NAMES, quantiles)})
    return result


def stability_summary(
    spectral_radius: np.ndarray,
    spectral_margin: np.ndarray,
    stable: np.ndarray,
    unstable_count: np.ndarray,
    pole_valid: np.ndarray,
    requested_count: int,
) -> Mapping[str, Any]:
    mask = np.asarray(pole_valid, dtype=bool)
    count = int(np.count_nonzero(mask))
    histogram: dict[str, int] = {}
    if count:
        unique, counts = np.unique(unstable_count[mask], return_counts=True)
        histogram = {str(int(key)): int(value) for key, value in zip(unique, counts)}
    return {
        "spectral_radius": _metric_summary(spectral_radius[mask]),
        "spectral_margin": _metric_summary(spectral_margin[mask]),
        "stable_fraction_among_pole_valid": float(np.mean(stable[mask])) if count else None,
        "unstable_fraction_among_pole_valid": float(np.mean(~stable[mask])) if count else None,
        "pole_valid_fraction_of_requested": float(count / requested_count),
        "pole_valid_samples": count,
        "requested_samples": int(requested_count),
        "unstable_pole_count_histogram": histogram,
        "unstable_pole_count_median": float(np.median(unstable_count[mask])) if count else None,
    }


def _plain_mode(mode: Any) -> Mapping[str, Any]:
    return {key: value for key, value in asdict(mode).items()}


def analyze_case(
    *,
    result_path: Path,
    arrays_path: Path,
    static_postprocess_path: Path,
    arguments_path: Path,
    bag_json_path: Path,
    controller_yaml_path: Path,
    vehicle_model_path: Path,
    covariance_mode: str,
    sample_count: int,
    seed: int,
    delay_mode: str,
    controller_dt_override: Optional[float] = None,
    fd_check_samples: int = DEFAULT_FD_CHECK_SAMPLES,
    flight_outcome: str = "unspecified",
) -> tuple[Mapping[str, Any], Mapping[str, np.ndarray], Mapping[str, Any]]:
    if covariance_mode not in COVARIANCE_MODES:
        raise PostprocessInputError("unsupported covariance mode")
    if delay_mode not in DELAY_MODES:
        raise PostprocessInputError("unsupported delay mode")
    if sample_count <= 0 or fd_check_samples < 0:
        raise PostprocessInputError("sample counts must be positive/non-negative")
    inputs = load_case_inputs(
        result_path=result_path,
        arrays_path=arrays_path,
        static_postprocess_path=static_postprocess_path,
        arguments_path=arguments_path,
        bag_json_path=bag_json_path,
        controller_yaml_path=controller_yaml_path,
        vehicle_model_path=vehicle_model_path,
        covariance_mode=covariance_mode,
    )
    timing = controller_timing_from_bag(inputs.bag)
    controller_dt = timing["median_seconds"] if controller_dt_override is None else float(controller_dt_override)
    if not np.isfinite(controller_dt) or controller_dt <= 0.0:
        raise PostprocessInputError("controller_dt override must be finite and positive")
    fitted_delay = float(inputs.result.plant.rotor_lag_seconds)
    selected_delay = fitted_delay if delay_mode == "fitted_thrust_delay" else 0.0
    delay = decompose_thrust_delay(selected_delay, controller_dt)

    coordinates = inputs.sampling_coordinates
    (
        quotient_samples,
        eigen_covariance,
        _covariance_eigenvectors,
        psd_tolerance,
    ) = draw_quotient_samples(
        coordinates.covariance,
        sample_count,
        seed,
    )
    dimension = 26 + 4 * delay.depth

    scale_free_samples = np.full((sample_count, 13), np.nan)
    trim_integral = np.full((sample_count, 6), np.nan)
    trim_gimbal = np.full((sample_count, 4), np.nan)
    trim_issued_thrust = np.full((sample_count, 4), np.nan)
    trim_issued_gimbal = np.full((sample_count, 4), np.nan)
    trim_actual_thrust = np.full((sample_count, 4), np.nan)
    trim_actual_gimbal = np.full((sample_count, 4), np.nan)
    trim_residual_norm = np.full(sample_count, np.nan)
    trim_one_step_defect = np.full(sample_count, np.nan)
    equilibrium_valid = np.zeros(sample_count, dtype=bool)
    eigenvalue_real = np.full((sample_count, dimension), np.nan)
    eigenvalue_imag = np.full((sample_count, dimension), np.nan)
    eigenvalue_magnitude = np.full((sample_count, dimension), np.nan)
    spectral_radius = np.full(sample_count, np.nan)
    spectral_margin = np.full(sample_count, np.nan)
    stable = np.zeros(sample_count, dtype=bool)
    unstable_pole_count = np.full(sample_count, -1, dtype=int)
    marginal_pole_count = np.full(sample_count, -1, dtype=int)
    numerical_valid = np.zeros(sample_count, dtype=bool)
    piecewise_linearization_near_kink = np.zeros(sample_count, dtype=bool)
    failure_stage = np.full(sample_count, "", dtype="U64")
    failure_type = np.full(sample_count, "", dtype="U96")
    failure_message = np.full(sample_count, "", dtype="U512")
    fd_diagnostics: list[Mapping[str, Any]] = []

    center = _analyze_plant(
        scale_free=coordinates.center_plant,
        inputs=inputs,
        controller_dt=controller_dt,
        delay=delay,
        fd_check=True,
    )
    if center["finite_difference_diagnostic"] is not None:
        fd_diagnostics.append({"sample": "center", **center["finite_difference_diagnostic"]})

    for index, quotient_delta in enumerate(quotient_samples):
        stage = "decode_scale_free_sample"
        try:
            scale_free = coordinates.decode(quotient_delta)
            scale_free_samples[index] = scale_free_vector(scale_free)
            stage = "trim_and_local_poles"
            evaluated = _analyze_plant(
                scale_free=scale_free,
                inputs=inputs,
                controller_dt=controller_dt,
                delay=delay,
                fd_check=index < fd_check_samples,
            )
            trim = evaluated["trim"]
            trim_integral[index] = trim.state.controller.integral_error
            trim_gimbal[index] = trim.state.actuators.gimbal_angle
            trim_issued_thrust[index] = trim.issued_command.thrust
            trim_issued_gimbal[index] = trim.issued_command.gimbal_angle
            trim_actual_thrust[index] = trim.state.actuators.thrust
            trim_actual_gimbal[index] = trim.state.actuators.gimbal_angle
            trim_residual_norm[index] = trim.residual_norm
            trim_one_step_defect[index] = trim.one_step_defect_norm
            equilibrium_valid[index] = trim.equilibrium_valid
            piecewise_linearization_near_kink[index] = (
                trim.piecewise_linearization_near_kink
            )
            numerical_valid[index] = True
            if evaluated["finite_difference_diagnostic"] is not None:
                fd_diagnostics.append({"sample": int(index), **evaluated["finite_difference_diagnostic"]})
            classification = evaluated["classification"]
            values = evaluated["eigenvalues"]
            if classification is not None and values is not None:
                eigenvalue_real[index] = values.real
                eigenvalue_imag[index] = values.imag
                eigenvalue_magnitude[index] = np.abs(values)
                spectral_radius[index] = classification["spectral_radius"]
                spectral_margin[index] = classification["spectral_margin"]
                stable[index] = classification["stable"]
                unstable_pole_count[index] = classification["unstable_pole_count"]
                marginal_pole_count[index] = classification["marginal_pole_count"]
        except NUMERICAL_SAMPLE_EXCEPTIONS as error:
            failure_stage[index] = stage
            failure_type[index] = type(error).__name__
            failure_message[index] = str(error)

    pole_valid = equilibrium_valid & numerical_valid & np.isfinite(spectral_radius)
    distribution = stability_summary(
        spectral_radius,
        spectral_margin,
        stable,
        unstable_pole_count,
        pole_valid,
        sample_count,
    )
    prefixes: dict[str, Any] = {}
    for prefix in PREFIX_COUNTS:
        if prefix <= sample_count:
            prefixes[str(prefix)] = stability_summary(
                spectral_radius[:prefix],
                spectral_margin[:prefix],
                stable[:prefix],
                unstable_pole_count[:prefix],
                pole_valid[:prefix],
                prefix,
            )

    snapshot = inputs.static_payload["controller_gain_snapshot"]
    center_classification = center["classification"]
    center_trim: TrimResult = center["trim"]
    warnings: list[str] = []
    if timing["standard_deviation_seconds"] > 0.0:
        warnings.append("constant_controller_dt_approximates_recorded_timing_jitter")
    if center_trim.piecewise_linearization_near_kink:
        warnings.append("center_piecewise_linearization_near_kink")
    if not center_trim.equilibrium_valid:
        warnings.append("center_hover_equilibrium_unresolved")
    if np.count_nonzero(~numerical_valid):
        warnings.append("one_or_more_numerical_samples_invalid")
    if np.count_nonzero(numerical_valid & ~equilibrium_valid):
        warnings.append("one_or_more_finite_trims_have_material_fixed_point_defect")
    if np.count_nonzero(piecewise_linearization_near_kink):
        warnings.append("one_or_more_samples_piecewise_linearization_near_kink")

    report: Mapping[str, Any] = {
        "schema": LOCAL_POLE_SCHEMA,
        "method": {
            "state_coordinate": "26 + 4 * thrust_delay_depth, right_orientation_tangent",
            "equilibrium": "unbounded_10d_nonlinear_least_squares",
            "jacobian": "full_central_finite_difference_of_augmented_forward_map",
            "finite_difference_base_step": "machine_epsilon**(1/3) times local state scale",
            "stability_criterion": "all(abs(discrete_pole) < 1) exactly",
            "coordinate_mode": "estimator_quotient",
        },
        "source_commit": source_commit(),
        "plan_base_commit": PLAN_BASE_COMMIT,
        "input": {
            "result": str(inputs.result_path),
            "arrays": str(inputs.arrays_path),
            "static_postprocess": str(inputs.static_path),
            "arguments_json": str(inputs.arguments_path),
            "bag_json": str(inputs.bag_json_path),
            "bag_path": inputs.bag.bag_path,
            "bag_interval_seconds": [inputs.bag.start_seconds, inputs.bag.end_seconds],
            "controller_yaml": str(inputs.controller_yaml_path),
            "controller_yaml_sha256": inputs.controller_yaml.sha256,
            "vehicle_model": str(inputs.vehicle_model_path),
            "estimator_source_commit": inputs.result.source_commit,
            "estimator_case_name": inputs.result.case_name,
            "static_postprocess_source_commit": inputs.static_baseline.source_commit,
        },
        "flight_outcome": str(flight_outcome),
        "controller": {
            "gain_source": snapshot["source"],
            "gains": inputs.recorded_gains.as_nested_mapping(),
            "record_times": list(snapshot.get("record_times", ())),
            "source_kinds": list(snapshot.get("source_kinds", ())),
            "pid_control_flags": list(snapshot.get("pid_control_flags", ())),
            "mode": _plain_mode(inputs.controller_yaml.mode),
            "nominal_allocation_uses_sampled_plant": False,
        },
        "plant_distribution": {
            "physical_gauge": "nominal_mass_gauge",
            "mass_kg": float(inputs.vehicle_model.parameters.mass),
            "coordinate_labels": list(inputs.sampling_coordinates.coordinate_labels),
            "scale_free_labels": list(SCALE_FREE_LABELS),
            "covariance_mode": covariance_mode,
            "covariance_source": inputs.sampling_coordinates.covariance_source,
            "rotor_lag_is_sampled": False,
            "torque_effectiveness_source": "nominal_vehicle_model",
            "linear_drag_source": "nominal_vehicle_model",
            "angular_drag_source": "nominal_vehicle_model",
        },
        "delay_model": {
            "mode": delay_mode,
            "kind": "exact_thrust_only_zero_order_hold_queue",
            "fitted_rotor_lag_seconds": fitted_delay,
            "selected_delay_seconds": selected_delay,
            "whole_steps": delay.whole_steps,
            "remainder_seconds": delay.remainder_seconds,
            "delay_depth": delay.depth,
            "queue_state_dimension": 4 * delay.depth,
            "gimbal_delay_seconds": 0.0,
            "generic_actuator_delay_seconds": inputs.actuator_parameters.delay,
        },
        "controller_timing": {
            **timing,
            "selected_controller_dt_seconds": controller_dt,
            "selection": "median" if controller_dt_override is None else "explicit_override",
        },
        "center_result": {
            **_trim_json(center_trim),
            "local_dimension": dimension,
            "spectral_radius": None if center_classification is None else center_classification["spectral_radius"],
            "spectral_margin": None if center_classification is None else center_classification["spectral_margin"],
            "stable": None if center_classification is None else center_classification["stable"],
            "unstable_pole_count": None if center_classification is None else center_classification["unstable_pole_count"],
            "marginal_pole_count": None if center_classification is None else center_classification["marginal_pole_count"],
            "eigenvalues": _eigenvalue_json(center["eigenvalues"]),
        },
        "sampling": {
            "requested_count": int(sample_count),
            "seed": int(seed),
            "covariance_eigenvalues": eigen_covariance.tolist(),
            "covariance_psd_tolerance": psd_tolerance,
            "numerical_valid_count": int(np.count_nonzero(numerical_valid)),
            "equilibrium_valid_count": int(np.count_nonzero(equilibrium_valid)),
            "pole_valid_count": int(np.count_nonzero(pole_valid)),
            "piecewise_linearization_near_kink_count": int(
                np.count_nonzero(piecewise_linearization_near_kink)
            ),
            "prefix_summaries": prefixes,
            "failures": [
                {
                    "sample_index": int(index),
                    "stage": str(failure_stage[index]),
                    "exception_type": str(failure_type[index]),
                    "message": str(failure_message[index]),
                }
                for index in np.flatnonzero(~numerical_valid)
            ],
        },
        "stability_distribution": distribution,
        "finite_difference_diagnostics": {
            "requested_monte_carlo_samples": int(fd_check_samples),
            "checks": fd_diagnostics,
            "used_as_sample_rejection": False,
        },
        "warnings": warnings,
    }
    arrays: Mapping[str, np.ndarray] = {
        "quotient_delta_samples": quotient_samples,
        "scale_free_samples": scale_free_samples,
        "trim_integral": trim_integral,
        "trim_gimbal": trim_gimbal,
        "trim_issued_thrust": trim_issued_thrust,
        "trim_issued_gimbal": trim_issued_gimbal,
        "trim_actual_thrust": trim_actual_thrust,
        "trim_actual_gimbal": trim_actual_gimbal,
        "trim_residual_norm": trim_residual_norm,
        "trim_one_step_defect": trim_one_step_defect,
        "equilibrium_valid": equilibrium_valid,
        "eigenvalue_real": eigenvalue_real,
        "eigenvalue_imag": eigenvalue_imag,
        "eigenvalue_magnitude": eigenvalue_magnitude,
        "spectral_radius": spectral_radius,
        "spectral_margin": spectral_margin,
        "stable": stable,
        "unstable_pole_count": unstable_pole_count,
        "marginal_pole_count": marginal_pole_count,
        "numerical_valid": numerical_valid,
        "piecewise_linearization_near_kink": piecewise_linearization_near_kink,
        "failure_stage": failure_stage,
        "failure_type": failure_type,
        "failure_message": failure_message,
    }
    status = {
        "schema": STATUS_SCHEMA,
        "requested_samples": int(sample_count),
        "numerical_valid_samples": int(np.count_nonzero(numerical_valid)),
        "equilibrium_valid_samples": int(np.count_nonzero(equilibrium_valid)),
        "pole_valid_samples": int(np.count_nonzero(pole_valid)),
        "numerical_failure_count": int(np.count_nonzero(~numerical_valid)),
        "trim_unresolved_count": int(np.count_nonzero(numerical_valid & ~equilibrium_valid)),
        "warnings": warnings,
        "status": "completed",
    }
    return report, arrays, status


def render_markdown(report: Mapping[str, Any]) -> str:
    center = report["center_result"]
    distribution = report["stability_distribution"]
    radius = distribution["spectral_radius"]
    lines = [
        "# Gimbalrotor local sampled-data pole validation",
        "",
        "- Flight outcome: `{}`".format(report["flight_outcome"]),
        "- Covariance: `{}`".format(report["plant_distribution"]["covariance_mode"]),
        "- Delay: `{}` ({:.9g} s; thrust only)".format(
            report["delay_model"]["mode"], report["delay_model"]["selected_delay_seconds"]
        ),
        "- Controller dt: {:.9g} s".format(report["controller_timing"]["selected_controller_dt_seconds"]),
        "- Recorded roll/pitch PID: P={:.9g}, I={:.9g}, D={:.9g}".format(
            *[report["controller"]["gains"]["roll_pitch"][name] for name in PID_GAIN_NAMES]
        ),
        "",
        "## Center plant",
        "",
        "- Equilibrium valid: `{}`".format(center["equilibrium_valid"]),
        "- One-step trim defect: {:.9g}".format(center["full_one_step_trim_defect_norm"]),
        "- Spectral radius: `{}`".format(center["spectral_radius"]),
        "- Stable: `{}`".format(center["stable"]),
        "",
        "## Monte Carlo distribution",
        "",
        "- Pole-valid samples: {}/{}".format(
            distribution["pole_valid_samples"], distribution["requested_samples"]
        ),
        "- Stable fraction among pole-valid: `{}`".format(distribution["stable_fraction_among_pole_valid"]),
    ]
    if radius is not None:
        lines.extend(
            (
                "- Spectral-radius median: {:.9g}".format(radius["q50"]),
                "- Spectral-radius 16–84%: [{:.9g}, {:.9g}]".format(radius["q16"], radius["q84"]),
                "- Spectral-radius 2.5–97.5%: [{:.9g}, {:.9g}]".format(radius["q025"], radius["q975"]),
            )
        )
    if report["warnings"]:
        lines.extend(("", "## Warnings", ""))
        lines.extend("- `{}`".format(value) for value in report["warnings"])
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    report: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    status: Mapping[str, Any],
) -> None:
    output = _resolved(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "local_pole_validation.json", report)
    (output / "local_pole_validation.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    np.savez_compressed(output / "local_pole_samples.npz", **arrays)
    write_json(output / "status.json", status)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--static-postprocess", type=Path, required=True)
    parser.add_argument("--arguments-json", type=Path, required=True)
    parser.add_argument("--bag-json", type=Path, required=True)
    parser.add_argument("--controller-yaml", type=Path, required=True)
    parser.add_argument("--vehicle-model", type=Path, required=True)
    parser.add_argument("--covariance-mode", choices=COVARIANCE_MODES, required=True)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--delay-mode", choices=DELAY_MODES, required=True)
    parser.add_argument("--controller-dt", type=float)
    parser.add_argument("--fd-check-samples", type=int, default=DEFAULT_FD_CHECK_SAMPLES)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    report, arrays, status = analyze_case(
        result_path=arguments.result,
        arrays_path=arguments.arrays,
        static_postprocess_path=arguments.static_postprocess,
        arguments_path=arguments.arguments_json,
        bag_json_path=arguments.bag_json,
        controller_yaml_path=arguments.controller_yaml,
        vehicle_model_path=arguments.vehicle_model,
        covariance_mode=arguments.covariance_mode,
        sample_count=arguments.samples,
        seed=arguments.seed,
        delay_mode=arguments.delay_mode,
        controller_dt_override=arguments.controller_dt,
        fd_check_samples=arguments.fd_check_samples,
    )
    write_outputs(arguments.output_dir, report, arrays, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
