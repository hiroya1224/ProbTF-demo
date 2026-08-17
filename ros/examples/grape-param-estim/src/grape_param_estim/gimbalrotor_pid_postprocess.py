"""Static, scale-free Gimbalrotor PID gain postprocessing.

The physical estimator and the controller retain independent models.  This
module compares the acceleration allocation of the nominal controller with a
scale-free identified plant at zero gimbal and proposes four multiplicative
PID gain-group corrections.  It never deploys gains or edits the source YAML.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from grape_param_estim.controller import (
    ControllerConfig,
    acceleration_allocation_matrix,
)
from grape_param_estim.controller_config import PID_GAIN_NAMES, PID_GROUPS
from grape_param_estim.system import GrapeGeometry, VehicleParameters


POSTPROCESS_SCHEMA = "grape-param-estim/gimbalrotor-pid-postprocess/v1"
METHOD = "scale_free_static_effectiveness_inverse"
SOURCE_SVD_THRESHOLD = 1.0e-4
GAIN_GROUP_AXES = {
    "xy": (0, 1),
    "z": (2,),
    "roll_pitch": (3, 4),
    "yaw": (5,),
}
DEFAULT_LARGE_SCALE_MIN = 0.5
DEFAULT_LARGE_SCALE_MAX = 2.0
DEFAULT_STRONG_COUPLING_THRESHOLD = 0.20
DEFAULT_ALLOCATION_CONDITION_WARNING = 1.0e8


class PostprocessInputError(ValueError):
    """An input artifact or controller branch violates the v1 contract."""


class PostprocessNumericalError(ValueError):
    """The requested static allocation correction is numerically invalid."""


def _readonly(value: Any, shape: Tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or np.any(~np.isfinite(result)):
        raise PostprocessInputError(
            "{} must be a finite {} array".format(name, shape)
        )
    copied = result.copy()
    copied.setflags(write=False)
    return copied


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PostprocessInputError("{} must be an object".format(name))
    return value


def _required(value: Mapping[str, Any], key: str, location: str) -> Any:
    if key not in value:
        raise PostprocessInputError(
            "{} is missing required key {!r}".format(location, key)
        )
    return value[key]


def _finite_scalar(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise PostprocessInputError("{} must be numeric".format(name))
    try:
        selected = float(value)
    except (TypeError, ValueError) as error:
        raise PostprocessInputError("{} must be numeric".format(name)) from error
    if not np.isfinite(selected) or (positive and selected <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise PostprocessInputError("{} must be {}".format(name, qualifier))
    return selected


def _read_json(path: Path, label: str) -> Tuple[Path, Mapping[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PostprocessInputError(
            "{} cannot be read: {}".format(label, source)
        ) from error
    return source, _mapping(payload, label)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PostprocessInputError(
            "cannot hash controller YAML: {}".format(path)
        ) from error
    return digest.hexdigest()


def _rotate_z(vector: np.ndarray, angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.asarray(
        (
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
            vector[2],
        ),
        dtype=float,
    )


def _deduplicate(values: Sequence[str]) -> Tuple[str, ...]:
    result = []
    seen = set()
    for value in values:
        selected = str(value)
        if selected not in seen:
            seen.add(selected)
            result.append(selected)
    return tuple(result)


@dataclass(frozen=True)
class ScaleFreePlant:
    inertia_over_mass: np.ndarray
    cog_position_body: np.ndarray
    force_effectiveness_over_mass: np.ndarray
    rotor_lag_seconds: float

    def __post_init__(self) -> None:
        inertia = _readonly(
            self.inertia_over_mass, (3, 3), "inertia_over_mass"
        )
        if not np.allclose(inertia, inertia.T, rtol=0.0, atol=1.0e-12):
            raise PostprocessInputError(
                "inertia_over_mass must be symmetric"
            )
        if np.any(np.linalg.eigvalsh(inertia) <= 0.0):
            raise PostprocessInputError(
                "inertia_over_mass must be positive definite"
            )
        force = _readonly(
            self.force_effectiveness_over_mass,
            (4,),
            "force_effectiveness_over_mass",
        )
        if np.any(force <= 0.0):
            raise PostprocessInputError(
                "force_effectiveness_over_mass must be positive"
            )
        lag = _finite_scalar(self.rotor_lag_seconds, "rotor_lag_seconds")
        if lag < 0.0:
            raise PostprocessInputError(
                "rotor_lag_seconds must be non-negative"
            )
        object.__setattr__(self, "inertia_over_mass", inertia)
        object.__setattr__(
            self,
            "cog_position_body",
            _readonly(self.cog_position_body, (3,), "cog_position_body"),
        )
        object.__setattr__(self, "force_effectiveness_over_mass", force)
        object.__setattr__(self, "rotor_lag_seconds", lag)


@dataclass(frozen=True)
class EstimatorResult:
    source_path: Path
    source_commit: str
    case_name: str
    overall_case_status: str
    optimization_status: str
    prior: Mapping[str, Any]
    plant: ScaleFreePlant
    warnings: Tuple[str, ...]


@dataclass(frozen=True)
class BagProvenance:
    source_path: Path
    bag_path: str
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        source = Path(self.source_path).expanduser().resolve()
        bag_path = str(self.bag_path)
        start = _finite_scalar(self.start_seconds, "bag start_seconds")
        end = _finite_scalar(self.end_seconds, "bag end_seconds")
        if not bag_path or not Path(bag_path).is_absolute():
            raise PostprocessInputError("bag_path must be an absolute path")
        if end <= start:
            raise PostprocessInputError(
                "bag end_seconds must exceed start_seconds"
            )
        object.__setattr__(self, "source_path", source)
        object.__setattr__(self, "bag_path", bag_path)
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)


@dataclass(frozen=True)
class VehicleModel:
    source_path: Path
    parameters: VehicleParameters
    body_geometry: GrapeGeometry


@dataclass(frozen=True)
class ControllerGainGroup:
    p_gain: float
    i_gain: float
    d_gain: float

    def __post_init__(self) -> None:
        values = np.asarray(
            (self.p_gain, self.i_gain, self.d_gain), dtype=float
        )
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise PostprocessInputError(
                "controller PID gains must be finite and non-negative"
            )
        object.__setattr__(self, "p_gain", float(values[0]))
        object.__setattr__(self, "i_gain", float(values[1]))
        object.__setattr__(self, "d_gain", float(values[2]))

    def as_dict(self) -> Mapping[str, float]:
        return {
            "p_gain": self.p_gain,
            "i_gain": self.i_gain,
            "d_gain": self.d_gain,
        }


@dataclass(frozen=True)
class ControllerMode:
    gimbal_dof: int
    gimbal_dof_source: str
    underactuate: bool
    underactuate_source: str
    gimbal_calc_in_fc: bool
    gimbal_calc_in_fc_source: str
    hovering_approximate: bool
    hovering_approximate_source: str
    yaw_need_d_control: bool


@dataclass(frozen=True)
class ControllerYaml:
    source_path: Path
    sha256: str
    document: Mapping[str, Any]
    gains: Mapping[str, ControllerGainGroup]
    mode: ControllerMode
    reference_values_differ: bool


@dataclass(frozen=True)
class RecordedControllerGains:
    """One fixed PID gain snapshot reconstructed from a selected ROS bag."""

    bag_path: str
    adapter_revision: str
    gains: Mapping[str, ControllerGainGroup]
    record_times: np.ndarray
    pid_control_flags: np.ndarray
    source_kinds: Tuple[str, ...]

    def __post_init__(self) -> None:
        bag_path = str(self.bag_path)
        adapter_revision = str(self.adapter_revision)
        groups = tuple(self.gains)
        times = np.asarray(self.record_times, dtype=float)
        flags = np.asarray(self.pid_control_flags, dtype=bool)
        source_kinds = tuple(str(value) for value in self.source_kinds)
        if (
            not bag_path
            or not Path(bag_path).is_absolute()
            or not adapter_revision
            or groups != tuple(PID_GROUPS)
            or any(
                not isinstance(self.gains[group], ControllerGainGroup)
                for group in PID_GROUPS
            )
            or times.shape != (len(PID_GROUPS),)
            or np.any(~np.isfinite(times))
            or flags.shape != (len(PID_GROUPS),)
            or len(source_kinds) != len(PID_GROUPS)
            or any(not value for value in source_kinds)
        ):
            raise PostprocessInputError(
                "recorded controller gain snapshot is invalid"
            )
        copied_times = times.copy()
        copied_times.setflags(write=False)
        copied_flags = flags.copy()
        copied_flags.setflags(write=False)
        object.__setattr__(self, "bag_path", bag_path)
        object.__setattr__(self, "adapter_revision", adapter_revision)
        object.__setattr__(self, "record_times", copied_times)
        object.__setattr__(self, "pid_control_flags", copied_flags)
        object.__setattr__(self, "source_kinds", source_kinds)


@dataclass(frozen=True)
class AllocationDiagnostics:
    matrix: np.ndarray
    singular_values: np.ndarray
    source_threshold_rank: int
    condition_number: float

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        singular = np.asarray(self.singular_values, dtype=float)
        if (
            matrix.shape != (6, 8)
            or np.any(~np.isfinite(matrix))
            or singular.shape != (6,)
            or np.any(~np.isfinite(singular))
            or np.any(singular < 0.0)
        ):
            raise PostprocessNumericalError(
                "allocation diagnostics are invalid"
            )
        if not isinstance(self.source_threshold_rank, (int, np.integer)):
            raise PostprocessNumericalError("allocation rank must be integral")
        condition = float(self.condition_number)
        if np.isnan(condition) or condition < 0.0:
            raise PostprocessNumericalError(
                "allocation condition number is invalid"
            )
        matrix_copy = matrix.copy()
        matrix_copy.setflags(write=False)
        singular_copy = singular.copy()
        singular_copy.setflags(write=False)
        object.__setattr__(self, "matrix", matrix_copy)
        object.__setattr__(self, "singular_values", singular_copy)
        object.__setattr__(
            self, "source_threshold_rank", int(self.source_threshold_rank)
        )
        object.__setattr__(self, "condition_number", condition)


@dataclass(frozen=True)
class GainCorrection:
    group: str
    axes: Tuple[int, ...]
    scale: float
    old: ControllerGainGroup
    proposed: ControllerGainGroup
    error_before: float
    error_after: float


@dataclass(frozen=True)
class StaticPidProposal:
    nominal_allocation: AllocationDiagnostics
    real_allocation: AllocationDiagnostics
    nominal_pseudoinverse: np.ndarray
    effectiveness: np.ndarray
    dimensionless_effectiveness: np.ndarray
    characteristic_length: float
    corrections: Mapping[str, GainCorrection]
    error_before: float
    error_after: float
    improvement_fraction: float
    coupling_ratio: float
    proposal_status: str
    warnings: Tuple[str, ...]


def load_scale_free_plant(payload: Mapping[str, Any]) -> ScaleFreePlant:
    parameters = _mapping(
        _required(payload, "parameters", "estimator result"),
        "estimator result parameters",
    )
    scale_free = _mapping(
        _required(parameters, "scale_free", "estimator result parameters"),
        "scale_free parameters",
    )
    return ScaleFreePlant(
        inertia_over_mass=_required(
            scale_free, "inertia_over_mass_m2", "scale_free parameters"
        ),
        cog_position_body=_required(
            scale_free, "cog_position_body_m", "scale_free parameters"
        ),
        force_effectiveness_over_mass=_required(
            scale_free,
            "force_effectiveness_over_mass",
            "scale_free parameters",
        ),
        rotor_lag_seconds=_required(
            parameters, "rotor_lag_seconds", "estimator result parameters"
        ),
    )


def load_estimator_result(
    path: Path,
    *,
    allow_non_prior_free_result: bool = False,
    allow_point_estimate_only: bool = False,
) -> EstimatorResult:
    source, payload = _read_json(path, "estimator result JSON")
    required = (
        "overall_case_status",
        "optimization_status",
        "success",
        "case_name",
        "source_commit",
    )
    for key in required:
        _required(payload, key, "estimator result")
    overall = str(payload["overall_case_status"])
    optimization = str(payload["optimization_status"])
    if optimization != "completed" or payload["success"] is not True:
        raise PostprocessInputError(
            "estimator optimization must be completed and successful"
        )
    warnings = []
    if overall == "point_estimate_completed":
        if not allow_point_estimate_only:
            raise PostprocessInputError(
                "point_estimate_completed requires "
                "--allow-point-estimate-only"
            )
        warnings.append("postfit_uncertainty_unavailable")
    elif overall != "completed":
        raise PostprocessInputError(
            "estimator overall_case_status must be completed"
        )
    case_name = str(payload["case_name"])
    prior = _mapping(payload.get("prior", {}), "estimator prior")
    prior_active = prior.get("active", False)
    if not isinstance(prior_active, (bool, np.bool_)):
        raise PostprocessInputError("estimator prior.active must be boolean")
    if case_name == "prior_free":
        if bool(prior_active):
            raise PostprocessInputError(
                "prior_free result cannot contain an active prior"
            )
    else:
        if not allow_non_prior_free_result:
            raise PostprocessInputError(
                "non-prior-free result requires "
                "--allow-non-prior-free-result"
            )
        warnings.append("non_prior_free_estimate")
    source_commit = str(payload["source_commit"])
    if not source_commit:
        raise PostprocessInputError(
            "estimator source_commit must be a non-empty string"
        )
    return EstimatorResult(
        source_path=source,
        source_commit=source_commit,
        case_name=case_name,
        overall_case_status=overall,
        optimization_status=optimization,
        prior=deepcopy(dict(prior)),
        plant=load_scale_free_plant(payload),
        warnings=tuple(warnings),
    )


def load_bag_provenance(path: Path) -> BagProvenance:
    source, payload = _read_json(path, "bag JSON")
    return BagProvenance(
        source_path=source,
        bag_path=_required(payload, "bag_path", "bag JSON"),
        start_seconds=_required(payload, "start_seconds", "bag JSON"),
        end_seconds=_required(payload, "end_seconds", "bag JSON"),
    )


def load_vehicle_model(path: Path) -> VehicleModel:
    source, payload = _read_json(path, "vehicle-model JSON")
    geometry_payload = _mapping(
        _required(payload, "geometry", "vehicle-model JSON"),
        "vehicle-model geometry",
    )
    try:
        parameters = VehicleParameters(
            mass=_required(payload, "mass_kg", "vehicle-model JSON"),
            inertia=_required(payload, "inertia_kg_m2", "vehicle-model JSON"),
            cog_offset=_required(
                payload, "cog_position_body_m", "vehicle-model JSON"
            ),
            force_effectiveness=_required(
                payload, "force_effectiveness", "vehicle-model JSON"
            ),
            torque_effectiveness=_required(
                payload, "torque_effectiveness", "vehicle-model JSON"
            ),
            linear_drag=_required(
                payload, "linear_drag", "vehicle-model JSON"
            ),
            angular_drag=_required(
                payload, "angular_drag", "vehicle-model JSON"
            ),
        )
        geometry = GrapeGeometry(
            rotor_origins=_required(
                geometry_payload,
                "rotor_origins_body_m",
                "vehicle-model geometry",
            ),
            arm_yaws=_required(
                geometry_payload, "arm_yaws_rad", "vehicle-model geometry"
            ),
            rotor_directions=_required(
                geometry_payload,
                "rotor_directions",
                "vehicle-model geometry",
            ),
            moment_force_rate=_required(
                geometry_payload,
                "moment_force_rate_m",
                "vehicle-model geometry",
            ),
            thrust_offset=_required(
                geometry_payload,
                "thrust_offset_m",
                "vehicle-model geometry",
            ),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, PostprocessInputError):
            raise
        raise PostprocessInputError(
            "vehicle-model JSON is invalid: {}".format(error)
        ) from error
    return VehicleModel(source, parameters, geometry)


def _yaml_bool(
    controller: Mapping[str, Any], key: str, default: bool
) -> Tuple[bool, str]:
    if key not in controller:
        return bool(default), "cpp_default"
    value = controller[key]
    if not isinstance(value, (bool, np.bool_)):
        raise PostprocessInputError(
            "controller.{} must be boolean".format(key)
        )
    return bool(value), "yaml"


def resolve_controller_mode(document: Mapping[str, Any]) -> ControllerMode:
    controller = _mapping(
        _required(document, "controller", "controller YAML"),
        "controller YAML controller",
    )
    if "gimbal_dof" in controller:
        value = controller["gimbal_dof"]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise PostprocessInputError("controller.gimbal_dof must be integer")
        gimbal_dof = int(value)
        gimbal_dof_source = "yaml"
    else:
        gimbal_dof = 1
        gimbal_dof_source = "cpp_default"
    underactuate, underactuate_source = _yaml_bool(
        controller, "underactuate", False
    )
    gimbal_calc, gimbal_calc_source = _yaml_bool(
        controller, "gimbal_calc_in_fc", True
    )
    hovering, hovering_source = _yaml_bool(
        controller, "hovering_approximate", False
    )
    yaw = _mapping(
        _required(controller, "yaw", "controller YAML controller"),
        "controller.yaw",
    )
    yaw_d = _required(yaw, "need_d_control", "controller.yaw")
    if not isinstance(yaw_d, (bool, np.bool_)):
        raise PostprocessInputError(
            "controller.yaw.need_d_control must be boolean"
        )
    unsupported = []
    if gimbal_dof != 1:
        unsupported.append("gimbal_dof != 1")
    if underactuate:
        unsupported.append("underactuate == true")
    if gimbal_calc:
        unsupported.append("gimbal_calc_in_fc == true")
    if not bool(yaw_d):
        unsupported.append("yaw.need_d_control == false")
    if unsupported:
        raise PostprocessInputError(
            "unsupported controller branch: {}".format(", ".join(unsupported))
        )
    return ControllerMode(
        gimbal_dof=gimbal_dof,
        gimbal_dof_source=gimbal_dof_source,
        underactuate=underactuate,
        underactuate_source=underactuate_source,
        gimbal_calc_in_fc=gimbal_calc,
        gimbal_calc_in_fc_source=gimbal_calc_source,
        hovering_approximate=hovering,
        hovering_approximate_source=hovering_source,
        yaw_need_d_control=bool(yaw_d),
    )


def _gain_group(controller: Mapping[str, Any], group: str) -> ControllerGainGroup:
    group_payload = _mapping(
        _required(controller, group, "controller YAML controller"),
        "controller.{}".format(group),
    )
    values = [
        _finite_scalar(
            _required(group_payload, gain, "controller.{}".format(group)),
            "controller.{}.{}".format(group, gain),
        )
        for gain in PID_GAIN_NAMES
    ]
    return ControllerGainGroup(*values)


def _reference_controller_gains() -> Mapping[str, ControllerGainGroup]:
    snapshot = ControllerConfig.grape()
    axis = {"xy": 0, "z": 2, "roll_pitch": 3, "yaw": 5}
    return {
        group: ControllerGainGroup(
            snapshot.pid[index].p_gain,
            snapshot.pid[index].i_gain,
            snapshot.pid[index].d_gain,
        )
        for group, index in axis.items()
    }


def load_controller_yaml(path: Path) -> ControllerYaml:
    source = Path(path).expanduser().resolve()
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise PostprocessInputError(
            "controller YAML cannot be read: {}".format(source)
        ) from error
    root = _mapping(document, "controller YAML")
    controller = _mapping(
        _required(root, "controller", "controller YAML"),
        "controller YAML controller",
    )
    gains = {group: _gain_group(controller, group) for group in PID_GROUPS}
    reference = _reference_controller_gains()
    differs = any(
        gains[group].as_dict() != reference[group].as_dict()
        for group in PID_GROUPS
    )
    return ControllerYaml(
        source_path=source,
        sha256=_sha256(source),
        document=deepcopy(dict(root)),
        gains=gains,
        mode=resolve_controller_mode(root),
        reference_values_differ=differs,
    )


def recorded_controller_gains_from_snapshot(
    snapshot: Any,
    *,
    bag_path: str,
    adapter_revision: str,
) -> RecordedControllerGains:
    """Adapt the existing rosbag snapshot contract without consulting YAML."""

    groups = tuple(str(value) for value in getattr(snapshot, "groups", ()))
    if groups != tuple(PID_GROUPS):
        raise PostprocessInputError(
            "ROS bag controller snapshot has non-canonical gain groups"
        )
    values = np.asarray(getattr(snapshot, "gains", None), dtype=float)
    if values.shape != (len(PID_GROUPS), len(PID_GAIN_NAMES)):
        raise PostprocessInputError(
            "ROS bag controller snapshot must contain four P/I/D rows"
        )
    gains = {
        group: ControllerGainGroup(*values[index])
        for index, group in enumerate(PID_GROUPS)
    }
    return RecordedControllerGains(
        bag_path=bag_path,
        adapter_revision=adapter_revision,
        gains=gains,
        record_times=np.asarray(getattr(snapshot, "record_times", None)),
        pid_control_flags=np.asarray(
            getattr(snapshot, "pid_control_flags", None)
        ),
        source_kinds=tuple(getattr(snapshot, "source_kinds", ())),
    )


def load_recorded_controller_gains(
    bag: BagProvenance,
) -> RecordedControllerGains:
    """Read the effective PID gains from the selected ROS bag interval.

    This deliberately reuses :func:`grape_param_estim.real_rosbag.load_flight_data`
    and its dynamic-reconfigure event policy.  The controller YAML is not a
    fallback source for flight-time gains.
    """

    if not isinstance(bag, BagProvenance):
        raise TypeError("bag must be a BagProvenance")
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
            "ROS bag controller gains cannot be reconstructed: {}".format(
                error
            )
        ) from error
    actual_path = str(Path(flight.provenance.bag_path).expanduser().resolve())
    expected_path = str(Path(bag.bag_path).expanduser().resolve())
    if actual_path != expected_path:
        raise PostprocessInputError(
            "ROS bag adapter returned a different bag path"
        )
    return recorded_controller_gains_from_snapshot(
        flight.controller_snapshot,
        bag_path=actual_path,
        adapter_revision=flight.provenance.adapter_revision,
    )


def build_controller_snapshot_geometry(model: VehicleModel) -> GrapeGeometry:
    """Convert BODY thrust-link origins to controller CoG-relative origins."""

    if not isinstance(model, VehicleModel):
        raise TypeError("model must be a VehicleModel")
    geometry = model.body_geometry
    return GrapeGeometry(
        rotor_origins=(
            geometry.rotor_origins - model.parameters.cog_offset[None, :]
        ),
        arm_yaws=geometry.arm_yaws,
        rotor_directions=geometry.rotor_directions,
        moment_force_rate=geometry.moment_force_rate,
        thrust_offset=geometry.thrust_offset,
    )


def _allocation_diagnostics(matrix: np.ndarray) -> AllocationDiagnostics:
    selected = np.asarray(matrix, dtype=float)
    if selected.shape != (6, 8) or np.any(~np.isfinite(selected)):
        raise PostprocessNumericalError(
            "allocation matrix must be finite and 6x8"
        )
    singular = np.linalg.svd(selected, compute_uv=False)
    rank = int(np.count_nonzero(singular > SOURCE_SVD_THRESHOLD))
    condition = (
        float(singular[0] / singular[-1])
        if singular[-1] > 0.0
        else float("inf")
    )
    return AllocationDiagnostics(selected, singular, rank, condition)


def build_nominal_controller_allocation(
    model: VehicleModel,
) -> AllocationDiagnostics:
    geometry = build_controller_snapshot_geometry(model)
    matrix = acceleration_allocation_matrix(
        model.parameters, geometry, np.zeros(4)
    )
    diagnostics = _allocation_diagnostics(matrix)
    if diagnostics.source_threshold_rank != 6:
        raise PostprocessNumericalError(
            "nominal controller allocation does not have source-threshold row rank 6"
        )
    return diagnostics


def build_real_scale_free_allocation(
    plant: ScaleFreePlant, model: VehicleModel
) -> AllocationDiagnostics:
    if not isinstance(plant, ScaleFreePlant):
        raise TypeError("plant must be a ScaleFreePlant")
    geometry = model.body_geometry
    result = np.zeros((6, 8), dtype=float)
    local_basis = (
        np.asarray((0.0, 1.0, 0.0)),
        np.asarray((0.0, 0.0, 1.0)),
    )
    for rotor in range(4):
        origin = geometry.rotor_origins[rotor] - plant.cog_position_body
        for component, local_force in enumerate(local_basis):
            column = 2 * rotor + component
            direction = _rotate_z(local_force, geometry.arm_yaws[rotor])
            torque_per_unit_force = np.cross(origin, direction) + (
                model.parameters.torque_effectiveness[rotor]
                * geometry.rotor_directions[rotor]
                * geometry.moment_force_rate
                * direction
            )
            force_over_mass = (
                plant.force_effectiveness_over_mass[rotor] * direction
            )
            torque_over_mass = (
                plant.force_effectiveness_over_mass[rotor]
                * torque_per_unit_force
            )
            result[:3, column] = force_over_mass
            try:
                result[3:, column] = np.linalg.solve(
                    plant.inertia_over_mass, torque_over_mass
                )
            except np.linalg.LinAlgError as error:
                raise PostprocessNumericalError(
                    "estimated inertia-over-mass solve failed"
                ) from error
    return _allocation_diagnostics(result)


def source_compatible_pseudoinverse(matrix: np.ndarray) -> np.ndarray:
    selected = np.asarray(matrix, dtype=float)
    if selected.ndim != 2 or np.any(~np.isfinite(selected)):
        raise PostprocessNumericalError(
            "pseudoinverse input must be a finite matrix"
        )
    left, singular, right_transpose = np.linalg.svd(
        selected, full_matrices=False
    )
    inverse = np.asarray(
        [
            1.0 / value if value > SOURCE_SVD_THRESHOLD else 0.0
            for value in singular
        ]
    )
    result = right_transpose.T @ np.diag(inverse) @ left.T
    if np.any(~np.isfinite(result)):
        raise PostprocessNumericalError("pseudoinverse is not finite")
    return result


def characteristic_length(
    model: VehicleModel, override: Optional[float] = None
) -> float:
    if override is not None:
        return _finite_scalar(
            override, "characteristic_length", positive=True
        )
    inertia_over_mass = model.parameters.inertia / model.parameters.mass
    value = float(np.sqrt(np.trace(inertia_over_mass) / 3.0))
    if not np.isfinite(value) or value <= 0.0:
        raise PostprocessNumericalError(
            "nominal characteristic length is invalid"
        )
    return value


def dimensionless_effectiveness(
    effectiveness: np.ndarray, length: float
) -> np.ndarray:
    matrix = np.asarray(effectiveness, dtype=float)
    if matrix.shape != (6, 6) or np.any(~np.isfinite(matrix)):
        raise PostprocessNumericalError(
            "effectiveness must be a finite 6x6 matrix"
        )
    selected_length = _finite_scalar(
        length, "characteristic_length", positive=True
    )
    scale = np.diag((1.0, 1.0, 1.0, selected_length, selected_length, selected_length))
    inverse = np.diag(
        (1.0, 1.0, 1.0, 1.0 / selected_length, 1.0 / selected_length, 1.0 / selected_length)
    )
    result = scale @ matrix @ inverse
    if np.any(~np.isfinite(result)):
        raise PostprocessNumericalError(
            "dimensionless effectiveness is not finite"
        )
    return result


def group_scale(matrix: np.ndarray, axes: Sequence[int]) -> float:
    selected = np.asarray(matrix, dtype=float)
    indices = tuple(int(value) for value in axes)
    if (
        selected.shape != (6, 6)
        or np.any(~np.isfinite(selected))
        or not indices
        or any(value < 0 or value >= 6 for value in indices)
        or len(set(indices)) != len(indices)
    ):
        raise PostprocessNumericalError("group-scale inputs are invalid")
    denominator = float(np.sum(selected[:, indices] ** 2))
    numerator = float(sum(selected[index, index] for index in indices))
    if denominator <= np.finfo(float).tiny:
        raise PostprocessNumericalError(
            "gain-group scale has a numerically zero denominator"
        )
    scale = numerator / denominator
    if not np.isfinite(scale) or scale <= 0.0:
        raise PostprocessNumericalError(
            "gain-group scale must be finite and positive"
        )
    return float(scale)


def calculate_gain_corrections(
    matrix: np.ndarray,
    gains: Mapping[str, ControllerGainGroup],
) -> Mapping[str, GainCorrection]:
    selected = np.asarray(matrix, dtype=float)
    identity = np.eye(6)
    corrections = {}
    for group in PID_GROUPS:
        axes = GAIN_GROUP_AXES[group]
        old = gains[group]
        scale = group_scale(selected, axes)
        proposed = ControllerGainGroup(
            scale * old.p_gain,
            scale * old.i_gain,
            scale * old.d_gain,
        )
        columns = np.asarray(axes, dtype=int)
        error_before = float(
            np.linalg.norm(
                selected[:, columns] - identity[:, columns], ord="fro"
            )
        )
        error_after = float(
            np.linalg.norm(
                scale * selected[:, columns] - identity[:, columns],
                ord="fro",
            )
        )
        if error_after > error_before + 1.0e-12 * max(1.0, error_before):
            raise PostprocessNumericalError(
                "group least-squares correction increased its objective"
            )
        corrections[group] = GainCorrection(
            group=group,
            axes=axes,
            scale=scale,
            old=old,
            proposed=proposed,
            error_before=error_before,
            error_after=error_after,
        )
    return corrections


def compute_static_pid_proposal(
    result: EstimatorResult,
    model: VehicleModel,
    controller: ControllerYaml,
    recorded_gains: RecordedControllerGains,
    *,
    characteristic_length_override: Optional[float] = None,
    large_scale_min: float = DEFAULT_LARGE_SCALE_MIN,
    large_scale_max: float = DEFAULT_LARGE_SCALE_MAX,
    strong_coupling_threshold: float = DEFAULT_STRONG_COUPLING_THRESHOLD,
) -> StaticPidProposal:
    lower = _finite_scalar(large_scale_min, "large_scale_min", positive=True)
    upper = _finite_scalar(large_scale_max, "large_scale_max", positive=True)
    coupling_threshold = _finite_scalar(
        strong_coupling_threshold,
        "strong_coupling_threshold",
        positive=True,
    )
    if lower >= upper:
        raise PostprocessInputError(
            "large_scale_min must be smaller than large_scale_max"
        )
    nominal = build_nominal_controller_allocation(model)
    real = build_real_scale_free_allocation(result.plant, model)
    pseudoinverse = source_compatible_pseudoinverse(nominal.matrix)
    effectiveness = real.matrix @ pseudoinverse
    length = characteristic_length(model, characteristic_length_override)
    dimensionless = dimensionless_effectiveness(effectiveness, length)
    if not isinstance(recorded_gains, RecordedControllerGains):
        raise TypeError("recorded_gains must be RecordedControllerGains")
    corrections = calculate_gain_corrections(
        dimensionless, recorded_gains.gains
    )
    group_scales = np.ones(6)
    for correction in corrections.values():
        group_scales[np.asarray(correction.axes, dtype=int)] = correction.scale
    correction_matrix = np.diag(group_scales)
    identity = np.eye(6)
    error_before = float(np.linalg.norm(dimensionless - identity, ord="fro"))
    error_after = float(
        np.linalg.norm(dimensionless @ correction_matrix - identity, ord="fro")
    )
    improvement = (
        (error_before - error_after) / error_before
        if error_before > 0.0
        else 0.0
    )
    denominator = float(np.linalg.norm(dimensionless, ord="fro"))
    coupling = (
        float(
            np.linalg.norm(
                dimensionless - np.diag(np.diag(dimensionless)), ord="fro"
            )
            / denominator
        )
        if denominator > 0.0
        else 0.0
    )
    warnings = list(result.warnings)
    mode = controller.mode
    if any(
        source == "cpp_default"
        for source in (
            mode.gimbal_dof_source,
            mode.underactuate_source,
            mode.gimbal_calc_in_fc_source,
            mode.hovering_approximate_source,
        )
    ):
        warnings.append("controller_mode_resolved_from_source_default")
    if controller.reference_values_differ:
        warnings.append("controller_reference_values_differ")
    if any(
        recorded_gains.gains[group].as_dict()
        != controller.gains[group].as_dict()
        for group in PID_GROUPS
    ):
        warnings.append("recorded_controller_gains_differ_from_yaml")
    if (
        real.source_threshold_rank < 6
        or not np.isfinite(real.condition_number)
        or real.condition_number > DEFAULT_ALLOCATION_CONDITION_WARNING
    ):
        warnings.append("estimated_allocation_ill_conditioned")
    if any(
        correction.scale < lower or correction.scale > upper
        for correction in corrections.values()
    ):
        warnings.append("large_static_gain_change")
    if coupling > coupling_threshold:
        warnings.append("strong_axis_coupling")
    warnings.extend(
        (
            "static_correction_does_not_cover_rotor_lag",
            "feedforward_not_corrected",
        )
    )
    warning_tuple = _deduplicate(warnings)
    review_warnings = {
        "non_prior_free_estimate",
        "postfit_uncertainty_unavailable",
        "estimated_allocation_ill_conditioned",
        "large_static_gain_change",
        "strong_axis_coupling",
        "controller_reference_values_differ",
        "recorded_controller_gains_differ_from_yaml",
    }
    proposal_status = (
        "review_required"
        if any(value in review_warnings for value in warning_tuple)
        else "valid"
    )
    return StaticPidProposal(
        nominal_allocation=nominal,
        real_allocation=real,
        nominal_pseudoinverse=pseudoinverse,
        effectiveness=effectiveness,
        dimensionless_effectiveness=dimensionless,
        characteristic_length=length,
        corrections=corrections,
        error_before=error_before,
        error_after=error_after,
        improvement_fraction=float(improvement),
        coupling_ratio=coupling,
        proposal_status=proposal_status,
        warnings=warning_tuple,
    )


def apply_gain_corrections_to_yaml(
    controller: ControllerYaml,
    corrections: Mapping[str, GainCorrection],
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    full = deepcopy(dict(controller.document))
    full_controller = _mapping(full["controller"], "controller YAML controller")
    overlay: dict[str, Any] = {"controller": {}}
    for group in PID_GROUPS:
        proposed = corrections[group].proposed.as_dict()
        destination = _mapping(
            full_controller[group], "controller.{}".format(group)
        )
        for gain in PID_GAIN_NAMES:
            destination[gain] = proposed[gain]
        overlay["controller"][group] = dict(proposed)
    return full, overlay


def _prior_report(prior: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "active": bool(prior.get("active", False)),
        "name": prior.get("name"),
        "role": prior.get("role"),
        "source": prior.get("source_path", prior.get("source")),
    }


def _diagnostic_report(value: AllocationDiagnostics) -> Mapping[str, Any]:
    return {
        "singular_values": value.singular_values.tolist(),
        "source_threshold_rank": value.source_threshold_rank,
        "source_absolute_svd_threshold": SOURCE_SVD_THRESHOLD,
        "condition_number": (
            value.condition_number
            if np.isfinite(value.condition_number)
            else None
        ),
    }


def build_report(
    *,
    source_commit: str,
    result: EstimatorResult,
    bag: BagProvenance,
    model: VehicleModel,
    controller: ControllerYaml,
    recorded_gains: RecordedControllerGains,
    proposal: StaticPidProposal,
) -> Mapping[str, Any]:
    controller_geometry = build_controller_snapshot_geometry(model)
    mode = controller.mode
    gains = {}
    for group, correction in proposal.corrections.items():
        gains[group] = {
            "axes": list(correction.axes),
            "scale": correction.scale,
            "old": dict(correction.old.as_dict()),
            "proposed": dict(correction.proposed.as_dict()),
            "error_before_frobenius": correction.error_before,
            "error_after_frobenius": correction.error_after,
        }
    nominal = model.parameters
    return {
        "schema": POSTPROCESS_SCHEMA,
        "method": METHOD,
        "source_commit": str(source_commit),
        "input": {
            "estimator_result_json": str(result.source_path),
            "estimator_source_commit": result.source_commit,
            "estimator_case_name": result.case_name,
            "estimator_overall_case_status": result.overall_case_status,
            "estimator_optimization_status": result.optimization_status,
            "estimator_prior": _prior_report(result.prior),
            "bag_json": str(bag.source_path),
            "bag_path": bag.bag_path,
            "bag_interval_seconds": [bag.start_seconds, bag.end_seconds],
            "vehicle_model_json": str(model.source_path),
            "controller_yaml": str(controller.source_path),
            "controller_yaml_sha256": controller.sha256,
            "controller_gain_source": "rosbag_recorded_dynamic_reconfigure",
        },
        "controller_gain_snapshot": {
            "source": "rosbag_recorded_dynamic_reconfigure",
            "bag_path": recorded_gains.bag_path,
            "adapter_revision": recorded_gains.adapter_revision,
            "group_order": list(PID_GROUPS),
            "gain_order": list(PID_GAIN_NAMES),
            "gains": {
                group: dict(recorded_gains.gains[group].as_dict())
                for group in PID_GROUPS
            },
            "record_times": recorded_gains.record_times.tolist(),
            "pid_control_flags": (
                recorded_gains.pid_control_flags.tolist()
            ),
            "source_kinds": list(recorded_gains.source_kinds),
            "controller_yaml_template_gains": {
                group: dict(controller.gains[group].as_dict())
                for group in PID_GROUPS
            },
            "recorded_gains_differ_from_yaml": any(
                recorded_gains.gains[group].as_dict()
                != controller.gains[group].as_dict()
                for group in PID_GROUPS
            ),
        },
        "controller_mode": {
            "gimbal_dof": mode.gimbal_dof,
            "gimbal_dof_source": mode.gimbal_dof_source,
            "underactuate": mode.underactuate,
            "underactuate_source": mode.underactuate_source,
            "gimbal_calc_in_fc": mode.gimbal_calc_in_fc,
            "gimbal_calc_in_fc_source": mode.gimbal_calc_in_fc_source,
            "hovering_approximate": mode.hovering_approximate,
            "hovering_approximate_source": mode.hovering_approximate_source,
            "yaw_need_d_control": mode.yaw_need_d_control,
        },
        "scale_free_plant": {
            "inertia_over_mass_m2": result.plant.inertia_over_mass.tolist(),
            "cog_position_body_m": result.plant.cog_position_body.tolist(),
            "force_effectiveness_over_mass": (
                result.plant.force_effectiveness_over_mass.tolist()
            ),
            "rotor_lag_seconds": result.plant.rotor_lag_seconds,
        },
        "nominal_controller_model": {
            "mass_kg": nominal.mass,
            "inertia_kg_m2": nominal.inertia.tolist(),
            "inertia_over_mass_m2": (
                nominal.inertia / nominal.mass
            ).tolist(),
            "cog_position_body_m": nominal.cog_offset.tolist(),
            "torque_effectiveness": nominal.torque_effectiveness.tolist(),
            "torque_effectiveness_source": "fixed_nominal_vehicle_model",
            "rotor_origins_body_m": model.body_geometry.rotor_origins.tolist(),
            "controller_rotor_origins_from_cog_m": (
                controller_geometry.rotor_origins.tolist()
            ),
            "arm_yaws_rad": model.body_geometry.arm_yaws.tolist(),
            "rotor_directions": model.body_geometry.rotor_directions.tolist(),
            "moment_force_rate_m": model.body_geometry.moment_force_rate,
        },
        "linearization": {
            "gimbal_angles_rad": [0.0, 0.0, 0.0, 0.0],
            "body_angular_velocity_rad_s": [0.0, 0.0, 0.0],
            "unsaturated_pid_region": True,
            "characteristic_length_m": proposal.characteristic_length,
            "characteristic_length_method": (
                "explicit_override"
                if not np.isclose(
                    proposal.characteristic_length,
                    characteristic_length(model),
                    rtol=0.0,
                    atol=0.0,
                )
                else "nominal_radius_of_gyration"
            ),
        },
        "allocation": {
            "A_cmd": proposal.nominal_allocation.matrix.tolist(),
            "A_real": proposal.real_allocation.matrix.tolist(),
            "A_cmd_pseudoinverse": proposal.nominal_pseudoinverse.tolist(),
            "A_cmd_diagnostics": _diagnostic_report(
                proposal.nominal_allocation
            ),
            "A_real_diagnostics": _diagnostic_report(proposal.real_allocation),
            "H": proposal.effectiveness.tolist(),
            "H_dimensionless": (
                proposal.dimensionless_effectiveness.tolist()
            ),
            "H_dimensionless_diagonal": np.diag(
                proposal.dimensionless_effectiveness
            ).tolist(),
        },
        "gain_groups": gains,
        "overall": {
            "error_before_frobenius": proposal.error_before,
            "error_after_frobenius": proposal.error_after,
            "improvement_fraction": proposal.improvement_fraction,
            "off_diagonal_coupling_ratio": proposal.coupling_ratio,
            "proposal_status": proposal.proposal_status,
            "warnings": list(proposal.warnings),
        },
    }


__all__ = [
    "AllocationDiagnostics",
    "BagProvenance",
    "ControllerGainGroup",
    "ControllerMode",
    "ControllerYaml",
    "DEFAULT_LARGE_SCALE_MAX",
    "DEFAULT_LARGE_SCALE_MIN",
    "DEFAULT_STRONG_COUPLING_THRESHOLD",
    "EstimatorResult",
    "GAIN_GROUP_AXES",
    "GainCorrection",
    "METHOD",
    "POSTPROCESS_SCHEMA",
    "PostprocessInputError",
    "PostprocessNumericalError",
    "RecordedControllerGains",
    "SOURCE_SVD_THRESHOLD",
    "ScaleFreePlant",
    "StaticPidProposal",
    "VehicleModel",
    "apply_gain_corrections_to_yaml",
    "build_controller_snapshot_geometry",
    "build_nominal_controller_allocation",
    "build_real_scale_free_allocation",
    "build_report",
    "calculate_gain_corrections",
    "characteristic_length",
    "compute_static_pid_proposal",
    "dimensionless_effectiveness",
    "group_scale",
    "load_bag_provenance",
    "load_controller_yaml",
    "load_estimator_result",
    "load_recorded_controller_gains",
    "load_scale_free_plant",
    "load_vehicle_model",
    "resolve_controller_mode",
    "recorded_controller_gains_from_snapshot",
    "source_compatible_pseudoinverse",
]
