#!/usr/bin/env python3
"""Independent geometric Savitzky--Golay rigid-body dynamics estimator.

This module owns its rigid-body dynamics, optimization, rollout, command-lag
search, and report generation. The spline estimator is a separate
ablation/reference implementation and is not imported.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import re
import os
from pathlib import Path
import shutil
import sys
import textwrap
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

import savgol_trajectory as sg


os.environ.setdefault("MPLCONFIGDIR", "/tmp/grape-minimal-matplotlib")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from legacies import deterministic_continuation_estimator as continuation  # noqa: E402
from legacies import deterministic_estimator as baseline  # noqa: E402
from legacies import (  # noqa: E402
    deterministic_multi_bag_multiple_shooting_estimator as multi,
)
from legacies import deterministic_multiple_shooting_estimator as strict  # noqa: E402
from legacies import (  # noqa: E402
    deterministic_smooth_lag_multiple_shooting_estimator as smooth,
)
from smooth_command import QuinticSmoothZoh  # noqa: E402
from spline_trajectory import (  # noqa: E402
    PoseSplineEvaluation,
    PoseSplineSelection,
    candidate_payload,
    select_pose_spline,
)
from grape_param_estim.dynamics import (  # noqa: E402
    FullSixDofPlant,
    advance_actuators,
)
from grape_param_estim.geometry import (  # noqa: E402
    matrix_to_quaternion,
    quaternion_to_matrix,
    skew,
    so3_log,
)
from grape_param_estim.real_rosbag import load_flight_data  # noqa: E402
from grape_param_estim.system import (  # noqa: E402
    GRAVITY,
    ActuatorParameters,
    ActuatorState,
    GrapeGeometry,
    RigidBodyState,
    VehicleParameters,
)


SCHEMA = "grape-param-estim/minimal-deterministic-spline-dynamics/v3"
OUTPUT_SUBDIRECTORY = "deterministic_spline_dynamics"
DATA_DICTIONARY_SOURCE = Path(__file__).resolve().with_name(
    "deterministic_spline_dynamics_data_dictionary.md"
)

# Current spline-dynamics physical chart.
#
# Rotor force effectiveness is deliberately represented by four independent
# log coordinates.  There is no sum-to-zero / geometric-mean normalization:
# common thrust scale is left in the parameter space so data-only ridge
# structure can be exposed later by the confidence analysis.
PHYSICAL_DIMENSION = 14
GLOBAL_DIMENSION = PHYSICAL_DIMENSION + 1
DELAY_INDEX = PHYSICAL_DIMENSION
PHYSICAL_PARAMETER_NAMES = (
    "log_mass_scale",
    "log_second_moment_cholesky_xx_scale",
    "log_second_moment_cholesky_yy_scale",
    "log_second_moment_cholesky_zz_scale",
    "normalized_second_moment_cholesky_yx_offset",
    "normalized_second_moment_cholesky_zx_offset",
    "normalized_second_moment_cholesky_zy_offset",
    "cog_position_body_x_m",
    "cog_position_body_y_m",
    "cog_position_body_z_m",
    "log_force_effectiveness_1",
    "log_force_effectiveness_2",
    "log_force_effectiveness_3",
    "log_force_effectiveness_4",
)

PHYSICAL_VALUE_NAMES = (
    "mass_kg",
    "inertia_xx_kg_m2",
    "inertia_yy_kg_m2",
    "inertia_zz_kg_m2",
    "inertia_xy_kg_m2",
    "inertia_xz_kg_m2",
    "inertia_yz_kg_m2",
    "cog_position_body_x_m",
    "cog_position_body_y_m",
    "cog_position_body_z_m",
    "force_effectiveness_1",
    "force_effectiveness_2",
    "force_effectiveness_3",
    "force_effectiveness_4",
)
_INERTIA_VALUE_COMPONENTS = (
    (0, 0),
    (1, 1),
    (2, 2),
    (0, 1),
    (0, 2),
    (1, 2),
)


@dataclass(frozen=True)
class VehicleModelInput:
    """Deterministic vehicle-model reference used to define the local chart."""

    source_path: Path
    parameters: VehicleParameters
    geometry: GrapeGeometry
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class GaussianPhysicalPrior:
    """Independent Gaussian prior in user-facing physical coordinates."""

    source_path: Path
    mean: np.ndarray
    std: np.ndarray
    raw: Mapping[str, Any]

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        std = np.asarray(self.std, dtype=float)
        expected = (len(PHYSICAL_VALUE_NAMES),)
        if (
            mean.shape != expected
            or std.shape != expected
            or np.any(~np.isfinite(mean))
            or np.any(~np.isfinite(std))
            or np.any(std <= 0.0)
        ):
            raise ValueError("physical Gaussian prior vectors are invalid")
        object.__setattr__(self, "mean", mean.copy())
        object.__setattr__(self, "std", std.copy())


def _read_json_object(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = path.expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("{} JSON cannot be read: {}".format(label, source)) from error
    if not isinstance(raw, dict):
        raise ValueError("{} JSON root must be an object".format(label))
    return source, raw


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or np.any(~np.isfinite(result)):
        raise ValueError("{} must contain {} finite values".format(name, size))
    return result


def load_vehicle_model(path: Path) -> VehicleModelInput:
    """Load the deterministic reference model.

    This file is not a prior and carries no uncertainty.  In particular,
    ``cog_position_body_m`` is the actual CoG position expressed from the
    physical body-frame origin used by the sensor extrinsics and geometry.
    """

    source, raw = _read_json_object(path, "vehicle model")
    required = (
        "mass_kg",
        "inertia_kg_m2",
        "cog_position_body_m",
        "force_effectiveness",
        "torque_effectiveness",
        "linear_drag",
        "angular_drag",
        "geometry",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(
            "vehicle model JSON is missing: {}".format(", ".join(missing))
        )

    inertia = np.asarray(raw["inertia_kg_m2"], dtype=float)
    if (
        inertia.shape != (3, 3)
        or np.any(~np.isfinite(inertia))
        or not np.allclose(inertia, inertia.T, atol=1.0e-12, rtol=0.0)
    ):
        raise ValueError("inertia_kg_m2 must be a finite symmetric 3x3 matrix")

    parameters = VehicleParameters(
        mass=float(raw["mass_kg"]),
        inertia=inertia,
        # VehicleParameters retains the historical field name ``cog_offset``.
        # In this estimator it means the absolute body-frame CoG position.
        cog_offset=_finite_vector(
            raw["cog_position_body_m"], 3, "cog_position_body_m"
        ),
        force_effectiveness=_finite_vector(
            raw["force_effectiveness"], 4, "force_effectiveness"
        ),
        torque_effectiveness=_finite_vector(
            raw["torque_effectiveness"], 4, "torque_effectiveness"
        ),
        linear_drag=_finite_vector(raw["linear_drag"], 3, "linear_drag"),
        angular_drag=_finite_vector(raw["angular_drag"], 3, "angular_drag"),
    )

    geometry_raw = raw["geometry"]
    if not isinstance(geometry_raw, dict):
        raise ValueError("geometry must be an object")
    geometry_required = (
        "rotor_origins_body_m",
        "arm_yaws_rad",
        "rotor_directions",
        "moment_force_rate_m",
        "thrust_offset_m",
    )
    missing_geometry = [
        name for name in geometry_required if name not in geometry_raw
    ]
    if missing_geometry:
        raise ValueError(
            "vehicle model geometry is missing: {}".format(
                ", ".join(missing_geometry)
            )
        )

    rotor_origins = np.asarray(
        geometry_raw["rotor_origins_body_m"], dtype=float
    )
    if rotor_origins.shape != (4, 3) or np.any(~np.isfinite(rotor_origins)):
        raise ValueError("rotor_origins_body_m must be a finite 4x3 array")
    geometry = GrapeGeometry(
        rotor_origins=rotor_origins,
        arm_yaws=_finite_vector(
            geometry_raw["arm_yaws_rad"], 4, "arm_yaws_rad"
        ),
        rotor_directions=_finite_vector(
            geometry_raw["rotor_directions"], 4, "rotor_directions"
        ),
        moment_force_rate=float(geometry_raw["moment_force_rate_m"]),
        thrust_offset=float(geometry_raw["thrust_offset_m"]),
    )
    return VehicleModelInput(
        source_path=source,
        parameters=parameters,
        geometry=geometry,
        raw=raw,
    )


def load_parameter_prior(path: Path) -> GaussianPhysicalPrior:
    """Load the deliberately small all-Gaussian physical prior."""

    source, raw = _read_json_object(path, "parameter prior")
    required = (
        "mass_kg",
        "inertia_kg_m2",
        "cog_position_body_m",
        "force_effectiveness",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(
            "parameter prior JSON is missing: {}".format(", ".join(missing))
        )

    def block(name: str) -> Mapping[str, Any]:
        value = raw[name]
        if (
            not isinstance(value, dict)
            or "mean" not in value
            or "std" not in value
        ):
            raise ValueError("{} prior requires mean and std".format(name))
        return value

    mass = block("mass_kg")
    inertia = block("inertia_kg_m2")
    cog = block("cog_position_body_m")
    effectiveness = block("force_effectiveness")

    mass_mean = float(mass["mean"])
    mass_std = float(mass["std"])
    inertia_mean = np.asarray(inertia["mean"], dtype=float)
    inertia_std = float(inertia["std"])
    cog_mean = _finite_vector(cog["mean"], 3, "cog_position_body_m.mean")
    cog_std = float(cog["std"])
    effectiveness_mean = _finite_vector(
        effectiveness["mean"], 4, "force_effectiveness.mean"
    )
    effectiveness_std = float(effectiveness["std"])

    if (
        not np.isfinite(mass_mean)
        or mass_mean <= 0.0
        or not np.isfinite(mass_std)
        or mass_std <= 0.0
        or inertia_mean.shape != (3, 3)
        or np.any(~np.isfinite(inertia_mean))
        or not np.allclose(inertia_mean, inertia_mean.T, atol=1.0e-12, rtol=0.0)
        or not np.isfinite(inertia_std)
        or inertia_std <= 0.0
        or not np.isfinite(cog_std)
        or cog_std <= 0.0
        or np.any(effectiveness_mean <= 0.0)
        or not np.isfinite(effectiveness_std)
        or effectiveness_std <= 0.0
    ):
        raise ValueError("parameter prior mean/std values are invalid")

    principal = np.linalg.eigvalsh(inertia_mean)
    if (
        np.any(principal <= 0.0)
        or principal[0] + principal[1] <= principal[2]
    ):
        raise ValueError(
            "inertia_kg_m2.mean must be physically admissible"
        )

    mean = np.concatenate(
        (
            np.asarray((mass_mean,), dtype=float),
            np.asarray(
                [inertia_mean[i, j] for i, j in _INERTIA_VALUE_COMPONENTS],
                dtype=float,
            ),
            cog_mean,
            effectiveness_mean,
        )
    )
    std = np.concatenate(
        (
            np.asarray((mass_std,), dtype=float),
            np.full(6, inertia_std, dtype=float),
            np.full(3, cog_std, dtype=float),
            np.full(4, effectiveness_std, dtype=float),
        )
    )
    return GaussianPhysicalPrior(
        source_path=source,
        mean=mean,
        std=std,
        raw=raw,
    )


def physical_parameter_vector(parameters: VehicleParameters) -> np.ndarray:
    inertia = np.asarray(parameters.inertia, dtype=float)
    return np.concatenate(
        (
            np.asarray((parameters.mass,), dtype=float),
            np.asarray(
                [inertia[i, j] for i, j in _INERTIA_VALUE_COMPONENTS],
                dtype=float,
            ),
            np.asarray(parameters.cog_offset, dtype=float),
            np.asarray(parameters.force_effectiveness, dtype=float),
        )
    )


def physical_parameter_jacobian(parameter_jacobian: Any) -> np.ndarray:
    mass = np.asarray(parameter_jacobian.mass, dtype=float)
    dimension = mass.size
    inertia = np.asarray(parameter_jacobian.inertia, dtype=float)
    cog = np.asarray(parameter_jacobian.cog_offset, dtype=float)
    effectiveness = np.asarray(
        parameter_jacobian.force_effectiveness, dtype=float
    )
    if (
        inertia.shape != (3, 3, dimension)
        or cog.shape != (3, dimension)
        or effectiveness.shape != (4, dimension)
    ):
        raise ValueError("physical parameter Jacobian has inconsistent shape")
    rows = [mass]
    rows.extend(inertia[i, j] for i, j in _INERTIA_VALUE_COMPONENTS)
    rows.extend(cog[index] for index in range(3))
    rows.extend(effectiveness[index] for index in range(4))
    return np.vstack(rows)

_CHOLESKY_OFF_DIAGONALS = ((1, 0), (2, 0), (2, 1))
COMPONENT_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class SplineSettings:
    knot_spacing_candidates_seconds: tuple[float, ...]
    collocation_step_seconds: float
    boundary_exclusion_knot_spans_each_side: float = 3.0
    cross_validation_block_seconds: float = 0.1


@dataclass(frozen=True)
class SplineEstimatorConfig:
    multi_bag: multi.MultiBagConfig
    spline: SplineSettings


@dataclass(frozen=True)
class BagSplineData:
    specification: multi.BagSpecification
    normalized_weight: float
    flight: Any
    direct_problem: baseline.DirectShootingProblem
    spline_selection: PoseSplineSelection
    collocation: PoseSplineEvaluation
    rotor_history: QuinticSmoothZoh
    gimbal_history: QuinticSmoothZoh
    initial_gimbal: np.ndarray
    boundary_exclusion_knot_spans_each_side: float = 0.0

    @property
    def collocation_time(self) -> np.ndarray:
        return self.collocation.time


@dataclass(frozen=True)
class BagDynamicsEvaluation:
    acceleration_residual: np.ndarray
    acceleration_jacobian: np.ndarray
    required_body_wrench: np.ndarray
    modeled_body_wrench: np.ndarray
    residual_body_wrench: np.ndarray
    residual_wrench_jacobian: np.ndarray
    cog_position: np.ndarray
    cog_velocity_world: np.ndarray
    cog_acceleration_world: np.ndarray
    actuator_thrust: np.ndarray
    actuator_gimbal: np.ndarray

    @property
    def dynamics_loss(self) -> float:
        residual = np.asarray(self.acceleration_residual, dtype=float)
        return 0.5 * float(np.mean(np.sum(residual * residual, axis=1)))


@dataclass(frozen=True)
class JointDynamicsEvaluation:
    residual: np.ndarray
    jacobian: np.ndarray
    bag_evaluations: tuple[BagDynamicsEvaluation, ...]
    decoded: Any
    physical_coordinate: np.ndarray
    delay_seconds: float
    data_loss: float
    prior_cost: float


@dataclass(frozen=True)
class DynamicsSolution:
    physical_coordinate: np.ndarray
    delay_seconds: float
    evaluation: JointDynamicsEvaluation
    optimizer: Mapping[str, Any]
    gimbal_delay_seconds: Optional[float] = None


class BodyWrenchHistory:
    """Body wrench interpolated only inside support and zero outside it."""

    def __init__(
        self,
        time_axis: Sequence[float],
        body_wrench: np.ndarray,
    ) -> None:
        time_value = np.asarray(time_axis, dtype=float)
        wrench_value = np.asarray(body_wrench, dtype=float)
        if (
            time_value.ndim != 1
            or time_value.size < 2
            or np.any(~np.isfinite(time_value))
            or np.any(np.diff(time_value) <= 0.0)
            or wrench_value.shape != (time_value.size, 6)
            or np.any(~np.isfinite(wrench_value))
        ):
            raise ValueError("external body wrench history is invalid")
        self.time = time_value.copy()
        self.body_wrench = wrench_value.copy()

    def value_at(self, time_value: float) -> np.ndarray:
        query = float(time_value)
        if not np.isfinite(query):
            raise ValueError("external body wrench query must be finite")
        return np.asarray(
            [
                np.interp(
                    query,
                    self.time,
                    self.body_wrench[:, component],
                    left=0.0,
                    right=0.0,
                )
                for component in range(6)
            ],
            dtype=float,
        )

    def __call__(
        self,
        time_value: float,
        _state: RigidBodyState,
    ) -> np.ndarray:
        return self.value_at(time_value)


def load_spline_config(path: Path) -> SplineEstimatorConfig:
    config_path = path.expanduser().resolve()
    base = multi.load_multi_bag_config(config_path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        spline_raw = raw.get("spline", {})
        candidates = tuple(
            float(value)
            for value in spline_raw.get(
                "knot_spacing_candidates_seconds", (0.05, 0.1, 0.2)
            )
        )
        collocation_step = float(
            spline_raw.get("collocation_step_seconds", 0.01)
        )
        boundary_exclusion = float(
            spline_raw.get(
                "boundary_exclusion_knot_spans_each_side", 3.0
            )
        )
        cross_validation_block = float(
            spline_raw.get("cross_validation_block_seconds", 0.1)
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError("spline settings in multi-bag config are invalid") from error
    positive_values = np.asarray(
        candidates + (collocation_step, cross_validation_block), dtype=float
    )
    if (
        not candidates
        or np.any(~np.isfinite(positive_values))
        or np.any(positive_values <= 0.0)
        or not np.isfinite(boundary_exclusion)
        or boundary_exclusion < 0.0
    ):
        raise ValueError(
            "spline spacings and cross-validation block duration must be "
            "positive; boundary exclusion must be finite and nonnegative"
        )
    return SplineEstimatorConfig(
        multi_bag=base,
        spline=SplineSettings(
            candidates,
            collocation_step,
            boundary_exclusion,
            cross_validation_block,
        ),
    )




def _collocation_grid(
    start: float,
    end: float,
    step: float,
) -> np.ndarray:
    count = int(math.floor((end - start) / step + 1.0e-12)) + 1
    if count < 3:
        raise ValueError("collocation support is too short")
    return start + step * np.arange(count, dtype=float)


def _spline_boundary_exclusion_seconds(
    selection: PoseSplineSelection,
    knot_spans_each_side: float,
) -> float:
    spline = selection.spline
    effective_spacing = max(
        spline.effective_position_knot_spacing_seconds,
        spline.effective_rotation_knot_spacing_seconds,
    )
    margin = float(knot_spans_each_side) * float(effective_spacing)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("spline boundary exclusion is invalid")
    return margin


def cog_kinematics_from_pose_spline(
    evaluation: PoseSplineEvaluation,
    pose_sensor_position_in_body: Sequence[float],
    cog_offset_in_body: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert sensor-pose spline derivatives to CoG world kinematics."""

    sensor_offset = np.asarray(pose_sensor_position_in_body, dtype=float)
    cog_offset = np.asarray(cog_offset_in_body, dtype=float)
    if (
        sensor_offset.shape != (3,)
        or cog_offset.shape != (3,)
        or np.any(~np.isfinite(sensor_offset))
        or np.any(~np.isfinite(cog_offset))
    ):
        raise ValueError("sensor and CoG offsets must be finite 3-vectors")
    lever = sensor_offset - cog_offset
    rotation = evaluation.body_rotation
    omega = evaluation.body_angular_velocity
    alpha = evaluation.body_angular_acceleration
    rotational_velocity = np.cross(omega, lever)
    rotational_acceleration = np.cross(alpha, lever) + np.cross(
        omega, np.cross(omega, lever)
    )
    position = evaluation.sensor_position - np.einsum(
        "nij,j->ni", rotation, lever
    )
    velocity = evaluation.sensor_velocity_world - np.einsum(
        "nij,nj->ni", rotation, rotational_velocity
    )
    acceleration = evaluation.sensor_acceleration_world - np.einsum(
        "nij,nj->ni", rotation, rotational_acceleration
    )
    return position, velocity, acceleration



def _build_bag_data(
    specification: multi.BagSpecification,
    normalized_weight: float,
    flight: Any,
    initial_delay: float,
    settings: SplineSettings,
    arguments: argparse.Namespace,
    reference_parameters: VehicleParameters,
    geometry: GrapeGeometry,
) -> BagSplineData:
    direct = baseline.DirectShootingProblem(
        flight=flight,
        sample_step=arguments.sample_step,
        integration_step=arguments.integration_step,
        command_delay=initial_delay,
        prior_weight=0.0,
        reference_parameters=reference_parameters,
        geometry=geometry,
    )
    selection = select_pose_spline(
        time_axis=direct.output_time,
        sensor_position=direct.observations.sensor_position,
        sensor_orientation_xyzw=(
            direct.observations.sensor_orientation_xyzw
        ),
        body_to_pose_sensor_rotation=(
            direct.pose_body_to_sensor_rotation
        ),
        knot_spacing_candidates_seconds=(
            settings.knot_spacing_candidates_seconds
        ),
        rotational_metric=(
            reference_parameters.inertia / reference_parameters.mass
        ),
        fold_count=arguments.spline_cv_folds,
        validation_block_duration_seconds=(
            settings.cross_validation_block_seconds
        ),
        derivative_check_step_seconds=settings.collocation_step_seconds,
        maximum_acceleration_m_per_s2=(
            arguments.maximum_spline_acceleration
        ),
        maximum_angular_acceleration_rad_per_s2=(
            arguments.maximum_spline_angular_acceleration
        ),
    )
    boundary_exclusion_seconds = _spline_boundary_exclusion_seconds(
        selection,
        settings.boundary_exclusion_knot_spans_each_side,
    )
    estimation_start = (
        selection.spline.start_time + boundary_exclusion_seconds
    )
    estimation_end = (
        selection.spline.end_time - boundary_exclusion_seconds
    )
    if (
        estimation_end - estimation_start
        < 2.0 * settings.collocation_step_seconds
    ):
        raise ValueError(
            "spline boundary exclusion leaves too little "
            "parameter-estimation support"
        )
    collocation_time = _collocation_grid(
        estimation_start,
        estimation_end,
        settings.collocation_step_seconds,
    )
    collocation = selection.spline.evaluate(collocation_time)
    initial_gimbal = baseline._linear_interpolate(
        flight.gimbal_position.times,
        flight.gimbal_position.values,
        np.asarray((collocation_time[0],), dtype=float),
    )[0]
    return BagSplineData(
        specification=specification,
        normalized_weight=float(normalized_weight),
        flight=flight,
        direct_problem=direct,
        spline_selection=selection,
        collocation=collocation,
        rotor_history=QuinticSmoothZoh(
            flight.rotor_command.all_times,
            flight.rotor_command.all_values,
        ),
        gimbal_history=QuinticSmoothZoh(
            flight.gimbal_command.all_times,
            flight.gimbal_command.all_values,
        ),
        initial_gimbal=initial_gimbal,
        boundary_exclusion_knot_spans_each_side=(
            settings.boundary_exclusion_knot_spans_each_side
        ),
    )



class SplinePhysicalParameterization:
    """14-D local chart around an explicit external vehicle-model reference.

    The reference values define only the coordinate chart.  They are not a
    prior.  The Gaussian prior is evaluated separately in physical coordinates.
    """

    def __init__(self, reference: VehicleParameters) -> None:
        if not isinstance(reference, VehicleParameters):
            raise TypeError("reference must be VehicleParameters")
        second_moment = (
            0.5 * float(np.trace(reference.inertia)) * np.eye(3)
            - reference.inertia
        )
        second_moment = 0.5 * (second_moment + second_moment.T)
        if np.any(np.linalg.eigvalsh(second_moment) <= 0.0):
            raise ValueError(
                "reference inertia must satisfy strict triangle inequalities"
            )
        self.reference = reference
        self.reference_cholesky = np.linalg.cholesky(second_moment)

    def _cholesky(self, coordinate: np.ndarray) -> np.ndarray:
        value = np.asarray(coordinate, dtype=float)
        if value.shape != (PHYSICAL_DIMENSION,):
            raise ValueError("physical coordinate must be 14-D")
        cholesky = self.reference_cholesky.copy()
        cholesky[(0, 1, 2), (0, 1, 2)] *= np.exp(value[1:4])
        for local_index, (row, column) in enumerate(
            _CHOLESKY_OFF_DIAGONALS
        ):
            scale = math.sqrt(
                self.reference_cholesky[row, row]
                * self.reference_cholesky[column, column]
            )
            cholesky[row, column] += value[4 + local_index] * scale
        return cholesky

    def decode(
        self,
        physical_coordinate: Sequence[float],
        delay: float,
    ) -> Any:
        decoded, _jacobian = self.decode_with_jacobian(
            physical_coordinate,
            delay,
        )
        return decoded

    def decode_with_jacobian(
        self,
        physical_coordinate: Sequence[float],
        delay: float,
    ) -> tuple[Any, Any]:
        value = np.asarray(physical_coordinate, dtype=float)
        if (
            value.shape != (PHYSICAL_DIMENSION,)
            or np.any(~np.isfinite(value))
            or not np.isfinite(delay)
            or delay < 0.0
        ):
            raise ValueError("physical coordinate or delay is invalid")

        cholesky = self._cholesky(value)
        second_moment = cholesky @ cholesky.T
        inertia = (
            np.trace(second_moment) * np.eye(3)
            - second_moment
        )
        inertia = 0.5 * (inertia + inertia.T)
        principal = np.linalg.eigvalsh(inertia)
        triangle_margin = float(
            principal[0] + principal[1] - principal[2]
        )

        force_effectiveness = (
            self.reference.force_effectiveness
            * np.exp(value[10:14])
        )
        parameters = VehicleParameters(
            mass=self.reference.mass * math.exp(float(value[0])),
            inertia=inertia,
            cog_offset=self.reference.cog_offset + value[7:10],
            force_effectiveness=force_effectiveness,
            torque_effectiveness=self.reference.torque_effectiveness,
            linear_drag=self.reference.linear_drag,
            angular_drag=self.reference.angular_drag,
        )
        actuator_parameters = ActuatorParameters(
            thrust_time_constant=0.0,
            gimbal_time_constant=0.0,
            delay=0.0,
        )
        decoded = strict.analytic.DecodedSearchPoint(
            parameters=parameters,
            actuator_parameters=actuator_parameters,
            delay=float(delay),
            inertia_principal_moments=principal,
            inertia_triangle_margin=triangle_margin,
        )

        dimension = PHYSICAL_DIMENSION
        mass = np.zeros(dimension, dtype=float)
        mass[0] = parameters.mass

        inertia_jacobian = np.zeros(
            (3, 3, dimension),
            dtype=float,
        )

        def store_inertia_derivative(
            coordinate_index: int,
            cholesky_derivative: np.ndarray,
        ) -> None:
            second_moment_derivative = (
                cholesky_derivative @ cholesky.T
                + cholesky @ cholesky_derivative.T
            )
            inertia_jacobian[:, :, coordinate_index] = (
                np.trace(second_moment_derivative) * np.eye(3)
                - second_moment_derivative
            )

        for axis in range(3):
            derivative = np.zeros((3, 3), dtype=float)
            derivative[axis, axis] = cholesky[axis, axis]
            store_inertia_derivative(1 + axis, derivative)
        for local_index, (row, column) in enumerate(
            _CHOLESKY_OFF_DIAGONALS
        ):
            derivative = np.zeros((3, 3), dtype=float)
            derivative[row, column] = math.sqrt(
                self.reference_cholesky[row, row]
                * self.reference_cholesky[column, column]
            )
            store_inertia_derivative(4 + local_index, derivative)

        cog_position = np.zeros((3, dimension), dtype=float)
        cog_position[:, 7:10] = np.eye(3)

        effectiveness_jacobian = np.zeros(
            (4, dimension),
            dtype=float,
        )
        effectiveness_jacobian[:, 10:14] = np.diag(
            force_effectiveness
        )

        zero = np.zeros(dimension, dtype=float)
        jacobian = strict.analytic.DecodedSearchJacobian(
            mass=mass,
            inertia=inertia_jacobian,
            # Historical field name in the shared dynamics API.
            cog_offset=cog_position,
            force_effectiveness=effectiveness_jacobian,
            thrust_time_constant=zero.copy(),
            gimbal_time_constant=zero.copy(),
        )
        return decoded, jacobian


def _extend_parameter_jacobian(
    source: Any,
    dimension: int,
) -> Any:
    """Append zero columns, e.g. for the smooth lag coordinate."""

    source_dimension = int(
        np.asarray(source.mass).size
    )
    if (
        source_dimension != PHYSICAL_DIMENSION
        or dimension < source_dimension
    ):
        raise ValueError(
            "extended derivative dimension is invalid"
        )

    def extend_vector(value: np.ndarray) -> np.ndarray:
        source_value = np.asarray(
            value,
            dtype=float,
        )
        if source_value.shape != (source_dimension,):
            raise ValueError(
                "parameter derivative vector has inconsistent width"
            )
        result = np.zeros(
            dimension,
            dtype=float,
        )
        result[:source_dimension] = source_value
        return result

    def extend_matrix(value: np.ndarray) -> np.ndarray:
        source_value = np.asarray(
            value,
            dtype=float,
        )
        if (
            source_value.ndim != 2
            or source_value.shape[1]
            != source_dimension
        ):
            raise ValueError(
                "parameter derivative matrix has inconsistent width"
            )
        result = np.zeros(
            (source_value.shape[0], dimension),
            dtype=float,
        )
        result[:, :source_dimension] = source_value
        return result

    source_inertia = np.asarray(
        source.inertia,
        dtype=float,
    )
    if source_inertia.shape != (
        3,
        3,
        source_dimension,
    ):
        raise ValueError(
            "inertia derivative has inconsistent width"
        )
    inertia = np.zeros(
        (3, 3, dimension),
        dtype=float,
    )
    inertia[:, :, :source_dimension] = (
        source_inertia
    )
    return strict.analytic.DecodedSearchJacobian(
        mass=extend_vector(source.mass),
        inertia=inertia,
        cog_offset=extend_matrix(
            source.cog_offset
        ),
        force_effectiveness=extend_matrix(
            source.force_effectiveness
        ),
        thrust_time_constant=extend_vector(
            source.thrust_time_constant
        ),
        gimbal_time_constant=extend_vector(
            source.gimbal_time_constant
        ),
    )



class SplineDynamicsProblem:
    """Shared physical parameters against fixed bag-local pose splines."""

    def __init__(
        self,
        bags: Sequence[BagSplineData],
        reference_parameters: VehicleParameters,
        parameter_prior: GaussianPhysicalPrior,
    ) -> None:
        self.bags = tuple(bags)
        if not self.bags:
            raise ValueError("spline dynamics requires at least one bag")
        weights = np.asarray(
            [bag.normalized_weight for bag in self.bags],
            dtype=float,
        )
        if not np.isclose(
            np.sum(weights), 1.0, atol=1.0e-12, rtol=0.0
        ):
            raise ValueError(
                "normalized bag weights must sum to one"
            )
        if not isinstance(reference_parameters, VehicleParameters):
            raise TypeError(
                "reference_parameters must be VehicleParameters"
            )
        if not isinstance(parameter_prior, GaussianPhysicalPrior):
            raise TypeError(
                "parameter_prior must be GaussianPhysicalPrior"
            )
        self.reference_parameters = reference_parameters
        self.parameter_prior = parameter_prior
        self.parameterization = SplinePhysicalParameterization(
            reference_parameters
        )
        self.angular_factor = np.linalg.cholesky(
            reference_parameters.inertia / reference_parameters.mass
        ).T

    def _decode(
        self,
        physical_coordinate: np.ndarray,
        delay: float,
        dimension: int,
    ) -> tuple[Any, Any]:
        decoded, jacobian = self.parameterization.decode_with_jacobian(
            physical_coordinate,
            delay,
        )
        return decoded, _extend_parameter_jacobian(
            jacobian,
            dimension,
        )

    @staticmethod
    def _command(
        bag: BagSplineData,
        time_value: float,
        delay: float,
        smooth_mode: bool,
        width_fraction: float,
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        if smooth_mode:
            rotor = bag.rotor_history.evaluate(
                time_value, delay, width_fraction
            )
            gimbal = bag.gimbal_history.evaluate(
                time_value, delay, width_fraction
            )
            return (
                smooth._command(rotor.value, gimbal.value),
                rotor.delay_derivative,
                gimbal.delay_derivative,
            )
        rotor_value = bag.rotor_history.exact_zoh(time_value, delay)
        gimbal_value = bag.gimbal_history.exact_zoh(time_value, delay)
        return (
            smooth._command(rotor_value, gimbal_value),
            np.zeros(4, dtype=float),
            np.zeros(4, dtype=float),
        )

    def _actuator_series(
        self,
        bag: BagSplineData,
        decoded: Any,
        parameter_jacobian: Any,
        dimension: int,
        smooth_mode: bool,
        width_fraction: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        time_axis = bag.collocation_time
        delay = float(decoded.delay)
        initial_command, rotor_derivative, _gimbal_derivative = self._command(
            bag,
            float(time_axis[0]),
            delay,
            smooth_mode,
            width_fraction,
        )
        state = ActuatorState(
            thrust=initial_command.thrust,
            gimbal_angle=bag.initial_gimbal,
        )
        sensitivity = np.zeros((8, dimension), dtype=float)
        if smooth_mode:
            sensitivity[:4, DELAY_INDEX] = rotor_derivative
        thrust = np.empty((time_axis.size, 4), dtype=float)
        gimbal = np.empty((time_axis.size, 4), dtype=float)
        state_jacobian = np.empty((time_axis.size, 8, dimension), dtype=float)
        for index, sample_time in enumerate(time_axis):
            thrust[index] = state.thrust
            gimbal[index] = state.gimbal_angle
            state_jacobian[index] = sensitivity
            if index == time_axis.size - 1:
                break
            dt = float(time_axis[index + 1] - sample_time)
            midpoint_time = float(sample_time + 0.5 * dt)
            command, thrust_derivative, gimbal_derivative = self._command(
                bag,
                midpoint_time,
                delay,
                smooth_mode,
                width_fraction,
            )
            command_sensitivity = np.zeros((8, dimension), dtype=float)
            if smooth_mode:
                command_sensitivity[:4, DELAY_INDEX] = thrust_derivative
                command_sensitivity[4:, DELAY_INDEX] = gimbal_derivative
            state, sensitivity = strict._actuator_step_with_sensitivity(
                state,
                sensitivity,
                command,
                decoded,
                parameter_jacobian,
                0.5 * dt,
                command_sensitivity,
            )
            state, sensitivity = strict._actuator_step_with_sensitivity(
                state,
                sensitivity,
                command,
                decoded,
                parameter_jacobian,
                0.5 * dt,
                command_sensitivity,
            )
        return thrust, gimbal, state_jacobian

    def _evaluate_bag(
        self,
        bag: BagSplineData,
        decoded: Any,
        parameter_jacobian: Any,
        dimension: int,
        smooth_mode: bool,
        width_fraction: float,
    ) -> BagDynamicsEvaluation:
        parameters = decoded.parameters
        collocation = bag.collocation
        count = collocation.time.size
        thrust, gimbal, actuator_jacobian = self._actuator_series(
            bag,
            decoded,
            parameter_jacobian,
            dimension,
            smooth_mode,
            width_fraction,
        )
        acceleration_residual = np.empty((count, 6), dtype=float)
        acceleration_jacobian = np.empty((count, 6, dimension), dtype=float)
        required_wrench = np.empty((count, 6), dtype=float)
        modeled_wrench = np.empty((count, 6), dtype=float)
        residual_wrench = np.empty((count, 6), dtype=float)
        residual_wrench_jacobian = np.empty(
            (count, 6, dimension), dtype=float
        )
        cog_position = np.empty((count, 3), dtype=float)
        cog_velocity = np.empty((count, 3), dtype=float)
        cog_acceleration = np.empty((count, 3), dtype=float)
        zero_rotation_sensitivity = np.zeros((3, dimension), dtype=float)
        zero_omega_sensitivity = np.zeros((3, dimension), dtype=float)
        gravity_world = np.asarray((0.0, 0.0, -GRAVITY), dtype=float)
        cog_position[:], cog_velocity[:], cog_acceleration[:] = (
            cog_kinematics_from_pose_spline(
                collocation,
                bag.direct_problem.pose_sensor_position,
                parameters.cog_offset,
            )
        )
        for index in range(count):
            rotation = collocation.body_rotation[index]
            omega = collocation.body_angular_velocity[index]
            alpha = collocation.body_angular_acceleration[index]
            lever = (
                bag.direct_problem.pose_sensor_position
                - parameters.cog_offset
            )
            angular_kinematics = (
                skew(alpha) + skew(omega) @ skew(omega)
            )
            velocity = cog_velocity[index]
            acceleration = cog_acceleration[index]
            velocity_sensitivity = (
                rotation
                @ skew(omega)
                @ parameter_jacobian.cog_offset
            )
            acceleration_body = rotation.T @ (
                acceleration - gravity_world
            )
            acceleration_body_sensitivity = (
                angular_kinematics @ parameter_jacobian.cog_offset
            )
            actuators = ActuatorState(
                thrust=thrust[index],
                gimbal_angle=gimbal[index],
            )
            wrench, wrench_jacobian = strict._body_wrench_with_sensitivity(
                bag.direct_problem,
                decoded,
                parameter_jacobian,
                rotation,
                zero_rotation_sensitivity,
                velocity,
                velocity_sensitivity,
                omega,
                zero_omega_sensitivity,
                actuators,
                actuator_jacobian[index],
            )
            force = wrench[:3]
            force_jacobian = wrench_jacobian[:3]
            predicted_acceleration = force / parameters.mass
            predicted_acceleration_jacobian = (
                force_jacobian / parameters.mass
                - np.outer(force, parameter_jacobian.mass)
                / parameters.mass**2
            )
            inertia_omega = parameters.inertia @ omega
            angular_rhs = wrench[3:] - np.cross(omega, inertia_omega)
            predicted_alpha = np.linalg.solve(
                parameters.inertia, angular_rhs
            )
            inertia_omega_jacobian = np.einsum(
                "ijk,j->ik", parameter_jacobian.inertia, omega
            )
            angular_rhs_jacobian = (
                wrench_jacobian[3:]
                - skew(omega) @ inertia_omega_jacobian
            )
            inertia_alpha_jacobian = np.einsum(
                "ijk,j->ik", parameter_jacobian.inertia, predicted_alpha
            )
            predicted_alpha_jacobian = np.linalg.solve(
                parameters.inertia,
                angular_rhs_jacobian - inertia_alpha_jacobian,
            )
            linear_error = acceleration_body - predicted_acceleration
            angular_error = alpha - predicted_alpha
            acceleration_residual[index, :3] = linear_error
            acceleration_residual[index, 3:] = (
                self.angular_factor @ angular_error
            )
            acceleration_jacobian[index, :3] = (
                acceleration_body_sensitivity
                - predicted_acceleration_jacobian
            )
            acceleration_jacobian[index, 3:] = (
                -self.angular_factor @ predicted_alpha_jacobian
            )
            required_force = parameters.mass * acceleration_body
            required_force_jacobian = (
                np.outer(acceleration_body, parameter_jacobian.mass)
                + parameters.mass * acceleration_body_sensitivity
            )
            required_torque = (
                parameters.inertia @ alpha
                + np.cross(omega, inertia_omega)
            )
            required_torque_jacobian = (
                np.einsum("ijk,j->ik", parameter_jacobian.inertia, alpha)
                + skew(omega) @ inertia_omega_jacobian
            )
            required_wrench[index, :3] = required_force
            required_wrench[index, 3:] = required_torque
            modeled_wrench[index] = wrench
            residual_wrench[index] = required_wrench[index] - wrench
            residual_wrench_jacobian[index, :3] = (
                required_force_jacobian - force_jacobian
            )
            residual_wrench_jacobian[index, 3:] = (
                required_torque_jacobian - wrench_jacobian[3:]
            )
        arrays = (
            acceleration_residual,
            acceleration_jacobian,
            required_wrench,
            modeled_wrench,
            residual_wrench,
            residual_wrench_jacobian,
            cog_position,
            cog_velocity,
            cog_acceleration,
            thrust,
            gimbal,
        )
        if any(np.any(~np.isfinite(value)) for value in arrays):
            raise FloatingPointError("spline dynamics evaluation is non-finite")
        return BagDynamicsEvaluation(
            acceleration_residual=acceleration_residual,
            acceleration_jacobian=acceleration_jacobian,
            required_body_wrench=required_wrench,
            modeled_body_wrench=modeled_wrench,
            residual_body_wrench=residual_wrench,
            residual_wrench_jacobian=residual_wrench_jacobian,
            cog_position=cog_position,
            cog_velocity_world=cog_velocity,
            cog_acceleration_world=cog_acceleration,
            actuator_thrust=thrust,
            actuator_gimbal=gimbal,
        )

    def evaluate_smooth(
        self,
        coordinate: Sequence[float],
        width_fraction: float,
    ) -> JointDynamicsEvaluation:
        value = np.asarray(coordinate, dtype=float)
        if value.shape != (GLOBAL_DIMENSION,) or np.any(~np.isfinite(value)):
            raise ValueError("smooth spline-dynamics coordinate must be 15-D")
        return self._evaluate_joint(
            physical_coordinate=value[: PHYSICAL_DIMENSION],
            delay=float(value[DELAY_INDEX]),
            dimension=GLOBAL_DIMENSION,
            smooth_mode=True,
            width_fraction=float(width_fraction),
        )

    def evaluate_strict(
        self,
        physical_coordinate: Sequence[float],
        delay: float,
    ) -> JointDynamicsEvaluation:
        physical = np.asarray(physical_coordinate, dtype=float)
        if (
            physical.shape != (PHYSICAL_DIMENSION,)
            or np.any(~np.isfinite(physical))
            or not np.isfinite(delay)
            or delay < 0.0
        ):
            raise ValueError("strict spline-dynamics coordinate is invalid")
        return self._evaluate_joint(
            physical_coordinate=physical,
            delay=float(delay),
            dimension=PHYSICAL_DIMENSION,
            smooth_mode=False,
            width_fraction=1.0,
        )

    def _evaluate_joint(
        self,
        *,
        physical_coordinate: np.ndarray,
        delay: float,
        dimension: int,
        smooth_mode: bool,
        width_fraction: float,
    ) -> JointDynamicsEvaluation:
        decoded, parameter_jacobian = self._decode(
            physical_coordinate, delay, dimension
        )
        residual_blocks = []
        jacobian_blocks = []
        bag_evaluations = []
        data_loss = 0.0
        for bag in self.bags:
            evaluation = self._evaluate_bag(
                bag,
                decoded,
                parameter_jacobian,
                dimension,
                smooth_mode,
                width_fraction,
            )
            bag_evaluations.append(evaluation)
            root_scale = math.sqrt(
                bag.normalized_weight / bag.collocation_time.size
            )
            residual_blocks.append(
                root_scale * evaluation.acceleration_residual.ravel()
            )
            jacobian_blocks.append(
                root_scale
                * evaluation.acceleration_jacobian.reshape(-1, dimension)
            )
            data_loss += bag.normalized_weight * evaluation.dynamics_loss
        prior_value = physical_parameter_vector(
            decoded.parameters
        )
        prior_jacobian = physical_parameter_jacobian(
            parameter_jacobian
        )
        prior_residual = (
            prior_value - self.parameter_prior.mean
        ) / self.parameter_prior.std
        prior_jacobian = (
            prior_jacobian / self.parameter_prior.std[:, None]
        )
        residual_blocks.append(prior_residual)
        jacobian_blocks.append(prior_jacobian)
        residual = np.concatenate(residual_blocks)
        jacobian = np.vstack(jacobian_blocks)
        prior_cost = 0.5 * float(
            prior_residual @ prior_residual
        )
        if np.any(~np.isfinite(residual)) or np.any(~np.isfinite(jacobian)):
            raise FloatingPointError("joint spline dynamics is non-finite")
        return JointDynamicsEvaluation(
            residual=residual,
            jacobian=jacobian,
            bag_evaluations=tuple(bag_evaluations),
            decoded=decoded,
            physical_coordinate=physical_coordinate.copy(),
            delay_seconds=delay,
            data_loss=float(data_loss),
            prior_cost=prior_cost,
        )


class _CachedObjective:
    def __init__(self, evaluator: Any) -> None:
        self.evaluator = evaluator
        self.coordinate: Optional[np.ndarray] = None
        self.evaluation: Optional[JointDynamicsEvaluation] = None

    def _get(self, coordinate: np.ndarray) -> JointDynamicsEvaluation:
        value = np.asarray(coordinate, dtype=float)
        if self.coordinate is None or not np.array_equal(value, self.coordinate):
            self.coordinate = value.copy()
            self.evaluation = self.evaluator(value)
        if self.evaluation is None:
            raise RuntimeError("cached objective has no evaluation")
        return self.evaluation

    def residual(self, coordinate: np.ndarray) -> np.ndarray:
        return self._get(coordinate).residual

    def jacobian(self, coordinate: np.ndarray) -> np.ndarray:
        return self._get(coordinate).jacobian


def _optimizer_payload(result: Any) -> dict[str, Any]:
    return {
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "njev": None if result.njev is None else int(result.njev),
        "status": int(result.status),
        "success": bool(result.success),
        "message": str(result.message),
    }



def _physical_bounds(
    initial_coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(initial_coordinate, dtype=float)
    if value.shape != (PHYSICAL_DIMENSION,):
        raise ValueError("initial physical coordinate must be 14-D")
    return (
        np.full(PHYSICAL_DIMENSION, -np.inf),
        np.full(PHYSICAL_DIMENSION, np.inf),
    )


def _solve_smooth(
    problem: SplineDynamicsProblem,
    initial: np.ndarray,
    width_fraction: float,
    lower: np.ndarray,
    upper: np.ndarray,
    arguments: argparse.Namespace,
) -> tuple[np.ndarray, JointDynamicsEvaluation, Mapping[str, Any]]:
    objective = _CachedObjective(
        lambda value: problem.evaluate_smooth(value, width_fraction)
    )
    result = least_squares(
        objective.residual,
        initial,
        jac=objective.jacobian,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        loss="linear",
        ftol=arguments.ftol,
        xtol=arguments.xtol,
        gtol=arguments.gtol,
        max_nfev=arguments.smooth_max_nfev,
        verbose=0,
    )
    return result.x, problem.evaluate_smooth(result.x, width_fraction), _optimizer_payload(result)


def _solve_strict(
    problem: SplineDynamicsProblem,
    initial: np.ndarray,
    delay: float,
    lower: np.ndarray,
    upper: np.ndarray,
    arguments: argparse.Namespace,
) -> DynamicsSolution:
    objective = _CachedObjective(
        lambda value: problem.evaluate_strict(value, delay)
    )
    result = least_squares(
        objective.residual,
        initial,
        jac=objective.jacobian,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        loss="linear",
        ftol=arguments.ftol,
        xtol=arguments.xtol,
        gtol=arguments.gtol,
        max_nfev=arguments.strict_max_nfev,
        verbose=0,
    )
    evaluation = problem.evaluate_strict(result.x, delay)
    return DynamicsSolution(
        physical_coordinate=result.x.copy(),
        delay_seconds=float(delay),
        evaluation=evaluation,
        optimizer=_optimizer_payload(result),
    )


def _observations_at_times(
    observations: baseline.Observations,
    query_time: Sequence[float],
) -> baseline.Observations:
    time_value = np.asarray(query_time, dtype=float)
    if (
        time_value.ndim != 1
        or time_value.size < 2
        or np.any(~np.isfinite(time_value))
        or np.any(np.diff(time_value) <= 0.0)
    ):
        raise ValueError("observation query times are invalid")

    position, orientation, valid = baseline.interpolate_observed_pose(
        observations.time,
        observations.sensor_position,
        observations.sensor_orientation_xyzw,
        time_value,
    )
    if not np.all(valid):
        raise ValueError("observed pose does not cover reconstruction time")

    return baseline.Observations(
        time=time_value.copy(),
        sensor_position=position,
        sensor_orientation_xyzw=orientation,
        sensor_velocity_world=baseline._linear_interpolate(
            observations.time,
            observations.sensor_velocity_world,
            time_value,
        ),
        angular_velocity_sensor=baseline._linear_interpolate(
            observations.time,
            observations.angular_velocity_sensor,
            time_value,
        ),
        specific_force_sensor=baseline._linear_interpolate(
            observations.time,
            observations.specific_force_sensor,
            time_value,
        ),
    )


def _simulation_at_times(
    simulation: baseline.Simulation,
    query_time: Sequence[float],
) -> baseline.Simulation:
    time_value = np.asarray(query_time, dtype=float)
    position, orientation, valid = baseline.interpolate_observed_pose(
        simulation.time,
        simulation.sensor_position,
        simulation.sensor_orientation_xyzw,
        time_value,
    )
    if not np.all(valid):
        raise ValueError("simulation does not cover requested times")

    def interp(values: np.ndarray) -> np.ndarray:
        return baseline._linear_interpolate(
            simulation.time,
            np.asarray(values, dtype=float),
            time_value,
        )

    return baseline.Simulation(
        time=time_value.copy(),
        sensor_position=position,
        sensor_orientation_xyzw=orientation,
        sensor_velocity_world=interp(simulation.sensor_velocity_world),
        angular_velocity_sensor=interp(simulation.angular_velocity_sensor),
        specific_force_sensor=interp(simulation.specific_force_sensor),
        cog_position=interp(simulation.cog_position),
        cog_velocity_world=interp(simulation.cog_velocity_world),
        actuator_thrust=interp(simulation.actuator_thrust),
        actuator_gimbal=interp(simulation.actuator_gimbal),
    )


def forward_rollout(
    bag: BagSplineData,
    physical_coordinate: Sequence[float],
    delay: float,
    reference_parameters: VehicleParameters,
    external_body_wrench: Optional[BodyWrenchHistory] = None,
    gimbal_delay: Optional[float] = None,
) -> baseline.Simulation:
    """Strict-ZOH rollout, optionally driven by an external body wrench."""

    if external_body_wrench is not None and not isinstance(
        external_body_wrench, BodyWrenchHistory
    ):
        raise TypeError("external_body_wrench must be BodyWrenchHistory")
    rotor_delay = float(delay)
    gimbal_delay_value = rotor_delay if gimbal_delay is None else float(gimbal_delay)

    parameterization = SplinePhysicalParameterization(
        reference_parameters
    )
    decoded = parameterization.decode(
        physical_coordinate,
        rotor_delay,
    )
    parameters = decoded.parameters
    direct = bag.direct_problem
    initial_spline = bag.spline_selection.spline.evaluate(
        np.asarray((direct.internal_time[0],), dtype=float)
    )
    rotation = initial_spline.body_rotation[0]
    omega = initial_spline.body_angular_velocity[0]
    lever_pose = direct.pose_sensor_position - parameters.cog_offset
    rigid = RigidBodyState(
        position=(
            initial_spline.sensor_position[0] - rotation @ lever_pose
        ),
        orientation_xyzw=matrix_to_quaternion(rotation),
        linear_velocity=(
            initial_spline.sensor_velocity_world[0]
            - rotation @ np.cross(omega, lever_pose)
        ),
        angular_velocity=omega,
    )
    actuator_parameters = ActuatorParameters(
        thrust_time_constant=0.0,
        gimbal_time_constant=0.0,
        delay=0.0,
    )
    actuators = ActuatorState(
        thrust=bag.rotor_history.exact_zoh(
            float(direct.internal_time[0]), rotor_delay
        ),
        gimbal_angle=bag.gimbal_history.exact_zoh(
            float(direct.internal_time[0]), gimbal_delay_value
        ),
    )
    plant = FullSixDofPlant(
        parameters,
        direct.geometry,
        model_discrepancy_wrench=external_body_wrench,
    )
    output_count = direct.output_time.size
    arrays = {
        "sensor_position": np.empty((output_count, 3)),
        "sensor_orientation_xyzw": np.empty((output_count, 4)),
        "sensor_velocity_world": np.empty((output_count, 3)),
        "angular_velocity_sensor": np.empty((output_count, 3)),
        "specific_force_sensor": np.empty((output_count, 3)),
        "cog_position": np.empty((output_count, 3)),
        "cog_velocity_world": np.empty((output_count, 3)),
        "actuator_thrust": np.empty((output_count, 4)),
        "actuator_gimbal": np.empty((output_count, 4)),
    }

    def store(output_index: int, simulation_time: float) -> None:
        body_rotation = quaternion_to_matrix(rigid.orientation_xyzw)
        pose_lever = direct.pose_sensor_position - parameters.cog_offset
        velocity_lever = (
            direct.velocity_sensor_position - parameters.cog_offset
        )
        imu_lever = direct.imu_sensor_position - parameters.cog_offset
        wrench = plant.total_body_wrench(simulation_time, rigid, actuators)
        angular_acceleration = np.linalg.solve(
            parameters.inertia,
            wrench[3:]
            - np.cross(
                rigid.angular_velocity,
                parameters.inertia @ rigid.angular_velocity,
            ),
        )
        specific_force_body = (
            wrench[:3] / parameters.mass
            + np.cross(angular_acceleration, imu_lever)
            + np.cross(
                rigid.angular_velocity,
                np.cross(rigid.angular_velocity, imu_lever),
            )
        )
        arrays["sensor_position"][output_index] = (
            rigid.position + body_rotation @ pose_lever
        )
        arrays["sensor_orientation_xyzw"][output_index] = (
            matrix_to_quaternion(
                body_rotation @ direct.pose_body_to_sensor_rotation
            )
        )
        arrays["sensor_velocity_world"][output_index] = (
            rigid.linear_velocity
            + body_rotation
            @ np.cross(rigid.angular_velocity, velocity_lever)
        )
        arrays["angular_velocity_sensor"][output_index] = (
            direct.body_to_imu_rotation @ rigid.angular_velocity
            + direct.gyro_bias
        )
        arrays["specific_force_sensor"][output_index] = (
            direct.body_to_imu_rotation @ specific_force_body
            + direct.accelerometer_bias
        )
        arrays["cog_position"][output_index] = rigid.position
        arrays["cog_velocity_world"][output_index] = rigid.linear_velocity
        arrays["actuator_thrust"][output_index] = actuators.thrust
        arrays["actuator_gimbal"][output_index] = actuators.gimbal_angle

    store(0, float(direct.internal_time[0]))
    output_index = 1
    for step_index in range(direct.internal_time.size - 1):
        start = float(direct.internal_time[step_index])
        dt = direct.integration_step
        midpoint = start + 0.5 * dt
        command = smooth._command(
            bag.rotor_history.exact_zoh(midpoint, rotor_delay),
            bag.gimbal_history.exact_zoh(midpoint, gimbal_delay_value),
        )
        midpoint_actuators = advance_actuators(
            actuators, command, actuator_parameters, 0.5 * dt
        )
        rigid = plant.step(start, rigid, midpoint_actuators, dt)
        actuators = advance_actuators(
            midpoint_actuators, command, actuator_parameters, 0.5 * dt
        )
        if (step_index + 1) % direct.output_stride == 0:
            store(output_index, start + dt)
            output_index += 1
    if output_index != output_count:
        raise RuntimeError("forward rollout grids disagree")
    return baseline.Simulation(time=direct.output_time, **arrays)


@dataclass(frozen=True)
class WrenchReplayEvaluation:
    residual: np.ndarray
    jacobian: np.ndarray
    simulation: baseline.Simulation
    knot_time: np.ndarray
    coefficients: np.ndarray


class WrenchReplayProblem:
    """Fit piecewise-linear external-wrench knot values to the pose spline."""

    def __init__(
        self,
        bag: BagSplineData,
        physical_coordinate: Sequence[float],
        delay: float,
        dynamics_evaluation: BagDynamicsEvaluation,
        reference_parameters: VehicleParameters,
        gimbal_delay: Optional[float] = None,
    ) -> None:
        self.bag = bag
        self.physical_coordinate = np.asarray(
            physical_coordinate, dtype=float
        ).copy()
        self.delay = float(delay)
        self.gimbal_delay = self.delay if gimbal_delay is None else float(gimbal_delay)
        self.dynamics_evaluation = dynamics_evaluation
        if (
            self.physical_coordinate.shape
            != (PHYSICAL_DIMENSION,)
            or np.any(~np.isfinite(self.physical_coordinate))
            or not np.isfinite(self.delay)
            or self.delay < 0.0
            or not np.isfinite(self.gimbal_delay)
            or self.gimbal_delay < 0.0
        ):
            raise ValueError("wrench-replay parameter input is invalid")

        collocation_time = np.asarray(bag.collocation_time, dtype=float)
        observed_time = np.asarray(
            bag.direct_problem.observations.time, dtype=float
        )
        mask = (
            (observed_time >= collocation_time[0] - 1.0e-10)
            & (observed_time <= collocation_time[-1] + 1.0e-10)
        )
        knot_time = observed_time[mask]
        if knot_time.size < 2:
            raise ValueError(
                "wrench reconstruction support has too few observations"
            )
        if not np.isclose(
            knot_time[0], collocation_time[0], atol=1.0e-10, rtol=0.0
        ):
            knot_time = np.concatenate(
                (np.asarray((collocation_time[0],)), knot_time)
            )
        if not np.isclose(
            knot_time[-1], collocation_time[-1], atol=1.0e-10, rtol=0.0
        ):
            knot_time = np.concatenate(
                (knot_time, np.asarray((collocation_time[-1],)))
            )
        self.knot_time = np.unique(knot_time)
        self.integration_time = np.unique(
            np.concatenate((collocation_time, self.knot_time))
        )
        self.knot_integration_indices = np.searchsorted(
            self.integration_time, self.knot_time
        )
        if not np.allclose(
            self.integration_time[self.knot_integration_indices],
            self.knot_time,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise RuntimeError("wrench knots are missing from integration grid")
        self.output_lookup = {
            int(integration_index): output_index
            for output_index, integration_index in enumerate(
                self.knot_integration_indices
            )
        }

        raw_wrench = np.asarray(
            dynamics_evaluation.residual_body_wrench, dtype=float
        )
        self.initial_coefficients = np.column_stack(
            [
                np.interp(
                    self.knot_time,
                    collocation_time,
                    raw_wrench[:, component],
                )
                for component in range(6)
            ]
        )
        self.dimension = int(6 * self.knot_time.size)

        parameterization = SplinePhysicalParameterization(
            reference_parameters
        )
        self.decoded = parameterization.decode(
            self.physical_coordinate,
            self.delay,
        )
        self.parameters = self.decoded.parameters
        self.actuator_parameters = self.decoded.actuator_parameters
        self.direct = bag.direct_problem
        self.zero_parameter_jacobian = (
            strict.analytic.DecodedSearchJacobian(
                mass=np.zeros(self.dimension, dtype=float),
                inertia=np.zeros(
                    (3, 3, self.dimension), dtype=float
                ),
                cog_offset=np.zeros(
                    (3, self.dimension), dtype=float
                ),
                force_effectiveness=np.zeros(
                    (4, self.dimension), dtype=float
                ),
                thrust_time_constant=np.zeros(
                    self.dimension, dtype=float
                ),
                gimbal_time_constant=np.zeros(
                    self.dimension, dtype=float
                ),
            )
        )
        self.zero_actuator_sensitivity = np.zeros(
            (8, self.dimension), dtype=float
        )
        self.inverse_inertia = np.linalg.solve(
            self.parameters.inertia, np.eye(3)
        )
        self.pose_factor = strict.inertia_radius_se3_factor(
            reference_parameters
        )
        self.target_spline = bag.spline_selection.spline.evaluate(
            self.knot_time
        )
        self.target_sensor_rotation = (
            bag.spline_selection.spline.sensor_rotation(self.knot_time)
        )
        _, target_cog_velocity, _ = cog_kinematics_from_pose_spline(
            self.target_spline,
            self.direct.pose_sensor_position,
            self.parameters.cog_offset,
        )
        self.target_terminal_cog_velocity = (
            target_cog_velocity[-1].copy()
        )
        self.target_terminal_angular_velocity = (
            self.target_spline.body_angular_velocity[-1].copy()
        )

    def _wrench_and_weights(
        self,
        coefficients: np.ndarray,
        time_value: float,
    ) -> tuple[np.ndarray, tuple[tuple[int, float], ...]]:
        query = float(time_value)
        if query <= self.knot_time[0]:
            return coefficients[0], ((0, 1.0),)
        if query >= self.knot_time[-1]:
            last = self.knot_time.size - 1
            return coefficients[last], ((last, 1.0),)

        right = int(np.searchsorted(
            self.knot_time, query, side="right"
        ))
        left = right - 1
        span = float(self.knot_time[right] - self.knot_time[left])
        fraction = (query - self.knot_time[left]) / span
        return (
            (1.0 - fraction) * coefficients[left]
            + fraction * coefficients[right],
            (
                (left, 1.0 - fraction),
                (right, fraction),
            ),
        )

    def _derivative_with_sensitivity(
        self,
        time_value: float,
        state_vector: np.ndarray,
        state_sensitivity: np.ndarray,
        actuators: ActuatorState,
        coefficients: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        derivative, derivative_sensitivity = (
            strict._rigid_derivative_with_sensitivity(
                self.direct,
                self.decoded,
                self.zero_parameter_jacobian,
                state_vector,
                state_sensitivity,
                actuators,
                self.zero_actuator_sensitivity,
            )
        )

        quaternion, normalization = (
            strict.analytic._normalise_quaternion_with_jacobian(
                state_vector[3:7]
            )
        )
        quaternion_sensitivity = (
            normalization @ state_sensitivity[3:7]
        )
        tangent = (
            strict.analytic._quaternion_right_tangent_matrix(
                quaternion
            )
        )
        rotation_right_sensitivity = (
            2.0 * tangent.T @ quaternion_sensitivity
        )
        rotation = quaternion_to_matrix(quaternion)

        wrench, active_weights = self._wrench_and_weights(
            coefficients, time_value
        )
        force_per_mass = wrench[:3] / self.parameters.mass
        derivative[7:10] += rotation @ force_per_mass
        derivative[10:13] += (
            self.inverse_inertia @ wrench[3:]
        )

        derivative_sensitivity[7:10] += (
            -rotation
            @ skew(force_per_mass)
            @ rotation_right_sensitivity
        )
        for knot_index, weight in active_weights:
            start = 6 * knot_index
            derivative_sensitivity[
                7:10, start : start + 3
            ] += weight * rotation / self.parameters.mass
            derivative_sensitivity[
                10:13, start + 3 : start + 6
            ] += weight * self.inverse_inertia

        return derivative, derivative_sensitivity

    def _step_with_sensitivity(
        self,
        time_value: float,
        state: RigidBodyState,
        state_sensitivity: np.ndarray,
        actuators: ActuatorState,
        time_step: float,
        coefficients: np.ndarray,
    ) -> tuple[RigidBodyState, np.ndarray]:
        vector = state.as_vector()
        sensitivity = np.asarray(
            state_sensitivity, dtype=float
        )
        dt = float(time_step)

        k1, j1 = self._derivative_with_sensitivity(
            time_value,
            vector,
            sensitivity,
            actuators,
            coefficients,
        )
        k2, j2 = self._derivative_with_sensitivity(
            time_value + 0.5 * dt,
            vector + 0.5 * dt * k1,
            sensitivity + 0.5 * dt * j1,
            actuators,
            coefficients,
        )
        k3, j3 = self._derivative_with_sensitivity(
            time_value + 0.5 * dt,
            vector + 0.5 * dt * k2,
            sensitivity + 0.5 * dt * j2,
            actuators,
            coefficients,
        )
        k4, j4 = self._derivative_with_sensitivity(
            time_value + dt,
            vector + dt * k3,
            sensitivity + dt * j3,
            actuators,
            coefficients,
        )

        next_vector = vector + dt / 6.0 * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
        next_sensitivity = sensitivity + dt / 6.0 * (
            j1 + 2.0 * j2 + 2.0 * j3 + j4
        )
        next_vector[3:7], normalization = (
            strict.analytic._normalise_quaternion_with_jacobian(
                next_vector[3:7]
            )
        )
        next_sensitivity[3:7] = (
            normalization @ next_sensitivity[3:7]
        )
        return (
            RigidBodyState.from_vector(next_vector),
            next_sensitivity,
        )

    def evaluate(
        self,
        correction_coordinate: Sequence[float],
    ) -> WrenchReplayEvaluation:
        correction = np.asarray(
            correction_coordinate, dtype=float
        )
        if (
            correction.shape != (self.dimension,)
            or np.any(~np.isfinite(correction))
        ):
            raise ValueError(
                "external-wrench correction has the wrong shape"
            )
        coefficients = (
            self.initial_coefficients
            + correction.reshape(-1, 6)
        )
        history = BodyWrenchHistory(
            self.knot_time, coefficients
        )
        plant = FullSixDofPlant(
            self.parameters,
            self.direct.geometry,
            model_discrepancy_wrench=history,
        )

        start_time = float(self.integration_time[0])
        initial_spline = (
            self.bag.spline_selection.spline.evaluate(
                np.asarray((start_time,), dtype=float)
            )
        )
        rotation = initial_spline.body_rotation[0]
        omega = initial_spline.body_angular_velocity[0]
        pose_lever = (
            self.direct.pose_sensor_position
            - self.parameters.cog_offset
        )
        rigid = RigidBodyState(
            position=(
                initial_spline.sensor_position[0]
                - rotation @ pose_lever
            ),
            orientation_xyzw=matrix_to_quaternion(rotation),
            linear_velocity=(
                initial_spline.sensor_velocity_world[0]
                - rotation @ np.cross(omega, pose_lever)
            ),
            angular_velocity=omega,
        )
        actuators = ActuatorState(
            thrust=np.asarray(
                self.dynamics_evaluation.actuator_thrust[0],
                dtype=float,
            ).copy(),
            gimbal_angle=np.asarray(
                self.dynamics_evaluation.actuator_gimbal[0],
                dtype=float,
            ).copy(),
        )
        rigid_sensitivity = np.zeros(
            (13, self.dimension), dtype=float
        )

        output_count = self.knot_time.size
        arrays = {
            "sensor_position": np.empty((output_count, 3)),
            "sensor_orientation_xyzw": np.empty(
                (output_count, 4)
            ),
            "sensor_velocity_world": np.empty(
                (output_count, 3)
            ),
            "angular_velocity_sensor": np.empty(
                (output_count, 3)
            ),
            "specific_force_sensor": np.empty(
                (output_count, 3)
            ),
            "cog_position": np.empty((output_count, 3)),
            "cog_velocity_world": np.empty(
                (output_count, 3)
            ),
            "actuator_thrust": np.empty((output_count, 4)),
            "actuator_gimbal": np.empty((output_count, 4)),
        }
        residual = np.empty((output_count, 6), dtype=float)
        jacobian = np.empty(
            (output_count, 6, self.dimension), dtype=float
        )

        def store(
            output_index: int,
            simulation_time: float,
        ) -> None:
            quaternion = rigid.orientation_xyzw
            body_rotation = quaternion_to_matrix(quaternion)
            tangent = (
                strict.analytic._quaternion_right_tangent_matrix(
                    quaternion
                )
            )
            rotation_right_sensitivity = (
                2.0 * tangent.T @ rigid_sensitivity[3:7]
            )
            lever = (
                self.direct.pose_sensor_position
                - self.parameters.cog_offset
            )
            sensor_position = (
                rigid.position + body_rotation @ lever
            )
            position_jacobian = (
                rigid_sensitivity[:3]
                - body_rotation
                @ skew(lever)
                @ rotation_right_sensitivity
            )
            sensor_rotation = (
                body_rotation
                @ self.direct.pose_body_to_sensor_rotation
            )
            sensor_rotation_right_sensitivity = (
                self.direct.pose_body_to_sensor_rotation.T
                @ rotation_right_sensitivity
            )
            pose_residual, pose_jacobian = (
                strict._se3_log_error_with_jacobian(
                    self.target_spline.sensor_position[
                        output_index
                    ],
                    self.target_sensor_rotation[output_index],
                    sensor_position,
                    sensor_rotation,
                    position_jacobian,
                    sensor_rotation_right_sensitivity,
                )
            )
            residual[output_index] = (
                self.pose_factor @ pose_residual
            )
            jacobian[output_index] = (
                self.pose_factor @ pose_jacobian
            )

            velocity_lever = (
                self.direct.velocity_sensor_position
                - self.parameters.cog_offset
            )
            imu_lever = (
                self.direct.imu_sensor_position
                - self.parameters.cog_offset
            )
            total_wrench = plant.total_body_wrench(
                simulation_time, rigid, actuators
            )
            angular_acceleration = self.inverse_inertia @ (
                total_wrench[3:]
                - np.cross(
                    rigid.angular_velocity,
                    self.parameters.inertia
                    @ rigid.angular_velocity,
                )
            )
            specific_force_body = (
                total_wrench[:3] / self.parameters.mass
                + np.cross(angular_acceleration, imu_lever)
                + np.cross(
                    rigid.angular_velocity,
                    np.cross(
                        rigid.angular_velocity, imu_lever
                    ),
                )
            )

            arrays["sensor_position"][output_index] = (
                sensor_position
            )
            arrays["sensor_orientation_xyzw"][output_index] = (
                matrix_to_quaternion(sensor_rotation)
            )
            arrays["sensor_velocity_world"][output_index] = (
                rigid.linear_velocity
                + body_rotation
                @ np.cross(
                    rigid.angular_velocity,
                    velocity_lever,
                )
            )
            arrays["angular_velocity_sensor"][output_index] = (
                self.direct.body_to_imu_rotation
                @ rigid.angular_velocity
                + self.direct.gyro_bias
            )
            arrays["specific_force_sensor"][output_index] = (
                self.direct.body_to_imu_rotation
                @ specific_force_body
                + self.direct.accelerometer_bias
            )
            arrays["cog_position"][output_index] = (
                rigid.position
            )
            arrays["cog_velocity_world"][output_index] = (
                rigid.linear_velocity
            )
            arrays["actuator_thrust"][output_index] = (
                actuators.thrust
            )
            arrays["actuator_gimbal"][output_index] = (
                actuators.gimbal_angle
            )

        if 0 not in self.output_lookup:
            raise RuntimeError(
                "reconstruction grid does not start at a wrench knot"
            )
        store(self.output_lookup[0], start_time)

        for integration_index in range(
            self.integration_time.size - 1
        ):
            start = float(
                self.integration_time[integration_index]
            )
            end = float(
                self.integration_time[integration_index + 1]
            )
            dt = end - start
            midpoint = start + 0.5 * dt
            command = smooth._command(
                self.bag.rotor_history.exact_zoh(
                    midpoint, self.delay
                ),
                self.bag.gimbal_history.exact_zoh(
                    midpoint, self.gimbal_delay
                ),
            )
            midpoint_actuators = advance_actuators(
                actuators,
                command,
                self.actuator_parameters,
                0.5 * dt,
            )
            rigid, rigid_sensitivity = (
                self._step_with_sensitivity(
                    start,
                    rigid,
                    rigid_sensitivity,
                    midpoint_actuators,
                    dt,
                    coefficients,
                )
            )
            actuators = advance_actuators(
                midpoint_actuators,
                command,
                self.actuator_parameters,
                0.5 * dt,
            )
            next_index = integration_index + 1
            if next_index in self.output_lookup:
                store(
                    self.output_lookup[next_index],
                    end,
                )

        simulation = baseline.Simulation(
            time=self.knot_time.copy(),
            **arrays,
        )

        # The initial pose is copied directly from the spline, so its six
        # residual rows are structurally zero.  Use those six rows instead
        # to impose the missing terminal state information: CoG linear
        # velocity in world coordinates and angular velocity in body
        # coordinates.
        terminal_velocity_error = np.concatenate(
            (
                rigid.linear_velocity
                - self.target_terminal_cog_velocity,
                rigid.angular_velocity
                - self.target_terminal_angular_velocity,
            )
        )
        terminal_velocity_jacobian = np.vstack(
            (
                rigid_sensitivity[7:10],
                rigid_sensitivity[10:13],
            )
        )
        residual[0] = self.pose_factor @ terminal_velocity_error
        jacobian[0] = self.pose_factor @ terminal_velocity_jacobian

        flat_residual = residual.ravel()
        flat_jacobian = jacobian.reshape(
            -1, self.dimension
        )
        if (
            np.any(~np.isfinite(flat_residual))
            or np.any(~np.isfinite(flat_jacobian))
        ):
            raise FloatingPointError(
                "external-wrench reconstruction is non-finite"
            )
        return WrenchReplayEvaluation(
            residual=flat_residual,
            jacobian=flat_jacobian,
            simulation=simulation,
            knot_time=self.knot_time.copy(),
            coefficients=coefficients.copy(),
        )


def _solve_wrench_replay(
    bag: BagSplineData,
    physical_coordinate: Sequence[float],
    delay: float,
    dynamics_evaluation: BagDynamicsEvaluation,
    arguments: argparse.Namespace,
    reference_parameters: VehicleParameters,
    gimbal_delay: Optional[float] = None,
) -> tuple[
    WrenchReplayProblem,
    WrenchReplayEvaluation,
    Mapping[str, Any],
]:
    problem = WrenchReplayProblem(
        bag,
        physical_coordinate,
        delay,
        dynamics_evaluation,
        reference_parameters,
        gimbal_delay=gimbal_delay,
    )
    objective = _CachedObjective(problem.evaluate)
    initial = np.zeros(problem.dimension, dtype=float)
    initial_evaluation = problem.evaluate(initial)
    result = least_squares(
        objective.residual,
        initial,
        jac=objective.jacobian,
        method="trf",
        x_scale="jac",
        loss="linear",
        ftol=arguments.ftol,
        xtol=arguments.xtol,
        gtol=arguments.gtol,
        max_nfev=arguments.strict_max_nfev,
        verbose=0,
    )
    evaluation = problem.evaluate(result.x)
    optimizer = dict(_optimizer_payload(result))
    optimizer.update(
        {
            "knot_count": int(problem.knot_time.size),
            "coefficient_dimension": int(problem.dimension),
            "uses_regularization": False,
            "initial_pose_cost": 0.5
            * float(
                initial_evaluation.residual
                @ initial_evaluation.residual
            ),
            "final_pose_cost": 0.5
            * float(
                evaluation.residual @ evaluation.residual
            ),
        }
    )
    return problem, evaluation, optimizer

def _orientation_errors(
    observed_xyzw: np.ndarray,
    predicted_xyzw: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            so3_log(
                quaternion_to_matrix(observed).T
                @ quaternion_to_matrix(predicted)
            )
            for observed, predicted in zip(observed_xyzw, predicted_xyzw)
        ],
        dtype=float,
    )


def _pose_metrics(
    observations: baseline.Observations,
    simulation: baseline.Simulation,
) -> dict[str, Any]:
    position_error = simulation.sensor_position - observations.sensor_position
    orientation_error = _orientation_errors(
        observations.sensor_orientation_xyzw,
        simulation.sensor_orientation_xyzw,
    )
    position_norm = np.linalg.norm(position_error, axis=1)
    orientation_norm = np.linalg.norm(orientation_error, axis=1)
    return {
        "position_rmse_m": float(np.sqrt(np.mean(position_norm**2))),
        "position_component_rmse_m": np.sqrt(
            np.mean(position_error**2, axis=0)
        ),
        "orientation_angle_rmse_rad": float(
            np.sqrt(np.mean(orientation_norm**2))
        ),
        "orientation_angle_rmse_deg": float(
            np.degrees(np.sqrt(np.mean(orientation_norm**2)))
        ),
        "terminal_position_error_m": float(position_norm[-1]),
        "terminal_orientation_error_deg": float(
            np.degrees(orientation_norm[-1])
        ),
    }


def _axis_validation_statistics(
    time_axis: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> tuple[dict[str, Any], ...]:
    dt = float(np.median(np.diff(time_axis)))
    statistics = []
    for component in range(3):
        reference = np.asarray(observed[:, component], dtype=float)
        estimate = np.asarray(predicted[:, component], dtype=float)
        error = estimate - reference
        centered_reference = reference - np.mean(reference)
        centered_estimate = estimate - np.mean(estimate)
        denominator = float(
            np.linalg.norm(centered_reference)
            * np.linalg.norm(centered_estimate)
        )
        correlation = (
            float(np.dot(centered_reference, centered_estimate) / denominator)
            if denominator > 1.0e-15
            else float("nan")
        )
        cross = np.correlate(
            centered_estimate, centered_reference, mode="full"
        )
        lags = np.arange(-reference.size + 1, reference.size, dtype=int)
        maximum_index = int(np.argmax(cross))
        statistics.append(
            {
                "axis": COMPONENT_NAMES[component],
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mean_bias": float(np.mean(error)),
                "pearson_correlation": correlation,
                "maximum_cross_correlation_time_shift_seconds": float(
                    lags[maximum_index] * dt
                ),
            }
        )
    return tuple(statistics)


def _sensor_metrics(
    observations: baseline.Observations,
    simulation: baseline.Simulation,
) -> dict[str, Any]:
    return {
        "gyro": _axis_validation_statistics(
            observations.time,
            observations.angular_velocity_sensor,
            simulation.angular_velocity_sensor,
        ),
        "specific_force": _axis_validation_statistics(
            observations.time,
            observations.specific_force_sensor,
            simulation.specific_force_sensor,
        ),
    }


def _sensor_pair_metrics(
    time_axis: np.ndarray,
    reference_gyro: np.ndarray,
    reference_specific_force: np.ndarray,
    predicted_gyro: np.ndarray,
    predicted_specific_force: np.ndarray,
) -> dict[str, Any]:
    return {
        "gyro": _axis_validation_statistics(
            time_axis,
            reference_gyro,
            predicted_gyro,
        ),
        "specific_force": _axis_validation_statistics(
            time_axis,
            reference_specific_force,
            predicted_specific_force,
        ),
    }


def _pose_spline_implied_sensor_series(
    bag: BagSplineData,
    parameters: VehicleParameters,
    time_axis: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    time_value = np.asarray(time_axis, dtype=float)
    if (
        time_value.ndim != 1
        or time_value.size < 2
        or np.any(~np.isfinite(time_value))
        or np.any(np.diff(time_value) <= 0.0)
    ):
        raise ValueError("diagnostic time axis is invalid")

    direct = bag.direct_problem
    spline = bag.spline_selection.spline.evaluate(time_value)
    _, _, cog_acceleration_world = cog_kinematics_from_pose_spline(
        spline,
        direct.pose_sensor_position,
        parameters.cog_offset,
    )

    gravity_world = np.asarray((0.0, 0.0, -GRAVITY), dtype=float)
    specific_force_cog_body = np.einsum(
        "nji,nj->ni",
        spline.body_rotation,
        cog_acceleration_world - gravity_world,
    )
    omega = spline.body_angular_velocity
    alpha = spline.body_angular_acceleration
    imu_lever = direct.imu_sensor_position - parameters.cog_offset
    specific_force_body = (
        specific_force_cog_body
        + np.cross(alpha, imu_lever)
        + np.cross(
            omega,
            np.cross(omega, imu_lever),
        )
    )

    gyro_sensor = (
        np.einsum(
            "ij,nj->ni",
            direct.body_to_imu_rotation,
            omega,
        )
        + direct.gyro_bias
    )
    specific_force_sensor = (
        np.einsum(
            "ij,nj->ni",
            direct.body_to_imu_rotation,
            specific_force_body,
        )
        + direct.accelerometer_bias
    )
    return gyro_sensor, specific_force_sensor


def _diagnostic_payload(
    observations: baseline.Observations,
    spline_implied_gyro: np.ndarray,
    spline_implied_specific_force: np.ndarray,
    reconstructed: baseline.Simulation,
) -> dict[str, Any]:
    time_axis = np.asarray(observations.time, dtype=float)
    return {
        "schema": SCHEMA + "/diagnostic/v1",
        "purpose": (
            "Persistent structured debugging output. Future debug series and "
            "metrics should be appended here, with matching pages in diagnostic.pdf."
        ),
        "time_seconds": time_axis,
        "series": {
            "measured": {
                "gyro_rad_per_s": observations.angular_velocity_sensor,
                "specific_force_m_per_s2": observations.specific_force_sensor,
            },
            "pose_spline_implied": {
                "gyro_rad_per_s": spline_implied_gyro,
                "specific_force_m_per_s2": spline_implied_specific_force,
            },
            "reconstructed": {
                "gyro_rad_per_s": reconstructed.angular_velocity_sensor,
                "specific_force_m_per_s2": reconstructed.specific_force_sensor,
            },
        },
        "comparisons": {
            "pose_spline_implied_vs_measured": _sensor_pair_metrics(
                time_axis,
                observations.angular_velocity_sensor,
                observations.specific_force_sensor,
                spline_implied_gyro,
                spline_implied_specific_force,
            ),
            "reconstructed_vs_measured": _sensor_metrics(
                observations,
                reconstructed,
            ),
            "reconstructed_vs_pose_spline_implied": _sensor_pair_metrics(
                time_axis,
                spline_implied_gyro,
                spline_implied_specific_force,
                reconstructed.angular_velocity_sensor,
                reconstructed.specific_force_sensor,
            ),
        },
    }


def _wrench_statistics(
    time_axis: np.ndarray,
    residual: np.ndarray,
) -> dict[str, Any]:
    relative_time = np.asarray(time_axis, dtype=float) - float(time_axis[0])
    duration = float(relative_time[-1])
    centered_time = relative_time - np.mean(relative_time)
    time_energy = float(centered_time @ centered_time)
    slopes = (
        centered_time @ (residual - np.mean(residual, axis=0)) / time_energy
        if time_energy > 0.0
        else np.zeros(6, dtype=float)
    )
    integral = np.trapz(residual, time_axis, axis=0)
    return {
        "component_names": ("F_x", "F_y", "F_z", "M_x", "M_y", "M_z"),
        "mean": np.mean(residual, axis=0),
        "rmse": np.sqrt(np.mean(residual**2, axis=0)),
        "linear_trend_per_second": slopes,
        "cumulative_impulse": integral,
        "duration_seconds": duration,
    }


def _common_3d_limits(*series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joined = np.vstack(series)
    lower = np.min(joined, axis=0)
    upper = np.max(joined, axis=0)
    center = 0.5 * (lower + upper)
    radius = max(0.5 * float(np.max(upper - lower)), 1.0e-3) * 1.05
    return center - radius, center + radius


def _write_trajectory_pdf(
    path: Path,
    bag: BagSplineData,
    reconstructed: baseline.Simulation,
) -> None:
    observed = _observations_at_times(
        bag.direct_problem.observations,
        reconstructed.time,
    )
    relative_time = reconstructed.time - reconstructed.time[0]
    observed_rpy = baseline._rpy_series(observed.sensor_orientation_xyzw)
    reconstructed_rpy = baseline._rpy_series(
        reconstructed.sensor_orientation_xyzw
    )
    lower, upper = _common_3d_limits(
        observed.sensor_position,
        reconstructed.sensor_position,
    )

    with PdfPages(path) as pdf:
        figure = plt.figure(figsize=(11.7, 8.3), constrained_layout=True)
        axis = figure.add_subplot(111, projection="3d")
        axis.plot(
            observed.sensor_position[:, 0],
            observed.sensor_position[:, 1],
            observed.sensor_position[:, 2],
            color="#1e5abe",
            linewidth=2.5,
            label="observed",
        )
        axis.plot(
            reconstructed.sensor_position[:, 0],
            reconstructed.sensor_position[:, 1],
            reconstructed.sensor_position[:, 2],
            color="#1e965f",
            linewidth=2.0,
            linestyle="--",
            label="estimated parameters + inferred external wrench",
        )
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.set_zlim(lower[2], upper[2])
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("z [m]")
        axis.set_title("Observed and reconstructed 3D trajectory")
        axis.legend(loc="best")
        pdf.savefig(figure)
        plt.close(figure)

        for title, reference, prediction, labels in (
            (
                "Observed and reconstructed sensor position",
                observed.sensor_position,
                reconstructed.sensor_position,
                ("x [m]", "y [m]", "z [m]"),
            ),
            (
                "Observed and reconstructed sensor orientation",
                observed_rpy,
                reconstructed_rpy,
                ("roll [rad]", "pitch [rad]", "yaw [rad]"),
            ),
        ):
            figure, axes = plt.subplots(
                3,
                1,
                figsize=(11.7, 8.3),
                sharex=True,
                constrained_layout=True,
            )
            for component, axis in enumerate(axes):
                axis.plot(
                    relative_time,
                    reference[:, component],
                    color="#1e5abe",
                    linewidth=2.2,
                    label="observed",
                )
                axis.plot(
                    relative_time,
                    prediction[:, component],
                    color="#1e965f",
                    linewidth=1.8,
                    linestyle="--",
                    label="reconstructed",
                )
                axis.set_ylabel(labels[component])
                axis.grid(True, alpha=0.25)
            axes[0].set_title(title)
            axes[0].legend(loc="best")
            axes[-1].set_xlabel(
                "time from reconstruction-support start [s]"
            )
            pdf.savefig(figure)
            plt.close(figure)

def _write_sensor_validation_pdf(
    path: Path,
    bag: BagSplineData,
    reconstructed: baseline.Simulation,
    metrics: Mapping[str, Any],
) -> None:
    del metrics
    observed = _observations_at_times(
        bag.direct_problem.observations,
        reconstructed.time,
    )
    relative_time = reconstructed.time - reconstructed.time[0]
    pages = (
        (
            "Gyroscope",
            observed.angular_velocity_sensor,
            reconstructed.angular_velocity_sensor,
            ("omega_x [rad/s]", "omega_y [rad/s]", "omega_z [rad/s]"),
        ),
        (
            "Specific force",
            observed.specific_force_sensor,
            reconstructed.specific_force_sensor,
            ("f_x [m/s^2]", "f_y [m/s^2]", "f_z [m/s^2]"),
        ),
    )

    with PdfPages(path) as pdf:
        for title, reference, prediction, labels in pages:
            figure, axes = plt.subplots(
                3,
                1,
                figsize=(11.7, 8.3),
                sharex=True,
                constrained_layout=True,
            )
            for component, axis in enumerate(axes):
                axis.plot(
                    relative_time,
                    reference[:, component],
                    color="#1e5abe",
                    linewidth=2.0,
                    label="measured",
                )
                axis.plot(
                    relative_time,
                    prediction[:, component],
                    color="#1e965f",
                    linestyle="--",
                    label="reconstructed",
                )
                axis.set_ylabel(labels[component])
                axis.grid(True, alpha=0.25)
            axes[0].set_title(
                title + " consistency (not used by parameter loss)"
            )
            axes[0].legend(loc="best")
            axes[-1].set_xlabel(
                "time from reconstruction-support start [s]"
            )
            pdf.savefig(figure)
            plt.close(figure)

def _write_diagnostic_pdf(
    path: Path,
    observations: baseline.Observations,
    spline_implied_gyro: np.ndarray,
    spline_implied_specific_force: np.ndarray,
    reconstructed: baseline.Simulation,
) -> None:
    relative_time = observations.time - observations.time[0]
    pages = (
        (
            "Diagnostic: gyroscope",
            observations.angular_velocity_sensor,
            spline_implied_gyro,
            reconstructed.angular_velocity_sensor,
            ("omega_x [rad/s]", "omega_y [rad/s]", "omega_z [rad/s]"),
        ),
        (
            "Diagnostic: specific force",
            observations.specific_force_sensor,
            spline_implied_specific_force,
            reconstructed.specific_force_sensor,
            ("f_x [m/s^2]", "f_y [m/s^2]", "f_z [m/s^2]"),
        ),
    )

    with PdfPages(path) as pdf:
        for title, measured, spline_implied, reconstructed_value, labels in pages:
            figure, axes = plt.subplots(
                3,
                1,
                figsize=(11.7, 8.3),
                sharex=True,
                constrained_layout=True,
            )
            for component, axis in enumerate(axes):
                axis.plot(
                    relative_time,
                    measured[:, component],
                    color="#1e5abe",
                    linewidth=2.0,
                    label="measured",
                )
                axis.plot(
                    relative_time,
                    spline_implied[:, component],
                    color="#8b4bb7",
                    linewidth=1.8,
                    linestyle=":",
                    label="pose-spline implied",
                )
                axis.plot(
                    relative_time,
                    reconstructed_value[:, component],
                    color="#1e965f",
                    linewidth=1.6,
                    linestyle="--",
                    label="reconstructed",
                )
                axis.set_ylabel(labels[component])
                axis.grid(True, alpha=0.25)
            axes[0].set_title(title)
            axes[0].legend(loc="best")
            axes[-1].set_xlabel(
                "time from reconstruction-support start [s]"
            )
            pdf.savefig(figure)
            plt.close(figure)


def _write_residual_wrench_pdf(
    path: Path,
    bag: BagSplineData,
    time_axis: np.ndarray,
    body_wrench: np.ndarray,
    statistics: Mapping[str, Any],
) -> None:
    del statistics
    observed = bag.direct_problem.observations
    time_value = np.asarray(time_axis, dtype=float)
    wrench_value = np.asarray(body_wrench, dtype=float)
    relative_time = time_value - observed.time[0]
    full_duration = float(
        observed.time[-1] - observed.time[0]
    )
    names = ("F_x", "F_y", "F_z", "M_x", "M_y", "M_z")
    units = ("N", "N", "N", "N m", "N m", "N m")

    with PdfPages(path) as pdf:
        for offset, title in (
            (0, "Trajectory-fitted external body force"),
            (3, "Trajectory-fitted external body torque"),
        ):
            figure, axes = plt.subplots(
                3,
                1,
                figsize=(11.7, 8.3),
                sharex=True,
                constrained_layout=True,
            )
            for local, axis in enumerate(axes):
                component = offset + local
                axis.plot(
                    relative_time,
                    wrench_value[:, component],
                    color="#1e965f",
                    linewidth=1.5,
                )
                axis.axhline(
                    0.0,
                    color="black",
                    linewidth=0.7,
                    alpha=0.5,
                )
                axis.set_xlim(0.0, full_duration)
                axis.set_ylabel(
                    "{} [{}]".format(
                        names[component], units[component]
                    )
                )
                axis.grid(True, alpha=0.25)
            axes[0].set_title(
                title
                + " (piecewise-linear; optimized from inverse-dynamics initialization)"
            )
            axes[-1].set_xlabel(
                "time from requested interval start [s]"
            )
            pdf.savefig(figure)
            plt.close(figure)

def _write_spline_fit_pdf(path: Path, bag: BagSplineData) -> None:
    direct = bag.direct_problem
    observed = direct.observations
    spline = bag.spline_selection.spline.evaluate(observed.time)
    spline_sensor_rotation = bag.spline_selection.spline.sensor_rotation(
        observed.time
    )
    spline_quaternion = np.asarray(
        [matrix_to_quaternion(value) for value in spline_sensor_rotation],
        dtype=float,
    )
    observed_rpy = baseline._rpy_series(observed.sensor_orientation_xyzw)
    spline_rpy = baseline._rpy_series(spline_quaternion)
    position_error = spline.sensor_position - observed.sensor_position
    orientation_error = _orientation_errors(
        observed.sensor_orientation_xyzw, spline_quaternion
    )
    relative_time = observed.time - observed.time[0]
    collocation = bag.collocation
    collocation_relative = collocation.time - collocation.time[0]
    with PdfPages(path) as pdf:
        for title, reference, prediction, labels in (
            (
                "Pose-only spline position fit",
                observed.sensor_position,
                spline.sensor_position,
                ("x [m]", "y [m]", "z [m]"),
            ),
            (
                "Pose-only spline orientation fit",
                observed_rpy,
                spline_rpy,
                ("roll [rad]", "pitch [rad]", "yaw [rad]"),
            ),
        ):
            figure, axes = plt.subplots(
                3, 1, figsize=(11.7, 8.3), sharex=True, constrained_layout=True
            )
            for component, axis in enumerate(axes):
                axis.plot(
                    relative_time,
                    reference[:, component],
                    label="observed",
                    color="#1e5abe",
                )
                axis.plot(
                    relative_time,
                    prediction[:, component],
                    label="spline",
                    color="#1e965f",
                    linestyle="--",
                )
                axis.set_ylabel(labels[component])
                axis.grid(True, alpha=0.25)
            axes[0].set_title(
                "{}; selected knot spacing={:.4g}s".format(
                    title, bag.spline_selection.selected_spacing_seconds
                )
            )
            axes[0].legend(loc="best")
            axes[-1].set_xlabel("time [s]")
            pdf.savefig(figure)
            plt.close(figure)

        for title, value, labels in (
            (
                "Spline pose residual",
                np.column_stack((position_error, orientation_error)),
                (
                    "dx [m]",
                    "dy [m]",
                    "dz [m]",
                    "dRx [rad]",
                    "dRy [rad]",
                    "dRz [rad]",
                ),
            ),
            (
                "Spline translational derivatives",
                np.column_stack(
                    (
                        collocation.sensor_velocity_world,
                        collocation.sensor_acceleration_world,
                    )
                ),
                (
                    "vx [m/s]",
                    "vy [m/s]",
                    "vz [m/s]",
                    "ax [m/s2]",
                    "ay [m/s2]",
                    "az [m/s2]",
                ),
            ),
            (
                "Spline rotational derivatives (body frame)",
                np.column_stack(
                    (
                        collocation.body_angular_velocity,
                        collocation.body_angular_acceleration,
                    )
                ),
                (
                    "wx [rad/s]",
                    "wy [rad/s]",
                    "wz [rad/s]",
                    "alphax [rad/s2]",
                    "alphay [rad/s2]",
                    "alphaz [rad/s2]",
                ),
            ),
        ):
            time_value = (
                relative_time
                if value.shape[0] == relative_time.size
                else collocation_relative
            )
            figure, axes = plt.subplots(
                3,
                2,
                figsize=(11.7, 8.3),
                sharex=True,
                constrained_layout=True,
            )
            for component, axis in enumerate(axes.ravel()):
                axis.plot(time_value, value[:, component], color="#8b4bb7")
                axis.set_ylabel(labels[component])
                axis.grid(True, alpha=0.25)
            axes[0, 0].set_title(title)
            axes[-1, 0].set_xlabel("time [s]")
            axes[-1, 1].set_xlabel("time [s]")
            pdf.savefig(figure)
            plt.close(figure)

        figure, axis = plt.subplots(figsize=(11.7, 8.3), constrained_layout=True)
        spacings = [
            candidate.requested_knot_spacing_seconds
            for candidate in bag.spline_selection.candidates
        ]
        scores = [candidate.score for candidate in bag.spline_selection.candidates]
        axis.plot(spacings, scores, marker="o")
        axis.axvline(
            bag.spline_selection.selected_spacing_seconds,
            color="#1e965f",
            linestyle="--",
            label="selected",
        )
        axis.set_xlabel("knot spacing [s]")
        axis.set_ylabel("blocked-CV pose score [m2]")
        axis.set_title("Pose-only blocked cross-validation")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
        pdf.savefig(figure)
        plt.close(figure)


def _write_delay_profile_pdf(
    path: Path,
    smooth_delay: float,
    candidate_delays: np.ndarray,
    screening_costs: np.ndarray,
    solutions: Sequence[DynamicsSolution],
    selected: DynamicsSolution,
) -> None:
    figure, axis = plt.subplots(figsize=(11.7, 8.3), constrained_layout=True)
    axis.plot(
        1000.0 * candidate_delays,
        screening_costs,
        marker="o",
        label="strict ZOH screening",
    )
    axis.scatter(
        [1000.0 * solution.delay_seconds for solution in solutions],
        [
            0.5
            * float(
                solution.evaluation.residual
                @ solution.evaluation.residual
            )
            for solution in solutions
        ],
        marker="s",
        s=70,
        label="strict ZOH, physical parameters reoptimized",
    )
    axis.axvline(
        1000.0 * smooth_delay,
        color="#9467bd",
        linestyle="--",
        label="smoothstep estimate",
    )
    axis.scatter(
        [1000.0 * selected.delay_seconds],
        [0.5 * float(selected.evaluation.residual @ selected.evaluation.residual)],
        marker="*",
        s=220,
        color="#1e965f",
        label="selected strict ZOH",
        zorder=5,
    )
    axis.set_xlabel("recorded-command lag [ms]")
    axis.set_ylabel("joint gradient-matching objective")
    axis.set_title("Smooth lag estimate and strict-ZOH local polish")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    with PdfPages(path) as pdf:
        pdf.savefig(figure)
    plt.close(figure)


def _spline_fit_metrics(bag: BagSplineData) -> dict[str, Any]:
    observations = bag.direct_problem.observations
    evaluation = bag.spline_selection.spline.evaluate(observations.time)
    sensor_rotation = bag.spline_selection.spline.sensor_rotation(
        observations.time
    )
    quaternion = np.asarray(
        [matrix_to_quaternion(value) for value in sensor_rotation],
        dtype=float,
    )
    position_error = evaluation.sensor_position - observations.sensor_position
    rotation_error = _orientation_errors(
        observations.sensor_orientation_xyzw, quaternion
    )
    return {
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "position_vector_rmse_m": float(
            np.sqrt(np.mean(np.sum(position_error**2, axis=1)))
        ),
        "orientation_angle_rmse_rad": float(
            np.sqrt(np.mean(np.sum(rotation_error**2, axis=1)))
        ),
        "orientation_angle_rmse_deg": float(
            np.degrees(np.sqrt(np.mean(np.sum(rotation_error**2, axis=1))))
        ),
        "maximum_acceleration_m_per_s2": float(
            np.max(np.linalg.norm(bag.collocation.sensor_acceleration_world, axis=1))
        ),
        "maximum_angular_acceleration_rad_per_s2": float(
            np.max(np.linalg.norm(bag.collocation.body_angular_acceleration, axis=1))
        ),
    }


def _parameter_lines(
    selected: DynamicsSolution,
    initial_delay: float,
    bags: Sequence[BagSplineData],
    bag_payloads: Sequence[Mapping[str, Any]],
    reference_parameters: VehicleParameters,
) -> list[str]:
    reference = reference_parameters
    estimated = selected.evaluation.decoded.parameters
    reference_principal = np.linalg.eigvalsh(reference.inertia)
    estimated_principal = np.linalg.eigvalsh(estimated.inertia)

    def update_line(
        name: str,
        before: float,
        after: float,
        unit: str = "",
        *,
        ratio: bool = True,
    ) -> str:
        delta = after - before
        if ratio and abs(before) > 1.0e-15:
            ratio_text = "{:.8g}".format(after / before)
        else:
            ratio_text = "-"
        unit_text = " [{}]".format(unit) if unit else ""
        return (
            "  {:32s} {: .10g} -> {: .10g}   delta={:+.6g}   ratio={}"
        ).format(name + unit_text, before, after, delta, ratio_text)

    lines = [
        "Deterministic pose-spline dynamics estimator",
        "",
        "Shared parameter update: reference -> estimated",
        update_line("mass", reference.mass, estimated.mass, "kg"),
    ]
    for component, name in enumerate(("CoG x", "CoG y", "CoG z")):
        lines.append(
            update_line(
                name,
                float(reference.cog_offset[component]),
                float(estimated.cog_offset[component]),
                "m",
                ratio=False,
            )
        )
    for component in range(4):
        lines.append(
            update_line(
                "rotor effectiveness {}".format(component + 1),
                float(reference.force_effectiveness[component]),
                float(estimated.force_effectiveness[component]),
            )
        )
    lines.append(
        update_line(
            "command lag",
            float(initial_delay),
            float(selected.delay_seconds),
            "s",
            ratio=False,
        )
    )

    lines.extend(["", "Inertia matrix [kg m^2]", "  reference:"])
    lines.extend(
        "    [{: .10g}, {: .10g}, {: .10g}]".format(*row)
        for row in reference.inertia
    )
    lines.append("  estimated:")
    lines.extend(
        "    [{: .10g}, {: .10g}, {: .10g}]".format(*row)
        for row in estimated.inertia
    )
    lines.extend(["", "Principal moments: reference -> estimated"])
    for component in range(3):
        lines.append(
            update_line(
                "principal moment {}".format(component + 1),
                float(reference_principal[component]),
                float(estimated_principal[component]),
                "kg m^2",
            )
        )

    lines.extend(
        [
            "",
            "Optimization",
            "  joint dynamics loss: {:.12g}".format(
                selected.evaluation.data_loss
            ),
            "  Gaussian-prior cost: {:.12g}".format(
                selected.evaluation.prior_cost
            ),
            "  selected strict-ZOH lag [s]: {:.12g}".format(
                selected.delay_seconds
            ),
        ]
    )

    for bag, payload in zip(bags, bag_payloads):
        pose = payload[
            "estimated_with_external_wrench_forward_metrics"
        ]
        sensors = payload["sensor_validation_with_external_wrench"]
        wrench = payload["residual_wrench_statistics"]
        gyro_rmse = np.asarray(
            [item["rmse"] for item in sensors["gyro"]],
            dtype=float,
        )
        force_rmse = np.asarray(
            [item["rmse"] for item in sensors["specific_force"]],
            dtype=float,
        )
        wrench_rms = np.asarray(wrench["rmse"], dtype=float)

        lines.extend(
            [
                "",
                "Bag {}".format(bag.specification.bag_id),
                "  reconstruction position RMSE [m]: {:.12g}".format(
                    pose["position_rmse_m"]
                ),
                "  reconstruction position component RMSE [m]: {}".format(
                    np.array2string(
                        np.asarray(pose["position_component_rmse_m"]),
                        precision=6,
                    )
                ),
                "  reconstruction orientation RMSE [deg]: {:.12g}".format(
                    pose["orientation_angle_rmse_deg"]
                ),
                "  gyro consistency RMSE xyz [rad/s]: {}".format(
                    np.array2string(gyro_rmse, precision=6)
                ),
                "  specific-force consistency RMSE xyz [m/s^2]: {}".format(
                    np.array2string(force_rmse, precision=6)
                ),
                "  inferred external force RMS xyz [N]: {}".format(
                    np.array2string(wrench_rms[:3], precision=6)
                ),
                "  inferred external torque RMS xyz [N m]: {}".format(
                    np.array2string(wrench_rms[3:], precision=6)
                ),
                "  bag dynamics loss: {:.12g}".format(
                    payload["dynamics_loss"]
                ),
            ]
        )
    return lines

def _write_parameters_pdf(path: Path, lines: Sequence[str]) -> None:
    wrapped_lines = []
    for line in lines:
        wrapped_lines.extend(
            textwrap.wrap(
                line,
                width=96,
                subsequent_indent="    ",
                replace_whitespace=False,
                drop_whitespace=False,
            )
            or [""]
        )
    page_size = 55
    with PdfPages(path) as pdf:
        for start in range(0, len(wrapped_lines), page_size):
            figure = plt.figure(figsize=(8.3, 11.7), constrained_layout=True)
            figure.text(
                0.04,
                0.98,
                "\n".join(wrapped_lines[start : start + page_size]),
                va="top",
                ha="left",
                family="monospace",
                fontsize=7.5,
            )
            pdf.savefig(figure)
            plt.close(figure)




def _solution_cost(solution: DynamicsSolution) -> float:
    residual = solution.evaluation.residual
    return 0.5 * float(residual @ residual)



def _rollout_pose_score(
    observations: baseline.Observations,
    simulation: baseline.Simulation,
    reference_parameters: VehicleParameters,
) -> float:
    position_error = (
        simulation.sensor_position - observations.sensor_position
    )
    orientation_error = _orientation_errors(
        observations.sensor_orientation_xyzw,
        simulation.sensor_orientation_xyzw,
    )
    rotational_metric = (
        reference_parameters.inertia / reference_parameters.mass
    )
    squared = np.sum(position_error**2, axis=1) + np.einsum(
        "ni,ij,nj->n",
        orientation_error,
        rotational_metric,
        orientation_error,
    )
    return 0.5 * float(np.mean(squared))


def _strict_candidate_payloads(
    candidate_delays: np.ndarray,
    screening_costs: np.ndarray,
    solutions: Sequence[DynamicsSolution],
    selected: DynamicsSolution,
) -> list[dict[str, Any]]:
    refined = {
        round(solution.delay_seconds, 12): solution for solution in solutions
    }
    payloads = []
    for delay, screening in zip(candidate_delays, screening_costs):
        solution = refined.get(round(float(delay), 12))
        payload: dict[str, Any] = {
            "delay_seconds": float(delay),
            "screening_cost": float(screening),
            "refined": solution is not None,
            "selected": bool(
                np.isclose(delay, selected.delay_seconds, atol=5.0e-13, rtol=0.0)
            ),
        }
        if solution is not None:
            payload.update(
                {
                    "refined_cost": _solution_cost(solution),
                    "physical_coordinate": solution.physical_coordinate,
                    "optimizer": solution.optimizer,
                }
            )
        payloads.append(payload)
    return payloads


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_sanitize(value.tolist())
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def run(arguments: argparse.Namespace) -> int:
    try:
        config = load_spline_config(arguments.config)
        vehicle_model = load_vehicle_model(
            arguments.vehicle_model_json
        )
        parameter_prior = load_parameter_prior(
            arguments.prior_json
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    reference_parameters = vehicle_model.parameters
    initial_physical_coordinate = np.zeros(
        PHYSICAL_DIMENSION,
        dtype=float,
    )
    initial_delay = (
        config.multi_bag.initial_delay_seconds
        if arguments.initial_delay is None
        else float(arguments.initial_delay)
    )
    _validate_arguments(arguments, config, initial_delay)
    started = time.perf_counter()
    print(
        "vehicle-model reference: {}".format(
            vehicle_model.source_path
        ),
        flush=True,
    )
    print(
        "Gaussian parameter prior: {}".format(
            parameter_prior.source_path
        ),
        flush=True,
    )

    raw_weights = np.asarray(
        [specification.weight for specification in config.multi_bag.bags],
        dtype=float,
    )
    normalized_weights = raw_weights / np.sum(raw_weights)
    bags = []
    for index, (specification, weight) in enumerate(
        zip(config.multi_bag.bags, normalized_weights)
    ):
        print(
            "loading and fitting pose spline {}/{} {}: {} [{:.3f}, {:.3f}] s".format(
                index + 1,
                len(config.multi_bag.bags),
                specification.bag_id,
                specification.path,
                specification.start,
                specification.end,
            ),
            flush=True,
        )
        flight = load_flight_data(
            str(specification.path),
            start_local=specification.start,
            end_local=specification.end,
            include_fc_specific_force=True,
            compute_sha256=False,
        )
        bag = _build_bag_data(
            specification,
            float(weight),
            flight,
            initial_delay,
            config.spline,
            arguments,
            reference_parameters,
            vehicle_model.geometry,
        )
        print(
            "  selected knot spacing {:.6g}s from {}; parameter support "
            "[{:.6f}, {:.6f}]s after excluding {:.6g} knot spans per side".format(
                bag.spline_selection.selected_spacing_seconds,
                tuple(config.spline.knot_spacing_candidates_seconds),
                bag.collocation_time[0],
                bag.collocation_time[-1],
                bag.boundary_exclusion_knot_spans_each_side,
            ),
            flush=True,
        )
        bags.append(bag)
    problem = SplineDynamicsProblem(
        bags,
        reference_parameters,
        parameter_prior,
    )

    physical_lower, physical_upper = _physical_bounds(
        initial_physical_coordinate
    )
    split_lag_result = _split_command_lag_search(
        problem,
        initial_physical_coordinate,
        physical_lower,
        physical_upper,
        arguments,
    )
    selected = split_lag_result["selected_solution"]
    command_lag_search = {
        key: value for key, value in split_lag_result.items()
        if key != "selected_solution"
    }
    smooth_stage_payloads = command_lag_search["smooth_stages"]
    strict_payloads = command_lag_search["strict_candidates"]
    smooth_delay = command_lag_search["smooth_result"]["rotor_delay_seconds"]
    candidate_delays = np.empty(0, dtype=float)
    screening_costs = np.empty(0, dtype=float)
    strict_solutions = [selected]
    print(
        "selected strict lags rotor={:.6f}s gimbal={:.6f}s; "
        "producing reconstruction and reports".format(
            selected.delay_seconds, selected.gimbal_delay_seconds
        ),
        flush=True,
    )


    output_directory = (
        arguments.output_dir.expanduser().resolve() / OUTPUT_SUBDIRECTORY
    )
    bags_directory = output_directory / "bags"
    bags_directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        DATA_DICTIONARY_SOURCE,
        output_directory / "DATA_DICTIONARY.md",
    )
    bag_payloads = []
    bag_sources = []
    bag_outputs: dict[str, Any] = {}
    selected_parameters = selected.evaluation.decoded.parameters
    selected_gimbal_delay = (
        selected.delay_seconds if selected.gimbal_delay_seconds is None else float(selected.gimbal_delay_seconds)
    )
    initial_gimbal_delay = float(getattr(arguments, "initial_gimbal_delay", initial_delay))
    for index, bag in enumerate(bags):
        bag_id = bag.specification.bag_id
        bag_directory = bags_directory / bag_id
        bag_directory.mkdir(parents=True, exist_ok=True)
        evaluation = selected.evaluation.bag_evaluations[index]
        print(
            "fitting piecewise-linear external wrench for {}".format(
                bag_id
            ),
            flush=True,
        )
        (
            wrench_replay_problem,
            wrench_replay_evaluation,
            wrench_replay_optimizer,
        ) = _solve_wrench_replay(
            bag,
            selected.physical_coordinate,
            selected.delay_seconds,
            evaluation,
            arguments,
            reference_parameters,
            gimbal_delay=selected_gimbal_delay,
        )
        external_wrench_time = (
            wrench_replay_evaluation.knot_time
        )
        external_wrench_value = (
            wrench_replay_evaluation.coefficients
        )
        external_wrench_initial = (
            wrench_replay_problem.initial_coefficients
        )
        external_wrench_correction = (
            external_wrench_value - external_wrench_initial
        )
        external_wrench_rollout = (
            wrench_replay_evaluation.simulation
        )
        print(
            "  external-wrench pose cost {:.9g} -> {:.9g}, nfev={}".format(
                wrench_replay_optimizer["initial_pose_cost"],
                wrench_replay_optimizer["final_pose_cost"],
                wrench_replay_optimizer["nfev"],
            ),
            flush=True,
        )
        estimated_rollout = forward_rollout(
            bag,
            selected.physical_coordinate,
            selected.delay_seconds,
            reference_parameters,
            gimbal_delay=selected_gimbal_delay,
        )
        reference_rollout = forward_rollout(
            bag,
            np.zeros(PHYSICAL_DIMENSION, dtype=float),
            initial_delay,
            reference_parameters,
            gimbal_delay=initial_gimbal_delay,
        )
        observations = bag.direct_problem.observations
        reconstruction_observations = _observations_at_times(
            observations,
            external_wrench_rollout.time,
        )
        estimated_metrics = _pose_metrics(observations, estimated_rollout)
        external_wrench_metrics = _pose_metrics(
            reconstruction_observations,
            external_wrench_rollout,
        )
        reference_metrics = _pose_metrics(observations, reference_rollout)
        estimated_score = _rollout_pose_score(observations, estimated_rollout, reference_parameters)
        external_wrench_score = _rollout_pose_score(
            reconstruction_observations,
            external_wrench_rollout,
            reference_parameters,
        )
        estimated_on_reconstruction_support = _simulation_at_times(
            estimated_rollout,
            external_wrench_rollout.time,
        )
        estimated_support_score = _rollout_pose_score(
            reconstruction_observations,
            estimated_on_reconstruction_support,
            reference_parameters,
        )
        reference_score = _rollout_pose_score(observations, reference_rollout, reference_parameters)
        sensor_metrics = _sensor_metrics(observations, estimated_rollout)
        external_wrench_sensor_metrics = _sensor_metrics(
            reconstruction_observations,
            external_wrench_rollout,
        )
        (
            spline_implied_gyro,
            spline_implied_specific_force,
        ) = _pose_spline_implied_sensor_series(
            bag,
            selected_parameters,
            external_wrench_rollout.time,
        )
        diagnostic_payload = _diagnostic_payload(
            reconstruction_observations,
            spline_implied_gyro,
            spline_implied_specific_force,
            external_wrench_rollout,
        )
        wrench_statistics = _wrench_statistics(
            external_wrench_time,
            external_wrench_value,
        )
        spline_metrics = _spline_fit_metrics(bag)
        spline_fit_bounds = np.asarray(
            (
                bag.spline_selection.spline.start_time,
                bag.spline_selection.spline.end_time,
            ),
            dtype=float,
        )
        estimation_bounds = np.asarray(
            (bag.collocation_time[0], bag.collocation_time[-1]),
            dtype=float,
        )
        boundary_exclusion_seconds = np.asarray(
            (
                estimation_bounds[0] - spline_fit_bounds[0],
                spline_fit_bounds[1] - estimation_bounds[1],
            ),
            dtype=float,
        )
        estimation_output_mask = (
            (observations.time >= estimation_bounds[0])
            & (observations.time <= estimation_bounds[1])
        )
        bag_payload = {
            "id": bag_id,
            "normalized_weight": bag.normalized_weight,
            "spline": {
                "degree": bag.spline_selection.spline.degree,
                "rotation_representation": "normalized quaternion B-spline",
                "selected_knot_spacing_seconds": (
                    bag.spline_selection.selected_spacing_seconds
                ),
                "fit_interval_seconds": spline_fit_bounds,
                "parameter_estimation_interval_seconds": estimation_bounds,
                "boundary_exclusion_knot_spans_each_side": (
                    bag.boundary_exclusion_knot_spans_each_side
                ),
                "actual_boundary_exclusion_seconds_start_end": (
                    boundary_exclusion_seconds
                ),
                "parameter_estimation_output_sample_count": int(
                    np.count_nonzero(estimation_output_mask)
                ),
                "uses_spline_fit_boundaries_for_parameter_loss": False,
                "uses_spline_extrapolation_for_parameter_loss": False,
                "blocked_cross_validation": [
                    candidate_payload(candidate)
                    for candidate in bag.spline_selection.candidates
                ],
                "fit_metrics": spline_metrics,
            },
            "collocation_count": int(bag.collocation_time.size),
            "dynamics_loss": evaluation.dynamics_loss,
            "residual_wrench_statistics": wrench_statistics,
            "inferred_external_wrench": {
                "definition": (
                    "piecewise-linear body wrench optimized so the fixed-parameter "
                    "forward pose matches the observed pose spline"
                ),
                "initialization": (
                    "required body wrench minus modeled body wrench"
                ),
                "frame": "body",
                "components": ("F_x", "F_y", "F_z", "M_x", "M_y", "M_z"),
                "force_unit": "N",
                "torque_unit": "N m",
                "knot_count": int(external_wrench_time.size),
                "uses_regularization": False,
                "optimizer": wrench_replay_optimizer,
                "rollout_interpolation": (
                    "continuous piecewise-linear interpolation between wrench knots"
                ),
            },
            "estimated_forward_metrics": estimated_metrics,
            "estimated_with_external_wrench_forward_metrics": (
                external_wrench_metrics
            ),
            "reconstruction_metrics": external_wrench_metrics,
            "reference_forward_metrics": reference_metrics,
            "estimated_forward_pose_score_m2": estimated_score,
            "estimated_with_external_wrench_forward_pose_score_m2": (
                external_wrench_score
            ),
            "external_wrench_improves_estimated_free_rollout": bool(
                external_wrench_score < estimated_support_score
            ),
            "reference_forward_pose_score_m2": reference_score,
            "estimated_improves_over_reference": bool(
                estimated_score < reference_score
            ),
            "sensor_validation": sensor_metrics,
            "sensor_validation_with_external_wrench": (
                external_wrench_sensor_metrics
            ),
            "sensor_consistency": external_wrench_sensor_metrics,
            "reference_rollout_lag_seconds": initial_delay,
        }
        _write_trajectory_pdf(
            bag_directory / "trajectory.pdf",
            bag,
            external_wrench_rollout,
        )
        _write_sensor_validation_pdf(
            bag_directory / "sensor_consistency.pdf",
            bag,
            external_wrench_rollout,
            external_wrench_sensor_metrics,
        )
        _write_diagnostic_pdf(
            bag_directory / "diagnostic.pdf",
            reconstruction_observations,
            spline_implied_gyro,
            spline_implied_specific_force,
            external_wrench_rollout,
        )
        baseline._write_json(
            bag_directory / "diagnostic.json",
            _json_sanitize(diagnostic_payload),
        )
        _write_residual_wrench_pdf(
            bag_directory / "external_wrench.pdf",
            bag,
            external_wrench_time,
            external_wrench_value,
            wrench_statistics,
        )
        np.savez_compressed(
            bag_directory / "spline_dynamics.npz",
            collocation_time=bag.collocation_time,
            output_time=observations.time,
            spline_fit_time_bounds=spline_fit_bounds,
            parameter_estimation_time_bounds=estimation_bounds,
            parameter_estimation_output_mask=estimation_output_mask,
            spline_boundary_exclusion_seconds_start_end=(
                boundary_exclusion_seconds
            ),
            spline_boundary_exclusion_knot_spans_each_side=np.asarray(
                bag.boundary_exclusion_knot_spans_each_side,
                dtype=float,
            ),
            observed_sensor_position=observations.sensor_position,
            observed_sensor_orientation_xyzw=(
                observations.sensor_orientation_xyzw
            ),
            observed_sensor_velocity_world=(
                observations.sensor_velocity_world
            ),
            observed_angular_velocity_sensor=(
                observations.angular_velocity_sensor
            ),
            observed_specific_force_sensor=(
                observations.specific_force_sensor
            ),
            spline_sensor_position=bag.collocation.sensor_position,
            spline_sensor_velocity_world=(
                bag.collocation.sensor_velocity_world
            ),
            spline_sensor_acceleration_world=(
                bag.collocation.sensor_acceleration_world
            ),
            spline_body_rotation=bag.collocation.body_rotation,
            spline_body_angular_velocity=(
                bag.collocation.body_angular_velocity
            ),
            spline_body_angular_acceleration=(
                bag.collocation.body_angular_acceleration
            ),
            required_body_wrench=evaluation.required_body_wrench,
            modeled_body_wrench=evaluation.modeled_body_wrench,
            residual_body_wrench=evaluation.residual_body_wrench,
            raw_inferred_external_body_wrench_time=bag.collocation_time,
            raw_inferred_external_body_wrench=(
                evaluation.residual_body_wrench
            ),
            inferred_external_body_wrench_time=external_wrench_time,
            inferred_external_body_wrench=external_wrench_value,
            inferred_external_body_wrench_initial=(
                external_wrench_initial
            ),
            inferred_external_body_wrench_correction=(
                external_wrench_correction
            ),
            estimated_forward_sensor_position=(
                estimated_rollout.sensor_position
            ),
            estimated_forward_sensor_orientation_xyzw=(
                estimated_rollout.sensor_orientation_xyzw
            ),
            estimated_forward_sensor_velocity_world=(
                estimated_rollout.sensor_velocity_world
            ),
            estimated_forward_angular_velocity_sensor=(
                estimated_rollout.angular_velocity_sensor
            ),
            estimated_forward_specific_force_sensor=(
                estimated_rollout.specific_force_sensor
            ),
            reconstruction_time=external_wrench_rollout.time,
            external_wrench_forward_time=external_wrench_rollout.time,
            external_wrench_forward_sensor_position=(
                external_wrench_rollout.sensor_position
            ),
            external_wrench_forward_sensor_orientation_xyzw=(
                external_wrench_rollout.sensor_orientation_xyzw
            ),
            external_wrench_forward_sensor_velocity_world=(
                external_wrench_rollout.sensor_velocity_world
            ),
            external_wrench_forward_angular_velocity_sensor=(
                external_wrench_rollout.angular_velocity_sensor
            ),
            external_wrench_forward_specific_force_sensor=(
                external_wrench_rollout.specific_force_sensor
            ),
            reference_forward_sensor_position=reference_rollout.sensor_position,
            reference_forward_sensor_orientation_xyzw=(
                reference_rollout.sensor_orientation_xyzw
            ),
        )
        bag_result = {
            "schema": SCHEMA + "/bag-result",
            "source": {
                "id": bag_id,
                "path": str(bag.specification.path),
                "sha256": baseline._sha256(bag.specification.path),
                "requested_interval_seconds": (
                    bag.specification.start,
                    bag.specification.end,
                ),
                "raw_weight": bag.specification.weight,
                "normalized_weight": bag.normalized_weight,
            },
            "shared_parameters": baseline._physical_parameters(
                selected_parameters
            ),
            "shared_physical_coordinate": selected.physical_coordinate,
            "shared_delay_seconds": selected.delay_seconds,
            "shared_rotor_delay_seconds": selected.delay_seconds,
            "shared_gimbal_delay_seconds": selected_gimbal_delay,
            "diagnostics": bag_payload,
            "outputs": {
                "trajectory_pdf": "trajectory.pdf",
                "sensor_consistency_pdf": "sensor_consistency.pdf",
                "diagnostic_pdf": "diagnostic.pdf",
                "diagnostic_json": "diagnostic.json",
                "external_wrench_pdf": "external_wrench.pdf",
                "spline_dynamics_npz": "spline_dynamics.npz",
            },
        }
        baseline._write_json(
            bag_directory / "result.json", _json_sanitize(bag_result)
        )
        bag_payloads.append(bag_payload)
        bag_sources.append(bag_result["source"])
        relative = "bags/{}/".format(bag_id)
        bag_outputs[bag_id] = {
            "result_json": relative + "result.json",
            "trajectory_pdf": relative + "trajectory.pdf",
            "sensor_consistency_pdf": relative + "sensor_consistency.pdf",
            "diagnostic_pdf": relative + "diagnostic.pdf",
            "diagnostic_json": relative + "diagnostic.json",
            "external_wrench_pdf": relative + "external_wrench.pdf",
            "spline_dynamics_npz": relative + "spline_dynamics.npz",
        }

    lines = _parameter_lines(
        selected,
        initial_delay,
        bags,
        bag_payloads,
        reference_parameters,
    )
    strict._write_text(output_directory / "parameters.txt", lines)
    _write_parameters_pdf(output_directory / "parameters.pdf", lines)
    _write_split_delay_report(output_directory, command_lag_search)
    elapsed = time.perf_counter() - started
    result = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(arguments.config.expanduser().resolve()),
        "vehicle_model_path": str(vehicle_model.source_path),
        "parameter_prior_path": str(parameter_prior.source_path),
        "vehicle_model": vehicle_model.raw,
        "parameter_prior": parameter_prior.raw,
        "method": {
            "name": "deterministic_spline_dynamics",
            "description": (
                "pose-only continuous-time spline gradient matching with "
                "shared physical parameters and command lag"
            ),
            "uses_multiple_shooting_nodes": False,
            "uses_continuity_constraints": False,
            "uses_augmented_lagrangian": False,
            "sensor_channels_in_parameter_loss": False,
            "pose_role": "constructs fixed bag-local splines only",
            "spline_extrapolation_used": False,
            "spline_fit_boundaries_used_for_parameter_loss": False,
            "pose_spline_degree": 5,
            "rotation_spline_representation": (
                "normalized quaternion quintic B-spline"
            ),
            "default_physical_initialization": (
                "zero coordinate of the external vehicle-model 14-D chart"
            ),
            "command_mode_during_search": "quintic smoothstep ZOH",
            "command_mode_final": "strict ZOH",
        },
        "initial_estimate": {
            "source_kind": "vehicle_model_reference",
            "physical_coordinate": initial_physical_coordinate,
            "delay_seconds": initial_delay,
            "rotor_delay_seconds": initial_delay,
            "gimbal_delay_seconds": initial_gimbal_delay,
        },
        "settings": {
            "sample_step_seconds": arguments.sample_step,
            "integration_step_seconds": arguments.integration_step,
            "collocation_step_seconds": (
                config.spline.collocation_step_seconds
            ),
            "knot_spacing_candidates_seconds": (
                config.spline.knot_spacing_candidates_seconds
            ),
            "pose_spline_degree": 5,
            "boundary_exclusion_knot_spans_each_side": (
                config.spline.boundary_exclusion_knot_spans_each_side
            ),
            "boundary_exclusion_policy": (
                "parameter loss and residual wrench use only the interior; "
                "quintic half-support is excluded at both spline boundaries"
            ),
            "physical_coordinate_bounds": "unbounded",
            "spline_cv_folds": arguments.spline_cv_folds,
            "spline_cross_validation_block_seconds": (
                config.spline.cross_validation_block_seconds
            ),
            "maximum_spline_acceleration_m_per_s2": (
                arguments.maximum_spline_acceleration
            ),
            "maximum_spline_angular_acceleration_rad_per_s2": (
                arguments.maximum_spline_angular_acceleration
            ),
            "delay_bounds_seconds": arguments.delay_bounds,
            "smoothstep_width_fractions": (
                arguments.smoothstep_width_fractions
            ),
        },
        "bags": bag_sources,
        "smoothstep_stages": smooth_stage_payloads,
        "strict_zoh_polish": strict_payloads,
        "command_lag_search": command_lag_search,
        "selection": {
            "physical_parameter_names": PHYSICAL_PARAMETER_NAMES,
            "physical_coordinate": selected.physical_coordinate,
            "delay_seconds": selected.delay_seconds,
            "rotor_delay_seconds": selected.delay_seconds,
            "gimbal_delay_seconds": selected_gimbal_delay,
            "parameters": baseline._physical_parameters(selected_parameters),
            "inertia_principal_moments_kg_m2": (
                selected.evaluation.decoded.inertia_principal_moments
            ),
            "inertia_triangle_margin_kg_m2": (
                selected.evaluation.decoded.inertia_triangle_margin
            ),
            "joint_dynamics_loss": selected.evaluation.data_loss,
            "gaussian_prior_cost": selected.evaluation.prior_cost,
            "joint_objective_cost": _solution_cost(selected),
            "optimizer": selected.optimizer,
        },
        "bag_diagnostics": bag_payloads,
        "elapsed_seconds": elapsed,
        "outputs": {
            "data_dictionary_md": "DATA_DICTIONARY.md",
            "parameters_txt": "parameters.txt",
            "parameters_pdf": "parameters.pdf",
            "delay_profile_pdf": "delay_profile.pdf",
            "bags": bag_outputs,
        },
    }
    baseline._write_json(
        output_directory / "result.json", _json_sanitize(result)
    )
    print("wrote {}".format(output_directory / "result.json"), flush=True)
    print("elapsed {:.3f}s".format(elapsed), flush=True)
    return 0

# Reusable downstream implementation copied into this module from the
# 8be7473 spline estimator. These aliases are local implementation
# snapshots; there is no runtime spline-estimator dependency.
_SharedSplineDynamicsProblem = SplineDynamicsProblem
_shared_parameter_lines = _parameter_lines
_shared_run = run

SCHEMA = "grape-param-estim/minimal-deterministic-savgol-dynamics/v4"
OUTPUT_SUBDIRECTORY = "deterministic_savgol_dynamics"
DATA_DICTIONARY_SOURCE = Path(__file__).resolve().with_name(
    "deterministic_savgol_dynamics_data_dictionary.md"
)
ROTOR_DELAY_INDEX = PHYSICAL_DIMENSION
GIMBAL_DELAY_INDEX = PHYSICAL_DIMENSION + 1
GLOBAL_DIMENSION = PHYSICAL_DIMENSION + 2
DELAY_INDEX = ROTOR_DELAY_INDEX
candidate_payload = sg.candidate_payload

_ACTIVE_WINDOW_SECONDS: Optional[float] = None
_ACTIVE_ROTOR_PERIOD_SECONDS: Optional[float] = None
_ACTIVE_GIMBAL_PERIOD_SECONDS: Optional[float] = None
_ACTIVE_INITIAL_GIMBAL_DELAY_SECONDS: Optional[float] = None
_ACTIVE_BAGS: dict[str, BagSplineData] = {}
_ACTIVE_RAW_WRENCH_STATS: dict[str, dict[str, Any]] = {}
_ORIGINAL_PARAMETER_LINES = _shared_parameter_lines


def _window() -> float:
    if _ACTIVE_WINDOW_SECONDS is None:
        raise RuntimeError("Savitzky-Golay window is not configured")
    return float(_ACTIVE_WINDOW_SECONDS)


def _compatibility_config(path: Path) -> SplineEstimatorConfig:
    """Load only the bag configuration; spline settings are intentionally inert."""

    multi_config = multi.load_multi_bag_config(path.expanduser().resolve())
    # run still serializes these historical fields before the JSON is
    # rewritten below.  NaN is used deliberately instead of inventing values.
    settings = SplineSettings(
        knot_spacing_candidates_seconds=(_window(),),
        collocation_step_seconds=math.nan,
        boundary_exclusion_knot_spans_each_side=0.0,
        cross_validation_block_seconds=math.nan,
    )
    return SplineEstimatorConfig(
        multi_bag=multi_config,
        spline=settings,
    )


def load_spline_config(path: Path) -> SplineEstimatorConfig:
    """Compatibility name used by the existing confidence/dynamics code."""

    return _compatibility_config(path)


def _validate_arguments(
    arguments: argparse.Namespace,
    config: SplineEstimatorConfig,
    initial_delay: float,
) -> None:
    del config
    positive = (
        arguments.window_seconds,
        arguments.sample_step,
        arguments.integration_step,
        arguments.smooth_max_nfev,
        arguments.strict_max_nfev,
        arguments.gtol,
    )
    optional_tolerances = (arguments.ftol, arguments.xtol)
    ratio = arguments.sample_step / arguments.integration_step
    bounds = np.asarray(arguments.delay_bounds, dtype=float)
    widths = np.asarray(arguments.smoothstep_width_fractions, dtype=float)
    if (
        any(
            not np.isfinite(value) or value <= 0.0
            for value in positive
        )
        or any(
            value is not None
            and (not np.isfinite(value) or value <= 0.0)
            for value in optional_tolerances
        )
        or not np.isclose(
            ratio,
            round(ratio),
            atol=1.0e-12,
            rtol=0.0,
        )
        or bounds.shape != (2,)
        or np.any(~np.isfinite(bounds))
        or bounds[0] < 0.0
        or bounds[1] <= bounds[0]
        or not bounds[0] <= initial_delay <= bounds[1]
        or not bounds[0] <= float(arguments.initial_gimbal_delay) <= bounds[1]
        or widths.ndim != 1
        or widths.size == 0
        or np.any(~np.isfinite(widths))
        or np.any(widths <= 0.0)
    ):
        raise SystemExit("Savitzky-Golay dynamics settings are invalid")


def _build_bag_data(
    specification: Any,
    normalized_weight: float,
    flight: Any,
    initial_delay: float,
    settings: Any,
    arguments: argparse.Namespace,
    reference_parameters: Any,
    geometry: Any,
) -> BagSplineData:
    del settings

    direct = baseline.DirectShootingProblem(
        flight=flight,
        sample_step=arguments.sample_step,
        integration_step=arguments.integration_step,
        command_delay=initial_delay,
        prior_weight=0.0,
        reference_parameters=reference_parameters,
        geometry=geometry,
    )

    raw_time = np.asarray(flight.pose.times, dtype=float)
    if not sg.window_is_feasible(
        raw_time,
        arguments.window_seconds,
        degree=sg.POLYNOMIAL_DEGREE,
    ):
        minimum = sg.minimum_feasible_window_seconds(
            raw_time,
            degree=sg.POLYNOMIAL_DEGREE,
        )
        raise ValueError(
            "W={:.12g}s is too short for degree-5 local polynomial on bag {}; "
            "minimum data-supported W is {:.12g}s".format(
                arguments.window_seconds,
                specification.bag_id,
                minimum,
            )
        )

    selection = sg.select_pose_spline(
        time_axis=raw_time,
        sensor_position=np.asarray(flight.pose.positions, dtype=float),
        sensor_orientation_xyzw=np.asarray(
            flight.pose.orientations_xyzw,
            dtype=float,
        ),
        body_to_pose_sensor_rotation=direct.pose_body_to_sensor_rotation,
        knot_spacing_candidates_seconds=(float(arguments.window_seconds),),
        rotational_metric=(
            reference_parameters.inertia / reference_parameters.mass
        ),
    )

    # The parameter loss uses every raw pose timestamp for which the complete
    # centered W-second window exists.  The DirectShootingProblem support is
    # intersected only because its extrinsics/actuator model are reused
    # downstream; no direct.output_time sample is used to fit SG.
    support_start = float(direct.output_time[0])
    support_end = float(direct.output_time[-1])

    # Delay search queries u(t - tau).  Exclude any initial interval that
    # cannot be covered at the largest allowed delay instead of relying on
    # invalid ZOH extrapolation.
    maximum_delay = float(arguments.delay_bounds[1])
    support_start = max(
        support_start,
        float(flight.rotor_command.all_times[0]) + maximum_delay,
        float(flight.gimbal_command.all_times[0]) + maximum_delay,
    )

    collocation_time = selection.spline.centered_raw_times(
        support_start=support_start,
        support_end=support_end,
    )
    collocation = selection.spline.evaluate(collocation_time)
    initial_gimbal = baseline._linear_interpolate(
        flight.gimbal_position.times,
        flight.gimbal_position.values,
        np.asarray((collocation_time[0],), dtype=float),
    )[0]

    bag = BagSplineData(
        specification=specification,
        normalized_weight=float(normalized_weight),
        flight=flight,
        direct_problem=direct,
        spline_selection=selection,
        collocation=collocation,
        rotor_history=QuinticSmoothZoh(
            flight.rotor_command.all_times,
            flight.rotor_command.all_values,
        ),
        gimbal_history=QuinticSmoothZoh(
            flight.gimbal_command.all_times,
            flight.gimbal_command.all_values,
        ),
        initial_gimbal=initial_gimbal,
        # Historical field retained only for ABI compatibility.  It has no
        # meaning in the SG estimator and is removed from emitted JSON.
        boundary_exclusion_knot_spans_each_side=0.0,
    )
    _ACTIVE_BAGS[str(specification.bag_id)] = bag
    return bag




class SplineDynamicsProblem(_SharedSplineDynamicsProblem):
    """Acceleration-residual SG problem with separate rotor/gimbal lags."""


    def _actuator_series(
        self,
        bag,
        decoded,
        parameter_jacobian,
        dimension,
        smooth_mode,
        width_fraction,
    ):
        """Evaluate actuator state at the same times as the SG dynamics samples.

        Rotor thrust is algebraic because this estimator fixes the thrust time
        constant to zero. Therefore thrust at an SG center t_i is the clipped
        delayed command evaluated at t_i itself, not a state left over from the
        previous integration interval.

        Gimbal angle retains the physical rate limit. Its initial state is the
        measured gimbal_position interpolated at the first SG center. For strict
        ZOH, propagation is split at every delayed recorded gimbal-command switch
        so the rate-limited state at each SG center is exact for the declared
        piecewise-constant command model.
        """

        time_axis = np.asarray(bag.collocation_time, dtype=float)
        if (
            time_axis.ndim != 1
            or time_axis.size < 1
            or np.any(~np.isfinite(time_axis))
            or np.any(np.diff(time_axis) <= 0.0)
        ):
            raise ValueError("SG actuator evaluation times are invalid")

        rotor_delay = float(self._active_rotor_delay)
        gimbal_delay = float(self._active_gimbal_delay)
        actuator_parameters = decoded.actuator_parameters
        if (
            actuator_parameters.thrust_time_constant != 0.0
            or actuator_parameters.gimbal_time_constant != 0.0
        ):
            raise RuntimeError(
                "SG lag timing assumes zero thrust/gimbal time constants"
            )

        count = time_axis.size
        thrust = np.empty((count, 4), dtype=float)
        gimbal = np.empty((count, 4), dtype=float)
        state_jacobian = np.zeros(
            (count, 8, dimension),
            dtype=float,
        )

        def rotor_state_at(time_value):
            if smooth_mode:
                evaluation = bag.rotor_history.evaluate(
                    float(time_value),
                    rotor_delay,
                    float(width_fraction),
                )
                raw = np.asarray(evaluation.value, dtype=float)
                derivative = np.asarray(
                    evaluation.delay_derivative,
                    dtype=float,
                )
                free = (
                    (raw > actuator_parameters.minimum_thrust)
                    & (raw < actuator_parameters.maximum_thrust)
                )
                clipped = np.clip(
                    raw,
                    actuator_parameters.minimum_thrust,
                    actuator_parameters.maximum_thrust,
                )
                return clipped, np.where(free, derivative, 0.0)

            raw = bag.rotor_history.exact_zoh(
                float(time_value),
                rotor_delay,
            )
            clipped = np.clip(
                raw,
                actuator_parameters.minimum_thrust,
                actuator_parameters.maximum_thrust,
            )
            return clipped, np.zeros(4, dtype=float)

        # Zero thrust time constant makes thrust a pointwise algebraic state.
        for index, sample_time in enumerate(time_axis):
            rotor_value, rotor_derivative = rotor_state_at(sample_time)
            thrust[index] = rotor_value
            if smooth_mode:
                state_jacobian[
                    index,
                    :4,
                    ROTOR_DELAY_INDEX,
                ] = rotor_derivative

        initial_gimbal = np.asarray(
            bag.initial_gimbal,
            dtype=float,
        )
        if (
            initial_gimbal.shape != (4,)
            or np.any(~np.isfinite(initial_gimbal))
        ):
            raise ValueError("measured initial gimbal state is invalid")

        # Condition on the measured actuator state at the first SG center.
        # It is data, not a delayed command evaluation, so its lag derivative
        # is exactly zero.
        current_state = ActuatorState(
            thrust=thrust[0].copy(),
            gimbal_angle=initial_gimbal.copy(),
        )
        current_sensitivity = np.zeros(
            (8, dimension),
            dtype=float,
        )
        gimbal[0] = current_state.gimbal_angle

        shifted_gimbal_switches = (
            np.asarray(bag.gimbal_history.times[1:], dtype=float)
            + gimbal_delay
        )

        for index in range(count - 1):
            left = float(time_axis[index])
            right = float(time_axis[index + 1])

            if smooth_mode:
                # Continuous smooth command: two midpoint substeps remove the
                # previous one-midpoint sample-phase error while preserving the
                # analytic lag sensitivity.
                middle = 0.5 * (left + right)
                subintervals = (
                    (left, middle),
                    (middle, right),
                )
                for sub_left, sub_right in subintervals:
                    dt = float(sub_right - sub_left)
                    if dt <= 0.0:
                        continue
                    query = 0.5 * (sub_left + sub_right)
                    rotor_evaluation = bag.rotor_history.evaluate(
                        query,
                        rotor_delay,
                        float(width_fraction),
                    )
                    gimbal_evaluation = bag.gimbal_history.evaluate(
                        query,
                        gimbal_delay,
                        float(width_fraction),
                    )
                    command = smooth._command(
                        rotor_evaluation.value,
                        gimbal_evaluation.value,
                    )
                    command_sensitivity = np.zeros(
                        (8, dimension),
                        dtype=float,
                    )
                    command_sensitivity[
                        :4,
                        ROTOR_DELAY_INDEX,
                    ] = rotor_evaluation.delay_derivative
                    command_sensitivity[
                        4:,
                        GIMBAL_DELAY_INDEX,
                    ] = gimbal_evaluation.delay_derivative
                    (
                        current_state,
                        current_sensitivity,
                    ) = strict._actuator_step_with_sensitivity(
                        current_state,
                        current_sensitivity,
                        command,
                        decoded,
                        parameter_jacobian,
                        dt,
                        command_sensitivity,
                    )
            else:
                # A source switch at t_cmd appears at t_cmd + tau_g in
                # evaluation time. Split exactly at those delayed switch times.
                first = int(
                    np.searchsorted(
                        shifted_gimbal_switches,
                        left,
                        side="right",
                    )
                )
                last = int(
                    np.searchsorted(
                        shifted_gimbal_switches,
                        right,
                        side="left",
                    )
                )
                internal = shifted_gimbal_switches[first:last]
                boundaries = np.concatenate(
                    (
                        np.asarray((left,), dtype=float),
                        internal,
                        np.asarray((right,), dtype=float),
                    )
                )
                for sub_left, sub_right in zip(
                    boundaries[:-1],
                    boundaries[1:],
                ):
                    dt = float(sub_right - sub_left)
                    if dt <= 0.0:
                        continue
                    query = 0.5 * (sub_left + sub_right)
                    command = smooth._command(
                        bag.rotor_history.exact_zoh(
                            query,
                            rotor_delay,
                        ),
                        bag.gimbal_history.exact_zoh(
                            query,
                            gimbal_delay,
                        ),
                    )
                    current_state = advance_actuators(
                        current_state,
                        command,
                        actuator_parameters,
                        dt,
                    )

            gimbal[index + 1] = current_state.gimbal_angle
            if smooth_mode:
                state_jacobian[
                    index + 1,
                    4:,
                    :,
                ] = current_sensitivity[4:]

        arrays = (thrust, gimbal, state_jacobian)
        if any(np.any(~np.isfinite(value)) for value in arrays):
            raise FloatingPointError(
                "SG actuator series contains non-finite values"
            )
        return thrust, gimbal, state_jacobian


    def evaluate_smooth(self, coordinate: Sequence[float], width_fraction: float):
        value = np.asarray(coordinate, dtype=float)
        if value.shape != (GLOBAL_DIMENSION,) or np.any(~np.isfinite(value)):
            raise ValueError("smooth SG dynamics coordinate must be 16-D")
        self._active_rotor_delay = float(value[ROTOR_DELAY_INDEX])
        self._active_gimbal_delay = float(value[GIMBAL_DELAY_INDEX])
        if self._active_rotor_delay < 0.0 or self._active_gimbal_delay < 0.0:
            raise ValueError("command lags must be non-negative")
        return self._evaluate_joint(
            physical_coordinate=value[:PHYSICAL_DIMENSION],
            delay=self._active_rotor_delay,
            dimension=GLOBAL_DIMENSION,
            smooth_mode=True,
            width_fraction=float(width_fraction),
        )

    def evaluate_strict(self, physical_coordinate, rotor_delay: float, gimbal_delay: Optional[float] = None):
        physical = np.asarray(physical_coordinate, dtype=float)
        rotor = float(rotor_delay)
        gimbal_value = rotor if gimbal_delay is None else float(gimbal_delay)
        if (
            physical.shape != (PHYSICAL_DIMENSION,)
            or np.any(~np.isfinite(physical))
            or not np.isfinite(rotor)
            or not np.isfinite(gimbal_value)
            or rotor < 0.0
            or gimbal_value < 0.0
        ):
            raise ValueError("strict SG dynamics coordinate is invalid")
        self._active_rotor_delay = rotor
        self._active_gimbal_delay = gimbal_value
        return self._evaluate_joint(
            physical_coordinate=physical,
            delay=rotor,
            dimension=PHYSICAL_DIMENSION,
            smooth_mode=False,
            width_fraction=1.0,
        )


def _jacobian_spectrum(jacobian: np.ndarray) -> dict[str, Any]:
    value = np.asarray(jacobian, dtype=float)
    singular = np.linalg.svd(value, compute_uv=False)
    if singular.size == 0:
        return {
            "singular_values": singular,
            "numerical_rank": 0,
            "numerical_tolerance": 0.0,
            "condition_number_nonzero_subspace": None,
        }
    tolerance = (
        max(value.shape)
        * np.finfo(float).eps
        * float(singular[0])
    )
    positive = singular > tolerance
    rank = int(np.count_nonzero(positive))
    condition = (
        None
        if rank == 0
        else float(singular[0] / singular[rank - 1])
    )
    return {
        "singular_values": singular,
        "numerical_rank": rank,
        "numerical_tolerance": tolerance,
        "condition_number_nonzero_subspace": condition,
    }


def _objective_point_diagnostics(evaluation: Any) -> dict[str, Any]:
    residual = np.asarray(evaluation.residual, dtype=float)
    jacobian = np.asarray(evaluation.jacobian, dtype=float)
    gradient = jacobian.T @ residual
    result = {
        "cost": 0.5 * float(residual @ residual),
        "residual_l2_norm": float(np.linalg.norm(residual)),
        "gradient_l2_norm": float(np.linalg.norm(gradient)),
        "gradient_inf_norm": float(np.linalg.norm(gradient, ord=np.inf)),
        "jacobian": _jacobian_spectrum(jacobian),
    }
    if gradient.size == GLOBAL_DIMENSION:
        result["command_lag_gradient"] = {
            "rotor": float(gradient[ROTOR_DELAY_INDEX]),
            "gimbal": float(gradient[GIMBAL_DELAY_INDEX]),
        }
    return result



def _directional_jacobian_checks(
    evaluator,
    coordinate,
    evaluation,
    lower,
    upper,
):
    """Check analytic Jacobian directions without collapsing at active bounds."""

    x = np.asarray(coordinate, dtype=float)
    residual = np.asarray(evaluation.residual, dtype=float)
    jacobian = np.asarray(evaluation.jacobian, dtype=float)
    lower_value = np.asarray(lower, dtype=float)
    upper_value = np.asarray(upper, dtype=float)
    gradient = jacobian.T @ residual

    directions = []
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm > 0.0:
        directions.append(
            ("negative_gradient", -gradient / gradient_norm)
        )
    if x.size == GLOBAL_DIMENSION:
        rotor_direction = np.zeros(x.size, dtype=float)
        rotor_direction[ROTOR_DELAY_INDEX] = 1.0
        gimbal_direction = np.zeros(x.size, dtype=float)
        gimbal_direction[GIMBAL_DELAY_INDEX] = 1.0
        directions.extend(
            (
                ("rotor_delay_seconds", rotor_direction),
                ("gimbal_delay_seconds", gimbal_direction),
            )
        )
    if not directions:
        direction = np.zeros(x.size, dtype=float)
        direction[0] = 1.0
        directions.append(
            (
                "coordinate:" + PHYSICAL_PARAMETER_NAMES[0],
                direction,
            )
        )

    checks = []
    scale = max(1.0, float(np.linalg.norm(x)))
    base_step = np.cbrt(np.finfo(float).eps) * scale
    minimum_usable_step = (
        1024.0 * np.finfo(float).eps * scale
    )

    for label, direction in directions:
        plus_limit = math.inf
        minus_limit = math.inf
        positive = direction > 0.0
        negative = direction < 0.0

        if np.any(positive):
            plus_limit = min(
                plus_limit,
                float(
                    np.min(
                        (upper_value[positive] - x[positive])
                        / direction[positive]
                    )
                ),
            )
            minus_limit = min(
                minus_limit,
                float(
                    np.min(
                        (x[positive] - lower_value[positive])
                        / direction[positive]
                    )
                ),
            )
        if np.any(negative):
            plus_limit = min(
                plus_limit,
                float(
                    np.min(
                        (x[negative] - lower_value[negative])
                        / (-direction[negative])
                    )
                ),
            )
            minus_limit = min(
                minus_limit,
                float(
                    np.min(
                        (upper_value[negative] - x[negative])
                        / (-direction[negative])
                    )
                ),
            )

        plus_step = (
            min(base_step, 0.5 * plus_limit)
            if plus_limit > minimum_usable_step
            else 0.0
        )
        minus_step = (
            min(base_step, 0.5 * minus_limit)
            if minus_limit > minimum_usable_step
            else 0.0
        )

        analytic = jacobian @ direction

        if (
            plus_step >= base_step
            and minus_step >= base_step
        ):
            step = base_step
            plus = evaluator(x + step * direction).residual
            minus = evaluator(x - step * direction).residual
            finite = (plus - minus) / (2.0 * step)
            scheme = "central"
        elif plus_step >= minus_step and plus_step > 0.0:
            step = plus_step
            finite = (
                evaluator(x + step * direction).residual
                - residual
            ) / step
            scheme = "forward"
        elif minus_step > 0.0:
            step = minus_step
            finite = (
                residual
                - evaluator(x - step * direction).residual
            ) / step
            scheme = "backward"
        else:
            checks.append(
                {
                    "direction": label,
                    "scheme": "unavailable_at_active_bounds",
                    "step": None,
                    "analytic_l2_norm": float(
                        np.linalg.norm(analytic)
                    ),
                    "finite_difference_l2_norm": None,
                    "difference_l2_norm": None,
                    "relative_error": None,
                }
            )
            continue

        difference = np.asarray(finite) - analytic
        denominator = max(
            float(np.linalg.norm(finite)),
            float(np.linalg.norm(analytic)),
            math.sqrt(np.finfo(float).eps),
        )
        checks.append(
            {
                "direction": label,
                "scheme": scheme,
                "step": float(step),
                "analytic_l2_norm": float(
                    np.linalg.norm(analytic)
                ),
                "finite_difference_l2_norm": float(
                    np.linalg.norm(finite)
                ),
                "difference_l2_norm": float(
                    np.linalg.norm(difference)
                ),
                "relative_error": float(
                    np.linalg.norm(difference) / denominator
                ),
            }
        )
    return checks


def _optimizer_diagnostics(
    evaluator: Any,
    initial: np.ndarray,
    result: Any,
    final_evaluation: Any,
    lower: np.ndarray,
    upper: np.ndarray,
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    initial_evaluation = evaluator(np.asarray(initial, dtype=float))
    initial_point = _objective_point_diagnostics(initial_evaluation)
    final_point = _objective_point_diagnostics(final_evaluation)
    checks = _directional_jacobian_checks(
        evaluator,
        np.asarray(result.x, dtype=float),
        final_evaluation,
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )
    finite_errors = [
        float(item["relative_error"])
        for item in checks
        if item.get("relative_error") is not None
    ]
    return {
        "tolerances": {
            "ftol": arguments.ftol,
            "xtol": arguments.xtol,
            "gtol": arguments.gtol,
        },
        "initial": initial_point,
        "final": final_point,
        "coordinate_step_l2_norm": float(
            np.linalg.norm(np.asarray(result.x, dtype=float) - np.asarray(initial, dtype=float))
        ),
        "cost_reduction": float(initial_point["cost"] - final_point["cost"]),
        "gradient_tolerance_satisfied": bool(
            float(result.optimality) <= float(arguments.gtol)
        ),
        "finite_difference_directional_checks": checks,
        "finite_difference_max_relative_error": (
            None if not finite_errors else max(finite_errors)
        ),
    }



def _solve_smooth(
    problem: SplineDynamicsProblem,
    initial: np.ndarray,
    width_fraction: float,
    lower: np.ndarray,
    upper: np.ndarray,
    arguments: argparse.Namespace,
) -> tuple[np.ndarray, Any, Mapping[str, Any]]:
    evaluator = lambda value: problem.evaluate_smooth(value, width_fraction)
    objective = _CachedObjective(evaluator)
    result = least_squares(
        objective.residual,
        initial,
        jac=objective.jacobian,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        loss="linear",
        ftol=arguments.ftol,
        xtol=arguments.xtol,
        gtol=arguments.gtol,
        max_nfev=arguments.smooth_max_nfev,
        verbose=0,
    )
    final_evaluation = problem.evaluate_smooth(result.x, width_fraction)
    payload = _optimizer_payload(result)
    payload["diagnostics"] = _optimizer_diagnostics(
        evaluator,
        np.asarray(initial, dtype=float),
        result,
        final_evaluation,
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
        arguments,
    )
    return result.x, final_evaluation, payload


def _solve_strict(
    problem: SplineDynamicsProblem,
    initial: np.ndarray,
    delay: float,
    lower: np.ndarray,
    upper: np.ndarray,
    arguments: argparse.Namespace,
) -> DynamicsSolution:
    evaluator = lambda value: problem.evaluate_strict(value, delay)
    objective = _CachedObjective(evaluator)
    result = least_squares(
        objective.residual,
        initial,
        jac=objective.jacobian,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        loss="linear",
        ftol=arguments.ftol,
        xtol=arguments.xtol,
        gtol=arguments.gtol,
        max_nfev=arguments.strict_max_nfev,
        verbose=0,
    )
    final_evaluation = problem.evaluate_strict(result.x, delay)
    payload = _optimizer_payload(result)
    payload["diagnostics"] = _optimizer_diagnostics(
        evaluator,
        np.asarray(initial, dtype=float),
        result,
        final_evaluation,
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
        arguments,
    )
    return DynamicsSolution(
        physical_coordinate=np.asarray(result.x, dtype=float).copy(),
        delay_seconds=float(delay),
        evaluation=final_evaluation,
        optimizer=payload,
    )


def _timestamp_interval_statistics(time_axis: Sequence[float]) -> dict[str, Any]:
    value = np.asarray(time_axis, dtype=float)
    differences = np.diff(value)
    positive = differences[np.isfinite(differences) & (differences > 0.0)]
    if positive.size == 0:
        return {
            "sample_count": int(value.size),
            "positive_interval_count": 0,
            "minimum_seconds": None,
            "median_seconds": None,
            "mean_seconds": None,
            "maximum_seconds": None,
            "median_frequency_hz": None,
        }
    median = float(np.median(positive))
    return {
        "sample_count": int(value.size),
        "positive_interval_count": int(positive.size),
        "minimum_seconds": float(np.min(positive)),
        "median_seconds": median,
        "mean_seconds": float(np.mean(positive)),
        "maximum_seconds": float(np.max(positive)),
        "median_frequency_hz": float(1.0 / median),
    }


def _recorded_command_periods(config_path: Path):
    config = multi.load_multi_bag_config(config_path.expanduser().resolve())
    rotor_periods = []
    gimbal_periods = []
    for specification in config.bags:
        flight = load_flight_data(
            str(specification.path), start_local=specification.start, end_local=specification.end,
            include_fc_specific_force=True, compute_sha256=False,
        )
        for target, axis in ((rotor_periods, flight.rotor_command.all_times), (gimbal_periods, flight.gimbal_command.all_times)):
            dt = np.diff(np.asarray(axis, dtype=float))
            positive = dt[np.isfinite(dt) & (dt > 0.0)]
            if positive.size == 0:
                raise ValueError("recorded command timestamps have no positive interval")
            target.append(float(np.median(positive)))
    return float(np.median(rotor_periods)), float(np.median(gimbal_periods))


def _resolve_lag_defaults(arguments):
    global _ACTIVE_ROTOR_PERIOD_SECONDS, _ACTIVE_GIMBAL_PERIOD_SECONDS, _ACTIVE_INITIAL_GIMBAL_DELAY_SECONDS
    rotor_period, gimbal_period = _recorded_command_periods(arguments.config)
    _ACTIVE_ROTOR_PERIOD_SECONDS = rotor_period
    _ACTIVE_GIMBAL_PERIOD_SECONDS = gimbal_period
    rotor_initial = rotor_period if arguments.initial_rotor_delay is None else float(arguments.initial_rotor_delay)
    gimbal_initial = gimbal_period if arguments.initial_gimbal_delay is None else float(arguments.initial_gimbal_delay)
    _ACTIVE_INITIAL_GIMBAL_DELAY_SECONDS = gimbal_initial
    arguments.initial_delay = rotor_initial
    arguments.initial_gimbal_delay = gimbal_initial


def _solve_smooth_pair(problem, initial, width, lower, upper, arguments):
    evaluator = lambda value: problem.evaluate_smooth(value, width)
    objective = _CachedObjective(evaluator)
    result = least_squares(
        objective.residual, initial, jac=objective.jacobian,
        bounds=(lower, upper), method="trf", x_scale="jac", loss="linear",
        ftol=arguments.ftol, xtol=arguments.xtol, gtol=arguments.gtol,
        max_nfev=arguments.smooth_max_nfev, verbose=0,
    )
    evaluation = evaluator(result.x)
    payload = _optimizer_payload(result)
    payload["diagnostics"] = _optimizer_diagnostics(
        evaluator, np.asarray(initial), result, evaluation,
        np.asarray(lower), np.asarray(upper), arguments,
    )
    return result.x, evaluation, payload


def _solve_strict_pair(problem, initial, rotor_delay, gimbal_delay, lower, upper, arguments):
    evaluator = lambda value: problem.evaluate_strict(value, rotor_delay, gimbal_delay)
    objective = _CachedObjective(evaluator)
    result = least_squares(
        objective.residual, initial, jac=objective.jacobian,
        bounds=(lower, upper), method="trf", x_scale="jac", loss="linear",
        ftol=arguments.ftol, xtol=arguments.xtol, gtol=arguments.gtol,
        max_nfev=arguments.strict_max_nfev, verbose=0,
    )
    evaluation = evaluator(result.x)
    payload = _optimizer_payload(result)
    payload["diagnostics"] = _optimizer_diagnostics(
        evaluator, np.asarray(initial), result, evaluation,
        np.asarray(lower), np.asarray(upper), arguments,
    )
    return DynamicsSolution(
        physical_coordinate=np.asarray(result.x).copy(), delay_seconds=float(rotor_delay),
        gimbal_delay_seconds=float(gimbal_delay), evaluation=evaluation, optimizer=payload,
    )


def _strict_pair_screen(problem, physical, center_rotor, center_gimbal, rotor_period, gimbal_period, delay_bounds):
    lower, upper = map(float, delay_bounds)
    rotor_values = {round(float(np.clip(center_rotor + d, lower, upper)), 12) for d in (-rotor_period, 0.0, rotor_period)}
    gimbal_values = {round(float(np.clip(center_gimbal + d, lower, upper)), 12) for d in (-gimbal_period, 0.0, gimbal_period)}
    costs = {}
    expansions = []
    def evaluate_new():
        for r in sorted(rotor_values):
            for g in sorted(gimbal_values):
                if (r, g) in costs:
                    continue
                ev = problem.evaluate_strict(physical, r, g)
                costs[(r, g)] = 0.5 * float(ev.residual @ ev.residual)
    while True:
        evaluate_new()
        best = min(costs, key=lambda key: (costs[key], key[0], key[1]))
        rs = sorted(rotor_values); gs = sorted(gimbal_values)
        additions = []
        if best[0] == rs[0] and best[0] > lower:
            value = round(max(lower, best[0] - rotor_period), 12)
            if value not in rotor_values:
                rotor_values.add(value); additions.append({"axis": "rotor", "delay_seconds": value})
        elif best[0] == rs[-1] and best[0] < upper:
            value = round(min(upper, best[0] + rotor_period), 12)
            if value not in rotor_values:
                rotor_values.add(value); additions.append({"axis": "rotor", "delay_seconds": value})
        if best[1] == gs[0] and best[1] > lower:
            value = round(max(lower, best[1] - gimbal_period), 12)
            if value not in gimbal_values:
                gimbal_values.add(value); additions.append({"axis": "gimbal", "delay_seconds": value})
        elif best[1] == gs[-1] and best[1] < upper:
            value = round(min(upper, best[1] + gimbal_period), 12)
            if value not in gimbal_values:
                gimbal_values.add(value); additions.append({"axis": "gimbal", "delay_seconds": value})
        if not additions:
            break
        expansions.append({"best_before_expansion": {"rotor_delay_seconds": best[0], "gimbal_delay_seconds": best[1], "cost": costs[best]}, "added": additions})
    evaluate_new()
    best = min(costs, key=lambda key: (costs[key], key[0], key[1]))
    return {
        "best_pair": best,
        "best_cost": costs[best],
        "rotor_range_seconds": (min(rotor_values), max(rotor_values)),
        "gimbal_range_seconds": (min(gimbal_values), max(gimbal_values)),
        "expansions": expansions,
        "candidates": [
            {"rotor_delay_seconds": r, "gimbal_delay_seconds": g, "cost": costs[(r,g)], "selected": (r,g) == best}
            for r in sorted(rotor_values) for g in sorted(gimbal_values)
        ],
    }



def _split_command_lag_search(
    problem,
    initial_physical,
    physical_lower,
    physical_upper,
    arguments,
):
    rotor_period = float(_ACTIVE_ROTOR_PERIOD_SECONDS)
    gimbal_period = float(_ACTIVE_GIMBAL_PERIOD_SECONDS)
    lower_lag, upper_lag = map(float, arguments.delay_bounds)

    coordinate = np.concatenate(
        (
            initial_physical,
            np.asarray(
                (
                    arguments.initial_delay,
                    arguments.initial_gimbal_delay,
                )
            ),
        )
    )
    lower = np.concatenate(
        (
            physical_lower,
            np.asarray((lower_lag, lower_lag)),
        )
    )
    upper = np.concatenate(
        (
            physical_upper,
            np.asarray((upper_lag, upper_lag)),
        )
    )

    smooth_stages = []
    for index, width in enumerate(
        arguments.smoothstep_width_fractions
    ):
        print(
            "smooth lag {}/{}: half-width={} publish periods".format(
                index + 1,
                len(arguments.smoothstep_width_fractions),
                width,
            ),
            flush=True,
        )
        coordinate, evaluation, optimizer = _solve_smooth_pair(
            problem,
            coordinate,
            float(width),
            lower,
            upper,
            arguments,
        )
        smooth_stages.append(
            {
                "half_width_period_multiplier": float(width),
                "rotor_transition_half_width_seconds": (
                    float(width) * rotor_period
                ),
                "gimbal_transition_half_width_seconds": (
                    float(width) * gimbal_period
                ),
                "objective_cost": 0.5
                * float(evaluation.residual @ evaluation.residual),
                "data_loss": evaluation.data_loss,
                "prior_cost": evaluation.prior_cost,
                "physical_coordinate": coordinate[
                    :PHYSICAL_DIMENSION
                ],
                "rotor_delay_seconds": float(
                    coordinate[ROTOR_DELAY_INDEX]
                ),
                "gimbal_delay_seconds": float(
                    coordinate[GIMBAL_DELAY_INDEX]
                ),
                "optimizer": optimizer,
            }
        )
        print(
            "  rotor={:.6f}s gimbal={:.6f}s cost={:.9g}".format(
                coordinate[ROTOR_DELAY_INDEX],
                coordinate[GIMBAL_DELAY_INDEX],
                smooth_stages[-1]["objective_cost"],
            ),
            flush=True,
        )

    physical = coordinate[:PHYSICAL_DIMENSION].copy()
    rotor = float(coordinate[ROTOR_DELAY_INDEX])
    gimbal = float(coordinate[GIMBAL_DELAY_INDEX])

    profile = []
    visited_pairs = set()
    refined_solutions = []
    termination = None
    maximum_iterations = 32

    for iteration_index in range(maximum_iterations):
        screen = _strict_pair_screen(
            problem,
            physical,
            rotor,
            gimbal,
            rotor_period,
            gimbal_period,
            arguments.delay_bounds,
        )
        best_rotor, best_gimbal = screen["best_pair"]
        current_pair = (
            round(float(best_rotor), 12),
            round(float(best_gimbal), 12),
        )
        visited_pairs.add(current_pair)

        solution = _solve_strict_pair(
            problem,
            physical,
            best_rotor,
            best_gimbal,
            physical_lower,
            physical_upper,
            arguments,
        )
        refined_solutions.append(solution)

        verify = _strict_pair_screen(
            problem,
            solution.physical_coordinate,
            best_rotor,
            best_gimbal,
            rotor_period,
            gimbal_period,
            arguments.delay_bounds,
        )
        profile.append(
            {
                "iteration": iteration_index + 1,
                "screening": screen,
                "refined_solution": {
                    "rotor_delay_seconds": best_rotor,
                    "gimbal_delay_seconds": best_gimbal,
                    "objective_cost": _solution_cost(solution),
                    "physical_coordinate": (
                        solution.physical_coordinate
                    ),
                    "optimizer": solution.optimizer,
                },
                "post_refinement_screening": verify,
            }
        )

        next_rotor, next_gimbal = verify["best_pair"]
        next_pair = (
            round(float(next_rotor), 12),
            round(float(next_gimbal), 12),
        )

        if (
            np.isclose(
                next_rotor,
                best_rotor,
                atol=5.0e-13,
                rtol=0.0,
            )
            and np.isclose(
                next_gimbal,
                best_gimbal,
                atol=5.0e-13,
                rtol=0.0,
            )
        ):
            selected = solution
            termination = "fixed_point"
            break

        if next_pair in visited_pairs:
            selected = min(
                refined_solutions,
                key=_solution_cost,
            )
            termination = "cycle_guard_best_refined_solution"
            break

        physical = solution.physical_coordinate.copy()
        rotor = float(next_rotor)
        gimbal = float(next_gimbal)
    else:
        selected = min(
            refined_solutions,
            key=_solution_cost,
        )
        termination = "maximum_iteration_guard_best_refined_solution"

    if not profile:
        raise RuntimeError("strict split-lag search produced no iteration")

    final_candidates = []
    for item in profile[-1][
        "post_refinement_screening"
    ]["candidates"]:
        value = dict(item)
        value["selected"] = bool(
            np.isclose(
                value["rotor_delay_seconds"],
                selected.delay_seconds,
                atol=5.0e-13,
                rtol=0.0,
            )
            and np.isclose(
                value["gimbal_delay_seconds"],
                selected.gimbal_delay_seconds,
                atol=5.0e-13,
                rtol=0.0,
            )
        )
        final_candidates.append(value)

    selected_rotor = float(selected.delay_seconds)
    selected_gimbal = float(selected.gimbal_delay_seconds)
    boundary_proximity = {
        "rotor_distance_to_lower_seconds": (
            selected_rotor - lower_lag
        ),
        "rotor_distance_to_upper_seconds": (
            upper_lag - selected_rotor
        ),
        "gimbal_distance_to_lower_seconds": (
            selected_gimbal - lower_lag
        ),
        "gimbal_distance_to_upper_seconds": (
            upper_lag - selected_gimbal
        ),
        "rotor_within_one_publish_period_of_bound": bool(
            selected_rotor - lower_lag <= rotor_period
            or upper_lag - selected_rotor <= rotor_period
        ),
        "gimbal_within_one_publish_period_of_bound": bool(
            selected_gimbal - lower_lag <= gimbal_period
            or upper_lag - selected_gimbal <= gimbal_period
        ),
    }

    return {
        "schema": SCHEMA + "/command-lag-search/v2",
        "recorded_command_period_seconds": {
            "rotor": rotor_period,
            "gimbal": gimbal_period,
        },
        "initial_delay_seconds": {
            "rotor": float(arguments.initial_delay),
            "gimbal": float(arguments.initial_gimbal_delay),
        },
        "delay_bounds_seconds": (lower_lag, upper_lag),
        "actuator_timing_policy": {
            "rotor": (
                "zero-time-constant thrust evaluated at each SG "
                "sample time using the delayed command"
            ),
            "gimbal_initial_state": (
                "measured gimbal_position at the first SG sample"
            ),
            "gimbal_strict_propagation": (
                "rate-limited propagation split at every delayed "
                "recorded gimbal-command switch"
            ),
        },
        "smooth_surrogate": {
            "representation": (
                "sum of overlapping quintic-smoothed ZOH jumps"
            ),
            "half_width_period_multipliers": list(
                arguments.smoothstep_width_fractions
            ),
            "gimbal_integration": (
                "two midpoint substeps per SG interval"
            ),
        },
        "smooth_stages": smooth_stages,
        "smooth_result": {
            "rotor_delay_seconds": float(
                coordinate[ROTOR_DELAY_INDEX]
            ),
            "gimbal_delay_seconds": float(
                coordinate[GIMBAL_DELAY_INDEX]
            ),
        },
        "strict_zoh": {
            "grid_step_seconds": {
                "rotor": rotor_period,
                "gimbal": gimbal_period,
            },
            "profile_iterations": profile,
            "termination": termination,
            "maximum_iterations": maximum_iterations,
        },
        "strict_candidates": final_candidates,
        "selected_solution": selected,
        "selected": {
            "rotor_delay_seconds": selected_rotor,
            "gimbal_delay_seconds": selected_gimbal,
            "objective_cost": _solution_cost(selected),
        },
        "selected_boundary_proximity": boundary_proximity,
    }


def _write_split_delay_report(output_directory, search):
    payload = {key: value for key, value in search.items() if key != "selected_solution"}
    (output_directory / "delay_profile.json").write_text(
        json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    periods = search["recorded_command_period_seconds"]
    selected = search["selected"]
    lines = [
        "Savitzky-Golay split command-lag search", "",
        f"rotor median publish period [s]: {periods['rotor']:.12g}",
        f"gimbal median publish period [s]: {periods['gimbal']:.12g}",
        f"initial lags [s]: {search['initial_delay_seconds']}",
        f"smooth half-width multipliers: {search['smooth_surrogate']['half_width_period_multipliers']}", "",
    ]
    for stage in search["smooth_stages"]:
        lines.append(
            "smooth width={:.6g}: rotor={:.9g}s gimbal={:.9g}s cost={:.12g} lag_gradient={} FD={}".format(
                stage["half_width_period_multiplier"], stage["rotor_delay_seconds"], stage["gimbal_delay_seconds"], stage["objective_cost"],
                stage["optimizer"]["diagnostics"]["final"].get("command_lag_gradient"),
                stage["optimizer"]["diagnostics"]["finite_difference_directional_checks"],
            )
        )
    lines.extend(["", f"strict profile iterations: {len(search['strict_zoh']['profile_iterations'])}"])
    for item in search["strict_zoh"]["profile_iterations"]:
        screen = item["screening"]
        lines.append(f"iteration {item['iteration']}: best={screen['best_pair']} cost={screen['best_cost']:.12g} rotor_range={screen['rotor_range_seconds']} gimbal_range={screen['gimbal_range_seconds']} expansions={screen['expansions']}")
    lines.extend([
        "",
        "selected rotor lag {:.12g} s ({:.6g} publish periods)".format(selected["rotor_delay_seconds"], selected["rotor_delay_seconds"] / periods["rotor"]),
        "selected gimbal lag {:.12g} s ({:.6g} publish periods)".format(selected["gimbal_delay_seconds"], selected["gimbal_delay_seconds"] / periods["gimbal"]),
        f"selected objective cost {selected['objective_cost']:.12g}",
    ])
    strict._write_text(output_directory / "delay_profile.txt", lines)
    _write_parameters_pdf(output_directory / "delay_profile.pdf", lines)


def _parameter_lines(
    selected: Any,
    initial_delay: float,
    bags: Sequence[Any],
    bag_payloads: Sequence[Mapping[str, Any]],
    reference_parameters: Any,
) -> list[str]:
    lines = _ORIGINAL_PARAMETER_LINES(
        selected,
        initial_delay,
        bags,
        bag_payloads,
        reference_parameters,
    )
    lines = [
        line.replace(
            "Deterministic pose-spline dynamics estimator",
            "Deterministic geometric Savitzky-Golay dynamics estimator",
        ).replace("command lag", "rotor command lag").replace(
            "selected strict-ZOH lag", "selected strict-ZOH rotor lag"
        )
        for line in lines
    ]
    initial_gimbal = initial_delay if _ACTIVE_INITIAL_GIMBAL_DELAY_SECONDS is None else float(_ACTIVE_INITIAL_GIMBAL_DELAY_SECONDS)
    selected_gimbal = selected.delay_seconds if selected.gimbal_delay_seconds is None else float(selected.gimbal_delay_seconds)
    lines.append("  gimbal command lag [s] {: .10g} -> {: .10g}".format(initial_gimbal, selected_gimbal))
    lines.extend(["", "Recorded command timestamp intervals (data-derived)"])
    for bag in bags:
        rotor = _timestamp_interval_statistics(
            bag.flight.rotor_command.all_times
        )
        gimbal = _timestamp_interval_statistics(
            bag.flight.gimbal_command.all_times
        )
        lines.append("  Bag {}".format(bag.specification.bag_id))
        for name, statistics in (("rotor", rotor), ("gimbal", gimbal)):
            median = statistics["median_seconds"]
            frequency = statistics["median_frequency_hz"]
            if median is None:
                lines.append("    {}: no positive timestamp interval".format(name))
            else:
                lines.append(
                    "    {}: median dt={:.9g} s ({:.6g} Hz), min={:.9g} s, max={:.9g} s".format(
                        name,
                        median,
                        frequency,
                        statistics["minimum_seconds"],
                        statistics["maximum_seconds"],
                    )
                )

    optimizer = selected.optimizer
    diagnostics = optimizer.get("diagnostics") if isinstance(optimizer, Mapping) else None
    lines.extend(
        [
            "",
            "Selected optimizer diagnostics",
            "  status={} success={} nfev={} njev={}".format(
                optimizer.get("status"),
                optimizer.get("success"),
                optimizer.get("nfev"),
                optimizer.get("njev"),
            ),
            "  message={}".format(optimizer.get("message")),
            "  reported optimality={}".format(optimizer.get("optimality")),
        ]
    )
    if isinstance(diagnostics, Mapping):
        initial_point = diagnostics.get("initial", {})
        final_point = diagnostics.get("final", {})
        final_jacobian = final_point.get("jacobian", {})
        lines.extend(
            [
                "  initial cost={}".format(initial_point.get("cost")),
                "  final cost={}".format(final_point.get("cost")),
                "  cost reduction={}".format(diagnostics.get("cost_reduction")),
                "  coordinate step L2={}".format(
                    diagnostics.get("coordinate_step_l2_norm")
                ),
                "  final gradient inf-norm={}".format(
                    final_point.get("gradient_inf_norm")
                ),
                "  gradient tolerance satisfied={}".format(
                    diagnostics.get("gradient_tolerance_satisfied")
                ),
                "  final Jacobian rank={}".format(
                    final_jacobian.get("numerical_rank")
                ),
                "  final Jacobian cond(nonzero)={}".format(
                    final_jacobian.get("condition_number_nonzero_subspace")
                ),
                "  max directional finite-difference Jacobian relative error={}".format(
                    diagnostics.get("finite_difference_max_relative_error")
                ),
            ]
        )
    return lines


class _RewritingStdout:
    """Keep run's useful progress output while removing spline wording."""

    def __init__(self, stream: Any) -> None:
        self.stream = stream

    def write(self, value: str) -> int:
        text = value.replace(
            "loading and fitting pose spline",
            "loading raw pose for geometric Savitzky-Golay",
        )
        text = text.replace(
            "selected strict lag",
            "selected strict lag",
        )
        pattern = re.compile(
            r"selected knot spacing ([0-9eE+.\-]+)s from .*?; parameter support "
            r"\[([0-9eE+.\-]+), ([0-9eE+.\-]+)\]s after excluding "
            r"[0-9eE+.\-]+ knot spans per side"
        )
        text = pattern.sub(
            r"SG window \1s; centered raw-pose parameter support "
            r"[\2, \3]s",
            text,
        )
        text = text.replace(
            "spline-dynamics",
            "Savitzky-Golay dynamics",
        )
        return self.stream.write(text)

    def flush(self) -> None:
        self.stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.stream, "isatty", lambda: False)())


def _write_savgol_fit_pdf(path: Path, bag: BagSplineData) -> None:
    """Preserve the old fit/derivative diagnostics and add SG-specific ones."""

    raw_time = np.asarray(bag.flight.pose.times, dtype=float)
    raw_position = np.asarray(bag.flight.pose.positions, dtype=float)
    raw_orientation = np.asarray(
        bag.flight.pose.orientations_xyzw,
        dtype=float,
    )
    trajectory = bag.spline_selection.spline
    filtered = trajectory.evaluate(raw_time)
    filtered_orientation = trajectory.sensor_orientation_xyzw(raw_time)

    raw_rpy = baseline._rpy_series(raw_orientation)
    filtered_rpy = baseline._rpy_series(filtered_orientation)
    position_error = filtered.sensor_position - raw_position
    orientation_error = _orientation_errors(
        raw_orientation,
        filtered_orientation,
    )
    relative_time = raw_time - raw_time[0]
    collocation = bag.collocation
    collocation_relative = collocation.time - raw_time[0]

    with PdfPages(path) as pdf:
        for title, reference, estimate, labels in (
            (
                "Raw mocap and local-polynomial position",
                raw_position,
                filtered.sensor_position,
                ("x [m]", "y [m]", "z [m]"),
            ),
            (
                "Raw mocap and geometric-SG orientation",
                raw_rpy,
                filtered_rpy,
                ("roll [rad]", "pitch [rad]", "yaw [rad]"),
            ),
        ):
            figure, axes = plt.subplots(
                3,
                1,
                figsize=(11.7, 8.3),
                sharex=True,
                constrained_layout=True,
            )
            for component, axis in enumerate(axes):
                axis.plot(
                    relative_time,
                    reference[:, component],
                    label="raw mocap",
                )
                axis.plot(
                    relative_time,
                    estimate[:, component],
                    linestyle="--",
                    label="degree-5 local polynomial",
                )
                axis.set_ylabel(labels[component])
                axis.grid(True, alpha=0.25)
            axes[0].set_title(
                "{}; W={:.6g}s".format(title, trajectory.window_seconds)
            )
            axes[0].legend(loc="best")
            axes[-1].set_xlabel("time from raw-pose start [s]")
            pdf.savefig(figure)
            plt.close(figure)

        value = np.column_stack((position_error, orientation_error))
        labels = (
            "dx [m]",
            "dy [m]",
            "dz [m]",
            "dRx [rad]",
            "dRy [rad]",
            "dRz [rad]",
        )
        figure, axes = plt.subplots(
            3,
            2,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        for component, axis in enumerate(axes.ravel()):
            axis.plot(relative_time, value[:, component])
            axis.set_ylabel(labels[component])
            axis.grid(True, alpha=0.25)
        axes[0, 0].set_title("Local-polynomial pose residual")
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        pdf.savefig(figure)
        plt.close(figure)

        for title, value, labels in (
            (
                "Translational derivatives at raw centered evaluation times",
                np.column_stack(
                    (
                        collocation.sensor_velocity_world,
                        collocation.sensor_acceleration_world,
                    )
                ),
                (
                    "vx [m/s]",
                    "vy [m/s]",
                    "vz [m/s]",
                    "ax [m/s2]",
                    "ay [m/s2]",
                    "az [m/s2]",
                ),
            ),
            (
                "Geometric-SG rotational derivatives (body frame)",
                np.column_stack(
                    (
                        collocation.body_angular_velocity,
                        collocation.body_angular_acceleration,
                    )
                ),
                (
                    "wx [rad/s]",
                    "wy [rad/s]",
                    "wz [rad/s]",
                    "alphax [rad/s2]",
                    "alphay [rad/s2]",
                    "alphaz [rad/s2]",
                ),
            ),
        ):
            figure, axes = plt.subplots(
                3,
                2,
                figsize=(11.7, 8.3),
                sharex=True,
                constrained_layout=True,
            )
            for component, axis in enumerate(axes.ravel()):
                axis.plot(
                    collocation_relative,
                    value[:, component],
                )
                axis.set_ylabel(labels[component])
                axis.grid(True, alpha=0.25)
            axes[0, 0].set_title(title)
            axes[-1, 0].set_xlabel("time [s]")
            axes[-1, 1].set_xlabel("time [s]")
            pdf.savefig(figure)
            plt.close(figure)

        acceleration_variance = np.diagonal(
            collocation.sensor_acceleration_world_covariance,
            axis1=1,
            axis2=2,
        )
        acceleration_std = np.sqrt(
            np.maximum(acceleration_variance, 0.0)
        )
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        for component, axis in enumerate(axes):
            axis.plot(
                collocation_relative,
                acceleration_std[:, component],
            )
            axis.set_ylabel(
                ("ax", "ay", "az")[component] + " local std [m/s2]"
            )
            axis.grid(True, alpha=0.25)
        axes[0].set_title(
            "Translation second-derivative covariance from local LS residuals"
        )
        axes[-1].set_xlabel("time [s]")
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(
            3,
            1,
            figsize=(11.7, 8.3),
            sharex=True,
            constrained_layout=True,
        )
        axes[0].plot(
            collocation_relative,
            collocation.window_sample_count,
        )
        axes[0].axhline(
            sg.MINIMUM_WINDOW_POINTS,
            linestyle="--",
            label="minimum required",
        )
        axes[0].set_ylabel("samples/window")
        axes[0].legend(loc="best")
        axes[1].semilogy(
            collocation_relative,
            collocation.position_fit_condition_number,
        )
        axes[1].set_ylabel("cond(position LS)")
        axes[2].semilogy(
            collocation_relative,
            collocation.rotation_fit_condition_number,
        )
        axes[2].set_ylabel("cond(SO3 LS)")
        axes[2].set_xlabel("time [s]")
        for axis in axes:
            axis.grid(True, alpha=0.25)
        axes[0].set_title(
            "Savitzky-Golay window diagnostics; W={:.6g}s".format(
                trajectory.window_seconds
            )
        )
        pdf.savefig(figure)
        plt.close(figure)


def _write_raw_residual_wrench_pdf(
    path: Path,
    time_axis: np.ndarray,
    body_wrench: np.ndarray,
    window_seconds: float,
) -> None:
    time_value = np.asarray(time_axis, dtype=float)
    wrench = np.asarray(body_wrench, dtype=float)
    if (
        time_value.ndim != 1
        or wrench.shape != (time_value.size, 6)
        or time_value.size < 1
    ):
        raise ValueError("raw residual-wrench diagnostic has invalid shape")
    relative = time_value - time_value[0]
    names = ("F_x", "F_y", "F_z", "M_x", "M_y", "M_z")
    units = ("N", "N", "N", "N m", "N m", "N m")
    with PdfPages(path) as pdf:
        for offset, title in (
            (0, "Raw inverse-dynamics residual body force"),
            (3, "Raw inverse-dynamics residual body torque"),
        ):
            figure, axes = plt.subplots(
                3, 1, figsize=(11.7, 8.3), sharex=True, constrained_layout=True
            )
            for local, axis in enumerate(axes):
                component = offset + local
                axis.plot(relative, wrench[:, component])
                axis.axhline(0.0, linewidth=0.7, alpha=0.5)
                axis.set_ylabel(
                    "{} [{}]".format(names[component], units[component])
                )
                axis.grid(True, alpha=0.25)
            axes[0].set_title(
                "{}; degree-5 SG W={:.6g}s; no replay-grid resampling".format(
                    title, window_seconds
                )
            )
            axes[-1].set_xlabel("time from first SG dynamics center [s]")
            pdf.savefig(figure)
            plt.close(figure)


def _rewrite_npz(
    bag_directory: Path,
    bag: BagSplineData,
) -> None:
    old_path = bag_directory / "spline_dynamics.npz"
    if not old_path.is_file():
        return
    with np.load(old_path, allow_pickle=False) as archive:
        values = {}
        for name in archive.files:
            new_name = (
                "savgol_" + name[len("spline_") :]
                if name.startswith("spline_")
                else name
            )
            values[new_name] = archive[name]

    collocation = bag.collocation
    values.update(
        {
            "raw_pose_time": np.asarray(
                bag.flight.pose.times,
                dtype=float,
            ),
            "raw_pose_sensor_position": np.asarray(
                bag.flight.pose.positions,
                dtype=float,
            ),
            "raw_pose_sensor_orientation_xyzw": np.asarray(
                bag.flight.pose.orientations_xyzw,
                dtype=float,
            ),
            "savgol_window_seconds": np.asarray(
                bag.spline_selection.spline.window_seconds,
                dtype=float,
            ),
            "savgol_polynomial_degree": np.asarray(
                sg.POLYNOMIAL_DEGREE,
                dtype=int,
            ),
            "savgol_window_sample_count": (
                collocation.window_sample_count
            ),
            "savgol_position_fit_condition_number": (
                collocation.position_fit_condition_number
            ),
            "savgol_rotation_fit_condition_number": (
                collocation.rotation_fit_condition_number
            ),
            "savgol_sensor_position_covariance": (
                collocation.sensor_position_covariance
            ),
            "savgol_sensor_velocity_world_covariance": (
                collocation.sensor_velocity_world_covariance
            ),
            "savgol_sensor_acceleration_world_covariance": (
                collocation.sensor_acceleration_world_covariance
            ),
        }
    )
    raw_wrench_time = np.asarray(
        values["raw_inferred_external_body_wrench_time"], dtype=float
    )
    raw_wrench = np.asarray(
        values["raw_inferred_external_body_wrench"], dtype=float
    )
    statistics = {
        "definition": "required_body_wrench - modeled_body_wrench at raw centered SG evaluation times",
        "sample_count": int(raw_wrench.shape[0]),
        "mean": np.mean(raw_wrench, axis=0),
        "covariance": (
            np.cov(raw_wrench, rowvar=False, ddof=1)
            if raw_wrench.shape[0] > 1
            else np.full((6, 6), np.nan)
        ),
        "second_moment": (raw_wrench.T @ raw_wrench) / raw_wrench.shape[0],
        "std": np.std(raw_wrench, axis=0, ddof=1) if raw_wrench.shape[0] > 1 else np.full(6, np.nan),
        "rms": np.sqrt(np.mean(raw_wrench * raw_wrench, axis=0)),
        "time_interval_seconds": (float(raw_wrench_time[0]), float(raw_wrench_time[-1])),
        "uses_replay_optimization": False,
        "uses_legacy_uniform_rollout_grid": False,
    }
    _ACTIVE_RAW_WRENCH_STATS[str(bag.specification.bag_id)] = statistics
    _write_raw_residual_wrench_pdf(
        bag_directory / "raw_residual_wrench.pdf",
        raw_wrench_time,
        raw_wrench,
        float(bag.spline_selection.spline.window_seconds),
    )

    new_path = bag_directory / "savgol_dynamics.npz"
    np.savez_compressed(new_path, **values)
    old_path.unlink()


def _rewrite_diagnostics(
    diagnostics: dict[str, Any],
    bag: BagSplineData,
) -> dict[str, Any]:
    old = diagnostics.pop("spline", {})
    trajectory = bag.spline_selection.spline
    candidate = bag.spline_selection.candidates[0]
    diagnostics.pop("collocation_count", None)
    diagnostics["evaluation_count"] = int(
        bag.collocation_time.size
    )
    diagnostics["savgol"] = {
        "degree": sg.POLYNOMIAL_DEGREE,
        "window_seconds": float(trajectory.window_seconds),
        "window_definition": (
            "physical time width; all raw mocap pose samples inside each "
            "local window are used"
        ),
        "translation": "degree-5 local least-squares polynomial in R^3",
        "rotation": (
            "degree-5 geometric Savitzky-Golay local polynomial on SO(3) "
            "(Jongeneel-Saccon IROS 2022)"
        ),
        "raw_pose_resampled_before_fit": False,
        "parameter_loss_evaluation_times": (
            "all raw pose timestamps with a complete centered W-second "
            "window, intersected with actuator/dynamics support"
        ),
        "minimum_required_points": sg.MINIMUM_WINDOW_POINTS,
        "minimum_window_sample_count": int(
            candidate.minimum_window_sample_count
        ),
        "maximum_window_sample_count": int(
            candidate.maximum_window_sample_count
        ),
        "raw_fit_interval_seconds": old.get(
            "fit_interval_seconds",
            np.asarray(
                (trajectory.start_time, trajectory.end_time),
                dtype=float,
            ),
        ),
        "parameter_estimation_interval_seconds": old.get(
            "parameter_estimation_interval_seconds",
            np.asarray(
                (
                    bag.collocation_time[0],
                    bag.collocation_time[-1],
                ),
                dtype=float,
            ),
        ),
        "actual_boundary_exclusion_seconds_start_end": old.get(
            "actual_boundary_exclusion_seconds_start_end",
        ),
        "centered_window_only_for_parameter_loss": True,
        "one_sided_shifted_full_window_used_only_for_edge_diagnostics": True,
        "fit_metrics": old.get("fit_metrics"),
        "translation_covariance": (
            "local ordinary-least-squares covariance estimated from position "
            "fit residuals when residual degrees of freedom are positive"
        ),
        "window_diagnostics": sg.candidate_payload(candidate),
    }
    rotor_timing = _timestamp_interval_statistics(
        bag.flight.rotor_command.all_times
    )
    gimbal_timing = _timestamp_interval_statistics(
        bag.flight.gimbal_command.all_times
    )
    diagnostics["command_timestamp_intervals"] = {
        "rotor": rotor_timing,
        "gimbal": gimbal_timing,
        "selected_rotor_lag_seconds": None,
        "selected_gimbal_lag_seconds": None,
        "note": (
            "Publish/update intervals are measured directly from rosbag "
            "timestamps; no command period is hard-coded into the SG front end."
        ),
    }
    raw_wrench_statistics = _ACTIVE_RAW_WRENCH_STATS.get(
        str(bag.specification.bag_id)
    )
    if raw_wrench_statistics is not None:
        diagnostics["raw_inverse_dynamics_residual_wrench_statistics"] = (
            raw_wrench_statistics
        )
    inferred = diagnostics.get("inferred_external_wrench")
    if isinstance(inferred, dict):
        definition = inferred.get("definition")
        if isinstance(definition, str):
            inferred["definition"] = definition.replace(
                "observed pose spline",
                "geometric Savitzky-Golay pose estimate",
            )
    return diagnostics


def _rewrite_json_outputs(
    output_directory: Path,
    initial_delay: float,
    arguments: argparse.Namespace,
) -> None:
    root_path = output_directory / "result.json"
    if not root_path.is_file():
        raise RuntimeError("base estimator did not produce result.json")

    root = json.loads(root_path.read_text(encoding="utf-8"))
    root["schema"] = SCHEMA
    root["method"] = {
        "name": "deterministic_savgol_dynamics",
        "description": (
            "raw-pose degree-5 local-polynomial gradient matching with "
            "geometric Savitzky-Golay filtering on SO(3), shared physical "
            "parameters, and command lag"
        ),
        "uses_multiple_shooting_nodes": False,
        "uses_continuity_constraints": False,
        "uses_augmented_lagrangian": False,
        "sensor_channels_in_parameter_loss": False,
        "pose_role": (
            "raw mocap pose supplies local degree-5 R3/SO3 polynomial fits"
        ),
        "raw_pose_resampled_before_local_polynomial_fit": False,
        "polynomial_degree": sg.POLYNOMIAL_DEGREE,
        "window_seconds": _window(),
        "parameter_data_objective": (
            "mean squared translational/angular acceleration residual; "
            "angular residual uses the reference inertia/mass metric"
        ),
        "command_mode_during_search": "overlapping quintic-smoothed ZOH jumps with separate rotor/gimbal lags",
        "command_mode_final": "strict ZOH with separate rotor/gimbal lags",
    }
    old_settings = root.get("settings", {})
    root["settings"] = {
        "window_seconds": _window(),
        "polynomial_degree": sg.POLYNOMIAL_DEGREE,
        "minimum_required_pose_samples_per_window": (
            sg.MINIMUM_WINDOW_POINTS
        ),
        "sample_step_seconds_for_rollout_diagnostics_only": old_settings.get(
            "sample_step_seconds"
        ),
        "integration_step_seconds": old_settings.get(
            "integration_step_seconds"
        ),
        "physical_coordinate_bounds": old_settings.get(
            "physical_coordinate_bounds"
        ),
        "delay_bounds_seconds": old_settings.get(
            "delay_bounds_seconds"
        ),
        "smoothstep_half_width_period_multipliers": list(arguments.smoothstep_width_fractions),
        "smoothstep_transitions_may_overlap": True,
        "strict_zoh_grid_step_seconds": {
            "rotor": _ACTIVE_ROTOR_PERIOD_SECONDS,
            "gimbal": _ACTIVE_GIMBAL_PERIOD_SECONDS,
        },
        "optimizer_tolerances": {
            "ftol": arguments.ftol,
            "xtol": arguments.xtol,
            "gtol": arguments.gtol,
            "note": (
                "ftol and xtol are disabled by default in the SG estimator; "
                "gtol or max_nfev terminates the solve unless explicitly overridden"
            ),
        },
        "pose_resampling_used_in_parameter_loss": False,
        "collocation_grid_used_in_parameter_loss": False,
    }
    initial = root.setdefault("initial_estimate", {})
    initial["delay_seconds"] = float(initial_delay)
    initial["delay_default"] = "one measured median publish period per command channel unless explicitly supplied"
    initial["rotor_delay_seconds"] = float(initial_delay)
    initial["gimbal_delay_seconds"] = float(arguments.initial_gimbal_delay)
    initial["recorded_command_period_seconds"] = {
        "rotor": _ACTIVE_ROTOR_PERIOD_SECONDS, "gimbal": _ACTIVE_GIMBAL_PERIOD_SECONDS
    }

    selection_root = root.get("selection", {})
    selected_rotor_lag = float(selection_root.get("rotor_delay_seconds", selection_root.get("delay_seconds", math.nan)))
    selected_gimbal_lag = float(selection_root.get("gimbal_delay_seconds", selected_rotor_lag))
    for diagnostics in root.get("bag_diagnostics", []):
        bag_id = str(diagnostics.get("id"))
        bag = _ACTIVE_BAGS.get(bag_id)
        if bag is not None:
            _rewrite_diagnostics(diagnostics, bag)
            timing = diagnostics.get("command_timestamp_intervals")
            if isinstance(timing, dict):
                timing["selected_rotor_lag_seconds"] = selected_rotor_lag
                timing["selected_gimbal_lag_seconds"] = selected_gimbal_lag
                for channel, lag in (("rotor", selected_rotor_lag), ("gimbal", selected_gimbal_lag)):
                    block = timing.get(channel)
                    if isinstance(block, dict):
                        median = block.get("median_seconds")
                        block["selected_lag_over_median_interval"] = (
                            None if median is None or not np.isfinite(lag)
                            else float(lag / float(median))
                        )

    selection = root.get("selection")

    optimizer_diagnostics = {
        "schema": SCHEMA + "/optimizer-diagnostics",
        "objective": root["method"]["parameter_data_objective"],
        "optimizer_tolerances": root["settings"]["optimizer_tolerances"],
        "smoothstep_stages": root.get("smoothstep_stages", []),
        "strict_zoh_polish": root.get("strict_zoh_polish", []),
        "selected_optimizer": (
            selection.get("optimizer") if isinstance(selection, dict) else None
        ),
        "command_lag_search": root.get("command_lag_search"),
    }
    optimizer_json = output_directory / "optimizer_diagnostics.json"
    optimizer_json.write_text(
        json.dumps(
            _json_sanitize(optimizer_diagnostics),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    optimizer_lines = [
        "Savitzky-Golay optimizer diagnostics",
        "",
        "Objective: {}".format(optimizer_diagnostics["objective"]),
        "Tolerances: {}".format(optimizer_diagnostics["optimizer_tolerances"]),
    ]
    selected_optimizer = optimizer_diagnostics.get("selected_optimizer")
    if isinstance(selected_optimizer, Mapping):
        optimizer_lines.extend(
            [
                "",
                "Selected strict-ZOH solve",
                "status={} success={} nfev={} njev={}".format(
                    selected_optimizer.get("status"),
                    selected_optimizer.get("success"),
                    selected_optimizer.get("nfev"),
                    selected_optimizer.get("njev"),
                ),
                "message={}".format(selected_optimizer.get("message")),
                "reported optimality={}".format(selected_optimizer.get("optimality")),
            ]
        )
        diagnostics = selected_optimizer.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            optimizer_lines.extend(
                [
                    "initial={}".format(diagnostics.get("initial")),
                    "final={}".format(diagnostics.get("final")),
                    "coordinate_step_l2_norm={}".format(
                        diagnostics.get("coordinate_step_l2_norm")
                    ),
                    "cost_reduction={}".format(diagnostics.get("cost_reduction")),
                    "gradient_tolerance_satisfied={}".format(
                        diagnostics.get("gradient_tolerance_satisfied")
                    ),
                    "finite_difference_max_relative_error={}".format(
                        diagnostics.get("finite_difference_max_relative_error")
                    ),
                    "finite_difference_directional_checks={}".format(
                        diagnostics.get("finite_difference_directional_checks")
                    ),
                ]
            )
    optimizer_lines.extend(["", "All smoothstep and strict-ZOH stage details are in optimizer_diagnostics.json."])
    strict._write_text(
        output_directory / "optimizer_diagnostics.txt",
        optimizer_lines,
    )
    _write_parameters_pdf(
        output_directory / "optimizer_diagnostics.pdf",
        optimizer_lines,
    )

    outputs = root.setdefault("outputs", {})
    outputs["delay_profile_json"] = "delay_profile.json"
    outputs["delay_profile_txt"] = "delay_profile.txt"
    outputs["optimizer_diagnostics_json"] = "optimizer_diagnostics.json"
    outputs["optimizer_diagnostics_txt"] = "optimizer_diagnostics.txt"
    outputs["optimizer_diagnostics_pdf"] = "optimizer_diagnostics.pdf"
    bag_outputs = outputs.get("bags", {})
    for bag_id, entry in bag_outputs.items():
        if isinstance(entry, dict):
            entry.pop("spline_dynamics_npz", None)
            entry["savgol_dynamics_npz"] = (
                "bags/{}/savgol_dynamics.npz".format(bag_id)
            )
            entry["savgol_fit_pdf"] = (
                "bags/{}/savgol_fit.pdf".format(bag_id)
            )
            entry["raw_residual_wrench_pdf"] = (
                "bags/{}/raw_residual_wrench.pdf".format(bag_id)
            )

    root_path.write_text(
        json.dumps(
            _json_sanitize(root),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    for bag_id, bag in _ACTIVE_BAGS.items():
        bag_directory = output_directory / "bags" / bag_id
        result_path = bag_directory / "result.json"
        if result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["schema"] = SCHEMA + "/bag-result"
            diagnostics = payload.get("diagnostics")
            if isinstance(diagnostics, dict):
                _rewrite_diagnostics(diagnostics, bag)
                timing = diagnostics.get("command_timestamp_intervals")
                if isinstance(timing, dict):
                    selected_rotor_lag = float(payload.get("shared_rotor_delay_seconds", payload.get("shared_delay_seconds", math.nan)))
                    selected_gimbal_lag = float(payload.get("shared_gimbal_delay_seconds", selected_rotor_lag))
                    timing["selected_rotor_lag_seconds"] = selected_rotor_lag
                    timing["selected_gimbal_lag_seconds"] = selected_gimbal_lag
                    for channel, lag in (("rotor", selected_rotor_lag), ("gimbal", selected_gimbal_lag)):
                        block = timing.get(channel)
                        if isinstance(block, dict):
                            median = block.get("median_seconds")
                            block["selected_lag_over_median_interval"] = (
                                None if median is None or not np.isfinite(lag)
                                else float(lag / float(median))
                            )
            outputs = payload.setdefault("outputs", {})
            outputs.pop("spline_dynamics_npz", None)
            outputs["savgol_dynamics_npz"] = "savgol_dynamics.npz"
            outputs["savgol_fit_pdf"] = "savgol_fit.pdf"
            outputs["raw_residual_wrench_pdf"] = "raw_residual_wrench.pdf"
            result_path.write_text(
                json.dumps(
                    _json_sanitize(payload),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate shared rigid-body parameters from raw mocap pose using "
            "degree-5 local polynomial / geometric Savitzky-Golay derivatives."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--vehicle-model-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prior-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        required=True,
        help=(
            "Physical SG window width W in seconds. All raw pose samples in "
            "each window are used; degree is fixed at 5."
        ),
    )
    parser.add_argument(
        "--sample-step",
        type=float,
        default=0.05,
        help=(
            "Uniform output step used only by legacy forward-rollout "
            "diagnostics; it does not resample the SG parameter-loss input."
        ),
    )
    parser.add_argument("--integration-step", type=float, default=0.025)
    parser.add_argument("--smooth-max-nfev", type=int, default=60)
    parser.add_argument("--strict-max-nfev", type=int, default=80)
    parser.add_argument(
        "--ftol",
        type=float,
        default=None,
        help="Function-change stopping tolerance. Disabled by default.",
    )
    parser.add_argument(
        "--xtol",
        type=float,
        default=None,
        help="Step-size stopping tolerance. Disabled by default.",
    )
    parser.add_argument("--gtol", type=float, default=1.0e-6)
    parser.add_argument(
        "--delay-bounds",
        type=float,
        nargs=2,
        default=(0.0, 0.20),
    )
    parser.add_argument(
        "--initial-rotor-delay",
        type=float,
        default=None,
        help="Initial rotor-command lag; default is one measured rotor publish period.",
    )
    parser.add_argument(
        "--initial-gimbal-delay",
        type=float,
        default=None,
        help="Initial gimbal-command lag; default is one measured gimbal publish period.",
    )
    parser.add_argument(
        "--smoothstep-width-fractions",
        type=float,
        nargs="+",
        default=(4.0, 2.0, 1.0, 0.5),
        help="Transition half-widths in measured publish periods; supports may overlap.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    return parser


def run(arguments: argparse.Namespace) -> int:
    global _ACTIVE_WINDOW_SECONDS
    _ACTIVE_WINDOW_SECONDS = float(arguments.window_seconds)
    _ACTIVE_BAGS.clear()
    _ACTIVE_RAW_WRENCH_STATS.clear()

    # Required by the shared report serializer but intentionally unused
    # by the SG front end.
    arguments.spline_cv_folds = 0
    arguments.maximum_spline_acceleration = math.inf
    arguments.maximum_spline_angular_acceleration = math.inf

    _resolve_lag_defaults(arguments)


    original_stdout = sys.stdout
    sys.stdout = _RewritingStdout(original_stdout)
    try:
        result = _shared_run(arguments)
    finally:
        sys.stdout = original_stdout

    output_directory = (
        arguments.output_dir.expanduser().resolve()
        / OUTPUT_SUBDIRECTORY
    )
    for bag_id, bag in _ACTIVE_BAGS.items():
        bag_directory = output_directory / "bags" / bag_id
        _write_savgol_fit_pdf(
            bag_directory / "savgol_fit.pdf",
            bag,
        )
        _rewrite_npz(bag_directory, bag)

    _rewrite_json_outputs(
        output_directory,
        float(arguments.initial_delay),
        arguments,
    )
    print(
        "Savitzky-Golay reports written to {}".format(output_directory),
        flush=True,
    )
    return int(result)


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(create_argument_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
