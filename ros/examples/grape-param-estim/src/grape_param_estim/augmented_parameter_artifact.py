"""Strict stage-2 artifact for fixed-Q augmented-parameter EnRTS.

This bundle is deliberately a scientific boundary rather than a Python
object dump.  It preserves raw member laws and per-time marginal smoother
coordinates, uses only JSON and pickle-free NPZ files, and binds the result
to the complete diagonal-Q artifact that supplied its fixed process noise.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, Union
import zipfile

import numpy as np

from grape_param_estim.artifact_io import (
    ArtifactStateError,
    ArtifactValidationError,
    CANCELLED_STATUS,
    COMPLETE_STATUS,
    IncompleteArtifactError,
    MANIFEST_NAME,
    UnsupportedArtifactSchema,
    WRITING_STATUS,
    canonical_json_bytes,
    read_json,
    request_fingerprint as compute_fingerprint,
    write_json_atomic,
    write_npz_atomic,
)
from grape_param_estim.augmented_parameter_state import (
    LOCAL_INITIAL_DIMENSION,
    SHARED_STATIC_DIMENSION,
    decode_shared_static_coordinates,
)
from grape_param_estim.diagonal_q import (
    BODY_WRENCH_COMPONENT_ORDER,
    BODY_WRENCH_FRAME,
    BODY_WRENCH_VARIANCE_UNITS,
)
from grape_param_estim.diagonal_q_artifact import (
    DIAGONAL_Q_ESTIMATE_SCHEMA,
    load_diagonal_q_artifact,
)
from grape_param_estim.geometry import (
    correction_transform_path,
    euler_xyz_to_matrix,
    matrix_to_quaternion,
)
from grape_param_estim.multi_bag_augmented_parameter import (
    MultiBagAugmentedParameterResult,
)
from grape_param_estim.parameterization import (
    PARAMETER_DIMENSION,
    VehicleParameterChart,
)
from grape_param_estim.real_rosbag import (
    ControllerGainSnapshot,
    EpisodeProvenance,
    RealFlightEpisode,
)
from grape_param_estim.strong_constraint import StrongConstraintProblem
from grape_param_estim.system import ClosedLoopTrajectory, VehicleParameters
from grape_param_estim.timing import BoundedDelayChart


AUGMENTED_PARAMETER_ESTIMATE_SCHEMA = (
    "grape-param-estim/fixed-q-augmented-parameter-estimate/v1"
)
SEQUENTIAL_ENRTS_PATH_SEMANTICS = (
    "sequential_enrts_marginal_with_time_varying_static_coordinates"
)

_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONFIGURATION_FINGERPRINT = re.compile(
    r"(?:(?:complete|incomplete):[0-9a-f]{64}|manual-group:sha256:[0-9a-f]{64})\Z"
)

_SHARED_KEYS = (
    "member_id",
    "initial_shared_coordinates",
    "final_shared_coordinates",
    "mass",
    "inertia",
    "cog_offset",
    "force_effectiveness",
    "torque_effectiveness",
    "constant_delay",
    "chart_nominal_mass",
    "chart_nominal_inertia",
    "chart_nominal_cog_offset",
    "chart_nominal_force_effectiveness",
    "chart_nominal_torque_effectiveness",
    "chart_nominal_linear_drag",
    "chart_nominal_angular_drag",
    "ridge_covariance",
    "ridge_eigenvalues",
    "ridge_eigenvectors",
    "expected_physical_ridge_direction",
    "expected_physical_ridge_variance",
    "ensemble_rank",
)

_BAG_KEYS = (
    "bag_id",
    "source_path",
    "source_sha256",
    "source_size_bytes",
    "episode_index",
    "configuration_fingerprint",
    "episode_provenance_fingerprint",
    "controller_snapshot_fingerprint",
    "controller_configuration_fingerprint",
    "model_provenance_fingerprint",
    "member_id",
    "times",
    "record_times",
    "requested_interval_record_seconds",
    "effective_interval_record_seconds",
    "effective_interval_local_seconds",
    "observation_position",
    "observation_orientation_xyzw",
    "reference_position",
    "reference_orientation_xyzw",
    "nominal_position",
    "nominal_orientation_xyzw",
    "nominal_linear_velocity",
    "nominal_angular_velocity",
    "nominal_controller_integral",
    "nominal_commanded_thrust",
    "nominal_commanded_gimbal_angle",
    "nominal_actuator_thrust",
    "nominal_actuator_gimbal_angle",
    "nominal_body_wrench",
    "fixed_q_stationary_variance",
    "fixed_r_translation_covariance",
    "fixed_r_rotation_covariance",
    "fixed_correlation_time",
    "initial_shared_coordinates",
    "initial_local_coordinates",
    "initial_position",
    "initial_orientation_xyzw",
    "initial_linear_velocity",
    "initial_angular_velocity",
    "initial_controller_integral",
    "initial_controller_roll_pitch_integration_active",
    "initial_actuator_thrust",
    "initial_actuator_gimbal_angle",
    "initial_residual_wrench",
    "static_smoothed_coordinates",
    "smoothed_position",
    "smoothed_orientation_xyzw",
    "smoothed_linear_velocity",
    "smoothed_angular_velocity",
    "smoothed_controller_integral",
    "smoothed_controller_roll_pitch_integration_active",
    "smoothed_actuator_thrust",
    "smoothed_actuator_gimbal_angle",
    "smoothed_residual_wrench",
    "observed_correction_translation",
    "observed_correction_rotation_vector",
    "reference_correction_translation",
    "reference_correction_rotation_vector",
    "smoothed_correction_translation",
    "smoothed_correction_rotation_vector",
    "filter_log_likelihood_by_time",
    "filter_nis",
    "applied_model_mass",
    "applied_model_delay",
)

_MANIFEST_KEYS = {
    "schema",
    "status",
    "run_id",
    "stage_id",
    "request_fingerprint",
    "project_fingerprint",
    "stage_input_fingerprint",
    "implementation",
    "upstream_diagonal_q",
    "selected_bag_ids",
    "member_count",
    "shared_static",
    "body_wrench",
    "path_semantics",
    "bags",
    "artifacts",
    "cancellation",
}


@dataclass(frozen=True)
class AugmentedParameterArtifactBagInput:
    """Real-flight data and immutable provenance for one stage-2 result."""

    bag_id: str
    episode_index: int
    episode: RealFlightEpisode
    problem: StrongConstraintProblem
    nominal_trajectory: ClosedLoopTrajectory
    configuration_fingerprint: str
    model_provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        identifier = _string(self.bag_id, "bag_id")
        if "\x00" in identifier:
            raise ArtifactValidationError("bag_id cannot contain NUL")
        episode_index = _nonnegative_integer(
            self.episode_index, "episode_index"
        )
        if not isinstance(self.episode, RealFlightEpisode):
            raise TypeError("episode must be a RealFlightEpisode")
        if not isinstance(self.problem, StrongConstraintProblem):
            raise TypeError("problem must be a StrongConstraintProblem")
        if not isinstance(self.nominal_trajectory, ClosedLoopTrajectory):
            raise TypeError(
                "nominal_trajectory must be a ClosedLoopTrajectory"
            )
        fingerprint = _configuration_fingerprint(
            self.configuration_fingerprint,
            "configuration_fingerprint",
        )
        provenance = _normalised_mapping(
            self.model_provenance, "model_provenance"
        )
        _validate_input_alignment(
            self.episode, self.problem, self.nominal_trajectory
        )
        source = self.episode.provenance
        _raw_sha256(source.bag_sha256, "episode.provenance.bag_sha256")
        _positive_integer(
            source.bag_size_bytes, "episode.provenance.bag_size_bytes"
        )
        object.__setattr__(self, "bag_id", identifier)
        object.__setattr__(self, "episode_index", episode_index)
        object.__setattr__(self, "configuration_fingerprint", fingerprint)
        object.__setattr__(self, "model_provenance", provenance)


@dataclass(frozen=True)
class AugmentedParameterArtifactBundle:
    """Detached arrays from one fully validated complete stage-2 bundle."""

    root: Path
    manifest: Mapping[str, Any]
    shared_posterior: Mapping[str, np.ndarray]
    bags: Mapping[str, Mapping[str, np.ndarray]]

    @property
    def bag_ids(self) -> Tuple[str, ...]:
        return tuple(self.manifest["selected_bag_ids"])


def _exact_keys(
    value: Any, expected: Sequence[str], location: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError("{} must be an object".format(location))
    expected_set = set(expected)
    actual = set(value)
    missing = sorted(expected_set - actual)
    extra = sorted(actual - expected_set)
    if missing or extra:
        raise ArtifactValidationError(
            "{} keys differ from schema; missing={}, extra={}".format(
                location, missing, extra
            )
        )
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactValidationError(
            "{} must be a non-empty string".format(location)
        )
    return value


def _fingerprint(value: Any, location: str) -> str:
    selected = _string(value, location)
    if _FINGERPRINT.fullmatch(selected) is None:
        raise ArtifactValidationError(
            "{} must be a lowercase SHA256 fingerprint".format(location)
        )
    return selected


def _raw_sha256(value: Any, location: str) -> str:
    selected = _string(value, location)
    if _RAW_SHA256.fullmatch(selected) is None:
        raise ArtifactValidationError(
            "{} must be a lowercase raw SHA256 digest".format(location)
        )
    return selected


def _configuration_fingerprint(value: Any, location: str) -> str:
    selected = _string(value, location)
    if _CONFIGURATION_FINGERPRINT.fullmatch(selected) is None:
        raise ArtifactValidationError(
            "{} must be a complete/incomplete or manual-group SHA256 "
            "fingerprint".format(
                location
            )
        )
    return selected


def _positive_integer(value: Any, location: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ArtifactValidationError(
            "{} must be a positive integer".format(location)
        )
    result = int(value)
    if result <= 0:
        raise ArtifactValidationError(
            "{} must be a positive integer".format(location)
        )
    return result


def _nonnegative_integer(value: Any, location: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ArtifactValidationError(
            "{} must be a non-negative integer".format(location)
        )
    result = int(value)
    if result < 0:
        raise ArtifactValidationError(
            "{} must be a non-negative integer".format(location)
        )
    return result


def _finite_float(value: Any, location: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ArtifactValidationError("{} must be finite".format(location))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ArtifactValidationError(
            "{} must be finite".format(location)
        ) from error
    if not np.isfinite(result):
        raise ArtifactValidationError("{} must be finite".format(location))
    return result


def _normalised_mapping(
    value: Mapping[str, Any], location: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError("{} must be an object".format(location))
    try:
        normalised = json.loads(canonical_json_bytes(value).decode("utf-8"))
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError(
            "{} is not finite JSON provenance".format(location)
        ) from error
    if not isinstance(normalised, dict):
        raise ArtifactValidationError("{} must be an object".format(location))
    return normalised


def _same_array(first: np.ndarray, second: np.ndarray) -> bool:
    return np.array_equal(np.asarray(first), np.asarray(second))


def _same_trajectory(
    first: ClosedLoopTrajectory, second: ClosedLoopTrajectory
) -> bool:
    return all(
        _same_array(getattr(first, field.name), getattr(second, field.name))
        for field in fields(ClosedLoopTrajectory)
    )


def _validate_input_alignment(
    episode: RealFlightEpisode,
    problem: StrongConstraintProblem,
    nominal: ClosedLoopTrajectory,
) -> None:
    observations = episode.observations
    problem_observations = problem.observations
    for name in (
        "times",
        "position",
        "orientation_xyzw",
        "translation_covariance",
        "rotation_covariance",
    ):
        if not _same_array(
            getattr(observations, name), getattr(problem_observations, name)
        ):
            raise ArtifactValidationError(
                "episode and problem observations differ in {}".format(name)
            )
    if len(episode.references) != len(problem.references) or any(
        any(
            not _same_array(getattr(left, field.name), getattr(right, field.name))
            for field in fields(left)
        )
        for left, right in zip(episode.references, problem.references)
    ):
        raise ArtifactValidationError(
            "episode and problem references do not match"
        )
    if not _same_trajectory(problem.nominal_trajectory, nominal):
        raise ArtifactValidationError(
            "explicit nominal trajectory differs from the filter problem"
        )


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_jsonable_value(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {
            field.name: _jsonable_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def _jsonable_dataclass(value: Any) -> Mapping[str, Any]:
    result: Dict[str, Any] = {
        field.name: _jsonable_value(getattr(value, field.name))
        for field in fields(value)
    }
    return _normalised_mapping(result, type(value).__name__)


def _controller_snapshot_mapping(
    value: ControllerGainSnapshot,
) -> Mapping[str, Any]:
    return _jsonable_dataclass(value)


def _episode_provenance_mapping(
    value: EpisodeProvenance,
) -> Mapping[str, Any]:
    return _jsonable_dataclass(value)


def _controller_configuration_mapping(value) -> Mapping[str, Any]:
    return _jsonable_dataclass(value)


def _interval(value: Any, location: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ArtifactValidationError(
            "{} must contain two bounds".format(location)
        )
    start = _finite_float(value[0], location + "[0]")
    end = _finite_float(value[1], location + "[1]")
    if end <= start:
        raise ArtifactValidationError(
            "{} must be strictly increasing".format(location)
        )
    return start, end


def _reference_pose(episode: RealFlightEpisode) -> Tuple[np.ndarray, np.ndarray]:
    position = np.asarray([value.position for value in episode.references])
    orientation = np.asarray(
        [
            matrix_to_quaternion(euler_xyz_to_matrix(value.rpy))
            for value in episode.references
        ]
    )
    return position, orientation


def _dynamic_arrays(states) -> Mapping[str, np.ndarray]:
    # Filter results store [time][member]; artifacts use [member, time, ...].
    result = {
        "position": np.asarray(
            [[value.rigid.position for value in ensemble] for ensemble in states]
        ),
        "orientation_xyzw": np.asarray(
            [
                [value.rigid.orientation_xyzw for value in ensemble]
                for ensemble in states
            ]
        ),
        "linear_velocity": np.asarray(
            [
                [value.rigid.linear_velocity for value in ensemble]
                for ensemble in states
            ]
        ),
        "angular_velocity": np.asarray(
            [
                [value.rigid.angular_velocity for value in ensemble]
                for ensemble in states
            ]
        ),
        "controller_integral": np.asarray(
            [
                [value.controller.integral_error for value in ensemble]
                for ensemble in states
            ]
        ),
        "controller_roll_pitch_integration_active": np.asarray(
            [
                [
                    value.controller.roll_pitch_integration_active
                    for value in ensemble
                ]
                for ensemble in states
            ],
            dtype=bool,
        ),
        "actuator_thrust": np.asarray(
            [[value.actuator.thrust for value in ensemble] for ensemble in states]
        ),
        "actuator_gimbal_angle": np.asarray(
            [
                [value.actuator.gimbal_angle for value in ensemble]
                for ensemble in states
            ]
        ),
        "residual_wrench": np.asarray(
            [[value.residual_wrench for value in ensemble] for ensemble in states]
        ),
    }
    return {
        key: np.transpose(value, (1, 0) + tuple(range(2, value.ndim)))
        for key, value in result.items()
    }


def _initial_dynamic_arrays(states) -> Mapping[str, np.ndarray]:
    return {
        "position": np.asarray([value.rigid.position for value in states]),
        "orientation_xyzw": np.asarray(
            [value.rigid.orientation_xyzw for value in states]
        ),
        "linear_velocity": np.asarray(
            [value.rigid.linear_velocity for value in states]
        ),
        "angular_velocity": np.asarray(
            [value.rigid.angular_velocity for value in states]
        ),
        "controller_integral": np.asarray(
            [value.controller.integral_error for value in states]
        ),
        "controller_roll_pitch_integration_active": np.asarray(
            [
                value.controller.roll_pitch_integration_active
                for value in states
            ],
            dtype=bool,
        ),
        "actuator_thrust": np.asarray(
            [value.actuator.thrust for value in states]
        ),
        "actuator_gimbal_angle": np.asarray(
            [value.actuator.gimbal_angle for value in states]
        ),
        "residual_wrench": np.asarray(
            [value.residual_wrench for value in states]
        ),
    }


def _smoothed_corrections(
    nominal: ClosedLoopTrajectory,
    position: np.ndarray,
    orientation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    translation = []
    rotation = []
    for member_position, member_orientation in zip(position, orientation):
        selected_translation, selected_rotation = correction_transform_path(
            nominal.position,
            nominal.orientation_xyzw,
            member_position,
            member_orientation,
        )
        translation.append(selected_translation)
        rotation.append(selected_rotation)
    return np.asarray(translation), np.asarray(rotation)


def _decoded_shared_arrays(
    result: MultiBagAugmentedParameterResult,
    inputs: Sequence[AugmentedParameterArtifactBagInput],
) -> Mapping[str, np.ndarray]:
    final = result.final_shared_posterior
    chart = BoundedDelayChart(result.maximum_delay)
    decoded = tuple(
        decode_shared_static_coordinates(inputs[0].problem, value, chart)
        for value in final
    )
    parameters = tuple(value[0] for value in decoded)
    delays = np.asarray([value[1] for value in decoded])
    for bag_input in inputs[1:]:
        check = tuple(
            decode_shared_static_coordinates(bag_input.problem, value, chart)
            for value in final
        )
        check_parameters = tuple(value[0] for value in check)
        check_mass = np.asarray([value.mass for value in check_parameters])
        check_delay = np.asarray([value[1] for value in check])
        if not np.allclose(
            check_mass,
            np.asarray([value.mass for value in parameters]),
            rtol=1.0e-12,
            atol=1.0e-14,
        ) or not np.array_equal(check_delay, delays) or any(
            not np.allclose(
                np.asarray(
                    [getattr(value, field_name) for value in check_parameters]
                ),
                np.asarray(
                    [getattr(value, field_name) for value in parameters]
                ),
                rtol=1.0e-12,
                atol=1.0e-14,
            )
            for field_name in (
                "inertia",
                "cog_offset",
                "force_effectiveness",
                "torque_effectiveness",
            )
        ):
            raise ArtifactValidationError(
                "bag parameter charts do not decode one shared posterior"
            )
    nominal = inputs[0].problem.parameter_chart.decode(
        np.zeros(PARAMETER_DIMENSION)
    )
    covariance = np.cov(final, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    expected = np.concatenate(
        (inputs[0].problem.parameter_chart.ridge_direction(), (0.0,))
    )
    expected /= np.linalg.norm(expected)
    centered = final - np.mean(final, axis=0)
    return {
        "member_id": result.member_id,
        "initial_shared_coordinates": result.initial_shared_ensemble,
        "final_shared_coordinates": final,
        "mass": np.asarray([value.mass for value in parameters]),
        "inertia": np.asarray([value.inertia for value in parameters]),
        "cog_offset": np.asarray([value.cog_offset for value in parameters]),
        "force_effectiveness": np.asarray(
            [value.force_effectiveness for value in parameters]
        ),
        "torque_effectiveness": np.asarray(
            [value.torque_effectiveness for value in parameters]
        ),
        "constant_delay": delays,
        "chart_nominal_mass": np.asarray((nominal.mass,)),
        "chart_nominal_inertia": nominal.inertia,
        "chart_nominal_cog_offset": nominal.cog_offset,
        "chart_nominal_force_effectiveness": nominal.force_effectiveness,
        "chart_nominal_torque_effectiveness": nominal.torque_effectiveness,
        "chart_nominal_linear_drag": nominal.linear_drag,
        "chart_nominal_angular_drag": nominal.angular_drag,
        "ridge_covariance": covariance,
        "ridge_eigenvalues": eigenvalues,
        "ridge_eigenvectors": eigenvectors,
        "expected_physical_ridge_direction": expected,
        "expected_physical_ridge_variance": np.asarray(
            (float(expected @ covariance @ expected),)
        ),
        "ensemble_rank": np.asarray(
            (np.linalg.matrix_rank(centered),), dtype=np.int64
        ),
    }


def _bag_arrays(
    bag_input: AugmentedParameterArtifactBagInput,
    bag_result,
    provenance_fingerprint: str,
    snapshot_fingerprint: str,
    controller_fingerprint: str,
    model_fingerprint: str,
) -> Mapping[str, np.ndarray]:
    episode = bag_input.episode
    provenance = episode.provenance
    nominal = bag_input.nominal_trajectory
    output = bag_result.filter_result
    reference_position, reference_orientation = _reference_pose(episode)
    smoothed = _dynamic_arrays(output.dynamic_smoothed_state_ensembles)
    initial = _initial_dynamic_arrays(bag_result.initial_ensemble.filter_states)
    observed_translation, observed_rotation = correction_transform_path(
        nominal.position,
        nominal.orientation_xyzw,
        episode.observations.position,
        episode.observations.orientation_xyzw,
    )
    reference_translation, reference_rotation = correction_transform_path(
        nominal.position,
        nominal.orientation_xyzw,
        reference_position,
        reference_orientation,
    )
    smoothed_translation, smoothed_rotation = _smoothed_corrections(
        nominal, smoothed["position"], smoothed["orientation_xyzw"]
    )
    arrays: Dict[str, np.ndarray] = {
        "bag_id": np.asarray((bag_input.bag_id,)),
        "source_path": np.asarray((provenance.bag_path,)),
        "source_sha256": np.asarray((provenance.bag_sha256,)),
        "source_size_bytes": np.asarray(
            (provenance.bag_size_bytes,), dtype=np.int64
        ),
        "episode_index": np.asarray(
            (bag_input.episode_index,), dtype=np.int64
        ),
        "configuration_fingerprint": np.asarray(
            (bag_input.configuration_fingerprint,)
        ),
        "episode_provenance_fingerprint": np.asarray(
            (provenance_fingerprint,)
        ),
        "controller_snapshot_fingerprint": np.asarray(
            (snapshot_fingerprint,)
        ),
        "controller_configuration_fingerprint": np.asarray(
            (controller_fingerprint,)
        ),
        "model_provenance_fingerprint": np.asarray((model_fingerprint,)),
        "member_id": output.member_id,
        "times": output.times,
        "record_times": episode.record_times,
        "requested_interval_record_seconds": np.asarray(
            (
                provenance.requested_window_start,
                provenance.requested_window_end,
            )
        ),
        "effective_interval_record_seconds": np.asarray(
            (
                episode.window_start_record_time,
                episode.window_end_record_time,
            )
        ),
        "effective_interval_local_seconds": np.asarray(
            (
                episode.window_start_local_time,
                episode.window_end_local_time,
            )
        ),
        "observation_position": episode.observations.position,
        "observation_orientation_xyzw": (
            episode.observations.orientation_xyzw
        ),
        "reference_position": reference_position,
        "reference_orientation_xyzw": reference_orientation,
        "nominal_position": nominal.position,
        "nominal_orientation_xyzw": nominal.orientation_xyzw,
        "nominal_linear_velocity": nominal.linear_velocity,
        "nominal_angular_velocity": nominal.angular_velocity,
        "nominal_controller_integral": nominal.controller_integral,
        "nominal_commanded_thrust": nominal.commanded_thrust,
        "nominal_commanded_gimbal_angle": nominal.commanded_gimbal_angle,
        "nominal_actuator_thrust": nominal.actuator_thrust,
        "nominal_actuator_gimbal_angle": nominal.actuator_gimbal_angle,
        "nominal_body_wrench": nominal.body_wrench,
        "fixed_q_stationary_variance": (
            bag_result.wrench_covariance.stationary_variance
        ),
        "fixed_r_translation_covariance": (
            bag_result.observation_covariance.translation
        ),
        "fixed_r_rotation_covariance": (
            bag_result.observation_covariance.rotation_tangent
        ),
        "fixed_correlation_time": np.asarray(
            (bag_result.correlation_time,)
        ),
        "initial_shared_coordinates": (
            bag_result.initial_ensemble.shared_coordinates
        ),
        "initial_local_coordinates": (
            bag_result.initial_ensemble.local_coordinates
        ),
        "static_smoothed_coordinates": output.static_smoothed_ensemble,
        "observed_correction_translation": observed_translation,
        "observed_correction_rotation_vector": observed_rotation,
        "reference_correction_translation": reference_translation,
        "reference_correction_rotation_vector": reference_rotation,
        "smoothed_correction_translation": smoothed_translation,
        "smoothed_correction_rotation_vector": smoothed_rotation,
        "filter_log_likelihood_by_time": (
            output.filter_log_likelihood_by_time
        ),
        "filter_nis": output.filter_nis,
        "applied_model_mass": output.applied_model_mass,
        "applied_model_delay": output.applied_model_delay,
    }
    for prefix, source in (("initial", initial), ("smoothed", smoothed)):
        for name, value in source.items():
            arrays["{}_{}".format(prefix, name)] = value
    _exact_keys(arrays, _BAG_KEYS, "generated bag arrays")
    return arrays


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise ArtifactValidationError(
            "cannot hash artifact {}: {}".format(path, error)
        ) from error
    return "sha256:{}".format(digest.hexdigest())


def diagonal_q_artifact_fingerprint(path: Union[str, Path]) -> str:
    """Fingerprint the validated complete upstream Q manifest."""

    bundle = load_diagonal_q_artifact(path)
    return compute_fingerprint(bundle.manifest)


def _empty_descriptor(relative: str) -> Mapping[str, Any]:
    return {"path": relative, "sha256": "sha256:" + "0" * 64, "size_bytes": 0}


def _complete_descriptor(path: Path, root: Path) -> Mapping[str, Any]:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise ArtifactValidationError(
            "artifact path is outside its bundle"
        ) from error
    return {
        "path": relative,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _descriptor(value: Any, location: str, complete: bool) -> Mapping[str, Any]:
    selected = _exact_keys(value, ("path", "sha256", "size_bytes"), location)
    path = _string(selected["path"], location + ".path")
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts or path in {".", ""}:
        raise ArtifactValidationError(
            "{}.path must stay inside the bundle".format(location)
        )
    digest = _fingerprint(selected["sha256"], location + ".sha256")
    size = selected["size_bytes"]
    if isinstance(size, (bool, np.bool_)) or not isinstance(
        size, (int, np.integer)
    ) or int(size) < 0:
        raise ArtifactValidationError(
            "{}.size_bytes must be a non-negative integer".format(location)
        )
    if complete and int(size) == 0:
        raise ArtifactValidationError(
            "{}.size_bytes must be positive when complete".format(location)
        )
    if not complete and (digest != "sha256:" + "0" * 64 or int(size) != 0):
        raise ArtifactValidationError(
            "{} writing descriptor must be empty".format(location)
        )
    return selected


def _safe_payload_path(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ArtifactValidationError("payload path escapes the bundle")
    candidate = root / relative_path
    candidate.parent.mkdir(parents=True, exist_ok=True)
    root_resolved = root.resolve()
    parent_resolved = candidate.parent.resolve()
    if parent_resolved != root_resolved and root_resolved not in parent_resolved.parents:
        raise ArtifactValidationError("payload parent escapes the bundle")
    if candidate.exists() or candidate.is_symlink():
        raise ArtifactStateError(
            "artifact payload already exists: {}".format(candidate)
        )
    return candidate


def _verified_payload(
    root: Path, descriptor: Mapping[str, Any], location: str
) -> Tuple[Path, bytes]:
    relative = descriptor["path"]
    candidate = root / relative
    root_resolved = root.resolve()
    resolved = candidate.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ArtifactValidationError("{} escapes the bundle".format(location))
    file_descriptor = None
    try:
        file_descriptor = os.open(
            str(candidate), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactValidationError(
                "{} must be a regular payload file".format(location)
            )
        blocks = []
        while True:
            block = os.read(file_descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
    except ArtifactValidationError:
        raise
    except OSError as error:
        raise ArtifactValidationError(
            "cannot read {}: {}".format(candidate, error)
        ) from error
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
    payload = b"".join(blocks)
    if len(payload) != descriptor["size_bytes"]:
        raise ArtifactValidationError(
            "{} size does not match the manifest".format(location)
        )
    digest = "sha256:{}".format(hashlib.sha256(payload).hexdigest())
    if digest != descriptor["sha256"]:
        raise ArtifactValidationError(
            "{} SHA256 does not match the manifest".format(location)
        )
    return candidate.resolve(), payload


def _load_npz_exact(
    payload: bytes, path: Path, keys: Sequence[str]
) -> Dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile) as error:
        raise ArtifactValidationError(
            "cannot inspect NPZ {}: {}".format(path, error)
        ) from error
    expected = {"{}.npy".format(key) for key in keys}
    if len(names) != len(set(names)) or set(names) != expected:
        raise ArtifactValidationError(
            "{} ZIP members differ from schema".format(path)
        )
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != set(keys):
                raise ArtifactValidationError(
                    "{} arrays differ from schema".format(path)
                )
            arrays = {key: np.asarray(archive[key]).copy() for key in keys}
    except ArtifactValidationError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ArtifactValidationError(
            "cannot load NPZ {}: {}".format(path, error)
        ) from error
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ArtifactValidationError(
            "{} contains forbidden object arrays".format(path)
        )
    return arrays


def _numeric(
    arrays: Mapping[str, np.ndarray],
    key: str,
    shape: Tuple[int, ...],
    location: str,
) -> np.ndarray:
    value = arrays[key]
    if (
        value.shape != shape
        or not np.issubdtype(value.dtype, np.floating)
        or np.any(~np.isfinite(value))
    ):
        raise ArtifactValidationError(
            "{}:{} must be a finite floating {} array".format(
                location, key, shape
            )
        )
    return value


def _integer(
    arrays: Mapping[str, np.ndarray],
    key: str,
    shape: Tuple[int, ...],
    location: str,
) -> np.ndarray:
    value = arrays[key]
    if value.shape != shape or not np.issubdtype(value.dtype, np.integer):
        raise ArtifactValidationError(
            "{}:{} must be an integer {} array".format(location, key, shape)
        )
    return value.astype(np.int64, copy=False)


def _boolean(
    arrays: Mapping[str, np.ndarray],
    key: str,
    shape: Tuple[int, ...],
    location: str,
) -> np.ndarray:
    value = arrays[key]
    if value.shape != shape or not np.issubdtype(value.dtype, np.bool_):
        raise ArtifactValidationError(
            "{}:{} must be a boolean {} array".format(location, key, shape)
        )
    return value


def _unicode_scalar(
    arrays: Mapping[str, np.ndarray], key: str, location: str
) -> str:
    value = arrays[key]
    if value.shape != (1,) or value.dtype.kind != "U":
        raise ArtifactValidationError(
            "{}:{} must contain one Unicode string".format(location, key)
        )
    return str(value[0])


def _covariance3(value: Any, location: str) -> np.ndarray:
    covariance = np.asarray(value, dtype=float)
    if (
        covariance.shape != (3, 3)
        or np.any(~np.isfinite(covariance))
        or not np.allclose(covariance, covariance.T, rtol=1.0e-12, atol=1.0e-14)
        or np.any(np.linalg.eigvalsh(covariance) <= 0.0)
    ):
        raise ArtifactValidationError(
            "{} must be positive-definite 3 by 3 covariance".format(location)
        )
    return covariance


def _finite_vector_json(value: Any, size: int, location: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if selected.shape != (size,) or np.any(~np.isfinite(selected)):
        raise ArtifactValidationError(
            "{} must contain {} finite values".format(location, size)
        )
    return selected


def _matrix_json(
    value: Any, shape: Tuple[int, ...], location: str
) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if selected.shape != shape or np.any(~np.isfinite(selected)):
        raise ArtifactValidationError(
            "{} must be a finite {} array".format(location, shape)
        )
    return selected


def _validate_manifest(
    manifest: Mapping[str, Any], require_complete: bool
) -> Tuple[Tuple[str, ...], Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(manifest, Mapping):
        raise ArtifactValidationError("manifest must be an object")
    if manifest.get("schema") != AUGMENTED_PARAMETER_ESTIMATE_SCHEMA:
        raise UnsupportedArtifactSchema(
            "unsupported artifact schema {!r}".format(manifest.get("schema"))
        )
    status = manifest.get("status")
    if status not in {WRITING_STATUS, COMPLETE_STATUS, CANCELLED_STATUS}:
        raise ArtifactValidationError("manifest.status is invalid")
    if require_complete and status != COMPLETE_STATUS:
        raise IncompleteArtifactError(
            "bundle status is {!r}; only complete bundles are loadable".format(
                status
            )
        )
    _exact_keys(manifest, _MANIFEST_KEYS, "manifest")
    complete = status == COMPLETE_STATUS
    cancellation = manifest["cancellation"]
    if status == CANCELLED_STATUS:
        selected_cancellation = _exact_keys(
            cancellation, ("reason",), "manifest.cancellation"
        )
        _string(selected_cancellation["reason"], "manifest.cancellation.reason")
    elif cancellation is not None:
        raise ArtifactValidationError(
            "manifest.cancellation must be null unless cancelled"
        )
    for name in ("run_id", "stage_id"):
        _string(manifest[name], "manifest." + name)
    for name in (
        "request_fingerprint",
        "project_fingerprint",
        "stage_input_fingerprint",
    ):
        _fingerprint(manifest[name], "manifest." + name)
    implementation = _exact_keys(
        manifest["implementation"],
        ("provenance", "fingerprint"),
        "manifest.implementation",
    )
    provenance = _normalised_mapping(
        implementation["provenance"], "manifest.implementation.provenance"
    )
    if compute_fingerprint(provenance) != _fingerprint(
        implementation["fingerprint"], "manifest.implementation.fingerprint"
    ):
        raise ArtifactValidationError("implementation fingerprint is stale")

    upstream = _exact_keys(
        manifest["upstream_diagonal_q"],
        (
            "schema",
            "path",
            "artifact_fingerprint",
            "run_id",
            "stage_id",
            "final_stationary_variance",
        ),
        "manifest.upstream_diagonal_q",
    )
    if upstream["schema"] != DIAGONAL_Q_ESTIMATE_SCHEMA:
        raise ArtifactValidationError("upstream Q schema is invalid")
    _string(upstream["path"], "manifest.upstream_diagonal_q.path")
    _fingerprint(
        upstream["artifact_fingerprint"],
        "manifest.upstream_diagonal_q.artifact_fingerprint",
    )
    _string(upstream["run_id"], "manifest.upstream_diagonal_q.run_id")
    _string(upstream["stage_id"], "manifest.upstream_diagonal_q.stage_id")
    q_variance = _finite_vector_json(
        upstream["final_stationary_variance"],
        6,
        "manifest.upstream_diagonal_q.final_stationary_variance",
    )
    if np.any(q_variance <= 0.0):
        raise ArtifactValidationError("upstream Q must be strictly positive")

    bag_ids_value = manifest["selected_bag_ids"]
    if (
        not isinstance(bag_ids_value, list)
        or not bag_ids_value
        or any(not isinstance(value, str) or not value for value in bag_ids_value)
    ):
        raise ArtifactValidationError(
            "manifest.selected_bag_ids must be non-empty strings"
        )
    bag_ids = tuple(bag_ids_value)
    if bag_ids != tuple(sorted(bag_ids)) or len(set(bag_ids)) != len(bag_ids):
        raise ArtifactValidationError(
            "manifest.selected_bag_ids must be sorted and unique"
        )
    member_count = _positive_integer(
        manifest["member_count"], "manifest.member_count"
    )
    shared = _exact_keys(
        manifest["shared_static"],
        (
            "dimension",
            "raw_coordinate_semantics",
            "decoded_fields",
            "maximum_delay_seconds",
            "ensemble_rank",
            "ridge_semantics",
        ),
        "manifest.shared_static",
    )
    if shared["dimension"] != SHARED_STATIC_DIMENSION:
        raise ArtifactValidationError("shared static dimension is invalid")
    if shared["raw_coordinate_semantics"] != (
        "18_vehicle_chart_coordinates_plus_1_bounded_delay_coordinate"
    ):
        raise ArtifactValidationError("shared coordinate semantics are invalid")
    if shared["decoded_fields"] != [
        "mass",
        "inertia",
        "cog_offset",
        "force_effectiveness",
        "torque_effectiveness",
        "constant_delay",
    ]:
        raise ArtifactValidationError("decoded shared fields are invalid")
    maximum_delay = _finite_float(
        shared["maximum_delay_seconds"],
        "manifest.shared_static.maximum_delay_seconds",
    )
    if maximum_delay <= 0.0:
        raise ArtifactValidationError("maximum delay must be positive")
    rank = _nonnegative_integer(
        shared["ensemble_rank"], "manifest.shared_static.ensemble_rank"
    )
    if rank >= member_count or rank > SHARED_STATIC_DIMENSION:
        raise ArtifactValidationError("shared ensemble rank is impossible")
    if shared["ridge_semantics"] != (
        "sample_covariance_eigendecomposition_of_final_raw_19d_coordinates"
    ):
        raise ArtifactValidationError("ridge semantics are invalid")

    body = _exact_keys(
        manifest["body_wrench"],
        ("frame", "component_order", "variance_units"),
        "manifest.body_wrench",
    )
    if (
        body["frame"] != BODY_WRENCH_FRAME
        or body["component_order"] != list(BODY_WRENCH_COMPONENT_ORDER)
        or body["variance_units"] != list(BODY_WRENCH_VARIANCE_UNITS)
    ):
        raise ArtifactValidationError("body-wrench convention is invalid")
    semantics = _exact_keys(
        manifest["path_semantics"],
        (
            "kind",
            "static_coordinates_at_each_time_are_actual",
            "earlier_bags_recomputed_with_final_shared_posterior",
        ),
        "manifest.path_semantics",
    )
    if (
        semantics["kind"] != SEQUENTIAL_ENRTS_PATH_SEMANTICS
        or semantics["static_coordinates_at_each_time_are_actual"] is not True
        or semantics["earlier_bags_recomputed_with_final_shared_posterior"]
        is not False
    ):
        raise ArtifactValidationError("path semantics are invalid")

    bag_metadata = _exact_keys(manifest["bags"], bag_ids, "manifest.bags")
    for bag_id in bag_ids:
        _validate_bag_metadata(
            bag_metadata[bag_id], bag_id, member_count, q_variance
        )
    artifacts = _exact_keys(
        manifest["artifacts"], ("shared_posterior", "bags"), "manifest.artifacts"
    )
    _descriptor(
        artifacts["shared_posterior"],
        "manifest.artifacts.shared_posterior",
        complete,
    )
    bag_artifacts = _exact_keys(
        artifacts["bags"], bag_ids, "manifest.artifacts.bags"
    )
    for bag_id in bag_ids:
        _descriptor(
            bag_artifacts[bag_id],
            "manifest.artifacts.bags.{!r}".format(bag_id),
            complete,
        )
    return bag_ids, bag_metadata, artifacts


def _validate_bag_metadata(
    value: Any,
    bag_id: str,
    member_count: int,
    upstream_q: np.ndarray,
) -> None:
    location = "manifest.bags.{!r}".format(bag_id)
    selected = _exact_keys(
        value,
        (
            "source_path",
            "source_sha256",
            "source_size_bytes",
            "episode_index",
            "configuration_fingerprint",
            "time_basis",
            "requested_interval_record_seconds",
            "effective_interval_record_seconds",
            "effective_interval_local_seconds",
            "episode_provenance",
            "episode_provenance_fingerprint",
            "controller_snapshot",
            "controller_snapshot_fingerprint",
            "controller_configuration",
            "controller_configuration_fingerprint",
            "model_provenance",
            "model_provenance_fingerprint",
            "boundary_count",
            "member_count",
            "fixed_q_stationary_variance",
            "fixed_r_translation_covariance",
            "fixed_r_rotation_covariance",
            "fixed_r_fingerprint",
            "fixed_correlation_time",
            "path_semantics",
        ),
        location,
    )
    _string(selected["source_path"], location + ".source_path")
    _raw_sha256(selected["source_sha256"], location + ".source_sha256")
    _positive_integer(selected["source_size_bytes"], location + ".source_size_bytes")
    _nonnegative_integer(selected["episode_index"], location + ".episode_index")
    _configuration_fingerprint(
        selected["configuration_fingerprint"],
        location + ".configuration_fingerprint",
    )
    if selected["time_basis"] != "episode_relative_seconds_with_rosbag_record_times":
        raise ArtifactValidationError("{} time basis is invalid".format(location))
    requested = _interval(
        selected["requested_interval_record_seconds"],
        location + ".requested_interval_record_seconds",
    )
    effective_record = _interval(
        selected["effective_interval_record_seconds"],
        location + ".effective_interval_record_seconds",
    )
    effective_local = _interval(
        selected["effective_interval_local_seconds"],
        location + ".effective_interval_local_seconds",
    )
    if effective_record[0] < requested[0] - 2.0e-7 or effective_record[1] > requested[1] + 2.0e-7:
        raise ArtifactValidationError(
            "{} effective interval must lie in requested interval".format(location)
        )
    provenance = _normalised_mapping(
        selected["episode_provenance"], location + ".episode_provenance"
    )
    provenance_fp = _fingerprint(
        selected["episode_provenance_fingerprint"],
        location + ".episode_provenance_fingerprint",
    )
    if compute_fingerprint(provenance) != provenance_fp:
        raise ArtifactValidationError("{} provenance fingerprint is stale".format(location))
    for key, expected in (
        ("bag_path", selected["source_path"]),
        ("bag_sha256", selected["source_sha256"]),
        ("bag_size_bytes", selected["source_size_bytes"]),
    ):
        if provenance.get(key) != expected:
            raise ArtifactValidationError(
                "{} episode provenance {} is inconsistent".format(location, key)
            )
    snapshot = _normalised_mapping(
        selected["controller_snapshot"], location + ".controller_snapshot"
    )
    snapshot_fp = _fingerprint(
        selected["controller_snapshot_fingerprint"],
        location + ".controller_snapshot_fingerprint",
    )
    if compute_fingerprint(snapshot) != snapshot_fp:
        raise ArtifactValidationError("{} controller snapshot is stale".format(location))
    controller = _normalised_mapping(
        selected["controller_configuration"],
        location + ".controller_configuration",
    )
    controller_fp = _fingerprint(
        selected["controller_configuration_fingerprint"],
        location + ".controller_configuration_fingerprint",
    )
    if compute_fingerprint(controller) != controller_fp:
        raise ArtifactValidationError(
            "{} controller configuration is stale".format(location)
        )
    model = _normalised_mapping(
        selected["model_provenance"], location + ".model_provenance"
    )
    model_fp = _fingerprint(
        selected["model_provenance_fingerprint"],
        location + ".model_provenance_fingerprint",
    )
    if compute_fingerprint(model) != model_fp:
        raise ArtifactValidationError("{} model provenance is stale".format(location))
    boundaries = _positive_integer(selected["boundary_count"], location + ".boundary_count")
    if boundaries < 2:
        raise ArtifactValidationError("{} requires at least two boundaries".format(location))
    if _positive_integer(selected["member_count"], location + ".member_count") != member_count:
        raise ArtifactValidationError("{} member count is inconsistent".format(location))
    q = _finite_vector_json(
        selected["fixed_q_stationary_variance"],
        6,
        location + ".fixed_q_stationary_variance",
    )
    if not np.array_equal(q, upstream_q):
        raise ArtifactValidationError("{} fixed Q differs from upstream".format(location))
    translation = _covariance3(
        selected["fixed_r_translation_covariance"],
        location + ".fixed_r_translation_covariance",
    )
    rotation = _covariance3(
        selected["fixed_r_rotation_covariance"],
        location + ".fixed_r_rotation_covariance",
    )
    r_fp = _fingerprint(selected["fixed_r_fingerprint"], location + ".fixed_r_fingerprint")
    if compute_fingerprint(
        {"translation": translation.tolist(), "rotation_tangent": rotation.tolist()}
    ) != r_fp:
        raise ArtifactValidationError("{} fixed R fingerprint is stale".format(location))
    correlation = _finite_float(
        selected["fixed_correlation_time"], location + ".fixed_correlation_time"
    )
    if correlation <= 0.0:
        raise ArtifactValidationError("{} correlation time must be positive".format(location))
    if selected["path_semantics"] != SEQUENTIAL_ENRTS_PATH_SEMANTICS:
        raise ArtifactValidationError("{} path semantics are invalid".format(location))


def _validate_shared_arrays(
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    location: str,
) -> Mapping[str, np.ndarray]:
    members = int(manifest["member_count"])
    member_id = _integer(arrays, "member_id", (members,), location)
    if np.unique(member_id).size != members:
        raise ArtifactValidationError("{} member IDs are not unique".format(location))
    initial = _numeric(
        arrays,
        "initial_shared_coordinates",
        (members, SHARED_STATIC_DIMENSION),
        location,
    )
    final = _numeric(
        arrays,
        "final_shared_coordinates",
        (members, SHARED_STATIC_DIMENSION),
        location,
    )
    mass = _numeric(arrays, "mass", (members,), location)
    inertia = _numeric(arrays, "inertia", (members, 3, 3), location)
    cog = _numeric(arrays, "cog_offset", (members, 3), location)
    force = _numeric(arrays, "force_effectiveness", (members, 4), location)
    torque = _numeric(arrays, "torque_effectiveness", (members, 4), location)
    delay = _numeric(arrays, "constant_delay", (members,), location)
    maximum_delay = manifest["shared_static"]["maximum_delay_seconds"]
    try:
        chart_nominal = VehicleParameters(
            mass=float(
                _numeric(arrays, "chart_nominal_mass", (1,), location)[0]
            ),
            inertia=_numeric(
                arrays, "chart_nominal_inertia", (3, 3), location
            ),
            cog_offset=_numeric(
                arrays, "chart_nominal_cog_offset", (3,), location
            ),
            force_effectiveness=_numeric(
                arrays,
                "chart_nominal_force_effectiveness",
                (4,),
                location,
            ),
            torque_effectiveness=_numeric(
                arrays,
                "chart_nominal_torque_effectiveness",
                (4,),
                location,
            ),
            linear_drag=_numeric(
                arrays, "chart_nominal_linear_drag", (3,), location
            ),
            angular_drag=_numeric(
                arrays, "chart_nominal_angular_drag", (3,), location
            ),
        )
        chart = VehicleParameterChart(chart_nominal)
        delay_chart = BoundedDelayChart(maximum_delay)
        decoded = tuple(
            chart.decode(value[:PARAMETER_DIMENSION]) for value in final
        )
        decoded_delay = np.asarray(
            [delay_chart.decode(value[-1]) for value in final]
        )
    except (TypeError, ValueError, FloatingPointError) as error:
        raise ArtifactValidationError(
            "{} stored chart cannot decode the raw ensemble".format(location)
        ) from error
    if (
        np.any(mass <= 0.0)
        or np.any(force <= 0.0)
        or np.any(torque <= 0.0)
        or np.any(delay <= 0.0)
        or np.any(delay >= maximum_delay)
        or any(
            not np.allclose(value, value.T, rtol=1.0e-12, atol=1.0e-14)
            or np.any(np.linalg.eigvalsh(value) <= 0.0)
            for value in inertia
        )
    ):
        raise ArtifactValidationError("{} decoded parameters are nonphysical".format(location))
    decoded_fields = (
        (mass, np.asarray([value.mass for value in decoded])),
        (inertia, np.asarray([value.inertia for value in decoded])),
        (cog, np.asarray([value.cog_offset for value in decoded])),
        (
            force,
            np.asarray([value.force_effectiveness for value in decoded]),
        ),
        (
            torque,
            np.asarray([value.torque_effectiveness for value in decoded]),
        ),
        (delay, decoded_delay),
    )
    if any(
        not np.allclose(stored, expected_value, rtol=1.0e-12, atol=1.0e-14)
        for stored, expected_value in decoded_fields
    ):
        raise ArtifactValidationError(
            "{} decoded physical ensemble differs from raw coordinates".format(
                location
            )
        )
    covariance = _numeric(
        arrays,
        "ridge_covariance",
        (SHARED_STATIC_DIMENSION, SHARED_STATIC_DIMENSION),
        location,
    )
    eigenvalues = _numeric(
        arrays, "ridge_eigenvalues", (SHARED_STATIC_DIMENSION,), location
    )
    eigenvectors = _numeric(
        arrays,
        "ridge_eigenvectors",
        (SHARED_STATIC_DIMENSION, SHARED_STATIC_DIMENSION),
        location,
    )
    expected = _numeric(
        arrays,
        "expected_physical_ridge_direction",
        (SHARED_STATIC_DIMENSION,),
        location,
    )
    expected_variance = _numeric(
        arrays, "expected_physical_ridge_variance", (1,), location
    )[0]
    rank = int(_integer(arrays, "ensemble_rank", (1,), location)[0])
    recomputed = np.cov(final, rowvar=False)
    values, vectors = np.linalg.eigh(recomputed)
    recomputed_rank = int(np.linalg.matrix_rank(final - np.mean(final, axis=0)))
    if (
        not np.allclose(covariance, recomputed, rtol=1.0e-12, atol=1.0e-14)
        or not np.allclose(eigenvalues, values, rtol=1.0e-11, atol=1.0e-14)
        or not np.allclose(
            eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T,
            covariance,
            rtol=1.0e-10,
            atol=1.0e-13,
        )
        or not np.allclose(eigenvectors.T @ eigenvectors, np.eye(19), atol=1.0e-10)
        or not np.isclose(np.linalg.norm(expected), 1.0, atol=1.0e-12)
        or not np.allclose(
            expected,
            np.concatenate((chart.ridge_direction(), (0.0,)))
            / np.linalg.norm(chart.ridge_direction()),
            rtol=0.0,
            atol=1.0e-14,
        )
        or not np.isclose(expected_variance, expected @ covariance @ expected, rtol=1.0e-12, atol=1.0e-14)
        or rank != recomputed_rank
        or rank != manifest["shared_static"]["ensemble_rank"]
    ):
        raise ArtifactValidationError("{} ridge diagnostics are inconsistent".format(location))
    # Silence accidental replacement by arrays with correct shapes but wrong field order.
    if cog.shape != (members, 3) or initial.shape != final.shape:
        raise ArtifactValidationError("{} shared arrays are misaligned".format(location))
    return {key: np.asarray(value).copy() for key, value in arrays.items()}


def _validate_quaternions(value: np.ndarray, location: str) -> None:
    norms = np.linalg.norm(value, axis=-1)
    if not np.allclose(norms, 1.0, rtol=1.0e-10, atol=1.0e-10):
        raise ArtifactValidationError(
            "{} contains non-unit quaternions".format(location)
        )


def _validate_bag_arrays(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    bag_id: str,
    common_member_id: np.ndarray,
    upstream_q: np.ndarray,
    location: str,
) -> Mapping[str, np.ndarray]:
    members = common_member_id.size
    boundaries = int(metadata["boundary_count"])
    string_fields = (
        ("bag_id", bag_id),
        ("source_path", metadata["source_path"]),
        ("source_sha256", metadata["source_sha256"]),
        ("configuration_fingerprint", metadata["configuration_fingerprint"]),
        (
            "episode_provenance_fingerprint",
            metadata["episode_provenance_fingerprint"],
        ),
        (
            "controller_snapshot_fingerprint",
            metadata["controller_snapshot_fingerprint"],
        ),
        (
            "controller_configuration_fingerprint",
            metadata["controller_configuration_fingerprint"],
        ),
        (
            "model_provenance_fingerprint",
            metadata["model_provenance_fingerprint"],
        ),
    )
    for key, expected in string_fields:
        if _unicode_scalar(arrays, key, location) != expected:
            raise ArtifactValidationError(
                "{}:{} does not match manifest".format(location, key)
            )
    source_size = int(_integer(arrays, "source_size_bytes", (1,), location)[0])
    episode_index = int(_integer(arrays, "episode_index", (1,), location)[0])
    if source_size != metadata["source_size_bytes"] or episode_index != metadata["episode_index"]:
        raise ArtifactValidationError(
            "{} source size or episode index differs from manifest".format(location)
        )
    member_id = _integer(arrays, "member_id", (members,), location)
    if not np.array_equal(member_id, common_member_id):
        raise ArtifactValidationError("{} member IDs are misaligned".format(location))
    times = _numeric(arrays, "times", (boundaries,), location)
    record_times = _numeric(arrays, "record_times", (boundaries,), location)
    if (
        times[0] != 0.0
        or np.any(np.diff(times) <= 0.0)
        or np.any(np.diff(record_times) <= 0.0)
        or not np.allclose(times, record_times - record_times[0], rtol=0.0, atol=2.0e-7)
    ):
        raise ArtifactValidationError("{} time bases are inconsistent".format(location))
    for key in (
        "requested_interval_record_seconds",
        "effective_interval_record_seconds",
        "effective_interval_local_seconds",
    ):
        stored = _numeric(arrays, key, (2,), location)
        if not np.array_equal(stored, np.asarray(metadata[key], dtype=float)):
            raise ArtifactValidationError(
                "{}:{} differs from manifest".format(location, key)
            )
    effective_record = arrays["effective_interval_record_seconds"]
    if not np.allclose(
        effective_record, (record_times[0], record_times[-1]), rtol=0.0, atol=2.0e-7
    ):
        raise ArtifactValidationError("{} effective record interval is stale".format(location))
    if not np.isclose(
        times[-1],
        metadata["effective_interval_local_seconds"][1]
        - metadata["effective_interval_local_seconds"][0],
        rtol=0.0,
        atol=2.0e-7,
    ):
        raise ArtifactValidationError("{} effective local duration is stale".format(location))

    observed_position = _numeric(arrays, "observation_position", (boundaries, 3), location)
    observed_orientation = _numeric(
        arrays, "observation_orientation_xyzw", (boundaries, 4), location
    )
    reference_position = _numeric(arrays, "reference_position", (boundaries, 3), location)
    reference_orientation = _numeric(
        arrays, "reference_orientation_xyzw", (boundaries, 4), location
    )
    nominal_position = _numeric(arrays, "nominal_position", (boundaries, 3), location)
    nominal_orientation = _numeric(
        arrays, "nominal_orientation_xyzw", (boundaries, 4), location
    )
    for key, width in (
        ("nominal_linear_velocity", 3),
        ("nominal_angular_velocity", 3),
        ("nominal_controller_integral", 6),
        ("nominal_commanded_thrust", 4),
        ("nominal_commanded_gimbal_angle", 4),
        ("nominal_actuator_thrust", 4),
        ("nominal_actuator_gimbal_angle", 4),
        ("nominal_body_wrench", 6),
    ):
        _numeric(arrays, key, (boundaries, width), location)
    for name, value in (
        ("observation", observed_orientation),
        ("reference", reference_orientation),
        ("nominal", nominal_orientation),
    ):
        _validate_quaternions(value, "{} {} orientation".format(location, name))

    q = _numeric(arrays, "fixed_q_stationary_variance", (6,), location)
    if not np.array_equal(q, upstream_q) or not np.array_equal(
        q, np.asarray(metadata["fixed_q_stationary_variance"])
    ):
        raise ArtifactValidationError("{} fixed Q is inconsistent".format(location))
    translation_r = _covariance3(
        _numeric(arrays, "fixed_r_translation_covariance", (3, 3), location),
        location + ":fixed_r_translation_covariance",
    )
    rotation_r = _covariance3(
        _numeric(arrays, "fixed_r_rotation_covariance", (3, 3), location),
        location + ":fixed_r_rotation_covariance",
    )
    if not np.array_equal(translation_r, np.asarray(metadata["fixed_r_translation_covariance"])) or not np.array_equal(
        rotation_r, np.asarray(metadata["fixed_r_rotation_covariance"])
    ):
        raise ArtifactValidationError("{} fixed R differs from manifest".format(location))
    correlation = float(_numeric(arrays, "fixed_correlation_time", (1,), location)[0])
    if correlation != metadata["fixed_correlation_time"]:
        raise ArtifactValidationError("{} correlation time differs from manifest".format(location))

    initial_shared = _numeric(
        arrays,
        "initial_shared_coordinates",
        (members, SHARED_STATIC_DIMENSION),
        location,
    )
    _numeric(
        arrays,
        "initial_local_coordinates",
        (members, LOCAL_INITIAL_DIMENSION),
        location,
    )
    initial_specs = (
        ("initial_position", 3),
        ("initial_orientation_xyzw", 4),
        ("initial_linear_velocity", 3),
        ("initial_angular_velocity", 3),
        ("initial_controller_integral", 6),
        ("initial_actuator_thrust", 4),
        ("initial_actuator_gimbal_angle", 4),
        ("initial_residual_wrench", 6),
    )
    for key, width in initial_specs:
        _numeric(arrays, key, (members, width), location)
    _boolean(
        arrays,
        "initial_controller_roll_pitch_integration_active",
        (members,),
        location,
    )
    _validate_quaternions(
        arrays["initial_orientation_xyzw"], location + " initial orientation"
    )

    static_smoothed = _numeric(
        arrays,
        "static_smoothed_coordinates",
        (members, boundaries, SHARED_STATIC_DIMENSION),
        location,
    )
    smoothed_specs = (
        ("smoothed_position", 3),
        ("smoothed_orientation_xyzw", 4),
        ("smoothed_linear_velocity", 3),
        ("smoothed_angular_velocity", 3),
        ("smoothed_controller_integral", 6),
        ("smoothed_actuator_thrust", 4),
        ("smoothed_actuator_gimbal_angle", 4),
        ("smoothed_residual_wrench", 6),
    )
    for key, width in smoothed_specs:
        _numeric(arrays, key, (members, boundaries, width), location)
    _boolean(
        arrays,
        "smoothed_controller_roll_pitch_integration_active",
        (members, boundaries),
        location,
    )
    _validate_quaternions(
        arrays["smoothed_orientation_xyzw"], location + " smoothed orientation"
    )
    for key in (
        "observed_correction_translation",
        "observed_correction_rotation_vector",
        "reference_correction_translation",
        "reference_correction_rotation_vector",
    ):
        _numeric(arrays, key, (boundaries, 3), location)
    for key in (
        "smoothed_correction_translation",
        "smoothed_correction_rotation_vector",
    ):
        _numeric(arrays, key, (members, boundaries, 3), location)
    observed_correction = correction_transform_path(
        nominal_position,
        nominal_orientation,
        observed_position,
        observed_orientation,
    )
    reference_correction = correction_transform_path(
        nominal_position,
        nominal_orientation,
        reference_position,
        reference_orientation,
    )
    smoothed_correction = _smoothed_corrections(
        ClosedLoopTrajectory(
            times,
            nominal_position,
            nominal_orientation,
            arrays["nominal_linear_velocity"],
            arrays["nominal_angular_velocity"],
            arrays["nominal_controller_integral"],
            arrays["nominal_commanded_thrust"],
            arrays["nominal_commanded_gimbal_angle"],
            arrays["nominal_actuator_thrust"],
            arrays["nominal_actuator_gimbal_angle"],
            arrays["nominal_body_wrench"],
        ),
        arrays["smoothed_position"],
        arrays["smoothed_orientation_xyzw"],
    )
    correction_pairs = (
        ("observed_correction_translation", observed_correction[0]),
        ("observed_correction_rotation_vector", observed_correction[1]),
        ("reference_correction_translation", reference_correction[0]),
        ("reference_correction_rotation_vector", reference_correction[1]),
        ("smoothed_correction_translation", smoothed_correction[0]),
        ("smoothed_correction_rotation_vector", smoothed_correction[1]),
    )
    if any(
        not np.allclose(arrays[key], expected, rtol=1.0e-12, atol=1.0e-13)
        for key, expected in correction_pairs
    ):
        raise ArtifactValidationError("{} correction transforms are stale".format(location))
    likelihood = _numeric(
        arrays, "filter_log_likelihood_by_time", (boundaries,), location
    )
    nis = _numeric(arrays, "filter_nis", (boundaries,), location)
    if np.any(nis < 0.0) or np.any(~np.isfinite(likelihood)):
        raise ArtifactValidationError("{} filter diagnostics are invalid".format(location))
    mass = _numeric(arrays, "applied_model_mass", (members, boundaries), location)
    delay = _numeric(arrays, "applied_model_delay", (members, boundaries), location)
    if np.any(mass <= 0.0) or np.any(delay < 0.0):
        raise ArtifactValidationError("{} applied model history is invalid".format(location))
    if initial_shared.shape != static_smoothed[:, 0, :].shape:
        raise ArtifactValidationError("{} static coordinates are misaligned".format(location))
    return {key: np.asarray(value).copy() for key, value in arrays.items()}


def _load_complete(
    root: Path, manifest: Mapping[str, Any]
) -> AugmentedParameterArtifactBundle:
    bag_ids, metadata, artifacts = _validate_manifest(
        manifest, require_complete=True
    )
    shared_path, shared_payload = _verified_payload(
        root,
        artifacts["shared_posterior"],
        "manifest.artifacts.shared_posterior",
    )
    shared_arrays = _load_npz_exact(shared_payload, shared_path, _SHARED_KEYS)
    shared = _validate_shared_arrays(shared_arrays, manifest, str(shared_path))
    member_id = shared["member_id"]
    upstream_q = np.asarray(
        manifest["upstream_diagonal_q"]["final_stationary_variance"],
        dtype=float,
    )
    used_paths = {shared_path}
    bags: Dict[str, Mapping[str, np.ndarray]] = {}
    preceding_final = None
    for index, bag_id in enumerate(bag_ids):
        descriptor = artifacts["bags"][bag_id]
        path, payload = _verified_payload(
            root,
            descriptor,
            "manifest.artifacts.bags.{!r}".format(bag_id),
        )
        if path in used_paths:
            raise ArtifactValidationError("artifact payload paths must be unique")
        used_paths.add(path)
        raw = _load_npz_exact(payload, path, _BAG_KEYS)
        arrays = _validate_bag_arrays(
            raw,
            metadata[bag_id],
            bag_id,
            member_id,
            upstream_q,
            str(path),
        )
        if index == 0 and not np.array_equal(
            arrays["initial_shared_coordinates"],
            shared["initial_shared_coordinates"],
        ):
            raise ArtifactValidationError(
                "first bag does not start at the shared prior"
            )
        if preceding_final is not None and not np.array_equal(
            arrays["initial_shared_coordinates"], preceding_final
        ):
            raise ArtifactValidationError(
                "sequential shared-member carry is broken"
            )
        preceding_final = arrays["static_smoothed_coordinates"][:, -1, :]
        bags[bag_id] = arrays
    if not np.array_equal(preceding_final, shared["final_shared_coordinates"]):
        raise ArtifactValidationError(
            "last bag marginal does not end at final shared posterior"
        )
    return AugmentedParameterArtifactBundle(
        root=root,
        manifest=json.loads(canonical_json_bytes(manifest).decode("utf-8")),
        shared_posterior=shared,
        bags=bags,
    )


def _ordered_inputs(
    values: Iterable[AugmentedParameterArtifactBagInput],
    expected_ids: Tuple[str, ...],
) -> Tuple[AugmentedParameterArtifactBagInput, ...]:
    try:
        inputs = tuple(values)
    except TypeError as error:
        raise TypeError(
            "bag_inputs must be an iterable of artifact bag inputs"
        ) from error
    if any(
        not isinstance(value, AugmentedParameterArtifactBagInput)
        for value in inputs
    ):
        raise TypeError(
            "bag_inputs must contain AugmentedParameterArtifactBagInput"
        )
    ordered = tuple(sorted(inputs, key=lambda value: value.bag_id))
    identifiers = tuple(value.bag_id for value in ordered)
    if identifiers != expected_ids:
        raise ArtifactValidationError(
            "artifact bag inputs must match result bag IDs exactly"
        )
    return ordered


def _validate_result_inputs(
    result: MultiBagAugmentedParameterResult,
    inputs: Sequence[AugmentedParameterArtifactBagInput],
) -> None:
    by_id = {value.bag_id: value for value in inputs}
    for bag_result in result.bags:
        bag_input = by_id[bag_result.bag_id]
        episode = bag_input.episode
        output = bag_result.filter_result
        if not np.array_equal(output.times, episode.observations.times):
            raise ArtifactValidationError(
                "filter and real episode times differ for {!r}".format(
                    bag_result.bag_id
                )
            )
        if not np.array_equal(
            output.member_id, result.member_id
        ) or not np.array_equal(
            bag_result.initial_ensemble.member_id, result.member_id
        ):
            raise ArtifactValidationError(
                "member identity differs for {!r}".format(bag_result.bag_id)
            )
        if not np.array_equal(
            bag_result.wrench_covariance.stationary_variance,
            result.wrench_covariance.stationary_variance,
        ):
            raise ArtifactValidationError(
                "bag {!r} did not use the fixed shared Q".format(
                    bag_result.bag_id
                )
            )


def read_augmented_parameter_manifest(
    root: Union[str, Path], *, require_complete: bool = False
) -> Mapping[str, Any]:
    """Read and strictly validate a stage-2 manifest."""

    destination = Path(root).expanduser().resolve()
    manifest = read_json(destination / MANIFEST_NAME)
    _validate_manifest(manifest, require_complete=require_complete)
    return manifest


def load_augmented_parameter_artifact(
    root: Union[str, Path],
) -> AugmentedParameterArtifactBundle:
    """Load a complete fixed-Q augmented-parameter artifact."""

    destination = Path(root).expanduser().resolve()
    manifest = read_json(destination / MANIFEST_NAME)
    # The incomplete status check intentionally precedes nested/payload work.
    if manifest.get("schema") != AUGMENTED_PARAMETER_ESTIMATE_SCHEMA:
        raise UnsupportedArtifactSchema(
            "unsupported artifact schema {!r}".format(manifest.get("schema"))
        )
    if manifest.get("status") != COMPLETE_STATUS:
        raise IncompleteArtifactError(
            "bundle status is {!r}; only complete bundles are loadable".format(
                manifest.get("status")
            )
        )
    return _load_complete(destination, manifest)


def write_augmented_parameter_artifact(
    root: Union[str, Path],
    *,
    run_id: str,
    stage_id: str,
    request_fingerprint: str,
    project_fingerprint: str,
    stage_input_fingerprint: str,
    implementation_provenance: Mapping[str, Any],
    upstream_diagonal_q_path: Union[str, Path],
    upstream_diagonal_q_fingerprint: str,
    bag_inputs: Iterable[AugmentedParameterArtifactBagInput],
    result: MultiBagAugmentedParameterResult,
) -> Path:
    """Atomically publish one complete fixed-Q stage-2 bundle."""

    if not isinstance(result, MultiBagAugmentedParameterResult):
        raise TypeError("result must be a MultiBagAugmentedParameterResult")
    selected_run_id = _string(run_id, "run_id")
    selected_stage_id = _string(stage_id, "stage_id")
    selected_request_fingerprint = _fingerprint(
        request_fingerprint, "request_fingerprint"
    )
    selected_project_fingerprint = _fingerprint(
        project_fingerprint, "project_fingerprint"
    )
    selected_stage_input_fingerprint = _fingerprint(
        stage_input_fingerprint, "stage_input_fingerprint"
    )
    selected_implementation = _normalised_mapping(
        implementation_provenance, "implementation_provenance"
    )
    if not selected_implementation:
        raise ArtifactValidationError(
            "implementation_provenance cannot be empty"
        )
    selected_upstream_fingerprint = _fingerprint(
        upstream_diagonal_q_fingerprint,
        "upstream_diagonal_q_fingerprint",
    )
    upstream_root = Path(upstream_diagonal_q_path).expanduser().resolve()
    upstream = load_diagonal_q_artifact(upstream_root)
    actual_upstream_fingerprint = compute_fingerprint(upstream.manifest)
    if selected_upstream_fingerprint != actual_upstream_fingerprint:
        raise ArtifactValidationError(
            "upstream diagonal-Q artifact fingerprint does not match"
        )
    if upstream.bag_ids != result.bag_ids:
        raise ArtifactValidationError(
            "upstream diagonal-Q and stage-2 bag IDs differ"
        )
    if not np.array_equal(
        upstream.covariance.stationary_variance,
        result.wrench_covariance.stationary_variance,
    ):
        raise ArtifactValidationError(
            "stage-2 fixed Q differs from its upstream artifact"
        )
    inputs = _ordered_inputs(bag_inputs, result.bag_ids)
    _validate_result_inputs(result, inputs)
    input_by_id = {value.bag_id: value for value in inputs}
    shared_arrays = _decoded_shared_arrays(result, inputs)

    destination = Path(root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / MANIFEST_NAME
    if manifest_path.exists() or manifest_path.is_symlink():
        raise ArtifactStateError(
            "bundle already has a manifest: {}".format(manifest_path)
        )
    shared_relative = "shared_posterior.npz"
    bag_relatives = {
        bag_id: "bags/{:04d}.npz".format(index)
        for index, bag_id in enumerate(result.bag_ids)
    }
    shared_path = _safe_payload_path(destination, shared_relative)
    bag_paths = {
        bag_id: _safe_payload_path(destination, relative)
        for bag_id, relative in bag_relatives.items()
    }

    per_bag_arrays: Dict[str, Mapping[str, np.ndarray]] = {}
    bag_metadata: Dict[str, Mapping[str, Any]] = {}
    for bag_result in result.bags:
        bag_input = input_by_id[bag_result.bag_id]
        episode = bag_input.episode
        provenance_mapping = _episode_provenance_mapping(episode.provenance)
        snapshot_mapping = _controller_snapshot_mapping(
            episode.controller_snapshot
        )
        controller_mapping = _controller_configuration_mapping(
            episode.controller_configuration
        )
        provenance_fingerprint = compute_fingerprint(provenance_mapping)
        snapshot_fingerprint = compute_fingerprint(snapshot_mapping)
        controller_fingerprint = compute_fingerprint(controller_mapping)
        model_fingerprint = compute_fingerprint(bag_input.model_provenance)
        arrays = _bag_arrays(
            bag_input,
            bag_result,
            provenance_fingerprint,
            snapshot_fingerprint,
            controller_fingerprint,
            model_fingerprint,
        )
        per_bag_arrays[bag_result.bag_id] = arrays
        r_mapping = {
            "translation": (
                bag_result.observation_covariance.translation.tolist()
            ),
            "rotation_tangent": (
                bag_result.observation_covariance.rotation_tangent.tolist()
            ),
        }
        provenance = episode.provenance
        bag_metadata[bag_result.bag_id] = {
            "source_path": provenance.bag_path,
            "source_sha256": provenance.bag_sha256,
            "source_size_bytes": provenance.bag_size_bytes,
            "episode_index": bag_input.episode_index,
            "configuration_fingerprint": (
                bag_input.configuration_fingerprint
            ),
            "time_basis": (
                "episode_relative_seconds_with_rosbag_record_times"
            ),
            "requested_interval_record_seconds": [
                provenance.requested_window_start,
                provenance.requested_window_end,
            ],
            "effective_interval_record_seconds": [
                episode.window_start_record_time,
                episode.window_end_record_time,
            ],
            "effective_interval_local_seconds": [
                episode.window_start_local_time,
                episode.window_end_local_time,
            ],
            "episode_provenance": provenance_mapping,
            "episode_provenance_fingerprint": provenance_fingerprint,
            "controller_snapshot": snapshot_mapping,
            "controller_snapshot_fingerprint": snapshot_fingerprint,
            "controller_configuration": controller_mapping,
            "controller_configuration_fingerprint": controller_fingerprint,
            "model_provenance": bag_input.model_provenance,
            "model_provenance_fingerprint": model_fingerprint,
            "boundary_count": int(bag_result.filter_result.times.size),
            "member_count": result.member_count,
            "fixed_q_stationary_variance": (
                result.wrench_covariance.stationary_variance.tolist()
            ),
            "fixed_r_translation_covariance": (
                bag_result.observation_covariance.translation.tolist()
            ),
            "fixed_r_rotation_covariance": (
                bag_result.observation_covariance.rotation_tangent.tolist()
            ),
            "fixed_r_fingerprint": compute_fingerprint(r_mapping),
            "fixed_correlation_time": bag_result.correlation_time,
            "path_semantics": SEQUENTIAL_ENRTS_PATH_SEMANTICS,
        }

    implementation_fingerprint = compute_fingerprint(
        selected_implementation
    )
    manifest: Dict[str, Any] = {
        "schema": AUGMENTED_PARAMETER_ESTIMATE_SCHEMA,
        "status": WRITING_STATUS,
        "run_id": selected_run_id,
        "stage_id": selected_stage_id,
        "request_fingerprint": selected_request_fingerprint,
        "project_fingerprint": selected_project_fingerprint,
        "stage_input_fingerprint": selected_stage_input_fingerprint,
        "implementation": {
            "provenance": selected_implementation,
            "fingerprint": implementation_fingerprint,
        },
        "upstream_diagonal_q": {
            "schema": DIAGONAL_Q_ESTIMATE_SCHEMA,
            "path": str(upstream_root),
            "artifact_fingerprint": actual_upstream_fingerprint,
            "run_id": upstream.manifest["run_id"],
            "stage_id": upstream.manifest["stage_id"],
            "final_stationary_variance": (
                upstream.covariance.stationary_variance.tolist()
            ),
        },
        "selected_bag_ids": list(result.bag_ids),
        "member_count": result.member_count,
        "shared_static": {
            "dimension": SHARED_STATIC_DIMENSION,
            "raw_coordinate_semantics": (
                "18_vehicle_chart_coordinates_plus_1_bounded_delay_coordinate"
            ),
            "decoded_fields": [
                "mass",
                "inertia",
                "cog_offset",
                "force_effectiveness",
                "torque_effectiveness",
                "constant_delay",
            ],
            "maximum_delay_seconds": result.maximum_delay,
            "ensemble_rank": int(shared_arrays["ensemble_rank"][0]),
            "ridge_semantics": (
                "sample_covariance_eigendecomposition_of_final_raw_19d_coordinates"
            ),
        },
        "body_wrench": {
            "frame": BODY_WRENCH_FRAME,
            "component_order": list(BODY_WRENCH_COMPONENT_ORDER),
            "variance_units": list(BODY_WRENCH_VARIANCE_UNITS),
        },
        "path_semantics": {
            "kind": SEQUENTIAL_ENRTS_PATH_SEMANTICS,
            "static_coordinates_at_each_time_are_actual": True,
            "earlier_bags_recomputed_with_final_shared_posterior": False,
        },
        "bags": bag_metadata,
        "artifacts": {
            "shared_posterior": _empty_descriptor(shared_relative),
            "bags": {
                bag_id: _empty_descriptor(bag_relatives[bag_id])
                for bag_id in result.bag_ids
            },
        },
        "cancellation": None,
    }
    _validate_manifest(manifest, require_complete=False)
    write_json_atomic(manifest_path, manifest)
    write_npz_atomic(shared_path, shared_arrays)
    for bag_id in result.bag_ids:
        write_npz_atomic(bag_paths[bag_id], per_bag_arrays[bag_id])

    candidate = json.loads(canonical_json_bytes(manifest).decode("utf-8"))
    candidate["status"] = COMPLETE_STATUS
    candidate["artifacts"]["shared_posterior"] = _complete_descriptor(
        shared_path, destination
    )
    candidate["artifacts"]["bags"] = {
        bag_id: _complete_descriptor(bag_paths[bag_id], destination)
        for bag_id in result.bag_ids
    }
    _load_complete(destination, candidate)
    current = read_json(manifest_path)
    if current.get("status") != WRITING_STATUS or canonical_json_bytes(
        current
    ) != canonical_json_bytes(manifest):
        raise ArtifactStateError("writing manifest changed before completion")
    write_json_atomic(manifest_path, candidate)
    return destination


def mark_augmented_parameter_artifact_cancelled(
    root: Union[str, Path], reason: str
) -> Path:
    """Atomically make cancellation authoritative for a writing bundle."""

    selected_reason = _string(reason, "cancellation reason")
    destination = Path(root).expanduser().resolve()
    manifest_path = destination / MANIFEST_NAME
    manifest = read_json(manifest_path)
    _validate_manifest(manifest, require_complete=False)
    status = manifest["status"]
    if status == COMPLETE_STATUS:
        raise ArtifactStateError("a complete artifact cannot be cancelled")
    if status == CANCELLED_STATUS:
        raise ArtifactStateError("artifact is already cancelled")
    candidate = json.loads(canonical_json_bytes(manifest).decode("utf-8"))
    candidate["status"] = CANCELLED_STATUS
    candidate["cancellation"] = {"reason": selected_reason}
    _validate_manifest(candidate, require_complete=False)
    current = read_json(manifest_path)
    if canonical_json_bytes(current) != canonical_json_bytes(manifest):
        raise ArtifactStateError("writing manifest changed before cancellation")
    write_json_atomic(manifest_path, candidate)
    return destination


__all__ = [
    "AUGMENTED_PARAMETER_ESTIMATE_SCHEMA",
    "SEQUENTIAL_ENRTS_PATH_SEMANTICS",
    "AugmentedParameterArtifactBagInput",
    "AugmentedParameterArtifactBundle",
    "diagonal_q_artifact_fingerprint",
    "load_augmented_parameter_artifact",
    "mark_augmented_parameter_artifact_cancelled",
    "read_augmented_parameter_manifest",
    "write_augmented_parameter_artifact",
]
