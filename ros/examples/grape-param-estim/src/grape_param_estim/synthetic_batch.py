"""ROS-free truth generators and strict artifacts for sparse batch estimation.

The helpers in this module generate data in the exact coordinates consumed by
the production batch factors.  They do not run a second, simplified estimator
and they do not use finite-difference derivatives.  In particular, the
perfect-model trajectory is advanced by Newton corrections formed from the
analytic rigid-body factor Jacobian itself.
"""

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence, Tuple

import numpy as np

from grape_param_estim.batch.factors.dynamics import (
    DynamicsResidualEvaluation,
    evaluate_raw_dynamics_residual,
)
from grape_param_estim.batch.factors.dynamics_factor import (
    SPECIFIC_ACCELERATION_QUANTITY,
    specific_acceleration_statistical_residual,
)
from grape_param_estim.batch.laplace_em import (
    DiagonalQDefinition,
    ExpectedResidualMoments,
    QIntervalModel,
)
from grape_param_estim.geometry import so3_exp
from grape_param_estim.parameterization import (
    PARAMETER_DIMENSION,
    VehicleParameterChart,
)
from grape_param_estim.system import GrapeGeometry, VehicleParameters


_GRAVITY_WORLD = np.asarray((0.0, 0.0, -9.80665), dtype=float)

SYNTHETIC_BATCH_TRUTH_SCHEMA = "grape-param-estim/synthetic-batch-truth/v1"
SYNTHETIC_BATCH_TRUTH_SUMMARY_SCHEMA = (
    "grape-param-estim/synthetic-batch-truth-summary/v1"
)

_GENERATOR_PROVENANCE = (
    "grape_param_estim.synthetic_batch.generate_perfect_model_batch_trajectory"
)
_DYNAMICS_FACTOR_PROVENANCE = (
    "grape_param_estim.batch.factors.dynamics.evaluate_raw_dynamics_residual"
)
_PARAMETER_COORDINATE_ORDER = (
    "log_mass_scale",
    "relative_log_inertia_xx",
    "relative_log_inertia_yy",
    "relative_log_inertia_zz",
    "relative_log_inertia_xy",
    "relative_log_inertia_xz",
    "relative_log_inertia_yz",
    "cog_offset_x_m",
    "cog_offset_y_m",
    "cog_offset_z_m",
    "log_force_effectiveness_0",
    "log_force_effectiveness_1",
    "log_force_effectiveness_2",
    "log_force_effectiveness_3",
    "log_torque_effectiveness_0",
    "log_torque_effectiveness_1",
    "log_torque_effectiveness_2",
    "log_torque_effectiveness_3",
)
_SYNTHETIC_BATCH_UNITS = {
    "times": "s",
    "position": "m; world frame",
    "rotation": "dimensionless SO(3); body-to-world",
    "linear_velocity": "m/s; world frame",
    "angular_velocity": "rad/s; body frame",
    "actuator_thrust": "N; per rotor",
    "gimbal_angle": "rad; per rotor",
    "truth_parameter_coordinates": "mixed chart coordinates; see coordinate_order",
    "truth_mass": "kg",
    "truth_inertia": "kg*m^2; body frame",
    "truth_cog_offset": "m; body frame",
    "truth_force_effectiveness": "dimensionless; per rotor",
    "truth_torque_effectiveness": "dimensionless; per rotor",
    "truth_linear_drag": "N/(m/s); body axes",
    "truth_angular_drag": "N*m/(rad/s); body axes",
}
_SYNTHETIC_BATCH_PAYLOAD_NAMES = (
    "metadata_json",
    "times",
    "position",
    "rotation",
    "linear_velocity",
    "angular_velocity",
    "actuator_thrust",
    "gimbal_angle",
    "truth_parameter_coordinates",
    "truth_mass",
    "truth_inertia",
    "truth_cog_offset",
    "truth_force_effectiveness",
    "truth_torque_effectiveness",
    "truth_linear_drag",
    "truth_angular_drag",
)
_SYNTHETIC_BATCH_ARCHIVE_NAMES = frozenset(
    _SYNTHETIC_BATCH_PAYLOAD_NAMES + ("payload_sha256",)
)


def _immutable_array(value: object, shape: Tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite array with shape {}".format(name, shape))
    result = result.copy()
    result.setflags(write=False)
    return result


def _positive_vector(value: object, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.shape != (size,)
        or not np.all(np.isfinite(result))
        or np.any(result <= 0.0)
    ):
        raise ValueError("{} must contain {} positive values".format(name, size))
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PerfectModelBatchTrajectory:
    """A variable-step trajectory satisfying the production dynamics factor."""

    times: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    actuator_thrust: np.ndarray
    gimbal_angle: np.ndarray
    parameter_chart: VehicleParameterChart
    truth_parameter_coordinates: np.ndarray
    geometry: GrapeGeometry

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        if (
            times.ndim != 1
            or times.size < 2
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("times must be a strictly increasing finite vector")
        count = int(times.size)
        times = times.copy()
        times.setflags(write=False)
        object.__setattr__(self, "times", times)
        for name, shape in (
            ("position", (count, 3)),
            ("rotation", (count, 3, 3)),
            ("linear_velocity", (count, 3)),
            ("angular_velocity", (count, 3)),
            ("actuator_thrust", (count, 4)),
            ("gimbal_angle", (count, 4)),
        ):
            object.__setattr__(
                self,
                name,
                _immutable_array(getattr(self, name), shape, name),
            )
        if not np.allclose(
            np.einsum("nji,njk->nik", self.rotation, self.rotation),
            np.broadcast_to(np.eye(3), (count, 3, 3)),
            rtol=0.0,
            atol=2.0e-10,
        ) or not np.allclose(
            np.linalg.det(self.rotation), np.ones(count), rtol=0.0, atol=2.0e-10
        ):
            raise ValueError("rotation must contain proper rotation matrices")
        if not isinstance(self.parameter_chart, VehicleParameterChart):
            raise TypeError("parameter_chart must be VehicleParameterChart")
        object.__setattr__(
            self,
            "truth_parameter_coordinates",
            _immutable_array(
                self.truth_parameter_coordinates,
                (PARAMETER_DIMENSION,),
                "truth_parameter_coordinates",
            ),
        )
        if not isinstance(self.geometry, GrapeGeometry):
            raise TypeError("geometry must be GrapeGeometry")

    @property
    def interval_count(self) -> int:
        return int(self.times.size - 1)

    @property
    def time_step(self) -> np.ndarray:
        result = np.diff(self.times)
        result.setflags(write=False)
        return result

    def dynamics_evaluation(
        self,
        interval_index: int,
        parameter_coordinates: Sequence[float],
    ) -> DynamicsResidualEvaluation:
        """Evaluate one interval with the production analytic dynamics factor."""

        index = int(interval_index)
        if index < 0 or index >= self.interval_count:
            raise IndexError("interval_index is outside the trajectory")
        return evaluate_raw_dynamics_residual(
            rotation_left=self.rotation[index],
            rotation_right=self.rotation[index + 1],
            linear_velocity_left=self.linear_velocity[index],
            linear_velocity_right=self.linear_velocity[index + 1],
            angular_velocity_left=self.angular_velocity[index],
            angular_velocity_right=self.angular_velocity[index + 1],
            actuator_thrust_left=self.actuator_thrust[index],
            actuator_thrust_right=self.actuator_thrust[index + 1],
            gimbal_angle_left=self.gimbal_angle[index],
            gimbal_angle_right=self.gimbal_angle[index + 1],
            time_step=self.time_step[index],
            parameter_chart=self.parameter_chart,
            parameter_coordinates=parameter_coordinates,
            geometry=self.geometry,
            gravity_world=_GRAVITY_WORLD,
        )

    def specific_acceleration_residual_and_jacobian(
        self,
        parameter_coordinates: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stack scale-invariant residuals and analytic 18-D Jacobians."""

        coordinates = np.asarray(parameter_coordinates, dtype=float)
        if coordinates.shape != (PARAMETER_DIMENSION,) or not np.all(
            np.isfinite(coordinates)
        ):
            raise ValueError("parameter_coordinates must contain 18 finite values")
        definition = DiagonalQDefinition(
            residual_quantity=SPECIFIC_ACCELERATION_QUANTITY,
            component_names=("x", "y", "z", "roll", "pitch", "yaw"),
            component_units=("m/s^2",) * 3 + ("rad/s^2",) * 3,
            interval_model=QIntervalModel.CONTINUOUS_SPECTRAL_DENSITY,
        )
        residuals = []
        jacobians = []
        for index in range(self.interval_count):
            raw = self.dynamics_evaluation(index, coordinates)
            statistical = specific_acceleration_statistical_residual(
                "synthetic-perfect",
                index,
                raw,
                definition,
                self.parameter_chart,
                coordinates,
            )
            residuals.append(statistical.residual)
            jacobians.append(statistical.jacobian.static_parameters)
        residual = np.concatenate(residuals)
        jacobian = np.vstack(jacobians)
        residual.setflags(write=False)
        jacobian.setflags(write=False)
        return residual, jacobian


@dataclass(frozen=True)
class SyntheticBatchTruthArtifact:
    """Strictly loaded solver-truth artifact and decoded physical parameters."""

    trajectory: PerfectModelBatchTrajectory
    truth_parameters: VehicleParameters
    generator_seed: int
    payload_sha256: str
    schema: str = SYNTHETIC_BATCH_TRUTH_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, PerfectModelBatchTrajectory):
            raise TypeError("trajectory must be PerfectModelBatchTrajectory")
        if not isinstance(self.truth_parameters, VehicleParameters):
            raise TypeError("truth_parameters must be VehicleParameters")
        if (
            isinstance(self.generator_seed, (bool, np.bool_))
            or not isinstance(self.generator_seed, Integral)
        ):
            raise TypeError("generator_seed must be an integer")
        if self.schema != SYNTHETIC_BATCH_TRUTH_SCHEMA:
            raise ValueError("synthetic batch truth schema is unsupported")
        digest = self.payload_sha256
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("payload_sha256 must be a lowercase SHA-256 digest")
        decoded = self.trajectory.parameter_chart.decode(
            self.trajectory.truth_parameter_coordinates
        )
        _assert_physical_parameters_equal(
            self.truth_parameters,
            decoded,
            "artifact physical parameters",
        )
        object.__setattr__(self, "generator_seed", int(self.generator_seed))

    @property
    def units(self) -> Mapping[str, str]:
        """Return the immutable unit contract recorded by schema v1."""

        return MappingProxyType(dict(_SYNTHETIC_BATCH_UNITS))

    @property
    def parameter_coordinate_order(self) -> Tuple[str, ...]:
        """Return the exact semantic order of the stored 18-D truth vector."""

        return _PARAMETER_COORDINATE_ORDER


def _truth_coordinates() -> np.ndarray:
    return np.asarray(
        (
            0.06,
            0.04,
            -0.03,
            0.02,
            0.015,
            -0.012,
            0.010,
            0.018,
            -0.012,
            0.009,
            0.05,
            -0.04,
            0.03,
            -0.02,
            0.04,
            -0.03,
            0.02,
            -0.01,
        ),
        dtype=float,
    )


def _newton_endpoint(
    *,
    chart: VehicleParameterChart,
    coordinates: np.ndarray,
    geometry: GrapeGeometry,
    rotation_left: np.ndarray,
    rotation_right: np.ndarray,
    velocity_left: np.ndarray,
    velocity_right: np.ndarray,
    omega_left: np.ndarray,
    omega_right: np.ndarray,
    thrust_left: np.ndarray,
    thrust_right: np.ndarray,
    gimbal_left: np.ndarray,
    gimbal_right: np.ndarray,
    time_step: float,
    rows: slice,
    field: str,
) -> np.ndarray:
    selected = np.asarray(
        omega_right if field == "angular_velocity_right" else velocity_right,
        dtype=float,
    ).copy()
    for _ in range(12):
        arguments = {
            "rotation_left": rotation_left,
            "rotation_right": rotation_right,
            "linear_velocity_left": velocity_left,
            "linear_velocity_right": velocity_right,
            "angular_velocity_left": omega_left,
            "angular_velocity_right": omega_right,
            "actuator_thrust_left": thrust_left,
            "actuator_thrust_right": thrust_right,
            "gimbal_angle_left": gimbal_left,
            "gimbal_angle_right": gimbal_right,
            "time_step": time_step,
            "parameter_chart": chart,
            "parameter_coordinates": coordinates,
            "geometry": geometry,
            "gravity_world": _GRAVITY_WORLD,
        }
        arguments[field] = selected
        evaluation = evaluate_raw_dynamics_residual(**arguments)
        residual = evaluation.residual[rows]
        if float(np.linalg.norm(residual, ord=np.inf)) <= 5.0e-13:
            return selected
        jacobian = getattr(evaluation.jacobian, field)[rows]
        selected -= np.linalg.solve(jacobian, residual)
    raise RuntimeError("analytic Newton construction did not converge")


def generate_perfect_model_batch_trajectory(
    interval_count: int = 36,
    seed: int = 917,
) -> PerfectModelBatchTrajectory:
    """Generate a noiseless, fully excited, variable-step batch trajectory."""

    if isinstance(interval_count, (bool, np.bool_)) or int(interval_count) < 18:
        raise ValueError("interval_count must be an integer at least 18")
    count = int(interval_count)
    random = np.random.RandomState(int(seed))
    chart = VehicleParameterChart(VehicleParameters.nominal())
    coordinates = _truth_coordinates()
    geometry = GrapeGeometry.grape()
    time_step = random.uniform(0.016, 0.032, size=count)
    times = np.concatenate((np.zeros(1), np.cumsum(time_step)))
    position = np.zeros((count + 1, 3), dtype=float)
    rotation = np.empty((count + 1, 3, 3), dtype=float)
    velocity = np.zeros((count + 1, 3), dtype=float)
    omega = np.zeros((count + 1, 3), dtype=float)
    thrust = np.empty((count + 1, 4), dtype=float)
    gimbal = np.empty((count + 1, 4), dtype=float)
    rotation[0] = so3_exp((0.08, -0.05, 0.12))
    velocity[0] = np.asarray((0.15, -0.08, 0.04))
    omega[0] = np.asarray((0.12, -0.09, 0.07))
    thrust[0] = np.asarray((5.9, 5.6, 5.8, 6.0))
    gimbal[0] = np.asarray((0.025, -0.018, 0.021, -0.016))

    rotor_phase = np.asarray((0.0, 0.7, 1.4, 2.1))
    gimbal_phase = np.asarray((0.2, 1.1, 2.0, 2.7))
    for index, dt in enumerate(time_step):
        phase = 0.53 * (index + 1)
        thrust[index + 1] = (
            5.8
            + 1.15 * np.sin(phase + rotor_phase)
            + 0.32 * random.normal(size=4)
        )
        gimbal[index + 1] = (
            0.11 * np.sin(0.71 * phase + gimbal_phase)
            + 0.025 * random.normal(size=4)
        )

        provisional_rotation = rotation[index]
        omega[index + 1] = _newton_endpoint(
            chart=chart,
            coordinates=coordinates,
            geometry=geometry,
            rotation_left=rotation[index],
            rotation_right=provisional_rotation,
            velocity_left=velocity[index],
            velocity_right=velocity[index],
            omega_left=omega[index],
            omega_right=omega[index],
            thrust_left=thrust[index],
            thrust_right=thrust[index + 1],
            gimbal_left=gimbal[index],
            gimbal_right=gimbal[index + 1],
            time_step=float(dt),
            rows=slice(3, 6),
            field="angular_velocity_right",
        )
        rotation[index + 1] = rotation[index] @ so3_exp(
            0.5 * float(dt) * (omega[index] + omega[index + 1])
        )
        velocity[index + 1] = _newton_endpoint(
            chart=chart,
            coordinates=coordinates,
            geometry=geometry,
            rotation_left=rotation[index],
            rotation_right=rotation[index + 1],
            velocity_left=velocity[index],
            velocity_right=velocity[index],
            omega_left=omega[index],
            omega_right=omega[index + 1],
            thrust_left=thrust[index],
            thrust_right=thrust[index + 1],
            gimbal_left=gimbal[index],
            gimbal_right=gimbal[index + 1],
            time_step=float(dt),
            rows=slice(0, 3),
            field="linear_velocity_right",
        )
        position[index + 1] = position[index] + 0.5 * float(dt) * (
            velocity[index] + velocity[index + 1]
        )

    return PerfectModelBatchTrajectory(
        times=times,
        position=position,
        rotation=rotation,
        linear_velocity=velocity,
        angular_velocity=omega,
        actuator_thrust=thrust,
        gimbal_angle=gimbal,
        parameter_chart=chart,
        truth_parameter_coordinates=coordinates,
        geometry=geometry,
    )


def _assert_physical_parameters_equal(
    first: VehicleParameters,
    second: VehicleParameters,
    name: str,
) -> None:
    if not np.isclose(first.mass, second.mass, rtol=2.0e-14, atol=0.0):
        raise ValueError("{} mass does not match the 18-D truth".format(name))
    for field in (
        "inertia",
        "cog_offset",
        "force_effectiveness",
        "torque_effectiveness",
        "linear_drag",
        "angular_drag",
    ):
        if not np.allclose(
            getattr(first, field),
            getattr(second, field),
            rtol=2.0e-14,
            atol=2.0e-15,
        ):
            raise ValueError(
                "{} {} does not match the 18-D truth".format(name, field)
            )


def _synthetic_batch_metadata(generator_seed: int, interval_count: int) -> dict:
    return {
        "schema": SYNTHETIC_BATCH_TRUTH_SCHEMA,
        "coordinate_order": list(_PARAMETER_COORDINATE_ORDER),
        "units": dict(_SYNTHETIC_BATCH_UNITS),
        "provenance": {
            "generator": _GENERATOR_PROVENANCE,
            "dynamics_factor": _DYNAMICS_FACTOR_PROVENANCE,
            "geometry": "grape_param_estim.system.GrapeGeometry.grape",
            "construction_derivatives": "analytic production factor Jacobian",
            "finite_difference_derivatives": False,
            "perfect_model": True,
            "variable_time_step": True,
            "generator_seed": int(generator_seed),
            "interval_count": int(interval_count),
        },
    }


def _canonical_metadata_json(generator_seed: int, interval_count: int) -> str:
    return json.dumps(
        _synthetic_batch_metadata(generator_seed, interval_count),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _payload_sha256(payload: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in _SYNTHETIC_BATCH_PAYLOAD_NAMES:
        value = np.ascontiguousarray(payload[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(item) for item in value.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _artifact_payload(
    trajectory: PerfectModelBatchTrajectory,
    generator_seed: int,
) -> dict:
    physical = trajectory.parameter_chart.decode(
        trajectory.truth_parameter_coordinates
    )
    payload = {
        "metadata_json": np.asarray(
            _canonical_metadata_json(generator_seed, trajectory.interval_count)
        ),
        "times": np.asarray(trajectory.times, dtype=np.float64),
        "position": np.asarray(trajectory.position, dtype=np.float64),
        "rotation": np.asarray(trajectory.rotation, dtype=np.float64),
        "linear_velocity": np.asarray(
            trajectory.linear_velocity, dtype=np.float64
        ),
        "angular_velocity": np.asarray(
            trajectory.angular_velocity, dtype=np.float64
        ),
        "actuator_thrust": np.asarray(
            trajectory.actuator_thrust, dtype=np.float64
        ),
        "gimbal_angle": np.asarray(trajectory.gimbal_angle, dtype=np.float64),
        "truth_parameter_coordinates": np.asarray(
            trajectory.truth_parameter_coordinates, dtype=np.float64
        ),
        "truth_mass": np.asarray((physical.mass,), dtype=np.float64),
        "truth_inertia": np.asarray(physical.inertia, dtype=np.float64),
        "truth_cog_offset": np.asarray(
            physical.cog_offset, dtype=np.float64
        ),
        "truth_force_effectiveness": np.asarray(
            physical.force_effectiveness, dtype=np.float64
        ),
        "truth_torque_effectiveness": np.asarray(
            physical.torque_effectiveness, dtype=np.float64
        ),
        "truth_linear_drag": np.asarray(
            physical.linear_drag, dtype=np.float64
        ),
        "truth_angular_drag": np.asarray(
            physical.angular_drag, dtype=np.float64
        ),
    }
    return payload


def save_synthetic_batch_truth_artifact(
    path: str,
    trajectory: PerfectModelBatchTrajectory,
    *,
    generator_seed: int,
) -> Path:
    """Save one canonical, pickle-free synthetic solver-truth NPZ artifact."""

    if not isinstance(trajectory, PerfectModelBatchTrajectory):
        raise TypeError("trajectory must be PerfectModelBatchTrajectory")
    if (
        isinstance(generator_seed, (bool, np.bool_))
        or not isinstance(generator_seed, Integral)
    ):
        raise TypeError("generator_seed must be an integer")
    destination = Path(path).expanduser().resolve()
    if destination.suffix != ".npz":
        raise ValueError("synthetic batch truth artifact must use a .npz suffix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _artifact_payload(trajectory, int(generator_seed))
    payload["payload_sha256"] = np.asarray(_payload_sha256(payload))
    np.savez_compressed(str(destination), **payload)
    return destination


def _strict_json_object(serialized: str) -> dict:
    def no_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("metadata_json contains duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(serialized, object_pairs_hook=no_duplicate_keys)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("metadata_json is not valid JSON") from error
    if type(value) is not dict:
        raise ValueError("metadata_json must contain one JSON object")
    return value


def _validate_metadata(metadata: dict) -> Tuple[int, int]:
    if set(metadata) != {"schema", "coordinate_order", "units", "provenance"}:
        raise ValueError("synthetic batch metadata fields do not match schema v1")
    if metadata["schema"] != SYNTHETIC_BATCH_TRUTH_SCHEMA:
        raise ValueError("synthetic batch truth schema is unsupported")
    if metadata["coordinate_order"] != list(_PARAMETER_COORDINATE_ORDER):
        raise ValueError("synthetic batch parameter coordinate order is invalid")
    if metadata["units"] != _SYNTHETIC_BATCH_UNITS:
        raise ValueError("synthetic batch unit contract is invalid")
    provenance = metadata["provenance"]
    expected_fields = {
        "generator",
        "dynamics_factor",
        "geometry",
        "construction_derivatives",
        "finite_difference_derivatives",
        "perfect_model",
        "variable_time_step",
        "generator_seed",
        "interval_count",
    }
    if type(provenance) is not dict or set(provenance) != expected_fields:
        raise ValueError("synthetic batch provenance is invalid")
    expected_fixed = {
        "generator": _GENERATOR_PROVENANCE,
        "dynamics_factor": _DYNAMICS_FACTOR_PROVENANCE,
        "geometry": "grape_param_estim.system.GrapeGeometry.grape",
        "construction_derivatives": "analytic production factor Jacobian",
        "finite_difference_derivatives": False,
        "perfect_model": True,
        "variable_time_step": True,
    }
    for name, expected in expected_fixed.items():
        if provenance[name] != expected or type(provenance[name]) is not type(expected):
            raise ValueError("synthetic batch provenance {} is invalid".format(name))
    seed = provenance["generator_seed"]
    interval_count = provenance["interval_count"]
    if type(seed) is not int:
        raise ValueError("synthetic batch generator_seed must be an integer")
    if type(interval_count) is not int or interval_count < 18:
        raise ValueError("synthetic batch interval_count must be at least 18")
    return seed, interval_count


def _require_float64_array(
    payload: Mapping[str, np.ndarray],
    name: str,
    shape: Tuple[int, ...],
) -> np.ndarray:
    value = payload[name]
    if value.dtype != np.dtype(np.float64) or value.shape != shape:
        raise ValueError(
            "{} must be a float64 array with shape {}".format(name, shape)
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("{} must contain only finite values".format(name))
    return value


def load_synthetic_batch_truth_artifact(
    path: str,
) -> SyntheticBatchTruthArtifact:
    """Strictly load and integrity-check a schema-v1 synthetic truth NPZ."""

    source = Path(path).expanduser().resolve()
    with np.load(str(source), allow_pickle=False) as archive:
        names = tuple(archive.files)
        if (
            len(names) != len(_SYNTHETIC_BATCH_ARCHIVE_NAMES)
            or set(names) != _SYNTHETIC_BATCH_ARCHIVE_NAMES
        ):
            raise ValueError(
                "synthetic batch truth archive members do not match schema v1"
            )
        payload = {
            name: np.asarray(archive[name]).copy()
            for name in _SYNTHETIC_BATCH_PAYLOAD_NAMES
        }
        digest_array = np.asarray(archive["payload_sha256"]).copy()

    for name, value in (
        ("metadata_json", payload["metadata_json"]),
        ("payload_sha256", digest_array),
    ):
        if value.shape != () or value.dtype.kind not in "US":
            raise ValueError("{} must be one scalar string".format(name))
    recorded_digest = str(digest_array.item())
    computed_digest = _payload_sha256(payload)
    if recorded_digest != computed_digest:
        raise ValueError("synthetic batch truth payload checksum does not match")

    metadata = _strict_json_object(str(payload["metadata_json"].item()))
    generator_seed, interval_count = _validate_metadata(metadata)
    sample_count = interval_count + 1
    times = _require_float64_array(payload, "times", (sample_count,))
    position = _require_float64_array(payload, "position", (sample_count, 3))
    rotation = _require_float64_array(
        payload, "rotation", (sample_count, 3, 3)
    )
    linear_velocity = _require_float64_array(
        payload, "linear_velocity", (sample_count, 3)
    )
    angular_velocity = _require_float64_array(
        payload, "angular_velocity", (sample_count, 3)
    )
    actuator_thrust = _require_float64_array(
        payload, "actuator_thrust", (sample_count, 4)
    )
    gimbal_angle = _require_float64_array(
        payload, "gimbal_angle", (sample_count, 4)
    )
    coordinates = _require_float64_array(
        payload,
        "truth_parameter_coordinates",
        (PARAMETER_DIMENSION,),
    )
    truth_mass = _require_float64_array(payload, "truth_mass", (1,))
    truth_inertia = _require_float64_array(payload, "truth_inertia", (3, 3))
    truth_cog = _require_float64_array(payload, "truth_cog_offset", (3,))
    truth_force = _require_float64_array(
        payload, "truth_force_effectiveness", (4,)
    )
    truth_torque = _require_float64_array(
        payload, "truth_torque_effectiveness", (4,)
    )
    truth_linear_drag = _require_float64_array(
        payload, "truth_linear_drag", (3,)
    )
    truth_angular_drag = _require_float64_array(
        payload, "truth_angular_drag", (3,)
    )

    chart = VehicleParameterChart(VehicleParameters.nominal())
    trajectory = PerfectModelBatchTrajectory(
        times=times,
        position=position,
        rotation=rotation,
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        actuator_thrust=actuator_thrust,
        gimbal_angle=gimbal_angle,
        parameter_chart=chart,
        truth_parameter_coordinates=coordinates,
        geometry=GrapeGeometry.grape(),
    )
    physical = VehicleParameters(
        mass=float(truth_mass[0]),
        inertia=truth_inertia,
        cog_offset=truth_cog,
        force_effectiveness=truth_force,
        torque_effectiveness=truth_torque,
        linear_drag=truth_linear_drag,
        angular_drag=truth_angular_drag,
    )
    return SyntheticBatchTruthArtifact(
        trajectory=trajectory,
        truth_parameters=physical,
        generator_seed=generator_seed,
        payload_sha256=recorded_digest,
    )


@dataclass(frozen=True)
class KnownQBatchMoments:
    """Synthetic normal-normal E-step moments with known diagonal Q."""

    definition: DiagonalQDefinition
    true_q: np.ndarray
    observation_noise_diagonal: np.ndarray
    time_step: np.ndarray
    bag_index: np.ndarray
    latent_residual: np.ndarray
    pseudo_observation: np.ndarray
    moments: ExpectedResidualMoments

    def __post_init__(self) -> None:
        if not isinstance(self.definition, DiagonalQDefinition):
            raise TypeError("definition must be DiagonalQDefinition")
        object.__setattr__(self, "true_q", _positive_vector(self.true_q, 6, "true_q"))
        object.__setattr__(
            self,
            "observation_noise_diagonal",
            _positive_vector(
                self.observation_noise_diagonal,
                6,
                "observation_noise_diagonal",
            ),
        )
        time_step = np.asarray(self.time_step, dtype=float)
        bag_index = np.asarray(self.bag_index)
        if (
            time_step.ndim != 1
            or time_step.size == 0
            or not np.all(np.isfinite(time_step))
            or np.any(time_step <= 0.0)
        ):
            raise ValueError("time_step must contain positive finite values")
        if bag_index.shape != time_step.shape or bag_index.dtype.kind not in "iu":
            raise ValueError("bag_index must contain one integer per interval")
        if not isinstance(self.moments, ExpectedResidualMoments):
            raise TypeError("moments must be ExpectedResidualMoments")
        if self.moments.interval_count != time_step.size:
            raise ValueError("moment count must match time_step")
        for name in ("latent_residual", "pseudo_observation"):
            object.__setattr__(
                self,
                name,
                _immutable_array(
                    getattr(self, name),
                    (time_step.size, 6),
                    name,
                ),
            )
        time_step = time_step.copy()
        bag_index = bag_index.copy()
        time_step.setflags(write=False)
        bag_index.setflags(write=False)
        object.__setattr__(self, "time_step", time_step)
        object.__setattr__(self, "bag_index", bag_index)


def generate_known_q_laplace_moments(
    true_q: Sequence[float],
    bag_time_steps: Sequence[Sequence[float]],
    *,
    interval_model: QIntervalModel = QIntervalModel.CONTINUOUS_SPECTRAL_DENSITY,
    observation_noise_ratio: float = 0.8,
    seed: int = 2718,
) -> KnownQBatchMoments:
    """Draw posterior E-step moments whose expected M-step target is ``true_q``.

    The latent discrepancy has interval covariance ``Q / dt`` for spectral
    density Q (or ``Q`` for fixed-interval Q).  A Gaussian noisy pseudo-
    observation is conditioned analytically.  Consequently the returned MAP
    residual is shrunken, while its Laplace covariance correction restores the
    known expected second moment.
    """

    q = _positive_vector(true_q, 6, "true_q")
    ratio = float(observation_noise_ratio)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("observation_noise_ratio must be finite and positive")
    if not isinstance(interval_model, QIntervalModel):
        raise TypeError("interval_model must be QIntervalModel")
    selected = tuple(np.asarray(value, dtype=float) for value in bag_time_steps)
    if not selected:
        raise ValueError("bag_time_steps cannot be empty")
    for value in selected:
        if (
            value.ndim != 1
            or value.size == 0
            or not np.all(np.isfinite(value))
            or np.any(value <= 0.0)
        ):
            raise ValueError("each bag must contain positive finite time steps")
    time_step = np.concatenate(selected)
    bag_index = np.concatenate(
        tuple(
            np.full(value.size, index, dtype=np.int64)
            for index, value in enumerate(selected)
        )
    )
    definition = DiagonalQDefinition(
        residual_quantity=SPECIFIC_ACCELERATION_QUANTITY,
        component_names=("x", "y", "z", "roll", "pitch", "yaw"),
        component_units=("m/s^2",) * 3 + ("rad/s^2",) * 3,
        interval_model=interval_model,
    )
    weights = definition.interval_weights(time_step)
    noise = ratio * q
    shrinkage = q / (q + noise)
    generator = np.random.RandomState(int(seed))
    latent_residual = generator.normal(size=(time_step.size, 6)) * np.sqrt(
        q[None, :] / weights[:, None]
    )
    observation_error = generator.normal(size=(time_step.size, 6)) * np.sqrt(
        noise[None, :] / weights[:, None]
    )
    pseudo_observation = latent_residual + observation_error
    map_residual = shrinkage[None, :] * pseudo_observation
    covariance_correction = np.broadcast_to(
        (q * noise / (q + noise))[None, :] / weights[:, None],
        map_residual.shape,
    ).copy()
    return KnownQBatchMoments(
        definition=definition,
        true_q=q,
        observation_noise_diagonal=noise,
        time_step=time_step,
        bag_index=bag_index,
        latent_residual=latent_residual,
        pseudo_observation=pseudo_observation,
        moments=ExpectedResidualMoments(
            map_residual=map_residual,
            covariance_correction=covariance_correction,
        ),
    )


def simulate_delayed_zoh_first_order(
    observation_times: Sequence[float],
    command_event_times: Sequence[float],
    command_values: np.ndarray,
    *,
    delay: float,
    time_constant: float,
    initial_state: Sequence[float] = None,
) -> np.ndarray:
    """Exactly integrate a delayed vector ZOH command through a first-order lag.

    The first command is the audited prehistory value and is active at the
    first observation.  Every later publish event takes effect at
    ``event_time + delay``.  Switches are integrated at their exact times and
    are never rounded to the observation or latent-knot grids.
    """

    observations = np.asarray(observation_times, dtype=float)
    events = np.asarray(command_event_times, dtype=float)
    commands = np.asarray(command_values, dtype=float)
    if (
        observations.ndim != 1
        or observations.size == 0
        or not np.all(np.isfinite(observations))
        or np.any(np.diff(observations) <= 0.0)
    ):
        raise ValueError("observation_times must be strictly increasing")
    if (
        events.ndim != 1
        or events.size == 0
        or not np.all(np.isfinite(events))
        or np.any(np.diff(events) <= 0.0)
        or events[0] > observations[0]
    ):
        raise ValueError("command_event_times must start before observations")
    if commands.ndim != 2 or commands.shape[0] != events.size or not np.all(
        np.isfinite(commands)
    ):
        raise ValueError("command_values must be a finite event-by-channel matrix")
    selected_delay = float(delay)
    tau = float(time_constant)
    if not np.isfinite(selected_delay) or selected_delay < 0.0:
        raise ValueError("delay must be finite and non-negative")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("time_constant must be finite and positive")
    if initial_state is None:
        state = commands[0].copy()
    else:
        state = np.asarray(initial_state, dtype=float)
        if state.shape != (commands.shape[1],) or not np.all(np.isfinite(state)):
            raise ValueError("initial_state must contain one finite value per channel")
        state = state.copy()

    current_time = float(observations[0])
    current_command = commands[0].copy()
    effective_times = events[1:] + selected_delay
    event_index = 0
    result = np.empty((observations.size, commands.shape[1]), dtype=float)

    def propagate(until: float) -> None:
        nonlocal current_time, state
        duration = float(until - current_time)
        if duration < -1.0e-12:
            raise ValueError("effective command events precede integration start")
        if duration > 0.0:
            decay = np.exp(-duration / tau)
            state = current_command + decay * (state - current_command)
            current_time = float(until)

    for observation_index, observation_time in enumerate(observations):
        while (
            event_index < effective_times.size
            and effective_times[event_index] <= observation_time
        ):
            propagate(float(effective_times[event_index]))
            current_command = commands[event_index + 1].copy()
            event_index += 1
        propagate(float(observation_time))
        result[observation_index] = state
    result.setflags(write=False)
    return result


__all__ = [
    "KnownQBatchMoments",
    "PerfectModelBatchTrajectory",
    "SYNTHETIC_BATCH_TRUTH_SCHEMA",
    "SYNTHETIC_BATCH_TRUTH_SUMMARY_SCHEMA",
    "SyntheticBatchTruthArtifact",
    "generate_known_q_laplace_moments",
    "generate_perfect_model_batch_trajectory",
    "load_synthetic_batch_truth_artifact",
    "save_synthetic_batch_truth_artifact",
    "simulate_delayed_zoh_first_order",
]
