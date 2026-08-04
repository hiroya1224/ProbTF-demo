"""Export converged sparse-batch solver results to strict artifact payloads.

This module is the scientific serialization boundary.  It accepts only the
validated request, inspected flight inputs, audited initializations, and
final solver products.  It does not rerun estimation, infer benchmark times,
or create a run directory; :func:`write_batch_estimation_run` owns the latter
atomic publication step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.batch.dynamics_moments import (
    DynamicsIntervalLinearization,
)
from grape_param_estim.batch.evidence import StaticLaplaceGeometry
from grape_param_estim.batch.em_loop import LaplaceEmResult
from grape_param_estim.batch.factors.dynamics_factor import (
    dynamics_square_root_information,
)
from grape_param_estim.batch.graph_builder import (
    GaussianCovariance,
    PreparedBagGraphData,
)
from grape_param_estim.batch.lag_profile import LagProfileResult
from grape_param_estim.batch.state import BatchState
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.batch_request import BatchEstimationRequest
from grape_param_estim.estimation import FixedGraphLaplaceSolution
from grape_param_estim.geometry import (
    correction_transform_path,
    matrix_to_quaternion,
)
from grape_param_estim.initialization import FlightInitialization
from grape_param_estim.parameterization import PARAMETER_DIMENSION
from grape_param_estim.posterior.diagnostics import McmcDiagnostics
from grape_param_estim.posterior.mcmc import McmcChainResult
from grape_param_estim.sensor_models import FlightData


_STATE_FIELDS = (
    ("position", VariableKind.POSITION, 3),
    ("orientation_xyzw", VariableKind.ORIENTATION_TANGENT, 4),
    ("linear_velocity", VariableKind.LINEAR_VELOCITY, 3),
    ("angular_velocity", VariableKind.ANGULAR_VELOCITY, 3),
    ("controller_integral", VariableKind.CONTROLLER_INTEGRAL, 6),
    ("actuator_thrust", VariableKind.ACTUATOR_THRUST, 4),
    ("actuator_gimbal", VariableKind.GIMBAL_ANGLE, 4),
)


def _canonical_string(value: object, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError("{} must be a canonical non-empty string".format(name))
    return value


def _sha256(value: object, name: str) -> str:
    selected = _canonical_string(value, name)
    if (
        len(selected) != 71
        or not selected.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in selected[7:])
    ):
        raise ValueError("{} must have form sha256:<64 lowercase hex>".format(name))
    return selected


def _normalized_flight_sha256(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("FlightData bag SHA-256 must be a string")
    digest = value[7:] if value.startswith("sha256:") else value
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(
            "FlightData bag SHA-256 must be raw lowercase hex or "
            "sha256:<lowercase hex>"
        )
    return "sha256:" + digest


def _nonnegative_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("{} must be a real scalar".format(name))
    selected = float(value)
    if not np.isfinite(selected) or selected < 0.0:
        raise ValueError("{} must be finite and non-negative".format(name))
    return selected


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value <= 0
    ):
        raise ValueError("{} must be a positive integer".format(name))
    return int(value)


def _timings(value: object, name: str, *, allow_empty: bool) -> Tuple[float, ...]:
    try:
        selected = tuple(
            _nonnegative_real(item, "{} member".format(name)) for item in value
        )
    except TypeError as error:
        raise TypeError("{} must be an iterable of timings".format(name)) from error
    if not allow_empty and not selected:
        raise ValueError("{} must not be empty".format(name))
    return selected


@dataclass(frozen=True)
class ArtifactRunIdentity:
    """Non-inferable implementation and project identity for one run."""

    estimator_revision: str
    configuration_fingerprint: str
    controller_snapshot_fingerprint: str
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "estimator_revision",
            _canonical_string(self.estimator_revision, "estimator_revision"),
        )
        for name in (
            "configuration_fingerprint",
            "controller_snapshot_fingerprint",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if type(self.warnings) is not tuple:
            raise TypeError("warnings must be a tuple")
        object.__setattr__(
            self,
            "warnings",
            tuple(
                _canonical_string(value, "warning") for value in self.warnings
            ),
        )


@dataclass(frozen=True)
class BagPerformanceMeasurements:
    """Measured sparse-system costs and sizes for one concrete bag."""

    bag_id: str
    knot_count: int
    factor_count: int
    residual_dimension: int
    jacobian_nnz: int
    assembly_seconds: float
    factorization_seconds: float
    schur_solve_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "bag_id", _canonical_string(self.bag_id, "bag_id"))
        for name in (
            "knot_count",
            "factor_count",
            "residual_dimension",
            "jacobian_nnz",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        for name in (
            "assembly_seconds",
            "factorization_seconds",
            "schur_solve_seconds",
        ):
            object.__setattr__(
                self, name, _nonnegative_real(getattr(self, name), name)
            )


@dataclass(frozen=True)
class RunPerformanceMeasurements:
    """Measured run costs that cannot be reconstructed from solver results."""

    bags: Tuple[BagPerformanceMeasurements, ...]
    nonlinear_iteration_seconds: Tuple[float, ...]
    em_iteration_seconds: Tuple[float, ...]
    mcmc_target_seconds: Tuple[float, ...]
    peak_memory_bytes: int

    def __post_init__(self) -> None:
        if (
            type(self.bags) is not tuple
            or not self.bags
            or any(
                not isinstance(value, BagPerformanceMeasurements)
                for value in self.bags
            )
        ):
            raise TypeError("bags must be a non-empty performance tuple")
        if len({value.bag_id for value in self.bags}) != len(self.bags):
            raise ValueError("bag performance IDs must be unique")
        object.__setattr__(
            self,
            "nonlinear_iteration_seconds",
            _timings(
                self.nonlinear_iteration_seconds,
                "nonlinear_iteration_seconds",
                allow_empty=False,
            ),
        )
        object.__setattr__(
            self,
            "em_iteration_seconds",
            _timings(
                self.em_iteration_seconds,
                "em_iteration_seconds",
                allow_empty=False,
            ),
        )
        object.__setattr__(
            self,
            "mcmc_target_seconds",
            _timings(
                self.mcmc_target_seconds,
                "mcmc_target_seconds",
                allow_empty=True,
            ),
        )
        if (
            isinstance(self.peak_memory_bytes, (bool, np.bool_))
            or not isinstance(self.peak_memory_bytes, Integral)
            or self.peak_memory_bytes < 0
        ):
            raise ValueError("peak_memory_bytes must be a non-negative integer")
        object.__setattr__(self, "peak_memory_bytes", int(self.peak_memory_bytes))


@dataclass(frozen=True)
class DelayLocalGeometry:
    """Explicit local delay uncertainty from the caller's profile analysis."""

    standard_deviation_seconds: float
    source: str
    curvature: Optional[float] = None

    def __post_init__(self) -> None:
        selected = _nonnegative_real(
            self.standard_deviation_seconds, "standard_deviation_seconds"
        )
        if selected == 0.0:
            raise ValueError("standard_deviation_seconds must be positive")
        object.__setattr__(self, "standard_deviation_seconds", selected)
        object.__setattr__(self, "source", _canonical_string(self.source, "source"))
        curvature = self.curvature
        if curvature is not None:
            curvature = _nonnegative_real(curvature, "curvature")
            if curvature == 0.0:
                raise ValueError("curvature must be positive when present")
        object.__setattr__(self, "curvature", curvature)


@dataclass(frozen=True)
class SelectedConditionalTrajectory:
    """One retained MCMC sample's conditional trajectory for one bag."""

    sample_id: str
    bag_id: str
    state: BatchState
    dynamics_residual: np.ndarray
    dynamics_residual_valid: np.ndarray
    conditional_objective: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sample_id", _canonical_string(self.sample_id, "sample_id")
        )
        object.__setattr__(self, "bag_id", _canonical_string(self.bag_id, "bag_id"))
        if not isinstance(self.state, BatchState):
            raise TypeError("state must be a BatchState")
        residual = np.asarray(self.dynamics_residual, dtype=float)
        valid = np.asarray(self.dynamics_residual_valid)
        if (
            residual.ndim != 2
            or residual.shape[1] != 6
            or not np.all(np.isfinite(residual))
        ):
            raise ValueError("dynamics_residual must be a finite (n, 6) array")
        if valid.shape != (residual.shape[0],) or valid.dtype != np.bool_:
            raise ValueError("dynamics_residual_valid must align and be boolean")
        residual = residual.copy()
        valid = valid.copy()
        residual.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "dynamics_residual", residual)
        object.__setattr__(self, "dynamics_residual_valid", valid)
        object.__setattr__(
            self,
            "conditional_objective",
            _nonnegative_real(self.conditional_objective, "conditional_objective"),
        )


@dataclass(frozen=True)
class BatchArtifactPayload:
    """All exact arguments accepted by the strict atomic run writer."""

    manifest_metadata: Mapping[str, Any]
    map_static: Mapping[str, np.ndarray]
    q_em: Mapping[str, np.ndarray]
    laplace: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, np.ndarray]
    bags: Mapping[str, Mapping[str, np.ndarray]]
    mcmc_samples: Optional[Mapping[str, np.ndarray]]
    trajectories: Mapping[str, Mapping[str, np.ndarray]]

    @property
    def writer_arguments(self) -> Dict[str, Any]:
        """Return keyword arguments for ``write_batch_estimation_run``."""

        return {
            "manifest_metadata": self.manifest_metadata,
            "map_static": self.map_static,
            "q_em": self.q_em,
            "laplace": self.laplace,
            "diagnostics": self.diagnostics,
            "bags": self.bags,
            "mcmc_samples": self.mcmc_samples,
            "trajectories": self.trajectories,
        }


def _plain(value: Any) -> Any:
    """Detach frozen request values into JSON-serializable plain objects."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _request_bags(request: BatchEstimationRequest) -> Mapping[str, Mapping[str, Any]]:
    return {
        str(value["bag_id"]): value for value in request.payload["bags"]
    }


def _by_bag_id(values: Sequence[Any], name: str) -> Mapping[str, Any]:
    result = {}
    for value in values:
        bag_id = getattr(value, "bag_id", None)
        if not isinstance(bag_id, str):
            raise TypeError("{} members must expose bag_id".format(name))
        if bag_id in result:
            raise ValueError("{} contains duplicate bag_id {!r}".format(name, bag_id))
        result[bag_id] = value
    return result


def _state_arrays(state: BatchState, bag_id: str, knot_count: int) -> Dict[str, np.ndarray]:
    result = {}
    for output_name, kind, width in _STATE_FIELDS:
        rows = []
        for index in range(knot_count):
            value = state.knot_value(bag_id, index, kind)
            if kind is VariableKind.ORIENTATION_TANGENT:
                value = matrix_to_quaternion(value)
            rows.append(value)
        array = np.asarray(rows, dtype=float)
        if array.shape != (knot_count, width):
            raise ValueError(
                "{} {} state has the wrong shape".format(bag_id, output_name)
            )
        result[output_name] = array
    return result


@dataclass(frozen=True)
class _FactorAudit:
    category: str
    group: str
    bag_id: Optional[str]
    instance: int
    whitening: np.ndarray
    dynamics: Optional[DynamicsIntervalLinearization] = None


def _whitening(covariance: GaussianCovariance) -> np.ndarray:
    if not isinstance(covariance, GaussianCovariance):
        raise TypeError("factor audit covariance is unavailable")
    return covariance.square_root_information


def _actuator_whitening(bag: PreparedBagGraphData) -> np.ndarray:
    value = np.zeros((8, 8), dtype=float)
    value[:4, :4] = _whitening(bag.covariances.actuator_thrust_transition)
    value[4:, 4:] = _whitening(bag.covariances.actuator_gimbal_transition)
    return value


def _factor_audits(solution: FixedGraphLaplaceSolution) -> Tuple[_FactorAudit, ...]:
    """Mirror the graph builder's documented deterministic factor order."""

    prepared = solution.prepared
    bags = tuple(sorted(prepared.bags, key=lambda value: value.bag_id))
    audits = []

    def add(
        category: str,
        group: str,
        bag_id: Optional[str],
        whitening: np.ndarray,
        instance: int = 0,
        dynamics: Optional[DynamicsIntervalLinearization] = None,
    ) -> None:
        audits.append(
            _FactorAudit(
                category,
                group,
                bag_id,
                instance,
                np.asarray(whitening, dtype=float),
                dynamics,
            )
        )

    add(
        "prior/static_parameters",
        "prior",
        None,
        _whitening(prepared.static_parameter_prior.covariance),
    )
    initial_priors = (
        ("position", "position"),
        ("orientation", "rotation"),
        ("linear_velocity", "linear_velocity"),
        ("angular_velocity", "angular_velocity"),
        ("controller_integral", "controller_integral"),
        ("actuator_thrust", "actuator_thrust"),
        ("actuator_gimbal", "gimbal_angle"),
    )
    for bag in bags:
        add(
            "prior/gyro_bias",
            "prior",
            bag.bag_id,
            _whitening(bag.priors.gyro_bias.covariance),
        )
        if bag.accelerometer.enabled:
            add(
                "prior/accelerometer_bias",
                "prior",
                bag.bag_id,
                _whitening(bag.priors.accelerometer_bias.covariance),
            )
        for category, field in initial_priors:
            prior = getattr(bag.priors.initial_knot, field)
            add(
                "prior/initial_{}".format(category),
                "prior",
                bag.bag_id,
                _whitening(prior.covariance),
            )

    for bag in bags:
        for index, _measurement in enumerate(bag.pose_measurements):
            add(
                "observation/pose_position",
                "likelihood",
                bag.bag_id,
                _whitening(bag.covariances.position_observation),
                index,
            )
            add(
                "observation/pose_orientation",
                "likelihood",
                bag.bag_id,
                _whitening(bag.covariances.orientation_observation),
                index,
            )
    for bag in bags:
        for index, _measurement in enumerate(bag.velocity_measurements):
            add(
                "observation/velocity",
                "likelihood",
                bag.bag_id,
                _whitening(bag.covariances.velocity_observation),
                index,
            )
    for bag in bags:
        for index, _measurement in enumerate(bag.gyro_measurements):
            add(
                "observation/gyro",
                "likelihood",
                bag.bag_id,
                _whitening(bag.covariances.gyro_observation),
                index,
            )
    for bag in bags:
        for index, _measurement in enumerate(bag.accelerometer_measurements):
            add(
                "observation/accelerometer",
                "likelihood",
                bag.bag_id,
                _whitening(bag.covariances.accelerometer_observation),
                index,
            )
    for bag in bags:
        for index, _measurement in enumerate(
            bag.controller_integral_measurements
        ):
            add(
                "observation/controller_integral",
                "likelihood",
                bag.bag_id,
                _whitening(bag.covariances.controller_integral_observation),
                index,
            )
    for bag in bags:
        for interval in bag.controller_intervals:
            index = interval.left_knot_index
            add(
                "model/controller_integral_transition",
                "likelihood",
                bag.bag_id,
                _whitening(bag.covariances.controller_integral_transition),
                index,
            )
            if interval.issued_thrust_observation is not None:
                add(
                    "observation/issued_thrust",
                    "likelihood",
                    bag.bag_id,
                    _whitening(bag.covariances.issued_thrust_observation),
                    index,
                )
            if interval.issued_gimbal_observation is not None:
                add(
                    "observation/issued_gimbal",
                    "likelihood",
                    bag.bag_id,
                    _whitening(bag.covariances.issued_gimbal_observation),
                    index,
                )
    for bag in bags:
        for interval in bag.actuator_intervals:
            add(
                "model/actuator_transition",
                "likelihood",
                bag.bag_id,
                _actuator_whitening(bag),
                interval.left_knot_index,
            )
        for index, _measurement in enumerate(bag.actual_gimbal_measurements):
            add(
                "observation/actual_gimbal",
                "likelihood",
                bag.bag_id,
                _whitening(bag.covariances.actual_gimbal_observation),
                index,
            )
    for bag in bags:
        for index in range(len(bag.knots) - 1):
            add(
                "model/position_kinematic",
                "likelihood",
                bag.bag_id,
                _whitening(bag.covariances.position_kinematic),
                index,
            )
            add(
                "model/orientation_kinematic",
                "likelihood",
                bag.bag_id,
                _whitening(bag.covariances.orientation_kinematic),
                index,
            )
    for interval in solution.dynamics.linearizations.intervals:
        add(
            "dynamics/dynamics_residual",
            "likelihood",
            interval.bag_id,
            dynamics_square_root_information(
                prepared.dynamics.q,
                interval.time_step,
                prepared.dynamics.q_definition,
            ),
            interval.left_knot_index,
            interval,
        )
    if len(audits) != len(solution.final_linearization.factors):
        raise ValueError(
            "factor audit count {} disagrees with final graph {}".format(
                len(audits), len(solution.final_linearization.factors)
            )
        )
    for audit, factor in zip(audits, solution.final_linearization.factors):
        if audit.whitening.shape != (factor.residual.size, factor.residual.size):
            raise ValueError(
                "factor audit dimension disagrees for {}".format(audit.category)
            )
    return tuple(audits)


def _factor_payload(
    solution: FixedGraphLaplaceSolution,
) -> Tuple[
    Tuple[_FactorAudit, ...],
    Mapping[str, float],
    Mapping[str, Mapping[str, np.ndarray]],
]:
    audits = _factor_audits(solution)
    components: Dict[str, float] = {}
    per_bag: Dict[str, Dict[str, Any]] = {
        value.bag_id: {
            "names": [],
            "raw": [],
            "normalized": [],
            "components": {},
        }
        for value in solution.prepared.bags
    }
    total = 0.0
    for audit, factor in zip(audits, solution.final_linearization.factors):
        normalized = np.asarray(factor.residual, dtype=float)
        raw = (
            audit.dynamics.residual
            if audit.dynamics is not None
            else np.linalg.solve(audit.whitening, normalized)
        )
        if not np.allclose(
            audit.whitening @ raw,
            normalized,
            rtol=2.0e-10,
            atol=2.0e-12,
        ):
            raise ValueError(
                "raw residual reconstruction failed for {}".format(
                    audit.category
                )
            )
        objective = 0.5 * float(normalized @ normalized)
        components[audit.category] = components.get(audit.category, 0.0) + objective
        total += objective
        if audit.bag_id is None:
            continue
        bag = per_bag[audit.bag_id]
        bag_components = bag["components"]
        bag_components[audit.category] = (
            bag_components.get(audit.category, 0.0) + objective
        )
        for coordinate in range(normalized.size):
            bag["names"].append(
                "{}/{:06d}/{:02d}".format(
                    audit.category, audit.instance, coordinate
                )
            )
            bag["raw"].append(float(raw[coordinate]))
            bag["normalized"].append(float(normalized[coordinate]))
    if not np.isclose(
        total, solution.lm.objective, rtol=2.0e-10, atol=2.0e-11
    ):
        raise ValueError("factor objectives do not reconstruct the MAP objective")
    result = {}
    for bag_id, value in per_bag.items():
        component_names = tuple(sorted(value["components"]))
        result[bag_id] = {
            "factor_names": np.asarray(value["names"]),
            "factor_residual_history": np.asarray(
                (value["raw"],), dtype=float
            ),
            "factor_normalized_residual_history": np.asarray(
                (value["normalized"],), dtype=float
            ),
            "objective_component_names": np.asarray(component_names),
            "objective_component_values": np.asarray(
                tuple(value["components"][name] for name in component_names),
                dtype=float,
            ),
        }
    return audits, components, result


def _validate_solver_alignment(
    final_solution: FixedGraphLaplaceSolution,
    em_result: LaplaceEmResult,
    static_geometry: StaticLaplaceGeometry,
) -> None:
    if not isinstance(final_solution, FixedGraphLaplaceSolution):
        raise TypeError("final_solution must be FixedGraphLaplaceSolution")
    if not isinstance(em_result, LaplaceEmResult):
        raise TypeError("em_result must be LaplaceEmResult")
    if not isinstance(static_geometry, StaticLaplaceGeometry):
        raise TypeError("static_geometry must be StaticLaplaceGeometry")
    final_step = em_result.final_step
    if not np.allclose(
        final_solution.prepared.dynamics.q,
        final_step.q,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("final solution Q disagrees with Laplace-EM")
    if not np.isclose(
        final_solution.prepared.fixed_delay,
        final_step.lag,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("final solution delay disagrees with Laplace-EM")
    if not np.isclose(
        final_solution.lm.objective,
        final_step.map_objective,
        rtol=2.0e-10,
        atol=2.0e-10,
    ):
        raise ValueError("final MAP objective disagrees with Laplace-EM")
    if not np.isclose(
        final_solution.marginal_objective.value,
        final_step.approximate_marginal_objective,
        rtol=2.0e-10,
        atol=2.0e-10,
    ):
        raise ValueError("final marginal objective disagrees with Laplace-EM")
    if final_solution.lm.state.layout != final_step.state.layout:
        raise ValueError("final solution state layout disagrees with Laplace-EM")
    for key in final_solution.lm.state.layout.variable_keys:
        if not np.allclose(
            final_solution.lm.state.value(key),
            final_step.state.value(key),
            rtol=1.0e-11,
            atol=1.0e-12,
        ):
            raise ValueError("final solution state disagrees with Laplace-EM")
    if not np.allclose(
        static_geometry.information.posterior.hessian,
        final_solution.factorization.reduced_hessian,
        rtol=2.0e-10,
        atol=2.0e-11,
    ):
        raise ValueError("static geometry does not describe the final Hessian")


def _validate_final_q_lag_profile(
    solution: FixedGraphLaplaceSolution,
    profile: Optional[LagProfileResult],
    delay_geometry: DelayLocalGeometry,
) -> None:
    if profile is None:
        if delay_geometry.curvature is not None:
            raise ValueError(
                "delay curvature cannot be supplied without a final-Q profile"
            )
        return
    if not isinstance(profile, LagProfileResult):
        raise TypeError("final_q_lag_profile must be LagProfileResult or None")
    if not np.isclose(
        profile.best_lag,
        solution.prepared.fixed_delay,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("final-Q profile best lag disagrees with final solution")
    if not np.isclose(
        profile.best_objective,
        solution.marginal_objective.value,
        rtol=2.0e-10,
        atol=2.0e-10,
    ):
        raise ValueError(
            "final-Q profile objective disagrees with final marginal objective"
        )
    if profile.best_state is None:
        raise ValueError("available final-Q profile must retain its best state")
    if profile.best_state.layout != solution.lm.state.layout:
        raise ValueError("final-Q profile state layout disagrees with final solution")
    for key in solution.lm.state.layout.variable_keys:
        if not np.allclose(
            profile.best_state.value(key),
            solution.lm.state.value(key),
            rtol=1.0e-11,
            atol=1.0e-12,
        ):
            raise ValueError(
                "final-Q profile state disagrees with final solution"
            )


def _physical_parameter_arrays(
    coordinates: np.ndarray, chart: object
) -> Mapping[str, np.ndarray]:
    value = np.asarray(coordinates, dtype=float)
    scalar = value.ndim == 1
    matrix = value.reshape((1, PARAMETER_DIMENSION)) if scalar else value
    if matrix.ndim != 2 or matrix.shape[1] != PARAMETER_DIMENSION:
        raise ValueError("parameter coordinates must have 18 columns")
    mass = []
    inertia = []
    cog = []
    force = []
    torque = []
    for coordinate in matrix:
        parameters = chart.decode(coordinate)
        mass.append(parameters.mass)
        inertia.append(parameters.inertia)
        cog.append(parameters.cog_offset)
        force.append(parameters.force_effectiveness)
        torque.append(parameters.torque_effectiveness)
    result = {
        "mass": np.asarray(mass, dtype=float),
        "inertia": np.asarray(inertia, dtype=float),
        "cog": np.asarray(cog, dtype=float),
        "force_effectiveness": np.asarray(force, dtype=float),
        "torque_effectiveness": np.asarray(torque, dtype=float),
    }
    if scalar:
        return {
            "mass": result["mass"],
            "inertia": result["inertia"][0],
            "cog": result["cog"][0],
            "force_effectiveness": result["force_effectiveness"][0],
            "torque_effectiveness": result["torque_effectiveness"][0],
        }
    return result


def _map_static_payload(
    solution: FixedGraphLaplaceSolution,
    audits: Tuple[_FactorAudit, ...],
    components: Mapping[str, float],
    bag_ids: Tuple[str, ...],
) -> Mapping[str, np.ndarray]:
    static_key = VariableKey(VariableKind.STATIC_PARAMETERS)
    coordinate = solution.lm.state.value(static_key)
    physical = _physical_parameter_arrays(
        coordinate, solution.prepared.parameter_chart
    )
    prior_objective = 0.0
    bag_objectives = {bag_id: 0.0 for bag_id in bag_ids}
    for audit, factor in zip(audits, solution.final_linearization.factors):
        objective = 0.5 * factor.squared_error
        if audit.group == "prior":
            prior_objective += objective
        if audit.bag_id is not None:
            bag_objectives[audit.bag_id] += objective
    component_names = tuple(sorted(components))
    return {
        "parameter_coordinate_map": np.asarray(coordinate, dtype=float),
        "mass": physical["mass"],
        "inertia": physical["inertia"],
        "cog": physical["cog"],
        "force_effectiveness": physical["force_effectiveness"],
        "torque_effectiveness": physical["torque_effectiveness"],
        "delay": np.asarray((solution.prepared.fixed_delay,), dtype=float),
        "q_diagonal": np.asarray(solution.prepared.dynamics.q, dtype=float),
        "objective_component_names": np.asarray(component_names),
        "objective_component_values": np.asarray(
            tuple(components[name] for name in component_names), dtype=float
        ),
        "prior_objective": np.asarray((prior_objective,), dtype=float),
        "likelihood_objective": np.asarray(
            (solution.lm.objective - prior_objective,), dtype=float
        ),
        "bag_id": np.asarray(bag_ids),
        "bag_objective": np.asarray(
            tuple(bag_objectives[bag_id] for bag_id in bag_ids), dtype=float
        ),
    }


def _q_em_payload(result: LaplaceEmResult) -> Mapping[str, np.ndarray]:
    iterations = result.iterations
    reasons = []
    for value in iterations:
        reason = value.q_update.termination_reason
        if value.lag_refinement_failed:
            reason = "{};lag_refinement_failed:{}".format(
                reason, value.lag_refinement_failure_reason
            )
        reasons.append(reason)
    return {
        "iteration": np.asarray(
            tuple(value.iteration for value in iterations), dtype=np.int64
        ),
        "input_q": np.asarray(
            tuple(value.input_step.q for value in iterations), dtype=float
        ),
        "target_q": np.asarray(
            tuple(value.q_target.target for value in iterations), dtype=float
        ),
        "accepted_q": np.asarray(
            tuple(value.q_update.accepted_q for value in iterations), dtype=float
        ),
        "alpha": np.asarray(
            tuple(value.q_update.accepted_alpha for value in iterations),
            dtype=float,
        ),
        "log_q_change": np.asarray(
            tuple(value.q_update.max_log_q_change for value in iterations),
            dtype=float,
        ),
        "map_objective": np.asarray(
            tuple(value.output_step.map_objective for value in iterations),
            dtype=float,
        ),
        "approximate_marginal_objective": np.asarray(
            tuple(
                value.output_step.approximate_marginal_objective
                for value in iterations
            ),
            dtype=float,
        ),
        "lag": np.asarray(
            tuple(value.output_step.lag for value in iterations), dtype=float
        ),
        "accepted": np.asarray(
            tuple(value.q_update.accepted for value in iterations), dtype=bool
        ),
        "reason": np.asarray(tuple(reasons)),
        "floor_activation": np.asarray(
            tuple(value.q_target.floor_active for value in iterations),
            dtype=bool,
        ),
        "expected_residual_second_moment": np.asarray(
            tuple(value.q_target.raw_target for value in iterations), dtype=float
        ),
        "map_residual_second_moment": np.asarray(
            tuple(value.q_target.map_second_moment for value in iterations),
            dtype=float,
        ),
        "covariance_correction": np.asarray(
            tuple(value.q_target.covariance_correction for value in iterations),
            dtype=float,
        ),
    }


def _delay_profile_arrays(
    final_q_lag_profile: Optional[LagProfileResult],
) -> Tuple[np.ndarray, np.ndarray]:
    if final_q_lag_profile is None:
        return np.asarray((), dtype=float), np.asarray((), dtype=float)
    if not isinstance(final_q_lag_profile, LagProfileResult):
        raise TypeError("final_q_lag_profile must be LagProfileResult or None")
    profile = {}
    for point in final_q_lag_profile.points:
        lag = float(point.lag)
        if not np.isfinite(lag) or lag < 0.0:
            raise ValueError("final-Q lag profile contains an invalid lag")
        objective = (
            float(point.objective)
            if point.converged and point.objective is not None
            else float("inf")
        )
        if np.isnan(objective):
            raise ValueError("final-Q lag profile objective cannot be NaN")
        current = profile.get(lag)
        profile[lag] = (
            objective if current is None else min(current, objective)
        )
    if not profile:
        raise ValueError("an available final-Q lag profile must contain points")
    grid = np.asarray(tuple(sorted(profile)), dtype=float)
    objective = np.asarray(tuple(profile[value] for value in grid), dtype=float)
    if not np.any(np.isfinite(objective)):
        raise ValueError("final-Q lag profile has no converged point")
    return grid, objective


def _laplace_payload(
    geometry: StaticLaplaceGeometry,
    final_q_lag_profile: Optional[LagProfileResult],
    delay_geometry: DelayLocalGeometry,
) -> Mapping[str, np.ndarray]:
    if final_q_lag_profile is None and delay_geometry.curvature is not None:
        raise ValueError(
            "delay curvature cannot be exported without a final-Q profile"
        )
    grid, objective = _delay_profile_arrays(final_q_lag_profile)
    likelihood = geometry.information.likelihood
    posterior = geometry.information.posterior
    curvature_valid = delay_geometry.curvature is not None
    return {
        "reduced_likelihood_hessian": np.asarray(
            likelihood.hessian, dtype=float
        ),
        "reduced_posterior_hessian": np.asarray(
            posterior.hessian, dtype=float
        ),
        "covariance": np.asarray(geometry.covariance, dtype=float),
        "eigenvalues": np.asarray(likelihood.eigenvalues, dtype=float),
        "eigenvectors": np.asarray(likelihood.eigenvectors, dtype=float),
        "effective_rank": np.asarray(
            (likelihood.effective_rank,), dtype=np.int64
        ),
        "exact_ridge_direction": np.asarray(
            geometry.exact_ridge_direction, dtype=float
        ),
        "ridge_alignment": np.asarray((geometry.ridge_alignment,), dtype=float),
        "condition_number": np.asarray(
            (likelihood.condition_number,), dtype=float
        ),
        "delay_profile_available": np.asarray(
            (final_q_lag_profile is not None,), dtype=bool
        ),
        "delay_profile_grid": grid,
        "delay_profile_objective": objective,
        "delay_local_uncertainty": np.asarray(
            (delay_geometry.standard_deviation_seconds,), dtype=float
        ),
        "delay_uncertainty_source": np.asarray((delay_geometry.source,)),
        "delay_profile_curvature": np.asarray(
            (
                delay_geometry.curvature
                if delay_geometry.curvature is not None
                else 0.0,
            ),
            dtype=float,
        ),
        "delay_profile_curvature_valid": np.asarray(
            (curvature_valid,), dtype=bool
        ),
    }


def _mcmc_diagnostic_arrays(
    diagnostics: McmcDiagnostics,
) -> Mapping[str, np.ndarray]:
    names = tuple(sorted(diagnostics.kernel_summaries))
    if not names:
        raise ValueError("MCMC diagnostics require kernel summaries")
    fields = (
        ("attempts", "mcmc_kernel_attempts"),
        ("stage_one_accepted", "mcmc_kernel_stage_one_accepted"),
        ("stage_two_attempted", "mcmc_kernel_stage_two_attempted"),
        ("stage_two_accepted", "mcmc_kernel_stage_two_accepted"),
        ("full_target_cache_hits", "mcmc_kernel_full_target_cache_hits"),
        ("inner_solve_failures", "mcmc_kernel_inner_solve_failures"),
        ("inner_iterations", "mcmc_kernel_inner_iterations"),
    )
    result = {
        "mcmc_chain_id": np.asarray(diagnostics.chain_ids),
        "mcmc_mode_id": np.asarray((diagnostics.mode_id,)),
        "mcmc_draws_per_chain": np.asarray(
            (diagnostics.draws_per_chain,), dtype=np.int64
        ),
        "mcmc_split_rhat": np.asarray(diagnostics.split_rhat, dtype=float),
        "mcmc_effective_sample_size": np.asarray(
            diagnostics.effective_sample_size, dtype=float
        ),
        "mcmc_integrated_autocorrelation_time": np.asarray(
            diagnostics.integrated_autocorrelation_time, dtype=float
        ),
        "mcmc_ridge_coordinate_trace": np.asarray(
            diagnostics.ridge_coordinate_trace, dtype=float
        ),
        "mcmc_delay_trace": np.asarray(diagnostics.delay_trace, dtype=float),
        "mcmc_log_posterior_trace": np.asarray(
            diagnostics.log_density_trace, dtype=float
        ),
        "mcmc_kernel_names": np.asarray(names),
        "mcmc_completed": np.asarray((diagnostics.completed,), dtype=bool),
        "mcmc_converged": np.asarray((diagnostics.converged,), dtype=bool),
        "mcmc_rhat_threshold": np.asarray(
            (diagnostics.rhat_threshold,), dtype=float
        ),
        "mcmc_minimum_effective_sample_size": np.asarray(
            (diagnostics.minimum_effective_sample_size,), dtype=float
        ),
    }
    for field, output in fields:
        result[output] = np.asarray(
            tuple(
                getattr(diagnostics.kernel_summaries[name], field)
                for name in names
            ),
            dtype=np.int64,
        )
    return result


def _diagnostics_payload(
    bag_ids: Tuple[str, ...],
    performance: RunPerformanceMeasurements,
    mcmc_diagnostics: Optional[McmcDiagnostics],
) -> Mapping[str, np.ndarray]:
    ordered = {value.bag_id: value for value in performance.bags}
    if tuple(ordered) != bag_ids:
        raise ValueError("performance bag order must match the request bag order")
    result = {
        "bag_id": np.asarray(bag_ids),
        "knot_count": np.asarray(
            tuple(ordered[value].knot_count for value in bag_ids),
            dtype=np.int64,
        ),
        "factor_count": np.asarray(
            tuple(ordered[value].factor_count for value in bag_ids),
            dtype=np.int64,
        ),
        "residual_dimension": np.asarray(
            tuple(ordered[value].residual_dimension for value in bag_ids),
            dtype=np.int64,
        ),
        "jacobian_nnz": np.asarray(
            tuple(ordered[value].jacobian_nnz for value in bag_ids),
            dtype=np.int64,
        ),
        "assembly_seconds": np.asarray(
            tuple(ordered[value].assembly_seconds for value in bag_ids),
            dtype=float,
        ),
        "factorization_seconds": np.asarray(
            tuple(ordered[value].factorization_seconds for value in bag_ids),
            dtype=float,
        ),
        "schur_solve_seconds": np.asarray(
            tuple(ordered[value].schur_solve_seconds for value in bag_ids),
            dtype=float,
        ),
        "nonlinear_iteration_seconds": np.asarray(
            performance.nonlinear_iteration_seconds, dtype=float
        ),
        "em_iteration_seconds": np.asarray(
            performance.em_iteration_seconds, dtype=float
        ),
        "mcmc_target_seconds": np.asarray(
            performance.mcmc_target_seconds, dtype=float
        ),
        "peak_memory_bytes": np.asarray(
            (performance.peak_memory_bytes,), dtype=np.int64
        ),
    }
    if mcmc_diagnostics is not None:
        result.update(_mcmc_diagnostic_arrays(mcmc_diagnostics))
    return result


def _factor_enabled(bag_request: Mapping[str, Any], name: str) -> bool:
    return bool(bag_request["observation_factors"][name]["enabled"])


def _support_mask(times: np.ndarray, knot_times: np.ndarray, enabled: bool) -> np.ndarray:
    selected = np.asarray(times, dtype=float)
    tolerance = 2.0e-10 * max(1.0, abs(knot_times[0]), abs(knot_times[-1]))
    return np.asarray(
        enabled
        & (selected >= knot_times[0] - tolerance)
        & (selected <= knot_times[-1] + tolerance),
        dtype=bool,
    )


def _covariance_series(
    count: int, covariance: Optional[GaussianCovariance], dimension: int
) -> Tuple[np.ndarray, np.ndarray]:
    if covariance is None:
        return (
            np.zeros((count, dimension, dimension), dtype=float),
            np.zeros(count, dtype=bool),
        )
    return (
        np.repeat(covariance.value[None, :, :], count, axis=0),
        np.ones(count, dtype=bool),
    )


def _pose_covariance_series(
    count: int, bag: PreparedBagGraphData
) -> Tuple[np.ndarray, np.ndarray]:
    if (
        bag.covariances.position_observation is None
        or bag.covariances.orientation_observation is None
    ):
        return np.zeros((count, 6, 6)), np.zeros(count, dtype=bool)
    covariance = np.zeros((6, 6), dtype=float)
    covariance[:3, :3] = bag.covariances.position_observation.value
    covariance[3:, 3:] = bag.covariances.orientation_observation.value
    return (
        np.repeat(covariance[None, :, :], count, axis=0),
        np.ones(count, dtype=bool),
    )


def _empty_series(width: int, covariance_width: int) -> Mapping[str, np.ndarray]:
    return {
        "time": np.asarray((), dtype=float),
        "record_time": np.asarray((), dtype=float),
        "value": np.empty((0, width), dtype=float),
        "valid": np.empty((0,), dtype=bool),
        "covariance": np.empty((0, covariance_width, covariance_width)),
        "covariance_valid": np.empty((0,), dtype=bool),
    }


def _vector_series_payload(
    series: Optional[object],
    knot_times: np.ndarray,
    enabled: bool,
    covariance: Optional[GaussianCovariance],
    width: int,
) -> Mapping[str, np.ndarray]:
    if series is None:
        return _empty_series(width, width)
    times = np.asarray(series.times, dtype=float)
    record = np.asarray(series.record_times, dtype=float)
    values = np.asarray(series.values, dtype=float)
    if values.shape != (times.size, width):
        raise ValueError("raw stream width changed before artifact export")
    covariances, covariance_valid = _covariance_series(
        times.size, covariance, width
    )
    return {
        "time": times,
        "record_time": record,
        "value": values,
        "valid": _support_mask(times, knot_times, enabled),
        "covariance": covariances,
        "covariance_valid": covariance_valid,
    }


def _insert_stream(
    target: Dict[str, np.ndarray],
    prefix: str,
    value_name: str,
    stream: Mapping[str, np.ndarray],
) -> None:
    target["{}_time".format(prefix)] = stream["time"]
    target["{}_record_time".format(prefix)] = stream["record_time"]
    target[value_name] = stream["value"]
    target["{}_valid".format(prefix)] = stream["valid"]
    target["{}_covariance".format(prefix)] = stream["covariance"]
    target["{}_covariance_valid".format(prefix)] = stream["covariance_valid"]


def _controller_integral_stream(
    flight: FlightData,
    knot_times: np.ndarray,
    enabled: bool,
    covariance: Optional[GaussianCovariance],
) -> Mapping[str, np.ndarray]:
    pid = flight.pid_debug
    if pid is None:
        return _empty_series(6, 6)
    gains = np.asarray(
        tuple(value.i_gain for value in flight.controller_configuration.pid),
        dtype=float,
    )
    if gains.shape != (6,) or np.any(np.abs(gains) <= 1.0e-12):
        if enabled:
            raise ValueError(
                "enabled controller-integral factor is not uniquely decodable"
            )
        return _empty_series(6, 6)
    values = np.asarray(pid.i_term, dtype=float) / gains[None, :]
    covariances, covariance_valid = _covariance_series(
        pid.times.size, covariance, 6
    )
    return {
        "time": np.asarray(pid.times, dtype=float),
        "record_time": np.asarray(pid.record_times, dtype=float),
        "value": values,
        "valid": _support_mask(pid.times, knot_times, enabled),
        "covariance": covariances,
        "covariance_valid": covariance_valid,
    }


def _dynamics_path(
    solution: FixedGraphLaplaceSolution,
    bag_id: str,
    interval_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    residual = np.zeros((interval_count, 6), dtype=float)
    valid = np.zeros(interval_count, dtype=bool)
    for value in solution.dynamics.linearizations.intervals:
        if value.bag_id == bag_id:
            residual[value.left_knot_index] = value.residual
            valid[value.left_knot_index] = True
    for value in solution.dynamics.linearizations.excluded_intervals:
        if value.bag_id == bag_id and valid[value.left_knot_index]:
            raise ValueError("valid and excluded dynamics intervals overlap")
    return residual, valid


def _bag_payload(
    *,
    solution: FixedGraphLaplaceSolution,
    prepared_bag: PreparedBagGraphData,
    request_bag: Mapping[str, Any],
    flight: FlightData,
    initialization: FlightInitialization,
    factor_payload: Mapping[str, np.ndarray],
    audits: Tuple[_FactorAudit, ...],
) -> Mapping[str, np.ndarray]:
    bag_id = prepared_bag.bag_id
    knot_times = np.asarray(
        tuple(value.time for value in prepared_bag.knots), dtype=float
    )
    if not np.array_equal(knot_times, initialization.grid.times):
        raise ValueError("prepared and initialized knot grids disagree")
    knot_count = knot_times.size
    nominal = _state_arrays(initialization.state, bag_id, knot_count)
    mapped = _state_arrays(solution.lm.state, bag_id, knot_count)
    correction_translation, correction_rotation = correction_transform_path(
        nominal["position"],
        nominal["orientation_xyzw"],
        mapped["position"],
        mapped["orientation_xyzw"],
    )
    map_dynamics, map_dynamics_valid = _dynamics_path(
        solution, bag_id, knot_count - 1
    )
    reference = flight.reference
    pose = flight.pose
    pose_covariance, pose_covariance_valid = _pose_covariance_series(
        pose.times.size, prepared_bag
    )
    pose_valid = _support_mask(
        pose.times, knot_times, _factor_enabled(request_bag, "pose")
    )
    if int(np.count_nonzero(pose_valid)) != len(prepared_bag.pose_measurements):
        raise ValueError("pose validity mask disagrees with prepared factors")
    result: Dict[str, np.ndarray] = {
        "bag_id": np.asarray((bag_id,)),
        "knot_time": knot_times,
        "knot_record_time": (
            float(flight.provenance.bag_record_start) + knot_times
        ),
        "reference_time": np.asarray(reference.times, dtype=float),
        "reference_record_time": np.asarray(reference.record_times, dtype=float),
        "reference_position": np.asarray(reference.position, dtype=float),
        "reference_linear_velocity": np.asarray(
            reference.linear_velocity, dtype=float
        ),
        "reference_linear_acceleration": np.asarray(
            reference.linear_acceleration, dtype=float
        ),
        "reference_rpy": np.asarray(reference.rpy, dtype=float),
        "reference_angular_velocity": np.asarray(
            reference.angular_velocity, dtype=float
        ),
        "reference_angular_acceleration": np.asarray(
            reference.angular_acceleration, dtype=float
        ),
        "pose_time": np.asarray(pose.times, dtype=float),
        "pose_record_time": np.asarray(pose.record_times, dtype=float),
        "pose_position": np.asarray(pose.positions, dtype=float),
        "pose_orientation_xyzw": np.asarray(
            pose.orientations_xyzw, dtype=float
        ),
        "pose_valid": pose_valid,
        "pose_covariance": pose_covariance,
        "pose_covariance_valid": pose_covariance_valid,
        "map_dynamics_residual": map_dynamics,
        "map_dynamics_residual_valid": map_dynamics_valid,
        "correction_translation": correction_translation,
        "correction_rotation_vector": correction_rotation,
    }
    for name, _kind, _width in _STATE_FIELDS:
        result["nominal_{}".format(name)] = nominal[name]
        result["map_{}".format(name)] = mapped[name]

    stream_specs = (
        (
            "velocity",
            "velocity",
            flight.velocity,
            "velocity",
            prepared_bag.covariances.velocity_observation,
            len(prepared_bag.velocity_measurements),
            3,
        ),
        (
            "gyro",
            "gyro",
            flight.gyro,
            "gyro",
            prepared_bag.covariances.gyro_observation,
            len(prepared_bag.gyro_measurements),
            3,
        ),
        (
            "accelerometer",
            "accelerometer",
            flight.accelerometer,
            "accelerometer",
            prepared_bag.covariances.accelerometer_observation,
            len(prepared_bag.accelerometer_measurements),
            3,
        ),
        (
            "gimbal_observation",
            "gimbal_observation",
            flight.gimbal_position,
            "actual_gimbal_position",
            prepared_bag.covariances.actual_gimbal_observation,
            len(prepared_bag.actual_gimbal_measurements),
            4,
        ),
        (
            "thrust_command",
            "thrust_command",
            flight.rotor_command,
            "issued_rotor_command",
            prepared_bag.covariances.issued_thrust_observation,
            None,
            4,
        ),
        (
            "gimbal_command",
            "gimbal_command",
            flight.gimbal_command,
            "issued_gimbal_command",
            prepared_bag.covariances.issued_gimbal_observation,
            None,
            4,
        ),
    )
    for (
        prefix,
        value_name,
        series,
        factor_name,
        covariance,
        expected_count,
        width,
    ) in stream_specs:
        stream = _vector_series_payload(
            series,
            knot_times,
            _factor_enabled(request_bag, factor_name),
            covariance,
            width,
        )
        if expected_count is not None and int(
            np.count_nonzero(stream["valid"])
        ) != expected_count:
            raise ValueError(
                "{} validity mask disagrees with prepared factors".format(prefix)
            )
        _insert_stream(result, prefix, value_name, stream)

    integral_stream = _controller_integral_stream(
        flight,
        knot_times,
        _factor_enabled(request_bag, "controller_integral"),
        prepared_bag.covariances.controller_integral_observation,
    )
    if int(np.count_nonzero(integral_stream["valid"])) != len(
        prepared_bag.controller_integral_measurements
    ):
        raise ValueError(
            "controller-integral validity mask disagrees with prepared factors"
        )
    _insert_stream(
        result,
        "controller_integral",
        "controller_integral_observation",
        integral_stream,
    )
    result.update(factor_payload)

    bag_audits = tuple(value for value in audits if value.bag_id == bag_id)
    active_count = 0
    for audit, factor in zip(audits, solution.final_linearization.factors):
        if audit.bag_id == bag_id:
            active_count += sum(
                int(np.count_nonzero(mask)) for mask in factor.active_set.values()
            )
    result["numerical_diagnostic_names"] = np.asarray(
        (
            "final_gradient_inf_norm",
            "final_damping",
            "lm_iteration_count",
            "valid_dynamics_interval_count",
            "excluded_dynamics_interval_count",
            "active_set_true_count",
        )
    )
    result["numerical_diagnostic_values"] = np.asarray(
        (
            solution.lm.final_gradient_inf_norm,
            solution.lm.final_damping,
            len(solution.lm.iterations),
            int(np.count_nonzero(map_dynamics_valid)),
            int(map_dynamics_valid.size - np.count_nonzero(map_dynamics_valid)),
            active_count,
        ),
        dtype=float,
    )
    if not bag_audits:
        raise ValueError("a selected bag has no graph factors")
    return result


def _validate_mcmc_inputs(
    chains: Sequence[McmcChainResult], diagnostics: McmcDiagnostics
) -> Tuple[McmcChainResult, ...]:
    if not isinstance(chains, (tuple, list)) or not chains:
        raise ValueError("enabled MCMC requires completed chains")
    selected = tuple(chains)
    if any(not isinstance(value, McmcChainResult) for value in selected):
        raise TypeError("mcmc_chains must contain McmcChainResult values")
    by_id = {value.chain_id: value for value in selected}
    if len(by_id) != len(selected):
        raise ValueError("MCMC chain IDs must be unique")
    if tuple(by_id) != diagnostics.chain_ids:
        raise ValueError("MCMC chain order must match diagnostics")
    if any(value.mode_id != diagnostics.mode_id for value in selected):
        raise ValueError("MCMC chains and diagnostics must use one mode")
    if any(
        value.sample_id.size != diagnostics.draws_per_chain
        for value in selected
    ):
        raise ValueError("MCMC retained draw count disagrees with diagnostics")
    if any(
        value.graph_objective is None
        or value.local_log_determinant is None
        or value.delay_log_prior is None
        for value in selected
    ):
        raise ValueError(
            "MCMC artifact export requires all target component traces"
        )
    all_ids = tuple(
        str(sample_id)
        for chain in selected
        for sample_id in chain.sample_id.tolist()
    )
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("MCMC sample IDs must be globally unique")
    return selected


def _mcmc_payload(
    solution: FixedGraphLaplaceSolution,
    chains: Sequence[McmcChainResult],
    diagnostics: McmcDiagnostics,
) -> Mapping[str, np.ndarray]:
    selected = _validate_mcmc_inputs(chains, diagnostics)
    coordinates = np.vstack(tuple(value.static_coordinate for value in selected))
    physical = _physical_parameter_arrays(
        coordinates, solution.prepared.parameter_chart
    )
    sample_id = np.concatenate(tuple(value.sample_id for value in selected))
    chain_id = np.concatenate(
        tuple(
            np.full(value.sample_id.size, value.chain_id)
            for value in selected
        )
    )
    source_mode = np.concatenate(
        tuple(
            np.full(value.sample_id.size, value.mode_id)
            for value in selected
        )
    )
    draw_index = np.concatenate(tuple(value.draw_index for value in selected))
    delay = np.concatenate(tuple(value.delay for value in selected))
    log_posterior = np.concatenate(
        tuple(value.log_density for value in selected)
    )
    graph_objective = np.concatenate(
        tuple(value.graph_objective for value in selected)
    )
    log_determinant = np.concatenate(
        tuple(value.local_log_determinant for value in selected)
    )
    delay_log_prior = np.concatenate(
        tuple(value.delay_log_prior for value in selected)
    )
    accepted_kernel = np.concatenate(
        tuple(value.accepted_kernel for value in selected)
    )
    prior = solution.prepared.static_parameter_prior
    centered = coordinates - prior.mean[None, :]
    normalized = centered @ prior.covariance.square_root_information.T
    prior_nll = 0.5 * np.sum(normalized * normalized, axis=1)
    log_likelihood = -graph_objective + prior_nll
    log_determinant_term = -0.5 * log_determinant
    reconstructed = (
        -prior_nll
        + delay_log_prior
        + log_likelihood
        + log_determinant_term
    )
    if not np.allclose(
        reconstructed, log_posterior, rtol=2.0e-11, atol=2.0e-11
    ):
        raise ValueError("exported MCMC terms do not reconstruct log posterior")
    return {
        "sample_id": np.asarray(sample_id),
        "chain_id": np.asarray(chain_id),
        "draw_index": np.asarray(draw_index, dtype=np.int64),
        "parameter_coordinate": coordinates,
        "mass": physical["mass"],
        "inertia": physical["inertia"],
        "cog": physical["cog"],
        "force_effectiveness": physical["force_effectiveness"],
        "torque_effectiveness": physical["torque_effectiveness"],
        "delay": delay,
        "log_posterior": log_posterior,
        "log_likelihood_approximation": log_likelihood,
        "log_determinant_term": log_determinant_term,
        "accepted_kernel": np.asarray(accepted_kernel),
        "source_mode_id": np.asarray(source_mode),
    }


def _trajectory_payloads(
    selected: Sequence[SelectedConditionalTrajectory],
    mcmc_samples: Optional[Mapping[str, np.ndarray]],
    prepared_bags: Mapping[str, PreparedBagGraphData],
    initializations: Mapping[str, FlightInitialization],
) -> Mapping[str, Mapping[str, np.ndarray]]:
    if not isinstance(selected, (tuple, list)):
        raise TypeError("selected_trajectories must be a tuple or list")
    if any(not isinstance(value, SelectedConditionalTrajectory) for value in selected):
        raise TypeError(
            "selected_trajectories must contain SelectedConditionalTrajectory"
        )
    if selected and mcmc_samples is None:
        raise ValueError("selected trajectories require MCMC samples")
    if not selected:
        return {}
    sample_ids = tuple(str(value) for value in mcmc_samples["sample_id"].tolist())
    sample_coordinate = {
        sample_id: mcmc_samples["parameter_coordinate"][index]
        for index, sample_id in enumerate(sample_ids)
    }
    groups: Dict[str, list] = {}
    pairs = set()
    for value in selected:
        pair = (value.sample_id, value.bag_id)
        if pair in pairs:
            raise ValueError("selected trajectory sample/bag pairs must be unique")
        pairs.add(pair)
        if value.sample_id not in sample_coordinate:
            raise ValueError("selected trajectory sample is absent from MCMC")
        if value.bag_id not in prepared_bags:
            raise ValueError("selected trajectory has an unknown bag ID")
        static_key = VariableKey(VariableKind.STATIC_PARAMETERS)
        if not np.allclose(
            value.state.value(static_key),
            sample_coordinate[value.sample_id],
            rtol=1.0e-12,
            atol=1.0e-13,
        ):
            raise ValueError(
                "selected trajectory static coordinate disagrees with sample"
            )
        groups.setdefault(value.bag_id, []).append(value)
    result = {}
    for bag_id, values in groups.items():
        prepared = prepared_bags[bag_id]
        initialization = initializations[bag_id]
        knot_times = np.asarray(
            tuple(value.time for value in prepared.knots), dtype=float
        )
        knot_count = knot_times.size
        nominal = _state_arrays(initialization.state, bag_id, knot_count)
        states = tuple(
            _state_arrays(value.state, bag_id, knot_count) for value in values
        )
        translations = []
        rotations = []
        for state in states:
            translation, rotation = correction_transform_path(
                nominal["position"],
                nominal["orientation_xyzw"],
                state["position"],
                state["orientation_xyzw"],
            )
            translations.append(translation)
            rotations.append(rotation)
        if any(
            value.dynamics_residual.shape != (knot_count - 1, 6)
            for value in values
        ):
            raise ValueError("selected trajectory residual does not match knots")
        result[bag_id] = {
            "sample_id": np.asarray(tuple(value.sample_id for value in values)),
            "knot_time": knot_times,
            "conditional_position": np.asarray(
                tuple(value["position"] for value in states), dtype=float
            ),
            "conditional_orientation_xyzw": np.asarray(
                tuple(value["orientation_xyzw"] for value in states), dtype=float
            ),
            "conditional_linear_velocity": np.asarray(
                tuple(value["linear_velocity"] for value in states), dtype=float
            ),
            "conditional_angular_velocity": np.asarray(
                tuple(value["angular_velocity"] for value in states), dtype=float
            ),
            "conditional_controller_integral": np.asarray(
                tuple(value["controller_integral"] for value in states),
                dtype=float,
            ),
            "conditional_actuator_thrust": np.asarray(
                tuple(value["actuator_thrust"] for value in states), dtype=float
            ),
            "conditional_actuator_gimbal": np.asarray(
                tuple(value["actuator_gimbal"] for value in states), dtype=float
            ),
            "correction_translation": np.asarray(translations, dtype=float),
            "correction_rotation_vector": np.asarray(rotations, dtype=float),
            "dynamics_residual": np.asarray(
                tuple(value.dynamics_residual for value in values), dtype=float
            ),
            "dynamics_residual_valid": np.asarray(
                tuple(value.dynamics_residual_valid for value in values),
                dtype=bool,
            ),
            "conditional_objective": np.asarray(
                tuple(value.conditional_objective for value in values),
                dtype=float,
            ),
        }
    return result


def _sensor_contract_payload(flight: FlightData) -> Mapping[str, Any]:
    topics = []
    for value in flight.sensor_contract:
        topics.append(
            {
                "topic": value.topic,
                "message_type": value.message_type,
                "timestamp_source": value.timestamp_source.value,
                "usage": value.usage.value,
                "frame_id": value.frame_id,
                "fields": list(value.fields),
                "units": list(value.units),
                "sample_rate_hz": value.sample_rate_hz,
                "median_gap_seconds": value.median_gap_seconds,
                "maximum_gap_seconds": value.maximum_gap_seconds,
                "duplicate_timestamp_count": value.duplicate_timestamp_count,
                "nonmonotonic_timestamp_count": value.nonmonotonic_timestamp_count,
                "covariance_provenance": value.covariance_provenance,
                "unavailable_reason": value.unavailable_reason,
                "mixed_frame_notes": value.mixed_frame_notes,
            }
        )
    extrinsics = flight.sensor_extrinsics
    return {
        "topics": topics,
        "extrinsics": {
            "body_frame": extrinsics.body_frame,
            "pose_sensor_frame": extrinsics.pose_sensor_frame,
            "velocity_sensor_frame": extrinsics.velocity_sensor_frame,
            "gyro_sensor_frame": extrinsics.gyro_sensor_frame,
            "pose_sensor_position_in_body": (
                extrinsics.pose_sensor_position_in_body.tolist()
            ),
            "pose_sensor_to_body_rotation": (
                extrinsics.pose_sensor_to_body_rotation.tolist()
            ),
            "velocity_sensor_position_in_body": (
                extrinsics.velocity_sensor_position_in_body.tolist()
            ),
            "velocity_sensor_to_body_rotation": (
                extrinsics.velocity_sensor_to_body_rotation.tolist()
            ),
            "gyro_sensor_position_in_body": (
                extrinsics.gyro_sensor_position_in_body.tolist()
            ),
            "body_to_gyro_sensor_rotation": (
                extrinsics.body_to_gyro_sensor_rotation.tolist()
            ),
            "source": extrinsics.source,
        },
        "pose_twist_cross_correlation_assumption": "ignored_independent_factors",
    }


def _manifest_metadata(
    *,
    request: BatchEstimationRequest,
    flights: Mapping[str, FlightData],
    identity: ArtifactRunIdentity,
    solution: FixedGraphLaplaceSolution,
    em_result: LaplaceEmResult,
    mcmc_diagnostics: Optional[McmcDiagnostics],
) -> Mapping[str, Any]:
    payload = request.payload
    bag_requests = _request_bags(request)
    bag_ids = request.bag_ids
    observation_factors = {}
    for bag_id in bag_ids:
        observation_factors[bag_id] = {
            name: {
                "enabled": bool(value["enabled"]),
                "disabled_reason": value["disabled_reason"],
            }
            for name, value in bag_requests[bag_id][
                "observation_factors"
            ].items()
        }
    substage_status = {
        "map": {
            "converged": bool(solution.lm.converged),
            "termination_reason": solution.lm.reason.value,
        },
        "laplace_em": {
            "converged": bool(em_result.converged),
            "termination_reason": em_result.reason.value,
        },
        "laplace": {
            "converged": True,
            "termination_reason": "completed",
        },
    }
    if mcmc_diagnostics is not None:
        substage_status["mcmc"] = {
            "converged": bool(mcmc_diagnostics.converged),
            "termination_reason": (
                "converged_diagnostics"
                if mcmc_diagnostics.converged
                else "completed_not_converged"
            ),
        }
    definition = solution.prepared.dynamics.q_definition
    return {
        "run_id": str(payload["run_id"]),
        "estimator_revision": identity.estimator_revision,
        "selected_bag_ids": list(bag_ids),
        "selected_intervals": {
            bag_id: list(bag_requests[bag_id]["interval_seconds"])
            for bag_id in bag_ids
        },
        "selected_bag_sha256": {
            bag_id: str(bag_requests[bag_id]["sha256"])
            for bag_id in bag_ids
        },
        "configuration_fingerprint": identity.configuration_fingerprint,
        "controller_snapshot_fingerprint": (
            identity.controller_snapshot_fingerprint
        ),
        "sensor_contracts": {
            bag_id: _sensor_contract_payload(flights[bag_id])
            for bag_id in bag_ids
        },
        "observation_factors": observation_factors,
        "parameter_prior": _plain(payload["parameter_prior"]),
        "delay_prior": _plain(payload["delay"]),
        "q_definition": {
            "definition": "{}/{}".format(
                definition.residual_quantity, definition.interval_model.value
            ),
            "components": list(definition.component_names),
            "units": list(definition.component_units),
        },
        "knot_policy": _plain(payload["knot_policy"]),
        "interpolation_policy": _plain(payload["interpolation_policy"]),
        "solver_settings": _plain(payload["solver_settings"]),
        "em_settings": _plain(payload["em_settings"]),
        "mcmc_settings": _plain(payload["mcmc_settings"]),
        "request_fingerprint": request.fingerprint,
        "substage_status": substage_status,
        "warnings": list(identity.warnings),
    }


def _validate_primary_inputs(
    request: BatchEstimationRequest,
    flight_data: Sequence[FlightData],
    initializations: Sequence[FlightInitialization],
    final_solution: FixedGraphLaplaceSolution,
) -> Tuple[
    Mapping[str, FlightData],
    Mapping[str, FlightInitialization],
    Mapping[str, PreparedBagGraphData],
]:
    if not isinstance(request, BatchEstimationRequest):
        raise TypeError("request must be a validated BatchEstimationRequest")
    if not isinstance(flight_data, (tuple, list)) or not flight_data:
        raise TypeError("flight_data must be a non-empty tuple or list")
    if any(not isinstance(value, FlightData) for value in flight_data):
        raise TypeError("flight_data must contain FlightData values")
    if not isinstance(initializations, (tuple, list)) or not initializations:
        raise TypeError("initializations must be a non-empty tuple or list")
    if any(
        not isinstance(value, FlightInitialization) for value in initializations
    ):
        raise TypeError(
            "initializations must contain FlightInitialization values"
        )
    flights = _by_bag_id(flight_data, "flight_data")
    initialized = _by_bag_id(initializations, "initializations")
    prepared = _by_bag_id(final_solution.prepared.bags, "prepared bags")
    expected = set(request.bag_ids)
    for name, values in (
        ("flight_data", flights),
        ("initializations", initialized),
        ("prepared bags", prepared),
    ):
        if set(values) != expected:
            raise ValueError("{} must exactly match request bag IDs".format(name))
    request_bags = _request_bags(request)
    for bag_id in request.bag_ids:
        flight = flights[bag_id]
        bag_request = request_bags[bag_id]
        if (
            _normalized_flight_sha256(flight.provenance.bag_sha256)
            != bag_request["sha256"]
        ):
            raise ValueError("FlightData SHA-256 disagrees with the request")
        selected_interval = tuple(
            float(value) for value in bag_request["interval_seconds"]
        )
        if not np.allclose(
            selected_interval,
            (flight.interval.start, flight.interval.end),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("FlightData interval disagrees with the request")
    return flights, initialized, prepared


def export_batch_estimation_artifact_payload(
    *,
    request: BatchEstimationRequest,
    flight_data: Sequence[FlightData],
    initializations: Sequence[FlightInitialization],
    final_solution: FixedGraphLaplaceSolution,
    em_result: LaplaceEmResult,
    static_geometry: StaticLaplaceGeometry,
    final_q_lag_profile: Optional[LagProfileResult],
    delay_geometry: DelayLocalGeometry,
    identity: ArtifactRunIdentity,
    performance: RunPerformanceMeasurements,
    mcmc_chains: Sequence[McmcChainResult] = (),
    mcmc_diagnostics: Optional[McmcDiagnostics] = None,
    selected_trajectories: Sequence[SelectedConditionalTrajectory] = (),
) -> BatchArtifactPayload:
    """Build exact pickle-free arrays for the strict batch artifact writer."""

    if not isinstance(identity, ArtifactRunIdentity):
        raise TypeError("identity must be ArtifactRunIdentity")
    if not isinstance(performance, RunPerformanceMeasurements):
        raise TypeError("performance must be RunPerformanceMeasurements")
    if not isinstance(delay_geometry, DelayLocalGeometry):
        raise TypeError("delay_geometry must be DelayLocalGeometry")
    _validate_solver_alignment(final_solution, em_result, static_geometry)
    _validate_final_q_lag_profile(
        final_solution, final_q_lag_profile, delay_geometry
    )
    flights, initialized, prepared_bags = _validate_primary_inputs(
        request, flight_data, initializations, final_solution
    )
    if em_result.definition != final_solution.prepared.dynamics.q_definition:
        raise ValueError("Laplace-EM Q definition disagrees with final graph")
    request_mcmc_enabled = bool(request.payload["mcmc_settings"]["enabled"])
    if request_mcmc_enabled:
        if mcmc_diagnostics is None:
            raise ValueError("enabled MCMC requires McmcDiagnostics")
        if not performance.mcmc_target_seconds:
            raise ValueError("enabled MCMC requires measured target timings")
        mcmc = _mcmc_payload(
            final_solution, mcmc_chains, mcmc_diagnostics
        )
    else:
        if mcmc_chains or mcmc_diagnostics is not None:
            raise ValueError("disabled MCMC cannot export chains or diagnostics")
        if performance.mcmc_target_seconds:
            raise ValueError("disabled MCMC cannot report target timings")
        mcmc = None

    audits, components, factor_payloads = _factor_payload(final_solution)
    bag_ids = request.bag_ids
    request_bags = _request_bags(request)
    bags = {
        bag_id: _bag_payload(
            solution=final_solution,
            prepared_bag=prepared_bags[bag_id],
            request_bag=request_bags[bag_id],
            flight=flights[bag_id],
            initialization=initialized[bag_id],
            factor_payload=factor_payloads[bag_id],
            audits=audits,
        )
        for bag_id in bag_ids
    }
    measured = {value.bag_id: value for value in performance.bags}
    if tuple(measured) != bag_ids:
        raise ValueError("performance bag order must match request bag order")
    for bag_id in bag_ids:
        actual_factor_count = sum(
            value.bag_id == bag_id for value in audits
        )
        actual_residual_dimension = bags[bag_id]["factor_names"].size
        if measured[bag_id].knot_count != bags[bag_id]["knot_time"].size:
            raise ValueError("measured knot_count disagrees with graph")
        if measured[bag_id].factor_count != actual_factor_count:
            raise ValueError("measured factor_count disagrees with graph")
        if measured[bag_id].residual_dimension != actual_residual_dimension:
            raise ValueError("measured residual_dimension disagrees with graph")

    trajectories = _trajectory_payloads(
        selected_trajectories, mcmc, prepared_bags, initialized
    )
    manifest = _manifest_metadata(
        request=request,
        flights=flights,
        identity=identity,
        solution=final_solution,
        em_result=em_result,
        mcmc_diagnostics=mcmc_diagnostics,
    )
    return BatchArtifactPayload(
        manifest_metadata=manifest,
        map_static=_map_static_payload(
            final_solution, audits, components, bag_ids
        ),
        q_em=_q_em_payload(em_result),
        laplace=_laplace_payload(
            static_geometry, final_q_lag_profile, delay_geometry
        ),
        diagnostics=_diagnostics_payload(
            bag_ids, performance, mcmc_diagnostics
        ),
        bags=bags,
        mcmc_samples=mcmc,
        trajectories=trajectories,
    )


__all__ = [
    "ArtifactRunIdentity",
    "BagPerformanceMeasurements",
    "BatchArtifactPayload",
    "DelayLocalGeometry",
    "RunPerformanceMeasurements",
    "SelectedConditionalTrajectory",
    "export_batch_estimation_artifact_payload",
]
