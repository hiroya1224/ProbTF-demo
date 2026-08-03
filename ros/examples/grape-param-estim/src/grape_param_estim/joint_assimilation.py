"""Joint weak-constraint smoothing with shared plant and delay members."""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.controller import GrapeController
from grape_param_estim.dynamics import FullSixDofPlant, simulate_closed_loop
from grape_param_estim.ensemble_solver import (
    EnsembleSpaceIteration,
    EnsembleSpaceResult,
    EstimationCancelled,
    InitialPriorForecastDiagnostics,
    run_ensemble_space_ienks,
)
from grape_param_estim.geometry import (
    correction_transform_path,
    quaternion_to_matrix,
    rotation_vector_from_matrix,
)
from grape_param_estim.parameterization import PARAMETER_DIMENSION
from grape_param_estim.real_assimilation import build_real_strong_problem
from grape_param_estim.real_calibration import (
    KnotResolution,
    ModelErrorCalibration,
    calibrate_model_error_from_closed_loop_pose,
    select_ou_knot_resolution,
)
from grape_param_estim.real_rosbag import (
    PID_AXIS_NAMES,
    PID_CONFIG_FIELD_NAMES,
    RealFlightEpisode,
)
from grape_param_estim.strong_constraint import (
    INITIAL_STATE_DIMENSION,
    PARAMETER_OFFSET,
    ParameterRidge,
    StrongConstraintPrior,
)
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    ClosedLoopTrajectory,
    ControllerState,
    RigidBodyState,
    VehicleParameters,
)
from grape_param_estim.timing import ConstantDelayChart
from grape_param_estim.weak_constraint import WeakConstraintProblem
from grape_param_estim.model_error import KnotGaussMarkovWrenchProcess


ACTUATOR_STATE_DIMENSION = 8
SHARED_STATIC_DIMENSION = PARAMETER_DIMENSION + 1


@dataclass(frozen=True)
class JointIEnKSConfig:
    ensemble_size: int = 128
    maximum_iterations: int = 5
    convergence_tolerance: float = 1.0e-3
    minimum_line_search_step: float = 1.0 / 64.0
    seed: int = 23
    maximum_initial_prior_backoff_trials: int = 8

    def __post_init__(self) -> None:
        if self.ensemble_size < 3:
            raise ValueError("joint ensemble requires at least three members")
        if self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive")
        if (
            isinstance(
                self.maximum_initial_prior_backoff_trials, (bool, np.bool_)
            )
            or int(self.maximum_initial_prior_backoff_trials)
            != self.maximum_initial_prior_backoff_trials
            or not 0 <= self.maximum_initial_prior_backoff_trials <= 30
        ):
            raise ValueError(
                "maximum_initial_prior_backoff_trials must be in [0, 30]"
            )
        if (
            not np.isfinite(self.convergence_tolerance)
            or self.convergence_tolerance <= 0.0
            or not np.isfinite(self.minimum_line_search_step)
            or not 0.0 < self.minimum_line_search_step <= 1.0
        ):
            raise ValueError("invalid joint IEnKS configuration")


@dataclass(frozen=True)
class JointBagControlLayout:
    bag_id: str
    initial_and_controller_slice: slice
    actuator_state_slice: slice
    innovation_slice: slice

    @property
    def local_dimension(self) -> int:
        return (
            self.initial_and_controller_slice.stop
            - self.initial_and_controller_slice.start
            + self.actuator_state_slice.stop
            - self.actuator_state_slice.start
            + self.innovation_slice.stop
            - self.innovation_slice.start
        )


@dataclass(frozen=True)
class JointControlLayout:
    shared_parameter_slice: slice
    shared_delay_index: int
    bags: Tuple[JointBagControlLayout, ...]
    dimension: int

    @classmethod
    def from_problems(
        cls, problems: Sequence[Tuple[str, WeakConstraintProblem]]
    ) -> "JointControlLayout":
        parameter_slice = slice(0, PARAMETER_DIMENSION)
        delay_index = PARAMETER_DIMENSION
        offset = SHARED_STATIC_DIMENSION
        layouts = []
        for bag_id, problem in problems:
            initial = slice(offset, offset + PARAMETER_OFFSET)
            offset = initial.stop
            actuator = slice(offset, offset + ACTUATOR_STATE_DIMENSION)
            offset = actuator.stop
            innovation = slice(
                offset, offset + problem.wrench_process.innovation_dimension
            )
            offset = innovation.stop
            layouts.append(
                JointBagControlLayout(
                    bag_id=str(bag_id),
                    initial_and_controller_slice=initial,
                    actuator_state_slice=actuator,
                    innovation_slice=innovation,
                )
            )
        return cls(parameter_slice, delay_index, tuple(layouts), offset)

    def for_bag(self, bag_id: str) -> JointBagControlLayout:
        for layout in self.bags:
            if layout.bag_id == bag_id:
                return layout
        raise KeyError("unknown joint bag: {}".format(bag_id))


@dataclass(frozen=True)
class JointBagProblem:
    bag_id: str
    problem: WeakConstraintProblem
    configuration_fingerprint: str = ""

    def __post_init__(self) -> None:
        identifier = str(self.bag_id)
        if not identifier:
            raise ValueError("joint bag ID cannot be empty")
        if not isinstance(self.problem, WeakConstraintProblem):
            raise TypeError("joint bag problem must be WeakConstraintProblem")
        object.__setattr__(self, "bag_id", identifier)
        object.__setattr__(
            self, "configuration_fingerprint", str(self.configuration_fingerprint)
        )


@dataclass(frozen=True)
class DecodedJointBagControl:
    bag_id: str
    initial_state: RigidBodyState
    initial_controller_state: ControllerState
    initial_actuator_state: ActuatorState
    residual_wrench: np.ndarray


@dataclass(frozen=True)
class DecodedJointControl:
    parameters: VehicleParameters
    constant_delay: float
    bags: Tuple[DecodedJointBagControl, ...]


@dataclass(frozen=True)
class SharedStaticParameterEnsemble:
    member_id: np.ndarray
    coordinates: np.ndarray
    physical_parameter_coordinates: np.ndarray
    constant_delay_coordinate: np.ndarray
    mass: np.ndarray
    inertia: np.ndarray
    cog_offset: np.ndarray
    force_effectiveness: np.ndarray
    torque_effectiveness: np.ndarray
    constant_delay: np.ndarray

    def __post_init__(self) -> None:
        member_id = np.asarray(self.member_id, dtype=np.int64)
        coordinates = np.asarray(self.coordinates, dtype=float)
        physical = np.asarray(
            self.physical_parameter_coordinates, dtype=float
        )
        delay_coordinate = np.asarray(
            self.constant_delay_coordinate, dtype=float
        )
        mass = np.asarray(self.mass, dtype=float)
        inertia = np.asarray(self.inertia, dtype=float)
        cog = np.asarray(self.cog_offset, dtype=float)
        force = np.asarray(self.force_effectiveness, dtype=float)
        torque = np.asarray(self.torque_effectiveness, dtype=float)
        delay = np.asarray(self.constant_delay, dtype=float)
        member_count = member_id.size
        if (
            member_id.ndim != 1
            or np.unique(member_id).size != member_count
            or coordinates.shape != (member_count, SHARED_STATIC_DIMENSION)
            or physical.shape != (member_count, PARAMETER_DIMENSION)
            or delay_coordinate.shape != (member_count,)
            or mass.shape != (member_count,)
            or inertia.shape != (member_count, 3, 3)
            or cog.shape != (member_count, 3)
            or force.shape != (member_count, 4)
            or torque.shape != (member_count, 4)
            or delay.shape != (member_count,)
            or np.any(~np.isfinite(coordinates))
            or np.any(~np.isfinite(physical))
            or np.any(~np.isfinite(delay_coordinate))
            or np.any(~np.isfinite(mass))
            or np.any(~np.isfinite(inertia))
            or np.any(~np.isfinite(cog))
            or np.any(~np.isfinite(force))
            or np.any(~np.isfinite(torque))
            or np.any(~np.isfinite(delay))
            or np.any(delay < 0.0)
        ):
            raise ValueError("shared static parameter ensemble is misaligned")
        for name, value in (
            ("member_id", member_id),
            ("coordinates", coordinates),
            ("physical_parameter_coordinates", physical),
            ("constant_delay_coordinate", delay_coordinate),
            ("mass", mass),
            ("inertia", inertia),
            ("cog_offset", cog),
            ("force_effectiveness", force),
            ("torque_effectiveness", torque),
            ("constant_delay", delay),
        ):
            object.__setattr__(self, name, value.copy())


@dataclass(frozen=True)
class JointBagPosterior:
    bag_id: str
    member_id: np.ndarray
    trajectory_ensemble: Tuple[ClosedLoopTrajectory, ...]
    prior_trajectory_ensemble: Tuple[ClosedLoopTrajectory, ...]
    initial_states: Tuple[RigidBodyState, ...]
    initial_controller_states: Tuple[ControllerState, ...]
    initial_actuator_states: Tuple[ActuatorState, ...]
    innovation_ensemble: np.ndarray
    residual_wrench_ensemble: np.ndarray
    correction_translation: np.ndarray
    correction_rotation_vector: np.ndarray
    objective_contribution: np.ndarray
    pose_component_coverage: float


@dataclass(frozen=True)
class JointWeakConstraintPosterior:
    member_id: np.ndarray
    control_ensemble: np.ndarray
    requested_prior_control_ensemble: np.ndarray
    prior_control_ensemble: np.ndarray
    shared_parameter_ensemble: SharedStaticParameterEnsemble
    ridge: ParameterRidge
    bags: Tuple[JointBagPosterior, ...]
    center_control: np.ndarray
    iterations: Tuple[EnsembleSpaceIteration, ...]
    ensemble_rank: int
    converged: bool
    termination_reason: str
    initial_prior_forecast: InitialPriorForecastDiagnostics

    def bag(self, bag_id: str) -> JointBagPosterior:
        for value in self.bags:
            if value.bag_id == bag_id:
                return value
        raise KeyError("unknown posterior bag: {}".format(bag_id))


class JointWeakConstraintProblem:
    """One likelihood with shared physical parameters and local bag paths."""

    def __init__(
        self,
        bags: Sequence[JointBagProblem],
        delay_chart: Optional[ConstantDelayChart] = None,
        allow_configuration_mismatch: bool = False,
    ):
        ordered = tuple(sorted(tuple(bags), key=lambda value: value.bag_id))
        if not ordered or len({value.bag_id for value in ordered}) != len(ordered):
            raise ValueError("joint problem requires unique non-empty bags")
        fingerprints = {
            value.configuration_fingerprint
            for value in ordered
            if value.configuration_fingerprint
        }
        if len(fingerprints) > 1 and not bool(allow_configuration_mismatch):
            raise ValueError(
                "selected bags have different configuration fingerprints"
            )
        self.bags = ordered
        self.configuration_mismatch_overridden = bool(
            len(fingerprints) > 1 and allow_configuration_mismatch
        )
        self.delay_chart = delay_chart or ConstantDelayChart()
        self.layout = JointControlLayout.from_problems(
            tuple((value.bag_id, value.problem) for value in ordered)
        )
        reference = ordered[0].problem.parameter_chart.decode(
            np.zeros(PARAMETER_DIMENSION)
        )
        for value in ordered[1:]:
            candidate = value.problem.parameter_chart.decode(
                np.zeros(PARAMETER_DIMENSION)
            )
            for name in (
                "mass",
                "inertia",
                "cog_offset",
                "force_effectiveness",
                "torque_effectiveness",
            ):
                if not np.allclose(
                    getattr(reference, name), getattr(candidate, name)
                ):
                    raise ValueError(
                        "joint bags must share one physical parameter chart"
                    )

    @property
    def control_dimension(self) -> int:
        return self.layout.dimension

    def _actuator_anchor(self, bag: JointBagProblem) -> ActuatorState:
        strong = bag.problem.strong_problem
        if strong.initial_actuator_state is not None:
            return strong.initial_actuator_state
        return ActuatorState(
            strong.nominal_trajectory.actuator_thrust[0],
            strong.nominal_trajectory.actuator_gimbal_angle[0],
        )

    def decode_control(self, control: Sequence[float]) -> DecodedJointControl:
        value = np.asarray(control, dtype=float)
        if (
            value.shape != (self.control_dimension,)
            or np.any(~np.isfinite(value))
        ):
            raise ValueError(
                "joint control must contain {} finite values".format(
                    self.control_dimension
                )
            )
        parameter_coordinates = value[self.layout.shared_parameter_slice]
        parameters = self.bags[0].problem.parameter_chart.decode(
            parameter_coordinates
        )
        constant_delay = self.delay_chart.decode(
            value[self.layout.shared_delay_index]
        )
        decoded_bags = []
        for bag in self.bags:
            layout = self.layout.for_bag(bag.bag_id)
            local_static = np.zeros(PARAMETER_OFFSET + PARAMETER_DIMENSION)
            local_static[:PARAMETER_OFFSET] = value[
                layout.initial_and_controller_slice
            ]
            local_static[PARAMETER_OFFSET:] = parameter_coordinates
            state, controller_state, _parameters = (
                bag.problem.strong_problem.decode_control(local_static)
            )
            anchor = self._actuator_anchor(bag)
            actuator_delta = value[layout.actuator_state_slice]
            limits = bag.problem.strong_problem.actuator_parameters
            actuator_state = ActuatorState(
                np.clip(
                    anchor.thrust + actuator_delta[:4],
                    limits.minimum_thrust,
                    limits.maximum_thrust,
                ),
                np.clip(
                    anchor.gimbal_angle + actuator_delta[4:],
                    -limits.maximum_gimbal_angle,
                    limits.maximum_gimbal_angle,
                ),
            )
            residual = bag.problem.wrench_process.decode(
                value[layout.innovation_slice]
            )
            decoded_bags.append(
                DecodedJointBagControl(
                    bag_id=bag.bag_id,
                    initial_state=state,
                    initial_controller_state=controller_state,
                    initial_actuator_state=actuator_state,
                    residual_wrench=residual,
                )
            )
        return DecodedJointControl(
            parameters=parameters,
            constant_delay=constant_delay,
            bags=tuple(decoded_bags),
        )

    def forecast(
        self,
        control: Sequence[float],
        member_index: Optional[int] = None,
        member_bag_callback: Optional[
            Callable[[int, str, int, int], None]
        ] = None,
        cancel_requested: Optional[Callable[[], bool]] = None,
    ) -> Tuple[ClosedLoopTrajectory, ...]:
        decoded = self.decode_control(control)
        trajectories = []
        total = len(self.bags)
        for bag_index, (bag, local) in enumerate(
            zip(self.bags, decoded.bags)
        ):
            if cancel_requested is not None and bool(cancel_requested()):
                raise EstimationCancelled(
                    "estimation cancelled at bag forecast boundary"
                )
            strong = bag.problem.strong_problem
            controller = GrapeController(
                strong.controller_configuration,
                strong.controller_parameters,
                strong.geometry,
                articulated_model=GrapeArticulatedModel(),
            )
            trajectories.append(
                simulate_closed_loop(
                    times=strong.observations.times,
                    references=strong.references,
                    initial_state=local.initial_state,
                    initial_controller_state=local.initial_controller_state,
                    controller=controller,
                    plant=FullSixDofPlant(decoded.parameters, strong.geometry),
                    actuator_parameters=replace(
                        strong.actuator_parameters,
                        delay=decoded.constant_delay,
                    ),
                    initial_actuator_state=local.initial_actuator_state,
                    interval_residual_wrench=local.residual_wrench,
                )
            )
            if member_bag_callback is not None:
                member_bag_callback(
                    -1 if member_index is None else int(member_index),
                    bag.bag_id,
                    bag_index + 1,
                    total,
                )
        return tuple(trajectories)

    def residual(
        self, trajectories: Sequence[ClosedLoopTrajectory]
    ) -> np.ndarray:
        values = tuple(trajectories)
        if len(values) != len(self.bags):
            raise ValueError("joint forecast must contain one path per bag")
        return np.concatenate(
            tuple(
                bag.problem.residual(trajectory)
                for bag, trajectory in zip(self.bags, values)
            )
        )

    def forecast_residual_batch(
        self,
        controls: np.ndarray,
        member_bag_callback: Optional[Callable[[int, str, int, int], None]] = None,
        cancel_requested: Optional[Callable[[], bool]] = None,
    ) -> np.ndarray:
        values = np.asarray(controls, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.control_dimension:
            raise ValueError("joint control batch has the wrong shape")
        total = values.shape[0] * len(self.bags)
        completed = 0
        rows = []
        for member, control in enumerate(values):
            def report(
                member_index: int,
                bag_id: str,
                _local_completed: int,
                _local_total: int,
            ) -> None:
                nonlocal completed
                completed += 1
                if member_bag_callback is not None:
                    member_bag_callback(
                        member_index, bag_id, completed, total
                    )

            member_trajectories = self.forecast(
                control,
                member_index=member,
                member_bag_callback=report,
                cancel_requested=cancel_requested,
            )
            rows.append(self.residual(member_trajectories))
        return np.asarray(rows)


@dataclass(frozen=True)
class JointWeakConstraintPrior:
    mean: np.ndarray
    standard_deviation: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        standard_deviation = np.asarray(self.standard_deviation, dtype=float)
        if (
            mean.ndim != 1
            or standard_deviation.shape != mean.shape
            or np.any(~np.isfinite(mean))
            or np.any(~np.isfinite(standard_deviation))
            or np.any(standard_deviation <= 0.0)
        ):
            raise ValueError("joint prior vectors are invalid")
        object.__setattr__(self, "mean", mean.copy())
        object.__setattr__(
            self, "standard_deviation", standard_deviation.copy()
        )

    @classmethod
    def grape(
        cls,
        problem: JointWeakConstraintProblem,
        delay_mean: float = 0.02,
        delay_standard_deviation: float = 0.015,
    ) -> "JointWeakConstraintPrior":
        selected_delay_mean = float(delay_mean)
        selected_delay_deviation = float(delay_standard_deviation)
        if (
            not np.isfinite(selected_delay_mean)
            or selected_delay_mean < 0.0
            or not np.isfinite(selected_delay_deviation)
            or selected_delay_deviation <= 0.0
        ):
            raise ValueError(
                "delay prior mean/deviation must be non-negative/positive"
            )
        layout = problem.layout
        mean = np.zeros(layout.dimension)
        deviation = np.ones(layout.dimension)
        static = StrongConstraintPrior.grape()
        mean[layout.shared_parameter_slice] = static.mean[PARAMETER_OFFSET:]
        deviation[layout.shared_parameter_slice] = np.sqrt(
            np.diag(static.covariance)[PARAMETER_OFFSET:]
        )
        mean[layout.shared_delay_index] = selected_delay_mean
        deviation[layout.shared_delay_index] = selected_delay_deviation
        for bag in problem.bags:
            local = layout.for_bag(bag.bag_id)
            mean[local.initial_and_controller_slice] = static.mean[
                :PARAMETER_OFFSET
            ]
            deviation[local.initial_and_controller_slice] = np.sqrt(
                np.diag(static.covariance)[:PARAMETER_OFFSET]
            )
            deviation[local.actuator_state_slice] = np.asarray(
                (0.30, 0.30, 0.30, 0.30, 0.03, 0.03, 0.03, 0.03)
            )
            deviation[local.innovation_slice] = 1.0
        return cls(mean, deviation)

    def ensemble(self, size: int, seed: int) -> np.ndarray:
        member_count = int(size)
        if member_count < 3:
            raise ValueError("joint ensemble requires at least three members")
        generator = np.random.RandomState(int(seed))
        standard = generator.normal(
            size=(member_count, self.mean.size)
        )
        standard -= np.mean(standard, axis=0, keepdims=True)
        scale = np.std(standard, axis=0, ddof=1)
        if np.any(scale <= 0.0):
            raise ValueError("joint random ensemble is degenerate")
        standard /= scale[None, :]
        return self.mean[None, :] + standard * self.standard_deviation[None, :]


class JointWeakConstraintIEnKSQ:
    def __init__(self, configuration):
        self.configuration = configuration

    def fit(
        self,
        problem: JointWeakConstraintProblem,
        prior: Optional[JointWeakConstraintPrior] = None,
        progress_callback=None,
        cancel_requested=None,
        member_bag_callback=None,
    ) -> JointWeakConstraintPosterior:
        selected_prior = prior or JointWeakConstraintPrior.grape(problem)
        if progress_callback is not None:
            progress_callback(
                "prior_ensemble_generation",
                0,
                1,
                "generating joint prior ensemble",
            )
        ensemble = selected_prior.ensemble(
            self.configuration.ensemble_size, self.configuration.seed
        )
        if progress_callback is not None:
            progress_callback(
                "prior_ensemble_generation",
                1,
                1,
                "joint prior ensemble generated",
            )
        result = run_ensemble_space_ienks(
            ensemble,
            lambda controls: problem.forecast_residual_batch(
                controls, member_bag_callback, cancel_requested
            ),
            self.configuration,
            progress_callback=progress_callback,
            cancel_requested=cancel_requested,
        )
        if progress_callback is not None:
            progress_callback(
                "posterior_diagnostics",
                0,
                1,
                "computing ridge, coverage, and member paths",
            )
        posterior = self._posterior(
            problem,
            result,
            member_bag_callback=member_bag_callback,
            cancel_requested=cancel_requested,
        )
        if progress_callback is not None:
            progress_callback(
                "posterior_diagnostics",
                1,
                1,
                "posterior diagnostics complete",
            )
        return posterior

    @staticmethod
    def _posterior(
        problem: JointWeakConstraintProblem,
        result: EnsembleSpaceResult,
        member_bag_callback=None,
        cancel_requested=None,
    ) -> JointWeakConstraintPosterior:
        member_count = result.posterior_ensemble.shape[0]
        member_id = np.arange(member_count, dtype=np.int64)
        decoded = tuple(
            problem.decode_control(value)
            for value in result.posterior_ensemble
        )
        parameters = tuple(value.parameters for value in decoded)
        delay_coordinate = result.posterior_ensemble[
            :, problem.layout.shared_delay_index
        ]
        physical_coordinates = result.posterior_ensemble[
            :, problem.layout.shared_parameter_slice
        ]
        shared = SharedStaticParameterEnsemble(
            member_id=member_id,
            coordinates=np.column_stack(
                (physical_coordinates, delay_coordinate)
            ),
            physical_parameter_coordinates=physical_coordinates,
            constant_delay_coordinate=delay_coordinate,
            mass=np.asarray([value.mass for value in parameters]),
            inertia=np.asarray([value.inertia for value in parameters]),
            cog_offset=np.asarray([value.cog_offset for value in parameters]),
            force_effectiveness=np.asarray(
                [value.force_effectiveness for value in parameters]
            ),
            torque_effectiveness=np.asarray(
                [value.torque_effectiveness for value in parameters]
            ),
            constant_delay=np.asarray(
                [value.constant_delay for value in decoded]
            ),
        )
        shared_covariance = np.cov(shared.coordinates, rowvar=False)
        ridge_eigenvalues, ridge_eigenvectors = np.linalg.eigh(
            shared_covariance
        )
        expected_ridge_direction = np.concatenate(
            (
                problem.bags[0].problem.parameter_chart.ridge_direction(),
                np.asarray((0.0,)),
            )
        )
        expected_ridge_direction /= np.linalg.norm(
            expected_ridge_direction
        )
        ridge = ParameterRidge(
            covariance=shared_covariance,
            eigenvalues=ridge_eigenvalues,
            eigenvectors=ridge_eigenvectors,
            expected_direction=expected_ridge_direction,
            expected_variance=float(
                expected_ridge_direction
                @ shared_covariance
                @ expected_ridge_direction
            ),
        )
        total_replay_units = 2 * member_count * len(problem.bags)
        replay_completed = [0]

        def report_replay(member, bag_id, _completed, _total):
            replay_completed[0] += 1
            if member_bag_callback is not None:
                member_bag_callback(
                    member,
                    bag_id,
                    replay_completed[0],
                    total_replay_units,
                )

        posterior_forecasts = tuple(
            problem.forecast(
                value,
                member_index=member,
                member_bag_callback=report_replay,
                cancel_requested=cancel_requested,
            )
            for member, value in enumerate(result.posterior_ensemble)
        )
        prior_forecasts = tuple(
            problem.forecast(
                value,
                member_index=member,
                member_bag_callback=report_replay,
                cancel_requested=cancel_requested,
            )
            for member, value in enumerate(result.prior_ensemble)
        )
        bag_results = []
        for bag_index, bag in enumerate(problem.bags):
            layout = problem.layout.for_bag(bag.bag_id)
            trajectories = tuple(
                value[bag_index] for value in posterior_forecasts
            )
            prior_trajectories = tuple(
                value[bag_index] for value in prior_forecasts
            )
            local_decoded = tuple(value.bags[bag_index] for value in decoded)
            translation = []
            rotation = []
            objective = []
            for trajectory in trajectories:
                delta_translation, delta_rotation = correction_transform_path(
                    bag.problem.nominal_trajectory.position,
                    bag.problem.nominal_trajectory.orientation_xyzw,
                    trajectory.position,
                    trajectory.orientation_xyzw,
                )
                translation.append(delta_translation)
                rotation.append(delta_rotation)
                residual = bag.problem.residual(trajectory)
                objective.append(0.5 * float(np.dot(residual, residual)))
            innovations = result.posterior_ensemble[
                :, layout.innovation_slice
            ]
            pose_error = np.empty(
                (
                    member_count,
                    bag.problem.strong_problem.observations.times.size,
                    6,
                ),
                dtype=float,
            )
            for member, trajectory in enumerate(trajectories):
                pose_error[member, :, :3] = (
                    trajectory.position
                    - bag.problem.strong_problem.observations.position
                )
                for sample in range(
                    bag.problem.strong_problem.observations.times.size
                ):
                    observed_rotation = quaternion_to_matrix(
                        bag.problem.strong_problem.observations
                        .orientation_xyzw[sample]
                    )
                    candidate_rotation = quaternion_to_matrix(
                        trajectory.orientation_xyzw[sample]
                    )
                    pose_error[member, sample, 3:] = (
                        rotation_vector_from_matrix(
                            observed_rotation.T @ candidate_rotation
                        )
                    )
            lower = np.percentile(pose_error, 2.5, axis=0)
            upper = np.percentile(pose_error, 97.5, axis=0)
            coverage = float(np.mean((lower <= 0.0) & (upper >= 0.0)))
            bag_results.append(
                JointBagPosterior(
                    bag_id=bag.bag_id,
                    member_id=member_id.copy(),
                    trajectory_ensemble=trajectories,
                    prior_trajectory_ensemble=prior_trajectories,
                    initial_states=tuple(
                        value.initial_state for value in local_decoded
                    ),
                    initial_controller_states=tuple(
                        value.initial_controller_state for value in local_decoded
                    ),
                    initial_actuator_states=tuple(
                        value.initial_actuator_state for value in local_decoded
                    ),
                    innovation_ensemble=innovations.copy(),
                    residual_wrench_ensemble=np.asarray(
                        [value.residual_wrench for value in local_decoded]
                    ),
                    correction_translation=np.asarray(translation),
                    correction_rotation_vector=np.asarray(rotation),
                    objective_contribution=np.asarray(objective),
                    pose_component_coverage=coverage,
                )
            )
        return JointWeakConstraintPosterior(
            member_id=member_id,
            control_ensemble=result.posterior_ensemble,
            requested_prior_control_ensemble=(
                result.requested_prior_ensemble
            ),
            prior_control_ensemble=result.prior_ensemble,
            shared_parameter_ensemble=shared,
            ridge=ridge,
            bags=tuple(bag_results),
            center_control=result.center_control,
            iterations=result.iterations,
            ensemble_rank=result.ensemble_rank,
            converged=result.converged,
            termination_reason=result.termination_reason,
            initial_prior_forecast=result.initial_prior_forecast,
        )


@dataclass(frozen=True)
class PreparedJointFlight:
    bag_id: str
    episode: RealFlightEpisode
    joint_bag_problem: JointBagProblem
    initial_state_anchor: RigidBodyState
    nominal_trajectory: ClosedLoopTrajectory
    actuator_parameters: ActuatorParameters
    nominal_parameters: VehicleParameters
    calibration: ModelErrorCalibration
    knot_resolution: KnotResolution
    wrench_process: KnotGaussMarkovWrenchProcess


@dataclass(frozen=True)
class JointAssimilationResult:
    prepared_flights: Tuple[PreparedJointFlight, ...]
    problem: JointWeakConstraintProblem
    prior: JointWeakConstraintPrior
    posterior: JointWeakConstraintPosterior

    def prepared(self, bag_id: str) -> PreparedJointFlight:
        for value in self.prepared_flights:
            if value.bag_id == bag_id:
                return value
        raise KeyError("unknown prepared flight: {}".format(bag_id))


def prepare_joint_flight(
    bag_id: str,
    episode: RealFlightEpisode,
    configuration_fingerprint: str,
    maximum_knots: Optional[int] = 12,
    bridge_standard_deviation_fraction: float = 0.5,
    initial_delay: float = 0.02,
) -> PreparedJointFlight:
    """Calibrate one independent bag path before sharing static parameters."""

    if not isinstance(episode, RealFlightEpisode):
        raise TypeError("episode must be a RealFlightEpisode")
    selected_delay = float(initial_delay)
    if not np.isfinite(selected_delay) or selected_delay < 0.0:
        raise ValueError("initial_delay must be finite and non-negative")
    (
        strong,
        initial_state,
        nominal,
        actuator_parameters,
        nominal_parameters,
    ) = build_real_strong_problem(
        episode, actuator_parameters=ActuatorParameters(delay=selected_delay)
    )
    calibration = calibrate_model_error_from_closed_loop_pose(
        episode.observations.times,
        episode.observations.position,
        episode.observations.orientation_xyzw,
        episode.references,
        episode.controller_configuration,
        episode.initial_controller_state,
        episode.initial_actuator_state,
        actuator_parameters,
        nominal_parameters,
        strong.geometry,
    )
    resolution = select_ou_knot_resolution(
        episode.observations.times,
        calibration.correlation_time,
        bridge_standard_deviation_fraction,
        maximum_knots,
    )
    process = KnotGaussMarkovWrenchProcess(
        integration_times=episode.observations.times,
        knot_indices=resolution.knot_indices,
        stationary_standard_deviation=(
            calibration.stationary_standard_deviation
        ),
        correlation_time=calibration.correlation_time,
    )
    weak = WeakConstraintProblem(strong, process)
    joint_bag = JointBagProblem(
        str(bag_id), weak, str(configuration_fingerprint)
    )
    return PreparedJointFlight(
        bag_id=str(bag_id),
        episode=episode,
        joint_bag_problem=joint_bag,
        initial_state_anchor=initial_state,
        nominal_trajectory=nominal,
        actuator_parameters=actuator_parameters,
        nominal_parameters=nominal_parameters,
        calibration=calibration,
        knot_resolution=resolution,
        wrench_process=process,
    )


def assimilate_joint_flights(
    prepared_flights: Sequence[PreparedJointFlight],
    configuration: Optional[JointIEnKSConfig] = None,
    delay_mean: float = 0.02,
    delay_standard_deviation: float = 0.015,
    allow_configuration_mismatch: bool = False,
    progress_callback=None,
    cancel_requested=None,
    member_bag_callback=None,
) -> JointAssimilationResult:
    """Solve one joint problem; no bag-level posterior averaging is performed."""

    prepared = tuple(
        sorted(tuple(prepared_flights), key=lambda value: value.bag_id)
    )
    if not prepared or any(
        not isinstance(value, PreparedJointFlight) for value in prepared
    ):
        raise ValueError("at least one prepared flight is required")
    problem = JointWeakConstraintProblem(
        tuple(value.joint_bag_problem for value in prepared),
        allow_configuration_mismatch=allow_configuration_mismatch,
    )
    prior = JointWeakConstraintPrior.grape(
        problem,
        delay_mean=delay_mean,
        delay_standard_deviation=delay_standard_deviation,
    )
    selected_configuration = configuration or JointIEnKSConfig()
    posterior = JointWeakConstraintIEnKSQ(selected_configuration).fit(
        problem,
        prior,
        progress_callback=progress_callback,
        cancel_requested=cancel_requested,
        member_bag_callback=member_bag_callback,
    )
    return JointAssimilationResult(prepared, problem, prior, posterior)


def assimilation_run_manifest(
    run_id: str,
    request_path: str,
    request_fingerprint: str,
    project_request_fingerprint: str,
    selected_intervals: Mapping[str, Sequence[float]],
    configuration_fingerprint: str,
    member_count: int,
    estimator_revision: str = "unknown",
) -> Dict[str, object]:
    """Build the complete writing manifest before expensive forecasts start."""

    bag_ids = tuple(sorted(str(value) for value in selected_intervals))
    return {
        "schema": "grape-param-estim/assimilation-run/v1",
        "run_id": str(run_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimator_revision": str(estimator_revision),
        "request_path": str(request_path),
        "request_fingerprint": str(request_fingerprint),
        "project_request_fingerprint": str(project_request_fingerprint),
        "selected_bag_ids": list(bag_ids),
        "selected_intervals": {
            bag_id: [
                float(selected_intervals[bag_id][0]),
                float(selected_intervals[bag_id][1]),
            ]
            for bag_id in bag_ids
        },
        "configuration_fingerprint": str(configuration_fingerprint),
        "shared_member_count": int(member_count),
        "termination_reason": "running",
        "converged": False,
        "artifacts": {
            "shared_posterior": "shared_posterior.npz",
            "diagnostics": "diagnostics.npz",
            "bags": {
                bag_id: "bags/{}.npz".format(bag_id)
                for bag_id in bag_ids
            },
        },
    }


def _trajectory_field(trajectories, field: str) -> np.ndarray:
    return np.asarray([getattr(value, field) for value in trajectories])


def initial_prior_forecast_manifest(
    posterior: JointWeakConstraintPosterior,
) -> Dict[str, object]:
    """Return the JSON audit summary matching ``diagnostics.npz``."""

    audit = posterior.initial_prior_forecast
    return {
        "strategy": "global_radial_dyadic_backoff",
        "radial_scale": float(audit.radial_scale),
        "backoff_trials": int(audit.backoff_trials),
        "maximum_backoff_trials": int(audit.maximum_backoff_trials),
        "requested_member_count": int(posterior.member_id.size),
        "effective_member_count": int(posterior.member_id.size),
        "requested_rank": int(audit.requested_rank),
        "effective_rank": int(audit.effective_rank),
        "failed_attempts": [
            {
                "radial_scale": float(value.radial_scale),
                "exception_type": str(value.exception_type),
                "reason": str(value.reason),
            }
            for value in audit.failures
        ],
        "effective_prior_source": (
            "diagnostics.npz:effective_prior_control_ensemble"
        ),
    }


def write_joint_assimilation_payloads(
    root: str, result: JointAssimilationResult
) -> Path:
    """Write pickle-free variable-length bag payloads into a begun bundle."""

    if not isinstance(result, JointAssimilationResult):
        raise TypeError("result must be a JointAssimilationResult")
    destination = Path(root).expanduser().resolve()
    (destination / "bags").mkdir(parents=True, exist_ok=True)
    posterior = result.posterior
    shared = posterior.shared_parameter_ensemble
    np.savez_compressed(
        str(destination / "shared_posterior.npz"),
        member_id=posterior.member_id,
        parameter_coordinates=shared.coordinates,
        physical_parameter_coordinates=shared.physical_parameter_coordinates,
        constant_delay_coordinate=shared.constant_delay_coordinate,
        mass=shared.mass,
        inertia=shared.inertia,
        cog=shared.cog_offset,
        force_effectiveness=shared.force_effectiveness,
        torque_effectiveness=shared.torque_effectiveness,
        constant_delay=shared.constant_delay,
        ridge_covariance=posterior.ridge.covariance,
        ridge_eigenvalues=posterior.ridge.eigenvalues,
        ridge_eigenvectors=posterior.ridge.eigenvectors,
        ridge_expected_direction=posterior.ridge.expected_direction,
        ridge_expected_variance=np.asarray(
            (posterior.ridge.expected_variance,)
        ),
        mode_id=np.asarray(("actuator_wiring_nominal",)),
        mode_weight=np.asarray((1.0,)),
        selected_mode_id=np.asarray(("actuator_wiring_nominal",)),
    )
    iteration = posterior.iterations
    initial_prior = posterior.initial_prior_forecast
    initial_failures = initial_prior.failures
    np.savez_compressed(
        str(destination / "diagnostics.npz"),
        iteration=np.asarray(
            [value.iteration for value in iteration], dtype=np.int64
        ),
        objective=np.asarray([value.objective for value in iteration]),
        accepted_objective=np.asarray(
            [value.accepted_objective for value in iteration]
        ),
        gradient_norm=np.asarray(
            [value.gradient_norm for value in iteration]
        ),
        step_norm=np.asarray([value.step_norm for value in iteration]),
        accepted_fraction=np.asarray(
            [value.accepted_fraction for value in iteration]
        ),
        ensemble_rank=np.asarray((posterior.ensemble_rank,), dtype=np.int64),
        converged=np.asarray((posterior.converged,), dtype=bool),
        termination_reason=np.asarray((posterior.termination_reason,)),
        initial_prior_member_id=posterior.member_id,
        requested_prior_control_ensemble=(
            posterior.requested_prior_control_ensemble
        ),
        effective_prior_control_ensemble=posterior.prior_control_ensemble,
        initial_prior_radial_scale=np.asarray(
            (initial_prior.radial_scale,)
        ),
        initial_prior_backoff_trials=np.asarray(
            (initial_prior.backoff_trials,), dtype=np.int64
        ),
        initial_prior_maximum_backoff_trials=np.asarray(
            (initial_prior.maximum_backoff_trials,), dtype=np.int64
        ),
        initial_prior_requested_rank=np.asarray(
            (initial_prior.requested_rank,), dtype=np.int64
        ),
        initial_prior_effective_rank=np.asarray(
            (initial_prior.effective_rank,), dtype=np.int64
        ),
        initial_prior_failed_scale=np.asarray(
            [value.radial_scale for value in initial_failures], dtype=float
        ),
        initial_prior_failure_type=np.asarray(
            [value.exception_type for value in initial_failures], dtype=str
        ),
        initial_prior_failure_reason=np.asarray(
            [value.reason for value in initial_failures], dtype=str
        ),
    )
    for bag_posterior in posterior.bags:
        prepared = result.prepared(bag_posterior.bag_id)
        episode = prepared.episode
        provenance = episode.provenance
        controller_configuration = episode.controller_configuration
        controller_pid_configuration = np.asarray(
            [
                [getattr(pid, name) for name in PID_CONFIG_FIELD_NAMES]
                for pid in controller_configuration.pid
            ]
        )
        references = episode.references
        observed_translation, observed_rotation = correction_transform_path(
            prepared.nominal_trajectory.position,
            prepared.nominal_trajectory.orientation_xyzw,
            episode.observations.position,
            episode.observations.orientation_xyzw,
        )
        np.savez_compressed(
            str(destination / "bags" / "{}.npz".format(bag_posterior.bag_id)),
            member_id=bag_posterior.member_id,
            times=episode.observations.times,
            record_times=episode.record_times,
            observed_position=episode.observations.position,
            observed_orientation_xyzw=episode.observations.orientation_xyzw,
            observation_translation_covariance=(
                episode.observations.translation_covariance
            ),
            observation_rotation_covariance=(
                episode.observations.rotation_covariance
            ),
            reference_position=np.asarray(
                [value.position for value in references]
            ),
            reference_linear_velocity=np.asarray(
                [value.linear_velocity for value in references]
            ),
            reference_linear_acceleration=np.asarray(
                [value.linear_acceleration for value in references]
            ),
            reference_rpy=np.asarray([value.rpy for value in references]),
            reference_angular_velocity=np.asarray(
                [value.angular_velocity for value in references]
            ),
            reference_angular_acceleration=np.asarray(
                [value.angular_acceleration for value in references]
            ),
            nominal_position=prepared.nominal_trajectory.position,
            nominal_orientation_xyzw=(
                prepared.nominal_trajectory.orientation_xyzw
            ),
            nominal_linear_velocity=(
                prepared.nominal_trajectory.linear_velocity
            ),
            nominal_angular_velocity=(
                prepared.nominal_trajectory.angular_velocity
            ),
            nominal_controller_integral=(
                prepared.nominal_trajectory.controller_integral
            ),
            nominal_commanded_thrust=(
                prepared.nominal_trajectory.commanded_thrust
            ),
            nominal_commanded_gimbal_angle=(
                prepared.nominal_trajectory.commanded_gimbal_angle
            ),
            nominal_actuator_thrust=(
                prepared.nominal_trajectory.actuator_thrust
            ),
            nominal_actuator_gimbal_angle=(
                prepared.nominal_trajectory.actuator_gimbal_angle
            ),
            nominal_body_wrench=prepared.nominal_trajectory.body_wrench,
            posterior_position=_trajectory_field(
                bag_posterior.trajectory_ensemble, "position"
            ),
            posterior_orientation_xyzw=_trajectory_field(
                bag_posterior.trajectory_ensemble, "orientation_xyzw"
            ),
            posterior_linear_velocity=_trajectory_field(
                bag_posterior.trajectory_ensemble, "linear_velocity"
            ),
            posterior_angular_velocity=_trajectory_field(
                bag_posterior.trajectory_ensemble, "angular_velocity"
            ),
            posterior_controller_integral=_trajectory_field(
                bag_posterior.trajectory_ensemble, "controller_integral"
            ),
            posterior_commanded_thrust=_trajectory_field(
                bag_posterior.trajectory_ensemble, "commanded_thrust"
            ),
            posterior_commanded_gimbal_angle=_trajectory_field(
                bag_posterior.trajectory_ensemble,
                "commanded_gimbal_angle",
            ),
            posterior_actuator_thrust=_trajectory_field(
                bag_posterior.trajectory_ensemble, "actuator_thrust"
            ),
            posterior_actuator_gimbal_angle=_trajectory_field(
                bag_posterior.trajectory_ensemble,
                "actuator_gimbal_angle",
            ),
            posterior_body_wrench=_trajectory_field(
                bag_posterior.trajectory_ensemble, "body_wrench"
            ),
            correction_translation=bag_posterior.correction_translation,
            correction_rotation_vector=(
                bag_posterior.correction_rotation_vector
            ),
            observed_correction_translation=observed_translation,
            observed_correction_rotation_vector=observed_rotation,
            residual_wrench_interval=(
                bag_posterior.residual_wrench_ensemble
            ),
            residual_wrench_knot=np.asarray(
                [
                    prepared.wrench_process.decode_knots(value)
                    for value in bag_posterior.innovation_ensemble
                ]
            ),
            innovation_ensemble=bag_posterior.innovation_ensemble,
            initial_position=np.asarray(
                [value.position for value in bag_posterior.initial_states]
            ),
            initial_orientation_xyzw=np.asarray(
                [
                    value.orientation_xyzw
                    for value in bag_posterior.initial_states
                ]
            ),
            initial_linear_velocity=np.asarray(
                [
                    value.linear_velocity
                    for value in bag_posterior.initial_states
                ]
            ),
            initial_angular_velocity=np.asarray(
                [
                    value.angular_velocity
                    for value in bag_posterior.initial_states
                ]
            ),
            initial_controller_integral=np.asarray(
                [
                    value.integral_error
                    for value in bag_posterior.initial_controller_states
                ]
            ),
            initial_controller_roll_pitch_integration_active=np.asarray(
                [
                    value.roll_pitch_integration_active
                    for value in bag_posterior.initial_controller_states
                ],
                dtype=bool,
            ),
            initial_actuator_thrust=np.asarray(
                [value.thrust for value in bag_posterior.initial_actuator_states]
            ),
            initial_actuator_gimbal_angle=np.asarray(
                [
                    value.gimbal_angle
                    for value in bag_posterior.initial_actuator_states
                ]
            ),
            actuator_thrust_time_constant=np.asarray(
                (prepared.actuator_parameters.thrust_time_constant,)
            ),
            actuator_gimbal_time_constant=np.asarray(
                (prepared.actuator_parameters.gimbal_time_constant,)
            ),
            actuator_minimum_thrust=np.asarray(
                (prepared.actuator_parameters.minimum_thrust,)
            ),
            actuator_maximum_thrust=np.asarray(
                (prepared.actuator_parameters.maximum_thrust,)
            ),
            actuator_maximum_gimbal_angle=np.asarray(
                (prepared.actuator_parameters.maximum_gimbal_angle,)
            ),
            actuator_maximum_gimbal_rate=np.asarray(
                (prepared.actuator_parameters.maximum_gimbal_rate,)
            ),
            objective_contribution=bag_posterior.objective_contribution,
            pose_component_coverage=np.asarray(
                (bag_posterior.pose_component_coverage,)
            ),
            q_resolution_sufficient=np.asarray(
                (prepared.knot_resolution.resolution_sufficient,), dtype=bool
            ),
            q_knot_indices=prepared.wrench_process.knot_indices,
            q_knot_times=prepared.wrench_process.knot_times,
            q_stationary_standard_deviation=(
                prepared.calibration.stationary_standard_deviation
            ),
            q_correlation_time=np.asarray(
                (prepared.calibration.correlation_time,)
            ),
            controller_snapshot_groups=np.asarray(
                episode.controller_snapshot.groups
            ),
            controller_snapshot_record_times=(
                episode.controller_snapshot.record_times
            ),
            controller_snapshot_gains=episode.controller_snapshot.gains,
            controller_snapshot_pid_control_flags=(
                episode.controller_snapshot.pid_control_flags
            ),
            controller_snapshot_source_kinds=np.asarray(
                episode.controller_snapshot.source_kinds
            ),
            controller_pid_axis_names=np.asarray(PID_AXIS_NAMES),
            controller_pid_field_names=np.asarray(PID_CONFIG_FIELD_NAMES),
            controller_pid_configuration=controller_pid_configuration,
            controller_xy_control_mode=np.asarray(
                (controller_configuration.xy_control_mode,)
            ),
            controller_need_yaw_d_control=np.asarray(
                (controller_configuration.need_yaw_d_control,), dtype=bool
            ),
            controller_start_roll_pitch_integration_height=np.asarray(
                (
                    controller_configuration
                    .start_roll_pitch_integration_height,
                )
            ),
            controller_initial_height=np.asarray(
                (controller_configuration.initial_height,)
            ),
            controller_source_compatible_gyro_term=np.asarray(
                (controller_configuration.source_compatible_gyro_term,),
                dtype=bool,
            ),
            provenance_bag_path=np.asarray((provenance.bag_path,)),
            provenance_bag_sha256=np.asarray((provenance.bag_sha256,)),
            provenance_bag_size_bytes=np.asarray(
                (provenance.bag_size_bytes,), dtype=np.int64
            ),
            provenance_time_basis=np.asarray((provenance.time_basis,)),
            provenance_requested_window=np.asarray(
                (
                    provenance.requested_window_start,
                    provenance.requested_window_end,
                )
            ),
            provenance_source_available_window=np.asarray(
                (
                    provenance.source_available_start,
                    provenance.source_available_end,
                )
            ),
            provenance_selected_flight_state=np.asarray(
                (provenance.selected_flight_state,), dtype=np.int64
            ),
            provenance_topic_names=np.asarray(
                provenance.topic_names, dtype=str
            ),
            provenance_topic_types=np.asarray(
                provenance.topic_types, dtype=str
            ),
        )
    return destination


__all__ = [
    "ACTUATOR_STATE_DIMENSION",
    "DecodedJointBagControl",
    "DecodedJointControl",
    "JointBagControlLayout",
    "JointBagPosterior",
    "JointBagProblem",
    "JointControlLayout",
    "JointIEnKSConfig",
    "JointAssimilationResult",
    "JointWeakConstraintIEnKSQ",
    "JointWeakConstraintPosterior",
    "JointWeakConstraintPrior",
    "JointWeakConstraintProblem",
    "PreparedJointFlight",
    "SHARED_STATIC_DIMENSION",
    "SharedStaticParameterEnsemble",
    "assimilate_joint_flights",
    "assimilation_run_manifest",
    "initial_prior_forecast_manifest",
    "prepare_joint_flight",
    "write_joint_assimilation_payloads",
]
