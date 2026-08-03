"""Experiment D: separate assimilation and conditioning of plant modes.

The discrete mode in this module is deliberately restricted to the plant.
The controller always uses the audited nominal actuator-channel convention,
while the plant uses either that convention or a 0/1 wiring permutation.
Consequently one ensemble never contains members from two modes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.controller import (
    ControllerConfig,
    GrapeController,
    initial_controller_state,
)
from grape_param_estim.dynamics import FullSixDofPlant, simulate_closed_loop
from grape_param_estim.geometry import (
    correction_transform_path,
    euler_xyz_to_matrix,
    matrix_to_quaternion,
)
from grape_param_estim.parameterization import VehicleParameterChart
from grape_param_estim.strong_constraint import (
    CONTROL_DIMENSION,
    IEnKSConfig,
    StrongConstraintIEnKS,
    StrongConstraintPosterior,
    StrongConstraintPrior,
    StrongConstraintProblem,
)
from grape_param_estim.synthetic import (
    SyntheticExperiment,
    generate_pose_observations,
)
from grape_param_estim.system import (
    ActuatorParameters,
    GrapeGeometry,
    PoseObservations,
    ReferenceState,
    RigidBodyState,
    VehicleParameters,
)


NOMINAL_MODE_ID = "actuator_wiring_nominal"
SWAPPED_MODE_ID = "actuator_wiring_swap_0_1"


@dataclass(frozen=True)
class PlantWiringMode:
    """A discrete mapping from actuator command channel to physical rotor."""

    mode_id: str
    channel_to_rotor: Tuple[int, int, int, int]

    def __post_init__(self) -> None:
        identifier = str(self.mode_id)
        if not identifier:
            raise ValueError("mode_id cannot be empty")
        permutation = tuple(int(value) for value in self.channel_to_rotor)
        if tuple(sorted(permutation)) != (0, 1, 2, 3):
            raise ValueError("channel_to_rotor must be a permutation of 0..3")
        object.__setattr__(self, "mode_id", identifier)
        object.__setattr__(self, "channel_to_rotor", permutation)

    def plant_geometry(
        self, controller_geometry: GrapeGeometry
    ) -> GrapeGeometry:
        """Apply wiring only to the plant-side actuator geometry."""

        order = np.asarray(self.channel_to_rotor, dtype=int)
        return GrapeGeometry(
            rotor_origins=controller_geometry.rotor_origins[order],
            arm_yaws=controller_geometry.arm_yaws[order],
            rotor_directions=controller_geometry.rotor_directions[order],
            moment_force_rate=controller_geometry.moment_force_rate,
            thrust_offset=controller_geometry.thrust_offset,
        )


PLANT_WIRING_MODES = (
    PlantWiringMode(NOMINAL_MODE_ID, (0, 1, 2, 3)),
    PlantWiringMode(SWAPPED_MODE_ID, (1, 0, 2, 3)),
)


def plant_wiring_mode(mode_id: str) -> PlantWiringMode:
    """Return one of the two registered Experiment-D modes."""

    identifier = str(mode_id)
    for mode in PLANT_WIRING_MODES:
        if mode.mode_id == identifier:
            return mode
    raise ValueError("unknown plant-wiring mode: {}".format(identifier))


@dataclass(frozen=True)
class ModeStrongConstraintProblem(StrongConstraintProblem):
    """A strong-constraint problem whose mode affects only the plant."""

    # Python 3.8 dataclass inheritance requires a syntactic default here
    # because the base problem ends in an optional actuator snapshot.  None is
    # rejected so construction sites must still select the mode explicitly.
    plant_mode: Optional[PlantWiringMode] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.plant_mode, PlantWiringMode):
            raise ValueError("plant_mode must be selected explicitly")

    def forecast(self, control: Sequence[float]):
        initial_state, controller_state, parameters = self.decode_control(
            control
        )
        # ``self.geometry`` remains the nominal/audited controller geometry.
        # Only the FullSixDofPlant receives the mode-dependent geometry.
        controller = GrapeController(
            self.controller_configuration,
            self.controller_parameters,
            self.geometry,
            articulated_model=GrapeArticulatedModel(),
        )
        plant = FullSixDofPlant(
            parameters,
            self.plant_mode.plant_geometry(self.geometry),
        )
        return simulate_closed_loop(
            times=self.observations.times,
            references=self.references,
            initial_state=initial_state,
            initial_controller_state=controller_state,
            controller=controller,
            plant=plant,
            actuator_parameters=self.actuator_parameters,
            initial_actuator_state=self.initial_actuator_state,
        )


@dataclass(frozen=True)
class LaplacePoseEvidence:
    """Relative pose evidence reconstructed around one mode's MAP cloud."""

    map_objective: float
    log_hessian_determinant: float
    relative_log_evidence: float
    regression_rank: int


@dataclass(frozen=True)
class ModePosterior:
    """One independent smoother result with stable member identities."""

    mode: PlantWiringMode
    problem: ModeStrongConstraintProblem
    posterior: StrongConstraintPosterior
    evidence: LaplacePoseEvidence
    member_ids: np.ndarray

    def __post_init__(self) -> None:
        identifiers = np.asarray(self.member_ids, dtype=np.int64)
        expected = np.arange(
            self.posterior.control_ensemble.shape[0], dtype=np.int64
        )
        if not np.array_equal(identifiers, expected):
            raise ValueError("member_ids must be the stable zero-based order")
        object.__setattr__(self, "member_ids", identifiers.copy())


@dataclass(frozen=True)
class ModeValidationResult:
    """Pose-only mixture of separately assimilated discrete modes."""

    synthetic: SyntheticExperiment
    truth_mode_id: str
    mode_posteriors: Tuple[ModePosterior, ...]
    prior_mode_probabilities: np.ndarray
    pose_mode_probabilities: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.mode_posteriors)
        identifiers = [value.mode.mode_id for value in self.mode_posteriors]
        if count < 2 or len(set(identifiers)) != count:
            raise ValueError("mode posterior identifiers must be unique")
        for field_name in (
            "prior_mode_probabilities",
            "pose_mode_probabilities",
        ):
            value = np.asarray(getattr(self, field_name), dtype=float)
            if (
                value.shape != (count,)
                or np.any(~np.isfinite(value))
                or np.any(value <= 0.0)
                or not np.isclose(np.sum(value), 1.0, atol=1.0e-12)
            ):
                raise ValueError("{} must be positive probabilities".format(
                    field_name
                ))
            object.__setattr__(self, field_name, value.copy())
        if str(self.truth_mode_id) not in identifiers:
            raise ValueError("truth mode must be among assimilated modes")

    @property
    def mode_ids(self) -> Tuple[str, ...]:
        return tuple(value.mode.mode_id for value in self.mode_posteriors)

    def for_mode(self, mode_id: str) -> ModePosterior:
        identifier = str(mode_id)
        for value in self.mode_posteriors:
            if value.mode.mode_id == identifier:
                return value
        raise KeyError(identifier)


@dataclass(frozen=True)
class ActuatorWiringMeasurement:
    """An independent channel-to-rotor inspection outside pose fitting."""

    channel_to_rotor: np.ndarray
    correctness_probability: float = 0.995

    def __post_init__(self) -> None:
        permutation = np.asarray(self.channel_to_rotor, dtype=np.int64)
        probability = float(self.correctness_probability)
        if (
            permutation.shape != (4,)
            or tuple(sorted(permutation.tolist())) != (0, 1, 2, 3)
        ):
            raise ValueError("channel_to_rotor must be a permutation of 0..3")
        if not np.isfinite(probability) or not 0.5 < probability < 1.0:
            raise ValueError(
                "correctness_probability must be strictly between 0.5 and 1"
            )
        object.__setattr__(self, "channel_to_rotor", permutation.copy())
        object.__setattr__(self, "correctness_probability", probability)


@dataclass(frozen=True)
class ModeConditioningResult:
    """Weights after an independent measurement; raw members are shared."""

    pose_result: ModeValidationResult
    measurement: ActuatorWiringMeasurement
    measurement_log_likelihood: np.ndarray
    conditioned_mode_probabilities: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.pose_result.mode_posteriors)
        likelihood = np.asarray(self.measurement_log_likelihood, dtype=float)
        probabilities = np.asarray(
            self.conditioned_mode_probabilities, dtype=float
        )
        if likelihood.shape != (count,) or np.any(~np.isfinite(likelihood)):
            raise ValueError("measurement log likelihood has wrong shape")
        if (
            probabilities.shape != (count,)
            or np.any(~np.isfinite(probabilities))
            or np.any(probabilities <= 0.0)
            or not np.isclose(np.sum(probabilities), 1.0, atol=1.0e-12)
        ):
            raise ValueError("conditioned mode probabilities are invalid")
        object.__setattr__(
            self, "measurement_log_likelihood", likelihood.copy()
        )
        object.__setattr__(
            self, "conditioned_mode_probabilities", probabilities.copy()
        )

    @property
    def selected_mode_id(self) -> str:
        index = int(np.argmax(self.conditioned_mode_probabilities))
        return self.pose_result.mode_posteriors[index].mode.mode_id

    @property
    def selected_posterior(self) -> ModePosterior:
        """Return the original raw posterior for the selected mode."""

        return self.pose_result.for_mode(self.selected_mode_id)


def _normalise_log_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    maximum = float(np.max(values))
    weights = np.maximum(
        np.exp(values - maximum), np.finfo(float).tiny
    )
    return weights / np.sum(weights)


def _mode_reference(times: np.ndarray) -> Tuple[ReferenceState, ...]:
    """A yaw-rich hover that makes actuator wiring observable in pose."""

    amplitude = 0.52
    frequency = 2.4
    yaw = amplitude * np.sin(frequency * times)
    yaw_rate = amplitude * frequency * np.cos(frequency * times)
    yaw_acceleration = -amplitude * frequency**2 * np.sin(
        frequency * times
    )
    zero = np.zeros((times.size, 3), dtype=float)
    position = zero.copy()
    position[:, 2] = 1.0
    rpy = zero.copy()
    rpy[:, 2] = yaw
    omega = zero.copy()
    omega[:, 2] = yaw_rate
    alpha = zero.copy()
    alpha[:, 2] = yaw_acceleration
    return tuple(
        ReferenceState(
            position=position[index],
            linear_velocity=zero[index],
            linear_acceleration=zero[index],
            rpy=rpy[index],
            angular_velocity=omega[index],
            angular_acceleration=alpha[index],
        )
        for index in range(times.size)
    )


def _initial_state(references: Tuple[ReferenceState, ...]) -> RigidBodyState:
    first = references[0]
    return RigidBodyState(
        position=first.position,
        orientation_xyzw=matrix_to_quaternion(
            euler_xyz_to_matrix(first.rpy)
        ),
        linear_velocity=first.linear_velocity,
        angular_velocity=first.angular_velocity,
    )


def generate_mode_synthetic_experiment(
    truth_mode_id: str = SWAPPED_MODE_ID,
    duration: float = 0.8,
    time_step: float = 0.04,
    translation_standard_deviation: float = 0.002,
    rotation_standard_deviation: float = np.deg2rad(0.12),
    realize_observation_noise: bool = False,
    seed: int = 71,
) -> SyntheticExperiment:
    """Generate perfect-model data except for the selected discrete mode."""

    duration = float(duration)
    time_step = float(time_step)
    translation_sigma = float(translation_standard_deviation)
    rotation_sigma = float(rotation_standard_deviation)
    if duration <= 0.0 or time_step <= 0.0:
        raise ValueError("duration and time_step must be positive")
    if translation_sigma <= 0.0 or rotation_sigma <= 0.0:
        raise ValueError("observation standard deviations must be positive")
    sample_count = int(np.floor(duration / time_step + 0.5)) + 1
    times = np.linspace(0.0, duration, sample_count)
    references = _mode_reference(times)
    geometry = GrapeGeometry.grape()
    parameters = VehicleParameters.nominal()
    configuration = ControllerConfig.grape()
    controller_state = initial_controller_state(configuration, trim_hover=True)
    initial_state = _initial_state(references)
    actuator_parameters = ActuatorParameters()

    def simulate(plant_geometry):
        return simulate_closed_loop(
            times=times,
            references=references,
            initial_state=initial_state,
            initial_controller_state=controller_state,
            controller=GrapeController(
                configuration,
                parameters,
                geometry,
                articulated_model=GrapeArticulatedModel(),
            ),
            plant=FullSixDofPlant(parameters, plant_geometry),
            actuator_parameters=actuator_parameters,
        )

    nominal = simulate(geometry)
    truth_mode = plant_wiring_mode(truth_mode_id)
    truth = simulate(truth_mode.plant_geometry(geometry))
    if realize_observation_noise:
        observations = generate_pose_observations(
            truth,
            translation_standard_deviation=translation_sigma,
            rotation_standard_deviation=rotation_sigma,
            seed=seed,
        )
    else:
        observations = PoseObservations(
            times=times,
            position=truth.position,
            orientation_xyzw=truth.orientation_xyzw,
            translation_covariance=np.eye(3) * translation_sigma**2,
            rotation_covariance=np.eye(3) * rotation_sigma**2,
        )
    correction_translation, correction_rotation = correction_transform_path(
        nominal.position,
        nominal.orientation_xyzw,
        truth.position,
        truth.orientation_xyzw,
    )
    return SyntheticExperiment(
        references=references,
        nominal=nominal,
        truth=truth,
        observations=observations,
        correction_translation=correction_translation,
        correction_rotation_vector=correction_rotation,
        controller_parameters=parameters,
        truth_parameters=parameters,
        nominal_actuator_parameters=actuator_parameters,
        truth_actuator_parameters=actuator_parameters,
    )


def _problem_for_mode(
    experiment: SyntheticExperiment, mode: PlantWiringMode
) -> ModeStrongConstraintProblem:
    configuration = ControllerConfig.grape()
    geometry = GrapeGeometry.grape()
    return ModeStrongConstraintProblem(
        references=experiment.references,
        observations=experiment.observations,
        nominal_trajectory=experiment.nominal,
        initial_state_anchor=_initial_state(experiment.references),
        initial_controller_anchor=initial_controller_state(
            configuration, trim_hover=True
        ),
        controller_configuration=configuration,
        controller_parameters=experiment.controller_parameters,
        geometry=geometry,
        actuator_parameters=experiment.nominal_actuator_parameters,
        parameter_chart=VehicleParameterChart(
            experiment.controller_parameters
        ),
        plant_mode=mode,
    )


def laplace_pose_evidence(
    problem: ModeStrongConstraintProblem,
    prior: StrongConstraintPrior,
    posterior: StrongConstraintPosterior,
) -> LaplacePoseEvidence:
    """Reconstruct a relative Laplace evidence from the final member cloud.

    The omitted likelihood normalisation is identical for every mode because
    all modes see the same pose samples and covariance.  The regression uses
    the member cloud as a black-box Jacobian approximation; it does not alter
    or re-run the existing IEnKS solver.
    """

    factor = np.linalg.cholesky(prior.covariance)
    standardized = np.linalg.solve(
        factor, (posterior.control_ensemble - prior.mean).T
    )
    center = np.linalg.solve(
        factor, posterior.center_control - prior.mean
    )
    coordinate_deviations = standardized - np.mean(
        standardized, axis=1, keepdims=True
    )
    residuals = np.column_stack(
        [problem.residual(value) for value in posterior.trajectory_ensemble]
    )
    residual_deviations = residuals - np.mean(
        residuals, axis=1, keepdims=True
    )
    # Solve Z.T @ J.T = R.T.  This is the same ensemble-cloud regression
    # contract as the smoother, expressed in proper-prior coordinates.
    transpose, _residual, rank, _singular = np.linalg.lstsq(
        coordinate_deviations.T,
        residual_deviations.T,
        rcond=1.0e-12,
    )
    sensitivity = transpose.T
    hessian = np.eye(CONTROL_DIMENSION) + sensitivity.T @ sensitivity
    sign, log_determinant = np.linalg.slogdet(hessian)
    if sign <= 0.0 or not np.isfinite(log_determinant):
        raise ValueError("Laplace pose Hessian must be positive definite")
    center_residual = problem.residual(posterior.center_trajectory)
    objective = 0.5 * float(
        np.dot(center, center) + np.dot(center_residual, center_residual)
    )
    return LaplacePoseEvidence(
        map_objective=objective,
        log_hessian_determinant=float(log_determinant),
        relative_log_evidence=float(objective * -1.0 - 0.5 * log_determinant),
        regression_rank=int(rank),
    )


def assimilate_plant_wiring_modes(
    experiment: SyntheticExperiment,
    mode_order: Sequence[str] = (NOMINAL_MODE_ID, SWAPPED_MODE_ID),
    ensemble_size: int = 40,
    maximum_iterations: int = 2,
    seed: int = 73,
    prior_mode_probabilities: Optional[Mapping[str, float]] = None,
    truth_mode_id: str = SWAPPED_MODE_ID,
) -> ModeValidationResult:
    """Run a completely independent StrongConstraintIEnKS for every mode."""

    modes = tuple(plant_wiring_mode(value) for value in mode_order)
    identifiers = tuple(value.mode_id for value in modes)
    if len(modes) != 2 or set(identifiers) != {
        NOMINAL_MODE_ID,
        SWAPPED_MODE_ID,
    }:
        raise ValueError("mode_order must contain each registered mode once")
    prior = StrongConstraintPrior.grape()
    selected_ensemble_size = int(ensemble_size)
    configuration = IEnKSConfig(
        ensemble_size=selected_ensemble_size,
        maximum_iterations=int(maximum_iterations),
        seed=int(seed),
    )
    results = []
    for mode in modes:
        problem = _problem_for_mode(experiment, mode)
        posterior = StrongConstraintIEnKS(configuration).fit(problem, prior)
        evidence = laplace_pose_evidence(problem, prior, posterior)
        results.append(
            ModePosterior(
                mode=mode,
                problem=problem,
                posterior=posterior,
                evidence=evidence,
                member_ids=np.arange(
                    selected_ensemble_size, dtype=np.int64
                ),
            )
        )

    if prior_mode_probabilities is None:
        prior_probabilities = np.full(len(modes), 1.0 / len(modes))
    else:
        prior_probabilities = np.asarray(
            [prior_mode_probabilities[value.mode_id] for value in modes],
            dtype=float,
        )
        if (
            np.any(~np.isfinite(prior_probabilities))
            or np.any(prior_probabilities <= 0.0)
        ):
            raise ValueError("mode priors must be finite and positive")
        prior_probabilities /= np.sum(prior_probabilities)
    log_probability = np.asarray(
        [value.evidence.relative_log_evidence for value in results]
    ) + np.log(prior_probabilities)
    return ModeValidationResult(
        synthetic=experiment,
        truth_mode_id=str(truth_mode_id),
        mode_posteriors=tuple(results),
        prior_mode_probabilities=prior_probabilities,
        pose_mode_probabilities=_normalise_log_probabilities(log_probability),
    )


def run_mode_validation_experiment(
    truth_mode_id: str = SWAPPED_MODE_ID,
    mode_order: Sequence[str] = (NOMINAL_MODE_ID, SWAPPED_MODE_ID),
    duration: float = 0.8,
    time_step: float = 0.04,
    ensemble_size: int = 40,
    maximum_iterations: int = 2,
    seed: int = 73,
) -> ModeValidationResult:
    """Generate the Experiment-D truth and perform pose-only mode fitting."""

    synthetic = generate_mode_synthetic_experiment(
        truth_mode_id=truth_mode_id,
        duration=duration,
        time_step=time_step,
        seed=seed + 1000,
    )
    return assimilate_plant_wiring_modes(
        synthetic,
        mode_order=mode_order,
        ensemble_size=ensemble_size,
        maximum_iterations=maximum_iterations,
        seed=seed,
        truth_mode_id=truth_mode_id,
    )


def condition_on_actuator_wiring(
    result: ModeValidationResult,
    measurement: ActuatorWiringMeasurement,
) -> ModeConditioningResult:
    """Condition only mixture weights on an independent wiring inspection.

    No posterior member array is copied into, resampled, sorted, or mutated.
    The returned selected posterior is the exact object produced by its
    mode-specific smoother.
    """

    probability = measurement.correctness_probability
    log_likelihood = []
    for value in result.mode_posteriors:
        expected = np.asarray(value.mode.channel_to_rotor, dtype=np.int64)
        matches = measurement.channel_to_rotor == expected
        log_likelihood.append(
            float(
                np.sum(
                    np.where(
                        matches,
                        np.log(probability),
                        np.log(1.0 - probability),
                    )
                )
            )
        )
    log_likelihood = np.asarray(log_likelihood)
    conditioned = _normalise_log_probabilities(
        np.log(result.pose_mode_probabilities) + log_likelihood
    )
    return ModeConditioningResult(
        pose_result=result,
        measurement=measurement,
        measurement_log_likelihood=log_likelihood,
        conditioned_mode_probabilities=conditioned,
    )


def save_mode_validation(
    path: str,
    result: ModeValidationResult,
    conditioning: Optional[ModeConditioningResult] = None,
) -> Path:
    """Save member-aligned mode posteriors in a pickle-free NPZ schema."""

    if conditioning is not None and conditioning.pose_result is not result:
        raise ValueError("conditioning must refer to the saved pose result")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = result.mode_posteriors
    mode_ids = np.asarray([value.mode.mode_id for value in values])
    member_count = values[0].posterior.control_ensemble.shape[0]
    if any(
        value.posterior.control_ensemble.shape[0] != member_count
        for value in values
    ):
        raise ValueError("all mode ensembles must have the same member count")
    conditioned_probability = (
        result.pose_mode_probabilities
        if conditioning is None
        else conditioning.conditioned_mode_probabilities
    )
    measurement_permutation = (
        np.empty((0,), dtype=np.int64)
        if conditioning is None
        else conditioning.measurement.channel_to_rotor
    )
    measurement_correctness = (
        np.empty((0,), dtype=float)
        if conditioning is None
        else np.asarray((conditioning.measurement.correctness_probability,))
    )
    measurement_log_likelihood = (
        np.empty((0,), dtype=float)
        if conditioning is None
        else conditioning.measurement_log_likelihood
    )
    np.savez_compressed(
        str(destination),
        schema=np.asarray(("grape-param-estim/mode-validation/v1",)),
        truth_mode_id=np.asarray((result.truth_mode_id,)),
        mode_id=mode_ids,
        mode_channel_to_rotor=np.asarray(
            [value.mode.channel_to_rotor for value in values],
            dtype=np.int64,
        ),
        prior_mode_probability=result.prior_mode_probabilities,
        pose_mode_probability=result.pose_mode_probabilities,
        conditioned_mode_probability=conditioned_probability,
        pose_relative_log_evidence=np.asarray(
            [value.evidence.relative_log_evidence for value in values]
        ),
        pose_map_objective=np.asarray(
            [value.evidence.map_objective for value in values]
        ),
        pose_log_hessian_determinant=np.asarray(
            [value.evidence.log_hessian_determinant for value in values]
        ),
        pose_regression_rank=np.asarray(
            [value.evidence.regression_rank for value in values],
            dtype=np.int64,
        ),
        actuator_wiring_measurement=measurement_permutation,
        actuator_wiring_correctness_probability=measurement_correctness,
        actuator_wiring_log_likelihood=measurement_log_likelihood,
        selected_mode_id=np.asarray(
            (
                result.mode_ids[
                    int(np.argmax(conditioned_probability))
                ],
            )
        ),
        member_mode_id=np.repeat(mode_ids, member_count),
        member_id=np.tile(
            np.arange(member_count, dtype=np.int64), len(values)
        ),
        prior_control_ensemble=np.stack(
            [value.posterior.prior_control_ensemble for value in values]
        ),
        posterior_control_ensemble=np.stack(
            [value.posterior.control_ensemble for value in values]
        ),
        posterior_parameter_coordinates=np.stack(
            [
                value.posterior.parameter_ensemble.coordinates
                for value in values
            ]
        ),
        posterior_position=np.stack(
            [
                np.asarray(
                    [
                        item.position
                        for item in value.posterior.trajectory_ensemble
                    ]
                )
                for value in values
            ]
        ),
        posterior_orientation_xyzw=np.stack(
            [
                np.asarray(
                    [
                        item.orientation_xyzw
                        for item in value.posterior.trajectory_ensemble
                    ]
                )
                for value in values
            ]
        ),
        posterior_linear_velocity=np.stack(
            [
                np.asarray(
                    [
                        item.linear_velocity
                        for item in value.posterior.trajectory_ensemble
                    ]
                )
                for value in values
            ]
        ),
        posterior_angular_velocity=np.stack(
            [
                np.asarray(
                    [
                        item.angular_velocity
                        for item in value.posterior.trajectory_ensemble
                    ]
                )
                for value in values
            ]
        ),
        posterior_controller_integral=np.stack(
            [
                np.asarray(
                    [
                        item.controller_integral
                        for item in value.posterior.trajectory_ensemble
                    ]
                )
                for value in values
            ]
        ),
        posterior_commanded_thrust=np.stack(
            [
                np.asarray(
                    [
                        item.commanded_thrust
                        for item in value.posterior.trajectory_ensemble
                    ]
                )
                for value in values
            ]
        ),
        posterior_commanded_gimbal_angle=np.stack(
            [
                np.asarray(
                    [
                        item.commanded_gimbal_angle
                        for item in value.posterior.trajectory_ensemble
                    ]
                )
                for value in values
            ]
        ),
        posterior_actuator_thrust=np.stack(
            [
                np.asarray(
                    [
                        item.actuator_thrust
                        for item in value.posterior.trajectory_ensemble
                    ]
                )
                for value in values
            ]
        ),
        posterior_actuator_gimbal_angle=np.stack(
            [
                np.asarray(
                    [
                        item.actuator_gimbal_angle
                        for item in value.posterior.trajectory_ensemble
                    ]
                )
                for value in values
            ]
        ),
        posterior_body_wrench=np.stack(
            [
                np.asarray(
                    [
                        item.body_wrench
                        for item in value.posterior.trajectory_ensemble
                    ]
                )
                for value in values
            ]
        ),
        posterior_correction_translation=np.stack(
            [value.posterior.correction_translation for value in values]
        ),
        posterior_correction_rotation_vector=np.stack(
            [value.posterior.correction_rotation_vector for value in values]
        ),
        times=result.synthetic.truth.times,
        observations_position=result.synthetic.observations.position,
        observations_orientation_xyzw=(
            result.synthetic.observations.orientation_xyzw
        ),
        observation_translation_covariance=(
            result.synthetic.observations.translation_covariance
        ),
        observation_rotation_covariance=(
            result.synthetic.observations.rotation_covariance
        ),
        reference_position=np.asarray(
            [value.position for value in result.synthetic.references]
        ),
        reference_rpy=np.asarray(
            [value.rpy for value in result.synthetic.references]
        ),
        truth_position=result.synthetic.truth.position,
        truth_orientation_xyzw=result.synthetic.truth.orientation_xyzw,
        truth_linear_velocity=result.synthetic.truth.linear_velocity,
        truth_angular_velocity=result.synthetic.truth.angular_velocity,
        truth_controller_integral=(
            result.synthetic.truth.controller_integral
        ),
        truth_commanded_thrust=result.synthetic.truth.commanded_thrust,
        truth_commanded_gimbal_angle=(
            result.synthetic.truth.commanded_gimbal_angle
        ),
        truth_actuator_thrust=result.synthetic.truth.actuator_thrust,
        truth_actuator_gimbal_angle=(
            result.synthetic.truth.actuator_gimbal_angle
        ),
        truth_body_wrench=result.synthetic.truth.body_wrench,
        nominal_position=result.synthetic.nominal.position,
        nominal_orientation_xyzw=result.synthetic.nominal.orientation_xyzw,
        nominal_linear_velocity=result.synthetic.nominal.linear_velocity,
        nominal_angular_velocity=result.synthetic.nominal.angular_velocity,
        nominal_controller_integral=(
            result.synthetic.nominal.controller_integral
        ),
        nominal_commanded_thrust=result.synthetic.nominal.commanded_thrust,
        nominal_commanded_gimbal_angle=(
            result.synthetic.nominal.commanded_gimbal_angle
        ),
        nominal_actuator_thrust=result.synthetic.nominal.actuator_thrust,
        nominal_actuator_gimbal_angle=(
            result.synthetic.nominal.actuator_gimbal_angle
        ),
        nominal_body_wrench=result.synthetic.nominal.body_wrench,
    )
    return destination


# A concise name for callers that follow the plan's Experiment-D terminology.
run_experiment_d = run_mode_validation_experiment
