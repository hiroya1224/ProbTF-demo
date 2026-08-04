"""ROS-free truth generators for sparse-batch statistical recovery tests.

The helpers in this module generate data in the exact coordinates consumed by
the production batch factors.  They do not run a second, simplified estimator
and they do not use finite-difference derivatives.  In particular, the
perfect-model trajectory is advanced by Newton corrections formed from the
analytic rigid-body factor Jacobian itself.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

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
    "generate_known_q_laplace_moments",
    "generate_perfect_model_batch_trajectory",
    "simulate_delayed_zoh_first_order",
]
