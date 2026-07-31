"""Data-derived model-error statistics and sparse-knot resolution checks."""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.controller import GrapeController
from grape_param_estim.dynamics import actuator_wrench, advance_actuators
from grape_param_estim.geometry import (
    quaternion_to_matrix,
    rotation_vector_from_matrix,
)
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    ClosedLoopTrajectory,
    ControllerState,
    GRAVITY,
    GrapeGeometry,
    ReferenceState,
    RigidBodyState,
    VehicleParameters,
)


@dataclass(frozen=True)
class ModelErrorCalibration:
    """Pilot residual-wrench calibration derived from pose and nominal replay."""

    stationary_standard_deviation: np.ndarray
    pilot_location: np.ndarray
    correlation_time: float
    proxy_wrench: np.ndarray
    valid_mask: np.ndarray
    derivative_window_samples: int
    method: str = "pose-only-local-polynomial-pilot/v1"

    def __post_init__(self) -> None:
        sigma = np.asarray(self.stationary_standard_deviation, dtype=float)
        location = np.asarray(self.pilot_location, dtype=float)
        proxy = np.asarray(self.proxy_wrench, dtype=float)
        mask = np.asarray(self.valid_mask, dtype=bool)
        if (
            sigma.shape != (6,)
            or location.shape != (6,)
            or np.any(~np.isfinite(sigma))
            or np.any(~np.isfinite(location))
            or np.any(sigma <= 0.0)
        ):
            raise ValueError("model-error sigma must contain six positive values")
        if (
            proxy.ndim != 2
            or proxy.shape[1] != 6
            or np.any(~np.isfinite(proxy))
            or mask.shape != (proxy.shape[0],)
        ):
            raise ValueError("proxy wrench and validity mask must align")
        if not np.isfinite(self.correlation_time) or self.correlation_time <= 0.0:
            raise ValueError("correlation time must be positive")
        if self.derivative_window_samples < 5:
            raise ValueError("derivative window must contain at least five samples")
        object.__setattr__(self, "stationary_standard_deviation", sigma.copy())
        object.__setattr__(self, "pilot_location", location.copy())
        object.__setattr__(self, "proxy_wrench", proxy.copy())
        object.__setattr__(self, "valid_mask", mask.copy())


@dataclass(frozen=True)
class KnotResolution:
    """Required and actually selected knot resolution for one OU process."""

    knot_indices: np.ndarray
    required_knot_count: int
    maximum_bridge_gap: float
    achieved_maximum_gap: float
    bridge_standard_deviation_fraction: float
    resolution_sufficient: bool

    def __post_init__(self) -> None:
        indices = np.asarray(self.knot_indices, dtype=np.int64)
        if (
            indices.ndim != 1
            or indices.size < 2
            or np.any(np.diff(indices) <= 0)
        ):
            raise ValueError("knot indices must be a strictly increasing vector")
        if self.required_knot_count < 2:
            raise ValueError("required knot count must be at least two")
        object.__setattr__(self, "knot_indices", indices.copy())


def _validate_pose_inputs(times, position, orientation):
    sample_times = np.asarray(times, dtype=float)
    positions = np.asarray(position, dtype=float)
    quaternions = np.asarray(orientation, dtype=float)
    if (
        sample_times.ndim != 1
        or sample_times.size < 7
        or np.any(~np.isfinite(sample_times))
        or np.any(np.diff(sample_times) <= 0.0)
        or positions.shape != (sample_times.size, 3)
        or quaternions.shape != (sample_times.size, 4)
        or np.any(~np.isfinite(positions))
        or np.any(~np.isfinite(quaternions))
    ):
        raise ValueError("pose calibration samples must be finite and aligned")
    return sample_times, positions, quaternions


def _local_polynomial_derivative(
    times: np.ndarray,
    values: np.ndarray,
    derivative_order: int,
    window_samples: int,
) -> np.ndarray:
    """Differentiate irregular samples with centered local cubic fits."""

    count = times.size
    width = int(window_samples)
    if width % 2 == 0:
        width += 1
    width = min(width, count if count % 2 == 1 else count - 1)
    if width < 5 or derivative_order not in (1, 2):
        raise ValueError("local derivative needs an odd window >=5 and order 1/2")
    half = width // 2
    result = np.empty_like(values, dtype=float)
    factorial = 1.0 if derivative_order == 1 else 2.0
    for index in range(count):
        left = min(max(0, index - half), count - width)
        selected = slice(left, left + width)
        relative = times[selected] - times[index]
        design = np.column_stack(
            (np.ones(width), relative, relative**2, relative**3)
        )
        coefficients, _residual, _rank, _singular = np.linalg.lstsq(
            design, values[selected], rcond=None
        )
        result[index] = factorial * coefficients[derivative_order]
    return result


def _observed_body_omega(times: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
    rotations = tuple(quaternion_to_matrix(value) for value in quaternions)
    midpoint = np.empty((times.size - 1, 3), dtype=float)
    for index in range(midpoint.shape[0]):
        midpoint[index] = rotation_vector_from_matrix(
            rotations[index].T @ rotations[index + 1]
        ) / (times[index + 1] - times[index])
    result = np.empty((times.size, 3), dtype=float)
    result[0] = midpoint[0]
    result[-1] = midpoint[-1]
    if times.size > 2:
        weights = (
            (times[1:-1] - times[:-2])
            / (times[2:] - times[:-2])
        )[:, None]
        result[1:-1] = (
            (1.0 - weights) * midpoint[:-1] + weights * midpoint[1:]
        )
    return result


def pose_derived_initial_state(
    times: Sequence[float],
    observed_position: np.ndarray,
    observed_orientation_xyzw: np.ndarray,
) -> RigidBodyState:
    """Build the latent-state anchor using pose samples only.

    This is an initial prior anchor, not an additional observation channel.
    Its velocity is the first value of the same local polynomial used for Q
    calibration and its body angular velocity comes from consecutive SO(3)
    increments.  Both remain adjustable coordinates in the smoother.
    """

    sample_times, position, quaternions = _validate_pose_inputs(
        times, observed_position, observed_orientation_xyzw
    )
    count = sample_times.size
    window = max(5, int(np.ceil(np.sqrt(float(count)))) | 1)
    if window >= count:
        window = count if count % 2 == 1 else count - 1
    velocity = _local_polynomial_derivative(
        sample_times, position, 1, window
    )
    omega = _observed_body_omega(sample_times, quaternions)
    return RigidBodyState(
        position=position[0],
        orientation_xyzw=quaternions[0],
        linear_velocity=velocity[0],
        angular_velocity=omega[0],
    )


def _robust_location_scale(values: np.ndarray):
    location = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - location), axis=0)
    numerical_floor = np.sqrt(np.finfo(float).eps) * np.maximum(
        1.0, np.max(np.abs(values), axis=0)
    )
    return location, np.maximum(scale, numerical_floor)


def _correlation_time(times: np.ndarray, values: np.ndarray) -> float:
    centered = values - np.median(values, axis=0)
    variance = np.sum(centered**2, axis=0)
    informative = variance > np.finfo(float).eps
    median_step = float(np.median(np.diff(times)))
    if not np.any(informative):
        return float(times[-1] - times[0])
    normalized = centered[:, informative] / np.sqrt(variance[informative])
    target = float(np.exp(-1.0))
    previous_correlation = 1.0
    previous_lag = 0.0
    maximum_lag = max(1, values.shape[0] // 2)
    for lag in range(1, maximum_lag + 1):
        channel_correlation = np.sum(
            normalized[:-lag] * normalized[lag:], axis=0
        )
        correlation = float(np.median(channel_correlation))
        lag_time = float(np.median(times[lag:] - times[:-lag]))
        if correlation <= target:
            denominator = previous_correlation - correlation
            fraction = (
                1.0
                if denominator <= np.finfo(float).eps
                else (previous_correlation - target) / denominator
            )
            return max(
                median_step,
                previous_lag + fraction * (lag_time - previous_lag),
            )
        previous_correlation = correlation
        previous_lag = lag_time
    return max(median_step, previous_lag)


def calibrate_model_error_from_pose(
    times: Sequence[float],
    observed_position: np.ndarray,
    observed_orientation_xyzw: np.ndarray,
    nominal_trajectory: ClosedLoopTrajectory,
    nominal_parameters: VehicleParameters,
) -> ModelErrorCalibration:
    """Estimate OU wrench statistics from a pose-only pilot discrepancy.

    The proxy is used only to set the proper Q prior.  Velocity, angular
    velocity, acceleration and IMU messages are never added to the smoother's
    likelihood.  Derivatives below are computed from the same position and
    orientation samples that define the pose observation contract.
    """

    sample_times, position, quaternions = _validate_pose_inputs(
        times, observed_position, observed_orientation_xyzw
    )
    if not np.array_equal(nominal_trajectory.times, sample_times):
        raise ValueError("nominal replay and pose calibration times must agree")
    count = sample_times.size
    window = max(5, int(np.ceil(np.sqrt(float(count)))) | 1)
    if window >= count:
        window = count if count % 2 == 1 else count - 1

    observed_acceleration = _local_polynomial_derivative(
        sample_times, position, 2, window
    )
    nominal_acceleration = _local_polynomial_derivative(
        sample_times, nominal_trajectory.linear_velocity, 1, window
    )
    observed_omega = _observed_body_omega(sample_times, quaternions)
    observed_alpha = _local_polynomial_derivative(
        sample_times, observed_omega, 1, window
    )
    nominal_alpha = _local_polynomial_derivative(
        sample_times, nominal_trajectory.angular_velocity, 1, window
    )

    proxy = np.empty((count, 6), dtype=float)
    for index in range(count):
        observed_rotation = quaternion_to_matrix(quaternions[index])
        proxy[index, :3] = nominal_parameters.mass * (
            observed_rotation.T
            @ (observed_acceleration[index] - nominal_acceleration[index])
        )
        proxy[index, 3:] = nominal_parameters.inertia @ (
            observed_alpha[index] - nominal_alpha[index]
        )

    half = window // 2
    valid = np.zeros(count, dtype=bool)
    valid[half:count - half] = True
    calibrated = proxy[valid]
    location, sigma = _robust_location_scale(calibrated)
    tau = _correlation_time(sample_times[valid], calibrated)
    return ModelErrorCalibration(
        stationary_standard_deviation=sigma,
        pilot_location=location,
        correlation_time=tau,
        proxy_wrench=proxy,
        valid_mask=valid,
        derivative_window_samples=window,
    )


def calibrate_model_error_from_closed_loop_pose(
    times: Sequence[float],
    observed_position: np.ndarray,
    observed_orientation_xyzw: np.ndarray,
    references: Sequence[ReferenceState],
    controller_configuration,
    initial_controller_state: ControllerState,
    initial_actuator_state: ActuatorState,
    actuator_parameters: ActuatorParameters,
    nominal_parameters: VehicleParameters,
    geometry: Optional[GrapeGeometry] = None,
) -> ModelErrorCalibration:
    """Calibrate Q from a causal nominal replay evaluated on observed pose.

    Pose derivatives provide the wrench required by the observed motion.  A
    nominal controller and actuator are then replayed causally on that latent
    pose-derived state path.  Their predicted actuator wrench is subtracted
    from the required wrench.  Recorded command, odometry twist and IMU fields
    are not used, and the resulting proxy only calibrates the Q prior; the
    smoother likelihood remains position and orientation only.
    """

    sample_times, position, quaternions = _validate_pose_inputs(
        times, observed_position, observed_orientation_xyzw
    )
    if len(references) != sample_times.size:
        raise ValueError("one reference is required per calibration sample")
    count = sample_times.size
    window = max(5, int(np.ceil(np.sqrt(float(count)))) | 1)
    if window >= count:
        window = count if count % 2 == 1 else count - 1
    velocity = _local_polynomial_derivative(
        sample_times, position, 1, window
    )
    acceleration = _local_polynomial_derivative(
        sample_times, position, 2, window
    )
    omega = _observed_body_omega(sample_times, quaternions)
    alpha = _local_polynomial_derivative(
        sample_times, omega, 1, window
    )
    states = tuple(
        RigidBodyState(
            position[index],
            quaternions[index],
            velocity[index],
            omega[index],
        )
        for index in range(count)
    )
    selected_geometry = geometry or GrapeGeometry.grape()
    controller = GrapeController(
        controller_configuration,
        nominal_parameters,
        selected_geometry,
        articulated_model=GrapeArticulatedModel(),
    )
    controller_state = ControllerState(
        initial_controller_state.integral_error,
        initial_controller_state.roll_pitch_integration_active,
    )
    actuators = ActuatorState(
        initial_actuator_state.thrust,
        initial_actuator_state.gimbal_angle,
    )
    predicted_actuator_wrench = np.empty((count, 6), dtype=float)
    for index in range(count):
        predicted_actuator_wrench[index] = actuator_wrench(
            actuators, nominal_parameters, selected_geometry
        )
        time_step = (
            sample_times[index + 1] - sample_times[index]
            if index + 1 < count
            else sample_times[index] - sample_times[index - 1]
        )
        command, next_controller_state = controller.step(
            states[index],
            references[index],
            controller_state,
            time_step,
            actuators.gimbal_angle,
        )
        if index + 1 < count:
            midpoint = advance_actuators(
                actuators, command, actuator_parameters, 0.5 * time_step
            )
            actuators = advance_actuators(
                midpoint, command, actuator_parameters, 0.5 * time_step
            )
            controller_state = next_controller_state

    required = np.empty((count, 6), dtype=float)
    gravity = np.asarray((0.0, 0.0, -GRAVITY))
    for index, state in enumerate(states):
        rotation = quaternion_to_matrix(state.orientation_xyzw)
        required[index, :3] = nominal_parameters.mass * (
            rotation.T @ (acceleration[index] - gravity)
        )
        required[index, 3:] = (
            nominal_parameters.inertia @ alpha[index]
            + np.cross(
                state.angular_velocity,
                nominal_parameters.inertia @ state.angular_velocity,
            )
        )
    proxy = required - predicted_actuator_wrench
    half = window // 2
    valid = np.zeros(count, dtype=bool)
    valid[half:count - half] = True
    calibrated = proxy[valid]
    location, scale = _robust_location_scale(calibrated)
    return ModelErrorCalibration(
        stationary_standard_deviation=scale,
        pilot_location=location,
        correlation_time=_correlation_time(
            sample_times[valid], calibrated
        ),
        proxy_wrench=proxy,
        valid_mask=valid,
        derivative_window_samples=window,
        method="pose-only-counterfactual-closed-loop-wrench/v1",
    )


def select_ou_knot_resolution(
    integration_times: Sequence[float],
    correlation_time: float,
    bridge_standard_deviation_fraction: float = 0.5,
    maximum_knots: Optional[int] = None,
) -> KnotResolution:
    """Select knots from an OU bridge-variance tolerance.

    For an OU bridge over gap ``h``, the midpoint conditional standard
    deviation divided by the stationary standard deviation is
    ``sqrt(tanh(h / (2 tau)))``.  Therefore the requested fraction ``eps``
    gives ``h <= 2 tau atanh(eps**2)``.
    """

    times = np.asarray(integration_times, dtype=float)
    tau = float(correlation_time)
    epsilon = float(bridge_standard_deviation_fraction)
    if (
        times.ndim != 1
        or times.size < 2
        or np.any(~np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("integration times must be finite and increasing")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("correlation time must be positive")
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError("bridge standard-deviation fraction must be in (0,1)")
    if maximum_knots is not None:
        if isinstance(maximum_knots, (bool, np.bool_)):
            raise ValueError("maximum_knots must be an integer >=2")
        maximum_knots = int(maximum_knots)
        if maximum_knots < 2:
            raise ValueError("maximum_knots must be an integer >=2")

    maximum_gap = float(2.0 * tau * np.arctanh(epsilon**2))
    required = [0]
    while required[-1] != times.size - 1:
        current = required[-1]
        candidate = int(
            np.searchsorted(
                times,
                times[current] + maximum_gap,
                side="right",
            ) - 1
        )
        candidate = max(current + 1, candidate)
        candidate = min(candidate, times.size - 1)
        required.append(candidate)

    selected = np.asarray(required, dtype=np.int64)
    sufficient = True
    if maximum_knots is not None and selected.size > maximum_knots:
        target_times = np.linspace(times[0], times[-1], maximum_knots)
        selected_values = [0]
        for target in target_times[1:-1]:
            index = int(np.argmin(np.abs(times - target)))
            index = max(selected_values[-1] + 1, index)
            remaining = maximum_knots - len(selected_values) - 1
            index = min(index, times.size - 1 - remaining)
            selected_values.append(index)
        selected_values.append(times.size - 1)
        selected = np.asarray(selected_values, dtype=np.int64)
        sufficient = False

    achieved = float(np.max(np.diff(times[selected])))
    sufficient = bool(sufficient and achieved <= maximum_gap * (1.0 + 1.0e-12))
    return KnotResolution(
        knot_indices=selected,
        required_knot_count=len(required),
        maximum_bridge_gap=maximum_gap,
        achieved_maximum_gap=achieved,
        bridge_standard_deviation_fraction=epsilon,
        resolution_sufficient=sufficient,
    )


__all__ = [
    "KnotResolution",
    "ModelErrorCalibration",
    "calibrate_model_error_from_closed_loop_pose",
    "calibrate_model_error_from_pose",
    "pose_derived_initial_state",
    "select_ou_knot_resolution",
]
