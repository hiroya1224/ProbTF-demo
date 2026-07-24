"""Mocap/IMU trajectory posterior with causal filtering and RTS smoothing.

This module intentionally does not turn a numerical second derivative of
mocap position into an independent observation.  Accelerometer and gyro
samples drive the state transition; mocap position and orientation are pose
evidence.  The offline backend applies an error-state Rauch--Tung--Striebel
smoother, while ``online_prefix=True`` exposes filter marginals only.

The implementation is application-local until the generic ProbTF temporal
model contract is available.  Its output contract is independent from ROS and
includes coherent trajectory samples for downstream target-tube events.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


ERROR_STATE_SIZE = 15
_P = slice(0, 3)
_V = slice(3, 6)
_THETA = slice(6, 9)
_BG = slice(9, 12)
_BA = slice(12, 15)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _readonly(values: np.ndarray, shape: Tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != shape:
        raise ValueError("{} must have shape {}".format(name, shape))
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy


def _times(values: np.ndarray, name: str, allow_empty: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if (not allow_empty and not array.size) or not np.all(np.isfinite(array)):
        raise ValueError("{} must be a finite non-empty vector".format(name))
    if array.size > 1 and np.any(np.diff(array) <= 0.0):
        raise ValueError("{} must be strictly increasing".format(name))
    return _readonly(array, (array.size,), name)


def _continuous_quaternions(values: np.ndarray, count: int, name: str) -> np.ndarray:
    quaternions = np.asarray(values, dtype=float)
    if quaternions.shape != (count, 4) or not np.all(np.isfinite(quaternions)):
        raise ValueError("{} must have shape ({}, 4) and be finite".format(name, count))
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms <= 1.0e-12):
        raise ValueError("{} contains a zero quaternion".format(name))
    output = np.array(quaternions / norms[:, None], copy=True)
    for index in range(1, count):
        if np.dot(output[index - 1], output[index]) < 0.0:
            output[index] *= -1.0
    output.setflags(write=False)
    return output


def _vectors(values: np.ndarray, count: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (count, 3) or not np.all(np.isfinite(array)):
        raise ValueError("{} must have shape ({}, 3) and be finite".format(name, count))
    return _readonly(array, (count, 3), name)


def _mask(values: Optional[np.ndarray], count: int, name: str) -> np.ndarray:
    array = np.ones(count, dtype=bool) if values is None else np.asarray(values, dtype=bool)
    return _readonly(array, (count,), name)


def _positive_semidefinite(matrix: np.ndarray, floor: float = 1.0e-12) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.maximum(eigenvalues, float(floor))
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _safe_cholesky(matrix: np.ndarray) -> np.ndarray:
    covariance = _positive_semidefinite(matrix)
    try:
        return np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        return eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1.0e-12)))


@dataclass(frozen=True)
class TrajectoryObservations:
    mocap_times: np.ndarray
    mocap_positions_world: np.ndarray
    mocap_quaternions_xyzw: np.ndarray
    imu_times: np.ndarray
    accelerometer_body: np.ndarray
    gyro_body: np.ndarray
    mocap_valid_mask: Optional[np.ndarray] = None
    imu_valid_mask: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        mocap_times = _times(self.mocap_times, "mocap_times")
        imu_times = _times(self.imu_times, "imu_times")
        positions = _vectors(
            self.mocap_positions_world, mocap_times.size, "mocap_positions_world"
        )
        quaternions = _continuous_quaternions(
            self.mocap_quaternions_xyzw,
            mocap_times.size,
            "mocap_quaternions_xyzw",
        )
        acceleration = _vectors(
            self.accelerometer_body, imu_times.size, "accelerometer_body"
        )
        gyro = _vectors(self.gyro_body, imu_times.size, "gyro_body")
        mocap_mask = _mask(
            self.mocap_valid_mask, mocap_times.size, "mocap_valid_mask"
        )
        imu_mask = _mask(self.imu_valid_mask, imu_times.size, "imu_valid_mask")
        if not np.any(mocap_mask):
            raise ValueError("at least one valid mocap sample is required")
        if not np.any(imu_mask):
            raise ValueError("at least one valid IMU sample is required")
        object.__setattr__(self, "mocap_times", mocap_times)
        object.__setattr__(self, "mocap_positions_world", positions)
        object.__setattr__(self, "mocap_quaternions_xyzw", quaternions)
        object.__setattr__(self, "imu_times", imu_times)
        object.__setattr__(self, "accelerometer_body", acceleration)
        object.__setattr__(self, "gyro_body", gyro)
        object.__setattr__(self, "mocap_valid_mask", mocap_mask)
        object.__setattr__(self, "imu_valid_mask", imu_mask)


@dataclass(frozen=True)
class SmootherConfig:
    backend: str = "error_state_ekf_rts"
    gravity_world: Tuple[float, float, float] = (0.0, 0.0, -9.80665)
    mocap_position_sigma: float = 0.01
    mocap_orientation_sigma: float = float(np.deg2rad(1.0))
    accelerometer_noise_sigma: float = 0.20
    gyro_noise_sigma: float = float(np.deg2rad(0.5))
    accelerometer_bias_random_walk_sigma: float = 0.02
    gyro_bias_random_walk_sigma: float = float(np.deg2rad(0.05))
    initial_position_sigma: float = 0.02
    initial_velocity_sigma: float = 0.50
    initial_orientation_sigma: float = float(np.deg2rad(2.0))
    initial_accelerometer_bias_sigma: float = 0.30
    initial_gyro_bias_sigma: float = float(np.deg2rad(1.0))
    mocap_nis_gate: float = 30.0
    max_accelerometer_norm: float = 100.0
    max_gyro_norm: float = 20.0
    max_propagation_step: float = 0.02
    trajectory_sample_count: int = 32
    seed: int = 7

    def __post_init__(self) -> None:
        if self.backend != "error_state_ekf_rts":
            raise ValueError("only backend='error_state_ekf_rts' is implemented")
        gravity = np.asarray(self.gravity_world, dtype=float)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("gravity_world must be a finite three-vector")
        positive_names = (
            "mocap_position_sigma",
            "mocap_orientation_sigma",
            "accelerometer_noise_sigma",
            "gyro_noise_sigma",
            "accelerometer_bias_random_walk_sigma",
            "gyro_bias_random_walk_sigma",
            "initial_position_sigma",
            "initial_velocity_sigma",
            "initial_orientation_sigma",
            "initial_accelerometer_bias_sigma",
            "initial_gyro_bias_sigma",
            "mocap_nis_gate",
            "max_accelerometer_norm",
            "max_gyro_norm",
            "max_propagation_step",
        )
        for name in positive_names:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("{} must be finite and positive".format(name))
            object.__setattr__(self, name, value)
        count = int(self.trajectory_sample_count)
        if count < 0:
            raise ValueError("trajectory_sample_count must be non-negative")
        object.__setattr__(self, "trajectory_sample_count", count)
        object.__setattr__(self, "gravity_world", tuple(float(item) for item in gravity))


@dataclass(frozen=True)
class TrajectoryPosterior:
    timestamps: np.ndarray
    position_world: np.ndarray
    velocity_world: np.ndarray
    velocity_body: np.ndarray
    quaternion_xyzw: np.ndarray
    angular_velocity_body: np.ndarray
    accelerometer_bias_body: np.ndarray
    gyro_bias_body: np.ndarray
    covariance: np.ndarray
    sample_ids: np.ndarray
    sample_weights: np.ndarray
    sample_position_world: np.ndarray
    sample_velocity_world: np.ndarray
    sample_quaternion_xyzw: np.ndarray
    sample_angular_velocity_body: np.ndarray
    sample_accelerometer_bias_body: np.ndarray
    sample_gyro_bias_body: np.ndarray
    mocap_used: np.ndarray
    mocap_rejected: np.ndarray
    imu_used: np.ndarray
    is_smoothed: bool
    causal_cutoff: float
    sampling_approximation: str

    def __post_init__(self) -> None:
        times = _times(self.timestamps, "timestamps")
        count = times.size
        samples = int(np.asarray(self.sample_ids).size)
        fields = (
            ("position_world", self.position_world, (count, 3)),
            ("velocity_world", self.velocity_world, (count, 3)),
            ("velocity_body", self.velocity_body, (count, 3)),
            ("quaternion_xyzw", self.quaternion_xyzw, (count, 4)),
            ("angular_velocity_body", self.angular_velocity_body, (count, 3)),
            (
                "accelerometer_bias_body",
                self.accelerometer_bias_body,
                (count, 3),
            ),
            ("gyro_bias_body", self.gyro_bias_body, (count, 3)),
            ("covariance", self.covariance, (count, ERROR_STATE_SIZE, ERROR_STATE_SIZE)),
            (
                "sample_position_world",
                self.sample_position_world,
                (samples, count, 3),
            ),
            (
                "sample_velocity_world",
                self.sample_velocity_world,
                (samples, count, 3),
            ),
            (
                "sample_quaternion_xyzw",
                self.sample_quaternion_xyzw,
                (samples, count, 4),
            ),
            (
                "sample_angular_velocity_body",
                self.sample_angular_velocity_body,
                (samples, count, 3),
            ),
            (
                "sample_accelerometer_bias_body",
                self.sample_accelerometer_bias_body,
                (samples, count, 3),
            ),
            (
                "sample_gyro_bias_body",
                self.sample_gyro_bias_body,
                (samples, count, 3),
            ),
        )
        for name, value, shape in fields:
            array = np.asarray(value, dtype=float)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError("{} must have finite shape {}".format(name, shape))
            if name == "covariance":
                if np.min(np.linalg.eigvalsh(array)) < -1.0e-8:
                    raise ValueError("covariance must be positive semidefinite")
            object.__setattr__(self, name, _readonly(array, shape, name))
        quaternion_norms = np.linalg.norm(self.quaternion_xyzw, axis=1)
        if not np.allclose(quaternion_norms, 1.0, atol=1.0e-8, rtol=0.0):
            raise ValueError("mean quaternions must be unit length")
        if samples:
            sample_norms = np.linalg.norm(self.sample_quaternion_xyzw, axis=2)
            if not np.allclose(sample_norms, 1.0, atol=1.0e-8, rtol=0.0):
                raise ValueError("sample quaternions must be unit length")
        ids = _readonly(np.asarray(self.sample_ids, dtype=np.uint64), (samples,), "sample_ids")
        weights = np.asarray(self.sample_weights, dtype=float)
        if weights.shape != (samples,) or np.any(weights < 0.0):
            raise ValueError("sample_weights must be a non-negative sample vector")
        if samples and not np.isclose(np.sum(weights), 1.0):
            raise ValueError("sample_weights must sum to one")
        masks = (
            ("mocap_used", self.mocap_used),
            ("mocap_rejected", self.mocap_rejected),
            ("imu_used", self.imu_used),
        )
        for name, values in masks:
            object.__setattr__(
                self, name, _readonly(np.asarray(values, dtype=bool), (count,), name)
            )
        cutoff = float(self.causal_cutoff)
        if not np.isfinite(cutoff):
            raise ValueError("causal_cutoff must be finite")
        object.__setattr__(self, "timestamps", times)
        object.__setattr__(self, "sample_ids", ids)
        object.__setattr__(self, "sample_weights", _readonly(weights, (samples,), "weights"))
        object.__setattr__(self, "causal_cutoff", cutoff)

    @property
    def sample_count(self) -> int:
        return int(self.sample_ids.size)


@dataclass
class _NominalState:
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    gyro_bias: np.ndarray
    accelerometer_bias: np.ndarray

    def copy(self) -> "_NominalState":
        return _NominalState(
            self.position.copy(),
            self.velocity.copy(),
            self.quaternion.copy(),
            self.gyro_bias.copy(),
            self.accelerometer_bias.copy(),
        )


def _state_error(reference: _NominalState, value: _NominalState) -> np.ndarray:
    result = np.empty(ERROR_STATE_SIZE)
    result[_P] = value.position - reference.position
    result[_V] = value.velocity - reference.velocity
    result[_THETA] = (
        Rotation.from_quat(reference.quaternion).inv()
        * Rotation.from_quat(value.quaternion)
    ).as_rotvec()
    result[_BG] = value.gyro_bias - reference.gyro_bias
    result[_BA] = value.accelerometer_bias - reference.accelerometer_bias
    return result


def _apply_error(state: _NominalState, error: np.ndarray) -> _NominalState:
    delta = np.asarray(error, dtype=float).reshape(ERROR_STATE_SIZE)
    quaternion = (
        Rotation.from_quat(state.quaternion)
        * Rotation.from_rotvec(delta[_THETA])
    ).as_quat()
    if np.dot(quaternion, state.quaternion) < 0.0:
        quaternion *= -1.0
    return _NominalState(
        position=state.position + delta[_P],
        velocity=state.velocity + delta[_V],
        quaternion=quaternion,
        gyro_bias=state.gyro_bias + delta[_BG],
        accelerometer_bias=state.accelerometer_bias + delta[_BA],
    )


class ErrorStateEkfRtsSmoother:
    """Error-state EKF/RTS implementation behind the common output contract."""

    def __init__(self, config: SmootherConfig = SmootherConfig()) -> None:
        if not isinstance(config, SmootherConfig):
            raise TypeError("config must be SmootherConfig")
        self.config = config

    def _initial_covariance(self) -> np.ndarray:
        config = self.config
        standard_deviations = np.concatenate(
            (
                np.full(3, config.initial_position_sigma),
                np.full(3, config.initial_velocity_sigma),
                np.full(3, config.initial_orientation_sigma),
                np.full(3, config.initial_gyro_bias_sigma),
                np.full(3, config.initial_accelerometer_bias_sigma),
            )
        )
        return np.diag(standard_deviations * standard_deviations)

    def _propagation_step(
        self,
        state: _NominalState,
        covariance: np.ndarray,
        acceleration_measurement: np.ndarray,
        gyro_measurement: np.ndarray,
        delta: float,
    ) -> Tuple[_NominalState, np.ndarray, np.ndarray]:
        config = self.config
        rotation = Rotation.from_quat(state.quaternion)
        rotation_matrix = rotation.as_matrix()
        omega = gyro_measurement - state.gyro_bias
        specific = acceleration_measurement - state.accelerometer_bias
        acceleration_world = rotation_matrix @ specific + np.asarray(config.gravity_world)

        propagated = _NominalState(
            position=state.position
            + state.velocity * delta
            + 0.5 * acceleration_world * delta * delta,
            velocity=state.velocity + acceleration_world * delta,
            quaternion=(rotation * Rotation.from_rotvec(omega * delta)).as_quat(),
            gyro_bias=state.gyro_bias.copy(),
            accelerometer_bias=state.accelerometer_bias.copy(),
        )
        transition = np.eye(ERROR_STATE_SIZE)
        transition[_P, _V] = np.eye(3) * delta
        transition[_P, _THETA] = (
            -0.5 * rotation_matrix @ _skew(specific) * delta * delta
        )
        transition[_P, _BA] = -0.5 * rotation_matrix * delta * delta
        transition[_V, _THETA] = -rotation_matrix @ _skew(specific) * delta
        transition[_V, _BA] = -rotation_matrix * delta
        transition[_THETA, _THETA] = np.eye(3) - _skew(omega) * delta
        transition[_THETA, _BG] = -np.eye(3) * delta

        process = np.zeros((ERROR_STATE_SIZE, ERROR_STATE_SIZE))
        accel_variance = config.accelerometer_noise_sigma ** 2
        gyro_variance = config.gyro_noise_sigma ** 2
        process[_P, _P] += np.eye(3) * accel_variance * delta ** 4 / 4.0
        process[_P, _V] += np.eye(3) * accel_variance * delta ** 3 / 2.0
        process[_V, _P] += np.eye(3) * accel_variance * delta ** 3 / 2.0
        process[_V, _V] += np.eye(3) * accel_variance * delta ** 2
        process[_THETA, _THETA] += np.eye(3) * gyro_variance * delta ** 2
        process[_BG, _BG] += (
            np.eye(3) * config.gyro_bias_random_walk_sigma ** 2 * delta
        )
        process[_BA, _BA] += (
            np.eye(3) * config.accelerometer_bias_random_walk_sigma ** 2 * delta
        )
        propagated_covariance = (
            transition @ covariance @ transition.T + process
        )
        return propagated, _positive_semidefinite(propagated_covariance), transition

    def _propagate(
        self,
        state: _NominalState,
        covariance: np.ndarray,
        acceleration_measurement: np.ndarray,
        gyro_measurement: np.ndarray,
        delta: float,
    ) -> Tuple[_NominalState, np.ndarray, np.ndarray]:
        if delta <= 0.0:
            return state.copy(), covariance.copy(), np.eye(ERROR_STATE_SIZE)
        count = max(1, int(np.ceil(delta / self.config.max_propagation_step)))
        step = delta / count
        current_state = state.copy()
        current_covariance = covariance.copy()
        total_transition = np.eye(ERROR_STATE_SIZE)
        for _ in range(count):
            current_state, current_covariance, transition = self._propagation_step(
                current_state,
                current_covariance,
                acceleration_measurement,
                gyro_measurement,
                step,
            )
            total_transition = transition @ total_transition
        return current_state, current_covariance, total_transition

    def _mocap_update(
        self,
        state: _NominalState,
        covariance: np.ndarray,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> Tuple[_NominalState, np.ndarray, bool]:
        residual = np.concatenate(
            (
                position - state.position,
                (
                    Rotation.from_quat(state.quaternion).inv()
                    * Rotation.from_quat(quaternion)
                ).as_rotvec(),
            )
        )
        observation = np.zeros((6, ERROR_STATE_SIZE))
        observation[:3, _P] = np.eye(3)
        observation[3:, _THETA] = np.eye(3)
        measurement = np.diag(
            np.concatenate(
                (
                    np.full(3, self.config.mocap_position_sigma ** 2),
                    np.full(3, self.config.mocap_orientation_sigma ** 2),
                )
            )
        )
        innovation_covariance = (
            observation @ covariance @ observation.T + measurement
        )
        normalized_innovation = float(
            residual @ np.linalg.solve(innovation_covariance, residual)
        )
        if (
            not np.isfinite(normalized_innovation)
            or normalized_innovation > self.config.mocap_nis_gate
        ):
            return state, covariance, False
        gain = np.linalg.solve(
            innovation_covariance.T, (covariance @ observation.T).T
        ).T
        correction = gain @ residual
        updated_state = _apply_error(state, correction)
        identity_minus = np.eye(ERROR_STATE_SIZE) - gain @ observation
        updated_covariance = (
            identity_minus @ covariance @ identity_minus.T
            + gain @ measurement @ gain.T
        )
        return updated_state, _positive_semidefinite(updated_covariance), True

    def smooth(
        self,
        observations: TrajectoryObservations,
        online_prefix: bool = False,
        cutoff: Optional[float] = None,
    ) -> TrajectoryPosterior:
        """Estimate the trajectory up to ``cutoff``.

        Offline mode applies RTS smoothing and backward trajectory sampling.
        Prefix mode never calls either backward pass; every reported marginal
        is a function only of messages at or before its timestamp.
        """

        if not isinstance(observations, TrajectoryObservations):
            raise TypeError("observations must be TrajectoryObservations")
        final_time = (
            min(
                float(cutoff),
                float(max(observations.mocap_times[-1], observations.imu_times[-1])),
            )
            if cutoff is not None
            else float(max(observations.mocap_times[-1], observations.imu_times[-1]))
        )
        if not np.isfinite(final_time):
            raise ValueError("cutoff must be finite")
        valid_mocap_indices = np.flatnonzero(
            observations.mocap_valid_mask
            & (observations.mocap_times <= final_time)
        )
        valid_imu_indices = np.flatnonzero(
            observations.imu_valid_mask & (observations.imu_times <= final_time)
        )
        if not valid_mocap_indices.size or not valid_imu_indices.size:
            raise ValueError("cutoff leaves no valid mocap or IMU samples")
        first_mocap_index = int(valid_mocap_indices[0])
        first_imu_index = int(valid_imu_indices[0])
        start_time = max(
            float(observations.mocap_times[first_mocap_index]),
            float(observations.imu_times[first_imu_index]),
        )
        initial_mocap_candidates = valid_mocap_indices[
            observations.mocap_times[valid_mocap_indices] <= start_time
        ]
        if not initial_mocap_candidates.size:
            raise ValueError("no causal mocap initialization at the first IMU time")
        initial_mocap = int(initial_mocap_candidates[-1])
        initial_imu_candidates = valid_imu_indices[
            observations.imu_times[valid_imu_indices] <= start_time
        ]
        if not initial_imu_candidates.size:
            raise ValueError("no causal IMU initialization at the first mocap time")
        initial_imu = int(initial_imu_candidates[-1])

        mocap_indices = np.flatnonzero(
            (observations.mocap_times >= start_time)
            & (observations.mocap_times <= final_time)
        )
        imu_indices = np.flatnonzero(
            (observations.imu_times >= start_time)
            & (observations.imu_times <= final_time)
        )
        event_times = np.unique(
            np.concatenate(
                (
                    observations.mocap_times[mocap_indices],
                    observations.imu_times[imu_indices],
                    np.array([start_time]),
                )
            )
        )
        event_times = event_times[event_times <= final_time]
        event_count = event_times.size
        mocap_by_time = {
            float(stamp): []
            for stamp in observations.mocap_times[mocap_indices]
        }
        for index in mocap_indices:
            mocap_by_time[float(observations.mocap_times[index])].append(int(index))
        imu_by_time = {
            float(stamp): []
            for stamp in observations.imu_times[imu_indices]
        }
        for index in imu_indices:
            imu_by_time[float(observations.imu_times[index])].append(int(index))

        current_state = _NominalState(
            position=observations.mocap_positions_world[initial_mocap].copy(),
            velocity=np.zeros(3),
            quaternion=observations.mocap_quaternions_xyzw[initial_mocap].copy(),
            gyro_bias=np.zeros(3),
            accelerometer_bias=np.zeros(3),
        )
        current_covariance = self._initial_covariance()
        current_acceleration = observations.accelerometer_body[initial_imu].copy()
        current_gyro = observations.gyro_body[initial_imu].copy()

        filtered_states = []
        filtered_covariances = []
        predicted_states = []
        predicted_covariances = []
        transitions = []
        gyro_at_event = []
        mocap_used = np.zeros(event_count, dtype=bool)
        mocap_rejected = np.zeros(event_count, dtype=bool)
        imu_used = np.zeros(event_count, dtype=bool)
        previous_time = float(event_times[0])
        for event_index, event_stamp_value in enumerate(event_times):
            event_stamp = float(event_stamp_value)
            if event_index:
                (
                    current_state,
                    current_covariance,
                    transition,
                ) = self._propagate(
                    current_state,
                    current_covariance,
                    current_acceleration,
                    current_gyro,
                    event_stamp - previous_time,
                )
            else:
                transition = np.eye(ERROR_STATE_SIZE)
            predicted_states.append(current_state.copy())
            predicted_covariances.append(current_covariance.copy())
            transitions.append(transition)

            for index in imu_by_time.get(event_stamp, ()):
                acceleration = observations.accelerometer_body[index]
                gyro = observations.gyro_body[index]
                valid = bool(observations.imu_valid_mask[index])
                valid &= np.linalg.norm(acceleration) <= self.config.max_accelerometer_norm
                valid &= np.linalg.norm(gyro) <= self.config.max_gyro_norm
                if valid:
                    current_acceleration = acceleration.copy()
                    current_gyro = gyro.copy()
                    imu_used[event_index] = True

            for index in mocap_by_time.get(event_stamp, ()):
                if not observations.mocap_valid_mask[index]:
                    mocap_rejected[event_index] = True
                    continue
                current_state, current_covariance, accepted = self._mocap_update(
                    current_state,
                    current_covariance,
                    observations.mocap_positions_world[index],
                    observations.mocap_quaternions_xyzw[index],
                )
                mocap_used[event_index] |= accepted
                mocap_rejected[event_index] |= not accepted

            filtered_states.append(current_state.copy())
            filtered_covariances.append(current_covariance.copy())
            gyro_at_event.append(current_gyro.copy())
            previous_time = event_stamp

        filtered_covariances_array = np.asarray(filtered_covariances)
        predicted_covariances_array = np.asarray(predicted_covariances)
        if online_prefix:
            output_states = [item.copy() for item in filtered_states]
            output_covariances = filtered_covariances_array.copy()
            is_smoothed = False
        else:
            output_states = [item.copy() for item in filtered_states]
            output_covariances = filtered_covariances_array.copy()
            for index in range(event_count - 2, -1, -1):
                transition = transitions[index + 1]
                predicted_covariance = predicted_covariances_array[index + 1]
                smoother_gain = np.linalg.solve(
                    predicted_covariance.T,
                    (filtered_covariances_array[index] @ transition.T).T,
                ).T
                difference = _state_error(
                    predicted_states[index + 1], output_states[index + 1]
                )
                output_states[index] = _apply_error(
                    filtered_states[index], smoother_gain @ difference
                )
                output_covariances[index] = _positive_semidefinite(
                    filtered_covariances_array[index]
                    + smoother_gain
                    @ (
                        output_covariances[index + 1]
                        - predicted_covariance
                    )
                    @ smoother_gain.T
                )
            is_smoothed = True

        sampled_states = self._sample_trajectories(
            filtered_states=filtered_states,
            filtered_covariances=filtered_covariances_array,
            predicted_states=predicted_states,
            predicted_covariances=predicted_covariances_array,
            transitions=transitions,
            output_states=output_states,
            output_covariances=output_covariances,
            online_prefix=online_prefix,
        )
        return self._posterior(
            event_times,
            output_states,
            output_covariances,
            sampled_states,
            np.asarray(gyro_at_event),
            mocap_used,
            mocap_rejected,
            imu_used,
            is_smoothed,
            final_time,
        )

    def _sample_trajectories(
        self,
        filtered_states,
        filtered_covariances,
        predicted_states,
        predicted_covariances,
        transitions,
        output_states,
        output_covariances,
        online_prefix,
    ):
        sample_count = self.config.trajectory_sample_count
        event_count = len(filtered_states)
        if not sample_count:
            return []
        rng = np.random.default_rng(int(self.config.seed))
        sampled = [
            [None for _ in range(event_count)] for _ in range(sample_count)
        ]
        if online_prefix:
            # A shared whitened vector keeps a stable sample ID across time.
            # It is an explicit approximation, but unlike a backward simulator
            # it never conditions an earlier state on a future observation.
            whitened = rng.normal(size=(sample_count, ERROR_STATE_SIZE))
            for index in range(event_count):
                cholesky = _safe_cholesky(filtered_covariances[index])
                for sample_index in range(sample_count):
                    sampled[sample_index][index] = _apply_error(
                        filtered_states[index],
                        cholesky @ whitened[sample_index],
                    )
            return sampled

        terminal_cholesky = _safe_cholesky(output_covariances[-1])
        for sample_index in range(sample_count):
            sampled[sample_index][-1] = _apply_error(
                output_states[-1],
                terminal_cholesky @ rng.normal(size=ERROR_STATE_SIZE),
            )
        for index in range(event_count - 2, -1, -1):
            transition = transitions[index + 1]
            predicted_covariance = predicted_covariances[index + 1]
            backward_gain = np.linalg.solve(
                predicted_covariance.T,
                (filtered_covariances[index] @ transition.T).T,
            ).T
            conditional_covariance = _positive_semidefinite(
                filtered_covariances[index]
                - backward_gain @ predicted_covariance @ backward_gain.T
            )
            cholesky = _safe_cholesky(conditional_covariance)
            for sample_index in range(sample_count):
                next_difference = _state_error(
                    predicted_states[index + 1],
                    sampled[sample_index][index + 1],
                )
                conditional_mean = _apply_error(
                    filtered_states[index], backward_gain @ next_difference
                )
                sampled[sample_index][index] = _apply_error(
                    conditional_mean,
                    cholesky @ rng.normal(size=ERROR_STATE_SIZE),
                )
        return sampled

    def _posterior(
        self,
        timestamps,
        states,
        covariances,
        sampled_states,
        gyro_at_event,
        mocap_used,
        mocap_rejected,
        imu_used,
        is_smoothed,
        cutoff,
    ) -> TrajectoryPosterior:
        positions = np.asarray([item.position for item in states])
        velocities = np.asarray([item.velocity for item in states])
        quaternions = np.asarray([item.quaternion for item in states])
        for index in range(1, len(quaternions)):
            if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
                quaternions[index] *= -1.0
        rotations = Rotation.from_quat(quaternions)
        velocities_body = rotations.inv().apply(velocities)
        gyro_bias = np.asarray([item.gyro_bias for item in states])
        accelerometer_bias = np.asarray(
            [item.accelerometer_bias for item in states]
        )
        angular_velocity = gyro_at_event - gyro_bias

        sample_count = len(sampled_states)
        event_count = len(states)
        sample_positions = np.empty((sample_count, event_count, 3))
        sample_velocities = np.empty((sample_count, event_count, 3))
        sample_quaternions = np.empty((sample_count, event_count, 4))
        sample_gyro_bias = np.empty((sample_count, event_count, 3))
        sample_accelerometer_bias = np.empty((sample_count, event_count, 3))
        for sample_index, trajectory in enumerate(sampled_states):
            for event_index, state in enumerate(trajectory):
                sample_positions[sample_index, event_index] = state.position
                sample_velocities[sample_index, event_index] = state.velocity
                sample_quaternions[sample_index, event_index] = state.quaternion
                sample_gyro_bias[sample_index, event_index] = state.gyro_bias
                sample_accelerometer_bias[sample_index, event_index] = (
                    state.accelerometer_bias
                )
            for event_index in range(1, event_count):
                if (
                    np.dot(
                        sample_quaternions[sample_index, event_index - 1],
                        sample_quaternions[sample_index, event_index],
                    )
                    < 0.0
                ):
                    sample_quaternions[sample_index, event_index] *= -1.0
        sample_angular_velocity = (
            gyro_at_event[None, :, :] - sample_gyro_bias
            if sample_count
            else np.empty((0, event_count, 3))
        )
        return TrajectoryPosterior(
            timestamps=np.asarray(timestamps),
            position_world=positions,
            velocity_world=velocities,
            velocity_body=velocities_body,
            quaternion_xyzw=quaternions,
            angular_velocity_body=angular_velocity,
            accelerometer_bias_body=accelerometer_bias,
            gyro_bias_body=gyro_bias,
            covariance=np.asarray(covariances),
            sample_ids=np.arange(sample_count, dtype=np.uint64),
            sample_weights=(
                np.full(sample_count, 1.0 / sample_count)
                if sample_count
                else np.empty(0)
            ),
            sample_position_world=sample_positions,
            sample_velocity_world=sample_velocities,
            sample_quaternion_xyzw=sample_quaternions,
            sample_angular_velocity_body=sample_angular_velocity,
            sample_accelerometer_bias_body=sample_accelerometer_bias,
            sample_gyro_bias_body=sample_gyro_bias,
            mocap_used=mocap_used,
            mocap_rejected=mocap_rejected,
            imu_used=imu_used,
            is_smoothed=is_smoothed,
            causal_cutoff=float(cutoff),
            sampling_approximation=(
                "shared_whitened_filter_marginals"
                if not is_smoothed
                else "linearized_backward_simulation"
            ),
        )


def smooth_trajectory(
    observations: TrajectoryObservations,
    config: SmootherConfig = SmootherConfig(),
    online_prefix: bool = False,
    cutoff: Optional[float] = None,
) -> TrajectoryPosterior:
    """Common backend entry point used by bag pipelines and tests."""

    return ErrorStateEkfRtsSmoother(config).smooth(
        observations, online_prefix=online_prefix, cutoff=cutoff
    )


__all__ = [
    "ERROR_STATE_SIZE",
    "ErrorStateEkfRtsSmoother",
    "SmootherConfig",
    "TrajectoryObservations",
    "TrajectoryPosterior",
    "smooth_trajectory",
]
