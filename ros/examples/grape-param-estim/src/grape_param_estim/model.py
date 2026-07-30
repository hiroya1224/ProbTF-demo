"""Minimal rigid-body model and short-segment replay for Grape."""

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from grape_param_estim.data import AnalysisData


GIMBAL_ORIGINS_MAIN = np.asarray(
    (
        (-0.22309, -0.22309, 0.0),
        (0.22309, -0.22309, 0.0),
        (0.22309, 0.22309, 0.0),
        (-0.22309, 0.22309, 0.0),
    ),
    dtype=float,
)
# Zero-joint CoG calculated from the source Grape URDF. The minimal model
# intentionally keeps it fixed while replaying the articulated vehicle.
NOMINAL_COG_MAIN = np.asarray(
    (-0.002024743002579445, -0.000030527098198356, 0.009509911360307566),
    dtype=float,
)
GIMBAL_ORIGINS_COG = GIMBAL_ORIGINS_MAIN - NOMINAL_COG_MAIN
ARM_YAWS = np.asarray((-2.3562, -0.7854, 0.7854, 2.3562), dtype=float)
ROTOR_DIRECTIONS = np.asarray((-1.0, 1.0, -1.0, 1.0), dtype=float)
THRUST_OFFSET = 0.056
MOMENT_FORCE_RATE = -0.0181
STANDARD_GRAVITY = 9.80665


def _rotate_z(vector: np.ndarray, angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    x_value, y_value, z_value = vector
    return np.asarray(
        (
            cosine * x_value - sine * y_value,
            sine * x_value + cosine * y_value,
            z_value,
        ),
        dtype=float,
    )


def command_to_wrench(
    base_thrust: Sequence[float],
    gimbal_angle: Sequence[float],
) -> np.ndarray:
    """Map four rotor/gimbal commands to body ``[F, tau]`` about the CoG."""

    thrust = np.asarray(base_thrust, dtype=float)
    angle = np.asarray(gimbal_angle, dtype=float)
    if thrust.shape != (4,) or angle.shape != (4,):
        raise ValueError("base_thrust and gimbal_angle must contain 4 values")
    if not np.all(np.isfinite(thrust)) or not np.all(np.isfinite(angle)):
        raise ValueError("recorded command values must be finite")

    wrench = np.zeros(6, dtype=float)
    for rotor in range(4):
        sine = float(np.sin(angle[rotor]))
        cosine = float(np.cos(angle[rotor]))
        force_arm = np.asarray(
            (0.0, -thrust[rotor] * sine, thrust[rotor] * cosine),
            dtype=float,
        )
        force = _rotate_z(force_arm, ARM_YAWS[rotor])
        thrust_point_arm = np.asarray(
            (0.0, -THRUST_OFFSET * sine, THRUST_OFFSET * cosine),
            dtype=float,
        )
        thrust_point = (
            GIMBAL_ORIGINS_COG[rotor]
            + _rotate_z(thrust_point_arm, ARM_YAWS[rotor])
        )
        wrench[:3] += force
        wrench[3:] += np.cross(thrust_point, force)
        wrench[3:] += (
            ROTOR_DIRECTIONS[rotor] * MOMENT_FORCE_RATE * force
        )
    return wrench


def quaternion_to_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(float).eps:
        raise ValueError("quaternion norm must be positive")
    x_value, y_value, z_value, w_value = quaternion / norm
    return np.asarray(
        (
            (
                1.0 - 2.0 * (y_value**2 + z_value**2),
                2.0 * (x_value * y_value - z_value * w_value),
                2.0 * (x_value * z_value + y_value * w_value),
            ),
            (
                2.0 * (x_value * y_value + z_value * w_value),
                1.0 - 2.0 * (x_value**2 + z_value**2),
                2.0 * (y_value * z_value - x_value * w_value),
            ),
            (
                2.0 * (x_value * z_value - y_value * w_value),
                2.0 * (y_value * z_value + x_value * w_value),
                1.0 - 2.0 * (x_value**2 + y_value**2),
            ),
        ),
        dtype=float,
    )


def _quaternion_multiply(
    left_xyzw: np.ndarray, right_xyzw: np.ndarray
) -> np.ndarray:
    left_vector = left_xyzw[:3]
    right_vector = right_xyzw[:3]
    left_scalar = left_xyzw[3]
    right_scalar = right_xyzw[3]
    return np.concatenate(
        (
            left_scalar * right_vector
            + right_scalar * left_vector
            + np.cross(left_vector, right_vector),
            np.asarray(
                (
                    left_scalar * right_scalar
                    - np.dot(left_vector, right_vector),
                )
            ),
        )
    )


def rotation_vector_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Return the shortest SO(3) logarithm as a rotation vector."""

    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3 by 3 matrix")
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew_vector = np.asarray(
        (
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ),
        dtype=float,
    )
    if angle < 1.0e-7:
        return 0.5 * skew_vector
    if np.pi - angle < 1.0e-5:
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (matrix + np.eye(3))
        )
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        if np.dot(axis, skew_vector) < 0.0:
            axis *= -1.0
        return angle * axis
    return (0.5 * angle / np.sin(angle)) * skew_vector


def _skew(vector: np.ndarray) -> np.ndarray:
    x_value, y_value, z_value = vector
    return np.asarray(
        (
            (0.0, -z_value, y_value),
            (z_value, 0.0, -x_value),
            (-y_value, x_value, 0.0),
        ),
        dtype=float,
    )


def se3_log(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    """Return ``Log(T)`` ordered as translation then rotation coordinates."""

    rotation_vector = rotation_vector_from_matrix(rotation)
    angle = float(np.linalg.norm(rotation_vector))
    omega = _skew(rotation_vector)
    if angle < 1.0e-7:
        inverse_left_jacobian = (
            np.eye(3) - 0.5 * omega + (1.0 / 12.0) * omega @ omega
        )
    else:
        coefficient = (
            1.0
            - 0.5 * angle / np.tan(0.5 * angle)
        ) / (angle * angle)
        inverse_left_jacobian = (
            np.eye(3) - 0.5 * omega + coefficient * omega @ omega
        )
    tangent_translation = inverse_left_jacobian @ np.asarray(
        translation, dtype=float
    )
    return np.concatenate((tangent_translation, rotation_vector))


@dataclass(frozen=True)
class RigidBodyParameters:
    """Nominal controller model plus Phase-2-compatible scale factors."""

    mass: float
    inertia: np.ndarray
    mass_scale: float = 1.0
    force_scale: float = 1.0
    inertia_scale: float = 1.0
    torque_scale: float = 1.0

    def __post_init__(self) -> None:
        mass = float(self.mass)
        inertia = np.asarray(self.inertia, dtype=float)
        scales = np.asarray(
            (
                self.mass_scale,
                self.force_scale,
                self.inertia_scale,
                self.torque_scale,
            ),
            dtype=float,
        )
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("mass must be finite and positive")
        if (
            inertia.shape != (3, 3)
            or not np.all(np.isfinite(inertia))
            or not np.allclose(inertia, inertia.T, atol=1.0e-12)
            or np.any(np.linalg.eigvalsh(inertia) <= 0.0)
        ):
            raise ValueError("inertia must be symmetric positive definite")
        if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("model scales must be finite and positive")

    @classmethod
    def from_diagonal(
        cls,
        mass: float,
        inertia_diagonal: Sequence[float],
        **scales,
    ):
        diagonal = np.asarray(inertia_diagonal, dtype=float)
        if diagonal.shape != (3,):
            raise ValueError("inertia_diagonal must contain 3 values")
        return cls(mass=mass, inertia=np.diag(diagonal), **scales)

    @property
    def effective_mass(self) -> float:
        return float(self.mass) * float(self.mass_scale)

    @property
    def effective_inertia(self) -> np.ndarray:
        return np.asarray(self.inertia, dtype=float) * float(
            self.inertia_scale
        )


@dataclass(frozen=True)
class SegmentReplay:
    times: np.ndarray
    position: np.ndarray
    orientation_xyzw: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    command_wrench: np.ndarray


@dataclass(frozen=True)
class ReplayResult:
    times: np.ndarray
    position: np.ndarray
    orientation_xyzw: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    command_wrench: np.ndarray
    residual_se3: np.ndarray
    correction_translation: np.ndarray
    correction_rotation_vector: np.ndarray
    segment_id: np.ndarray

    @property
    def translation_residual_norm(self) -> np.ndarray:
        return np.linalg.norm(self.residual_se3[:, :3], axis=1)

    @property
    def rotation_residual_norm(self) -> np.ndarray:
        return np.linalg.norm(self.residual_se3[:, 3:], axis=1)


class MotionModel(Protocol):
    def simulate_segment(
        self,
        data: AnalysisData,
        parameters: RigidBodyParameters,
        segment: slice,
    ) -> SegmentReplay:
        ...


class GrapeRigidBodyModel:
    """Open-loop six-DoF model driven by the command stored in the bag."""

    def __init__(self, maximum_time_step: float = 0.005):
        step = float(maximum_time_step)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("maximum_time_step must be positive")
        self.maximum_time_step = step
        self.gravity = np.asarray((0.0, 0.0, -STANDARD_GRAVITY))

    @staticmethod
    def _normalise_state(state: np.ndarray) -> np.ndarray:
        result = np.asarray(state, dtype=float).copy()
        quaternion_norm = float(np.linalg.norm(result[3:7]))
        if quaternion_norm <= np.finfo(float).eps:
            raise FloatingPointError("simulated quaternion collapsed")
        result[3:7] /= quaternion_norm
        return result

    def _derivative(
        self,
        state: np.ndarray,
        command_wrench: np.ndarray,
        parameters: RigidBodyParameters,
        inertia: np.ndarray,
        inverse_inertia: np.ndarray,
    ) -> np.ndarray:
        quaternion = state[3:7]
        linear_velocity = state[7:10]
        angular_velocity = state[10:13]
        body_force = (
            float(parameters.force_scale) * command_wrench[:3]
        )
        body_torque = (
            float(parameters.torque_scale) * command_wrench[3:]
        )
        quaternion_rate = 0.5 * _quaternion_multiply(
            quaternion,
            np.concatenate((angular_velocity, np.asarray((0.0,)))),
        )
        linear_acceleration = (
            self.gravity
            + quaternion_to_matrix(quaternion)
            @ body_force
            / parameters.effective_mass
        )
        angular_acceleration = inverse_inertia @ (
            body_torque
            - np.cross(angular_velocity, inertia @ angular_velocity)
        )
        return np.concatenate(
            (
                linear_velocity,
                quaternion_rate,
                linear_acceleration,
                angular_acceleration,
            )
        )

    def _rk4_step(
        self,
        state: np.ndarray,
        start_wrench: np.ndarray,
        end_wrench: np.ndarray,
        step: float,
        parameters: RigidBodyParameters,
        inertia: np.ndarray,
        inverse_inertia: np.ndarray,
    ) -> np.ndarray:
        midpoint_wrench = 0.5 * (start_wrench + end_wrench)
        first = self._derivative(
            state,
            start_wrench,
            parameters,
            inertia,
            inverse_inertia,
        )
        second_state = self._normalise_state(state + 0.5 * step * first)
        second = self._derivative(
            second_state,
            midpoint_wrench,
            parameters,
            inertia,
            inverse_inertia,
        )
        third_state = self._normalise_state(state + 0.5 * step * second)
        third = self._derivative(
            third_state,
            midpoint_wrench,
            parameters,
            inertia,
            inverse_inertia,
        )
        fourth_state = self._normalise_state(state + step * third)
        fourth = self._derivative(
            fourth_state,
            end_wrench,
            parameters,
            inertia,
            inverse_inertia,
        )
        return self._normalise_state(
            state
            + (step / 6.0)
            * (first + 2.0 * second + 2.0 * third + fourth)
        )

    def simulate_segment(
        self,
        data: AnalysisData,
        parameters: RigidBodyParameters,
        segment: slice,
    ) -> SegmentReplay:
        times = np.asarray(data.times[segment], dtype=float)
        if times.size < 1:
            raise ValueError("cannot simulate an empty segment")
        command_wrench = np.vstack(
            [
                command_to_wrench(thrust, angle)
                for thrust, angle in zip(
                    data.base_thrust[segment],
                    data.gimbal_angle[segment],
                )
            ]
        )
        states = np.empty((times.size, 13), dtype=float)
        start = int(segment.start or 0)
        states[0] = np.concatenate(
            (
                data.position[start],
                data.orientation_xyzw[start],
                data.linear_velocity[start],
                data.angular_velocity[start],
            )
        )
        states[0] = self._normalise_state(states[0])
        inertia = parameters.effective_inertia
        inverse_inertia = np.linalg.inv(inertia)

        for index in range(1, times.size):
            interval = float(times[index] - times[index - 1])
            substeps = max(
                1, int(np.ceil(interval / self.maximum_time_step))
            )
            step = interval / substeps
            state = states[index - 1]
            for substep in range(substeps):
                start_fraction = substep / substeps
                end_fraction = (substep + 1) / substeps
                start_wrench = (
                    (1.0 - start_fraction) * command_wrench[index - 1]
                    + start_fraction * command_wrench[index]
                )
                end_wrench = (
                    (1.0 - end_fraction) * command_wrench[index - 1]
                    + end_fraction * command_wrench[index]
                )
                state = self._rk4_step(
                    state,
                    start_wrench,
                    end_wrench,
                    step,
                    parameters,
                    inertia,
                    inverse_inertia,
                )
            states[index] = state

        return SegmentReplay(
            times=times,
            position=states[:, 0:3],
            orientation_xyzw=states[:, 3:7],
            linear_velocity=states[:, 7:10],
            angular_velocity=states[:, 10:13],
            command_wrench=command_wrench,
        )


def _segment_residual(
    data: AnalysisData,
    replay: SegmentReplay,
    segment: slice,
) -> tuple:
    observed_position = data.position[segment]
    observed_orientation = data.orientation_xyzw[segment]
    observed_start_rotation = quaternion_to_matrix(observed_orientation[0])
    nominal_start_rotation = quaternion_to_matrix(
        replay.orientation_xyzw[0]
    )
    observed_start_position = observed_position[0]
    nominal_start_position = replay.position[0]

    residual = np.empty((replay.times.size, 6), dtype=float)
    correction_translation = np.empty((replay.times.size, 3), dtype=float)
    correction_rotation = np.empty((replay.times.size, 3), dtype=float)
    for index in range(replay.times.size):
        observed_rotation = quaternion_to_matrix(
            observed_orientation[index]
        )
        nominal_rotation = quaternion_to_matrix(
            replay.orientation_xyzw[index]
        )
        observed_relative_rotation = (
            observed_start_rotation.T @ observed_rotation
        )
        nominal_relative_rotation = (
            nominal_start_rotation.T @ nominal_rotation
        )
        observed_relative_translation = observed_start_rotation.T @ (
            observed_position[index] - observed_start_position
        )
        nominal_relative_translation = nominal_start_rotation.T @ (
            replay.position[index] - nominal_start_position
        )
        residual_rotation = (
            observed_relative_rotation.T @ nominal_relative_rotation
        )
        residual_translation = observed_relative_rotation.T @ (
            nominal_relative_translation - observed_relative_translation
        )
        residual[index] = se3_log(
            residual_rotation, residual_translation
        )

        correction_rotation_matrix = (
            nominal_rotation.T @ observed_rotation
        )
        correction_translation[index] = nominal_rotation.T @ (
            observed_position[index] - replay.position[index]
        )
        correction_rotation[index] = rotation_vector_from_matrix(
            correction_rotation_matrix
        )
    return residual, correction_translation, correction_rotation


def replay_segments(
    data: AnalysisData,
    model: MotionModel,
    parameters: RigidBodyParameters,
) -> ReplayResult:
    """Replay every segment from its observed initial state."""

    sample_count = data.times.size
    position = np.empty((sample_count, 3), dtype=float)
    orientation = np.empty((sample_count, 4), dtype=float)
    linear_velocity = np.empty((sample_count, 3), dtype=float)
    angular_velocity = np.empty((sample_count, 3), dtype=float)
    command_wrench = np.empty((sample_count, 6), dtype=float)
    residual = np.empty((sample_count, 6), dtype=float)
    correction_translation = np.empty((sample_count, 3), dtype=float)
    correction_rotation = np.empty((sample_count, 3), dtype=float)

    for _, segment in data.segments():
        replay = model.simulate_segment(data, parameters, segment)
        position[segment] = replay.position
        orientation[segment] = replay.orientation_xyzw
        linear_velocity[segment] = replay.linear_velocity
        angular_velocity[segment] = replay.angular_velocity
        command_wrench[segment] = replay.command_wrench
        (
            residual[segment],
            correction_translation[segment],
            correction_rotation[segment],
        ) = _segment_residual(data, replay, segment)

    return ReplayResult(
        times=data.times.copy(),
        position=position,
        orientation_xyzw=orientation,
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        command_wrench=command_wrench,
        residual_se3=residual,
        correction_translation=correction_translation,
        correction_rotation_vector=correction_rotation,
        segment_id=data.segment_id.copy(),
    )


__all__ = [
    "GrapeRigidBodyModel",
    "MotionModel",
    "ReplayResult",
    "RigidBodyParameters",
    "SegmentReplay",
    "command_to_wrench",
    "quaternion_to_matrix",
    "replay_segments",
    "rotation_vector_from_matrix",
    "se3_log",
]
