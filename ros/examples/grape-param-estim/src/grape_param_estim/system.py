"""Data contracts for the Grape closed-loop forecast operator."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from grape_param_estim.geometry import normalise_quaternion


GRAVITY = 9.80665


def _finite_vector(
    value: Sequence[float], size: int, name: str
) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError("{} must contain {} finite values".format(name, size))
    return result.copy()


def _positive_vector(
    value: Sequence[float], size: int, name: str, allow_zero: bool = False
) -> np.ndarray:
    result = _finite_vector(value, size, name)
    invalid = result < 0.0 if allow_zero else result <= 0.0
    if np.any(invalid):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError("{} must be {}".format(name, qualifier))
    return result


@dataclass(frozen=True)
class GrapeGeometry:
    """The four one-DoF vectoring rotors used by the Grape controller."""

    rotor_origins: np.ndarray
    arm_yaws: np.ndarray
    rotor_directions: np.ndarray
    moment_force_rate: float = -0.0181
    thrust_offset: float = 0.056

    def __post_init__(self) -> None:
        origins = np.asarray(self.rotor_origins, dtype=float)
        yaws = np.asarray(self.arm_yaws, dtype=float)
        directions = np.asarray(self.rotor_directions, dtype=float)
        if origins.shape != (4, 3) or not np.all(np.isfinite(origins)):
            raise ValueError("rotor_origins must be a finite 4 by 3 array")
        if yaws.shape != (4,) or not np.all(np.isfinite(yaws)):
            raise ValueError("arm_yaws must contain four finite values")
        if directions.shape != (4,) or not np.all(np.isfinite(directions)):
            raise ValueError(
                "rotor_directions must contain four finite values"
            )
        if (
            not np.isfinite(self.moment_force_rate)
            or not np.isfinite(self.thrust_offset)
            or self.thrust_offset < 0.0
        ):
            raise ValueError("moment_force_rate and thrust_offset are invalid")
        object.__setattr__(self, "rotor_origins", origins.copy())
        object.__setattr__(self, "arm_yaws", yaws.copy())
        object.__setattr__(self, "rotor_directions", directions.copy())

    @classmethod
    def grape(cls):
        nominal_cog = np.asarray(
            (-0.002024708562282, -0.000030526578941,
             0.009509749599446),
            dtype=float,
        )
        origins_main = np.asarray(
            (
                (-0.22309, -0.22309, 0.0),
                (0.22309, -0.22309, 0.0),
                (0.22309, 0.22309, 0.0),
                (-0.22309, 0.22309, 0.0),
            ),
            dtype=float,
        )
        return cls(
            # ``getRotorsOriginFromCog`` returns each thrust-link origin.
            # At q=0 it is 56 mm above the one-DoF gimbal origin.
            rotor_origins=(
                origins_main + np.asarray((0.0, 0.0, 0.056))
                - nominal_cog
            ),
            arm_yaws=np.asarray((-2.3562, -0.7854, 0.7854, 2.3562)),
            rotor_directions=np.asarray((-1.0, 1.0, -1.0, 1.0)),
            moment_force_rate=-0.0181,
            thrust_offset=0.056,
        )

    def thrust_origins(self, gimbal_angles: Sequence[float]) -> np.ndarray:
        """Return articulated thrust-link origins about the nominal CoG."""

        angles = _finite_vector(gimbal_angles, 4, "gimbal_angles")
        result = self.rotor_origins.copy()
        for rotor, angle in enumerate(angles):
            local_displacement = np.asarray(
                (
                    0.0,
                    -self.thrust_offset * np.sin(angle),
                    self.thrust_offset * (np.cos(angle) - 1.0),
                )
            )
            cosine = float(np.cos(self.arm_yaws[rotor]))
            sine = float(np.sin(self.arm_yaws[rotor]))
            result[rotor] += np.asarray(
                (
                    cosine * local_displacement[0]
                    - sine * local_displacement[1],
                    sine * local_displacement[0]
                    + cosine * local_displacement[1],
                    local_displacement[2],
                )
            )
        return result


@dataclass(frozen=True)
class RigidBodyState:
    position: np.ndarray
    orientation_xyzw: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position", _finite_vector(self.position, 3, "position")
        )
        object.__setattr__(
            self,
            "orientation_xyzw",
            normalise_quaternion(self.orientation_xyzw),
        )
        object.__setattr__(
            self,
            "linear_velocity",
            _finite_vector(self.linear_velocity, 3, "linear_velocity"),
        )
        object.__setattr__(
            self,
            "angular_velocity",
            _finite_vector(self.angular_velocity, 3, "angular_velocity"),
        )

    def as_vector(self) -> np.ndarray:
        return np.concatenate(
            (
                self.position,
                self.orientation_xyzw,
                self.linear_velocity,
                self.angular_velocity,
            )
        )

    @classmethod
    def from_vector(cls, value: Sequence[float]):
        vector = _finite_vector(value, 13, "rigid-body state")
        return cls(vector[:3], vector[3:7], vector[7:10], vector[10:13])


@dataclass(frozen=True)
class ReferenceState:
    position: np.ndarray
    linear_velocity: np.ndarray
    linear_acceleration: np.ndarray
    rpy: np.ndarray
    angular_velocity: np.ndarray
    angular_acceleration: np.ndarray

    def __post_init__(self) -> None:
        for field_name in (
            "position",
            "linear_velocity",
            "linear_acceleration",
            "rpy",
            "angular_velocity",
            "angular_acceleration",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_vector(getattr(self, field_name), 3, field_name),
            )


@dataclass(frozen=True)
class VehicleParameters:
    mass: float
    inertia: np.ndarray
    cog_offset: np.ndarray
    force_effectiveness: np.ndarray
    torque_effectiveness: np.ndarray
    linear_drag: np.ndarray
    angular_drag: np.ndarray

    def __post_init__(self) -> None:
        mass = float(self.mass)
        inertia = np.asarray(self.inertia, dtype=float)
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("mass must be finite and positive")
        if (
            inertia.shape != (3, 3)
            or not np.all(np.isfinite(inertia))
            or not np.allclose(inertia, inertia.T, atol=1.0e-12)
            or np.any(np.linalg.eigvalsh(inertia) <= 0.0)
        ):
            raise ValueError("inertia must be symmetric positive definite")
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "inertia", inertia.copy())
        object.__setattr__(
            self,
            "cog_offset",
            _finite_vector(self.cog_offset, 3, "cog_offset"),
        )
        object.__setattr__(
            self,
            "force_effectiveness",
            _positive_vector(
                self.force_effectiveness, 4, "force_effectiveness"
            ),
        )
        object.__setattr__(
            self,
            "torque_effectiveness",
            _positive_vector(
                self.torque_effectiveness, 4, "torque_effectiveness"
            ),
        )
        object.__setattr__(
            self,
            "linear_drag",
            _positive_vector(
                self.linear_drag, 3, "linear_drag", allow_zero=True
            ),
        )
        object.__setattr__(
            self,
            "angular_drag",
            _positive_vector(
                self.angular_drag, 3, "angular_drag", allow_zero=True
            ),
        )

    @classmethod
    def nominal(cls):
        return cls(
            mass=2.3515975908123767,
            inertia=np.asarray(
                (
                    (0.065000061483315, -7.27899253e-7, 1.9015080033e-5),
                    (-7.27899253e-7, 0.064952656340165, 5.9167305e-8),
                    (1.9015080033e-5, 5.9167305e-8, 0.128992110664428),
                ),
                dtype=float,
            ),
            cog_offset=np.zeros(3),
            force_effectiveness=np.ones(4),
            torque_effectiveness=np.ones(4),
            linear_drag=np.zeros(3),
            angular_drag=np.zeros(3),
        )


@dataclass(frozen=True)
class ActuatorParameters:
    thrust_time_constant: float = 0.0
    gimbal_time_constant: float = 0.0
    delay: float = 0.0
    minimum_thrust: float = 1.5
    maximum_thrust: float = 27.6145
    maximum_gimbal_angle: float = 3.14
    maximum_gimbal_rate: float = 6.0

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.thrust_time_constant,
                self.gimbal_time_constant,
                self.delay,
                self.minimum_thrust,
                self.maximum_thrust,
                self.maximum_gimbal_angle,
                self.maximum_gimbal_rate,
            ),
            dtype=float,
        )
        if np.any(~np.isfinite(values)) or np.any(values[:4] < 0.0):
            raise ValueError("actuator parameters must be finite")
        if self.maximum_thrust <= self.minimum_thrust:
            raise ValueError("maximum_thrust must exceed minimum_thrust")
        if (
            self.maximum_gimbal_angle <= 0.0
            or self.maximum_gimbal_rate <= 0.0
        ):
            raise ValueError("gimbal angle and rate limits must be positive")


@dataclass(frozen=True)
class ActuatorCommand:
    thrust: np.ndarray
    gimbal_angle: np.ndarray
    virtual_force: np.ndarray
    desired_acceleration: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "thrust", _finite_vector(self.thrust, 4, "thrust")
        )
        object.__setattr__(
            self,
            "gimbal_angle",
            _finite_vector(self.gimbal_angle, 4, "gimbal_angle"),
        )
        object.__setattr__(
            self,
            "virtual_force",
            _finite_vector(self.virtual_force, 8, "virtual_force"),
        )
        object.__setattr__(
            self,
            "desired_acceleration",
            _finite_vector(
                self.desired_acceleration, 6, "desired_acceleration"
            ),
        )


@dataclass(frozen=True)
class ActuatorState:
    thrust: np.ndarray
    gimbal_angle: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "thrust", _finite_vector(self.thrust, 4, "thrust")
        )
        object.__setattr__(
            self,
            "gimbal_angle",
            _finite_vector(self.gimbal_angle, 4, "gimbal_angle"),
        )


@dataclass(frozen=True)
class ControllerState:
    integral_error: np.ndarray
    roll_pitch_integration_active: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "integral_error",
            _finite_vector(self.integral_error, 6, "integral_error"),
        )


@dataclass(frozen=True)
class ClosedLoopTrajectory:
    times: np.ndarray
    position: np.ndarray
    orientation_xyzw: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    controller_integral: np.ndarray
    commanded_thrust: np.ndarray
    commanded_gimbal_angle: np.ndarray
    actuator_thrust: np.ndarray
    actuator_gimbal_angle: np.ndarray
    body_wrench: np.ndarray

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        if (
            times.ndim != 1
            or times.size < 2
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("times must be a strictly increasing vector")
        expected = {
            "position": (times.size, 3),
            "orientation_xyzw": (times.size, 4),
            "linear_velocity": (times.size, 3),
            "angular_velocity": (times.size, 3),
            "controller_integral": (times.size, 6),
            "commanded_thrust": (times.size, 4),
            "commanded_gimbal_angle": (times.size, 4),
            "actuator_thrust": (times.size, 4),
            "actuator_gimbal_angle": (times.size, 4),
            "body_wrench": (times.size, 6),
        }
        object.__setattr__(self, "times", times.copy())
        for field_name, shape in expected.items():
            value = np.asarray(getattr(self, field_name), dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(
                    "{} must be a finite {} array".format(field_name, shape)
                )
            object.__setattr__(self, field_name, value.copy())


@dataclass(frozen=True)
class PoseObservations:
    """The complete observation contract: time, position and orientation."""

    times: np.ndarray
    position: np.ndarray
    orientation_xyzw: np.ndarray
    translation_covariance: np.ndarray
    rotation_covariance: np.ndarray

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        position = np.asarray(self.position, dtype=float)
        orientation = np.asarray(self.orientation_xyzw, dtype=float)
        if (
            times.ndim != 1
            or times.size < 2
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
            or position.shape != (times.size, 3)
            or not np.all(np.isfinite(position))
        ):
            raise ValueError("pose observation times and positions must align")
        if (
            orientation.shape != (times.size, 4)
            or not np.all(np.isfinite(orientation))
        ):
            raise ValueError("pose observation orientations must align")
        for name in ("translation_covariance", "rotation_covariance"):
            covariance = np.asarray(getattr(self, name), dtype=float)
            if (
                covariance.shape != (3, 3)
                or not np.all(np.isfinite(covariance))
                or not np.allclose(covariance, covariance.T)
                or np.any(np.linalg.eigvalsh(covariance) < 0.0)
            ):
                raise ValueError("{} must be positive semidefinite".format(name))
            object.__setattr__(self, name, covariance.copy())
        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "position", position.copy())
        object.__setattr__(
            self,
            "orientation_xyzw",
            np.asarray(
                [normalise_quaternion(value) for value in orientation]
            ),
        )
