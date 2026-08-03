"""Euclidean/SO(3) chart for the 32-D stochastic Grape filter state."""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence, Tuple

import numpy as np

from grape_param_estim.geometry import (
    matrix_to_quaternion,
    normalise_quaternion,
    quaternion_to_matrix,
    rotation_matrix_from_vector,
    rotation_vector_from_matrix,
)
from grape_param_estim.system import (
    ActuatorState,
    ControllerState,
    RigidBodyState,
)


FILTER_STATE_DIMENSION = 32
POSE_COORDINATE_DIMENSION = 6


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or np.any(~np.isfinite(result)):
        raise ValueError(
            "{} must contain {} finite values".format(name, size)
        )
    return result.copy()


def _canonical_quaternion(value: Sequence[float]) -> np.ndarray:
    result = normalise_quaternion(value)
    tolerance = 16.0 * np.finfo(float).eps
    if abs(result[3]) <= tolerance:
        for component in result[:3]:
            if abs(component) <= tolerance:
                continue
            if component < 0.0:
                result = -result
            break
    return result


@dataclass(frozen=True)
class GrapeFilterStateLayout:
    """Fixed coordinate slices for one dynamic filter state."""

    @property
    def position_slice(self) -> slice:
        return slice(0, 3)

    @property
    def orientation_tangent_slice(self) -> slice:
        return slice(3, 6)

    @property
    def linear_velocity_slice(self) -> slice:
        return slice(6, 9)

    @property
    def angular_velocity_slice(self) -> slice:
        return slice(9, 12)

    @property
    def controller_integral_slice(self) -> slice:
        return slice(12, 18)

    @property
    def actuator_thrust_slice(self) -> slice:
        return slice(18, 22)

    @property
    def actuator_gimbal_slice(self) -> slice:
        return slice(22, 26)

    @property
    def residual_wrench_slice(self) -> slice:
        return slice(26, 32)

    @property
    def dimension(self) -> int:
        return FILTER_STATE_DIMENSION


GRAPE_FILTER_STATE_LAYOUT = GrapeFilterStateLayout()


@dataclass(frozen=True)
class GrapeFilterState:
    """Physical/controller/actuator state plus one body residual wrench."""

    rigid: RigidBodyState
    controller: ControllerState
    actuator: ActuatorState
    residual_wrench: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.rigid, RigidBodyState):
            raise TypeError("rigid must be a RigidBodyState")
        if not isinstance(self.controller, ControllerState):
            raise TypeError("controller must be a ControllerState")
        if not isinstance(self.actuator, ActuatorState):
            raise TypeError("actuator must be an ActuatorState")
        active = self.controller.roll_pitch_integration_active
        if not isinstance(active, (bool, np.bool_)):
            raise TypeError(
                "controller roll_pitch_integration_active must be boolean"
            )
        rigid = RigidBodyState(
            self.rigid.position,
            self.rigid.orientation_xyzw,
            self.rigid.linear_velocity,
            self.rigid.angular_velocity,
        )
        controller = ControllerState(
            self.controller.integral_error, bool(active)
        )
        actuator = ActuatorState(
            self.actuator.thrust, self.actuator.gimbal_angle
        )
        wrench = _finite_vector(
            self.residual_wrench, 6, "residual_wrench"
        )
        object.__setattr__(self, "rigid", rigid)
        object.__setattr__(self, "controller", controller)
        object.__setattr__(self, "actuator", actuator)
        object.__setattr__(self, "residual_wrench", wrench)


def _state_tuple(
    states: Iterable[GrapeFilterState],
) -> Tuple[GrapeFilterState, ...]:
    values = tuple(states)
    if not values:
        raise ValueError("state ensemble cannot be empty")
    if any(not isinstance(value, GrapeFilterState) for value in values):
        raise TypeError("state ensemble must contain GrapeFilterState values")
    return values


def orientation_anchor_from_quaternions(
    quaternions_xyzw: Iterable[Sequence[float]],
) -> np.ndarray:
    """Return a sign- and permutation-invariant Markley orientation anchor."""

    quaternions = tuple(
        _canonical_quaternion(value) for value in quaternions_xyzw
    )
    if not quaternions:
        raise ValueError("orientation ensemble cannot be empty")
    outer_products = tuple(
        sorted(
            (np.outer(value, value) for value in quaternions),
            key=lambda matrix: tuple(
                float(item) for item in matrix.reshape(-1)
            ),
        )
    )
    accumulator = np.empty((4, 4), dtype=float)
    for row in range(4):
        for column in range(4):
            accumulator[row, column] = math.fsum(
                float(value[row, column]) for value in outer_products
            ) / float(len(outer_products))
    accumulator = 0.5 * (accumulator + accumulator.T)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    tolerance = (
        64.0
        * np.finfo(float).eps
        * max(1.0, float(abs(eigenvalues[-1])))
    )
    if eigenvalues[-1] - eigenvalues[-2] <= tolerance:
        raise ValueError(
            "orientation ensemble does not define a unique anchor"
        )
    return _canonical_quaternion(eigenvectors[:, -1])


@dataclass(frozen=True)
class GrapeFilterStateChart:
    """Chart using absolute Euclidean fields and one shared SO(3) anchor."""

    orientation_anchor_xyzw: np.ndarray

    def __post_init__(self) -> None:
        anchor = _canonical_quaternion(self.orientation_anchor_xyzw)
        object.__setattr__(self, "orientation_anchor_xyzw", anchor)
        object.__setattr__(
            self, "_orientation_anchor", quaternion_to_matrix(anchor)
        )

    @classmethod
    def from_ensemble(
        cls, states: Iterable[GrapeFilterState]
    ) -> "GrapeFilterStateChart":
        values = _state_tuple(states)
        return cls(
            orientation_anchor_from_quaternions(
                value.rigid.orientation_xyzw for value in values
            )
        )

    @property
    def layout(self) -> GrapeFilterStateLayout:
        return GRAPE_FILTER_STATE_LAYOUT

    @property
    def orientation_anchor_matrix(self) -> np.ndarray:
        return self._orientation_anchor.copy()

    def _orientation_coordinates(
        self, orientation_xyzw: Sequence[float]
    ) -> np.ndarray:
        rotation = quaternion_to_matrix(orientation_xyzw)
        return rotation_vector_from_matrix(
            self._orientation_anchor.T @ rotation
        )

    def _decode_orientation(
        self, tangent: Sequence[float]
    ) -> np.ndarray:
        value = _finite_vector(tangent, 3, "orientation tangent")
        return matrix_to_quaternion(
            self._orientation_anchor @ rotation_matrix_from_vector(value)
        )

    def encode(self, state: GrapeFilterState) -> np.ndarray:
        """Encode one state into the fixed 32-D layout."""

        if not isinstance(state, GrapeFilterState):
            raise TypeError("state must be a GrapeFilterState")
        layout = self.layout
        result = np.empty(layout.dimension, dtype=float)
        result[layout.position_slice] = state.rigid.position
        result[layout.orientation_tangent_slice] = (
            self._orientation_coordinates(state.rigid.orientation_xyzw)
        )
        result[layout.linear_velocity_slice] = state.rigid.linear_velocity
        result[layout.angular_velocity_slice] = state.rigid.angular_velocity
        result[layout.controller_integral_slice] = (
            state.controller.integral_error
        )
        result[layout.actuator_thrust_slice] = state.actuator.thrust
        result[layout.actuator_gimbal_slice] = state.actuator.gimbal_angle
        result[layout.residual_wrench_slice] = state.residual_wrench
        if np.any(~np.isfinite(result)):
            raise ValueError("encoded filter state is not finite")
        return result

    def decode(
        self,
        coordinates: Sequence[float],
        template: GrapeFilterState,
    ) -> GrapeFilterState:
        """Decode one state while preserving the template controller flag."""

        if not isinstance(template, GrapeFilterState):
            raise TypeError("template must be a GrapeFilterState")
        value = _finite_vector(
            coordinates, self.layout.dimension, "filter coordinates"
        )
        layout = self.layout
        return GrapeFilterState(
            rigid=RigidBodyState(
                position=value[layout.position_slice],
                orientation_xyzw=self._decode_orientation(
                    value[layout.orientation_tangent_slice]
                ),
                linear_velocity=value[layout.linear_velocity_slice],
                angular_velocity=value[layout.angular_velocity_slice],
            ),
            controller=ControllerState(
                integral_error=value[layout.controller_integral_slice],
                roll_pitch_integration_active=(
                    template.controller.roll_pitch_integration_active
                ),
            ),
            actuator=ActuatorState(
                thrust=value[layout.actuator_thrust_slice],
                gimbal_angle=value[layout.actuator_gimbal_slice],
            ),
            residual_wrench=value[layout.residual_wrench_slice],
        )

    def encode_ensemble(
        self, states: Iterable[GrapeFilterState]
    ) -> np.ndarray:
        """Encode a non-empty ensemble with member-first shape ``(M, 32)``."""

        values = _state_tuple(states)
        return np.asarray([self.encode(value) for value in values])

    def decode_ensemble(
        self,
        coordinates: np.ndarray,
        templates: Iterable[GrapeFilterState],
    ) -> Tuple[GrapeFilterState, ...]:
        """Decode member-first coordinates using one template per member."""

        values = np.asarray(coordinates, dtype=float)
        selected_templates = _state_tuple(templates)
        if (
            values.shape
            != (len(selected_templates), self.layout.dimension)
            or np.any(~np.isfinite(values))
        ):
            raise ValueError(
                "filter ensemble coordinates must have finite member-first "
                "shape (M, 32)"
            )
        return tuple(
            self.decode(value, template)
            for value, template in zip(values, selected_templates)
        )

    def pose_coordinates(
        self,
        position: Sequence[float],
        orientation_xyzw: Sequence[float],
    ) -> np.ndarray:
        """Encode a pose into absolute position and anchored SO(3) tangent."""

        return np.concatenate(
            (
                _finite_vector(position, 3, "pose position"),
                self._orientation_coordinates(orientation_xyzw),
            )
        )

    def predicted_pose_coordinates(
        self, state: GrapeFilterState
    ) -> np.ndarray:
        if not isinstance(state, GrapeFilterState):
            raise TypeError("state must be a GrapeFilterState")
        return self.pose_coordinates(
            state.rigid.position, state.rigid.orientation_xyzw
        )

    def predicted_pose_ensemble(
        self, states: Iterable[GrapeFilterState]
    ) -> np.ndarray:
        """Return predicted pose coordinates with shape ``(M, 6)``."""

        values = _state_tuple(states)
        return np.asarray(
            [self.predicted_pose_coordinates(value) for value in values]
        )

    def observed_pose_coordinates(
        self,
        position: Sequence[float],
        orientation_xyzw: Sequence[float],
    ) -> np.ndarray:
        """Encode one observation in the same 6-D chart as predictions."""

        return self.pose_coordinates(position, orientation_xyzw)


__all__ = [
    "FILTER_STATE_DIMENSION",
    "GRAPE_FILTER_STATE_LAYOUT",
    "POSE_COORDINATE_DIMENSION",
    "GrapeFilterState",
    "GrapeFilterStateChart",
    "GrapeFilterStateLayout",
    "orientation_anchor_from_quaternions",
]
