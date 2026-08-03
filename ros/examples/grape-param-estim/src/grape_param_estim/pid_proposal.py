"""Member-wise PID compensation derived from the physical posterior."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.controller import acceleration_allocation_matrix
from grape_param_estim.controller_config import (
    PID_GROUPS,
    PidGainConfiguration,
)
from grape_param_estim.system import GrapeGeometry, VehicleParameters


GROUP_AXIS_INDICES = ((0, 1), (2,), (3, 4), (5,))


def _rotate_z(vector: np.ndarray, angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.asarray(
        (
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
            vector[2],
        )
    )


def physical_acceleration_allocation_matrix(
    parameters: VehicleParameters,
    geometry: GrapeGeometry,
    gimbal_angles: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Map controller virtual rotor-force components to real accelerations."""

    if not isinstance(parameters, VehicleParameters):
        raise TypeError("parameters must be VehicleParameters")
    if not isinstance(geometry, GrapeGeometry):
        raise TypeError("geometry must be GrapeGeometry")
    selected_angles = (
        np.zeros(4)
        if gimbal_angles is None
        else np.asarray(gimbal_angles, dtype=float)
    )
    if selected_angles.shape != (4,) or np.any(~np.isfinite(selected_angles)):
        raise ValueError("gimbal angles must contain four finite values")
    origins = geometry.thrust_origins(selected_angles)
    inverse_inertia = np.linalg.inv(parameters.inertia)
    result = np.zeros((6, 8), dtype=float)
    local_basis = (
        np.asarray((0.0, 1.0, 0.0)),
        np.asarray((0.0, 0.0, 1.0)),
    )
    for rotor in range(4):
        for component, local_force in enumerate(local_basis):
            column = 2 * rotor + component
            force = (
                parameters.force_effectiveness[rotor]
                * _rotate_z(local_force, geometry.arm_yaws[rotor])
            )
            origin = origins[rotor] - parameters.cog_offset
            torque = np.cross(origin, force) + (
                parameters.torque_effectiveness[rotor]
                * geometry.rotor_directions[rotor]
                * geometry.moment_force_rate
                * force
            )
            result[:3, column] = force / parameters.mass
            result[3:, column] = inverse_inertia @ torque
    return result


def closed_loop_acceleration_response(
    physical_parameters: VehicleParameters,
    controller_nominal_parameters: VehicleParameters,
    geometry: GrapeGeometry,
) -> np.ndarray:
    """Return the algebraic real response to nominal generalized acceleration."""

    nominal = acceleration_allocation_matrix(
        controller_nominal_parameters, geometry
    )
    physical = physical_acceleration_allocation_matrix(
        physical_parameters, geometry
    )
    return physical @ np.linalg.pinv(nominal, rcond=1.0e-4)


def group_compensation_scales(response: np.ndarray) -> np.ndarray:
    """Least-squares positive scales for xy/z/roll-pitch/yaw subspaces."""

    matrix = np.asarray(response, dtype=float)
    if matrix.shape != (6, 6) or np.any(~np.isfinite(matrix)):
        raise ValueError("closed-loop response must be a finite 6x6 matrix")
    identity = np.eye(6)
    scales = []
    for axes in GROUP_AXIS_INDICES:
        indices = np.asarray(axes, dtype=int)
        actual = matrix[:, indices]
        target = identity[:, indices]
        denominator = float(np.sum(actual * actual))
        numerator = float(np.sum(actual * target))
        if denominator <= 0.0 or numerator <= 0.0:
            raise ValueError(
                "physical member has no positive response for {}".format(
                    PID_GROUPS[len(scales)]
                )
            )
        scales.append(numerator / denominator)
    values = np.asarray(scales)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("PID compensation scales must be finite and positive")
    return values


@dataclass(frozen=True)
class PidProposalEnsemble:
    member_id: np.ndarray
    source_mode_id: Tuple[str, ...]
    group_scales: np.ndarray
    exact_gain_values: np.ndarray
    source_member_id: np.ndarray
    current: PidGainConfiguration
    constant_delay: np.ndarray
    acceleration_response: np.ndarray

    def __post_init__(self) -> None:
        member_id = np.asarray(self.member_id, dtype=np.int64)
        source_mode_id = tuple(str(value) for value in self.source_mode_id)
        scales = np.asarray(self.group_scales, dtype=float)
        exact = np.asarray(self.exact_gain_values, dtype=float)
        source_member_id = np.asarray(self.source_member_id, dtype=np.int64)
        delay = np.asarray(self.constant_delay, dtype=float)
        response = np.asarray(self.acceleration_response, dtype=float)
        count = member_id.size
        if (
            count < 1
            or member_id.shape != (count,)
            or np.unique(member_id).size != count
            or len(source_mode_id) != count
            or any(not value for value in source_mode_id)
            or scales.shape != (count, len(PID_GROUPS))
            or exact.shape != (count, len(PID_GROUPS), 3)
            or source_member_id.shape != (count,)
            or not np.array_equal(source_member_id, member_id)
            or delay.shape != (count,)
            or response.shape != (count, 6, 6)
            or np.any(~np.isfinite(scales))
            or np.any(scales <= 0.0)
            or np.any(~np.isfinite(exact))
            or np.any(exact < 0.0)
            or np.any(~np.isfinite(delay))
            or np.any(delay < 0.0)
            or np.any(~np.isfinite(response))
            or not isinstance(self.current, PidGainConfiguration)
        ):
            raise ValueError("PID proposal ensemble is not member-aligned")
        expected = self.current.values[None, :, :] * scales[:, :, None]
        if not np.allclose(exact, expected, rtol=1.0e-12, atol=1.0e-14):
            raise ValueError("exact PID proposals must preserve group P:I:D ratios")
        object.__setattr__(self, "member_id", member_id.copy())
        object.__setattr__(self, "source_mode_id", source_mode_id)
        object.__setattr__(self, "group_scales", scales.copy())
        object.__setattr__(self, "exact_gain_values", exact.copy())
        object.__setattr__(self, "source_member_id", source_member_id.copy())
        object.__setattr__(self, "constant_delay", delay.copy())
        object.__setattr__(self, "acceleration_response", response.copy())

    def member_index(self, member_id: int) -> int:
        indices = np.flatnonzero(self.member_id == int(member_id))
        if indices.size != 1:
            raise KeyError("unknown proposal source member")
        return int(indices[0])

    def configuration_for_member(self, member_id: int) -> PidGainConfiguration:
        return PidGainConfiguration(
            self.exact_gain_values[self.member_index(member_id)]
        )

    def percentile_ranges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return raw member 50% and 95% marginal intervals, not a candidate."""

        return (
            np.percentile(self.exact_gain_values, (25.0, 75.0), axis=0),
            np.percentile(self.exact_gain_values, (2.5, 97.5), axis=0),
        )


@dataclass(frozen=True)
class PidGainCandidate:
    candidate_id: str
    source: str
    configuration: PidGainConfiguration
    source_member_id: int = -1
    source_mode_id: str = ""

    def __post_init__(self) -> None:
        identifier = str(self.candidate_id)
        source = str(self.source)
        if not identifier:
            raise ValueError("candidate ID cannot be empty")
        if source not in {"current", "member-derived", "user"}:
            raise ValueError("unknown PID candidate source")
        if not isinstance(self.configuration, PidGainConfiguration):
            raise TypeError("candidate configuration has the wrong type")
        member = int(self.source_member_id)
        mode = str(self.source_mode_id)
        if source == "member-derived" and (member < 0 or not mode):
            raise ValueError("member-derived candidates need member and mode IDs")
        if source != "member-derived" and (member != -1 or mode):
            raise ValueError("only member-derived candidates have source members")
        object.__setattr__(self, "candidate_id", identifier)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "source_member_id", member)
        object.__setattr__(self, "source_mode_id", mode)


def derive_pid_proposal_ensemble(
    member_id: Sequence[int],
    physical_parameter_members: Sequence[VehicleParameters],
    constant_delay: Sequence[float],
    source_mode_id: Sequence[str],
    controller_nominal_parameters: VehicleParameters,
    geometry: GrapeGeometry,
    current: PidGainConfiguration,
) -> PidProposalEnsemble:
    """Derive one correlated four-group PID proposal for every raw member."""

    ids = np.asarray(member_id, dtype=np.int64)
    physical = tuple(physical_parameter_members)
    delays = np.asarray(constant_delay, dtype=float)
    modes = tuple(str(value) for value in source_mode_id)
    count = ids.size
    if (
        count < 1
        or len(physical) != count
        or delays.shape != (count,)
        or len(modes) != count
    ):
        raise ValueError("physical posterior inputs must stay member-aligned")
    responses = np.asarray(
        [
            closed_loop_acceleration_response(
                parameters, controller_nominal_parameters, geometry
            )
            for parameters in physical
        ]
    )
    scales = np.asarray(
        [group_compensation_scales(value) for value in responses]
    )
    exact = current.values[None, :, :] * scales[:, :, None]
    return PidProposalEnsemble(
        member_id=ids,
        source_mode_id=modes,
        group_scales=scales,
        exact_gain_values=exact,
        source_member_id=ids.copy(),
        current=current,
        constant_delay=delays,
        acceleration_response=responses,
    )


def current_pid_candidate(current: PidGainConfiguration) -> PidGainCandidate:
    return PidGainCandidate("current", "current", current)


def member_pid_candidate(
    proposals: PidProposalEnsemble, member_id: int
) -> PidGainCandidate:
    index = proposals.member_index(member_id)
    return PidGainCandidate(
        candidate_id="member_{}".format(int(member_id)),
        source="member-derived",
        configuration=PidGainConfiguration(
            proposals.exact_gain_values[index]
        ),
        source_member_id=int(member_id),
        source_mode_id=proposals.source_mode_id[index],
    )


def user_pid_candidate(
    candidate_id: str, configuration: PidGainConfiguration
) -> PidGainCandidate:
    return PidGainCandidate(str(candidate_id), "user", configuration)


__all__ = [
    "GROUP_AXIS_INDICES",
    "PidGainCandidate",
    "PidProposalEnsemble",
    "closed_loop_acceleration_response",
    "current_pid_candidate",
    "derive_pid_proposal_ensemble",
    "group_compensation_scales",
    "member_pid_candidate",
    "physical_acceleration_allocation_matrix",
    "user_pid_candidate",
]
