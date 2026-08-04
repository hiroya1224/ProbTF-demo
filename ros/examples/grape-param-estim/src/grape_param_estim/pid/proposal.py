"""Exact PID proposals derived from equal-weight physical posterior samples.

This module deliberately has no legacy compatibility layer.  Every row is
identified by the retained MCMC ``sample_id`` that supplied one exact physical
plant and one delay.  Posterior summaries may be displayed, but are never
turned into a component-wise averaged gain proposal here.
"""

import base64
from dataclasses import dataclass
import re
from typing import Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.controller import acceleration_allocation_matrix
from grape_param_estim.controller_config import (
    PID_GROUPS,
    PidGainConfiguration,
)
from grape_param_estim.system import GrapeGeometry, VehicleParameters


GROUP_AXIS_INDICES = ((0, 1), (2,), (3, 4), (5,))
_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


def _canonical_identifier(value: str, name: str) -> str:
    result = str(value)
    if not _CANDIDATE_ID.match(result):
        raise ValueError("{} must be a safe non-empty identifier".format(name))
    return result


def _canonical_sample_id(value: str) -> str:
    result = str(value)
    if not result or result.strip() != result or "\x00" in result:
        raise ValueError("sample_id must be a canonical non-empty string")
    return result


@dataclass(frozen=True)
class PhysicalPlantSample:
    """One equal-weight retained MCMC draw in physical coordinates."""

    sample_id: str
    parameters: VehicleParameters
    delay: float
    source_mode_id: str

    def __post_init__(self) -> None:
        sample_id = _canonical_sample_id(self.sample_id)
        if not isinstance(self.parameters, VehicleParameters):
            raise TypeError("parameters must be VehicleParameters")
        delay = float(self.delay)
        mode = str(self.source_mode_id)
        if not np.isfinite(delay) or delay < 0.0:
            raise ValueError("delay must be finite and non-negative")
        if not mode or mode.strip() != mode:
            raise ValueError("source_mode_id must be a canonical string")
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "delay", delay)
        object.__setattr__(self, "source_mode_id", mode)


@dataclass(frozen=True)
class PhysicalPlantPosterior:
    """An equal-weight population of physical plants and their delays.

    No weight field is accepted: retained MCMC draws are equal-weight samples.
    Repeated chain states are valid draws, but every stored draw still needs a
    unique artifact-level ``sample_id``.
    """

    samples: Tuple[PhysicalPlantSample, ...]

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if not samples or any(
            not isinstance(value, PhysicalPlantSample) for value in samples
        ):
            raise ValueError("samples must contain physical MCMC samples")
        identifiers = tuple(value.sample_id for value in samples)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("sample_id values must be unique")
        object.__setattr__(self, "samples", samples)

    @classmethod
    def from_aligned_values(
        cls,
        sample_id: Sequence[str],
        physical_parameters: Sequence[VehicleParameters],
        delay: Sequence[float],
        source_mode_id: Sequence[str],
    ) -> "PhysicalPlantPosterior":
        identifiers = tuple(str(value) for value in sample_id)
        parameters = tuple(physical_parameters)
        delays = np.asarray(delay, dtype=float)
        modes = tuple(str(value) for value in source_mode_id)
        count = len(identifiers)
        if (
            count < 1
            or len(parameters) != count
            or delays.shape != (count,)
            or len(modes) != count
        ):
            raise ValueError("physical posterior inputs must remain aligned")
        return cls(
            tuple(
                PhysicalPlantSample(
                    identifiers[index],
                    parameters[index],
                    delays[index],
                    modes[index],
                )
                for index in range(count)
            )
        )

    @property
    def sample_id(self) -> np.ndarray:
        result = np.asarray(
            tuple(value.sample_id for value in self.samples), dtype=str
        )
        result.setflags(write=False)
        return result

    @property
    def delay(self) -> np.ndarray:
        result = np.asarray(tuple(value.delay for value in self.samples))
        result.setflags(write=False)
        return result

    @property
    def equal_weight(self) -> float:
        return 1.0 / len(self.samples)

    def sample(self, sample_id: str) -> PhysicalPlantSample:
        identifier = str(sample_id)
        matches = tuple(
            value for value in self.samples if value.sample_id == identifier
        )
        if len(matches) != 1:
            raise KeyError("unknown physical posterior sample")
        return matches[0]


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
    """Map controller virtual rotor-force components to real acceleration."""

    if not isinstance(parameters, VehicleParameters):
        raise TypeError("parameters must be VehicleParameters")
    if not isinstance(geometry, GrapeGeometry):
        raise TypeError("geometry must be GrapeGeometry")
    angles = (
        np.zeros(4)
        if gimbal_angles is None
        else np.asarray(gimbal_angles, dtype=float)
    )
    if angles.shape != (4,) or np.any(~np.isfinite(angles)):
        raise ValueError("gimbal_angles must contain four finite values")
    origins = geometry.thrust_origins(angles)
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
    """Return the real response to nominal generalized acceleration."""

    nominal = acceleration_allocation_matrix(
        controller_nominal_parameters, geometry
    )
    physical = physical_acceleration_allocation_matrix(
        physical_parameters, geometry
    )
    result = physical @ np.linalg.pinv(nominal, rcond=1.0e-4)
    if result.shape != (6, 6) or np.any(~np.isfinite(result)):
        raise ValueError("closed-loop acceleration response is invalid")
    return result


def group_compensation_scales(response: np.ndarray) -> np.ndarray:
    """Find one positive least-squares scale per coupled PID group."""

    matrix = np.asarray(response, dtype=float)
    if matrix.shape != (6, 6) or np.any(~np.isfinite(matrix)):
        raise ValueError("response must be a finite 6x6 matrix")
    identity = np.eye(6)
    scales = []
    for group, axes in zip(PID_GROUPS, GROUP_AXIS_INDICES):
        indices = np.asarray(axes, dtype=int)
        actual = matrix[:, indices]
        target = identity[:, indices]
        denominator = float(np.sum(actual * actual))
        numerator = float(np.sum(actual * target))
        if denominator <= 0.0 or numerator <= 0.0:
            raise ValueError(
                "sample has no positive acceleration response for {}".format(
                    group
                )
            )
        scales.append(numerator / denominator)
    result = np.asarray(scales)
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("PID compensation scales must be positive")
    return result


@dataclass(frozen=True)
class PidProposalPopulation:
    """One exact, sample-aligned PID proposal per physical MCMC draw."""

    source_sample_id: np.ndarray
    source_mode_id: Tuple[str, ...]
    source_delay: np.ndarray
    group_scales: np.ndarray
    exact_gain_values: np.ndarray
    acceleration_response: np.ndarray
    current: PidGainConfiguration

    def __post_init__(self) -> None:
        sample_id = np.asarray(self.source_sample_id)
        if sample_id.ndim != 1 or sample_id.dtype.kind not in "US":
            raise ValueError("source_sample_id must be a string vector")
        count = sample_id.size
        modes = tuple(str(value) for value in self.source_mode_id)
        delay = np.asarray(self.source_delay, dtype=float)
        scales = np.asarray(self.group_scales, dtype=float)
        gains = np.asarray(self.exact_gain_values, dtype=float)
        response = np.asarray(self.acceleration_response, dtype=float)
        if (
            count < 1
            or len(set(sample_id.tolist())) != count
            or len(modes) != count
            or any(not value for value in modes)
            or delay.shape != (count,)
            or scales.shape != (count, len(PID_GROUPS))
            or gains.shape != (count, len(PID_GROUPS), 3)
            or response.shape != (count, 6, 6)
            or np.any(~np.isfinite(delay))
            or np.any(delay < 0.0)
            or np.any(~np.isfinite(scales))
            or np.any(scales <= 0.0)
            or np.any(~np.isfinite(gains))
            or np.any(gains < 0.0)
            or np.any(~np.isfinite(response))
            or not isinstance(self.current, PidGainConfiguration)
        ):
            raise ValueError("PID proposal population is not sample-aligned")
        expected = self.current.values[None, :, :] * scales[:, :, None]
        if not np.allclose(gains, expected, rtol=1.0e-12, atol=1.0e-14):
            raise ValueError("exact gains must retain each raw sample proposal")
        for name, value in (
            ("source_sample_id", sample_id),
            ("source_delay", delay),
            ("group_scales", scales),
            ("exact_gain_values", gains),
            ("acceleration_response", response),
        ):
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        object.__setattr__(self, "source_mode_id", modes)

    def sample_index(self, sample_id: str) -> int:
        matches = np.flatnonzero(self.source_sample_id == str(sample_id))
        if matches.size != 1:
            raise KeyError("unknown PID proposal source sample")
        return int(matches[0])

    def configuration_for_sample(
        self, sample_id: str
    ) -> PidGainConfiguration:
        return PidGainConfiguration(
            self.exact_gain_values[self.sample_index(sample_id)]
        )


@dataclass(frozen=True)
class PidCandidate:
    """An exact PID gain candidate with auditable generation provenance."""

    candidate_id: str
    source: str
    configuration: PidGainConfiguration
    source_sample_id: str = ""
    source_mode_id: str = ""
    generation: int = 0
    parent_candidate_id: str = ""

    def __post_init__(self) -> None:
        identifier = _canonical_identifier(self.candidate_id, "candidate_id")
        source = str(self.source)
        allowed = ("current", "sample-derived", "user", "mutation")
        if source not in allowed:
            raise ValueError("unknown PID candidate source")
        if not isinstance(self.configuration, PidGainConfiguration):
            raise TypeError("configuration must be PidGainConfiguration")
        sample_id = str(self.source_sample_id)
        mode = str(self.source_mode_id)
        generation = self.generation
        parent = str(self.parent_candidate_id)
        if (
            isinstance(generation, (bool, np.bool_))
            or not isinstance(generation, (int, np.integer))
            or generation < 0
        ):
            raise ValueError("generation must be a non-negative integer")
        if source == "current":
            valid = (
                identifier == "current"
                and not sample_id
                and not mode
                and generation == 0
                and not parent
            )
        elif source == "sample-derived":
            valid = bool(sample_id and mode and generation == 0 and not parent)
        elif source == "user":
            valid = not sample_id and not mode and generation == 0 and not parent
        else:
            valid = bool(not sample_id and not mode and generation > 0 and parent)
        if not valid:
            raise ValueError("candidate provenance disagrees with its source")
        if source != "current" and identifier == "current":
            raise ValueError("candidate_id=current is reserved for the baseline")
        if sample_id:
            sample_id = _canonical_sample_id(sample_id)
        if parent:
            parent = _canonical_identifier(parent, "parent_candidate_id")
        object.__setattr__(self, "candidate_id", identifier)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "source_sample_id", sample_id)
        object.__setattr__(self, "source_mode_id", mode)
        object.__setattr__(self, "generation", int(generation))
        object.__setattr__(self, "parent_candidate_id", parent)


def derive_pid_proposals(
    posterior: PhysicalPlantPosterior,
    controller_nominal_parameters: VehicleParameters,
    geometry: GrapeGeometry,
    current: PidGainConfiguration,
) -> PidProposalPopulation:
    """Derive one correlated exact-gain proposal for every retained draw."""

    if not isinstance(posterior, PhysicalPlantPosterior):
        raise TypeError("posterior must be PhysicalPlantPosterior")
    if not isinstance(current, PidGainConfiguration):
        raise TypeError("current must be PidGainConfiguration")
    responses = np.asarray(
        tuple(
            closed_loop_acceleration_response(
                sample.parameters,
                controller_nominal_parameters,
                geometry,
            )
            for sample in posterior.samples
        )
    )
    scales = np.asarray(
        tuple(group_compensation_scales(value) for value in responses)
    )
    return PidProposalPopulation(
        source_sample_id=posterior.sample_id,
        source_mode_id=tuple(
            value.source_mode_id for value in posterior.samples
        ),
        source_delay=posterior.delay,
        group_scales=scales,
        exact_gain_values=current.values[None, :, :] * scales[:, :, None],
        acceleration_response=responses,
        current=current,
    )


def current_pid_candidate(current: PidGainConfiguration) -> PidCandidate:
    return PidCandidate("current", "current", current)


def sample_pid_candidate(
    proposals: PidProposalPopulation, sample_id: str
) -> PidCandidate:
    if not isinstance(proposals, PidProposalPopulation):
        raise TypeError("proposals must be PidProposalPopulation")
    index = proposals.sample_index(sample_id)
    identifier = str(proposals.source_sample_id[index])
    encoded_identifier = base64.urlsafe_b64encode(
        identifier.encode("utf-8")
    ).decode("ascii").rstrip("=")
    return PidCandidate(
        candidate_id="sample_{}".format(encoded_identifier),
        source="sample-derived",
        configuration=PidGainConfiguration(
            proposals.exact_gain_values[index]
        ),
        source_sample_id=identifier,
        source_mode_id=proposals.source_mode_id[index],
    )


def user_pid_candidate(
    candidate_id: str, configuration: PidGainConfiguration
) -> PidCandidate:
    return PidCandidate(str(candidate_id), "user", configuration)


__all__ = [
    "GROUP_AXIS_INDICES",
    "PhysicalPlantPosterior",
    "PhysicalPlantSample",
    "PidCandidate",
    "PidProposalPopulation",
    "closed_loop_acceleration_response",
    "current_pid_candidate",
    "derive_pid_proposals",
    "group_compensation_scales",
    "physical_acceleration_allocation_matrix",
    "sample_pid_candidate",
    "user_pid_candidate",
]
