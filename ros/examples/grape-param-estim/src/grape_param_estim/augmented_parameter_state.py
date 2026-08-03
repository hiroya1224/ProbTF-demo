"""The fixed-size augmented state used after diagonal-Q calibration.

The estimated unknowns are 19 shared static coordinates and 26 bag-local
initial coordinates.  The six residual-wrench values are a Markov process
state driven by Q; they are never appended once per sample to an optimisation
vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from grape_param_estim.diagonal_q import BodyWrenchDiagonalCovariance
from grape_param_estim.ensemble_state_smoother import exact_gaussian_ensemble
from grape_param_estim.filter_state import (
    FILTER_STATE_DIMENSION,
    GrapeFilterState,
)
from grape_param_estim.parameterization import PARAMETER_DIMENSION
from grape_param_estim.strong_constraint import (
    PARAMETER_OFFSET,
    StrongConstraintPrior,
    StrongConstraintProblem,
)
from grape_param_estim.system import ActuatorState
from grape_param_estim.timing import ConstantDelayChart


SHARED_STATIC_DIMENSION = PARAMETER_DIMENSION + 1
LOCAL_INITIAL_DIMENSION = PARAMETER_OFFSET + 8
ESTIMATED_UNKNOWN_DIMENSION = (
    SHARED_STATIC_DIMENSION + LOCAL_INITIAL_DIMENSION
)
AUGMENTED_FILTER_DIMENSION = (
    SHARED_STATIC_DIMENSION + FILTER_STATE_DIMENSION
)
MINIMUM_FULL_RANK_MEMBER_COUNT = AUGMENTED_FILTER_DIMENSION + 1
MINIMUM_PROCESS_NOISE_MEMBER_COUNT = AUGMENTED_FILTER_DIMENSION + 7


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or np.any(~np.isfinite(result)):
        raise ValueError(
            "{} must contain {} finite values".format(name, size)
        )
    return result.copy()


@dataclass(frozen=True)
class AugmentedParameterPrior:
    """Independent Gaussian prior for shared and bag-local coordinates."""

    shared_mean: np.ndarray
    shared_covariance: np.ndarray
    local_mean: np.ndarray
    local_covariance: np.ndarray

    def __post_init__(self) -> None:
        fields = (
            ("shared", SHARED_STATIC_DIMENSION),
            ("local", LOCAL_INITIAL_DIMENSION),
        )
        for prefix, size in fields:
            mean = _finite_vector(
                getattr(self, prefix + "_mean"), size, prefix + "_mean"
            )
            covariance = np.asarray(
                getattr(self, prefix + "_covariance"), dtype=float
            )
            if (
                covariance.shape != (size, size)
                or np.any(~np.isfinite(covariance))
                or not np.allclose(
                    covariance, covariance.T, rtol=1.0e-12, atol=1.0e-14
                )
                or np.any(np.linalg.eigvalsh(covariance) <= 0.0)
            ):
                raise ValueError(
                    "{}_covariance must be finite positive definite".format(
                        prefix
                    )
                )
            object.__setattr__(self, prefix + "_mean", mean)
            object.__setattr__(
                self,
                prefix + "_covariance",
                0.5 * (covariance + covariance.T),
            )

    @classmethod
    def grape(
        cls,
        delay_mean: float = 0.02,
        delay_standard_deviation: float = 0.015,
    ) -> "AugmentedParameterPrior":
        selected_delay_mean = float(delay_mean)
        selected_delay_deviation = float(delay_standard_deviation)
        if (
            not np.isfinite(selected_delay_mean)
            or selected_delay_mean < 0.0
            or not np.isfinite(selected_delay_deviation)
            or selected_delay_deviation <= 0.0
        ):
            raise ValueError(
                "delay mean/deviation must be non-negative/positive"
            )
        base = StrongConstraintPrior.grape()
        shared_mean = np.concatenate(
            (
                base.mean[PARAMETER_OFFSET:],
                np.asarray((selected_delay_mean,)),
            )
        )
        shared_covariance = np.zeros(
            (SHARED_STATIC_DIMENSION, SHARED_STATIC_DIMENSION), dtype=float
        )
        shared_covariance[:PARAMETER_DIMENSION, :PARAMETER_DIMENSION] = (
            base.covariance[PARAMETER_OFFSET:, PARAMETER_OFFSET:]
        )
        shared_covariance[-1, -1] = selected_delay_deviation**2

        actuator_deviation = np.asarray(
            (0.30, 0.30, 0.30, 0.30, 0.03, 0.03, 0.03, 0.03)
        )
        local_mean = np.zeros(LOCAL_INITIAL_DIMENSION, dtype=float)
        local_mean[:PARAMETER_OFFSET] = base.mean[:PARAMETER_OFFSET]
        local_covariance = np.zeros(
            (LOCAL_INITIAL_DIMENSION, LOCAL_INITIAL_DIMENSION), dtype=float
        )
        local_covariance[:PARAMETER_OFFSET, :PARAMETER_OFFSET] = (
            base.covariance[:PARAMETER_OFFSET, :PARAMETER_OFFSET]
        )
        local_covariance[PARAMETER_OFFSET:, PARAMETER_OFFSET:] = np.diag(
            actuator_deviation**2
        )
        return cls(
            shared_mean,
            shared_covariance,
            local_mean,
            local_covariance,
        )


@dataclass(frozen=True)
class AugmentedInitialEnsemble:
    """Member-aligned 45-D unknowns and initial stochastic state."""

    member_id: np.ndarray
    shared_coordinates: np.ndarray
    local_coordinates: np.ndarray
    filter_states: Tuple[GrapeFilterState, ...]

    def __post_init__(self) -> None:
        member_id = np.asarray(self.member_id, dtype=np.int64)
        shared = np.asarray(self.shared_coordinates, dtype=float)
        local = np.asarray(self.local_coordinates, dtype=float)
        states = tuple(self.filter_states)
        members = member_id.size
        if (
            member_id.shape != (members,)
            or np.unique(member_id).size != members
            or shared.shape != (members, SHARED_STATIC_DIMENSION)
            or local.shape != (members, LOCAL_INITIAL_DIMENSION)
            or len(states) != members
            or members < MINIMUM_FULL_RANK_MEMBER_COUNT
            or np.any(~np.isfinite(shared))
            or np.any(~np.isfinite(local))
            or any(not isinstance(state, GrapeFilterState) for state in states)
        ):
            raise ValueError("augmented initial ensemble is misaligned")
        object.__setattr__(self, "member_id", member_id.copy())
        object.__setattr__(self, "shared_coordinates", shared.copy())
        object.__setattr__(self, "local_coordinates", local.copy())
        object.__setattr__(self, "filter_states", states)

    @property
    def member_count(self) -> int:
        return int(self.member_id.size)

    @property
    def estimated_unknown_coordinates(self) -> np.ndarray:
        return np.concatenate(
            (self.shared_coordinates, self.local_coordinates), axis=1
        )


def draw_augmented_initial_ensemble(
    problem: StrongConstraintProblem,
    covariance: BodyWrenchDiagonalCovariance,
    member_count: int,
    seed: int,
    prior: AugmentedParameterPrior | None = None,
) -> AugmentedInitialEnsemble:
    """Draw one exact, mutually uncorrelated 19+26+6 initial ensemble."""

    if not isinstance(problem, StrongConstraintProblem):
        raise TypeError("problem must be a StrongConstraintProblem")
    if not isinstance(covariance, BodyWrenchDiagonalCovariance):
        raise TypeError(
            "covariance must be a BodyWrenchDiagonalCovariance"
        )
    count = int(member_count)
    if count != member_count or count < MINIMUM_FULL_RANK_MEMBER_COUNT:
        raise ValueError(
            "member_count must be at least {}".format(
                MINIMUM_FULL_RANK_MEMBER_COUNT
            )
        )
    selected_prior = prior or AugmentedParameterPrior.grape()
    shared = exact_gaussian_ensemble(
        selected_prior.shared_mean,
        selected_prior.shared_covariance,
        count,
        int(seed),
    )
    local = exact_gaussian_ensemble(
        selected_prior.local_mean,
        selected_prior.local_covariance,
        count,
        int(seed) + 1,
        orthogonal_to=shared,
    )
    unknowns = np.concatenate((shared, local), axis=1)
    wrench = exact_gaussian_ensemble(
        np.zeros(6),
        covariance.matrix,
        count,
        int(seed) + 2,
        orthogonal_to=unknowns,
    )

    anchor = problem.initial_actuator_state
    if anchor is None:
        anchor = ActuatorState(
            problem.nominal_trajectory.actuator_thrust[0],
            problem.nominal_trajectory.actuator_gimbal_angle[0],
        )
    limits = problem.actuator_parameters
    states = []
    for member in range(count):
        strong_coordinates = np.concatenate(
            (
                local[member, :PARAMETER_OFFSET],
                shared[member, :PARAMETER_DIMENSION],
            )
        )
        rigid, controller, _parameters = problem.decode_control(
            strong_coordinates
        )
        actuator_delta = local[member, PARAMETER_OFFSET:]
        actuator = ActuatorState(
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
        states.append(
            GrapeFilterState(rigid, controller, actuator, wrench[member])
        )
    return AugmentedInitialEnsemble(
        np.arange(count, dtype=np.int64), shared, local, tuple(states)
    )


def decode_shared_static_coordinates(
    problem: StrongConstraintProblem, coordinates: np.ndarray
):
    """Decode one 19-D member into plant parameters and physical delay."""

    if not isinstance(problem, StrongConstraintProblem):
        raise TypeError("problem must be a StrongConstraintProblem")
    value = _finite_vector(
        coordinates, SHARED_STATIC_DIMENSION, "shared coordinates"
    )
    return (
        problem.parameter_chart.decode(value[:PARAMETER_DIMENSION]),
        ConstantDelayChart().decode(value[-1]),
    )


__all__ = [
    "AUGMENTED_FILTER_DIMENSION",
    "ESTIMATED_UNKNOWN_DIMENSION",
    "LOCAL_INITIAL_DIMENSION",
    "MINIMUM_FULL_RANK_MEMBER_COUNT",
    "MINIMUM_PROCESS_NOISE_MEMBER_COUNT",
    "SHARED_STATIC_DIMENSION",
    "AugmentedInitialEnsemble",
    "AugmentedParameterPrior",
    "decode_shared_static_coordinates",
    "draw_augmented_initial_ensemble",
]
