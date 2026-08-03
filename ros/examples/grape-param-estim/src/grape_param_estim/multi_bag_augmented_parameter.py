"""Canonical multi-bag orchestration for fixed-Q parameter assimilation.

The shared 19-D static ensemble is updated by one bag at a time in canonical
``bag_id`` order.  Member pairing is never changed between bags.  Every bag
receives a fresh exact 26-D local prior and six-dimensional initial wrench
that are sample-orthogonal to the shared ensemble entering that bag.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Optional, Tuple

import numpy as np

from grape_param_estim.augmented_parameter_filter import (
    AugmentedParameterFilterResult,
    run_augmented_parameter_filter,
)
from grape_param_estim.augmented_parameter_state import (
    MINIMUM_FULL_RANK_MEMBER_COUNT,
    MINIMUM_PROCESS_NOISE_MEMBER_COUNT,
    SHARED_STATIC_DIMENSION,
    AugmentedInitialEnsemble,
    AugmentedParameterPrior,
)
from grape_param_estim.diagonal_q import BodyWrenchDiagonalCovariance
from grape_param_estim.ensemble_state_smoother import exact_gaussian_ensemble
from grape_param_estim.filter_state import GrapeFilterState
from grape_param_estim.parameterization import PARAMETER_DIMENSION
from grape_param_estim.progress import (
    CancellationToken,
    ProgressCallback,
    ProgressEvent,
    ProgressTracker,
)
from grape_param_estim.stochastic_closed_loop import (
    PoseObservationCovariance,
)
from grape_param_estim.strong_constraint import (
    PARAMETER_OFFSET,
    StrongConstraintProblem,
)
from grape_param_estim.system import ActuatorState
from grape_param_estim.timing import BoundedDelayChart


def _bag_id(value) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("bag_id must be a non-empty string")
    return value


def _seed(value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("seed must be an unsigned 32-bit integer")
    result = int(value)
    if result < 0 or result >= 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    return result


def _member_count(value) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            "ensemble_size must be at least {}".format(
                MINIMUM_PROCESS_NOISE_MEMBER_COUNT
            )
        )
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "ensemble_size must be at least {}".format(
                MINIMUM_PROCESS_NOISE_MEMBER_COUNT
            )
        ) from error
    if result != value or result < MINIMUM_PROCESS_NOISE_MEMBER_COUNT:
        raise ValueError(
            "ensemble_size must be at least {}".format(
                MINIMUM_PROCESS_NOISE_MEMBER_COUNT
            )
        )
    return result


def _derived_seed(root_seed: int, namespace: str) -> int:
    payload = root_seed.to_bytes(4, byteorder="big", signed=False)
    payload += b"\0" + namespace.encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _filter_work_units(member_count: int, time_count: int) -> int:
    return time_count + member_count * (time_count - 1) + time_count - 1


def _draw_initial_ensemble_from_shared(
    problem: StrongConstraintProblem,
    covariance: BodyWrenchDiagonalCovariance,
    shared_coordinates: np.ndarray,
    member_id: np.ndarray,
    seed: int,
    prior: AugmentedParameterPrior,
) -> AugmentedInitialEnsemble:
    """Draw one exact local+wrench prior without changing shared pairing."""

    shared = np.asarray(shared_coordinates, dtype=float)
    identifiers = np.asarray(member_id, dtype=np.int64)
    if (
        shared.ndim != 2
        or shared.shape[1] != SHARED_STATIC_DIMENSION
        or shared.shape[0] < MINIMUM_FULL_RANK_MEMBER_COUNT
        or np.any(~np.isfinite(shared))
        or identifiers.shape != (shared.shape[0],)
        or np.unique(identifiers).size != identifiers.size
    ):
        raise ValueError("shared ensemble and member IDs are misaligned")
    count = int(shared.shape[0])
    local = exact_gaussian_ensemble(
        prior.local_mean,
        prior.local_covariance,
        count,
        seed,
        orthogonal_to=shared,
    )
    unknowns = np.concatenate((shared, local), axis=1)
    wrench = exact_gaussian_ensemble(
        np.zeros(6),
        covariance.matrix,
        count,
        int((seed + 1) % (2**32)),
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
        identifiers, shared, local, tuple(states)
    )


def _nominal_parameter_vector(problem: StrongConstraintProblem) -> np.ndarray:
    value = problem.parameter_chart.decode(
        np.zeros(PARAMETER_DIMENSION, dtype=float)
    )
    return np.concatenate(
        (
            np.asarray((value.mass,), dtype=float),
            np.asarray(value.inertia, dtype=float).reshape(-1),
            np.asarray(value.cog_offset, dtype=float).reshape(-1),
            np.asarray(value.force_effectiveness, dtype=float).reshape(-1),
            np.asarray(value.torque_effectiveness, dtype=float).reshape(-1),
        )
    )


@dataclass(frozen=True)
class PreparedAugmentedParameterBag:
    """One stage-2 bag with explicitly fixed ``R`` and OU time scale."""

    bag_id: str
    problem: StrongConstraintProblem
    observation_covariance: PoseObservationCovariance
    correlation_time: float
    configuration_fingerprint: str = ""

    def __post_init__(self) -> None:
        identifier = _bag_id(self.bag_id)
        if not isinstance(self.problem, StrongConstraintProblem):
            raise TypeError("problem must be a StrongConstraintProblem")
        if not isinstance(
            self.observation_covariance, PoseObservationCovariance
        ):
            raise TypeError(
                "observation_covariance must be PoseObservationCovariance"
            )
        correlation_time = float(self.correlation_time)
        if not np.isfinite(correlation_time) or correlation_time <= 0.0:
            raise ValueError("correlation_time must be finite and positive")
        if not isinstance(self.configuration_fingerprint, str):
            raise TypeError("configuration_fingerprint must be a string")
        covariance = PoseObservationCovariance(
            self.observation_covariance.translation,
            self.observation_covariance.rotation_tangent,
        )
        object.__setattr__(self, "bag_id", identifier)
        object.__setattr__(self, "observation_covariance", covariance)
        object.__setattr__(self, "correlation_time", correlation_time)

    @classmethod
    def from_diagonal_q_bag(cls, value):
        """Retain the exact fixed-model inputs used by stage-1 Q EM."""

        from grape_param_estim.real_diagonal_q_estimation import (
            PreparedDiagonalQBag,
        )

        if not isinstance(value, PreparedDiagonalQBag):
            raise TypeError("value must be a PreparedDiagonalQBag")
        return cls(
            bag_id=value.bag_id,
            problem=value.problem,
            observation_covariance=value.observation_covariance,
            correlation_time=value.calibration.correlation_time,
            configuration_fingerprint=value.configuration_fingerprint,
        )


def _ordered_bags(
    values: Iterable[PreparedAugmentedParameterBag],
) -> Tuple[PreparedAugmentedParameterBag, ...]:
    try:
        bags = tuple(values)
    except TypeError as error:
        raise TypeError(
            "bags must be an iterable of PreparedAugmentedParameterBag"
        ) from error
    if not bags:
        raise ValueError("at least one augmented-parameter bag is required")
    if any(
        not isinstance(value, PreparedAugmentedParameterBag)
        for value in bags
    ):
        raise TypeError(
            "bags must contain PreparedAugmentedParameterBag values"
        )
    ordered = tuple(sorted(bags, key=lambda value: value.bag_id))
    identifiers = tuple(value.bag_id for value in ordered)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("augmented-parameter bags must have unique bag IDs")
    fingerprints = {
        value.configuration_fingerprint
        for value in ordered
        if value.configuration_fingerprint
    }
    if len(fingerprints) > 1:
        raise ValueError(
            "augmented-parameter bags must share one configuration "
            "fingerprint"
        )
    reference = _nominal_parameter_vector(ordered[0].problem)
    for bag in ordered[1:]:
        if not np.allclose(
            reference,
            _nominal_parameter_vector(bag.problem),
            rtol=1.0e-12,
            atol=1.0e-14,
        ):
            raise ValueError(
                "augmented-parameter bags must share one physical "
                "parameter chart"
            )
    return ordered


@dataclass(frozen=True)
class AugmentedParameterBagResult:
    """The member-aligned input and filter/smoother output for one bag."""

    bag_id: str
    wrench_covariance: BodyWrenchDiagonalCovariance
    observation_covariance: PoseObservationCovariance
    correlation_time: float
    initial_ensemble: AugmentedInitialEnsemble
    filter_result: AugmentedParameterFilterResult

    def __post_init__(self) -> None:
        identifier = _bag_id(self.bag_id)
        if not isinstance(
            self.wrench_covariance, BodyWrenchDiagonalCovariance
        ):
            raise TypeError(
                "wrench_covariance must be BodyWrenchDiagonalCovariance"
            )
        if not isinstance(
            self.observation_covariance, PoseObservationCovariance
        ):
            raise TypeError(
                "observation_covariance must be PoseObservationCovariance"
            )
        correlation_time = float(self.correlation_time)
        if not np.isfinite(correlation_time) or correlation_time <= 0.0:
            raise ValueError("correlation_time must be finite and positive")
        if not isinstance(self.initial_ensemble, AugmentedInitialEnsemble):
            raise TypeError(
                "initial_ensemble must be AugmentedInitialEnsemble"
            )
        if not isinstance(
            self.filter_result, AugmentedParameterFilterResult
        ):
            raise TypeError(
                "filter_result must be AugmentedParameterFilterResult"
            )
        if not np.array_equal(
            self.initial_ensemble.member_id, self.filter_result.member_id
        ):
            raise ValueError("bag result changed member identity")
        if not np.array_equal(
            self.initial_ensemble.shared_coordinates,
            self.filter_result.prior_static_ensemble,
        ):
            raise ValueError("bag result does not start from its shared input")
        object.__setattr__(self, "bag_id", identifier)
        object.__setattr__(
            self,
            "wrench_covariance",
            BodyWrenchDiagonalCovariance(
                self.wrench_covariance.stationary_variance
            ),
        )
        object.__setattr__(
            self,
            "observation_covariance",
            PoseObservationCovariance(
                self.observation_covariance.translation,
                self.observation_covariance.rotation_tangent,
            ),
        )
        object.__setattr__(self, "correlation_time", correlation_time)

    @property
    def initial_shared_ensemble(self) -> np.ndarray:
        return self.initial_ensemble.shared_coordinates.copy()

    @property
    def final_shared_ensemble(self) -> np.ndarray:
        return self.filter_result.final_static_ensemble.copy()


@dataclass(frozen=True)
class MultiBagAugmentedParameterResult:
    """A canonical sequential pass and its final shared posterior."""

    wrench_covariance: BodyWrenchDiagonalCovariance
    bags: Tuple[AugmentedParameterBagResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.wrench_covariance, BodyWrenchDiagonalCovariance
        ):
            raise TypeError(
                "wrench_covariance must be BodyWrenchDiagonalCovariance"
            )
        bags = tuple(self.bags)
        if not bags or any(
            not isinstance(value, AugmentedParameterBagResult)
            for value in bags
        ):
            raise ValueError(
                "bags must contain at least one AugmentedParameterBagResult"
            )
        identifiers = tuple(value.bag_id for value in bags)
        if (
            identifiers != tuple(sorted(identifiers))
            or len(set(identifiers)) != len(identifiers)
        ):
            raise ValueError("bag results must be sorted and unique")
        member_id = bags[0].initial_ensemble.member_id
        maximum_delay = bags[0].filter_result.maximum_delay
        for index, value in enumerate(bags):
            if not np.array_equal(
                value.wrench_covariance.stationary_variance,
                self.wrench_covariance.stationary_variance,
            ):
                raise ValueError("all bag results must use the shared fixed Q")
            if not np.array_equal(value.initial_ensemble.member_id, member_id):
                raise ValueError("member identity must be common to all bags")
            if value.filter_result.maximum_delay != maximum_delay:
                raise ValueError("all bag results must use one delay chart")
            if index and not np.array_equal(
                bags[index - 1].filter_result.final_static_ensemble,
                value.initial_ensemble.shared_coordinates,
            ):
                raise ValueError(
                    "each bag must receive the preceding shared posterior"
                )
        covariance = BodyWrenchDiagonalCovariance(
            self.wrench_covariance.stationary_variance
        )
        object.__setattr__(self, "wrench_covariance", covariance)
        object.__setattr__(self, "bags", bags)

    @property
    def bag_ids(self) -> Tuple[str, ...]:
        return tuple(value.bag_id for value in self.bags)

    @property
    def member_id(self) -> np.ndarray:
        return self.bags[0].initial_ensemble.member_id.copy()

    @property
    def member_count(self) -> int:
        return int(self.member_id.size)

    @property
    def maximum_delay(self) -> float:
        return float(self.bags[0].filter_result.maximum_delay)

    @property
    def initial_shared_ensemble(self) -> np.ndarray:
        return self.bags[0].initial_shared_ensemble

    @property
    def final_shared_posterior(self) -> np.ndarray:
        return self.bags[-1].final_shared_ensemble

    def bag(self, bag_id: str) -> AugmentedParameterBagResult:
        identifier = _bag_id(bag_id)
        for value in self.bags:
            if value.bag_id == identifier:
                return value
        raise KeyError(
            "unknown augmented-parameter bag {!r}".format(identifier)
        )


def run_multi_bag_augmented_parameter_filter(
    bags: Iterable[PreparedAugmentedParameterBag],
    wrench_covariance: BodyWrenchDiagonalCovariance,
    *,
    ensemble_size: int = 64,
    seed: int = 23,
    prior: Optional[AugmentedParameterPrior] = None,
    delay_chart: Optional[BoundedDelayChart] = None,
    covariance_rcond: float = 1.0e-12,
    progress_callback: Optional[ProgressCallback] = None,
    cancellation_token: Optional[CancellationToken] = None,
    run_id: str = "multi-bag-augmented-parameter",
) -> MultiBagAugmentedParameterResult:
    """Assimilate bags sequentially while carrying the shared 19-D members."""

    ordered = _ordered_bags(bags)
    if not isinstance(wrench_covariance, BodyWrenchDiagonalCovariance):
        raise TypeError(
            "wrench_covariance must be BodyWrenchDiagonalCovariance"
        )
    members = _member_count(ensemble_size)
    selected_seed = _seed(seed)
    selected_delay_chart = delay_chart or BoundedDelayChart()
    if not isinstance(selected_delay_chart, BoundedDelayChart):
        raise TypeError("delay_chart must be a BoundedDelayChart")
    selected_prior = (
        AugmentedParameterPrior.grape(
            maximum_delay=selected_delay_chart.maximum_delay
        )
        if prior is None
        else prior
    )
    if not isinstance(selected_prior, AugmentedParameterPrior):
        raise TypeError("prior must be an AugmentedParameterPrior")
    selected_rcond = float(covariance_rcond)
    if not np.isfinite(selected_rcond) or selected_rcond <= 0.0:
        raise ValueError("covariance_rcond must be finite and positive")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    cancellation = cancellation_token or CancellationToken()
    if not isinstance(cancellation, CancellationToken):
        raise TypeError("cancellation_token must be CancellationToken")

    total_units = sum(
        _filter_work_units(
            members, int(value.problem.observations.times.size)
        )
        for value in ordered
    )
    tracker = ProgressTracker(
        run_id,
        total_units,
        callback=progress_callback,
        cancellation_token=cancellation,
    )
    tracker.emit(
        0,
        "multi_bag_augmented_parameter",
        "Multi-bag augmented parameter assimilation",
        message="starting canonical multi-bag pass",
    )

    shared = exact_gaussian_ensemble(
        selected_prior.shared_mean,
        selected_prior.shared_covariance,
        members,
        _derived_seed(selected_seed, "shared-static-prior"),
    )
    member_id = np.arange(members, dtype=np.int64)
    completed_before_bag = 0
    outputs = []
    for bag in ordered:
        tracker.checkpoint()
        local_seed = _derived_seed(
            selected_seed, "{}\0local-and-wrench".format(bag.bag_id)
        )
        filter_seed = _derived_seed(
            selected_seed, "{}\0filter-process".format(bag.bag_id)
        )
        initial = _draw_initial_ensemble_from_shared(
            bag.problem,
            wrench_covariance,
            shared,
            member_id,
            local_seed,
            prior=selected_prior,
        )

        def forward_progress(
            event: ProgressEvent,
            offset=completed_before_bag,
        ) -> None:
            tracker.emit(
                offset + event.completed_units,
                "multi_bag_{}".format(event.stage_id),
                event.stage_label,
                bag_id=bag.bag_id,
                member_id=event.member_id,
                message=event.message,
            )

        output = run_augmented_parameter_filter(
            problem=bag.problem,
            initial_ensemble=initial,
            wrench_covariance=wrench_covariance,
            correlation_time=bag.correlation_time,
            observation_covariance=bag.observation_covariance,
            seed=filter_seed,
            delay_chart=selected_delay_chart,
            covariance_rcond=selected_rcond,
            progress_callback=forward_progress,
            cancellation_token=cancellation,
            progress_run_id="{}-{}".format(run_id, bag.bag_id),
            bag_id=bag.bag_id,
        )
        outputs.append(
            AugmentedParameterBagResult(
                bag_id=bag.bag_id,
                wrench_covariance=wrench_covariance,
                observation_covariance=bag.observation_covariance,
                correlation_time=bag.correlation_time,
                initial_ensemble=initial,
                filter_result=output,
            )
        )
        shared = output.final_static_ensemble.copy()
        completed_before_bag += _filter_work_units(
            members, int(bag.problem.observations.times.size)
        )
        tracker.emit(
            completed_before_bag,
            "multi_bag_augmented_parameter",
            "Multi-bag augmented parameter assimilation",
            bag_id=bag.bag_id,
            message="completed bag {}".format(bag.bag_id),
        )

    return MultiBagAugmentedParameterResult(
        wrench_covariance=wrench_covariance,
        bags=tuple(outputs),
    )


__all__ = [
    "AugmentedParameterBagResult",
    "MultiBagAugmentedParameterResult",
    "PreparedAugmentedParameterBag",
    "run_multi_bag_augmented_parameter_filter",
]
