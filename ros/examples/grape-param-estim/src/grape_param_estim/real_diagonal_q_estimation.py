"""Real-flight adapter for diagonal-Q ensemble EM estimation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Iterable, Optional, Tuple

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.controller import GrapeController
from grape_param_estim.diagonal_q import BodyWrenchDiagonalCovariance
from grape_param_estim.diagonal_q_em import (
    DiagonalQBagExpectation,
    DiagonalQEmConfig,
    DiagonalQEmResult,
    DiagonalQInitialPilot,
    run_diagonal_q_em,
)
from grape_param_estim.dynamics import FullSixDofPlant
from grape_param_estim.ensemble_state_smoother import exact_gaussian_ensemble
from grape_param_estim.filter_state import GrapeFilterState
from grape_param_estim.progress import CancellationToken, ProgressCallback
from grape_param_estim.real_assimilation import build_real_strong_problem
from grape_param_estim.real_calibration import (
    ModelErrorCalibration,
    calibrate_model_error_from_closed_loop_pose,
)
from grape_param_estim.real_rosbag import RealFlightEpisode
from grape_param_estim.stochastic_closed_loop import (
    PoseObservationCovariance,
    StochasticClosedLoopEStepResult,
    run_stochastic_closed_loop_e_step,
)
from grape_param_estim.strong_constraint import (
    PARAMETER_OFFSET,
    StrongConstraintPrior,
    StrongConstraintProblem,
)
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
)


Q_ONLY_LOCAL_DIMENSION = PARAMETER_OFFSET + 8
Q_ONLY_MINIMUM_MEMBER_COUNT = 39


def _bag_id(value) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("bag_id must be a non-empty string")
    return value


def _member_count(value) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            "ensemble_size must be at least {}".format(
                Q_ONLY_MINIMUM_MEMBER_COUNT
            )
        )
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "ensemble_size must be at least {}".format(
                Q_ONLY_MINIMUM_MEMBER_COUNT
            )
        ) from error
    if result != value or result < Q_ONLY_MINIMUM_MEMBER_COUNT:
        raise ValueError(
            "ensemble_size must be at least {}".format(
                Q_ONLY_MINIMUM_MEMBER_COUNT
            )
        )
    return result


def _seed(value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("seed must be an unsigned 32-bit integer")
    result = int(value)
    if result < 0 or result >= 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    return result


def _bag_seed(seed: int, bag_id: str) -> int:
    digest = hashlib.sha256(bag_id.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return int((seed + offset) % (2**32))


@dataclass(frozen=True)
class PreparedDiagonalQBag:
    """Fixed-model inputs and pose-only pilot calibration for one bag."""

    bag_id: str
    problem: StrongConstraintProblem
    calibration: ModelErrorCalibration
    configuration_fingerprint: str = ""

    def __post_init__(self) -> None:
        identifier = _bag_id(self.bag_id)
        if not isinstance(self.problem, StrongConstraintProblem):
            raise TypeError("problem must be a StrongConstraintProblem")
        if not isinstance(self.calibration, ModelErrorCalibration):
            raise TypeError("calibration must be ModelErrorCalibration")
        fingerprint = str(self.configuration_fingerprint)
        object.__setattr__(self, "bag_id", identifier)
        object.__setattr__(self, "configuration_fingerprint", fingerprint)

    @property
    def boundary_count(self) -> int:
        return int(self.problem.observations.times.size)

    @property
    def observation_covariance(self) -> PoseObservationCovariance:
        return PoseObservationCovariance(
            self.problem.observations.translation_covariance,
            self.problem.observations.rotation_covariance,
        )

    @property
    def initial_pilot(self) -> DiagonalQInitialPilot:
        return DiagonalQInitialPilot(
            self.bag_id,
            self.boundary_count,
            self.calibration.stationary_standard_deviation,
        )


def prepare_real_diagonal_q_bag(
    bag_id: str,
    episode: RealFlightEpisode,
    configuration_fingerprint: str,
    *,
    initial_delay: float = 0.02,
) -> PreparedDiagonalQBag:
    """Prepare fixed nominal dynamics, fixed R, and a Q pilot for one bag."""

    if not isinstance(episode, RealFlightEpisode):
        raise TypeError("episode must be a RealFlightEpisode")
    delay = float(initial_delay)
    if not np.isfinite(delay) or delay < 0.0:
        raise ValueError("initial_delay must be finite and non-negative")
    (
        problem,
        _initial_state,
        _nominal,
        actuator_parameters,
        nominal_parameters,
    ) = build_real_strong_problem(
        episode, actuator_parameters=ActuatorParameters(delay=delay)
    )
    calibration = calibrate_model_error_from_closed_loop_pose(
        episode.observations.times,
        episode.observations.position,
        episode.observations.orientation_xyzw,
        episode.references,
        episode.controller_configuration,
        episode.initial_controller_state,
        episode.initial_actuator_state,
        actuator_parameters,
        nominal_parameters,
        problem.geometry,
    )
    return PreparedDiagonalQBag(
        _bag_id(bag_id),
        problem,
        calibration,
        str(configuration_fingerprint),
    )


@dataclass(frozen=True)
class QOnlyInitialEnsemble:
    """The 26-D local initial prior and its decoded stochastic states."""

    local_coordinates: np.ndarray
    filter_states: Tuple[GrapeFilterState, ...]

    def __post_init__(self) -> None:
        local = np.asarray(self.local_coordinates, dtype=float)
        states = tuple(self.filter_states)
        if (
            local.ndim != 2
            or local.shape[0] < Q_ONLY_MINIMUM_MEMBER_COUNT
            or local.shape[1] != Q_ONLY_LOCAL_DIMENSION
            or np.any(~np.isfinite(local))
            or len(states) != local.shape[0]
            or any(not isinstance(value, GrapeFilterState) for value in states)
        ):
            raise ValueError("Q-only initial ensemble is misaligned")
        object.__setattr__(self, "local_coordinates", local.copy())
        object.__setattr__(self, "filter_states", states)

    @property
    def member_count(self) -> int:
        return int(self.local_coordinates.shape[0])


def _local_prior():
    base = StrongConstraintPrior.grape()
    mean = np.zeros(Q_ONLY_LOCAL_DIMENSION, dtype=float)
    mean[:PARAMETER_OFFSET] = base.mean[:PARAMETER_OFFSET]
    covariance = np.zeros(
        (Q_ONLY_LOCAL_DIMENSION, Q_ONLY_LOCAL_DIMENSION), dtype=float
    )
    covariance[:PARAMETER_OFFSET, :PARAMETER_OFFSET] = (
        base.covariance[:PARAMETER_OFFSET, :PARAMETER_OFFSET]
    )
    actuator_standard_deviation = np.asarray(
        (0.30, 0.30, 0.30, 0.30, 0.03, 0.03, 0.03, 0.03)
    )
    covariance[PARAMETER_OFFSET:, PARAMETER_OFFSET:] = np.diag(
        actuator_standard_deviation**2
    )
    return mean, covariance


def draw_q_only_initial_ensemble(
    bag: PreparedDiagonalQBag,
    covariance: BodyWrenchDiagonalCovariance,
    ensemble_size: int,
    seed: int,
) -> QOnlyInitialEnsemble:
    """Draw exact local and wrench priors with zero sample cross covariance."""

    if not isinstance(bag, PreparedDiagonalQBag):
        raise TypeError("bag must be PreparedDiagonalQBag")
    if not isinstance(covariance, BodyWrenchDiagonalCovariance):
        raise TypeError(
            "covariance must be BodyWrenchDiagonalCovariance"
        )
    members = _member_count(ensemble_size)
    selected_seed = _seed(seed)
    mean, local_covariance = _local_prior()
    local = exact_gaussian_ensemble(
        mean, local_covariance, members, selected_seed
    )
    wrench = exact_gaussian_ensemble(
        np.zeros(6),
        covariance.matrix,
        members,
        int((selected_seed + 1) % (2**32)),
        orthogonal_to=local,
    )
    problem = bag.problem
    anchor = problem.initial_actuator_state
    if anchor is None:
        anchor = ActuatorState(
            problem.nominal_trajectory.actuator_thrust[0],
            problem.nominal_trajectory.actuator_gimbal_angle[0],
        )
    limits = problem.actuator_parameters
    states = []
    for member in range(members):
        strong_coordinates = np.concatenate(
            (
                local[member, :PARAMETER_OFFSET],
                np.zeros(18, dtype=float),
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
    return QOnlyInitialEnsemble(local, tuple(states))


BagProgressCallback = Callable[[int, str, object], None]


@dataclass(frozen=True)
class RealDiagonalQEstimationResult:
    """Shared diagonal-Q EM result and the final per-bag E-step paths."""

    prepared_bags: Tuple[PreparedDiagonalQBag, ...]
    em_result: DiagonalQEmResult
    final_e_steps: Tuple[
        Tuple[str, StochasticClosedLoopEStepResult], ...
    ]

    def __post_init__(self) -> None:
        bags = tuple(self.prepared_bags)
        if not bags or any(
            not isinstance(value, PreparedDiagonalQBag) for value in bags
        ):
            raise ValueError("prepared_bags cannot be empty")
        identifiers = tuple(value.bag_id for value in bags)
        if (
            identifiers != tuple(sorted(identifiers))
            or len(set(identifiers)) != len(identifiers)
        ):
            raise ValueError("prepared_bags must be sorted and unique")
        if not isinstance(self.em_result, DiagonalQEmResult):
            raise TypeError("em_result must be DiagonalQEmResult")
        steps = tuple(self.final_e_steps)
        if tuple(value[0] for value in steps) != identifiers or any(
            not isinstance(value[1], StochasticClosedLoopEStepResult)
            for value in steps
        ):
            raise ValueError("final_e_steps must align with prepared_bags")
        expectations = {
            value.bag_id: value
            for value in self.em_result.final_expectations
        }
        for bag_id, result in steps:
            expectation = expectations[bag_id]
            if (
                not np.array_equal(result.times, expectation.times)
                or not np.array_equal(
                    result.smoothed_wrench_ensemble,
                    expectation.smoothed_wrench,
                )
                or result.filter_log_likelihood
                != expectation.approx_log_likelihood
            ):
                raise ValueError("final E-step output does not match EM trace")
        object.__setattr__(self, "prepared_bags", bags)
        object.__setattr__(self, "final_e_steps", steps)

    @property
    def covariance(self) -> BodyWrenchDiagonalCovariance:
        return self.em_result.covariance

    def e_step(self, bag_id: str) -> StochasticClosedLoopEStepResult:
        identifier = _bag_id(bag_id)
        for current_id, value in self.final_e_steps:
            if current_id == identifier:
                return value
        raise KeyError("unknown Q-estimation bag {!r}".format(identifier))


def _ordered_bags(
    values: Iterable[PreparedDiagonalQBag],
) -> Tuple[PreparedDiagonalQBag, ...]:
    bags = tuple(values)
    if not bags or any(
        not isinstance(value, PreparedDiagonalQBag) for value in bags
    ):
        raise ValueError("at least one PreparedDiagonalQBag is required")
    ordered = tuple(sorted(bags, key=lambda value: value.bag_id))
    if len({value.bag_id for value in ordered}) != len(ordered):
        raise ValueError("prepared Q bags must have unique bag IDs")
    fingerprints = {
        value.configuration_fingerprint
        for value in ordered
        if value.configuration_fingerprint
    }
    if len(fingerprints) > 1:
        raise ValueError(
            "diagonal Q bags must share one configuration fingerprint"
        )
    return ordered


def run_real_diagonal_q_em(
    prepared_bags: Iterable[PreparedDiagonalQBag],
    config: DiagonalQEmConfig,
    *,
    ensemble_size: int = 64,
    forecast_workers: int = 1,
    seed: int = 23,
    progress_callback: Optional[ProgressCallback] = None,
    bag_progress_callback: Optional[BagProgressCallback] = None,
    cancellation_token: Optional[CancellationToken] = None,
    run_id: str = "real-diagonal-q",
) -> RealDiagonalQEstimationResult:
    """Run shared-Q EM while holding theta, delay, R, and tau-c fixed."""

    bags = _ordered_bags(prepared_bags)
    if not isinstance(config, DiagonalQEmConfig):
        raise TypeError("config must be DiagonalQEmConfig")
    members = _member_count(ensemble_size)
    selected_seed = _seed(seed)
    cancellation = cancellation_token or CancellationToken()
    if not isinstance(cancellation, CancellationToken):
        raise TypeError("cancellation_token must be CancellationToken")
    if bag_progress_callback is not None and not callable(
        bag_progress_callback
    ):
        raise TypeError("bag_progress_callback must be callable")
    final_steps = {}

    def expectation_step(covariance, iteration):
        expectations = []
        for bag in bags:
            cancellation.raise_if_cancelled()
            bag_seed = _bag_seed(selected_seed, bag.bag_id)
            initial = draw_q_only_initial_ensemble(
                bag, covariance, members, bag_seed
            )
            problem = bag.problem

            def forward_event(event):
                if bag_progress_callback is not None:
                    bag_progress_callback(iteration, bag.bag_id, event)

            result = run_stochastic_closed_loop_e_step(
                times=problem.observations.times,
                references=problem.references,
                observed_position=problem.observations.position,
                observed_orientation_xyzw=(
                    problem.observations.orientation_xyzw
                ),
                initial_state_ensemble=initial.filter_states,
                controller=GrapeController(
                    problem.controller_configuration,
                    problem.controller_parameters,
                    problem.geometry,
                    articulated_model=GrapeArticulatedModel(),
                ),
                plant=FullSixDofPlant(
                    problem.parameter_chart.decode(np.zeros(18)),
                    problem.geometry,
                ),
                actuator_parameters=problem.actuator_parameters,
                wrench_covariance=covariance,
                correlation_time=bag.calibration.correlation_time,
                observation_covariance=bag.observation_covariance,
                seed=int((bag_seed + 2) % (2**32)),
                forecast_workers=forecast_workers,
                progress_callback=(
                    forward_event
                    if bag_progress_callback is not None
                    else None
                ),
                cancellation_token=cancellation,
                progress_run_id="{}-{}-{}".format(
                    run_id, iteration, bag.bag_id
                ),
                bag_id=bag.bag_id,
            )
            final_steps[bag.bag_id] = result
            expectations.append(
                DiagonalQBagExpectation(
                    bag_id=bag.bag_id,
                    times=result.times,
                    correlation_time=bag.calibration.correlation_time,
                    smoothed_wrench=result.smoothed_wrench_ensemble,
                    approx_log_likelihood=result.filter_log_likelihood,
                )
            )
        return tuple(expectations)

    em_result = run_diagonal_q_em(
        tuple(value.initial_pilot for value in bags),
        expectation_step,
        config,
        progress_callback=progress_callback,
        cancellation_token=cancellation,
        run_id=run_id,
    )
    return RealDiagonalQEstimationResult(
        prepared_bags=bags,
        em_result=em_result,
        final_e_steps=tuple(
            (bag.bag_id, final_steps[bag.bag_id]) for bag in bags
        ),
    )


__all__ = [
    "Q_ONLY_LOCAL_DIMENSION",
    "Q_ONLY_MINIMUM_MEMBER_COUNT",
    "BagProgressCallback",
    "PreparedDiagonalQBag",
    "QOnlyInitialEnsemble",
    "RealDiagonalQEstimationResult",
    "draw_q_only_initial_ensemble",
    "prepare_real_diagonal_q_bag",
    "run_real_diagonal_q_em",
]
