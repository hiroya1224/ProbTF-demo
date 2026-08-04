"""Posterior cross-evaluation and bounded PID candidate particle search."""

from dataclasses import dataclass
import hashlib
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.controller_config import PidGainConfiguration
from grape_param_estim.pid.metrics import (
    CandidateMetricSummary,
    ForecastMetricRecord,
    ForecastMetrics,
    RecommendationDecision,
    decide_recommendation,
    summarize_forecast_records,
)
from grape_param_estim.pid.proposal import (
    PhysicalPlantPosterior,
    PhysicalPlantSample,
    PidCandidate,
    PidProposalPopulation,
    current_pid_candidate,
    sample_pid_candidate,
)


ZERO_MODEL_DISCREPANCY = "zero_model_discrepancy"
SAMPLE_MODEL_DISCREPANCY = "sample_model_discrepancy"
MODEL_DISCREPANCY_POLICIES = (
    ZERO_MODEL_DISCREPANCY,
    SAMPLE_MODEL_DISCREPANCY,
)
BODY_WRENCH_MODEL_DISCREPANCY = "body_wrench"
SPECIFIC_ACCELERATION_MODEL_DISCREPANCY = "specific_acceleration"
MODEL_DISCREPANCY_QUANTITIES = (
    BODY_WRENCH_MODEL_DISCREPANCY,
    SPECIFIC_ACCELERATION_MODEL_DISCREPANCY,
)


@dataclass(frozen=True)
class ModelDiscrepancyConfiguration:
    """Future discrepancy policy with an auditable common-random-number seed."""

    policy: str
    diagonal_q: np.ndarray
    base_seed: int
    residual_quantity: str
    replicates: int = 1

    def __post_init__(self) -> None:
        policy = str(self.policy)
        quantity = str(self.residual_quantity)
        q = np.asarray(self.diagonal_q, dtype=float)
        seed = self.base_seed
        replicates = self.replicates
        if policy not in MODEL_DISCREPANCY_POLICIES:
            raise ValueError("unknown model discrepancy policy")
        if quantity not in MODEL_DISCREPANCY_QUANTITIES:
            raise ValueError(
                "residual_quantity must explicitly be body_wrench or "
                "specific_acceleration"
            )
        if q.shape != (6,) or np.any(~np.isfinite(q)) or np.any(q < 0.0):
            raise ValueError("diagonal_q must contain six non-negative values")
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
            or seed >= 2 ** 64
        ):
            raise ValueError("base_seed must be an unsigned 64-bit integer")
        if (
            isinstance(replicates, (bool, np.bool_))
            or not isinstance(replicates, (int, np.integer))
            or replicates < 1
        ):
            raise ValueError("replicates must be a positive integer")
        copied = q.copy()
        copied.setflags(write=False)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "residual_quantity", quantity)
        object.__setattr__(self, "diagonal_q", copied)
        object.__setattr__(self, "base_seed", int(seed))
        object.__setattr__(self, "replicates", int(replicates))

    def seed_for(
        self, sample_id: str, bag_id: str, replicate_index: int
    ) -> int:
        """Derive a stable seed independent of candidate and loop ordering."""

        sample = str(sample_id)
        bag = str(bag_id)
        replicate = int(replicate_index)
        if not sample or not bag or replicate < 0 or replicate >= self.replicates:
            raise ValueError("discrepancy seed identity is invalid")
        payload = "{}\x1f{}\x1f{}\x1f{}".format(
            self.base_seed, sample, bag, replicate
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")

    def realization(
        self, sample_id: str, bag_id: str, replicate_index: int
    ) -> "ModelDiscrepancyRealization":
        return ModelDiscrepancyRealization(
            policy=self.policy,
            diagonal_q=self.diagonal_q,
            seed=self.seed_for(sample_id, bag_id, replicate_index),
            replicate_index=replicate_index,
            residual_quantity=self.residual_quantity,
        )


@dataclass(frozen=True)
class ModelDiscrepancyRealization:
    """One repeatable future discrepancy stream shared by every candidate."""

    policy: str
    diagonal_q: np.ndarray
    seed: int
    replicate_index: int
    residual_quantity: str

    def __post_init__(self) -> None:
        policy = str(self.policy)
        quantity = str(self.residual_quantity)
        q = np.asarray(self.diagonal_q, dtype=float)
        if policy not in MODEL_DISCREPANCY_POLICIES:
            raise ValueError("unknown model discrepancy policy")
        if quantity not in MODEL_DISCREPANCY_QUANTITIES:
            raise ValueError(
                "residual_quantity must explicitly be body_wrench or "
                "specific_acceleration"
            )
        if q.shape != (6,) or np.any(~np.isfinite(q)) or np.any(q < 0.0):
            raise ValueError("diagonal_q must contain six non-negative values")
        for name in ("seed", "replicate_index"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError("{} must be non-negative".format(name))
            object.__setattr__(self, name, int(value))
        copied = q.copy()
        copied.setflags(write=False)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "residual_quantity", quantity)
        object.__setattr__(self, "diagonal_q", copied)

    def interval_average_residual(
        self, time_step: Sequence[float]
    ) -> np.ndarray:
        """Sample the configured residual with covariance ``Q / dt``."""

        dt = np.asarray(time_step, dtype=float)
        if (
            dt.ndim != 1
            or dt.size < 1
            or np.any(~np.isfinite(dt))
            or np.any(dt <= 0.0)
        ):
            raise ValueError("time_step must contain positive interval lengths")
        if self.policy == ZERO_MODEL_DISCREPANCY:
            return np.zeros((dt.size, 6), dtype=float)
        # The generator is intentionally recreated: a realization is a value
        # object, and repeated calls with the same horizon must be identical.
        random_state = np.random.default_rng(self.seed)
        standard = random_state.standard_normal((dt.size, 6))
        return standard * np.sqrt(self.diagonal_q[None, :] / dt[:, None])


PidForecastEvaluator = Callable[
    [
        PidCandidate,
        PhysicalPlantSample,
        str,
        ModelDiscrepancyRealization,
    ],
    ForecastMetrics,
]


class PidEvaluationCancelled(RuntimeError):
    """Cancellation observed at a PID forecast boundary."""

    def __init__(self, completed_forecasts: int):
        self.completed_forecasts = int(completed_forecasts)
        super().__init__(
            "PID evaluation cancelled after {} forecasts".format(
                self.completed_forecasts
            )
        )


def _canonical_bag_ids(bag_ids: Sequence[str]) -> Tuple[str, ...]:
    result = tuple(str(value) for value in bag_ids)
    if (
        not result
        or len(set(result)) != len(result)
        or any(not value or value.strip() != value for value in result)
    ):
        raise ValueError("bag_ids must contain unique canonical strings")
    return result


def _selected_samples(
    posterior: PhysicalPlantPosterior,
    sample_ids: Optional[Sequence[str]],
) -> Tuple[PhysicalPlantSample, ...]:
    if sample_ids is None:
        return posterior.samples
    identifiers = tuple(str(value) for value in sample_ids)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("sample subset IDs must be non-empty and unique")
    return tuple(posterior.sample(value) for value in identifiers)


def _candidates_with_current(
    candidates: Sequence[PidCandidate], current: PidGainConfiguration
) -> Tuple[PidCandidate, ...]:
    selected = tuple(candidates)
    if any(not isinstance(value, PidCandidate) for value in selected):
        raise TypeError("candidates must contain PidCandidate values")
    current_candidates = tuple(
        value for value in selected if value.candidate_id == "current"
    )
    if len(current_candidates) > 1:
        raise ValueError("current baseline may be supplied only once")
    if current_candidates:
        baseline = current_candidates[0]
        if baseline.source != "current" or not np.array_equal(
            baseline.configuration.values, current.values
        ):
            raise ValueError("supplied current candidate disagrees with baseline")
        selected = (baseline,) + tuple(
            value for value in selected if value.candidate_id != "current"
        )
    else:
        selected = (current_pid_candidate(current),) + selected
    identifiers = tuple(value.candidate_id for value in selected)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate IDs must be unique")
    return selected


@dataclass(frozen=True)
class PidCandidateEvaluation:
    """Complete candidate x plant sample x bag x replicate cross-evaluation."""

    candidates: Tuple[PidCandidate, ...]
    plant_sample_ids: Tuple[str, ...]
    bag_ids: Tuple[str, ...]
    records: Tuple[ForecastMetricRecord, ...]
    summaries: Tuple[CandidateMetricSummary, ...]
    decision: RecommendationDecision
    discrepancy: ModelDiscrepancyConfiguration
    plant_sample_subset_method: str

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        sample_ids = tuple(str(value) for value in self.plant_sample_ids)
        bag_ids = tuple(str(value) for value in self.bag_ids)
        records = tuple(self.records)
        summaries = tuple(self.summaries)
        if (
            not candidates
            or candidates[0].candidate_id != "current"
            or any(not isinstance(value, PidCandidate) for value in candidates)
            or len({value.candidate_id for value in candidates}) != len(candidates)
            or not sample_ids
            or len(set(sample_ids)) != len(sample_ids)
            or not bag_ids
            or len(set(bag_ids)) != len(bag_ids)
            or any(not isinstance(value, ForecastMetricRecord) for value in records)
            or any(not isinstance(value, CandidateMetricSummary) for value in summaries)
            or not isinstance(self.decision, RecommendationDecision)
            or not isinstance(self.discrepancy, ModelDiscrepancyConfiguration)
            or not str(self.plant_sample_subset_method)
        ):
            raise ValueError("PID candidate evaluation contract is invalid")
        candidate_ids = tuple(value.candidate_id for value in candidates)
        if tuple(value.candidate_id for value in summaries) != candidate_ids:
            raise ValueError("candidate summary order must match candidates")
        expected = {
            (candidate, sample, bag, replicate)
            for candidate in candidate_ids
            for sample in sample_ids
            for bag in bag_ids
            for replicate in range(self.discrepancy.replicates)
        }
        actual = {
            (
                value.candidate_id,
                value.sample_id,
                value.bag_id,
                value.replicate_index,
            )
            for value in records
        }
        if actual != expected or len(records) != len(expected):
            raise ValueError("forecasts must cover the full Cartesian product")
        for candidate in candidate_ids:
            selected_seeds = {
                (value.sample_id, value.bag_id, value.replicate_index): value.discrepancy_seed
                for value in records
                if value.candidate_id == candidate
            }
            reference_seeds = {
                (sample, bag, replicate): self.discrepancy.seed_for(
                    sample, bag, replicate
                )
                for sample in sample_ids
                for bag in bag_ids
                for replicate in range(self.discrepancy.replicates)
            }
            if selected_seeds != reference_seeds:
                raise ValueError("candidate forecasts did not use common random numbers")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "plant_sample_ids", sample_ids)
        object.__setattr__(self, "bag_ids", bag_ids)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "summaries", summaries)
        object.__setattr__(
            self, "plant_sample_subset_method", str(self.plant_sample_subset_method)
        )

    @property
    def recommendation_available(self) -> bool:
        return self.decision.recommendation_available

    def summary(self, candidate_id: str) -> CandidateMetricSummary:
        matches = tuple(
            value for value in self.summaries if value.candidate_id == str(candidate_id)
        )
        if len(matches) != 1:
            raise KeyError("unknown candidate summary")
        return matches[0]


def evaluate_pid_candidates(
    candidates: Sequence[PidCandidate],
    posterior: PhysicalPlantPosterior,
    bag_ids: Sequence[str],
    evaluator: PidForecastEvaluator,
    current: PidGainConfiguration,
    discrepancy: ModelDiscrepancyConfiguration,
    *,
    sample_ids: Optional[Sequence[str]] = None,
    plant_sample_subset_method: Optional[str] = None,
    quantile_level: float = 0.95,
    cvar_level: float = 0.90,
    cancellation_requested: Optional[Callable[[], bool]] = None,
) -> PidCandidateEvaluation:
    """Cross-evaluate every candidate on every selected sample and bag."""

    if not isinstance(posterior, PhysicalPlantPosterior):
        raise TypeError("posterior must be PhysicalPlantPosterior")
    if not isinstance(current, PidGainConfiguration):
        raise TypeError("current must be PidGainConfiguration")
    if not isinstance(discrepancy, ModelDiscrepancyConfiguration):
        raise TypeError("discrepancy has the wrong type")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    selected_candidates = _candidates_with_current(candidates, current)
    samples = _selected_samples(posterior, sample_ids)
    selected_bags = _canonical_bag_ids(bag_ids)
    known_sample_ids = set(posterior.sample_id.tolist())
    for candidate in selected_candidates:
        if (
            candidate.source == "sample-derived"
            and candidate.source_sample_id not in known_sample_ids
        ):
            raise ValueError("derived candidate source sample is absent")
    records = []
    completed = 0
    for candidate in selected_candidates:
        for sample in samples:
            for bag_id in selected_bags:
                for replicate in range(discrepancy.replicates):
                    if cancellation_requested is not None and cancellation_requested():
                        raise PidEvaluationCancelled(completed)
                    realization = discrepancy.realization(
                        sample.sample_id, bag_id, replicate
                    )
                    metrics = evaluator(candidate, sample, bag_id, realization)
                    if not isinstance(metrics, ForecastMetrics):
                        raise TypeError("evaluator must return ForecastMetrics")
                    records.append(
                        ForecastMetricRecord(
                            candidate_id=candidate.candidate_id,
                            sample_id=sample.sample_id,
                            bag_id=bag_id,
                            replicate_index=replicate,
                            discrepancy_seed=realization.seed,
                            metrics=metrics,
                        )
                    )
                    completed += 1
    summaries = tuple(
        summarize_forecast_records(
            tuple(
                value
                for value in records
                if value.candidate_id == candidate.candidate_id
            ),
            current,
            candidate.configuration,
            quantile_level=quantile_level,
            cvar_level=cvar_level,
        )
        for candidate in selected_candidates
    )
    method = (
        str(plant_sample_subset_method)
        if plant_sample_subset_method is not None
        else (
            "all_equal_weight_mcmc_samples"
            if sample_ids is None
            else "explicit_equal_weight_mcmc_subset"
        )
    )
    if not method:
        raise ValueError("plant_sample_subset_method cannot be empty")
    return PidCandidateEvaluation(
        candidates=selected_candidates,
        plant_sample_ids=tuple(value.sample_id for value in samples),
        bag_ids=selected_bags,
        records=tuple(records),
        summaries=summaries,
        decision=decide_recommendation(summaries),
        discrepancy=discrepancy,
        plant_sample_subset_method=method,
    )


def _pairwise_squared_distance(features: np.ndarray) -> np.ndarray:
    difference = features[:, None, :] - features[None, :, :]
    return np.sum(difference * difference, axis=2)


def _medoid_indices(features: np.ndarray, count: int) -> Tuple[int, ...]:
    """Deterministic unweighted k-medoids; returned points are always raw rows."""

    values = np.asarray(features, dtype=float)
    selected_count = int(count)
    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] < 1
        or np.any(~np.isfinite(values))
        or selected_count < 1
        or selected_count > values.shape[0]
    ):
        raise ValueError("k-medoids inputs are invalid")
    if selected_count == values.shape[0]:
        return tuple(range(values.shape[0]))
    scale = np.std(values, axis=0)
    normalized = values / np.where(scale > 0.0, scale, 1.0)
    distance = _pairwise_squared_distance(normalized)
    medoids = [int(np.argmin(np.sum(distance, axis=1)))]
    while len(medoids) < selected_count:
        nearest = np.min(distance[:, medoids], axis=1)
        nearest[np.asarray(medoids, dtype=int)] = -1.0
        medoids.append(int(np.argmax(nearest)))
    for _iteration in range(25):
        assignment = np.argmin(distance[:, medoids], axis=1)
        updated = []
        for cluster in range(selected_count):
            cluster_rows = np.flatnonzero(assignment == cluster)
            if cluster_rows.size == 0:
                updated.append(medoids[cluster])
            else:
                within = distance[np.ix_(cluster_rows, cluster_rows)]
                updated.append(
                    int(
                        cluster_rows[
                            int(np.argmin(np.sum(within, axis=1)))
                        ]
                    )
                )
        if updated == medoids:
            break
        if len(set(updated)) != len(updated):
            break
        medoids = updated
    return tuple(sorted(medoids))


def select_proposal_medoids(
    proposals: PidProposalPopulation, maximum_candidates: int
) -> Tuple[str, ...]:
    """Select raw sample proposals in log-scale space without averaging gains."""

    if not isinstance(proposals, PidProposalPopulation):
        raise TypeError("proposals must be PidProposalPopulation")
    maximum = maximum_candidates
    if (
        isinstance(maximum, (bool, np.bool_))
        or not isinstance(maximum, (int, np.integer))
        or maximum < 1
    ):
        raise ValueError("maximum_candidates must be positive")
    count = min(int(maximum), proposals.source_sample_id.size)
    features = np.concatenate(
        (
            np.log(proposals.group_scales),
            proposals.source_delay[:, None],
        ),
        axis=1,
    )
    indices = _medoid_indices(features, count)
    return tuple(str(proposals.source_sample_id[index]) for index in indices)


def build_initial_candidate_population(
    proposals: PidProposalPopulation,
    *,
    maximum_derived_candidates: Optional[int] = None,
    user_candidates: Sequence[PidCandidate] = tuple(),
) -> Tuple[PidCandidate, ...]:
    """Build current + raw sample-derived + user exact candidate population."""

    if not isinstance(proposals, PidProposalPopulation):
        raise TypeError("proposals must be PidProposalPopulation")
    if maximum_derived_candidates is None:
        sample_ids = tuple(str(value) for value in proposals.source_sample_id)
    else:
        sample_ids = select_proposal_medoids(
            proposals, maximum_derived_candidates
        )
    users = tuple(user_candidates)
    if any(
        not isinstance(value, PidCandidate) or value.source != "user"
        for value in users
    ):
        raise ValueError("user_candidates must contain exact user candidates")
    result = (
        (current_pid_candidate(proposals.current),)
        + tuple(sample_pid_candidate(proposals, value) for value in sample_ids)
        + users
    )
    identifiers = tuple(value.candidate_id for value in result)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("initial candidate IDs must be unique")
    return result


@dataclass(frozen=True)
class ParticleRefinementSettings:
    maximum_generations: int
    survivor_count: int
    mutations_per_survivor: int
    log_gain_standard_deviation: float
    maximum_log_gain_step: float
    random_seed: int
    stagnation_generations: int = 2
    maximum_finalists: int = 12

    def __post_init__(self) -> None:
        for name, minimum in (
            ("maximum_generations", 0),
            ("survivor_count", 1),
            ("mutations_per_survivor", 1),
            ("stagnation_generations", 1),
            ("maximum_finalists", 1),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError("{} must be >= {}".format(name, minimum))
            object.__setattr__(self, name, int(value))
        for name in ("log_gain_standard_deviation", "maximum_log_gain_step"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("{} must be positive".format(name))
            object.__setattr__(self, name, value)
        seed = self.random_seed
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
            or seed >= 2 ** 32
        ):
            raise ValueError("random_seed must be an unsigned 32-bit integer")
        object.__setattr__(self, "random_seed", int(seed))


def _select_survivors(
    evaluation: PidCandidateEvaluation, count: int
) -> Tuple[PidCandidate, ...]:
    nondominated = set(evaluation.decision.nondominated_candidate_ids)
    pool = tuple(
        candidate
        for candidate in evaluation.candidates
        if candidate.candidate_id in nondominated
    )
    if len(pool) <= count:
        return pool
    summary_by_id = {
        value.candidate_id: value for value in evaluation.summaries
    }
    features = np.vstack(
        tuple(
            summary_by_id[value.candidate_id].performance_cost_vector()
            for value in pool
        )
    )
    indices = _medoid_indices(features, count)
    return tuple(pool[index] for index in indices)


def _mutate_candidate(
    parent: PidCandidate,
    generation: int,
    mutation_index: int,
    settings: ParticleRefinementSettings,
    random_state: np.random.RandomState,
) -> PidCandidate:
    step = random_state.normal(
        0.0,
        settings.log_gain_standard_deviation,
        size=parent.configuration.values.shape,
    )
    step = np.clip(
        step,
        -settings.maximum_log_gain_step,
        settings.maximum_log_gain_step,
    )
    values = parent.configuration.values.copy()
    configured = values > 0.0
    values[configured] *= np.exp(step[configured])
    return PidCandidate(
        candidate_id="mutation_g{:03d}_{:05d}".format(
            generation, mutation_index
        ),
        source="mutation",
        configuration=PidGainConfiguration(values),
        generation=generation,
        parent_candidate_id=parent.candidate_id,
    )


def _front_progressed(
    previous: PidCandidateEvaluation,
    current: PidCandidateEvaluation,
    tolerance: float = 1.0e-12,
) -> bool:
    previous_ids = set(previous.decision.nondominated_candidate_ids)
    previous_costs = tuple(
        value.performance_cost_vector()
        for value in previous.summaries
        if value.candidate_id in previous_ids
    )
    current_ids = set(current.decision.nondominated_candidate_ids)
    for summary in current.summaries:
        if summary.candidate_id not in current_ids:
            continue
        cost = summary.performance_cost_vector()
        if not any(
            np.all(previous_cost <= cost + tolerance)
            for previous_cost in previous_costs
        ):
            return True
    return False


@dataclass(frozen=True)
class PidParticleSearchResult:
    initial_evaluation: PidCandidateEvaluation
    generation_evaluations: Tuple[PidCandidateEvaluation, ...]
    final_evaluation: PidCandidateEvaluation
    completed_generations: int
    stop_reason: str

    def __post_init__(self) -> None:
        generations = tuple(self.generation_evaluations)
        if (
            not isinstance(self.initial_evaluation, PidCandidateEvaluation)
            or any(not isinstance(value, PidCandidateEvaluation) for value in generations)
            or not isinstance(self.final_evaluation, PidCandidateEvaluation)
            or self.completed_generations != len(generations)
            or not str(self.stop_reason)
        ):
            raise ValueError("PID particle search result is invalid")
        object.__setattr__(self, "generation_evaluations", generations)
        object.__setattr__(self, "stop_reason", str(self.stop_reason))


def refine_pid_candidate_particles(
    initial_candidates: Sequence[PidCandidate],
    posterior: PhysicalPlantPosterior,
    bag_ids: Sequence[str],
    evaluator: PidForecastEvaluator,
    current: PidGainConfiguration,
    discrepancy: ModelDiscrepancyConfiguration,
    settings: ParticleRefinementSettings,
    *,
    coarse_sample_ids: Optional[Sequence[str]] = None,
    quantile_level: float = 0.95,
    cvar_level: float = 0.90,
    cancellation_requested: Optional[Callable[[], bool]] = None,
) -> PidParticleSearchResult:
    """Optionally mutate the Pareto population, then reevaluate finalists fully."""

    if not isinstance(settings, ParticleRefinementSettings):
        raise TypeError("settings must be ParticleRefinementSettings")
    initial = evaluate_pid_candidates(
        initial_candidates,
        posterior,
        bag_ids,
        evaluator,
        current,
        discrepancy,
        sample_ids=coarse_sample_ids,
        plant_sample_subset_method=(
            "all_equal_weight_mcmc_samples"
            if coarse_sample_ids is None
            else "coarse_explicit_equal_weight_mcmc_subset"
        ),
        quantile_level=quantile_level,
        cvar_level=cvar_level,
        cancellation_requested=cancellation_requested,
    )
    random_state = np.random.RandomState(settings.random_seed)
    generation_evaluations = []
    active = initial
    stagnation = 0
    stop_reason = "maximum_generations"
    mutation_serial = 0
    for generation in range(1, settings.maximum_generations + 1):
        survivors = _select_survivors(active, settings.survivor_count)
        mutations = []
        for parent in survivors:
            for _repeat in range(settings.mutations_per_survivor):
                mutations.append(
                    _mutate_candidate(
                        parent,
                        generation,
                        mutation_serial,
                        settings,
                        random_state,
                    )
                )
                mutation_serial += 1
        population = survivors + tuple(mutations)
        evaluated = evaluate_pid_candidates(
            population,
            posterior,
            bag_ids,
            evaluator,
            current,
            discrepancy,
            sample_ids=coarse_sample_ids,
            plant_sample_subset_method=(
                "all_equal_weight_mcmc_samples"
                if coarse_sample_ids is None
                else "coarse_explicit_equal_weight_mcmc_subset"
            ),
            quantile_level=quantile_level,
            cvar_level=cvar_level,
            cancellation_requested=cancellation_requested,
        )
        generation_evaluations.append(evaluated)
        if _front_progressed(active, evaluated):
            stagnation = 0
        else:
            stagnation += 1
        active = evaluated
        if stagnation >= settings.stagnation_generations:
            stop_reason = "pareto_front_stagnated"
            break
    finalist_pool = _select_survivors(active, settings.maximum_finalists)
    final_evaluation = evaluate_pid_candidates(
        finalist_pool,
        posterior,
        bag_ids,
        evaluator,
        current,
        discrepancy,
        sample_ids=None,
        plant_sample_subset_method="final_all_equal_weight_mcmc_samples",
        quantile_level=quantile_level,
        cvar_level=cvar_level,
        cancellation_requested=cancellation_requested,
    )
    return PidParticleSearchResult(
        initial_evaluation=initial,
        generation_evaluations=tuple(generation_evaluations),
        final_evaluation=final_evaluation,
        completed_generations=len(generation_evaluations),
        stop_reason=stop_reason,
    )


__all__ = [
    "BODY_WRENCH_MODEL_DISCREPANCY",
    "MODEL_DISCREPANCY_POLICIES",
    "MODEL_DISCREPANCY_QUANTITIES",
    "ModelDiscrepancyConfiguration",
    "ModelDiscrepancyRealization",
    "ParticleRefinementSettings",
    "PidCandidateEvaluation",
    "PidEvaluationCancelled",
    "PidForecastEvaluator",
    "PidParticleSearchResult",
    "SAMPLE_MODEL_DISCREPANCY",
    "SPECIFIC_ACCELERATION_MODEL_DISCREPANCY",
    "ZERO_MODEL_DISCREPANCY",
    "build_initial_candidate_population",
    "evaluate_pid_candidates",
    "refine_pid_candidate_particles",
    "select_proposal_medoids",
]
