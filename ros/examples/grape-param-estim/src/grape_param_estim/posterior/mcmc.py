"""Single-chain execution and immutable trace contracts for posterior MCMC."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Tuple

import numpy as np

from grape_param_estim.posterior.delayed_acceptance import (
    DelayedAcceptanceSampler,
    DelayedAcceptanceStep,
    STATIC_PARAMETER_DIMENSION,
    TargetEvaluation,
)


@dataclass(frozen=True)
class McmcChainSettings:
    """Fixed proposal counts and identities for one resumable chain."""

    chain_id: str
    mode_id: str
    warmup_steps: int
    retained_draws: int
    thinning: int = 1

    def __post_init__(self) -> None:
        for name in ("chain_id", "mode_id"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or value.strip() != value
            ):
                raise ValueError("{} must be a canonical non-empty string".format(name))
        for name, minimum in (
            ("warmup_steps", 0),
            ("retained_draws", 1),
            ("thinning", 1),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(
                    "{} must be an integer >= {}".format(name, minimum)
                )
            object.__setattr__(self, name, int(value))

    @property
    def total_transitions(self) -> int:
        return self.warmup_steps + self.retained_draws * self.thinning


@dataclass(frozen=True)
class KernelAcceptanceSummary:
    """Proposal and nonlinear-solve counts for one named mixture component."""

    attempts: int
    stage_one_accepted: int
    stage_two_attempted: int
    stage_two_accepted: int
    full_target_cache_hits: int
    inner_solve_failures: int
    inner_iterations: int

    def __post_init__(self) -> None:
        names = (
            "attempts",
            "stage_one_accepted",
            "stage_two_attempted",
            "stage_two_accepted",
            "full_target_cache_hits",
            "inner_solve_failures",
            "inner_iterations",
        )
        for name in names:
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError("{} must be a non-negative integer".format(name))
            object.__setattr__(self, name, int(value))
        if not (
            self.stage_two_accepted
            <= self.stage_two_attempted
            <= self.stage_one_accepted
            <= self.attempts
        ):
            raise ValueError("acceptance counts are not nested")
        if self.full_target_cache_hits > self.stage_two_attempted:
            raise ValueError("cache hits cannot exceed full target attempts")
        if self.inner_solve_failures > self.stage_two_attempted:
            raise ValueError("inner failures cannot exceed full target attempts")

    @property
    def stage_one_acceptance_rate(self) -> float:
        return (
            float(self.stage_one_accepted) / self.attempts
            if self.attempts
            else float("nan")
        )

    @property
    def stage_two_acceptance_rate(self) -> float:
        return (
            float(self.stage_two_accepted) / self.stage_two_attempted
            if self.stage_two_attempted
            else float("nan")
        )

    @property
    def overall_acceptance_rate(self) -> float:
        return (
            float(self.stage_two_accepted) / self.attempts
            if self.attempts
            else float("nan")
        )

    @property
    def average_inner_iterations(self) -> float:
        uncached = self.stage_two_attempted - self.full_target_cache_hits
        return (
            float(self.inner_iterations) / uncached
            if uncached
            else float("nan")
        )


@dataclass(frozen=True)
class McmcChainResult:
    """Equal-weight retained draws and all-transition acceptance summaries."""

    chain_id: str
    mode_id: str
    sample_id: np.ndarray
    draw_index: np.ndarray
    static_coordinate: np.ndarray
    delay: np.ndarray
    log_density: np.ndarray
    attempted_kernel: np.ndarray
    accepted_kernel: np.ndarray
    accepted: np.ndarray
    stage_one_accepted: np.ndarray
    stage_two_attempted: np.ndarray
    full_target_cache_hit: np.ndarray
    inner_solve_failed: np.ndarray
    inner_iterations: np.ndarray
    warmup_steps: int
    thinning: int
    total_transitions: int
    kernel_summaries: Mapping[str, KernelAcceptanceSummary]
    graph_objective: Optional[np.ndarray] = None
    local_log_determinant: Optional[np.ndarray] = None
    delay_log_prior: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        for name in ("chain_id", "mode_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError("{} must be a non-empty string".format(name))
        sample_id = np.asarray(self.sample_id)
        if sample_id.ndim != 1 or sample_id.dtype.kind not in "US":
            raise ValueError("sample_id must be a one-dimensional string array")
        draw_count = sample_id.size
        if draw_count == 0 or len(set(sample_id.tolist())) != draw_count:
            raise ValueError("sample_id must be non-empty and unique")
        specifications = (
            ("draw_index", (draw_count,), "iu"),
            (
                "static_coordinate",
                (draw_count, STATIC_PARAMETER_DIMENSION),
                "f",
            ),
            ("delay", (draw_count,), "f"),
            ("log_density", (draw_count,), "f"),
            ("attempted_kernel", (draw_count,), "US"),
            ("accepted_kernel", (draw_count,), "US"),
            ("accepted", (draw_count,), "b"),
            ("stage_one_accepted", (draw_count,), "b"),
            ("stage_two_attempted", (draw_count,), "b"),
            ("full_target_cache_hit", (draw_count,), "b"),
            ("inner_solve_failed", (draw_count,), "b"),
            ("inner_iterations", (draw_count,), "iu"),
        )
        arrays = {"sample_id": sample_id}
        for name, shape, kinds in specifications:
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype.kind not in kinds:
                raise ValueError(
                    "{} must have shape {} and dtype kind {}".format(
                        name, shape, kinds
                    )
                )
            if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
                raise ValueError("{} must be finite".format(name))
            arrays[name] = value
        if np.any(np.diff(arrays["draw_index"]) <= 0):
            raise ValueError("draw_index must be strictly increasing")
        if np.any(arrays["inner_iterations"] < 0):
            raise ValueError("inner_iterations must be non-negative")
        for index, was_accepted in enumerate(arrays["accepted"]):
            expected = (
                str(arrays["attempted_kernel"][index])
                if was_accepted
                else ""
            )
            if str(arrays["accepted_kernel"][index]) != expected:
                raise ValueError(
                    "accepted_kernel must name only accepted proposals"
                )
        for name, value in arrays.items():
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        for name in ("warmup_steps", "thinning", "total_transitions"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError("{} must be non-negative".format(name))
            object.__setattr__(self, name, int(value))
        if self.thinning < 1:
            raise ValueError("thinning must be positive")
        expected_transitions = self.warmup_steps + int(
            arrays["draw_index"][-1]
        ) * self.thinning
        if self.total_transitions != expected_transitions:
            raise ValueError("total_transitions disagrees with retained draws")
        summaries = {}
        for name, summary in self.kernel_summaries.items():
            if not isinstance(name, str) or not name:
                raise ValueError("kernel summary names must be non-empty strings")
            if not isinstance(summary, KernelAcceptanceSummary):
                raise TypeError("kernel summaries have invalid values")
            summaries[name] = summary
        if sum(value.attempts for value in summaries.values()) != self.total_transitions:
            raise ValueError("kernel attempts must equal total transitions")
        object.__setattr__(self, "kernel_summaries", MappingProxyType(summaries))
        component_names = (
            "graph_objective",
            "local_log_determinant",
            "delay_log_prior",
        )
        component_values = tuple(getattr(self, name) for name in component_names)
        supplied = tuple(value is not None for value in component_values)
        if any(supplied) and not all(supplied):
            raise ValueError(
                "target density component traces must be supplied together"
            )
        if all(supplied):
            for name, value in zip(component_names, component_values):
                array = np.asarray(value, dtype=float)
                if array.shape != (draw_count,) or not np.all(
                    np.isfinite(array)
                ):
                    raise ValueError(
                        "{} must contain one finite value per draw".format(
                            name
                        )
                    )
                copied = array.copy()
                copied.setflags(write=False)
                object.__setattr__(self, name, copied)
            reconstructed = (
                self.delay_log_prior
                - self.graph_objective
                - 0.5 * self.local_log_determinant
            )
            if not np.allclose(
                self.log_density,
                reconstructed,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "target component traces do not reconstruct log_density"
                )


class McmcCancelled(RuntimeError):
    """Cancellation observed at a proposal boundary."""

    def __init__(self, completed_transitions: int):
        self.completed_transitions = int(completed_transitions)
        super().__init__(
            "MCMC cancelled after {} transitions".format(
                self.completed_transitions
            )
        )


def _new_counter() -> dict:
    return {
        "attempts": 0,
        "stage_one_accepted": 0,
        "stage_two_attempted": 0,
        "stage_two_accepted": 0,
        "full_target_cache_hits": 0,
        "inner_solve_failures": 0,
        "inner_iterations": 0,
    }


def _update_counter(counter: dict, step: DelayedAcceptanceStep) -> None:
    counter["attempts"] += 1
    counter["stage_one_accepted"] += int(step.stage_one_accepted)
    counter["stage_two_attempted"] += int(step.stage_two_attempted)
    counter["stage_two_accepted"] += int(step.stage_two_accepted)
    counter["full_target_cache_hits"] += int(step.full_target_cache_hit)
    counter["inner_solve_failures"] += int(step.inner_solve_failed)
    if step.candidate_evaluation is not None and not step.full_target_cache_hit:
        counter["inner_iterations"] += (
            step.candidate_evaluation.inner_iterations
        )


def run_mcmc_chain(
    sampler: DelayedAcceptanceSampler,
    initial: TargetEvaluation,
    settings: McmcChainSettings,
    random_state: object,
    *,
    cancellation_requested: Optional[Callable[[], bool]] = None,
    progress: Optional[
        Callable[[int, int, DelayedAcceptanceStep], None]
    ] = None,
) -> McmcChainResult:
    """Run one fixed-kernel chain and check cancellation between proposals."""

    if not isinstance(sampler, DelayedAcceptanceSampler):
        raise TypeError("sampler must be a DelayedAcceptanceSampler")
    if not isinstance(initial, TargetEvaluation) or not initial.successful:
        raise ValueError("initial target evaluation must be successful")
    if not isinstance(settings, McmcChainSettings):
        raise TypeError("settings must be McmcChainSettings")
    if cancellation_requested is not None and not callable(
        cancellation_requested
    ):
        raise TypeError("cancellation_requested must be callable")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")

    sampler.cache.store(initial)
    current = initial
    counters = {
        kernel.name: _new_counter() for kernel in sampler.proposal.kernels
    }
    retained = []
    total = settings.total_transitions
    for transition_index in range(1, total + 1):
        if cancellation_requested is not None and cancellation_requested():
            raise McmcCancelled(transition_index - 1)
        step = sampler.step(current, random_state)
        current = step.current
        _update_counter(counters[step.kernel_name], step)
        if progress is not None:
            progress(transition_index, total, step)
        after_warmup = transition_index - settings.warmup_steps
        if after_warmup > 0 and after_warmup % settings.thinning == 0:
            retained.append((after_warmup // settings.thinning, current, step))

    sample_id = np.asarray(
        tuple(
            "{}:{:08d}".format(settings.chain_id, draw_index)
            for draw_index, _, _ in retained
        )
    )
    draw_index = np.asarray(
        tuple(item[0] for item in retained), dtype=np.int64
    )
    static_coordinate = np.vstack(
        tuple(item[1].point.static_coordinate for item in retained)
    )
    delay = np.asarray(tuple(item[1].point.delay for item in retained))
    log_density = np.asarray(
        tuple(item[1].log_density for item in retained)
    )
    component_availability = tuple(
        item[1].has_density_components for item in retained
    )
    if any(component_availability) and not all(component_availability):
        raise RuntimeError(
            "retained target evaluations mix decomposed and opaque densities"
        )
    if all(component_availability):
        graph_objective = np.asarray(
            tuple(item[1].graph_objective for item in retained), dtype=float
        )
        local_log_determinant = np.asarray(
            tuple(
                item[1].local_log_determinant for item in retained
            ),
            dtype=float,
        )
        delay_log_prior = np.asarray(
            tuple(item[1].delay_log_prior for item in retained), dtype=float
        )
    else:
        graph_objective = None
        local_log_determinant = None
        delay_log_prior = None
    attempted_kernel = np.asarray(
        tuple(item[2].kernel_name for item in retained)
    )
    accepted = np.asarray(
        tuple(item[2].accepted for item in retained), dtype=bool
    )
    accepted_kernel = np.asarray(
        tuple(
            item[2].kernel_name if item[2].accepted else ""
            for item in retained
        )
    )
    stage_one_accepted = np.asarray(
        tuple(item[2].stage_one_accepted for item in retained), dtype=bool
    )
    stage_two_attempted = np.asarray(
        tuple(item[2].stage_two_attempted for item in retained), dtype=bool
    )
    full_target_cache_hit = np.asarray(
        tuple(item[2].full_target_cache_hit for item in retained), dtype=bool
    )
    inner_solve_failed = np.asarray(
        tuple(item[2].inner_solve_failed for item in retained), dtype=bool
    )
    inner_iterations = np.asarray(
        tuple(
            item[2].candidate_evaluation.inner_iterations
            if item[2].candidate_evaluation is not None
            else 0
            for item in retained
        ),
        dtype=np.int64,
    )
    summaries = {
        name: KernelAcceptanceSummary(**counter)
        for name, counter in counters.items()
    }
    return McmcChainResult(
        chain_id=settings.chain_id,
        mode_id=settings.mode_id,
        sample_id=sample_id,
        draw_index=draw_index,
        static_coordinate=static_coordinate,
        delay=delay,
        log_density=log_density,
        attempted_kernel=attempted_kernel,
        accepted_kernel=accepted_kernel,
        accepted=accepted,
        stage_one_accepted=stage_one_accepted,
        stage_two_attempted=stage_two_attempted,
        full_target_cache_hit=full_target_cache_hit,
        inner_solve_failed=inner_solve_failed,
        inner_iterations=inner_iterations,
        warmup_steps=settings.warmup_steps,
        thinning=settings.thinning,
        total_transitions=total,
        kernel_summaries=summaries,
        graph_objective=graph_objective,
        local_log_determinant=local_log_determinant,
        delay_log_prior=delay_log_prior,
    )


__all__ = [
    "KernelAcceptanceSummary",
    "McmcCancelled",
    "McmcChainResult",
    "McmcChainSettings",
    "run_mcmc_chain",
]
