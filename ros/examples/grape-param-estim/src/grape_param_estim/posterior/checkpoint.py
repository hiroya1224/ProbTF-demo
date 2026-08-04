"""Strict pickle-free checkpoints for one posterior MCMC chain."""

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Mapping, Tuple
import zipfile

import numpy as np

from grape_param_estim.posterior.delayed_acceptance import (
    PosteriorPoint,
    STATIC_PARAMETER_DIMENSION,
    TargetEvaluation,
)


MCMC_CHECKPOINT_SCHEMA = "grape-param-estim/mcmc-chain-checkpoint/v1"
KERNEL_COUNTER_FIELDS = (
    "attempts",
    "stage_one_accepted",
    "stage_two_attempted",
    "stage_two_accepted",
    "full_target_cache_hits",
    "inner_solve_failures",
    "inner_iterations",
)


class McmcCheckpointError(ValueError):
    """A checkpoint is corrupt, unsafe, or incompatible."""


def _nonnegative_integer(value, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise ValueError("{} must be a non-negative integer".format(name))
    return int(value)


def _canonical_string(value, name: str) -> str:
    result = str(value)
    if not result or result.strip() != result or "\x00" in result:
        raise ValueError("{} must be a canonical non-empty string".format(name))
    return result


@dataclass(frozen=True)
class KernelCounterCheckpoint:
    """All-transition counters for one fixed proposal kernel."""

    attempts: int
    stage_one_accepted: int
    stage_two_attempted: int
    stage_two_accepted: int
    full_target_cache_hits: int
    inner_solve_failures: int
    inner_iterations: int

    def __post_init__(self) -> None:
        for name in KERNEL_COUNTER_FIELDS:
            object.__setattr__(
                self, name, _nonnegative_integer(getattr(self, name), name)
            )
        if not (
            self.stage_two_accepted
            <= self.stage_two_attempted
            <= self.stage_one_accepted
            <= self.attempts
        ):
            raise ValueError("kernel acceptance counters are not nested")
        if self.full_target_cache_hits > self.stage_two_attempted:
            raise ValueError("cache hits cannot exceed exact target attempts")
        if self.inner_solve_failures > self.stage_two_attempted:
            raise ValueError("inner failures cannot exceed exact target attempts")

    @classmethod
    def from_mapping(cls, value: Mapping[str, int]) -> "KernelCounterCheckpoint":
        if set(value) != set(KERNEL_COUNTER_FIELDS):
            raise ValueError("kernel counter fields are not exact")
        return cls(**{name: value[name] for name in KERNEL_COUNTER_FIELDS})

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in KERNEL_COUNTER_FIELDS}


@dataclass(frozen=True)
class RetainedMcmcDraw:
    """One retained exact target and the transition diagnostics that stored it."""

    draw_index: int
    evaluation: TargetEvaluation
    attempted_kernel: str
    accepted: bool
    stage_one_accepted: bool
    stage_two_attempted: bool
    full_target_cache_hit: bool
    inner_solve_failed: bool
    inner_iterations: int

    def __post_init__(self) -> None:
        draw_index = _nonnegative_integer(self.draw_index, "draw_index")
        if draw_index < 1:
            raise ValueError("draw_index must be positive")
        if not isinstance(self.evaluation, TargetEvaluation) or not self.evaluation.successful:
            raise ValueError("retained evaluation must be a successful exact target")
        kernel = _canonical_string(self.attempted_kernel, "attempted_kernel")
        boolean_names = (
            "accepted",
            "stage_one_accepted",
            "stage_two_attempted",
            "full_target_cache_hit",
            "inner_solve_failed",
        )
        for name in boolean_names:
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError("{} must be boolean".format(name))
            object.__setattr__(self, name, bool(value))
        inner_iterations = _nonnegative_integer(
            self.inner_iterations, "inner_iterations"
        )
        if self.accepted and (
            not self.stage_one_accepted or not self.stage_two_attempted
        ):
            raise ValueError("accepted transition flags are inconsistent")
        if self.stage_two_attempted and not self.stage_one_accepted:
            raise ValueError("stage two requires stage-one acceptance")
        if self.full_target_cache_hit and not self.stage_two_attempted:
            raise ValueError("cache hits require an exact target attempt")
        if self.inner_solve_failed and not self.stage_two_attempted:
            raise ValueError("inner failure requires an exact target attempt")
        if self.inner_solve_failed and self.accepted:
            raise ValueError("an inner-solve failure cannot be accepted")
        if not self.stage_two_attempted and inner_iterations != 0:
            raise ValueError("an unattempted exact target has zero iterations")
        object.__setattr__(self, "draw_index", draw_index)
        object.__setattr__(self, "attempted_kernel", kernel)
        object.__setattr__(self, "inner_iterations", inner_iterations)

    @property
    def accepted_kernel(self) -> str:
        return self.attempted_kernel if self.accepted else ""

    @property
    def stage_two_accepted(self) -> bool:
        return self.accepted


@dataclass(frozen=True)
class NumpyRandomStateCheckpoint:
    """Complete, pickle-free state of NumPy's legacy MT19937 RandomState."""

    algorithm: str
    keys: np.ndarray
    position: int
    has_gauss: int
    cached_gaussian: float

    def __post_init__(self) -> None:
        algorithm = str(self.algorithm)
        keys = np.asarray(self.keys)
        position = _nonnegative_integer(self.position, "position")
        has_gauss = _nonnegative_integer(self.has_gauss, "has_gauss")
        cached = float(self.cached_gaussian)
        if algorithm != "MT19937":
            raise ValueError("only NumPy MT19937 RandomState is supported")
        if keys.shape != (624,) or keys.dtype != np.dtype(np.uint32):
            raise ValueError("RandomState keys must be 624 uint32 values")
        if position > 624:
            raise ValueError("RandomState position must be at most 624")
        if has_gauss not in (0, 1):
            raise ValueError("RandomState has_gauss must be zero or one")
        if not np.isfinite(cached):
            raise ValueError("RandomState cached Gaussian must be finite")
        copied = keys.copy()
        copied.setflags(write=False)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "keys", copied)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "has_gauss", has_gauss)
        object.__setattr__(self, "cached_gaussian", cached)

    @classmethod
    def capture(cls, random_state: np.random.RandomState) -> "NumpyRandomStateCheckpoint":
        if not isinstance(random_state, np.random.RandomState):
            raise TypeError("checkpointing requires numpy.random.RandomState")
        algorithm, keys, position, has_gauss, cached = random_state.get_state()
        return cls(algorithm, keys, position, has_gauss, cached)

    def restore(self, random_state: np.random.RandomState) -> None:
        if not isinstance(random_state, np.random.RandomState):
            raise TypeError("resuming requires numpy.random.RandomState")
        random_state.set_state(
            (
                self.algorithm,
                self.keys.copy(),
                self.position,
                self.has_gauss,
                self.cached_gaussian,
            )
        )


@dataclass(frozen=True)
class McmcChainCheckpoint:
    """Complete proposal-boundary state for one fixed MCMC chain."""

    chain_id: str
    mode_id: str
    warmup_steps: int
    retained_draws: int
    thinning: int
    kernel_names: Tuple[str, ...]
    completed_transition: int
    current: TargetEvaluation
    retained: Tuple[RetainedMcmcDraw, ...]
    kernel_counters: Mapping[str, KernelCounterCheckpoint]
    random_state: NumpyRandomStateCheckpoint

    def __post_init__(self) -> None:
        chain_id = _canonical_string(self.chain_id, "chain_id")
        mode_id = _canonical_string(self.mode_id, "mode_id")
        warmup = _nonnegative_integer(self.warmup_steps, "warmup_steps")
        retained_draws = _nonnegative_integer(
            self.retained_draws, "retained_draws"
        )
        thinning = _nonnegative_integer(self.thinning, "thinning")
        if retained_draws < 1 or thinning < 1:
            raise ValueError("retained_draws and thinning must be positive")
        kernel_names = tuple(
            _canonical_string(value, "kernel name") for value in self.kernel_names
        )
        if not kernel_names or len(set(kernel_names)) != len(kernel_names):
            raise ValueError("kernel names must be non-empty and unique")
        completed = _nonnegative_integer(
            self.completed_transition, "completed_transition"
        )
        total = warmup + retained_draws * thinning
        if completed > total:
            raise ValueError("completed transition exceeds chain length")
        if not isinstance(self.current, TargetEvaluation) or not self.current.successful:
            raise ValueError("current must be a successful exact target")
        retained = tuple(self.retained)
        if any(not isinstance(value, RetainedMcmcDraw) for value in retained):
            raise TypeError("retained trace has the wrong value type")
        expected_retained = max(0, completed - warmup) // thinning
        expected_draw_index = tuple(range(1, expected_retained + 1))
        if tuple(value.draw_index for value in retained) != expected_draw_index:
            raise ValueError("retained trace disagrees with completed transition")
        counters = dict(self.kernel_counters)
        if set(counters) != set(kernel_names) or any(
            not isinstance(value, KernelCounterCheckpoint)
            for value in counters.values()
        ):
            raise ValueError("kernel counters must match kernel names exactly")
        if sum(value.attempts for value in counters.values()) != completed:
            raise ValueError("kernel attempts must equal completed transitions")
        if any(value.attempted_kernel not in counters for value in retained):
            raise ValueError("retained trace names an unknown kernel")
        if not isinstance(self.random_state, NumpyRandomStateCheckpoint):
            raise TypeError("random_state has the wrong checkpoint type")
        component_flags = (self.current.has_density_components,) + tuple(
            value.evaluation.has_density_components for value in retained
        )
        if any(component_flags) and not all(component_flags):
            raise ValueError("checkpoint cannot mix decomposed and opaque targets")
        object.__setattr__(self, "chain_id", chain_id)
        object.__setattr__(self, "mode_id", mode_id)
        object.__setattr__(self, "warmup_steps", warmup)
        object.__setattr__(self, "retained_draws", retained_draws)
        object.__setattr__(self, "thinning", thinning)
        object.__setattr__(self, "kernel_names", kernel_names)
        object.__setattr__(self, "completed_transition", completed)
        object.__setattr__(self, "retained", retained)
        ordered_counters = {
            name: counters[name] for name in kernel_names
        }
        object.__setattr__(
            self, "kernel_counters", MappingProxyType(ordered_counters)
        )

    @property
    def total_transitions(self) -> int:
        return self.warmup_steps + self.retained_draws * self.thinning


_NPZ_FIELDS = frozenset(
    (
        "schema",
        "chain_id",
        "mode_id",
        "warmup_steps",
        "retained_draws",
        "thinning",
        "kernel_names",
        "kernel_counters",
        "completed_transition",
        "current_static_coordinate",
        "current_delay",
        "current_log_density",
        "current_inner_iterations",
        "current_has_components",
        "current_graph_objective",
        "current_local_log_determinant",
        "current_delay_log_prior",
        "retained_draw_index",
        "retained_static_coordinate",
        "retained_delay",
        "retained_log_density",
        "retained_target_inner_iterations",
        "retained_has_components",
        "retained_graph_objective",
        "retained_local_log_determinant",
        "retained_delay_log_prior",
        "retained_attempted_kernel",
        "retained_accepted",
        "retained_stage_one_accepted",
        "retained_stage_two_attempted",
        "retained_full_target_cache_hit",
        "retained_inner_solve_failed",
        "retained_inner_iterations",
        "random_algorithm",
        "random_keys",
        "random_position",
        "random_has_gauss",
        "random_cached_gaussian",
    )
)


def _target_components(evaluation: TargetEvaluation) -> Tuple[bool, float, float, float]:
    if evaluation.has_density_components:
        return (
            True,
            float(evaluation.graph_objective),
            float(evaluation.local_log_determinant),
            float(evaluation.delay_log_prior),
        )
    return False, 0.0, 0.0, 0.0


def _checkpoint_arrays(checkpoint: McmcChainCheckpoint) -> dict:
    if not isinstance(checkpoint, McmcChainCheckpoint):
        raise TypeError("checkpoint must be McmcChainCheckpoint")
    current_components = _target_components(checkpoint.current)
    retained_components = tuple(
        _target_components(value.evaluation) for value in checkpoint.retained
    )
    count = len(checkpoint.retained)
    string_size = max(
        (1,) + tuple(len(value.attempted_kernel) for value in checkpoint.retained)
    )
    return {
        "schema": np.asarray((MCMC_CHECKPOINT_SCHEMA,)),
        "chain_id": np.asarray((checkpoint.chain_id,)),
        "mode_id": np.asarray((checkpoint.mode_id,)),
        "warmup_steps": np.asarray((checkpoint.warmup_steps,), dtype=np.int64),
        "retained_draws": np.asarray((checkpoint.retained_draws,), dtype=np.int64),
        "thinning": np.asarray((checkpoint.thinning,), dtype=np.int64),
        "kernel_names": np.asarray(checkpoint.kernel_names),
        "kernel_counters": np.asarray(
            tuple(
                tuple(getattr(checkpoint.kernel_counters[name], field) for field in KERNEL_COUNTER_FIELDS)
                for name in checkpoint.kernel_names
            ),
            dtype=np.int64,
        ),
        "completed_transition": np.asarray((checkpoint.completed_transition,), dtype=np.int64),
        "current_static_coordinate": checkpoint.current.point.static_coordinate,
        "current_delay": np.asarray((checkpoint.current.point.delay,)),
        "current_log_density": np.asarray((checkpoint.current.log_density,)),
        "current_inner_iterations": np.asarray((checkpoint.current.inner_iterations,), dtype=np.int64),
        "current_has_components": np.asarray((current_components[0],), dtype=bool),
        "current_graph_objective": np.asarray((current_components[1],)),
        "current_local_log_determinant": np.asarray((current_components[2],)),
        "current_delay_log_prior": np.asarray((current_components[3],)),
        "retained_draw_index": np.asarray(tuple(value.draw_index for value in checkpoint.retained), dtype=np.int64),
        "retained_static_coordinate": (
            np.vstack(tuple(value.evaluation.point.static_coordinate for value in checkpoint.retained))
            if count
            else np.empty((0, STATIC_PARAMETER_DIMENSION), dtype=float)
        ),
        "retained_delay": np.asarray(tuple(value.evaluation.point.delay for value in checkpoint.retained), dtype=float),
        "retained_log_density": np.asarray(tuple(value.evaluation.log_density for value in checkpoint.retained), dtype=float),
        "retained_target_inner_iterations": np.asarray(tuple(value.evaluation.inner_iterations for value in checkpoint.retained), dtype=np.int64),
        "retained_has_components": np.asarray(tuple(value[0] for value in retained_components), dtype=bool),
        "retained_graph_objective": np.asarray(tuple(value[1] for value in retained_components), dtype=float),
        "retained_local_log_determinant": np.asarray(tuple(value[2] for value in retained_components), dtype=float),
        "retained_delay_log_prior": np.asarray(tuple(value[3] for value in retained_components), dtype=float),
        "retained_attempted_kernel": np.asarray(tuple(value.attempted_kernel for value in checkpoint.retained), dtype="<U{}".format(string_size)),
        "retained_accepted": np.asarray(tuple(value.accepted for value in checkpoint.retained), dtype=bool),
        "retained_stage_one_accepted": np.asarray(tuple(value.stage_one_accepted for value in checkpoint.retained), dtype=bool),
        "retained_stage_two_attempted": np.asarray(tuple(value.stage_two_attempted for value in checkpoint.retained), dtype=bool),
        "retained_full_target_cache_hit": np.asarray(tuple(value.full_target_cache_hit for value in checkpoint.retained), dtype=bool),
        "retained_inner_solve_failed": np.asarray(tuple(value.inner_solve_failed for value in checkpoint.retained), dtype=bool),
        "retained_inner_iterations": np.asarray(tuple(value.inner_iterations for value in checkpoint.retained), dtype=np.int64),
        "random_algorithm": np.asarray((checkpoint.random_state.algorithm,)),
        "random_keys": checkpoint.random_state.keys,
        "random_position": np.asarray((checkpoint.random_state.position,), dtype=np.int64),
        "random_has_gauss": np.asarray((checkpoint.random_state.has_gauss,), dtype=np.int64),
        "random_cached_gaussian": np.asarray((checkpoint.random_state.cached_gaussian,)),
    }


def save_mcmc_checkpoint(path: str, checkpoint: McmcChainCheckpoint) -> None:
    """Atomically replace one checkpoint with a compressed, pickle-free NPZ."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = _checkpoint_arrays(checkpoint)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=".{}.".format(destination.name),
            suffix=".tmp",
            dir=str(destination.parent),
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(str(temporary_path), str(destination))
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _array(archive, name: str) -> np.ndarray:
    try:
        value = np.asarray(archive[name])
    except (KeyError, ValueError) as error:
        raise McmcCheckpointError(
            "checkpoint field {} cannot be read safely".format(name)
        ) from error
    if value.dtype.hasobject:
        raise McmcCheckpointError(
            "checkpoint field {} has forbidden object dtype".format(name)
        )
    return value


def _string_scalar(archive, name: str) -> str:
    value = _array(archive, name)
    if value.shape != (1,) or value.dtype.kind != "U":
        raise McmcCheckpointError("{} must be one string".format(name))
    try:
        return _canonical_string(str(value[0]), name)
    except ValueError as error:
        raise McmcCheckpointError(str(error)) from error


def _integer_array(archive, name: str, shape: Tuple[int, ...]) -> np.ndarray:
    value = _array(archive, name)
    if value.shape != shape or value.dtype != np.dtype(np.int64):
        raise McmcCheckpointError(
            "{} must be an integer array with shape {}".format(name, shape)
        )
    converted = value.astype(np.int64, copy=True)
    if np.any(converted < 0):
        raise McmcCheckpointError("{} cannot contain negative values".format(name))
    return converted


def _float_array(archive, name: str, shape: Tuple[int, ...]) -> np.ndarray:
    value = _array(archive, name)
    if value.shape != shape or value.dtype != np.dtype(np.float64):
        raise McmcCheckpointError(
            "{} must be a floating array with shape {}".format(name, shape)
        )
    converted = value.astype(float, copy=True)
    if np.any(~np.isfinite(converted)):
        raise McmcCheckpointError("{} must be finite".format(name))
    return converted


def _boolean_array(archive, name: str, shape: Tuple[int, ...]) -> np.ndarray:
    value = _array(archive, name)
    if value.shape != shape or value.dtype.kind != "b":
        raise McmcCheckpointError(
            "{} must be a boolean array with shape {}".format(name, shape)
        )
    return value.astype(bool, copy=True)


def _evaluation_from_arrays(
    static_coordinate: np.ndarray,
    delay: float,
    log_density: float,
    inner_iterations: int,
    has_components: bool,
    graph_objective: float,
    local_log_determinant: float,
    delay_log_prior: float,
) -> TargetEvaluation:
    keyword_arguments = {}
    if has_components:
        keyword_arguments = {
            "graph_objective": graph_objective,
            "local_log_determinant": local_log_determinant,
            "delay_log_prior": delay_log_prior,
        }
    elif any(
        value != 0.0
        for value in (graph_objective, local_log_determinant, delay_log_prior)
    ):
        raise McmcCheckpointError(
            "inactive target decomposition fields must be zero"
        )
    try:
        return TargetEvaluation(
            point=PosteriorPoint(static_coordinate, delay),
            log_density=log_density,
            successful=True,
            failure_reason="",
            inner_iterations=inner_iterations,
            warm_start=None,
            **keyword_arguments
        )
    except (TypeError, ValueError) as error:
        raise McmcCheckpointError("target evaluation is invalid") from error


def load_mcmc_checkpoint(path: str) -> McmcChainCheckpoint:
    """Strictly load one v1 checkpoint without enabling NumPy pickle."""

    try:
        with np.load(str(Path(path)), allow_pickle=False) as archive:
            fields = set(archive.files)
            if len(archive.files) != len(fields) or fields != _NPZ_FIELDS:
                missing = sorted(_NPZ_FIELDS - fields)
                unknown = sorted(fields - _NPZ_FIELDS)
                raise McmcCheckpointError(
                    "checkpoint fields differ; missing={}, unknown={}".format(
                        missing, unknown
                    )
                )
            schema = _string_scalar(archive, "schema")
            if schema != MCMC_CHECKPOINT_SCHEMA:
                raise McmcCheckpointError("unsupported MCMC checkpoint schema")
            chain_id = _string_scalar(archive, "chain_id")
            mode_id = _string_scalar(archive, "mode_id")
            warmup = int(_integer_array(archive, "warmup_steps", (1,))[0])
            retained_draws = int(
                _integer_array(archive, "retained_draws", (1,))[0]
            )
            thinning = int(_integer_array(archive, "thinning", (1,))[0])
            completed = int(
                _integer_array(archive, "completed_transition", (1,))[0]
            )
            kernel_names_array = _array(archive, "kernel_names")
            if (
                kernel_names_array.ndim != 1
                or kernel_names_array.size < 1
                or kernel_names_array.dtype.kind != "U"
            ):
                raise McmcCheckpointError("kernel_names must be a string vector")
            kernel_names = tuple(str(value) for value in kernel_names_array)
            counter_matrix = _integer_array(
                archive,
                "kernel_counters",
                (len(kernel_names), len(KERNEL_COUNTER_FIELDS)),
            )
            counters = {
                name: KernelCounterCheckpoint(
                    **{
                        field: int(counter_matrix[index, field_index])
                        for field_index, field in enumerate(KERNEL_COUNTER_FIELDS)
                    }
                )
                for index, name in enumerate(kernel_names)
            }

            current_static = _float_array(
                archive,
                "current_static_coordinate",
                (STATIC_PARAMETER_DIMENSION,),
            )
            current_delay = float(
                _float_array(archive, "current_delay", (1,))[0]
            )
            current_log_density = float(
                _float_array(archive, "current_log_density", (1,))[0]
            )
            current_inner = int(
                _integer_array(archive, "current_inner_iterations", (1,))[0]
            )
            current_has_components = bool(
                _boolean_array(archive, "current_has_components", (1,))[0]
            )
            current_graph = float(
                _float_array(archive, "current_graph_objective", (1,))[0]
            )
            current_logdet = float(
                _float_array(archive, "current_local_log_determinant", (1,))[0]
            )
            current_delay_prior = float(
                _float_array(archive, "current_delay_log_prior", (1,))[0]
            )
            current = _evaluation_from_arrays(
                current_static,
                current_delay,
                current_log_density,
                current_inner,
                current_has_components,
                current_graph,
                current_logdet,
                current_delay_prior,
            )

            retained_draw_index_array = _array(archive, "retained_draw_index")
            if retained_draw_index_array.ndim != 1:
                raise McmcCheckpointError("retained_draw_index must be a vector")
            count = retained_draw_index_array.size
            retained_draw_index = _integer_array(
                archive, "retained_draw_index", (count,)
            )
            retained_static = _float_array(
                archive,
                "retained_static_coordinate",
                (count, STATIC_PARAMETER_DIMENSION),
            )
            retained_delay = _float_array(
                archive, "retained_delay", (count,)
            )
            retained_log_density = _float_array(
                archive, "retained_log_density", (count,)
            )
            retained_target_inner = _integer_array(
                archive, "retained_target_inner_iterations", (count,)
            )
            retained_has_components = _boolean_array(
                archive, "retained_has_components", (count,)
            )
            retained_graph = _float_array(
                archive, "retained_graph_objective", (count,)
            )
            retained_logdet = _float_array(
                archive, "retained_local_log_determinant", (count,)
            )
            retained_delay_prior = _float_array(
                archive, "retained_delay_log_prior", (count,)
            )
            attempted_kernel = _array(archive, "retained_attempted_kernel")
            if attempted_kernel.shape != (count,) or attempted_kernel.dtype.kind != "U":
                raise McmcCheckpointError(
                    "retained_attempted_kernel must be a string vector"
                )
            retained_accepted = _boolean_array(
                archive, "retained_accepted", (count,)
            )
            retained_stage_one = _boolean_array(
                archive, "retained_stage_one_accepted", (count,)
            )
            retained_stage_two = _boolean_array(
                archive, "retained_stage_two_attempted", (count,)
            )
            retained_cache_hit = _boolean_array(
                archive, "retained_full_target_cache_hit", (count,)
            )
            retained_inner_failed = _boolean_array(
                archive, "retained_inner_solve_failed", (count,)
            )
            retained_inner_iterations = _integer_array(
                archive, "retained_inner_iterations", (count,)
            )
            retained = []
            for index in range(count):
                evaluation = _evaluation_from_arrays(
                    retained_static[index],
                    retained_delay[index],
                    retained_log_density[index],
                    int(retained_target_inner[index]),
                    bool(retained_has_components[index]),
                    retained_graph[index],
                    retained_logdet[index],
                    retained_delay_prior[index],
                )
                retained.append(
                    RetainedMcmcDraw(
                        draw_index=int(retained_draw_index[index]),
                        evaluation=evaluation,
                        attempted_kernel=str(attempted_kernel[index]),
                        accepted=bool(retained_accepted[index]),
                        stage_one_accepted=bool(retained_stage_one[index]),
                        stage_two_attempted=bool(retained_stage_two[index]),
                        full_target_cache_hit=bool(retained_cache_hit[index]),
                        inner_solve_failed=bool(retained_inner_failed[index]),
                        inner_iterations=int(retained_inner_iterations[index]),
                    )
                )

            random_algorithm = _string_scalar(archive, "random_algorithm")
            random_keys = _array(archive, "random_keys")
            if random_keys.shape != (624,) or random_keys.dtype != np.dtype(np.uint32):
                raise McmcCheckpointError(
                    "random_keys must contain exactly 624 uint32 values"
                )
            random_state = NumpyRandomStateCheckpoint(
                algorithm=random_algorithm,
                keys=random_keys,
                position=int(
                    _integer_array(archive, "random_position", (1,))[0]
                ),
                has_gauss=int(
                    _integer_array(archive, "random_has_gauss", (1,))[0]
                ),
                cached_gaussian=float(
                    _float_array(archive, "random_cached_gaussian", (1,))[0]
                ),
            )
    except McmcCheckpointError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        EOFError,
        zipfile.BadZipFile,
    ) as error:
        raise McmcCheckpointError("cannot load MCMC checkpoint") from error
    try:
        return McmcChainCheckpoint(
            chain_id=chain_id,
            mode_id=mode_id,
            warmup_steps=warmup,
            retained_draws=retained_draws,
            thinning=thinning,
            kernel_names=kernel_names,
            completed_transition=completed,
            current=current,
            retained=tuple(retained),
            kernel_counters=counters,
            random_state=random_state,
        )
    except (TypeError, ValueError) as error:
        raise McmcCheckpointError("MCMC checkpoint contract is invalid") from error


__all__ = [
    "KERNEL_COUNTER_FIELDS",
    "MCMC_CHECKPOINT_SCHEMA",
    "KernelCounterCheckpoint",
    "McmcChainCheckpoint",
    "McmcCheckpointError",
    "NumpyRandomStateCheckpoint",
    "RetainedMcmcDraw",
    "load_mcmc_checkpoint",
    "save_mcmc_checkpoint",
]
