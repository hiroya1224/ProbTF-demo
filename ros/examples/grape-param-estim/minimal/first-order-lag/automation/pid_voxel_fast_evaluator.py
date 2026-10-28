"""Gain-aware hover seeds; exact workers and batched plant scheduling are unchanged."""
from __future__ import annotations
import time
from typing import Any, Optional, Sequence
import numpy as np
from tqdm import tqdm
from pid_safe_margin_slices import GAIN_CAPS, MarginEvaluation, _controller_configuration
from pid_spectral_gradient_path import BatchedForwardMarginEvaluator, _keyed_safe_chunk_task

TRIM_INTEGRAL_COLUMNS = ((0, 1), (2,), (3, 4), (5,))


def scale_trim_seed(trims, source_gains, target_gains):
    """Preserve K_I * integral_error while changing only the solver's initial guess.

    Zero target I retains the conservative original seed and solver fallbacks.
    The production search boxes have strictly positive I on every axis.
    """
    if trims is None:
        return None
    warm = np.asarray(trims, dtype=float)
    source = np.asarray(source_gains, dtype=float)
    target = np.asarray(target_gains, dtype=float)
    if warm.ndim != 2 or warm.shape[1] != 10:
        raise ValueError('trim seeds must have shape (sample_count, 10)')
    for gains in (source, target):
        if gains.shape != (4, 3) or np.any(~np.isfinite(gains)) or np.any(gains < 0):
            raise ValueError('source and target gains must be finite nonnegative 4-by-3 arrays')
    scaled = warm.copy()
    for group, columns in enumerate(TRIM_INTEGRAL_COLUMNS):
        if target[group, 1] > 0:
            ratio = source[group, 1] / target[group, 1]
            if ratio != 1.0:
                scaled[:, columns] *= ratio
    return scaled


class FastGainEvaluator(BatchedForwardMarginEvaluator):
    """Use the supplied recorded failure trim seeds for each gain wave.

    evaluate_many follows the frozen batched scheduler, with gain-aware copies
    of the explicit and nearest seeds. No pole/model approximation is used.
    """
    def __init__(self, *, reference_gains, **kwargs: Any):
        reference = np.asarray(reference_gains, dtype=float)
        if reference.shape != (4, 3) or np.any(~np.isfinite(reference)) or np.any(reference < 0):
            raise ValueError('reference_gains must be a finite nonnegative 4-by-3 array')
        self.reference_gains = reference.copy()
        super().__init__(**kwargs)

    @staticmethod
    def _key(q: Sequence[float]) -> tuple[float, ...]:
        selected = np.asarray(q, dtype=float)
        if selected.shape != (12,) or np.any(~np.isfinite(selected)) or np.any(selected < 0):
            raise ValueError('normalized PID coordinates must be finite nonnegative length 12')
        return tuple(float(value) for value in selected)

    def evaluate_many(
        self,
        q_values: Sequence[Sequence[float]],
        *,
        initial_trims: Optional[np.ndarray] = None,
        progress: bool = True,
        description: str = "Batched gain wave",
    ) -> list[MarginEvaluation]:
        q_matrix = np.asarray(q_values, dtype=float)
        if q_matrix.ndim != 2 or q_matrix.shape[1] != 12:
            raise ValueError("batched gain coordinates must be M-by-12")

        results: list[Optional[MarginEvaluation]] = [None] * q_matrix.shape[0]
        uncached: list[
            tuple[
                int,
                tuple[float, ...],
                np.ndarray,
                Optional[MarginEvaluation],
                np.ndarray,
            ]
        ] = []
        sample_count = self.active_sample_count
        for row_index, q in enumerate(q_matrix):
            key = self._key(q)
            cached = self._cache.get(key)
            cached_count = (
                0
                if cached is None
                else int(np.asarray(cached.spectral_radius).size)
            )
            if cached_count == sample_count:
                results[row_index] = cached
                self._cache_hit_count += sample_count
            elif cached_count < sample_count:
                self._cache_hit_count += cached_count
                missing = np.arange(
                    cached_count,
                    sample_count,
                    dtype=int,
                )
                uncached.append((
                    row_index,
                    key,
                    np.asarray(key, dtype=float),
                    cached,
                    missing,
                ))
            else:
                raise RuntimeError("cached plant prefix exceeds active prefix")

        if uncached:
            warm = None
            if initial_trims is not None:
                warm = np.asarray(initial_trims, dtype=float)
                if warm.shape != (sample_count, 10):
                    raise ValueError("initial_trims must have shape (sample_count, 10)")

            tasks: list[tuple[int, tuple[Any, ...]]] = []
            for row_index, _key, q, _cached, missing in uncached:
                configuration = _controller_configuration(q.reshape(4, 3) * GAIN_CAPS)
                row_warm = scale_trim_seed(warm, self.reference_gains, q.reshape(4, 3) * GAIN_CAPS)
                chunk_count = min(
                    missing.size,
                    max(1, 2 * self.workers),
                )
                chunks = [
                    chunk
                    for chunk in np.array_split(missing, chunk_count)
                    if chunk.size
                ]

                eligible = [
                    row
                    for row in self._cache.values()
                    if np.asarray(row.trim_vectors).shape[0] >= sample_count
                ]
                nearest = (
                    None
                    if not eligible
                    else min(
                        eligible,
                        key=lambda row: float(np.linalg.norm(row.q - q)),
                    )
                )
                nearest = (None if nearest is None else
                           scale_trim_seed(nearest.trim_vectors, nearest.gains, q.reshape(4, 3) * GAIN_CAPS))
                for chunk in chunks:
                    tasks.append(
                        (
                            row_index,
                            (
                                tuple(int(i) for i in chunk),
                                tuple(self.plants[int(i)] for i in chunk),
                                self.vehicle_model,
                                self.actuator_parameters,
                                self.controller_dt,
                                configuration,
                                None if row_warm is None else row_warm[chunk],
                                None if nearest is None else nearest[chunk],
                            ),
                        )
                    )

            started = time.perf_counter()
            if self._executor is None:
                iterator = (_keyed_safe_chunk_task(task) for task in tasks)
            else:
                iterator = self._executor.map(_keyed_safe_chunk_task, tasks)

            grouped: dict[int, list[tuple[Any, ...]]] = {
                row_index: []
                for row_index, _key, _q, _cached, _missing in uncached
            }
            bar = tqdm(
                iterator,
                total=len(tasks),
                desc=description,
                unit="chunk",
                dynamic_ncols=True,
                leave=False,
                disable=not progress,
            )
            for row_index, row in bar:
                grouped[int(row_index)].append(row)
            total_wall = time.perf_counter() - started

            per_gain_wall = total_wall / max(1, len(uncached))
            for row_index, key, q, cached, missing in uncached:
                result = self._assemble_rows(
                    q,
                    grouped[row_index],
                    per_gain_wall,
                    prior=cached,
                )
                self._cache[key] = result
                results[row_index] = result
                self._new_plant_evaluation_count += int(missing.size)

        final = [row for row in results if row is not None]
        if len(final) != q_matrix.shape[0]:
            raise RuntimeError("batched gain evaluation lost a result")
        return final
