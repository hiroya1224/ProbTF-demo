#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
FIRST_ORDER = HERE.parent
if str(FIRST_ORDER) not in sys.path:
    sys.path.insert(0, str(FIRST_ORDER))

from pid_safe_margin_slices import ForwardMarginEvaluator  # noqa: E402
import pid_spectral_gradient_path as spectral_path  # noqa: E402
from pid_spectral_gradient_path import (  # noqa: E402
    BatchedForwardMarginEvaluator,
    _finite_difference_jacobian,
)
from pid_spectral_gradient_sharpen import (  # noqa: E402
    _build_gain_update_table,
    _nested_prefix_counts,
    _normal_newton_step,
    _softmax_summary,
    _soft_target_margin,
    _tangent_direction,
    _target_residual,
)


class _LinearRhoEvaluator:
    def __init__(
        self,
        center: np.ndarray,
        slopes: np.ndarray,
        unresolved: object,
    ) -> None:
        self.center = np.asarray(center, dtype=float)
        self.slopes = np.asarray(slopes, dtype=float)
        self.unresolved = unresolved
        self.calls: list[np.ndarray] = []

    def row(self, q: np.ndarray) -> SimpleNamespace:
        selected = np.asarray(q, dtype=float)
        valid = np.ones(self.slopes.shape[0], dtype=bool)
        invalid = np.asarray(self.unresolved(selected), dtype=bool)
        valid[invalid] = False
        rho = 1.0 + self.slopes @ (selected - self.center)
        rho[~valid] = np.nan
        return SimpleNamespace(
            spectral_radius=rho,
            pole_valid_mask=valid,
            trim_vectors=np.zeros((self.slopes.shape[0], 10), dtype=float),
        )

    def evaluate_many(
        self,
        q_values: list[np.ndarray],
        **_kwargs: object,
    ) -> list[SimpleNamespace]:
        self.calls.extend(np.asarray(q, dtype=float).copy() for q in q_values)
        return [self.row(np.asarray(q, dtype=float)) for q in q_values]


def test_finite_difference_uses_resolved_side_per_plant() -> None:
    center = np.full(12, 0.5, dtype=float)
    center[1] = 1.0 / 120.0
    center[6] = 1.0
    slopes = np.vstack((np.arange(1.0, 13.0), -np.arange(1.0, 13.0)))

    def unresolved(q: np.ndarray) -> np.ndarray:
        # The lower xy:I perturbation lands at the controller singularity,
        # while the baseline and upper perturbation remain resolved.
        return np.full(2, q[1] == 0.0, dtype=bool)

    evaluator = _LinearRhoEvaluator(center, slopes, unresolved)
    baseline = evaluator.row(center)
    result = _finite_difference_jacobian(
        evaluator,
        center,
        baseline,
        0.01,
        progress=False,
    )

    assert np.allclose(result["jacobian"], slopes)
    assert np.array_equal(result["coverage"], np.ones(12))
    assert result["refinement_count"][1] == 0


def test_finite_difference_refines_until_perturbation_resolves() -> None:
    center = np.full(12, 0.5, dtype=float)
    slopes = np.vstack((np.arange(1.0, 13.0), -np.arange(1.0, 13.0)))

    def unresolved(q: np.ndarray) -> np.ndarray:
        # Both initial +/- probes on axis 2 fail.  Their first geometric
        # refinement is close enough to the resolved baseline.
        failed = (
            np.count_nonzero(q != center) == 1
            and q[2] != center[2]
            and abs(q[2] - center[2]) > 0.0051
        )
        return np.full(2, failed, dtype=bool)

    evaluator = _LinearRhoEvaluator(center, slopes, unresolved)
    result = _finite_difference_jacobian(
        evaluator,
        center,
        evaluator.row(center),
        0.01,
        progress=False,
    )

    assert np.allclose(result["jacobian"], slopes)
    assert result["refinement_count"][2] == 1


def test_exact_cache_key_does_not_merge_neighboring_floats() -> None:
    q = np.full(12, 0.5, dtype=float)
    neighbor = q.copy()
    neighbor[0] = np.nextafter(neighbor[0], 1.0)

    assert ForwardMarginEvaluator._key(q) != ForwardMarginEvaluator._key(neighbor)


def test_normal_predictor_cancels_linearized_residual() -> None:
    gradient = np.linspace(-0.6, 0.5, 12)
    residual = -4.0e-5
    step = _normal_newton_step(residual, gradient)

    assert abs(residual + float(np.dot(gradient, step))) < 1.0e-18


def test_failure_direction_is_projected_into_boundary_tangent() -> None:
    q = np.linspace(0.1, 0.8, 12)
    failure_q = np.linspace(0.2, 0.7, 12)
    gradient = np.linspace(-0.4, 0.7, 12)
    tangent = _tangent_direction(q, failure_q, gradient)

    assert abs(float(np.dot(gradient, tangent))) < 1.0e-15


def test_tau_sharpening_reuses_the_same_plant_radii() -> None:
    rho = np.asarray((0.9998, 1.0000, 1.0002))
    broad = _softmax_summary(rho, 1.0e-4)
    sharp = _softmax_summary(rho, 5.0e-5)

    assert sharp["smooth_max"] >= broad["smooth_max"]
    assert np.isclose(np.sum(broad["weights"]), 1.0)
    assert np.isclose(np.sum(sharp["weights"]), 1.0)


def test_nested_prefix_schedule_reaches_exact_requested_maximum() -> None:
    assert _nested_prefix_counts(128) == (16, 32, 64, 128)
    assert _nested_prefix_counts(20) == (16, 20)
    assert _nested_prefix_counts(8) == (8,)


def test_batched_evaluator_extends_cached_gain_with_suffix_only() -> None:
    original_task = spectral_path._keyed_safe_chunk_task

    def fake_task(task: tuple[int, tuple[object, ...]]) -> tuple[int, tuple[object, ...]]:
        row_index, payload = task
        indices = np.asarray(payload[0], dtype=int)
        plants = payload[1]
        matrices = np.zeros((indices.size, 26, 26), dtype=float)
        for local_index, plant in enumerate(plants):
            matrices[local_index] = np.eye(26) * (0.5 + 0.01 * int(plant))
        return row_index, (
            tuple(int(index) for index in indices),
            matrices,
            np.ones(indices.size, dtype=bool),
            np.zeros(indices.size, dtype=bool),
            np.zeros((indices.size, 10), dtype=float),
        )

    spectral_path._keyed_safe_chunk_task = fake_task
    evaluator = BatchedForwardMarginEvaluator(
        plants=tuple(range(4)),
        vehicle_model=None,
        actuator_parameters=None,
        controller_dt=0.01,
        workers=1,
    )
    try:
        q = np.full(12, 0.5, dtype=float)
        evaluator.set_active_sample_count(2)
        first = evaluator.evaluate(q)
        evaluator.set_active_sample_count(4)
        extended = evaluator.evaluate(q)

        assert first.spectral_radius.shape == (2,)
        assert extended.spectral_radius.shape == (4,)
        assert np.array_equal(
            extended.spectral_radius[:2],
            first.spectral_radius,
        )
        assert evaluator.cache_diagnostics()[
            "new_plant_evaluation_count"
        ] == 4
        assert evaluator.cache_diagnostics()["cache_hit_count"] == 2
    finally:
        evaluator.close()
        spectral_path._keyed_safe_chunk_task = original_task


def test_soft_target_guarantees_requested_hard_margin_bound() -> None:
    tau = 1.0e-4
    sample_count = 128
    hard_margin_target = 5.0e-5
    soft_target = _soft_target_margin(
        tau,
        sample_count,
        hard_margin_target,
    )
    summary = {"delta_soft": soft_target}

    assert _target_residual(summary, soft_target) == 0.0
    assert np.isclose(
        soft_target - tau * np.log(sample_count),
        hard_margin_target,
    )


def test_gain_update_table_reports_physical_and_normalized_changes() -> None:
    failure = np.zeros(12, dtype=float)
    proposal = np.full(12, 0.1, dtype=float)
    rows = _build_gain_update_table(failure, proposal)

    assert len(rows) == 12
    assert rows[0][:2] == ["xy", "P"]
    assert rows[-1][:2] == ["yaw", "D"]
    assert rows[0][-1] == "+0.1"
