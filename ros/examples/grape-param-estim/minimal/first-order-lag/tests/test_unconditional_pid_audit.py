"""Regressions for the 2026-10-27 success-conditioning audit.

The population is the complete, fixed collection drawn from a failure fit.
A known successful controller is a reference, never a population filter or a
choice of the other nine gains in a prediction slice.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

FIRST_ORDER = Path(__file__).resolve().parents[1]
if str(FIRST_ORDER) not in sys.path:
    sys.path.insert(0, str(FIRST_ORDER))

import pid_gain_contour as contour  # noqa: E402


def _fixed_population_evaluator(stable, valid=None, log_radius=None):
    """Supply fixed pole outcomes without doing expensive plant identification."""
    stable = np.asarray(stable, dtype=bool)
    count = len(stable)
    evaluator = object.__new__(contour.ExactGroupGainEvaluator)
    evaluator.plants = tuple(object() for _ in range(count))
    evaluator._break_cache = {}
    # An old saved success-reference mask must have no authority to change N.
    evaluator.success_stable_mask = np.arange(count) != count - 1
    evaluator.success_stable_count = count - 1
    evaluator.evaluate = lambda _coordinate: SimpleNamespace(
        stable_mask=stable.copy(),
        pole_valid_mask=(np.ones(count, dtype=bool) if valid is None else valid),
        log_spectral_radius=(
            np.where(stable, np.log(0.9), np.log(1.1))
            if log_radius is None else log_radius
        ),
    )
    return evaluator


def test_known_success_is_not_made_one_hundred_percent_by_dropping_a_plant():
    evaluator = _fixed_population_evaluator([True, True, False])
    assert evaluator.survival_fraction(np.zeros(3)) == pytest.approx(2.0 / 3.0)


def test_plant_unstable_at_the_reference_still_counts_when_candidate_stabilizes_it():
    evaluator = _fixed_population_evaluator([False, True, True])
    # The last sample was not stable at the historical successful reference.
    # It nevertheless belongs to the population, and is stable at this point.
    assert evaluator.survival_fraction(np.zeros(3)) == pytest.approx(2.0 / 3.0)


def test_unresolved_and_nonfinite_samples_do_not_shrink_the_denominator():
    evaluator = _fixed_population_evaluator(
        [True, False, False, False],
        valid=np.array([True, True, False, False]),
        log_radius=np.array([np.log(0.9), np.log(1.1), np.inf, np.nan]),
    )
    assert evaluator.survival_fraction(np.zeros(3)) == pytest.approx(1.0 / 4.0)


def test_nonfinite_matrix_and_unresolved_trim_remain_in_the_population():
    evaluator = _fixed_population_evaluator([True, False, False, False, True])
    matrices = np.stack([value * np.eye(26)
                         for value in (0.9, 1.1, np.nan, 0.8, 0.0)])
    flags = np.zeros(5, dtype=bool)
    stack = (
        matrices, np.array([True, True, True, False, True]), flags,
        np.zeros((5, 10)), np.zeros(5, dtype=int), np.zeros((5, 3), dtype=int),
        flags, flags, flags, flags, np.zeros(5), np.zeros(5), np.zeros(5), flags,
    )
    point = evaluator._gain_point_from_matrix_stack((0.0, 0.0, 0.0), stack, "audit")
    assert np.array_equal(point.pole_valid_mask, [True, True, False, False, True])
    assert np.array_equal(point.stable_mask, [True, False, False, False, True])
    # A valid zero spectral radius is stable; its log is -inf, not a failed pole.
    assert point.log_spectral_radius[-1] == -np.inf
    evaluator.evaluate = lambda _coordinate: point
    assert evaluator.survival_fraction(np.zeros(3)) == pytest.approx(2.0 / 5.0)


def test_success_reference_does_not_seed_the_constructor_trim_cache():
    failure = {group: contour.GainTriple(2.0, 1.0, 3.0)
               for group in contour.PID_GROUPS}
    success = {group: contour.GainTriple(20.0, 5.0, 30.0)
               for group in contour.PID_GROUPS}
    requested = []

    def evaluate(_self, coordinate, **_kwargs):
        requested.append(np.asarray(coordinate).copy())
        return SimpleNamespace(
            log_spectral_radius=np.log([0.9, 0.9, 1.1]),
            pole_valid_mask=np.ones(3, dtype=bool),
            stable_mask=np.array([True, True, False]),
        )

    with patch.object(contour.ExactGroupGainEvaluator, "evaluate", evaluate):
        evaluator = contour.ExactGroupGainEvaluator(
            plants=(object(),) * 3, hybrid_failure_gains=failure,
            success_gains=success, group="roll_pitch", vehicle_model=object(),
            actuator_parameters=object(), controller_dt=0.01, workers=1,
        )
        try:
            assert requested == []
            assert evaluator.evaluated_gain_point_count == 0
        finally:
            evaluator.close()


def test_worker_receives_one_complete_controller_with_failure_fixed_other_groups():
    failure = {
        group: contour.GainTriple(2.0 + index, 1.0 + index, 3.0 + index)
        for index, group in enumerate(contour.PID_GROUPS)
    }
    success = {group: contour.GainTriple(100.0, 50.0, 200.0)
               for group in contour.PID_GROUPS}
    evaluator = contour.ExactGroupGainEvaluator(
        plants=(object(),) * 3, hybrid_failure_gains=failure,
        success_gains=success, group="roll_pitch", vehicle_model=object(),
        actuator_parameters=object(), controller_dt=0.01, workers=1,
    )
    try:
        with patch.object(contour, "_configuration", side_effect=lambda gains:
                          np.stack([gains[group].array() for group in contour.PID_GROUPS])):
            tasks = evaluator._matrix_tasks(np.log([2.0, 3.0, 0.5]), None, None)
        expected = np.stack([failure[group].array() for group in contour.PID_GROUPS])
        expected[contour.PID_GROUPS.index("roll_pitch")] *= [2.0, 3.0, 0.5]
        assert sum(len(task[0]) for task in tasks) == 3
        for task in tasks:
            assert np.allclose(task[5], expected, rtol=0.0, atol=1e-14)
    finally:
        evaluator.close()


class _ReachedPredictionGrid(Exception):
    pass


def _estimate(gain_rows):
    return {
        "input": {"vehicle_model": "/tmp/audit-model-not-loaded.urdf"},
        "controller_timing": {"median_seconds": 0.01},
        "controller": {
            "gains": {
                group: dict(zip(("p_gain", "i_gain", "d_gain"), row))
                for group, row in zip(contour.PID_GROUPS, gain_rows)
            }
        },
    }


def _capture_prediction_setup(success_rows):
    failure_rows = np.array([
        [3.0, 0.1, 1.0], [5.0, 1.0, 2.5],
        [20.0, 1.0, 8.0], [4.0, 1.0, 2.0],
    ])
    failure = _estimate(failure_rows)
    success = _estimate(success_rows)
    captured = {"evaluation_coordinates": [], "loaded_paths": []}

    def load_estimate(path):
        captured["loaded_paths"].append(str(path))
        return success if "success" in str(path) else failure

    class _PredictorSpy:
        def __init__(self, **kwargs):
            captured["constructor"] = kwargs
            base = kwargs.get("failure_gains", kwargs.get("hybrid_failure_gains"))
            self.base_gain = base[kwargs["group"]].array()
            reference = kwargs.get("success_gains") or base
            self.success_log_ratio = np.log(reference[kwargs["group"]].array() / self.base_gain)

        def break_evaluation(self, coordinate):
            captured["evaluation_coordinates"].append(np.asarray(coordinate).copy())
            return SimpleNamespace()

        def evaluate(self, coordinate, **_kwargs):
            return self.break_evaluation(coordinate)

        def close(self):
            pass

    def record_grid(_slice, **kwargs):
        captured["grid"] = kwargs
        raise _ReachedPredictionGrid

    args = argparse.Namespace(
        estimate_json="/tmp/failure-audit.json",
        success_json="/tmp/success-audit.json",
        group="roll_pitch", scale_min=0.5, scale_max=2.0,
        covariance="likelihood", samples=3, seed=0, workers=1,
        alpha=0.95, max_grid_size=3, final_boundary_grid_size=3,
    )
    patches = [
        patch.object(contour, "load_estimate_json", side_effect=load_estimate),
        patch.object(contour, "load_vehicle_model", return_value=SimpleNamespace(parameters=object())),
        patch.object(contour, "actuator_parameters_from_estimate", return_value=object()),
        patch.object(contour, "draw_quotient_coordinates", return_value=np.zeros((3, 13))),
        patch.object(contour, "quotient_to_scale_free_plants", return_value=(object(),) * 3),
        patch.object(contour, "ExactGroupGainEvaluator", _PredictorSpy),
        patch.object(contour, "_adaptive_projection_grid", side_effect=record_grid),
    ]
    with ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        with pytest.raises(_ReachedPredictionGrid):
            contour.analyze(args)
    return failure_rows, captured


def test_prediction_slice_keeps_other_nine_gains_at_recorded_failure_values():
    rows, result = _capture_prediction_setup(np.full((4, 3), 100.0))
    kwargs = result["constructor"]
    base = kwargs.get("failure_gains", kwargs.get("hybrid_failure_gains"))
    for group, expected in zip(contour.PID_GROUPS, rows):
        assert np.array_equal(base[group].array(), expected)


def test_changing_success_reference_does_not_change_prediction_bounds_or_preseed_cache():
    _, first = _capture_prediction_setup(np.full((4, 3), 100.0))
    _, second = _capture_prediction_setup(np.full((4, 3), 0.001))
    for result in (first, second):
        assert result["grid"]["lower"] == pytest.approx(np.log(0.5))
        assert result["grid"]["upper"] == pytest.approx(np.log(2.0))
        # A success-seeded trim cache could change numerical branch selection.
        assert all(np.array_equal(value, np.zeros(3))
                   for value in result["evaluation_coordinates"])
        assert result["loaded_paths"] == ["/tmp/failure-audit.json"]
