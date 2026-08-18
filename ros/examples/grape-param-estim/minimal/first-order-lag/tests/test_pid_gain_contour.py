#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
FIRST_ORDER = HERE.parent
if str(FIRST_ORDER) not in sys.path:
    sys.path.insert(0, str(FIRST_ORDER))

from pid_gain_contour import (  # noqa: E402
    SliceGridEvaluator,
    _adaptive_projection_grid,
    _boundary_present,
    _global_candidate_cells,
)


class _SyntheticBreakEvaluation:
    def __init__(self, fraction: float) -> None:
        self.caused_break_fraction = float(fraction)


class _SyntheticGroupEvaluator:
    def __init__(self) -> None:
        self.base_gain = np.ones(3)
        self.calls = 0

    def break_evaluation(self, coordinate):
        self.calls += 1
        selected = np.asarray(coordinate, dtype=float)
        # Circular break boundary in the two visible coordinates.
        fraction = float(np.clip(0.05 + 0.2 * (selected[0] ** 2 + selected[1] ** 2 - 0.5), 0.0, 1.0))
        return _SyntheticBreakEvaluation(fraction)


def test_nested_grid_reuses_all_previous_points() -> None:
    group = _SyntheticGroupEvaluator()
    evaluator = SliceGridEvaluator(
        group,
        first_axis=0,
        second_axis=1,
        hidden_axis=2,
    )
    evaluator.regular_grid(-1.0, 1.0, 3)
    assert group.calls == 9
    evaluator.regular_grid(-1.0, 1.0, 5)
    assert group.calls == 25
    evaluator.regular_grid(-1.0, 1.0, 9)
    assert group.calls == 81


def test_adaptive_grid_refines_only_boundary_cells_after_global_17() -> None:
    group = _SyntheticGroupEvaluator()
    evaluator = SliceGridEvaluator(
        group,
        first_axis=0,
        second_axis=1,
        hidden_axis=2,
    )
    result = _adaptive_projection_grid(
        evaluator,
        name="pi",
        first_axis=0,
        second_axis=1,
        hidden_axis=2,
        lower=-1.0,
        upper=1.0,
        threshold=0.05,
        maximum_grid_size=17,
        local_refinement_levels=3,
    )
    assert result.axis_log_ratio.size == 17
    assert [level.size for level in result.levels] == [3, 5, 9, 17]
    assert [level.new_point_count for level in result.levels] == [9, 16, 56, 208]
    assert len(result.local_refinement_levels) == 3
    assert result.boundary_segments_log_ratio.shape[0] > 0
    assert result.effective_local_equivalent_grid_size == 129
    assert result.stop_reason == "boundary_refined_locally"
    assert evaluator.cached_point_count == group.calls
    assert group.calls > 17 * 17
    assert group.calls < 129 * 129


def test_boundary_detection_requires_a_crossing_cell() -> None:
    below = np.zeros((3, 3), dtype=float)
    above = np.ones((3, 3), dtype=float)
    mixed = below.copy()
    mixed[1:, 1:] = 0.1
    assert not _boundary_present(below, 0.05)
    assert not _boundary_present(above, 0.05)
    assert _boundary_present(mixed, 0.05)


def test_center_probe_detects_subcell_boundary_island() -> None:
    class _IslandGroup(_SyntheticGroupEvaluator):
        def break_evaluation(self, coordinate):
            self.calls += 1
            selected = np.asarray(coordinate, dtype=float)
            radius_squared = (selected[0] - 0.0625) ** 2 + (selected[1] - 0.0625) ** 2
            fraction = 0.1 if radius_squared < 0.02 ** 2 else 0.0
            return _SyntheticBreakEvaluation(fraction)

    group = _IslandGroup()
    evaluator = SliceGridEvaluator(
        group,
        first_axis=0,
        second_axis=1,
        hidden_axis=2,
    )
    axis, field = evaluator.regular_grid(-1.0, 1.0, 17)
    assert not _boundary_present(field, 0.05)
    candidates, new_points, center_detected = _global_candidate_cells(
        evaluator, axis, field, 0.05
    )
    assert new_points > 0
    assert center_detected >= 1
    assert candidates
