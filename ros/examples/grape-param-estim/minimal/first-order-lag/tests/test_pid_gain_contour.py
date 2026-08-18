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
)


class _SyntheticGroupEvaluator:
    def __init__(self) -> None:
        self.base_gain = np.ones(3)
        self.calls = 0

    def survival_fraction(self, coordinate):
        self.calls += 1
        selected = np.asarray(coordinate, dtype=float)
        # Circular 95% survival boundary in the two visible coordinates.
        return float(
            np.clip(
                0.95 + 0.1 * (selected[0] ** 2 + selected[1] ** 2 - 0.5),
                0.0,
                1.0,
            )
        )


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


def test_adaptive_grid_stops_global_search_then_refines_only_boundary_cells() -> None:
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
        threshold=0.95,
        maximum_grid_size=17,
        final_boundary_grid_size=33,
    )
    assert result.axis_log_ratio.size == 3
    assert [level.size for level in result.global_levels] == [3]
    assert [level.new_point_count for level in result.global_levels] == [9]
    assert len(result.local_refinement_levels) == 4
    assert result.boundary_segments_log_ratio.shape[0] > 0
    assert result.effective_local_equivalent_grid_size == 33
    assert result.stop_reason == "boundary_found_then_refined_locally_to_target_spacing"
    assert evaluator.cached_point_count == group.calls
    assert group.calls > 3 * 3
    assert group.calls < 33 * 33


def test_boundary_detection_requires_a_crossing_cell() -> None:
    below = np.zeros((3, 3), dtype=float)
    above = np.ones((3, 3), dtype=float)
    mixed = below.copy()
    mixed[1:, 1:] = 0.1
    assert not _boundary_present(below, 0.95)
    assert not _boundary_present(above, 0.95)
    assert _boundary_present(mixed, 0.05)


def test_no_boundary_uses_full_nested_global_search() -> None:
    class _IslandGroup(_SyntheticGroupEvaluator):
        def survival_fraction(self, coordinate):
            self.calls += 1
            selected = np.asarray(coordinate, dtype=float)
            radius_squared = (selected[0] - 0.0625) ** 2 + (selected[1] - 0.0625) ** 2
            return 0.9 if radius_squared < 0.02 ** 2 else 1.0

    group = _IslandGroup()
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
        threshold=0.95,
        maximum_grid_size=17,
        final_boundary_grid_size=33,
    )
    assert [level.size for level in result.global_levels] == [3, 5, 9, 17]
    assert [level.new_point_count for level in result.global_levels] == [9, 16, 56, 208]
    assert result.boundary_first_seen_grid_size is None
    assert not result.local_refinement_levels
    assert result.boundary_segments_log_ratio.shape == (0, 2, 2)
    assert result.stop_reason == "no_boundary_detected_through_global_17x17_search"
    assert evaluator.cached_point_count == group.calls == 17 * 17
