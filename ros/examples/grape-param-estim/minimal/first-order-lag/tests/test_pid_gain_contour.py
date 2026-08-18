#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
FIRST_ORDER = HERE.parent
if str(FIRST_ORDER) not in sys.path:
    sys.path.insert(0, str(FIRST_ORDER))

from pid_gain_contour import (  # noqa: E402
    AdaptiveCell,
    ExactGroupGainEvaluator,
    SliceGridEvaluator,
    _average_trim_predictors,
    _adaptive_projection_grid,
    _bilinear_trim_predictor,
    _boundary_present,
    _evaluate_coordinates_from_parent_cells,
    _exact_matrix_with_configuration,
    _exact_matrix_chunk_task,
    _gain_scaled_bilinear_trim_predictor,
)


class _SyntheticGroupEvaluator:
    def __init__(self) -> None:
        self.base_gain = np.ones(3)
        self.calls = 0
        self.coordinates = []

    def survival_fraction(self, coordinate):
        self.calls += 1
        selected = np.asarray(coordinate, dtype=float)
        self.coordinates.append(selected.copy())
        # Circular 95% survival boundary in the two visible coordinates.
        return float(
            np.clip(
                0.95 + 0.1 * (selected[0] ** 2 + selected[1] ** 2 - 0.5),
                0.0,
                1.0,
            )
        )


def test_unresolved_trim_is_returned_as_an_invalid_pole_sample() -> None:
    trim = SimpleNamespace(
        equilibrium_valid=False,
        piecewise_linearization_near_kink=False,
        root_nfev=11,
        trim_vector=np.zeros(10),
        root_initial_step_infinity_norm=0.0,
        root_initial_unchanged=True,
    )
    task = ((7,), (object(),), object(), object(), 0.01, object(), None, None)
    with patch(
        "pid_gain_contour._analyze_plant",
        return_value={
            "jacobian": None,
            "trim": trim,
            "analytic_piecewise_near_kink": False,
        },
    ):
        result = _exact_matrix_chunk_task(task)
    indices, matrices, pole_valid, near_kink = result[:4]
    assert np.array_equal(indices, np.asarray((7,)))
    assert matrices.shape == (1, 26, 26)
    assert not pole_valid[0]
    assert not near_kink[0]


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
        maximum_grid_size=9,
        final_boundary_grid_size=33,
    )
    assert result.axis_log_ratio.size == 3
    assert np.array_equal(group.coordinates[0], np.zeros(3))
    assert sum(np.array_equal(value, np.zeros(3)) for value in group.coordinates) == 1
    assert [level.size for level in result.global_levels] == [3]
    assert [level.new_point_count for level in result.global_levels] == [8]
    assert len(result.local_refinement_levels) == 4
    assert result.boundary_segments_log_ratio.shape[0] > 0
    assert result.effective_local_equivalent_grid_size == 33
    assert result.stop_reason == (
        "boundary_found_on_global_grid_then_refined_locally_to_target_spacing"
    )
    assert result.center_probe is None
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


def test_no_boundary_stops_after_9x9_and_all_cell_center_probes() -> None:
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
        maximum_grid_size=9,
        final_boundary_grid_size=33,
    )
    assert [level.size for level in result.global_levels] == [3, 5, 9]
    assert [level.new_point_count for level in result.global_levels] == [8, 16, 56]
    assert result.boundary_first_seen_grid_size is None
    assert not result.local_refinement_levels
    assert result.boundary_segments_log_ratio.shape == (0, 2, 2)
    assert result.stop_reason == (
        "no_boundary_detected_through_global_9x9_and_cell_center_probes"
    )
    assert result.center_probe is not None
    assert result.center_probe.coordinates.shape == (64, 2)
    assert not np.any(result.center_probe.suspicious_mask)
    assert evaluator.cached_point_count == group.calls == 9 * 9 + 64


def test_bilinear_trim_predictor_has_expected_midpoint_algebra() -> None:
    shape = (3, 10)
    corners = [np.full(shape, value, dtype=float) for value in (0.0, 2.0, 6.0, 4.0)]
    edge = _bilinear_trim_predictor(
        x=0.5, y=0.0, x0=0.0, x1=1.0, y0=0.0, y1=1.0,
        corner_trims=corners,
    )
    center = _bilinear_trim_predictor(
        x=0.5, y=0.5, x0=0.0, x1=1.0, y0=0.0, y1=1.0,
        corner_trims=corners,
    )
    general = _bilinear_trim_predictor(
        x=0.25, y=0.75, x0=0.0, x1=1.0, y0=0.0, y1=1.0,
        corner_trims=corners,
    )
    assert np.array_equal(edge, np.full(shape, 1.0))
    assert np.array_equal(center, np.full(shape, 3.0))
    assert np.allclose(general, np.full(shape, 3.5))


def test_shared_predictor_average_is_order_independent() -> None:
    first = np.arange(20, dtype=float).reshape(2, 10)
    second = first + 10.0
    expected = 0.5 * (first + second)
    assert np.array_equal(_average_trim_predictors((first, second)), expected)
    assert np.array_equal(_average_trim_predictors((second, first)), expected)


def test_gain_scaled_predictor_tracks_reciprocal_integral_state() -> None:
    corner_gains = (
        np.full(6, 1.0),
        np.full(6, 4.0),
        np.full(6, 4.0),
        np.full(6, 1.0),
    )
    corner_trims = []
    for gain in corner_gains:
        trim = np.zeros((2, 10), dtype=float)
        trim[:, :6] = 12.0 / gain
        trim[:, 6:] = 3.0
        corner_trims.append(trim)
    target_gain = np.full(6, 2.0)
    predicted = _gain_scaled_bilinear_trim_predictor(
        x=0.5, y=0.5, x0=0.0, x1=1.0, y0=0.0, y1=1.0,
        corner_trims=corner_trims,
        corner_integral_gains=corner_gains,
        target_integral_gains=target_gain,
    )
    assert np.array_equal(predicted[:, :6], np.full((2, 6), 6.0))
    assert np.array_equal(predicted[:, 6:], np.full((2, 4), 3.0))


def test_gain_scaled_predictor_preserves_constant_i_raw_result_bitwise() -> None:
    rng = np.random.default_rng(7)
    corners = [rng.normal(size=(3, 10)) for _ in range(4)]
    raw = _bilinear_trim_predictor(
        x=0.37, y=0.61, x0=-1.0, x1=2.0, y0=-2.0, y1=3.0,
        corner_trims=corners,
    )
    gain = np.asarray((1.0, 1.0, 2.0, 3.0, 3.0, 4.0))
    scaled = _gain_scaled_bilinear_trim_predictor(
        x=0.37, y=0.61, x0=-1.0, x1=2.0, y0=-2.0, y1=3.0,
        corner_trims=corners,
        corner_integral_gains=(gain, gain, gain, gain),
        target_integral_gains=gain,
    )
    assert np.array_equal(scaled, raw)


def test_refinement_coordinates_are_submitted_as_one_frozen_wave() -> None:
    class _WaveGroup:
        def __init__(self) -> None:
            self.requests = []

        @staticmethod
        def _key(coordinate):
            return tuple(float(value) for value in np.round(coordinate, 14))

        def evaluate_wave(self, requests):
            self.requests.append(tuple(requests))
            # Return in reverse insertion order to ensure visible results are
            # associated by canonical gain key rather than completion order.
            return {
                self._key(request.log_ratio): SimpleNamespace(
                    trim_vectors=np.full(
                        (2, 10), float(np.sum(request.log_ratio))
                    )
                )
                for request in reversed(requests)
            }

        def survival_fraction(self, coordinate):
            return float(np.sum(coordinate))

        @staticmethod
        def integral_gain_vector(coordinate):
            del coordinate
            return np.ones(6, dtype=float)

    group = _WaveGroup()
    evaluator = SliceGridEvaluator(
        group, first_axis=0, second_axis=1, hidden_axis=2
    )
    corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    for index, coordinate in enumerate(corners):
        key = evaluator._visible_key(*coordinate)
        evaluator._cache[key] = float(index)
        evaluator._trim_cache[key] = np.full((2, 10), float(index))
    parent = AdaptiveCell(0.0, 1.0, 0.0, 1.0, (0.0, 1.0, 2.0, 3.0))
    coordinates = ((0.5, 0.0), (1.0, 0.5), (0.5, 0.5), (0.5, 1.0))
    _evaluate_coordinates_from_parent_cells(
        evaluator, coordinates, (parent,), stage="test:wave"
    )
    assert len(group.requests) == 1
    requests = group.requests[0]
    assert len(requests) == len(coordinates)
    assert all(request.stage == "test:wave" for request in requests)
    # Every nearest fallback was frozen from the four predecessor corners;
    # no current-wave trim can appear in another request.
    assert all(
        request.nearest_trims is not None
        and set(np.unique(request.nearest_trims)).issubset({0.0, 1.0, 2.0, 3.0})
        for request in requests
    )
    center_request = next(
        request
        for request in requests
        if np.array_equal(request.log_ratio, np.asarray((0.5, 0.5, 0.0)))
    )
    assert np.array_equal(center_request.initial_trims, np.full((2, 10), 1.5))
    assert evaluator.cached_point_count == len(corners) + len(coordinates)


def test_chunk_assembly_restores_sample_index_order() -> None:
    def row(indices):
        selected = np.asarray(indices, dtype=int)
        count = selected.size
        matrices = np.asarray(
            [np.full((26, 26), float(index)) for index in selected]
        )
        return (
            selected,
            matrices,
            np.ones(count, dtype=bool),
            np.zeros(count, dtype=bool),
            np.repeat(selected[:, None], 10, axis=1).astype(float),
            np.ones(count, dtype=int),
            np.ones((count, 3), dtype=int),
            np.ones(count, dtype=bool),
            np.zeros(count, dtype=bool),
            np.zeros(count, dtype=bool),
            np.zeros(count, dtype=bool),
            np.zeros(count),
            np.zeros(count),
            np.zeros(count),
            np.ones(count, dtype=bool),
        )

    evaluator = SimpleNamespace(plants=(None,) * 4)
    stack = ExactGroupGainEvaluator._assemble_matrix_rows(
        evaluator, (row((2, 0)), row((3, 1)))
    )
    assert np.array_equal(stack[0][:, 0, 0], np.arange(4, dtype=float))
    assert np.array_equal(stack[3][:, 0], np.arange(4, dtype=float))


def test_trim_fallback_retries_nearest_then_generic_without_dropping_sample() -> None:
    invalid = SimpleNamespace(
        equilibrium_valid=False,
        piecewise_linearization_near_kink=False,
        root_nfev=9,
        trim_vector=np.zeros(10),
        root_initial_step_infinity_norm=1.0,
        root_initial_unchanged=False,
    )
    valid = SimpleNamespace(
        equilibrium_valid=True,
        piecewise_linearization_near_kink=False,
        root_nfev=4,
        trim_vector=np.arange(10, dtype=float),
        root_initial_step_infinity_norm=2.0,
        root_initial_unchanged=False,
    )
    calls = []

    def fake_analyze(**kwargs):
        calls.append(kwargs["initial_trim"])
        trim = invalid if len(calls) < 3 else valid
        return {
            "trim": trim,
            "jacobian": None if not trim.equilibrium_valid else np.eye(26),
            "analytic_piecewise_near_kink": False,
            "trim_wall_seconds": 0.01,
            "analytic_jacobian_wall_seconds": 0.02 if trim.equilibrium_valid else 0.0,
        }

    warm = np.ones(10)
    nearest = np.full(10, 2.0)
    with patch("pid_gain_contour._analyze_plant", side_effect=fake_analyze):
        result = _exact_matrix_with_configuration(
            plant=object(), vehicle_model=object(), actuator_parameters=object(),
            controller_dt=0.01, controller_configuration=object(),
            initial_trim=warm, nearest_trim=nearest,
        )
    matrix, _near, trim_vector, _nfev, attempts = result[:5]
    assert np.array_equal(calls[0], warm)
    assert np.array_equal(calls[1], nearest)
    assert calls[2] is None
    assert matrix is not None
    assert np.array_equal(trim_vector, valid.trim_vector)
    assert np.array_equal(attempts, np.asarray((9, 9, 4)))
    assert result[6] and result[7]


def test_boundary_discovery_schedule_reaches_same_33_equivalent_spacing() -> None:
    expected = {
        0.5: (5, 3, "global_grid"),
        0.25: (9, 2, "global_grid"),
        0.125: (None, 2, "center_probe"),
    }
    for center, (first_seen, local_count, detection_source) in expected.items():
        class _LateBoundary(_SyntheticGroupEvaluator):
            def survival_fraction(self, coordinate):
                self.calls += 1
                selected = np.asarray(coordinate, dtype=float)
                radius_squared = (selected[0] - center) ** 2 + (selected[1] - center) ** 2
                return 0.9 if radius_squared < 0.01 ** 2 else 1.0

        group = _LateBoundary()
        evaluator = SliceGridEvaluator(group, first_axis=0, second_axis=1, hidden_axis=2)
        result = _adaptive_projection_grid(
            evaluator, name="pi", first_axis=0, second_axis=1, hidden_axis=2,
            lower=-1.0, upper=1.0, threshold=0.95,
            maximum_grid_size=9, final_boundary_grid_size=33,
        )
        assert result.boundary_first_seen_grid_size == first_seen
        assert len(result.local_refinement_levels) == local_count
        assert result.effective_local_equivalent_grid_size == 33
        if detection_source == "center_probe":
            assert result.center_probe is not None
            assert np.count_nonzero(result.center_probe.suspicious_mask) == 1
            assert [
                level.input_cell_kind for level in result.local_refinement_levels
            ] == ["suspicious", "boundary"]
            assert result.stop_reason == (
                "center_probe_detected_suspicious_cells_then_refined_"
                "locally_to_target_spacing"
            )
        else:
            assert result.center_probe is None
            assert all(
                level.input_cell_kind == "boundary"
                for level in result.local_refinement_levels
            )
