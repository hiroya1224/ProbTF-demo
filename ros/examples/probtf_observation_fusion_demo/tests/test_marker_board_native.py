"""Independent integral parity and real native image-mixture board replays."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from probtf.bingham import (
    _normalizer_derivatives, bingham_second_moment as core_moment,
)
from probtf_observation_fusion_demo import marker_board_native as native
from probtf_observation_fusion_demo.marker_board import fuse_marker_board
from test_marker_board import detection, SETTINGS


PACKAGE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('marker_board_replay_benchmark',
    PACKAGE/'validation/tools/benchmark_marker_board_native.py')
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


@pytest.mark.parametrize('values', [np.zeros(4), [-30., -15., -5., 0.],
    [-1e6, -1e4, -1e2, 0.], [-80., -80., -80., 0.]])
@pytest.mark.parametrize('steps', [31, 120])
def test_same_deterministic_integral_and_both_derivatives(values, steps):
    reference = _normalizer_derivatives(values, steps, need_second_derivative=True)
    actual = native.normalizer_derivatives(values, steps, need_second_derivative=True)
    for a, b in zip(actual, reference):
        np.testing.assert_allclose(a, b, rtol=2e-13, atol=2e-18)


def test_diagonal_residual_keeps_coordinate_order_ties_and_scalar_gauge():
    values = np.array([-9e3, -2e4, -9e3, 0.])
    expected = np.diag(core_moment(np.diag(values)))
    np.testing.assert_allclose(native.diagonal_second_moment(values), expected,
                               rtol=2e-13, atol=1e-16)
    np.testing.assert_allclose(native.diagonal_second_moment(values+17.), expected,
                               rtol=2e-13, atol=1e-16)


@pytest.mark.parametrize('record_index', [0, 1])
def test_recorded_nine_tag_and_blurred_partial_native_mixtures_keep_same_law(record_index):
    fixture = json.loads(BENCH.DEFAULT_FIXTURE.read_text())
    record = fixture['records'][record_index]
    detections = BENCH.load_recorded_detections(record)
    reference = fuse_marker_board(detections, fixture['settings'], arithmetic_backend='python')
    actual = fuse_marker_board(detections, fixture['settings'], arithmetic_backend='native')
    comparison = BENCH.numerical_comparison(reference, actual)
    assert all(comparison['structural_checks'].values())
    differences = comparison['maximum_absolute_differences']
    assert differences['translation'] < 2e-12
    assert differences['covariance'] < 2e-13
    assert differences['quaternion_second_moment'] < 2e-10
    assert differences['projection_log_mass'] < 2e-8
    assert differences['log_weight'] < 1e-12
    assert differences['native_center_query_mean'] < 2e-12
    assert differences['native_center_query_covariance'] < 2e-13


def test_underflowed_mode_and_broad_fallback_retain_all_branches_in_native_path():
    observations = {17: detection(17, center=(0., 0., .8), angles=(0.,)),
        18: detection(18, center=(0., 0., .8), angles=(0., 0.),
            offsets=([1., 0., 0.], [0., 0., 0.]), log_masses=(0., -1000.))}
    reference = fuse_marker_board(observations, SETTINGS, arithmetic_backend='python')
    actual = fuse_marker_board(observations, SETTINGS, arithmetic_backend='native')
    assert actual.record['raw_observations'][1]['components'][1]['weight'] == 0.
    assert actual.record['raw_observations'][1]['components'][1]['log_weight'] == -1000.
    assert len(actual.record['steps'][1]['branch_products']) == 2
    np.testing.assert_allclose(actual.detected.translations, reference.detected.translations,
                               atol=2e-12, rtol=2e-12)
    broad = {21: detection(21, angles=(-1.2, 1.2), log_masses=np.log([.17, .83]))}
    reference = fuse_marker_board(broad, SETTINGS, arithmetic_backend='python')
    actual = fuse_marker_board(broad, SETTINGS, arithmetic_backend='native')
    assert actual.record['final_mixture_fallback_used']
    assert actual.record['central_component_count'] == 2
    assert actual.detected.log_masses == pytest.approx(reference.detected.log_masses, abs=1e-12)
    assert all(BENCH.numerical_comparison(reference, actual)['structural_checks'].values())


def test_explicit_python_reference_environment_and_cached_backend_metadata(monkeypatch):
    monkeypatch.setenv('PROBTF_MARKER_BOARD_BACKEND', 'python_reference')
    reference = fuse_marker_board({21: detection(21)}, SETTINGS)
    assert reference.record['arithmetic_backend'] == 'python'
    assert reference.record['arithmetic_backend_source']['arithmetic_backend'] == 'python_reference'
    assert native.backend_record()['arithmetic_backend'] == 'python_reference'
    record = native.backend_record('native')
    assert record['arithmetic_backend'] == 'native'
    assert len(record['library_sha256']) == 64
    record['library_sha256'] = 'changed external copy'
    assert len(native.backend_record('native')['library_sha256']) == 64


def test_nonfinite_values_fail_without_a_fallback_or_modified_moments():
    with pytest.raises(ValueError, match='finite'):
        native.normalizer_derivatives([0., 0., np.nan, 0.])
