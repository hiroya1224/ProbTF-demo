"""Per-mode optical bearing, its derivative and explicit unavailability."""

from copy import deepcopy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from probtf_observation_fusion_demo.visibility_factor import VisibilityFactor, direction_penalty


def component(weight=1., log_weight=None, label='a'):
    return SimpleNamespace(weight=weight, log_weight=log_weight, label=label,
                           state=SimpleNamespace(state_id='source_body_'+label))


def query(position, *, state='trial_body_a', stamp=10., command=(.2, -.3), version=7):
    return SimpleNamespace(mean=np.r_[position, [0., 0., 0.]], covariance=np.eye(6)*.01,
        query_id='visibility:camera_relative_marker', state_id=state, component_id='marker_a',
        frame='camera_optical', stamp=stamp, command=command, version=version)


def factor(weight=3., **config):
    return VisibilityFactor(dict(enabled=True, weight=weight, **config))


@pytest.mark.parametrize('position,expected', [([0., 0., 2.], 0.), ([1., 0., 0.], 1.),
                                               ([0., 0., -2.], 2.)])
def test_extrema_and_log_factor_sign(position, expected):
    record = factor().evaluate([component()], [query(position)])
    assert record['status'] == 'ok'
    assert record['penalty'] == expected
    assert record['log_factor'] == -3.*expected
    assert record['weighted_penalty'] == 3.*expected
    if position[:2] == [0., 0.]:
        assert record['per_mode_record'][0]['directional_stationary']


def test_factor_increases_when_marker_direction_approaches_front_optical_axis():
    points = ([2., 0., -.5], [2., 0., .5], [1., 0., 1.], [.1, 0., 1.], [0., 0., 1.])
    scores = [factor().evaluate([component()], [query(point)])['log_factor'] for point in points]
    assert all(a < b for a, b in zip(scores, scores[1:]))
    assert scores[-1] == 0.


@pytest.mark.parametrize('position', [[.3, -.4, 2.], [-.3, .1, -.9], [1., 0., 0.],
                                      [0., 0., 2.], [0., 0., -2.]])
def test_analytic_position_derivative_matches_independent_central_difference(position):
    position = np.asarray(position)
    _, analytic = direction_penalty(position)
    step = 1e-5
    finite = []
    for index in range(3):
        offset = np.eye(3)[index]*step
        plus = 1.-(position+offset)[2]/np.linalg.norm(position+offset)
        minus = 1.-(position-offset)[2]/np.linalg.norm(position-offset)
        finite.append((plus-minus)/(2*step))
    np.testing.assert_allclose(analytic, finite, rtol=2e-9, atol=2e-11)


def test_direction_gradient_avoids_large_norm_cubing_overflow():
    ordinary, ordinary_gradient = direction_penalty([1., -2., 3.])
    large, large_gradient = direction_penalty([1e200, -2e200, 3e200])
    assert math.isclose(large, ordinary, abs_tol=2e-16)
    np.testing.assert_allclose(large_gradient*1e200, ordinary_gradient, rtol=2e-15)


def test_two_directions_are_not_replaced_by_their_single_mean():
    modes = [component(.25, label='left'), component(.75, label='right')]
    queries = [query([-2., 0., 1.], state='left'), query([2., 0., 1.], state='right')]
    record = factor().evaluate(modes, queries)
    expected = 1.-1./math.sqrt(5.)
    assert record['status'] == 'ok'
    assert math.isclose(record['penalty'], expected)
    collapsed, _ = direction_penalty([1., 0., 1.])
    assert abs(record['penalty']-collapsed) > .2
    np.testing.assert_allclose([item['normalized_weight'] for item in record['per_mode_record']], [.25, .75], atol=2e-16)
    assert len(record['per_mode_record']) == 2
    assert math.isclose(sum(item['penalty_contribution'] for item in record['per_mode_record']), record['penalty'])
    assert math.isclose(sum(item['weighted_penalty_contribution'] for item in record['per_mode_record']), record['weighted_penalty'])
    # The complete query covariance, including translation/orientation entries,
    # is reported unchanged, even though this direction factor uses the mean.
    np.testing.assert_array_equal(record['per_mode_record'][0]['covariance'], queries[0].covariance)


def test_finite_log_weights_survive_underflowed_linear_weights_and_global_scale():
    queries = [query([0., 0., 1.]), query([0., 0., -1.], state='back')]
    a = factor().evaluate([component(0., -10000.), component(0., -10001.)], queries)
    b = factor().evaluate([component(math.exp(-1.), -1.), component(math.exp(-2.), -2.)], queries)
    assert a['status'] == 'ok'
    assert len(a['per_mode_record']) == 2
    np.testing.assert_allclose(a['penalty'], b['penalty'], rtol=0., atol=1e-16)
    assert a['per_mode_record'][0]['input_log_weight'] == -10000.
    assert a['per_mode_record'][1]['positive_optical_depth'] is False
    assert a['per_mode_record'][1]['normalized_weight'] > 0.
    assert math.isclose(a['penalty'], 2./(math.e+1.))


def test_exact_zero_mass_is_recorded_without_nonstandard_json_or_branch_loss():
    record = factor().evaluate([component(1.), component(0.)],
                              [query([1., 0., 1.]), query([0., 0., -1.])])
    assert record['status'] == 'ok'
    assert len(record['per_mode_record']) == 2
    assert record['per_mode_record'][1]['input_log_weight'] == '-Infinity'
    assert record['per_mode_record'][1]['normalized_weight'] == 0.
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize('config,reason', [({'enabled': False, 'weight': 3.}, 'disabled'),
                                          ({'enabled': True, 'weight': 0.}, 'zero_weight')])
def test_inactive_factor_is_exact_unit_and_does_not_inspect_invalid_candidates(config, reason):
    class Forbidden:
        def __iter__(self):
            raise AssertionError('Inactive view factor must not evaluate a candidate query.')
    record = VisibilityFactor(config).evaluate(Forbidden(), Forbidden())
    assert record['status'] == 'unit_factor'
    assert record['reason'] == reason
    assert record['log_factor'] == 0.
    assert record['weighted_penalty'] == 0.
    assert record['penalty'] is None
    assert not record['query_evaluation_performed']
    assert record['unit_factor_applied']


@pytest.mark.parametrize('bad', [[0., 0., 0.], [np.nan, 0., 1.],
                                 [np.inf, 0., 1.]])
def test_bad_active_mode_makes_complete_factor_unavailable_without_renormalizing_survivors(bad):
    record = factor().evaluate([component(.9), component(.1)],
                              [query([0., 0., 1.]), query(bad)])
    assert record['status'] == 'unavailable'
    assert record['penalty'] is None
    assert record['log_factor'] is None
    assert len(record['per_mode_record']) == 2
    assert math.isclose(record['per_mode_record'][0]['normalized_weight'], .9, abs_tol=2e-16)
    assert record['per_mode_record'][1]['status'] == 'unavailable'
    json.dumps(record, allow_nan=False)


def test_finite_bearing_below_numerical_radius_threshold_remains_usable():
    position = np.array([1e-12, 0., 1e-12])
    penalty, gradient = direction_penalty(position)
    assert np.isfinite(penalty) and np.all(np.isfinite(gradient))
    record = factor().evaluate([component()], [query(position)])
    assert record['status'] == 'ok'
    mode = record['per_mode_record'][0]
    assert mode['position_norm_below_diagnostic_threshold']
    assert mode['minimum_norm_is_acceptance_gate'] is False
    np.testing.assert_array_equal(mode['analytic_position_gradient'], gradient)


def test_no_candidates_and_all_zero_mass_are_explicit():
    assert factor().evaluate([], [])['reason'] == 'no_candidates'
    record = factor().evaluate([component(0.)], [query([0., 0., 1.])])
    assert record['status'] == 'unavailable'
    assert any(item['reason'] == 'zero_or_invalid_total_component_mass' for item in record['unavailable_modes'])
    assert factor().evaluate([component()], [])['reason'] == 'component_query_count_mismatch'


def test_single_rear_mode_remains_finite_valid_and_stationary_on_exact_reverse_axis():
    record = factor().evaluate([component()], [query([0., 0., -1.])])
    mode = record['per_mode_record'][0]
    assert record['status'] == 'ok'
    assert mode['front_or_back'] == 'back'
    assert mode['optical_axis_angle_rad'] == math.pi
    assert mode['directional_stationary']
    assert mode['analytic_position_gradient'] == [-0., -0., 0.]
    assert math.exp(record['log_factor']) > 0.


def test_optional_pixel_diagnostics_do_not_reject_rear_or_optical_plane_modes():
    camera = dict(fx=720., fy=720., cx=480., cy=360., width=960, height=720)
    for position, projection_reason in (([0., 0., 1.], 'front_projection'),
                                        ([0., 0., -1.], 'behind_camera_algebraic_projection'),
                                        ([1., 0., 0.], 'zero_optical_depth')):
        record = factor(intrinsics=camera).evaluate([component()], [query(position)])
        assert record['status'] == 'ok'
        mode = record['per_mode_record'][0]
        assert mode['pixel_projection_reason'] == projection_reason
        assert mode['predicted_pixel_inside_image'] == (True if position[2] > 0. else False if position[2] < 0. else None)


def test_source_ids_metadata_and_queries_are_read_without_mutation():
    ids = dict(body=['body_7'], marker=['marker_3'], density='density_2', K=[100., 120.],
               observation_stamp=7., source_stamp=10., command=[.2, -.3])
    ids_saved = deepcopy(ids)
    q = query([.1, .2, 1.])
    mean, covariance = q.mean.copy(), q.covariance.copy()
    record = factor().evaluate([component()], [q], source_ids=ids)
    assert record['source_ids'] == ids_saved
    mode = record['per_mode_record'][0]
    assert mode['source_state_id'] == 'trial_body_a'
    assert mode['source_stamp'] == 10.
    assert mode['source_version'] == 7
    assert mode['source_command'] == [.2, -.3]
    record['source_ids']['body'][0] = 'mutated_record_only'
    assert ids == ids_saved
    np.testing.assert_array_equal(q.mean, mean)
    np.testing.assert_array_equal(q.covariance, covariance)


def test_mismatched_source_time_across_modes_is_not_silently_combined():
    record = factor().evaluate([component(.5), component(.5)],
                              [query([0., 0., 1.]), query([1., 0., 1.], stamp=11.)])
    assert record['status'] == 'unavailable'
    assert any(item['reason'] == 'inconsistent_query_stamp_command_or_version' for item in record['unavailable_modes'])


@pytest.mark.parametrize('config', [dict(weight=-1.), dict(weight=np.nan),
                                   dict(minimum_norm_m=0.), dict(minimum_norm_m=np.inf)])
def test_invalid_configuration_is_explicit(config):
    with pytest.raises(ValueError):
        VisibilityFactor(dict(enabled=True, **config))
