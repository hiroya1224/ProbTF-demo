"""Memoization preserves complete core laws, clocks and nested records."""
from copy import deepcopy
from dataclasses import replace
import json
import pickle

import numpy as np
import pytest

from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, bingham_from_rotation_coordinates,
    reexpress_state as core_reexpress_state,
)
from probtf_observation_fusion_demo.immutable_body_cache import ReexpressionCache, freeze_json_record


def _state():
    transforms = tuple(BinghamTransform('link_'+str(index), np.array([.1, .2, .3]),
        bingham=bingham_from_rotation_coordinates([1., 0., 0., 0.], .001*np.eye(3)),
        translation_covariance=.01*np.eye(3)) for index in range(2))
    return JointTransformState.from_independent(transforms, stamp=2., command=(.1, .2),
                                                component_id='retained')


def _same_state(left, right):
    assert left.state_id == right.state_id
    assert (left.stamp, left.version, left.command, left.component_id) == (right.stamp, right.version, right.command, right.component_id)
    np.testing.assert_array_equal(left.mean, right.mean)
    np.testing.assert_array_equal(left.covariance, right.covariance)
    for a, b in zip(left.transforms, right.transforms):
        for name in ('translation', 'translation_covariance', 'rotation_wxyz', 'bingham'):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_exact_repeated_core_reexpression_is_reused_with_every_cross_block():
    state = _state(); mean = np.linspace(-.01, .01, 12)
    covariance = .8*state.covariance.copy()
    covariance[3, 9] = covariance[9, 3] = .0001
    expected = core_reexpress_state(state, mean, covariance, version=5, component_id='observed')
    cache = ReexpressionCache(3)
    actual = cache.reexpress(state, mean, covariance, version=5, component_id='observed')
    _same_state(actual, expected)
    assert actual is cache.reexpress(state, mean.copy(), covariance.copy(), version=5, component_id='observed')
    assert cache.statistics()['hits'] == 1
    assert actual.covariance[3, 9] != 0.
    assert not actual.covariance.flags.writeable


@pytest.mark.parametrize('change', ['mean', 'cross', 'version', 'component', 'stamp', 'command'])
def test_any_law_or_provenance_change_is_a_miss(change):
    state = _state(); mean = np.zeros(12); covariance = state.covariance.copy()
    cache = ReexpressionCache(3)
    original = cache.reexpress(state, mean, covariance, version=4, component_id='result')
    version, component = 4, 'result'
    if change == 'mean': mean[0] = .001
    if change == 'cross': covariance[3, 9] = covariance[9, 3] = .0001
    if change == 'version': version = 5
    if change == 'component': component = 'other'
    if change == 'stamp': state = replace(state, stamp=3.)
    if change == 'command': state = replace(state, command=(.2, .3))
    actual = cache.reexpress(state, mean, covariance, version=version, component_id=component)
    expected = core_reexpress_state(state, mean, covariance, version=version, component_id=component)
    _same_state(actual, expected)
    assert actual is not original and actual.state_id != original.state_id
    assert cache.statistics()['hits'] == 0 and cache.statistics()['misses'] == 2


def test_eviction_discards_computation_without_changing_previously_retained_laws():
    state = _state(); cache = ReexpressionCache(2); mean = np.zeros(12)
    first = cache.reexpress(state, mean, state.covariance, component_id='first')
    first_id = first.state_id
    for name in ('second', 'third'):
        cache.reexpress(state, mean, state.covariance, component_id=name)
    assert cache.statistics()['entries'] == 2 and cache.statistics()['evictions'] == 1
    again = cache.reexpress(state, mean, state.covariance, component_id='first')
    _same_state(first, again)
    assert again is not first and first.state_id == first_id


def test_separately_stored_marginal_bytes_are_part_of_the_complete_source_key():
    state = _state(); cache = ReexpressionCache(2)
    first = cache.reexpress(state, state.mean, state.covariance)
    marginal = state.transforms[0].translation_covariance+1e-14*np.eye(3)
    edge = replace(state.transforms[0], translation_covariance=marginal, rotation_wxyz=None)
    other = replace(state, transforms=(edge, state.transforms[1]))
    # Core identity uses the joint covariance; its tolerated separate edge
    # marginal is deliberately included in the demo memo key as well.
    assert other.state_id == state.state_id
    second = cache.reexpress(other, other.mean, other.covariance)
    assert second is not first and cache.statistics()['hits'] == 0
    _same_state(second, core_reexpress_state(other, other.mean, other.covariance))


def test_actual_core_failure_is_never_cached_as_a_result():
    state = _state(); cache = ReexpressionCache(2)
    for _ in range(2):
        with pytest.raises(ValueError):
            cache.reexpress(state, np.full(12, float('nan')), state.covariance)
    assert cache.statistics()['misses'] == 2 and cache.statistics()['entries'] == 0


def test_immutable_record_detaches_inputs_and_preserves_JSON_deepcopy_and_pickle():
    source = dict(stamp=1., command=[.1, .2], modes=[dict(ids=['event:1'], covariance=[[1., .2], [.2, 2.]])])
    encoded = json.dumps(source, sort_keys=True)
    frozen = freeze_json_record(source)
    source['modes'][0]['ids'].append('later')
    source['modes'][0]['covariance'][0][0] = 99.
    assert json.dumps(frozen, sort_keys=True) == encoded
    assert deepcopy(frozen) is frozen
    restored = pickle.loads(pickle.dumps(frozen))
    assert json.dumps(restored, sort_keys=True) == encoded
    for value in (frozen, restored):
        with pytest.raises(TypeError, match='immutable'): value['stamp'] = 2.
        with pytest.raises(TypeError, match='immutable'): value['modes'][0]['ids'].append('change')
        with pytest.raises(TypeError, match='immutable'): value['modes'][0]['covariance'][0][0] = 3.
        with pytest.raises(TypeError, match='immutable'): value.update(stamp=2.)
        with pytest.raises(TypeError, match='immutable'): value['command'].__imul__(2)
