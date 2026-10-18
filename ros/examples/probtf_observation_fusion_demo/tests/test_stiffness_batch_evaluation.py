"""Batch arithmetic retains complete factors, recurrence and frozen charts."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from probtf_observation_fusion_demo.body_native import reexpress_state
from probtf.spherical_law.fusion import Gaussian, PosteriorComponent
from probtf_observation_fusion_demo.model import FrozenBodyBranch
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.stiffness_bayes_factor import ContinuousStiffnessBFS, ResetMeasure
from probtf_observation_fusion_demo.stiffness_identification import (
    ChartedBodyDensityRatio, FrozenStiffnessFactor, StiffnessIdentifier,
)
from legacy_marker import legacy_marker_scene

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]


@pytest.fixture(scope='module')
def context(tmp_path_factory):
    scene = legacy_marker_scene(tmp_path_factory.mktemp('stiffness_batch'))
    frame = scene.step(0., 1000.)
    inference = MovingInference(scene.robot_description, (), config_path=scene.legacy_inference_config)
    state = inference.prior_at(frame.command, frame.stamp, 0)
    settings = dict(prior_config_path=str(REPO/'ros/examples/deflecomp/deflecomp_ros/config/estimator.yaml'))
    native = StiffnessIdentifier(scene.urdf, inference.body_ids, settings)
    reference = StiffnessIdentifier(scene.urdf, inference.body_ids,
        dict(settings, auxiliary_equilibrium_backend='python_reference'))
    native._freeze_branch(state, {})
    reference._branch_anchor = tuple(value.copy() if isinstance(value, np.ndarray) else value
                                    for value in native._branch_anchor)
    return native, reference, state


def factors(identifier, state):
    # The second chart shares the physical problem but has a genuinely
    # different reference orientation; its covariance and all cross terms stay.
    delta = np.zeros_like(state.mean)
    delta[4] = .004
    other = reexpress_state(state, delta, state.covariance, component_id='second_chart')
    branches = []
    for label, weight, source in (('keep', .7, state), ('reset', .3, other)):
        prior = Gaussian(source.mean, source.covariance)
        branches.append(FrozenBodyBranch(label, weight, source, prior, ()))
    result = []
    for index, displacement in enumerate((.001, -.003)):
        mean = state.mean.copy()
        mean[4] += displacement
        child = PosteriorComponent('observed:'+str(index), Gaussian(mean, .8*state.covariance), 1.)
        projected = replace(branches[0], weight=1., posterior_components=(child,))
        ratio = ChartedBodyDensityRatio(tuple(branches), identifier.body_edge_ids,
                                       posterior_branch=projected)
        result.append(FrozenStiffnessFactor('factor:'+str(index), ('obs:'+str(index),), ratio,
            (state, other), identifier._branch_anchor, dict(verified=True)))
    return tuple(result)


def test_native_factor_batch_matches_python_reference_without_collapsing_charts(context):
    native, reference, state = context
    history = factors(native, state)
    x = native.initial_components[0].distribution.mean+np.array([0., -.012, .017, -.009, .006, .02])
    native.factor_cache.clear()
    reference.factor_cache.clear()
    native._physical_prediction_query = None
    native.factors = reference.factors = history
    updates = tuple(SimpleNamespace(factor_id=f.factor_id) for f in history)
    before = dict(native._evaluation_cache_statistics)
    actual = native._batch_factor_evaluation(x, updates)
    expected = tuple(reference._factor_evaluation(f, x) for f in history)
    for got, wanted in zip(actual, expected):
        np.testing.assert_allclose(got[0], wanted[0], rtol=1e-10, atol=2e-10)
        np.testing.assert_allclose(got[1], wanted[1], rtol=1e-8, atol=2e-9)
    assert actual[0][0] != actual[1][0]
    assert native._evaluation_cache_statistics['native_batch_calls'] == before['native_batch_calls']+1
    assert native._evaluation_cache_statistics['native_batch_rows'] == before['native_batch_rows']+1
    assert len(native.factor_cache) == len(history)
    assert history[0].ratio.branches[0].reference_state.state_id != history[0].ratio.branches[1].reference_state.state_id
    np.testing.assert_array_equal(history[0].ratio.branches[1].prior.covariance,
                                  history[0].ratio.branches[1].reference_state.covariance)


def test_factor_cache_survives_physical_query_eviction_and_preserves_full_history(context):
    native, _, state = context
    history = factors(native, state)
    native.factors = history
    updates = tuple(SimpleNamespace(factor_id=f.factor_id) for f in history)
    x = native.initial_components[0].distribution.mean+np.full(6, .013)
    first = native._batch_factor_evaluation(x, updates)
    native._batch_factor_evaluation(x+.0007, updates)
    before = dict(native._evaluation_cache_statistics)
    repeated = native._batch_factor_evaluation(x, updates)
    assert native._evaluation_cache_statistics['native_batch_calls'] == before['native_batch_calls']
    for a, b in zip(first, repeated):
        assert a[0] == b[0]
        np.testing.assert_array_equal(a[1], b[1])
    assert native.factors is history
    assert [f.observation_ids for f in native.factors] == [('obs:0',), ('obs:1',)]


def test_native_failed_row_is_explicit_and_cannot_become_a_cached_density(context, monkeypatch):
    native, _, state = context
    history = factors(native, state)
    native.factors = history
    native.factor_cache.clear()
    native._physical_prediction_query = None
    original = native._auxiliary_native.evaluate_batch
    def unavailable(*args, **kwargs):
        result = original(*args, **kwargs)
        result['diagnostics'][0].update(status='failed', error='actual native solve unavailable')
        return result
    monkeypatch.setattr(native._auxiliary_native, 'evaluate_batch', unavailable)
    x = native.initial_components[0].distribution.mean+np.full(6, .037)
    with pytest.raises(ValueError, match='actual native solve unavailable'):
        native._batch_factor_evaluation(x, tuple(SimpleNamespace(factor_id=f.factor_id) for f in history))
    assert not native.factor_cache
    assert native.factors is history


def _finite_bfs():
    initial = lambda x: (-float(x@x)/2., -np.asarray(x))
    bfs = ContinuousStiffnessBFS(initial, initial_source_id='initial', rho0=0.)
    for i, coefficient in enumerate((.2, -.7, .4)):
        def factor(x, a=coefficient):
            return float(a*x[0]-.1*x[0]**2), np.array([a-.2*x[0]])
        bfs.advance(str(i), ('observation:'+str(i),), factor, rho0=0.)
    return bfs


def test_default_none_and_ordered_batch_produce_bitwise_identical_density_and_score():
    bfs = _finite_bfs()
    x = np.array([.31])
    expected = bfs.evaluate(x)
    calls = []
    def batch(query, updates):
        calls.append(tuple(u.factor_id for u in updates))
        return tuple(u.factor(query) for u in updates)
    bfs.batch_factor_evaluator = batch
    actual = bfs.evaluate(x)
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert calls == [('0', '1', '2')]
    assert bfs.record()['factor_ids'] == ['0', '1', '2']


def test_batch_keeps_each_reset_hazard_scale_and_score_in_original_order():
    bfs = _finite_bfs()
    reset = ResetMeasure('explicit-reset', bfs.initial)
    # Distinct finite branch scales exercise the actual recursive weighting;
    # these are declared test scales, not claims of exact evidence integrals.
    bfs.updates = tuple(replace(update, rho0=rho, keep_log_mass=.37*index,
        reset=reset if rho else None)
        for index, (update, rho) in enumerate(zip(bfs.updates, (.1, 1., .25))))
    before = bfs.updates
    x = np.array([-.42])
    expected = bfs.evaluate(x)
    bfs.batch_factor_evaluator = lambda q, updates: tuple(u.factor(q) for u in updates)
    actual = bfs.evaluate(x)
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert bfs.updates is before
    assert [u.rho0 for u in bfs.updates] == [.1, 1., .25]


@pytest.mark.parametrize('bad', [(), ((0., np.array([np.nan])),)*3])
def test_invalid_batch_cannot_silently_omit_factors_or_supply_nonfinite_scores(bad):
    bfs = _finite_bfs()
    bfs.batch_factor_evaluator = lambda x, updates: bad
    with pytest.raises(ValueError, match='retain every|finite value and score'):
        bfs.evaluate(np.array([.2]))
