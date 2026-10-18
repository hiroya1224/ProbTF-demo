"""Analytic change-kernel moments, chart transport and fixed-goal covariance."""

from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from probtf.geometry import quat_to_rotmat, rotmat_to_quat
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, _right_jacobian, bingham_from_rotation_coordinates, point_query,
)
from probtf_observation_fusion_demo.body_native import reexpress_state
from probtf.spherical_law.fusion import FusedMixture, Gaussian, PosteriorComponent
from probtf.spherical_law.support import covariance_support
from probtf_observation_fusion_demo.bayes_factor import (
    common_chart_gaussian, projection_input, retain_transition_goals,
    transition_priors,
)
from probtf_observation_fusion_demo.body_projection import project_update
from probtf_observation_fusion_demo.model import (
    DemoResult, FrozenBodyBranch, Stage, WeightedState,
)


def _state(*, variance=.002, angle=0., component_id='retained', version=7):
    def edge(name, translation, rotation, rotation_variance, translation_variance):
        return BinghamTransform(name, translation,
            bingham=bingham_from_rotation_coordinates(
                rotmat_to_quat(rotation), rotation_variance*np.eye(3)),
            translation_covariance=translation_variance*np.eye(3))
    transforms = (
        edge('joint_a', [0., 0., .1],
             Rotation.from_rotvec([angle, -.4*angle, .2*angle]).as_matrix(), variance, .0001),
        edge('joint_b', [.5, 0., 0.], np.eye(3), variance, .0001),
        BinghamTransform('tool', [.2, 0., 0.], rotation_wxyz=[1., 0., 0., 0.]),
        edge('marker', [1., 0., .5], np.eye(3), .004, .003),
    )
    independent = JointTransformState.from_independent(transforms,
        component_id=component_id, version=version, stamp=1000., command=(.1, .2))
    covariance = independent.covariance.copy()
    covariance[3, 9] = covariance[9, 3] = .15*variance
    covariance[3, 18] = covariance[18, 3] = .1*np.sqrt(variance*.003)
    covariance[9, 19] = covariance[19, 9] = -.08*np.sqrt(variance*.003)
    return replace(independent, covariance=covariance)


@pytest.mark.parametrize('distance_from_cut', [1e-9, 1e-12])
def test_finite_near_cut_chart_transport_is_not_rejected_by_angular_margin(distance_from_cut):
    axis = np.array([1., -.4, .2])
    source = _state(angle=(math.pi-distance_from_cut)/np.linalg.norm(axis),
                    component_id='near_cut_source')
    target = _state(component_id='target')
    distribution = Gaussian(source.mean, source.covariance)
    moved = common_chart_gaussian(source, distribution, target)
    assert np.all(np.isfinite(moved.mean))
    assert np.all(np.isfinite(moved.covariance))
    assert np.linalg.norm(moved.mean[3:6]) == pytest.approx(
        math.pi-distance_from_cut, rel=0., abs=2e-15)
    assert covariance_support(moved.covariance).rank == covariance_support(source.covariance).rank
    np.testing.assert_array_equal(source.mean, 0.)


def _raw_result(*, keep_mass=.2):
    retained = _state()
    direct = _state(variance=.008, angle=.08, component_id='direct')
    priors, _ = transition_priors(WeightedState(retained.component_id, 1., retained, 0.), direct)
    branches, raw, actual = [], [], []
    for index, (prior, mass) in enumerate(zip(priors, (keep_mass, 1.-keep_mass))):
        reference = prior.state
        mean = np.zeros(24)
        mean[3:6] = ([.01, -.003, .002] if index == 0 else [-.015, .002, -.004])
        mean[9] = .006 if index == 0 else -.02
        child = PosteriorComponent(prior.label+'|imu',
            Gaussian(mean, reference.covariance*(.5 if index == 0 else .6)),
            mass, log_weight=math.log(mass))
        state = reexpress_state(reference, child.distribution.mean,
            child.distribution.covariance, version=9, component_id=child.label)
        raw.append(child)
        actual.append(WeightedState(child.label, mass, state, math.log(mass)))
        branches.append(FrozenBodyBranch(prior.label, prior.weight, reference,
            Gaussian(reference.mean, reference.covariance), (child,), prior.log_weight))
    fused = FusedMixture(tuple(raw), (), ('imu:7',), retained.version,
                         retained.stamp, retained.command)
    path = (('joint_a', False), ('joint_b', False), ('tool', False))
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    result = DemoResult(retained, (Stage('prior', priors), Stage('fused', tuple(actual), ('imu',))),
        image, image, {'joint_a': path[:1], 'joint_b': path[:2]}, path,
        path, (('marker', False),), tuple((name, True) for name, _ in reversed(path))
            + (('marker', False),),
        'base', (), (), {'observation_ids': ['imu:7']},
        tuple(branch.prior for branch in branches), fused, tuple(branches))
    return result


def _fixed_goals(reference):
    goals = []
    for index, weight in enumerate((.4, .6)):
        marker = reference.transforms[-1]
        moved = BinghamTransform(marker.edge_id,
            marker.translation+[0., .02*index, -.01*index],
            bingham=marker.bingham,
            translation_covariance=marker.translation_covariance)
        transforms = reference.transforms[:-1]+(moved,)
        label = 'goal:{}'.format(index)
        state = replace(reference, transforms=transforms, component_id=label)
        goals.append(WeightedState(label, weight, state, math.log(weight)))
    return tuple(goals)


def test_reset_kernel_reuses_actual_environment_and_only_resets_body_cross():
    retained, direct = _state(), _state(variance=.008, angle=.08, component_id='direct')
    source = WeightedState(retained.component_id, 1., retained, 0.)
    priors, record = transition_priors(source, direct)
    keep, reset = priors
    assert keep.state is retained
    assert (keep.weight, reset.weight) == (.75, .25)
    assert keep.log_weight == math.log(.75) and reset.log_weight == math.log(.25)
    assert reset.state.transforms[-1] is retained.transforms[-1]
    assert all(actual is supplied for actual, supplied in
               zip(reset.state.transforms[:-1], direct.transforms[:-1]))
    np.testing.assert_array_equal(reset.state.covariance[:18, :18], direct.covariance[:18, :18])
    np.testing.assert_array_equal(reset.state.covariance[18:, 18:], retained.covariance[18:, 18:])
    np.testing.assert_array_equal(reset.state.covariance[:18, 18:], 0.)
    np.testing.assert_array_equal(keep.state.covariance, retained.covariance)
    assert reset.label == reset.state.component_id == 'reset:7'
    assert record['change_prior_odds'] == pytest.approx(1./3.)
    assert record['retained_source_state_id'] == retained.state_id
    assert record['direct_family_state_id'] == direct.state_id
    assert not record['sampling_used'] and not record['physical_truth_received']


@pytest.mark.parametrize('probability', [0., 1., -.1, math.nan, math.inf])
def test_transition_rejects_invalid_probabilities(probability):
    retained = _state()
    with pytest.raises(ValueError, match='probability'):
        transition_priors(WeightedState(retained.component_id, 1., retained, 0.),
                          _state(component_id='direct'), p=probability)


def test_transition_and_chart_transport_reject_mismatched_time_geometry_and_mass():
    retained = _state()
    source = WeightedState(retained.component_id, 1., retained, 0.)
    for foreign in (replace(retained, stamp=1001.), replace(retained, version=8),
                    replace(retained, command=(.2, .3))):
        with pytest.raises(ValueError, match='geometry, version, time and command'):
            transition_priors(source, foreign)
        with pytest.raises(ValueError, match='geometry, version, time and command'):
            common_chart_gaussian(retained, Gaussian(retained.mean, retained.covariance), foreign)
    with pytest.raises(ValueError, match='unit mass'):
        transition_priors(replace(source, weight=.5, log_weight=math.log(.5)), retained)
    with pytest.raises(ValueError, match='label'):
        transition_priors(replace(source, label='other'), retained)


def test_same_chart_preserves_gaussian_object_and_all_bits():
    reference = _state()
    distribution = Gaussian(np.arange(24)*.0001, reference.covariance*.8)
    # Exact mounts have no uncertain coordinates.
    mean = distribution.mean.copy(); mean[12:18] = 0.
    distribution = Gaussian(mean, distribution.covariance)
    assert common_chart_gaussian(reference, distribution, reference) is distribution
    assert common_chart_gaussian(reference, distribution, replace(reference)) is distribution


def test_nonzero_mean_chart_transport_matches_explicit_differential_and_all_cross():
    target, source = _state(), _state(angle=.09, component_id='different-chart')
    mean = np.zeros(24)
    mean[3:6] = [.018, -.027, .009]
    mean[9:12] = [-.01, .02, .03]
    mean[18:21] = [.005, -.004, .002]
    law = Gaussian(mean, source.covariance*.8)
    transported = common_chart_gaussian(source, law, target)
    expected_mean = mean.copy()
    jacobian = np.eye(24)
    for index, (old, new) in enumerate(zip(source.transforms, target.transforms)):
        translation, rotation = slice(6*index, 6*index+3), slice(6*index+3, 6*index+6)
        expected_mean[translation] += old.translation-new.translation
        value = Rotation.from_matrix(quat_to_rotmat(new.rotation_wxyz).T
            @ quat_to_rotmat(old.rotation_wxyz)
            @ Rotation.from_rotvec(mean[rotation]).as_matrix()).as_rotvec()
        expected_mean[rotation] = value
        jacobian[rotation, rotation] = np.linalg.solve(_right_jacobian(value),
                                                       _right_jacobian(mean[rotation]))
    np.testing.assert_allclose(transported.mean, expected_mean, atol=2e-16)
    np.testing.assert_allclose(transported.covariance,
                               jacobian @ law.covariance @ jacobian.T, atol=2e-18)
    assert transported.covariance[3, 9] != 0.
    assert transported.covariance[3, 18] != 0.
    # Independent central-difference check of this coordinate map's derivative;
    # the production operation itself performs no numerical differentiation.
    epsilon = 1e-6
    numerical = np.zeros((3, 3))
    relative = quat_to_rotmat(target.transforms[0].rotation_wxyz).T @ quat_to_rotmat(source.transforms[0].rotation_wxyz)
    for axis in range(3):
        step = np.eye(3)[axis]*epsilon
        plus = Rotation.from_matrix(relative @ Rotation.from_rotvec(mean[3:6]+step).as_matrix()).as_rotvec()
        minus = Rotation.from_matrix(relative @ Rotation.from_rotvec(mean[3:6]-step).as_matrix()).as_rotvec()
        numerical[:, axis] = (plus-minus)/(2.*epsilon)
    np.testing.assert_allclose(jacobian[3:6, 3:6], numerical, atol=2e-10, rtol=2e-9)


def test_temporary_projection_view_preserves_original_branches_and_full_joint_moments():
    result = _raw_result()
    original_branches, original_stage = result.frozen_body_branches, result.stages[-1]
    temporary = projection_input(result)
    assert result.frozen_body_branches is original_branches
    assert result.stages[-1] is original_stage
    assert 'bayes_factor_transport' not in result.records
    assert len(temporary.frozen_body_branches) == 1
    projection = project_update(temporary)
    children = temporary.frozen_body_branches[0].posterior_components
    mean = sum(child.weight*child.distribution.mean for child in children)
    within = sum(child.weight*child.distribution.covariance for child in children)
    between = sum(child.weight*np.outer(child.distribution.mean-mean,
        child.distribution.mean-mean) for child in children)
    np.testing.assert_allclose(projection.posterior.mean, mean, atol=2e-17)
    np.testing.assert_allclose(projection.posterior.covariance, within+between, atol=2e-17)
    assert projection.posterior.covariance[3, 18] != 0.
    assert np.linalg.norm(between[:12, :12]) > 0.
    provenance = temporary.records['bayes_factor_transport']
    assert provenance['source_posterior_state_ids'] == [item.state.state_id for item in original_stage.components]
    for item, record in zip(temporary.stages[-1].components, provenance['sources']):
        assert record['label'] == item.label
        assert record['common_posterior_state_id'] == item.state.state_id
        np.testing.assert_array_equal(item.state.covariance[12:18], 0.)


def test_projection_transport_rejects_original_source_parameter_substitution():
    result = _raw_result()
    original = result.stages[-1].components[1]
    mean = np.zeros(24); mean[3] = .001
    substituted = reexpress_state(original.state, mean, original.state.covariance,
        version=original.state.version, component_id=original.label)
    wrong = replace(original, state=substituted)
    stage = replace(result.stages[-1], components=(result.stages[-1].components[0], wrong))
    corrupted = replace(result, stages=result.stages[:-1]+(stage,))
    with pytest.raises(ValueError, match='original source state'):
        projection_input(corrupted)


def test_fixed_goal_marginals_survive_reset_expansion_with_schmidt_cross_and_full_psd():
    result = _raw_result()
    projection = project_update(projection_input(result))
    reference = result.prior
    goals = _fixed_goals(reference)
    # This is the actual failure mode of the old contraction-only hold helper.
    assert np.linalg.eigvalsh(reference.covariance[:12, :12]
        -projection.posterior.covariance[:12, :12])[0] < -.001
    retained = retain_transition_goals(goals, result, projection)
    chart = np.eye(18)
    for index in range(2):
        block = slice(6*index+3, 6*index+6)
        chart[block, block] = _right_jacobian(projection.posterior.mean[block])
    for old, actual in zip(goals, retained):
        state = actual.state
        assert (actual.label, actual.weight, actual.log_weight) == (old.label, old.weight, old.log_weight)
        assert state.transforms[-1] is old.state.transforms[-1]
        assert all(a is b for a, b in zip(state.transforms[:-1], projection.state.transforms[:-1]))
        np.testing.assert_array_equal(state.covariance[18:, 18:], old.state.covariance[18:, 18:])
        np.testing.assert_array_equal(state.covariance[:18, :18], projection.state.covariance[:18, :18])
        np.testing.assert_allclose(state.covariance[:18, 18:],
            .2*.5*chart @ old.state.covariance[:18, 18:], atol=2e-18, rtol=2e-12)
        assert np.linalg.norm(state.covariance[:18, 18:]) > 0.
        covariance_support(state.covariance)
        for endpoint in ([0., 0., 0.], [.05, 0., 0.], [0., -.24, .045]):
            before = point_query(old.state, (('marker', False),), endpoint)
            after = point_query(state, (('marker', False),), endpoint)
            np.testing.assert_array_equal(before.mean, after.mean)
            np.testing.assert_allclose(before.covariance, after.covariance, atol=2e-18)
    record = result.records['bayes_factor_goal_retention']
    assert record['keep_posterior_mass'] == pytest.approx(.2)
    assert not record['goal_marginals_reconditioned']
    assert not record['full_environment_bayes_posterior_claimed']


def test_positive_underflowed_branches_and_goal_modes_are_not_pruned():
    result = _raw_result()
    branches = list(result.frozen_body_branches)
    children = [replace(branches[0].posterior_components[0], weight=1., log_weight=0.),
                replace(branches[1].posterior_components[0], weight=0., log_weight=-1000.)]
    branches = tuple(replace(branch, posterior_components=(child,))
                     for branch, child in zip(branches, children))
    stage = replace(result.stages[-1], components=tuple(replace(actual,
        weight=child.weight, log_weight=child.log_weight)
        for actual, child in zip(result.stages[-1].components, children)))
    result = replace(result, frozen_body_branches=branches,
        frozen_body_posterior=replace(result.frozen_body_posterior, components=tuple(children)),
        stages=result.stages[:-1]+(stage,))
    temporary = projection_input(result)
    assert len(temporary.stages[-1].components) == 2
    assert temporary.stages[-1].components[1].log_weight == -1000.
    projection = project_update(temporary)
    goals = _fixed_goals(result.prior)
    goals = (replace(goals[0], weight=1., log_weight=0.),
             replace(goals[1], weight=0., log_weight=-1000.))
    retained = retain_transition_goals(goals, result, projection)
    assert len(retained) == 2 and retained[1].log_weight == -1000.
    assert retained[1].state.transforms[-1] is goals[1].state.transforms[-1]


def test_transition_transport_projection_and_hold_do_not_use_monte_carlo(monkeypatch):
    result = _raw_result()
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Bayes factor transition operations must be deterministic.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    reference = result.prior
    transition_priors(WeightedState(reference.component_id, 1., reference, 0.),
                      _state(variance=.008, angle=.08, component_id='direct'))
    projection = project_update(projection_input(result))
    retained = retain_transition_goals(_fixed_goals(reference), result, projection)
    assert len(retained) == 2
