"""Real-model regressions for K factors on the joint local-rotation law.

Only declared geometry, estimated body laws and sent commands enter these
tests.  Realized simulator configurations and physical stiffness are excluded.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal

from probtf.geometry import quat_to_rotmat, skew
from probtf.spherical_law.fusion import FusedMixture, Gaussian, PosteriorComponent
from probtf_observation_fusion_demo.body_native import reexpress_state
from probtf_observation_fusion_demo.joint_rotation_stiffness import JointRotationDensityRatio
from probtf_observation_fusion_demo.model import FrozenBodyBranch, Stage, WeightedState
from probtf_observation_fusion_demo.stiffness_identification import (
    FrozenTimeIndexedStiffnessFactor, StiffnessIdentifier,
)
from test_hand_stiffness_identification import hand_model, observed_result


PACKAGE = Path(__file__).resolve().parents[1]


def identifier(model):
    urdf, inference, _, settings = model
    # The shared geometry fixture explicitly preserves the historical hand
    # comparison.  Remove its target override to exercise this constructor's
    # current production default, rather than inheriting that comparison mode.
    settings = {name: value for name, value in settings.items() if name != 'matching_target'}
    return StiffnessIdentifier(urdf, inference.body_ids, settings)


def rotation_indices(state, body_ids):
    return [6*state.edge_ids.index(name)+coordinate
        for name in body_ids for coordinate in (3, 4, 5)]


def result_with_law(result, law, *, label='rotation:test:posterior'):
    source = result.prior
    child = PosteriorComponent(label, law)
    state = (source if np.array_equal(law.mean, source.mean)
        and np.array_equal(law.covariance, source.covariance) else reexpress_state(
            source, law.mean, law.covariance, version=source.version+1, component_id=label))
    prior = Gaussian(source.mean, source.covariance)
    branch = FrozenBodyBranch(source.component_id, 1., source, prior, (child,))
    fields = dict(vars(result))
    fields.update(frozen_body_prior=prior,
        frozen_body_posterior=FusedMixture((child,), (),
            tuple(result.records['observation_ids']), source.version, source.stamp, source.command),
        frozen_body_branches=(branch,), projected_body_numerator=branch,
        stages=(Stage('rotation_estimated_body', (WeightedState(label, 1., state),)),))
    # The rotation model uses actual local link orientations, without any
    # dependency on a hand path or the separately visualized task goal.
    fields.pop('hand_path', None)
    return SimpleNamespace(**fields)


def updated(model, result=None):
    estimate = identifier(model)
    result = observed_result(model) if result is None else result
    if hasattr(result, 'hand_path'):
        del result.hand_path
    record = estimate.update(result)
    assert record['status'] == 'updated_density_factor', record.get('failure_reason')
    return estimate, result, record


def right_jacobian(vector):
    angle = np.linalg.norm(vector)
    if angle == 0.:
        return np.eye(3)
    cross = skew(vector)
    return (np.eye(3)-(1.-np.cos(angle))/angle**2*cross
        +(angle-np.sin(angle))/angle**3*(cross @ cross))


def test_constructor_and_runtime_configuration_select_joint_rotation_without_hand_path(hand_model):
    estimate, result, _ = updated(hand_model)
    assert estimate.matching_target == 'joint_rotation'
    assert yaml.safe_load((PACKAGE/'config/stiffness_identification.yaml').read_text())[
        'matching_target'] == 'joint_rotation'
    assert not hasattr(result, 'hand_path')
    ratio = estimate.factors[0].ratio
    assert isinstance(ratio, JointRotationDensityRatio)
    assert ratio.marginals[0][0].mean.shape == (18,)
    assert ratio.marginals[0][0].covariance.shape == (18, 18)


def test_frozen_joint_rotation_covariance_is_the_canonical_display_law_in_its_source_chart(hand_model):
    estimate, result, record = updated(hand_model)
    ratio = estimate.factors[0].ratio
    index = rotation_indices(result.prior, estimate.body_edge_ids)
    raw = result.projected_body_numerator.posterior_components[0].distribution
    marginal = ratio.marginals[0][1][0][1]
    np.testing.assert_array_equal(marginal.mean, raw.mean[index])
    np.testing.assert_array_equal(marginal.covariance, raw.covariance[np.ix_(index, index)])
    assert marginal.covariance[1, 4] != 0.
    # Verify the full cross blocks in the displayed canonical chart through
    # the independently assembled right-coordinate transport, not by treating
    # its zero coordinate mean as the original uncentered observation chart.
    chart = np.zeros((18, 18))
    displayed = result.stages[-1].components[0].state
    displayed_by_id = {edge.edge_id: edge for edge in displayed.transforms}
    for j, name in enumerate(estimate.body_edge_ids):
        block = slice(3*j, 3*j+3)
        chart[block, block] = right_jacobian(marginal.mean[block])
        reference = result.prior.transforms[result.prior.edge_ids.index(name)]
        actual_rotation = quat_to_rotmat(displayed_by_id[name].rotation_wxyz)
        expected_rotation = (quat_to_rotmat(reference.rotation_wxyz)
            @ Rotation.from_rotvec(marginal.mean[block].copy()).as_matrix())
        np.testing.assert_allclose(actual_rotation, expected_rotation, atol=2e-13)
    np.testing.assert_allclose(chart @ marginal.covariance @ chart.T,
        displayed.covariance[np.ix_(index, index)], rtol=2e-12, atol=2e-13)
    assert record['body_posterior_state_ids'] == [displayed.state_id]


def test_factor_is_the_normalized_joint_gaussian_ratio_not_independent_link_factors(hand_model):
    estimate, result, _ = updated(hand_model)
    factor = estimate.factors[0]
    x = estimate.initial_components[0].distribution.mean
    saved = estimate._branch_anchor
    try:
        estimate._branch_anchor = factor.branch_anchor
        value, _, _, _ = estimate._prediction(x, result.prior)
    finally:
        estimate._branch_anchor = saved
    local_rotation_rows = [6*j+coordinate for j in range(6) for coordinate in (3, 4, 5)]
    rotation = value[local_rotation_rows]
    prior, children, _ = factor.ratio.marginals[0]
    after = children[0][1]
    expected = (multivariate_normal.logpdf(rotation, after.mean, after.covariance)
        -multivariate_normal.logpdf(rotation, prior.mean, prior.covariance))
    actual = estimate._factor_evaluation(factor, x)[0]
    assert actual == pytest.approx(expected, rel=2e-12, abs=2e-11)
    independent = 0.
    for j in range(6):
        block = slice(3*j, 3*j+3)
        independent += multivariate_normal.logpdf(rotation[block], after.mean[block],
            after.covariance[block, block])-multivariate_normal.logpdf(
                rotation[block], prior.mean[block], prior.covariance[block, block])
    assert not np.isclose(actual, independent, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize('offset', [np.zeros(6), np.array([0., -.08, .06, .03, -.04, .02])])
def test_continuous_joint_rotation_factor_score_matches_implicit_equilibrium_finite_difference(hand_model, offset):
    estimate, _, _ = updated(hand_model)
    factor = estimate.factors[0]
    x = estimate.initial_components[0].distribution.mean+offset
    _, score = estimate._factor_evaluation(factor, x)
    numerical = []
    for direction in np.eye(6):
        plus = estimate._factor_evaluation(factor, x+1e-5*direction)[0]
        minus = estimate._factor_evaluation(factor, x-1e-5*direction)[0]
        numerical.append((plus-minus)/2e-5)
    np.testing.assert_allclose(score, numerical, rtol=3e-4, atol=3e-5)
    assert np.linalg.norm(score[1:]) > 1e-7
    assert score[0] == pytest.approx(0., abs=1e-10)


def test_downstream_stiffness_changes_upstream_equilibrium_rotation(hand_model):
    estimate, result, _ = updated(hand_model)
    factor = estimate.factors[0]
    x = estimate.initial_components[0].distribution.mean
    saved = estimate._branch_anchor
    try:
        estimate._branch_anchor = factor.branch_anchor
        value, jacobian, _, _ = estimate._prediction(x, result.prior)
        direction = np.eye(6)[2]
        numerical = (estimate._prediction(x+1e-5*direction, result.prior)[0]
            -estimate._prediction(x-1e-5*direction, result.prior)[0])/2e-5
    finally:
        estimate._branch_anchor = saved
    # Local link two's orientation depends on the third spring parameter
    # through the coupled gravitational equilibrium, despite FK path order.
    upstream_rotation = slice(9, 12)
    assert np.linalg.norm(jacobian[upstream_rotation, 2]) > 1e-5
    np.testing.assert_allclose(jacobian[upstream_rotation, 2],
        numerical[upstream_rotation], rtol=3e-4, atol=3e-6)


def test_joint_rotation_score_keeps_a_near_stop_interior_equilibrium_free(hand_model):
    estimate = identifier(hand_model)
    _, inference, initial, _ = hand_model
    x = np.full(6, np.log(500.))
    theta = np.asarray(initial.command).copy()
    theta[1] = estimate.robot.model.lowerPositionLimit[1]+1e-6
    command = theta+estimate.robot.tau_gravity(theta)/np.exp(x)
    assert np.all(command >= estimate.robot.model.lowerPositionLimit)
    assert np.all(command <= estimate.robot.model.upperPositionLimit)
    source = inference.prior_at(command, 15., 1)
    result = observed_result(hand_model, source, 'rotation:near_stop')
    del result.hand_path
    record = estimate.update(result)
    assert record['update_applied'], record.get('failure_reason')
    factor = estimate.factors[0]
    saved = estimate._branch_anchor
    try:
        estimate._branch_anchor = factor.branch_anchor
        physical = estimate._physical_prediction(x, source)
    finally:
        estimate._branch_anchor = saved
    np.testing.assert_allclose(physical['theta'], theta, atol=2e-10)
    assert not physical['active'][1]
    _, score = estimate._factor_evaluation(factor, x)
    direction = np.eye(6)[1]
    numerical = (estimate._factor_evaluation(factor, x+1e-6*direction)[0]
        -estimate._factor_evaluation(factor, x-1e-6*direction)[0])/2e-6
    assert abs(score[1]) > 1e-7
    assert score[1] == pytest.approx(numerical, rel=3e-4, abs=3e-5)


def test_translation_mean_covariance_and_translation_rotation_cross_do_not_penalize_K(hand_model):
    original = observed_result(hand_model)
    law = original.projected_body_numerator.posterior_components[0].distribution
    mean = law.mean.copy()
    mean[0] += .02
    scales = np.ones(mean.size)
    scales[:3] = 2.
    changed = Gaussian(mean, law.covariance*np.outer(scales, scales))
    modified = result_with_law(original, changed, label='changed_translation_only')
    first, _, _ = updated(hand_model, original)
    second, _, _ = updated(hand_model, modified)
    x = first.initial_components[0].distribution.mean
    before = first._factor_evaluation(first.factors[0], x)
    after = second._factor_evaluation(second.factors[0], x)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert original.stages[-1].components[0].state.state_id != modified.stages[-1].components[0].state.state_id


@pytest.mark.parametrize('translation_changed', [False, True])
def test_unchanged_rotation_law_is_exact_identity_without_any_equilibrium_query(
        hand_model, monkeypatch, translation_changed):
    estimate = identifier(hand_model)
    source = hand_model[2]
    mean, covariance = source.mean.copy(), source.covariance.copy()
    if translation_changed:
        mean[0] += .02
        scale = np.ones(mean.size)
        scale[:3] = 2.
        covariance *= np.outer(scale, scale)
    result = result_with_law(observed_result(hand_model),
        Gaussian(mean, covariance), label='unchanged_rotations')
    def forbidden(*args, **kwargs):
        pytest.fail('An identity rotation factor must not query a physical candidate.')
    monkeypatch.setattr(estimate, '_prediction', forbidden)
    monkeypatch.setattr(estimate, '_physical_prediction', forbidden)
    monkeypatch.setattr(estimate, '_freeze_branch', forbidden)
    record = estimate.update(result)
    assert record['status'] == 'identity_density_ratio'
    assert record['prior'] == record['posterior']
    assert not estimate.factors


def test_history_and_queued_product_preserve_each_acquisition_command_clock_and_factor(hand_model):
    estimate = identifier(hand_model)
    _, inference, initial, _ = hand_model
    old = observed_result(hand_model)
    del old.hand_path
    assert estimate.update(old)['update_applied']
    old_factor = estimate.factors[0]
    x = estimate.initial_components[0].distribution.mean
    old_value = estimate._factor_evaluation(old_factor, x)
    events, specifications, results = [], [], []
    for j, displacement in enumerate((0., .13)):
        command = tuple(np.asarray(initial.command)+[0., displacement, -.5*displacement, 0., 0., 0.])
        source = inference.prior_at(command, 20.+j, j)
        identity = 'rotation:queued:'+str(j)
        result = observed_result(hand_model, source, identity)
        del result.hand_path
        results.append(result)
        events.append(dict(event_id=identity, kind='imu', stamp=source.stamp,
            sequence=j, command=source.command, payload_hash=identity+':raw'))
        specifications.append(dict(observation_id=identity,
            source_stamp=source.stamp, source_command=list(source.command),
            branches=result.frozen_body_branches,
            posterior_branch=result.projected_body_numerator,
            provenance=dict(raw_hash=events[-1]['payload_hash'])))
    terminal = results[-1]
    terminal.records['observation_ids'] = [event['event_id'] for event in events]
    terminal.records['joint_batch'] = dict(batch_id='rotation:queued')
    ledger = dict(kind='actual test command ledger', source_stamp=20., target_stamp=21., steps=[])
    record = estimate.update_batch(terminal, events=events, command_ledger=ledger,
        batch_id='rotation:queued', event_body_ratios=specifications)
    assert record['update_applied'], record.get('failure_reason')
    assert len(estimate.factors) == 2 and estimate.factors[0] is old_factor
    product = estimate.factors[1]
    assert isinstance(product, FrozenTimeIndexedStiffnessFactor)
    assert [part.states[0].stamp for part in product.parts] == [20., 21.]
    assert [part.states[0].command for part in product.parts] == [event['command'] for event in events]
    assert all(isinstance(part.ratio, JointRotationDensityRatio) for part in product.parts)
    actual = estimate._factor_evaluation(product, x)
    contributions = [estimate._factor_evaluation(part, x) for part in product.parts]
    assert actual[0] == pytest.approx(sum(value for value, _ in contributions), abs=1e-12)
    np.testing.assert_allclose(actual[1], sum(score for _, score in contributions), atol=1e-12)
    assert not np.isclose(contributions[0][0], contributions[1][0])
    repeated = estimate._factor_evaluation(old_factor, x)
    assert repeated[0] == old_value[0]
    np.testing.assert_array_equal(repeated[1], old_value[1])
    assert estimate.observation_ids == frozenset(('hand:observation:0', 'rotation:queued:0', 'rotation:queued:1'))
