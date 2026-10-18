"""Endpoint stiffness factors retain propagated pose laws and their full cross."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from probtf.geometry import rotation_vector_to_quaternion, quat_to_rotmat
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, bingham_from_rotation_coordinates,
)
from probtf.spherical_law.fusion import Gaussian, PosteriorComponent
from probtf_observation_fusion_demo.body_native import materialize_state_in_chart, pose_query
from probtf_observation_fusion_demo.model import FrozenBodyBranch
from probtf_observation_fusion_demo.hand_stiffness import HandPoseDensityRatio


def state():
    scale = np.array([.012, .015, .013, .08, .06, .07]*2)
    vector = np.sin(np.arange(12)+.4)
    covariance = np.zeros((18, 18))
    covariance[:12, :12] = (np.eye(12)+.15*np.outer(vector, vector))*np.outer(scale, scale)
    transforms = []
    for i in range(2):
        quaternion = rotation_vector_to_quaternion([.1+.2*i, -.3+.1*i, .2])
        transforms.append(BinghamTransform('link_'+str(i), np.array([.2, .04*i, .1]),
            bingham_from_rotation_coordinates(quaternion, covariance[6*i+3:6*i+6, 6*i+3:6*i+6]),
            covariance[6*i:6*i+3, 6*i:6*i+3]))
    transforms.append(BinghamTransform('tool', np.array([.13, 0., .02]),
        rotation_wxyz=rotation_vector_to_quaternion([.2, 0., 0.])))
    return JointTransformState(tuple(transforms), covariance, version=9, stamp=1.25,
        command=(.2, -.3), component_id='full-cross-estimated-law')


def branch(reference, mean=None, covariance=None, *, weight=1., label='branch', log_weight=None):
    mean = reference.mean if mean is None else mean
    covariance = reference.covariance if covariance is None else covariance
    child = PosteriorComponent(label+':observed', Gaussian(mean, covariance), weight,
        log_weight=log_weight)
    return FrozenBodyBranch(label, weight, reference,
        Gaussian(reference.mean, reference.covariance), (child,), log_weight=log_weight)


def ratio_for(reference, mean=None, covariance=None):
    return HandPoseDensityRatio((branch(reference, mean, covariance),), ('link_0', 'link_1'),
                                ('link_0', 'link_1', 'tool'))


def test_prior_and_posterior_use_existing_pose_query_with_complete_cross():
    source = state()
    mean = np.zeros(18); mean[:12] = .02*np.sin(np.arange(12))
    covariance = .7*source.covariance
    ratio = ratio_for(source, mean, covariance)
    expected_prior = pose_query(source, source.edge_ids)
    expected_state = materialize_state_in_chart(source, mean, covariance)
    expected_posterior = pose_query(expected_state, source.edge_ids)
    np.testing.assert_array_equal(ratio.marginals[0][0].mean, expected_prior.mean)
    np.testing.assert_array_equal(ratio.marginals[0][0].covariance, expected_prior.covariance)
    np.testing.assert_allclose(ratio.marginals[0][1][0][1].mean, expected_posterior.mean, atol=1.e-14)
    np.testing.assert_allclose(ratio.marginals[0][1][0][1].covariance, expected_posterior.covariance, atol=1.e-14)
    assert np.linalg.norm(expected_prior.covariance[:3, 3:]) > 1.e-6
    record = ratio.record()
    assert record['hand_dimension'] == 6 and not record['intermediate_link_density_scored']
    assert not record['physical_truth_received'] and not record['monte_carlo_used']
    assert record['branches'][0]['prior']['source_stamp'] == source.stamp
    assert record['branches'][0]['prior']['source_command'] == list(source.command)


def test_pose_score_matches_translation_and_noncommuting_rotation_derivatives():
    source = state()
    mean = np.zeros(18); mean[:12] = .035*np.cos(np.arange(12)+.2)
    ratio = ratio_for(source, mean, .65*source.covariance)
    p = np.array([.31, -.06, .21])
    rotation = Rotation.from_rotvec([.15, -.4, .5]).as_matrix()
    dp = np.array([[.12, -.04], [.05, .02], [-.08, .03]])
    omega = np.array([[.2, -.1], [.13, .08], [-.17, .04]])
    value, score = ratio.evaluate_pose(p, rotation, dp, omega)
    assert np.isfinite(value)
    for i in range(2):
        step = 1.e-6
        plus = ratio.evaluate_pose(p+step*dp[:, i],
            rotation @ Rotation.from_rotvec(step*omega[:, i]).as_matrix(), dp, omega)[0]
        minus = ratio.evaluate_pose(p-step*dp[:, i],
            rotation @ Rotation.from_rotvec(-step*omega[:, i]).as_matrix(), dp, omega)[0]
        assert score[i] == pytest.approx((plus-minus)/(2.*step), rel=2.e-7, abs=3.e-7)


def test_keep_reset_own_charts_and_shared_canonical_numerator_are_retained():
    source = state()
    reset_mean = np.zeros(18); reset_mean[:12] = .08*np.sin(np.arange(12)+.7)
    reset = materialize_state_in_chart(source, reset_mean, 1.4*source.covariance,
                                     component_id='reset-reference')
    keep = branch(source, covariance=.7*source.covariance, weight=.7, label='keep')
    reset_branch = branch(reset, covariance=.8*reset.covariance, weight=.3, label='reset')
    canonical_mean = np.zeros(18); canonical_mean[:12] = .04*np.cos(np.arange(12))
    canonical = branch(source, canonical_mean, .6*source.covariance, label='canonical')
    ratio = HandPoseDensityRatio((keep, reset_branch), ('link_0', 'link_1'), source.edge_ids,
                                posterior_branch=canonical)
    assert len(ratio.marginals[0][1]) == 1 and not ratio.marginals[1][1]
    assert not np.allclose(ratio.pose_charts[0], ratio.pose_charts[1])
    p = np.array([.32, .04, .21]); rotation = Rotation.from_rotvec([.2, -.2, .15]).as_matrix()
    actual = ratio.evaluate_pose(p, rotation, np.eye(3), np.eye(3))[0]
    def density(query, distribution):
        delta = Rotation.from_matrix(quat_to_rotmat(query.rotation_wxyz).T @ rotation).as_rotvec()
        angle = np.linalg.norm(delta)
        volume = 2.*np.log(np.sin(angle/2.)/(angle/2.))
        return multivariate_normal.logpdf(np.r_[p, delta], distribution.mean, distribution.covariance)-volume
    denominator = logsumexp([np.log(item.weight)+density(query, marginal[0])
        for item, query, marginal in zip(ratio.branches, ratio.prior_queries, ratio.marginals)])
    numerator = density(ratio.child_queries[0][0], ratio.marginals[0][1][0][1])
    assert actual == pytest.approx(numerator-denominator, rel=1.e-12, abs=2.e-12)


def test_unchanged_law_is_exact_identity_without_model_reconstruction():
    source = state(); ratio = ratio_for(source)
    assert ratio.identity
    query = ratio.prior_queries[0]
    value, score = ratio.evaluate_pose(query.mean[:3], quat_to_rotmat(query.rotation_wxyz),
                                     np.eye(3), np.eye(3))
    assert value == 0.; np.testing.assert_array_equal(score, np.zeros(3))
    assert ratio.properness_bounds(2.)['log_factor_upper_bound'] == 0.
    assert ratio.prior_queries[0].state_id == source.state_id


def test_distinct_intermediate_widths_with_same_endpoint_law_have_zero_pose_ratio():
    covariance = np.eye(12)*.01
    transforms = tuple(BinghamTransform('link_'+str(i), np.zeros(3),
        bingham_from_rotation_coordinates(np.array([1., 0., 0., 0.]), covariance[6*i+3:6*i+6, 6*i+3:6*i+6]),
        covariance[6*i:6*i+3, 6*i:6*i+3]) for i in range(2))
    source = JointTransformState(transforms, covariance, stamp=2., command=(.1, .2))
    changed = np.diag([.005]*6+[.015]*6)
    ratio = HandPoseDensityRatio((branch(source, covariance=changed),), source.edge_ids, source.edge_ids)
    np.testing.assert_allclose(ratio.marginals[0][0].covariance, ratio.marginals[0][1][0][1].covariance,
                               rtol=1.e-12, atol=1.e-14)
    value, score = ratio.evaluate_pose(np.array([.03, -.01, .04]),
        Rotation.from_rotvec([.1, -.04, .03]).as_matrix(), np.eye(3), np.eye(3))
    assert value == pytest.approx(0., abs=3.e-12)
    np.testing.assert_allclose(score, 0., atol=3.e-12)
    # A full-body score would instead penalize these harmless width changes.
    assert abs(multivariate_normal.logpdf(np.zeros(12), cov=changed)
        -multivariate_normal.logpdf(np.zeros(12), cov=covariance)) > .5


def test_removing_joint_cross_changes_endpoint_factor():
    source = state(); mean = np.zeros(18); mean[:12] = .02
    full = ratio_for(source, mean, .7*source.covariance)
    diagonal = np.diag(np.diag(source.covariance))
    independent = ratio_for(materialize_state_in_chart(source, source.mean, diagonal), mean, .7*diagonal)
    inputs = (np.array([.4, .1, .3]), Rotation.from_rotvec([.3, -.1, .2]).as_matrix(), np.eye(3), np.eye(3))
    assert abs(full.evaluate_pose(*inputs)[0]-independent.evaluate_pose(*inputs)[0]) > .01


def test_compact_hand_bounds_include_finite_pose_ratio_queries():
    source = state(); mean = np.zeros(18); mean[:12] = .02
    ratio = ratio_for(source, mean, .7*source.covariance)
    bounds = ratio.properness_bounds(2.)
    for p, angle in (([0., 0., 0.], [0., 0., 0.]), ([1., .2, -.5], [.2, -.1, .4]),
                     ([-.7, .4, .2], [2., .5, -.4])):
        value = ratio.evaluate_pose(p, Rotation.from_rotvec(angle).as_matrix(), np.eye(3), np.eye(3))[0]
        assert bounds['log_factor_lower_bound'] <= value <= bounds['log_factor_upper_bound']
    assert bounds['verified'] and not bounds['intermediate_link_density_scored']


def test_underflowed_mixture_branch_keeps_its_native_log_mass():
    source = state()
    mean = np.zeros(18); mean[:12] = .02
    common = branch(source, mean, .7*source.covariance)
    first = common.posterior_components[0]
    children = (PosteriorComponent('ordinary', first.distribution, 1., log_weight=0.),
                PosteriorComponent('underflow', Gaussian(-mean, .8*source.covariance), 0., log_weight=-1000.))
    parent = FrozenBodyBranch('mixture', 1., source, common.prior, children)
    ratio = HandPoseDensityRatio((parent,), ('link_0', 'link_1'), source.edge_ids)
    assert len(ratio.child_queries[0]) == 2
    assert ratio.record()['branches'][0]['components'][1]['log_weight'] == -1000.


@pytest.mark.parametrize('problem', ['nan', 'dimension'])
def test_actual_nonfinite_and_dimension_errors_are_not_replaced_by_zero_information(problem):
    ratio = ratio_for(state(), covariance=.7*state().covariance)
    p = np.array([0., 0., np.nan]) if problem == 'nan' else np.zeros(2)
    with pytest.raises(ValueError):
        ratio.evaluate_pose(p, np.eye(3), np.eye(3), np.eye(3))
