"""Independent numerical checks of frozen comparison diagnostics."""
from dataclasses import replace

import numpy as np
import pytest

from probtf.geometry import quat_left_matrix
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, bingham_from_rotation_coordinates,
)
from probtf.spherical_law.fusion import Gaussian, PosteriorComponent
from probtf_observation_fusion_demo.model import FrozenBodyBranch
from probtf_observation_fusion_demo.joint_rotation_stiffness import JointRotationDensityRatio
from compare_joint_rotation_factors import (
    IndependentLocalBinghamRatio, bingham_value_score, block_diagonal_rotation_ratio,
    ChartedBodyDensityRatio, HandPoseDensityRatio, physical_local_rotation_errors,
)


def fixture(cross):
    edges = tuple(BinghamTransform('link_'+str(i+1), [.08, 0., 0.],
        bingham_from_rotation_coordinates([1., 0., 0., 0.], np.diag([.008, .009, .012])),
        np.eye(3)*.0001) for i in range(6))
    covariance = np.diag(np.tile([.0001, .0001, .0001, .008, .009, .012], 6))
    covariance[4, 10] = covariance[10, 4] = cross*np.sqrt(covariance[4, 4]*covariance[10, 10])
    source = JointTransformState(edges, covariance, stamp=100., command=(0.,)*6)
    prior = Gaussian(source.mean, covariance)
    mean = prior.mean.copy()
    mean[4], mean[10] = .04, -.025
    child = PosteriorComponent('after', Gaussian(mean, .7*covariance))
    branch = FrozenBodyBranch('before', 1., source, prior, (child,))
    return (branch,), tuple(edge.edge_id for edge in edges)


def prediction():
    value = np.zeros(36)
    value[4], value[10] = .015, -.01
    jacobian = np.zeros((36, 6))
    for i in range(6):
        jacobian[6*i+4, i] = 1.
    return value, jacobian


def quaternions(angles):
    values = [np.array([np.cos(a/2.), 0., np.sin(a/2.), 0.]) for a in angles]
    derivatives = [.5*quat_left_matrix(q)[:, 1:] @ np.eye(3)[:, 1:2] for q in values]
    return values, derivatives


def test_normalized_bingham_density_has_gauge_and_antipodal_invariance_and_correct_score():
    parameter = np.diag([0., -3., -4., -8.])
    q, derivative = quaternions([.31])
    value, score = bingham_value_score(parameter, q[0], derivative[0])
    shifted = bingham_value_score(parameter+37.*np.eye(4), q[0], derivative[0])
    flipped = bingham_value_score(parameter, -q[0], -derivative[0])
    assert shifted[0] == pytest.approx(value, abs=1.e-12)
    np.testing.assert_allclose(shifted[1], score, atol=1.e-12)
    assert flipped[0] == value
    np.testing.assert_array_equal(flipped[1], score)
    def at(angle):
        q, derivative = quaternions([angle])
        return bingham_value_score(parameter, q[0], derivative[0])[0]
    numeric = (at(.31+1.e-6)-at(.31-1.e-6))/(2.e-6)
    assert score[0] == pytest.approx(numeric, rel=1.e-8)


def test_zero_cross_joint_gaussian_is_the_product_of_local_gaussian_ratios():
    branches, ids = fixture(0.)
    ratio = JointRotationDensityRatio(branches, ids)
    value, jacobian = prediction()
    actual = ratio.evaluate(((value, jacobian),))
    prior, children, _ = ratio.marginals[0]
    after = children[0][1]
    rotation = value[ratio.prediction_indices]
    expected = 0.
    for start in range(0, 18, 3):
        part = slice(start, start+3)
        def density(distribution):
            p = distribution.covariance[part, part]
            error = rotation[part]-distribution.mean[part]
            return -.5*(3*np.log(2.*np.pi)+np.linalg.slogdet(p)[1]+error @ np.linalg.solve(p, error))
        expected += density(after)-density(prior)
    assert actual[0] == pytest.approx(expected, abs=1.e-12)
    diagonal = block_diagonal_rotation_ratio(ratio).evaluate(((value, jacobian),))
    assert diagonal[0] == actual[0]
    np.testing.assert_array_equal(diagonal[1], actual[1])


def test_cross_ablation_changes_joint_factor_and_leaves_original_laws_unchanged():
    branches, ids = fixture(.6)
    prior_bytes = branches[0].prior.covariance.tobytes()
    after_bytes = branches[0].posterior_components[0].distribution.covariance.tobytes()
    ratio = JointRotationDensityRatio(branches, ids)
    diagonal = block_diagonal_rotation_ratio(ratio)
    actual, independent = ratio.evaluate((prediction(),)), diagonal.evaluate((prediction(),))
    assert abs(actual[0]-independent[0]) > .01
    assert np.linalg.norm(actual[1]-independent[1]) > .1
    assert branches[0].prior.covariance.tobytes() == prior_bytes
    assert branches[0].posterior_components[0].distribution.covariance.tobytes() == after_bytes
    np.testing.assert_array_equal(np.diag(ratio.marginals[0][0].covariance),
                                  np.diag(diagonal.marginals[0][0].covariance))


def test_independent_bingham_product_cannot_observe_interlink_rotation_cross():
    independent, ids = fixture(0.)
    correlated, _ = fixture(.6)
    first = IndependentLocalBinghamRatio(independent, ids)
    second = IndependentLocalBinghamRatio(correlated, ids)
    q, derivative = quaternions([.015, -.01, 0., 0., 0., 0.])
    a, b = first.evaluate(q, derivative), second.evaluate(q, derivative)
    assert a[0] == pytest.approx(b[0], abs=1.e-12)
    np.testing.assert_allclose(a[1], b[1], atol=1.e-12)


def test_joint_rotation_score_matches_finite_difference_and_ignores_translation_query():
    branches, ids = fixture(.6)
    ratio = JointRotationDensityRatio(branches, ids)
    value, jacobian = prediction()
    baseline = ratio.evaluate(((value, jacobian),))
    numeric = []
    for i in range(6):
        offset = 1.e-6*jacobian[:, i]
        numeric.append((ratio.evaluate(((value+offset, jacobian),))[0]
                       -ratio.evaluate(((value-offset, jacobian),))[0])/(2.e-6))
    np.testing.assert_allclose(baseline[1], numeric, rtol=1.e-8, atol=1.e-8)
    shifted = value.copy()
    shifted.reshape(6, 6)[:, :3] = [100., -200., 300.]
    changed = ratio.evaluate(((shifted, jacobian),))
    assert changed[0] == baseline[0]
    np.testing.assert_array_equal(changed[1], baseline[1])


def test_split_identical_native_components_keep_the_same_independent_bingham_density():
    branches, ids = fixture(.3)
    child = branches[0].posterior_components[0]
    split = replace(branches[0], posterior_components=(
        replace(child, label='first', weight=.25, log_weight=np.log(.25)),
        replace(child, label='second', weight=.75, log_weight=np.log(.75))))
    q, derivative = quaternions([.015, -.01, 0., 0., 0., 0.])
    a = IndependentLocalBinghamRatio(branches, ids).evaluate(q, derivative)
    b = IndependentLocalBinghamRatio((split,), ids).evaluate(q, derivative)
    assert b[0] == pytest.approx(a[0], abs=1.e-12)
    np.testing.assert_allclose(a[1], b[1], atol=1.e-12)


def test_identical_before_after_is_identity_for_all_four_comparison_spaces():
    branches, ids = fixture(.6)
    branch = branches[0]
    branch = replace(branch, posterior_components=(PosteriorComponent('unchanged', branch.prior),))
    for ratio in (JointRotationDensityRatio((branch,), ids), ChartedBodyDensityRatio((branch,), ids)):
        value, gradient = ratio.evaluate((prediction(),))
        assert value == 0.
        np.testing.assert_array_equal(gradient, np.zeros(6))
    q, derivative = quaternions([.1]*6)
    value, gradient = IndependentLocalBinghamRatio((branch,), ids).evaluate(q, derivative)
    assert value == 0.
    np.testing.assert_array_equal(gradient, np.zeros(1))
    hand = HandPoseDensityRatio((branch,), ids, tuple((name, False) for name in ids))
    value, gradient = hand.evaluate_pose(np.array([.2, .1, .3]), np.eye(3),
                                         np.zeros((3, 6)), np.zeros((3, 6)))
    assert value == 0.
    np.testing.assert_array_equal(gradient, np.zeros(6))


def test_physical_rotation_error_uses_URDF_joint_indices_and_wraps_full_turns():
    edges = tuple((name, None, None, index, None)
                  for name, index in (('distal', 2), ('root', 0), ('middle', 1)))
    theta = np.array([.1, 2.*np.pi-.2, -2.*np.pi+.3])
    actual = np.zeros(3)
    before = actual.copy()
    np.testing.assert_allclose(physical_local_rotation_errors(theta, actual, edges),
                              [.3, .1, .2], atol=1.e-14)
    np.testing.assert_array_equal(actual, before)
