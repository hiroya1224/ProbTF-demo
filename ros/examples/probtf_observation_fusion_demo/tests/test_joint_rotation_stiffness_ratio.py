"""Joint local-rotation marginal factors preserve correlations and support."""

from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal

from probtf.geometry import quat_to_rotmat, rotation_vector_to_quaternion
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, bingham_from_rotation_coordinates,
)
from probtf.spherical_law.fusion import Gaussian, PosteriorComponent
from probtf_observation_fusion_demo.body_native import materialize_state_in_chart
from probtf_observation_fusion_demo.model import FrozenBodyBranch
from probtf_observation_fusion_demo.joint_rotation_stiffness import JointRotationDensityRatio
from probtf_observation_fusion_demo.stiffness_identification import (
    ChartedBodyDensityRatio, _right_log_derivative, _rotation_chart_volume,
)


BODY = tuple('link_'+str(i) for i in range(6))
ROT = np.asarray([6*i+j for i in range(6) for j in (3, 4, 5)])


def source(sign=1., covariance=None):
    if covariance is None:
        scales = np.array([.02, .025, .03, .07, .08, .09]*7)
        vector = np.sin(np.arange(42)+.4)
        covariance = (np.eye(42)+.04*np.outer(vector, vector))*np.outer(scales, scales)
    transforms = []
    for i in range(7):
        quaternion = sign*rotation_vector_to_quaternion([.03*i, -.02*i, .04])
        transforms.append(BinghamTransform(('link_'+str(i)) if i < 6 else 'marker',
            np.array([.2, .01*i, .03]),
            bingham_from_rotation_coordinates(quaternion, covariance[6*i+3:6*i+6, 6*i+3:6*i+6]),
            covariance[6*i:6*i+3, 6*i:6*i+3]))
    return JointTransformState(tuple(transforms), covariance, version=5, stamp=123.5,
                               command=(.1, -.2, .3, -.1, .2, .05), component_id='estimated-body')


def branch(state, *, before=None, mean=None, covariance=None, label='body', weight=1., log_weight=None):
    before = Gaussian(state.mean, state.covariance) if before is None else before
    mean = .015*np.cos(np.arange(42)+.3) if mean is None else mean
    covariance = .65*state.covariance if covariance is None else covariance
    child = PosteriorComponent(label+':after', Gaussian(mean, covariance), weight,
                               log_weight=log_weight)
    return FrozenBodyBranch(label, weight, state, before, (child,), log_weight)


def prediction():
    value = .03*np.sin(np.arange(36)+.2)
    jacobian = .1*np.cos(np.arange(72).reshape(36, 2)+.7)
    return value, jacobian


def test_six_link_rotational_marginal_retains_every_cross_and_original_chart():
    state = source(); frozen = branch(state)
    ratio = JointRotationDensityRatio((frozen,), BODY)
    prior, children, _ = ratio.marginals[0]
    posterior = children[0][1]
    np.testing.assert_array_equal(prior.mean, frozen.prior.mean[ROT])
    np.testing.assert_array_equal(prior.covariance, frozen.prior.covariance[np.ix_(ROT, ROT)])
    np.testing.assert_array_equal(posterior.mean, frozen.posterior_components[0].distribution.mean[ROT])
    np.testing.assert_array_equal(posterior.covariance,
        frozen.posterior_components[0].distribution.covariance[np.ix_(ROT, ROT)])
    assert prior.mean.shape == (18,) and prior.covariance.shape == (18, 18)
    assert prior.covariance[1, 5] != 0.
    description = ratio.record()
    assert description['factor_domain'] == 'joint_local_rotations'
    assert description['rotation_dimension'] == 18
    assert description['inter_link_rotation_cross_retained']
    assert not description['translation_density_scored']
    assert not description['translation_conditioning_used']
    assert not description['independent_link_product_used']
    assert description['branches'][0]['source_rotation_coordinate_indices'] == ROT.tolist()
    assert description['branches'][0]['source_stamp'] == state.stamp
    assert description['branches'][0]['source_command'] == list(state.command)


def test_full_joint_gaussian_density_and_analytic_score_match_direct_evaluation():
    frozen = branch(source()); ratio = JointRotationDensityRatio((frozen,), BODY)
    value, jacobian = prediction()
    actual, score = ratio.evaluate(((value, jacobian),))
    prior, children, _ = ratio.marginals[0]; posterior = children[0][1]
    expected = (multivariate_normal.logpdf(value[ROT], posterior.mean, posterior.covariance)
        -multivariate_normal.logpdf(value[ROT], prior.mean, prior.covariance))
    assert actual == pytest.approx(expected, abs=1.e-12)
    for i in range(2):
        h = 1.e-6
        plus = ratio.evaluate(((value+h*jacobian[:, i], jacobian),))[0]
        minus = ratio.evaluate(((value-h*jacobian[:, i], jacobian),))[0]
        assert score[i] == pytest.approx((plus-minus)/(2.*h), rel=3.e-7, abs=2.e-8)
    direct = ratio.evaluate(((value[ROT], jacobian[ROT]),))
    assert direct[0] == actual
    np.testing.assert_array_equal(direct[1], score)


def test_translation_mean_width_and_rotation_cross_changes_do_not_condition_rotation_factor():
    state = source(); frozen = branch(state)
    ratio = JointRotationDensityRatio((frozen,), BODY)
    t = np.ones(42); t[np.setdiff1d(np.arange(42), ROT)] = 2.5
    before_covariance = frozen.prior.covariance*np.outer(t, t)
    t[np.setdiff1d(np.arange(42), ROT)] = .3
    after_covariance = frozen.posterior_components[0].distribution.covariance*np.outer(t, t)
    before_mean = frozen.prior.mean.copy(); before_mean[np.setdiff1d(np.arange(42), ROT)] = 200.
    after_mean = frozen.posterior_components[0].distribution.mean.copy()
    after_mean[np.setdiff1d(np.arange(42), ROT)] = -150.
    changed = branch(state, before=Gaussian(before_mean, before_covariance),
                     mean=after_mean, covariance=after_covariance)
    alternate = JointRotationDensityRatio((changed,), BODY)
    value, jacobian = prediction()
    original = ratio.evaluate(((value, jacobian),))
    value2 = value.copy(); value2[np.setdiff1d(np.arange(36), ROT)] = 300.
    jacobian2 = jacobian.copy(); jacobian2[np.setdiff1d(np.arange(36), ROT)] = -900.
    revised = alternate.evaluate(((value2, jacobian2),))
    assert revised[0] == original[0]
    np.testing.assert_array_equal(revised[1], original[1])
    np.testing.assert_array_equal(alternate.marginals[0][0].covariance, ratio.marginals[0][0].covariance)
    # A full-body factor does change, demonstrating that no conditional
    # translation information has silently entered the new marginal.
    full = ChartedBodyDensityRatio((frozen,), BODY)
    changed_full = ChartedBodyDensityRatio((changed,), BODY)
    assert abs(full.evaluate(((value, jacobian),))[0]-changed_full.evaluate(((value, jacobian),))[0]) > 100.


def test_inter_link_rotation_cross_changes_score_and_diagonal_blocks_reduce_to_product():
    state = source(); frozen = branch(state)
    full = JointRotationDensityRatio((frozen,), BODY)
    diagonal = np.diag(np.diag(state.covariance))
    independent_state = source(covariance=diagonal)
    independent = JointRotationDensityRatio((branch(independent_state),), BODY)
    value, jacobian = prediction()
    assert abs(full.evaluate(((value, jacobian),))[0]-independent.evaluate(((value, jacobian),))[0]) > 1.e-4
    prior, children, _ = independent.marginals[0]; posterior = children[0][1]
    expected = sum(multivariate_normal.logpdf(value[ROT][3*i:3*i+3],
        posterior.mean[3*i:3*i+3], posterior.covariance[3*i:3*i+3, 3*i:3*i+3])
        -multivariate_normal.logpdf(value[ROT][3*i:3*i+3], prior.mean[3*i:3*i+3],
            prior.covariance[3*i:3*i+3, 3*i:3*i+3]) for i in range(6))
    assert independent.evaluate(((value, jacobian),))[0] == pytest.approx(expected, abs=2.e-12)


def test_identity_is_exact_when_only_translations_have_been_updated():
    state = source(); mean = state.mean.copy(); mean[0] = .5
    covariance = state.covariance.copy()
    scale = np.ones(42); scale[[0, 1, 2, 6, 7, 8]] = 2.
    covariance *= np.outer(scale, scale)
    ratio = JointRotationDensityRatio((branch(state, mean=mean, covariance=covariance),), BODY)
    assert ratio.identity
    value, jacobian = prediction()
    result, score = ratio.evaluate(((value, jacobian),))
    assert result == 0.; np.testing.assert_array_equal(score, np.zeros(2))
    assert ratio.properness_bounds()['log_factor_lower_bound'] == 0.


def test_native_chart_mixture_and_canonical_numerator_keep_physical_volume_inside_lse():
    first = source(); change = np.zeros(42); change[ROT] = .08*np.sin(np.arange(18)+.3)
    second = materialize_state_in_chart(first, change, 1.3*first.covariance,
                                       component_id='reset-reference')
    keep = branch(first, weight=.7, label='keep')
    reset = branch(second, weight=.3, label='reset')
    canonical = branch(first, label='canonical')
    ratio = JointRotationDensityRatio((keep, reset), BODY, posterior_branch=canonical)
    base_rotations = [quat_to_rotmat(edge.rotation_wxyz)
        @ Rotation.from_rotvec([.12, -.04, .08]).as_matrix() for edge in first.transforms[:6]]
    omega = .06*np.sin(np.arange(36).reshape(6, 3, 2)+.6)
    def query(parameters):
        predictions = []
        for parent in (keep, reset):
            full = np.zeros(36); jacobian = np.zeros((36, 2))
            for i, edge in enumerate(parent.reference_state.transforms[:6]):
                actual = base_rotations[i] @ Rotation.from_rotvec(omega[i] @ parameters).as_matrix()
                delta = Rotation.from_matrix(quat_to_rotmat(edge.rotation_wxyz).T @ actual).as_rotvec()
                full[6*i+3:6*i+6] = delta
                jacobian[6*i+3:6*i+6] = _right_log_derivative(delta) @ omega[i]
            predictions.append((full, jacobian))
        return predictions
    predictions = query(np.zeros(2)); actual, score = ratio.evaluate(predictions)
    minus, plus = [], []
    for parent, (prior, children, _), (value, _) in zip(ratio.branches, ratio.marginals, predictions):
        volume = _rotation_chart_volume(value)[0]
        minus.append(math.log(parent.weight)+multivariate_normal.logpdf(value[ROT], prior.mean, prior.covariance)-volume)
        plus.extend(math.log(child.weight)+multivariate_normal.logpdf(value[ROT], law.mean, law.covariance)-volume
                    for child, law in children)
    assert actual == pytest.approx(logsumexp(plus)-logsumexp(minus), abs=2.e-12)
    assert len(ratio.marginals[0][1]) == 1 and not ratio.marginals[1][1]
    for i in range(2):
        step = np.eye(2)[i]*1.e-6
        numeric = (ratio.evaluate(query(step))[0]-ratio.evaluate(query(-step))[0])/2.e-6
        assert score[i] == pytest.approx(numeric, rel=2.e-7, abs=2.e-8)


def test_quaternion_sign_changes_do_not_change_local_rotation_factor():
    positive, negative = source(sign=1.), source(sign=-1.)
    for a, b in zip(positive.transforms, negative.transforms):
        np.testing.assert_array_equal(a.bingham, b.bingham)
    first = JointRotationDensityRatio((branch(positive),), BODY)
    second = JointRotationDensityRatio((branch(negative),), BODY)
    value, jacobian = prediction()
    a, b = first.evaluate(((value, jacobian),)), second.evaluate(((value, jacobian),))
    assert a[0] == b[0]; np.testing.assert_array_equal(a[1], b[1])


def test_full_cross_coordinate_transport_is_used_before_rotational_marginal_extraction():
    state = source(); mean = np.zeros(42); mean[ROT] = .2*np.sin(np.arange(18))
    transported = materialize_state_in_chart(state, mean, .8*state.covariance)
    ratio = JointRotationDensityRatio((branch(transported),), BODY)
    np.testing.assert_array_equal(ratio.marginals[0][0].covariance,
                                  transported.covariance[np.ix_(ROT, ROT)])
    assert not np.allclose(transported.covariance[np.ix_(ROT, ROT)],
                           .8*state.covariance[np.ix_(ROT, ROT)], rtol=1.e-6, atol=1.e-12)


def test_singular_shared_rotation_support_uses_actual_rank_and_does_not_add_jitter():
    covariance = np.eye(42)*.01
    covariance[np.ix_(ROT, ROT)] = np.tile(np.diag([.01, .02, .03]), (6, 6))
    state = source(covariance=covariance)
    mean = np.zeros(42); mean[ROT] = np.tile([.01, -.02, .03], 6)
    ratio = JointRotationDensityRatio((branch(state, mean=mean, covariance=.7*covariance),), BODY)
    assert ratio.marginals[0][2].rank == 3
    value = np.tile([.02, -.01, .04], 6); jacobian = np.tile(np.eye(3), (6, 1))
    result, score = ratio.evaluate(((value, jacobian),))
    assert np.isfinite(result) and np.all(np.isfinite(score))
    np.testing.assert_array_equal(ratio.marginals[0][0].covariance, covariance[np.ix_(ROT, ROT)])
    bounds = ratio.properness_bounds()
    assert not bounds['verified'] and bounds['global_model_support_proof_required']
    assert bounds['log_factor_lower_bound'] <= result <= bounds['log_factor_upper_bound']
    assert ratio.properness_bounds(model_support_verified=True)['verified']
    bad = value.copy(); bad[0] += .1
    with pytest.raises(ValueError):
        ratio.evaluate(((bad, jacobian),))


def test_rotation_dirac_and_zero_coordinate_identity_have_exact_zero_factor():
    covariance = np.eye(42)*.01
    covariance[np.ix_(ROT, ROT)] = 0.
    reference = source()
    before = Gaussian(reference.mean, covariance)
    same = branch(reference, before=before, mean=reference.mean, covariance=covariance)
    ratio = JointRotationDensityRatio((same,), BODY)
    assert ratio.marginals[0][2].rank == 0 and ratio.identity
    assert ratio.evaluate(((np.zeros(18), np.zeros((18, 2))),))[0] == 0.
    empty = JointRotationDensityRatio((same,), ())
    result, score = empty.evaluate(((np.zeros(0), np.zeros((0, 2))),))
    assert result == 0.; np.testing.assert_array_equal(score, [0., 0.])


def test_underflowed_native_component_retains_log_mass_and_compact_bounds():
    state = source(); parent = branch(state)
    ordinary = parent.posterior_components[0]
    tiny = PosteriorComponent('tiny', Gaussian(-ordinary.distribution.mean,
        .8*state.covariance), 0., log_weight=-1000.)
    parent = replace(parent, posterior_components=(ordinary, tiny))
    ratio = JointRotationDensityRatio((parent,), BODY)
    assert len(ratio.marginals[0][1]) == 2
    assert ratio.record()['branches'][0]['components'][1]['log_weight'] == -1000.
    value, jacobian = prediction(); actual = ratio.evaluate(((value, jacobian),))[0]
    bounds = ratio.properness_bounds()
    assert bounds['verified']
    assert bounds['log_factor_lower_bound'] <= actual <= bounds['log_factor_upper_bound']


def shared_rotation_covariance():
    covariance = np.eye(42)*.01
    covariance[np.ix_(ROT, ROT)] = np.tile(np.diag([.01, .02, .03]), (6, 6))
    return covariance


def test_same_chart_singular_mixture_retains_common_rank_density_and_score():
    covariance = shared_rotation_covariance()
    first = source(covariance=covariance)
    second = source(sign=-1., covariance=covariance)
    before2 = np.zeros(42); before2[ROT] = np.tile([-.01, .02, .005], 6)
    after1 = np.zeros(42); after1[ROT] = np.tile([.01, -.02, .03], 6)
    after2 = before2.copy(); after2[ROT] += np.tile([.015, -.005, .01], 6)
    keep = branch(first, mean=after1, covariance=.7*covariance, weight=.7, label='keep')
    reset = branch(second, before=Gaussian(before2, .8*covariance), mean=after2,
                   covariance=.6*covariance, weight=.3, label='reset')
    ratio = JointRotationDensityRatio((keep, reset), BODY)
    assert ratio.common_rotation_chart and ratio.marginals[0][2].rank == 3
    value = np.tile([.02, -.01, .04], 6); jacobian = np.tile(np.eye(3), (6, 1))
    actual, score = ratio.evaluate(((value, jacobian), (value, jacobian)))
    minus, plus = [], []
    for parent, (prior, children, _) in zip(ratio.branches, ratio.marginals):
        minus.append(math.log(parent.weight)+multivariate_normal.logpdf(value,
            prior.mean, prior.covariance, allow_singular=True))
        plus.extend(math.log(child.weight)+multivariate_normal.logpdf(value,
            distribution.mean, distribution.covariance, allow_singular=True)
            for child, distribution in children)
    assert actual == pytest.approx(logsumexp(plus)-logsumexp(minus), abs=2.e-12)
    for i in range(3):
        step = 1.e-6*jacobian[:, i]
        upper = ratio.evaluate(((value+step, jacobian), (value+step, jacobian)))[0]
        lower = ratio.evaluate(((value-step, jacobian), (value-step, jacobian)))[0]
        assert score[i] == pytest.approx((upper-lower)/2.e-6, rel=3.e-7, abs=2.e-8)
    assert ratio.record()['common_affine_rotation_support_verified']
    bounds = ratio.properness_bounds()
    assert not bounds['verified'] and bounds['conditional_on_model_support']
    assert bounds['log_factor_lower_bound'] <= actual <= bounds['log_factor_upper_bound']


def test_same_chart_singular_mixture_identity_is_exact_including_rotation_dirac():
    for covariance in (shared_rotation_covariance(), np.diag(
            [0. if index in ROT else .01 for index in range(42)])):
        # A reference geometry can describe separate Gaussian proxy laws in
        # the same chart; no Bingham reexpression or support widening occurs.
        reference = source()
        first = branch(reference, before=Gaussian(reference.mean, covariance),
            mean=reference.mean, covariance=covariance, weight=.7, label='same_a')
        second = branch(reference, before=Gaussian(reference.mean, .8*covariance),
            mean=reference.mean, covariance=.8*covariance, weight=.3, label='same_b')
        ratio = JointRotationDensityRatio((first, second), BODY)
        assert ratio.common_rotation_chart and ratio.identity
        value = np.zeros(18); jacobian = np.zeros((18, 2))
        actual, score = ratio.evaluate(((value, jacobian), (value, jacobian)))
        assert actual == 0.; np.testing.assert_array_equal(score, [0., 0.])
        assert ratio.properness_bounds()['verified']


@pytest.mark.parametrize('problem', ['linear_support', 'affine_support'])
def test_same_chart_singular_mixture_rejects_actual_incompatible_supports(problem):
    covariance = shared_rotation_covariance(); reference = source(covariance=covariance)
    first = branch(reference, mean=reference.mean, covariance=covariance, weight=.7, label='first')
    mean2, covariance2 = reference.mean.copy(), covariance.copy()
    if problem == 'linear_support':
        signs = np.ones(42); signs[ROT[3:6]] = -1.
        covariance2 *= np.outer(signs, signs)
    else:
        mean2[ROT[0]] = .1
    second = branch(reference, before=Gaussian(mean2, covariance2), mean=mean2,
                    covariance=covariance2, weight=.3, label='second')
    with pytest.raises(ValueError):
        JointRotationDensityRatio((first, second), BODY)


def test_different_singular_rotation_charts_require_a_real_common_physical_measure():
    covariance = shared_rotation_covariance(); first = source(covariance=covariance)
    shift = np.zeros(42); shift[ROT] = np.tile([.1, -.02, .03], 6)
    second = materialize_state_in_chart(first, shift, covariance,
                                       component_id='different-rotation-reference')
    keep = branch(first, mean=first.mean, covariance=first.covariance, weight=.7, label='keep')
    reset = branch(second, mean=second.mean, covariance=second.covariance, weight=.3, label='reset')
    with pytest.raises(ValueError, match='common physical support measure'):
        JointRotationDensityRatio((keep, reset), BODY)
