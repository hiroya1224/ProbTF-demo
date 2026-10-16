"""Analytic acceptance examples for the explicitly first-order query path."""

import numpy as np
import pytest

from probtf.geometry import (
    axis_angle_to_quat, quat_mul, quat_to_rotmat, skew, rotation_vector_to_quaternion,
)
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState,
    bingham_from_rotation_coordinates, bingham_rotation_coordinates,
    conditional_total_covariance, direction_query, linear_query,
    point_query, pose_query, query_cross_covariance, reexpress_state,
)
from probtf.spherical_law.tangent import induced_vector_moments_tangent


def law(edge_id, translation=(0., 0., 0.), quaternion=(1., 0., 0., 0.), covariance=None):
    covariance = np.diag([0.007, 0.009, 0.012]) if covariance is None else covariance
    return BinghamTransform(edge_id, translation, bingham_from_rotation_coordinates(quaternion, covariance))


def test_direction_proxy_keeps_contraction_and_only_leading_covariance():
    quaternion = axis_angle_to_quat([1., 2., -1.], 0.6)
    covariance = np.array([[.007, .001, 0.], [.001, .011, .002], [0., .002, .014]])
    matrix = bingham_from_rotation_coordinates(quaternion, covariance)
    vector = np.array([.3, -.4, .8])
    r = quat_to_rotmat(quaternion)
    sensitivity = -r @ skew(vector)
    expected_covariance = sensitivity @ covariance @ sensitivity.T
    expected_mean = (1.-np.trace(expected_covariance)/(2.*(vector@vector))) * r @ vector
    result = induced_vector_moments_tangent(vector, matrix, order=1)
    np.testing.assert_allclose(result.mean, expected_mean, atol=1e-13)
    np.testing.assert_allclose(result.cov, expected_covariance, atol=1e-13)
    np.testing.assert_allclose(result.cov @ result.mode, 0., atol=1e-14)
    doubled_concentration = induced_vector_moments_tangent(vector, 2.*matrix, order=1)
    np.testing.assert_allclose(doubled_concentration.cov, .5*result.cov, atol=1e-13)
    np.testing.assert_allclose(r@vector-doubled_concentration.mean, .5*(r@vector-result.mean), atol=1e-13)
    # Existing callers retain the finite-precision/second-order default.
    assert not np.allclose(induced_vector_moments_tangent(vector, matrix).cov, result.cov, rtol=1e-7, atol=0.)


def test_same_rotation_cross_covariance_comes_from_common_bingham_block():
    edge = law("one")
    state = JointTransformState.from_independent([edge])
    a, b = np.array([1., 2., 0.]), np.array([0., 1., 3.])
    qa = direction_query(state, ["one"], a)
    qb = direction_query(state, ["one"], b)
    _, c = bingham_rotation_coordinates(edge.bingham)
    r = quat_to_rotmat(edge.rotation_wxyz)
    np.testing.assert_allclose(query_cross_covariance(state, qa, qb), r@skew(a)@c@skew(b).T@r.T, atol=1e-13)
    np.testing.assert_allclose(qa.covariance, induced_vector_moments_tangent(a, edge.bingham, order=1).cov, atol=1e-13)


def test_independent_two_link_formula_and_global_order():
    q0 = axis_angle_to_quat([0., 0., 1.], .35)
    q1 = axis_angle_to_quat([1., 1., 0.], .2)
    q2 = axis_angle_to_quat([0., 1., 1.], -.4)
    a, b = np.array([.4, .2, -.1]), np.array([.2, -.1, .3])
    transforms = [BinghamTransform("root", np.zeros(3), rotation_wxyz=q0), law("a", quaternion=q1), law("b", a, q2)]
    state = JointTransformState.from_independent(transforms)
    result = point_query(state, ["root", "a", "b"], b)
    r0, r1, r2 = (quat_to_rotmat(q) for q in (q0, q1, q2))
    v = a+r2@b
    m1 = induced_vector_moments_tangent(v, transforms[1].bingham, order=1)
    m2 = induced_vector_moments_tangent(b, transforms[2].bingham, order=1)
    expected = r0 @ (r1@v+(m1.mean-r1@v)+r1@(m2.mean-r2@b))
    expected_covariance = r0 @ (m1.cov+r1@m2.cov@r1.T) @ r0.T
    np.testing.assert_allclose(result.mean, expected, atol=1e-13)
    np.testing.assert_allclose(result.covariance, expected_covariance, atol=1e-13)
    # Scaling the whole P changes only order-one corrections.
    scaled = JointTransformState.from_independent([
        transforms[0], law("a", quaternion=q1, covariance=.5*state.covariance[9:12, 9:12]),
        law("b", a, q2, covariance=.5*state.covariance[15:18, 15:18]),
    ])
    half = point_query(scaled, ["root", "a", "b"], b)
    np.testing.assert_allclose(half.mean-half.mode, .5*(result.mean-result.mode), atol=1e-13)
    np.testing.assert_allclose(half.covariance, .5*result.covariance, atol=1e-13)


def test_conditional_two_terms_include_nonzero_cross_and_rank_deficiency():
    covariance = np.array([[.1, 0., .03], [0., 0., 0.], [.03, 0., .2]])
    jacobian = np.array([[1., 5., -2.], [.3, -4., .7]])
    total, within, between = conditional_total_covariance(covariance, jacobian, split=2)
    b = .03/.1
    expected_within = np.outer(jacobian[:, 2], jacobian[:, 2])*(.2-.03**2/.1)
    combined = jacobian[:, 0]+jacobian[:, 2]*b
    expected_between = .1*np.outer(combined, combined)
    np.testing.assert_allclose(within, expected_within, atol=1e-14)
    np.testing.assert_allclose(between, expected_between, atol=1e-14)
    np.testing.assert_allclose(total, jacobian@covariance@jacobian.T, atol=1e-14)


def test_cross_link_mean_and_rotation_translation_mixed_derivatives():
    transforms = [law("a"), law("b")]
    independent = JointTransformState.from_independent(transforms)
    covariance = independent.covariance.copy()
    c = .001
    covariance[5, 11] = covariance[11, 5] = c
    correlated = JointTransformState(transforms, covariance)
    x = np.array([1., 0., 0.])
    baseline = point_query(independent, ["a", "b"], x)
    result = point_query(correlated, ["a", "b"], x)
    np.testing.assert_allclose(result.mean-baseline.mean, [-c, 0., 0.], atol=1e-13)
    np.testing.assert_allclose(result.covariance, result.jacobian@covariance@result.jacobian.T, atol=1e-13)
    # Correlated translation of b and z rotation of a gives [ez]x ey=-ex.
    second = BinghamTransform("b", np.zeros(3), transforms[1].bingham, .01*np.eye(3))
    independent = JointTransformState.from_independent([transforms[0], second])
    covariance = independent.covariance.copy()
    covariance[5, 7] = covariance[7, 5] = c
    correlated = JointTransformState([transforms[0], second], covariance)
    result = point_query(correlated, ["a", "b"])
    np.testing.assert_allclose(result.mean, [-c, 0., 0.], atol=1e-13)


def test_inverse_shared_variable_cancellation_and_fixed_tangent_query():
    state = JointTransformState.from_independent([law("a", [.2, .1, -.2]), law("b")])
    vector = np.array([.4, -.2, .1])
    cancelled = point_query(state, [("a", True), ("a", False)], vector)
    np.testing.assert_array_equal(cancelled.mean, vector)
    np.testing.assert_array_equal(cancelled.covariance, np.zeros((3, 3)))
    inverse = point_query(state, [("a", True)], vector)
    r = quat_to_rotmat(state.transforms[0].rotation_wxyz)
    np.testing.assert_allclose(inverse.mode, r.T@(vector-state.transforms[0].translation), atol=1e-13)
    imu = direction_query(state, [("a", True)], [0., 0., -1.])
    intrinsic = linear_query(imu, np.eye(3)[:2])
    assert intrinsic.covariance.shape == (2, 2)
    np.testing.assert_allclose(intrinsic.within+intrinsic.between, intrinsic.covariance, atol=1e-13)


def test_pose_query_and_bingham_joint_reexpression_keep_cross_blocks():
    transforms = [law("a", [.2, 0., 0.]), law("b", [.3, .1, 0.])]
    prior = JointTransformState.from_independent(transforms, stamp=12.3, command=(.1, .2), version=7)
    covariance = prior.covariance.copy()
    covariance[4, 10] = covariance[10, 4] = .001
    mu = np.zeros(12)
    mu[3:6] = [.02, -.01, .03]
    mu[9:12] = [-.01, .02, -.03]
    posterior = reexpress_state(prior, mu, covariance)
    assert posterior.version == 8 and posterior.stamp == 12.3 and posterior.command == (.1, .2)
    assert np.linalg.norm(posterior.covariance[:6, 6:]) > 0.
    for i, edge in enumerate(posterior.transforms):
        q, recovered = bingham_rotation_coordinates(edge.bingham)
        expected_q = quat_mul(transforms[i].rotation_wxyz, rotation_vector_to_quaternion(mu[6*i+3:6*i+6]))
        assert abs(q@expected_q) > 1.-1e-13
        np.testing.assert_allclose(recovered, posterior.covariance[6*i+3:6*i+6, 6*i+3:6*i+6], atol=1e-13)
    relative = pose_query(posterior, [("a", True), ("b", False)])
    assert relative.mean.shape == (6,) and relative.rotation_wxyz.shape == (4,)
    np.testing.assert_allclose(relative.covariance, relative.jacobian@posterior.covariance@relative.jacobian.T, atol=1e-13)
    independentized = posterior.covariance.copy()
    independentized[:6, 6:] = independentized[6:, :6] = 0.
    lost_dependency = JointTransformState(posterior.transforms, independentized)
    assert np.linalg.norm(point_query(posterior, ["a", "b"], [.1, 0., 0.]).covariance-point_query(lost_dependency, ["a", "b"], [.1, 0., 0.]).covariance) > 1e-8


def test_outside_concentration_domain_and_different_state_queries_are_rejected():
    with pytest.raises(ValueError, match="concentration"):
        law("too_broad", covariance=.2*np.eye(3))
    state = JointTransformState.from_independent([law("a")], version=4, stamp=1.2, command=(.3,))
    a = point_query(state, ["a"], [.1, 0., 0.])
    other = JointTransformState(state.transforms, state.covariance, version=4, stamp=1.2, command=(.3,), component_id="different")
    b = point_query(other, ["a"], [.1, 0., 0.])
    with pytest.raises(ValueError, match="same joint state"):
        query_cross_covariance(state, a, b)


def test_inverse_rotation_translation_and_nonadjacent_shared_rotation():
    edge = law("a")
    edge = BinghamTransform("a", np.zeros(3), edge.bingham, .01*np.eye(3))
    prior = JointTransformState.from_independent([edge])
    covariance = prior.covariance.copy()
    c = .001
    covariance[5, 1] = covariance[1, 5] = c
    correlated = JointTransformState([edge], covariance)
    x = np.array([1., 0., 0.])
    base = point_query(prior, [("a", True)], x)
    inverse = point_query(correlated, [("a", True)], x)
    np.testing.assert_allclose(inverse.mean-base.mean, [-c, 0., 0.], atol=1e-13)
    s = quat_to_rotmat(axis_angle_to_quat([1., 1., 0.], .3))
    fixed = BinghamTransform("s", np.zeros(3), rotation_wxyz=axis_angle_to_quat([1., 1., 0.], .3))
    state = JointTransformState.from_independent([law("a"), fixed])
    output = point_query(state, ["a", "s", "a"], x)
    rotcov = state.covariance[3:6, 3:6]
    independent_mean = (induced_vector_moments_tangent(s@x, state.transforms[0].bingham, order=1).mean
                        + s@(induced_vector_moments_tangent(x, state.transforms[0].bingham, order=1).mean-x))
    cross_mean = sum(rotcov[i, j]*skew(ei)@s@skew(ej)@x
                     for i, ei in enumerate(np.eye(3)) for j, ej in enumerate(np.eye(3)))
    np.testing.assert_allclose(output.mean, independent_mean+cross_mean, atol=1e-13)
    expected_j = -skew(s@x)-s@skew(x)
    np.testing.assert_allclose(output.covariance, expected_j@rotcov@expected_j.T, atol=1e-13)
    shifted = np.zeros(12)
    shifted[6] = .1  # Known attachment is outside the random support.
    with pytest.raises(ValueError, match="outside"):
        reexpress_state(state, shifted, state.covariance)
