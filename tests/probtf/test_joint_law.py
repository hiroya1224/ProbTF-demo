"""Initial transform source laws and their exact shared control pushforwards."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal

from probtf.candidates import JointFiniteTransformLaw, IndependentTransformLaw
from probtf.distributions import (
    BinghamOrientation, ConditionalGaussianTranslation, DistributionStatus,
    TransformComponent, TransformDistribution, TransformDistributionStamped,
)
from probtf.joint import JointTransformLaw, JointTransformExpression, PointQuadrature, PointSamples
from probtf.kernels import KernelEvaluator, KernelRepresentation
from probtf.provenance import ApproximationKind


def make_law(revision=1):
    angles = np.array([-.4, .1, .7])
    local = np.broadcast_to(np.eye(4), (3, 2, 4, 4)).copy()
    local[:, 0, :3, :3] = Rotation.from_rotvec(np.column_stack((angles*0., angles*0., angles))).as_matrix()
    local[:, 1, 0, 3] = [1., 1.4, .8]
    source = JointFiniteTransformLaw(local, [.2, .5, .3], ("left", "middle", "right"),
                                     ("joint", "tip"), revision=revision)
    def generate(initial, command):
        result = initial.copy()
        delta = np.eye(4)
        delta[:3, :3] = Rotation.from_rotvec([0., 0., command[0]]).as_matrix()
        result[:, 0] = result[:, 0] @ delta
        return result
    return JointTransformLaw(source, generate, source.frame_names, "known_z_right_action_v1")


def continuous_law():
    component = TransformComponent("native", 1.,
        BinghamOrientation.from_parameter_matrix(np.diag([0., -2., -3., -4.])),
        ConditionalGaussianTranslation(np.zeros(3), np.zeros((3, 3)), np.zeros((3, 9))))
    edge = TransformDistributionStamped("base", "joint", 0., "edge", "test", TransformDistribution((component,)))
    source = IndependentTransformLaw((edge,), True, sample_count=32, seed=3)
    return JointTransformLaw(source, lambda initial, command: initial.copy(), source.frame_names, "identity_control_v1")


def test_complete_joint_tuple_pushforward_preserves_weights_ids_and_correlations():
    law = make_law()
    query = law.at([.2]).root(-1).evaluate().apply([1., 0., 0.])
    angles, lengths = np.array([-.4, .1, .7])+.2, np.array([1., 1.4, .8])+1.
    expected = np.column_stack((lengths*np.cos(angles), lengths*np.sin(angles), np.zeros(3)))
    np.testing.assert_allclose(query.points, expected, atol=1e-15)
    np.testing.assert_array_equal(query.weights, [.2, .5, .3])
    assert query.source_law is law.source_law
    assert isinstance(query, PointQuadrature)
    assert query.approximation.kind is ApproximationKind.EXACT
    assert not query.approximation.lossy
    assert law.at([.2]).evaluate().body_ids == ("left", "middle", "right")
    assert query.metadata["result_kind"] == "joint_finite_pushforward"
    assert query.metadata["source_revision"] == 1
    # This is the configured three-tuple source, not nine independent pairs.
    assert len(query.points) == 3


def test_native_continuous_source_returns_samples_and_rejects_finite_query_type():
    law = continuous_law()
    expression = law.at([0.]).root(0)
    query = expression.samples().apply([1., 0., 0.])
    assert isinstance(query, PointSamples)
    assert query.source_law is law.source_law
    assert query.approximation.kind is ApproximationKind.MONTE_CARLO
    assert query.approximation.lossy
    assert query.metadata["sample_count"] == 32 and query.metadata["seed"] == 3
    assert law.at([1.]).evaluate().body_ids == law.at([0.]).evaluate().body_ids
    np.testing.assert_array_equal(law.at([1.]).evaluate().local_transforms, law.at([0.]).evaluate().local_transforms)
    with pytest.raises(ValueError, match="joint samples"):
        expression.quadrature()
    with pytest.raises(ValueError, match="exact sum"):
        make_law().at([0.]).root(0).samples()
    with pytest.raises(ValueError, match="finite-source"):
        PointQuadrature(query.points, query.weights, query.metadata)


def test_expression_and_finite_query_keep_original_law_snapshot():
    state = make_law().at([.1])
    kernel = state.root(-1)
    evaluator = KernelEvaluator()
    expression = evaluator.apply_to_point(kernel, [1, 0, 0])
    assert expression.status is DistributionStatus.OK
    assert expression.value.kernel is kernel
    query = evaluator.apply_to_point(kernel, [1, 0, 0], KernelRepresentation.QUADRATURE)
    assert query.status is DistributionStatus.OK
    assert isinstance(query.value, PointQuadrature)
    assert query.value.source_law is state.law.source_law


def test_shared_realization_cancels_before_marginalization_including_native_samples():
    for law in (make_law(), continuous_law()):
        state = law.at([.1])
        identity = state.relative(state, 0, reference_index=0).evaluate()
        np.testing.assert_array_equal(identity.transforms, np.tile(np.eye(4), (len(identity.weights), 1, 1)))
    law = make_law()
    before, after = law.at([.1]), law.at([.3])
    relative = after.relative(before, 1, reference_index=0).evaluate()
    independent = np.linalg.inv(before.evaluate().root_transforms[:, 0]) @ after.evaluate().root_transforms[:, 1]
    np.testing.assert_allclose(relative.transforms, independent, atol=1e-15)
    with pytest.raises(ValueError, match="same shared source law snapshot"):
        after.relative(make_law(revision=2).at([.1]), 0)
    with pytest.raises(ValueError, match="same shared source law snapshot"):
        JointTransformExpression(after, 0, make_law().at([.1]))


def test_commands_source_and_evaluation_are_immutable_query_snapshots():
    law = make_law()
    command = np.array([.1])
    state = law.at(command)
    command[0] = 2.
    assert state.command == (.1,)
    with pytest.raises(FrozenInstanceError):
        law.model_id = "other"
    query = state.evaluate()
    with pytest.raises(ValueError):
        query.local_transforms[0] = 0.
    with pytest.raises(TypeError):
        query.metadata["source_revision"] = 99
    with pytest.raises(ValueError):
        law.source_law.initial_local_transforms[0] = 0.
    assert state.evaluate() is query
    assert state.query_id != law.at([.2]).query_id
    assert state.query_id != make_law(revision=2).at([.1]).query_id


def test_full_finite_expected_density_and_spatial_gradient_match_independent_gaussian():
    query = make_law().at([.1]).root(-1).quadrature().apply(np.zeros(3))
    mean, covariance = np.array([1., .2, -.1]), np.diag([.2, .3, .4])**2
    class Density:
        def log_density_and_gradient(self, points):
            return multivariate_normal.logpdf(points, mean, covariance), -(points-mean) @ np.linalg.inv(covariance)
    score, per_point_gradient = query.expected_log_density(Density())
    logpdf = multivariate_normal.logpdf(query.points, mean, covariance)
    expected = logsumexp(np.log([.2, .5, .3])+logpdf)
    assert score == pytest.approx(expected, abs=1e-14)
    for axis in range(3):
        step = np.eye(3)[axis]*1e-6
        plus = logsumexp(np.log(query.weights)+multivariate_normal.logpdf(query.points+step, mean, covariance))
        minus = logsumexp(np.log(query.weights)+multivariate_normal.logpdf(query.points-step, mean, covariance))
        assert per_point_gradient[:, axis].sum() == pytest.approx((plus-minus)/2e-6, abs=2e-9)


def test_declared_relative_map_cancels_shared_prefix_in_both_query_interfaces():
    initial = np.broadcast_to(np.eye(4), (2, 3, 4, 4)).copy()
    initial[:, 0, :3, :3] = Rotation.from_rotvec([[.3, -.2, .1], [-.5, .4, -.7]]).as_matrix()
    initial[:, 0, :3, 3] = [[.1, -.3, .7], [-.4, .8, .2]]
    initial[:, 1, 0, 3] = .4
    initial[:, 2, 0, 3] = .3
    source = JointFiniteTransformLaw(initial, [.25, .75], ("left", "right"), ("first", "hand", "tip"))

    def generate(rows, command):
        local = rows.copy()
        local[:, 0, :3, :3] = local[:, 0, :3, :3] @ Rotation.from_rotvec([0., 0., command[0]]).as_matrix()
        return local

    calls = []
    def root(local):
        output, current = np.empty_like(local), np.broadcast_to(np.eye(4), (len(local), 4, 4))
        for index in range(local.shape[1]):
            current = current @ local[:, index]
            output[:, index] = current
        return output

    def relative(rows, before, after, reference_index, index=None):
        calls.append((rows is source.initial_local_transforms, reference_index, index))
        # All roots share the SAME initial left prefix. Cancel it as a map;
        # this leaves each complete candidate row and its weight untouched.
        reduced = rows.copy()
        reduced[:, 0] = np.eye(4)
        first, second = root(generate(reduced, before)), root(generate(reduced, after))
        all_links = np.linalg.inv(first[:, reference_index])[:, None] @ second
        return all_links if index is None else all_links[:, index]

    law = JointTransformLaw(source, generate, source.frame_names, "right_action_v1",
        relative_generator=relative, relative_generator_id="shared_left_prefix_v1")
    before, after = law.at([.17]), law.at([-.21])
    selected = after.relative(before, -1).evaluate()
    all_links = after.evaluate().relative_transforms(before.evaluate())
    np.testing.assert_array_equal(selected.transforms, all_links[:, -1])
    np.testing.assert_array_equal(selected.transforms[0], selected.transforms[1])
    generic = np.linalg.inv(before.evaluate().root_transforms[:, -2]) @ after.evaluate().root_transforms[:, -1]
    np.testing.assert_allclose(selected.transforms, generic, atol=8e-16)
    assert calls == [(True, 1, 2), (True, 1, None)]
    assert after.evaluate().body_ids == source.node_ids
    np.testing.assert_array_equal(selected.weights, [.25, .75])
    np.testing.assert_array_equal(source.quadrature()[0], initial)
    assert law.descriptor()["relative_generator_id"] == "shared_left_prefix_v1"
    another_version = JointTransformLaw(source, generate, source.frame_names, "right_action_v1",
        relative_generator=relative, relative_generator_id="shared_left_prefix_v2")
    assert after.query_id != another_version.at([-.21]).query_id


def test_relative_hook_keeps_uncertain_attachment_and_prioritizes_exact_identity():
    original = make_law()
    def must_not_run(*args):
        raise AssertionError("exact adjacent cancellation must precede the optional map")
    law = JointTransformLaw(original.source_law, original.generator, original.frame_names, "right_action_v1",
        relative_generator=must_not_run, relative_generator_id="declared_unused_map_v1")
    state = law.at([.1])
    identity = state.relative(state, 0, reference_index=0).evaluate()
    np.testing.assert_array_equal(identity.transforms, np.tile(np.eye(4), (3, 1, 1)))
    adjacent = state.relative(state, 1, reference_index=0).evaluate()
    np.testing.assert_array_equal(adjacent.transforms, original.source_law.quadrature()[0][:, 1])
    assert np.ptp(adjacent.transforms[:, 0, 3]) > .5
    with pytest.raises(ValueError, match="explicit model ID"):
        JointTransformLaw(original.source_law, original.generator, original.frame_names, "right_action_v1",
                          relative_generator=must_not_run)
