"""Shared native transform candidates survive propagation and expected IK."""
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal

from probtf.distributions import (
    BinghamOrientation, ConditionalGaussianTranslation, TransformComponent,
    TransformDistribution, TransformDistributionStamped,
)
from probtf.candidates import JointFiniteTransformLaw
from probtf_switch_press_demo.beliefs import BODY_FRAMES, JointMeasure
from probtf_switch_press_demo.control import GoalLikelihood
from probtf_switch_press_demo.kinematics import CandidateKinematics
from probtf_switch_press_demo.scene import inverse, load_config, write_robot


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "ros/examples/probtf_switch_press_demo"


def initial_source(count, reference_command=(0.,) * 6):
    """An explicit non-Gaussian joint prior, with correlated free SE(3) atoms."""
    coordinate = np.linspace(-1., 1., count)
    local = np.broadcast_to(np.eye(4), (count, len(BODY_FRAMES), 4, 4)).copy()
    for link in range(6):
        rotation_vectors = np.column_stack((.12 * coordinate,
            .07 * np.sin((link + 1) * coordinate), .05 * np.cos(2 * coordinate)))
        local[:, link, :3, :3] = Rotation.from_rotvec(rotation_vectors).as_matrix()
        local[:, link, :3, 3] = np.column_stack((.12 + .004 * (link + 1) * coordinate,
            .003 * np.sin(coordinate + link), .04 + .002 * coordinate**2))
    local[:, -2, 0, 3] = .06
    local[:, -1, 0, 3] = .325
    return JointFiniteTransformLaw(local, np.arange(1, count + 1, dtype=float),
        tuple("candidate_{:02d}".format(index) for index in range(count)), BODY_FRAMES,
        "correlated_local_transform_atoms", 1, tuple(reference_command))


def joint_candidates(count, command):
    source = initial_source(count)
    local = source.initial_local_transforms.copy()
    axes = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0],
                     [1, 0, 0], [0, 1, 0], [1, 0, 0]])
    for link in range(6):
        local[:, link, :3, :3] = local[:, link, :3, :3] @ Rotation.from_rotvec(
            np.asarray(command)[link] * axes[link]).as_matrix()
    return JointMeasure(source.node_ids, source.weights, local, metadata=source.query_metadata())


@pytest.mark.parametrize("count", [16, 64])
def test_all_body_rows_and_weights_survive_root_and_paired_pushforwards(count):
    before = joint_candidates(count, np.zeros(6))
    after = joint_candidates(count, [.02, -.03, .01, -.01, .04, .02])
    root_records = after.root_records(2.)
    relative_records = after.relative_records(before, 2.)
    assert len(root_records) == len(relative_records) == len(BODY_FRAMES)
    np.testing.assert_allclose(after.weights, np.arange(1, count + 1) / sum(range(1, count + 1)))
    for row in range(count):
        before_hand = np.eye(4)
        for local in before.local_transforms[row, :-1]:
            before_hand = before_hand @ local
        product = np.eye(4)
        for link, local in enumerate(after.local_transforms[row]):
            product = product @ local
            np.testing.assert_allclose(after.root_transforms[row, link], product, atol=1e-12)
            for record, expected in ((root_records[link], product),
                                     (relative_records[link], inverse(before_hand) @ product)):
                assert len(record.distribution.components) == count
                component = record.distribution.components[row]
                assert component.component_id == after.body_ids[row]
                assert component.raw_weight == pytest.approx(after.weights[row], abs=1e-16)
                np.testing.assert_allclose(component.translation.mean_at_reference,
                                           expected[:3, 3], atol=1e-12)
    # The entire body prior is preserved, including its final and heaviest row.
    assert root_records[-1].distribution.components[-1].component_id == after.body_ids[-1]
    zero = before.relative_records(before, 1.)[-1]
    np.testing.assert_array_equal(np.array([item.translation.mean_at_reference for item in
                                          zero.distribution.components]),
                                  np.tile([.325, 0, 0], (count, 1)))


def test_expected_likelihood_and_gradient_use_every_one_of_64_weighted_rows():
    before = joint_candidates(64, np.zeros(6))
    after = joint_candidates(64, [.02, -.03, .01, -.01, .04, .02])
    hand = after.relative_records(before, 1.)[-1]
    points = np.array([item.translation.mean_at_reference for item in hand.distribution.components])
    covariance = np.diag([.004, .005, .006]) ** 2
    means = np.array([points[0] + [.001, -.002, .001],
                      points[-1] + [-.001, .001, -.002]])
    mixture_weights = np.array([.13, .87])
    marker = TransformDistributionStamped("camera", "marker", 1., "marker", "test",
        TransformDistribution(tuple(TransformComponent(str(index), weight,
            BinghamOrientation.dirac([1, 0, 0, 0]),
            ConditionalGaussianTranslation(mean, covariance, np.zeros((3, 9))))
            for index, (weight, mean) in enumerate(zip(mixture_weights, means)))))
    goal = GoalLikelihood(marker, np.eye(4), np.zeros(3), 3)
    expected_terms = np.stack([multivariate_normal.logpdf(points, mean, covariance) +
                              np.log(after.weights) + np.log(weight)
                              for weight, mean in zip(mixture_weights, means)], axis=1)
    expected = logsumexp(expected_terms)
    score, gradient = goal.value_and_tip_gradient(hand)
    assert score == pytest.approx(expected, abs=1e-11)
    expected_gradient = -np.sum(np.exp(expected_terms - expected)[:, :, None] *
        ((points[:, None, :] - means[None, :, :]) @ np.linalg.inv(covariance)), axis=1)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-11, atol=1e-10)
    assert np.linalg.norm(gradient[-1]) > .1
    assert abs(score - logsumexp(expected_terms[:3])) > 1.


def test_64_free_native_transform_candidates_keep_joint_rows_through_actuation_and_ik(tmp_path):
    config = load_config(PACKAGE / "config/default.yaml")
    urdf = write_robot(ROOT / "ros/examples/symaware_grasp/urdf/simple_six_dof_arm.urdf",
                       tmp_path / "rod.urdf", config)
    command = np.array(config["initial_command"])
    source = initial_source(64, command)
    candidates = CandidateKinematics(urdf, source, config["rod"]["length"], source.reference_command)
    increment = np.array([.01, -.03, .02, -.005, -.02, .01])
    prediction = candidates.predict(command, increment, 1.)
    assert prediction.joint_before.body_ids == prediction.joint_after.body_ids == candidates.ids
    # The finite input law is exact. IK and publication must retain the same
    # joint atoms, weights and expected-density gradient.
    points = prediction.tips.copy()
    covariance = np.diag([.004, .005, .006]) ** 2
    means = np.array([points[0] + [.001, -.002, .001],
                      points[-1] + [-.001, .001, -.002]])
    mixture_weights = np.array([.13, .87])
    marker = TransformDistributionStamped("camera", "marker", 1., "marker", "test",
        TransformDistribution(tuple(TransformComponent(str(index), weight,
            BinghamOrientation.dirac([1, 0, 0, 0]),
            ConditionalGaussianTranslation(mean, covariance, np.zeros((3, 9))))
            for index, (weight, mean) in enumerate(zip(mixture_weights, means)))))
    goal = GoalLikelihood(marker, np.eye(4), np.zeros(3), 3)
    direct_score, direct_gradient = goal.value_and_prediction_gradient(prediction)
    native_score, native_gradient = goal.value_and_tip_gradient(prediction.hand_record)
    expected_terms = np.stack([multivariate_normal.logpdf(points, mean, covariance) +
                              np.log(prediction.joint_after.weights) + np.log(weight)
                              for weight, mean in zip(mixture_weights, means)], axis=1)
    expected_score = logsumexp(expected_terms)
    expected_gradient = -np.sum(np.exp(expected_terms - expected_score)[:, :, None] *
        ((points[:, None, :] - means[None, :, :]) @ np.linalg.inv(covariance)), axis=1)
    assert direct_score == pytest.approx(native_score, abs=1e-12)
    assert direct_score == pytest.approx(expected_score, abs=1e-11)
    np.testing.assert_allclose(direct_gradient, native_gradient, rtol=1e-12, atol=1e-11)
    np.testing.assert_allclose(direct_gradient, expected_gradient, rtol=1e-11, atol=1e-10)
    assert np.linalg.norm(direct_gradient[-1]) > .1
    assert len(prediction.hand_record.distribution.components) == 64
    assert prediction.joint_after.local_transforms.shape == (64, 8, 4, 4)
    expected_local = source.initial_local_transforms.copy()
    for link, axis in enumerate(candidates.axes):
        expected_local[:, link, :3, :3] = expected_local[:, link, :3, :3] @ Rotation.from_rotvec(
            axis * increment[link]).as_matrix()
    np.testing.assert_allclose(prediction.joint_after.local_transforms, expected_local, atol=1e-14)
    assert prediction.joint_before.expression.law is prediction.joint_after.expression.law
    assert prediction.joint_after.expression.law.source_law is source
    assert prediction.joint_after.metadata["result_kind"] == "joint_finite_pushforward"
    for row in range(64):
        expected = np.eye(4)
        for link, frame in enumerate(BODY_FRAMES):
            expected = expected @ expected_local[row, link]
            np.testing.assert_allclose(prediction.joint_after.root_transforms[row, link], expected, atol=1e-10)
        expected_tip = (inverse(prediction.joint_before.root_transforms[row, -2]) @
                        prediction.joint_after.root_transforms[row, -1])[:3, 3]
        np.testing.assert_allclose(prediction.tips[row], expected_tip, atol=1e-10)
    # Free initial translations survive unchanged; actuation changes rotations
    # about the known axes without reconstructing a stiffness/angle prior.
    np.testing.assert_array_equal(prediction.joint_after.local_transforms[:, :, :3, 3],
                                  source.initial_local_transforms[:, :, :3, 3])
    assert np.ptp(source.initial_local_transforms[:, :6, :3, 3], axis=0).max() > .01
