"""A displayed moment must summarize the law without changing its support."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from probtf.distributions import (
    BinghamOrientation, ConditionalGaussianTranslation, RepresentativeKind,
    TransformComponent, TransformDistribution, TransformDistributionStamped,
)
from probtf.geometry import DeterministicTransform, quat_to_rotmat, rotmat_to_quat
from probtf.candidates import JointFiniteTransformLaw, load_initial_transform_law
from probtf_switch_press_demo.beliefs import BODY_FRAMES, finite_pose_record, left_compose_record
from probtf_switch_press_demo.control import GoalLikelihood
from probtf_switch_press_demo.kinematics import CandidateKinematics
from probtf_switch_press_demo.scene import load_config, transform, write_robot


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "ros/examples/probtf_switch_press_demo"


def support():
    values = np.linspace(-1., 1., 64)
    poses = np.broadcast_to(np.eye(4), (64, 4, 4)).copy()
    poses[:, :3, :3] = Rotation.from_rotvec(np.column_stack(
        (.3 * values, .2 * np.sin(values), .4 * values ** 2))).as_matrix()
    poses[:, :3, 3] = np.column_stack((.3 * values, .1 * values ** 2, .05 * np.sin(values)))
    return poses, tuple("body_{:04d}".format(index) for index in range(64))


def assert_support_unchanged(record, poses, body_ids, weights):
    assert len(record.distribution.components) == len(poses)
    for component, pose, name, weight in zip(record.distribution.components, poses, body_ids, weights):
        assert component.component_id == name
        assert component.raw_weight == weight
        np.testing.assert_array_equal(component.translation.mean_at_reference, pose[:3, 3])
        np.testing.assert_allclose(quat_to_rotmat(component.orientation.reference_quaternion_wxyz),
                                   pose[:3, :3], atol=1e-14, rtol=0.)
        np.testing.assert_array_equal(component.translation.residual_covariance, np.zeros((3, 3)))
        np.testing.assert_array_equal(component.translation.rotation_coupling, np.zeros((3, 9)))


@pytest.mark.parametrize("uniform", [True, False])
def test_moment_uses_weighted_support_and_is_invariant_under_candidate_permutation(uniform):
    poses, body_ids = support()
    weights = np.ones(64) if uniform else np.arange(1., 65.)
    weights /= weights.sum()
    record = finite_pose_record("observation_hand", "tip", 1., poses, body_ids, weights)
    assert record.representative_kind is RepresentativeKind.MOMENT_REPRESENTATIVE
    np.testing.assert_allclose(record.representative.translation,
                               np.average(poses[:, :3, 3], axis=0, weights=weights), atol=1e-14, rtol=0.)
    # SciPy independently computes the chordal rotation mean; its rotation
    # equals the Markley quaternion moment, irrespective of quaternion signs.
    expected_rotation = Rotation.from_matrix(poses[:, :3, :3]).mean(weights=weights).as_matrix()
    np.testing.assert_allclose(quat_to_rotmat(record.representative.rotation_wxyz),
                               expected_rotation, atol=1e-14, rtol=0.)
    assert_support_unchanged(record, poses, body_ids, weights)
    permutation = np.random.default_rng(13).permutation(64)
    permuted = finite_pose_record("observation_hand", "tip", 1., poses[permutation],
        tuple(body_ids[index] for index in permutation), weights[permutation])
    np.testing.assert_allclose(permuted.representative.translation,
                               record.representative.translation, atol=1e-14, rtol=0.)
    np.testing.assert_allclose(quat_to_rotmat(permuted.representative.rotation_wxyz),
                               quat_to_rotmat(record.representative.rotation_wxyz), atol=1e-14, rtol=0.)
    assert_support_unchanged(permuted, poses[permutation],
                            tuple(body_ids[index] for index in permutation), weights[permutation])


def test_moment_commutes_with_a_deterministic_left_coordinate_change():
    poses, body_ids = support()
    weights = np.arange(1., 65.)
    weights /= weights.sum()
    source = finite_pose_record("observation_hand", "tip", 1., poses, body_ids, weights)
    anchor = transform([.6, -.2, .4], [.3, -.4, .6])
    transformed = left_compose_record(source, anchor, "base_link")
    recomputed = finite_pose_record("base_link", "tip", 1., anchor @ poses, body_ids, weights)
    assert transformed.representative_kind is RepresentativeKind.MOMENT_REPRESENTATIVE
    np.testing.assert_allclose(transformed.representative.translation,
                               recomputed.representative.translation, atol=1e-14, rtol=0.)
    np.testing.assert_allclose(quat_to_rotmat(transformed.representative.rotation_wxyz),
                               quat_to_rotmat(recomputed.representative.rotation_wxyz), atol=1e-14, rtol=0.)
    assert_support_unchanged(recomputed, anchor @ poses, body_ids, weights)


def test_expected_likelihood_ignores_representative_metadata():
    poses, body_ids = support()
    weights = np.arange(1., 65.)
    weights /= weights.sum()
    record = finite_pose_record("observation_hand", "tip", 1., poses, body_ids, weights)
    marker = TransformDistributionStamped("camera", "marker", 1., "marker", "test",
        TransformDistribution((TransformComponent("goal", 1., BinghamOrientation.dirac([1., 0., 0., 0.]),
            ConditionalGaussianTranslation(np.array([.01, .02, .03]), np.diag([.02, .03, .04]) ** 2,
                                           np.zeros((3, 9)))),)))
    goal = GoalLikelihood(marker, np.eye(4), np.zeros(3), 3)
    original_score, original_gradient = goal.value_and_tip_gradient(record)
    arbitrary_pose = DeterministicTransform([8., -4., 3.], rotmat_to_quat(poses[0, :3, :3]))
    for changed in (replace(record, representative=arbitrary_pose,
                            representative_kind=RepresentativeKind.PRODUCER_SUPPLIED),
                    replace(record, representative=None, representative_kind=RepresentativeKind.NONE)):
        assert changed.distribution is record.distribution
        score, gradient = goal.value_and_tip_gradient(changed)
        assert score == original_score
        np.testing.assert_array_equal(gradient, original_gradient)


def test_64_transform_candidate_axis_uses_moment_when_first_candidate_is_far_from_it(tmp_path):
    config = load_config(PACKAGE / "config/default.yaml")
    urdf = write_robot(ROOT / "ros/examples/symaware_grasp/urdf/simple_six_dof_arm.urdf",
                       tmp_path / "rod.urdf", config)
    configured = load_initial_transform_law(PACKAGE / "config/initial_transform_law.yaml")
    local, weights = configured.quadrature()
    local = local.copy()
    # A freely configured joint prior may correlate translations and rotations.
    # Its first atom must never be substituted for the law's position moment.
    local[:, 0, 0, 3] += np.linspace(-.4, .4, len(local))
    source = JointFiniteTransformLaw(local, weights, configured.node_ids, BODY_FRAMES,
        "representative_joint_transform_prior", 1, configured.reference_command)
    candidates = CandidateKinematics(urdf, source, config["rod"]["length"], source.reference_command)
    command = np.array([-.0512, -1.2074, 1.2138, .0044, -.8778, .0108])
    candidates.current(command)
    poses = candidates.current_measure.root_transforms[:, -1]
    record = candidates.current_measure.root_records(1.)[-1]
    expected = candidates.weights @ poses[:, :3, 3]
    assert len(record.distribution.components) == 64
    assert np.linalg.norm(poses[0, :3, 3] - expected) > .15
    assert record.representative_kind is RepresentativeKind.MOMENT_REPRESENTATIVE
    np.testing.assert_allclose(record.representative.translation, expected, atol=1e-14, rtol=0.)
    assert_support_unchanged(record, poses, candidates.ids, candidates.weights)
