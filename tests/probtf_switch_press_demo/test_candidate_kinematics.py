"""The configured joint transform law is the sole input to IK kinematics."""
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from probtf.candidates import JointFiniteTransformLaw
from probtf_switch_press_demo.beliefs import BODY_FRAMES
from probtf_switch_press_demo.body import clearance
from probtf_switch_press_demo.kinematics import CandidateKinematics
from probtf_switch_press_demo.scene import inverse, transform, write_robot
from symaware_grasp.grasp_collision import BottleCollisionModel


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = np.array([-.15, -1.85, 2.35, 0., -1.25, 0.])


@pytest.fixture
def arm(tmp_path):
    with open(ROOT / "ros/examples/probtf_switch_press_demo/config/default.yaml") as handle:
        config = yaml.safe_load(handle)
    urdf = write_robot(ROOT / "ros/examples/symaware_grasp/urdf/simple_six_dof_arm.urdf",
                       tmp_path / "configured_arm.urdf", config)
    geometry = BottleCollisionModel(urdf)
    frames = geometry.transforms(REFERENCE, .185)
    local, previous = [], np.eye(4)
    for frame in BODY_FRAMES:
        local.append(inverse(previous) @ frames[frame])
        previous = frames[frame]
    return urdf, config, geometry, np.asarray(local)


def make_source(local, varied=True, vary_tip=True):
    rows = np.broadcast_to(local, (2, len(BODY_FRAMES), 4, 4)).copy()
    if varied:
        # These are arbitrary SE(3) link candidates, including transverse
        # translations and rotations about axes different from the actuator.
        rows[0, 0] = rows[0, 0] @ transform([.004, -.003, .002], [.03, -.02, .04])
        rows[0, 2] = rows[0, 2] @ transform([-.003, .002, .001], [-.01, .04, -.02])
        rows[1, 1] = rows[1, 1] @ transform([.002, .004, -.003], [.04, .01, -.03])
        rows[1, 5] = rows[1, 5] @ transform([.001, -.002, .003], [-.02, .03, .01])
        if vary_tip:
            rows[1, -1] = rows[1, -1] @ transform([.004, -.003, .002], [.01, -.02, .03])
    return JointFiniteTransformLaw(rows, [.35, .65], ("body_A", "body_B"), BODY_FRAMES,
                                   law_id="configured_body", reference_command=tuple(REFERENCE))


def cumulative(local):
    result = np.empty_like(local)
    for sample, row in enumerate(local):
        root = np.eye(4)
        for index, edge in enumerate(row):
            root = root @ edge
            result[sample, index] = root
    return result


def test_actuation_preserves_arbitrary_initial_transforms_and_joint_candidate_pairing(arm):
    urdf, config, _geometry, local = arm
    source = make_source(local)
    initial = source.initial_local_transforms.copy()
    kinematics = CandidateKinematics(urdf, source, config["rod"]["length"], REFERENCE)
    at_reference = kinematics.current(REFERENCE)
    np.testing.assert_allclose(at_reference.local_transforms, initial, rtol=0., atol=0.)
    command = REFERENCE + [.08, -.07, .06, -.05, .04, -.03]
    expected = initial.copy()
    for index, axis in enumerate(kinematics.axes):
        action = np.eye(4)
        action[:3, :3] = Rotation.from_rotvec(axis*(command[index]-REFERENCE[index])).as_matrix()
        expected[:, index] = initial[:, index] @ action
    actual = kinematics.current(command)
    np.testing.assert_allclose(actual.local_transforms, expected, rtol=0., atol=3e-16)
    np.testing.assert_allclose(actual.root_transforms, cumulative(expected), rtol=0., atol=3e-16)
    np.testing.assert_array_equal(source.initial_local_transforms, initial)
    np.testing.assert_array_equal(actual.local_transforms[:, -2:], initial[:, -2:])
    np.testing.assert_array_equal(actual.weights, source.weights)
    assert actual.body_ids == source.node_ids
    assert kinematics.law.source_law is source
    assert kinematics.current(command) is actual
    # Cross-combining marginal edge candidates would introduce a third body,
    # which this two-row source never contained.
    mixed = expected[0].copy()
    mixed[1] = expected[1, 1]
    mixed_tip = cumulative(mixed[None])[0, -1]
    assert all(np.linalg.norm(mixed_tip - tip) > 1e-4 for tip in actual.root_transforms[:, -1])


def test_paired_relative_action_and_full_actuator_jacobians_match_finite_difference(arm):
    urdf, config, _geometry, local = arm
    source = make_source(local)
    kinematics = CandidateKinematics(urdf, source, config["rod"]["length"], REFERENCE)
    command = REFERENCE + [.04, -.03, .02, .01, -.02, .03]
    increment = np.array([.009, -.008, .007, -.006, .005, -.004])
    prediction = kinematics.predict(command, increment, stamp=12.5)
    references = np.array([inverse(pose) for pose in prediction.joint_before.root_transforms[:, -2]])
    expected_relative = references[:, None] @ prediction.joint_after.root_transforms
    np.testing.assert_allclose(prediction.relative_transforms, expected_relative, rtol=0., atol=1e-14)
    actual = kinematics.tip_jacobians(command, prediction)
    step = 2e-7
    finite_difference = np.empty_like(actual)
    for axis in range(6):
        delta = np.zeros(6)
        delta[axis] = step
        plus = kinematics.predict(command, increment+delta).tips
        minus = kinematics.predict(command, increment-delta).tips
        finite_difference[:, :, axis] = (plus-minus)/(2*step)
    np.testing.assert_allclose(actual, finite_difference, rtol=0., atol=2e-9)
    assert prediction.stamp == 12.5


def test_algebraic_prefix_cancellation_matches_every_direct_relative_frame(arm):
    urdf, config, _geometry, local = arm
    source = make_source(local)
    kinematics = CandidateKinematics(urdf, source, config["rod"]["length"], REFERENCE)
    before_command = REFERENCE + [.03, -.02, .04, -.05, .06, -.07]
    after_command = REFERENCE + [-.04, .05, -.06, .07, -.03, .02]
    before = cumulative(kinematics.generate(source.initial_local_transforms, before_command))
    after = cumulative(kinematics.generate(source.initial_local_transforms, after_command))
    for reference in range(len(BODY_FRAMES)):
        actual = kinematics.relative_generate(source.initial_local_transforms, before_command,
                                              after_command, reference)
        expected = np.linalg.inv(before[:, reference])[:, None] @ after
        np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-14)
        for index in range(len(BODY_FRAMES)):
            np.testing.assert_array_equal(kinematics.relative_generate(source.initial_local_transforms,
                before_command, after_command, reference, index), actual[:, index])


def test_first_initial_transform_cancels_exactly_without_coalescing_source_rows(arm):
    from probtf.candidates import load_initial_transform_law

    urdf, config, _geometry, local = arm
    source = make_source(local)
    kinematics = CandidateKinematics(urdf, source, config["rod"]["length"], REFERENCE)
    changed = source.initial_local_transforms.copy()
    changed[0, 0] = transform([.31, -.22, .43], [.17, -.29, .36]) @ changed[0, 0]
    changed[1, 0] = transform([-.24, .13, -.32], [-.26, .38, -.41]) @ changed[1, 0]
    before = REFERENCE + [.04, -.03, .02, -.01, .05, -.06]
    after = REFERENCE + [-.02, .01, -.04, .03, -.06, .05]
    for reference in range(len(BODY_FRAMES)):
        original = kinematics.relative_generate(source.initial_local_transforms, before, after, reference)
        moved = kinematics.relative_generate(changed, before, after, reference)
        np.testing.assert_array_equal(original, moved)
    default = load_initial_transform_law(ROOT / "ros/examples/probtf_switch_press_demo/config/initial_transform_law.yaml")
    default_kinematics = CandidateKinematics(urdf, default, config["rod"]["length"], REFERENCE)
    prediction = default_kinematics.predict(before, after-before)
    assert prediction.joint_after.body_ids == default.node_ids
    assert len(default.node_ids) == 64
    # The first edge's two possibilities differ, but their paired inverse
    # action cancels that edge exactly. All64 weighted rows are retained.
    np.testing.assert_array_equal(prediction.relative_transforms[:32], prediction.relative_transforms[32:])
    assert np.unique(prediction.tips, axis=0).shape[0] == 32
    expression_points = default_kinematics.law.at(before+prediction.increment).relative(
        default_kinematics.law.at(before), -1).evaluate().apply(np.zeros(3))
    np.testing.assert_array_equal(expression_points.points, prediction.tips)


def test_zero_command_cancels_the_same_hand_but_preserves_uncertain_tool_tip(arm):
    urdf, config, _geometry, local = arm
    source = make_source(local, vary_tip=True)
    kinematics = CandidateKinematics(urdf, source, config["rod"]["length"], REFERENCE)
    prediction = kinematics.predict(REFERENCE, np.zeros(6))
    assert prediction.joint_after is prediction.joint_before
    np.testing.assert_array_equal(prediction.relative_transforms[:, -2], np.broadcast_to(np.eye(4), (2, 4, 4)))
    np.testing.assert_array_equal(prediction.relative_transforms[:, -1], source.initial_local_transforms[:, -1])
    assert not np.array_equal(prediction.tips[0], prediction.tips[1])
    assert len(BODY_FRAMES)-1 not in kinematics.law.fixed_local_indices
    fixed_source = make_source(local, vary_tip=False)
    fixed_kinematics = CandidateKinematics(urdf, fixed_source, config["rod"]["length"], REFERENCE)
    assert len(BODY_FRAMES)-1 in fixed_kinematics.law.fixed_local_indices


def test_expected_goal_density_gradient_uses_every_paired_candidate(arm):
    urdf, config, _geometry, local = arm
    source = make_source(local)
    kinematics = CandidateKinematics(urdf, source, config["rod"]["length"], REFERENCE)
    increment = np.array([.009, -.008, .007, -.006, .005, -.004])
    prediction = kinematics.predict(REFERENCE, increment)

    class GoalDensity:
        def log_density_and_gradient(self, points):
            residual = points-np.array([config["rod"]["length"]+.01, -.008, .006])
            variance = np.array([.02, .018, .025])**2
            return -.5*np.sum(residual**2/variance, axis=1), -residual/variance

    before = kinematics.law.at(REFERENCE)

    def score(delta):
        expression = kinematics.law.at(REFERENCE+delta).relative(before, -1)
        points = expression.evaluate().apply(np.zeros(3))
        assert points.source_law is source
        return points.expected_log_density(GoalDensity())

    _value, weighted_spatial_gradient = score(increment)
    analytic = np.einsum("si,sij->j", weighted_spatial_gradient,
                         kinematics.tip_jacobians(REFERENCE, prediction))
    step, numerical = 2e-7, np.zeros(6)
    for axis in range(6):
        delta = np.zeros(6)
        delta[axis] = step
        numerical[axis] = (score(increment+delta)[0]-score(increment-delta)[0])/(2*step)
    np.testing.assert_allclose(analytic, numerical, rtol=0., atol=2e-6)


def test_nominal_candidate_collision_geometry_matches_independent_urdf_kinematics(arm):
    urdf, config, geometry, local = arm
    source = make_source(local, varied=False)
    kinematics = CandidateKinematics(urdf, source, config["rod"]["length"], REFERENCE)
    command = REFERENCE + [.2, -.1, .12, -.13, .14, -.15]
    frames = geometry.transforms(command, .185)
    references = np.array([transform([.3, -.2, .1], [.2, -.1, .4]), np.eye(4)])
    actual = kinematics.boxes(command, references)
    assert len(actual) == 2*len(geometry.bodies)
    for sample in range(2):
        for index, (link, _name, local_collision, half) in enumerate(geometry.bodies):
            name, pose, actual_half = actual[sample*len(geometry.bodies)+index]
            assert name == source.node_ids[sample]+":"+link
            np.testing.assert_allclose(pose, references[sample] @ frames[link] @ local_collision,
                                       rtol=0., atol=1e-14)
            np.testing.assert_array_equal(actual_half, half)


def test_collision_guard_checks_all_candidate_rows_and_intermediate_motion(arm):
    urdf, config, _geometry, local = arm
    source = make_source(local, varied=False)
    kinematics = CandidateKinematics(urdf, source, config["rod"]["length"], REFERENCE)
    start, middle, end = [REFERENCE + [angle, 0, 0, 0, 0, 0] for angle in (-.06, 0., .06)]
    before = kinematics.current(REFERENCE)
    reference = np.array([inverse(pose) for pose in before.root_transforms[:, -2]])
    middle_tip = kinematics.current(middle).root_transforms[0, -1, :3, 3]
    center = (reference[0] @ np.r_[middle_tip, 1.])[:3]
    obstacles = [dict(name="mid_path_cube", center=center.tolist(), size=[.003]*3)]
    # Endpoint-only checking misses this collision with the middle of a swing.
    endpoint_gaps = [clearance(kinematics.boxes(command, reference), np.eye(4), obstacles)[0]
                     for command in (start, end)]
    assert min(endpoint_gaps) > .008
    settings = dict(clearance=.004, spatial_resolution=.002, joint_resolution=.2)
    safe, gap = kinematics.safe_motion(start, end, REFERENCE, [np.eye(4)], obstacles, settings)
    assert not safe and gap < settings["clearance"]
    # One complete candidate may collide while the other remains distant.
    shifted = source.initial_local_transforms.copy()
    shifted[1, 0, 1, 3] += .3
    separated_source = JointFiniteTransformLaw(shifted, [.99, .01], source.node_ids, BODY_FRAMES)
    separated = CandidateKinematics(urdf, separated_source, config["rod"]["length"], REFERENCE)
    moved = REFERENCE + [.1, 0, 0, 0, 0, 0]
    refposes = np.array([inverse(pose) for pose in separated.current(REFERENCE).root_transforms[:, -2]])
    candidate_boxes = separated.boxes(moved, refposes)
    per_candidate = len(candidate_boxes)//2
    # Use the known base geometry: its observation coordinates differ with
    # each candidate's uncertain upstream translation.
    body_B_base = candidate_boxes[per_candidate]
    obstacle = [dict(name="candidate_B_base", center=body_B_base[1][:3, 3].tolist(), size=[.002]*3)]
    assert clearance(candidate_boxes[:per_candidate], np.eye(4), obstacle)[0] > .05
    assert clearance(candidate_boxes[per_candidate:], np.eye(4), obstacle)[0] < 0.
    safe, _gap = separated.safe_motion(moved, moved, REFERENCE, [np.eye(4)], obstacle, settings)
    assert not safe


def test_initial_transform_reference_and_known_command_limits_are_validated(arm):
    from dataclasses import replace
    from probtf.candidates import load_initial_transform_law

    urdf, config, _geometry, local = arm
    source = make_source(local)
    with pytest.raises(ValueError, match="reference commands"):
        CandidateKinematics(urdf, source, config["rod"]["length"], REFERENCE+.01)
    kinematics = CandidateKinematics(urdf, source, config["rod"]["length"], REFERENCE)
    with pytest.raises(ValueError, match="joint limits"):
        kinematics.current(kinematics.upper+.1)
    with pytest.raises(ValueError, match="six finite"):
        kinematics.predict(REFERENCE, [np.nan]*6)
    native = load_initial_transform_law(ROOT / "ros/examples/probtf_switch_press_demo/config/initial_transform_law.yaml")
    wrong_root = replace(native, edge_records=(replace(native.edge_records[0], parent_frame_id="world"),)
                         + native.edge_records[1:])
    with pytest.raises(ValueError, match="rooted at base_link"):
        CandidateKinematics(urdf, wrong_root, config["rod"]["length"], REFERENCE)


def test_zero_mass_rows_do_not_create_collision_constraints_or_false_attachment_variation(arm):
    urdf, config, geometry, local = arm
    varied = make_source(local)
    source = JointFiniteTransformLaw(varied.initial_local_transforms, [1., 0.], varied.node_ids, BODY_FRAMES)
    kinematics = CandidateKinematics(urdf, source, config["rod"]["length"], REFERENCE)
    assert len(BODY_FRAMES)-1 in kinematics.law.fixed_local_indices
    assert len(kinematics.boxes(kinematics.current(REFERENCE))) == len(geometry.bodies)
