from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from probtf.distributions import (
    BinghamOrientation, ConditionalGaussianTranslation, TransformComponent,
    TransformDistribution, TransformDistributionStamped,
)
from probtf.geometry import quat_mul, quat_to_rotmat, rpy_to_quat
from probtf_switch_press_demo.body import EquilibriumBody, ReachMonitor, clearance, motion_samples
from probtf_switch_press_demo.control import BodyCandidates, CommandServo, Controller, GoalLikelihood
from probtf_switch_press_demo.scene import apply, inverse, load_config, transform, write_robot
from probtf_switch_press_demo.vision import PlanarCamera
from probtf_switch_press_demo.beliefs import BODY_FRAMES, body_frame_records, observation_pose_record

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "ros/examples/probtf_switch_press_demo"


@pytest.fixture
def scene(tmp_path):
    config = load_config(PACKAGE / "config/default.yaml")
    urdf = write_robot(ROOT / "ros/examples/symaware_grasp/urdf/simple_six_dof_arm.urdf",
                       tmp_path / "rod.urdf", config)
    return config, urdf


def test_zero_motion_has_no_false_body_uncertainty_and_cache_is_per_body(scene):
    config, urdf = scene
    candidates = BodyCandidates(urdf, config["control"]["bodies"], config["rod"]["length"])
    command = np.array(config["initial_command"])
    poses, _flanges = candidates.current(command)
    assert np.max(np.abs(poses[0] - poses[-1])) > .1
    assert candidates.current(command)[0] is poses
    prediction = candidates.predict(command, np.zeros(6))
    np.testing.assert_array_equal(prediction.tips, np.tile(candidates.tip, (3, 1)))
    for old, new in zip(poses, prediction.poses):
        np.testing.assert_array_equal(old, new)


def test_relative_prediction_pairs_before_and_after_for_each_body(scene):
    config, urdf = scene
    candidates = BodyCandidates(urdf, config["control"]["bodies"], config["rod"]["length"])
    command = np.array(config["initial_command"])
    increment = np.array([.014, -.015, .043, -.007, -.014, 0.])
    predicted = candidates.predict(command, increment)
    for index, setting in enumerate(config["control"]["bodies"]):
        independent = EquilibriumBody(urdf, setting["stiffness"])
        before = independent.equilibrium(command)
        after = independent.equilibrium(command + increment, before)
        expected = apply(inverse(independent.flange(before)) @ independent.flange(after), candidates.tip)
        np.testing.assert_allclose(predicted.tips[index], expected, atol=1e-8)
    assert np.linalg.norm(predicted.tips[0] - predicted.tips[-1]) > .005


def gaussian_record(covariance):
    quaternion = rpy_to_quat(.1, -.2, .3)
    components = tuple(TransformComponent(
        str(index), weight, BinghamOrientation.dirac(quaternion),
        ConditionalGaussianTranslation(np.array(mean), covariance, np.zeros((3, 9))))
        for index, weight, mean in ((0, .3, [.02, .01, .6]), (1, .7, [-.01, .04, .62])))
    return TransformDistributionStamped("camera", "marker", 1., "edge", "test", TransformDistribution(components))


def test_expected_density_matches_analytic_gaussian_mixture_without_kde():
    covariance = np.diag([.002, .003, .004]) ** 2
    record = gaussian_record(covariance)
    hand_camera = transform([.1, -.03, .08], [.2, -.1, .4])
    offset = np.array([.0, -.24, .08])
    likelihood = GoalLikelihood(record, hand_camera, offset, 5, np.random.default_rng(1))
    points = np.array([[.12, .12, .7], [.10, .15, .72], [.09, .13, .71]])
    body_weights = np.array([.25, .5, .25])
    reference_rotation = quat_to_rotmat(record.distribution.components[0].orientation.reference_quaternion_wxyz)
    covariance_hand = hand_camera[:3, :3] @ covariance @ hand_camera[:3, :3].T
    expected_terms = []
    for component in record.distribution.components:
        mean = apply(hand_camera, component.translation.mean_at_reference + reference_rotation @ offset)
        expected_terms.extend(multivariate_normal.logpdf(points, mean, covariance_hand) +
                              np.log(body_weights) + np.log(component.raw_weight))
    assert likelihood.log_expected_density(points, body_weights) == pytest.approx(logsumexp(expected_terms), abs=1e-9)


def test_expected_density_gradient_matches_central_difference():
    likelihood = GoalLikelihood(gaussian_record(np.diag([.002, .003, .004]) ** 2),
                                np.eye(4), np.zeros(3), 5, np.random.default_rng(1))
    tips = np.array([[.018, .012, .601], [-.011, .039, .622]])
    weights = np.array([.4, .6])
    _score, gradient = likelihood.value_and_tip_gradient(tips, weights)
    finite = np.zeros_like(tips)
    epsilon = 1e-7
    for index in range(len(tips)):
        for axis in range(3):
            plus, minus = tips.copy(), tips.copy()
            plus[index, axis] += epsilon
            minus[index, axis] -= epsilon
            finite[index, axis] = (likelihood.log_expected_density(plus, weights)
                                   - likelihood.log_expected_density(minus, weights)) / (2 * epsilon)
    np.testing.assert_allclose(gradient, finite, rtol=2e-6, atol=1e-5)


def test_equilibrium_tip_jacobian_matches_command_perturbations(scene):
    config, urdf = scene
    candidates = BodyCandidates(urdf, config["control"]["bodies"], config["rod"]["length"])
    command = np.array(config["initial_command"])
    increment = np.array([.01, -.01, -.01, .005, .01, .005])
    prediction = candidates.predict(command, increment)
    analytic = candidates.tip_jacobians(command, prediction)
    finite = np.empty_like(analytic)
    epsilon = 5e-4
    for axis in range(6):
        delta = np.eye(6)[axis] * epsilon
        finite[:, :, axis] = (candidates.predict(command, increment + delta).tips
                               - candidates.predict(command, increment - delta).tips) / (2 * epsilon)
    np.testing.assert_allclose(analytic, finite, atol=2e-3, rtol=2e-3)


def test_degenerate_residual_is_rejected_without_adding_width():
    with pytest.raises(np.linalg.LinAlgError):
        GoalLikelihood(gaussian_record(np.zeros((3, 3))), np.eye(4), np.zeros(3), 4, np.random.default_rng(1))


def test_intermediate_collision_is_found_when_both_endpoints_are_clear(scene):
    config, urdf = scene
    body = EquilibriumBody(urdf, config["plant"]["stiffness"])
    middle = np.array(config["initial_command"])
    middle[0] = 0.
    start, end = middle.copy(), middle.copy()
    start[0], end[0] = -.08, .08
    middle_pose = body.equilibrium(middle)
    obstacle_pose = np.eye(4)
    obstacle_pose[:3, 3] = apply(body.flange(middle_pose), [config["rod"]["length"], 0, 0])
    obstacles = [dict(name="test_box", center=[0, 0, 0], size=[.012] * 3)]
    initial = body.equilibrium(start)
    final = body.equilibrium(end)
    assert clearance(body.boxes(initial), obstacle_pose, obstacles)[0] > config["motion"]["clearance"]
    assert clearance(body.boxes(final), obstacle_pose, obstacles)[0] > config["motion"]["clearance"]
    samples = list(motion_samples(body, start, end, initial, config["motion"]))
    gaps = [clearance(body.boxes(pose), obstacle_pose, obstacles)[0] for _command, pose in samples]
    assert min(gaps) < 0
    assert len(samples) > 20


def test_lamp_requires_actual_reach_and_collision_failure_is_permanent():
    monitor = ReachMonitor(np.zeros(3), .009)
    assert not monitor.update([.02, 0, 0], .02, .004)
    assert monitor.update([.008, 0, 0], .02, .004)
    monitor = ReachMonitor(np.zeros(3), .009)
    assert not monitor.update(np.zeros(3), -.001, .004)
    assert monitor.collided
    assert not monitor.update(np.zeros(3), .02, .004)


def test_pixels_to_native_candidates_and_truth_boundary(scene):
    config, urdf = scene
    plant = EquilibriumBody(urdf, config["plant"]["stiffness"])
    pose = plant.equilibrium(config["initial_command"])
    hand = plant.flange(pose)
    marker = inverse(transform(config["base_translation"])) @ transform(config["marker"]["world_translation"], config["marker"]["world_rpy"])
    camera = PlanarCamera(ROOT / "ros/examples/prob_artag_detector/samples/apriltag_36h11_id_021.png",
                         config["marker"]["size"], config["camera"], config["seed"])
    image = camera.render(hand @ transform(config["camera"]["translation"], config["camera"]["rpy"]), marker)
    # Controller receives no actual joints, stiffness, or marker world pose.
    marker_settings = {key: value for key, value in config["marker"].items() if not key.startswith("world_")}
    controller = Controller(urdf, config["control"], config["motion"], marker_settings, config["camera"],
                            config["rod"]["length"], config["obstacles"], config["seed"])
    selected, goal, result, _debug = controller.choose(image, np.array(config["initial_command"]), 1.)
    assert result is not None and selected is not None
    assert selected.ik["method"] == "L-BFGS-B expected-likelihood IK"
    assert selected.ik["iterations"] > 0
    assert selected.ik["trace"][-1]["log_expected_density"] > selected.ik["trace"][0]["log_expected_density"]
    assert abs(sum(result.weights) - 1) < 1e-12
    goal_in_world = apply(hand, goal.mean)
    np.testing.assert_allclose(goal_in_world, apply(marker, config["marker"]["goal_offset"]), atol=.01)
    assert all(np.linalg.eigvalsh(item.component.translation.residual_covariance)[0] > 0
               for item in result.record.distribution.normalized_components())


def test_rod_mass_contributes_to_gravity_and_urdf_inertia(scene, tmp_path):
    config, urdf = scene
    loaded = EquilibriumBody(urdf, config["plant"]["stiffness"])
    original_mass = config["rod"]["mass"]
    config["rod"]["mass"] = .00001
    light_urdf = write_robot(ROOT / "ros/examples/symaware_grasp/urdf/simple_six_dof_arm.urdf",
                             tmp_path / "light.urdf", config)
    light = EquilibriumBody(light_urdf, config["plant"]["stiffness"])
    assert loaded.robot.total_mass - light.robot.total_mass == pytest.approx(original_mass - .00001)
    command = np.array(config["initial_command"])
    assert np.linalg.norm(loaded.robot.tau_gravity(command) - light.robot.tau_gravity(command)) > 1
    assert np.linalg.norm(loaded.equilibrium(command) - light.equilibrium(command)) > .01


def test_servo_preserves_velocity_and_acceleration_limits_across_retargets(scene):
    config, _urdf = scene
    settings = config["motion"]
    servo = CommandServo(settings)
    command = np.zeros(6)
    previous_velocity = servo.velocity.copy()
    dt = settings["frame_period"]
    for index in range(160):
        target = np.full(6, .3 if index < 40 else -.1)
        command = servo.step(command, target, dt)
        assert np.max(np.abs(servo.velocity)) <= settings["max_joint_velocity"] + 1e-12
        assert np.max(np.abs(servo.velocity - previous_velocity)) <= settings["max_joint_acceleration"] * dt + 1e-12
        if index == 40:
            assert np.min(servo.velocity) > 0  # Retargeting does not stop or reverse it instantly.
        previous_velocity = servo.velocity.copy()
    np.testing.assert_allclose(command, target, atol=.002)


def test_spliced_motion_is_checked_in_the_old_observation_frame(scene):
    config, urdf = scene
    candidates = BodyCandidates(urdf, config["control"]["bodies"][1:2], config["rod"]["length"])
    observed = np.array(config["initial_command"])
    observed_hand = candidates.current(observed)[1][0].copy()
    start, middle, end = (observed.copy() for _ in range(3))
    start[0] += .15
    middle[0] += .175
    end[0] += .20
    body = candidates.bodies[0]
    middle_hand = body.flange(body.equilibrium(middle))
    obstacle = np.eye(4)
    obstacle[:3, 3] = apply(inverse(observed_hand) @ middle_hand, candidates.tip)
    boxes = [dict(name="splice_obstacle", center=[0, 0, 0], size=[.008] * 3)]
    safe, gap = candidates.safe_motion(start, end, observed, [obstacle], boxes, config["motion"])
    assert not safe and gap < config["motion"]["clearance"]
    # Incorrectly treating the old observation as a new one moves the obstacle away.
    wrong, _gap = candidates.safe_motion(start, end, start, [obstacle], boxes, config["motion"])
    assert wrong


def test_motion_continues_while_ik_waits_and_old_increment_uses_its_source_command(scene, tmp_path):
    from threading import Event
    from probtf_switch_press_demo.trial import Trial

    config, _urdf = scene
    trial = Trial(config, ROOT / "ros/examples/symaware_grasp/urdf/simple_six_dof_arm.urdf",
                  ROOT / "ros/examples/prob_artag_detector/samples/apriltag_36h11_id_021.png",
                  tmp_path / "continuous", PACKAGE)
    assert trial.guard.bodies[0].robot.data is not trial.controller.candidates.bodies[0].robot.data
    choose = trial.controller.choose
    blocked, moved, done = Event(), Event(), Event()
    state = dict(calls=0, moving_frames=0)

    def delayed_choose(image, command, stamp, on_iteration):
        state["calls"] += 1
        if state["calls"] == 2:
            state["source_command"] = command.copy()
            blocked.set()
            assert moved.wait(timeout=4), "motion loop blocked waiting for IK"
        return choose(image, command, stamp, on_iteration)

    def frame(value):
        if blocked.is_set() and np.max(np.abs(value.command - state["source_command"])) > .001:
            state["moving_frames"] += 1
            if state["moving_frames"] >= 3:
                moved.set()
        if len(value.cycles) >= 2:
            done.set()

    trial.controller.choose = delayed_choose
    trial.run(frame, done.is_set)
    assert moved.is_set()
    second = trial.cycles[1]
    assert second["moving_ticks_during_ik"] >= 3
    assert np.max(np.abs(second["command_at_receipt"] - second["command_before"])) > .001
    expected = second["command_before"] + second["increment"]
    np.testing.assert_allclose(second["reference_hand_world"],
                               trial.world_base @ second["reference_hand_base"], atol=1e-12)
    assert trial.summary["trajectory_frame"] == "base_link"
    tick = next(item for item in trial.servo_ticks if item["plan_cycle"] == 1)
    np.testing.assert_allclose(tick["target"], expected, atol=1e-12)


def test_native_joint_tree_representatives_match_every_visible_robot_frame(scene):
    from probtf.distributions import OrientationKind
    from symaware_grasp.probtf_config import load_prob_tf_mapping

    config, urdf = scene
    body = EquilibriumBody(urdf, config["plant"]["stiffness"])
    pose = body.equilibrium(config["initial_command"])
    mapping = load_prob_tf_mapping(ROOT / "ros/examples/symaware_grasp/configs/simple_six_dof_prob_tf.yaml")
    records = {record.child_frame_id: record for record in
               body_frame_records(mapping, pose, config["rod"]["length"], 1.25)}
    assert set(records) == set(BODY_FRAMES)
    body.robot._fk_update(pose)
    for frame in BODY_FRAMES:
        record = records[frame]
        assert record.stamp == 1.25 and not record.is_static
        assert len(record.distribution.components) == 1
        if frame.startswith("link_"):
            assert record.distribution.components[0].orientation.kind is OrientationKind.FINITE_BINGHAM
        chain, name = [], frame
        while name != "base_link":
            chain.append(records[name])
            name = records[name].parent_frame_id
        actual = np.eye(4)
        for edge in reversed(chain):
            value = np.eye(4)
            value[:3, :3] = quat_to_rotmat(edge.representative.rotation_wxyz)
            value[:3, 3] = edge.representative.translation
            actual = actual @ value
        expected = np.asarray(body.robot.data.oMf[body.robot.get_frame_id(frame)].homogeneous)
        np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_marker_pose_display_exactly_transforms_the_coupled_law_without_helper_frames():
    original = gaussian_record(np.diag([.002, .003, .004]) ** 2)
    marker = GoalLikelihood(original, np.eye(4), np.array([0, -.24, .045]), 5, np.random.default_rng(1)).record
    hand, hand_camera = transform([.6, .1, .4], [.1, .2, -.3]), transform([.02, -.1, .12])
    preview = dict(hand=hand, marker_record=marker, stamp=marker.stamp)
    shown = observation_pose_record(preview, hand_camera, 4.)
    assert shown.parent_frame_id == "base_link" and shown.stamp == 4.
    assert "source_stamp=1.000000000" in shown.provenance.detail
    assert marker.stamp == 1. and marker.parent_frame_id == "camera"
    camera = hand @ hand_camera
    left = rpy_to_quat(.1, .2, -.3)
    for old, new in zip(marker.distribution.components, shown.distribution.components):
        assert old.raw_weight == new.raw_weight
        for q in (old.orientation.reference_quaternion_wxyz, rpy_to_quat(.2, -.4, .1)):
            np.testing.assert_allclose(new.conditional_translation_mean(quat_mul(left, q)),
                                       apply(camera, old.conditional_translation_mean(q)), atol=1e-12)
        np.testing.assert_allclose(new.translation.residual_covariance,
            camera[:3, :3] @ old.translation.residual_covariance @ camera[:3, :3].T, atol=1e-12)


def test_joint_display_samples_form_a_local_cloud_instead_of_three_dirac_points(scene):
    from probtf.probability.sampling import apply_transform_samples, sample_transform_distribution
    from symaware_grasp.probtf_config import load_prob_tf_mapping

    config, _urdf = scene
    mapping = load_prob_tf_mapping(ROOT / "ros/examples/symaware_grasp/configs/simple_six_dof_prob_tf.yaml")
    records = body_frame_records(mapping, np.array(config["initial_command"]), config["rod"]["length"], 1.)
    record = next(item for item in records if item.child_frame_id == "link_2")
    samples = sample_transform_distribution(record.distribution, 100, np.random.default_rng(4))
    points = apply_transform_samples(samples, np.array([.075, 0, 0]))
    assert len(np.unique(np.round(points, 7), axis=0)) > 90
    center = record.representative.translation + quat_to_rotmat(record.representative.rotation_wxyz) @ [.075, 0, 0]
    assert 1e-4 < np.mean(np.linalg.norm(points - center, axis=1)) < .015


def test_motion_exposure_blurs_pixels_and_detector_covariance_uses_the_degraded_image(scene):
    from probtf_switch_press_demo.vision import ImageObservation

    config, urdf = scene
    body = EquilibriumBody(urdf, config["plant"]["stiffness"])
    command = np.array(config["initial_command"])
    first = body.flange(body.equilibrium(command)) @ transform(config["camera"]["translation"], config["camera"]["rpy"])
    command[0] += .018
    last = body.flange(body.equilibrium(command)) @ transform(config["camera"]["translation"], config["camera"]["rpy"])
    marker = inverse(transform(config["base_translation"])) @ transform(config["marker"]["world_translation"], config["marker"]["world_rpy"])
    texture = ROOT / "ros/examples/prob_artag_detector/samples/apriltag_36h11_id_021.png"
    static, moving = [PlanarCamera(texture, config["marker"]["size"], config["camera"], 7) for _ in range(2)]
    period = config["camera"]["exposure_time"]
    still_image = static.render(last, marker, [(0., last), (period, last)])
    moving_image = moving.render(last, marker, [(0., first), (period, last)])
    assert static.diagnostics["motion_displacement_px"] == pytest.approx(0., abs=1e-10)
    assert moving.diagnostics["motion_displacement_px"] > 10
    np.testing.assert_array_equal(static.clean_image, moving.clean_image)
    assert np.mean(np.abs(still_image.astype(float) - moving_image)) > .1
    observation = ImageObservation(config["marker"], config["camera"])
    result, _debug = observation.estimate(moving_image, 1.)
    assert result is not None
    covariance = observation.last_observation.image_covariance
    floor = np.eye(8) * config["marker"]["corner_sigma_px"] ** 2
    assert np.linalg.eigvalsh(covariance - floor)[0] >= -1e-10
    assert np.trace(covariance) > np.trace(floor) + 1e-4
