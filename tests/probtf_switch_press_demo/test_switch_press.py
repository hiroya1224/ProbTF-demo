from pathlib import Path
from dataclasses import replace
import json

import cv2
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from probtf.bingham import bingham_log_normalizer
from probtf.distributions import (
    BinghamOrientation, ConditionalGaussianTranslation, TransformComponent,
    TransformDistribution, TransformDistributionStamped, compose_with_deterministic_right,
)
from probtf.geometry import DeterministicTransform, quat_mul, quat_to_rotmat, rpy_to_quat
from probtf_switch_press_demo.body import EquilibriumBody, ReachMonitor, clearance, motion_samples
from probtf_switch_press_demo.control import CommandServo, Controller, GeometryGuard, GoalLikelihood, Prediction
from probtf_switch_press_demo.model import BODY_FRAMES, COMPARISON_METHOD, Model
from probtf_switch_press_demo.scene import apply, inverse, load_config, transform, write_robot
from probtf_switch_press_demo.vision import PlanarCamera

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "ros/examples/probtf_switch_press_demo"


@pytest.fixture
def scene(tmp_path):
    config = load_config(PACKAGE / "config/default.yaml")
    urdf = write_robot(ROOT / "ros/examples/symaware_grasp/urdf/simple_six_dof_arm.urdf",
                       tmp_path / "rod.urdf", config)
    return config, urdf


@pytest.fixture
def model(scene):
    config, urdf = scene
    return Model.from_config(urdf, rod_length=config["rod"]["length"],
        hand_camera=transform(config["camera"]["translation"], config["camera"]["rpy"]))


def test_same_command_retains_configured_candidate_uncertainty_without_plant_hypotheses(scene, model):
    config, _urdf = scene
    command = np.array(config["initial_command"])
    first = model.hand(command, stamp=1.25)
    same = model.hand(command + np.zeros(6), stamp=1.25)
    np.testing.assert_array_equal(same.mean, first.mean)
    np.testing.assert_array_equal(same.covariance, first.covariance)
    np.testing.assert_array_equal(same.orientation_parameter, first.orientation_parameter)
    assert np.linalg.eigvalsh(first.covariance)[0] > 0
    assert "stiffness" not in json.dumps(model.descriptor)
    assert first.metadata["command"] == tuple(command)


def test_command_changes_the_frozen_configured_family_before_full_hand_projection(scene, model):
    config, _urdf = scene
    command = np.array(config["initial_command"])
    increment = np.array([.014, -.015, .043, -.007, -.014, 0.])
    before, after = model.hand(command), model.hand(command + increment)
    assert np.linalg.norm(before.mean - after.mean) > .005
    assert np.linalg.norm(before.covariance - after.covariance) > 1e-6
    assert np.linalg.norm(before.orientation_moment - after.orientation_moment) > .001
    assert before.metadata["source_fingerprint"] == after.metadata["source_fingerprint"]
    assert before.metadata["revision"] == after.metadata["revision"]
    np.testing.assert_array_equal(command, config["initial_command"])


def gaussian_record(covariance):
    quaternion = rpy_to_quat(.1, -.2, .3)
    components = tuple(TransformComponent(
        str(index), weight, BinghamOrientation.dirac(quaternion),
        ConditionalGaussianTranslation(np.array(mean), covariance, np.zeros((3, 9))))
        for index, weight, mean in ((0, .3, [.02, .01, .6]), (1, .7, [-.01, .04, .62])))
    return TransformDistributionStamped("camera_optical_frame", "marker", 1., "edge", "test", TransformDistribution(components))


def test_goal_likelihood_uses_common_gaussian_and_all_bingham_overlaps(scene, model):
    config, _urdf = scene
    covariance = np.diag([.002, .003, .004]) ** 2
    record = gaussian_record(covariance)
    command = np.array(config["initial_command"])
    likelihood = GoalLikelihood(model, record, command, [.0, -.24, .08], rpy_to_quat(0., .4, 0.))
    hand = model.hand(command + [.01, -.01, .02, 0., 0., 0.], stamp=record.stamp)
    goal = likelihood.summary
    score, diagnostics = likelihood.objective.evaluate(hand)
    expected_position = multivariate_normal.logpdf(hand.mean, goal.mean, hand.covariance + goal.covariance)
    expected_orientation = logsumexp([
        np.log(weight) + bingham_log_normalizer(hand.orientation_parameter + parameter, 480)
        - bingham_log_normalizer(hand.orientation_parameter, 480) - bingham_log_normalizer(parameter, 480)
        for weight, parameter in zip(goal.weights, goal.orientation_parameters)])
    assert score == pytest.approx(expected_position + expected_orientation, abs=2e-8)
    assert diagnostics["log_position_overlap"] == pytest.approx(expected_position, abs=1e-10)
    assert likelihood.objective.goal is goal
    np.testing.assert_array_equal(goal.weights, [.3, .7])
    for component in likelihood.record.distribution.components:
        np.testing.assert_array_equal(component.translation.mean_at_reference, goal.mean)
        np.testing.assert_array_equal(component.translation.residual_covariance, goal.covariance)


def test_goal_is_fixed_while_full_hand_overlap_gradient_changes_command(scene, model):
    config, _urdf = scene
    observed = np.array(config["initial_command"])
    likelihood = GoalLikelihood(model, gaussian_record(np.diag([.002, .003, .004]) ** 2),
                                observed, [.0, -.24, .045], rpy_to_quat(0., .4, 0.))
    frozen_mean, frozen_covariance = likelihood.summary.mean.copy(), likelihood.summary.covariance.copy()
    command = observed + [.01, -.01, .02, .003, -.005, 0.]
    gradient = likelihood.objective.finite_difference_gradient(model.hand, command, step=2e-5)
    finite = np.empty(6)
    for axis in range(6):
        delta = np.eye(6)[axis] * 1e-5
        finite[axis] = (likelihood.objective.evaluate(model.hand(command + delta))[0]
                        - likelihood.objective.evaluate(model.hand(command - delta))[0]) / 2e-5
    np.testing.assert_allclose(gradient, finite, rtol=2e-3, atol=2e-3)
    np.testing.assert_array_equal(likelihood.summary.mean, frozen_mean)
    np.testing.assert_array_equal(likelihood.summary.covariance, frozen_covariance)
    assert likelihood.summary.metadata["observation_command"] == tuple(observed)


def test_equilibrium_tip_jacobian_matches_command_perturbations(scene):
    config, urdf = scene
    body = EquilibriumBody(urdf, config["plant"]["stiffness"])
    command = np.array(config["initial_command"])
    pose = body.equilibrium(command)
    analytic = body.tip_command_jacobian(pose, np.eye(3))
    finite = np.empty_like(analytic)
    epsilon = 5e-4
    for axis in range(6):
        delta = np.eye(6)[axis] * epsilon
        first, second = body.equilibrium(command + delta, pose), body.equilibrium(command - delta, pose)
        finite[:, axis] = (apply(body.flange(first), [config["rod"]["length"], 0, 0])
                          - apply(body.flange(second), [config["rod"]["length"], 0, 0])) / (2 * epsilon)
    np.testing.assert_allclose(analytic, finite, atol=2e-3, rtol=2e-3)


def test_degenerate_comparison_covariance_is_rejected_without_adding_width(scene, model):
    from probtf_switch_press_demo.objective import PoseOverlapObjective
    config, _urdf = scene
    command = np.array(config["initial_command"])
    likelihood = GoalLikelihood(model, gaussian_record(np.zeros((3, 3))), command,
                                np.zeros(3), [1., 0., 0., 0.])
    # Exact leaf translation remains valid; propagated camera uncertainty is
    # retained. No artificial epsilon is needed to construct the candidate law.
    assert np.linalg.eigvalsh(likelihood.summary.covariance)[0] > 0
    target = replace(likelihood.summary, covariance=np.zeros((3, 3)))
    hand = replace(model.hand(command), covariance=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="positive definite"):
        PoseOverlapObjective(target).evaluate(hand)


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
    updates = []
    command = np.array(config["initial_command"])
    selected, goal, result, _debug = controller.choose(image, command, 1.,
        lambda prediction, frozen_goal, native_result, _debug, _iteration:
            updates.append((prediction, frozen_goal, native_result)))
    assert result is not None and selected is not None
    assert selected.ik["method"] == "SLSQP Gaussian-Bingham expected-likelihood IK"
    assert selected.ik["iterations"] > 0
    assert selected.ik["trace"]
    assert selected.score == pytest.approx(goal.objective.evaluate(selected.hand)[0], abs=1e-12)
    initial_hand = controller.model.hand(command, stamp=1.)
    assert updates
    initial_distance = np.linalg.norm(initial_hand.mean - goal.summary.mean)
    trust_limit = min(config["control"]["max_joint_step"],
                      max(.1 * config["control"]["max_joint_step"], .5 * initial_distance))
    assert selected.ik["joint_trust_limit"] == pytest.approx(trust_limit, abs=1e-12)
    observed_camera = controller.model.nominal_roots(command)["camera_optical_frame"]
    half = config["marker"]["size"] / 2
    corners = np.array([[-half, half, 0.], [half, half, 0.], [half, -half, 0.], [-half, -half, 0.]])
    camera_matrix = camera.camera.camera_matrix
    image_size = np.array([config["camera"]["width"], config["camera"]["height"]])
    # Check every safe update that can reach the servo, as well as the final
    # selection. OpenCV projection independently checks the declared image
    # margin/recovery constraint; these are nominal candidates, not a
    # Gaussian-support bound.
    for prediction, frozen_goal, native_result in updates + [(selected, goal, result)]:
        assert frozen_goal is goal and native_result is result
        np.testing.assert_allclose(prediction.command, command + prediction.increment, atol=1e-14)
        assert np.max(np.abs(prediction.increment)) <= trust_limit + 1e-10
        assert np.linalg.norm(prediction.hand.mean - goal.summary.mean)**2 <= (.95 * initial_distance)**2 + 1e-8
        future_camera = controller.model.nominal_roots(prediction.command)["camera_optical_frame"]
        for branch in result.record.distribution.normalized_components():
            component = branch.component
            quaternion = component.orientation.reference_quaternion_wxyz
            marker_pose = np.eye(4)
            marker_pose[:3, :3] = quat_to_rotmat(quaternion)
            marker_pose[:3, 3] = component.conditional_translation_mean(quaternion)
            camera_marker = inverse(future_camera) @ observed_camera @ marker_pose
            source_pixels = cv2.projectPoints(corners, cv2.Rodrigues(marker_pose[:3, :3])[0],
                marker_pose[:3, 3], camera_matrix, np.zeros(5))[0].reshape(-1, 2)
            source_normalized = source_pixels / image_size
            lower_margin = np.minimum(.15, source_normalized + .02)
            upper_margin = np.minimum(.15, 1 - source_normalized + .02)
            depths = apply(camera_marker, corners)[:, 2]
            assert np.min(depths) >= .02 - 1e-8
            pixels = cv2.projectPoints(corners, cv2.Rodrigues(camera_marker[:3, :3])[0],
                camera_marker[:3, 3], camera_matrix, np.zeros(5))[0].reshape(-1, 2)
            assert np.all(pixels >= lower_margin * image_size - 1e-5)
            assert np.all(pixels <= (1 - upper_margin) * image_size + 1e-5)
    assert abs(sum(result.weights) - 1) < 1e-12
    direct = controller.model.target(np.array(config["initial_command"]), result.record,
        config["marker"]["goal_offset"], config["marker"]["goal_orientation_wxyz"])
    np.testing.assert_array_equal(goal.summary.mean, direct.mean)
    np.testing.assert_array_equal(goal.summary.covariance, direct.covariance)
    assert goal.summary.metadata["frame_id"] == "base_link"
    assert "stiffness" not in json.dumps(controller.model.descriptor)
    assert not hasattr(controller, "plant")
    assert not hasattr(controller, "candidates")
    assert all(np.linalg.eigvalsh(item.component.translation.residual_covariance)[0] > 0
               for item in result.record.distribution.normalized_components())


def test_controller_rejects_stalled_command_and_requires_two_point_view_recovery(scene, monkeypatch):
    from types import SimpleNamespace
    import probtf_switch_press_demo.control as control_module

    config, urdf = scene
    marker_settings = {key: value for key, value in config["marker"].items() if not key.startswith("world_")}
    controller = Controller(urdf, config["control"], config["motion"], marker_settings, config["camera"],
                            config["rod"]["length"], config["obstacles"], config["seed"])
    command = np.array(config["initial_command"])
    # At depth .6m this tag has a top corner at y=36px=5% of image
    # height. Recovering to 7% is required; demanding 15% at once is wrong.
    component = TransformComponent("visible_near_top", 1., BinghamOrientation.dirac([1., 0., 0., 0.]),
        ConditionalGaussianTranslation([0., -.18, .6], np.eye(3) * 1e-6, np.zeros((3, 9))))
    sensor = TransformDistributionStamped("camera_optical_frame", "marker", 1., "test_image", "detector",
        TransformDistribution((component,)))
    result = SimpleNamespace(record=sensor)
    controller.observation.estimate = lambda image, stamp: (result, np.zeros((8, 8, 3), dtype=np.uint8))
    controller.propose = lambda q, mean: [np.zeros(6)]
    checked = {}

    def zero_optimizer(function, seed, *, callback, constraints, **_kwargs):
        margins = constraints[0]["fun"](np.zeros(6))
        checked["progress_margin"] = margins[0]
        checked["view_margin"] = np.min(margins[1:])
        callback(np.zeros(6))
        return SimpleNamespace(x=np.zeros(6), fun=function(np.zeros(6)), success=True,
                               message="controlled zero command", nit=1, nfev=1)

    monkeypatch.setattr(control_module, "minimize", zero_optimizer)
    updates = []
    selected, goal, native, _debug = controller.choose(np.zeros((8, 8), dtype=np.uint8), command, 1.,
        lambda *update: updates.append(update))
    assert selected is None and native is result and goal is not None
    source_mean = controller.model.position_moments(command)[0]
    distance_squared = np.sum((source_mean - goal.mean)**2)
    assert distance_squared > .01
    assert checked["progress_margin"] == pytest.approx((.95**2 - 1.) * distance_squared, abs=1e-12)
    assert checked["view_margin"] == pytest.approx(-.02, abs=1e-12)
    assert updates == []


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


def test_spliced_motion_checks_the_fixed_root_goal_without_reanchoring(scene, model):
    config, _urdf = scene
    geometry = GeometryGuard(model)
    observed = np.array(config["initial_command"])
    start, middle, end = (observed.copy() for _ in range(3))
    start[0] += .15
    middle[0] += .175
    end[0] += .20
    obstacle = np.eye(4)
    obstacle[:3, 3] = model.nominal_roots(middle)["rod_tip"][:3, 3]
    original = obstacle.copy()
    boxes = [dict(name="splice_obstacle", center=[0, 0, 0], size=[.008] * 3)]
    safe, gap = geometry.safe_motion(start, end, [obstacle], boxes, config["motion"])
    assert not safe and gap < config["motion"]["clearance"]
    np.testing.assert_array_equal(obstacle, original)
    # A later command does not move this observation-time root obstacle.
    assert geometry.safe_motion(start, start, [obstacle], boxes, config["motion"])[0]
    assert geometry.safe_motion(end, end, [obstacle], boxes, config["motion"])[0]


def test_motion_continues_while_ik_waits_and_old_increment_uses_its_source_command(scene, tmp_path):
    from threading import Event
    from probtf_switch_press_demo.trial import Trial

    config, _urdf = scene
    trial = Trial(config, ROOT / "ros/examples/symaware_grasp/urdf/simple_six_dof_arm.urdf",
                  ROOT / "ros/examples/prob_artag_detector/samples/apriltag_36h11_id_021.png",
                  tmp_path / "continuous", PACKAGE)
    assert "stiffness" not in json.dumps(trial.controller.model.descriptor)
    assert not hasattr(trial.controller, "plant")
    blocked, moved, done = Event(), Event(), Event()
    state = dict(calls=0, moving_frames=0)

    def delayed_choose(image, command, stamp, on_iteration):
        state["calls"] += 1
        if state["calls"] == 2:
            state["source_command"] = command.copy()
            blocked.set()
            assert moved.wait(timeout=4), "motion loop blocked waiting for IK"
        # A controlled safe planner leaves a known unfinished target. This
        # checks thread/command semantics independently of optimizer latency;
        # a full solve may otherwise finish after its earlier preview arrives.
        controller = trial.controller
        result, debug = controller.observation.estimate(image, stamp)
        assert result is not None
        goal = GoalLikelihood(controller.model, result.record, command,
                              controller.goal_offset, controller.goal_orientation)
        increment = np.array([.06, 0., 0., 0., 0., 0.])
        target = command + increment
        hand = controller.model.hand(target, stamp)
        score, diagnostics = goal.objective.evaluate(hand)
        safe, gap = controller.geometry.safe_motion(command, target, goal.marker_modes,
                                                    controller.obstacles, controller.motion)
        assert safe
        prediction = Prediction(increment, target, hand, score, diagnostics, gap,
                                ik=dict(iterations=1, method="controlled async test planner"))
        prediction.records = controller.model.display_records(target, stamp, hand_summary=hand)
        roots = controller.model.nominal_roots(target)
        prediction.rods = np.array([[roots["tool0"][:3, 3], hand.mean]])
        on_iteration(prediction, goal, result, debug, 0)
        return prediction, goal, result, debug

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
    assert trial.summary["trajectory_frame"] == "base_link"
    tick = next(item for item in trial.servo_ticks if item["plan_cycle"] == 1)
    np.testing.assert_allclose(tick["target"], expected, atol=1e-12)


def test_native_display_packet_keeps_local_sources_and_only_the_requested_hand_projection(scene, model):
    config, _urdf = scene
    command = np.array(config["initial_command"])
    hand = model.hand(command, stamp=1.25)
    records = model.display_records(command, stamp=1.25, hand_summary=hand)
    assert tuple(record.child_frame_id for record in records) == BODY_FRAMES
    primitives = tuple(model.chain(command, stamp=1.25))
    for record, (parent, child), primitive in zip(records[:-1], model.edges[:-1], primitives):
        assert (record.parent_frame_id, record.child_frame_id) == (parent, child)
        assert record.stamp == 1.25 and not record.is_static
        assert record.provenance.method == "configured_native_local_transform_family"
        assert len(record.distribution.components) == 1
        component = record.distribution.components[0]
        assert component is primitive
        assert component.orientation is primitive.orientation
        assert component.translation is primitive.translation
        context = json.loads(record.provenance.detail)
        assert context["command"] == list(command)
        assert context["stamp"] == 1.25
        assert context["primitive_dependence"] == "independent"
        assert context["source_fingerprint"] == hand.metadata["source_fingerprint"]
        assert context["revision"] == hand.metadata["revision"]
    record = records[-1]
    assert record.parent_frame_id == "base_link"
    assert record.provenance.method == COMPARISON_METHOD
    component = record.distribution.components[0]
    assert component.orientation is hand.orientation
    np.testing.assert_array_equal(record.representative.translation, hand.mean)
    np.testing.assert_array_equal(component.translation.mean_at_reference, hand.mean)
    np.testing.assert_array_equal(component.translation.residual_covariance, hand.covariance)
    np.testing.assert_array_equal(component.orientation.backend_parameter_matrix(), hand.orientation_parameter)


def test_marker_display_queries_the_same_fixed_goal_comparison_without_replacing_the_sensor_law(scene, model):
    config, _urdf = scene
    original = gaussian_record(np.diag([.002, .003, .004]) ** 2)
    observed = np.array(config["initial_command"])
    offset = np.array([0, -.24, .045])
    quaternion = rpy_to_quat(.2, -.1, .3)
    original_components = original.distribution.components
    likelihood = GoalLikelihood(model, original, observed, offset, quaternion)
    attachment = DeterministicTransform(offset, quaternion)
    expected = compose_with_deterministic_right(likelihood.record, attachment.inverse(),
        child_frame_id="observed_ar_marker", edge_id="goal_comparison_marker_coordinate_query")
    shown = likelihood.marker_record
    assert likelihood.sensor_record is original
    assert original.distribution.components is original_components
    assert shown.parent_frame_id == "base_link" and shown.stamp == original.stamp
    assert original.parent_frame_id == "camera_optical_frame" and original.stamp == 1.
    assert shown.provenance.method == "goal_comparison_marker_coordinate_query"
    assert likelihood.record.edge_id in shown.provenance.derived_from_edge_ids
    for component, exact_query, source in zip(shown.distribution.components,
            expected.distribution.components, likelihood.record.distribution.components):
        assert component.raw_weight == exact_query.raw_weight == source.raw_weight
        np.testing.assert_array_equal(component.translation.mean_at_reference,
                                      exact_query.translation.mean_at_reference)
        np.testing.assert_array_equal(component.translation.residual_covariance,
                                      source.translation.residual_covariance)
        np.testing.assert_array_equal(component.translation.rotation_coupling,
                                      exact_query.translation.rotation_coupling)
        np.testing.assert_array_equal(component.orientation.backend_parameter_matrix(),
                                      exact_query.orientation.backend_parameter_matrix())
        assert np.linalg.norm(component.translation.rotation_coupling) > .1
        # Point-wise right composition is exact, including its nonzero X/R
        # dependence. An independently refitted marker product fails here.
        for goal_rotation in (source.orientation.mode_wxyz, rpy_to_quat(.7, -.3, .2)):
            marker_rotation = quat_mul(goal_rotation, attachment.inverse().rotation_wxyz)
            expected_translation = (source.conditional_translation_mean(goal_rotation)
                + quat_to_rotmat(goal_rotation) @ attachment.inverse().translation)
            np.testing.assert_allclose(component.conditional_translation_mean(marker_rotation),
                                       expected_translation, atol=2e-16)
    context = json.loads(shown.provenance.detail)
    assert context["observation_command"] == list(observed)
    assert context["coordinate_query"] == "goal_comparison_marker_coordinate_query"


def test_display_prefix_point_moments_use_the_same_continuous_native_laws(scene, model):
    from probtf.probability.moments import PointMomentSummary, forward_component_point_moments
    config, _urdf = scene
    command = np.array(config["initial_command"])
    records = model.display_records(command, stamp=1.)
    for index, frame in enumerate(BODY_FRAMES[:-1]):
        internal = model.chain(command, frame, stamp=1.)
        for endpoint in .1 * np.eye(3):
            shown = PointMomentSummary(endpoint, np.zeros((3, 3)))
            expected = PointMomentSummary(endpoint, np.zeros((3, 3)))
            for record, prepared in zip(reversed(records[:index + 1]), reversed(internal.prepared)):
                shown = forward_component_point_moments(record.distribution.components[0], shown, 480)
                expected = prepared.forward(expected)
            np.testing.assert_allclose(shown.mean, expected.mean, atol=1e-13)
            np.testing.assert_allclose(shown.covariance, expected.covariance, atol=1e-13)
            assert np.linalg.eigvalsh(shown.covariance)[0] > 0


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
