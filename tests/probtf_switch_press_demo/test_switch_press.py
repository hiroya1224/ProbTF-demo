from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from probtf.distributions import (
    BinghamOrientation, ConditionalGaussianTranslation, TransformComponent,
    TransformDistribution, TransformDistributionStamped, compose_with_deterministic_right,
)
from probtf.geometry import DeterministicTransform, quat_mul, quat_to_rotmat, rpy_to_quat
from probtf.probability.point_density import DensityIntegrationError
from probtf.candidates import JointFiniteTransformLaw, load_initial_transform_law
from probtf_switch_press_demo.body import EquilibriumBody, ReachMonitor, clearance, motion_samples
from probtf_switch_press_demo.control import CommandServo, Controller, GoalLikelihood
from probtf_switch_press_demo.kinematics import CandidateKinematics
from probtf_switch_press_demo.scene import apply, inverse, load_config, transform, write_robot
from probtf_switch_press_demo.vision import PlanarCamera
from probtf_switch_press_demo.beliefs import (
    BODY_FRAMES, JointMeasure, display_record, finite_pose_record, left_compose_record,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "ros/examples/probtf_switch_press_demo"


@pytest.fixture
def scene(tmp_path):
    config = load_config(PACKAGE / "config/default.yaml")
    # A smaller explicit JOINT transform law keeps the image/async regression
    # cases quick. The default 64-tuple input is verified separately.
    source = load_initial_transform_law(PACKAGE / config["control"]["initial_transform_law"])
    rows, _weights = source.quadrature()
    law_path = tmp_path / "initial_transforms.yaml"
    law_path.write_text(yaml.safe_dump(dict(kind="joint_finite", law_id="regression_body",
        revision=1, reference_command=source.reference_command, frame_names=BODY_FRAMES,
        scenarios=[dict(id=name, weight=weight, local_transforms=rows[index].tolist())
                   for index, name, weight in ((0, "body_a", .25), (21, "body_b", .5), (63, "body_c", .25))])))
    config["control"]["initial_transform_law"] = str(law_path)
    urdf = write_robot(ROOT / "ros/examples/symaware_grasp/urdf/simple_six_dof_arm.urdf",
                       tmp_path / "rod.urdf", config)
    return config, urdf


def configured_kinematics(config, urdf):
    source = load_initial_transform_law(config["control"]["initial_transform_law"])
    return CandidateKinematics(urdf, source, config["rod"]["length"], source.reference_command)


def initial_image(config, urdf):
    plant = EquilibriumBody(urdf, config["plant"]["stiffness"])
    hand = plant.flange(plant.equilibrium(config["initial_command"]))
    marker = inverse(transform(config["base_translation"])) @ transform(config["marker"]["world_translation"], config["marker"]["world_rpy"])
    camera = PlanarCamera(ROOT / "ros/examples/prob_artag_detector/samples/apriltag_36h11_id_021.png",
                         config["marker"]["size"], config["camera"], config["seed"])
    image = camera.render(hand @ transform(config["camera"]["translation"], config["camera"]["rpy"]), marker)
    return image, hand, marker


def gaussian_record(covariance):
    quaternion = rpy_to_quat(.1, -.2, .3)
    components = tuple(TransformComponent(
        str(index), weight, BinghamOrientation.dirac(quaternion),
        ConditionalGaussianTranslation(np.array(mean), covariance, np.zeros((3, 9))))
        for index, weight, mean in ((0, .3, [.02, .01, .6]), (1, .7, [-.01, .04, .62])))
    return TransformDistributionStamped("camera", "marker", 1., "edge", "test", TransformDistribution(components))


def finite_hand(points, weights):
    return finite_pose_record("observation_hand", "predicted_rod_tip", 1.,
                              [transform(point) for point in points],
                              tuple(str(index) for index in range(len(points))), weights)


def test_expected_density_matches_analytic_gaussian_mixture_without_kde():
    covariance = np.diag([.002, .003, .004]) ** 2
    record = gaussian_record(covariance)
    hand_camera = transform([.1, -.03, .08], [.2, -.1, .4])
    offset = np.array([.0, -.24, .08])
    likelihood = GoalLikelihood(record, hand_camera, offset, 5)
    points = np.array([[.12, .12, .7], [.10, .15, .72], [.09, .13, .71]])
    body_weights = np.array([.25, .5, .25])
    reference_rotation = quat_to_rotmat(record.distribution.components[0].orientation.reference_quaternion_wxyz)
    covariance_hand = hand_camera[:3, :3] @ covariance @ hand_camera[:3, :3].T
    expected_terms = []
    for component in record.distribution.components:
        mean = apply(hand_camera, component.translation.mean_at_reference + reference_rotation @ offset)
        expected_terms.extend(multivariate_normal.logpdf(points, mean, covariance_hand) +
                              np.log(body_weights) + np.log(component.raw_weight))
    assert likelihood.log_expected_density(finite_hand(points, body_weights)) == pytest.approx(logsumexp(expected_terms), abs=1e-9)


def test_expected_density_gradient_matches_central_difference():
    likelihood = GoalLikelihood(gaussian_record(np.diag([.002, .003, .004]) ** 2),
                                np.eye(4), np.zeros(3), 5)
    tips = np.array([[.018, .012, .601], [-.011, .039, .622]])
    weights = np.array([.4, .6])
    _score, gradient = likelihood.value_and_tip_gradient(finite_hand(tips, weights))
    finite = np.zeros_like(tips)
    epsilon = 1e-7
    for index in range(len(tips)):
        for axis in range(3):
            plus, minus = tips.copy(), tips.copy()
            plus[index, axis] += epsilon
            minus[index, axis] -= epsilon
            finite[index, axis] = (likelihood.log_expected_density(finite_hand(plus, weights))
                                   - likelihood.log_expected_density(finite_hand(minus, weights))) / (2 * epsilon)
    np.testing.assert_allclose(gradient, finite, rtol=2e-6, atol=1e-5)


def test_goal_quadrature_transforms_the_same_original_law_after_a_frame_change():
    from dataclasses import replace

    record = gaussian_record(np.diag([.002, .003, .004]) ** 2)
    orientation = BinghamOrientation.from_parameter_matrix(np.diag([0., -200., -300., -400.]))
    record = replace(record, distribution=TransformDistribution(tuple(
        replace(component, orientation=orientation) for component in record.distribution.components)))
    anchor = transform([.4, -.1, .2], [.2, -.3, .4])
    offset = np.array([0, -.24, .045])
    local = GoalLikelihood(record, np.eye(4), offset, 9)
    moved = GoalLikelihood(record, anchor, offset, 9)
    points = np.array([[.1, .2, .6], [-.1, -.1, .7]])
    old_values, old_gradients = local.density.log_density_and_gradient(points)
    new_values, new_gradients = moved.density.log_density_and_gradient(apply(anchor, points))
    np.testing.assert_allclose(new_values, old_values, atol=1e-7)
    np.testing.assert_allclose(new_gradients, old_gradients @ anchor[:3, :3].T, rtol=1e-6, atol=1e-5)


def test_degenerate_residual_is_rejected_without_adding_width():
    with pytest.raises(DensityIntegrationError, match="density query unavailable"):
        GoalLikelihood(gaussian_record(np.zeros((3, 3))), np.eye(4), np.zeros(3), 4)


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
    image, hand, marker = initial_image(config, urdf)
    # Controller receives no actual joints, stiffness, or marker world pose.
    marker_settings = {key: value for key, value in config["marker"].items() if not key.startswith("world_")}
    controller = Controller(urdf, config["control"], config["motion"], marker_settings, config["camera"],
                            config["rod"]["length"], config["obstacles"], config["seed"])
    selected, goal, result, _debug = controller.choose(image, np.array(config["initial_command"]), 1.)
    assert result is not None and selected is not None
    assert selected.ik["method"] == "L-BFGS-B expected-likelihood IK"
    assert selected.ik["iterations"] > 0
    assert selected.ik["trace"][-1]["log_expected_density"] > selected.ik["trace"][0]["log_expected_density"]
    assert selected.hand_record.parent_frame_id == goal.record.parent_frame_id == "observation_hand"
    assert selected.hand_record.stamp == goal.record.stamp == 1.
    assert goal.log_expected_density(selected.hand_record) == pytest.approx(selected.score, abs=1e-8)
    assert abs(sum(result.weights) - 1) < 1e-12
    goal_in_world = apply(hand, goal.mean)
    np.testing.assert_allclose(goal_in_world, apply(marker, config["marker"]["goal_offset"]), atol=.01)
    assert all(np.linalg.eigvalsh(item.component.translation.residual_covariance)[0] > 0
               for item in result.record.distribution.normalized_components())


def test_unresolved_observation_and_unsafe_iterates_never_publish_motion(scene, monkeypatch):
    config, urdf = scene
    image, _hand, _marker = initial_image(config, urdf)
    marker_settings = {key: value for key, value in config["marker"].items() if not key.startswith("world_")}
    controller = Controller(urdf, config["control"], config["motion"], marker_settings, config["camera"],
                            config["rod"]["length"], config["obstacles"], config["seed"])
    updates = []
    monkeypatch.setattr(controller.candidates, "safe_path", lambda *_args: False)
    selected, _goal, result, _debug = controller.choose(image, np.array(config["initial_command"]), 1.,
                                                       lambda *update: updates.append(update))
    assert result is not None and selected is None and not updates
    monkeypatch.setattr(controller.observation, "estimate", lambda *_args: (None, None))
    selected, goal, result, _debug = controller.choose(image, np.array(config["initial_command"]), 2.,
                                                      lambda *update: updates.append(update))
    assert selected is goal is result is None and not updates


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
    source = load_initial_transform_law(config["control"]["initial_transform_law"])
    source = JointFiniteTransformLaw(source.initial_local_transforms[1:2], [1.], (source.node_ids[1],),
                                    BODY_FRAMES, reference_command=source.reference_command)
    candidates = CandidateKinematics(urdf, source, config["rod"]["length"], source.reference_command)
    observed = np.array(config["initial_command"])
    observed_hand = candidates.current(observed).root_transforms[0, -2].copy()
    start, middle, end = (observed.copy() for _ in range(3))
    start[0] += .15
    middle[0] += .175
    end[0] += .20
    middle_hand = candidates.current(middle).root_transforms[0, -2]
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
    assert trial.guard is not trial.controller.candidates
    assert trial.guard.law is not trial.controller.candidates.law
    assert trial.guard.source_law is trial.controller.candidates.source_law
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
        if sum(cycle["action"] == "retarget" for cycle in value.cycles) >= 2:
            done.set()

    trial.controller.choose = delayed_choose
    trial.run(frame, done.is_set)
    assert moved.is_set()
    assert trial.cycles[1]["moving_ticks_during_ik"] >= 3
    # Motion blur can make an intervening observation unresolved. The next
    # selected plan must still act on the command of its own source image.
    second = [cycle for cycle in trial.cycles if cycle["action"] == "retarget"][1]
    assert np.max(np.abs(second["command_at_receipt"] - second["command_before"])) > .001
    expected = second["command_before"] + second["increment"]
    np.testing.assert_allclose(second["reference_hand_world"],
                               trial.world_base @ second["reference_hand_base"], atol=1e-12)
    assert trial.summary["trajectory_frame"] == "base_link"
    ticks = [item for item in trial.servo_ticks if item["plan_cycle"] == second["cycle"]]
    assert any(np.allclose(tick["target"], expected, rtol=0., atol=1e-12) for tick in ticks)


def test_first_ik_iterate_moves_robot_before_the_first_solve_returns(scene, tmp_path):
    from threading import Event
    from probtf_switch_press_demo.trial import Trial

    config, _urdf = scene
    trial = Trial(config, ROOT / "ros/examples/symaware_grasp/urdf/simple_six_dof_arm.urdf",
                  ROOT / "ros/examples/prob_artag_detector/samples/apriltag_36h11_id_021.png",
                  tmp_path / "first_progress", PACKAGE)
    choose = trial.controller.choose
    blocked, moved, returned, done = Event(), Event(), Event(), Event()
    state = dict(calls=0, moving_frames=0, preview_checked=False)

    def delayed_first_choose(image, command, stamp, on_iteration):
        state["calls"] += 1

        def progress(prediction, goal, result, debug, iteration):
            on_iteration(prediction, goal, result, debug, iteration)
            if state["calls"] == 1 and not blocked.is_set() and np.max(np.abs(prediction.increment)) > .001:
                state["source_command"] = command.copy()
                blocked.set()
                assert moved.wait(timeout=4), "first feasible iterate waited for the completed IK solve"

        result = choose(image, command, stamp, progress)
        returned.set()
        return result

    def frame(value):
        if blocked.is_set() and not returned.is_set() and np.max(np.abs(value.command-state["source_command"])) > .001:
            update = value.applied_ik_updates[-1]
            assert value.preview["prediction"].joint_after.metadata["query_id"] == update["joint_measure_after_query"]["query_id"]
            np.testing.assert_array_equal(value.preview["prediction"].increment, update["increment"])
            np.testing.assert_array_equal(update["absolute_target"], update["reference_command"]+update["increment"])
            for query in (update["joint_measure_before_query"], update["joint_measure_after_query"]):
                assert query["source_law_id"] == value.controller.candidates.source_law.law_id
                assert query["source_revision"] == value.controller.candidates.source_law.revision
            assert update["expected_density_diagnostics"]["relative_standard_error"] == 0.
            summary = str(update["goal_density_diagnostics"])
            assert "proposal_parameters" not in summary and "mode_quaternions" not in summary
            state["preview_checked"] = True
            state["moving_frames"] += 1
            if state["moving_frames"] >= 3:
                moved.set()
        if value.cycles:
            done.set()

    trial.controller.choose = delayed_first_choose
    trial.run(frame, done.is_set)
    assert moved.is_set() and returned.is_set()
    assert state["moving_frames"] >= 3 and state["preview_checked"]
    import json
    archived = json.loads((trial.output / "applied_ik_updates.json").read_text())
    assert len(archived) == len(trial.applied_ik_updates) > 0


@pytest.mark.parametrize("completion", ["density_error", "stale_final"])
def test_accepted_progress_survives_later_solver_failure_and_stale_final_is_not_adopted(scene, tmp_path, completion):
    from threading import Event
    from probtf_switch_press_demo.trial import Trial

    config, _urdf = scene
    if completion == "stale_final":
        config["control"]["max_plan_age"] = .5
    trial = Trial(config, ROOT / "ros/examples/symaware_grasp/urdf/simple_six_dof_arm.urdf",
                  ROOT / "ros/examples/prob_artag_detector/samples/apriltag_36h11_id_021.png",
                  tmp_path / completion, PACKAGE)
    choose = trial.controller.choose
    blocked, release, done = Event(), Event(), Event()
    state = dict(calls=0, moving_frames=0)
    expected_action = "continue_last_target" if completion == "density_error" else "discarded_stale"

    def controlled_choose(image, command, stamp, on_iteration):
        state["calls"] += 1
        if state["calls"] > 1:
            return None, None, None, np.repeat(image[:, :, None], 3, axis=2)

        def progress(prediction, goal, result, debug, iteration):
            on_iteration(prediction, goal, result, debug, iteration)
            if not blocked.is_set() and np.max(np.abs(prediction.increment)) > .001:
                state["source_command"] = command.copy()
                blocked.set()
                assert release.wait(timeout=4), "safe partial target was not applied"
                if completion == "density_error":
                    raise DensityIntegrationError("test density refinement unavailable", {})

        return choose(image, command, stamp, progress)

    def frame(value):
        if blocked.is_set() and value.applied_ik_updates:
            update = value.applied_ik_updates[-1]
            if np.max(np.abs(value.command-state["source_command"])) > .001:
                state["moving_frames"] += 1
            ready = (state["moving_frames"] >= 3 if completion == "density_error" else
                     value.time-update["source_time"] > config["control"]["max_plan_age"]+.1)
            if ready:
                release.set()
        if any(cycle["action"] == expected_action for cycle in value.cycles):
            done.set()

    trial.controller.choose = controlled_choose
    trial.run(frame, done.is_set)
    assert state["moving_frames"] >= 3
    entry = next(cycle for cycle in trial.cycles if cycle["action"] == expected_action)
    assert entry["applied_updates"]
    update = trial.applied_ik_updates[entry["applied_updates"][-1]]
    assert trial.preview["prediction"].joint_after.metadata["query_id"] == update["joint_measure_after_query"]["query_id"]
    assert all(item["applied_time"]-item["source_time"] <= config["control"]["max_plan_age"]
               for item in trial.applied_ik_updates)
    if completion == "density_error":
        assert "test density refinement unavailable" in entry["observation_error"]
    else:
        assert entry["discard_reason"] == "stale"
        assert entry["received_time"]-entry["time"] > config["control"]["max_plan_age"]


def test_all_link_display_serializes_the_same_configured_joint_pushforward(scene):
    config, urdf = scene
    candidates = configured_kinematics(config, urdf)
    command = np.array(config["initial_command"])
    prediction = candidates.predict(command, np.array([.01, -.03, .02, 0, -.02, .01]), 1.25)
    joint = prediction.joint_after
    assert joint.body_ids == candidates.ids == prediction.joint_before.body_ids
    np.testing.assert_array_equal(joint.weights, candidates.weights)
    assert prediction.hand_record is prediction.link_records[-1]
    assert joint.expression.law.source_law is candidates.source_law
    for row, name in enumerate(candidates.ids):
        reference = inverse(prediction.joint_before.root_transforms[row, -2])
        product = np.eye(4)
        for link, frame in enumerate(BODY_FRAMES):
            product = product @ joint.local_transforms[row, link]
            np.testing.assert_allclose(joint.root_transforms[row, link], product, atol=1e-13)
            record = prediction.link_records[link]
            assert record.parent_frame_id == "observation_hand" and record.stamp == 1.25 and not record.is_static
            assert len(record.distribution.components) == len(candidates.ids)
            component = record.distribution.components[row]
            assert component.component_id == name
            assert component.raw_weight == candidates.weights[row]
            np.testing.assert_array_equal(component.translation.residual_covariance, np.zeros((3, 3)))
            expected = reference @ product
            np.testing.assert_allclose(component.translation.mean_at_reference, expected[:3, 3], atol=1e-13)
            np.testing.assert_allclose(quat_to_rotmat(component.orientation.reference_quaternion_wxyz),
                                       expected[:3, :3], atol=1e-13)


def test_relative_joint_propagation_rejects_mismatched_body_correspondence(scene):
    config, urdf = scene
    candidates = configured_kinematics(config, urdf)
    prediction = candidates.predict(np.array(config["initial_command"]), np.zeros(6))
    joint = prediction.joint_after
    mismatched = JointMeasure(tuple(reversed(joint.body_ids)), joint.weights, joint.local_transforms[::-1])
    with pytest.raises(ValueError, match="same source snapshot"):
        joint.relative_records(mismatched, 1.)


def test_root_link_display_keeps_absolute_uncertainty_separate_from_relative_ik(scene):
    config, urdf = scene
    candidates = configured_kinematics(config, urdf)
    prediction = candidates.predict(np.array(config["initial_command"]), np.zeros(6), 1.)
    absolute = prediction.joint_after.root_records(1.)
    assert len(absolute) == len(BODY_FRAMES)
    for link, record in enumerate(absolute):
        assert record.parent_frame_id == "base_link"
        assert record.child_frame_id == "root_predicted/" + BODY_FRAMES[link]
        for row, component in enumerate(record.distribution.components):
            assert component.component_id == prediction.joint_after.body_ids[row]
            assert component.raw_weight == prediction.joint_after.weights[row]
            np.testing.assert_allclose(component.translation.mean_at_reference,
                prediction.joint_after.root_transforms[row, link, :3, 3], atol=1e-12)
            np.testing.assert_allclose(quat_to_rotmat(component.orientation.reference_quaternion_wxyz),
                prediction.joint_after.root_transforms[row, link, :3, :3], atol=1e-12)
    root_tips = np.array([component.translation.mean_at_reference for component in absolute[-1].distribution.components])
    assert np.linalg.norm(root_tips[0] - root_tips[-1]) > .05
    # Zero motion has NO relative tip uncertainty even though root-based
    # uncertainty remains. Rendering the latter must not reset its center.
    np.testing.assert_array_equal(prediction.tips, np.tile(candidates.tip, (3, 1)))


def test_marker_pose_display_exactly_transforms_the_coupled_law_without_helper_frames():
    original = gaussian_record(np.diag([.002, .003, .004]) ** 2)
    marker = compose_with_deterministic_right(original,
        DeterministicTransform(np.array([0, -.24, .045]), [1, 0, 0, 0]), "goal", "goal_edge")
    hand, hand_camera = transform([.6, .1, .4], [.1, .2, -.3]), transform([.02, -.1, .12])
    shown = display_record(marker, hand @ hand_camera, 4., "observed_ar_marker")
    assert shown.parent_frame_id == "base_link" and shown.stamp == original.stamp
    import json
    assert json.loads(shown.provenance.detail)["source_observation_stamp"] == original.stamp
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


def test_published_hand_and_goal_preserve_the_kernel_under_the_same_frozen_anchor(scene):
    from probtf.probability.sampling import sample_transform_distribution

    config, urdf = scene
    candidates = configured_kinematics(config, urdf)
    prediction = candidates.predict(np.array(config["initial_command"]),
                                    np.array([.01, -.03, .02, 0, -.02, .01]), 1.)
    goal = GoalLikelihood(gaussian_record(np.diag([.01, .015, .02]) ** 2),
                          transform([.1, .2, -.1], [.2, -.3, .1]), np.array([0, -.24, .045]),
                          5)
    anchor = transform([.6, .1, .4], [.1, .2, -.3])
    shown_hand = display_record(prediction.hand_record, anchor, 4.)
    shown_goal = display_record(goal.record, anchor, 4.)
    assert shown_hand.stamp == prediction.hand_record.stamp
    assert shown_goal.stamp == goal.record.stamp
    import json
    metadata = json.loads(shown_hand.provenance.detail)
    assert metadata["source_observation_stamp"] == prediction.hand_record.stamp
    assert metadata["temporal_prediction"] is False
    assert metadata["display_result_kind"] == "frozen_law_query"
    assert shown_hand.parent_frame_id == shown_goal.parent_frame_id == "base_link"
    samples = sample_transform_distribution(shown_hand.distribution, 100, np.random.default_rng(4))
    support = apply(anchor, prediction.tips)
    for point in samples.translations:
        assert np.min(np.linalg.norm(support - point, axis=1)) < 1e-12
    for before, after in zip(prediction.hand_record.distribution.components, shown_hand.distribution.components):
        assert before.component_id == after.component_id and before.raw_weight == after.raw_weight
        np.testing.assert_array_equal(after.translation.residual_covariance, np.zeros((3, 3)))
    # An independent Gaussian-mixture calculation must give the same score
    # using the ACTUAL published support, weights, and target law.
    terms = []
    for hand in shown_hand.distribution.normalized_components():
        for target in shown_goal.distribution.normalized_components():
            component = target.component
            mean = component.conditional_translation_mean(component.orientation.reference_quaternion_wxyz)
            terms.append(np.log(hand.weight) + np.log(target.weight) +
                multivariate_normal.logpdf(hand.component.translation.mean_at_reference, mean,
                                            component.translation.residual_covariance))
    assert goal.log_expected_density(prediction.hand_record) == pytest.approx(logsumexp(terms), abs=1e-8)
    wrong_frame = left_compose_record(prediction.hand_record, anchor, "different_frame")
    with pytest.raises(ValueError, match="same observation frame"):
        goal.log_expected_density(wrong_frame)


@pytest.mark.parametrize("rotation, detection_required", [(.009, True), (.018, False)],
                         ids=["moderate_motion", "strong_motion"])
def test_motion_exposure_blurs_pixels_and_detector_covariance_uses_the_degraded_image(scene, rotation, detection_required):
    from probtf_switch_press_demo.vision import ImageObservation

    config, urdf = scene
    body = EquilibriumBody(urdf, config["plant"]["stiffness"])
    command = np.array(config["initial_command"])
    first = body.flange(body.equilibrium(command)) @ transform(config["camera"]["translation"], config["camera"]["rpy"])
    command[0] += rotation
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
    if result is None:
        # Strong exposure blur may prevent a valid observation altogether.
        assert not detection_required and observation.last_observation is None
        return
    assert result is not None
    covariance = observation.last_observation.image_covariance
    floor = np.eye(8) * config["marker"]["corner_sigma_px"] ** 2
    assert np.linalg.eigvalsh(covariance - floor)[0] >= -1e-10
    assert np.trace(covariance) > np.trace(floor) + 1e-4
    static_observation = ImageObservation(config["marker"], config["camera"])
    static_result, _debug = static_observation.estimate(still_image, 1.)
    assert static_result is not None
    assert np.trace(covariance) > np.trace(static_observation.last_observation.image_covariance) + 1e-4
