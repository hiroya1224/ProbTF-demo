"""Command selection keeps the observed joint law and the truth boundary."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import least_squares
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from probtf.bingham import bingham_log_normalizer
from probtf.geometry import (
    axis_angle_to_quat, quat_mul, quat_to_rotmat, rotation_vector_to_quaternion,
)
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState,
    bingham_from_rotation_coordinates, point_query, pose_query,
)
from probtf_observation_fusion_demo.model import Stage, WeightedState, load_inputs
from probtf_observation_fusion_demo.moving_control import MovingController, command_state
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene


CONFIG = Path(__file__).resolve().parents[1] / "config"
GOAL_OFFSET = [0.0, -0.24, 0.045]
GOAL_ORIENTATION = [0.5, -0.5, 0.5, 0.5]
LIMITS = {"max_joint_velocity": 0.20, "max_joint_acceleration": 0.60,
          "damping": 0.025}


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    scene = MovingSwitchScene(tmp_path_factory.mktemp("moving_control"))
    frame = scene.step(0.0, 1000.0)
    inferences, results = {}, {}
    for case in ("no_sensors", "imu", "marker", "imu_marker"):
        _, observations = load_inputs(CONFIG / "common.yaml", CONFIG / (case + ".yaml"))
        inference = MovingInference(scene.robot_description, observations)
        inferences[case] = inference
        results[case] = inference.update(scene.sensor_inputs(frame), 0)
    return scene, frame, inferences, results


def controller(scene, inference):
    return MovingController(inference, scene.robot_description, GOAL_OFFSET,
                            GOAL_ORIENTATION, LIMITS)


def test_future_command_preserves_joint_cross_blocks_and_frozen_marker(inputs):
    _, _, inferences, results = inputs
    inference = inferences["imu_marker"]
    state = results["imu_marker"].stages[-1].components[0].state
    trial_command = np.asarray(state.command) + [.013, -.007, .011, .009, -.006, .014]
    future = command_state(state, trial_command, inference.body_ids, inference.actions)
    chart = np.eye(state.mean.size)
    for name, action in zip(inference.body_ids, inference.actions):
        index, joint = state.edge_ids.index(name), int(action["joint_index"])
        delta = axis_angle_to_quat(action["axis"], trial_command[joint]-state.command[joint])
        chart[6*index+3:6*index+6, 6*index+3:6*index+6] = quat_to_rotmat(delta).T
    np.testing.assert_allclose(future.covariance, chart @ state.covariance @ chart.T,
                               rtol=1e-12, atol=1e-15)
    marker_index = state.edge_ids.index("marker")
    assert future.transforms[marker_index] is state.transforms[marker_index]
    assert np.linalg.norm(state.covariance[:36, 6*marker_index:6*marker_index+6]) > 1e-5
    assert np.linalg.norm(future.covariance[:36, 6*marker_index:6*marker_index+6]) > 1e-5
    returned = command_state(future, state.command, inference.body_ids, inference.actions)
    np.testing.assert_allclose(returned.covariance, state.covariance, atol=1e-14)
    for before, after in zip(state.transforms, returned.transforms):
        np.testing.assert_allclose(after.translation, before.translation, atol=1e-14)
        if before.bingham is not None:
            np.testing.assert_allclose(after.bingham, before.bingham, atol=1e-10)
    assert future.command == tuple(trial_command)
    assert future.stamp == state.stamp
    assert future.state_id != state.state_id


def test_same_controller_uses_all_fused_candidates_and_respects_command_bounds(inputs):
    scene, frame, inferences, results = inputs
    commands = {}
    for case, result in results.items():
        selected, diagnostics = controller(scene, inferences[case]).step(result, .05)
        commands[case] = selected
        components = result.stages[-1].components
        assert diagnostics["source_state_ids"] == [item.state.state_id for item in components]
        assert diagnostics["source_weights"] == [item.weight for item in components]
        assert len(diagnostics["selected_branches"]) == len(components)
        assert diagnostics["source_stamp"] == frame.stamp
        assert diagnostics["source_command"] == list(result.prior.command)
        assert diagnostics["source_observation_ids"] == result.records["observation_ids"]
        assert diagnostics["global_optimum_claimed"] is False
        increment = selected-np.asarray(result.prior.command)
        assert np.max(np.abs(increment)) <= LIMITS["max_joint_velocity"]*.05+1e-12
        assert np.max(np.abs(increment/.05)) <= LIMITS["max_joint_acceleration"]*.05+1e-12
        assert -diagnostics["log_expected_density"] <= diagnostics["feasible_baseline_objective"]+1e-12
        assert diagnostics["convergence_claimed"] is False
        assert diagnostics["gradient_evaluation_count"] == 1
        assert diagnostics["objective_evaluation_count"] <= 11
        for source, branch in zip(components, diagnostics["selected_branches"]):
            assert branch["parent_state_id"] == source.state.state_id
            assert branch["parent_stamp"] == frame.stamp
            assert branch["query_command"] == selected.tolist()
            assert branch["query_frame"] == "marker"
            np.testing.assert_allclose(np.array(branch["query_within"])
                                       +np.array(branch["query_between"]),
                                       branch["query_covariance"], atol=1e-12)
    # Identical physics, command and prior do not force identical control when
    # the actual posterior distributions differ.
    assert np.linalg.norm(commands["no_sensors"]-commands["imu_marker"]) > 1e-5


def test_empty_observations_are_identity_without_a_sensor_specific_control_rule(inputs):
    scene, frame, inferences, results = inputs
    result = results["no_sensors"]
    assert result.stages[-1].components[0].state is result.prior
    first = controller(scene, inferences["no_sensors"]).step(result, .05)
    # Sensor names and counts do not choose the action. The same law always
    # produces the same action, even with unrelated descriptive stage labels.
    stage = replace(result.stages[-1], label="any descriptive stage", sensor_ids=("unused",))
    renamed = replace(result, stages=result.stages[:-1]+(stage,))
    second = controller(scene, inferences["imu_marker"]).step(renamed, .05)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]
    assert np.linalg.norm(point_query(result.prior, result.hand_path).mode-frame.tip) > .2


def test_three_short_observation_updates_use_fresh_same_time_posterior(inputs):
    scene, frame, _, _ = inputs
    _, observations = load_inputs(CONFIG/"common.yaml", CONFIG/"imu.yaml")
    inference = MovingInference(scene.robot_description, observations)
    selector = controller(scene, inference)
    sensor_inputs = scene.sensor_inputs(frame)
    command = np.array(frame.command)
    previous_velocity = np.zeros(6)
    state_ids = []
    for index in range(3):
        observed = replace(sensor_inputs, time=.05*index, stamp=frame.stamp+.05*index,
                           command=command)
        result = inference.update(observed, index)
        selected, diagnostics = selector.step(result, .05)
        np.testing.assert_array_equal(diagnostics["source_command"], command)
        velocity = (selected-command)/.05
        assert np.max(np.abs(velocity)) <= LIMITS["max_joint_velocity"]+1e-12
        assert np.max(np.abs(velocity-previous_velocity)) <= LIMITS["max_joint_acceleration"]*.05+1e-12
        state_ids.append(diagnostics["source_state_ids"])
        previous_velocity, command = velocity, selected
    assert state_ids[0] != state_ids[1] != state_ids[2]


def test_no_observation_candidate_reaching_does_not_imply_physical_success(inputs):
    scene, _, inferences, results = inputs
    inference = inferences["no_sensors"]
    result = results["no_sensors"]
    selector = controller(scene, inference)
    state = result.prior
    def relative_position(command):
        future = command_state(state, command, inference.body_ids, inference.actions)
        return point_query(future, selector.relative_path).mean-np.asarray(GOAL_OFFSET)
    solution = least_squares(relative_position, state.command,
                             bounds=(selector.lower, selector.upper),
                             xtol=1e-9, ftol=1e-9, gtol=1e-9, max_nfev=80)
    assert np.linalg.norm(relative_position(solution.x)) < .006
    # The simulator is used only to check the selected-law prediction here;
    # its stiffness, equilibrium and true goal are never selector inputs.
    physical = scene.plant.equilibrium(solution.x)
    actual_tip = scene.plant.flange(physical)[:3, 3] + (
        scene.plant.flange(physical)[:3, :3] @ scene.tip_offset)
    assert np.linalg.norm(actual_tip-scene.goal) > .04


def test_optimizer_does_not_use_modal_query_jacobian_as_a_mean_derivative(inputs, monkeypatch):
    scene, _, inferences, results = inputs
    result = results["no_sensors"]
    first = controller(scene, inferences["no_sensors"]).step(result, .05)
    original = MovingController._query
    def without_modal_sensitivity(self, state):
        query = original(self, state)
        # The modal Jacobian can change the numerical preconditioner, but it
        # must never change the independently computed full-objective gradient.
        return replace(query, jacobian=np.zeros_like(query.jacobian))
    monkeypatch.setattr(MovingController, "_query", without_modal_sensitivity)
    second = controller(scene, inferences["no_sensors"]).step(result, .05)
    np.testing.assert_array_equal(first[1]["objective_gradient"], second[1]["objective_gradient"])
    assert first[1]["gradient_at_command"] == second[1]["gradient_at_command"]
    for _, diagnostics in (first, second):
        assert diagnostics["improved_from_feasible_baseline"]
        assert diagnostics["objective_evaluation_count"] <= 11


def test_single_gradient_step_uses_every_objective_term_and_improves_full_density(inputs):
    scene, _, inferences, results = inputs
    inference, result = inferences["no_sensors"], results["no_sensors"]
    selector = controller(scene, inference)
    selected, diagnostics = selector.step(result, .05)
    def cost(command):
        terms = []
        for component in result.stages[-1].components:
            future = command_state(component.state, command, inference.body_ids, inference.actions)
            query = pose_query(future, selector.relative_path)
            quaternion = quat_mul(query.rotation_wxyz,
                rotation_vector_to_quaternion(query.correction[3:]))
            matrix = bingham_from_rotation_coordinates(quaternion, query.covariance[3:, 3:])
            terms.append(np.log(component.weight)
                +multivariate_normal.logpdf(GOAL_OFFSET, query.mean[:3], query.covariance[:3, :3])
                +np.array(GOAL_ORIENTATION) @ matrix @ np.array(GOAL_ORIENTATION)
                -bingham_log_normalizer(matrix, integration_steps=480))
        return -float(logsumexp(terms))
    source = np.asarray(result.prior.command)
    selected_cost = cost(selected)
    np.testing.assert_allclose(selected_cost, -diagnostics["log_expected_density"], atol=1e-10)
    assert diagnostics["line_search_accepted"]
    assert selected_cost < cost(source)
    assert diagnostics["convergence_claimed"] is False
    baseline = np.asarray(diagnostics["gradient_at_command"])
    np.testing.assert_array_equal(baseline, source)
    base_cost = cost(baseline)
    expected = []
    for index, difference in enumerate(diagnostics["gradient_difference_steps"]):
        probe = baseline.copy()
        probe[index] += difference
        expected.append((cost(probe)-base_cost)/difference)
    np.testing.assert_allclose(diagnostics["objective_gradient"], expected,
                               rtol=1e-6, atol=1e-7)
    assert diagnostics["objective_evaluation_count"] <= 11


def test_near_target_gradient_step_shrinks_without_normalizing_to_speed_limit():
    # A one-joint arm near a real core-law density maximum. A forced normalized
    # speed step would overshoot by an order of magnitude; the scaled gradient
    # instead takes a smaller corrective command.
    angle = 1e-4
    quaternion = axis_angle_to_quat([0., 0., 1.], angle)
    body = BinghamTransform("body", [0., 0., 0.],
        bingham_from_rotation_coordinates(quaternion, .002*np.eye(3)), .0001*np.eye(3))
    tip = BinghamTransform("tip", [1., 0., 0.], rotation_wxyz=[1., 0., 0., 0.])
    marker = BinghamTransform("marker", [0., 0., 0.], rotation_wxyz=[1., 0., 0., 0.])
    state = JointTransformState.from_independent([body, tip, marker], command=(angle,), stamp=1.)
    inference = SimpleNamespace(body_ids=["body"],
        actions=[{"joint_index": 0, "axis": [0., 0., 1.]}],
        paths={"hand": ("body", "tip"), "marker": (("marker", False),)},
        settings={"root_frame": "base"})
    urdf = '<robot name="one"><joint name="one" type="revolute"><limit lower="-3" upper="3"/></joint></robot>'
    # E[R ex] has the required first-order radial contraction of 0.002.
    selector = MovingController(inference, urdf, [.998, 0., 0.], [1., 0., 0., 0.], LIMITS)
    result = SimpleNamespace(prior=state, stages=(Stage("fused", (WeightedState("one", 1., state),)),),
                             records={"observation_ids": []})
    selected, diagnostics = selector.step(result, .05)
    assert abs(selected[0]) < angle
    assert abs(selected[0]-angle) < LIMITS["max_joint_acceleration"]*.05**2/2
    assert diagnostics["line_search_accepted"]
    assert diagnostics["convergence_claimed"] is False


def test_gradient_metadata_names_feasible_coasting_point_when_zero_is_infeasible(inputs):
    scene, _, inferences, results = inputs
    selector = controller(scene, inferences["no_sensors"])
    selector.velocity[:] = .1
    result = results["no_sensors"]
    _, diagnostics = selector.step(result, .05)
    assert diagnostics["zero_increment_feasible"] is False
    np.testing.assert_allclose(diagnostics["gradient_at_command"],
        np.asarray(result.prior.command)+np.asarray(diagnostics["feasible_baseline_increment"]))
    assert diagnostics["gradient_at_command"] != diagnostics["source_command"]
    assert all(value != 0.0 for value in diagnostics["gradient_difference_steps"])
