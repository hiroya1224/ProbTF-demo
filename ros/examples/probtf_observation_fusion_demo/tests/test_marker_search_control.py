"""Marker-loss actions preserve the estimated laws and feasible command history."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from probtf_observation_fusion_demo.display import make_markers, verify_wire_identity
from probtf_observation_fusion_demo.model import load_inputs
from probtf_observation_fusion_demo.marker_search import MarkerSearchPolicy, marker_observed
from probtf_observation_fusion_demo.moving_control import MovingController, command_state
from probtf_observation_fusion_demo.moving_inference import MovingInference
from legacy_marker import legacy_marker_scene
from probtf_observation_fusion_demo.node import build_snapshot, verify_adapter_identity
from probtf_observation_fusion_demo.stiffness_command import StiffnessCommandMap
from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier


PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def search_inputs(tmp_path_factory):
    scene = legacy_marker_scene(tmp_path_factory.mktemp('marker_search_control'))
    inputs = scene.sensor_inputs(scene.step(0., 1000.))
    contexts = {}
    for case in ('no_sensors', 'imu', 'marker', 'imu_marker'):
        _, observations = load_inputs(PACKAGE/'config/common.yaml', PACKAGE/'config'/(case+'.yaml'))
        inference = MovingInference(scene.robot_description, observations, config_path=scene.legacy_inference_config)
        result = inference.update(inputs, 0)
        contexts[case] = inference, result
    return scene, inputs, contexts


def make_controller(scene, inference):
    settings = inference.settings['control']
    return MovingController(inference, scene.robot_description,
        settings['goal_offset'], settings['goal_orientation_wxyz'], settings,
        integration_steps=settings['integration_steps'])


def advance_without_new_sensor_truth(inference, inputs, command, index, command_map=None):
    """New timestamped blank image; the sensor boundary has no world pose/K."""
    missing = replace(inputs, stamp=inputs.stamp+.05*index, time=.05*index,
        command=np.asarray(command), image=np.full_like(inputs.image, 255))
    return inference.update(missing, index, command_map=command_map)


SEARCH = {'enabled': True, 'joint_indices': [3, 4], 'amplitude_rad': .02,
          'max_joint_velocity': .025, 'settle_time_s': .25,
          'amplitude_increment_rad': .01, 'max_amplitude_rad': .12}


def policy_result(command=None, *, posterior=True, detector=True, stamp=1000., version=1):
    return SimpleNamespace(prior=SimpleNamespace(command=tuple(np.zeros(6) if command is None else command),
        stamp=stamp, version=version), local_estimates=(SimpleNamespace(
            posterior=object() if posterior else None,
            detector_result=object() if detector else None),))


def plan(policy, time, *, command=None, velocity=None, visible=False, lower=None, upper=None,
         observation_stamp=None, detected=None):
    result = policy_result(command, posterior=visible, detector=visible, stamp=1000.+time)
    if observation_stamp is not None:
        result.records = dict(fixed_goal_memory=dict(available=True,
            observation_stamp=1000.+observation_stamp))
    return policy.plan(result, np.zeros(6) if velocity is None else velocity, time,
        joint_lower=np.full(6, -3.) if lower is None else lower,
        joint_upper=np.full(6, 3.) if upper is None else upper,
        marker_detected=detected)


@pytest.mark.parametrize('posterior,detector,expected', [
    (True, True, True), (True, False, False), (False, True, False), (False, False, False)])
def test_only_a_completed_marker_observation_counts_as_reacquisition(posterior, detector, expected):
    assert marker_observed(policy_result(posterior=posterior, detector=detector)) is expected
    assert marker_observed(SimpleNamespace(local_estimates=())) is False


@pytest.mark.parametrize('enabled,marker_enabled', [(False, True), (True, False), (False, False)])
def test_marker_absence_does_not_start_search_when_the_sensor_is_unused(enabled, marker_enabled):
    policy = MarkerSearchPolicy(dict(SEARCH, enabled=enabled), marker_enabled=marker_enabled)
    for index in range(5):
        chosen = plan(policy, .05*index, velocity=np.full(6, .01))
        assert chosen.mode == 'reaching'
        assert chosen.record['marker_visible'] is None
        assert chosen.record['episode'] == 0
        assert chosen.record['event'] == 'marker_observation_not_enabled'
        assert chosen.target_command is chosen.position_lower is chosen.position_upper is None


def test_marker_loss_brakes_before_settling_even_if_the_marker_reappears():
    policy = MarkerSearchPolicy(SEARCH, marker_enabled=True, exposure_time=.18)
    initial = np.array([.1, -.2, .3, -.4, .5, -.6])
    lost = plan(policy, 0., command=initial, velocity=np.full(6, .06))
    assert lost.mode == 'braking' and lost.target_command is None
    assert lost.record['stop_anchor_command'] is None
    seen_but_moving = plan(policy, .05, command=initial+.001, velocity=np.full(6, .03), visible=True)
    assert seen_but_moving.mode == 'braking'
    stopped_command = initial+.002
    stopped = plan(policy, .10, command=stopped_command, visible=True)
    assert stopped.mode == 'settling'
    np.testing.assert_array_equal(stopped.target_command, stopped_command)
    assert not stopped.target_command.flags.writeable
    not_yet_clear = plan(policy, .30, command=stopped_command, visible=True)
    assert not_yet_clear.mode == 'settling'
    resumed = plan(policy, .35, command=stopped_command, visible=True)
    assert resumed.mode == 'reaching'
    assert resumed.record['event'] == 'marker_reacquired_after_stopping'
    assert resumed.target_command is None


def test_settling_holds_long_enough_to_clear_the_actual_exposure_history():
    policy = MarkerSearchPolicy(dict(SEARCH, settle_time_s=.05), marker_enabled=True, exposure_time=.18)
    assert plan(policy, 0.).mode == 'settling'
    assert plan(policy, .179).mode == 'settling'
    searching = plan(policy, .180)
    assert searching.mode == 'searching'
    assert searching.record['settle_time_s'] == .18
    assert searching.record['stopped_before_search']


def test_default_large_steps_stop_at_outward_and_return_viewpoints_and_inspect_joint_five_first():
    policy = MarkerSearchPolicy({}, marker_enabled=True)
    anchor = np.zeros(6)
    assert plan(policy, 0., command=anchor).mode == 'settling'
    outward = plan(policy, .35, command=anchor)
    assert outward.mode == 'searching'
    assert outward.max_speed == .048
    assert outward.record['joint_indices'] == [4, 3]
    assert outward.record['amplitude_rad'] == .06
    np.testing.assert_array_equal(outward.target_command, [0., 0., 0., 0., .06, 0.])
    point = outward.target_command
    moving = plan(policy, .40, command=point, velocity=[0., 0., 0., 0., 1e-5, 0.])
    assert moving.mode == 'searching'
    stopped = plan(policy, .45, command=point)
    assert stopped.mode == 'settling' and stopped.record['settle_reason'] == 'outward_observation'
    np.testing.assert_array_equal(stopped.target_command, point)
    assert plan(policy, .799, command=point).mode == 'settling'
    returning = plan(policy, .80, command=point)
    assert returning.mode == 'searching'
    np.testing.assert_array_equal(returning.target_command, anchor)
    center = plan(policy, .85, command=anchor)
    assert center.mode == 'settling' and center.record['completed_roundtrips'] == 1
    assert plan(policy, 1.199, command=anchor).mode == 'settling'
    expanded = plan(policy, 1.20, command=anchor)
    assert expanded.mode == 'searching'
    assert expanded.record['amplitude_rad'] == .09
    np.testing.assert_array_equal(expanded.target_command, [0., 0., 0., .09, 0., 0.])
    assert expanded.record['max_amplitude_rad'] == .18


def test_search_keeps_the_clipped_stop_anchor_and_has_no_elapsed_time_expiry():
    policy = MarkerSearchPolicy(SEARCH, marker_enabled=True)
    anchor = np.array([.1, -.2, .3, 2.995, -2.995, -.6])
    assert plan(policy, 0., command=anchor).mode == 'settling'
    searching = plan(policy, .25, command=anchor)
    assert searching.mode == 'searching'
    np.testing.assert_array_equal(searching.target_command, [*anchor[:3], 3., anchor[4], anchor[5]])
    expected_lower = anchor.copy(); expected_lower[[3, 4]] = [2.975, -3.]
    expected_upper = anchor.copy(); expected_upper[[3, 4]] = [3., -2.975]
    np.testing.assert_allclose(searching.position_lower, expected_lower, atol=1e-15)
    np.testing.assert_allclose(searching.position_upper, expected_upper, atol=1e-15)
    assert not searching.position_lower.flags.writeable and not searching.position_upper.flags.writeable
    # Elapsed time cannot pretend a roundtrip has completed or stop the new
    # expanding search after the old eight-second lifetime.
    moved = anchor.copy(); moved[3] -= .01
    for time in (1., 3., 8.25, 100., 1000.):
        chosen = plan(policy, time, command=moved, velocity=np.array([0., 0., 0., .01, 0., 0.]))
        assert chosen.mode == 'searching' and chosen.target_command is not None
        assert chosen.record['max_duration_s'] is None
        assert chosen.record['completed_roundtrips'] == 0
        assert chosen.record['amplitude_rad'] == SEARCH['amplitude_rad']
        np.testing.assert_array_equal(chosen.record['stop_anchor_command'], anchor)
        np.testing.assert_array_equal(chosen.position_lower, searching.position_lower)
        np.testing.assert_array_equal(chosen.position_upper, searching.position_upper)


def test_each_roundtrip_stops_and_settles_before_expansion_then_repeats_at_the_radius_cap():
    policy = MarkerSearchPolicy(SEARCH, marker_enabled=True)
    anchor = np.array([.1, -.2, .3, -.4, .5, -.6])
    initial = plan(policy, 0., command=anchor)
    assert initial.record['settle_reason'] == 'after_marker_loss'
    chosen = plan(policy, .25, command=anchor)
    time = .25
    directions = ((1., 0.), (0., 1.), (-1., 0.), (0., -1.))
    for roundtrip in range(80):
        radius = min(.12, .02+.01*roundtrip)
        assert chosen.mode == 'searching'
        assert chosen.record['settle_reason'] is None
        assert chosen.record['amplitude_rad'] == pytest.approx(radius)
        assert chosen.record['initial_amplitude_rad'] == .02
        assert chosen.record['amplitude_increment_rad'] == .01
        assert chosen.record['max_amplitude_rad'] == .12
        assert chosen.record['completed_roundtrips'] == roundtrip
        assert chosen.record['completed_pattern_cycles'] == roundtrip//4
        assert chosen.record['max_duration_s'] is None
        wanted = anchor.copy(); wanted[[3, 4]] += radius*np.array(directions[roundtrip%4])
        np.testing.assert_allclose(chosen.target_command, wanted, atol=1e-15)
        np.testing.assert_array_equal(chosen.record['stop_anchor_command'], anchor)
        np.testing.assert_array_equal(chosen.target_command[[0, 1, 2, 5]], anchor[[0, 1, 2, 5]])
        time += .05
        outward = chosen.target_command.copy()
        outward_hold = plan(policy, time, command=outward)
        assert outward_hold.mode == 'settling'
        assert outward_hold.record['settle_reason'] == 'outward_observation'
        assert outward_hold.record['event'] == 'stopped_at_outward_viewpoint'
        np.testing.assert_array_equal(outward_hold.target_command, outward)
        assert plan(policy, time+.249, command=outward).mode == 'settling'
        time += .25
        returning = plan(policy, time, command=outward)
        assert returning.mode == 'searching'
        np.testing.assert_array_equal(returning.target_command, anchor)
        assert returning.record['event'] == 'returning_to_search_center'
        # Proximity alone cannot begin a settle interval while the wrist is
        # still moving. This prevents a radius change during a reversal.
        time += .05
        moving = plan(policy, time, command=anchor,
            velocity=np.array([0., 0., 0., 1e-5, 0., 0.]))
        assert moving.mode == 'searching'
        assert moving.record['completed_roundtrips'] == roundtrip
        time += .05
        stopped = plan(policy, time, command=anchor)
        assert stopped.mode == 'settling'
        assert stopped.record['settle_reason'] == 'between_sweeps'
        assert stopped.record['event'] == 'stopped_between_search_sweeps'
        assert stopped.record['completed_roundtrips'] == roundtrip+1
        assert stopped.record['amplitude_rad'] == pytest.approx(radius)
        np.testing.assert_array_equal(stopped.target_command, anchor)
        before = plan(policy, time+.249, command=anchor)
        assert before.mode == 'settling'
        assert before.record['amplitude_rad'] == pytest.approx(radius)
        time += .25
        chosen = plan(policy, time, command=anchor)
        expected_radius = min(.12, .02+.01*(roundtrip+1))
        assert chosen.record['amplitude_rad'] == pytest.approx(expected_radius)
        assert chosen.record['event'] == ('search_amplitude_expanded' if radius < .12
                                          else 'bounded_wrist_search_repeated')
    assert chosen.record['completed_pattern_cycles'] == 20
    assert chosen.mode == 'searching' and chosen.record['amplitude_rad'] == .12


def test_reacquisition_at_an_inter_sweep_stop_resumes_without_another_outward_command():
    policy = MarkerSearchPolicy(SEARCH, marker_enabled=True)
    anchor = np.zeros(6)
    plan(policy, 0., command=anchor)
    outward = plan(policy, .25, command=anchor)
    plan(policy, .30, command=outward.target_command)
    returning = plan(policy, .55, command=outward.target_command)
    assert returning.mode == 'searching'
    stopped = plan(policy, .60, command=anchor)
    assert stopped.mode == 'settling' and stopped.record['settle_reason'] == 'between_sweeps'
    held = plan(policy, .65, command=anchor, visible=True)
    assert held.mode == 'settling'
    reacquired = plan(policy, .85, command=anchor, visible=True)
    assert reacquired.mode == 'reaching'
    assert reacquired.record['event'] == 'marker_reacquired_after_stopping'
    assert reacquired.target_command is None
    assert reacquired.record['completed_roundtrips'] == 1
    assert reacquired.record['amplitude_rad'] == .02


def test_reacquisition_brakes_holds_and_requires_a_post_stop_fused_image_then_new_loss_gets_new_anchor():
    policy = MarkerSearchPolicy(SEARCH, marker_enabled=True)
    anchor = np.zeros(6)
    plan(policy, 0., command=anchor)
    first = plan(policy, .25, command=anchor)
    plan(policy, .30, command=first.target_command)
    plan(policy, .55, command=first.target_command)
    plan(policy, .60, command=anchor)
    expanded = plan(policy, .85, command=anchor)
    assert expanded.record['amplitude_rad'] == .03
    assert expanded.record['completed_roundtrips'] == 1
    moving_command = expanded.target_command/2.
    velocity = np.array([0., 0., 0., 0., .01, 0.])
    original_velocity = velocity.copy()
    visible = plan(policy, .90, command=moving_command, velocity=velocity, visible=True,
                   observation_stamp=.89)
    assert visible.mode == 'braking' and visible.record['event'] == 'braking_after_marker_reacquisition'
    assert visible.target_command is None
    np.testing.assert_array_equal(visible.record['stop_anchor_command'], anchor)
    np.testing.assert_array_equal(velocity, original_velocity)
    stopped = plan(policy, .95, command=moving_command, visible=True, observation_stamp=.89)
    assert stopped.mode == 'settling'
    np.testing.assert_array_equal(stopped.target_command, moving_command)
    stale = plan(policy, 1.20, command=moving_command, visible=True, observation_stamp=.89)
    assert stale.mode == 'settling'
    assert not stale.record['marker_observation_since_stop']
    assert stale.record['marker_detected'] and stale.record['waiting_for_fused_marker_goal']
    assert stale.record['episode'] == 1
    # The stopped viewpoint and completed hold are retained while inference
    # catches up; a new fused goal can then resume without another hold.
    fresh = plan(policy, 1.21, command=moving_command, visible=True, observation_stamp=1.21)
    assert fresh.mode == 'reaching'
    resumed = plan(policy, 1.46, command=moving_command, visible=True, observation_stamp=1.40)
    assert resumed.mode == 'reaching'
    lost_again = plan(policy, 1.50, command=moving_command, velocity=velocity)
    assert lost_again.mode == 'braking'
    assert lost_again.record['episode'] == 2
    assert lost_again.record['stop_anchor_command'] is None
    assert lost_again.record['amplitude_rad'] == .02
    assert lost_again.record['completed_roundtrips'] == 0
    assert lost_again.record['completed_pattern_cycles'] == 0
    stopped_again = plan(policy, 1.55, command=moving_command)
    assert stopped_again.mode == 'settling'
    np.testing.assert_array_equal(stopped_again.record['stop_anchor_command'], moving_command)
    restarted = plan(policy, 1.80, command=moving_command)
    assert restarted.mode == 'searching'
    assert restarted.record['amplitude_rad'] == .02
    assert restarted.record['search_elapsed_s'] == 0.


def test_raw_detection_holds_the_stopped_viewpoint_until_a_post_stop_fused_goal():
    policy = MarkerSearchPolicy(SEARCH, marker_enabled=True)
    anchor = np.array([.1, -.2, .3, -.4, .5, -.6])
    stopped = plan(policy, 0., command=anchor, detected=True)
    assert stopped.mode == 'settling'
    assert stopped.record['marker_detected'] and not stopped.record['marker_observed']
    assert stopped.record['marker_detection_source'] == 'raw_image_preview'
    for time in (.25, .50, 1., 2.):
        waiting = plan(policy, time, command=anchor, detected=True)
        assert waiting.mode == 'settling'
        assert waiting.record['event'] == 'marker_detected_waiting_for_fused_goal'
        assert waiting.record['waiting_for_fused_marker_goal']
        np.testing.assert_array_equal(waiting.target_command, anchor)
        np.testing.assert_array_equal(waiting.record['stop_anchor_command'], anchor)
        assert waiting.record['observation_hold_started'] == 0.
        assert waiting.record['observation_hold_source_stamp'] == 1000.
        assert waiting.record['waypoint_index'] == 0
        assert waiting.record['search_elapsed_s'] is None
    stale = plan(policy, 2.1, command=anchor, detected=True, visible=True, observation_stamp=-.1)
    assert stale.mode == 'settling' and stale.record['waiting_for_fused_marker_goal']
    fresh = plan(policy, 2.2, command=anchor, detected=True, visible=True, observation_stamp=2.1)
    assert fresh.mode == 'reaching'
    assert not fresh.record['waiting_for_fused_marker_goal']
    assert fresh.record['marker_observation_since_stop']


def test_raw_reacquisition_brakes_search_before_inference_then_waits_without_moving_again():
    policy = MarkerSearchPolicy(SEARCH, marker_enabled=True)
    anchor = np.zeros(6)
    plan(policy, 0., command=anchor, detected=False)
    searching = plan(policy, .25, command=anchor, detected=False)
    assert searching.mode == 'searching'
    moved = .4*searching.target_command
    velocity = np.zeros(6); velocity[3] = .01
    raw = plan(policy, .30, command=moved, velocity=velocity, detected=True)
    assert raw.mode == 'braking'
    assert raw.record['event'] == 'braking_after_marker_reacquisition'
    assert raw.record['marker_detected'] and not raw.record['marker_observed']
    stopped = plan(policy, .35, command=moved, detected=True)
    assert stopped.mode == 'settling'
    waited = plan(policy, .60, command=moved, detected=True)
    assert waited.mode == 'settling'
    assert waited.record['waiting_for_fused_marker_goal']
    assert waited.record['observation_hold_source_stamp'] == 1000.35
    np.testing.assert_array_equal(waited.target_command, moved)
    np.testing.assert_array_equal(waited.record['stop_anchor_command'], anchor)
    resumed = plan(policy, .65, command=moved, detected=True, visible=True, observation_stamp=.60)
    assert resumed.mode == 'reaching'


def test_raw_loss_after_waiting_resumes_search_and_old_fused_success_cannot_override_pixels():
    policy = MarkerSearchPolicy(SEARCH, marker_enabled=True)
    anchor = np.zeros(6)
    plan(policy, 0., command=anchor, detected=True)
    assert plan(policy, .25, command=anchor, detected=True).mode == 'settling'
    vanished = plan(policy, .30, command=anchor, detected=False, visible=True, observation_stamp=.29)
    assert vanished.mode == 'searching'
    assert not vanished.record['marker_detected'] and vanished.record['marker_observed']
    assert vanished.record['marker_observation_since_stop']
    assert not vanished.record['waiting_for_fused_marker_goal']


def test_raw_loss_brakes_reaching_even_if_a_completed_fused_observation_is_still_positive():
    policy = MarkerSearchPolicy(SEARCH, marker_enabled=True)
    assert plan(policy, 0., visible=True, detected=True).mode == 'reaching'
    lost = plan(policy, .05, visible=True, detected=False, velocity=np.full(6, .01))
    assert lost.mode == 'braking'
    assert lost.record['event'] == 'braking_after_marker_loss'
    assert lost.record['marker_observed'] and not lost.record['marker_detected']


def test_repeated_repaint_does_not_advance_a_search_episode_or_count_as_new_evidence():
    policy = MarkerSearchPolicy(SEARCH, marker_enabled=True)
    result = policy_result(posterior=False, detector=False)
    first = policy.plan(result, np.zeros(6), 0., joint_lower=np.full(6, -3.), joint_upper=np.full(6, 3.))
    with pytest.raises(ValueError, match='increasing times'):
        policy.plan(result, np.zeros(6), 0., joint_lower=np.full(6, -3.), joint_upper=np.full(6, 3.))
    assert policy.episode == first.record['episode'] == 1


def test_search_planning_uses_no_sampling_or_physical_pose_and_does_not_mutate_the_input(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError('The marker reacquisition policy must be deterministic.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    policy = MarkerSearchPolicy(SEARCH, marker_enabled=True)
    command = np.linspace(-.1, .1, 6); velocity = np.zeros(6)
    result = policy_result(command, posterior=False, detector=False)
    untouched = dict(vars(result.prior)); estimate = result.local_estimates[0]
    for time in (0., .25, .3, 8.5, 9.):
        chosen = policy.plan(result, velocity, time, joint_lower=np.full(6, -3.), joint_upper=np.full(6, 3.))
        assert chosen.record['physical_truth_received'] is False
        assert chosen.record['monte_carlo_used'] is False
    assert vars(result.prior) == untouched
    assert result.local_estimates[0] is estimate
    np.testing.assert_array_equal(command, np.linspace(-.1, .1, 6))
    np.testing.assert_array_equal(velocity, np.zeros(6))


@pytest.mark.parametrize('mode', ['braking', 'settling', 'searching', 'search_exhausted'])
def test_local_actions_use_the_exact_same_K_corrected_laws_next_prior_and_published_queries(search_inputs, mode):
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal
    from probtf.bingham import bingham_log_normalizer
    from probtf.geometry import quat_mul, rotation_vector_to_quaternion
    from probtf.spherical_law.first_order import bingham_from_rotation_coordinates, pose_query

    scene, inputs, contexts = search_inputs
    inference = MovingInference(scene.robot_description, contexts['imu_marker'][0].templates, config_path=scene.legacy_inference_config)
    result = inference.update(inputs, 0)
    settings = yaml.safe_load((PACKAGE/'config/stiffness_identification.yaml').read_text())
    identifier = StiffnessIdentifier(scene.urdf, inference.body_ids, settings)
    result.records['stiffness'] = identifier.update(result)
    proposal = identifier.command_proposal(result)
    assert proposal is not None
    mapping = StiffnessCommandMap(identifier, result, proposal)
    selector = make_controller(scene, inference)
    source_command = np.array(result.prior.command)
    stage = result.stages[-1]
    body_before = stage.components[0].state
    goals_before = stage.goal_components
    memory_before = result.records['fixed_goal_memory']
    fresh_modes = result.records['fresh_goal_construction']['components']
    assert len(goals_before) == len(fresh_modes) == 2
    assert [item.log_weight for item in goals_before] == [item['image_log_weight'] for item in fresh_modes]
    selector.velocity[3] = .02 if mode in ('braking', 'search_exhausted') else 0.
    before_velocity = selector.velocity.copy()
    target = source_command.copy() if mode == 'settling' else None
    lower = upper = None
    if mode == 'searching':
        target = source_command.copy(); target[3] += .003
        lower = source_command.copy(); upper = source_command.copy()
        lower[[3, 4]] -= .02; upper[[3, 4]] += .02
    selected, diagnostics = selector.local_step(result, .05, mode=mode,
        target_command=target, max_speed=.025 if mode != 'settling' else None,
        position_lower=lower, position_upper=upper, command_map=mapping)
    velocity = (selected-source_command)/.05
    assert np.max(np.abs(velocity-before_velocity)) <= selector.acceleration*.05+1e-12
    assert diagnostics['action_mode'] == mode
    assert diagnostics['objective_evaluation_count'] == 1
    assert diagnostics['gradient_evaluation_count'] == 0
    assert diagnostics['objective_used_for_command_selection'] is False
    assert diagnostics['objective_gradient'] is None
    assert diagnostics['physical_truth_received'] is False
    assert result.stages[-1].components[0].state is body_before
    assert result.stages[-1].goal_components is goals_before
    assert result.records['fixed_goal_memory'] is memory_before
    assert diagnostics['body_source_state_ids'] == [body_before.state_id]
    assert diagnostics['goal_source_state_ids'] == [candidate.state.state_id for candidate in goals_before]
    assert diagnostics['source_log_weights'] == [candidate.log_weight for candidate in goals_before]
    assert diagnostics['selected_command_action'] == mapping.selection_record(selected)
    assert diagnostics['stiffness_command_correction'] == mapping.record()
    expected_body = mapping.push(body_before, selected, inference.body_ids, inference.actions)
    assert diagnostics['selected_body_state_ids'] == [expected_body.state_id]
    terms = []
    for candidate, branch in zip(goals_before, diagnostics['selected_branches']):
        expected = mapping.push(candidate.state, selected, inference.body_ids, inference.actions)
        query = pose_query(expected, selector.relative_path)
        assert branch['state_id'] == expected.state_id
        assert branch['parent_state_id'] == candidate.state.state_id
        np.testing.assert_array_equal(branch['query_mean'], query.mean)
        np.testing.assert_array_equal(branch['query_covariance'], query.covariance)
        parameter = bingham_from_rotation_coordinates(quat_mul(query.rotation_wxyz,
            rotation_vector_to_quaternion(query.correction[3:])), query.covariance[3:, 3:])
        position = multivariate_normal.logpdf(selector.goal_offset,
            query.mean[:3], query.covariance[:3, :3])
        orientation = (selector.goal_orientation@parameter@selector.goal_orientation
            -bingham_log_normalizer(parameter, integration_steps=selector.integration_steps))
        terms.append(candidate.log_weight+position+orientation)
        marker = candidate.state.edge_ids.index('marker')
        assert expected.transforms[marker] is candidate.state.transforms[marker]
        coordinates = slice(6*marker, 6*marker+6)
        np.testing.assert_array_equal(expected.covariance[coordinates, coordinates],
                                      candidate.state.covariance[coordinates, coordinates])
    np.testing.assert_allclose(diagnostics['log_expected_density'], logsumexp(terms), atol=2e-10)
    updated = advance_without_new_sensor_truth(inference, inputs, selected, 1, mapping)
    assert updated.records['prediction_command_correction'] == mapping.record()
    assert updated.records['prediction_command_action'] == diagnostics['selected_command_action']
    predicted = updated.stages[0].components[0].state
    np.testing.assert_array_equal(predicted.covariance, expected_body.covariance)
    for actual, expected in zip(predicted.transforms, expected_body.transforms):
        np.testing.assert_array_equal(actual.translation, expected.translation)
        if expected.bingham is not None:
            np.testing.assert_array_equal(actual.bingham, expected.bingham)
    assert updated.records['fixed_goal_memory'] == memory_before
    assert updated.records['fixed_goal_memory_event'] == 'image_missing_hold'
    snapshot = build_snapshot(updated, updated.stages[-1], inference.common)
    verify_adapter_identity(snapshot, updated.stages[-1])
    markers = make_markers(snapshot, inference.common['display']['seed'], inference.common['display']['point_size'])
    verify_wire_identity(snapshot, markers, seed=inference.common['display']['seed'],
                         point_size=inference.common['display']['point_size'])


def test_progressive_controller_sweeps_repeat_at_the_cap_without_anchor_drift_or_velocity_jumps(search_inputs, monkeypatch):
    scene, _, contexts = search_inputs
    inference, result = contexts['no_sensors']
    selector = make_controller(scene, inference)
    # Isolate the real constrained command interface with a flat diagnostic
    # density. No geometric objective can supply a lucky stopping direction.
    selector._query = lambda state: SimpleNamespace(command=state.command)
    selector._selection_sources = lambda *_args, **_kwargs: {}
    monkeypatch.setattr('probtf_observation_fusion_demo.moving_control._relative_density',
        lambda *_args: (0., {}))
    policy = MarkerSearchPolicy(SEARCH, marker_enabled=True, exposure_time=.18)
    anchor = np.array(result.prior.command)
    previous_velocity = selector.velocity.copy()
    cap_roundtrips = set(); modes = set(); last_roundtrips = 0
    maximum_deviation = 0.
    dt = .05
    for index in range(6000):
        command = np.array(result.prior.command)
        missing = SimpleNamespace(prior=result.prior, local_estimates=())
        chosen = policy.plan(missing, selector.velocity, index*dt,
            joint_lower=selector.lower, joint_upper=selector.upper)
        modes.add(chosen.mode)
        assert chosen.mode in ('settling', 'searching')
        radius = chosen.record['amplitude_rad']
        assert .02 <= radius <= .12
        assert radius == pytest.approx(min(.12, .02+.01*(
            chosen.record['completed_roundtrips']-(chosen.mode == 'settling'
                and chosen.record['settle_reason'] == 'between_sweeps'))))
        np.testing.assert_array_equal(chosen.record['stop_anchor_command'], anchor)
        if chosen.record['completed_roundtrips'] != last_roundtrips:
            assert chosen.mode == 'settling'
            assert np.max(np.abs(previous_velocity)) <= 1e-6
            assert np.max(np.abs(command-anchor)) <= 1e-4
            last_roundtrips = chosen.record['completed_roundtrips']
        selected, diagnostic = selector.local_step(result, dt, mode=chosen.mode,
            target_command=chosen.target_command, max_speed=chosen.max_speed,
            position_lower=chosen.position_lower, position_upper=chosen.position_upper)
        velocity = (selected-command)/dt
        assert np.max(np.abs(velocity)) <= .025+1e-12
        assert np.max(np.abs(velocity-previous_velocity)) <= selector.acceleration*dt+1e-12
        assert np.all(selected >= selector.lower-1e-12) and np.all(selected <= selector.upper+1e-12)
        np.testing.assert_array_equal(selected[[0, 1, 2, 5]], anchor[[0, 1, 2, 5]])
        assert np.max(np.abs(selected-anchor)) <= radius+1e-12
        maximum_deviation = max(maximum_deviation, float(np.max(np.abs(selected-anchor))))
        # Independently continue maximal braking from every selected command;
        # the active neighborhood must still admit a feasible stop.
        lower = selector.lower if chosen.position_lower is None else chosen.position_lower
        upper = selector.upper if chosen.position_upper is None else chosen.position_upper
        stopped_position, stopped_velocity = selected.copy(), velocity.copy()
        for _ in range(10):
            stopped_velocity = np.sign(stopped_velocity)*np.maximum(
                np.abs(stopped_velocity)-selector.acceleration*dt, 0.)
            stopped_position += dt*stopped_velocity
            assert np.all(stopped_position >= lower-1e-12)
            assert np.all(stopped_position <= upper+1e-12)
        assert np.max(np.abs(stopped_velocity)) < 1e-12
        if radius == .12:
            cap_roundtrips.add(chosen.record['completed_roundtrips'])
        stage = result.stages[-1]
        def predicted(candidate):
            future = command_state(candidate.state, selected, inference.body_ids, inference.actions)
            return replace(candidate, state=replace(future, stamp=1000.+(index+1)*dt, version=index+2))
        internal = tuple(predicted(candidate) for candidate in stage.components)
        goals = tuple(predicted(candidate) for candidate in stage.goal_components)
        result = SimpleNamespace(**dict(vars(result), prior=internal[0].state,
            stages=(replace(stage, components=internal, goal_components=goals),)))
        previous_velocity = velocity
        assert diagnostic['gradient_evaluation_count'] == 0
        if chosen.record['completed_roundtrips'] >= 24 and chosen.mode == 'searching':
            break
    else:
        pytest.fail('The bounded policy stopped making roundtrips before reaching the cap and repeating.')
    assert modes == {'settling', 'searching'}
    assert chosen.record['completed_pattern_cycles'] >= 6
    assert len(cap_roundtrips) >= 10
    assert maximum_deviation > .119
    assert chosen.record['search_elapsed_s'] > 8.
    assert chosen.record['max_duration_s'] is None


def test_fresh_rendered_reacquisition_after_search_updates_every_goal_mode_and_resumes_reaching(tmp_path):
    scene = legacy_marker_scene(tmp_path/'search_reacquire')
    inputs = scene.sensor_inputs(scene.step(0., 1000.))
    _, observations = load_inputs(PACKAGE/'config/common.yaml', PACKAGE/'config/marker.yaml')
    inference = MovingInference(scene.robot_description, observations, config_path=scene.legacy_inference_config)
    initial = inference.update(inputs, 0)
    assert marker_observed(initial)
    selector = make_controller(scene, inference)
    policy = MarkerSearchPolicy(SEARCH, marker_enabled=True, exposure_time=.18)
    first = policy.plan(initial, selector.velocity, 0., joint_lower=selector.lower, joint_upper=selector.upper)
    assert first.mode == 'reaching'
    command = np.array(initial.prior.command)
    missing = advance_without_new_sensor_truth(inference, inputs, command, 1)
    stopped = policy.plan(missing, selector.velocity, .05, joint_lower=selector.lower, joint_upper=selector.upper)
    assert stopped.mode == 'settling'
    # A test-only image dropout leaves the already observed root target fixed.
    still_missing_inputs = replace(inputs, time=.30, stamp=1000.30, command=command,
        image=np.full_like(inputs.image, 255))
    still_missing = inference.update(still_missing_inputs, 2)
    search = policy.plan(still_missing, selector.velocity, .30,
        joint_lower=selector.lower, joint_upper=selector.upper)
    assert search.mode == 'searching'
    selected, local = selector.local_step(still_missing, .05, mode=search.mode,
        target_command=search.target_command, max_speed=search.max_speed,
        position_lower=search.position_lower, position_upper=search.position_upper)
    assert np.linalg.norm(selected-command) > 0.
    assert still_missing.records['fixed_goal_memory'] == initial.records['fixed_goal_memory']
    # Render a new image from the same physical simulator after that actual
    # local command. No old image is relabelled or reused as a new observation.
    new_inputs = scene.sensor_inputs(scene.step(.35, 1000.35, command=selected))
    reacquired = inference.update(new_inputs, 3)
    assert marker_observed(reacquired)
    previous_velocity = selector.velocity.copy()
    resume = policy.plan(reacquired, selector.velocity, .35,
        joint_lower=selector.lower, joint_upper=selector.upper)
    assert resume.mode == 'braking' and resume.record['event'] == 'braking_after_marker_reacquisition'
    np.testing.assert_array_equal(selector.velocity, previous_velocity)
    assert reacquired.records['fixed_goal_memory_event'] == 'marker_observation_update'
    memory = reacquired.records['fixed_goal_memory']
    assert memory['observation_stamp'] == new_inputs.stamp
    assert memory['observation_command'] == selected.tolist()
    native_modes = next(e for e in reacquired.records['sensor_estimates'] if e['sensor'] == 'marker')['components']
    goals = reacquired.stages[-1].goal_components
    assert len(native_modes) == 2
    assert len(goals) == len(memory['goal_components']) == len(native_modes)
    assert [item.log_weight for item in goals] == [item['image_log_weight']
        for item in reacquired.records['fresh_goal_construction']['components']]
    assert [item.log_weight for item in goals] == [item['log_weight'] for item in memory['goal_components']]
    stopped_command, brake = selector.local_step(reacquired, .05, mode=resume.mode,
        target_command=resume.target_command, max_speed=resume.max_speed,
        position_lower=resume.position_lower, position_upper=resume.position_upper)
    assert np.max(np.abs(selector.velocity)) <= 1e-6
    stopped_inputs = scene.sensor_inputs(scene.step(.40, 1000.40, command=stopped_command))
    at_stop = inference.update(stopped_inputs, 4)
    stop = policy.plan(at_stop, selector.velocity, .40,
        joint_lower=selector.lower, joint_upper=selector.upper)
    assert stop.mode == 'settling'
    observed_inputs = scene.sensor_inputs(scene.step(.65, 1000.65, command=stopped_command))
    observed = inference.update(observed_inputs, 5)
    assert marker_observed(observed)
    resume = policy.plan(observed, selector.velocity, .65,
        joint_lower=selector.lower, joint_upper=selector.upper)
    assert resume.mode == 'reaching' and resume.record['event'] == 'marker_reacquired_after_stopping'
    goals = observed.stages[-1].goal_components
    previous_velocity = selector.velocity.copy()
    next_command, normal = selector.step(observed, .05)
    assert normal['gradient_evaluation_count'] == 1
    assert normal['source_state_ids'] == [item.state.state_id for item in goals]
    assert normal['fixed_goal_memory_available'] is True
    velocity = (next_command-stopped_command)/.05
    assert np.max(np.abs(velocity-previous_velocity)) <= selector.acceleration*.05+1e-12
