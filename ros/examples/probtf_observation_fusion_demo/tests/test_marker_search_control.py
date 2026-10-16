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
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.node import build_snapshot, verify_adapter_identity
from probtf_observation_fusion_demo.stiffness_command import StiffnessCommandMap
from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier


PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def search_inputs(tmp_path_factory):
    scene = MovingSwitchScene(tmp_path_factory.mktemp('marker_search_control'))
    inputs = scene.sensor_inputs(scene.step(0., 1000.))
    contexts = {}
    for case in ('no_sensors', 'imu', 'marker', 'imu_marker'):
        _, observations = load_inputs(PACKAGE/'config/common.yaml', PACKAGE/'config'/(case+'.yaml'))
        inference = MovingInference(scene.robot_description, observations)
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
          'max_joint_velocity': .025, 'settle_time_s': .25, 'max_duration_s': 8.}


def policy_result(command=None, *, posterior=True, detector=True, stamp=1000., version=1):
    return SimpleNamespace(prior=SimpleNamespace(command=tuple(np.zeros(6) if command is None else command),
        stamp=stamp, version=version), local_estimates=(SimpleNamespace(
            posterior=object() if posterior else None,
            detector_result=object() if detector else None),))


def plan(policy, time, *, command=None, velocity=None, visible=False, lower=None, upper=None):
    result = policy_result(command, posterior=visible, detector=visible, stamp=1000.+time)
    return policy.plan(result, np.zeros(6) if velocity is None else velocity, time,
        joint_lower=np.full(6, -3.) if lower is None else lower,
        joint_upper=np.full(6, 3.) if upper is None else upper)


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


def test_search_has_a_fixed_clipped_anchor_and_expires_to_brake_and_hold():
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
    # A moving wrist does not move the centre or restart its search clock.
    moved = anchor.copy(); moved[3] -= .01
    for time in (1., 3., 8.24):
        chosen = plan(policy, time, command=moved, velocity=np.array([0., 0., 0., .01, 0., 0.]))
        assert chosen.mode == 'searching'
        np.testing.assert_array_equal(chosen.record['stop_anchor_command'], anchor)
        np.testing.assert_array_equal(chosen.position_lower, searching.position_lower)
        np.testing.assert_array_equal(chosen.position_upper, searching.position_upper)
    expired = plan(policy, 8.25, command=moved, velocity=np.array([0., 0., 0., .01, 0., 0.]))
    assert expired.mode == 'search_exhausted'
    assert expired.target_command is None
    assert expired.record['event'] == 'search_limit_brake_and_hold'
    for index in range(1, 1001):
        held = plan(policy, 8.25+.05*index, command=moved)
        assert held.mode == 'search_exhausted' and held.target_command is None
        assert held.record['episode'] == 1
        np.testing.assert_array_equal(held.record['stop_anchor_command'], anchor)
        np.testing.assert_array_equal(held.position_lower, expired.position_lower)
        np.testing.assert_array_equal(held.position_upper, expired.position_upper)


def test_each_small_sweep_returns_through_anchor_and_the_pattern_runs_once():
    policy = MarkerSearchPolicy(SEARCH, marker_enabled=True)
    anchor = np.array([.1, -.2, .3, -.4, .5, -.6])
    plan(policy, 0., command=anchor)
    chosen = plan(policy, .25, command=anchor)
    targets = []
    for index in range(8):
        assert chosen.mode == 'searching'
        target = chosen.target_command.copy(); targets.append(target)
        assert np.max(np.abs(target-anchor)) <= SEARCH['amplitude_rad']+1e-15
        np.testing.assert_array_equal(target[[0, 1, 2, 5]], anchor[[0, 1, 2, 5]])
        chosen = plan(policy, .30+.05*index, command=target)
    assert chosen.mode == 'search_exhausted' and chosen.target_command is None
    for index in (1, 3, 5, 7):
        np.testing.assert_array_equal(targets[index], anchor)
    assert targets[0][3] > anchor[3] and targets[2][3] < anchor[3]
    assert targets[4][4] > anchor[4] and targets[6][4] < anchor[4]
    assert chosen.record['waypoint_index'] == chosen.record['waypoint_count'] == 8


def test_reacquisition_resumes_from_current_velocity_then_a_new_loss_gets_a_new_anchor():
    policy = MarkerSearchPolicy(SEARCH, marker_enabled=True)
    anchor = np.zeros(6)
    plan(policy, 0., command=anchor)
    first = plan(policy, .25, command=anchor)
    moving_command = first.target_command/2.
    velocity = np.array([0., 0., 0., .01, 0., 0.])
    original_velocity = velocity.copy()
    visible = plan(policy, .30, command=moving_command, velocity=velocity, visible=True)
    assert visible.mode == 'reaching' and visible.record['event'] == 'marker_reacquired'
    assert visible.target_command is visible.position_lower is visible.position_upper is None
    np.testing.assert_array_equal(velocity, original_velocity)
    lost_again = plan(policy, .35, command=moving_command, velocity=velocity)
    assert lost_again.mode == 'braking'
    assert lost_again.record['episode'] == 2
    assert lost_again.record['stop_anchor_command'] is None
    stopped_again = plan(policy, .40, command=moving_command)
    assert stopped_again.mode == 'settling'
    np.testing.assert_array_equal(stopped_again.record['stop_anchor_command'], moving_command)


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
    inference = MovingInference(scene.robot_description, contexts['imu_marker'][0].templates)
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
    assert len(goals_before) == 2
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


def test_long_local_sweeps_and_expiry_braking_stay_in_the_neighborhood_and_within_acceleration(monkeypatch):
    from test_goal_mixture_criterion import _problem
    inference, result, selector = _problem()
    # Isolate the real command interface with a flat diagnostic density. The
    # action cannot benefit from a geometric objective that happens to brake.
    selector._query = lambda state: SimpleNamespace(command=state.command)
    selector._selection_sources = lambda *_args, **_kwargs: {}
    monkeypatch.setattr('probtf_observation_fusion_demo.moving_control._relative_density',
        lambda *_args: (0., {}))
    anchor = np.array(result.prior.command)
    lower, upper = anchor-.02, anchor+.02
    previous_velocity = selector.velocity.copy()
    reversals, previous_sign = 0, 0.
    extrema = [anchor[0], anchor[0]]
    target = upper.copy()
    dt = .05
    for index in range(1200):
        command = np.array(result.prior.command)
        mode = 'searching' if index < 900 else 'search_exhausted'
        if mode == 'searching' and abs(command[0]-target[0]) < 1e-4 and abs(previous_velocity[0]) < 1e-3:
            target = lower.copy() if target[0] > anchor[0] else upper.copy()
        selected, diagnostic = selector.local_step(result, dt, mode=mode,
            target_command=target if mode == 'searching' else None,
            max_speed=.025, position_lower=lower, position_upper=upper)
        velocity = (selected-command)/dt
        assert lower[0]-1e-12 <= selected[0] <= upper[0]+1e-12
        assert abs(velocity[0]) <= .025+1e-12
        assert abs(velocity[0]-previous_velocity[0]) <= selector.acceleration*dt+1e-12
        # Every selected command admits an independent maximal-braking
        # continuation without crossing either local boundary.
        stopped_position, stopped_velocity = selected.copy(), velocity.copy()
        for _ in range(10):
            stopped_velocity = np.sign(stopped_velocity)*np.maximum(
                np.abs(stopped_velocity)-selector.acceleration*dt, 0.)
            stopped_position += dt*stopped_velocity
            assert lower[0]-1e-12 <= stopped_position[0] <= upper[0]+1e-12
        assert abs(stopped_velocity[0]) < 1e-12
        sign = np.sign(velocity[0])
        if sign and previous_sign and sign != previous_sign:
            reversals += 1
        if sign:
            previous_sign = sign
        extrema = [min(extrema[0], selected[0]), max(extrema[1], selected[0])]
        stage = result.stages[-1]
        components = tuple(replace(candidate, state=command_state(candidate.state,
            selected, inference.body_ids, inference.actions)) for candidate in stage.components)
        result = SimpleNamespace(**dict(vars(result), prior=components[0].state,
            stages=(replace(stage, components=components),)))
        previous_velocity = velocity
        if index > 902:
            np.testing.assert_array_equal(velocity, np.zeros(1))
        assert diagnostic['gradient_evaluation_count'] == 0
    assert reversals >= 10
    assert extrema[0] < anchor[0]-.019 and extrema[1] > anchor[0]+.019


def test_fresh_rendered_reacquisition_after_search_updates_every_goal_mode_and_resumes_reaching(tmp_path):
    scene = MovingSwitchScene(tmp_path/'search_reacquire')
    inputs = scene.sensor_inputs(scene.step(0., 1000.))
    _, observations = load_inputs(PACKAGE/'config/common.yaml', PACKAGE/'config/marker.yaml')
    inference = MovingInference(scene.robot_description, observations)
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
    assert resume.mode == 'reaching' and resume.record['event'] == 'marker_reacquired'
    np.testing.assert_array_equal(selector.velocity, previous_velocity)
    assert reacquired.records['fixed_goal_memory_event'] == 'marker_observation_update'
    memory = reacquired.records['fixed_goal_memory']
    assert memory['observation_stamp'] == new_inputs.stamp
    assert memory['observation_command'] == selected.tolist()
    native_modes = next(e for e in reacquired.records['sensor_estimates'] if e['sensor'] == 'marker')['components']
    goals = reacquired.stages[-1].goal_components
    assert len(goals) == len(memory['goal_components']) == len(native_modes) == 2
    assert [item.log_weight for item in goals] == [item['log_weight'] for item in memory['goal_components']]
    next_command, normal = selector.step(reacquired, .05)
    assert normal['gradient_evaluation_count'] == 1
    assert normal['source_state_ids'] == [item.state.state_id for item in goals]
    assert normal['fixed_goal_memory_available'] is True
    velocity = (next_command-selected)/.05
    assert np.max(np.abs(velocity-previous_velocity)) <= selector.acceleration*.05+1e-12
