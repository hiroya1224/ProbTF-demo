"""Centering proposes a start; complete-objective IK selects every command."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from probtf.geometry import axis_angle_to_quat
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, bingham_from_rotation_coordinates,
    pose_query,
)
from probtf_observation_fusion_demo.model import Stage, WeightedState
from probtf_observation_fusion_demo.moving_control import MovingController, command_state
from probtf_observation_fusion_demo.visual_approach import VisualApproach


PACKAGE = Path(__file__).resolve().parents[1]


def _problem(markers=((.3, .08, 1.),), weights=None, *, position_variance=1e-5,
             rotation_variance=1e-5, log_weights=None):
    command = (0., 0.)
    axes = ([0., 1., 0.], [1., 0., 0.])
    bodies = tuple(BinghamTransform('joint_'+str(index), [0., 0., 0.],
        bingham_from_rotation_coordinates(axis_angle_to_quat(axis, command[index]),
                                          rotation_variance*np.eye(3)),
        position_variance*np.eye(3)) for index, axis in enumerate(axes))
    tip = BinghamTransform('tip', [0., 0., .4], rotation_wxyz=[1., 0., 0., 0.])
    states = []
    for index, marker in enumerate(markers):
        transform = BinghamTransform('marker', marker,
            bingham_from_rotation_coordinates([1., 0., 0., 0.],
                                              rotation_variance*np.eye(3)),
            position_variance*np.eye(3))
        states.append(JointTransformState.from_independent(bodies+(tip, transform),
            command=command, stamp=1234.25, version=9,
            component_id='native_'+str(index)))
    weights = tuple([1./len(markers)]*len(markers)) if weights is None else tuple(weights)
    log_weights = tuple(np.log(weights)) if log_weights is None else tuple(log_weights)
    goals = tuple(WeightedState('native_'+str(index), weight, state, log_weight)
        for index, (state, weight, log_weight) in enumerate(zip(states, weights, log_weights)))
    internal = WeightedState('internal', 1., replace(states[0], component_id='internal'), 0.)
    camera_path = (('joint_0', False), ('joint_1', False))
    hand_path = camera_path+(('tip', False),)
    marker_path = (('marker', False),)
    inference = SimpleNamespace(body_ids=('joint_0', 'joint_1'),
        actions=tuple(dict(joint_index=index, axis=axis) for index, axis in enumerate(axes)),
        paths=dict(hand=hand_path, camera=camera_path, marker=marker_path),
        settings=dict(root_frame='base'))
    result = SimpleNamespace(prior=states[0],
        stages=(Stage('fused', (internal,), goal_components=goals),),
        camera_path=camera_path, marker_path=marker_path,
        records=dict(observation_ids=['image:1'],
                     fixed_goal_memory=dict(available=True, observation_stamp=1234.25)))
    joint = '<joint name="{}" type="revolute"><limit lower="-3" upper="3"/></joint>'
    urdf = '<robot name="two">'+''.join(joint.format(index) for index in range(2))+'</robot>'
    controller = MovingController(inference, urdf, [0., 0., 0.], [1., 0., 0., 0.],
        dict(max_joint_velocity=.048, max_joint_acceleration=.3, damping=.025))
    return inference, result, controller


def test_all_modes_full_covariances_and_source_metadata_are_actual_core_queries(monkeypatch):
    _, result, controller = _problem(markers=((.3, .08, 1.), (.2, -.04, 1.1)),
                                      weights=(.2, .8))
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Visual control must not sample any source candidate.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    before = [(item.state.state_id, item.state.covariance.copy())
              for item in result.stages[-1].goal_components]
    target, record = VisualApproach(controller).propose(result, .5)
    assert target is not None
    assert record['goal_source_weights'] == [.2, .8]
    assert record['goal_source_log_weights'] == list(np.log([.2, .8]))
    assert record['mixture_collapsed'] is False
    assert record['physical_truth_received'] is record['monte_carlo_used'] is False
    assert record['full_gaussian_bingham_criterion_used_for_selection'] is False
    assert record['field_of_view_guaranteed'] is False
    camera_path = tuple((name, not inverse) for name, inverse in reversed(result.camera_path))+result.marker_path
    squared = 0.
    for component, branch, (source_id, covariance) in zip(
            result.stages[-1].goal_components, record['branches'], before):
        expected = pose_query(component.state, camera_path,
            query_id='visual:camera_relative_marker', frame='camera')
        for name in ('mean', 'covariance', 'jacobian', 'cross_covariance', 'within', 'between'):
            np.testing.assert_array_equal(branch['camera_relative_marker'][name], getattr(expected, name))
        source = branch['camera_relative_marker']
        assert source['source_state_id'] == source_id == component.state.state_id
        assert source['source_stamp'] == result.prior.stamp
        assert source['source_command'] == list(result.prior.command)
        assert source['source_version'] == result.prior.version
        np.testing.assert_array_equal(component.state.covariance, covariance)
        hand = pose_query(component.state, controller.relative_path)
        squared += component.weight*(hand.mean[:3] @ hand.mean[:3]+np.trace(hand.covariance[:3, :3]))
    np.testing.assert_allclose(record['hand_goal_rms_distance_m'], np.sqrt(squared), rtol=1e-14)
    json.dumps(record, allow_nan=False)


def test_edge_marker_gates_progression_and_requested_velocity_reduces_bearing():
    inference, result, controller = _problem()
    target, record = VisualApproach(controller).propose(result, .5)
    assert record['progression_gate'] == 0.
    velocity = np.asarray(record['requested_velocity'])
    assert np.linalg.norm(velocity) > 0.
    assert np.max(np.abs(velocity)) <= controller.speed
    np.testing.assert_allclose(target, np.array(result.prior.command)+.5*velocity)
    branch = record['branches'][0]
    bearing = np.array(branch['normalized_bearing_mean'])
    derivative = np.array(branch['normalized_bearing_command_jacobian'])
    assert bearing @ derivative @ velocity < 0.
    state = command_state(result.stages[-1].goal_components[0].state,
        np.array(result.prior.command)+.001*velocity, inference.body_ids, inference.actions)
    path = tuple((name, not inverse) for name, inverse in reversed(result.camera_path))+result.marker_path
    actual = pose_query(state, path).mean[:3]
    assert np.linalg.norm(actual[:2]/actual[2]) < np.linalg.norm(bearing)


def test_centered_marker_allows_position_progression_and_large_error_stops_it():
    _, centered, controller = _problem(markers=((.02, .01, 1.),))
    _, small = VisualApproach(controller).propose(centered, .5)
    assert small['progression_gate'] == 1.
    _, middle, middle_controller = _problem(markers=((.125, 0., 1.),))
    _, partial = VisualApproach(middle_controller).propose(middle, .5)
    assert 0. < partial['progression_gate'] < 1.
    _, edge, edge_controller = _problem(markers=((.4, 0., 1.),))
    _, stopped = VisualApproach(edge_controller).propose(edge, .5)
    assert stopped['progression_gate'] == 0.


def test_opposing_native_modes_keep_expected_loss_and_both_weights_affect_control():
    _, equal, controller = _problem(markers=((.3, .08, 1.), (-.3, -.08, 1.)), weights=(.5, .5))
    _, record = VisualApproach(controller).propose(equal, .5)
    bearings = np.array([item['normalized_bearing_mean'] for item in record['branches']])
    np.testing.assert_allclose(np.array([.5, .5]) @ bearings, 0., atol=1e-14)
    assert record['centering_squared_loss_expectation'] > .09
    assert record['normalized_bearing_mean_rms'] > .3
    assert len(record['branches']) == 2
    _, unequal, other = _problem(markers=((.3, .08, 1.), (-.3, -.08, 1.)), weights=(.9, .1))
    _, changed = VisualApproach(other).propose(unequal, .5)
    assert np.linalg.norm(np.array(record['requested_velocity'])-changed['requested_velocity']) > .01


def test_near_stops_visual_proposals_using_every_mode_and_within_covariance_then_latches():
    _, result, controller = _problem(markers=((0., 0., .41),), position_variance=1e-7)
    servo = VisualApproach(controller, {'near_distance_m': .18})
    target, record = servo.propose(result, .5)
    assert target is None and record['near_latched']
    assert record['hand_goal_rms_distance_m'] < .18
    _, far, _ = _problem(markers=((.3, 0., 1.),))
    target, retained = servo.propose(far, .5)
    assert target is None and retained['selection_reason'] == 'near phase already latched'

    _, two, controller = _problem(markers=((0., 0., .41), (0., 0., 1.)), weights=(.5, .5))
    target, record = VisualApproach(controller, {'near_distance_m': .18}).propose(two, .5)
    assert target is not None and not record['near_latched']
    assert record['hand_goal_rms_distance_m'] > .4
    _, broad, controller = _problem(markers=((0., 0., .41),), position_variance=.02)
    target, record = VisualApproach(controller, {'near_distance_m': .18}).propose(broad, .5)
    assert target is not None and not record['near_latched']
    assert record['hand_goal_rms_distance_m'] > .18


def test_worse_visual_start_is_rejected_and_reaching_still_computes_full_gradient(monkeypatch):
    _, result, controller = _problem(markers=((0., 0., .4),),
                                      position_variance=1e-7)
    # The current hand is already at this target with the requested orientation.
    # A deliberately bad centering nomination must not become an executed step.
    _, visual_record = VisualApproach(controller).propose(result, .5)
    proposed = np.array([10., -10.])
    controller.visual_approach_policy = SimpleNamespace(
        propose=lambda *_args, **_kwargs: (proposed, dict(visual_record)))
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Reaching cannot execute a visual velocity directly.')
    monkeypatch.setattr(controller, 'local_step', forbidden)
    selected, diagnostics = controller.reaching_step(result, .5)
    initial = diagnostics['approach_initial_proposal']
    assert initial['available'] and not initial['accepted_as_initial_point']
    assert initial['proposed_objective'] > initial['comparison_objective']
    np.testing.assert_array_equal(initial['requested_command'], proposed)
    bounds = np.asarray(diagnostics['increment_bounds'])
    np.testing.assert_array_equal(initial['projected_command'],
        np.asarray(result.prior.command)+np.clip(proposed-np.asarray(result.prior.command),
                                                bounds[:, 0], bounds[:, 1]))
    assert not np.array_equal(selected, initial['projected_command'])
    assert diagnostics['gradient_evaluation_count'] == 1
    np.testing.assert_array_equal(diagnostics['gradient_at_command'], result.prior.command)
    assert diagnostics['reaching_phase'] == 'indet_ik'
    assert diagnostics['visual_approach']['full_gaussian_bingham_criterion_used_for_selection']
    assert diagnostics['visual_approach']['visual_velocity_executed_directly'] is False
    assert -diagnostics['log_expected_density'] <= diagnostics['braking_baseline_objective']+1e-12


def test_improving_visual_start_is_followed_by_one_full_objective_gradient():
    _, result, ordinary = _problem(markers=((.1, .08, .7),))
    useful, reference = ordinary.step(result, .5)
    assert -reference['log_expected_density'] < reference['braking_baseline_objective']
    _, _, restarted = _problem(markers=((.1, .08, .7),))
    selected, diagnostics = restarted.step(result, .5, initial_command=useful)
    initial = diagnostics['approach_initial_proposal']
    assert initial['accepted_as_initial_point']
    np.testing.assert_array_equal(diagnostics['gradient_at_command'], useful)
    assert diagnostics['gradient_evaluation_count'] == 1
    assert diagnostics['log_expected_density'] >= reference['log_expected_density']-1e-12
    assert -diagnostics['log_expected_density'] <= diagnostics['feasible_baseline_objective']+1e-12
    assert np.max(np.abs(selected-np.asarray(result.prior.command))) <= restarted.speed*.5+1e-12


@pytest.mark.parametrize('proposal', ([np.nan, 0.], [np.inf, 0.], [0., 0., 0.]))
def test_invalid_visual_start_cannot_enter_complete_objective_IK(proposal):
    _, result, controller = _problem()
    with pytest.raises(ValueError, match='finite target per joint'):
        controller.step(result, .5, initial_command=proposal)


def test_underflow_mode_is_recorded_and_invalid_depth_never_prunes_or_approaches():
    _, result, controller = _problem(markers=((.3, .08, 1.), (.1, .05, -1.)),
        weights=(1., 0.), log_weights=(0., -1000.))
    target, record = VisualApproach(controller).propose(result, .5)
    np.testing.assert_array_equal(target, result.prior.command)
    assert len(record['branches']) == 2
    assert record['branches'][1]['log_weight'] == -1000.
    assert record['branches'][1]['weight'] == 0.
    assert not record['all_modes_positive_depth']
    np.testing.assert_array_equal(record['requested_velocity'], [0., 0.])


@pytest.fixture(scope='module')
def corrected(tmp_path_factory):
    from probtf_observation_fusion_demo.moving_inference import MovingInference
    from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
    from probtf_observation_fusion_demo.stiffness_command import StiffnessCommandMap
    from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier
    scene = MovingSwitchScene(tmp_path_factory.mktemp('visual_corrected'), imu_target_link='tool0')
    inputs = scene.sensor_inputs(scene.step(0., 1000.))
    observations = yaml.safe_load((PACKAGE/'config/imu_marker.yaml').read_text())['observations']
    inference = MovingInference(scene.robot_description, observations)
    result = inference.update(inputs, 0)
    settings = yaml.safe_load((PACKAGE/'config/stiffness_identification.yaml').read_text())
    identifier = StiffnessIdentifier(scene.urdf, inference.body_ids, settings)
    result.records['stiffness'] = identifier.update(result)
    proposal = identifier.command_proposal(result)
    mapping = StiffnessCommandMap(identifier, result, proposal)
    control = inference.settings['control']
    controller = MovingController(inference, scene.robot_description,
        control['goal_offset'], control['goal_orientation_wxyz'], control)
    return inference, result, controller, mapping


def test_real_same_K_map_drives_analytic_camera_jacobian_and_retains_all_native_sources(corrected):
    inference, result, controller, mapping = corrected
    _, record = VisualApproach(controller, {'near_distance_m': .001}).propose(
        result, .5, command_map=mapping)
    np.testing.assert_array_equal(record['command_jacobian'], mapping.jacobian(result.prior.command))
    assert np.linalg.norm(np.array(record['command_jacobian'])-np.eye(6)) > 1e-3
    assert record['stiffness_command_correction'] == mapping.record()
    assert record['command_action'] == 'same frozen estimated-K command action'
    assert record['source_version'] == result.stages[-1].components[0].state.version
    assert record['prior_source_version'] == result.prior.version
    memory = result.records['fixed_goal_memory']
    assert record['goal_acquisition']['observation_stamp'] == memory['observation_stamp']
    assert record['goal_acquisition']['observation_command'] == memory['observation_command']
    image = memory['image_relative_source']
    assert record['native_image_source']['observation_id'] == image['observation_id']
    assert record['native_image_source']['capture_command'] == image['capture_command']
    assert len(record['native_image_components']) == len(image['components'])
    for recorded, original in zip(record['native_image_components'], image['components']):
        for key in ('native_component_id', 'weight', 'log_weight', 'image_only_covariance', 'bingham'):
            assert recorded[key] == original[key]
    if record['native_image_components']:
        saved = image['components'][0]['image_only_covariance'][0][0]
        record['native_image_components'][0]['image_only_covariance'][0][0] += 1.
        assert image['components'][0]['image_only_covariance'][0][0] == saved
    components = result.stages[-1].goal_components
    assert len(record['branches']) == len(components)
    assert record['goal_source_state_ids'] == [item.state.state_id for item in components]
    for component, branch in zip(components, record['branches']):
        numerical = np.zeros((3, 6))
        step = 1e-5
        path = tuple((name, not inverse) for name, inverse in reversed(result.camera_path))+result.marker_path
        for index in range(6):
            delta = step*np.eye(6)[index]
            plus = mapping.push(component.state, np.array(result.prior.command)+delta,
                                inference.body_ids, inference.actions)
            minus = mapping.push(component.state, np.array(result.prior.command)-delta,
                                 inference.body_ids, inference.actions)
            numerical[:, index] = (pose_query(plus, path).mode[:3]
                                  -pose_query(minus, path).mode[:3])/(2*step)
        np.testing.assert_allclose(branch['camera_position_command_jacobian'], numerical,
                                   rtol=3e-5, atol=3e-7)


@pytest.mark.parametrize('dt', [0., -.1, np.nan])
def test_invalid_intervals_fail_instead_of_issuing_a_command(dt):
    _, result, controller = _problem()
    with pytest.raises(ValueError, match='positive finite'):
        VisualApproach(controller).propose(result, dt)


def test_foreign_goal_metadata_and_bad_K_jacobian_are_rejected():
    _, result, controller = _problem()
    stage = result.stages[-1]
    goal = stage.goal_components[0]
    foreign = replace(goal, state=replace(goal.state, stamp=goal.state.stamp+.1))
    changed = SimpleNamespace(**dict(vars(result), stages=(replace(stage, goal_components=(foreign,)),)))
    with pytest.raises(ValueError, match='same internal body marginal'):
        VisualApproach(controller).propose(changed, .5)
    class BadMap:
        def validate_result(self, _result):
            pass
        def jacobian(self, _command):
            return np.full((2, 2), np.nan)
    with pytest.raises(ValueError, match='command-action Jacobian'):
        VisualApproach(controller).propose(result, .5, command_map=BadMap())
