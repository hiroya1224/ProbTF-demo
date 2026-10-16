"""Recorded asynchronous transport preserves the actual inferred laws."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import pickle

import numpy as np
import pytest

from probtf.spherical_law.first_order import point_query
from probtf_observation_fusion_demo.display import make_markers, verify_wire_identity
from probtf_observation_fusion_demo.model import load_inputs
from probtf_observation_fusion_demo.moving_control import command_state, control_components
from probtf_observation_fusion_demo.moving_inference import MovingInference
from legacy_marker import legacy_marker_scene
from probtf_observation_fusion_demo.node import build_snapshot, verify_adapter_identity
from probtf_observation_fusion_demo.async_state import ActionStep, RecordedCommandPath, predict_result


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    scene = legacy_marker_scene(tmp_path_factory.mktemp('async_laws'))
    package = Path(__file__).resolve().parents[1]
    _, observations = load_inputs(package/'config/common.yaml', package/'config/imu_marker.yaml')
    inference = MovingInference(scene.robot_description, observations, config_path=scene.legacy_inference_config)
    result = inference.update(scene.sensor_inputs(scene.step(0., 1000.)), 0)
    assert len(result.stages[-1].components) == 1
    assert len(result.stages[-1].goal_components) > 1
    return inference, result


def steps_for(result, *, closed=False):
    initial = np.asarray(result.prior.command)
    increments = (np.array([.01, -.02, .003, 0., .001, .003]),
                  np.array([-.005, .01, .001, .002, 0., -.001]),
                  np.array([.002, 0., -.005, -.001, .002, 0.]))
    command, steps = initial.copy(), []
    for index, delta in enumerate(increments):
        end = initial.copy() if closed and index == 2 else command+delta
        # This fixture supplies declared recorded increments, not new physical
        # measurements or equilibrium-generated candidate distributions.
        action = (end-command)*(.8 if index < 2 else 1.1)
        steps.append(ActionStep(index, 1000.+.05*index, 1000.+.05*(index+1),
            command, end, action, density_id='K0' if index < 2 else 'K1',
            estimate_version=3 if index < 2 else 6, selected_K=np.full(6, 120.+index),
            anchor_command=command, anchor_equilibrium=command*.8,
            source_state_ids=tuple(item.state.state_id for item in result.stages[-1].components),
            source_stamp=1000., source_version=3,
            provenance={'fixture': ['known executed action', {'index': index}]}))
        command = end
    return tuple(steps)


def assert_law_equal(left, right):
    assert left.edge_ids == right.edge_ids
    np.testing.assert_allclose(left.covariance, right.covariance, rtol=1e-11, atol=1e-15)
    for a, b in zip(left.transforms, right.transforms):
        np.testing.assert_array_equal(a.translation, b.translation)
        np.testing.assert_array_equal(a.translation_covariance, b.translation_covariance)
        if a.bingham is not None:
            np.testing.assert_allclose(a.bingham, b.bingham, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('closed', [False, True])
def test_summed_recorded_action_matches_every_ordered_step_and_preserves_joint_goal_law(source, closed):
    inference, result = source
    steps = steps_for(result, closed=closed)
    path = RecordedCommandPath(result, steps, body_ids=inference.body_ids, actions=inference.actions)
    if closed:
        np.testing.assert_array_equal(path.target_command, result.prior.command)
        assert np.linalg.norm(path.joint_increment) > .001
    body, goals = control_components(result)
    for component in body+goals:
        state = component.state
        sequential = state
        for step in steps:
            sequential = command_state(sequential, step.end_command, inference.body_ids,
                inference.actions, joint_increment=step.joint_increment)
        summed = path.push(state, path.target_command, inference.body_ids, inference.actions)
        assert_law_equal(summed, sequential)
        expected = point_query(sequential, result.hand_path)
        query = point_query(summed, result.hand_path)
        np.testing.assert_allclose(query.mean, expected.mean, atol=1e-13)
        np.testing.assert_allclose(query.covariance, expected.covariance, rtol=1e-10, atol=1e-15)
        environment = state.edge_ids.index('marker')
        block = np.arange(6*environment, 6*environment+6)
        np.testing.assert_array_equal(summed.covariance[np.ix_(block, block)],
                                      state.covariance[np.ix_(block, block)])
        assert summed.transforms[environment] is state.transforms[environment]
        root_before, root_after = point_query(state, result.marker_path), point_query(summed, result.marker_path)
        np.testing.assert_array_equal(root_after.mean, root_before.mean)
        # The exact root parameters/P_EE above are bit-identical; the core's
        # full-state support decomposition can differ at floating roundoff.
        np.testing.assert_allclose(root_after.covariance, root_before.covariance,
                                   rtol=1e-13, atol=1e-17)
    assert path.record()['steps'][2]['density_id'] == 'K1'
    assert path.selection_record(path.target_command)['command_jacobian'] is None
    with pytest.raises(ValueError, match='not a future IK'):
        path.jacobian(path.target_command)


def test_action_step_is_deeply_immutable_and_records_have_no_aliases(source):
    _, result = source
    step = steps_for(result)[0]
    with pytest.raises(FrozenInstanceError):
        step.index = 8
    with pytest.raises(TypeError):
        step.provenance['fixture'] = ()
    with pytest.raises(TypeError):
        step.provenance['fixture'][1]['index'] = 9
    record = step.record()
    record['provenance']['fixture'][1]['index'] = 100
    record['selected_K'][0] = -1.
    assert step.record()['provenance']['fixture'][1]['index'] == 0
    assert step.selected_K[0] == 120.
    roundtrip = pickle.loads(pickle.dumps(step))
    assert roundtrip.record() == step.record()
    with pytest.raises(TypeError):
        roundtrip.provenance['fixture'][1]['index'] = 8


@pytest.mark.parametrize('failure', ['index_gap', 'duplicate', 'time_gap', 'time_overlap', 'command_gap'])
def test_command_history_requires_exact_intervals_without_gaps_overlaps_or_duplicates(source, failure):
    inference, result = source
    original = steps_for(result)
    steps = list(original)
    if failure == 'index_gap':
        steps[1] = replace(steps[1], index=9)
    elif failure == 'duplicate':
        steps[1] = original[0]
    elif failure == 'time_gap':
        steps[1] = replace(steps[1], start_stamp=steps[1].start_stamp+.001)
    elif failure == 'time_overlap':
        steps[1] = replace(steps[1], start_stamp=steps[1].start_stamp-.001)
    else:
        steps[1] = replace(steps[1], start_command=np.asarray(steps[1].start_command)+.001)
    with pytest.raises(ValueError, match='gap, overlap, duplicate or mismatched command'):
        RecordedCommandPath(result, steps, body_ids=inference.body_ids, actions=inference.actions)


def test_recorded_path_rejects_foreign_source_endpoint_or_axis(source):
    inference, result = source
    steps = steps_for(result)
    path = RecordedCommandPath(result, steps, body_ids=inference.body_ids, actions=inference.actions)
    state = result.stages[-1].components[0].state
    with pytest.raises(ValueError, match='exact frozen source'):
        path.push(replace(state, stamp=state.stamp+.001), path.target_command,
                  inference.body_ids, inference.actions)
    with pytest.raises(ValueError, match='another endpoint or joint action'):
        path.push(state, np.asarray(path.target_command)+.001, inference.body_ids, inference.actions)
    changed = [dict(item) for item in inference.actions]
    changed[0]['axis'] = [1., 0., 0.]
    with pytest.raises(ValueError, match='another endpoint or joint action'):
        path.push(state, path.target_command, inference.body_ids, changed)
    with pytest.raises(ValueError, match='different frozen source'):
        path.validate_result(replace(result, prior=replace(result.prior, stamp=result.prior.stamp+.001)))


def test_prediction_is_same_current_control_display_law_and_never_another_sensor_or_K_update(source, monkeypatch):
    inference, result = source
    steps = steps_for(result)
    original_ids = [item.state.state_id for item in result.stages[-1].components]
    original_goal_ids = [item.state.state_id for item in result.stages[-1].goal_components]
    original_records = result.records.copy()
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Prediction must not assimilate or sample anything.')
    monkeypatch.setattr(inference, 'update', forbidden)
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)
    predicted = predict_result(result, steps, steps[-1].end_stamp, 3,
                               inference.body_ids, inference.actions)
    body, goals = control_components(predicted)
    assert predicted.prior is body[0].state
    assert len(goals) == len(result.stages[-1].goal_components)
    assert [item.weight for item in goals] == [item.weight for item in result.stages[-1].goal_components]
    assert [item.log_weight for item in goals] == [item.log_weight for item in result.stages[-1].goal_components]
    assert all(item.state.stamp == steps[-1].end_stamp and item.state.command == steps[-1].end_command
               for item in body+goals)
    assert predicted.records['fixed_goal_memory'] == result.records['fixed_goal_memory']
    metadata = predicted.records['async_prediction']
    assert metadata['observation_source']['stamp'] == 1000.
    assert metadata['observation_source']['body_state_ids'] == original_ids
    assert metadata['present_stamp'] == steps[-1].end_stamp
    assert metadata['observations_assimilated'] == metadata['bayes_factor_updates'] == metadata['stiffness_factor_updates'] == 0
    assert predicted.sensor_posteriors == predicted.local_estimates == ()
    assert predicted.frozen_body_prior is predicted.frozen_body_posterior is None
    assert predicted.frozen_body_branches == ()
    snapshot = build_snapshot(predicted, predicted.stages[-1], inference.common)
    verify_adapter_identity(snapshot, predicted.stages[-1])
    settings = inference.common['display']
    message = make_markers(snapshot, settings['seed'], settings['point_size'])
    assert verify_wire_identity(snapshot, message, seed=settings['seed'], point_size=settings['point_size'])
    assert {point.query.state_id for point in snapshot.points if point.frame_label == 'hand'} == {body[0].state.state_id}
    assert {point.query.state_id for point in snapshot.points if point.frame_label == 'switch_goal'} == {item.state.state_id for item in goals}
    assert [item.state.state_id for item in result.stages[-1].components] == original_ids
    assert [item.state.state_id for item in result.stages[-1].goal_components] == original_goal_ids
    assert result.records == original_records


def test_hold_intervals_are_recorded_and_a_prediction_cannot_skip_time(source):
    inference, result = source
    hold = ActionStep(5, 1000., 1001., result.prior.command, result.prior.command, np.zeros(6))
    path = RecordedCommandPath(result, [hold], body_ids=inference.body_ids, actions=inference.actions)
    state = result.stages[-1].components[0].state
    assert path.push(state, result.prior.command, inference.body_ids, inference.actions) is state
    with pytest.raises(ValueError, match='timestamp must equal'):
        predict_result(result, [hold], 1001.001, 5, inference.body_ids, inference.actions)
    with pytest.raises(ValueError, match='timestamp must equal'):
        predict_result(result, [], 1001., 5, inference.body_ids, inference.actions)
    predicted = predict_result(result, [hold], 1001., 5, inference.body_ids, inference.actions)
    np.testing.assert_array_equal(predicted.prior.covariance, state.covariance)
    assert predicted.prior.stamp == 1001.


def test_prediction_of_a_prediction_preserves_original_observation_source_time(source):
    inference, result = source
    original = steps_for(result)
    first = predict_result(result, original[:1], original[0].end_stamp, 1,
                           inference.body_ids, inference.actions)
    second = predict_result(first, original[1:], original[-1].end_stamp, 3,
                            inference.body_ids, inference.actions)
    once = predict_result(result, original, original[-1].end_stamp, 3,
                          inference.body_ids, inference.actions)
    assert_law_equal(second.prior, once.prior)
    assert second.records['async_prediction']['observation_source'] == once.records['async_prediction']['observation_source']
    assert second.records['async_prediction']['observation_source']['stamp'] == 1000.
