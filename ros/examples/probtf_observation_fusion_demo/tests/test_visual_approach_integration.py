"""Every reaching update selects the complete criterion on the displayed law/K."""
from dataclasses import replace
import io
import json
from pathlib import Path
from queue import Queue

import numpy as np
import pytest
import yaml
from visualization_msgs.msg import MarkerArray

from probtf_observation_fusion_demo.async_motion import control_worker
from probtf_observation_fusion_demo.execution_stiffness import LogStiffnessDamper
from probtf_observation_fusion_demo.async_state import predict_result
from probtf_observation_fusion_demo.async_stiffness import (
    PhysicalActionEngine, capture_stiffness_packet, reanchor_retained_stiffness,
)
from probtf_observation_fusion_demo.display import make_markers, verify_wire_identity
from probtf_observation_fusion_demo.moving_control import MovingController, _relative_density
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.node import build_snapshot
from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier
from probtf_observation_fusion_demo.visual_approach import VisualApproach

from test_async_execution import session, _complete_control, _complete_estimate


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE_KEYS = ('source_stamp', 'source_command', 'source_version',
               'body_source_state_ids', 'goal_source_state_ids')


@pytest.fixture(scope='module')
def problems(tmp_path_factory):
    scene = MovingSwitchScene(tmp_path_factory.mktemp('visual_integration'),
                              imu_target_link='tool0')
    inputs = scene.sensor_inputs(scene.step(0., 1792080000.))
    stiffness = yaml.safe_load((PACKAGE/'config/stiffness_identification.yaml').read_text())
    result = {}
    for name in ('imu_marker', 'imu', 'no_sensors'):
        templates = yaml.safe_load((PACKAGE/('config/'+name+'.yaml')).read_text())['observations']
        inference = MovingInference(scene.robot_description, templates)
        source = inference.update(inputs, 0)
        identifier = StiffnessIdentifier(scene.urdf, inference.body_ids, stiffness)
        source.records['stiffness'] = identifier.update(source)
        proposal = identifier.command_proposal(source)
        packet = None if proposal is None else capture_stiffness_packet(source)
        predicted = predict_result(source, (), target_stamp=source.prior.stamp,
            frame_index=0, body_ids=inference.body_ids, actions=inference.actions)
        if packet is None:
            view, mapping = predicted, None
        else:
            engine = PhysicalActionEngine(scene.urdf, inference.body_ids, stiffness)
            view, mapping = reanchor_retained_stiffness(engine, predicted, packet)
        result[name] = scene, inference, source, view, mapping, packet, templates
    return result


def controller(problem, **visual):
    scene, inference, *_ = problem
    settings = dict(inference.settings['control'])
    settings['visual_approach'] = dict(settings.get('visual_approach', {}), **visual)
    return MovingController(inference, scene.robot_description, settings['goal_offset'],
        settings['goal_orientation_wxyz'], settings,
        integration_steps=settings['integration_steps'])


def forbid_sampling(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('The control/inference candidate law must never be sampled.')
    for name in ('default_rng', 'normal', 'random', 'randn', 'multivariate_normal', 'RandomState'):
        monkeypatch.setattr(np.random, name, forbidden)


def test_far_wrapper_uses_full_IK_keeps_every_goal_same_K_law_and_shared_bounds(problems, monkeypatch):
    problem = problems['imu_marker']
    _, inference, _, view, mapping, _, _ = problem
    current = controller(problem, near_distance_m=.001)
    original = [(item.state.state_id, item.state.covariance.copy())
                for item in view.stages[-1].components+view.stages[-1].goal_components]
    forbid_sampling(monkeypatch)
    target, _ = VisualApproach(current, {'near_distance_m': .001}).propose(
        view, .5, command_map=mapping)
    reference = controller(problem, near_distance_m=.001)
    expected, expected_record = reference.step(view, .5, command_map=mapping,
                                               initial_command=target)
    def direct_visual_execution_forbidden(*args, **kwargs):
        raise AssertionError('A visual proposal must not bypass complete-objective IK.')
    monkeypatch.setattr(current, 'local_step', direct_visual_execution_forbidden)
    command, record = current.reaching_step(view, .5, command_map=mapping)
    visual = record['visual_approach']
    np.testing.assert_array_equal(command, expected)
    assert {key: value for key, value in record.items()
            if key not in ('reaching_phase', 'visual_approach')} == expected_record
    assert record['reaching_phase'] == 'indet_ik'
    assert record['gradient_evaluation_count'] == 1
    assert record['objective_evaluation_count'] > 1
    assert visual['full_gaussian_bingham_criterion_used_for_selection'] is True
    assert visual['visual_velocity_executed_directly'] is False
    assert visual['initial_proposal'] == record['approach_initial_proposal']
    assert record['approach_initial_proposal']['available']
    assert -record['log_expected_density'] <= record['feasible_baseline_objective']+1e-12
    assert visual['mixture_collapsed'] is visual['source_laws_modified'] is False
    assert visual['physical_truth_received'] is visual['monte_carlo_used'] is False
    for key in SOURCE_KEYS:
        assert visual[key] == record[key]
    for key in ('goal_source_weights', 'goal_source_log_weights'):
        assert visual[key] == record[key]
    # This moving default has the explicitly projected board-center input.
    # Individual detector modes remain in its full image-only fusion record.
    image = view.records['fixed_goal_memory']['image_relative_source']
    assert image['observation_kind'] == 'fused_board_center'
    central = image['marker_board_fusion']['central_distribution']['components']
    assert len(visual['branches']) == len(view.stages[-1].goal_components) == len(central)
    assert image['central_component_count'] == len(central)
    assert [item['native_component_id'] for item in image['components']] == [item['component_id'] for item in central]
    raw = image['marker_board_fusion']['raw_observations']
    assert image['raw_marker_count'] == len(raw)
    assert image['raw_native_component_count'] == sum(item['native_component_count'] for item in raw)
    for item in raw:
        assert item['native_component_count'] == len(item['components'])
    assert visual['stiffness_command_correction'] == record['stiffness_command_correction']
    np.testing.assert_array_equal(visual['command_jacobian'], mapping.jacobian(view.prior.command))
    assert 'requested_visual_velocity' not in record
    np.testing.assert_array_equal(record['approach_initial_proposal']['requested_command'],
                                  visual['target_command'])
    assert record['selected_command_action'] == mapping.selection_record(command)
    # Every selected objective branch is the exact same frozen-K push of the
    # goal law used by the current wire; all moments and normalizers are used.
    for component, selected in zip(view.stages[-1].goal_components, record['selected_branches']):
        pushed = mapping.push(component.state, command, inference.body_ids, inference.actions)
        query = current._query(pushed)
        _, density = _relative_density(query, current.goal_offset,
                                      current.goal_orientation, current.integration_steps)
        assert selected['state_id'] == pushed.state_id
        assert selected['parent_state_id'] == component.state.state_id
        for key, value in density.items():
            assert selected[key] == value
    np.testing.assert_allclose(command, np.asarray(view.prior.command)+.5*current.velocity)
    assert np.max(np.abs(current.velocity)) <= current.speed+1e-12
    assert np.max(np.abs(current.velocity)) <= current.acceleration*.5+1e-12
    assert np.all(command >= current.lower) and np.all(command <= current.upper)
    for component, (state_id, covariance) in zip(
            view.stages[-1].components+view.stages[-1].goal_components, original):
        assert component.state.state_id == state_id
        np.testing.assert_array_equal(component.state.covariance, covariance)
    snapshot = build_snapshot(view, view.stages[-1], inference.common)
    display = inference.common['display']
    markers = make_markers(snapshot, display['seed'], display['point_size'])
    wire = io.BytesIO(); markers.serialize(wire)
    verify_wire_identity(snapshot, MarkerArray().deserialize(wire.getvalue()),
                         seed=display['seed'], point_size=display['point_size'])
    assert {p.query.state_id for p in snapshot.points if p.frame_label == 'hand'} == set(record['body_source_state_ids'])
    assert {p.query.state_id for p in snapshot.points if p.frame_label == 'switch_goal'} == set(record['goal_source_state_ids'])


def test_near_dispatch_is_identical_to_complete_criterion_IK_and_latches(problems):
    problem = problems['imu_marker']; view, mapping = problem[3:5]
    ordinary = controller(problem)
    near = controller(problem, near_distance_m=10.)
    expected, original = ordinary.step(view, .5, command_map=mapping)
    actual, record = near.reaching_step(view, .5, command_map=mapping)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(near.velocity, ordinary.velocity)
    assert record['reaching_phase'] == 'indet_ik'
    assert record['visual_approach']['near_latched']
    assert {key: value for key, value in record.items()
            if key not in ('reaching_phase', 'visual_approach')} == original
    near.visual_approach_policy.near_distance = .001
    _, retained = near.reaching_step(view, .5, command_map=mapping)
    assert retained['reaching_phase'] == 'indet_ik'
    assert retained['visual_approach']['selection_reason'] == 'near phase already latched'


@pytest.mark.parametrize('name', ['imu', 'no_sensors'])
def test_marker_disabled_never_enters_visual_phase_or_changes_ordinary_IK(problems, name):
    problem = problems[name]; view, mapping = problem[3:5]
    ordinary, actual = controller(problem), controller(problem)
    expected, original = ordinary.step(view, .5, command_map=mapping)
    command, record = actual.reaching_step(view, .5, command_map=mapping)
    np.testing.assert_array_equal(command, expected)
    assert actual.visual_approach_policy is None
    assert record['reaching_phase'] == 'indet_ik'
    assert record['visual_approach']['phase'] == 'unavailable'
    assert {key: value for key, value in record.items()
            if key not in ('reaching_phase', 'visual_approach')} == original


def test_actual_control_worker_selects_full_K_IK_and_serializes_the_same_law(problems, tmp_path, monkeypatch):
    scene, inference, source, _, _, packet, templates = problems['imu_marker']
    requests, replies = Queue(), Queue()
    sample = LogStiffnessDamper(inference.settings['control']['execution_stiffness']).advance(
        packet.proposal['selected_K'], source.prior.stamp)
    requests.put(dict(result=source, steps=(), stamp=source.prior.stamp,
        index=0, capture_index=0, packet=packet, execution_sample=sample,
        execution_packet=None, velocity=np.zeros(6)))
    requests.put(None)
    forbid_sampling(monkeypatch)
    control_worker(requests, replies, scene.robot_description, str(scene.urdf), templates,
                   str(tmp_path), .5)
    reply = replies.get_nowait()
    assert not reply.get('failed'), reply
    record = json.loads((tmp_path/'control.jsonl').read_text())
    diagnostics = reply['diagnostics']
    assert diagnostics['reaching_phase'] == 'indet_ik'
    assert record['controller'] == diagnostics
    assert diagnostics['gradient_evaluation_count'] == 1
    assert diagnostics['ik_update_count'] == 1
    assert diagnostics['future_ik_updates'] == 0 and not diagnostics['lookahead_enabled']
    assert 'trajectory_commands' not in diagnostics and 'rollout_steps' not in diagnostics
    assert diagnostics['visual_approach']['full_gaussian_bingham_criterion_used_for_selection'] is True
    assert diagnostics['visual_approach']['visual_velocity_executed_directly'] is False
    assert diagnostics['stiffness_command_correction']['selected_K'] == packet.record['command_proposal']['selected_K']
    assert diagnostics['selected_command_action']['density_id'] == packet.record['command_proposal']['density_id']
    snapshot = reply['snapshot']
    assert {p.query.state_id for p in snapshot.points if p.frame_label == 'hand'} == set(diagnostics['body_source_state_ids'])
    assert {p.query.state_id for p in snapshot.points if p.frame_label == 'switch_goal'} == set(diagnostics['goal_source_state_ids'])
    assert snapshot.stiffness_record['captured_packet_id'] == reply['packet'].packet_id
    assert snapshot.stiffness_record['captured_estimate_packet_id'] == packet.packet_id
    assert snapshot.stiffness_record['command_proposal'] == packet.proposal
    assert snapshot.stiffness_record['execution_command_proposal']['selected_K'] == sample['execution_K']
    assert record['stiffness'] == snapshot.stiffness_record
    display = inference.common['display']
    verify_wire_identity(snapshot, MarkerArray().deserialize(reply['markers_wire']),
                         seed=display['seed'], point_size=display['point_size'])
    prefix = tmp_path/record['artifact_prefix']
    assert prefix.with_suffix('.markers.bin').read_bytes() == reply['markers_wire']


def test_phase_source_commits_with_active_law_and_incoming_interval_is_preserved(session):
    current, clock, initial, identifier = session
    assert current.reaching_phase() == ('waiting_for_control', None)
    current.submit_control()
    first = _complete_control(current)
    current.control_replies.put(first); current.poll()
    expected = {key: first['diagnostics'][key] for key in SOURCE_KEYS}
    assert current.reaching_phase() == ('indet_ik', expected)
    clock[0] = 100.05; current.advance()
    interval = current.scene.trajectory[-1]
    assert interval['reaching_phase'] == 'indet_ik'
    assert interval['reaching_phase_source'] == expected
    # A newer inference result is pending; it cannot switch the active law,
    # selected K, or the phase source of the already executed interval.
    clock[0] = 100.10; current.advance()
    updated, packet, _ = _complete_estimate(current, initial, identifier)
    assert current.reaching_phase() == ('indet_ik', expected)
    current.submit_control()
    second = _complete_control(current)
    current.control_replies.put(second); current.poll()
    selected = {key: second['diagnostics'][key] for key in SOURCE_KEYS}
    assert selected != expected
    manifest = current.manifest()
    assert manifest['reaching_phase_source'] == expected
    assert manifest['planned_reaching_phase_source'] == selected
    assert current.current_control['snapshot'].stiffness_record['captured_packet_id'] == second['packet'].packet_id
    assert current.latest_packet.packet_id == packet.packet_id
    assert current.latest_result is updated
    rows = [json.loads(row) for row in (current.output/'commands.jsonl').read_text().splitlines()]
    assert rows[-1]['reaching_phase_source'] == expected
