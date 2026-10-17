"""The published measure and direct-K HUD describe their actual plotted law."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from probtf_observation_fusion_demo.async_state import ActionStep, predict_result
from probtf_observation_fusion_demo.async_stiffness import (
    capture_direct_stiffness_action, capture_stiffness_packet, reanchor_retained_stiffness,
)
from probtf_observation_fusion_demo.display import (
    make_markers, save_snapshot, snapshot_record, verify_measure_identity,
    verify_wire_identity,
)
from probtf_observation_fusion_demo.motion_display import MotionDisplay
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.node import build_snapshot
from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore
from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier

from test_async_state import steps_for
from test_async_stiffness import engine_for
from test_sensor_event_inference import acquired


def wire(message):
    output = io.BytesIO()
    message.serialize(output)
    return output.getvalue()


@pytest.fixture(scope='module')
def measured_source(acquired, tmp_path_factory):
    description, inputs, templates = acquired
    inference = MovingInference(description, templates)
    directory = tmp_path_factory.mktemp('measure_provenance')
    with SensorEventStore(directory/'events.sqlite3') as store:
        event = store.append('image:acquired:0', 'image', inputs.stamp, inputs.command,
            dict(time=inputs.time, frame_index=0, image_stamp=inputs.stamp,
                 image=inputs.image, image_path='capture_000000.png',
                 image_sha256=hashlib.sha256(inputs.image.tobytes()).hexdigest()))
        result = inference.update_sensor_event(event, event_index=0,
            probability_unit='per_effective_sensor_event')
    assert result.records['valid_observation_count'] == 1
    return inference, result


def test_prediction_transports_shape_but_preserves_mass_evidence_and_acquisition_bitwise(measured_source):
    inference, source = measured_source
    original = deepcopy(source.records)
    steps = steps_for(source)
    prediction = predict_result(source, steps, steps[-1].end_stamp, 3,
                                inference.body_ids, inference.actions)
    measure = prediction.records['candidate_measure']
    original_measure = source.records['candidate_measure']
    assert struct.pack('!d', measure['log_total_mass']) == struct.pack('!d', original_measure['log_total_mass'])
    for key in ('previous_log_total_mass', 'log_mass_increment', 'observation_evidence',
                'evidence_measure', 'bayes_factor_hypothesis_scale'):
        assert measure[key] == original_measure[key]
    assert prediction.records['sensor_event'] == source.records['sensor_event']
    assert prediction.records['sensor_event'] is not source.records['sensor_event']
    assert measure['shape_source_state_ids'] == [item.state.state_id for item in prediction.stages[-1].components]
    assert measure['shape_source_versions'] == [item.state.version for item in prediction.stages[-1].components]
    assert measure['shape_source_stamps'] == [steps[-1].end_stamp]
    assert measure['shape_source_commands'] == [list(steps[-1].end_command)]
    assert measure['prediction_log_mass_increment'] == 0.
    assert measure['observation_source']['shape_source_state_ids'] == original_measure['shape_source_state_ids']
    assert measure['observation_source']['sensor_event'] == source.records['sensor_event']
    assert prediction.records['async_prediction']['observations_assimilated'] == 0
    assert source.records == original


def test_repeated_prediction_keeps_original_observation_not_previous_prediction(measured_source):
    inference, source = measured_source
    steps = steps_for(source)
    first = predict_result(source, steps, steps[-1].end_stamp, 3,
                           inference.body_ids, inference.actions)
    command = np.asarray(first.prior.command)
    end = command+np.array([.001, 0., 0., 0., 0., 0.])
    step = ActionStep(3, first.prior.stamp, first.prior.stamp+.05,
        command, end, end-command, density_id='declared_test_action', estimate_version=0,
        source_state_ids=tuple(item.state.state_id for item in first.stages[-1].components),
        source_stamp=first.prior.stamp, source_version=first.prior.version)
    second = predict_result(first, (step,), step.end_stamp, 4,
                            inference.body_ids, inference.actions)
    assert second.records['candidate_measure']['observation_source'] == first.records['candidate_measure']['observation_source']
    assert second.records['sensor_event'] == source.records['sensor_event']
    assert second.records['candidate_measure']['log_total_mass'] == source.records['candidate_measure']['log_total_mass']
    assert second.records['candidate_measure']['shape_source_state_ids'] != first.records['candidate_measure']['shape_source_state_ids']


def test_actual_published_string_and_marker_wire_share_predicted_measure_without_style_changes(measured_source, tmp_path):
    inference, source = measured_source
    steps = steps_for(source)
    prediction = predict_result(source, steps, steps[-1].end_stamp, 3,
                                inference.body_ids, inference.actions)
    snapshot = build_snapshot(prediction, prediction.stages[-1], inference.common)
    assert snapshot.candidate_measure is prediction.records['candidate_measure']
    assert snapshot.sensor_event is prediction.records['sensor_event']
    settings = inference.common['display']
    message = make_markers(snapshot, settings['seed'], settings['point_size'])
    actual = MarkerArray().deserialize(wire(message))
    assert verify_wire_identity(snapshot, actual, seed=settings['seed'], point_size=settings['point_size'])
    bare = replace(snapshot, candidate_measure=None, sensor_event=None)
    assert wire(message) == wire(make_markers(bare, settings['seed'], settings['point_size']))
    published = String().deserialize(wire(String(json.dumps(snapshot_record(snapshot)))))
    metadata = json.loads(published.data)
    assert metadata['candidate_measure'] == prediction.records['candidate_measure']
    assert metadata['sensor_event'] == source.records['sensor_event']
    ids = {point['source_state_id'] for point in metadata['queries']
           if point['frame_label'] not in ('marker', 'switch_goal')}
    assert ids == set(metadata['candidate_measure']['shape_source_state_ids'])
    assert metadata['sensor_event']['acquisition_stamp'] < metadata['stamp']
    assert metadata['candidate_measure']['shape_source_stamps'] == [metadata['stamp']]
    save_snapshot(tmp_path/'published', snapshot, actual,
                  seed=settings['seed'], point_size=settings['point_size'])
    saved = json.loads((tmp_path/'published.json').read_text())
    assert saved['candidate_measure'] == metadata['candidate_measure']
    assert saved['sensor_event'] == metadata['sensor_event']


@pytest.mark.parametrize('field,value', [
    ('shape_source_state_ids', ['foreign']), ('shape_source_versions', [-1]),
    ('shape_source_stamps', [0.]), ('shape_source_commands', [[0.]*6]),
    ('shape_log_weights', [-100.]),
])
def test_published_snapshot_and_wire_reject_foreign_measure_shape(measured_source, field, value):
    inference, source = measured_source
    snapshot = build_snapshot(source, source.stages[-1], inference.common)
    measure = deepcopy(snapshot.candidate_measure)
    measure[field] = value
    wrong = replace(snapshot, candidate_measure=measure)
    with pytest.raises(ValueError, match='plotted body law'):
        snapshot_record(wrong)
    with pytest.raises(ValueError, match='plotted body law'):
        make_markers(wrong)
    foreign_source = replace(source, records=dict(source.records, candidate_measure=measure))
    steps = steps_for(source)
    with pytest.raises(ValueError, match='actual source body law'):
        predict_result(foreign_source, steps, steps[-1].end_stamp, 3,
                       inference.body_ids, inference.actions)


def test_final_measure_is_not_misattributed_to_intermediate_prior_stage(measured_source):
    inference, source = measured_source
    snapshot = build_snapshot(source, source.stages[0], inference.common)
    assert snapshot.candidate_measure is None
    assert 'candidate_measure' not in snapshot_record(snapshot)


@pytest.fixture(scope='module')
def observed(measured_source, acquired, tmp_path_factory):
    """A real single acquired image update, including the current K BFS path."""
    inference, original = measured_source
    result = replace(original, records=deepcopy(original.records))
    directory = tmp_path_factory.mktemp('direct_measure_K')
    urdf = directory/'robot.urdf'
    urdf.write_text(acquired[0])
    settings = yaml.safe_load((Path(__file__).resolve().parents[1]/
                               'config/stiffness_identification.yaml').read_text())
    identifier = StiffnessIdentifier(urdf, inference.body_ids, settings)
    result.records['stiffness'] = identifier.update(result)
    assert result.records['stiffness']['bayes_factor_surprise']['hazard_applied']
    assert result.records['stiffness']['bayes_factor_surprise']['rho0'] == .1
    assert identifier.command_proposal(result) is not None
    packet = capture_stiffness_packet(result)
    return SimpleNamespace(urdf=urdf), inference, result, identifier, settings, packet


def direct_view(observed):
    engine = engine_for(observed)
    _, inference, source, _, _, raw = observed
    command = np.asarray(source.prior.command)
    target = command+np.array([.003, -.002, .001, .002, -.001, .001])
    action = engine.increment(command, target)
    step = ActionStep(0, source.prior.stamp, source.prior.stamp+.3,
        command, target, action['joint_increment'], density_id=raw.proposal['density_id'],
        selected_K=raw.proposal['selected_K'], estimate_version=source.prior.version,
        source_state_ids=tuple(item.state.state_id for item in source.stages[-1].components),
        source_stamp=source.prior.stamp, source_version=source.prior.version)
    current = predict_result(source, (step,), step.end_stamp, 36,
                             inference.body_ids, inference.actions)
    packet = capture_direct_stiffness_action(engine, current, observed[-1])
    view, mapping = reanchor_retained_stiffness(engine, current, packet)
    return view, mapping, packet


def test_direct_estimated_K_HUD_wire_uses_exact_same_K_and_preserves_raw_proposal(observed):
    view, mapping, packet = direct_view(observed)
    inference = observed[1]
    snapshot = build_snapshot(view, view.stages[-1], inference.common)
    record = view.records['stiffness']
    assert record['command_proposal'] == packet.proposal == observed[-1].proposal
    assert record['action_command_proposal']['selected_K'] == packet.proposal['selected_K']
    assert record['action_stiffness'] == packet.action
    assert mapping.record()['selected_K'] == packet.proposal['selected_K']
    hud = MotionDisplay.stiffness_marker(snapshot, record)
    text = ', '.join('{:.4g}'.format(value) for value in packet.proposal['selected_K'])
    assert 'Estimated K [Nm/rad]: '+text in hud.text
    assert 'IK K_est [Nm/rad]: '+text in hud.text
    assert 'Direct estimated representative' in hud.text
    assert 'damping' not in hud.text
    actual = MarkerArray().deserialize(wire(MarkerArray([hud]))).markers[0]
    assert actual.text == hud.text
    assert actual.scale.__getstate__() == hud.scale.__getstate__()
    assert record['command_proposal'] == observed[-1].proposal


@pytest.mark.parametrize('tamper', ['action_K', 'raw_K', 'source_stamp', 'map_action', 'proposal_K'])
def test_direct_HUD_rejects_mismatched_estimate_or_action_provenance(observed, tamper):
    view, _, _ = direct_view(observed)
    snapshot = build_snapshot(view, view.stages[-1], observed[1].common)
    record = deepcopy(view.records['stiffness'])
    if tamper == 'action_K':
        record['action_stiffness']['selected_K'][0] *= 1.01
    elif tamper == 'raw_K':
        record['command_proposal']['selected_K'][0] *= 1.01
    elif tamper == 'source_stamp':
        record['action_stiffness']['estimated_source_stamp'] += .01
    elif tamper == 'map_action':
        record['command_correction']['captured_parameter_packet']['action_stiffness']['selected_K'][0] *= 1.01
    else:
        record['action_command_proposal']['selected_K'][0] *= 1.01
    with pytest.raises(ValueError, match='different stiffness|same source and action'):
        MotionDisplay.stiffness_marker(snapshot, record)
