"""Immutable candidate identities and deferred display/file construction.

The execution reply contains small identities and the completed gradient.
The same frozen candidate law is serialized separately after that reply;
the artifact process verifies its complete numeric law before drawing it.
"""
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np

from .async_stiffness import FrozenActionKPacket
from .display import make_markers, save_snapshot, snapshot_record
from .model import component_log_weight
from .motion_display import MotionDisplay
from .moving_control import control_components
from .node import build_snapshot


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity_hash(identity):
    return hashlib.sha256(_canonical(identity).encode()).hexdigest()


def _state_numeric_hash(state):
    digest = hashlib.sha256(b'ProbTF complete retained transform law v1')
    for array in (state.mean, state.covariance):
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    for edge in state.transforms:
        digest.update(edge.edge_id.encode()+b'\0')
        for array in (edge.translation, edge.translation_covariance, edge.rotation_wxyz):
            digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        digest.update(b'dirac' if edge.bingham is None else
            np.ascontiguousarray(edge.bingham, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _component_identity(item):
    log_weight = float(component_log_weight(item))
    if log_weight == -np.inf:
        log_weight = '-Infinity'
    return dict(label=item.label, state_id=item.state.state_id,
        component_id=item.state.component_id, weight=float(item.weight), log_weight=log_weight,
        numeric_law_sha256=_state_numeric_hash(item.state))


def action_packet_reference(packet):
    if packet is None:
        return None
    if not isinstance(packet, FrozenActionKPacket):
        raise TypeError('Deferred execution requires a captured direct-action K packet.')
    return dict(estimated_packet_id=packet.estimated_packet.packet_id,
                action_json=packet.action_json, packet_id=packet.packet_id)


def restore_action_packet(reference, estimated_packet):
    if reference is None:
        return None
    if estimated_packet is None or reference['estimated_packet_id'] != estimated_packet.packet_id:
        raise ValueError('A direct action reference lost its exact retained K packet.')
    expected = hashlib.sha256(('direct-action\n'+estimated_packet.packet_id+'\n'+
        reference['action_json']).encode()).hexdigest()
    if expected != reference['packet_id']:
        raise ValueError('A direct action reference checksum differs.')
    return FrozenActionKPacket(estimated_packet, reference['action_json'], expected)


def capture_candidate_identity(result, *, parameter_packet=None, action_packet=None,
                               parameter_identity=None):
    body, goals = control_components(result)
    if parameter_identity is None:
        proposal = None if parameter_packet is None else parameter_packet.proposal
        parameter_identity = dict(
            estimated_packet_id=None if parameter_packet is None else parameter_packet.packet_id,
            density_id=None if proposal is None else proposal['density_id'],
            selected_K=None if proposal is None else proposal['selected_K'],
            action_packet_ref=action_packet_reference(action_packet))
    measure = result.records.get('candidate_measure') or {}
    prediction = result.records.get('async_prediction') or {}
    ledger = prediction.get('command_path')
    return dict(schema='immutable_control_candidate_v1', frame=result.root_frame,
        stamp=float(result.prior.stamp), command=list(result.prior.command),
        version=result.prior.version, prior_state_id=result.prior.state_id,
        body=[_component_identity(item) for item in body],
        goals=[_component_identity(item) for item in goals],
        candidate_log_total_mass=measure.get('log_total_mass'),
        observation_ids_sha256=identity_hash(list(result.records.get('observation_ids', ()))),
        observation_id_count=len(result.records.get('observation_ids', ())),
        original_sensor_event=json.loads(_canonical(result.records.get('sensor_event'))),
        action_ledger_sha256=identity_hash(ledger),
        parameter=json.loads(_canonical(parameter_identity)))


def freeze_work_item(kind, payload):
    """Freeze after sending readiness; a queue feeder cannot see later mutation."""
    encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    return dict(kind=kind, payload=encoded,
        payload_sha256=hashlib.sha256(encoded).hexdigest(), payload_bytes=len(encoded))


def thaw_work_item(item, kind):
    if item.get('kind') != kind or not isinstance(item.get('payload'), bytes):
        raise ValueError('Unexpected immutable asynchronous work item.')
    if hashlib.sha256(item['payload']).hexdigest() != item['payload_sha256']:
        raise ValueError('An immutable asynchronous work payload changed.')
    return pickle.loads(item['payload'])


def archive_stiffness_job(output, job):
    """Retain every complete ratio/ledger even if the K consumer has failed."""
    directory = Path(output)/'stiffness_jobs'
    directory.mkdir(parents=True, exist_ok=True)
    frozen = freeze_work_item('stiffness_job',job)
    name = '{:08d}_{}'.format(job['job_sequence'],hashlib.sha256(
        job['batch_id'].encode()).hexdigest()[:16])
    path = directory/(name+'.pickle')
    if path.exists():
        raise ValueError('A stiffness job sequence cannot be overwritten.')
    temporary = path.with_suffix('.pickle.tmp')
    with temporary.open('wb') as stream:
        stream.write(frozen['payload'])
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    reference = dict(kind='stiffness_job_reference',job_sequence=job['job_sequence'],
        batch_id=job['batch_id'],source_candidate_id=job['source_candidate_id'],
        source_stamp=job['result'].prior.stamp,source_command=list(job['result'].prior.command),
        source_version=job['result'].prior.version,
        path=str(path),payload_sha256=frozen['payload_sha256'],
        payload_bytes=frozen['payload_bytes'],initialize_only=job['initialize_only'])
    metadata = path.with_suffix('.json')
    tmp = metadata.with_suffix('.json.tmp')
    tmp.write_text(_canonical(dict(reference,status='pending',
        archived_wall_time_s=time.time()))+'\n')
    tmp.replace(metadata)
    return reference


def load_stiffness_job(reference):
    if reference.get('kind') != 'stiffness_job_reference':
        raise ValueError('The K worker requires an archived immutable job reference.')
    payload = Path(reference['path']).read_bytes()
    job = thaw_work_item(dict(kind='stiffness_job',payload=payload,
        payload_sha256=reference['payload_sha256']),'stiffness_job')
    for key in ('job_sequence','batch_id','source_candidate_id','initialize_only'):
        if job[key] != reference[key]:
            raise ValueError('An archived stiffness job changed its FIFO/source identity.')
    return job


def construct_control_artifacts(job, common, output, monitor, *, consistency_evaluator=None):
    """Construct and save the exact source candidate, never a newer estimate."""
    from .async_workers import _append, _freeze_arrays
    predicted = _freeze_arrays(job['result'])
    expected = job['candidate_identity']
    actual = capture_candidate_identity(predicted, parameter_identity=expected['parameter'])
    if actual != expected or identity_hash(actual) != job['candidate_id']:
        raise ValueError('Deferred display candidate law/source differs from its execution packet.')
    request = job['request']
    context = dict(request_id=request.get('request_id'), candidate_id=job['candidate_id'])
    started = time.monotonic()
    with monitor.stage('snapshot_and_marker_construction', **context):
        snapshot = replace(build_snapshot(predicted, predicted.stages[-1], common),
            summary='', show_labels=False, stiffness_record=predicted.records['stiffness'])
        display = common['display']
        markers = make_markers(snapshot, display['seed'], display['point_size'])
    identity = str(request.get('request_id') or job['candidate_id'][:12])
    path = Path(output)/'control_{:05d}_capture_{:05d}_request_{}'.format(
        request['index'], request['capture_index'], identity)
    with monitor.stage('snapshot_artifact_serialization', **context):
        save_snapshot(path, snapshot, markers, seed=display['seed'], point_size=display['point_size'])
        MotionDisplay.stiffness_marker(snapshot, snapshot.stiffness_record)
        wire = io.BytesIO()
        markers.serialize(wire)
    if consistency_evaluator is not None:
        # The execution reply already used this exact frozen candidate.  An
        # auxiliary model-fit diagnostic may neither delay that reply nor
        # mutate the inferred law, gradient, stiffness packet or display.
        with monitor.stage('K_body_consistency_diagnostic', **context):
            try:
                consistency = consistency_evaluator.evaluate(
                    predicted, expected, request=request)
            except Exception as error:
                consistency = dict(status='unavailable',
                    reason=type(error).__name__+': '+str(error),
                    candidate_id=job['candidate_id'], request_id=request.get('request_id'),
                    stamp=expected['stamp'], command=expected['command'],
                    diagnostic_is_control_acceptance_gate=False,
                    body_reconstructed_from_stiffness=False,
                    physical_truth_received=False, monte_carlo_used=False)
            _append(Path(output)/'K_body_consistency.jsonl', consistency)
    with monitor.stage('control_record_json_write', **context):
        _append(Path(output)/'control.jsonl', dict(index=request['index'],
            request_id=request.get('request_id'),event_id=request.get('event_id'),
            capture_index=request['capture_index'], artifact_prefix=path.name,
            markers_wire_sha256=hashlib.sha256(wire.getvalue()).hexdigest(),
            prediction=predicted.records['async_prediction'], controller=job['diagnostics'],
            control_mode=job['control_mode'], gradient_evaluation=job['gradient_evaluation'],
            stiffness=snapshot.stiffness_record,snapshot=snapshot_record(snapshot),
            candidate_identity=expected,candidate_id=job['candidate_id'],
            execution_ready_seconds=job['ready_seconds'],
            artifact_work_seconds=time.monotonic()-started,
            work_seconds=job['ready_seconds']+time.monotonic()-started))
    return dict(kind='control_artifacts_ready',index=request['index'],
        request_id=request.get('request_id'),event_id=request.get('event_id'),
        capture_index=request['capture_index'],snapshot=snapshot,markers_wire=wire.getvalue(),
        diagnostics=job['diagnostics'],candidate_identity=expected,candidate_id=job['candidate_id'],
        control_mode=job['control_mode'],gradient_evaluation=job['gradient_evaluation'],
        artifact_work_seconds=time.monotonic()-started)
