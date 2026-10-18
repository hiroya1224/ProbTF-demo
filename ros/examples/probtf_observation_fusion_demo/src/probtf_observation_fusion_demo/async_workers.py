"""Independent numerical and acquisition workers for timestamped event execution."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
import rospkg
import yaml

from .async_state import RecordedCommandPath, predict_result
from .async_stiffness import PhysicalActionEngine, capture_stiffness_packet, capture_direct_stiffness_action, reanchor_retained_stiffness
from .board_camera import BoardCamera
from .display import make_markers, save_snapshot, snapshot_record
from .marker_search import marker_observed
from .motion_display import MotionDisplay
from .model import component_log_weight
from .moving_control import MovingController
from .moving_control import control_components
from .moving_inference import MovingInference
from .selection_gradient import selection_gradient_at_command
from .runtime_monitor import RuntimeMonitor
from .visibility_factor import VisibilityFactor
from .node import build_snapshot
from .sensor_event_store import SensorEventStore
from .stiffness_identification import StiffnessIdentifier
from probtf_switch_press_demo.vision import PlanarCamera

def _append(path, record):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(record, separators=(',', ':'))+'\n')

def _freeze_arrays(value, seen=None):
    """Restore immutable array flags after process serialization, not values."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return value
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    elif isinstance(value, dict):
        for item in value.values():
            _freeze_arrays(item, seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _freeze_arrays(item, seen)
    elif hasattr(value, '__dict__'):
        _freeze_arrays(vars(value), seen)
    return value

def _worker_error(output, worker):
    output.put(dict(failed=True, worker=worker, traceback=traceback.format_exc()))

def _archive_image(path, image, index, stamp, dictionary, parameters, detector, marker_id):
    if not cv2.imwrite(str(path), image):
        raise IOError('Failed to archive camera frame '+str(path))
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = (detector.detectMarkers(gray) if detector is not None else
        cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters))
    allowed = {int(marker_id)} if np.isscalar(marker_id) else set(map(int, marker_id))
    visible_ids = [] if ids is None else [int(value) for value in ids.ravel() if int(value) in allowed]
    visible = bool(visible_ids)
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else np.array(image, copy=True)
    if visible:
        for points, identity in zip(corners, ids.ravel()):
            if int(identity) in allowed:
                cv2.polylines(preview, [np.round(points.reshape(-1, 2)).astype(np.int32)],
                              True, (0, 255, 0), 2, cv2.LINE_AA)
    return dict(index=index, stamp=stamp, visible=bool(visible), image=preview,
                visible_marker_ids=visible_ids)

def _handoff_result(result):
    """Transfer the identical final law, excluding already saved raw charts."""
    keys = ('observation_ids', 'fixed_goal_memory', 'fixed_goal_memory_event',
            'stiffness', 'bayes_factor_surprise', 'sensor_event', 'candidate_measure', 'joint_batch_source')
    return replace(result, stages=(result.stages[-1],), sensor_posteriors=(),
        local_estimates=(), records={k: result.records[k] for k in keys if k in result.records},
        image=np.empty((0, 0), dtype=np.uint8), detection_image=np.empty((0, 0), dtype=np.uint8),
        frozen_body_prior=None, frozen_body_posterior=None, frozen_body_branches=(),
        projected_body_numerator=None, time_resolved_body_ratios=())

def _unselected_parameter_view(result, captured):
    """No observation has yet selected K; retain the declared command family."""
    stage = result.stages[-1]
    density = captured.get('posterior') or captured.get('retained_prior')
    record = dict(captured, prior=density, posterior=None, retained_prior=density,
        prior_retained=True, update_applied=False, command_proposal=None,
        command_correction=None, status='prediction_without_selected_parameters',
        source_stamp=result.prior.stamp, source_command=list(result.prior.command),
        source_prior_state_id=result.prior.state_id,
        body_posterior_state_ids=[item.state.state_id for item in stage.components],
        body_posterior_labels=[item.label for item in stage.components],
        body_posterior_versions=[item.state.version for item in stage.components],
        body_posterior_weights=[item.weight for item in stage.components],
        body_posterior_log_weights=[component_log_weight(item) for item in stage.components],
        observation_ids=[], captured_source_stamp=captured['source_stamp'],
        captured_source_command=captured['source_command'],
        captured_status=captured['status'], information_rank=None,
        feedback_to_control=False, feedback_to_body_prediction=False)
    return replace(result, records=dict(result.records, stiffness=record))

def control_worker(requests, replies, description, urdf, templates, output, horizon,
                   control_mode=None, visibility_options=None):
    """Evaluate a frozen-law gradient, or the isolated comparison endpoint IK."""
    cv2.setNumThreads(1)
    monitor = RuntimeMonitor(output, 'control')
    try:
        inference = MovingInference(description, templates)
        settings = inference.settings['control']
        mode = control_mode or settings.get('mode', 'single_step_ik')
        if mode not in ('running_gradient', 'single_step_ik'):
            raise ValueError('Unknown asynchronous control mode: '+str(mode))
        view_settings=dict(settings.get('visibility_factor',{}))
        if visibility_options:
            view_settings.update(visibility_options)
        camera=inference.settings['camera']
        matrix=camera['matrix']
        view_settings.setdefault('intrinsics',dict(fx=matrix[0][0],fy=matrix[1][1],
            cx=matrix[0][2],cy=matrix[1][2],width=camera['width'],height=camera['height']))
        view_factor=VisibilityFactor(view_settings)
        controller = MovingController(inference, description, settings['goal_offset'],
            settings['goal_orientation_wxyz'], settings,
            integration_steps=int(settings['integration_steps']))
        package = Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
        parameter_settings = yaml.safe_load((package/'config/stiffness_identification.yaml').read_text())
        engine = PhysicalActionEngine(Path(urdf), tuple(inference.body_ids), parameter_settings)
        action_ledger = []
        ledger_hashes=[hashlib.sha256(b'ProbTF actual-command ledger v1').hexdigest()]
        while True:
            request = requests.get()
            if request is None:
                return
            started = time.monotonic()
            monitor.mark('request_received', request_id=request.get('request_id'),
                         source_stamp=request['stamp'])
            with monitor.stage('actual_ledger_transport', request_id=request.get('request_id'), source_stamp=request['stamp']):
                source = _freeze_arrays(request['result'])
                ledger_reference=None
                if 'ledger_delta' in request:
                    for step in request['ledger_delta']:
                        if step.index != len(action_ledger):
                            raise ValueError('IK ledger transfer has a missing or duplicated actual interval.')
                        action_ledger.append(step)
                        encoded=json.dumps(step.record(),sort_keys=True,
                            separators=(',',':'),allow_nan=False).encode()
                        ledger_hashes.append(hashlib.sha256(
                            bytes.fromhex(ledger_hashes[-1])+encoded).hexdigest())
                    end=request['ledger_target_index']
                    begin=request['ledger_source_index']
                    if not 0 <= begin <= end == len(action_ledger):
                        raise ValueError('IK ledger range differs from the actual source/target frames.')
                    steps=tuple(action_ledger[begin:end])
                    ledger_reference=dict(path='commands.jsonl',start_index=begin,
                        end_index_exclusive=end,step_count=end-begin,
                        prefix_chain_sha256=ledger_hashes[end],
                        source_prefix_chain_sha256=ledger_hashes[begin])
                else:
                    steps=request['steps']  # explicit standalone diagnostic call
                predicted = predict_result(source, steps,
                    target_stamp=request['stamp'], frame_index=request['index'],
                    body_ids=inference.body_ids, actions=inference.actions,
                    ledger_reference=ledger_reference)
            try:
                with monitor.stage('capture_K_and_reanchor', request_id=request.get('request_id')):
                    if request['packet'] is None:
                        action_packet = None
                        predicted = _unselected_parameter_view(predicted, source.records['stiffness'])
                        command_map = None
                    else:
                        action_packet = capture_direct_stiffness_action(engine, predicted, request['packet'],
                            request.get('action_packet'))
                        predicted, command_map = reanchor_retained_stiffness(engine, predicted, action_packet)
            except Exception as error:
                if mode != 'running_gradient':
                    raise
                # An unavailable branch is a failed gradient evaluation, not
                # a zero gradient or permission to execute an uncorrected law.
                reason = type(error).__name__+': '+str(error)
                failed_body,failed_goals=control_components(predicted)
                failure = dict(status='failed', reason=reason,
                    gradient=None, body_state_ids=[item.state.state_id for item in failed_body],
                    goal_state_ids=[item.state.state_id for item in failed_goals],
                    parameter_density_id=request['packet'].proposal['density_id'],
                    selected_K=request['packet'].proposal['selected_K'],
                    objective_at_source=None, computation_seconds=time.monotonic()-started,
                    finite_difference_diagnostics={},
                    source_stamp=request['stamp'], source_command=list(predicted.prior.command),
                    observation_source_stamp=source.prior.stamp,
                    observation_source_stamps=[source.prior.stamp],
                    snapshot_update_count=request.get('snapshot_update_count',0),
                    command_interval_id=request['index'],
                    request_id=request.get('request_id'), event_id=request.get('event_id'),
                    source_action_ledger=ledger_reference,
                    physical_truth_received=False, monte_carlo_used=False)
                _append(Path(output)/'gradient_failures.jsonl', failure)
                replies.put(dict(index=request['index'], request_id=request.get('request_id'),
                    event_id=request.get('event_id'), capture_index=request['capture_index'],
                    control_mode=mode, gradient_evaluation=failure,
                    work_seconds=time.monotonic()-started))
                continue
            gradient_evaluation = None
            if mode == 'running_gradient':
                if predicted.records.get('fixed_goal_memory', {}).get('available', False):
                    with monitor.stage('task_and_view_gradient', request_id=request.get('request_id'), source_stamp=request['stamp']):
                        gradient_evaluation = selection_gradient_at_command(
                            controller,predicted,view_factor,command_map=command_map).record
                else:
                    body, goals = control_components(predicted)
                    gradient_evaluation = dict(status='waiting_for_goal',
                        reason='No valid observed goal has been retained; J is not fabricated.',
                        gradient=None, source_command=list(predicted.prior.command),
                        source_stamp=predicted.prior.stamp,
                        body_state_ids=[item.state.state_id for item in body],
                        goal_state_ids=[item.state.state_id for item in goals],
                        parameter_density_id=None if command_map is None else command_map.record()['density_id'],
                        selected_K=None if command_map is None else command_map.record()['selected_K'],
                        objective_at_source=None, computation_seconds=0.,
                        finite_difference_diagnostics={}, fixed_goal_memory_available=False,
                        physical_truth_received=False, monte_carlo_used=False)
                goal_stamp=predicted.records.get('fixed_goal_memory',{}).get('observation_stamp')
                body_stamp=(source.records['joint_batch_source']['last_effective_observation_stamp']
                    if 'joint_batch_source' in source.records else
                    source.prior.stamp if source.records.get('observation_ids') else None)
                gradient_evaluation.update(
                    snapshot_update_count=request.get('snapshot_update_count', 0),
                    command_interval_id=request['index'],
                    observation_source_stamps=sorted({stamp for stamp in (body_stamp,goal_stamp)
                                                      if stamp is not None}),
                    body_observation_source_stamp=body_stamp,
                    goal_observation_source_stamp=goal_stamp,
                    source_action_ledger=ledger_reference,
                    source_observation_ids=list(source.records.get('observation_ids', ())))
                diagnostics = dict(gradient_evaluation,
                    method='asynchronous independent full-J task and visibility factor gradients; no endpoint selection',
                    control_mode=mode, reaching_phase='running_gradient',
                    source_version=predicted.prior.version,
                    body_source_state_ids=gradient_evaluation['body_state_ids'],
                    goal_source_state_ids=gradient_evaluation['goal_state_ids'],
                    ik_update_count=0, gradient_evaluation_count=1 if gradient_evaluation['status']=='ok' else 0,
                    future_ik_updates=0, lookahead_enabled=False,
                    endpoint_selection_performed=False, velocity_clipping_enabled=False,
                    acceleration_clipping_enabled=False)
                planned_velocity = None
            else:
                controller.velocity = np.array(request['velocity'], copy=True)
                command, diagnostics = controller.reaching_step(predicted, horizon, command_map=command_map)
                diagnostics.update(ik_update_count=1, future_ik_updates=0, lookahead_enabled=False)
                planned_velocity = (np.asarray(command)-np.asarray(predicted.prior.command))/horizon
            predicted.records['controller'] = diagnostics
            predicted.records['async_control'] = dict(
                source_capture_index=request['capture_index'], prediction_index=request['index'],
                prediction_stamp=request['stamp'], control_mode=mode,
                horizon_s=None if mode=='running_gradient' else diagnostics['duration'],
                ik_update_step_s=None if mode=='running_gradient' else horizon,
                ik_update_count=0 if mode=='running_gradient' else 1,
                source_capture_stamp=source.prior.stamp,
                source_capture_command=list(source.prior.command),
                requested_velocity=None if planned_velocity is None else (planned_velocity*horizon).tolist(),
                velocity_units='joint displacement over horizon; see desired_velocity_rad_s',
                desired_velocity_rad_s=None if planned_velocity is None else planned_velocity.tolist(),
                physical_truth_received=False, monte_carlo_used=False)
            with monitor.stage('snapshot_and_marker_construction', request_id=request.get('request_id')):
                snapshot = replace(build_snapshot(predicted, predicted.stages[-1], inference.common),
                    summary='', show_labels=False, stiffness_record=predicted.records['stiffness'])
                display = inference.common['display']
                markers = make_markers(snapshot, display['seed'], display['point_size'])
            identity = str(request.get('request_id') or hashlib.sha256(
                (str(request.get('event_id'))+source.prior.state_id).encode()).hexdigest()[:12])
            path = Path(output)/'control_{:05d}_capture_{:05d}_request_{}'.format(
                request['index'], request['capture_index'], identity)
            with monitor.stage('snapshot_artifact_serialization', request_id=request.get('request_id')):
                save_snapshot(path, snapshot, markers, seed=display['seed'],
                              point_size=display['point_size'])
                MotionDisplay.stiffness_marker(snapshot, snapshot.stiffness_record)
                wire = io.BytesIO()
                markers.serialize(wire)
            with monitor.stage('control_record_json_write', request_id=request.get('request_id')):
                _append(Path(output)/'control.jsonl', dict(index=request['index'],
                    request_id=request.get('request_id'),event_id=request.get('event_id'),
                    capture_index=request['capture_index'], artifact_prefix=path.name,
                    markers_wire_sha256=hashlib.sha256(wire.getvalue()).hexdigest(),
                    prediction=predicted.records['async_prediction'],
                    controller=diagnostics, control_mode=mode,
                    gradient_evaluation=gradient_evaluation, stiffness=snapshot.stiffness_record,
                    snapshot=snapshot_record(snapshot), work_seconds=time.monotonic()-started))
            replies.put(dict(index=request['index'], capture_index=request['capture_index'],
                event_id=request.get('event_id'),
                request_id=request.get('request_id'),
                packet=action_packet, control_mode=mode,
                gradient_evaluation=gradient_evaluation,
                velocity=None if planned_velocity is None else planned_velocity.copy(),
                snapshot=snapshot, markers_wire=wire.getvalue(), diagnostics=diagnostics,
                work_seconds=time.monotonic()-started))
            monitor.mark('reply_enqueued', request_id=request.get('request_id'),
                         source_stamp=request['stamp'], total_seconds=time.monotonic()-started)
    except Exception:
        _worker_error(replies, 'control')


def event_inference_worker(requests, replies, description, urdf, templates, output,
                           probability_unit):
    """Fuse every event in an immutable acquisition batch before its handoff.

    The joint path retains acquisition-specific density ratios and computes
    stiffness evidence/proposal once after the full frontier. The explicit
    reference path retains each event's original update. Metadata checkpoints
    precede the final law/K handoff; joint input payloads are loaded together.
    """
    cv2.setNumThreads(1)
    monitor = RuntimeMonitor(output, 'inference')
    try:
        inference = MovingInference(description, templates)
        package = Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
        settings = yaml.safe_load((package/'config/stiffness_identification.yaml').read_text())
        stiffness = StiffnessIdentifier(Path(urdf), tuple(inference.body_ids), settings)
        stiffness.runtime_monitor = monitor
        enabled = {item['sensor'] for item in templates}
        previous = None
        previous_index = None
        def evaluate(event, index, steps):
            nonlocal previous, previous_index
            started = time.monotonic()
            command_path = (None if previous is None else RecordedCommandPath(previous, steps,
                body_ids=inference.body_ids, actions=inference.actions))
            with monitor.stage('body_event_update', event_id=event.event_id, source_stamp=event.stamp):
                result = inference.update_sensor_event(event, event_index=index,
                    command_map=command_path, probability_unit=probability_unit)
            inference_seconds = time.monotonic()-started
            with monitor.stage('K_factor_and_evidence', event_id=event.event_id):
                record = stiffness.update(result)
            with monitor.stage('K_representative_selection', event_id=event.event_id):
                proposal = stiffness.command_proposal(result)
            result.records['stiffness'] = record
            if (record.get('stiffness_posterior_defined') is False or
                    (stiffness.factors and proposal is None and
                     record.get('proposal_status')!='no_informative_density_factor')):
                _append(Path(output)/'estimation_failure.jsonl',dict(
                    event_id=event.event_id,event_sequence=event.sequence,
                    sensor_kind=event.kind,capture_index=index,stiffness=record,
                    sensor_event=result.records.get('sensor_event'),
                    body_source_state_ids=[item.state.state_id for item in result.stages[-1].components]))
                raise ValueError('No stable estimated-K representative: '+str(
                    record.get('failure_reason') or record.get('proposal_failure_reason')))
            packet = None if proposal is None else capture_stiffness_packet(result)
            result.records['async_estimation'] = dict(
                event_id=event.event_id, event_sequence=event.sequence,
                sensor_kind=event.kind, capture_index=index,
                capture_stamp=event.stamp, capture_command=list(event.command),
                inference_seconds=inference_seconds,
                total_seconds=time.monotonic()-started, command_action_count=len(steps),
                physical_truth_received=False, posterior_retained=True,
                scheduling='all events in acquisition timestamp and sequence order',
                newest_only_selection=False, monte_carlo_used=False)
            with monitor.stage('estimation_record_and_handoff', event_id=event.event_id):
                _append(Path(output)/'estimation.jsonl', result.records)
                message = dict(result=_handoff_result(result), packet=packet,
                    event_id=event.event_id, event_sequence=event.sequence,
                    sensor_kind=event.kind, capture_index=index, capture_stamp=event.stamp,
                    marker_observed=marker_observed(result),
                    work_seconds=time.monotonic()-started)
            previous, previous_index = result, index
            return result, message

        while True:
            request = requests.get()
            if request is None:
                return
            if not isinstance(request, dict) or request.get('kind') != 'sensor_event_batch':
                event, index, steps = request
                _, message = evaluate(event, index, steps)
                replies.put(message)
                continue
            batch = request['batch']
            batch_identity = batch.batch_id
            monitor.mark('batch_received', batch_id=batch_identity,
                         event_count=len(batch.events), source_stamp=batch.events[-1].stamp)
            by_index = {step.index: step for step in request['ledger_steps']}
            if len(by_index) != len(request['ledger_steps']):
                raise ValueError('The batch action ledger contains duplicate intervals.')
            store = SensorEventStore(request['database'])
            started_batch = time.monotonic()
            last_message = None
            last_image_observed = None
            processed = []
            evaluated = []
            try:
                joint_mode = bool(request.get('joint_body_batch', True))
                joint_result = None
                joint_fused_ids = set()
                if joint_mode:
                    with monitor.stage('raw_batch_load_and_validate', batch_id=batch_identity, event_count=len(batch.events)):
                        events = tuple(store.get(metadata.event_id) for metadata in batch.events)
                        for event, metadata in zip(events, batch.events):
                            if (event.sequence != metadata.sequence or event.kind != metadata.kind
                                    or event.stamp != metadata.stamp
                                    or event.command != metadata.command or event.payload_hash != metadata.payload_hash
                                    or event.payload_json != metadata.payload_json
                                    or event.status != 'processing' or event.worker_id != batch.worker_id):
                                raise ValueError('An immutable joint-batch acquisition changed after claim.')
                    with monitor.stage('body_joint_batch_update', batch_id=batch_identity, event_count=len(events), source_stamp=events[-1].stamp):
                        joint_result = inference.update_sensor_batch(events,
                            request['ledger_steps'], batch_id=batch_identity)
                    inference_seconds = time.monotonic()-started_batch
                    with monitor.stage('K_factor_and_evidence', batch_id=batch_identity, source_stamp=events[-1].stamp):
                        record = stiffness.update_batch(joint_result, events=events,
                            command_ledger=joint_result.records['joint_batch']['actual_ledger'],
                            batch_id=batch_identity,
                            event_body_ratios=joint_result.time_resolved_body_ratios)
                    with monitor.stage('K_representative_selection', batch_id=batch_identity, source_stamp=events[-1].stamp):
                        proposal = stiffness.command_proposal(joint_result)
                    joint_result.records['stiffness'] = record
                    if (record.get('stiffness_posterior_defined') is False or
                            (stiffness.factors and proposal is None and
                             record.get('proposal_status') != 'no_informative_density_factor')):
                        _append(Path(output)/'estimation_failure.jsonl',dict(
                            batch_id=batch_identity, event_ids=list(batch.event_ids),
                            stiffness=record,
                            body_source_state_ids=[item.state.state_id
                                for item in joint_result.stages[-1].components]))
                        raise ValueError('No finite batch stiffness representative: '+str(
                            record.get('failure_reason') or record.get('proposal_failure_reason')))
                    packet = None if proposal is None else capture_stiffness_packet(joint_result)
                    endpoint_event = events[-1]
                    endpoint_index = int(endpoint_event.payload['frame_index'])
                    joint_fused_ids = set(joint_result.records['observation_ids'])
                    joint_result.records['async_estimation'] = dict(batch_id=batch_identity,
                        event_id=endpoint_event.event_id, event_sequence=endpoint_event.sequence,
                        sensor_kind=endpoint_event.kind, capture_index=endpoint_index,
                        capture_stamp=endpoint_event.stamp, capture_command=list(endpoint_event.command),
                        inference_seconds=inference_seconds, total_seconds=time.monotonic()-started_batch,
                        command_action_count=len(request['ledger_steps']),
                        scheduling='all acquired likelihoods through the immutable frontier jointly',
                        joint_body_batch=True, intermediate_body_projections=0,
                        terminal_body_projections=1, stiffness_evidence_updates=1 if joint_fused_ids else 0,
                        physical_truth_received=False, posterior_retained=True, monte_carlo_used=False)
                    with monitor.stage('estimation_record_and_handoff', batch_id=batch_identity):
                        _append(Path(output)/'estimation.jsonl', joint_result.records)
                        last_message = dict(result=_handoff_result(joint_result), packet=packet,
                            event_id=endpoint_event.event_id, event_sequence=endpoint_event.sequence,
                            sensor_kind=endpoint_event.kind, capture_index=endpoint_index,
                            capture_stamp=endpoint_event.stamp, marker_observed=marker_observed(joint_result),
                            work_seconds=time.monotonic()-started_batch)
                    previous, previous_index = joint_result, endpoint_index
                with monitor.stage('raw_batch_completion_and_checkpoints', batch_id=batch_identity, event_count=len(batch.events)):
                    for offset, metadata in enumerate(batch.events):
                        event = store.get(metadata.event_id)
                        if (event.sequence != metadata.sequence or event.kind != metadata.kind
                                or event.stamp != metadata.stamp
                                or event.command != metadata.command or event.payload_hash != metadata.payload_hash
                                or event.payload_json != metadata.payload_json
                                or event.status != 'processing' or event.worker_id != batch.worker_id):
                            raise ValueError('An immutable batch observation changed after its claim.')
                        sensor = {'imu': 'imu', 'image': 'marker'}.get(event.kind)
                        initialize = previous is None and event.kind == 'joint'
                        use = initialize or sensor in enabled
                        try:
                            if use:
                                index = int(event.payload['frame_index'])
                                if joint_mode:
                                    result = joint_result
                                else:
                                    source_index = index if previous_index is None else previous_index
                                    if index < source_index or any(i not in by_index for i in range(source_index, index)):
                                        raise ValueError('The batch is missing an actually executed command interval.')
                                    steps = tuple(by_index[i] for i in range(source_index, index))
                                    result, last_message = evaluate(event, index, steps)
                                if event.kind == 'image':
                                    last_image_observed = (event.event_id in joint_fused_ids if joint_mode
                                        else last_message['marker_observed'])
                                evaluated.append(event.event_id)
                                completion = dict(observation_fused=(event.event_id in joint_fused_ids if joint_mode
                                    else bool(result.records.get('observation_ids'))),
                                    body_source_state_id=result.prior.state_id,
                                    inference_seconds=None if joint_mode else last_message['work_seconds'])
                                if joint_mode:
                                    completion.update(batch_work_seconds=last_message['work_seconds'],
                                        processing_time_scope='shared batch; per-event duration not measured')
                            else:
                                completion = dict(observation_fused=False, reason=(
                                    'command notification' if event.kind == 'joint' else 'sensor not enabled in this comparison case'))
                            completion.update(batch_id=batch_identity, batch_offset=offset,
                                              batch_snapshot_id=batch.snapshot_id)
                            store.complete(event.event_id, batch.worker_id, result_metadata=completion)
                        except Exception as error:
                            store.fail(event.event_id, batch.worker_id, str(error))
                            raise
                        processed.append(event.event_id)
                        checkpoint = dict(kind='inference_batch_progress', batch_id=batch_identity,
                            batch_offset=offset, event_id=event.event_id, event_sequence=event.sequence,
                            sensor_kind=event.kind, capture_stamp=event.stamp,
                            capture_index=event.payload_metadata['frame_index'], evaluated=use,
                            result_metadata=completion, completed_event_count=len(processed))
                        _append(Path(output)/'inference_batch_events.jsonl', checkpoint)
                        replies.put(checkpoint)
                record = dict(batch.record(), completed_event_ids=processed,
                    evaluated_event_ids=evaluated, completed=True,
                    last_candidate_event_id=None if last_message is None else last_message['event_id'],
                    last_image_observed=last_image_observed,
                    work_seconds=time.monotonic()-started_batch,
                    joint_body_batch=joint_mode,
                    per_event_evidence_and_proposal_preserved=not joint_mode,
                    terminal_body_projection_only=joint_mode,
                    all_claimed_events_completed=tuple(processed) == batch.event_ids,
                    all_raw_payloads_loaded_together=joint_mode, numerical_history_discarded=False)
                with monitor.stage('batch_completion_handoff', batch_id=batch_identity):
                    _append(Path(output)/'inference_batches.jsonl', record)
                    replies.put(dict(kind='inference_batch_completed', batch_id=batch_identity,
                        batch_record=record, final_message=last_message,
                        last_image_observed=last_image_observed,
                        evaluated_event_count=len(evaluated)))
            except Exception as error:
                # A failed joint computation precedes all per-event commits.
                # Keep every raw payload and name the blocked frontier in the
                # database, rather than leave it apparently running forever.
                unfinished = [item for item in batch.events if item.event_id not in processed]
                if unfinished:
                    blocked = store.get(unfinished[0].event_id, include_payload=False)
                    if blocked.status == 'processing' and blocked.worker_id == batch.worker_id:
                        store.fail(blocked.event_id, batch.worker_id, str(error))
                _append(Path(output)/'inference_batch_failures.jsonl', dict(
                    batch_id=batch_identity, batch_snapshot_id=batch.snapshot_id,
                    error_type=type(error).__name__, error=str(error),
                    completed_event_count=len(processed), unresolved_event_count=len(unfinished),
                    first_unresolved_event_id=None if not unfinished else unfinished[0].event_id,
                    original_raw_events_retained=True, final_law_handed_off=False))
                raise
            finally:
                store.close()
    except Exception:
        _worker_error(replies, 'inference')


def acquisition_worker(requests, replies, specification, database, output):
    """Physical-side camera rendering and durable raw-event acquisition.

    One producer queue supplies already timestamped joint/IMU readings and
    camera exposure requests. The latter's private world geometry is used
    solely here to produce pixels; only pixels/measurement metadata enter the
    public observation database. Neither the simulator nor its servo waits
    for rendering, PNG encoding or a database BLOB commit.
    """
    cv2.setNumThreads(1)
    try:
        board = specification['board']
        camera = (BoardCamera(specification['texture'], board,
                    specification['camera'], specification['seed'])
            if board.get('enabled', False) else PlanarCamera(specification['texture'],
                specification['marker_size'], specification['camera'], specification['seed']))
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        parameters = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco,
            'DetectorParameters') else cv2.aruco.DetectorParameters_create())
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = (cv2.aruco.ArucoDetector(dictionary, parameters)
                    if hasattr(cv2.aruco, 'ArucoDetector') else None)
        store = SensorEventStore(database)
        while True:
            request = requests.get()
            if request is None:
                store.close()
                return
            kind = request['kind']
            if kind=='database_operation':
                if request['operation']=='select_batch':
                    batch=store.claim_batch('inference', max_stamp=request['through_stamp'],
                        snapshot_id=request['snapshot_id'], include_payload=False,
                        initial_command_only=request.get('initial_command_only', False))
                    replies.put(dict(kind='database_batch_selection', batch=batch,
                        acquisition_watermark=request.get('acquisition_watermark')))
                elif request['operation']=='select':
                    selected=None
                    while True:
                        event=store.claim_next('inference',through_stamp=request['through_stamp'])
                        if event is None:
                            break
                        sensor={'imu':'imu','image':'marker'}.get(event.kind)
                        initialize=request['initialize'] and event.kind=='joint'
                        if initialize or (sensor is not None and sensor in request['enabled_sensors']):
                            selected=event
                            break
                        store.complete(event.event_id,'inference',result_metadata=dict(
                            observation_fused=False,reason=('command notification' if event.kind=='joint'
                                else 'sensor not enabled in this comparison case')))
                    replies.put(dict(kind='database_selection',event=selected,
                        acquisition_watermark=request.get('acquisition_watermark')))
                elif request['operation']=='complete':
                    store.complete(request['event_id'],'inference',
                        result_metadata=request['result_metadata'])
                    replies.put(dict(kind='database_completed',event_id=request['event_id']))
                elif request['operation']=='fail':
                    store.fail(request['event_id'],'inference',request['error'])
                    replies.put(dict(kind='database_failed',event_id=request['event_id']))
                else:
                    raise ValueError('Unknown observation database operation.')
                continue
            metadata = dict(time=request['time'], frame_index=request['frame_index'])
            received = request['received_wall_stamp']
            if kind != 'image':
                metadata.update(request['payload'])
                event = store.append(request['event_id'], kind, request['stamp'],
                    request['command'], metadata, received_wall_stamp=received)
                replies.put(dict(kind='raw_ready', sensor_kind=kind,
                    event_id=event.event_id, event_sequence=event.sequence,
                    frame_index=request['frame_index'], stamp=request['stamp']))
                continue
            store.reserve(request['event_id'], kind, request['stamp'],
                request['command'], received_wall_stamp=received)
            started = time.monotonic()
            image = camera.render(request['camera_pose'], request['marker_pose'],
                                  request['camera_history'])
            path = Path(output)/('capture_{:06d}.png'.format(request['frame_index']))
            preview = _archive_image(path, image, request['frame_index'], request['stamp'],
                dictionary, parameters, detector, specification['marker_ids'])
            metadata.update(image=image, image_stamp=request['stamp'],
                image_path=path.name,
                image_sha256=hashlib.sha256(image.tobytes()).hexdigest())
            event = store.supply_payload(request['event_id'], metadata)
            replies.put(dict(kind='image_ready', sensor_kind=kind,
                event_id=event.event_id, event_sequence=event.sequence,
                frame_index=request['frame_index'], stamp=request['stamp'],
                image=image, preview=preview, camera_diagnostics=dict(camera.diagnostics),
                work_seconds=time.monotonic()-started,
                image_sha256=metadata['image_sha256'], image_path=path.name))
    except Exception:
        _worker_error(replies, 'acquisition')
