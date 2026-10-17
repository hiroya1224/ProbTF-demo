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
            'stiffness', 'bayes_factor_surprise', 'sensor_event', 'candidate_measure')
    return replace(result, stages=(result.stages[-1],), sensor_posteriors=(),
        local_estimates=(), records={k: result.records[k] for k in keys if k in result.records},
        image=np.empty((0, 0), dtype=np.uint8), detection_image=np.empty((0, 0), dtype=np.uint8),
        frozen_body_prior=None, frozen_body_posterior=None, frozen_body_branches=(),
        projected_body_numerator=None)

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
                   control_mode=None):
    """Evaluate a frozen-law gradient, or the isolated comparison endpoint IK."""
    cv2.setNumThreads(1)
    try:
        inference = MovingInference(description, templates)
        settings = inference.settings['control']
        mode = control_mode or settings.get('mode', 'single_step_ik')
        if mode not in ('running_gradient', 'single_step_ik'):
            raise ValueError('Unknown asynchronous control mode: '+str(mode))
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
                    gradient_evaluation = controller.gradient_at_command(
                        predicted, command_map=command_map).record
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
                body_stamp=(source.prior.stamp if source.records.get('observation_ids') else None)
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
                    method='asynchronous full-J gradient only; no endpoint selection',
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
            snapshot = replace(build_snapshot(predicted, predicted.stages[-1], inference.common),
                summary='', show_labels=False, stiffness_record=predicted.records['stiffness'])
            display = inference.common['display']
            markers = make_markers(snapshot, display['seed'], display['point_size'])
            identity = str(request.get('request_id') or hashlib.sha256(
                (str(request.get('event_id'))+source.prior.state_id).encode()).hexdigest()[:12])
            path = Path(output)/'control_{:05d}_capture_{:05d}_request_{}'.format(
                request['index'], request['capture_index'], identity)
            save_snapshot(path, snapshot, markers, seed=display['seed'],
                          point_size=display['point_size'])
            MotionDisplay.stiffness_marker(snapshot, snapshot.stiffness_record)
            wire = io.BytesIO()
            markers.serialize(wire)
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
    except Exception:
        _worker_error(replies, 'control')


def event_inference_worker(requests, replies, description, urdf, templates, output,
                           probability_unit):
    """Apply each claimed sensor event once, in acquisition order.

    Only SensorEvent observations and the actual immutable action ledger cross
    this boundary. Camera/world poses and physical stiffness never enter it.
    """
    cv2.setNumThreads(1)
    try:
        inference = MovingInference(description, templates)
        package = Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
        settings = yaml.safe_load((package/'config/stiffness_identification.yaml').read_text())
        stiffness = StiffnessIdentifier(Path(urdf), tuple(inference.body_ids), settings)
        previous = None
        while True:
            request = requests.get()
            if request is None:
                return
            event, index, steps = request
            started = time.monotonic()
            command_path = (None if previous is None else RecordedCommandPath(previous, steps,
                body_ids=inference.body_ids, actions=inference.actions))
            result = inference.update_sensor_event(event, event_index=index,
                command_map=command_path, probability_unit=probability_unit)
            inference_seconds = time.monotonic()-started
            record = stiffness.update(result)
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
                    record.get('proposal_failure_reason')))
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
            _append(Path(output)/'estimation.jsonl', result.records)
            replies.put(dict(result=_handoff_result(result), packet=packet,
                event_id=event.event_id, event_sequence=event.sequence,
                sensor_kind=event.kind, capture_index=index, capture_stamp=event.stamp,
                marker_observed=marker_observed(result),
                work_seconds=time.monotonic()-started))
            previous = result
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
                if request['operation']=='select':
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
