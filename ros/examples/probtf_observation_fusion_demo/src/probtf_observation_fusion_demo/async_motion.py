"""Wall-clock motion, independent inference and repeated estimated-K IK.

Camera acquisition is archived in full. Inference takes the newest available
capture when idle, retaining its previous posterior and every intervening
command action. Selection and omission of captures are explicit in the log.
No simulator pose is sent to either numerical worker.
"""
from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
from threading import RLock
import time
import traceback
from types import SimpleNamespace

import cv2
import numpy as np
import rospy
import rospkg
import yaml
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import MarkerArray

from .async_state import ActionStep, RecordedCommandPath, predict_result
from .async_stiffness import (PhysicalActionEngine, capture_stiffness_packet,
                              execution_sample_record, execution_stiffness_packet, reanchor_retained_stiffness)
from .display import make_markers, save_snapshot, snapshot_record
from .execution_stiffness import LogStiffnessDamper
from .marker_search import MarkerSearchPolicy, marker_observed
from .model import load_inputs, component_log_weight
from .motion_display import MotionDisplay
from .moving_control import MovingController
from .moving_inference import MovingInference
from .moving_scene import MovingSwitchScene, SensorInputs
from .node import build_snapshot, image_message
from .stiffness_identification import StiffnessIdentifier
from .short_command import command_reference


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
            'stiffness', 'bayes_factor_surprise')
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


def inference_worker(requests, replies, description, urdf, templates, output):
    """Own the sole recursive posterior and K factor history in this process."""
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
            started = time.monotonic()
            inputs, index, steps = request
            command_path = (None if previous is None else RecordedCommandPath(previous, steps,
                body_ids=inference.body_ids, actions=inference.actions))
            result = inference.update(inputs, index, command_map=command_path)
            inference_seconds = time.monotonic()-started
            record = stiffness.update(result)
            proposal = stiffness.command_proposal(result)
            result.records['stiffness'] = record
            if stiffness.factors and proposal is None:
                raise ValueError('No stable estimated-K representative: '+str(
                    record.get('proposal_failure_reason')))
            packet = None if proposal is None else capture_stiffness_packet(result)
            result.records['async_estimation'] = dict(capture_index=index,
                capture_stamp=inputs.stamp, capture_command=list(inputs.command),
                inference_seconds=inference_seconds,
                total_seconds=time.monotonic()-started,
                command_action_count=len(steps), physical_truth_received=False,
                scheduling='newest archived capture when worker is idle',
                posterior_retained=True, monte_carlo_used=False)
            _append(Path(output)/'estimation.jsonl', result.records)
            replies.put(dict(result=_handoff_result(result), packet=packet,
                marker_observed=marker_observed(result), capture_index=index,
                capture_stamp=inputs.stamp, work_seconds=time.monotonic()-started))
            previous = result
    except Exception:
        _worker_error(replies, 'inference')


def control_worker(requests, replies, description, urdf, templates, output, horizon):
    """Freeze one damped execution K with each complete-criterion IK plan."""
    cv2.setNumThreads(1)
    try:
        inference = MovingInference(description, templates)
        settings = inference.settings['control']
        controller = MovingController(inference, description, settings['goal_offset'],
            settings['goal_orientation_wxyz'], settings,
            integration_steps=int(settings['integration_steps']))
        package = Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
        parameter_settings = yaml.safe_load((package/'config/stiffness_identification.yaml').read_text())
        engine = PhysicalActionEngine(Path(urdf), tuple(inference.body_ids), parameter_settings)
        while True:
            request = requests.get()
            if request is None:
                return
            started = time.monotonic()
            source = _freeze_arrays(request['result'])
            predicted = predict_result(source, request['steps'],
                target_stamp=request['stamp'], frame_index=request['index'],
                body_ids=inference.body_ids, actions=inference.actions)
            if request['packet'] is None:
                execution_packet = None
                predicted = _unselected_parameter_view(predicted, source.records['stiffness'])
                command_map = None
            else:
                execution_packet = execution_stiffness_packet(engine, predicted, request['packet'],
                    request['execution_sample'], request.get('execution_packet'))
                predicted, command_map = reanchor_retained_stiffness(engine, predicted, execution_packet)
            controller.velocity = np.array(request['velocity'], copy=True)
            command, diagnostics = controller.reaching_step(predicted, horizon, command_map=command_map)
            diagnostics.update(ik_update_count=1, future_ik_updates=0, lookahead_enabled=False)
            planned_velocity = (np.asarray(command)-np.asarray(predicted.prior.command))/horizon
            predicted.records['controller'] = diagnostics
            predicted.records['async_control'] = dict(
                source_capture_index=request['capture_index'], prediction_index=request['index'],
                prediction_stamp=request['stamp'], horizon_s=diagnostics['duration'],
                ik_update_step_s=horizon, ik_update_count=1,
                source_capture_stamp=source.prior.stamp,
                source_capture_command=list(source.prior.command),
                requested_velocity=(np.asarray(command)-np.asarray(predicted.prior.command)).tolist(),
                velocity_units='joint displacement over horizon; see desired_velocity_rad_s',
                desired_velocity_rad_s=planned_velocity.tolist(),
                physical_truth_received=False, monte_carlo_used=False)
            snapshot = replace(build_snapshot(predicted, predicted.stages[-1], inference.common),
                summary='', show_labels=False, stiffness_record=predicted.records['stiffness'])
            display = inference.common['display']
            markers = make_markers(snapshot, display['seed'], display['point_size'])
            path = Path(output)/'control_{:05d}_capture_{:05d}'.format(
                request['index'], request['capture_index'])
            save_snapshot(path, snapshot, markers, seed=display['seed'],
                          point_size=display['point_size'])
            MotionDisplay.stiffness_marker(snapshot, snapshot.stiffness_record)
            wire = io.BytesIO()
            markers.serialize(wire)
            _append(Path(output)/'control.jsonl', dict(index=request['index'],
                capture_index=request['capture_index'], artifact_prefix=path.name,
                markers_wire_sha256=hashlib.sha256(wire.getvalue()).hexdigest(),
                prediction=predicted.records['async_prediction'],
                controller=diagnostics, stiffness=snapshot.stiffness_record,
                snapshot=snapshot_record(snapshot), work_seconds=time.monotonic()-started))
            replies.put(dict(index=request['index'], capture_index=request['capture_index'],
                packet=execution_packet, velocity=planned_velocity.copy(),
                snapshot=snapshot, markers_wire=wire.getvalue(), diagnostics=diagnostics,
                work_seconds=time.monotonic()-started))
    except Exception:
        _worker_error(replies, 'control')


class AsyncMotionSession:
    """Nonblocking coordinator; only geometry, servo limits and cameras here."""
    def __init__(self, observations_path, output, *, imu_target_link=None,
                 duration=None, clock=time.monotonic, context=None):
        self.clock, self.output = clock, Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.package = Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
        _, self.templates = load_inputs(self.package/'config/common.yaml', observations_path)
        self.scene = MovingSwitchScene(self.output, imu_target_link=imu_target_link)
        if duration is not None:
            self.scene.duration = float(duration)
        self.inference_description = MovingInference(self.scene.robot_description, self.templates)
        self.settings = self.inference_description.settings
        board = self.settings.get('marker_board', {})
        self.preview_marker_ids = (tuple(int(value) for row in board['ids'] for value in row)
                                   if board.get('enabled', False) else (int(self.settings['marker']['id']),))
        self.options = dict(self.settings.get('async_execution', {}))
        self.period = float(self.options.get('motion_period_s', .05))
        self.camera_period = 1./float(self.options.get('camera_hz', 15.))
        self.horizon = float(self.options.get('ik_horizon_s', .5))
        if not (self.period > 0. and self.camera_period > 0. and self.horizon >= self.period):
            raise ValueError('Positive asynchronous periods and an IK horizon >= motion period are required.')
        settings = self.settings['control']
        self.execution_filter = LogStiffnessDamper(settings.get('execution_stiffness', {}))
        self.execution_sample = self.execution_target_packet = None
        self.servo = MovingController(self.inference_description, self.scene.robot_description,
            settings['goal_offset'], settings['goal_orientation_wxyz'], settings)
        parameter_settings = yaml.safe_load((self.package/'config/stiffness_identification.yaml').read_text())
        self.engine = PhysicalActionEngine(self.scene.urdf,
            tuple(self.inference_description.body_ids), parameter_settings)
        self.behavior = MarkerSearchPolicy(settings.get('marker_search', {}),
            marker_enabled=any(item['sensor'] == 'marker' for item in self.templates),
            exposure_time=self.scene.config['camera']['exposure_time'])
        self.context = context or mp.get_context('spawn')
        self.estimate_requests, self.estimate_replies = self.context.Queue(), self.context.Queue()
        self.control_requests, self.control_replies = self.context.Queue(), self.context.Queue()
        common = (self.scene.robot_description, str(self.scene.urdf), self.templates, str(self.output))
        self.workers = (
            self.context.Process(target=inference_worker,
                args=(self.estimate_requests, self.estimate_replies)+common, daemon=True),
            self.context.Process(target=control_worker,
                args=(self.control_requests, self.control_replies)+common+(self.horizon,), daemon=True))
        for worker in self.workers:
            worker.start()
        self.command = self.scene.initial_command.copy()
        self.velocity = np.zeros_like(self.command)
        self.desired_velocity = self.velocity.copy()
        self.actions, self.frames, self.capture_indices, self.assimilated = [], [], [], []
        self.latest_result = self.latest_packet = self.latest_capture = None
        self.latest_marker_observed = False
        self.current_control = None
        self.inference_busy = self.control_busy = False
        self.last_submitted_capture = self.last_assimilated_capture = None
        self.last_capture_time = -np.inf
        self.next_capture_time = 0.
        self.started = None
        self.last_action_mode = 'initial_hold'
        self.last_marker_missing_capture = -1
        self.marker_preview_visible = None
        self.failure = None
        self.closed = False
        self.archive_executor = ThreadPoolExecutor(max_workers=1)
        self.archive_jobs = []
        self.archives_collected = 0
        self.preview_capture_index = None
        self.last_tick_timings = {}
        self.publish_times, self.capture_wall_times, self.control_times, self.estimate_times = [], [], [], []
        self.source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(Path(__file__).parent.glob('*.py'))}
        self._dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self._detector_parameters = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco,
            'DetectorParameters') else cv2.aruco.DetectorParameters_create())
        self._detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._preview_detector = (cv2.aruco.ArucoDetector(self._dictionary, self._detector_parameters)
                                  if hasattr(cv2.aruco, 'ArucoDetector') else None)
        self.frame = self.scene.step(0., stamp=1792080000., command=self.command)
        self.frames.append(self.frame)
        self.scene.trajectory[-1].update(action_mode='reaching', command_interval_mode='initial_hold',
            reaching_phase='waiting_for_control', reaching_phase_source=None)
        _append(self.output/'physical_trajectory.jsonl', self.scene.trajectory[-1])
        self.preview_image = self.frame.image
        self.capture()
        self.submit_estimate()

    @property
    def motion_finished(self):
        return self.frame.time >= self.scene.duration

    @property
    def estimation_finished(self):
        return (self.motion_finished and not self.inference_busy
                and self.last_assimilated_capture == self.capture_indices[-1]
                and self.archives_collected == len(self.archive_jobs))

    def start(self):
        if self.started is None and self.latest_result is not None:
            self.started = self.clock()
            self.publish_times.clear()
            self.capture_wall_times.clear()
            return True
        return False

    def capture(self):
        """Archive every acquired image; distinguish acquisition and assimilation."""
        index = len(self.frames)-1
        inputs = self.scene.sensor_inputs(self.frame)
        path = self.output/'capture_{:05d}.png'.format(index)
        self.archive_jobs.append(self.archive_executor.submit(_archive_image, path, inputs.image,
            index, inputs.stamp, self._dictionary, self._detector_parameters,
            self._preview_detector, self.preview_marker_ids))
        self.latest_capture = (index, inputs)
        self.capture_indices.append(index)
        self.last_capture_time = self.frame.time
        while self.next_capture_time <= self.frame.time+1e-9:
            self.next_capture_time += self.camera_period
        self.capture_wall_times.append(self.clock())
        _append(self.output/'captures.jsonl', dict(index=index, stamp=inputs.stamp,
            time=inputs.time, command=inputs.command.tolist(), image=str(path.name),
            image_sha256=hashlib.sha256(inputs.image.tobytes()).hexdigest(),
            gravity_in_sensor=inputs.gravity_in_sensor.tolist(), imu_target_link=inputs.imu_target_link,
            imu_sigma_rad=inputs.imu_sigma_rad, image_space_marker_visible=None,
            observation_assimilated_at_capture=False))

    def collect_archives(self):
        """Read completed previews without delaying physics for PNG/detection."""
        while self.archives_collected < len(self.archive_jobs):
            job = self.archive_jobs[self.archives_collected]
            if not job.done():
                break
            result = job.result()
            self.archives_collected += 1
            self.preview_image = result['image']
            self.preview_capture_index = result['index']
            self.marker_preview_visible = result['visible']
            if not result['visible']:
                self.last_marker_missing_capture = result['index']
            _append(self.output/'camera_previews.jsonl', {k: result[k] for k in
                ('index', 'stamp', 'visible', 'visible_marker_ids') if k in result})

    def _steps_since(self, frame_index, target_index=None):
        target_index = len(self.frames)-1 if target_index is None else target_index
        return tuple(self.actions[frame_index:target_index])

    def submit_estimate(self):
        if self.inference_busy or self.latest_capture is None:
            return
        index, inputs = self.latest_capture
        if index == self.last_submitted_capture:
            return
        steps = () if self.last_submitted_capture is None else self._steps_since(
            self.last_submitted_capture, index)
        omitted = [i for i in self.capture_indices if (
            (self.last_submitted_capture is None or i > self.last_submitted_capture) and i < index)]
        _append(self.output/'assimilation_selection.jsonl', dict(selected_capture_index=index,
            selected_stamp=inputs.stamp, previous_capture_index=self.last_submitted_capture,
            archived_not_assimilated_capture_indices=omitted,
            scheduling='latest available capture; all captured data archived',
            command_interval_count=len(steps), posterior_reset=False))
        self.estimate_requests.put((inputs, index, steps))
        self.last_submitted_capture, self.inference_busy = index, True

    def submit_control(self):
        if self.control_busy or self.latest_result is None:
            return
        index = len(self.frames)-1
        if self.current_control is not None and (index == self.current_control['index']
                and self.current_control['capture_index'] == self.last_assimilated_capture):
            return
        self._sample_execution_stiffness(self.frame.stamp)
        self.control_requests.put(dict(result=self.latest_result, packet=self.execution_target_packet,
            execution_sample=self.execution_sample,
            execution_packet=None if self.current_control is None else self.current_control['packet'],
            capture_index=self.last_assimilated_capture, index=index, stamp=self.frame.stamp,
            steps=self._steps_since(self.last_assimilated_capture), velocity=self.velocity.copy()))
        self.control_busy = True

    def _sample_execution_stiffness(self, stamp):
        """Advance at motion cadence, even while both numerical workers wait."""
        target = self.latest_packet if self.latest_packet is not None else self.execution_target_packet
        if target is None:
            return
        if (self.execution_sample is not None and self.execution_sample['stamp'] == stamp
                and target is self.execution_target_packet):
            return
        self.execution_sample = execution_sample_record(
            self.execution_filter.advance(target.proposal['selected_K'], stamp), target)
        self.execution_target_packet = target
        _append(self.output/'execution_stiffness.jsonl', self.execution_sample)

    def _reaching_plan_velocity(self, dt, interval_end_stamp):
        """Follow the single selected position command without extrapolating it."""
        if self.current_control is None:
            return self.desired_velocity.copy(), dict(method='initial hold; no completed IK plan')
        diagnostics = self.current_control['diagnostics']
        reference = command_reference(diagnostics, interval_end_stamp)
        requested = (np.asarray(reference['reference_command'])-self.command)/dt
        reference.update(method='single selected IK position reference; ordinary actuator bounds apply',
            source_capture_index=self.current_control['capture_index'],
            execution_packet_id=None if self.current_control['packet'] is None else self.current_control['packet'].packet_id,
            interval_end_stamp=interval_end_stamp, requested_velocity=requested.tolist(),
            actual_command_before=self.command.tolist(),
            new_IK_or_distribution_evaluation=False, physical_truth_received=False)
        return requested, reference

    def poll(self):
        """Queue reads never wait for numerical work or share its mutable models."""
        changed = False
        for kind, queue in (('inference', self.estimate_replies), ('control', self.control_replies)):
            try:
                message = queue.get_nowait()
            except Empty:
                continue
            if message.get('failed'):
                self.failure = message
                _append(self.output/'execution_failure.jsonl', message)
                raise RuntimeError(message['worker']+' worker failed: '+message['traceback'])
            if kind == 'inference':
                self.inference_busy = False
                self.latest_result = _freeze_arrays(message['result'])
                self.latest_marker_observed = message.get('marker_observed', marker_observed(self.latest_result))
                self.latest_packet = message['packet']
                if self.latest_packet is not None and self.current_control is None:
                    self.engine.set(self.latest_packet)
                self.last_assimilated_capture = message['capture_index']
                self.assimilated.append(message['capture_index'])
                self.estimate_times.append(self.clock())
                changed = True
            else:
                self.control_busy = False
                # Commit a completed law/K/IK snapshot atomically. Rejecting
                # every plan whenever inference overtakes IK can starve all
                # motion. The next request uses the newest available update.
                if (self.current_control is None or message['capture_index']
                        >= self.current_control['capture_index']):
                    self.current_control = _freeze_arrays(message)
                    self.desired_velocity = np.array(message['velocity'], copy=True)
                    if message['packet'] is not None:
                        self.engine.set(message['packet'])
                    self.control_times.append(self.clock())
                    changed = True
        for kind, worker in zip(('inference', 'control'), self.workers):
            if not worker.is_alive() and worker.exitcode is not None and self.failure is None:
                raise RuntimeError(kind+' worker exited unexpectedly: '+str(worker.exitcode))
        return changed

    def advance(self):
        started = previous = self.clock()
        self.poll()
        self.collect_archives()
        timings = dict(receive_worker_results_s=self.clock()-previous)
        if self.started is not None and not self.motion_finished:
            elapsed = min(self.scene.duration, self.clock()-self.started)
            dt = elapsed-self.frame.time
            if dt > 1e-6:
                self._sample_execution_stiffness(1792080000.+elapsed)
                previous = self.clock()
                self.servo.velocity = self.velocity.copy()
                # Pixel detection stops an unnecessary search while the
                # numerical workers catch up. Reaching additionally needs
                # the matching fused goal and committed IK snapshot.
                active_goal = ({} if self.current_control is None else
                    self.current_control['snapshot'].goal_memory or {})
                loss_stamp = (-np.inf if self.last_marker_missing_capture < 0 else
                    self.frames[self.last_marker_missing_capture].stamp)
                visible = (self.marker_preview_visible
                    and self.latest_marker_observed
                    and self.current_control is not None
                    and self.current_control['capture_index'] > self.last_marker_missing_capture
                    and active_goal.get('available', False)
                    and active_goal['observation_stamp'] > loss_stamp
                    and self.last_assimilated_capture > self.last_marker_missing_capture)
                observation = SimpleNamespace(posterior=True, detector_result=True)
                policy_input = SimpleNamespace(prior=SimpleNamespace(command=tuple(self.command),
                    stamp=self.frame.stamp, version=3*(len(self.frames)-1)+3),
                    local_estimates=(observation,) if visible else (),
                    records={'fixed_goal_memory': active_goal})
                plan = self.behavior.plan(policy_input, self.velocity, elapsed,
                    joint_lower=self.servo.lower, joint_upper=self.servo.upper,
                    marker_detected=bool(self.marker_preview_visible))
                plan.record.update(marker_detection_capture_index=self.preview_capture_index,
                    marker_detection_stamp=(None if self.preview_capture_index is None else
                        float(self.frames[self.preview_capture_index].stamp)))
                reaching_execution = None
                if plan.mode == 'reaching':
                    lower, upper, _, _ = self.servo._command_bounds(self.command, dt)
                    requested_velocity, reaching_execution = self._reaching_plan_velocity(
                        dt, 1792080000.+elapsed)
                    delta = np.clip(dt*requested_velocity, lower, upper)
                    command = self.command+delta
                    self.velocity = delta/dt
                else:
                    # Local control has no objective/finite-difference work.
                    # Use its identical servo bounds and braking formula,
                    # without asking it to evaluate a distribution.
                    lower, upper, _, _ = self.servo._command_bounds(self.command, dt,
                        position_lower=plan.position_lower, position_upper=plan.position_upper,
                        max_speed=plan.max_speed)
                    target = self.command if plan.target_command is None else plan.target_command
                    displacement = np.asarray(target)-self.command
                    distance = np.abs(displacement)
                    braking_change = self.servo.acceleration*dt
                    stopping_speed = (2*self.servo.acceleration*distance /
                        (np.sqrt(braking_change**2+2*self.servo.acceleration*distance)+braking_change))
                    speed = (self.servo.speed if plan.max_speed is None else
                             min(self.servo.speed, plan.max_speed))
                    desired = np.sign(displacement)*np.minimum(stopping_speed, speed)
                    delta = np.clip(dt*desired, lower, upper)
                    command = self.command+delta
                    self.velocity = delta/dt
                timings['servo_s'] = self.clock()-previous
                previous = self.clock()
                action = (dict(joint_increment=(command-self.command).tolist(),
                    density_id=None, selected_K=None,
                    captured_body_posterior_versions=[None],
                    method='direct input family right action; no observed K representative yet',
                    physical_truth_received=False, body_reconstructed_from_stiffness=False)
                    if self.engine.packet is None else self.engine.increment(self.command, command))
                step = ActionStep(index=len(self.actions), start_stamp=self.frame.stamp,
                    end_stamp=1792080000.+elapsed, start_command=tuple(self.command),
                    end_command=tuple(command), joint_increment=tuple(action['joint_increment']),
                    density_id=action.get('density_id'),
                    estimate_version=action['captured_body_posterior_versions'][0],
                    selected_K=action.get('selected_K'), anchor_command=action.get('numerical_branch_anchor_command'),
                    anchor_equilibrium=action.get('numerical_branch_anchor'),
                    source_state_ids=tuple(action.get('captured_body_posterior_state_ids', ())),
                    source_stamp=action.get('captured_source_stamp'),
                    source_version=action['captured_body_posterior_versions'][0], provenance=action)
                self.actions.append(step)
                self.command = command
                timings['model_action_s'] = self.clock()-previous
                previous = self.clock()
                capture_due = elapsed >= self.next_capture_time-1e-9 or elapsed >= self.scene.duration
                self.frame = self.scene.step(elapsed, stamp=step.end_stamp, command=command,
                                             render_camera=capture_due)
                timings['physics_and_camera_s'] = self.clock()-previous
                self.frames.append(self.frame)
                phase, phase_source = self.reaching_phase()
                self.scene.trajectory[-1].update(action_mode=plan.mode,
                    command_interval_mode=plan.mode, search_amplitude_rad=plan.record['amplitude_rad'],
                    search_completed_roundtrips=plan.record['completed_roundtrips'],
                    reaching_phase=phase, reaching_phase_source=phase_source)
                _append(self.output/'physical_trajectory.jsonl', self.scene.trajectory[-1])
                self.last_action_mode = plan.mode
                self.publish_times.append(self.clock())
                _append(self.output/'commands.jsonl', dict(step=step.record(),
                    velocity=self.velocity.tolist(), desired_velocity=self.desired_velocity.tolist(),
                    behavior=plan.record, reaching_execution=reaching_execution,
                    reaching_phase=phase, reaching_phase_source=phase_source,
                    wall_elapsed_s=self.clock()-self.started))
                previous = self.clock()
                if capture_due:
                    self.capture()
                timings['capture_archive_and_preview_s'] = self.clock()-previous
        self.submit_estimate()
        self.submit_control()
        timings['coordinator_s'] = self.clock()-started
        self.last_tick_timings = timings
        _append(self.output/'coordinator_timing.jsonl', dict(frame_index=len(self.frames)-1,
            physical_time=self.frame.time, timings_s=timings))
        return self.frame

    def reaching_phase(self):
        if self.current_control is None:
            return 'waiting_for_control', None
        diagnostics = self.current_control['diagnostics']
        return diagnostics.get('reaching_phase', 'indet_ik'), {
            key: diagnostics[key] for key in ('source_stamp', 'source_command', 'source_version',
                                              'body_source_state_ids', 'goal_source_state_ids')}

    def manifest(self):
        def hz(values):
            # Initial inference/IK can precede a long user-requested wait.
            # That pause is not computation time during the moving experiment.
            values = values if self.started is None else [v for v in values if v >= self.started]
            return (len(values)-1)/(values[-1]-values[0]) if len(values)>1 else None
        snapshot = None if self.current_control is None else self.current_control['snapshot']
        planned_phase, planned_phase_source = self.reaching_phase()
        row = self.scene.trajectory[-1]
        phase = row['reaching_phase']
        phase_source = row['reaching_phase_source']
        return dict(execution_mode='async', offline_precomputation=False,
            reaching_phase=phase, reaching_phase_source=phase_source,
            planned_reaching_phase=planned_phase, planned_reaching_phase_source=planned_phase_source,
            sample_index=len(self.frames)-1, estimate_index=self.last_assimilated_capture,
            simulated_elapsed_s=self.frame.time, duration=self.scene.duration,
            finished=self.motion_finished, estimation_drained=self.estimation_finished,
            actual_motion_hz=hz(self.publish_times), actual_camera_hz=hz(self.capture_wall_times),
            actual_control_hz=hz(self.control_times), actual_estimate_hz=hz(self.estimate_times),
            actual_update_hz=hz(self.estimate_times), simulated_step_s=self.period,
            physical_stamp=self.frame.stamp, command=self.command.tolist(),
            estimate_stamp=None if self.latest_result is None else self.latest_result.prior.stamp,
            displayed_prediction_stamp=None if snapshot is None else snapshot.stamp,
            displayed_prediction_command=None if snapshot is None else list(snapshot.command),
            displayed_prediction_age_s=None if snapshot is None else self.frame.stamp-snapshot.stamp,
            observation_age_s=None if self.latest_result is None else self.frame.stamp-self.latest_result.prior.stamp,
            acquired_captures=len(self.capture_indices), assimilated_captures=list(self.assimilated),
            archived_not_assimilated_captures=[i for i in self.capture_indices if i not in self.assimilated],
            inference_busy=self.inference_busy, control_busy=self.control_busy,
            coordinator_last_timings_s=self.last_tick_timings,
            active_parameter_packet_id=None if self.engine.packet is None else self.engine.packet.packet_id,
            latest_estimated_parameter_packet_id=None if self.latest_packet is None else self.latest_packet.packet_id,
            latest_estimated_K=None if self.latest_packet is None else self.latest_packet.proposal['selected_K'],
            interpolated_execution_K=None if self.execution_sample is None else self.execution_sample['execution_K'],
            interpolated_execution_K_stamp=None if self.execution_sample is None else self.execution_sample['stamp'],
            active_execution_K=None if self.engine.packet is None else self.engine.stiffness.tolist(),
            active_ik_capture_index=None if self.current_control is None else self.current_control['capture_index'],
            ik_plan_horizon_s=(self.horizon
                if self.current_control is None else self.current_control['diagnostics']['duration']),
            ik_update_step_s=self.horizon,
            ik_update_count=1,
            future_ik_updates=0, lookahead_enabled=False,
            ik_command_nominal_end_stamp=(None if self.current_control is None else
                self.current_control['diagnostics']['source_stamp']+self.current_control['diagnostics']['duration']),
            ik_command_past_selected_duration=(self.current_control is not None and
                self.frame.stamp >= self.current_control['diagnostics']['source_stamp']
                    +self.current_control['diagnostics']['duration']),
            marker_search_mode=self.last_action_mode,
            implementation_sha256=self.source_hashes,
            implementation_unchanged_during_execution=self.source_hashes == {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob('*.py'))},
            physics_and_camera_do_not_wait_for_inference=True,
            all_captured_sensor_data_archived=True, posterior_retained=True,
            archived_capture_files_written=sum(job.done() and job.exception() is None for job in self.archive_jobs),
            monte_carlo_used=False, body_reconstructed_from_stiffness=False,
            reached=self.frame.reached, collided=self.frame.collided,
            final_physical_error_m=self.scene.trajectory[-1]['physical_error_m'])

    def close(self):
        if self.closed:
            return
        self.closed = True
        for queue in (self.estimate_requests, self.control_requests):
            queue.put(None)
        for worker in self.workers:
            worker.join(timeout=2.)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1.)
        for queue in (self.estimate_requests, self.estimate_replies,
                      self.control_requests, self.control_replies):
            queue.cancel_join_thread()
            queue.close()
        self.archive_executor.shutdown(wait=True)


class AsyncMotionLive:
    """ROS publication of independent physical and law sources with their ages."""
    def __init__(self, observations_path, output, case, autostart, *, imu_target_link=None,
                 max_updates=0, duration=None):
        if max_updates:
            raise ValueError('max_updates is a synchronous audit limit; use async_duration_s for async runs.')
        self.lock, self.case = RLock(), case
        self.observations_path, self.output = Path(observations_path), Path(output)
        self.imu_target_link, self.duration = imu_target_link, duration
        self.autostart, self.replays = bool(autostart), 0
        cv2.setNumThreads(1)
        self.session = AsyncMotionSession(observations_path, output,
            imu_target_link=imu_target_link, duration=duration)
        self.distributions = rospy.Publisher('/observation_fusion/distributions', MarkerArray, queue_size=4, latch=True)
        self.snapshots = rospy.Publisher('/observation_fusion/snapshot', String, queue_size=4, latch=True)
        self.stiffness = rospy.Publisher('/observation_fusion/stiffness', String, queue_size=4, latch=True)
        self.images = rospy.Publisher('/observation_fusion/detection_image', Image, queue_size=2, latch=True)
        self.motion = rospy.Publisher('/observation_fusion/motion_metadata', String, queue_size=4, latch=True)
        self.display = MotionDisplay(self.session.scene, case, [i['sensor'] for i in self.session.templates])
        self.published_control = self.published_image = None
        self.last_scene_publication = -np.inf
        self.service = rospy.Service('/observation_fusion/replay', Trigger, self.replay)
        rospy.on_shutdown(lambda: self.session.close())
        self.timer = rospy.Timer(rospy.Duration(self.session.period), self.advance)

    def advance(self, _event):
        with self.lock:
            try:
                self.session.advance()
                if self.autostart and self.session.current_control is not None:
                    self.session.start()
                current = self.session.current_control
                if current is not None:
                    key = (current['capture_index'], current['index'])
                    if key != self.published_control:
                        snapshot = current['snapshot']
                        self.distributions.publish(rospy.AnyMsg().deserialize(current['markers_wire']))
                        self.snapshots.publish(String(json.dumps(snapshot_record(snapshot))))
                        self.stiffness.publish(String(json.dumps(snapshot.stiffness_record)))
                        self.published_control = key
                    metadata = self.session.manifest()
                    metadata['displayed_snapshot'] = dict(stamp=current['snapshot'].stamp,
                        command=list(current['snapshot'].command), version=current['snapshot'].version,
                        body_source_ids=current['diagnostics']['body_source_state_ids'],
                        goal_source_ids=current['diagnostics']['goal_source_state_ids'])
                    scene_due = (time.monotonic()-self.last_scene_publication >=
                        1./float(self.session.settings['display']['publish_hz']))
                    self.display.publish(self.session.frame, len(self.session.frames)-1,
                        current['snapshot'], current['capture_index'],
                        stiffness_record=current['snapshot'].stiffness_record, execution=metadata,
                        publish_scene=scene_due)
                    if scene_due:
                        self.last_scene_publication = time.monotonic()
                    self.motion.publish(String(json.dumps(metadata)))
                    (self.session.output/'motion_manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
                image_index = self.session.preview_capture_index
                if image_index is not None and image_index != self.published_image:
                    self.images.publish(image_message(self.session.preview_image,
                        self.session.frames[image_index].stamp, 'camera_optical_frame'))
                    self.published_image = image_index
            except Exception as error:
                self.timer.shutdown()
                self.session.failure = dict(failed=True, error=str(error), traceback=traceback.format_exc())
                _append(self.session.output/'execution_failure.jsonl', self.session.failure)
                self.motion.publish(String(json.dumps(self.session.failure)))
                self.session.close()
                rospy.logerr('Async execution stopped: %s', error)

    def replay(self, _request):
        with self.lock:
            if self.session.started is None:
                self.autostart = True
                ready = (self.session.current_control is not None and self.session.start())
                return TriggerResponse(True, 'Starting asynchronous motion.' if ready else 'Starting after initial estimate.')
            self.session.close()
            self.replays += 1
            path = self.output.with_name(self.output.name+'_replay_{:03d}'.format(self.replays))
            self.session = AsyncMotionSession(self.observations_path, path,
                imu_target_link=self.imu_target_link, duration=self.duration)
            self.display.scene = self.session.scene
            self.published_control = self.published_image = None
            self.autostart = True
            return TriggerResponse(True, 'Starting a fresh asynchronous experiment.')
