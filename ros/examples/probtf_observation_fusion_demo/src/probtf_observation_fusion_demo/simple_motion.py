"""Minimal asynchronous reference: one joint body/environment law, one log-K law.

The mathematical recurrence has one owner. The coordinator only executes finite
position segments and records the actual command/K intervals. Acquisition and
ROS drawing reuse the unchanged adapters in separate processes.
"""
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import time
import traceback

import numpy as np
import rospkg
import yaml

from .async_state import ActionStep, RecordedCommandPath
from .async_stiffness import PhysicalActionEngine, capture_stiffness_packet
from .async_workers import _append, acquisition_worker
from .event_motion import AsyncMotionSession
from .model import load_inputs
from .moving_scene import MovingSwitchScene
from .sensor_event_store import SensorEventStore


def simple_worker(requests, replies, description, urdf, templates, output,
                  body_surprise_enabled=None):
    """Own the two recursive laws; publish completed laws without batch waits."""
    import cv2
    from .simple_inference import SimpleInference
    from .simple_stiffness import SimpleStiffnessIdentifier
    cv2.setNumThreads(1)
    try:
        inference = SimpleInference(description, templates,
                                    body_surprise_enabled=body_surprise_enabled)
        package = Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
        settings = yaml.safe_load((package/'config/stiffness_identification.yaml').read_text())
        settings.update(inference.settings['simple_stiffness'])
        settings['bayes_factor_surprise'] = dict(enabled=False)
        stiffness = SimpleStiffnessIdentifier(urdf, tuple(inference.body_ids), settings)
        (Path(output)/'simple_model.json').write_text(json.dumps(dict(
            body_edge_ids=list(inference.body_ids), hand_path=inference.paths['hand'],
            marker_path=inference.paths['marker'], goal_offset=inference.settings['control']['goal_offset'],
            root_frame=inference.settings['root_frame'], urdf_path=urdf), indent=2)+'\n')
        enabled = {item['sensor'] for item in templates}
        previous, previous_index, packet, last_effective = None, None, None, None
        while True:
            request = requests.get()
            if request is None:
                return
            if previous_index != request['source_index']:
                raise ValueError('The simple recurrence lost its actual preceding acquisition.')
            steps = {step.index: step for step in request['steps']}
            selected = frozenset(request['selection']['selected_event_ids'])
            processed = []
            with SensorEventStore(request['database']) as store:
                for metadata in request['batch'].events:
                    event = store.get(metadata.event_id)
                    if (event.stamp != metadata.stamp or event.command != metadata.command
                            or event.sequence != metadata.sequence or event.payload_hash != metadata.payload_hash):
                        raise ValueError('A claimed original acquisition changed before processing.')
                    index = int(event.payload['frame_index'])
                    interval = (() if previous_index is None else
                        tuple(steps[i] for i in range(previous_index, index)))
                    path = (None if previous is None else RecordedCommandPath(previous, interval,
                        body_ids=inference.body_ids, actions=inference.actions))
                    eligible = {'imu': 'imu', 'image': 'marker'}.get(event.kind) in enabled
                    saved_templates = inference.templates
                    if event.event_id not in selected:
                        inference.templates = ()
                    before = time.monotonic()
                    try:
                        result = inference.update_sensor_event(event, event_index=index, command_map=path)
                    finally:
                        inference.templates = saved_templates
                    body_seconds = time.monotonic()-before
                    before = time.monotonic()
                    record = None
                    observed = bool(result.records.get('observation_ids'))
                    if previous is None or observed:
                        record = stiffness.update(result)
                        proposal = stiffness.command_proposal(result)
                        result.records['stiffness'] = record
                        if proposal is not None:
                            packet = capture_stiffness_packet(result)
                    K_seconds = time.monotonic()-before if record is not None else 0.
                    if observed:
                        last_effective = (event.stamp, list(event.command))
                    result.records['last_effective_observation_stamp'] = None if last_effective is None else last_effective[0]
                    result.records['last_effective_observation_command'] = None if last_effective is None else last_effective[1]
                    _append(Path(output)/'simple_updates.jsonl', dict(
                        event_id=event.event_id, sequence=event.sequence, kind=event.kind,
                        acquisition_stamp=event.stamp, acquisition_command=list(event.command),
                        frame_index=index, body_seconds=body_seconds, K_seconds=K_seconds,
                        observation_eligible=eligible,
                        observation_selected=event.event_id in selected,
                        selection_policy=request['selection']['mode'],
                        K_update_skipped_no_observation=record is None,
                        body_state_ids=[item.state.state_id for item in result.stages[-1].components],
                        body_state_version=result.stages[-1].components[0].state.version,
                        body_component_id=result.stages[-1].components[0].state.component_id,
                        body_chart_mean=result.stages[-1].components[0].state.mean.tolist(),
                        local_transforms=[dict(edge_id=edge.edge_id, translation=edge.translation.tolist(),
                            rotation_wxyz=edge.rotation_wxyz.tolist(),
                            translation_covariance=edge.translation_covariance.tolist(),
                            bingham=None if edge.bingham is None else edge.bingham.tolist())
                            for edge in result.stages[-1].components[0].state.transforms],
                        body_covariance=result.stages[-1].components[0].state.covariance.tolist(),
                        stiffness=record, simple_inference=result.records.get('simple_inference'),
                        body_surprise=result.records.get('bayes_factor_surprise'),
                        command_path=None if path is None else path.record(),
                        observation_ids=result.records.get('observation_ids', [])))
                    processed.append(dict(event_id=event.event_id,
                        result_metadata=dict(observation_fused=observed,
                            observation_eligible=eligible, observation_selected=event.event_id in selected,
                            reason=('temporal_thinning' if eligible and event.event_id not in selected else
                                'fused' if observed else 'no_valid_observation_or_command_context'),
                            source_stamp=event.stamp, source_command=list(event.command))))
                    previous, previous_index = result, index
                    # This is a read-only handoff of the same complete law,
                    # not selection of a new raw observation or another Bayes update.
                    if event.kind in ('image', 'imu') or len(processed) == len(request['batch'].events):
                        keep = ('simple_inference', 'fixed_goal_memory', 'sensor_event',
                            'bayes_factor_surprise',
                            'last_effective_observation_stamp', 'last_effective_observation_command')
                        records = {key: result.records[key] for key in keep if key in result.records}
                        records['observation_ids'] = list(result.records.get('observation_ids', ()))
                        records['candidate_measure'] = {key: value for key, value in
                            result.records['candidate_measure'].items() if key != 'observation_evidence'}
                        view = replace(result, stages=(result.stages[-1],), records=records,
                            image=np.empty((0,0), dtype=np.uint8), detection_image=np.empty((0,0), dtype=np.uint8),
                            local_estimates=(), sensor_posteriors=(), frozen_body_prior=None,
                            frozen_body_posterior=None, frozen_body_branches=(),
                            projected_body_numerator=None, time_resolved_body_ratios=())
                        replies.put(dict(kind='body', result=view, packet=packet, capture_index=index,
                            goal_observed=inference.goal_observed))
            replies.put(dict(kind='batch_complete', processed=processed, last_source_index=previous_index))
    except Exception:
        replies.put(dict(failed=True, worker='simple_recurrence', traceback=traceback.format_exc()))


class SimpleMotionSession:
    """Finite position execution; no velocity integrator or goal/PDF side state."""
    _acquire_due = AsyncMotionSession._acquire_due
    _accept_camera_preview = AsyncMotionSession._accept_camera_preview

    def __init__(self, observations_path, output, *, imu_target_link=None, duration=None,
                 clock=time.monotonic, context=None, direct_camera_preview=False,
                 body_surprise_enabled=None, **_unused):
        from .simple_inference import SimpleInference
        self.clock, self.output = clock, Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.package = Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
        _, self.templates = load_inputs(self.package/'config/common.yaml', observations_path)
        self.scene = MovingSwitchScene(self.output, imu_target_link=imu_target_link)
        if duration is not None:
            self.scene.duration = float(duration)
        self.inference_description = SimpleInference(self.scene.robot_description, self.templates,
            body_surprise_enabled=body_surprise_enabled)
        self.body_surprise_enabled = self.inference_description.surprise_enabled
        self.settings = self.inference_description.settings
        self.options = self.settings.get('async_execution', {})
        self.period = 1./float(self.options.get('physics_hz', 120.))
        self.rates = {kind: float(self.options.get(kind+'_hz', rate))
            for kind, rate in (('joint', 40.), ('imu', 30.), ('image', 12.))}
        self.control_mode = 'simple_reference'
        self.direct_camera_preview = bool(direct_camera_preview)
        settings = yaml.safe_load((self.package/'config/stiffness_identification.yaml').read_text())
        settings['bayes_factor_surprise'] = dict(enabled=False)
        self.engine = PhysicalActionEngine(self.scene.urdf,
            tuple(self.inference_description.body_ids), settings)
        self.store = SensorEventStore(self.output/'observations.sqlite3')
        self.context = context or mp.get_context('spawn')
        self.acquisition_requests, self.acquisition_replies = self.context.Queue(), self.context.Queue()
        self.requests, self.replies = self.context.Queue(), self.context.Queue()
        self.control_requests, self.control_replies = self.context.Queue(), self.context.Queue()
        board = self.scene.config.get('marker_board', {})
        ids = (tuple(int(i) for row in board['ids'] for i in row) if board.get('enabled', False)
               else (int(self.scene.config['marker']['id']),))
        specification = dict(board=board, camera=self.scene.config['camera'],
            texture=str(self.scene.texture_path), seed=self.scene.config['seed'],
            marker_size=self.scene.config['marker']['size'], marker_ids=ids,
            direct_camera_preview=self.direct_camera_preview)
        from .simple_selection_worker import selection_worker
        self.workers = (
            self.context.Process(target=acquisition_worker,
                args=(self.acquisition_requests, self.acquisition_replies, specification,
                      str(self.output/'observations.sqlite3'), str(self.output)), daemon=True),
            self.context.Process(target=simple_worker,
                args=(self.requests, self.replies, self.scene.robot_description, str(self.scene.urdf),
                      self.templates, str(self.output), self.body_surprise_enabled), daemon=True),
            self.context.Process(target=selection_worker,
                args=(self.control_requests, self.control_replies, self.scene.robot_description,
                      str(self.scene.urdf), self.templates, str(self.output),
                      self.body_surprise_enabled), daemon=True))
        for worker in self.workers:
            worker.start()
        self.command = self.scene.initial_command.copy()
        self.started = self.failure = None
        self.closed = False
        self.actions, self.frames = [], []
        self.current_control = self.current_display = None
        self.raw_publications, self.capture_indices, self.capture_wall_times = [], [], []
        self.acquisition_times = {kind: [] for kind in self.rates}
        self.next_acquisition = {kind: 0. for kind in self.rates}
        self.missed_acquisition_deadlines = {kind: 0 for kind in self.rates}
        self.acquisition_submitted = self.acquisition_completed = 0
        self.preview_capture_index = None
        self.preview_image = np.zeros((self.scene.config['camera']['height'], self.scene.config['camera']['width'], 3), np.uint8)
        self.marker_preview_visible = None
        self.last_marker_missing_capture = -1
        self.inference_busy = self.selection_pending = False
        self.ack_pending = set()
        self.empty_watermark = None
        self.request_counter = 0
        self.last_assimilated_capture = None
        self.processed_events = 0
        self.fused_observations = self.thinned_observations = 0
        self.latest_body = self.latest_packet = None
        self.latest_body_index = None
        self.control_busy = False
        self.control_counter = 0
        self.last_control_body_id = None
        self.plan = None
        self.source_hashes = {str(p.relative_to(self.package)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (self.package/'src').rglob('*.py')}
        self.frame = self.scene.step(0., stamp=1792080000., command=self.command, render_camera=False)
        self.frames.append(self.frame)
        self.scene.trajectory[-1].update(action_mode='waiting_for_goal', command_interval_mode='waiting_for_goal')
        _append(self.output/'physical_trajectory.jsonl', self.scene.trajectory[-1])
        self._acquire_due()

    @property
    def velocity(self):
        # Derived only for the raw command notification; never an independent
        # dynamical state, optimization variable or source of momentum.
        if self.plan is None or self.frame.time >= self.plan['end_time']:
            return np.zeros_like(self.command)
        return (self.plan['target']-self.plan['start'])/self.plan['duration']

    @property
    def motion_finished(self):
        return self.frame.time >= self.scene.duration-1e-6

    @property
    def estimation_finished(self):
        return (self.motion_finished and not self.inference_busy and not self.selection_pending
            and not self.control_busy and not self.ack_pending and self.acquisition_completed == self.acquisition_submitted
            and self.store.stats()['earliest_unfinished'] is None)

    def start(self):
        if self.started is None and self.current_control is not None:
            self.started = self.clock()
            return True
        return False

    def _submit(self):
        if self.inference_busy or self.selection_pending or self.ack_pending:
            return
        if self.last_assimilated_capture is not None and self.started is None:
            return
        watermark = self.acquisition_submitted, self.acquisition_completed
        if self.empty_watermark == watermark:
            return
        self.request_counter += 1
        self.acquisition_requests.put(dict(kind='database_operation', operation='select_batch',
            through_stamp=self.frame.stamp, snapshot_id='simple:'+str(self.request_counter),
            acquisition_watermark=watermark, initial_command_only=self.last_assimilated_capture is None))
        self.selection_pending = True

    def poll(self):
        while True:
            try:
                message = self.acquisition_replies.get_nowait()
            except Empty:
                break
            if message.get('failed'):
                raise RuntimeError(message['traceback'])
            kind = message['kind']
            if kind == 'preview_ready':
                self._accept_camera_preview(message)
            elif kind == 'database_batch_selection':
                self.selection_pending = False
                batch = message['batch']
                if batch:
                    start = (int(batch.events[0].payload_metadata['frame_index'])
                        if self.last_assimilated_capture is None else self.last_assimilated_capture)
                    from .observation_selection import select_observations
                    selection = select_observations(batch.events, {item['sensor'] for item in self.templates},
                        self.options.get('observation_selection', {}))
                    _append(self.output/'observation_selection.jsonl', dict(batch_id=batch.batch_id, **selection))
                    self.requests.put(dict(batch=batch, selection=selection, database=str(self.output/'observations.sqlite3'),
                        request_id=self.request_counter, source_index=self.last_assimilated_capture,
                        steps=tuple(self.actions[start:]), index=len(self.frames)-1, stamp=self.frame.stamp))
                    self.inference_busy = True
                else:
                    self.empty_watermark = message['acquisition_watermark']
            elif kind == 'database_completed':
                self.ack_pending.remove(message['event_id'])
            elif kind in ('raw_ready', 'image_ready'):
                self.acquisition_completed += 1
                if kind == 'image_ready':
                    self._accept_camera_preview(message)
                    self.raw_publications.append(dict(kind='image', stamp=message['stamp'],
                        frame_index=message['frame_index'], image=message['image']))
            else:
                raise ValueError('Unexpected simple acquisition reply: '+str(kind))
        while True:
            try:
                message = self.replies.get_nowait()
            except Empty:
                break
            if message.get('failed'):
                raise RuntimeError(message['traceback'])
            if message['kind'] == 'body':
                self.latest_body, self.latest_packet = message['result'], message['packet']
                self.latest_body_index = message['capture_index']
            elif message['kind'] == 'batch_complete':
                self.inference_busy = False
                self.last_assimilated_capture = message['last_source_index']
                for item in message['processed']:
                    self.acquisition_requests.put(dict(kind='database_operation', operation='complete', **item))
                    self.ack_pending.add(item['event_id'])
                self.processed_events += len(message['processed'])
                self.fused_observations += sum(item['result_metadata']['observation_fused'] for item in message['processed'])
                self.thinned_observations += sum(item['result_metadata'].get('reason') == 'temporal_thinning' for item in message['processed'])
            else:
                raise ValueError('Unexpected simple recurrence reply.')
        while True:
            try:
                message = self.control_replies.get_nowait()
            except Empty:
                break
            if message.get('failed'):
                raise RuntimeError(message['traceback'])
            if message['kind'] == 'plan':
                if message['selection_performed'] and not self.motion_finished:
                    if not np.array_equal(message['source_command'], self.command):
                        raise ValueError('A finite step must be selected at the held current command; its source command changed before adoption.')
                    if message['packet'] is not None:
                        self.engine.set(message['packet'])
                    duration = float(message['duration'])
                    self.plan = dict(start=self.command.copy(), target=np.asarray(message['q_next']).copy(),
                        start_time=self.frame.time, end_time=self.frame.time+duration, duration=duration,
                        request_id=message['request_id'],
                        source={key: message['diagnostics'][key] for key in
                            ('source_stamp', 'source_command', 'source_version', 'body_source_state_ids', 'goal_source_state_ids')},
                        goal_observed=message['diagnostics']['goal_observed'])
                    _append(self.output/'simple_plans.jsonl', dict(
                        request_id=message['request_id'], adoption_stamp=self.frame.stamp,
                        start_command=self.command.tolist(), target_command=message['q_next'].tolist(),
                        duration=duration, diagnostics=message['diagnostics'], work_seconds=message['work_seconds']))
            elif message['kind'] == 'display':
                self.current_display = self.current_control = message
                self.control_busy = False
            else:
                raise ValueError('Unexpected simple selection reply.')
        for worker in self.workers:
            if worker.exitcode is not None and not self.closed:
                raise RuntimeError('Simple worker exited: '+str(worker.exitcode))

    def _submit_control(self):
        if self.control_busy or self.latest_body is None:
            return
        if self.plan is not None and self.frame.time < self.plan['end_time'] and not self.motion_finished:
            return
        identity = self.latest_body.stages[-1].components[0].state.state_id
        if self.motion_finished and identity == self.last_control_body_id:
            return
        self.control_counter += 1
        self.control_requests.put(dict(result=self.latest_body, packet=self.latest_packet,
            steps=tuple(self.actions[self.latest_body_index:]),
            index=len(self.frames)-1, stamp=self.frame.stamp,
            request_id=self.control_counter, capture_index=self.latest_body_index,
            goal_observed=self.latest_body.records['simple_inference']['goal_observed'],
            select=not self.motion_finished))
        self.control_busy = True
        self.last_control_body_id = identity

    def advance(self):
        # Advance old position reference to this receipt boundary, then adopt
        # any new finite plan. No command is applied retroactively.
        if self.started is not None and not self.motion_finished:
            elapsed = min(self.scene.duration, self.clock()-self.started)
            if elapsed-self.frame.time > 1e-6:
                command = self.command.copy()
                if self.plan is not None:
                    fraction = min(1., max(0., (elapsed-self.plan['start_time'])/self.plan['duration']))
                    command = self.plan['start']+fraction*(self.plan['target']-self.plan['start'])
                if self.engine.packet is None:
                    action = dict(joint_increment=(command-self.command).tolist(), selected_K=None,
                        density_id=None, captured_body_posterior_versions=[None],
                        physical_truth_received=False, method='declared rigid increment before K representative')
                else:
                    action = self.engine.increment(self.command, command)
                step = ActionStep(index=len(self.actions), start_stamp=self.frame.stamp,
                    end_stamp=1792080000.+elapsed, start_command=tuple(self.command), end_command=tuple(command),
                    joint_increment=tuple(action['joint_increment']), density_id=action.get('density_id'),
                    selected_K=action.get('selected_K'),
                    estimate_version=action['captured_body_posterior_versions'][0],
                    anchor_command=action.get('numerical_branch_anchor_command'),
                    anchor_equilibrium=action.get('numerical_branch_anchor'),
                    source_state_ids=tuple(action.get('captured_body_posterior_state_ids', ())),
                    source_stamp=action.get('captured_source_stamp'), provenance=action)
                frame = self.scene.step(elapsed, stamp=step.end_stamp, command=command, render_camera=False)
                self.actions.append(step)
                self.command, self.frame = command, frame
                self.frames.append(frame)
                mode = ('reaching' if self.plan is not None and self.plan['goal_observed'] else 'waiting_for_goal')
                self.scene.trajectory[-1].update(action_mode=mode, command_interval_mode=mode,
                    reaching_phase='simple_reference',
                    reaching_phase_source=None if self.plan is None else self.plan['source'])
                _append(self.output/'physical_trajectory.jsonl', self.scene.trajectory[-1])
                _append(self.output/'commands.jsonl', step.record())
                self._acquire_due()
        self.poll()
        self._submit()
        self._submit_control()
        return self.frame

    def manifest(self):
        def hz(values):
            return ((len(values)-1)/(values[-1]-values[0]) if len(values)>1 and values[-1]>values[0] else 0.)
        display = self.current_display
        snapshot = None if display is None else display['snapshot']
        phase = {name: self.scene.trajectory[-1][name] for name in
            ('reaching_phase', 'reaching_phase_source') if name in self.scene.trajectory[-1]}
        return dict(execution_mode='simple', control_mode='finite_expected_position_error',
            physical_stamp=self.frame.stamp, command=self.command.tolist(),
            displayed_prediction_stamp=None if snapshot is None else snapshot.stamp,
            displayed_prediction_command=None if snapshot is None else list(snapshot.command),
            displayed_prediction_age_s=None if snapshot is None else self.frame.stamp-snapshot.stamp,
            observation_age_s=None if display is None else self.frame.stamp-display['diagnostics']['original_observation_stamp'],
            duration=self.scene.duration, simulated_elapsed_s=self.frame.time, finished=self.motion_finished,
            startup_phase='initial_prior' if self.started is None else 'moving_or_draining',
            estimation_drained=self.estimation_finished, body_estimation_drained=self.estimation_finished,
            stiffness_estimation_drained=self.estimation_finished,
            acquisition_requests=self.acquisition_submitted, acquisition_archived=self.acquisition_completed,
            all_captured_sensor_data_archived=self.acquisition_completed==self.acquisition_submitted,
            processed_events=self.processed_events, fused_observations=self.fused_observations,
            thinned_observations=self.thinned_observations,
            observation_selection=self.options.get('observation_selection'), actual_motion_hz=(len(self.actions)/self.frame.time if self.frame.time else 0.),
            actual_camera_hz=hz(self.acquisition_times['image']), actual_joint_hz=hz(self.acquisition_times['joint']),
            actual_imu_hz=hz(self.acquisition_times['imu']),
            camera_preview_publication_route='acquisition_worker_direct_ros' if self.direct_camera_preview else 'publication_worker',
            body_bfs_enabled=self.body_surprise_enabled,
            body_bayes_factor_surprise=dict(self.inference_description.surprise_settings),
            K_bfs_enabled=False, K_evidence_enabled=False,
            cumulative_K_factor_reevaluation=False, independent_fresh_goal_enabled=False,
            running_gradient_enabled=False, view_factor_enabled=False,
            physical_truth_used_for_inference=False, monte_carlo_used=False,
            collision_checking_enabled=self.scene.config['plant']['collision_checking'],
            active_command_proposal=None if self.engine.packet is None else self.engine.packet.proposal,
            active_plan_request_id=None if self.plan is None else self.plan['request_id'],
            active_plan_source=None if self.plan is None else self.plan['source'],
            displayed_request_id=None if self.current_display is None else self.current_display['request_id'],
            displayed_source=None if self.current_display is None else self.current_display['diagnostics'],
            source_sha256=self.source_hashes, failure=self.failure, **phase)

    def close(self):
        if self.closed:
            return
        self.closed = True
        for requests in (self.requests, self.control_requests, self.acquisition_requests):
            requests.put(None)
        for worker in self.workers:
            worker.join(timeout=3.)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1.)
        _append(self.output/'shutdown.jsonl', dict(motion_finished=self.motion_finished,
            estimation_drained=self.estimation_finished, processed_events=self.processed_events,
            acquisition_submitted=self.acquisition_submitted, acquisition_archived=self.acquisition_completed))
        self.store.close()
        for queue in (self.requests, self.replies, self.control_requests, self.control_replies, self.acquisition_requests, self.acquisition_replies):
            queue.cancel_join_thread()
            queue.close()
