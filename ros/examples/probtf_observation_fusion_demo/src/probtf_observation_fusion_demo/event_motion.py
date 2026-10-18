"""Motion and acquisition continue while every timestamped event is processed.

The observation database is the source of work. No latest-only selection is
performed. Raw acquisition/rendering, sequential inference and IK own separate
processes. All physical command intervals retain their actual captured K_est.
"""
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import time
from types import SimpleNamespace

import numpy as np
import rospkg
import yaml

from .async_state import ActionStep
from .async_stiffness import PhysicalActionEngine
from .async_workers import (_append, _freeze_arrays, acquisition_worker,
                            control_worker, event_inference_worker)
from .marker_search import MarkerSearchPolicy
from .model import load_inputs
from .moving_control import MovingController
from .moving_inference import MovingInference
from .moving_scene import MovingSwitchScene
from .sensor_event_store import SensorEventStore
from .short_command import command_reference
from .running_gradient import GradientPacket, RunningGradientTracker


def _gradient_identity(value):
    """Hash non-finite log mass exactly without modifying the candidate law.

    A zero-mass mixture may legitimately have log_weight=-infinity. This
    tagged representation is used only for content identity; packet gradient
    finiteness is independently checked before adoption.
    """
    if isinstance(value,float) and not np.isfinite(value):
        return {'nonfinite_float_identity':repr(value)}
    if isinstance(value,dict):
        return {key:_gradient_identity(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):
        return [_gradient_identity(item) for item in value]
    return value


class AsyncMotionSession:
    """One physical experiment with a durable, acquisition-ordered event log."""
    def __init__(self, observations_path, output, *, imu_target_link=None,
                 duration=None, clock=time.monotonic, context=None, control_mode=None,
                 running_gradient_options=None, visibility_factor_options=None):
        self.clock, self.output = clock, Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.package = Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
        _, self.templates = load_inputs(self.package/'config/common.yaml', observations_path)
        self.scene = MovingSwitchScene(self.output, imu_target_link=imu_target_link)
        if duration is not None:
            self.scene.duration = float(duration)
        self.inference_description = MovingInference(self.scene.robot_description, self.templates)
        self.settings = self.inference_description.settings
        self.options = dict(self.settings.get('async_execution', {}))
        self.period = 1./float(self.options.get('physics_hz', 120.))
        self.rates = {kind: float(self.options.get(kind+'_hz', rate))
                      for kind, rate in (('joint',40.),('imu',30.),('image',12.))}
        if any(not np.isfinite(v) or v <= 0 for v in (*self.rates.values(),1/self.period)):
            raise ValueError('Acquisition and physics rates must be finite and positive.')
        self.camera_period = 1./self.rates['image']
        self.horizon = float(self.options.get('ik_horizon_s', .5))
        self.probability_unit = self.settings.get('bayes_factor_surprise', {}).get('event_probability_unit')
        control = self.settings['control']
        self.control_mode = control_mode or control.get('mode', 'single_step_ik')
        self.visibility_options=dict(control.get('visibility_factor',{}))
        if visibility_factor_options:
            self.visibility_options.update(visibility_factor_options)
        if self.control_mode not in ('running_gradient', 'single_step_ik'):
            raise ValueError('Unknown asynchronous control mode: '+str(self.control_mode))
        self.servo = MovingController(self.inference_description, self.scene.robot_description,
            control['goal_offset'], control['goal_orientation_wxyz'], control)
        parameter = yaml.safe_load((self.package/'config/stiffness_identification.yaml').read_text())
        self.engine = PhysicalActionEngine(self.scene.urdf,
            tuple(self.inference_description.body_ids), parameter)
        self.enabled_sensors = {item['sensor'] for item in self.templates}
        self.behavior = MarkerSearchPolicy(control.get('marker_search', {}),
            marker_enabled='marker' in self.enabled_sensors,
            exposure_time=self.scene.config['camera']['exposure_time'])
        self.store = SensorEventStore(self.output/'observations.sqlite3')
        self.context = context or mp.get_context('spawn')
        self.estimate_requests, self.estimate_replies = self.context.Queue(), self.context.Queue()
        self.control_requests, self.control_replies = self.context.Queue(), self.context.Queue()
        self.acquisition_requests, self.acquisition_replies = self.context.Queue(), self.context.Queue()
        common = (self.scene.robot_description, str(self.scene.urdf), self.templates, str(self.output))
        board = self.scene.config.get('marker_board', {})
        ids = (tuple(int(i) for row in board['ids'] for i in row)
               if board.get('enabled', False) else (int(self.scene.config['marker']['id']),))
        specification = dict(board=board, camera=self.scene.config['camera'],
            texture=str(self.scene.texture_path), seed=self.scene.config['seed'],
            marker_size=self.scene.config['marker']['size'], marker_ids=ids)
        self.workers = (
            self.context.Process(target=event_inference_worker,
                args=(self.estimate_requests,self.estimate_replies)+common+(self.probability_unit,),daemon=True),
            self.context.Process(target=control_worker,
                args=(self.control_requests,self.control_replies)+common+(
                    self.horizon,self.control_mode,self.visibility_options),daemon=True),
            self.context.Process(target=acquisition_worker,
                args=(self.acquisition_requests,self.acquisition_replies,specification,
                      str(self.output/'observations.sqlite3'),str(self.output)),daemon=True))
        for worker in self.workers:
            worker.start()
        self.command = self.scene.initial_command.copy()
        self.velocity = np.zeros_like(self.command)
        gradient_options = dict(control.get('running_gradient', {}))
        if running_gradient_options:
            gradient_options.update(running_gradient_options)
        self.gradient_tracker = (RunningGradientTracker(self.command,
            eta=gradient_options['eta'], gamma=gradient_options['gamma'],
            joint_lower=self.servo.lower, joint_upper=self.servo.upper,
            initial_stamp=1792080000.,
            max_source_age=gradient_options.get('max_source_age_s'),
            max_command_distance=gradient_options.get('max_command_distance_rad'),
            max_snapshot_update_lag=gradient_options.get('max_snapshot_update_lag'))
            if self.control_mode=='running_gradient' else None)
        self.last_gradient_adoption = self.last_gradient_tick = None
        self.gradient_evaluations_completed = 0
        self.desired_velocity = self.velocity.copy()
        self.actions, self.frames, self.capture_indices, self.assimilated = [], [], [], []
        self.latest_result = self.latest_packet = self.current_control = None
        self.latest_marker_observed = False
        self.inference_busy = self.control_busy = False
        self.control_request_counter = 0
        self.control_ledger_transmitted = 0
        self.pending_control_request = None
        self.pending_inference_event = None
        self.database_selection_pending = False
        self.database_ack_pending = None
        self.selected_sensor_event = None
        self.database_empty_acquisition_watermark = -1
        self.last_submitted_capture = self.last_assimilated_capture = None
        self.last_assimilated_event = None
        self.next_acquisition = {kind: 0. for kind in self.rates}
        self.acquisition_times = {kind: [] for kind in self.rates}
        self.missed_acquisition_deadlines = {kind: 0 for kind in self.rates}
        self.acquisition_submitted = self.acquisition_completed = 0
        self.raw_publications = []
        self.started = None
        self.last_action_mode = 'initial_hold'
        self.last_marker_missing_capture = -1
        self.marker_preview_visible = None
        self.preview_capture_index = None
        self.preview_image = np.zeros((self.scene.config['camera']['height'],
                                       self.scene.config['camera']['width'],3),np.uint8)
        self.failure = None
        self.closed = False
        self.last_tick_timings = {}
        self.publish_times, self.capture_wall_times, self.control_times, self.estimate_times = [], [], [], []
        self.source_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}
        self.source_integrity_checked_at = -np.inf
        self.source_integrity_matches = True
        self.frame = self.scene.step(0.,stamp=1792080000.,command=self.command,render_camera=False)
        self.frames.append(self.frame)
        self.scene.trajectory[-1].update(action_mode='reaching',command_interval_mode='initial_hold',
            reaching_phase='waiting_for_control',reaching_phase_source=None)
        _append(self.output/'physical_trajectory.jsonl',self.scene.trajectory[-1])
        self._acquire_due()

    @property
    def motion_finished(self):
        # Matches the minimum physical advance; do not leave a sub-microsecond
        # floating-point remainder in a permanently unfinished drain state.
        return self.frame.time >= self.scene.duration-1e-6

    @property
    def estimation_finished(self):
        stats = self.store.stats()
        return (self.motion_finished and not self.inference_busy
            and not self.database_selection_pending and self.database_ack_pending is None
            and self.selected_sensor_event is None
            and self.acquisition_completed == self.acquisition_submitted
            and stats['earliest_unfinished'] is None)

    def start(self):
        if self.started is None and self.latest_result is not None:
            self.started = self.clock()
            for values in (self.publish_times,self.capture_wall_times,*self.acquisition_times.values()):
                values.clear()
            return True
        return False

    def _acquire_due(self):
        """Sample each independent source at its first available physics tick.

        Missed nominal deadlines are measured, not fabricated as observations.
        Every actually acquired observation is queued and durably archived.
        """
        index = len(self.frames)-1
        for kind in ('joint','imu','image'):
            if self.frame.time+1e-9 < self.next_acquisition[kind]:
                continue
            deadlines = 0
            while self.next_acquisition[kind] <= self.frame.time+1e-9:
                self.next_acquisition[kind] += 1./self.rates[kind]
                deadlines += 1
            self.missed_acquisition_deadlines[kind] += max(0,deadlines-1)
            event_id = '{}:{:08d}'.format(kind,index)
            request = dict(kind=kind,event_id=event_id,frame_index=index,
                stamp=self.frame.stamp,time=self.frame.time,command=self.command.tolist(),
                received_wall_stamp=time.time())
            if kind == 'joint':
                request['payload'] = dict(command_velocity=self.velocity.tolist())
            elif kind == 'imu':
                request['payload'] = dict(gravity_in_sensor=self.frame.gravity_in_sensor,
                    imu_target_link=self.scene.imu_target_link,imu_sigma_rad=self.scene.imu_sigma_rad)
            else:
                request.update(camera_pose=self.frame.camera_pose,
                    marker_pose=self.scene.marker,camera_history=tuple(self.scene.camera_history))
                self.capture_indices.append(index)
                self.capture_wall_times.append(self.clock())
                _append(self.output/'captures.jsonl',dict(index=index,event_id=event_id,
                    stamp=self.frame.stamp,time=self.frame.time,command=self.command.tolist(),
                    image='capture_{:06d}.png'.format(index),
                    observation_assimilated_at_capture=False))
            # Physical-side recovery spool, separate from observation payloads.
            def serial(value):
                if isinstance(value,np.ndarray): return value.tolist()
                if isinstance(value,dict): return {k:serial(v) for k,v in value.items()}
                if isinstance(value,(tuple,list)): return [serial(v) for v in value]
                return value
            _append(self.output/'acquisition_requests.jsonl',serial(request))
            self.acquisition_requests.put(request)
            self.acquisition_submitted += 1
            self.acquisition_times[kind].append(self.clock())
            if kind != 'image':
                self.raw_publications.append(request)

    def collect_archives(self):
        while True:
            try: message = self.acquisition_replies.get_nowait()
            except Empty: break
            if message.get('failed'):
                self.failure = message
                raise RuntimeError('Acquisition failed: '+message['traceback'])
            if message['kind']=='database_selection':
                if not self.database_selection_pending:
                    raise ValueError('Unexpected database selection reply.')
                self.database_selection_pending=False
                self.selected_sensor_event=message['event']
                if message['event'] is None:
                    self.database_empty_acquisition_watermark=message['acquisition_watermark']
                continue
            if message['kind']=='database_completed':
                if self.database_ack_pending != message['event_id']:
                    raise ValueError('Database acknowledgement belongs to another event.')
                self.database_ack_pending=None
                continue
            if message['kind']=='database_failed':
                continue
            self.acquisition_completed += 1
            if message['kind'] != 'image_ready':
                continue
            preview = message['preview']
            self.preview_image = preview['image']
            self.preview_capture_index = message['frame_index']
            self.marker_preview_visible = preview['visible']
            if not preview['visible']:
                self.last_marker_missing_capture = message['frame_index']
            self.raw_publications.append(dict(kind='image',stamp=message['stamp'],
                frame_index=message['frame_index'],image=message['image']))
            _append(self.output/'camera_previews.jsonl',{k:preview[k] for k in
                ('index','stamp','visible','visible_marker_ids')})

    def _steps_since(self,index,target_index=None):
        target_index = len(self.frames)-1 if target_index is None else target_index
        return tuple(self.actions[index:target_index])

    def submit_estimate(self):
        if self.inference_busy or self.database_ack_pending is not None:
            return
        event=self.selected_sensor_event
        if event is None:
            if (not self.database_selection_pending and
                    self.database_empty_acquisition_watermark != self.acquisition_submitted):
                self.acquisition_requests.put(dict(kind='database_operation',operation='select',
                    through_stamp=self.frame.stamp,initialize=self.latest_result is None,
                    acquisition_watermark=self.acquisition_submitted,
                    enabled_sensors=tuple(self.enabled_sensors)))
                self.database_selection_pending=True
            return
        self.selected_sensor_event=None
        index=int(event.payload['frame_index'])
        steps=(() if self.last_assimilated_capture is None else
               self._steps_since(self.last_assimilated_capture,index))
        _append(self.output/'assimilation_selection.jsonl',dict(
            event_id=event.event_id,event_sequence=event.sequence,sensor_kind=event.kind,
            selected_capture_index=index,selected_stamp=event.stamp,
            command_interval_count=len(steps),posterior_reset=False,
            archived_not_assimilated_capture_indices=[],
            scheduling='oldest unprocessed acquisition timestamp; no latest-only selection'))
        self.estimate_requests.put((event,index,steps))
        self.pending_inference_event=event.event_id
        self.last_submitted_capture,self.inference_busy=index,True

    def submit_control(self):
        if self.control_busy or self.latest_result is None:
            return
        index = len(self.frames)-1
        if self.current_control is not None and index == self.current_control['index'] and (
                self.current_control.get('event_id') == self.last_assimilated_event):
            return
        self.control_request_counter += 1
        self.pending_control_request = self.control_request_counter
        self.control_requests.put(dict(result=self.latest_result,packet=self.latest_packet,
            request_id=self.pending_control_request,
            action_packet=None if self.current_control is None else self.current_control['packet'],
            event_id=self.last_assimilated_event,capture_index=self.last_assimilated_capture,
            index=index,stamp=self.frame.stamp,
            ledger_delta=tuple(self.actions[self.control_ledger_transmitted:index]),
            ledger_source_index=self.last_assimilated_capture,ledger_target_index=index,
            velocity=self.velocity.copy(),snapshot_update_count=len(self.assimilated)))
        self.control_ledger_transmitted=index
        self.control_busy = True

    def poll(self):
        changed=False
        self.collect_archives()
        for kind,queue in (('inference',self.estimate_replies),('control',self.control_replies)):
            try: message=queue.get_nowait()
            except Empty: continue
            if message.get('failed'):
                self.failure=message
                _append(self.output/'execution_failure.jsonl',message)
                if kind=='inference' and self.pending_inference_event is not None:
                    self.acquisition_requests.put(dict(kind='database_operation',operation='fail',
                        event_id=self.pending_inference_event,error=message['traceback']))
                raise RuntimeError(kind+' worker failed: '+message['traceback'])
            if kind == 'inference':
                if not self.inference_busy or message['event_id'] != self.pending_inference_event:
                    raise ValueError('Inference reply does not match the claimed acquisition event.')
                self.inference_busy=False
                self.pending_inference_event=None
                self.latest_result=_freeze_arrays(message['result'])
                self.latest_packet=message['packet']
                if message['sensor_kind']=='image':
                    self.latest_marker_observed=message['marker_observed']
                self.last_assimilated_capture=message['capture_index']
                self.last_assimilated_event=message['event_id']
                self.assimilated.append(message['event_id'])
                self.database_ack_pending=message['event_id']
                self.acquisition_requests.put(dict(kind='database_operation',operation='complete',
                    event_id=message['event_id'],result_metadata=dict(
                    observation_fused=bool(self.latest_result.records.get('observation_ids')),
                    body_source_state_id=self.latest_result.prior.state_id,
                    inference_seconds=message['work_seconds'])))
                self.estimate_times.append(self.clock())
            else:
                if (not self.control_busy
                        or message.get('request_id') != self.pending_control_request):
                    raise ValueError('IK reply does not match the outstanding frozen-law request.')
                self.control_busy=False
                self.pending_control_request=None
                adopt = True
                if self.control_mode=='running_gradient':
                    self.gradient_evaluations_completed += 1
                    evaluation=message['gradient_evaluation']
                    fields=GradientPacket.__dataclass_fields__
                    packet_fields={name:evaluation[name] for name in fields if name in evaluation}
                    packet_fields['packet_id']=hashlib.sha256(json.dumps(_gradient_identity(evaluation),
                        sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                    gradient_packet=GradientPacket(**packet_fields)
                    before_command=self.gradient_tracker.position.tolist()
                    before_velocity=self.gradient_tracker.velocity.tolist()
                    adoption=self.gradient_tracker.accept(gradient_packet,
                        adoption_stamp=self.frame.stamp,snapshot_update_count=len(self.assimilated))
                    adoption.update(request_id=message.get('request_id'),
                        state_before=dict(command=before_command,velocity=before_velocity),
                        state_after=dict(command=self.gradient_tracker.position.tolist(),
                                         velocity=self.gradient_tracker.velocity.tolist()),
                        evaluation_status=evaluation['status'],evaluation_reason=evaluation.get('reason'),
                        gradient_packet=gradient_packet.as_dict(),
                        source_action_packet_id=(None if message.get('packet') is None else message['packet'].packet_id))
                    self.last_gradient_adoption=adoption
                    _append(self.output/'gradient_adoptions.jsonl',adoption)
                    adopt=adoption['accepted']
                    # Before the first valid input, publish the waiting-law
                    # snapshot with an explicit hold status, never as a gradient.
                    if (not adopt and self.current_control is None and 'snapshot' in message
                            and evaluation['status']=='waiting_for_goal'):
                        self.current_control=_freeze_arrays(message)
                if adopt:
                    self.current_control=_freeze_arrays(message)
                    if self.control_mode!='running_gradient':
                        self.desired_velocity=np.array(message['velocity'],copy=True)
                    if message['packet'] is not None:
                        self.engine.set(message['packet'])
                    else:
                        # A captured unselected-K action remains explicit.
                        self.engine.packet=None
                self.control_times.append(self.clock())
            changed=True
        for kind,worker in zip(('inference','control','acquisition'),self.workers):
            if not worker.is_alive() and worker.exitcode is not None and self.failure is None:
                raise RuntimeError(kind+' worker exited unexpectedly: '+str(worker.exitcode))
        return changed

    def _reaching_plan_velocity(self,dt,stamp):
        if self.current_control is None:
            return self.desired_velocity.copy(),dict(method='initial hold; no completed IK plan')
        reference=command_reference(self.current_control['diagnostics'],stamp)
        requested=(np.asarray(reference['reference_command'])-self.command)/dt
        reference.update(method='single selected IK position reference; ordinary actuator bounds apply',
            source_capture_index=self.current_control['capture_index'],
            action_packet_id=None if self.current_control['packet'] is None else self.current_control['packet'].packet_id,
            interval_end_stamp=stamp,requested_velocity=requested.tolist(),
            actual_command_before=self.command.tolist(),new_IK_or_distribution_evaluation=False,
            physical_truth_received=False)
        return requested,reference

    def _running_gradient_motion(self, dt, elapsed):
        """Only integrate the held gradient; no FK/J/endpoint or servo here."""
        tracker=self.gradient_tracker
        if not np.array_equal(self.command,tracker.position):
            raise ValueError('Running-gradient state and actually sent command disagree.')
        before_velocity=tracker.velocity.tolist()
        step=tracker.advance(dt,stamp=1792080000.+elapsed,
            snapshot_update_count=len(self.assimilated))
        record=dict(step.record)
        # Full immutable packets are saved once per adoption; intervals keep
        # their ID and evaluation provenance without repeatedly copying them.
        packet=record.pop('gradient_packet',None)
        record.update(gradient_packet_id=None if packet is None else packet['packet_id'],
            position_before=self.command.tolist(),velocity_before=before_velocity,
            eta=tracker.eta,gamma=tracker.gamma,
            source_command=None if packet is None else packet['source_command'],
            source_stamp=None if packet is None else packet['source_stamp'],
            parameter_density_id=None if packet is None else packet['parameter_density_id'],
            selected_K=None if packet is None else packet['selected_K'],
            method='exact held-gradient damped second-order command trajectory',
            control_mode='running_gradient',
            velocity_clipping_enabled=False,acceleration_clipping_enabled=False,
            endpoint_selection_performed=False,physical_truth_received=False)
        self.last_gradient_tick=record
        if step.record['halted']:
            _append(self.output/'running_gradient_fail_safe.jsonl',record)
            raise RuntimeError('Running-gradient explicit fail-safe: '+str(record.get('reason')))
        self.velocity=np.asarray(step.velocity)
        self.desired_velocity=self.velocity.copy()
        mode='waiting_for_goal' if tracker.last_packet is None else 'reaching'
        goal_acquired=bool(self.latest_result is not None and
            self.latest_result.records.get('fixed_goal_memory',{}).get('available',False))
        behavior=dict(mode=mode,event=('waiting_for_valid_goal_and_gradient' if mode=='waiting_for_goal'
            else 'tracking_retained_goal'),marker_detected=self.marker_preview_visible,
            marker_observed=self.latest_marker_observed,
            goal_available=bool(self.current_control is not None and
                (self.current_control['snapshot'].goal_memory or {}).get('available',False)),
            marker_acquisition_state=('never_acquired' if not goal_acquired else
                'acquired_visible' if self.marker_preview_visible else 'acquired_missing'),
            observation_loss_policy='continue against retained valid goal; no fresh-since-stop gate',
            amplitude_rad=0.,completed_roundtrips=0,physical_truth_received=False,
            reaching_criterion_used_for_selection=mode=='reaching')
        return np.asarray(step.command),SimpleNamespace(mode=mode,record=behavior),record

    def advance(self):
        started=self.clock()
        timings={}
        self.poll()
        if self.started is not None and not self.motion_finished:
            elapsed=min(self.scene.duration,self.clock()-self.started)
            dt=elapsed-self.frame.time
            if dt>1e-6:
                generation_started=time.perf_counter()
                if self.control_mode=='running_gradient':
                    command,plan,reaching_execution=self._running_gradient_motion(dt,elapsed)
                else:
                    self.servo.velocity=self.velocity.copy()
                    active_goal={} if self.current_control is None else self.current_control['snapshot'].goal_memory or {}
                    loss_stamp=(-np.inf if self.last_marker_missing_capture<0 else
                                self.frames[self.last_marker_missing_capture].stamp)
                    visible=(self.marker_preview_visible and self.latest_marker_observed
                        and self.current_control is not None
                        and self.current_control['capture_index']>self.last_marker_missing_capture
                        and active_goal.get('available',False) and active_goal['observation_stamp']>loss_stamp
                        and self.last_assimilated_capture>self.last_marker_missing_capture)
                    observation=SimpleNamespace(posterior=True,detector_result=True)
                    inputs=SimpleNamespace(prior=SimpleNamespace(command=tuple(self.command),
                        stamp=self.frame.stamp,version=3*(len(self.frames)-1)+3),
                        local_estimates=(observation,) if visible else (),records={'fixed_goal_memory':active_goal})
                    plan=self.behavior.plan(inputs,self.velocity,elapsed,joint_lower=self.servo.lower,
                        joint_upper=self.servo.upper,marker_detected=bool(self.marker_preview_visible))
                    plan.record.update(marker_detection_capture_index=self.preview_capture_index,
                        marker_detection_stamp=None if self.preview_capture_index is None else
                            self.frames[self.preview_capture_index].stamp)
                    reaching_execution=None
                    if plan.mode=='reaching':
                        lower,upper,_,_=self.servo._command_bounds(self.command,dt)
                        requested,reaching_execution=self._reaching_plan_velocity(dt,1792080000.+elapsed)
                    else:
                        lower,upper,_,_=self.servo._command_bounds(self.command,dt,
                            position_lower=plan.position_lower,position_upper=plan.position_upper,max_speed=plan.max_speed)
                        target=self.command if plan.target_command is None else np.asarray(plan.target_command)
                        distance=np.abs(target-self.command)
                        change=self.servo.acceleration*dt
                        stop=2*self.servo.acceleration*distance/(np.sqrt(change**2+2*self.servo.acceleration*distance)+change)
                        speed=self.servo.speed if plan.max_speed is None else min(self.servo.speed,plan.max_speed)
                        requested=np.sign(target-self.command)*np.minimum(stop,speed)
                    delta=np.clip(dt*requested,lower,upper)
                    command=self.command+delta
                    self.velocity=delta/dt
                timings['command_generation_s']=time.perf_counter()-generation_started
                delta=command-self.command
                action_started=time.perf_counter()
                action=(dict(joint_increment=delta.tolist(),density_id=None,selected_K=None,
                    captured_body_posterior_versions=[None],physical_truth_received=False,
                    method='direct supplied command action; no selected K representative')
                    if self.engine.packet is None else self.engine.increment(self.command,command))
                timings['frozen_K_ledger_action_s']=time.perf_counter()-action_started
                step=ActionStep(index=len(self.actions),start_stamp=self.frame.stamp,
                    end_stamp=1792080000.+elapsed,start_command=tuple(self.command),end_command=tuple(command),
                    joint_increment=tuple(action['joint_increment']),density_id=action.get('density_id'),
                    estimate_version=action['captured_body_posterior_versions'][0],selected_K=action.get('selected_K'),
                    anchor_command=action.get('numerical_branch_anchor_command'),
                    anchor_equilibrium=action.get('numerical_branch_anchor'),
                    source_state_ids=tuple(action.get('captured_body_posterior_state_ids',())),
                    source_stamp=action.get('captured_source_stamp'),
                    source_version=action['captured_body_posterior_versions'][0],provenance=action)
                physics_started=time.perf_counter()
                advanced_frame=self.scene.step(elapsed,stamp=step.end_stamp,command=command,render_camera=False)
                timings['native_physics_and_endpoint_FK_s']=time.perf_counter()-physics_started
                # A rejected native advance did not execute this interval.
                # Commit command/ledger only after physical integration succeeds.
                self.actions.append(step)
                self.command=command
                self.frame=advanced_frame
                self.frames.append(self.frame)
                phase,source=self.reaching_phase()
                self.scene.trajectory[-1].update(action_mode=plan.mode,command_interval_mode=plan.mode,
                    reaching_phase=phase,reaching_phase_source=source,
                    search_amplitude_rad=plan.record['amplitude_rad'],
                    search_completed_roundtrips=plan.record['completed_roundtrips'])
                _append(self.output/'physical_trajectory.jsonl',self.scene.trajectory[-1])
                _append(self.output/'commands.jsonl',dict(step=step.record(),velocity=self.velocity.tolist(),
                    desired_velocity=self.desired_velocity.tolist(),behavior=plan.record,
                    reaching_execution=reaching_execution,reaching_phase=phase,reaching_phase_source=source,
                    coordinator_timings_s=timings,
                    wall_elapsed_s=self.clock()-self.started))
                self.last_action_mode=plan.mode
                self.publish_times.append(self.clock())
                self._acquire_due()
        self.submit_estimate()
        self.submit_control()
        self.last_tick_timings=dict(timings,coordinator_s=self.clock()-started)
        return self.frame

    def reaching_phase(self):
        if self.current_control is None:
            return 'waiting_for_control',None
        diagnostics=self.current_control['diagnostics']
        if self.control_mode=='running_gradient' and self.gradient_tracker.last_packet is None:
            return 'waiting_for_goal',None
        return diagnostics.get('reaching_phase','indet_ik'),{key:diagnostics[key] for key in
            ('source_stamp','source_command','source_version','body_source_state_ids','goal_source_state_ids')}

    def manifest(self):
        def hz(values):
            values=values if self.started is None else [v for v in values if v>=self.started]
            duration=values[-1]-values[0] if len(values)>1 else 0.
            return (len(values)-1)/duration if duration>0. else None
        stats=self.store.stats()
        checked_at=time.monotonic()
        if checked_at-self.source_integrity_checked_at >= 1.:
            self.source_integrity_matches = self.source_hashes=={
                p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob('*.py'))}
            self.source_integrity_checked_at=checked_at
        snapshot=None if self.current_control is None else self.current_control['snapshot']
        row=self.scene.trajectory[-1]
        planned_phase,planned_source=self.reaching_phase()
        gradient=None if self.gradient_tracker is None else self.gradient_tracker.last_packet
        return dict(execution_mode='async',execution_architecture='timestamped_sensor_events',
            control_mode=self.control_mode,
            offline_precomputation=False,sample_index=len(self.frames)-1,estimate_index=self.last_assimilated_capture,
            simulated_elapsed_s=self.frame.time,duration=self.scene.duration,finished=self.motion_finished,
            estimation_drained=self.estimation_finished,physical_stamp=self.frame.stamp,command=self.command.tolist(),
            estimate_stamp=None if self.latest_result is None else self.latest_result.prior.stamp,
            displayed_prediction_stamp=None if snapshot is None else snapshot.stamp,
            displayed_prediction_command=None if snapshot is None else list(snapshot.command),
            displayed_prediction_age_s=None if snapshot is None else self.frame.stamp-snapshot.stamp,
            observation_age_s=None if self.latest_result is None else self.frame.stamp-self.latest_result.prior.stamp,
            reaching_phase=row['reaching_phase'],reaching_phase_source=row['reaching_phase_source'],
            planned_reaching_phase=planned_phase,planned_reaching_phase_source=planned_source,
            motion_terminal_tolerance_s=1e-6,
            actual_motion_hz=hz(self.publish_times),actual_camera_hz=hz(self.capture_wall_times),
            actual_control_hz=hz(self.control_times),actual_estimate_hz=hz(self.estimate_times),
            actual_update_hz=hz(self.estimate_times),actual_joint_hz=hz(self.acquisition_times['joint']),
            actual_imu_hz=hz(self.acquisition_times['imu']),target_physics_hz=1/self.period,
            target_joint_hz=self.rates['joint'],target_imu_hz=self.rates['imu'],target_image_hz=self.rates['image'],
            physical_model='inertial_spring_damper',physical_integrator='implicit_midpoint',
            physical_model_source=self.scene.physics_source,
            collision_checking_enabled=self.scene.collision_checking,
            reach_condition=self.frame.physics_diagnostics['reach_condition'],
            inference_command_model='quasistatic_equilibrium_increment',
            missed_acquisition_deadlines=dict(self.missed_acquisition_deadlines),sensor_event_database=stats,
            acquisition_requests=self.acquisition_submitted,acquisition_archived=self.acquisition_completed,
            all_captured_sensor_data_archived=self.acquisition_completed==self.acquisition_submitted,
            all_enabled_observations_processed=self.estimation_finished,
            observation_scheduling='all events in acquisition order; no latest-only selection',
            acquired_captures=len(self.capture_indices),assimilated_event_ids=list(self.assimilated),
            inference_busy=self.inference_busy,control_busy=self.control_busy,
            database_selection_pending=self.database_selection_pending,
            database_ack_pending=self.database_ack_pending,
            active_parameter_packet_id=None if self.engine.packet is None else self.engine.packet.packet_id,
            latest_estimated_parameter_packet_id=None if self.latest_packet is None else self.latest_packet.packet_id,
            latest_estimated_K=None if self.latest_packet is None else self.latest_packet.proposal['selected_K'],
            active_ik_K_est=None if self.engine.packet is None else self.engine.stiffness.tolist(),
            active_gradient_K_est=(None if gradient is None or gradient.selected_K is None
                                  else list(gradient.selected_K)),
            active_gradient_packet_id=None if gradient is None else gradient.packet_id,
            gradient_source_stamp=None if gradient is None else gradient.source_stamp,
            gradient_source_command=None if gradient is None else list(gradient.source_command),
            gradient_source_age_s=None if gradient is None else self.frame.stamp-gradient.source_stamp,
            gradient_observation_source_stamps=None if gradient is None else list(gradient.observation_source_stamps),
            gradient_source_command_distance_rad=None if gradient is None else
                float(np.linalg.norm(self.command-np.asarray(gradient.source_command))),
            gradient_evaluations_completed=self.gradient_evaluations_completed,
            gradient_input_policy='sample_and_hold_last_accepted; age, distance and lag diagnostics never stop finite updates',
            running_gradient_gains=None if self.gradient_tracker is None else dict(
                eta=self.gradient_tracker.eta,gamma=self.gradient_tracker.gamma),
            running_gradient_stale_thresholds=None if self.gradient_tracker is None else dict(
                max_source_age_s=self.gradient_tracker.max_source_age,
                max_command_distance_rad=self.gradient_tracker.max_command_distance,
                max_snapshot_update_lag=self.gradient_tracker.max_snapshot_update_lag),
            visibility_factor_config=dict(self.visibility_options),
            active_selection_factors=None if gradient is None else gradient.as_dict()['selection_factors'],
            velocity_clipping_enabled=self.control_mode!='running_gradient',
            acceleration_clipping_enabled=self.control_mode!='running_gradient',
            endpoint_selection_enabled=self.control_mode!='running_gradient',
            last_gradient_adoption=self.last_gradient_adoption,
            last_gradient_tick=self.last_gradient_tick,
            K_exec_enabled=False,position_s_constraint_enabled=False,SG_interpolation_used=False,
            candidate_measure=None if self.latest_result is None else self.latest_result.records.get('candidate_measure'),
            marker_search_mode=self.last_action_mode,simulated_step_s=self.period,
            coordinator_last_timings_s=dict(self.last_tick_timings),implementation_sha256=self.source_hashes,
            implementation_unchanged_during_execution=self.source_integrity_matches,
            source_integrity_check_monotonic_s=self.source_integrity_checked_at,
            physics_and_camera_do_not_wait_for_inference=True,posterior_retained=True,
            monte_carlo_used=False,body_reconstructed_from_stiffness=False,
            ik_plan_horizon_s=None if self.control_mode=='running_gradient' else self.horizon,
            ik_update_step_s=None if self.control_mode=='running_gradient' else self.horizon,
            ik_update_count=0 if self.control_mode=='running_gradient' else 1,
            future_ik_updates=0,lookahead_enabled=False,
            reached=self.frame.reached,collided=self.frame.collided,
            final_physical_error_m=row['physical_error_m'])

    def close(self):
        if self.closed:
            return
        self.closed=True
        for queue in (self.acquisition_requests,self.estimate_requests,self.control_requests):
            queue.put(None)
        # Drain camera replies while joining: a multiprocessing Queue feeder
        # otherwise waits forever on a pipe full of the final raw image.
        camera=self.workers[2]
        deadline=time.monotonic()+3.
        while camera.is_alive() and time.monotonic()<deadline:
            self.collect_archives()
            camera.join(timeout=.01)
        self.collect_archives()
        for worker in reversed(self.workers):
            worker.join(timeout=.1 if worker is camera else 3.)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1.)
        # A shutdown is not an assertion that inference drained. The immutable
        # DB plus physical-side acquisition journal retain pending work; record
        # exactly what remains before closing queues/connections.
        _append(self.output/'shutdown.jsonl',dict(
            closed_wall_stamp=time.time(),motion_finished=self.motion_finished,
            estimation_drained=self.estimation_finished,
            sensor_event_database=self.store.stats(),
            acquisition_submitted=self.acquisition_submitted,
            acquisition_acknowledged=self.acquisition_completed,
            acquisition_recovery_journal='acquisition_requests.jsonl',
            interrupted_inference_event=self.pending_inference_event,
            selected_but_not_dispatched_event=(None if self.selected_sensor_event is None
                                              else self.selected_sensor_event.event_id),
            interrupted_control_request=self.pending_control_request,
            workers=[dict(exitcode=worker.exitcode) for worker in self.workers]))
        for queue in (self.estimate_requests,self.estimate_replies,self.control_requests,
                      self.control_replies,self.acquisition_requests,self.acquisition_replies):
            queue.cancel_join_thread()
            queue.close()
        self.store.close()
