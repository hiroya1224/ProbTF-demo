"""Observation-driven physical motion with same-tick inference and display."""
from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
from threading import RLock
import time

import cv2
import numpy as np
import rospy
import rospkg
import yaml
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import Image

from .display import make_markers, save_snapshot, snapshot_record
from .model import load_inputs
from .moving_scene import MovingSwitchScene
from .moving_inference import MovingInference
from .moving_control import MovingController
from .motion_display import MotionDisplay
from .stiffness_identification import StiffnessIdentifier
from .stiffness_command import StiffnessCommandMap
from .node import build_snapshot, image_message


def require_current_update(frame, snapshot, image, frame_index, update_index):
    """Reject mixing one physical tick with another command's distribution.

    Only the shared command/time metadata is compared.  Physical joint angles
    and poses are neither inputs to the estimator nor candidate centres.
    """
    if frame_index != update_index:
        raise AssertionError('Every physical frame requires its own inference update.')
    if (snapshot.stamp != frame.stamp
            or tuple(snapshot.command) != tuple(frame.command)
            or image.header.stamp != rospy.Time.from_sec(frame.stamp)):
        raise AssertionError('Physical frame, estimate and observed image are not simultaneous.')
    suffix = ':frame:{:05d}'.format(frame_index)
    if any(not observation_id.endswith(suffix) for observation_id in snapshot.observation_ids):
        raise AssertionError('An observation belongs to a different physical frame.')
    for point in snapshot.points:
        query = point.query
        if (query.stamp != frame.stamp or tuple(query.command) != tuple(frame.command)
                or query.version != snapshot.version or query.frame != snapshot.frame_id):
            raise AssertionError('A displayed query belongs to a different command/time snapshot.')
    if snapshot.stiffness_record is not None:
        MotionDisplay.stiffness_marker(snapshot, snapshot.stiffness_record)


def read_image(snapshot, image_path):
    """Decode one saved observation without changing its source timestamp."""
    pixels = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if pixels is None:
        raise IOError('Could not read the saved observation image: ' + str(image_path))
    return image_message(pixels, snapshot.stamp, 'camera_optical_frame')


def read_update(update):
    """Read the verified wire message and lossless pixels for one frame only."""
    snapshot, markers_path, image_path = update
    message = MarkerArray().deserialize(markers_path.read_bytes())
    return snapshot, message, read_image(snapshot, image_path)


@dataclass
class MotionTick:
    index: int
    frame: object
    result: object
    snapshot: object
    markers: object
    image: object
    path: Path
    timings: dict
    command_map: object = None


class MotionSession:
    """One sequential experiment, shared by live execution and explicit replay.

    ``observe`` produces the current immutable estimate. The caller publishes
    that exact tick before ``finish`` selects the next short command. Physics
    advances only on the next observe call, never on a separate wall timer.
    """
    def __init__(self, observations_path, output, *, imu_target_link=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.package = Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
        _, self.templates = load_inputs(self.package/'config/common.yaml', observations_path)
        self.scene = (MovingSwitchScene(self.output) if imu_target_link is None
                      else MovingSwitchScene(self.output, imu_target_link=imu_target_link))
        self.inference = MovingInference(self.scene.robot_description, self.templates)
        self.stiffness = StiffnessIdentifier(self.scene.urdf, tuple(self.inference.body_ids),
            yaml.safe_load((self.package/'config/stiffness_identification.yaml').read_text()))
        control = self.inference.settings['control']
        self.controller = MovingController(self.inference, self.scene.robot_description,
            control['goal_offset'], control['goal_orientation_wxyz'], control,
            integration_steps=int(control['integration_steps']))
        self.period = float(self.scene.config['motion']['frame_period'])
        self.times = list(np.arange(0., self.scene.duration, self.period)) + [self.scene.duration]
        self.command = self.scene.initial_command.copy()
        self.index, self.pending = 0, None
        self.last_tick = None
        self.sent_commands, self.stiffness_counts = [], {}
        self.initial_stiffness = None
        self.command_proposal_used = False
        self.command_correction_used = False
        self.command_map_for_prediction = None
        self.work_seconds = 0.
        self.sources = tuple(sorted(Path(__file__).parent.glob('*.py')))
        self.source_hashes = self._source_hashes()

    def _source_hashes(self):
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.sources}

    @property
    def finished(self):
        return self.index == len(self.times)

    def observe(self):
        if self.pending is not None or self.finished:
            raise RuntimeError('A motion tick must finish before the next observation.')
        started = previous = time.monotonic()
        elapsed = self.times[self.index]
        frame = self.scene.step(elapsed, stamp=1792080000.+elapsed, command=self.command)
        timings = {'physics_and_camera': time.monotonic()-previous}
        previous = time.monotonic()
        result = self.inference.update(self.scene.sensor_inputs(frame), self.index,
                                       command_map=self.command_map_for_prediction)
        timings['inference'] = time.monotonic()-previous
        previous = time.monotonic()
        stiffness_record = self.stiffness.update(result)
        timings['stiffness_update'] = time.monotonic()-previous
        previous = time.monotonic()
        proposal = self.stiffness.command_proposal(result)
        result.records['stiffness'] = stiffness_record
        if self.stiffness.factors and proposal is None:
            raise ValueError('The estimated stiffness cannot define the requested corrected IK action: '
                             + str(stiffness_record.get('proposal_failure_reason')))
        command_map = (None if proposal is None
                       else StiffnessCommandMap(self.stiffness, result, proposal))
        stiffness_record['command_correction'] = None if command_map is None else command_map.record()
        stiffness_record['body_reconstructed_from_stiffness'] = False
        stiffness_record['feedback_to_observation_update'] = False
        stiffness_record['feedback_to_body_prediction'] = command_map is not None
        if command_map is not None:
            stiffness_record['feedback_to_body'] = True
            stiffness_record['feedback_to_control'] = 'one-step IK and the same next-prior equilibrium increment'
        timings['stiffness_command_proposal'] = time.monotonic()-previous
        previous = time.monotonic()
        snapshot = replace(build_snapshot(result, result.stages[-1], self.inference.common),
                           summary='', show_labels=False, stiffness_record=stiffness_record)
        timings['snapshot_core_queries'] = time.monotonic()-previous
        previous = time.monotonic()
        settings = self.inference.common['display']
        markers = make_markers(snapshot, settings['seed'], settings['point_size'])
        image = image_message(result.detection_image, frame.stamp, 'camera_optical_frame')
        require_current_update(frame, snapshot, image, self.index, self.index)
        path = self.output/'estimate_{:03d}'.format(self.index)
        save_snapshot(path, snapshot, markers, seed=settings['seed'], point_size=settings['point_size'])
        if not cv2.imwrite(str(path.with_suffix('.png')), result.detection_image):
            raise IOError('Could not save the observation image: ' + str(path))
        timings['render_verify_and_save'] = time.monotonic()-previous
        timings['observation_work'] = time.monotonic()-started
        self.pending = MotionTick(self.index, frame, result, snapshot, markers, image, path,
                                  timings, command_map)
        return self.pending

    def finish(self, tick):
        if tick is not self.pending:
            raise AssertionError('Command selection requires the same observed/displayed tick.')
        started = time.monotonic()
        if tick.index+1 < len(self.times):
            proposal = tick.snapshot.stiffness_record.get('command_proposal')
            self.command, diagnostics = self.controller.step(
                tick.result, self.times[tick.index+1]-self.times[tick.index],
                stiffness_proposal=proposal, command_map=tick.command_map)
            self.command_map_for_prediction = tick.command_map
            self.command_correction_used = self.command_correction_used or tick.command_map is not None
            self.command_proposal_used = self.command_proposal_used or proposal is not None
        else:
            diagnostics = dict(status='common_duration_complete',
                source_stamp=tick.snapshot.stamp, source_command=list(tick.snapshot.command),
                source_version=tick.snapshot.version)
        tick.timings['command_selection'] = time.monotonic()-started
        tick.timings['cycle_work'] = tick.timings['observation_work']+tick.timings['command_selection']
        self.work_seconds += tick.timings['cycle_work']
        tick.result.records['controller'] = diagnostics
        tick.result.records['execution'] = dict(
            sample_index=tick.index, simulated_elapsed_s=tick.frame.time,
            simulated_step_s=self.period, timings_s=tick.timings,
            skipped_simulation_ticks=0, asynchronous_estimate_reuse=False)
        with (self.output/'estimation.jsonl').open('a') as records:
            records.write(json.dumps(tick.result.records)+'\n')
        self.sent_commands.append(dict(time=tick.frame.time, command=tick.frame.command.tolist()))
        stiffness = tick.snapshot.stiffness_record
        if self.initial_stiffness is None:
            self.initial_stiffness = stiffness['prior']
        status = stiffness['status']
        self.stiffness_counts[status] = self.stiffness_counts.get(status, 0)+1
        self.last_tick, self.pending = tick, None
        self.index += 1
        if self.finished:
            self.write_manifest()

    def write_manifest(self, mode='live'):
        if self.last_tick is None:
            raise RuntimeError('A manifest requires at least one completed observation.')
        if self._source_hashes() != self.source_hashes:
            raise RuntimeError('Implementation changed during the experiment; repeat with frozen sources.')
        physical_bytes = (json.dumps(self.scene.trajectory, separators=(',', ':'))+'\n').encode()
        command_bytes = (json.dumps(self.sent_commands, separators=(',', ':'))+'\n').encode()
        (self.output/'physical_trajectory.json').write_bytes(physical_bytes)
        (self.output/'sent_commands.json').write_bytes(command_bytes)
        stiffness = self.last_tick.snapshot.stiffness_record
        manifest = dict(
            execution_mode=mode, offline_precomputation=mode == 'prepared_playback',
            output_dir=str(self.output), duration=self.scene.duration, frames=len(self.times),
            updates=self.index, completed=self.finished,
            implementation_sha256=self.source_hashes, implementation_unchanged_during_execution=True,
            implementation_unchanged_during_preparation=(True if mode == 'prepared_playback' else None),
            frame_period_s=self.period, inference_period_s=self.period,
            playback_hz=1./self.period if mode == 'prepared_playback' else None,
            preparation_wall_seconds=self.work_seconds if mode == 'prepared_playback' else 0.,
            preparation_frames_per_second=self.index/self.work_seconds if mode == 'prepared_playback' else None,
            computation_wall_seconds=self.work_seconds,
            computation_frames_per_second=self.index/self.work_seconds,
            simulated_elapsed_s=self.last_tick.frame.time,
            physical_estimate_time_offset_s=0.,
            initial_command=self.scene.initial_command.tolist(),
            scene_config_sha256=self.scene.config_sha256,
            estimation_config_sha256=hashlib.sha256((self.package/'config/moving_estimation.yaml').read_bytes()).hexdigest(),
            stiffness_config_sha256=hashlib.sha256((self.package/'config/stiffness_identification.yaml').read_bytes()).hexdigest(),
            command_sha256=hashlib.sha256(command_bytes).hexdigest(),
            physics_sha256=hashlib.sha256(physical_bytes).hexdigest(),
            sensors=[item['sensor'] for item in self.templates],
            imu_target_link=self.scene.imu_target_link,
            imu_direction_sigma_rad=self.scene.imu_sigma_rad,
            imu_mount='known identity attachment to '+self.scene.imu_target_link,
            same_motion_sensor_estimation_comparison=False, same_physical_model_and_initial_prior=True,
            sensors_drive_commands=True,
            command_source='common controller applied to the current fused estimate; no command recording',
            final_physical_error_m=self.scene.trajectory[-1]['physical_error_m'],
            final_rigid_error_m=self.scene.trajectory[-1]['rigid_command_error_m'],
            reached=self.last_tick.frame.reached, collided=self.last_tick.frame.collided,
            minimum_clearance_m=self.scene.monitor.minimum_clearance,
            stiffness_prior=self.initial_stiffness, stiffness_posterior=stiffness['posterior'],
            stiffness_retained_memory=(stiffness['posterior'] if stiffness['posterior'] is not None else stiffness['retained_prior']),
            stiffness_status_counts=dict(self.stiffness_counts),
            stiffness_feeds_body=self.command_correction_used,
            stiffness_feeds_body_prediction=self.command_correction_used,
            body_reconstructed_from_stiffness=False,
            stiffness_command_proposal_enabled=True, stiffness_feeds_body_or_control=self.command_proposal_used)
        (self.output/'motion_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        return manifest


def prepare_motion(observations_path, output, *, imu_target_link=None):
    """Explicit offline mode; live uses exactly the same observe/finish path."""
    session = MotionSession(observations_path, output, imu_target_link=imu_target_link)
    frames, updates = [], []
    while not session.finished:
        tick = session.observe()
        session.finish(tick)
        updates.append((tick.snapshot, tick.path.with_suffix('.markers.bin'), tick.path.with_suffix('.png')))
        frames.append(replace(tick.frame, image=np.empty((0, 0), dtype=np.uint8)))
    manifest = session.write_manifest(mode='prepared_playback')
    return session.scene, frames, updates, list(range(len(frames))), manifest


class MotionPlayback:
    def __init__(self, scene, frames, updates, update_for_frame, manifest, case, autostart):
        self.scene, self.frames, self.updates = scene, frames, updates
        self.update_for_frame, self.manifest = update_for_frame, manifest
        if len(updates) != len(frames) or tuple(update_for_frame) != tuple(range(len(frames))):
            raise AssertionError('Playback requires one same-time inference update per physical frame.')
        self.started = None
        self.index = self.update_index = None
        self.current_update = None
        self.lock = RLock()
        # Forward the already verified MarkerArray bytes directly. rospy
        # accepts AnyMsg on this typed publisher without changing its advertised
        # ROS type, payload, or nested source timestamps. Rebuilding thousands
        # of Point objects only to serialize them again wastes the frame budget.
        self.marker_messages = tuple(rospy.AnyMsg().deserialize(update[1].read_bytes())
                                     for update in updates)
        self.snapshot_messages = tuple(String(json.dumps(snapshot_record(update[0])))
                                       for update in updates)
        self.stiffness_messages = tuple(String(json.dumps(update[0].stiffness_record))
                                        for update in updates)
        # Keep only two decoded future images, rather than the entire movie.
        # PNG decoding happens outside the timer callback while the previous
        # exact frame is displayed.
        self.image_executor = ThreadPoolExecutor(max_workers=1,
                                                  thread_name_prefix='observation-image')
        self.image_futures = {}
        self.distributions = rospy.Publisher('/observation_fusion/distributions', MarkerArray, queue_size=4,latch=True)
        self.snapshots = rospy.Publisher('/observation_fusion/snapshot', String, queue_size=4,latch=True)
        self.stiffness = rospy.Publisher('/observation_fusion/stiffness', String, queue_size=4,latch=True)
        self.motion = rospy.Publisher('/observation_fusion/motion_metadata', String, queue_size=4,latch=True)
        self.images = rospy.Publisher('/observation_fusion/detection_image', Image, queue_size=4,latch=True)
        self.display = MotionDisplay(scene, case, manifest['sensors'])
        self.service = rospy.Service('/observation_fusion/replay', Trigger, self.replay)
        rospy.on_shutdown(self.stop_prefetch)
        self._prefetch_images(0)
        self.started = time.monotonic() if autostart else None
        self.timer = rospy.Timer(rospy.Duration(float(scene.config['motion']['frame_period'])), self.advance)
        self.advance(None)

    def replay(self, _request):
        with self.lock:
            self.started = time.monotonic()
            self.index = self.update_index = None
            for future in self.image_futures.values():
                future.cancel()
            self.image_futures.clear()
            self._prefetch_images(0)
            self.advance(None)
        return TriggerResponse(True, 'Replaying this sensor case from its common initial command.')

    def advance(self, _event):
        # The replay service and timer use different ROS threads.  Publish one
        # complete frame before allowing either thread to select another tick.
        with self.lock:
            self._advance()

    def _advance(self):
        elapsed = 0. if self.started is None else time.monotonic()-self.started
        period = float(self.scene.config['motion']['frame_period'])
        target = min(int(elapsed/period), len(self.frames)-1)
        first = 0 if self.index is None else self.index+1
        if first <= target:
            # Retain the wall-clock playback rate. If one callback arrives
            # late, publish each due source frame instead of silently skipping
            # its command, observation, and verified distribution message.
            for index in range(first, target+1):
                self._publish_frame(index)
        else:
            snapshot = self.current_update[0]
            self.display.publish(self.frames[target], target, snapshot, self.update_index,
                                 stiffness_record=snapshot.stiffness_record)

    def _prefetch_images(self, index):
        for old in tuple(self.image_futures):
            if old < index:
                self.image_futures.pop(old).cancel()
        for selected in range(index, min(index+3, len(self.updates))):
            if selected not in self.image_futures:
                snapshot, _, image_path = self.updates[selected]
                self.image_futures[selected] = self.image_executor.submit(
                    read_image, snapshot, image_path)

    def stop_prefetch(self):
        self.image_executor.shutdown(wait=False)

    def _publish_frame(self, index):
        update_index = self.update_for_frame[index]
        self._prefetch_images(update_index)
        image = self.image_futures.pop(update_index).result()
        snapshot = self.updates[update_index][0]
        self.current_update = (snapshot, self.marker_messages[update_index], image)
        snapshot, message, image = self.current_update
        frame = self.frames[index]
        require_current_update(frame, snapshot, image, index, update_index)
        if update_index != self.update_index:
            self.distributions.publish(message)
            self.snapshots.publish(self.snapshot_messages[update_index])
            self.stiffness.publish(self.stiffness_messages[update_index])
            self.images.publish(image)
            self.update_index = update_index
        # Publish TF heartbeats even while holding the first/final pose.
        self.display.publish(frame, index, snapshot, update_index,
                             stiffness_record=snapshot.stiffness_record)
        if index != self.index:
            metadata = dict(self.manifest, sample_index=index, scene_elapsed=frame.time,
                            finished=index==len(self.frames)-1, estimate_index=update_index,
                            estimate_stamp=snapshot.stamp, physical_stamp=frame.stamp,
                            command=frame.command.tolist(), valid_observations=len(snapshot.observation_ids))
            self.motion.publish(String(json.dumps(metadata)))
            self.index = index


class MotionLive:
    """Compute and publish every logical tick before choosing its next command.

    Simulation time has an explicit fixed step. Expensive computation slows
    the experiment's wall rate; it never advances physics under an old cloud.
    No prepared trajectory, image, or distribution is read by this executor.
    """
    def __init__(self, observations_path, output, case, autostart, *, imu_target_link=None,
                 max_updates=0):
        self.observations_path, self.output, self.case = observations_path, Path(output), case
        self.imu_target_link = imu_target_link
        self.max_updates = int(max_updates)
        if self.max_updates < 0:
            raise ValueError('max_updates must be zero (unlimited) or a positive observation count.')
        self.lock = RLock()
        self.replays = 0
        self.failure = None
        self.started = time.monotonic() if autostart else None
        self.publish_times = []
        self.session = MotionSession(observations_path, self.output, imu_target_link=imu_target_link)
        self.distributions = rospy.Publisher('/observation_fusion/distributions', MarkerArray, queue_size=4, latch=True)
        self.snapshots = rospy.Publisher('/observation_fusion/snapshot', String, queue_size=4, latch=True)
        self.stiffness = rospy.Publisher('/observation_fusion/stiffness', String, queue_size=4, latch=True)
        self.images = rospy.Publisher('/observation_fusion/detection_image', Image, queue_size=4, latch=True)
        self.motion = rospy.Publisher('/observation_fusion/motion_metadata', String, queue_size=4, latch=True)
        self.display = MotionDisplay(self.session.scene, case, [item['sensor'] for item in self.session.templates])
        self.current_tick = None
        self.common_manifest = None
        self._run_tick()
        self.service = rospy.Service('/observation_fusion/replay', Trigger, self.replay)
        self.timer = rospy.Timer(rospy.Duration(self.session.period), self.advance)

    def timing(self):
        now = time.monotonic()
        elapsed = 0. if self.started is None else now-self.started
        hz = ((len(self.publish_times)-1)/(self.publish_times[-1]-self.publish_times[0])
              if len(self.publish_times) > 1 else None)
        simulated = 0. if self.current_tick is None else self.current_tick.frame.time
        return dict(execution_mode='live', simulated_step_s=self.session.period,
                    wall_elapsed_s=elapsed, actual_update_hz=hz,
                    failure_reason=None if self.failure is None else self.failure['failure_reason'],
                    realtime_factor=simulated/elapsed if elapsed > 0 else None,
                    requested_update_limit=self.max_updates,
                    paused_at_update_limit=(self.max_updates > 0
                        and self.session.index >= self.max_updates and not self.session.finished))

    def _run_tick(self):
        tick = self.session.observe()
        self.current_tick = tick
        # Use precisely the wire object verified and archived in observe.
        self.distributions.publish(tick.markers)
        self.snapshots.publish(String(json.dumps(snapshot_record(tick.snapshot))))
        self.stiffness.publish(String(json.dumps(tick.snapshot.stiffness_record)))
        self.images.publish(tick.image)
        self.publish_times.append(time.monotonic())
        self.display.publish(tick.frame, tick.index, tick.snapshot, tick.index,
                             stiffness_record=tick.snapshot.stiffness_record, execution=self.timing())
        # Every same-time publication above precedes next-command selection.
        self.session.finish(tick)
        if (self.common_manifest is None or self.session.finished
                or self.timing()['paused_at_update_limit']):
            self.common_manifest = self.session.write_manifest(mode='live')
        metadata = dict(self.common_manifest, **self.timing(),
            sample_index=tick.index, scene_elapsed=tick.frame.time,
            simulated_elapsed_s=tick.frame.time, updates=self.session.index,
            completed=self.session.finished, finished=self.session.finished,
            estimate_index=tick.index, estimate_stamp=tick.snapshot.stamp,
            physical_stamp=tick.frame.stamp, command=tick.frame.command.tolist(),
            valid_observations=len(tick.snapshot.observation_ids),
            tick_timings_s=tick.timings,
            stiffness_posterior=tick.snapshot.stiffness_record['posterior'],
            stiffness_retained_memory=(tick.snapshot.stiffness_record['posterior']
                if tick.snapshot.stiffness_record['posterior'] is not None
                else tick.snapshot.stiffness_record['retained_prior']),
            final_physical_error_m=self.session.scene.trajectory[-1]['physical_error_m'],
            final_rigid_error_m=self.session.scene.trajectory[-1]['rigid_command_error_m'],
            minimum_clearance_m=self.session.scene.monitor.minimum_clearance,
            reached=tick.frame.reached, collided=tick.frame.collided,
            computation_wall_seconds=self.session.work_seconds,
            computation_frames_per_second=self.session.index/self.session.work_seconds,
            stiffness_feeds_body_or_control=self.session.command_proposal_used,
            stiffness_feeds_body=self.session.command_correction_used,
            stiffness_feeds_body_prediction=self.session.command_correction_used,
            stiffness_status_counts=dict(self.session.stiffness_counts))
        if not self.session.finished:
            metadata.update(command_sha256=None, physics_sha256=None)
        self.motion.publish(String(json.dumps(metadata)))

    def advance(self, _event):
        with self.lock:
            if (self.failure is None and self.started is not None and not self.session.finished
                    and not self.timing()['paused_at_update_limit']):
                try:
                    self._run_tick()
                except Exception as error:
                    failed_index = min(self.session.index, len(self.session.times)-1)
                    failed_elapsed = self.session.times[failed_index]
                    self.failure = dict(execution_mode='live', failed=True, finished=False,
                        failure_reason='{}: {}'.format(type(error).__name__, error),
                        failed_sample_index=failed_index,
                        failed_simulated_elapsed_s=failed_elapsed,
                        failed_source_stamp=1792080000.+failed_elapsed,
                        failed_command=self.session.command.tolist(),
                        last_published_sample_index=self.current_tick.index,
                        last_published_stamp=self.current_tick.snapshot.stamp,
                        last_published_command=list(self.current_tick.snapshot.command))
                    (self.session.output/'execution_failure.json').write_text(json.dumps(self.failure, indent=2)+'\n')
                    self.motion.publish(String(json.dumps(self.failure)))
                    rospy.logerr('Live experiment stopped without a replacement estimate: %s', self.failure['failure_reason'])
            else:
                # Heartbeats retain the same physical and estimate source.
                tick = self.current_tick
                self.display.publish(tick.frame, tick.index, tick.snapshot, tick.index,
                    stiffness_record=tick.snapshot.stiffness_record, execution=self.timing())

    def replay(self, _request):
        with self.lock:
            if self.failure is None and self.started is None and self.session.index == 1:
                # Start the waiting experiment without assimilating t=0 twice.
                self.started = time.monotonic()
                self.publish_times.clear()
            else:
                self.replays += 1
                output = self.output.with_name(self.output.name+'_replay_{:03d}'.format(self.replays))
                self.session = MotionSession(self.observations_path, output,
                                             imu_target_link=self.imu_target_link)
                self.display.scene = self.session.scene
                self.failure = None
                self.started = time.monotonic()
                self.publish_times.clear()
                self.common_manifest = None
                self._run_tick()
        return TriggerResponse(True, 'Starting a fresh sequential experiment from the common initial prior.')


def main():
    rospy.init_node('observation_fusion_demo')
    package = Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
    observations = Path(rospy.get_param('~observations_config', str(package/'config/imu_marker.yaml')))
    output = Path(rospy.get_param('~output_dir', '/tmp/probtf_observation_fusion_motion/'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    mode = rospy.get_param('~execution_mode', 'live')
    autostart = bool(rospy.get_param('~autostart', True))
    imu_target_link = rospy.get_param('~imu_target_link', '') or None
    max_updates = int(rospy.get_param('~max_updates', 0))
    if mode == 'live':
        playback = MotionLive(observations, output, observations.stem, autostart,
                              imu_target_link=imu_target_link, max_updates=max_updates)
        rospy.loginfo('Live sequential execution: simulation dt %.3f s; actual wall rate is measured.',
                      playback.session.period)
    elif mode == 'prepared_playback':
        rospy.loginfo('Preparing explicitly requested offline playback: %s', output)
        scene, frames, updates, mapping, manifest = prepare_motion(
            observations, output, imu_target_link=imu_target_link)
        playback = MotionPlayback(scene,frames,updates,mapping,manifest,observations.stem, autostart)
        rospy.loginfo('Playback ready: %.3f s, %d same-time inference updates', scene.duration, len(updates))
    else:
        raise ValueError('execution_mode must be live or prepared_playback.')
    rospy.spin()
