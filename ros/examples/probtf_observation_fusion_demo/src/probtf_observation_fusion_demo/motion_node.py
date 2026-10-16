"""Observation-driven physical motion with same-tick inference and display."""
from dataclasses import replace
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


def prepare_motion(observations_path, output):
    started = time.monotonic()
    package = Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
    _, templates = load_inputs(package/'config/common.yaml', observations_path)
    output.mkdir(parents=True, exist_ok=False)
    scene = MovingSwitchScene(output)
    inference = MovingInference(scene.robot_description, templates)
    control = inference.settings['control']
    controller = MovingController(
        inference, scene.robot_description, control['goal_offset'],
        control['goal_orientation_wxyz'], control,
        integration_steps=int(control['integration_steps']))
    period = float(scene.config['motion']['frame_period'])
    times = list(np.arange(0., scene.duration, period)) + [scene.duration]
    frames, updates, update_for_frame = [], [], []
    command = scene.initial_command.copy()
    with (output/'estimation.jsonl').open('w') as records:
        for index, elapsed in enumerate(times):
            frame = scene.step(elapsed, stamp=1792080000.+elapsed, command=command)
            # Every drawn physical tick has a fresh fixed-command estimate.
            # Preparation happens before playback, so holding an older estimate
            # here would introduce an artificial command/time mismatch.
            result = inference.update(scene.sensor_inputs(frame), index)
            stage = result.stages[-1]
            snapshot = replace(build_snapshot(result, stage, inference.common),
                               summary='', show_labels=False)
            settings = inference.common['display']
            message = make_markers(snapshot, settings['seed'], settings['point_size'])
            path = output/'estimate_{:03d}'.format(index)
            save_snapshot(path, snapshot, message,
                          seed=settings['seed'], point_size=settings['point_size'])
            debug = result.detection_image.copy()
            label = 'Camera used for estimation' if any(t['sensor']=='marker' for t in templates) else 'Camera reference only; NOT used for estimation'
            cv2.putText(debug, label, (12,debug.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (255,0,255) if debug.ndim == 3 else 50, 1, cv2.LINE_AA)
            image = image_message(debug, frame.stamp, 'camera_optical_frame')
            require_current_update(frame, snapshot, image, index, index)
            image_path = path.with_suffix('.png')
            if not cv2.imwrite(str(image_path), debug):
                raise IOError('Could not save the observation image: ' + str(image_path))
            # Thousands of Python Point objects and full-resolution images per
            # frame must not remain resident throughout the trajectory.  Read
            # the exact verified wire data and lossless image on the next tick.
            updates.append((snapshot, path.with_suffix('.markers.bin'), image_path))
            if index + 1 < len(times):
                command, diagnostics = controller.step(result, times[index+1]-elapsed)
                result.records['controller'] = diagnostics
            else:
                # The common time limit ends the experiment.  Physical success
                # or collision never changes the controller or its run length.
                result.records['controller'] = {
                    'status': 'common_duration_complete',
                    'source_stamp': snapshot.stamp,
                    'source_command': list(snapshot.command),
                    'source_version': snapshot.version,
                }
            records.write(json.dumps(result.records)+'\n')
            update_for_frame.append(index)
            # The exact observed/detected pixels are retained in updates.
            frames.append(replace(frame, image=np.empty((0,0), dtype=np.uint8)))
    physical_bytes = (json.dumps(scene.trajectory, separators=(',',':'))+'\n').encode()
    (output/'physical_trajectory.json').write_bytes(physical_bytes)
    command_bytes = (json.dumps([
        dict(time=frame.time, command=frame.command.tolist()) for frame in frames
    ], separators=(',', ':'))+'\n').encode()
    (output/'sent_commands.json').write_bytes(command_bytes)
    preparation_seconds = time.monotonic()-started
    manifest = {
        'duration': scene.duration, 'frames': len(frames), 'updates': len(updates),
        'frame_period_s': period, 'inference_period_s': period,
        'playback_hz': 1./period,
        'preparation_wall_seconds': preparation_seconds,
        'preparation_frames_per_second': len(frames)/preparation_seconds,
        'physical_estimate_time_offset_s': 0.,
        'initial_command': scene.initial_command.tolist(),
        'scene_config_sha256': scene.config_sha256,
        'estimation_config_sha256': hashlib.sha256(
            (package/'config/moving_estimation.yaml').read_bytes()).hexdigest(),
        'command_sha256': hashlib.sha256(command_bytes).hexdigest(),
        'physics_sha256': hashlib.sha256(physical_bytes).hexdigest(),
        'sensors': [t['sensor'] for t in templates],
        'same_motion_sensor_estimation_comparison': False,
        'same_physical_model_and_initial_prior': True,
        'sensors_drive_commands': True,
        'command_source': 'common controller applied to the current fused estimate; no command recording',
        'final_physical_error_m': scene.trajectory[-1]['physical_error_m'],
        'final_rigid_error_m': scene.trajectory[-1]['rigid_command_error_m'],
        'reached': frames[-1].reached, 'collided': frames[-1].collided,
        'minimum_clearance_m': scene.monitor.minimum_clearance,
    }
    (output/'motion_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return scene, frames, updates, update_for_frame, manifest


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
        # Keep only two decoded future images, rather than the entire movie.
        # PNG decoding happens outside the timer callback while the previous
        # exact frame is displayed.
        self.image_executor = ThreadPoolExecutor(max_workers=1,
                                                  thread_name_prefix='observation-image')
        self.image_futures = {}
        self.distributions = rospy.Publisher('/observation_fusion/distributions', MarkerArray, queue_size=4,latch=True)
        self.snapshots = rospy.Publisher('/observation_fusion/snapshot', String, queue_size=4,latch=True)
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
            self.display.publish(self.frames[target], target, snapshot, self.update_index)

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
            self.images.publish(image)
            self.update_index = update_index
        # Publish TF heartbeats even while holding the first/final pose.
        self.display.publish(frame, index, snapshot, update_index)
        if index != self.index:
            metadata = dict(self.manifest, sample_index=index, scene_elapsed=frame.time,
                            finished=index==len(self.frames)-1, estimate_index=update_index,
                            estimate_stamp=snapshot.stamp, physical_stamp=frame.stamp,
                            command=frame.command.tolist(), valid_observations=len(snapshot.observation_ids))
            self.motion.publish(String(json.dumps(metadata)))
            self.index = index


def main():
    rospy.init_node('observation_fusion_demo')
    package = Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
    observations = Path(rospy.get_param('~observations_config', str(package/'config/imu_marker.yaml')))
    output = Path(rospy.get_param('~output_dir', '/tmp/probtf_observation_fusion_motion/'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    rospy.loginfo('Preparing observation-driven switch motion and same-time estimates: %s', output)
    scene, frames, updates, mapping, manifest = prepare_motion(observations, output)
    playback = MotionPlayback(scene,frames,updates,mapping,manifest,observations.stem,
                               bool(rospy.get_param('~autostart',True)))
    rospy.loginfo('Ready: %.3f s, %d physical frames, %d inference updates',scene.duration,len(frames),len(updates))
    rospy.spin()
