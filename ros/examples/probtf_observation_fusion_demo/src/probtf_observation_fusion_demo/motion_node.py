"""Fixed-time physics replay and same-snapshot inference for four sensor lists."""
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
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
from .motion_display import MotionDisplay
from .node import build_snapshot, image_message


def prepare_motion(observations_path, output):
    package = Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
    _, templates = load_inputs(package/'config/common.yaml', observations_path)
    output.mkdir(parents=True, exist_ok=False)
    scene = MovingSwitchScene(output)
    inference = MovingInference(scene.robot_description, templates)
    period = float(scene.config['motion']['frame_period'])
    times = list(np.arange(0., scene.duration, period)) + [scene.duration]
    frames, updates, update_for_frame = [], [], []
    last_update = -1.
    with (output/'estimation.jsonl').open('w') as records:
        for index, elapsed in enumerate(times):
            frame = scene.step(elapsed, stamp=1792080000.+elapsed)
            if index == 0 or elapsed-last_update >= .5-1e-9 or index == len(times)-1:
                update_index = len(updates)
                result = inference.update(scene.sensor_inputs(frame), update_index)
                stage = result.stages[-1]
                snapshot = replace(build_snapshot(result, stage, inference.common),
                                   summary='', show_labels=False)
                settings = inference.common['display']
                message = make_markers(snapshot, settings['seed'], settings['point_size'])
                save_snapshot(output/'estimate_{:03d}'.format(update_index), snapshot, message,
                              seed=settings['seed'], point_size=settings['point_size'])
                debug = result.detection_image.copy()
                label = 'Camera used for estimation' if any(t['sensor']=='marker' for t in templates) else 'Camera reference only; NOT used for estimation'
                cv2.putText(debug, label, (12,debug.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (255,0,255) if debug.ndim == 3 else 50, 1, cv2.LINE_AA)
                updates.append((snapshot, message, image_message(debug, frame.stamp, 'camera_optical_frame')))
                records.write(json.dumps(result.records)+'\n')
                last_update = elapsed
            update_for_frame.append(len(updates)-1)
            # Intermediate images are not observations; retain only poses for 20 Hz playback.
            frames.append(replace(frame, image=np.empty((0,0), dtype=np.uint8)))
    physical_bytes = (json.dumps(scene.trajectory, separators=(',',':'))+'\n').encode()
    (output/'physical_trajectory.json').write_bytes(physical_bytes)
    manifest = {
        'duration': scene.duration, 'frames': len(frames), 'updates': len(updates),
        'command_sha256': scene.command_file_sha256,
        'physics_sha256': hashlib.sha256(physical_bytes).hexdigest(),
        'sensors': [t['sensor'] for t in templates],
        'same_motion_sensor_estimation_comparison': True,
        'sensors_drive_commands': False,
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
        self.started = time.monotonic() if autostart else None
        self.index = self.update_index = None
        self.distributions = rospy.Publisher('/observation_fusion/distributions', MarkerArray, queue_size=1,latch=True)
        self.snapshots = rospy.Publisher('/observation_fusion/snapshot', String, queue_size=1,latch=True)
        self.motion = rospy.Publisher('/observation_fusion/motion_metadata', String, queue_size=1,latch=True)
        self.images = rospy.Publisher('/observation_fusion/detection_image', Image, queue_size=1,latch=True)
        self.display = MotionDisplay(scene, case, manifest['sensors'])
        self.service = rospy.Service('/observation_fusion/replay', Trigger, self.replay)
        self.timer = rospy.Timer(rospy.Duration(.05), self.advance)
        self.advance(None)

    def replay(self, _request):
        self.started = time.monotonic()
        self.index = self.update_index = None
        self.advance(None)
        return TriggerResponse(True, 'Replaying the common physical motion from its initial frame.')

    def advance(self, _event):
        elapsed = 0. if self.started is None else time.monotonic()-self.started
        period = float(self.scene.config['motion']['frame_period'])
        index = min(int(elapsed/period), len(self.frames)-1)
        update_index = self.update_for_frame[index]
        snapshot, message, image = self.updates[update_index]
        if update_index != self.update_index:
            self.distributions.publish(message)
            self.snapshots.publish(String(json.dumps(snapshot_record(snapshot))))
            self.images.publish(image)
            self.update_index = update_index
        frame = self.frames[index]
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
    rospy.loginfo('Preparing the identical switch motion and sensor snapshots: %s', output)
    scene, frames, updates, mapping, manifest = prepare_motion(observations, output)
    playback = MotionPlayback(scene,frames,updates,mapping,manifest,observations.stem,
                               bool(rospy.get_param('~autostart',True)))
    rospy.loginfo('Ready: %.3f s, %d physical frames, %d inference updates',scene.duration,len(frames),len(updates))
    rospy.spin()
