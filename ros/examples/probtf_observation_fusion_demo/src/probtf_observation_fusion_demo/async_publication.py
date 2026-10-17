"""ROS drawing and serialization in a process separate from physical ticks.

Packets retain immutable physical/core snapshots. The trajectory is appended
by contiguous index and each raw acquisition is published, without latest-only
replacement. This process owns its ROS node, models and display objects; its
physical reference geometry never becomes an inference input.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, Imu, JointState
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from probtf_switch_press_demo.body import EquilibriumBody
from probtf_switch_press_demo.vision import PlanarCamera
from .async_workers import _append, _freeze_arrays
from .board_camera import BoardCamera
from .display import snapshot_record
from .motion_display import MotionDisplay
from .node import image_message


@dataclass(frozen=True)
class PublicationPacket:
    sequence: int
    enqueued_monotonic_s: float
    frame: object
    frame_index: int
    trajectory_start: int
    trajectory_rows: tuple
    raw_events: tuple
    metadata: object
    control: object = None
    preview: object = None


def publication_specification(session, case):
    scene = session.scene
    return dict(output=str(session.output), case=case,
        sensors=[item['sensor'] for item in session.templates],
        config=scene.config, robot_description=scene.robot_description,
        urdf=str(scene.urdf), texture_path=str(scene.texture_path),
        world_base=scene.world_base, marker=scene.marker, goal=scene.goal,
        tip_offset=scene.tip_offset, imu_target_link=scene.imu_target_link,
        duration=scene.duration,
        display_hz=float(session.settings['display']['publish_hz']))


def _publication_scene(specification):
    """Own a reference-only geometry/FK workspace, never a second estimate."""
    config = specification['config']
    board = config.get('marker_board', {})
    camera = (BoardCamera(specification['texture_path'], board, config['camera'], config['seed'])
        if board.get('enabled', False) else PlanarCamera(specification['texture_path'],
            config['marker']['size'], config['camera'], config['seed']))
    return SimpleNamespace(config=config, camera=camera,
        robot_description=specification['robot_description'],
        texture_path=Path(specification['texture_path']),
        world_base=specification['world_base'], marker=specification['marker'],
        goal=specification['goal'], tip_offset=specification['tip_offset'],
        imu_target_link=specification['imu_target_link'], duration=specification['duration'], trajectory=[],
        plant=EquilibriumBody(Path(specification['urdf']), config['plant']['stiffness']))


def control_key(control):
    return (control.get('request_id'), control.get('event_id'),
            control['capture_index'], control['index'])


class PublicationRenderer:
    """The old ROS adapter, driven by ordered delta packets in its own process."""
    def __init__(self, specification, *, scene_factory=None, display_factory=None,
                 publisher_factory=None, clock=time.monotonic):
        self.specification, self.clock = specification, clock
        self.output = Path(specification['output'])
        self.scene = (scene_factory or _publication_scene)(specification)
        self.display = (display_factory or MotionDisplay)(self.scene,
            specification['case'], specification['sensors'])
        publisher = publisher_factory or rospy.Publisher
        self.distributions = publisher('/observation_fusion/distributions', MarkerArray, queue_size=4, latch=True)
        self.snapshots = publisher('/observation_fusion/snapshot', String, queue_size=4, latch=True)
        self.stiffness = publisher('/observation_fusion/stiffness', String, queue_size=4, latch=True)
        self.images = publisher('/observation_fusion/detection_image', Image, queue_size=2, latch=True)
        self.raw_images = publisher('/observation_fusion/raw_image', Image, queue_size=4)
        self.raw_imu = publisher('/observation_fusion/imu', Imu, queue_size=32)
        self.joint_commands = publisher('/observation_fusion/joint_command', JointState, queue_size=32)
        self.motion = publisher('/observation_fusion/motion_metadata', String, queue_size=4, latch=True)
        self.trajectory = []
        self.current = None
        self.published_control = self.published_image = None
        self.last_scene_publication = -np.inf
        self.sequence = -1
        self.raw_counts = dict(joint=0, imu=0, image=0)

    def _publish_raw_events(self, events):
        for event in events:
            kind = event['kind']
            if kind == 'joint':
                message = JointState()
                message.header.stamp = rospy.Time.from_sec(event['stamp'])
                message.name = ['joint_'+str(i+1) for i in range(len(event['command']))]
                message.position = list(event['command'])
                message.velocity = list(event['payload']['command_velocity'])
                self.joint_commands.publish(message)
            elif kind == 'imu':
                payload = event['payload']
                message = Imu()
                message.header.stamp = rospy.Time.from_sec(event['stamp'])
                message.header.frame_id = payload['imu_target_link']
                gravity = 9.80665*np.asarray(payload['gravity_in_sensor'])
                (message.linear_acceleration.x, message.linear_acceleration.y,
                 message.linear_acceleration.z) = map(float, gravity)
                direction = gravity/np.linalg.norm(gravity)
                covariance = (9.80665*payload['imu_sigma_rad'])**2*(np.eye(3)-np.outer(direction,direction))
                message.linear_acceleration_covariance = covariance.ravel().tolist()
                message.orientation_covariance[0] = message.angular_velocity_covariance[0] = -1.
                self.raw_imu.publish(message)
            elif kind == 'image':
                self.raw_images.publish(image_message(event['image'], event['stamp'], 'camera_optical_frame'))
            else:
                raise ValueError('Unknown raw publication kind: '+str(kind))
            self.raw_counts[kind] += 1

    def process(self, packet):
        if not isinstance(packet, PublicationPacket) or packet.sequence != self.sequence+1:
            raise ValueError('Publication packets must have contiguous sequence numbers.')
        if packet.trajectory_start != len(self.trajectory):
            raise ValueError('Publication trajectory has a gap, overlap or duplicate delta.')
        if packet.frame_index != len(self.trajectory)+len(packet.trajectory_rows)-1:
            raise ValueError('Publication physical frame and trajectory endpoint differ.')
        _freeze_arrays(packet)
        self.trajectory.extend(packet.trajectory_rows)
        row = self.trajectory[packet.frame_index]
        if row['stamp'] != packet.frame.stamp or tuple(row['command']) != tuple(packet.frame.command):
            raise ValueError('Published physical frame has a different trajectory command/time.')
        self.sequence = packet.sequence
        self._publish_raw_events(packet.raw_events)
        if packet.control is not None:
            self.current = packet.control
        current = self.current
        delay = max(0., self.clock()-packet.enqueued_monotonic_s)
        if current is not None:
            key = control_key(current)
            if key != self.published_control:
                snapshot = current['snapshot']
                self.distributions.publish(rospy.AnyMsg().deserialize(current['markers_wire']))
                self.snapshots.publish(String(json.dumps(snapshot_record(snapshot))))
                self.stiffness.publish(String(json.dumps(snapshot.stiffness_record)))
                self.published_control = key
            if packet.metadata is None:
                raise ValueError('An active plotted control requires its matching physical metadata.')
            metadata = dict(packet.metadata)
            metadata['displayed_snapshot'] = dict(stamp=current['snapshot'].stamp,
                command=list(current['snapshot'].command), version=current['snapshot'].version,
                body_source_ids=current['diagnostics']['body_source_state_ids'],
                goal_source_ids=current['diagnostics']['goal_source_state_ids'])
            metadata['publication_process'] = dict(pid=os.getpid(), packet_sequence=packet.sequence,
                queue_delay_s=delay, trajectory_rows=len(self.trajectory),
                raw_events_published=dict(self.raw_counts),
                physical_frame_is_captured_source=True,
                rendering_changes_candidate_source_clock=False)
            due = self.clock()-self.last_scene_publication >= 1./self.specification['display_hz']
            self.display.scene.trajectory = self.trajectory
            self.display.publish(packet.frame, packet.frame_index, current['snapshot'],
                current['capture_index'], stiffness_record=current['snapshot'].stiffness_record,
                execution=metadata, publish_scene=due)
            if due:
                self.last_scene_publication = self.clock()
            self.motion.publish(String(json.dumps(metadata)))
            path = self.output/'motion_manifest.json'
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(metadata, indent=2)+'\n')
            temporary.replace(path)
        if packet.preview is not None:
            index, image, stamp = packet.preview
            if index != self.published_image:
                self.images.publish(image_message(image, stamp, 'camera_optical_frame'))
                self.published_image = index
        record = dict(kind='published', pid=os.getpid(), sequence=packet.sequence,
            frame_index=packet.frame_index, physical_stamp=packet.frame.stamp,
            queue_delay_s=delay, trajectory_rows=len(self.trajectory),
            raw_events_in_packet=len(packet.raw_events), raw_counts=dict(self.raw_counts),
            published_control=self.published_control, published_image=self.published_image)
        _append(self.output/'publication.jsonl', record)
        return record


def publication_worker(requests, replies, specification):
    """Own ROS and its GIL in a fresh process; report every failure to physics."""
    renderer = None
    try:
        cv2.setNumThreads(1)
        # roslaunch's __name remap must not steal the coordinator's node name.
        arguments = [sys.argv[0]]+[arg for arg in sys.argv[1:]
            if not arg.startswith(('__name:=', '__log:='))]
        rospy.init_node('probtf_async_publication', argv=arguments,
                        anonymous=True, disable_signals=True)
        renderer = PublicationRenderer(specification)
        replies.put(dict(kind='ready', pid=os.getpid(), node=rospy.get_name()))
        while True:
            packet = requests.get()
            if packet is None:
                break
            if isinstance(packet, dict) and packet.get('kind') == 'failure':
                renderer.motion.publish(String(json.dumps(packet['record'])))
                continue
            if rospy.is_shutdown():
                raise RuntimeError('The independent ROS publication node shut down.')
            replies.put(renderer.process(packet))
        stopped = dict(kind='stopped', pid=os.getpid(), graceful=True,
            last_sequence=renderer.sequence, trajectory_rows=len(renderer.trajectory),
            raw_counts=dict(renderer.raw_counts))
        _append(Path(specification['output'])/'publication_shutdown.jsonl', stopped)
        replies.put(stopped)
    except Exception:
        failure = dict(failed=True, worker='publication', pid=os.getpid(),
                       traceback=traceback.format_exc())
        _append(Path(specification['output'])/'publication_failure.jsonl', failure)
        replies.put(failure)
    finally:
        if not rospy.is_shutdown():
            rospy.signal_shutdown('independent publication process stopped')
