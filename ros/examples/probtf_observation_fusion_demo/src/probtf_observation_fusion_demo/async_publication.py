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
from .board_camera import BoardCamera, make_camera
from .display import snapshot_record
from .motion_display import MotionDisplay
from .node import image_message
from .runtime_monitor import RuntimeMonitor


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
        direct_camera_preview=bool(getattr(session, 'direct_camera_preview', False)),
        display_hz=float(session.settings['display']['publish_hz']))


def _publication_scene(specification):
    """Own a reference-only geometry/FK workspace, never a second estimate."""
    config = specification['config']
    board = config.get('marker_board', {})
    camera = make_camera(specification['texture_path'], board, config['camera'],
        config['seed'], marker_size=config['marker']['size'])
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
        self.monitor = RuntimeMonitor(self.output, 'publication')
        self.scene = (scene_factory or _publication_scene)(specification)
        self.display = (display_factory or MotionDisplay)(self.scene,
            specification['case'], specification['sensors'])
        publisher = publisher_factory or rospy.Publisher
        self.distributions = publisher('/observation_fusion/distributions', MarkerArray, queue_size=4, latch=True)
        self.snapshots = publisher('/observation_fusion/snapshot', String, queue_size=4, latch=True)
        self.stiffness = publisher('/observation_fusion/stiffness', String, queue_size=4, latch=True)
        # A live session publishes the same captured preview directly in its
        # acquisition process.  A second publisher here could replay older
        # queued images after that independent live image has already arrived.
        self.images = (None if specification.get('direct_camera_preview', False) else
            publisher('/observation_fusion/detection_image', Image, queue_size=1, latch=True))
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

    def process(self, packet, *, received_monotonic_s=None, received_wall_stamp=None):
        # Record queue receipt before validation, raw publication or rendering.
        # The older queue_delay_s below intentionally keeps its original
        # boundary after raw publication for historical comparisons.
        received = self.clock() if received_monotonic_s is None else received_monotonic_s
        received_wall = time.time() if received_wall_stamp is None else received_wall_stamp
        if not isinstance(packet, PublicationPacket) or packet.sequence != self.sequence+1:
            raise ValueError('Publication packets must have contiguous sequence numbers.')
        context = dict(sequence=packet.sequence, frame_index=packet.frame_index,
            physical_stamp=packet.frame.stamp)
        received_delay = max(0., received-packet.enqueued_monotonic_s)
        self.monitor.mark('queue_received', **context,
            queue_received_monotonic_s=received, queue_received_wall_stamp=received_wall,
            queue_received_delay_s=received_delay, raw_event_count=len(packet.raw_events))
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
        # The live image has already been acquired and archived.  Send its
        # unchanged pixels/source stamp before scene rendering and snapshot
        # serialization, without delaying or discarding any raw publication.
        if self.images is not None and packet.preview is not None:
            index, image, stamp = packet.preview
            if index != self.published_image:
                with self.monitor.stage('preview_publish', **context,
                                        image_index=index, image_source_stamp=stamp):
                    self.images.publish(image_message(image, stamp, 'camera_optical_frame'))
                self.published_image = index
        with self.monitor.stage('raw_publish', **context,
                                raw_event_count=len(packet.raw_events)):
            self._publish_raw_events(packet.raw_events)
        if packet.control is not None:
            self.current = packet.control
        current = self.current
        delay = max(0., self.clock()-packet.enqueued_monotonic_s)
        if current is not None:
            key = control_key(current)
            if key != self.published_control:
                snapshot = current['snapshot']
                source = dict(context, control_request_id=current.get('request_id'),
                    control_event_id=current.get('event_id'), snapshot_stamp=snapshot.stamp,
                    snapshot_version=snapshot.version)
                with self.monitor.stage('snapshot_wire_publish', **source,
                                        wire_bytes=len(current['markers_wire'])):
                    self.distributions.publish(rospy.AnyMsg().deserialize(current['markers_wire']))
                with self.monitor.stage('snapshot_json_publish', **source):
                    self.snapshots.publish(String(json.dumps(snapshot_record(snapshot))))
                with self.monitor.stage('stiffness_json_publish', **source):
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
                queue_received_delay_s=received_delay,
                queue_received_monotonic_s=received, queue_received_wall_stamp=received_wall,
                raw_events_published=dict(self.raw_counts),
                physical_frame_is_captured_source=True,
                rendering_changes_candidate_source_clock=False)
            due = self.clock()-self.last_scene_publication >= 1./self.specification['display_hz']
            self.display.scene.trajectory = self.trajectory
            # This measures the publisher-side scene adapter, including its
            # ROS calls. RViz's asynchronous GPU rendering is not observed.
            with self.monitor.stage('scene_publish', **context, publish_scene=due,
                                    snapshot_stamp=current['snapshot'].stamp):
                self.display.publish(packet.frame, packet.frame_index, current['snapshot'],
                    current['capture_index'], stiffness_record=current['snapshot'].stiffness_record,
                    execution=metadata, publish_scene=due)
            if due:
                self.last_scene_publication = self.clock()
            with self.monitor.stage('motion_metadata_publish', **context):
                self.motion.publish(String(json.dumps(metadata)))
            with self.monitor.stage('manifest_json_save', **context):
                path = self.output/'motion_manifest.json'
                temporary = path.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(metadata, indent=2)+'\n')
                temporary.replace(path)
        record = dict(kind='published', pid=os.getpid(), sequence=packet.sequence,
            frame_index=packet.frame_index, physical_stamp=packet.frame.stamp,
            queue_delay_s=delay, trajectory_rows=len(self.trajectory),
            queue_received_delay_s=received_delay,
            queue_received_monotonic_s=received, queue_received_wall_stamp=received_wall,
            raw_events_in_packet=len(packet.raw_events), raw_counts=dict(self.raw_counts),
            published_control=self.published_control, published_image=self.published_image)
        with self.monitor.stage('publication_record_save', **context):
            _append(self.output/'publication.jsonl', record)
        self.monitor.mark('packet_completed', **context,
            published_control=self.published_control, published_image=self.published_image)
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
            received = renderer.clock()
            received_wall = time.time()
            if packet is None:
                break
            if isinstance(packet, dict) and packet.get('kind') == 'failure':
                renderer.motion.publish(String(json.dumps(packet['record'])))
                continue
            if rospy.is_shutdown():
                raise RuntimeError('The independent ROS publication node shut down.')
            replies.put(renderer.process(packet, received_monotonic_s=received,
                                         received_wall_stamp=received_wall))
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
