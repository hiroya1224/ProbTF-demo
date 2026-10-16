"""Physical robots/scene are references; candidate clouds use only core queries."""
from types import SimpleNamespace

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from probtf_switch_press_demo.ros_display import Display as SwitchDisplay


def transform_message(parent, child, pose, stamp):
    message = TransformStamped()
    message.header.frame_id, message.header.stamp = parent, stamp
    message.child_frame_id = child
    message.transform.translation.x, message.transform.translation.y, message.transform.translation.z = pose[:3, 3]
    quaternion = Rotation.from_matrix(pose[:3, :3].copy()).as_quat()
    (message.transform.rotation.x, message.transform.rotation.y,
     message.transform.rotation.z, message.transform.rotation.w) = quaternion
    return message


class MotionDisplay:
    # Reuse the original switch's geometry drawing and AprilTag bitmap.
    marker = SwitchDisplay.marker
    points = staticmethod(SwitchDisplay.points)

    def __init__(self, scene, case, sensor_names):
        self.scene, self.case, self.sensor_names = scene, case, tuple(sensor_names)
        self.tag_texture = cv2.imread(str(scene.texture_path), cv2.IMREAD_GRAYSCALE)
        self.marker_id = 0
        rospy.set_param('/observation_fusion/robot_description', scene.robot_description)
        self.transforms = TransformBroadcaster()
        self.static = StaticTransformBroadcaster()
        stamp = rospy.Time.now()
        # The original setting explicitly gives the base placement in world.
        self.static.sendTransform([
            transform_message('world', 'base_link', scene.world_base, stamp),
            transform_message('base_link', 'rigid/base_link', np.eye(4), stamp),
        ])
        self.publisher = rospy.Publisher('/observation_fusion/scene', MarkerArray,
                                         queue_size=1, latch=True)

    def publish(self, frame, frame_index, snapshot, inference_index):
        stamp = rospy.Time.now()
        messages = []
        for prefix, poses in (('', frame.link_poses), ('rigid/', frame.rigid_link_poses)):
            for name, pose in poses.items():
                if name != 'base_link':
                    messages.append(transform_message('base_link', prefix + name, pose, stamp))
        self.transforms.sendTransform(messages)
        self.trial = SimpleNamespace(
            config=self.scene.config, marker=self.scene.marker, goal=self.scene.goal,
            status='reached' if frame.reached else 'running',
            monitor=SimpleNamespace(reached=frame.reached, collided=frame.collided),
            plant=self.scene.plant, pose=frame.joints, tip_offset=self.scene.tip_offset,
            trajectory=self.scene.trajectory[:frame_index+1], phase='closed_loop', cycle=0,
            ik_iteration=0,
        )
        output = SwitchDisplay.scene_markers(self)
        for marker in output.markers:
            if marker.ns == 'status':
                marker.text = (
                    '{} | {:.1f} / {:.1f} s\n'
                    'Observations -> estimate -> command | 20 fps\n'
                    'Opaque: gravity-flexed | Translucent: rigid FK\n'
                    'RGB: estimated XYZ endpoint distributions\n'
                    'Actual tip {:.1f} mm | Rigid tip {:.1f} mm\n'
                    'Observations {} | estimate t={:.1f} s | frame {}'
                ).format(self.case, frame.time, self.scene.duration,
                         1000*np.linalg.norm(frame.tip-self.scene.goal),
                         1000*np.linalg.norm(frame.rigid_tip-self.scene.goal),
                         len(snapshot.observation_ids), snapshot.stamp-1792080000., inference_index)
        imu = self.marker('imu_mount', Marker.CUBE,
                          (.55,.12,.75,1.) if 'imu' in self.sensor_names else (.4,.4,.4,.5),
                          (.025,.025,.025), pose=frame.link_poses['link_3'].copy())
        output.markers.append(imu)
        self.publisher.publish(output)
