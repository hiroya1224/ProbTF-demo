#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

from probtf.sensor_config import load_sensor_mounts


class SensorMountTfNode:
    def __init__(self):
        mounts = load_sensor_mounts(rospy.get_param("~config_file"))
        transforms = []
        for mount in mounts:
            if mount.parent_frame_id == mount.frame_id:
                continue
            message = TransformStamped()
            message.header.stamp = rospy.Time.now()
            message.header.frame_id = mount.parent_frame_id
            message.child_frame_id = mount.frame_id
            position = mount.position_xyz
            message.transform.translation.x = float(position[0])
            message.transform.translation.y = float(position[1])
            message.transform.translation.z = float(position[2])
            quaternion = mount.orientation_wxyz
            message.transform.rotation.w = float(quaternion[0])
            message.transform.rotation.x = float(quaternion[1])
            message.transform.rotation.y = float(quaternion[2])
            message.transform.rotation.z = float(quaternion[3])
            transforms.append(message)
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        if transforms:
            self.broadcaster.sendTransform(transforms)
        rospy.loginfo("Loaded %d sensor mounts and published %d static transforms", len(mounts), len(transforms))


if __name__ == "__main__":
    rospy.init_node("probtf_sensor_mount_tf")
    SensorMountTfNode()
    rospy.spin()
