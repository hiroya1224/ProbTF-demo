#!/usr/bin/env python3

import message_filters
import numpy as np
import rospy

from probtf_estimators.imu_relative_pose import ImuRelativePoseEstimator
from probtf_estimators.ros_conversions import imu_kinematics_from_msg
from probtf_msgs.msg import ImuKinematics, ProbabilisticTransformStamped
from probtf_ros.v2_conversions import transform_distribution_to_msg


class ImuRelativePoseNode:
    def __init__(self):
        parent_frame = rospy.get_param("~parent_frame_id")
        child_frame = rospy.get_param("~child_frame_id")
        self.estimator = ImuRelativePoseEstimator(
            parent_frame_id=parent_frame,
            child_frame_id=child_frame,
            rotation_forgetting_factor=rospy.get_param(
                "~rotation_forgetting_factor",
                1.0,
            ),
            position_forgetting_factor=rospy.get_param(
                "~position_forgetting_factor",
                1.0,
            ),
            prior_position_variance=rospy.get_param(
                "~prior_position_variance",
                1e6,
            ),
            integration_steps=rospy.get_param("~integration_steps", 120),
            source_id=rospy.get_param("~source_id", "imu_relative_pose"),
            edge_id=rospy.get_param(
                "~edge_id",
                "{}__to__{}".format(parent_frame, child_frame),
            ),
            authority=rospy.get_name(),
        )
        self.publisher = rospy.Publisher(
            "~relative_pose",
            ProbabilisticTransformStamped,
            queue_size=10,
        )
        parent_subscriber = message_filters.Subscriber(
            "~parent_kinematics",
            ImuKinematics,
        )
        child_subscriber = message_filters.Subscriber(
            "~child_kinematics",
            ImuKinematics,
        )
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [parent_subscriber, child_subscriber],
            queue_size=rospy.get_param("~synchronizer_queue_size", 20),
            slop=rospy.get_param("~synchronizer_slop", 0.02),
        )
        self.synchronizer.registerCallback(self._update)

    def _update(self, parent_message, child_message):
        try:
            transform = self.estimator.update(
                imu_kinematics_from_msg(parent_message),
                imu_kinematics_from_msg(child_message),
            )
            self.publisher.publish(
                transform_distribution_to_msg(
                    transform,
                    time_factory=rospy.Time.from_sec,
                )
            )
        except (TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            rospy.logwarn_throttle(2.0, "ProbTF IMU relative-pose update rejected: %s", error)


if __name__ == "__main__":
    rospy.init_node("probtf_imu_relative_pose")
    ImuRelativePoseNode()
    rospy.spin()
