#!/usr/bin/env python3

import numpy as np
import rospy
from sensor_msgs.msg import Imu

from probtf.imu_preprocessing import ImuKinematicsPreprocessor
from probtf_msgs.msg import ImuKinematics


def _vector(vector):
    return np.array([vector.x, vector.y, vector.z], dtype=float)


def _covariance(values, fallback_variance):
    covariance = np.asarray(values, dtype=float).reshape(3, 3)
    if covariance[0, 0] < 0.0 or not np.all(np.isfinite(covariance)):
        return np.eye(3, dtype=float) * float(fallback_variance)
    return 0.5 * (covariance + covariance.T)


class ImuKinematicsNode:
    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id", "")
        self.default_gyro_variance = rospy.get_param("~default_gyro_variance", 1e-4)
        self.default_force_variance = rospy.get_param("~default_force_variance", 1e-3)
        self.processor = None
        self.window_size = rospy.get_param("~window_size", 9)
        self.polynomial_order = rospy.get_param("~polynomial_order", 2)
        self.minimum_samples = rospy.get_param("~minimum_samples", 5)
        self.publisher = rospy.Publisher("~kinematics", ImuKinematics, queue_size=10)
        self.subscriber = rospy.Subscriber("~imu", Imu, self._update, queue_size=50)

    def _update(self, message):
        frame_id = self.frame_id or message.header.frame_id
        if self.processor is None:
            self.processor = ImuKinematicsPreprocessor(
                frame_id,
                window_size=self.window_size,
                polynomial_order=self.polynomial_order,
                minimum_samples=self.minimum_samples,
            )
        elif frame_id.lstrip("/") != self.processor.frame_id:
            rospy.logwarn_throttle(2.0, "Ignoring IMU sample after frame_id changed")
            return
        try:
            result = self.processor.update(
                message.header.stamp.to_sec(),
                _vector(message.angular_velocity),
                _vector(message.linear_acceleration),
                _covariance(
                    message.angular_velocity_covariance,
                    self.default_gyro_variance,
                ),
                _covariance(
                    message.linear_acceleration_covariance,
                    self.default_force_variance,
                ),
            )
        except ValueError as error:
            rospy.logwarn_throttle(2.0, "ProbTF IMU preprocessing rejected a sample: %s", error)
            return
        if result is None:
            return
        output = ImuKinematics()
        output.header = message.header
        output.header.frame_id = result.frame_id
        output.angular_velocity.x, output.angular_velocity.y, output.angular_velocity.z = result.angular_velocity
        output.angular_velocity_covariance = result.angular_velocity_covariance.reshape(-1).tolist()
        output.angular_acceleration.x, output.angular_acceleration.y, output.angular_acceleration.z = result.angular_acceleration
        output.angular_acceleration_covariance = result.angular_acceleration_covariance.reshape(-1).tolist()
        output.specific_force.x, output.specific_force.y, output.specific_force.z = result.specific_force
        output.specific_force_covariance = result.specific_force_covariance.reshape(-1).tolist()
        self.publisher.publish(output)


if __name__ == "__main__":
    rospy.init_node("probtf_imu_kinematics")
    ImuKinematicsNode()
    rospy.spin()
