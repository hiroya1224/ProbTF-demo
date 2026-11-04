"""Message-field conversion for this example's ROS boundary.

ROS message classes are supplied by the nodes. Importing the numerical
package or this duck-typed adapter does not require rospy or generated types.
"""

import numpy as np

from .imu_kinematics import ImuKinematics


def imu_kinematics_from_msg(message):
    def vector(value):
        return np.array([value.x, value.y, value.z], dtype=float)

    stamp = message.header.stamp
    return ImuKinematics(
        frame_id=message.header.frame_id,
        angular_velocity=vector(message.angular_velocity),
        angular_acceleration=vector(message.angular_acceleration),
        specific_force=vector(message.specific_force),
        angular_velocity_covariance=np.asarray(message.angular_velocity_covariance, dtype=float).reshape(3, 3),
        angular_acceleration_covariance=np.asarray(message.angular_acceleration_covariance, dtype=float).reshape(3, 3),
        specific_force_covariance=np.asarray(message.specific_force_covariance, dtype=float).reshape(3, 3),
        stamp=stamp.to_sec() if hasattr(stamp, "to_sec") else float(stamp),
    )
