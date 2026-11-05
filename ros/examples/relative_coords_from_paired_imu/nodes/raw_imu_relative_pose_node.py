#!/usr/bin/env python3
"""Transport adapter: multiplexed sensor_msgs/Imu -> native ProbTF v2."""

from queue import SimpleQueue
from threading import Thread
import sys
import sysconfig

# This workstation puts system Python packages on PYTHONPATH. When catkin
# selects a venv interpreter, prefer its numerical packages over that older
# system NumPy, while retaining the ROS/generated-message search paths.
if sys.prefix != sys.base_prefix:
    site_packages = sysconfig.get_path("purelib")
    if site_packages in sys.path:
        sys.path.remove(site_packages)
    sys.path.insert(0, site_packages)

import numpy as np
import rospy
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty

from relative_coords_from_paired_imu.imu_pair_pipeline import ImuRelativePosePipeline, imu_measurement_covariance
from probtf_msgs.msg import ProbabilisticTransformStamped
from probtf_ros.v2_conversions import transform_distribution_to_msg


def _vector(value):
    return np.array([value.x, value.y, value.z], dtype=float)


class RawImuRelativePoseNode:
    def __init__(self):
        self.input_queue = SimpleQueue()
        self._reset_token = object()
        self.pipeline = ImuRelativePosePipeline(
            parent_frame_id=rospy.get_param("~parent_frame_id"),
            child_frame_id=rospy.get_param("~child_frame_id"),
            window_size=rospy.get_param("~window_size", 11),
            polynomial_order=rospy.get_param("~polynomial_order", 7),
            fit_history_size=rospy.get_param("~fit_history_size", 10),
            use_child_gyro=rospy.get_param("~use_child_gyro", True),
            rotation_forgetting_factor=rospy.get_param("~rotation_forgetting_factor", 1.0),
            position_observation_weight=rospy.get_param("~position_observation_weight", 1.0),
            position_residual_variance=rospy.get_param("~position_residual_variance", 1.0),
            prior_position_variance=rospy.get_param("~prior_position_variance", 1e30),
            integration_steps=rospy.get_param("~integration_steps", 50),
            source_id=rospy.get_param("~source_id", "imu_extended_leastsq"),
            edge_id=rospy.get_param("~edge_id", None),
            authority=rospy.get_name(),
        )
        self.default_gyro_variance = rospy.get_param("~default_gyro_variance", 1e-4)
        self.default_force_variance = rospy.get_param("~default_force_variance", 1e-3)
        self.biases = {}
        for role, frame in zip(("parent", "child"), self.pipeline.preprocessor.frames):
            gyro_bias = np.asarray(rospy.get_param("~{}_gyro_bias".format(role), [0.0] * 3), dtype=float).reshape(3)
            force_bias = np.asarray(rospy.get_param("~{}_force_bias".format(role), [0.0] * 3), dtype=float).reshape(3)
            if not np.all(np.isfinite(gyro_bias)) or not np.all(np.isfinite(force_bias)):
                raise ValueError("IMU calibration biases must be finite.")
            self.biases[frame] = (gyro_bias, force_bias)
        # Validate configured fallback variances even if input already has covariances.
        imu_measurement_covariance(np.zeros(9), self.default_gyro_variance)
        imu_measurement_covariance(np.zeros(9), self.default_force_variance)
        self.publisher = rospy.Publisher(
            "~relative_pose", ProbabilisticTransformStamped, queue_size=0
        )
        self.reset_subscriber = rospy.Subscriber("~reset", Empty, self._reset, queue_size=1)
        self.subscriber = rospy.Subscriber(
            "~imu", Imu, self.input_queue.put, queue_size=None, buff_size=8 * 1024 * 1024,
            tcp_nodelay=True,
        )
        self.worker = Thread(target=self._run, name="imu_relative_pose_estimation", daemon=True)
        rospy.on_shutdown(lambda: self.input_queue.put(None))
        self.worker.start()
        rospy.loginfo(
            "IMU ProbTF: %s -> %s, raw=%s, output=%s; continuously estimating a rigid IMU pair",
            self.pipeline.estimator.child_frame_id,
            self.pipeline.estimator.parent_frame_id,
            self.subscriber.resolved_name,
            self.publisher.resolved_name,
        )

    def _reset(self, _message):
        # Reset in FIFO order with already received observations, in the same
        # worker that owns the posterior and SG histories.
        self.input_queue.put(self._reset_token)

    def _run(self):
        while not rospy.is_shutdown():
            message = self.input_queue.get()
            if message is None:
                return
            if message is self._reset_token:
                self.pipeline.reset()
                rospy.loginfo("IMU relative-transform posterior and input windows reset")
            else:
                self._update(message)

    def _update(self, message):
        try:
            selector = self.pipeline.preprocessor.select_frame(message.header.frame_id)
            if selector is None:
                return
            gyro_bias, force_bias = self.biases[selector]
            records = self.pipeline.update_all(
                message.header.frame_id,
                message.header.stamp.to_sec(),
                _vector(message.angular_velocity) - gyro_bias,
                _vector(message.linear_acceleration) - force_bias,
                imu_measurement_covariance(message.angular_velocity_covariance, self.default_gyro_variance),
                imu_measurement_covariance(message.linear_acceleration_covariance, self.default_force_variance),
            )
            for record in records:
                # A delayed input may release several chronological updates.
                # Publish ALL their original laws and evaluation timestamps.
                self.publisher.publish(transform_distribution_to_msg(record, time_factory=rospy.Time.from_sec))
        except (TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            rospy.logwarn_throttle(2.0, "IMU relative-transform input/update failed: %s", error)


if __name__ == "__main__":
    rospy.init_node("imu_relative_pose")
    RawImuRelativePoseNode()
    rospy.spin()
