#!/usr/bin/env python3
"""Transport adapter: multiplexed sensor_msgs/Imu -> native ProbTF v2."""

from threading import Event, Thread
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

from relative_coords_from_paired_imu.imu_pair_pipeline import imu_measurement_covariance
from relative_coords_from_paired_imu.latest_imu_pair import LatestImuRelativePosePipeline
from probtf_msgs.msg import ProbabilisticTransformStamped
from probtf_ros.v2_conversions import transform_distribution_to_msg


def _vector(value):
    return np.array([value.x, value.y, value.z], dtype=float)


class RawImuRelativePoseNode:
    def __init__(self):
        self.wake = Event()
        self.reset_requested = Event()
        self.stopping = Event()
        self.pipeline = LatestImuRelativePosePipeline(
            parent_frame_id=rospy.get_param("~parent_frame_id"),
            child_frame_id=rospy.get_param("~child_frame_id"),
            window_size=rospy.get_param("~window_size", 11),
            polynomial_order=rospy.get_param("~polynomial_order", 7),
            fit_history_size=rospy.get_param("~fit_history_size", 10),
            raw_history_capacity=rospy.get_param("~raw_history_capacity", 16384),
            max_sample_gap_s=rospy.get_param("~max_sample_gap_s", None),
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
            "~relative_pose", ProbabilisticTransformStamped, queue_size=1
        )
        self.reset_subscriber = rospy.Subscriber("~reset", Empty, self._reset, queue_size=1)
        self.subscriber = rospy.Subscriber(
            "~imu", Imu, self._receive, queue_size=None, buff_size=8 * 1024 * 1024,
            tcp_nodelay=True,
        )
        self.worker = Thread(target=self._run, name="imu_relative_pose_estimation", daemon=True)
        rospy.on_shutdown(self._shutdown)
        self.worker.start()
        rospy.loginfo(
            "IMU ProbTF: %s -> %s, raw=%s, output=%s; continuously estimating a rigid IMU pair",
            self.pipeline.estimator.child_frame_id,
            self.pipeline.estimator.parent_frame_id,
            self.subscriber.resolved_name,
            self.publisher.resolved_name,
        )

    def _reset(self, _message):
        self.reset_requested.set()
        self.wake.set()

    def _shutdown(self):
        self.stopping.set()
        self.wake.set()

    def _receive(self, message):
        try:
            if self.pipeline.ingest(message.header.frame_id, message.header.stamp.to_nsec(), message):
                self.wake.set()
        except (TypeError, ValueError) as error:
            rospy.logwarn_throttle(2.0, "IMU input failed: %s", error)

    def _decode(self, sample):
        message = sample.payload
        selector = self.pipeline.preprocessor.select_frame(sample.frame_id)
        gyro_bias, force_bias = self.biases[selector]
        return (
            _vector(message.angular_velocity) - gyro_bias,
            _vector(message.linear_acceleration) - force_bias,
            imu_measurement_covariance(message.angular_velocity_covariance, self.default_gyro_variance),
            imu_measurement_covariance(message.linear_acceleration_covariance, self.default_force_variance),
        )

    def _run(self):
        while not self.stopping.is_set():
            self.wake.wait()
            self.wake.clear()
            if self.stopping.is_set():
                return
            if self.reset_requested.is_set():
                self.reset_requested.clear()
                self.pipeline.reset()
                rospy.loginfo("IMU relative-transform posterior and raw histories reset")
                continue
            try:
                record = self.pipeline.update_latest(self._decode)
                if record is None or self.reset_requested.is_set():
                    continue
                message = transform_distribution_to_msg(record, time_factory=rospy.Time.from_sec)
                ns = self.pipeline.last_selection.stamp_ns
                message.header.stamp = rospy.Time(ns // 1000000000, ns % 1000000000)
                self.publisher.publish(message)
            except (TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                rospy.logwarn_throttle(2.0, "IMU relative-transform update failed: %s", error)


if __name__ == "__main__":
    rospy.init_node("imu_relative_pose")
    RawImuRelativePoseNode()
    rospy.spin()
