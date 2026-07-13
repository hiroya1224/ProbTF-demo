#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import rospy
import rospkg
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from probtf.distributions import DistributionStatus
from probtf.temporal import TemporalPolicy
from probtf_ros import RosProbTfListener
from probtf_ros.bridge import PROBTF_STATIC_TOPIC, PROBTF_TOPIC
from symaware_grasp.probtf_config import load_prob_tf_config
from symaware_grasp.visualization import pack_rgb


def _default_config_path():
    return Path(rospkg.RosPack().get_path("symaware_grasp")) / "configs" / "simple_six_dof_prob_tf.yaml"


class ProbTfLinkCloudNode:
    def __init__(self):
        config = load_prob_tf_config(
            Path(rospy.get_param("~config_path", str(_default_config_path())))
        )
        self.root_frame = config.root_frame
        self.frames = tuple(frame for frame in config.frames if frame != config.root_frame)
        self.axis_length = float(rospy.get_param("~axis_length", 0.08))
        self.sample_count = int(rospy.get_param("~sample_count", 60))
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive.")
        self.publish_rate = float(rospy.get_param("~publish_rate", 1.0))
        self.lookup_timeout = float(rospy.get_param("~lookup_timeout", 5.0))
        seed = int(rospy.get_param("~seed", 29))
        self.rng = np.random.default_rng(seed if seed >= 0 else None)
        self.listener = RosProbTfListener(
            dynamic_topic=rospy.get_param("~probtf_topic", PROBTF_TOPIC),
            static_topic=rospy.get_param("~probtf_static_topic", PROBTF_STATIC_TOPIC),
        )
        self.publisher = rospy.Publisher(
            rospy.get_param("~cloud_topic", "/symaware_grasp/link_pointcloud"),
            PointCloud2,
            queue_size=1,
            latch=True,
        )
        self.point_fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]
        self.axis_colors = (pack_rgb(255, 64, 64), pack_rgb(64, 255, 64), pack_rgb(64, 64, 255))
        self.ready = False
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / max(self.publish_rate, 1e-3)),
            self.publish_cloud,
        )

    @staticmethod
    def regularized_covariance(covariance):
        covariance = np.asarray(covariance, dtype=float)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        return eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-10)) @ eigenvectors.T

    def _wait_until_ready(self):
        if self.ready:
            return True
        self.ready = all(
            self.listener.wait_for_lookup(
                self.root_frame,
                frame,
                policy=TemporalPolicy.LATEST,
                timeout=self.lookup_timeout,
            )
            for frame in self.frames
        )
        if not self.ready:
            rospy.logwarn_throttle(5.0, "Waiting for static ProbTF arm records.")
        return self.ready

    def publish_cloud(self, _event):
        if not self._wait_until_ready():
            return
        cloud_points = []
        for frame in self.frames:
            for axis_index, color in enumerate(self.axis_colors):
                local_point = np.zeros(3, dtype=float)
                local_point[axis_index] = self.axis_length
                result = self.listener.lookup_point_moments(
                    self.root_frame,
                    frame,
                    local_point,
                    policy=TemporalPolicy.LATEST,
                )
                if result.status is not DistributionStatus.OK:
                    rospy.logwarn_throttle(5.0, "Point moments unavailable for frame '%s'.", frame)
                    continue
                endpoints = self.rng.multivariate_normal(
                    result.value.mean,
                    self.regularized_covariance(result.value.covariance),
                    size=self.sample_count,
                )
                cloud_points.extend(
                    [float(point[0]), float(point[1]), float(point[2]), color]
                    for point in endpoints
                )
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.root_frame
        self.publisher.publish(point_cloud2.create_cloud(header, self.point_fields, cloud_points))


def main():
    rospy.init_node("prob_tf_link_cloud_node")
    ProbTfLinkCloudNode()
    rospy.spin()


if __name__ == "__main__":
    main()
