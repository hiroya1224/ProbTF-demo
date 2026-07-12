#!/usr/bin/env python3

import os

import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from symaware_grasp.prob_tf.urdf_override import build_tree_from_prob_tf_yaml, load_prob_tf_yaml
from symaware_grasp.ptf_utils import pack_rgb


def _default_config_path():
    try:
        import rospkg

        return os.path.join(rospkg.RosPack().get_path("symaware_grasp"), "configs", "simple_six_dof_prob_tf.yaml")
    except Exception:
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "simple_six_dof_prob_tf.yaml")


class ProbTfLinkCloudNode:
    def __init__(self):
        self.config_path = rospy.get_param(
            "~config_path",
            _default_config_path(),
        )
        self.axis_length = float(rospy.get_param("~axis_length", 0.08))
        self.sample_count = int(rospy.get_param("~sample_count", 60))
        self.publish_rate = float(rospy.get_param("~publish_rate", 1.0))
        seed = int(rospy.get_param("~seed", 29))
        self.rng = np.random.default_rng(seed if seed >= 0 else None)
        self.frame_id = rospy.get_param("~frame_id", "base_link")

        self.tree = build_tree_from_prob_tf_yaml(self.config_path)
        self.config = load_prob_tf_yaml(self.config_path)
        self.frames = [frame for frame in self.config.get("frames", []) if frame != self.tree.root]
        self.endpoint_results = self._build_endpoint_results()
        self.publisher = rospy.Publisher("prob_tf_link_cloud", PointCloud2, queue_size=1, latch=True)
        self.point_fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]
        self.axis_colors = [pack_rgb(255, 64, 64), pack_rgb(64, 255, 64), pack_rgb(64, 64, 255)]
        rospy.Timer(rospy.Duration(1.0 / max(self.publish_rate, 1e-3)), self.publish_cloud)

    @staticmethod
    def regularized_covariance(covariance):
        covariance = 0.5 * (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def _build_endpoint_results(self):
        endpoint_results = {}
        unit_axes = np.eye(3, dtype=float)
        for frame in self.frames:
            endpoint_results[frame] = [
                self.tree.lookup_point_tangent_surrogate(
                    self.tree.root,
                    frame,
                    self.axis_length * unit_axes[:, axis_index],
                    return_bingham=False,
                    summarize=True,
                )
                for axis_index in range(3)
            ]
        return endpoint_results

    def publish_cloud(self, _event):
        cloud_points = []
        for _ in range(max(self.sample_count, 1)):
            for frame in self.frames:
                for axis_index, color in enumerate(self.axis_colors):
                    result = self.endpoint_results[frame][axis_index]
                    endpoint = self.rng.multivariate_normal(
                        np.asarray(result.mean_translation, dtype=float),
                        self.regularized_covariance(result.cov_translation),
                    )
                    cloud_points.append([float(endpoint[0]), float(endpoint[1]), float(endpoint[2]), color])

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id
        self.publisher.publish(point_cloud2.create_cloud(header, self.point_fields, cloud_points))


def main():
    rospy.init_node("prob_tf_link_cloud_node")
    ProbTfLinkCloudNode()
    rospy.spin()


if __name__ == "__main__":
    main()
