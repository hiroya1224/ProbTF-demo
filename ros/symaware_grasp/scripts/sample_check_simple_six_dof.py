#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import numpy as np
import quaternion
import rospy
import rospkg

from symaware_grasp.prob_tf.geometry import quat_to_rotmat
from symaware_grasp.prob_tf.urdf_override import build_tree_from_prob_tf_yaml


DEFAULT_BINGHAM_SOURCE_DIR = os.environ.get("BINGHAM_SOURCE_DIR", "/home/leus/BinghamNLL/src")
if os.path.isdir(DEFAULT_BINGHAM_SOURCE_DIR) and DEFAULT_BINGHAM_SOURCE_DIR not in sys.path:
    sys.path.insert(0, DEFAULT_BINGHAM_SOURCE_DIR)

from bingham.distribution import BinghamDistribution


def _default_config_path():
    return Path(rospkg.RosPack().get_path("symaware_grasp")) / "configs" / "simple_six_dof_prob_tf.yaml"


class SampleCheckSimpleSixDofNode:
    def __init__(self):
        self.config_path = Path(rospy.get_param("~config_path", str(_default_config_path())))
        self.target = str(rospy.get_param("~target", "tool0"))
        self.sample_count = int(rospy.get_param("~samples", 1000))
        self.seed = int(rospy.get_param("~seed", 19))

    def run(self):
        tree = build_tree_from_prob_tf_yaml(self.config_path)
        analytic = tree.lookup(tree.root, self.target, return_bingham=False, summarize=True)
        path = tree.lookup_path(tree.root, self.target)
        edges = [tree.edges[view.edge_id] for view in path]
        rng = np.random.default_rng(self.seed)

        sampled_positions = []
        for _ in range(max(self.sample_count, 1)):
            position = np.zeros(3, dtype=float)
            rotation = np.eye(3, dtype=float)
            for edge in edges:
                position = position + rotation @ edge.translation
                if edge.joint_type == "fixed":
                    sample_rotation = np.eye(3, dtype=float)
                else:
                    distribution = BinghamDistribution(A=edge.bingham_param)
                    quat = quaternion.as_float_array(distribution.update_sample())
                    sample_rotation = quat_to_rotmat(quat)
                rotation = rotation @ sample_rotation
            sampled_positions.append(position)

        sampled_positions = np.asarray(sampled_positions, dtype=float)
        sample_mean = sampled_positions.mean(axis=0)
        sample_cov = np.cov(sampled_positions.T, bias=True)

        rospy.loginfo("target: %s", self.target)
        rospy.loginfo("analytic mean: %s", analytic.mean_translation.tolist())
        rospy.loginfo("sample mean: %s", sample_mean.tolist())
        rospy.loginfo("analytic cov: %s", np.asarray(analytic.cov_translation).tolist())
        rospy.loginfo("sample cov: %s", np.asarray(sample_cov).tolist())
        rospy.loginfo("mean error norm: %.6e", float(np.linalg.norm(sample_mean - analytic.mean_translation)))
        rospy.signal_shutdown("done")


def main():
    rospy.init_node("sample_check_simple_six_dof")
    SampleCheckSimpleSixDofNode().run()


if __name__ == "__main__":
    main()
