#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import rospy
import rospkg

from probik_demo.prob_tf.urdf_override import build_tree_from_prob_tf_yaml, load_prob_tf_yaml
from probik_demo.prob_tf.visualize import plot_link_prob_tf, write_results_csv, write_results_json


def _default_config_path():
    return Path(rospkg.RosPack().get_path("probik_demo")) / "configs" / "simple_six_dof_prob_tf.yaml"


def _default_output_dir():
    return Path(rospkg.RosPack().get_path("probik_demo")) / "outputs" / "simple_six_dof"


class ShowLinkProbTfNode:
    def __init__(self):
        self.config_path = Path(rospy.get_param("~config_path", str(_default_config_path())))
        self.out_dir = Path(rospy.get_param("~out_dir", str(_default_output_dir())))

    def run(self):
        config = load_prob_tf_yaml(self.config_path)
        tree = build_tree_from_prob_tf_yaml(self.config_path)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for frame in config.get("frames", []):
            if frame == tree.root:
                continue
            results.append(tree.lookup(tree.root, frame, return_bingham=True, summarize=True))

        write_results_csv(results, self.out_dir / "link_prob_tf.csv")
        write_results_json(results, self.out_dir / "link_prob_tf.json")
        plot_link_prob_tf(results, self.out_dir / "link_prob_tf.png")

        for result in results:
            mean = np.asarray(result.mean_translation, dtype=float)
            covariance_trace = float(np.trace(result.cov_translation))
            rospy.loginfo(
                "%s mean=(%.6f, %.6f, %.6f) trace_cov=%.6e",
                result.target,
                mean[0],
                mean[1],
                mean[2],
                covariance_trace,
            )
        rospy.loginfo("Wrote Prob-TF outputs under %s", self.out_dir)
        rospy.signal_shutdown("done")


def main():
    rospy.init_node("show_link_prob_tf")
    ShowLinkProbTfNode().run()

if __name__ == "__main__":
    main()
