#!/usr/bin/env python3

from pathlib import Path

import rospy
import rospkg
import yaml


DEFAULT_CONFIG = {
    "root": "base_link",
    "frames": [
        "base_link",
        "link_1",
        "link_2",
        "link_3",
        "link_4",
        "link_5",
        "link_6",
        "tool0",
    ],
    "edges": [
        {
            "joint": "joint_1",
            "parent": "base_link",
            "child": "link_1",
            "type": "revolute",
            "translation": [0.0, 0.0, 0.12],
            "axis": [0.0, 0.0, 1.0],
            "nominal_angle": 0.0,
            "bingham_kappa": 500.0,
            "bingham_eigenvalues": [1500.0, -500.0, -500.0, -500.0],
        },
        {
            "joint": "joint_2",
            "parent": "link_1",
            "child": "link_2",
            "type": "revolute",
            "translation": [0.0, 0.0, 0.18],
            "axis": [0.0, 1.0, 0.0],
            "nominal_angle": 0.0,
            "bingham_kappa": 450.0,
            "bingham_eigenvalues": [1350.0, -450.0, -450.0, -450.0],
        },
        {
            "joint": "joint_3",
            "parent": "link_2",
            "child": "link_3",
            "type": "revolute",
            "translation": [0.36, 0.0, 0.0],
            "axis": [0.0, 1.0, 0.0],
            "nominal_angle": 0.0,
            "bingham_kappa": 400.0,
            "bingham_eigenvalues": [1200.0, -400.0, -400.0, -400.0],
        },
        {
            "joint": "joint_4",
            "parent": "link_3",
            "child": "link_4",
            "type": "revolute",
            "translation": [0.30, 0.0, 0.0],
            "axis": [1.0, 0.0, 0.0],
            "nominal_angle": 0.0,
            "bingham_kappa": 350.0,
            "bingham_eigenvalues": [1050.0, -350.0, -350.0, -350.0],
        },
        {
            "joint": "joint_5",
            "parent": "link_4",
            "child": "link_5",
            "type": "revolute",
            "translation": [0.14, 0.0, 0.0],
            "axis": [0.0, 1.0, 0.0],
            "nominal_angle": 0.0,
            "bingham_kappa": 300.0,
            "bingham_eigenvalues": [900.0, -300.0, -300.0, -300.0],
        },
        {
            "joint": "joint_6",
            "parent": "link_5",
            "child": "link_6",
            "type": "revolute",
            "translation": [0.12, 0.0, 0.0],
            "axis": [1.0, 0.0, 0.0],
            "nominal_angle": 0.0,
            "bingham_kappa": 250.0,
            "bingham_eigenvalues": [750.0, -250.0, -250.0, -250.0],
        },
        {
            "joint": "tool0_joint",
            "parent": "link_6",
            "child": "tool0",
            "type": "fixed",
            "translation": [0.10, 0.0, 0.0],
        },
    ],
}

def _default_output_path():
    package_path = Path(rospkg.RosPack().get_path("symaware_grasp"))
    return package_path / "configs" / "simple_six_dof_prob_tf.yaml"


class WriteSimpleSixDofProbTfNode:
    def __init__(self):
        self.output_path = Path(rospy.get_param("~output_path", str(_default_output_path())))

    def run(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(DEFAULT_CONFIG, handle, sort_keys=False)
        rospy.loginfo("Wrote Prob-TF config to %s", self.output_path)
        rospy.signal_shutdown("done")


def main():
    rospy.init_node("write_simple_six_dof_prob_tf")
    WriteSimpleSixDofProbTfNode().run()


if __name__ == "__main__":
    main()
