from pathlib import Path

import numpy as np
import yaml

from symaware_grasp.prob_tf.geometry import axis_angle_to_quat, complete_orthonormal_basis, quat_normalize
from symaware_grasp.prob_tf.tree import ProbTfEdge, ProbTfTree


def load_prob_tf_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_bingham_param_from_mode(q_mode, eigenvalues):
    quaternion = quat_normalize(q_mode)
    basis = complete_orthonormal_basis(quaternion)
    eigs = np.asarray(eigenvalues, dtype=float).reshape(4)
    eigs = eigs - np.mean(eigs)
    return basis @ np.diag(eigs) @ basis.T


def build_tree_from_prob_tf_yaml(path):
    config = load_prob_tf_yaml(path)
    root = config["root"]
    tree = ProbTfTree(root=root)
    tree.frame_order = list(config.get("frames", [root]))

    for edge_spec in config.get("edges", []):
        joint_type = edge_spec["type"]
        nominal_quaternion = (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            if joint_type == "fixed"
            else axis_angle_to_quat(edge_spec["axis"], float(edge_spec.get("nominal_angle", 0.0)))
        )
        edge = ProbTfEdge(
            edge_id=edge_spec["joint"],
            parent=edge_spec["parent"],
            child=edge_spec["child"],
            translation=edge_spec["translation"],
            joint_type=joint_type,
            axis=edge_spec.get("axis"),
            nominal_angle=float(edge_spec.get("nominal_angle", 0.0)),
            nominal_quaternion=nominal_quaternion,
            bingham_param=(
                None
                if joint_type == "fixed"
                else make_bingham_param_from_mode(
                    q_mode=edge_spec.get("mode_quaternion", nominal_quaternion),
                    eigenvalues=edge_spec["bingham_eigenvalues"],
                )
            ),
        )
        tree.add_edge(edge)

    tree.config_path = str(Path(path))
    return tree
