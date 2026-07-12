import copy
from pathlib import Path

import numpy as np
import yaml

from symaware_grasp.arm_kinematics import ToyArm6DOF
from symaware_grasp.prob_tf.urdf_override import build_tree_from_prob_tf_yaml, load_prob_tf_yaml


def _config_path():
    return (
        Path(__file__).resolve().parents[2]
        / "ros"
        / "examples"
        / "symaware_grasp"
        / "configs"
        / "simple_six_dof_prob_tf.yaml"
    )


def test_tool0_mean_approaches_deterministic_fk_under_small_uncertainty(tmp_path):
    config = load_prob_tf_yaml(_config_path())
    config = copy.deepcopy(config)
    for edge in config["edges"]:
        if edge["type"] == "revolute":
            edge["bingham_eigenvalues"] = [30000.0, -10000.0, -10000.0, -10000.0]

    config_path = tmp_path / "small_uncertainty.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    tree = build_tree_from_prob_tf_yaml(config_path)
    result = tree.lookup("base_link", "tool0", summarize=True, return_bingham=False)
    expected_position, _, _ = ToyArm6DOF().forward_kinematics(np.zeros(6, dtype=float))
    assert np.allclose(result.mean_translation, expected_position, atol=2e-2)


def test_lookup_point_zero_offset_matches_link_origin():
    tree = build_tree_from_prob_tf_yaml(_config_path())
    origin_result = tree.lookup("base_link", "link_4", summarize=True, return_bingham=False)
    point_result = tree.lookup_point("base_link", "link_4", [0.0, 0.0, 0.0], summarize=True, return_bingham=False)

    assert np.allclose(point_result.mean_translation, origin_result.mean_translation, atol=1e-10)
    assert np.allclose(point_result.cov_translation, origin_result.cov_translation, atol=1e-10)


def test_lookup_point_mean_approaches_deterministic_attached_point_under_small_uncertainty(tmp_path):
    config = load_prob_tf_yaml(_config_path())
    config = copy.deepcopy(config)
    for edge in config["edges"]:
        if edge["type"] == "revolute":
            edge["bingham_eigenvalues"] = [30000.0, -10000.0, -10000.0, -10000.0]

    config_path = tmp_path / "small_uncertainty_attached_point.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    tree = build_tree_from_prob_tf_yaml(config_path)
    local_point = np.array([0.05, -0.01, 0.02], dtype=float)
    result = tree.lookup_point("base_link", "tool0", local_point, summarize=True, return_bingham=False)
    expected_position, _, _ = ToyArm6DOF().forward_kinematics(np.zeros(6, dtype=float))
    expected_attached_point = expected_position + local_point
    assert np.allclose(result.mean_translation, expected_attached_point, atol=2e-2)


def test_lookup_point_tangent_surrogate_zero_offset_matches_link_origin():
    tree = build_tree_from_prob_tf_yaml(_config_path())
    origin_result = tree.lookup("base_link", "link_4", summarize=True, return_bingham=False)
    point_result = tree.lookup_point_tangent_surrogate(
        "base_link",
        "link_4",
        [0.0, 0.0, 0.0],
        summarize=True,
        return_bingham=False,
    )

    assert np.allclose(point_result.mean_translation, origin_result.mean_translation, atol=1e-10)
    assert np.allclose(point_result.cov_translation, origin_result.cov_translation, atol=1e-10)


def test_lookup_point_tangent_surrogate_matches_exact_covariance_scale():
    tree = build_tree_from_prob_tf_yaml(_config_path())
    local_point = np.array([0.08, 0.0, 0.0], dtype=float)
    exact_result = tree.lookup_point("base_link", "tool0", local_point, summarize=True, return_bingham=False)
    tangent_result = tree.lookup_point_tangent_surrogate(
        "base_link",
        "tool0",
        local_point,
        summarize=True,
        return_bingham=False,
    )

    assert np.allclose(tangent_result.mean_translation, exact_result.mean_translation, atol=5e-3)
    exact_trace = float(np.trace(exact_result.cov_translation))
    tangent_trace = float(np.trace(tangent_result.cov_translation))
    assert 0.8 <= tangent_trace / exact_trace <= 1.2
