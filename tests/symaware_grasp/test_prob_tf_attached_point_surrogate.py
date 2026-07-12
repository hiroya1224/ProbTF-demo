from pathlib import Path

import numpy as np

from symaware_grasp.prob_tf.urdf_override import build_tree_from_prob_tf_yaml


def _config_path():
    return (
        Path(__file__).resolve().parents[2]
        / "ros"
        / "examples"
        / "symaware_grasp"
        / "configs"
        / "simple_six_dof_prob_tf.yaml"
    )


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
