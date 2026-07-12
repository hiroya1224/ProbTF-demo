from pathlib import Path

import pytest

from symaware_grasp.prob_tf.path_expression import EdgeView, PathExpression
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


def test_lookup_path_forward_chain():
    tree = build_tree_from_prob_tf_yaml(_config_path())
    path = tree.lookup_path("base_link", "link_3")
    assert path.views == [
        EdgeView("joint_1", +1),
        EdgeView("joint_2", +1),
        EdgeView("joint_3", +1),
    ]


def test_lookup_path_inverse_chain_uses_same_edge_ids():
    tree = build_tree_from_prob_tf_yaml(_config_path())
    path = tree.lookup_path("link_3", "base_link")
    assert path.views == [
        EdgeView("joint_3", -1),
        EdgeView("joint_2", -1),
        EdgeView("joint_1", -1),
    ]


def test_edge_view_inverse_does_not_create_new_edge_id():
    assert EdgeView("joint_3", +1).inverse() == EdgeView("joint_3", -1)


def test_adjacent_inverse_cancellation():
    reduced = PathExpression([EdgeView("joint_3", +1), EdgeView("joint_3", -1)]).reduce_adjacent_inverses()
    assert reduced.views == []


def test_repeated_edges_after_reduction_are_rejected():
    path = PathExpression([EdgeView("joint_2", +1), EdgeView("joint_1", +1), EdgeView("joint_2", -1)])
    with pytest.raises(NotImplementedError):
        path.assert_no_repeated_edge_ids()


def test_summarized_results_cannot_be_re_registered_as_edges():
    tree = build_tree_from_prob_tf_yaml(_config_path())
    result = tree.lookup("base_link", "tool0", return_bingham=True, summarize=True)
    with pytest.raises(ValueError):
        tree.add_edge(result)
