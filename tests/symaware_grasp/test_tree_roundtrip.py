from pathlib import Path

import numpy as np
import pytest

from probtf.distributions import OrientationKind
from probtf.kernels import KernelEvaluator, KernelRepresentation
from symaware_grasp.prob_tf.path_expression import EdgeView, PathExpression
from symaware_grasp.prob_tf.tree import ProbTfEdge, ProbTfTree
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


def test_legacy_tree_adapter_swaps_pose_query_into_core_action_semantics():
    tree = ProbTfTree(root="world")
    tree.add_edge(
        ProbTfEdge(
            edge_id="joint",
            parent="world",
            child="tool",
            translation=np.array([1.0, 2.0, 3.0]),
            joint_type="fixed",
        )
    )
    kernel = tree.lookup_core_kernel("world", "tool", stamp=0.0)
    result = KernelEvaluator().apply_to_point(
        kernel,
        np.zeros(3),
        KernelRepresentation.MOMENTS,
    )
    np.testing.assert_allclose(result.value.mean, [1.0, 2.0, 3.0])
    record = tree.to_core_graph().edge_buffer("joint").records[0]
    assert record.distribution.components[0].orientation.kind is OrientationKind.DIRAC


def test_legacy_uncertain_edge_maps_to_finite_bingham_without_closure():
    tree = ProbTfTree(root="world")
    tree.add_edge(
        ProbTfEdge(
            edge_id="joint",
            parent="world",
            child="tool",
            translation=np.zeros(3),
            joint_type="revolute",
            axis=np.array([0.0, 0.0, 1.0]),
            bingham_param=np.diag([0.0, -5.0, -10.0, -15.0]),
        )
    )
    record = tree.to_core_graph().edge_buffer("joint").records[0]
    component = record.distribution.components[0]
    assert component.orientation.kind is OrientationKind.FINITE_BINGHAM
    np.testing.assert_allclose(component.translation.rotation_coupling, np.zeros((3, 9)))
