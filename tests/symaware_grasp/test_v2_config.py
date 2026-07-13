from pathlib import Path
import copy

import numpy as np
import pytest
import yaml

from probtf.distributions import OrientationKind
from probtf.graph import EdgeDirection
from probtf.kernels import KernelEvaluator, KernelRepresentation
from probtf.temporal import TemporalPolicy
from probtf_ros import ProbTfListener
from symaware_grasp.arm_kinematics import ToyArm6DOF
from symaware_grasp.probtf_config import config_from_mapping, load_prob_tf_config


def _config_path():
    return (
        Path(__file__).resolve().parents[2]
        / "ros"
        / "examples"
        / "symaware_grasp"
        / "configs"
        / "simple_six_dof_prob_tf.yaml"
    )


def test_yaml_loads_directly_as_native_v2_static_records():
    config = load_prob_tf_config(_config_path())

    assert config.root_frame == "base_link"
    assert len(config.records) == 7
    assert all(record.is_static for record in config.records)
    assert config.records[0].distribution.components[0].orientation.kind is OrientationKind.FINITE_BINGHAM
    assert config.records[-1].distribution.components[0].orientation.kind is OrientationKind.DIRAC
    assert all(
        np.array_equal(
            record.distribution.components[0].translation.rotation_coupling,
            np.zeros((3, 9)),
        )
        for record in config.records
    )


def test_native_graph_lookup_replaces_legacy_tree_path_and_propagation():
    config = load_prob_tf_config(_config_path())
    graph = config.build_graph()

    path = graph.lookup_path(
        target_frame="base_link",
        source_frame="link_3",
        stamp=0.0,
        policy=TemporalPolicy.EXACT,
    )
    assert tuple(view.edge_id for view in path.edge_views) == (
        "joint_3",
        "joint_2",
        "joint_1",
    )

    endpoint = KernelEvaluator().apply_to_point(
        graph.lookup_kernel(
            target_frame="base_link",
            source_frame="link_3",
            stamp=0.0,
            policy=TemporalPolicy.EXACT,
        ),
        np.array([0.08, 0.0, 0.0]),
        KernelRepresentation.MOMENTS,
    )
    assert endpoint.value.mean.shape == (3,)
    assert endpoint.value.covariance.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(endpoint.value.covariance) >= -1e-10)


def test_listener_provides_link_point_moments_for_every_configured_link():
    config = load_prob_tf_config(_config_path())
    listener = ProbTfListener()
    listener.receive_records(config.records)

    for frame in config.frames:
        if frame == config.root_frame:
            continue
        result = listener.lookup_point_moments(
            config.root_frame,
            frame,
            np.array([0.08, 0.0, 0.0]),
            policy=TemporalPolicy.LATEST,
        )
        assert result.value.mean.shape == (3,)
        assert result.value.covariance.shape == (3, 3)
        assert np.all(np.linalg.eigvalsh(result.value.covariance) >= -1e-10)


def test_forward_and_inverse_queries_reuse_physical_edge_ids():
    graph = load_prob_tf_config(_config_path()).build_graph()
    forward = graph.lookup_path("base_link", "link_3", stamp=0.0)
    inverse = graph.lookup_path("link_3", "base_link", stamp=0.0)

    assert tuple(view.edge_id for view in forward.edge_views) == ("joint_3", "joint_2", "joint_1")
    assert tuple(view.edge_id for view in inverse.edge_views) == ("joint_1", "joint_2", "joint_3")
    assert all(view.direction is EdgeDirection.FORWARD for view in forward.edge_views)
    assert all(view.direction is EdgeDirection.INVERSE for view in inverse.edge_views)


def test_small_uncertainty_link_and_attached_point_approach_deterministic_fk():
    mapping = yaml.safe_load(_config_path().read_text(encoding="utf-8"))
    mapping = copy.deepcopy(mapping)
    for edge in mapping["edges"]:
        if edge["type"] == "revolute":
            edge["bingham_eigenvalues"] = [30000.0, -10000.0, -10000.0, -10000.0]
    listener = ProbTfListener(config_from_mapping(mapping).build_graph())
    local_point = np.array([0.05, -0.01, 0.02])

    origin = listener.lookup_point_moments(
        "base_link",
        "tool0",
        np.zeros(3),
        policy=TemporalPolicy.LATEST,
    )
    attached = listener.lookup_point_moments(
        "base_link",
        "tool0",
        local_point,
        policy=TemporalPolicy.LATEST,
    )
    expected_position, _, _ = ToyArm6DOF().forward_kinematics(np.zeros(6))
    np.testing.assert_allclose(origin.value.mean, expected_position, atol=2e-2)
    np.testing.assert_allclose(attached.value.mean, expected_position + local_point, atol=2e-2)


def test_static_records_resolve_at_arbitrary_query_stamp():
    listener = ProbTfListener(load_prob_tf_config(_config_path()).build_graph())
    path = listener.lookup_path(
        "base_link",
        "tool0",
        stamp=123.0,
        policy=TemporalPolicy.EXACT,
    )
    assert path.resolved_stamp == 123.0
    assert all(view.sample_stamp == 0.0 for view in path.edge_views)


def test_config_rejects_frame_topology_mismatch_and_unknown_joint_type():
    mapping = yaml.safe_load(_config_path().read_text(encoding="utf-8"))
    mapping["frames"] = mapping["frames"][:-1]
    with pytest.raises(ValueError, match="frame list"):
        config_from_mapping(mapping)

    mapping = yaml.safe_load(_config_path().read_text(encoding="utf-8"))
    mapping["edges"][0]["type"] = "floating"
    with pytest.raises(ValueError, match="Unsupported"):
        config_from_mapping(mapping)
