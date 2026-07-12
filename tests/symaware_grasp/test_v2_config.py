from pathlib import Path

import numpy as np

from probtf.distributions import OrientationKind
from probtf.kernels import KernelEvaluator, KernelRepresentation
from probtf.temporal import TemporalPolicy
from symaware_grasp.probtf_config import load_prob_tf_config


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
