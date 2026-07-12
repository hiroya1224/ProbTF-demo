import numpy as np
import pytest

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.graph import (
    EdgeDirection,
    EdgeTimeBuffer,
    EdgeView,
    GraphErrorCode,
    PathExpression,
    ProbTfGraph,
    TemporalResolutionError,
    TopologyError,
)
from probtf.temporal import (
    AuthorityConflictPolicy,
    ParentChangePolicy,
    TemporalPolicy,
)
from probtf.kernels import (
    ComposedTransformKernel,
    ForwardEdgeKernel,
    InverseEdgeKernel,
    MixtureTransformKernel,
)


def _distribution(translation=(0.0, 0.0, 0.0)):
    return TransformDistribution(
        (
            TransformComponent(
                "deterministic",
                1.0,
                BinghamOrientation.dirac([1.0, 0.0, 0.0, 0.0]),
                ConditionalGaussianTranslation(
                    np.asarray(translation, dtype=float),
                    np.zeros((3, 3)),
                    np.zeros((3, 9)),
                ),
            ),
        )
    )


def _record(edge_id, parent, child, stamp, authority="test", is_static=False, translation=(0, 0, 0)):
    return TransformDistributionStamped(
        parent,
        child,
        float(stamp),
        edge_id,
        authority,
        _distribution(translation),
        is_static=is_static,
    )


def test_forest_path_uses_transform_action_directions_and_same_latent_ids():
    graph = ProbTfGraph()
    graph.insert(_record("world_a", "world", "a", 1.0))
    graph.insert(_record("a_tool", "a", "tool", 1.0))
    graph.insert(_record("a_camera", "a", "camera", 1.0))

    to_world = graph.lookup_path("world", "tool", 1.0)
    assert to_world.source_frame == "tool"
    assert [(view.edge_id, view.direction) for view in to_world] == [
        ("a_tool", EdgeDirection.FORWARD),
        ("world_a", EdgeDirection.FORWARD),
    ]

    to_tool = graph.lookup_path("tool", "world", 1.0)
    assert [(view.edge_id, view.direction) for view in to_tool] == [
        ("world_a", EdgeDirection.INVERSE),
        ("a_tool", EdgeDirection.INVERSE),
    ]
    assert to_tool == to_world.reversed()

    sibling = graph.lookup_path("camera", "tool", 1.0)
    assert [(view.edge_id, view.direction) for view in sibling] == [
        ("a_tool", EdgeDirection.FORWARD),
        ("a_camera", EdgeDirection.INVERSE),
    ]


def test_identity_path_and_disconnected_forest_behavior():
    graph = ProbTfGraph()
    graph.insert(_record("world_a", "world", "a", 1.0))
    graph.insert(_record("map_b", "map", "b", 1.0))
    identity = graph.lookup_path("a", "a", 2.0)
    assert identity.edge_views == ()
    assert identity.resolved_stamp == 2.0

    with pytest.raises(TopologyError) as error:
        graph.lookup_path("b", "a", 1.0)
    assert error.value.code is GraphErrorCode.DISCONNECTED
    with pytest.raises(TopologyError) as error:
        graph.lookup_path("unknown", "a", 1.0)
    assert error.value.code is GraphErrorCode.UNKNOWN_FRAME


def test_topology_rejects_cycles_duplicate_parent_and_mismatched_edge_id():
    graph = ProbTfGraph()
    graph.insert(_record("world_a", "world", "a", 1.0))
    graph.insert(_record("a_b", "a", "b", 1.0))
    with pytest.raises(TopologyError) as error:
        graph.insert(_record("b_world", "b", "world", 1.0))
    assert error.value.code is GraphErrorCode.CYCLE

    with pytest.raises(TopologyError) as error:
        graph.insert(_record("map_a", "map", "a", 1.0))
    assert error.value.code is GraphErrorCode.MULTIPLE_PARENT

    with pytest.raises(TopologyError) as error:
        graph.insert(_record("world_a", "world", "other", 1.0))
    assert error.value.code is GraphErrorCode.DUPLICATE_EDGE


def test_parent_replacement_requires_explicit_policy_and_leaves_diagnostic():
    graph = ProbTfGraph(parent_change_policy=ParentChangePolicy.REPLACE_WITH_DIAGNOSTIC)
    graph.insert(_record("world_a", "world", "a", 1.0))
    graph.insert(_record("map_a", "map", "a", 1.0))
    assert graph.topology.diagnostics[-1].code == "PARENT_REPLACED"
    path = graph.lookup_path("map", "a", 1.0)
    assert path.edge_views[0].edge_id == "map_a"


def test_edge_buffer_out_of_order_exact_nearest_latest_and_tie_break():
    buffer = EdgeTimeBuffer()
    for stamp in (3.0, 1.0, 2.0):
        buffer.insert(_record("edge", "world", "tool", stamp))
    assert [record.stamp for record in buffer.records] == [1.0, 2.0, 3.0]
    assert buffer.resolve(2.0, TemporalPolicy.EXACT).sample_stamp == 2.0
    assert buffer.resolve(2.5, TemporalPolicy.NEAREST_WITHIN_TOLERANCE, 0.5).sample_stamp == 2.0
    assert buffer.resolve(2.9, TemporalPolicy.LATEST).sample_stamp == 2.0
    assert buffer.resolve(None, TemporalPolicy.LATEST).sample_stamp == 3.0

    with pytest.raises(TemporalResolutionError) as error:
        buffer.resolve(2.5, TemporalPolicy.NEAREST_WITHIN_TOLERANCE, 0.49)
    assert error.value.code is GraphErrorCode.TEMPORAL_OUT_OF_RANGE
    with pytest.raises(TemporalResolutionError):
        buffer.resolve(0.5, TemporalPolicy.LATEST)


def test_same_stamp_authority_conflict_policy_is_explicit():
    rejecting = EdgeTimeBuffer(conflict_policy=AuthorityConflictPolicy.REJECT)
    rejecting.insert(_record("edge", "world", "tool", 1.0, "first"))
    with pytest.raises(TemporalResolutionError) as error:
        rejecting.insert(_record("edge", "world", "tool", 1.0, "second"))
    assert error.value.code is GraphErrorCode.AUTHORITY_CONFLICT

    replacing = EdgeTimeBuffer(conflict_policy=AuthorityConflictPolicy.REPLACE)
    replacing.insert(_record("edge", "world", "tool", 1.0, "first"))
    replacing.insert(_record("edge", "world", "tool", 1.0, "second"))
    assert replacing.records[0].authority == "second"

    same_authority = EdgeTimeBuffer()
    same_authority.insert(_record("edge", "world", "tool", 1.0, "same", translation=(1, 0, 0)))
    same_authority.insert(_record("edge", "world", "tool", 1.0, "same", translation=(2, 0, 0)))
    deterministic = same_authority.records[0].distribution.deterministic_transform()
    np.testing.assert_allclose(deterministic.translation, [2.0, 0.0, 0.0])


def test_static_edge_is_time_invariant_and_identical_republish_is_idempotent():
    buffer = EdgeTimeBuffer()
    original = _record(
        "edge",
        "world",
        "tool",
        0.0,
        is_static=True,
        translation=(1.0, 0.0, 0.0),
    )
    buffer.insert(original)
    buffer.insert(
        _record(
            "edge",
            "world",
            "tool",
            10.0,
            is_static=True,
            translation=(1.0, 0.0, 0.0),
        )
    )
    assert len(buffer) == 1
    with pytest.raises(TemporalResolutionError) as error:
        buffer.insert(
            _record(
                "edge",
                "world",
                "tool",
                10.0,
                is_static=True,
                translation=(2.0, 0.0, 0.0),
            )
        )
    assert error.value.code is GraphErrorCode.STATIC_EDGE_CONFLICT
    np.testing.assert_allclose(
        buffer.resolve(100.0, TemporalPolicy.EXACT).record.distribution.deterministic_transform().translation,
        [1.0, 0.0, 0.0],
    )


def test_latest_max_age_reports_stale_sample():
    buffer = EdgeTimeBuffer()
    buffer.insert(_record("edge", "world", "tool", 1.0))
    with pytest.raises(TemporalResolutionError) as error:
        buffer.resolve(10.0, TemporalPolicy.LATEST, max_age=2.0)
    assert error.value.code is GraphErrorCode.TEMPORAL_STALE
    assert buffer.resolve(10.0, TemporalPolicy.LATEST, max_age=10.0).sample_stamp == 1.0


def test_invalid_graph_buffer_configuration_fails_before_topology_exists():
    with pytest.raises(ValueError, match="max_records"):
        ProbTfGraph(max_records_per_edge=0)


def test_cross_time_inverse_views_do_not_cancel_as_one_realization():
    path = PathExpression(
        "tool",
        "tool",
        2.0,
        (
            EdgeView("edge", EdgeDirection.FORWARD, 1.0),
            EdgeView("edge", EdgeDirection.INVERSE, 2.0),
        ),
    )
    assert len(path.reduce_adjacent_inverses()) == 2


def test_latest_common_uses_common_availability_and_actual_sample_stamps():
    graph = ProbTfGraph()
    for stamp in (1.0, 3.0):
        graph.insert(_record("world_a", "world", "a", stamp))
    for stamp in (2.0, 4.0):
        graph.insert(_record("a_tool", "a", "tool", stamp))

    path = graph.lookup_path("world", "tool", policy=TemporalPolicy.LATEST_COMMON)
    assert path.resolved_stamp == 3.0
    assert [(view.edge_id, view.sample_stamp) for view in path] == [
        ("a_tool", 2.0),
        ("world_a", 3.0),
    ]
    assert "a_tool:LATEST_COMMON_ZERO_ORDER_HOLD" in path.diagnostics


def test_latest_common_rejects_nonoverlap_and_accepts_static_uncertain_edge():
    no_overlap = ProbTfGraph()
    no_overlap.insert(_record("world_a", "world", "a", 1.0))
    no_overlap.insert(_record("a_tool", "a", "tool", 4.0))
    with pytest.raises(TemporalResolutionError) as error:
        no_overlap.lookup_path("world", "tool", policy=TemporalPolicy.LATEST_COMMON)
    assert error.value.code is GraphErrorCode.TEMPORAL_OUT_OF_RANGE

    graph = ProbTfGraph()
    graph.insert(_record("world_a", "world", "a", 0.0, is_static=True))
    graph.insert(_record("a_tool", "a", "tool", 2.0))
    graph.insert(_record("a_tool", "a", "tool", 5.0))
    path = graph.lookup_path("world", "tool", policy=TemporalPolicy.LATEST_COMMON)
    assert path.resolved_stamp == 5.0
    assert [(view.edge_id, view.sample_stamp) for view in path] == [
        ("a_tool", 5.0),
        ("world_a", 0.0),
    ]


def test_temporal_model_interfaces_fail_explicitly():
    buffer = EdgeTimeBuffer()
    buffer.insert(_record("edge", "world", "tool", 1.0))
    for policy in (TemporalPolicy.INTERPOLATE_WITH_MODEL, TemporalPolicy.PREDICT_WITH_MODEL):
        with pytest.raises(TemporalResolutionError) as error:
            buffer.resolve(1.0, policy)
        assert error.value.code is GraphErrorCode.UNSUPPORTED_TEMPORAL_POLICY


def test_lookup_kernel_is_lazy_and_preserves_direction_order():
    graph = ProbTfGraph()
    graph.insert(_record("world_a", "world", "a", 1.0))
    graph.insert(_record("a_tool", "a", "tool", 1.0))

    forward = graph.lookup_kernel("world", "tool", 1.0)
    assert isinstance(forward, ComposedTransformKernel)
    assert [type(kernel) for kernel in forward.kernels] == [
        ForwardEdgeKernel,
        ForwardEdgeKernel,
    ]
    inverse = graph.lookup_kernel("tool", "world", 1.0)
    assert [type(kernel) for kernel in inverse.kernels] == [
        InverseEdgeKernel,
        InverseEdgeKernel,
    ]
    assert [kernel.edge_record.edge_id for kernel in inverse.kernels] == [
        "world_a",
        "a_tool",
    ]


def test_lookup_kernel_preserves_mixture_components_without_reduction():
    finite = BinghamOrientation.from_parameter_matrix(np.diag([0.0, -1.0, -2.0, -3.0]))
    components = tuple(
        TransformComponent(
            "c{}".format(index),
            weight,
            finite,
            ConditionalGaussianTranslation(np.zeros(3), np.eye(3), np.zeros((3, 9))),
        )
        for index, weight in enumerate((2.0, 3.0))
    )
    graph = ProbTfGraph()
    graph.insert(
        TransformDistributionStamped(
            "world",
            "tool",
            1.0,
            "edge",
            "test",
            TransformDistribution(components),
        )
    )
    expression = graph.lookup_kernel("world", "tool", 1.0)
    assert isinstance(expression.kernels[0], MixtureTransformKernel)
    assert len(expression.kernels[0].edge_record.distribution.components) == 2
