import numpy as np
import pytest

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.geometry import DeterministicTransform, se3_exp
from probtf.graph import (
    EdgeDirection,
    EdgeView,
    GraphErrorCode,
    PathExpression,
    ProbTfGraph,
    TemporalResolutionError,
)
from probtf.kernels import (
    KernelEvaluationOptions,
    KernelEvaluator,
    KernelRepresentation,
    kernel_from_path,
)
from probtf.temporal import (
    ConstantBodyAccelerationModel,
    ConstantBodyTwistModel,
    TemporalDiagnosticCode,
    TemporalEvaluationKind,
    TemporalPolicy,
    TemporalQueryMode,
    TemporalUncertaintyBackend,
)


def _record(
    edge_id,
    parent,
    child,
    stamp,
    transform=None,
    *,
    authority="authority",
    covariance=None,
    is_static=False,
):
    transform = DeterministicTransform.identity() if transform is None else transform
    covariance = np.zeros((3, 3)) if covariance is None else covariance
    return TransformDistributionStamped(
        parent,
        child,
        float(stamp),
        edge_id,
        authority,
        TransformDistribution(
            (
                TransformComponent(
                    "component",
                    1.0,
                    BinghamOrientation.dirac(transform.rotation_wxyz),
                    ConditionalGaussianTranslation(
                        transform.translation,
                        covariance,
                        np.zeros((3, 9)),
                    ),
                ),
            )
        ),
        is_static=is_static,
    )


def _twist_model(model_id="twist", maximum_horizon=2.0, process_noise=None):
    return ConstantBodyTwistModel(
        np.zeros((6, 6)) if process_noise is None else process_noise,
        maximum_horizon,
        model_id=model_id,
    )


def _insert_motion_history(graph, edge_id="edge", parent="world", child="tool"):
    graph.insert(
        _record(
            edge_id,
            parent,
            child,
            0.0,
            DeterministicTransform.identity(),
        )
    )
    graph.insert(
        _record(
            edge_id,
            parent,
            child,
            1.0,
            se3_exp(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.2])),
        )
    )


def test_models_are_bound_by_edge_authority_and_named_override_is_provenance():
    graph = ProbTfGraph()
    _insert_motion_history(graph)
    twist = _twist_model("twist")
    acceleration = ConstantBodyAccelerationModel(
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "imu",
        "tool",
        np.zeros((6, 6)),
        2.0,
        model_id="acceleration",
    )
    graph.register_temporal_model("edge", "authority", twist)
    graph.register_temporal_model(
        "edge",
        "authority",
        acceleration,
        make_default=False,
    )

    default_path = graph.lookup_path(
        "world",
        "tool",
        1.5,
        TemporalPolicy.PREDICT_WITH_MODEL,
        max_age=1.0,
        max_prediction_horizon=1.0,
    )
    assert default_path.edge_evaluations[0].model_id == "twist"
    override_path = graph.lookup_path(
        "world",
        "tool",
        1.5,
        TemporalPolicy.PREDICT_WITH_MODEL,
        max_age=1.0,
        max_prediction_horizon=1.0,
        model_id="acceleration",
    )
    assert override_path.edge_evaluations[0].model_id == "acceleration"
    assert (
        override_path.edge_evaluations[0].record.representative.translation[0]
        > default_path.edge_evaluations[0].record.representative.translation[0]
    )
    with pytest.raises(TemporalResolutionError) as error:
        graph.lookup_path(
            "world",
            "tool",
            1.5,
            TemporalPolicy.PREDICT_WITH_MODEL,
            max_age=1.0,
            max_prediction_horizon=1.0,
            model_id="missing",
        )
    assert error.value.code is GraphErrorCode.MODEL_NOT_REGISTERED


def test_multiple_nondefault_models_require_an_explicit_selector():
    graph = ProbTfGraph()
    _insert_motion_history(graph)
    graph.register_temporal_model(
        "edge",
        "authority",
        _twist_model("first"),
        make_default=False,
    )
    graph.register_temporal_model(
        "edge",
        "authority",
        _twist_model("second"),
        make_default=False,
    )
    with pytest.raises(TemporalResolutionError) as error:
        graph.lookup_path(
            "world",
            "tool",
            1.5,
            TemporalPolicy.PREDICT_WITH_MODEL,
            max_age=1.0,
            max_prediction_horizon=1.0,
        )
    assert error.value.code is GraphErrorCode.MODEL_AMBIGUOUS


def test_interpolation_requires_strict_bracket_and_offline_smoothing_mode():
    graph = ProbTfGraph()
    _insert_motion_history(graph)
    graph.register_temporal_model("edge", "authority", _twist_model())

    exact = graph.lookup_path(
        "world",
        "tool",
        1.0,
        TemporalPolicy.INTERPOLATE_WITH_MODEL,
    )
    assert exact.edge_evaluations[0].evaluation_kind is TemporalEvaluationKind.SAMPLE_SELECTION
    assert exact.edge_evaluations[0].model_id == "sample_selection"

    with pytest.raises(TemporalResolutionError) as error:
        graph.lookup_path(
            "world",
            "tool",
            0.5,
            TemporalPolicy.INTERPOLATE_WITH_MODEL,
        )
    assert error.value.code is GraphErrorCode.NON_CAUSAL_INPUT_REJECTED

    path = graph.lookup_path(
        "world",
        "tool",
        0.5,
        TemporalPolicy.INTERPOLATE_WITH_MODEL,
        query_mode=TemporalQueryMode.OFFLINE_SMOOTHING,
    )
    evaluation = path.edge_evaluations[0]
    assert evaluation.evaluation_kind is TemporalEvaluationKind.MODEL_INTERPOLATION
    assert evaluation.source_stamps == (0.0, 1.0)
    assert path.edge_views[0].sample_stamp == 0.5

    with pytest.raises(TemporalResolutionError) as error:
        graph.lookup_path(
            "world",
            "tool",
            2.0,
            TemporalPolicy.INTERPOLATE_WITH_MODEL,
            query_mode=TemporalQueryMode.OFFLINE_SMOOTHING,
        )
    assert error.value.code is GraphErrorCode.MODEL_SUPPORT_EXCEEDED


def test_prediction_requires_bounded_fresh_history_and_model_support():
    graph = ProbTfGraph()
    _insert_motion_history(graph)
    graph.register_temporal_model(
        "edge",
        "authority",
        _twist_model(maximum_horizon=0.6),
    )
    with pytest.raises(ValueError, match="max_prediction_horizon"):
        graph.lookup_path(
            "world",
            "tool",
            1.2,
            TemporalPolicy.PREDICT_WITH_MODEL,
            max_age=1.0,
        )
    with pytest.raises(ValueError, match="max_age"):
        graph.lookup_path(
            "world",
            "tool",
            1.2,
            TemporalPolicy.PREDICT_WITH_MODEL,
            max_prediction_horizon=1.0,
        )
    with pytest.raises(TemporalResolutionError) as error:
        graph.lookup_path(
            "world",
            "tool",
            1.7,
            TemporalPolicy.PREDICT_WITH_MODEL,
            max_age=2.0,
            max_prediction_horizon=0.5,
        )
    assert error.value.code is GraphErrorCode.PREDICTION_HORIZON_EXCEEDED
    with pytest.raises(TemporalResolutionError) as error:
        graph.lookup_path(
            "world",
            "tool",
            1.7,
            TemporalPolicy.PREDICT_WITH_MODEL,
            max_age=2.0,
            max_prediction_horizon=1.0,
        )
    assert error.value.code is GraphErrorCode.MODEL_SUPPORT_EXCEEDED
    with pytest.raises(TemporalResolutionError) as error:
        graph.lookup_path(
            "world",
            "tool",
            1.4,
            TemporalPolicy.PREDICT_WITH_MODEL,
            max_age=0.2,
            max_prediction_horizon=1.0,
        )
    assert error.value.code is GraphErrorCode.TEMPORAL_STALE

    insufficient = ProbTfGraph()
    insufficient.insert(_record("edge", "world", "tool", 1.0))
    insufficient.register_temporal_model("edge", "authority", _twist_model())
    with pytest.raises(TemporalResolutionError) as error:
        insufficient.lookup_path(
            "world",
            "tool",
            1.2,
            TemporalPolicy.PREDICT_WITH_MODEL,
            max_age=1.0,
            max_prediction_horizon=1.0,
        )
    assert error.value.code is GraphErrorCode.INSUFFICIENT_HISTORY


def test_online_prediction_is_unchanged_by_future_record_insertion():
    graph = ProbTfGraph()
    _insert_motion_history(graph)
    graph.register_temporal_model("edge", "authority", _twist_model())
    before = graph.lookup_path(
        "world",
        "tool",
        2.0,
        TemporalPolicy.PREDICT_WITH_MODEL,
        max_age=2.0,
        max_prediction_horizon=2.0,
    )
    graph.insert(
        _record(
            "edge",
            "world",
            "tool",
            3.0,
            se3_exp(np.array([-100.0, 2.0, 1.0, 0.5, 0.2, -0.8])),
        )
    )
    after = graph.lookup_path(
        "world",
        "tool",
        2.0,
        TemporalPolicy.PREDICT_WITH_MODEL,
        max_age=2.0,
        max_prediction_horizon=2.0,
    )
    left = before._record_snapshot[0].distribution.deterministic_transform()
    right = after._record_snapshot[0].distribution.deterministic_transform()
    np.testing.assert_array_equal(left.translation, right.translation)
    np.testing.assert_array_equal(left.rotation_wxyz, right.rotation_wxyz)
    assert max(after.edge_evaluations[0].source_stamps) <= 2.0


def test_static_uncertain_edge_is_retimed_without_process_noise_or_model():
    graph = ProbTfGraph()
    record = _record(
        "static",
        "world",
        "sensor",
        0.0,
        covariance=np.diag([0.1, 0.2, 0.3]),
        is_static=True,
    )
    graph.insert(record)
    path = graph.lookup_path(
        "world",
        "sensor",
        100.0,
        TemporalPolicy.PREDICT_WITH_MODEL,
    )
    evaluated = path._record_snapshot[0]
    evaluation = path.edge_evaluations[0]
    assert evaluated.stamp == 100.0
    assert evaluated.distribution is record.distribution
    assert evaluation.evaluation_kind is TemporalEvaluationKind.STATIC
    assert evaluation.horizon == 0.0
    assert evaluation.uncertainty_increase == pytest.approx(0.0)
    assert TemporalDiagnosticCode.STATIC_EDGE in evaluation.diagnostics


def test_unknown_uniform_uncertainty_is_infinite_and_never_passes_a_limit():
    component = TransformComponent(
        "uniform",
        1.0,
        BinghamOrientation.uniform(),
        ConditionalGaussianTranslation(
            np.zeros(3),
            np.zeros((3, 3)),
            np.zeros((3, 9)),
        ),
    )
    record = TransformDistributionStamped(
        "world",
        "sensor",
        0.0,
        "static",
        "authority",
        TransformDistribution((component,)),
        is_static=True,
    )
    graph = ProbTfGraph()
    graph.insert(record)
    with pytest.raises(TemporalResolutionError) as error:
        graph.lookup_path(
            "world",
            "sensor",
            10.0,
            TemporalPolicy.PREDICT_WITH_MODEL,
            max_uncertainty_trace=100.0,
        )
    assert error.value.code is GraphErrorCode.UNCERTAINTY_LIMIT_EXCEEDED

    degraded = graph.lookup_path(
        "world",
        "sensor",
        10.0,
        TemporalPolicy.PREDICT_WITH_MODEL,
        max_uncertainty_trace=100.0,
        allow_degraded=True,
    ).edge_evaluations[0]
    assert degraded.result_uncertainty_trace == float("inf")
    assert degraded.initial_uncertainty_trace == float("inf")
    assert degraded.uncertainty_increase == 0.0
    assert TemporalDiagnosticCode.UNCERTAINTY_LIMIT_EXCEEDED in degraded.diagnostics


def test_uncertainty_limit_rejects_or_returns_explicit_degraded_result():
    graph = ProbTfGraph()
    _insert_motion_history(graph)
    graph.register_temporal_model(
        "edge",
        "authority",
        _twist_model(process_noise=np.diag([0.2] * 6)),
    )
    arguments = dict(
        target_frame="world",
        source_frame="tool",
        stamp=1.5,
        policy=TemporalPolicy.PREDICT_WITH_MODEL,
        max_age=1.0,
        max_prediction_horizon=1.0,
        max_uncertainty_trace=0.01,
    )
    with pytest.raises(TemporalResolutionError) as error:
        graph.lookup_path(**arguments)
    assert error.value.code is GraphErrorCode.UNCERTAINTY_LIMIT_EXCEEDED

    degraded = graph.lookup_path(**arguments, allow_degraded=True)
    evaluation = degraded.edge_evaluations[0]
    assert TemporalDiagnosticCode.UNCERTAINTY_LIMIT_EXCEEDED in evaluation.diagnostics
    assert "degraded" in evaluation.warnings[-1]


def test_latest_common_can_use_one_unique_anchor_for_model_evaluation():
    graph = ProbTfGraph()
    for stamp in (0.0, 1.0):
        graph.insert(
            _record(
                "world_a",
                "world",
                "a",
                stamp,
                DeterministicTransform([stamp, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            )
        )
    for stamp in (0.0, 0.5, 1.5):
        graph.insert(
            _record(
                "a_tool",
                "a",
                "tool",
                stamp,
                DeterministicTransform([0.0, stamp, 0.0], [1.0, 0.0, 0.0, 0.0]),
            )
        )
    graph.register_temporal_model("world_a", "authority", _twist_model("world_model"))
    graph.register_temporal_model("a_tool", "authority", _twist_model("tool_model"))
    path = graph.lookup_path(
        "world",
        "tool",
        policy=TemporalPolicy.LATEST_COMMON,
        latest_common_model_policy=TemporalPolicy.PREDICT_WITH_MODEL,
        max_age=2.0,
        max_prediction_horizon=2.0,
    )
    assert path.resolved_stamp == 1.0
    assert [view.sample_stamp for view in path.edge_views] == [1.0, 1.0]
    assert [record.stamp for record in path._record_snapshot] == [1.0, 1.0]
    assert [
        evaluation.evaluation_kind for evaluation in path.edge_evaluations
    ] == [
        TemporalEvaluationKind.MODEL_PREDICTION,
        TemporalEvaluationKind.SAMPLE_SELECTION,
    ]
    assert path.edge_evaluations[0].source_stamps == (0.0, 0.5)


def test_path_keeps_immutable_record_snapshot_after_history_eviction():
    graph = ProbTfGraph(max_records_per_edge=2)
    graph.insert(_record("edge", "world", "tool", 0.0))
    graph.insert(
        _record(
            "edge",
            "world",
            "tool",
            1.0,
            DeterministicTransform([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        )
    )
    path = graph.lookup_path("world", "tool", 1.0)
    graph.insert(_record("edge", "world", "tool", 2.0))
    graph.insert(_record("edge", "world", "tool", 3.0))
    assert [item.stamp for item in graph.edge_buffer("edge").records] == [2.0, 3.0]
    snapshot = graph.resolved_records(path)
    assert snapshot[0].stamp == 1.0
    np.testing.assert_allclose(
        snapshot[0].distribution.deterministic_transform().translation,
        [1.0, 0.0, 0.0],
    )


def test_interpolation_rejects_authority_change_between_endpoints():
    graph = ProbTfGraph()
    graph.insert(_record("edge", "world", "tool", 0.0, authority="first"))
    graph.insert(_record("edge", "world", "tool", 1.0, authority="second"))
    with pytest.raises(TemporalResolutionError) as error:
        graph.lookup_path(
            "world",
            "tool",
            0.5,
            TemporalPolicy.INTERPOLATE_WITH_MODEL,
            query_mode=TemporalQueryMode.OFFLINE_SMOOTHING,
        )
    assert error.value.code is GraphErrorCode.AUTHORITY_CONFLICT


def test_sample_ids_resolve_shared_path_dependency_with_common_random_numbers():
    finite = BinghamOrientation.from_parameter_matrix(
        np.diag([0.0, -8.0, -5.0, -3.0])
    )
    start = _record("world_tool", "world", "tool", 0.0)
    anchor = _record(
        "world_tool",
        "world",
        "tool",
        1.0,
        DeterministicTransform([0.5, 0.0, 0.0], finite.mode_wxyz),
    )
    start_component = start.distribution.components[0]
    start = TransformDistributionStamped(
        start.parent_frame_id,
        start.child_frame_id,
        start.stamp,
        start.edge_id,
        start.authority,
        TransformDistribution(
            (
                TransformComponent(
                    start_component.component_id,
                    1.0,
                    finite,
                    start_component.translation,
                ),
            )
        ),
    )
    anchor_component = anchor.distribution.components[0]
    anchor = TransformDistributionStamped(
        anchor.parent_frame_id,
        anchor.child_frame_id,
        anchor.stamp,
        anchor.edge_id,
        anchor.authority,
        TransformDistribution(
            (
                TransformComponent(
                    anchor_component.component_id,
                    1.0,
                    finite,
                    anchor_component.translation,
                ),
            )
        ),
    )
    model = ConstantBodyTwistModel(
        np.diag([0.01] * 6),
        1.0,
        model_id="samples",
        backend=TemporalUncertaintyBackend.SAMPLE,
        sample_count=32,
    )
    graph = ProbTfGraph()
    graph.insert(start)
    graph.insert(anchor)
    graph.register_temporal_model("world_tool", "authority", model)
    record = graph.lookup_path(
        "world",
        "tool",
        1.2,
        TemporalPolicy.PREDICT_WITH_MODEL,
        max_age=1.0,
        max_prediction_horizon=1.0,
        random_seed=19,
        random_stream="common",
    )._record_snapshot[0]
    path = PathExpression(
        "tool",
        "tool",
        1.2,
        (
            EdgeView("world_tool", EdgeDirection.FORWARD, 1.2),
            EdgeView("world_tool", EdgeDirection.INVERSE, 1.2),
        ),
    )
    kernel = kernel_from_path(path, (record, record))
    result = KernelEvaluator().apply_to_point(
        kernel,
        [0.2, -0.1, 0.4],
        KernelRepresentation.SAMPLES,
        KernelEvaluationOptions(
            KernelRepresentation.SAMPLES,
            sample_count=80,
            rng=3,
        ),
    )
    np.testing.assert_allclose(
        result.value,
        np.tile([0.2, -0.1, 0.4], (80, 1)),
        atol=1.0e-12,
    )
