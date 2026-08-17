import numpy as np
import pytest

from probtf.dependency import (
    DependencyAwareMomentEvaluator,
    DependencyMomentError,
    EdgeLatentBinding,
    GaussianLatentStore,
    GaussianObservationFactor,
    apply_mixed_pose_perturbation,
    inverse_mixed_pose_jacobian,
)
from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.geometry import (
    DeterministicTransform,
    axis_angle_to_quat,
    compose_transforms,
    relative_transform,
    se3_log,
)
from probtf.graph import EdgeDirection, EdgeView, PathExpression, ProbTfGraph
from probtf.kernels import (
    KernelEvaluator,
    KernelRepresentation,
    kernel_from_path,
)
from probtf.provenance import (
    ApproximationInfo,
    ApproximationKind,
    ComponentProvenance,
    Provenance,
    TransformProvenance,
)
from probtf.temporal.backends import (
    component_from_pose_covariance,
    component_pose_covariance,
)


def _record(
    edge_id,
    parent,
    child,
    translation=(0.0, 0.0, 0.0),
    rotation=(1.0, 0.0, 0.0, 0.0),
    residual_covariance=None,
    dependency_id=None,
    stamp=1.0,
):
    provenance = (
        TransformProvenance()
        if dependency_id is None
        else TransformProvenance(derived_from_edge_ids=(dependency_id,))
    )
    component = TransformComponent(
        edge_id + ":component",
        1.0,
        BinghamOrientation.dirac(rotation),
        ConditionalGaussianTranslation(
            np.asarray(translation, dtype=float),
            (
                np.zeros((3, 3), dtype=float)
                if residual_covariance is None
                else np.asarray(residual_covariance, dtype=float)
            ),
            np.zeros((3, 9), dtype=float),
        ),
    )
    return TransformDistributionStamped(
        parent,
        child,
        stamp,
        edge_id,
        "test",
        TransformDistribution((component,)),
        provenance=provenance,
    )


def _pose(record):
    return record.distribution.components[0].deterministic_transform()


def _binding(record, factor, sensitivity):
    return EdgeLatentBinding(
        record.edge_id,
        factor.factor_id,
        sensitivity,
        factor.version,
        record.stamp,
        _pose(record),
    )


def _mixed_difference(reference, value):
    return np.concatenate(
        (
            value.translation - reference.translation,
            se3_log(relative_transform(reference, value))[3:],
        )
    )


def test_latent_store_versions_snapshots_and_binding_validation():
    store = GaussianLatentStore()
    factor = store.put_factor(
        "bias",
        np.zeros(2),
        np.diag([0.2, 0.3]),
        1.0,
        Provenance(source_ids=("prior",)),
    )
    snapshot = store.snapshot()
    assert factor.version == 1
    assert snapshot.factor("bias").version == 1
    with pytest.raises(ValueError):
        snapshot.factor("bias").mean[0] = 1.0

    record = _record("edge", "world", "tool")
    store.bind_edge(_binding(record, factor, np.zeros((6, 2))))
    replacement = store.put_factor(
        "bias",
        np.ones(2),
        np.diag([0.1, 0.1]),
        2.0,
    )
    assert replacement.version == 2
    refreshed = store.snapshot().bindings_for_edge("edge")[0]
    assert refreshed.factor_version == 2
    with pytest.raises(ValueError, match="stale"):
        store.bind_edge(
            EdgeLatentBinding(
                "other",
                "bias",
                np.zeros((6, 2)),
                1,
                1.0,
                DeterministicTransform.identity(),
            )
        )
    with pytest.raises(ValueError, match="Unsupported perturbation"):
        EdgeLatentBinding(
            "other",
            "bias",
            np.zeros((6, 2)),
            2,
            1.0,
            DeterministicTransform.identity(),
            "translation_parent_rotation_left_local",
        )


def test_two_variable_observation_preserves_negative_cross_covariance():
    sigma2 = 0.4
    store = GaussianLatentStore()
    store.put_factor(
        "joint",
        np.zeros(2),
        sigma2 * np.eye(2),
        0.0,
    )
    result = store.apply_observation(
        GaussianObservationFactor(
            "sum",
            ("joint",),
            np.zeros(1),
            (np.array([[1.0, 1.0]]),),
            np.array([[1.0e-12]]),
            1.0,
        )
    )
    posterior = store.snapshot().factor("joint")
    expected = 0.5 * sigma2 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(posterior.covariance, expected, atol=2.0e-12)
    assert result.prior_versions == (("joint", 1),)
    assert result.posterior_versions == (("joint", 2),)
    assert np.array([1.0, 1.0]) @ posterior.covariance @ np.array(
        [1.0, 1.0]
    ) < 2.0e-12
    assert np.array([1.0, -1.0]) @ posterior.covariance @ np.array(
        [1.0, -1.0]
    ) == pytest.approx(2.0 * sigma2, abs=2.0e-12)


def test_multi_factor_observation_is_atomic_and_keeps_cross_block():
    store = GaussianLatentStore()
    store.put_factor("left", [0.0], [[1.0]], 0.0)
    store.put_factor("right", [0.0], [[1.0]], 0.0)
    store.apply_observation(
        GaussianObservationFactor(
            "sum",
            ("left", "right"),
            [0.0],
            (np.ones((1, 1)), np.ones((1, 1))),
            [[1.0e-8]],
            1.0,
        ),
        expected_versions={"left": 1, "right": 1},
    )
    snapshot = store.snapshot()
    _, covariance, slices = snapshot.joint_mean_covariance(("left", "right"))
    assert snapshot.factor("left").version == snapshot.factor("right").version == 2
    assert covariance[slices["left"], slices["right"]][0, 0] < -0.49
    with pytest.raises(RuntimeError, match="version changed"):
        store.apply_observation(
            GaussianObservationFactor(
                "stale",
                ("left",),
                [0.0],
                (np.ones((1, 1)),),
                [[1.0]],
                2.0,
            ),
            expected_versions={"left": 1},
        )


def test_repeated_dependency_fails_closed_until_every_binding_is_present():
    store = GaussianLatentStore()
    factor = store.put_factor("shared", [0.0], [[0.04]], 1.0)
    first = _record(
        "world_a",
        "world",
        "a",
        translation=(1.0, 0.0, 0.0),
        dependency_id="shared",
    )
    second = _record(
        "a_tool",
        "a",
        "tool",
        translation=(0.0, 1.0, 0.0),
        dependency_id="shared",
    )
    graph = ProbTfGraph(latent_store=store)
    graph.insert(first)
    graph.insert(second)
    sensitivity = np.zeros((6, 1))
    sensitivity[0, 0] = 1.0
    store.bind_edge(_binding(first, factor, sensitivity))

    with pytest.raises(DependencyMomentError) as error:
        graph.lookup_transform_moments("world", "tool", 1.0)
    assert error.value.code == "DEPENDENCY_UNRESOLVED"

    store.bind_edge(_binding(second, factor, sensitivity))
    result = graph.lookup_transform_moments("world", "tool", 1.0)
    np.testing.assert_allclose(result.mean.translation, [1.0, 1.0, 0.0])
    assert result.covariance[0, 0] == pytest.approx(0.16)
    assert result.factor_versions == (("shared", 1),)
    assert result.approximation.kind is ApproximationKind.MOMENT_SUMMARY
    assert result.approximation.lossy
    assert (
        result.approximation.source
        == "probtf.dependency.DependencyAwareMomentEvaluator"
    )
    assert result.provenance.method == "dependency_aware_local_gaussian_moments"
    assert result.provenance.detail == result.perturbation_convention
    assert result.provenance.derived_from_edge_ids == result.edge_ids
    assert result.diagnostics == ("resolved repeated dependencies: shared",)

    kernel_result = KernelEvaluator(latent_store=store).apply_to_point(
        graph.lookup_kernel("world", "tool", 1.0),
        [0.0, 0.0, 0.0],
        KernelRepresentation.MOMENTS,
    )
    assert kernel_result.status.value == "ok"
    assert kernel_result.value.covariance[0, 0] == pytest.approx(0.16)


def test_forward_inverse_same_edge_cancels_shared_latent_exactly():
    store = GaussianLatentStore()
    factor = store.put_factor("edge", [0.2], [[0.3]], 1.0)
    record = _record(
        "edge",
        "world",
        "tool",
        translation=(0.4, -0.2, 0.1),
        rotation=axis_angle_to_quat([0.0, 0.0, 1.0], 0.3),
    )
    sensitivity = np.zeros((6, 1))
    sensitivity[1, 0] = 0.4
    sensitivity[5, 0] = 1.0
    store.bind_edge(_binding(record, factor, sensitivity))
    path = PathExpression(
        "tool",
        "tool",
        1.0,
        (
            EdgeView("edge", EdgeDirection.FORWARD, 1.0),
            EdgeView("edge", EdgeDirection.INVERSE, 1.0),
        ),
    )
    result = DependencyAwareMomentEvaluator().evaluate(
        kernel_from_path(path, (record, record)),
        store,
    )
    np.testing.assert_allclose(result.mean.translation, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(
        result.mean.rotation_wxyz,
        [1.0, 0.0, 0.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(result.covariance, np.zeros((6, 6)), atol=1e-12)


def test_inverse_mixed_jacobian_matches_finite_difference():
    transform = DeterministicTransform(
        np.array([0.4, -0.3, 0.2]),
        axis_angle_to_quat([1.0, 2.0, -1.0], 0.45),
    )
    inverse = transform.inverse()
    analytic = inverse_mixed_pose_jacobian(transform)
    numerical = np.zeros((6, 6))
    step = 1.0e-7
    for column in range(6):
        perturbation = np.zeros(6)
        perturbation[column] = step
        changed = apply_mixed_pose_perturbation(
            transform,
            perturbation,
        ).inverse()
        numerical[:, column] = _mixed_difference(inverse, changed) / step
    np.testing.assert_allclose(analytic, numerical, atol=8.0e-8)


def test_forward_inverse_sibling_path_matches_monte_carlo_oracle():
    store = GaussianLatentStore()
    variance = 2.5e-6
    factor = store.put_factor("shared", [0.0], [[variance]], 1.0)
    left = _record(
        "root_left",
        "root",
        "left",
        translation=(0.3, 0.1, -0.2),
        rotation=axis_angle_to_quat([0.0, 0.0, 1.0], 0.2),
        dependency_id="shared",
    )
    right = _record(
        "root_right",
        "root",
        "right",
        translation=(-0.2, 0.4, 0.1),
        rotation=axis_angle_to_quat([1.0, 0.0, 0.0], -0.25),
        dependency_id="shared",
    )
    left_sensitivity = np.zeros((6, 1))
    left_sensitivity[[0, 5], 0] = [0.7, 1.0]
    right_sensitivity = np.zeros((6, 1))
    right_sensitivity[[1, 3], 0] = [-0.4, 0.6]
    store.bind_edge(_binding(left, factor, left_sensitivity))
    store.bind_edge(_binding(right, factor, right_sensitivity))
    graph = ProbTfGraph(latent_store=store)
    graph.insert(left)
    graph.insert(right)
    result = graph.lookup_transform_moments("right", "left", 1.0)

    rng = np.random.default_rng(31)
    samples = rng.normal(scale=np.sqrt(variance), size=8000)
    nominal = compose_transforms(_pose(right).inverse(), _pose(left))
    deltas = np.empty((samples.size, 6))
    for index, value in enumerate(samples):
        left_sample = apply_mixed_pose_perturbation(
            _pose(left),
            left_sensitivity[:, 0] * value,
        )
        right_sample = apply_mixed_pose_perturbation(
            _pose(right),
            right_sensitivity[:, 0] * value,
        )
        sample = compose_transforms(right_sample.inverse(), left_sample)
        deltas[index] = _mixed_difference(nominal, sample)
    empirical = np.cov(deltas, rowvar=False)
    np.testing.assert_allclose(
        result.covariance,
        empirical,
        rtol=0.05,
        atol=2.0e-8,
    )


def test_independent_translation_regression_matches_current_point_moments():
    residual = np.diag([0.04, 0.03, 0.02])
    record = _record(
        "edge",
        "world",
        "tool",
        translation=(0.2, -0.1, 0.4),
        residual_covariance=residual,
    )
    graph = ProbTfGraph()
    graph.insert(record)
    kernel = graph.lookup_kernel("world", "tool", 1.0)
    current = KernelEvaluator().apply_to_point(
        kernel,
        [0.3, 0.0, -0.2],
        KernelRepresentation.MOMENTS,
    )
    local = graph.lookup_transform_moments(
        "world",
        "tool",
        1.0,
    ).apply_to_point([0.3, 0.0, -0.2])
    np.testing.assert_allclose(local.mean, current.value.mean, atol=1e-12)
    np.testing.assert_allclose(
        local.covariance,
        current.value.covariance,
        atol=1e-12,
    )


def test_native_component_covariance_round_trip():
    transform = DeterministicTransform(
        np.array([0.1, -0.2, 0.3]),
        axis_angle_to_quat([1.0, -1.0, 2.0], 0.2),
    )
    covariance = np.diag([0.01, 0.02, 0.03, 0.0004, 0.0005, 0.0006])
    covariance[0, 4] = covariance[4, 0] = 0.0002
    component = component_from_pose_covariance(
        component_id="round_trip",
        raw_weight=1.0,
        transform=transform,
        covariance=covariance,
        provenance=ComponentProvenance(method="test"),
        approximation=ApproximationInfo(),
    )
    np.testing.assert_allclose(
        component_pose_covariance(component),
        covariance,
        atol=2.0e-9,
    )


def test_six_joint_chain_matches_direct_j_sigma_j_transpose():
    store = GaussianLatentStore()
    rng = np.random.default_rng(7)
    root = rng.normal(size=(6, 6))
    covariance = 2.0e-5 * (root @ root.T) / 6.0
    factor = store.put_factor("joints", np.zeros(6), covariance, 1.0)
    graph = ProbTfGraph(latent_store=store)
    records = []
    sensitivities = []
    parent = "world"
    for index in range(6):
        child = "joint{}".format(index + 1)
        record = _record(
            "edge{}".format(index + 1),
            parent,
            child,
            translation=(0.12, 0.01 * index, 0.0),
            rotation=axis_angle_to_quat([0.0, 0.0, 1.0], 0.04 * index),
            dependency_id="joints",
        )
        sensitivity = np.zeros((6, 6))
        sensitivity[5, index] = 1.0
        store.bind_edge(_binding(record, factor, sensitivity))
        graph.insert(record)
        records.append(record)
        sensitivities.append(sensitivity)
        parent = child
    result = graph.lookup_transform_moments("world", parent, 1.0)

    directed_nominal = tuple(_pose(record) for record in reversed(records))
    nominal = DeterministicTransform.identity()
    for transform in directed_nominal:
        nominal = compose_transforms(transform, nominal)
    numerical = np.zeros((6, 6))
    step = 1.0e-7
    for column in range(6):
        changed = []
        for record, sensitivity in zip(reversed(records), reversed(sensitivities)):
            changed.append(
                apply_mixed_pose_perturbation(
                    _pose(record),
                    sensitivity[:, column] * step,
                )
            )
        output = DeterministicTransform.identity()
        for transform in changed:
            output = compose_transforms(transform, output)
        numerical[:, column] = _mixed_difference(nominal, output) / step
    np.testing.assert_allclose(
        result.covariance,
        numerical @ covariance @ numerical.T,
        rtol=2.0e-6,
        atol=2.0e-11,
    )


def test_observation_update_is_visible_and_invalidates_query_cache():
    store = GaussianLatentStore()
    factor = store.put_factor("joint", np.zeros(2), np.eye(2), 1.0)
    first = _record(
        "first",
        "world",
        "a",
        dependency_id="joint",
    )
    second = _record(
        "second",
        "a",
        "tool",
        dependency_id="joint",
    )
    first_sensitivity = np.zeros((6, 2))
    first_sensitivity[0, 0] = 1.0
    second_sensitivity = np.zeros((6, 2))
    second_sensitivity[1, 1] = 1.0
    store.bind_edge(_binding(first, factor, first_sensitivity))
    store.bind_edge(_binding(second, factor, second_sensitivity))
    graph = ProbTfGraph(latent_store=store)
    graph.insert(first)
    graph.insert(second)

    before = graph.lookup_transform_moments("world", "tool", 1.0)
    again = graph.lookup_transform_moments("world", "tool", 1.0)
    assert again is before
    assert graph._dependency_moment_evaluator.cache_hits == 1
    store.apply_observation(
        GaussianObservationFactor(
            "camera_x",
            ("joint",),
            [0.5],
            (np.array([[1.0, 0.0]]),),
            [[0.01]],
            2.0,
        )
    )
    after_first = graph.lookup_transform_moments("world", "tool", 1.0)
    assert after_first.factor_versions == (("joint", 2),)
    assert after_first.covariance[0, 0] < 0.02 * before.covariance[0, 0]
    assert after_first.mean.translation[0] < -0.49
    store.apply_observation(
        GaussianObservationFactor(
            "camera_y",
            ("joint",),
            [0.0],
            (np.array([[0.0, 1.0]]),),
            [[0.01]],
            3.0,
        )
    )
    after_second = graph.lookup_transform_moments("world", "tool", 1.0)
    assert after_second.covariance[1, 1] < 0.02 * before.covariance[1, 1]
    assert graph._dependency_moment_evaluator.cache_misses == 3
