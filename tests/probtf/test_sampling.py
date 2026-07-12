import numpy as np
import pytest

from probtf.bingham import bingham_second_moment
from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    DistributionStatus,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.geometry import axis_angle_to_quat, quat_to_rotmat
from probtf.graph import ProbTfGraph
from probtf.kernels import KernelEvaluationOptions, KernelEvaluator, KernelRepresentation
from probtf.probability import (
    PointMomentSummary,
    apply_transform_samples,
    forward_component_point_moments,
    sample_bingham_orientation,
    sample_transform_component,
    sample_transform_distribution,
)
from probtf.provenance import ApproximationKind


def _component(
    component_id,
    weight=1.0,
    orientation=None,
    translation=(0.0, 0.0, 0.0),
    covariance=None,
    coupling=None,
):
    return TransformComponent(
        component_id=component_id,
        raw_weight=weight,
        orientation=(
            BinghamOrientation.dirac([1.0, 0.0, 0.0, 0.0])
            if orientation is None
            else orientation
        ),
        translation=ConditionalGaussianTranslation(
            mean_at_reference=np.asarray(translation, dtype=float),
            residual_covariance=(
                np.zeros((3, 3), dtype=float) if covariance is None else covariance
            ),
            rotation_coupling=(
                np.zeros((3, 9), dtype=float) if coupling is None else coupling
            ),
        ),
    )


def _record(component):
    return TransformDistributionStamped(
        parent_frame_id="world",
        child_frame_id="tool",
        stamp=1.0,
        edge_id="world_tool",
        authority="sampling_test",
        distribution=TransformDistribution((component,)),
    )


def test_dirac_orientation_and_transform_samples_are_exact():
    quaternion = axis_angle_to_quat([0.0, 0.0, 1.0], np.pi / 2.0)
    orientation = BinghamOrientation.dirac(quaternion)

    rotations = sample_bingham_orientation(orientation, 8, rng=3)
    np.testing.assert_allclose(rotations, np.repeat(quaternion[None, :], 8, axis=0))
    assert rotations.flags.writeable is False

    component = _component(
        "dirac",
        orientation=orientation,
        translation=[1.0, -2.0, 0.5],
    )
    samples = sample_transform_component(component, 8, rng=4)
    np.testing.assert_allclose(samples.translations, np.tile([1.0, -2.0, 0.5], (8, 1)))
    np.testing.assert_allclose(samples.rotations_wxyz, rotations)
    assert samples.count == 8


def test_finite_bingham_samples_match_second_moment():
    parameter = np.diag([0.0, -2.0, -4.0, -6.0])
    orientation = BinghamOrientation.from_parameter_matrix(parameter)

    samples = sample_bingham_orientation(orientation, 20000, rng=19)
    sample_second_moment = samples.T @ samples / float(len(samples))
    expected_second_moment = bingham_second_moment(parameter, integration_steps=120)

    np.testing.assert_allclose(np.linalg.norm(samples, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(sample_second_moment, expected_second_moment, atol=1.2e-2)


def test_mixture_samples_follow_normalized_positive_weights():
    distribution = TransformDistribution(
        (
            _component("left", weight=1.0, translation=[0.0, 0.0, 0.0]),
            _component("right", weight=3.0, translation=[4.0, 0.0, 0.0]),
        )
    )

    samples = sample_transform_distribution(distribution, 20000, rng=23)
    right_fraction = np.mean(samples.translations[:, 0] == 4.0)

    assert right_fraction == pytest.approx(0.75, abs=1.5e-2)
    np.testing.assert_allclose(samples.translations[:, 1:], 0.0)
    assert np.mean(samples.translations[:, 0]) == pytest.approx(3.0, abs=6e-2)


def test_coupled_translation_samples_match_conditional_and_marginal_moments():
    coupling = np.array(
        [
            [0.2, 0.0, 0.1, -0.3, 0.2, 0.0, 0.0, 0.1, -0.2],
            [0.0, -0.2, 0.1, 0.2, 0.1, 0.0, -0.1, 0.3, 0.0],
            [0.1, 0.0, -0.1, 0.0, 0.2, -0.2, 0.3, 0.0, 0.1],
        ]
    )
    residual_covariance = np.diag([0.03, 0.02, 0.01])
    component = _component(
        "coupled",
        orientation=BinghamOrientation.uniform(),
        translation=[0.2, -0.1, 0.3],
        covariance=residual_covariance,
        coupling=coupling,
    )

    samples = sample_transform_component(component, 30000, rng=29)
    conditional_means = np.asarray(
        [
            component.conditional_translation_mean(quaternion)
            for quaternion in samples.rotations_wxyz
        ]
    )
    residuals = samples.translations - conditional_means
    exact = forward_component_point_moments(
        component,
        PointMomentSummary(np.zeros(3), np.zeros((3, 3))),
    )

    np.testing.assert_allclose(residuals.mean(axis=0), np.zeros(3), atol=3e-3)
    np.testing.assert_allclose(
        np.cov(residuals, rowvar=False, bias=True),
        residual_covariance,
        atol=1.5e-3,
    )
    np.testing.assert_allclose(samples.translations.mean(axis=0), exact.mean, atol=1.2e-2)
    np.testing.assert_allclose(
        np.cov(samples.translations, rowvar=False, bias=True),
        exact.covariance,
        atol=1.5e-2,
    )


def test_forward_then_inverse_with_same_samples_recovers_points():
    orientation = BinghamOrientation.from_parameter_matrix(
        np.diag([0.0, -2.0, -4.0, -6.0])
    )
    component = _component(
        "stochastic",
        orientation=orientation,
        translation=[0.3, -0.2, 0.4],
        covariance=np.diag([0.01, 0.02, 0.03]),
        coupling=np.arange(27, dtype=float).reshape(3, 9) * 1e-3,
    )
    samples = sample_transform_component(component, 4000, rng=31)
    points = np.random.default_rng(37).normal(size=(samples.count, 3))

    transformed = apply_transform_samples(samples, points)
    restored = apply_transform_samples(samples, transformed, inverse=True)

    np.testing.assert_allclose(restored, points, atol=2e-12)


def test_kernel_forward_and_inverse_sample_statistics():
    quaternion = axis_angle_to_quat([0.0, 0.0, 1.0], np.pi / 2.0)
    rotation = quat_to_rotmat(quaternion)
    translation = np.array([1.0, -0.5, 0.25])
    covariance = np.diag([0.04, 0.09, 0.01])
    component = _component(
        "gaussian_translation",
        orientation=BinghamOrientation.dirac(quaternion),
        translation=translation,
        covariance=covariance,
    )
    graph = ProbTfGraph()
    graph.insert(_record(component))
    evaluator = KernelEvaluator()
    options = KernelEvaluationOptions(
        KernelRepresentation.SAMPLES,
        sample_count=30000,
        rng=41,
    )

    child_point = np.array([0.4, 0.2, -0.1])
    forward = evaluator.apply_to_point(
        graph.lookup_kernel("world", "tool", 1.0),
        child_point,
        KernelRepresentation.SAMPLES,
        options,
    )
    assert forward.status is DistributionStatus.OK
    assert forward.approximation.kind is ApproximationKind.MONTE_CARLO
    np.testing.assert_allclose(
        forward.value.mean(axis=0),
        rotation @ child_point + translation,
        atol=4e-3,
    )
    np.testing.assert_allclose(
        np.cov(forward.value, rowvar=False, bias=True),
        covariance,
        atol=3e-3,
    )

    world_point = np.array([-0.2, 0.7, 0.3])
    inverse = evaluator.apply_to_point(
        graph.lookup_kernel("tool", "world", 1.0),
        world_point,
        KernelRepresentation.SAMPLES,
        KernelEvaluationOptions(
            KernelRepresentation.SAMPLES,
            sample_count=30000,
            rng=43,
        ),
    )
    assert inverse.status is DistributionStatus.OK
    np.testing.assert_allclose(
        inverse.value.mean(axis=0),
        rotation.T @ (world_point - translation),
        atol=4e-3,
    )
    np.testing.assert_allclose(
        np.cov(inverse.value, rowvar=False, bias=True),
        rotation.T @ covariance @ rotation,
        atol=3e-3,
    )
