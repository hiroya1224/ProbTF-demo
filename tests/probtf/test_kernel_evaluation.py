import numpy as np
import pytest

import probtf.isl as isl
from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    DistributionStatus,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.geometry import axis_angle_to_quat, rotation_vector_from_quaternion
from probtf.graph import EdgeDirection, EdgeView, PathExpression, ProbTfGraph
from probtf.kernels import (
    DiracPointLaw,
    GaussianPointLaw,
    KernelDiagnosticCode,
    KernelEvaluationOptions,
    KernelEvaluator,
    KernelRepresentation,
    UnavailableKernelValue,
    UncoupledPointActionLaw,
    kernel_from_path,
)
from probtf.probability import PointMomentSummary, forward_component_point_moments
from probtf.provenance import ApproximationKind
from probtf.provenance import ComponentProvenance
from probtf.spherical_law import (
    DiracVectorLaw,
    IslBackendUnavailableError,
    IslEvaluationOptions,
    TangentSurrogateIslBackend,
    UnavailableExactIslBackend,
    UniformDirectionLaw,
)


def _component(
    component_id="component",
    weight=1.0,
    orientation=None,
    translation=(0.0, 0.0, 0.0),
    covariance=None,
    coupling=None,
):
    return TransformComponent(
        component_id,
        weight,
        BinghamOrientation.dirac([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation,
        ConditionalGaussianTranslation(
            np.asarray(translation, dtype=float),
            np.zeros((3, 3)) if covariance is None else covariance,
            np.zeros((3, 9)) if coupling is None else coupling,
        ),
    )


def _record(edge_id, parent, child, components, stamp=1.0):
    return TransformDistributionStamped(
        parent,
        child,
        stamp,
        edge_id,
        "test",
        TransformDistribution(tuple(components)),
    )


class FailingIslBackend:
    def rotate_direction(self, *args, **kwargs):
        raise AssertionError("deterministic fast path called the ISL backend")

    def rotate_vector(self, *args, **kwargs):
        raise AssertionError("deterministic fast path called the ISL backend")


def test_deterministic_forward_inverse_composition_bypasses_isl_backend():
    graph = ProbTfGraph()
    graph.insert(
        _record(
            "world_a",
            "world",
            "a",
            [
                _component(
                    orientation=BinghamOrientation.dirac(
                        axis_angle_to_quat([0.0, 0.0, 1.0], np.pi / 2.0)
                    ),
                    translation=[1.0, 0.0, 0.0],
                )
            ],
        )
    )
    graph.insert(_record("a_tool", "a", "tool", [_component(translation=[1.0, 0.0, 0.0])]))
    evaluator = KernelEvaluator(isl_backend=FailingIslBackend())

    forward_kernel = graph.lookup_kernel("world", "tool", 1.0)
    forward = evaluator.apply_to_point(
        forward_kernel,
        [0.0, 0.0, 0.0],
        KernelRepresentation.NUMERICAL_LAW,
    )
    assert forward.status is DistributionStatus.OK
    assert isinstance(forward.value, DiracPointLaw)
    np.testing.assert_allclose(forward.value.point, [1.0, 1.0, 0.0], atol=1e-12)

    inverse_kernel = graph.lookup_kernel("tool", "world", 1.0)
    inverse = evaluator.apply_to_point(
        inverse_kernel,
        forward.value.point,
        KernelRepresentation.MOMENTS,
    )
    np.testing.assert_allclose(inverse.value.mean, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(inverse.value.covariance, np.zeros((3, 3)), atol=1e-12)


def test_deterministic_gaussian_and_samples_are_exact():
    graph = ProbTfGraph()
    graph.insert(_record("edge", "world", "tool", [_component(translation=[1.0, 2.0, 3.0])]))
    kernel = graph.lookup_kernel("world", "tool", 1.0)
    evaluator = KernelEvaluator(isl_backend=FailingIslBackend())
    gaussian = evaluator.apply(
        kernel,
        GaussianPointLaw(np.array([1.0, 0.0, 0.0]), np.diag([1.0, 2.0, 3.0])),
        KernelEvaluationOptions(KernelRepresentation.NUMERICAL_LAW),
    )
    assert isinstance(gaussian.value, GaussianPointLaw)
    np.testing.assert_allclose(gaussian.value.mean, [2.0, 2.0, 3.0])
    np.testing.assert_allclose(gaussian.value.covariance, np.diag([1.0, 2.0, 3.0]))

    samples = evaluator.apply_to_point(
        kernel,
        [0.0, 0.0, 0.0],
        KernelRepresentation.SAMPLES,
        KernelEvaluationOptions(KernelRepresentation.SAMPLES, sample_count=4),
    )
    np.testing.assert_allclose(samples.value, np.tile([1.0, 2.0, 3.0], (4, 1)))


def test_mixture_moment_summary_uses_normalized_raw_weights():
    graph = ProbTfGraph()
    graph.insert(
        _record(
            "edge",
            "world",
            "tool",
            [
                _component("zero", 1.0, translation=[0.0, 0.0, 0.0]),
                _component("two", 3.0, translation=[2.0, 0.0, 0.0]),
            ],
        )
    )
    result = KernelEvaluator().apply_to_point(
        graph.lookup_kernel("world", "tool", 1.0),
        [0.0, 0.0, 0.0],
        KernelRepresentation.MOMENTS,
    )
    np.testing.assert_allclose(result.value.mean, [1.5, 0.0, 0.0])
    np.testing.assert_allclose(result.value.covariance, np.diag([0.75, 0.0, 0.0]))
    assert result.approximation.kind is ApproximationKind.MOMENT_SUMMARY
    assert result.approximation.lossy


def test_uniform_orientation_point_moments_are_isotropic():
    component = _component(orientation=BinghamOrientation.uniform())
    result = forward_component_point_moments(
        component,
        PointMomentSummary(np.array([2.0, 0.0, 0.0]), np.zeros((3, 3))),
    )
    np.testing.assert_allclose(result.mean, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(result.covariance, (4.0 / 3.0) * np.eye(3), atol=1e-12)


def test_uniform_coupling_moments_match_monte_carlo_cross_covariance():
    rng = np.random.default_rng(19)
    coupling = np.array(
        [
            [0.2, 0.0, 0.1, -0.3, 0.2, 0.0, 0.0, 0.1, -0.2],
            [0.0, -0.2, 0.1, 0.2, 0.1, 0.0, -0.1, 0.3, 0.0],
            [0.1, 0.0, -0.1, 0.0, 0.2, -0.2, 0.3, 0.0, 0.1],
        ]
    )
    orientation = BinghamOrientation.uniform()
    component = _component(orientation=orientation, coupling=coupling, translation=[0.2, -0.1, 0.3])
    point = np.array([0.4, -0.2, 0.8])
    exact = forward_component_point_moments(
        component,
        PointMomentSummary(point, np.zeros((3, 3))),
    )

    quaternions = rng.normal(size=(60000, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    w, x, y, z = quaternions.T
    rotations = np.empty((len(quaternions), 3, 3))
    rotations[:, 0, 0] = w * w + x * x - y * y - z * z
    rotations[:, 0, 1] = 2.0 * (x * y - w * z)
    rotations[:, 0, 2] = 2.0 * (x * z + w * y)
    rotations[:, 1, 0] = 2.0 * (x * y + w * z)
    rotations[:, 1, 1] = w * w - x * x + y * y - z * z
    rotations[:, 1, 2] = 2.0 * (y * z - w * x)
    rotations[:, 2, 0] = 2.0 * (x * z - w * y)
    rotations[:, 2, 1] = 2.0 * (y * z + w * x)
    rotations[:, 2, 2] = w * w - x * x - y * y + z * z
    rotation_vectors = rotations.transpose(0, 2, 1).reshape(-1, 9)
    reference_vector = rotation_vector_from_quaternion(
        orientation.reference_quaternion_wxyz
    )
    outputs = (
        np.einsum("nij,j->ni", rotations, point)
        + component.translation.mean_at_reference
        + (rotation_vectors - reference_vector) @ coupling.T
    )
    np.testing.assert_allclose(exact.mean, outputs.mean(axis=0), atol=1.2e-2)
    np.testing.assert_allclose(exact.covariance, np.cov(outputs, rowvar=False), atol=1.5e-2)


def test_finite_exact_isl_is_explicitly_unavailable_but_zero_vector_is_safe():
    finite = BinghamOrientation.from_parameter_matrix(np.diag([0.0, -2.0, -3.0, -4.0]))
    graph = ProbTfGraph()
    graph.insert(_record("edge", "world", "tool", [_component(orientation=finite)]))
    kernel = graph.lookup_kernel("world", "tool", 1.0)
    evaluator = KernelEvaluator(isl_backend=UnavailableExactIslBackend())

    unavailable = evaluator.apply_to_point(
        kernel,
        [1.0, 0.0, 0.0],
        KernelRepresentation.NUMERICAL_LAW,
    )
    assert unavailable.status is DistributionStatus.INVALID
    assert isinstance(unavailable.value, UnavailableKernelValue)
    assert unavailable.value.code == "UNAVAILABLE_EXACT_ISL_BACKEND"
    assert KernelDiagnosticCode.UNAVAILABLE_BACKEND in unavailable.diagnostics.codes

    zero = evaluator.apply_to_point(
        kernel,
        [0.0, 0.0, 0.0],
        KernelRepresentation.NUMERICAL_LAW,
    )
    assert zero.status is DistributionStatus.OK
    assert isinstance(zero.value, UncoupledPointActionLaw)
    assert isinstance(zero.value.induced_vector_law, DiracVectorLaw)
    np.testing.assert_allclose(zero.value.induced_vector_law.vector, np.zeros(3))


def test_uniform_direction_and_tangent_surrogate_are_typed_distinctly():
    backend = UnavailableExactIslBackend()
    uniform = backend.rotate_direction(
        BinghamOrientation.uniform(),
        np.array([1.0, 0.0, 0.0]),
        IslEvaluationOptions(),
    )
    assert isinstance(uniform, UniformDirectionLaw)
    assert uniform.density([0.0, 1.0, 0.0]) == pytest.approx(1.0 / (4.0 * np.pi))

    finite = BinghamOrientation.from_parameter_matrix(np.diag([0.0, -20.0, -30.0, -40.0]))
    tangent = TangentSurrogateIslBackend().rotate_vector(
        finite,
        np.array([1.0, 0.0, 0.0]),
        IslEvaluationOptions(),
    )
    assert tangent.approximation.kind is ApproximationKind.TANGENT_SURROGATE
    assert tangent.approximation.lossy


def test_isl_special_cases_preserve_vector_scale_and_quaternion_antipodes():
    assert isl.UnavailableExactIslBackend is UnavailableExactIslBackend
    backend = UnavailableExactIslBackend()
    quaternion = axis_angle_to_quat([0.0, 0.0, 1.0], np.pi / 2.0)
    positive = BinghamOrientation.dirac(quaternion)
    negative = BinghamOrientation.dirac(-quaternion)
    options = IslEvaluationOptions()
    law_positive = backend.rotate_vector(positive, np.array([2.0, 0.0, 0.0]), options)
    law_negative = backend.rotate_vector(negative, np.array([2.0, 0.0, 0.0]), options)
    np.testing.assert_allclose(law_positive.vector, [0.0, 2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(law_negative.vector, law_positive.vector, atol=1e-12)
    np.testing.assert_allclose(positive.shape_matrix, negative.shape_matrix, atol=1e-12)

    finite = BinghamOrientation.from_parameter_matrix(np.diag([0.0, -2.0, -3.0, -4.0]))
    with pytest.raises(IslBackendUnavailableError) as error:
        backend.rotate_vector(finite, np.array([1.0, 0.0, 0.0]), options)
    assert error.value.code == "UNAVAILABLE_EXACT_ISL_BACKEND"


def test_zero_mass_stochastic_inverse_and_closure_fail_without_fallback():
    zero_graph = ProbTfGraph()
    zero_graph.insert(_record("zero", "world", "tool", [_component(weight=0.0)]))
    zero_result = KernelEvaluator().apply_to_point(
        zero_graph.lookup_kernel("world", "tool", 1.0),
        [0.0, 0.0, 0.0],
        KernelRepresentation.MOMENTS,
    )
    assert zero_result.status is DistributionStatus.ZERO_MASS
    assert zero_result.value is None

    finite = BinghamOrientation.from_parameter_matrix(np.diag([0.0, -2.0, -3.0, -4.0]))
    graph = ProbTfGraph()
    graph.insert(_record("edge", "world", "tool", [_component(orientation=finite)]))
    inverse = KernelEvaluator().apply_to_point(
        graph.lookup_kernel("tool", "world", 1.0),
        [0.0, 0.0, 0.0],
        KernelRepresentation.MOMENTS,
    )
    assert inverse.value.code == "UNAVAILABLE_INVERSE_STOCHASTIC_MOMENTS"

    closure = KernelEvaluator().apply_to_point(
        graph.lookup_kernel("world", "tool", 1.0),
        [0.0, 0.0, 0.0],
        KernelRepresentation.CLOSED_MIXTURE,
    )
    assert closure.value.code == "UNAVAILABLE_CLOSED_MIXTURE_BACKEND"


def test_repeated_latent_edge_is_not_silently_sampled_independently():
    finite = BinghamOrientation.from_parameter_matrix(np.diag([0.0, -2.0, -3.0, -4.0]))
    record = _record(
        "edge",
        "world",
        "tool",
        [_component(orientation=finite)],
        stamp=1.0,
    )
    path = PathExpression(
        "tool",
        "tool",
        1.0,
        (
            EdgeView("edge", EdgeDirection.FORWARD, 1.0),
            EdgeView("edge", EdgeDirection.INVERSE, 1.0),
        ),
    )
    expression = kernel_from_path(path, (record, record))
    result = KernelEvaluator().apply_to_point(
        expression,
        [0.0, 0.0, 0.0],
        KernelRepresentation.SAMPLES,
        KernelEvaluationOptions(KernelRepresentation.SAMPLES, sample_count=3),
    )
    assert result.status is DistributionStatus.INVALID
    assert result.value.code == "DEPENDENCY_UNRESOLVED"
    assert result.diagnostics.repeated_dependency_ids == ("edge",)


def test_repeated_deterministic_edge_uses_the_same_known_realization():
    record = _record("edge", "world", "tool", [_component(translation=[1.0, 2.0, 3.0])])
    path = PathExpression(
        "tool",
        "tool",
        1.0,
        (
            EdgeView("edge", EdgeDirection.FORWARD, 1.0),
            EdgeView("edge", EdgeDirection.INVERSE, 1.0),
        ),
    )
    expression = kernel_from_path(path, (record, record))
    result = KernelEvaluator().apply_to_point(
        expression,
        [0.5, -0.25, 1.0],
        KernelRepresentation.MOMENTS,
    )
    assert result.status is DistributionStatus.OK
    np.testing.assert_allclose(result.value.mean, [0.5, -0.25, 1.0], atol=1e-12)


def test_shared_component_provenance_is_a_latent_dependency():
    finite = BinghamOrientation.from_parameter_matrix(np.diag([0.0, -2.0, -3.0, -4.0]))

    def dependent_component(component_id):
        return TransformComponent(
            component_id,
            1.0,
            finite,
            ConditionalGaussianTranslation(np.zeros(3), np.zeros((3, 3)), np.zeros((3, 9))),
            provenance=ComponentProvenance(derived_from_edge_ids=("shared_latent",)),
        )

    graph = ProbTfGraph()
    graph.insert(_record("world_a", "world", "a", [dependent_component("first")]))
    graph.insert(_record("a_tool", "a", "tool", [dependent_component("second")]))
    result = KernelEvaluator().apply_to_point(
        graph.lookup_kernel("world", "tool", 1.0),
        [1.0, 0.0, 0.0],
        KernelRepresentation.MOMENTS,
    )
    assert result.status is DistributionStatus.INVALID
    assert result.value.code == "DEPENDENCY_UNRESOLVED"
    assert result.diagnostics.repeated_dependency_ids == ("shared_latent",)
