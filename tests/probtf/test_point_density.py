"""Numerical density queries retain the source law and full pose coupling."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

import probtf.probability.point_density as point_density
from probtf.bingham import bingham_log_normalizer, bingham_second_moment, rotation_first_moment
from probtf.distributions import (
    BinghamOrientation, ConditionalGaussianTranslation, DistributionStatus,
    TransformComponent, TransformDistribution, TransformDistributionStamped,
)
from probtf.geometry import quat_left_matrix, quat_mul, quat_to_rotmat, rpy_to_quat
from probtf.graph import ProbTfGraph
from probtf.kernels import KernelEvaluator, KernelRepresentation
from probtf.probability import PointMomentSummary, forward_component_point_moments
from probtf.probability.point_density import ConditionalPointDensity, DensityIntegrationError
from probtf.probability.quadrature import (
    orientation_quadrature, orientation_quadrature_diagnostics,
    quaternion_rotation_matrices,
)
from probtf.provenance import ApproximationKind


def record(orientation=None, coupling=None, covariance=None):
    orientation = (BinghamOrientation.from_parameter_matrix(np.diag([-4., -3., -2., 0.]))
                   if orientation is None else orientation)
    coupling = (np.arange(27).reshape(3, 9) * .001 - .013 if coupling is None else coupling)
    covariance = np.diag([.12, .09, .15])**2 if covariance is None else covariance
    component = TransformComponent("pose", 1., orientation,
        ConditionalGaussianTranslation(np.array([.2, -.1, .4]), covariance, coupling))
    return TransformDistributionStamped("parent", "child", 2., "observed_pose", "test",
                                         TransformDistribution((component,)))


@pytest.mark.parametrize("eigenvalues, order, tolerance", [
    ([-4., -3., -2., 0.], 11, 2e-7),
    ([-400., -300., -200., 0.], 21, 1e-6),
    ([-40., -30., 0., 0.], 21, 1e-7),
    ([0., 0., 0., 0.], 9, 1e-12),
])
def test_full_sphere_quadrature_agrees_with_independent_bingham_moments(eigenvalues, order, tolerance):
    parameter = np.diag(eigenvalues)
    orientation = BinghamOrientation.from_parameter_matrix(parameter)
    quaternions, weights = orientation_quadrature(orientation, order)
    np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1., atol=1e-15)
    assert np.all(weights >= 0) and weights.sum() == pytest.approx(1., abs=2e-15)
    np.testing.assert_allclose((quaternions.T * weights) @ quaternions,
                              bingham_second_moment(parameter), rtol=0., atol=tolerance)
    again = orientation_quadrature(orientation, order)
    np.testing.assert_array_equal(quaternions, again[0])
    np.testing.assert_array_equal(weights, again[1])
    diagnostics = orientation_quadrature_diagnostics(orientation, order)
    assert diagnostics["node_count"] == 4 * order**3
    assert diagnostics["relative_normalizer_error"] < 1e-4
    assert diagnostics["error_bound"] is None


def test_dirac_quadrature_and_conditional_density_have_exact_gaussian_limit():
    orientation = BinghamOrientation.dirac(rpy_to_quat(.2, -.3, .4))
    quaternions, weights = orientation_quadrature(orientation)
    np.testing.assert_array_equal(quaternions, orientation.reference_quaternion_wxyz[None, :])
    np.testing.assert_array_equal(weights, [1.])
    source = record(orientation)
    point = np.array([.08, -.04, .06])
    law = ConditionalPointDensity(source, point)
    component = source.distribution.components[0]
    mean = component.translation.mean_at_reference + quat_to_rotmat(quaternions[0]) @ point
    locations = np.array([[.2, -.1, .4], [.3, -.2, .5]])
    expected = multivariate_normal.logpdf(locations, mean, component.translation.residual_covariance)
    np.testing.assert_allclose(law.log_density(locations), expected, atol=1e-14)
    assert law.approximation.kind is ApproximationKind.EXACT
    assert law.source_record is source


def test_quadrature_density_preserves_rotation_translation_coupling_and_point_moments():
    source = record()
    point = np.array([.08, -.04, .06])
    law = ConditionalPointDensity(source, point, order=15)
    _weight, means, weights, cholesky, _normalizer = law.terms[0]
    expected = forward_component_point_moments(source.distribution.components[0],
        PointMomentSummary(point, np.zeros((3, 3))))
    mean = weights @ means
    residuals = means - mean
    covariance = np.einsum("n,ni,nj->ij", weights, residuals, residuals) + cholesky @ cholesky.T
    np.testing.assert_allclose(mean, expected.mean, atol=2e-8)
    np.testing.assert_allclose(covariance, expected.covariance, atol=2e-8)
    assert law.source_record is source
    np.testing.assert_array_equal(source.distribution.components[0].translation.rotation_coupling,
                                  np.arange(27).reshape(3, 9) * .001 - .013)
    assert law.approximation.kind is ApproximationKind.NUMERICAL_INTEGRATION
    assert law.approximation.error_bound is None


def test_density_gradient_matches_finite_difference_and_actual_refinement_is_reported():
    source = record()
    point = [.08, -.04, .06]
    locations = np.array([[.2, -.1, .4], [.3, -.2, .5]])
    law = ConditionalPointDensity(source, point, order=11)
    _log_density, gradient = law.log_density_and_gradient(locations)
    epsilon = 1e-6
    for axis in range(3):
        shift = np.eye(3)[axis] * epsilon
        expected = (law.log_density(locations + shift) - law.log_density(locations - shift)) / (2*epsilon)
        # Both sides include numerical integration and numerical normalizers.
        np.testing.assert_allclose(gradient[:, axis], expected, rtol=1e-6, atol=1e-6)
    diagnostics = law.refinement_diagnostics(locations, 15)
    refined = ConditionalPointDensity(source, point, 15)
    assert diagnostics["maximum_log_density_change"] == pytest.approx(
        np.max(np.abs(refined.log_density(locations) - law.log_density(locations))))
    assert diagnostics["maximum_log_density_change"] < 1e-5
    assert diagnostics["error_bound"] is None
    with pytest.raises(DensityIntegrationError) as failure:
        law.refinement_diagnostics(locations, 15, log_tolerance=0.)
    assert failure.value.diagnostics["maximum_log_density_change"] > 0.


@pytest.mark.parametrize("eigenvalues", [[-4., -3., -2., 0.], [-5., -5., 0., 0.], [0., 0., 0., 0.]])
def test_finite_rule_and_density_are_equivariant_under_left_transform(eigenvalues):
    reference = rpy_to_quat(.1, -.2, .3)
    orientation = BinghamOrientation.from_parameter_matrix(np.diag(eigenvalues), reference)
    source = record(orientation)
    fixed_q = rpy_to_quat(-.4, .3, -.2)
    fixed_rotation = quat_to_rotmat(fixed_q)
    fixed_translation = np.array([.1, .3, -.2])
    left = quat_left_matrix(fixed_q)
    rotated_orientation = BinghamOrientation.from_parameter_matrix(
        left @ orientation.parameter_matrix() @ left.T, quat_mul(fixed_q, reference))
    component = source.distribution.components[0]
    translation = component.translation
    rotated_translation = ConditionalGaussianTranslation(
        fixed_rotation @ translation.mean_at_reference + fixed_translation,
        fixed_rotation @ translation.residual_covariance @ fixed_rotation.T,
        fixed_rotation @ translation.rotation_coupling @ np.kron(np.eye(3), fixed_rotation.T))
    rotated_source = replace(source, distribution=TransformDistribution((
        replace(component, orientation=rotated_orientation, translation=rotated_translation),)))
    locations = np.array([[.2, -.1, .4], [.3, -.2, .5]])
    law = ConditionalPointDensity(source, [.08, -.04, .06], 9)
    rotated_law = ConditionalPointDensity(rotated_source, [.08, -.04, .06], 9)
    values, gradients = law.log_density_and_gradient(locations)
    rotated_values, rotated_gradients = rotated_law.log_density_and_gradient(
        locations @ fixed_rotation.T + fixed_translation)
    np.testing.assert_allclose(rotated_values, values, atol=2e-12)
    np.testing.assert_allclose(rotated_gradients, gradients @ fixed_rotation.T, atol=5e-12)


def test_two_pose_modes_are_integrated_as_a_mixture_instead_of_collapsed():
    first = record(BinghamOrientation.dirac([1., 0., 0., 0.]))
    component = first.distribution.components[0]
    second = replace(component, component_id="other", raw_weight=3.,
        orientation=BinghamOrientation.dirac(rpy_to_quat(0., 0., np.pi)))
    source = replace(first, distribution=TransformDistribution((component, second)))
    point = np.array([.3, 0., 0.])
    locations = np.array([[.5, -.1, .4], [-.1, -.1, .4]])
    law = ConditionalPointDensity(source, point)
    expected = []
    for item, weight in ((component, .25), (second, .75)):
        mean = (quat_to_rotmat(item.orientation.reference_quaternion_wxyz) @ point
                + item.conditional_translation_mean(item.orientation.reference_quaternion_wxyz))
        expected.append(np.log(weight) + multivariate_normal.logpdf(
            locations, mean, item.translation.residual_covariance))
    np.testing.assert_allclose(law.log_density(locations), logsumexp(expected, axis=0), atol=1e-14)


def test_high_concentration_tends_to_the_conditional_gaussian_without_substituting_it():
    orientation = BinghamOrientation.from_parameter_matrix(np.diag([0., -2e8, -3e8, -4e8]))
    source = record(orientation)
    point = [.08, -.04, .06]
    locations = np.array([[.2, -.1, .4], [.3, -.2, .5]])
    finite = ConditionalPointDensity(source, point, 15)
    deterministic = ConditionalPointDensity(record(BinghamOrientation.dirac([1., 0., 0., 0.])), point)
    np.testing.assert_allclose(finite.log_density(locations), deterministic.log_density(locations), atol=2e-7)
    assert finite.source_record.distribution.components[0].orientation is orientation
    assert finite.approximation.kind is ApproximationKind.NUMERICAL_INTEGRATION


def test_kernel_numerical_point_query_retains_original_coupled_law():
    source = record()
    graph = ProbTfGraph()
    graph.insert(source)
    kernel = graph.lookup_kernel("parent", "child", 2.)
    result = KernelEvaluator(point_density_order=11).apply_to_point(
        kernel, [.08, -.04, .06], KernelRepresentation.NUMERICAL_LAW)
    assert result.status is DistributionStatus.OK
    assert isinstance(result.value, ConditionalPointDensity)
    assert result.value.source_record is source
    assert result.approximation.kind is ApproximationKind.NUMERICAL_INTEGRATION


def test_singular_residual_is_rejected_without_artificial_variance():
    with pytest.raises(ValueError, match="positive-definite"):
        ConditionalPointDensity(record(covariance=np.zeros((3, 3))))
    for order in (0, 1, True, 3.5):
        with pytest.raises(ValueError, match="order"):
            orientation_quadrature(BinghamOrientation.uniform(), order)
    with pytest.raises(ValueError, match="final dimension"):
        ConditionalPointDensity(record()).log_density([np.nan, 0., 0.])


def test_narrow_gaussian_tail_matches_analytic_bingham_normalizer_ratio():
    """A sharp off-mode kernel has an independent closed-form integral.

    Isotropic noise and a rigid lever arm make |x-R(q)z|^2 quadratic in q;
    its exact integral is a ratio of Bingham normalizers. This catches prior-
    centered rules that have accurate moments but entirely miss density tails.
    """
    parameter = np.diag([0., -2000., -4000., -6000.])
    sigma = .0005
    point = np.array([.325, 0., 0.])
    orientation = BinghamOrientation.from_parameter_matrix(parameter)
    source = record(orientation, coupling=np.zeros((3, 9)), covariance=np.eye(3)*sigma**2)
    translation = source.distribution.components[0].translation.mean_at_reference
    offsets = np.array([[0., .325, 0.], [.01, .31, .02]])
    locations = offsets + translation
    law = ConditionalPointDensity(source, point, 9)
    actual, actual_gradient = law.log_density_and_gradient(locations)
    for index, offset in enumerate(offsets):
        # Recover this scalar rotation quadratic directly from quaternion
        # actions, independently of the density evaluator's stored forms.
        form = np.empty((4, 4))
        basis = np.eye(4)
        for row in range(4):
            form[row, row] = offset @ quat_to_rotmat(basis[row]) @ point
        for row in range(4):
            for column in range(row+1, 4):
                form[row, column] = form[column, row] = (
                    offset @ quat_to_rotmat((basis[row]+basis[column])/np.sqrt(2.)) @ point
                    - .5*(form[row, row]+form[column, column]))
        posterior = parameter + form/sigma**2
        expected = (-1.5*np.log(2*np.pi*sigma**2)
                    - .5*(offset @ offset + point @ point)/sigma**2
                    + bingham_log_normalizer(posterior, 480) - bingham_log_normalizer(parameter, 480))
        expected_gradient = (-offset + rotation_first_moment(bingham_second_moment(posterior, 480)) @ point)/sigma**2
        assert actual[index] == pytest.approx(expected, abs=2e-8)
        np.testing.assert_allclose(actual_gradient[index], expected_gradient, atol=1e-4, rtol=1e-8)
    refinement = law.refinement_diagnostics(locations, 13, log_tolerance=1e-7, gradient_tolerance=1e-4)
    assert refinement["maximum_log_density_change"] < 1e-8
    assert any("gauss_hermite" in rule for query in law.last_diagnostics
               for component in query["components"] for rule in component["integration_rules"])


@pytest.mark.parametrize("parameter", [np.zeros((4, 4)), np.diag([0., -400., -600., -800.])])
def test_positive_partition_preserves_orientation_independent_gaussian(parameter):
    source = record(BinghamOrientation.from_parameter_matrix(parameter), coupling=np.zeros((3, 9)))
    law = ConditionalPointDensity(source, order=13)
    locations = np.array([[.2, -.1, .4], [.22, -.12, .41]])
    values, gradients = law.log_density_and_gradient(locations)
    component = source.distribution.components[0]
    expected = multivariate_normal(component.translation.mean_at_reference,
                                   component.translation.residual_covariance)
    np.testing.assert_allclose(values, expected.logpdf(locations), rtol=0., atol=1e-10)
    expected_gradient = -(locations-component.translation.mean_at_reference) @ np.linalg.inv(component.translation.residual_covariance)
    np.testing.assert_allclose(gradients, expected_gradient, rtol=0., atol=1e-10)
    if not np.any(parameter):
        assert all(query['components'][0]['modes'] > 1 for query in law.last_diagnostics)


def test_gauss_hermite_partition_needs_no_proposal_normalizer(monkeypatch):
    orientation = BinghamOrientation.from_parameter_matrix(np.diag([0., -2e8, -3e8, -4e8]))
    law = ConditionalPointDensity(record(orientation), [.08, -.04, .06], 9)
    locations = np.array([[.2, -.1, .4], [.3, -.2, .5]])
    reference, reference_gradient = law.log_density_and_gradient(locations)
    for component in law._components:
        component['warm'] = None
    def unavailable_normalizer(*_args, **_kwargs):
        raise AssertionError("Gaussian chart partition needs no Bingham proposal normalizer")
    monkeypatch.setattr(point_density, 'bingham_log_normalizer', unavailable_normalizer)
    values, gradients = law.log_density_and_gradient(locations)
    np.testing.assert_array_equal(values, reference)
    np.testing.assert_array_equal(gradients, reference_gradient)
