"""Chapter-4 spherical fiber density, with independent deterministic checks."""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import i0e

from probtf.distributions import BinghamOrientation
from probtf.geometry import quat_left_matrix, quat_right_matrix, quat_to_rotmat, rpy_to_quat
from probtf.probability.moments import rotation_vector_moments
from probtf.provenance import ApproximationKind
from probtf.spherical_law import (
    DiracDirectionLaw, DiracVectorLaw, ExactIslBackend,
    FiniteBinghamInducedDirectionLaw, IslBackendUnavailableError,
    IslEvaluationOptions, NumericalIslBackend, ScaledDirectionVectorLaw,
    UniformDirectionLaw, UnavailableExactIslBackend,
)


OPTIONS = IslEvaluationOptions(absolute_tolerance=1e-11, relative_tolerance=1e-11)


def parameter():
    left = quat_left_matrix(rpy_to_quat(.2, -.3, .4))
    return left @ np.diag([0., -2., -4., -7.]) @ left.T


def unit(value):
    value = np.asarray(value, dtype=float)
    return value / np.linalg.norm(value)


def independent_normalizer(matrix):
    """Integrate the two circular angles analytically, then integrate u."""
    values = np.linalg.eigvalsh(matrix)[::-1]
    shift = values[0]
    values = values - shift
    first_gap, second_gap = (values[0] - values[1]) / 2., (values[2] - values[3]) / 2.
    integral = quad(lambda u: np.exp(u * values[0] + (1. - u) * values[2]) *
                    i0e(u * first_gap) * i0e((1. - u) * second_gap),
                    0., 1., epsabs=1e-13, epsrel=1e-13)[0]
    return 2. * np.pi ** 2 * integral * np.exp(shift)


def sphere_grid():
    z, weights = leggauss(48)
    azimuth = np.arange(128) * (2. * np.pi / 128)
    points = np.stack((np.sqrt(1. - z[:, None] ** 2) * np.cos(azimuth),
                       np.sqrt(1. - z[:, None] ** 2) * np.sin(azimuth),
                       np.broadcast_to(z[:, None], (len(z), len(azimuth)))), axis=-1)
    return points.reshape(-1, 3), np.repeat(weights * 2. * np.pi / 128, 128)


def test_density_matches_the_direct_fiber_integral_including_the_antipode():
    matrix, direction = parameter(), unit([.2, .4, .8])
    orientation = BinghamOrientation.from_parameter_matrix(matrix)
    law = ExactIslBackend().rotate_direction(orientation, direction, OPTIONS)
    assert isinstance(law, FiniteBinghamInducedDirectionLaw)
    assert law.orientation is orientation
    assert law.approximation.kind is ApproximationKind.NUMERICAL_INTEGRATION
    assert not law.approximation.lossy and law.approximation.error_bound is None
    normalizer = independent_normalizer(matrix)
    for output in (direction, -direction, unit([-.4, .7, .2])):
        if np.dot(direction, output) < -1. + 1e-12:
            first = unit(np.cross(direction, [1., 0., 0.]))
            second = np.cross(direction, first)
            q1, q2 = np.r_[0., first], np.r_[0., second]
        else:
            cosine = np.sqrt((1. + np.dot(direction, output)) / 2.)
            q1 = np.r_[cosine, np.cross(direction, output) / (2. * cosine)]
            q2 = np.r_[0., (direction + output) / (2. * cosine)]
        def fiber(t):
            quaternion = q1 * np.cos(t) + q2 * np.sin(t)
            return np.exp(quaternion @ matrix @ quaternion)
        expected = quad(fiber, 0., 2. * np.pi, epsabs=1e-12, epsrel=1e-12)[0] / (4. * normalizer)
        assert law.density(output) == pytest.approx(expected, rel=2e-10, abs=1e-12)
    # Quaternion antipodes represent one rotation; output directions need not
    # have antipodally symmetric probabilities on the two-dimensional sphere.
    assert not np.isclose(law.density(direction), law.density(-direction), rtol=.01)


def test_density_normalizes_and_its_moments_match_native_second_and_fourth_moments():
    orientation = BinghamOrientation.from_parameter_matrix(parameter())
    direction = unit([.2, .4, .8])
    law = ExactIslBackend().rotate_direction(orientation, direction, OPTIONS)
    points, area_weights = sphere_grid()
    weighted = area_weights * law.density(points)
    assert weighted.sum() == pytest.approx(1., abs=2e-10)
    mean = weighted @ points
    covariance = np.einsum("s,si,sj->ij", weighted, points - mean, points - mean)
    rotations = rotation_vector_moments(orientation, integration_steps=960)
    action = np.kron(direction[None], np.eye(3))
    expected_mean = action @ rotations.mean_vector
    expected_covariance = action @ (rotations.second_vector -
                                    np.outer(rotations.mean_vector, rotations.mean_vector)) @ action.T
    np.testing.assert_allclose(mean, expected_mean, rtol=0., atol=2e-9)
    np.testing.assert_allclose(covariance, expected_covariance, rtol=0., atol=2e-9)


def test_scalar_gauge_left_and_right_rotations_preserve_the_induced_law():
    matrix, direction = parameter(), unit([.2, .4, .8])
    query = unit([-.4, .7, .2])
    backend = ExactIslBackend()
    original = backend.rotate_direction(BinghamOrientation.from_parameter_matrix(matrix), direction, OPTIONS)
    shifted = backend.rotate_direction(BinghamOrientation.from_parameter_matrix(matrix + 1234. * np.eye(4)),
                                       direction, OPTIONS)
    assert shifted.log_density(query) == pytest.approx(original.log_density(query), abs=1e-10)
    quaternion = rpy_to_quat(.4, -.2, .6)
    rotation = quat_to_rotmat(quaternion)
    for sign in (-1., 1.):
        left = quat_left_matrix(sign * quaternion)
        moved = backend.rotate_direction(BinghamOrientation.from_parameter_matrix(left @ matrix @ left.T),
                                         direction, OPTIONS)
        assert moved.log_density(rotation @ query) == pytest.approx(original.log_density(query), abs=1e-10)
    right = quat_right_matrix(quaternion)
    moved = backend.rotate_direction(BinghamOrientation.from_parameter_matrix(right @ matrix @ right.T),
                                     rotation.T @ direction, OPTIONS)
    assert moved.log_density(query) == pytest.approx(original.log_density(query), abs=1e-10)


def test_high_concentration_density_stays_finite_and_has_the_correct_local_peak_scale():
    backend = ExactIslBackend()
    for concentration in (100., 10000.):
        orientation = BinghamOrientation.from_parameter_matrix(np.diag(
            [0., -concentration, -2. * concentration, -3. * concentration]))
        law = backend.rotate_direction(orientation, [1., 0., 0.], OPTIONS)
        assert np.isfinite(law.log_density([-1., 0., 0.]))
        peak = law.density([1., 0., 0.])
        expected_local_peak = concentration * np.sqrt(1.5) / (2. * np.pi)
        assert peak == pytest.approx(expected_local_peak, rel=.01)


def test_exact_special_cases_vector_surface_measure_and_compatibility_backend():
    backend = ExactIslBackend()
    uniform = backend.rotate_direction(BinghamOrientation.uniform(), [1., 0., 0.], OPTIONS)
    assert isinstance(uniform, UniformDirectionLaw)
    assert uniform.log_density([0., 1., 0.]) == pytest.approx(-np.log(4. * np.pi))
    quaternion = rpy_to_quat(.2, -.3, .4)
    positive = backend.rotate_direction(BinghamOrientation.dirac(quaternion), [1., 0., 0.], OPTIONS)
    negative = backend.rotate_direction(BinghamOrientation.dirac(-quaternion), [1., 0., 0.], OPTIONS)
    assert isinstance(positive, DiracDirectionLaw)
    np.testing.assert_array_equal(positive.direction, negative.direction)
    orientation = BinghamOrientation.from_parameter_matrix(parameter())
    zero = backend.rotate_vector(orientation, np.zeros(3), OPTIONS)
    assert isinstance(zero, DiracVectorLaw)
    vector = backend.rotate_vector(orientation, [2., 0., 0.], OPTIONS)
    assert isinstance(vector, ScaledDirectionVectorLaw)
    assert vector.surface_density([0., 2., 0.]) == pytest.approx(vector.direction_law.density([0., 1., 0.]) / 4.)
    with pytest.raises(ValueError, match="fixed-radius"):
        vector.surface_density([0., 1., 0.])
    tiny = backend.rotate_vector(orientation, [1e-14, 0., 0.], OPTIONS)
    assert isinstance(tiny, ScaledDirectionVectorLaw) and tiny.radius == 1e-14
    assert isinstance(NumericalIslBackend().rotate_direction(orientation, [1., 0., 0.], OPTIONS),
                      FiniteBinghamInducedDirectionLaw)
    with pytest.raises(IslBackendUnavailableError):
        UnavailableExactIslBackend().rotate_direction(orientation, [1., 0., 0.], OPTIONS)


@pytest.mark.parametrize("direction", [[0., 0., 0.], [2., 0., 0.], [np.nan, 0., 0.], [1., 0.]])
def test_exact_law_rejects_invalid_output_directions(direction):
    orientation = BinghamOrientation.from_parameter_matrix(parameter())
    law = ExactIslBackend().rotate_direction(orientation, [1., 0., 0.], OPTIONS)
    with pytest.raises(ValueError):
        law.density(direction)
