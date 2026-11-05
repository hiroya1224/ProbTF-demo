"""Distribution and wrapper checks for the single C++ BACG implementation."""

import numpy as np
import pytest

from probtf import _sampling
from probtf.distributions import BinghamOrientation
from probtf.probability import sample_bingham_orientation


def native_samples(parameter, count, seed=42):
    parameter = np.ascontiguousarray(parameter, dtype=np.float64)
    output = np.empty((count, parameter.shape[0]))
    b, proposals = _sampling.sample_bingham(parameter, output, seed)
    return output, b, proposals


@pytest.mark.parametrize("bit_generator", [np.random.PCG64, np.random.MT19937])
def test_python_wrapper_calls_native_with_full_width_seed(bit_generator):
    orientation = BinghamOrientation.from_parameter_matrix(np.diag([-3., -2., -1., 0.]))
    generator = np.random.Generator(bit_generator(123))
    expected, _, _ = native_samples(
        orientation.backend_parameter_matrix(), 200,
        int(generator.integers(0, 1 << 64, dtype=np.uint64)))
    actual = sample_bingham_orientation(orientation, 200, np.random.Generator(bit_generator(123)))
    np.testing.assert_array_equal(actual, expected)
    assert not actual.flags.writeable


@pytest.mark.parametrize("parameter", [np.zeros((4, 4)), np.diag([-3., -2., -1., 0.]),
                                      np.diag([-1e12, -1e12, -1e12, 0.])])
def test_count_independent_prefixes(parameter):
    small, _, _ = native_samples(parameter, 30)
    large, _, _ = native_samples(parameter, 300)
    np.testing.assert_array_equal(small, large[:30])


def test_shift_and_orthogonal_equivariance_of_second_moments():
    basis, _ = np.linalg.qr(np.random.default_rng(12).normal(size=(4, 4)))
    diagonal = np.diag([-3., -2., -1., 0.])
    rotated = basis @ diagonal @ basis.T
    samples, _, _ = native_samples(rotated, 50000)
    shifted, _, _ = native_samples(rotated + 123 * np.eye(4), 50000)
    np.testing.assert_allclose(samples, shifted, atol=2e-12)
    from probtf.bingham import bingham_second_moment
    expected = basis @ bingham_second_moment(diagonal) @ basis.T
    np.testing.assert_allclose(samples.T @ samples / len(samples), expected, atol=.005)


def test_girdle_is_not_replaced_by_single_mode():
    samples, b, proposals = native_samples(np.diag([-1e8, -1e8, 0., 0.]), 30000)
    moment = samples.T @ samples / len(samples)
    np.testing.assert_allclose(np.diag(moment)[2:], [.5, .5], atol=.008)
    np.testing.assert_allclose(1e8 * np.diag(moment)[:2], [.5, .5], atol=.02)
    assert 1 <= b <= 4
    assert proposals < 3 * len(samples)


def test_concentrated_arc_matches_the_von_mises_limit_without_dead_zones():
    from scipy.stats import vonmises
    # q=(cos(theta),0,0,sin(theta)) and phi=2*theta in the concentrated
    # plane. exp(-4*sin(theta)^2) gives a von Mises law with kappa=2.
    samples, _, _ = native_samples(np.diag([0.,-1e6,-1e6,-4.]), 50000, 20261106)
    angles = np.sort((2*np.arctan2(samples[:,3],samples[:,0])+np.pi) % (2*np.pi)-np.pi)
    cdf = vonmises.cdf(angles, 2.)
    empirical = np.arange(1,len(angles)+1)/len(angles)
    assert np.max(np.abs(empirical-cdf)) < .012


def test_buffers_and_matrix_validation():
    with pytest.raises(ValueError):
        native_samples(np.array([[0., 1.], [0., 0.]]), 2)
    with pytest.raises(ValueError):
        _sampling.sample_bingham(np.eye(4), np.empty((10, 3)), 0)
    with pytest.raises((ValueError, BufferError)):
        _sampling.sample_bingham(np.eye(4), np.empty((10, 4), dtype=np.float32), 0)
    for invalid_seed in (-1, 1 << 64):
        with pytest.raises(OverflowError):
            native_samples(np.eye(4), 2, invalid_seed)
    lower, _, _ = native_samples(np.zeros((4, 4)), 2, 42)
    higher, _, _ = native_samples(np.zeros((4, 4)), 2, 42 + (1 << 63))
    assert not np.array_equal(lower, higher)
