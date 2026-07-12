import numpy as np

from symaware_grasp.prob_tf.geometry import axis_angle_to_quat
from symaware_grasp.prob_tf.tangent_surrogate import induced_vector_moments_tangent
from symaware_grasp.prob_tf.urdf_override import make_bingham_param_from_mode


def test_tangent_surrogate_is_finite_for_concentrated_identity_mode():
    parameter = make_bingham_param_from_mode([1.0, 0.0, 0.0, 0.0], [3000.0, -1000.0, -1000.0, -1000.0])
    result = induced_vector_moments_tangent([1.0, 0.0, 0.0], parameter)

    assert np.all(np.isfinite(result.mean))
    assert np.all(np.isfinite(result.cov))
    assert np.all(np.isfinite(result.sigma_mat))
    assert np.linalg.norm(result.mean - np.array([1.0, 0.0, 0.0], dtype=float)) < 5e-2
    assert float(np.trace(result.cov)) < 5e-2


def test_tangent_surrogate_is_locally_isotropic_for_isotropic_bingham():
    parameter = make_bingham_param_from_mode([1.0, 0.0, 0.0, 0.0], [3000.0, -1000.0, -1000.0, -1000.0])
    result = induced_vector_moments_tangent([1.0, 0.0, 0.0], parameter)

    assert np.allclose(result.sigma_mat, result.sigma_mat.T, atol=1e-10)
    assert abs(float(result.sigma_mat[0, 0] - result.sigma_mat[1, 1])) < 1e-3
    assert abs(float(result.sigma_mat[0, 1])) < 1e-3


def test_tangent_surrogate_mean_tracks_mode_rotation():
    mode_quaternion = axis_angle_to_quat([0.0, 0.0, 1.0], 0.5 * np.pi)
    parameter = make_bingham_param_from_mode(mode_quaternion, [3000.0, -1000.0, -1000.0, -1000.0])
    result = induced_vector_moments_tangent([1.0, 0.0, 0.0], parameter)

    assert np.linalg.norm(result.mean - np.array([0.0, 1.0, 0.0], dtype=float)) < 5e-2


def test_tangent_surrogate_polar_branch_is_finite():
    parameter = make_bingham_param_from_mode([1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, -2.0])
    result = induced_vector_moments_tangent([1.0, 0.0, 0.0], parameter)

    assert np.all(np.isfinite(result.mean))
    assert np.all(np.isfinite(result.cov))
    assert np.all(np.isfinite(result.sigma_mat))
    assert np.allclose(result.mean / np.linalg.norm(result.mean), np.array([1.0, 0.0, 0.0]), atol=1e-8)
