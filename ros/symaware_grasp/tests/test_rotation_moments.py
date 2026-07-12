import numpy as np

from symaware_grasp.prob_tf.rotation_moments import rotation_moment_from_bingham
from symaware_grasp.prob_tf.urdf_override import make_bingham_param_from_mode


def test_rotation_moment_is_near_identity_for_concentrated_identity_mode():
    parameter = make_bingham_param_from_mode([1.0, 0.0, 0.0, 0.0], [3000.0, -1000.0, -1000.0, -1000.0])
    moment = rotation_moment_from_bingham(parameter, integration_steps=60)
    assert np.allclose(moment.mean_rot, np.eye(3), atol=5e-2)


def test_rotation_moment_preserves_identity_matrix():
    parameter = make_bingham_param_from_mode([1.0, 0.0, 0.0, 0.0], [2100.0, -700.0, -700.0, -700.0])
    moment = rotation_moment_from_bingham(parameter, integration_steps=60)
    propagated = moment.apply_second(np.eye(3))
    assert np.allclose(propagated, np.eye(3), atol=5e-2)


def test_rotation_moment_keeps_symmetric_inputs_symmetric():
    parameter = make_bingham_param_from_mode([1.0, 0.0, 0.0, 0.0], [1800.0, -600.0, -600.0, -600.0])
    moment = rotation_moment_from_bingham(parameter, integration_steps=60)
    symmetric = np.array([[1.0, 0.2, -0.1], [0.2, 2.0, 0.3], [-0.1, 0.3, 3.0]], dtype=float)
    propagated = moment.apply_second(symmetric)
    assert np.allclose(propagated, propagated.T, atol=1e-8)
