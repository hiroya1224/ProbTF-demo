import numpy as np

from probik_demo.prob_tf.geometry import axis_angle_to_quat, quat_conj, quat_mul, quat_to_rotmat


def test_quaternion_conjugate_product_is_identity():
    quaternion = axis_angle_to_quat([0.3, -0.4, 0.5], 0.8)
    product = quat_mul(quaternion, quat_conj(quaternion))
    assert np.allclose(product, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-8)


def test_rotation_matrix_is_orthogonal():
    quaternion = axis_angle_to_quat([0.0, 1.0, 0.0], -0.7)
    rotation = quat_to_rotmat(quaternion)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)


def test_rotation_matrix_respects_quaternion_multiplication():
    quaternion_a = axis_angle_to_quat([1.0, 0.0, 0.0], 0.3)
    quaternion_b = axis_angle_to_quat([0.0, 0.0, 1.0], -0.4)
    lhs = quat_to_rotmat(quat_mul(quaternion_a, quaternion_b))
    rhs = quat_to_rotmat(quaternion_a) @ quat_to_rotmat(quaternion_b)
    assert np.allclose(lhs, rhs, atol=1e-8)
