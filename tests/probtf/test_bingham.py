import itertools

import numpy as np
import pytest

from probtf.bingham import (
    bingham_fourth_moment,
    bingham_log_normalizer,
    bingham_mode,
    bingham_second_moment,
    canonical_bingham_parameter,
    deterministic_rotation_moment_from_quaternion,
    match_bingham_to_second_moment,
    quaternion_product_second_moment,
    rotation_first_moment,
    rotation_kronecker_moment,
    rotation_moment_from_bingham,
    validate_bingham_parameter,
)
from probtf.geometry import quat_mul, quat_to_rotmat


def _uniform_fourth_moment():
    moment = np.zeros((4, 4, 4, 4), dtype=float)
    delta = np.eye(4, dtype=float)
    for i, j, k, l in itertools.product(range(4), repeat=4):
        moment[i, j, k, l] = (
            delta[i, j] * delta[k, l]
            + delta[i, k] * delta[j, l]
            + delta[i, l] * delta[j, k]
        ) / 24.0
    return moment


def _outer_fourth(quaternion):
    return np.einsum("i,j,k,l->ijkl", quaternion, quaternion, quaternion, quaternion)


def test_parameter_validation_and_canonical_gauge():
    parameter = np.array(
        [
            [-4.0, 0.2, 0.0, 0.0],
            [0.2, -2.0, 0.1, 0.0],
            [0.0, 0.1, -1.0, 0.0],
            [0.0, 0.0, 0.0, 3.0],
        ],
        dtype=float,
    )
    canonical = canonical_bingham_parameter(parameter)

    assert canonical.shape == (4, 4)
    assert np.allclose(canonical, canonical.T)
    assert np.isclose(np.linalg.eigvalsh(canonical)[-1], 0.0, atol=1e-12)
    assert np.allclose(canonical_bingham_parameter(parameter + 17.0 * np.eye(4)), canonical)
    assert np.allclose(validate_bingham_parameter(parameter), parameter)

    with pytest.raises(ValueError, match="shape"):
        canonical_bingham_parameter(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="symmetric"):
        canonical_bingham_parameter(np.triu(np.ones((4, 4))))
    with pytest.raises(ValueError, match="finite"):
        canonical_bingham_parameter(np.diag([0.0, 0.0, 0.0, np.nan]))


def test_mode_is_wxyz_and_gauge_invariant():
    expected_mode = np.array([0.8, -0.4, 0.2, 0.4], dtype=float)
    expected_mode /= np.linalg.norm(expected_mode)
    parameter = -12.0 * (np.eye(4) - np.outer(expected_mode, expected_mode))

    actual_mode = bingham_mode(parameter + 9.0 * np.eye(4))

    assert np.isclose(abs(float(actual_mode @ expected_mode)), 1.0, atol=1e-12)
    assert actual_mode[int(np.argmax(np.abs(actual_mode)))] >= 0.0


def test_uniform_distribution_has_known_second_and_fourth_moments():
    parameter = np.zeros((4, 4), dtype=float)

    second = bingham_second_moment(parameter, integration_steps=80)
    fourth = bingham_fourth_moment(parameter, integration_steps=80)

    assert np.allclose(second, np.eye(4) / 4.0, atol=2e-5)
    assert np.allclose(fourth, _uniform_fourth_moment(), atol=2e-5)
    assert np.isclose(np.trace(second), 1.0, atol=1e-12)
    assert np.isclose(np.einsum("iikk->", fourth), 1.0, atol=1e-12)


def test_moments_are_invariant_to_parameter_gauge():
    parameter = np.diag([-8.0, -4.0, -1.5, 0.0])
    shifted = parameter + 100.0 * np.eye(4)

    assert np.allclose(
        bingham_second_moment(parameter, integration_steps=60),
        bingham_second_moment(shifted, integration_steps=60),
        atol=1e-12,
    )
    assert np.allclose(
        bingham_fourth_moment(parameter, integration_steps=60),
        bingham_fourth_moment(shifted, integration_steps=60),
        atol=1e-12,
    )


def test_log_normalizer_tracks_parameter_gauge_shift():
    parameter = np.diag([-8.0, -4.0, -1.5, 0.0])
    baseline = bingham_log_normalizer(parameter, integration_steps=60)
    shifted = bingham_log_normalizer(parameter + 7.5 * np.eye(4), integration_steps=60)

    assert np.isclose(shifted - baseline, 7.5, atol=1e-12)


def test_matching_round_trip_preserves_second_moment_and_mode():
    expected_parameter = np.diag([-9.0, -4.0, -1.0, 0.0])
    expected_moment = bingham_second_moment(expected_parameter, integration_steps=80)

    fitted_parameter = match_bingham_to_second_moment(
        expected_moment,
        integration_steps=80,
        max_iterations=100,
    )
    fitted_moment = bingham_second_moment(fitted_parameter, integration_steps=80)

    assert np.isclose(np.linalg.eigvalsh(fitted_parameter)[-1], 0.0, atol=1e-12)
    assert np.allclose(fitted_moment, expected_moment, atol=5e-5)
    assert np.isclose(abs(float(bingham_mode(fitted_parameter)[3])), 1.0, atol=1e-8)


def test_quaternion_product_second_moment_matches_deterministic_product():
    left = np.array([0.8, 0.2, -0.4, 0.4], dtype=float)
    right = np.array([0.5, -0.5, 0.5, 0.5], dtype=float)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    expected = quat_mul(left, right)

    product_moment = quaternion_product_second_moment(
        np.outer(left, left),
        np.outer(right, right),
    )

    assert np.allclose(product_moment, np.outer(expected, expected), atol=1e-12)
    assert np.allclose(
        quaternion_product_second_moment(np.eye(4) / 4.0, np.outer(right, right)),
        np.eye(4) / 4.0,
        atol=1e-12,
    )


def test_quaternion_product_rejects_invalid_second_moment():
    with pytest.raises(ValueError, match="trace one"):
        quaternion_product_second_moment(np.eye(4), np.eye(4) / 4.0)
    invalid = np.diag([0.5, 0.5, 0.1, -0.1])
    with pytest.raises(ValueError, match="positive semidefinite"):
        quaternion_product_second_moment(invalid, np.eye(4) / 4.0)


def test_rotation_moments_match_a_deterministic_quaternion():
    quaternion = np.array([0.5, -0.5, 0.5, 0.5], dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    rotation = quat_to_rotmat(quaternion)
    second = np.outer(quaternion, quaternion)
    fourth = _outer_fourth(quaternion)

    assert np.allclose(rotation_first_moment(second), rotation, atol=1e-12)
    assert np.allclose(rotation_kronecker_moment(fourth), np.kron(rotation, rotation), atol=1e-12)

    deterministic = deterministic_rotation_moment_from_quaternion(quaternion)
    matrix = np.array([[1.0, 0.2, 0.3], [0.2, 2.0, -0.1], [0.3, -0.1, 3.0]])
    assert np.allclose(deterministic.apply_second(matrix), rotation @ matrix @ rotation.T, atol=1e-12)


def test_rotation_moment_from_concentrated_bingham_is_near_mode_rotation():
    parameter = np.diag([0.0, -800.0, -800.0, -800.0])
    moment = rotation_moment_from_bingham(parameter, integration_steps=80)

    assert np.allclose(moment.mean_rot, np.eye(3), atol=1e-2)
    assert np.allclose(moment.apply_second(np.eye(3)), np.eye(3), atol=1e-2)
