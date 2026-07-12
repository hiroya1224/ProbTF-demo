import numpy as np
import pytest

from probtf.models import (
    BinghamRotation,
    GaussianPosition,
    ImuKinematics,
    ProbabilisticTransform,
)


def test_probabilistic_transform_normalizes_wire_conventions():
    transform = ProbabilisticTransform.from_arrays(
        "/base",
        "/imu",
        [1.0, 2.0, 3.0],
        np.diag([0.1, 0.2, 0.3]),
        np.diag([4.0, 1.0, 0.0, -2.0]),
        [-2.0, 0.0, 0.0, 0.0],
        source_id="calibrator",
        evidence_source_ids=("gyro", "force"),
    )

    assert transform.parent_frame_id == "base"
    assert transform.child_frame_id == "imu"
    assert transform.edge_id == "base__to__imu"
    np.testing.assert_allclose(transform.orientation_mode_wxyz, [-1.0, 0.0, 0.0, 0.0])
    assert np.max(np.linalg.eigvalsh(transform.orientation_bingham)) == pytest.approx(0.0)


def test_covariances_must_be_positive_semidefinite():
    with pytest.raises(ValueError, match="positive semidefinite"):
        GaussianPosition(np.zeros(3), np.diag([1.0, 1.0, -0.1]))

    with pytest.raises(ValueError, match="positive semidefinite"):
        ImuKinematics(
            "imu",
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.diag([1.0, 1.0, -0.1]),
            np.eye(3),
            np.eye(3),
        )


def test_bingham_rotation_uses_wxyz_mode_and_canonical_gauge():
    parameter = np.diag([-8.0, -4.0, 3.0, -2.0])
    rotation = BinghamRotation(parameter)

    np.testing.assert_allclose(np.abs(rotation.mode_wxyz), [0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(
        rotation.parameter,
        np.diag([-11.0, -7.0, 0.0, -5.0]),
    )


def test_transform_rejects_duplicate_evidence_sources():
    with pytest.raises(ValueError, match="unique"):
        ProbabilisticTransform(
            "base",
            "imu",
            GaussianPosition(np.zeros(3), np.eye(3)),
            BinghamRotation(np.zeros((4, 4))),
            evidence_source_ids=("imu", "imu"),
        )
