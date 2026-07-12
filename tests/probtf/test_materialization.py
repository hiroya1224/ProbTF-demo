import numpy as np
import pytest

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.geometry import rotation_action_matrix
from probtf_estimators.materialization import summarize_transform_for_materialization


def _record(components):
    return TransformDistributionStamped(
        "world",
        "sensor",
        1.0,
        "world_sensor",
        "test",
        TransformDistribution(tuple(components)),
    )


def test_materialization_position_summary_includes_rotation_coupling():
    orientation = BinghamOrientation.uniform()
    offset = np.array([0.3, -0.2, 0.1])
    component = TransformComponent(
        "sensor",
        1.0,
        orientation,
        ConditionalGaussianTranslation(
            offset,
            np.diag([0.01, 0.02, 0.03]),
            rotation_action_matrix(offset),
        ),
    )

    summary = summarize_transform_for_materialization(_record((component,)))

    np.testing.assert_allclose(summary.position_mean, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(
        summary.position_covariance,
        np.diag([0.01, 0.02, 0.03])
        + (np.dot(offset, offset) / 3.0) * np.eye(3),
        atol=1e-12,
    )
    assert summary.orientation_concentration_gap == 0.0


def test_materialization_rejects_mixture_instead_of_selecting_component():
    component = TransformComponent(
        "first",
        1.0,
        BinghamOrientation.dirac([1.0, 0.0, 0.0, 0.0]),
        ConditionalGaussianTranslation(np.zeros(3), np.zeros((3, 3)), np.zeros((3, 9))),
    )
    second = TransformComponent(
        "second",
        1.0,
        component.orientation,
        component.translation,
    )

    with pytest.raises(ValueError, match="exactly one"):
        summarize_transform_for_materialization(_record((component, second)))
