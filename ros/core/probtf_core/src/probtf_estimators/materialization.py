"""Terminal summaries used when materializing a stochastic transform."""

from dataclasses import dataclass

import numpy as np

from probtf.distributions import (
    DistributionStatus,
    OrientationKind,
    TransformDistributionStamped,
)
from probtf.probability import PointMomentSummary, forward_component_point_moments


@dataclass(frozen=True)
class TransformMaterializationSummary:
    position_mean: np.ndarray
    position_covariance: np.ndarray
    orientation_mode_wxyz: np.ndarray
    orientation_concentration_gap: float


def summarize_transform_for_materialization(record, integration_steps=120):
    """Return a one-component terminal moment/mode summary.

    Mixtures are rejected instead of silently choosing a component. The
    position moments include the component's rotation/translation coupling.
    """

    if not isinstance(record, TransformDistributionStamped):
        raise TypeError("record must be a TransformDistributionStamped.")
    normalized = record.distribution.normalize_weights()
    if normalized.status is not DistributionStatus.OK:
        raise ValueError("Transform distribution is not usable.")
    if len(normalized.components) != 1:
        raise ValueError("Materialization requires exactly one usable component.")
    component = normalized.components[0].component
    point = forward_component_point_moments(
        component,
        PointMomentSummary(np.zeros(3), np.zeros((3, 3))),
        integration_steps=integration_steps,
    )
    orientation = component.orientation
    if orientation.kind is OrientationKind.DIRAC:
        concentration_gap = np.inf
    elif orientation.kind is OrientationKind.UNIFORM:
        concentration_gap = 0.0
    else:
        eigenvalues = np.linalg.eigvalsh(orientation.backend_parameter_matrix())
        concentration_gap = float(eigenvalues[-1] - eigenvalues[-2])
    return TransformMaterializationSummary(
        point.mean,
        point.covariance,
        orientation.mode_wxyz,
        concentration_gap,
    )
