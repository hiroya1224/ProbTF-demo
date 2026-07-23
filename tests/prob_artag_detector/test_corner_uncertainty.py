import numpy as np
import pytest

from prob_artag_detector.corner_uncertainty import (
    AdaptiveCornerCovariance,
    _Track,
    affine_explained_fraction,
    constant_velocity_innovation,
)
from prob_artag_detector.models import MarkerObservation


BASE_CORNERS = np.array(
    [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]
)
SPATIAL = np.eye(8, dtype=float) * 0.25


def _observation(corners, marker_id=7, covariance=SPATIAL):
    return MarkerObservation(marker_id, np.asarray(corners), covariance)


def _assert_spd(covariance):
    np.testing.assert_allclose(covariance, covariance.T, rtol=0.0, atol=1e-12)
    assert np.all(np.isfinite(covariance))
    assert float(np.min(np.linalg.eigvalsh(covariance))) > 0.0


def test_constant_velocity_innovation_is_zero_for_irregular_timestamps():
    velocity = np.array([3.0, -1.0] * 4)
    z0 = BASE_CORNERS.reshape(8)
    z1 = z0 + 0.04 * velocity
    z2 = z0 + 0.11 * velocity
    innovation, predicted, residual, ratio = constant_velocity_innovation(
        (1.0, z0), (1.04, z1), (1.11, z2)
    )
    np.testing.assert_allclose(innovation, 0.0, atol=1e-13)
    np.testing.assert_allclose(predicted, z2, atol=1e-13)
    np.testing.assert_allclose(residual, 0.0, atol=1e-13)
    np.testing.assert_allclose(ratio, 0.07 / 0.04)


def test_noise_free_constant_velocity_never_exceeds_spatial_floor():
    adaptive = AdaptiveCornerCovariance(
        warmup_samples=4,
        temporal_half_life_sec=0.1,
    )
    velocity = np.array([1.7, -0.6])
    result = None
    diagnostics = None
    for index in range(40):
        stamp = index / 30.0
        corners = BASE_CORNERS + stamp * velocity
        result, diagnostics = adaptive.update(_observation(corners), stamp)
        _assert_spd(result.image_covariance)
    assert diagnostics.temporal_ready
    np.testing.assert_allclose(result.image_covariance, SPATIAL, atol=1e-12)


def test_non_affine_single_corner_jitter_adds_temporal_excess():
    adaptive = AdaptiveCornerCovariance(
        warmup_samples=4,
        temporal_half_life_sec=0.08,
        motion_gate_px=0.5,
        affine_motion_fraction=0.9,
        shrinkage=0.1,
    )
    result = None
    statuses = []
    for index in range(80):
        corners = BASE_CORNERS.copy()
        corners[0, 0] += 1.5 if index % 2 else -1.5
        result, diagnostics = adaptive.update(
            _observation(corners), index / 30.0
        )
        statuses.append(diagnostics.status)
    _assert_spd(result.image_covariance)
    assert diagnostics.temporal_ready
    assert statuses.count("accepted") > 60
    assert result.image_covariance[0, 0] > 2.0 * SPATIAL[0, 0]
    assert np.trace(result.image_covariance) > np.trace(SPATIAL)


def test_correlated_corner_jitter_retains_full_covariance():
    adaptive = AdaptiveCornerCovariance(
        warmup_samples=4,
        temporal_half_life_sec=0.08,
        motion_gate_px=100.0,
        shrinkage=0.1,
    )
    result = None
    for index in range(80):
        corners = BASE_CORNERS.copy()
        shared = 1.2 if index % 2 else -1.2
        corners[0, 0] += shared
        corners[1, 0] += shared
        result, _ = adaptive.update(_observation(corners), index / 30.0)
    _assert_spd(result.image_covariance)
    assert result.image_covariance[0, 2] > 0.25


def test_coherent_acceleration_is_frozen_as_motion():
    adaptive = AdaptiveCornerCovariance(
        warmup_samples=3,
        motion_gate_px=0.1,
        motion_gate_edge_fraction=1e-5,
        freeze_affine_motion=True,
    )
    statuses = []
    result = None
    for index in range(20):
        stamp = index / 30.0
        displacement = 400.0 * stamp * stamp
        corners = BASE_CORNERS + np.array([displacement, 0.0])
        result, diagnostics = adaptive.update(_observation(corners), stamp)
        statuses.append(diagnostics.status)
    assert "motion_frozen" in statuses
    np.testing.assert_allclose(result.image_covariance, SPATIAL, atol=1e-12)
    assert not diagnostics.temporal_ready


def test_default_conservative_mode_learns_coherent_corner_jitter():
    adaptive = AdaptiveCornerCovariance(
        warmup_samples=4,
        temporal_half_life_sec=0.08,
    )
    result = None
    statuses = []
    for index in range(60):
        corners = BASE_CORNERS.copy()
        corners[:, 0] += 0.6 if index % 2 else -0.6
        result, diagnostics = adaptive.update(
            _observation(corners), index / 30.0
        )
        statuses.append(diagnostics.status)
    assert diagnostics.temporal_ready
    assert "motion_frozen" not in statuses
    assert result.image_covariance[0, 2] > 0.1
    assert np.trace(result.image_covariance) > np.trace(SPATIAL)


def test_gap_resets_learned_jitter_and_restores_spatial_covariance():
    adaptive = AdaptiveCornerCovariance(
        warmup_samples=3,
        temporal_half_life_sec=0.05,
        max_gap_sec=0.2,
        motion_gate_px=0.5,
    )
    result = None
    for index in range(30):
        corners = BASE_CORNERS.copy()
        corners[0, 0] += 1.0 if index % 2 else -1.0
        result, _ = adaptive.update(_observation(corners), index / 30.0)
    assert np.trace(result.image_covariance) > np.trace(SPATIAL)

    result, diagnostics = adaptive.update(_observation(BASE_CORNERS), 2.0)
    assert diagnostics.status == "gap_reset"
    assert diagnostics.accepted_samples == 0
    assert not diagnostics.temporal_ready
    np.testing.assert_allclose(result.image_covariance, SPATIAL, atol=1e-12)


def test_outputs_remain_bounded_spd_under_correlated_jitter_and_outlier():
    adaptive = AdaptiveCornerCovariance(
        maximum_excess_sigma_px=3.0,
        warmup_samples=3,
        temporal_half_life_sec=0.05,
        motion_gate_px=2.0,
        hard_outlier_px=4.0,
        hard_outlier_edge_fraction=0.04,
    )
    rng = np.random.RandomState(5)
    for index in range(60):
        common = rng.normal(scale=0.8, size=2)
        independent = rng.normal(scale=0.2, size=(4, 2))
        result, _ = adaptive.update(
            _observation(BASE_CORNERS + common + independent),
            index / 30.0,
        )
        _assert_spd(result.image_covariance)
        excess = result.image_covariance - SPATIAL
        assert float(np.min(np.linalg.eigvalsh(excess))) >= -1e-10
        assert float(np.max(np.linalg.eigvalsh(excess))) <= 9.0 + 1e-10

    outlier = BASE_CORNERS.copy()
    outlier[0] += [100.0, -100.0]
    result, diagnostics = adaptive.update(_observation(outlier), 2.0)
    assert diagnostics.status == "outlier"
    np.testing.assert_allclose(
        result.image_covariance, np.eye(8) * 9.25, atol=1e-12
    )


def test_hard_affine_jump_is_outlier_before_optional_motion_freeze_and_recovers():
    adaptive = AdaptiveCornerCovariance(
        freeze_affine_motion=True,
        hard_outlier_px=8.0,
        hard_outlier_edge_fraction=0.15,
    )
    adaptive.update(_observation(BASE_CORNERS), 0.0)
    adaptive.update(_observation(BASE_CORNERS), 1.0 / 30.0)
    jumped = BASE_CORNERS + np.array([20.0, 0.0])
    result, diagnostics = adaptive.update(_observation(jumped), 2.0 / 30.0)
    assert diagnostics.status == "outlier"
    assert diagnostics.accepted_samples == 0
    assert np.trace(result.image_covariance) > np.trace(SPATIAL)

    recovery_statuses = []
    for index in range(3, 6):
        _, diagnostics = adaptive.update(
            _observation(BASE_CORNERS), index / 30.0
        )
        recovery_statuses.append(diagnostics.status)
    assert recovery_statuses == ["warmup", "warmup", "accepted"]


def test_shrinkage_does_not_create_excess_from_correlated_spatial_covariance():
    correlated = SPATIAL.copy()
    correlated[0, 1] = correlated[1, 0] = 0.2
    adaptive = AdaptiveCornerCovariance(shrinkage=0.25, warmup_samples=4)
    track = _Track(
        accepted_samples=4,
        innovation_covariance=correlated.copy(),
        spatial_mean=correlated.copy(),
    )
    np.testing.assert_allclose(
        adaptive._excess_covariance(track), np.zeros((8, 8)), atol=1e-12
    )


def test_temporal_cap_never_reduces_large_spatial_covariance():
    spatial = np.eye(8) * 100.0
    adaptive = AdaptiveCornerCovariance(maximum_excess_sigma_px=3.0)
    result, _ = adaptive.update(
        _observation(BASE_CORNERS, covariance=spatial), 0.0
    )
    np.testing.assert_allclose(result.image_covariance, spatial, atol=1e-12)


def test_affine_explained_fraction_separates_shared_motion_from_corner_glitch():
    shared = np.tile([3.0, -2.0], (4, 1))
    single = np.zeros((4, 2), dtype=float)
    single[0, 0] = 3.0
    assert affine_explained_fraction(BASE_CORNERS, shared) > 0.999
    assert affine_explained_fraction(BASE_CORNERS, single) < 0.9


def test_invalid_spatial_covariance_is_not_silently_regularized():
    invalid = SPATIAL.copy()
    invalid[0, 0] = -1.0
    adaptive = AdaptiveCornerCovariance()
    with pytest.raises(ValueError, match="positive definite"):
        adaptive.update(_observation(BASE_CORNERS, covariance=invalid), 0.0)
