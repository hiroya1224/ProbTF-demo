"""Temporal, full-covariance uncertainty for ordered AprilTag corners.

The estimator in this module deliberately changes only the covariance of the
current observation.  It never smooths or delays the observed corners: doing
so would turn an image measurement model into an implicit motion filter.
"""

from dataclasses import dataclass, field
import math

import numpy as np

from prob_artag_detector.models import MarkerObservation


def _symmetric(value):
    matrix = np.asarray(value, dtype=float)
    return 0.5 * (matrix + matrix.T)


def _positive_part(value):
    """Return the positive-semidefinite part of a symmetric matrix."""

    eigenvalues, eigenvectors = np.linalg.eigh(_symmetric(value))
    return _symmetric(
        (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    )


def _spatial_covariance(value, minimum_variance):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (8, 8) or not np.all(np.isfinite(matrix)):
        raise ValueError("corner covariance must be a finite 8x8 matrix.")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-10):
        raise ValueError("corner covariance must be symmetric.")
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetric(matrix))
    if eigenvalues[0] <= 0.0:
        raise ValueError("corner covariance must be positive definite.")
    eigenvalues = np.maximum(eigenvalues, minimum_variance)
    return _symmetric((eigenvectors * eigenvalues) @ eigenvectors.T)


def constant_velocity_innovation(older, previous, current):
    """Return a timestamp-aware, variance-normalized second difference.

    Each argument is ``(stamp, corners)`` where corners has shape ``(4, 2)``.
    A signal that is linear in time cancels exactly.  If the three measurements
    have the same independent covariance ``R``, the returned innovation also
    has covariance ``R``.
    """

    t0, z0 = older
    t1, z1 = previous
    t2, z2 = current
    t0, t1, t2 = float(t0), float(t1), float(t2)
    dt0 = t1 - t0
    dt1 = t2 - t1
    if not np.all(np.isfinite([t0, t1, t2])) or dt0 <= 0.0 or dt1 <= 0.0:
        raise ValueError("corner timestamps must be finite and strictly increasing.")
    z0 = np.asarray(z0, dtype=float).reshape(8)
    z1 = np.asarray(z1, dtype=float).reshape(8)
    z2 = np.asarray(z2, dtype=float).reshape(8)
    if not np.all(np.isfinite(np.concatenate((z0, z1, z2)))):
        raise ValueError("corner coordinates must be finite.")
    ratio = dt1 / dt0
    predicted = (1.0 + ratio) * z1 - ratio * z0
    residual = z2 - predicted
    variance_scale = 1.0 + (1.0 + ratio) ** 2 + ratio ** 2
    return residual / math.sqrt(variance_scale), predicted, residual, ratio


def affine_explained_fraction(predicted_corners, residual):
    """Fraction of corner-displacement energy explained by one affine warp."""

    corners = np.asarray(predicted_corners, dtype=float).reshape(4, 2)
    displacement = np.asarray(residual, dtype=float).reshape(4, 2)
    total = float(np.sum(displacement * displacement))
    if not np.isfinite(total) or total <= np.finfo(float).eps:
        return 0.0
    design = np.column_stack((corners, np.ones(4, dtype=float)))
    coefficients, _, _, _ = np.linalg.lstsq(design, displacement, rcond=None)
    explained = design @ coefficients
    fraction = float(np.sum(explained * explained)) / total
    return min(1.0, max(0.0, fraction))


@dataclass(frozen=True)
class CornerNoiseDiagnostics:
    """Summary of one adaptive covariance update."""

    status: str
    temporal_ready: bool
    accepted_samples: int
    equivalent_sigma_px: float
    affine_fraction: float = 0.0


@dataclass
class _Track:
    history: list = field(default_factory=list)
    accepted_samples: int = 0
    innovation_mean: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=float)
    )
    innovation_covariance: np.ndarray = field(
        default_factory=lambda: np.zeros((8, 8), dtype=float)
    )
    innovation_m2: np.ndarray = field(
        default_factory=lambda: np.zeros((8, 8), dtype=float)
    )
    spatial_mean: np.ndarray = field(
        default_factory=lambda: np.zeros((8, 8), dtype=float)
    )
    last_seen: float = float("-inf")


class AdaptiveCornerCovariance:
    """Estimate excess corner jitter independently for each decoded tag.

    ``spatial_covariance`` is the covariance already inferred from the current
    image (or ``observation.image_covariance`` when omitted).  Temporal
    innovations estimate only covariance in excess of that spatial model.
    ``maximum_excess_sigma_px`` caps the added temporal term and never narrows
    the current spatial covariance.  Affine innovations are learned by default;
    ``freeze_affine_motion`` trades that conservatism for motion rejection.
    """

    def __init__(
        self,
        minimum_sigma_px=0.5,
        maximum_excess_sigma_px=5.0,
        warmup_samples=8,
        temporal_half_life_sec=0.5,
        shrinkage=0.25,
        motion_gate_px=1.5,
        motion_gate_edge_fraction=0.02,
        affine_motion_fraction=0.90,
        freeze_affine_motion=False,
        huber_chi2=20.09,
        hard_outlier_px=8.0,
        hard_outlier_edge_fraction=0.15,
        max_gap_sec=0.25,
        max_dt_ratio=2.5,
        track_ttl_sec=2.0,
    ):
        self.minimum_sigma_px = float(minimum_sigma_px)
        self.maximum_excess_sigma_px = float(maximum_excess_sigma_px)
        self.warmup_samples = int(warmup_samples)
        self.temporal_half_life_sec = float(temporal_half_life_sec)
        self.shrinkage = float(shrinkage)
        self.motion_gate_px = float(motion_gate_px)
        self.motion_gate_edge_fraction = float(motion_gate_edge_fraction)
        self.affine_motion_fraction = float(affine_motion_fraction)
        self.freeze_affine_motion = bool(freeze_affine_motion)
        self.huber_chi2 = float(huber_chi2)
        self.hard_outlier_px = float(hard_outlier_px)
        self.hard_outlier_edge_fraction = float(hard_outlier_edge_fraction)
        self.max_gap_sec = float(max_gap_sec)
        self.max_dt_ratio = float(max_dt_ratio)
        self.track_ttl_sec = float(track_ttl_sec)
        finite_positive = (
            self.minimum_sigma_px,
            self.maximum_excess_sigma_px,
            self.temporal_half_life_sec,
            self.motion_gate_px,
            self.motion_gate_edge_fraction,
            self.huber_chi2,
            self.hard_outlier_px,
            self.hard_outlier_edge_fraction,
            self.max_gap_sec,
            self.max_dt_ratio,
            self.track_ttl_sec,
        )
        if not np.all(np.isfinite(finite_positive)) or min(finite_positive) <= 0.0:
            raise ValueError("corner uncertainty scales must be positive and finite.")
        if self.warmup_samples < 2:
            raise ValueError("warmup_samples must be at least two.")
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage must lie in [0, 1].")
        if not 0.0 <= self.affine_motion_fraction <= 1.0:
            raise ValueError("affine_motion_fraction must lie in [0, 1].")
        if self.max_dt_ratio < 1.0:
            raise ValueError("max_dt_ratio must be at least one.")
        self._minimum_variance = self.minimum_sigma_px ** 2
        self._maximum_excess_variance = self.maximum_excess_sigma_px ** 2
        self._tracks = {}

    @staticmethod
    def _key(observation):
        return str(observation.family), int(observation.marker_id)

    @staticmethod
    def _edge_length(corners):
        points = np.asarray(corners, dtype=float).reshape(4, 2)
        edges = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(edges, axis=1)
        return float(np.median(lengths))

    def _new_track(self, stamp, corners, covariance):
        track = _Track(last_seen=float(stamp))
        track.history.append(
            (float(stamp), np.asarray(corners, dtype=float).copy(), covariance.copy())
        )
        return track

    def _evict_stale(self, stamp, current_key):
        stale = [
            key
            for key, track in self._tracks.items()
            if key != current_key
            and stamp >= track.last_seen
            and stamp - track.last_seen > self.track_ttl_sec
        ]
        for key in stale:
            del self._tracks[key]

    def reset(self, marker_id=None, family=None):
        """Reset all tracks, or tracks matching the supplied ID/family."""

        if marker_id is None and family is None:
            self._tracks.clear()
            return
        marker_id = None if marker_id is None else int(marker_id)
        family = None if family is None else str(family)
        keys = [
            key
            for key in self._tracks
            if (marker_id is None or key[1] == marker_id)
            and (family is None or key[0] == family)
        ]
        for key in keys:
            del self._tracks[key]

    def _excess_covariance(self, track):
        if track.accepted_samples < self.warmup_samples:
            return np.zeros((8, 8), dtype=float)
        raw_excess = _symmetric(
            track.innovation_covariance - track.spatial_mean
        )
        diagonal = np.diag(np.diag(raw_excess))
        shrunk_excess = (
            (1.0 - self.shrinkage) * raw_excess
            + self.shrinkage * diagonal
        )
        excess = _positive_part(shrunk_excess)
        eigenvalues, eigenvectors = np.linalg.eigh(excess)
        eigenvalues = np.minimum(
            eigenvalues, self._maximum_excess_variance
        )
        return _symmetric((eigenvectors * eigenvalues) @ eigenvectors.T)

    def _output(self, observation, covariance, status, track, affine_fraction=0.0):
        bounded = _spatial_covariance(covariance, self._minimum_variance)
        result = MarkerObservation(
            observation.marker_id,
            observation.corners_px,
            bounded,
            observation.family,
        )
        diagnostics = CornerNoiseDiagnostics(
            status=str(status),
            temporal_ready=track.accepted_samples >= self.warmup_samples,
            accepted_samples=int(track.accepted_samples),
            equivalent_sigma_px=float(math.sqrt(np.trace(bounded) / 8.0)),
            affine_fraction=float(affine_fraction),
        )
        return result, diagnostics

    def _update_statistics(self, track, innovation, spatial, dt):
        count = track.accepted_samples + 1
        if count == 1:
            track.innovation_mean = innovation.copy()
            track.innovation_m2.fill(0.0)
            track.innovation_covariance.fill(0.0)
            track.spatial_mean = spatial.copy()
        elif count <= self.warmup_samples:
            delta = innovation - track.innovation_mean
            track.innovation_mean += delta / float(count)
            track.innovation_m2 += np.outer(
                delta, innovation - track.innovation_mean
            )
            track.innovation_covariance = _symmetric(
                track.innovation_m2 / float(count - 1)
            )
            track.spatial_mean += (spatial - track.spatial_mean) / float(count)
        else:
            alpha = 1.0 - math.exp(
                -math.log(2.0) * float(dt) / self.temporal_half_life_sec
            )
            alpha = min(1.0, max(np.finfo(float).eps, alpha))
            delta = innovation - track.innovation_mean
            new_mean = track.innovation_mean + alpha * delta
            track.innovation_covariance = _symmetric(
                (1.0 - alpha) * track.innovation_covariance
                + alpha * np.outer(delta, innovation - new_mean)
            )
            track.innovation_mean = new_mean
            track.spatial_mean = _symmetric(
                (1.0 - alpha) * track.spatial_mean + alpha * spatial
            )
        track.accepted_samples = count

    def update(self, observation, stamp, spatial_covariance=None):
        """Return the current observation with adaptive full 8x8 covariance."""

        if not isinstance(observation, MarkerObservation):
            raise TypeError("observation must be MarkerObservation.")
        stamp = float(stamp)
        if not np.isfinite(stamp):
            raise ValueError("stamp must be finite.")
        spatial = (
            observation.image_covariance
            if spatial_covariance is None
            else spatial_covariance
        )
        spatial = _spatial_covariance(spatial, self._minimum_variance)
        key = self._key(observation)
        self._evict_stale(stamp, key)
        corners = np.asarray(observation.corners_px, dtype=float).reshape(4, 2)
        track = self._tracks.get(key)
        if track is None:
            track = self._new_track(stamp, corners, spatial)
            self._tracks[key] = track
            return self._output(observation, spatial, "warmup", track)

        elapsed = stamp - track.last_seen
        if elapsed <= 0.0 or elapsed > self.max_gap_sec:
            track = self._new_track(stamp, corners, spatial)
            self._tracks[key] = track
            return self._output(observation, spatial, "gap_reset", track)

        track.last_seen = stamp
        if len(track.history) < 2:
            track.history.append((stamp, corners.copy(), spatial.copy()))
            return self._output(observation, spatial, "warmup", track)

        older = track.history[-2]
        previous = track.history[-1]
        innovation, predicted, residual, ratio = constant_velocity_innovation(
            (older[0], older[1]),
            (previous[0], previous[1]),
            (stamp, corners),
        )
        if ratio > self.max_dt_ratio or ratio < 1.0 / self.max_dt_ratio:
            track = self._new_track(stamp, corners, spatial)
            self._tracks[key] = track
            return self._output(observation, spatial, "gap_reset", track)

        variance_scale = 1.0 + (1.0 + ratio) ** 2 + ratio ** 2
        innovation_spatial = _symmetric(
            (
                spatial
                + (1.0 + ratio) ** 2 * previous[2]
                + ratio ** 2 * older[2]
            )
            / variance_scale
        )
        residual_vectors = residual.reshape(4, 2)
        residual_rms = float(
            np.sqrt(np.mean(np.sum(residual_vectors * residual_vectors, axis=1)))
        )
        edge_length = self._edge_length(corners)
        motion_threshold = max(
            self.motion_gate_px, self.motion_gate_edge_fraction * edge_length
        )
        hard_threshold = max(
            self.hard_outlier_px,
            self.hard_outlier_edge_fraction * edge_length,
        )
        affine_fraction = affine_explained_fraction(
            predicted.reshape(4, 2), residual.reshape(4, 2)
        )
        track.history = track.history[-1:] + [
            (stamp, corners.copy(), spatial.copy())
        ]

        if residual_rms > hard_threshold:
            # Do not let a grossly inconsistent corner set contaminate future
            # covariance or prediction.  Two subsequent valid observations
            # establish a fresh velocity before innovations resume.
            track = _Track(last_seen=stamp)
            self._tracks[key] = track
            covariance = (
                spatial
                + np.eye(8, dtype=float) * self._maximum_excess_variance
            )
            return self._output(
                observation,
                covariance,
                "outlier",
                track,
                affine_fraction,
            )

        if (
            self.freeze_affine_motion
            and residual_rms > motion_threshold
            and affine_fraction >= self.affine_motion_fraction
        ):
            covariance = spatial + self._excess_covariance(track)
            return self._output(
                observation,
                covariance,
                "motion_frozen",
                track,
                affine_fraction,
            )

        reference = innovation_spatial + self._excess_covariance(track)
        centered = innovation - track.innovation_mean
        mahalanobis = float(centered @ np.linalg.solve(reference, centered))
        # Learn the initial scale before robust clipping; otherwise a genuinely
        # under-modelled jitter process can never grow beyond the spatial prior.
        if (
            track.accepted_samples >= self.warmup_samples
            and mahalanobis > self.huber_chi2
        ):
            scale = math.sqrt(self.huber_chi2 / mahalanobis)
            innovation = track.innovation_mean + scale * centered

        self._update_statistics(track, innovation, innovation_spatial, elapsed)
        covariance = spatial + self._excess_covariance(track)
        return self._output(
            observation,
            covariance,
            "accepted",
            track,
            affine_fraction,
        )
