"""Sampling for native v2 transform components and mixtures."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from probtf.distributions import (
    BinghamOrientation,
    DistributionStatus,
    OrientationKind,
    TransformComponent,
    TransformDistribution,
)
from probtf.geometry import quat_to_rotmat


@dataclass(frozen=True)
class TransformSampleBatch:
    translations: np.ndarray
    rotations_wxyz: np.ndarray

    def __post_init__(self):
        translations = np.asarray(self.translations, dtype=float)
        rotations = np.asarray(self.rotations_wxyz, dtype=float)
        if translations.ndim != 2 or translations.shape[1:] != (3,):
            raise ValueError("translations must have shape (N, 3).")
        if rotations.shape != (translations.shape[0], 4):
            raise ValueError("rotations_wxyz must have shape (N, 4).")
        if not np.all(np.isfinite(translations)) or not np.all(np.isfinite(rotations)):
            raise ValueError("transform samples must contain only finite values.")
        if not np.allclose(
            np.linalg.norm(rotations, axis=1),
            1.0,
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError("rotations_wxyz must contain unit quaternions.")
        translations = translations.copy()
        rotations = rotations.copy()
        translations.setflags(write=False)
        rotations.setflags(write=False)
        object.__setattr__(self, "translations", translations)
        object.__setattr__(self, "rotations_wxyz", rotations)

    @property
    def count(self):
        return self.translations.shape[0]


def _count(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("sample count must be a non-negative integer.")
    count = int(value)
    if count < 0 or count != value:
        raise ValueError("sample count must be a non-negative integer.")
    return count


def _generator(rng):
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _readonly(values):
    output = np.asarray(values, dtype=float)
    output.setflags(write=False)
    return output


def _bingham_proposal_scale(eigenvalues):
    values = np.asarray(eigenvalues, dtype=float)

    def equation(scale):
        return float(np.sum(1.0 / (scale + 2.0 * values)) - 1.0)

    return float(brentq(equation, 1e-12, 4.0 + 1e-12))


def _sample_finite_bingham(parameter, count, rng):
    precision = -np.asarray(parameter, dtype=float)
    minimum = float(np.linalg.eigvalsh(precision)[0])
    precision -= minimum * np.eye(4)
    eigenvalues = np.linalg.eigvalsh(precision)
    scale = _bingham_proposal_scale(eigenvalues)
    omega = np.eye(4) + 2.0 * precision / scale
    inverse_sqrt = np.linalg.cholesky(np.linalg.inv(omega))
    envelope = np.exp(-(4.0 - scale) / 2.0) * (4.0 / scale) ** 2.0

    accepted = []
    remaining = count
    while remaining:
        # A count-independent batch keeps fixed-seed sample prefixes stable:
        # requesting N and M>N samples yields the same first N accepted
        # orientations.  Temporal common-random-number comparisons rely on
        # that sample_id contract.
        batch_size = 256
        gaussian = inverse_sqrt @ rng.normal(size=(4, batch_size))
        candidates = (gaussian / np.linalg.norm(gaussian, axis=0)).T
        proposal = np.einsum("ni,ij,nj->n", candidates, omega, candidates) ** -2.0
        target = np.exp(-np.einsum("ni,ij,nj->n", candidates, precision, candidates))
        probability = np.clip(target / (envelope * proposal), 0.0, 1.0)
        selected = candidates[rng.random(batch_size) < probability]
        if len(selected):
            accepted.append(selected[:remaining])
            remaining -= min(len(selected), remaining)
    return np.concatenate(accepted, axis=0)


def sample_bingham_orientation(orientation, count, rng=None):
    """Sample unit quaternions from a v2 orientation law."""

    if not isinstance(orientation, BinghamOrientation):
        raise TypeError("orientation must be a BinghamOrientation.")
    count = _count(count)
    generator = _generator(rng)
    if count == 0:
        return _readonly(np.empty((0, 4), dtype=float))
    if orientation.kind is OrientationKind.DIRAC:
        samples = np.repeat(orientation.reference_quaternion_wxyz[None, :], count, axis=0)
        return _readonly(samples)
    if orientation.kind is OrientationKind.UNIFORM:
        samples = generator.normal(size=(count, 4))
        return _readonly(samples / np.linalg.norm(samples, axis=1, keepdims=True))
    return _readonly(
        _sample_finite_bingham(
            orientation.backend_parameter_matrix(),
            count,
            generator,
        )
    )


def sample_transform_component(component, count, rng=None):
    """Jointly sample orientation and conditional Gaussian translation."""

    if not isinstance(component, TransformComponent):
        raise TypeError("component must be a TransformComponent.")
    count = _count(count)
    generator = _generator(rng)
    rotations = sample_bingham_orientation(component.orientation, count, generator)
    translations = np.asarray(
        [component.conditional_translation_mean(quaternion) for quaternion in rotations],
        dtype=float,
    ).reshape(count, 3)
    covariance = component.translation.residual_covariance
    if count and not np.allclose(covariance, 0.0, rtol=0.0, atol=0.0):
        translations += generator.multivariate_normal(np.zeros(3), covariance, size=count)
    return TransformSampleBatch(translations, rotations)


def sample_transform_distribution(distribution, count, rng=None):
    """Sample a component according to normalized positive mixture weights."""

    if not isinstance(distribution, TransformDistribution):
        raise TypeError("distribution must be a TransformDistribution.")
    count = _count(count)
    normalized = distribution.normalize_weights()
    if normalized.status is not DistributionStatus.OK:
        raise ValueError("Cannot sample a {} transform distribution.".format(normalized.status.value))
    generator = _generator(rng)
    weights = np.array([item.weight for item in normalized.components], dtype=float)
    choices = generator.choice(len(normalized.components), size=count, p=weights)
    translations = np.empty((count, 3), dtype=float)
    rotations = np.empty((count, 4), dtype=float)
    for index, item in enumerate(normalized.components):
        selected = np.flatnonzero(choices == index)
        if not len(selected):
            continue
        samples = sample_transform_component(item.component, len(selected), generator)
        translations[selected] = samples.translations
        rotations[selected] = samples.rotations_wxyz
    return TransformSampleBatch(translations, rotations)


def sample_transform_distribution_components(distribution, component_indices, rng=None):
    """Sample explicitly selected mixture components in the supplied order.

    This is used by dependency-aware temporal sample paths.  The same
    component-index vector can be reused across multiple records so matching
    ``sample_id`` components remain the same latent realization.
    """

    if not isinstance(distribution, TransformDistribution):
        raise TypeError("distribution must be a TransformDistribution.")
    indices = np.asarray(component_indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("component_indices must be a one-dimensional integer array.")
    count = indices.shape[0]
    normalized = distribution.normalize_weights()
    if normalized.status is not DistributionStatus.OK:
        raise ValueError("Cannot sample a {} transform distribution.".format(normalized.status.value))
    if count and (int(np.min(indices)) < 0 or int(np.max(indices)) >= len(normalized.components)):
        raise ValueError("component_indices contains an out-of-range component.")
    generator = _generator(rng)
    translations = np.empty((count, 3), dtype=float)
    rotations = np.empty((count, 4), dtype=float)
    for index, item in enumerate(normalized.components):
        selected = np.flatnonzero(indices == index)
        if not len(selected):
            continue
        samples = sample_transform_component(item.component, len(selected), generator)
        translations[selected] = samples.translations
        rotations[selected] = samples.rotations_wxyz
    return TransformSampleBatch(translations, rotations)


def apply_transform_samples(samples, points, inverse=False):
    """Apply sampled transforms to one fixed point or corresponding points."""

    if not isinstance(samples, TransformSampleBatch):
        raise TypeError("samples must be a TransformSampleBatch.")
    points = np.asarray(points, dtype=float)
    if points.shape == (3,):
        points = np.broadcast_to(points, (samples.count, 3))
    elif points.shape != samples.translations.shape:
        raise ValueError("points must have shape (3,) or (N, 3) matching samples.")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values.")
    rotations = np.asarray(
        [quat_to_rotmat(quaternion) for quaternion in samples.rotations_wxyz],
        dtype=float,
    ).reshape(samples.count, 3, 3)
    if inverse:
        output = np.einsum("nji,nj->ni", rotations, points - samples.translations)
    else:
        output = np.einsum("nij,nj->ni", rotations, points) + samples.translations
    return _readonly(output)
