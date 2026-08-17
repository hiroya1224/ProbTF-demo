"""Immutable Gaussian latent-state and observation contracts."""

from dataclasses import dataclass, field
import math
from typing import Tuple

import numpy as np

from probtf.provenance import Provenance


def _identifier(value, name):
    result = str(value).strip()
    if not result:
        raise ValueError("{} must not be empty.".format(name))
    return result


def _stamp(value, name="stamp"):
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("{} must be finite and non-negative.".format(name))
    return result


def _version(value, name="version"):
    result = int(value)
    if result < 1 or result != value:
        raise ValueError("{} must be a positive integer.".format(name))
    return result


def immutable_vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size < 1 or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a non-empty finite vector.".format(name))
    result = result.copy()
    result.setflags(write=False)
    return result


def immutable_matrix(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite matrix with shape {}.".format(name, shape))
    result = result.copy()
    result.setflags(write=False)
    return result


def immutable_psd_matrix(value, dimension, name, *, positive_definite=False):
    result = np.asarray(value, dtype=float)
    if result.shape != (dimension, dimension) or not np.all(np.isfinite(result)):
        raise ValueError(
            "{} must be a finite {}x{} matrix.".format(name, dimension, dimension)
        )
    result = 0.5 * (result + result.T)
    eigenvalues, eigenvectors = np.linalg.eigh(result)
    scale = max(1.0, float(np.linalg.norm(result, ord=np.inf)))
    tolerance = 1.0e-10 * scale
    if positive_definite:
        if float(eigenvalues[0]) <= np.finfo(float).eps * scale:
            raise ValueError("{} must be positive definite.".format(name))
    elif float(eigenvalues[0]) < -tolerance:
        raise ValueError("{} must be positive semidefinite.".format(name))
    clipped = np.maximum(eigenvalues, 0.0)
    result = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    result = 0.5 * (result + result.T)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GaussianLatentFactor:
    """One versioned local Gaussian latent variable."""

    factor_id: str
    mean: np.ndarray
    covariance: np.ndarray
    stamp: float
    version: int
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self):
        factor_id = _identifier(self.factor_id, "factor_id")
        mean = immutable_vector(self.mean, "mean")
        covariance = immutable_psd_matrix(
            self.covariance,
            mean.size,
            "covariance",
        )
        if not isinstance(self.provenance, Provenance):
            raise TypeError("provenance must be Provenance.")
        object.__setattr__(self, "factor_id", factor_id)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "stamp", _stamp(self.stamp))
        object.__setattr__(self, "version", _version(self.version))

    @property
    def dimension(self):
        return int(self.mean.size)


@dataclass(frozen=True)
class GaussianObservationFactor:
    """Linearized residual/noise relation for an atomic posterior update.

    residual + H delta_eta is conditioned to zero. Jacobian blocks are in the
    same order as latent_factor_ids.
    """

    observation_id: str
    latent_factor_ids: Tuple[str, ...]
    residual: np.ndarray
    jacobian_blocks: Tuple[np.ndarray, ...]
    noise_covariance: np.ndarray
    stamp: float
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self):
        observation_id = _identifier(self.observation_id, "observation_id")
        factor_ids = tuple(
            _identifier(value, "latent_factor_ids")
            for value in self.latent_factor_ids
        )
        if not factor_ids:
            raise ValueError("latent_factor_ids must not be empty.")
        if len(set(factor_ids)) != len(factor_ids):
            raise ValueError("latent_factor_ids must be unique.")
        residual = immutable_vector(self.residual, "residual")
        blocks = tuple(np.asarray(value, dtype=float) for value in self.jacobian_blocks)
        if len(blocks) != len(factor_ids):
            raise ValueError("jacobian_blocks must align with latent_factor_ids.")
        immutable_blocks = []
        for index, block in enumerate(blocks):
            if (
                block.ndim != 2
                or block.shape[0] != residual.size
                or block.shape[1] < 1
                or not np.all(np.isfinite(block))
            ):
                raise ValueError(
                    "jacobian_blocks[{}] must have shape ({}, d) and be finite.".format(
                        index,
                        residual.size,
                    )
                )
            immutable = block.copy()
            immutable.setflags(write=False)
            immutable_blocks.append(immutable)
        noise = immutable_psd_matrix(
            self.noise_covariance,
            residual.size,
            "noise_covariance",
            positive_definite=True,
        )
        if not isinstance(self.provenance, Provenance):
            raise TypeError("provenance must be Provenance.")
        object.__setattr__(self, "observation_id", observation_id)
        object.__setattr__(self, "latent_factor_ids", factor_ids)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "jacobian_blocks", tuple(immutable_blocks))
        object.__setattr__(self, "noise_covariance", noise)
        object.__setattr__(self, "stamp", _stamp(self.stamp))


@dataclass(frozen=True)
class GaussianUpdateResult:
    observation_id: str
    prior_versions: Tuple[Tuple[str, int], ...]
    posterior_versions: Tuple[Tuple[str, int], ...]
    innovation_covariance: np.ndarray
    kalman_gain: np.ndarray
    store_revision: int

    def __post_init__(self):
        innovation = np.asarray(self.innovation_covariance, dtype=float)
        gain = np.asarray(self.kalman_gain, dtype=float)
        if innovation.ndim != 2 or innovation.shape[0] != innovation.shape[1]:
            raise ValueError("innovation_covariance must be square.")
        if gain.ndim != 2 or gain.shape[1] != innovation.shape[0]:
            raise ValueError("kalman_gain has incompatible dimensions.")
        if not np.all(np.isfinite(innovation)) or not np.all(np.isfinite(gain)):
            raise ValueError("update diagnostics must be finite.")
        innovation = innovation.copy()
        gain = gain.copy()
        innovation.setflags(write=False)
        gain.setflags(write=False)
        object.__setattr__(
            self,
            "observation_id",
            _identifier(self.observation_id, "observation_id"),
        )
        object.__setattr__(self, "prior_versions", tuple(self.prior_versions))
        object.__setattr__(self, "posterior_versions", tuple(self.posterior_versions))
        object.__setattr__(self, "innovation_covariance", innovation)
        object.__setattr__(self, "kalman_gain", gain)
        object.__setattr__(self, "store_revision", int(self.store_revision))


__all__ = [
    "GaussianLatentFactor",
    "GaussianObservationFactor",
    "GaussianUpdateResult",
    "immutable_matrix",
    "immutable_psd_matrix",
    "immutable_vector",
]
