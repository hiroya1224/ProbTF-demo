"""Physical-edge bindings to versioned Gaussian latent factors."""

from dataclasses import dataclass
import math

import numpy as np

from probtf.dependency.gaussian import immutable_matrix
from probtf.geometry import DeterministicTransform


POSE_PERTURBATION_CONVENTION = "translation_parent_rotation_right_local"


def _identifier(value, name):
    result = str(value).strip()
    if not result:
        raise ValueError("{} must not be empty.".format(name))
    return result


@dataclass(frozen=True)
class EdgeLatentBinding:
    edge_id: str
    factor_id: str
    sensitivity: np.ndarray
    factor_version: int
    linearization_stamp: float
    linearization_pose: DeterministicTransform
    perturbation_convention: str = POSE_PERTURBATION_CONVENTION

    def __post_init__(self):
        edge_id = _identifier(self.edge_id, "edge_id")
        factor_id = _identifier(self.factor_id, "factor_id")
        sensitivity = np.asarray(self.sensitivity, dtype=float)
        if sensitivity.ndim != 2 or sensitivity.shape[0] != 6 or sensitivity.shape[1] < 1:
            raise ValueError("sensitivity must have shape (6, d) with d >= 1.")
        sensitivity = immutable_matrix(
            sensitivity,
            sensitivity.shape,
            "sensitivity",
        )
        version = int(self.factor_version)
        if version < 1 or version != self.factor_version:
            raise ValueError("factor_version must be a positive integer.")
        stamp = float(self.linearization_stamp)
        if not math.isfinite(stamp) or stamp < 0.0:
            raise ValueError("linearization_stamp must be finite and non-negative.")
        if not isinstance(self.linearization_pose, DeterministicTransform):
            raise TypeError("linearization_pose must be DeterministicTransform.")
        convention = str(self.perturbation_convention).strip()
        if convention != POSE_PERTURBATION_CONVENTION:
            raise ValueError(
                "Unsupported perturbation convention {!r}.".format(convention)
            )
        object.__setattr__(self, "edge_id", edge_id)
        object.__setattr__(self, "factor_id", factor_id)
        object.__setattr__(self, "sensitivity", sensitivity)
        object.__setattr__(self, "factor_version", version)
        object.__setattr__(self, "linearization_stamp", stamp)
        object.__setattr__(self, "perturbation_convention", convention)

    @property
    def latent_dimension(self):
        return int(self.sensitivity.shape[1])


__all__ = ["EdgeLatentBinding", "POSE_PERTURBATION_CONVENTION"]
