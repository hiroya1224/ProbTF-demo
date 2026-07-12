"""Translation conditioned affinely on column-major ``vec(R(q))``."""

from dataclasses import dataclass

import numpy as np

from probtf.distributions.validation import immutable_array, immutable_symmetric_matrix
from probtf.geometry import rotation_vector_from_quaternion


@dataclass(frozen=True)
class ConditionalGaussianTranslation:
    mean_at_reference: np.ndarray
    residual_covariance: np.ndarray
    rotation_coupling: np.ndarray

    def __post_init__(self):
        object.__setattr__(
            self,
            "mean_at_reference",
            immutable_array(self.mean_at_reference, (3,), "mean_at_reference"),
        )
        object.__setattr__(
            self,
            "residual_covariance",
            immutable_symmetric_matrix(
                self.residual_covariance,
                3,
                "residual_covariance",
                positive_semidefinite=True,
            ),
        )
        object.__setattr__(
            self,
            "rotation_coupling",
            immutable_array(self.rotation_coupling, (3, 9), "rotation_coupling"),
        )

    def conditional_mean(self, quat_wxyz, reference_quaternion_wxyz=None):
        """Return the conditional mean for ``quat_wxyz``.

        The coupling reference is component-level orientation metadata.  It is
        therefore explicit here; callers normally use
        :meth:`TransformComponent.conditional_translation_mean`.
        """

        if reference_quaternion_wxyz is None:
            if np.allclose(self.rotation_coupling, 0.0, rtol=0.0, atol=0.0):
                return self.mean_at_reference.copy()
            raise ValueError("reference_quaternion_wxyz is required for coupled translation.")
        rotation_vector = rotation_vector_from_quaternion(quat_wxyz)
        reference_vector = rotation_vector_from_quaternion(reference_quaternion_wxyz)
        return self.mean_at_reference + self.rotation_coupling @ (rotation_vector - reference_vector)

    def conditional_covariance(self, quat_wxyz):
        rotation_vector_from_quaternion(quat_wxyz)
        return self.residual_covariance.copy()

