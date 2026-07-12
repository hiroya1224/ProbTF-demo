from dataclasses import dataclass, field

import numpy as np

from probtf.distributions.bingham_orientation import BinghamOrientation
from probtf.distributions.conditional_translation import ConditionalGaussianTranslation
from probtf.distributions.status import OrientationKind
from probtf.distributions.validation import identifier
from probtf.geometry import DeterministicTransform
from probtf.provenance import ApproximationInfo, ComponentProvenance


@dataclass(frozen=True)
class TransformComponent:
    """One joint orientation/conditional-translation pose hypothesis."""

    component_id: str
    raw_weight: float
    orientation: BinghamOrientation
    translation: ConditionalGaussianTranslation
    provenance: ComponentProvenance = field(default_factory=ComponentProvenance)
    approximation: ApproximationInfo = field(default_factory=ApproximationInfo)

    def __post_init__(self):
        object.__setattr__(self, "component_id", identifier(self.component_id, "component_id"))
        object.__setattr__(self, "raw_weight", float(self.raw_weight))
        if not isinstance(self.orientation, BinghamOrientation):
            raise TypeError("orientation must be a BinghamOrientation.")
        if not isinstance(self.translation, ConditionalGaussianTranslation):
            raise TypeError("translation must be a ConditionalGaussianTranslation.")
        if not isinstance(self.provenance, ComponentProvenance):
            raise TypeError("provenance must be ComponentProvenance.")
        if not isinstance(self.approximation, ApproximationInfo):
            raise TypeError("approximation must be ApproximationInfo.")

    def conditional_translation_mean(self, quat_wxyz):
        return self.translation.conditional_mean(
            quat_wxyz,
            self.orientation.reference_quaternion_wxyz,
        )

    @property
    def is_deterministic(self):
        return (
            self.orientation.kind is OrientationKind.DIRAC
            and np.allclose(self.translation.rotation_coupling, 0.0, rtol=0.0, atol=0.0)
            and np.allclose(self.translation.residual_covariance, 0.0, rtol=0.0, atol=0.0)
        )

    def deterministic_transform(self):
        if not self.is_deterministic:
            raise ValueError("Component is not deterministic.")
        return DeterministicTransform(
            self.translation.mean_at_reference,
            self.orientation.reference_quaternion_wxyz,
        )

