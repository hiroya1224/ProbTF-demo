from dataclasses import dataclass
from typing import Tuple

import numpy as np

from probtf.distributions.status import (
    DistributionStatus,
    RepresentativeKind,
    RepresentativePolicy,
)
from probtf.distributions.transform_component import TransformComponent
from probtf.geometry import DeterministicTransform
from probtf.provenance import ApproximationInfo, ApproximationKind


@dataclass(frozen=True)
class WeightDiagnostic:
    component_id: str
    raw_weight: float
    used_weight: float
    code: str


@dataclass(frozen=True)
class WeightedTransformComponent:
    component: TransformComponent
    weight: float


@dataclass(frozen=True)
class NormalizedTransformDistribution:
    components: Tuple[WeightedTransformComponent, ...]
    diagnostics: Tuple[WeightDiagnostic, ...] = ()
    status: DistributionStatus = DistributionStatus.OK


@dataclass(frozen=True)
class RepresentativeResult:
    transform: DeterministicTransform
    kind: RepresentativeKind
    approximation: ApproximationInfo


@dataclass(frozen=True)
class TransformDistribution:
    components: Tuple[TransformComponent, ...]

    def __post_init__(self):
        components = tuple(self.components)
        if any(not isinstance(component, TransformComponent) for component in components):
            raise TypeError("components must contain only TransformComponent objects.")
        identifiers = tuple(component.component_id for component in components)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("component_id values must be unique within a distribution.")
        object.__setattr__(self, "components", components)

    def status(self):
        raw_weights = np.array([component.raw_weight for component in self.components], dtype=float)
        if not np.all(np.isfinite(raw_weights)):
            return DistributionStatus.INVALID
        if not np.any(raw_weights > 0.0):
            return DistributionStatus.ZERO_MASS
        return DistributionStatus.OK

    def normalize_weights(self):
        status = self.status()
        if status is DistributionStatus.INVALID:
            diagnostics = tuple(
                WeightDiagnostic(component.component_id, component.raw_weight, 0.0, "NONFINITE_WEIGHT")
                for component in self.components
                if not np.isfinite(component.raw_weight)
            )
            return NormalizedTransformDistribution((), diagnostics, status)

        clamped = np.array([max(component.raw_weight, 0.0) for component in self.components])
        diagnostics = tuple(
            WeightDiagnostic(component.component_id, component.raw_weight, 0.0, "NEGATIVE_WEIGHT_CLAMPED")
            for component in self.components
            if component.raw_weight < 0.0
        )
        scale = float(np.max(clamped)) if clamped.size else 0.0
        if scale <= 0.0:
            return NormalizedTransformDistribution((), diagnostics, DistributionStatus.ZERO_MASS)
        scaled = clamped / scale
        total = float(np.sum(scaled))
        weighted = tuple(
            WeightedTransformComponent(component, float(weight / total))
            for component, weight in zip(self.components, scaled)
            if weight > 0.0
        )
        return NormalizedTransformDistribution(weighted, diagnostics, DistributionStatus.OK)

    def normalized_components(self):
        return self.normalize_weights().components

    def deterministic_transform(self):
        normalized = self.normalize_weights()
        if normalized.status is not DistributionStatus.OK or len(normalized.components) != 1:
            return None
        component = normalized.components[0].component
        return component.deterministic_transform() if component.is_deterministic else None

    def representative(self, policy):
        if not isinstance(policy, RepresentativePolicy):
            raise TypeError("policy must be a RepresentativePolicy.")
        deterministic = self.deterministic_transform()
        if deterministic is not None:
            return RepresentativeResult(
                deterministic,
                RepresentativeKind.EXACT_MAP,
                ApproximationInfo(),
            )
        if policy is RepresentativePolicy.EXACT_ONLY:
            raise ValueError("Distribution has no exact deterministic representative.")

        normalized = self.normalize_weights()
        if normalized.status is not DistributionStatus.OK:
            raise ValueError("A non-OK distribution has no representative.")
        selected = max(normalized.components, key=lambda item: item.weight).component
        mode = selected.orientation.mode_wxyz
        transform = DeterministicTransform(selected.conditional_translation_mean(mode), mode)
        return RepresentativeResult(
            transform,
            RepresentativeKind.COMPONENT_MODE_APPROXIMATION,
            ApproximationInfo(
                kind=ApproximationKind.REPRESENTATIVE_PROJECTION,
                lossy=True,
                detail="Mode of the highest normalized-weight component; this is not a global MAP claim.",
            ),
        )
