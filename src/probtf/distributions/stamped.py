from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from probtf.distributions.status import RepresentativeKind
from probtf.distributions.transform_distribution import TransformDistribution
from probtf.distributions.validation import frame_id, identifier
from probtf.geometry import DeterministicTransform
from probtf.provenance import ApproximationInfo, TransformProvenance


@dataclass(frozen=True)
class TransformDistributionStamped:
    """A timestamped physical edge mapping child coordinates into parent."""

    parent_frame_id: str
    child_frame_id: str
    stamp: float
    edge_id: str
    authority: str
    distribution: TransformDistribution
    representative: Optional[DeterministicTransform] = None
    representative_kind: RepresentativeKind = RepresentativeKind.NONE
    provenance: TransformProvenance = field(default_factory=TransformProvenance)
    is_static: bool = False
    approximation: ApproximationInfo = field(default_factory=ApproximationInfo)

    def __post_init__(self):
        parent = frame_id(self.parent_frame_id, "parent_frame_id")
        child = frame_id(self.child_frame_id, "child_frame_id")
        if parent == child:
            raise ValueError("parent_frame_id and child_frame_id must differ.")
        stamp = float(self.stamp)
        if not np.isfinite(stamp) or stamp < 0.0:
            raise ValueError("stamp must be finite and non-negative.")
        if not isinstance(self.distribution, TransformDistribution):
            raise TypeError("distribution must be a TransformDistribution.")
        if self.representative is not None and not isinstance(
            self.representative,
            DeterministicTransform,
        ):
            raise TypeError("representative must be a DeterministicTransform or None.")
        if not isinstance(self.representative_kind, RepresentativeKind):
            raise TypeError("representative_kind must be RepresentativeKind.")
        if self.representative is None and self.representative_kind is not RepresentativeKind.NONE:
            raise ValueError("representative_kind must be NONE when representative is absent.")
        if self.representative is not None and self.representative_kind is RepresentativeKind.NONE:
            raise ValueError("A representative requires an explicit representative_kind.")
        if not isinstance(self.provenance, TransformProvenance):
            raise TypeError("provenance must be TransformProvenance.")
        if not isinstance(self.approximation, ApproximationInfo):
            raise TypeError("approximation must be ApproximationInfo.")

        object.__setattr__(self, "parent_frame_id", parent)
        object.__setattr__(self, "child_frame_id", child)
        object.__setattr__(self, "stamp", stamp)
        object.__setattr__(self, "edge_id", identifier(self.edge_id, "edge_id"))
        object.__setattr__(self, "authority", identifier(self.authority, "authority"))
        object.__setattr__(self, "is_static", bool(self.is_static))
