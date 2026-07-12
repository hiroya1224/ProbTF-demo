import math
from dataclasses import dataclass
from enum import Enum

from probtf.distributions.validation import frame_id, identifier


class EdgeDirection(Enum):
    FORWARD = "forward"
    INVERSE = "inverse"

    def inverse(self):
        return EdgeDirection.INVERSE if self is EdgeDirection.FORWARD else EdgeDirection.FORWARD


@dataclass(frozen=True)
class PhysicalEdge:
    """A latent transform mapping child coordinates into parent coordinates."""

    edge_id: str
    parent_frame_id: str
    child_frame_id: str

    def __post_init__(self):
        parent = frame_id(self.parent_frame_id, "parent_frame_id")
        child = frame_id(self.child_frame_id, "child_frame_id")
        if parent == child:
            raise ValueError("A physical edge cannot connect a frame to itself.")
        object.__setattr__(self, "edge_id", identifier(self.edge_id, "edge_id"))
        object.__setattr__(self, "parent_frame_id", parent)
        object.__setattr__(self, "child_frame_id", child)


@dataclass(frozen=True)
class EdgeView:
    edge_id: str
    direction: EdgeDirection
    sample_stamp: float

    def __post_init__(self):
        object.__setattr__(self, "edge_id", identifier(self.edge_id, "edge_id"))
        if not isinstance(self.direction, EdgeDirection):
            raise TypeError("direction must be an EdgeDirection.")
        stamp = float(self.sample_stamp)
        if not math.isfinite(stamp) or stamp < 0.0:
            raise ValueError("sample_stamp must be finite and non-negative.")
        object.__setattr__(self, "sample_stamp", stamp)

    def inverse(self):
        return EdgeView(self.edge_id, self.direction.inverse(), self.sample_stamp)
