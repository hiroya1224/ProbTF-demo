import math
from dataclasses import dataclass
from typing import Tuple

from probtf.distributions.validation import frame_id
from probtf.graph.edge import EdgeView
from probtf.graph.status import DependencyUnresolvedError


@dataclass(frozen=True)
class PathExpression:
    """A resolved source-to-target traversal over physical latent edges."""

    source_frame: str
    target_frame: str
    resolved_stamp: float
    edge_views: Tuple[EdgeView, ...]
    diagnostics: Tuple[str, ...] = ()

    def __post_init__(self):
        views = tuple(self.edge_views)
        if any(not isinstance(view, EdgeView) for view in views):
            raise TypeError("edge_views must contain only EdgeView objects.")
        object.__setattr__(self, "source_frame", frame_id(self.source_frame, "source_frame"))
        object.__setattr__(self, "target_frame", frame_id(self.target_frame, "target_frame"))
        stamp = float(self.resolved_stamp)
        if not math.isfinite(stamp) or stamp < 0.0:
            raise ValueError("resolved_stamp must be finite and non-negative.")
        object.__setattr__(self, "resolved_stamp", stamp)
        object.__setattr__(self, "edge_views", views)
        object.__setattr__(self, "diagnostics", tuple(str(item) for item in self.diagnostics))

    def __iter__(self):
        return iter(self.edge_views)

    def __len__(self):
        return len(self.edge_views)

    def reversed(self):
        return PathExpression(
            self.target_frame,
            self.source_frame,
            self.resolved_stamp,
            tuple(view.inverse() for view in reversed(self.edge_views)),
            tuple(reversed(self.diagnostics)),
        )

    def reduce_adjacent_inverses(self):
        stack = []
        diagnostic_stack = []
        aligned_diagnostics = len(self.diagnostics) == len(self.edge_views)
        for index, view in enumerate(self.edge_views):
            if (
                stack
                and stack[-1].edge_id == view.edge_id
                and stack[-1].direction is view.direction.inverse()
                and stack[-1].sample_stamp == view.sample_stamp
            ):
                stack.pop()
                if aligned_diagnostics:
                    diagnostic_stack.pop()
            else:
                stack.append(view)
                if aligned_diagnostics:
                    diagnostic_stack.append(self.diagnostics[index])
        return PathExpression(
            self.source_frame,
            self.target_frame,
            self.resolved_stamp,
            tuple(stack),
            tuple(diagnostic_stack) if aligned_diagnostics else self.diagnostics,
        )

    def repeated_edge_ids(self):
        seen = set()
        repeated = []
        for view in self.edge_views:
            if view.edge_id in seen and view.edge_id not in repeated:
                repeated.append(view.edge_id)
            seen.add(view.edge_id)
        return tuple(repeated)

    def assert_dependencies_resolved(self):
        repeated = self.repeated_edge_ids()
        if repeated:
            raise DependencyUnresolvedError(repeated)
