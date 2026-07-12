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

    def __post_init__(self):
        views = tuple(self.edge_views)
        if any(not isinstance(view, EdgeView) for view in views):
            raise TypeError("edge_views must contain only EdgeView objects.")
        object.__setattr__(self, "source_frame", frame_id(self.source_frame, "source_frame"))
        object.__setattr__(self, "target_frame", frame_id(self.target_frame, "target_frame"))
        object.__setattr__(self, "resolved_stamp", float(self.resolved_stamp))
        object.__setattr__(self, "edge_views", views)

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
        )

    def reduce_adjacent_inverses(self):
        stack = []
        for view in self.edge_views:
            if (
                stack
                and stack[-1].edge_id == view.edge_id
                and stack[-1].direction is view.direction.inverse()
            ):
                stack.pop()
            else:
                stack.append(view)
        return PathExpression(
            self.source_frame,
            self.target_frame,
            self.resolved_stamp,
            tuple(stack),
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

