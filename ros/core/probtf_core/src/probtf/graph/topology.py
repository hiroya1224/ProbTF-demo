from dataclasses import dataclass

from probtf.graph.edge import EdgeDirection, PhysicalEdge
from probtf.graph.status import GraphErrorCode, TopologyError
from probtf.temporal import ParentChangePolicy


@dataclass(frozen=True)
class TopologyDiagnostic:
    code: str
    detail: str


class ProbTfTopology:
    """A mutable TF-style forest of immutable physical edge identities."""

    def __init__(self, parent_change_policy=ParentChangePolicy.REJECT):
        if not isinstance(parent_change_policy, ParentChangePolicy):
            raise TypeError("parent_change_policy must be ParentChangePolicy.")
        self.parent_change_policy = parent_change_policy
        self._edges = {}
        self._parent_edge_by_child = {}
        self._frames = set()
        self._diagnostics = []

    @property
    def frames(self):
        return frozenset(self._frames)

    @property
    def diagnostics(self):
        return tuple(self._diagnostics)

    def edge(self, edge_id):
        return self._edges[edge_id]

    def _edge_to_parent(self, frame):
        edge_id = self._parent_edge_by_child.get(frame)
        return None if edge_id is None else self._edges[edge_id]

    def _would_create_cycle(self, edge):
        current = edge.parent_frame_id
        visited = set()
        while current in self._parent_edge_by_child:
            if current == edge.child_frame_id or current in visited:
                return True
            visited.add(current)
            current = self._edge_to_parent(current).parent_frame_id
        return current == edge.child_frame_id

    def add_edge(self, edge):
        if not isinstance(edge, PhysicalEdge):
            raise TypeError("edge must be a PhysicalEdge.")
        existing = self._edges.get(edge.edge_id)
        if existing is not None:
            if existing == edge:
                return
            raise TopologyError(
                GraphErrorCode.DUPLICATE_EDGE,
                "edge_id '{}' already identifies different endpoints.".format(edge.edge_id),
            )

        old_parent_id = self._parent_edge_by_child.get(edge.child_frame_id)
        if old_parent_id is not None:
            old_edge = self._edges[old_parent_id]
            if old_edge.parent_frame_id != edge.parent_frame_id:
                if self.parent_change_policy is ParentChangePolicy.REJECT:
                    raise TopologyError(
                        GraphErrorCode.MULTIPLE_PARENT,
                        "Frame '{}' already has parent '{}'.".format(
                            edge.child_frame_id,
                            old_edge.parent_frame_id,
                        ),
                    )
                del self._parent_edge_by_child[edge.child_frame_id]
                self._diagnostics.append(
                    TopologyDiagnostic(
                        "PARENT_REPLACED",
                        "Child '{}' changed parent from '{}' to '{}'.".format(
                            edge.child_frame_id,
                            old_edge.parent_frame_id,
                            edge.parent_frame_id,
                        ),
                    )
                )
            else:
                raise TopologyError(
                    GraphErrorCode.MULTIPLE_PARENT,
                    "Frame '{}' already has physical edge '{}'.".format(
                        edge.child_frame_id,
                        old_parent_id,
                    ),
                )

        if self._would_create_cycle(edge):
            if old_parent_id is not None and self.parent_change_policy is ParentChangePolicy.REPLACE_WITH_DIAGNOSTIC:
                self._parent_edge_by_child[edge.child_frame_id] = old_parent_id
                self._diagnostics.pop()
            raise TopologyError(GraphErrorCode.CYCLE, "Physical edge would create a cycle.")

        self._edges[edge.edge_id] = edge
        self._parent_edge_by_child[edge.child_frame_id] = edge.edge_id
        self._frames.update((edge.parent_frame_id, edge.child_frame_id))

    def traversal(self, source_frame, target_frame):
        if source_frame not in self._frames:
            raise TopologyError(GraphErrorCode.UNKNOWN_FRAME, "Unknown source frame '{}'.".format(source_frame))
        if target_frame not in self._frames:
            raise TopologyError(GraphErrorCode.UNKNOWN_FRAME, "Unknown target frame '{}'.".format(target_frame))
        if source_frame == target_frame:
            return ()

        source_ancestors = {}
        source_up = []
        current = source_frame
        while True:
            source_ancestors[current] = len(source_up)
            edge = self._edge_to_parent(current)
            if edge is None:
                break
            source_up.append(edge)
            current = edge.parent_frame_id

        target_up = []
        current = target_frame
        while current not in source_ancestors:
            edge = self._edge_to_parent(current)
            if edge is None:
                raise TopologyError(
                    GraphErrorCode.DISCONNECTED,
                    "Frames '{}' and '{}' are disconnected.".format(source_frame, target_frame),
                )
            target_up.append(edge)
            current = edge.parent_frame_id
        lca = current

        views = []
        current = source_frame
        while current != lca:
            edge = self._edge_to_parent(current)
            views.append((edge, EdgeDirection.FORWARD))
            current = edge.parent_frame_id
        views.extend((edge, EdgeDirection.INVERSE) for edge in reversed(target_up))
        return tuple(views)

