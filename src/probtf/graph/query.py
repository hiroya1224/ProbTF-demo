from probtf.distributions import TransformDistributionStamped
from probtf.graph.buffer import EdgeTimeBuffer
from probtf.graph.edge import EdgeView, PhysicalEdge
from probtf.graph.path import PathExpression
from probtf.graph.status import GraphErrorCode, TemporalResolutionError
from probtf.graph.topology import ProbTfTopology
from probtf.temporal import (
    AuthorityConflictPolicy,
    ParentChangePolicy,
    TemporalPolicy,
)


class ProbTfGraph:
    """TF-style forest plus timestamped Prob-TF edge histories."""

    def __init__(
        self,
        max_records_per_edge=None,
        authority_conflict_policy=AuthorityConflictPolicy.REJECT,
        parent_change_policy=ParentChangePolicy.REJECT,
    ):
        self.topology = ProbTfTopology(parent_change_policy)
        self.max_records_per_edge = max_records_per_edge
        self.authority_conflict_policy = authority_conflict_policy
        self._buffers = {}

    def edge_buffer(self, edge_id):
        return self._buffers[edge_id]

    def insert(self, record):
        if not isinstance(record, TransformDistributionStamped):
            raise TypeError("record must be a TransformDistributionStamped.")
        physical = PhysicalEdge(record.edge_id, record.parent_frame_id, record.child_frame_id)
        self.topology.add_edge(physical)
        buffer = self._buffers.get(record.edge_id)
        if buffer is None:
            buffer = EdgeTimeBuffer(
                self.max_records_per_edge,
                self.authority_conflict_policy,
            )
            self._buffers[record.edge_id] = buffer
        buffer.insert(record)

    def _resolved_traversal(self, target_frame, source_frame, stamp, policy, tolerance):
        traversal = self.topology.traversal(source_frame, target_frame)
        if not traversal:
            resolved_stamp = 0.0 if stamp is None else float(stamp)
            return resolved_stamp, ()

        if policy is TemporalPolicy.LATEST_COMMON:
            dynamic_buffers = [
                self._buffers[edge.edge_id]
                for edge, _ in traversal
                if not self._buffers[edge.edge_id].is_static
            ]
            if dynamic_buffers:
                common = min(buffer.latest_stamp for buffer in dynamic_buffers)
                earliest_common = max(buffer.earliest_stamp for buffer in dynamic_buffers)
                if common < earliest_common:
                    raise TemporalResolutionError(
                        GraphErrorCode.TEMPORAL_OUT_OF_RANGE,
                        "Path edge histories have no common availability interval.",
                    )
            else:
                common = 0.0 if stamp is None else float(stamp)
            resolved = tuple(
                (
                    edge,
                    direction,
                    self._buffers[edge.edge_id].resolve(common, TemporalPolicy.LATEST),
                )
                for edge, direction in traversal
            )
            return common, resolved

        resolved = tuple(
            (
                edge,
                direction,
                self._buffers[edge.edge_id].resolve(stamp, policy, tolerance),
            )
            for edge, direction in traversal
        )
        if stamp is not None:
            resolved_stamp = float(stamp)
        else:
            resolved_stamp = max(item.sample_stamp for _, _, item in resolved)
        return resolved_stamp, resolved

    def lookup_path(
        self,
        target_frame,
        source_frame,
        stamp=None,
        policy=TemporalPolicy.EXACT,
        tolerance=0.0,
    ):
        resolved_stamp, traversal = self._resolved_traversal(
            target_frame,
            source_frame,
            stamp,
            policy,
            tolerance,
        )
        return PathExpression(
            source_frame,
            target_frame,
            resolved_stamp,
            tuple(
                EdgeView(edge.edge_id, direction, resolved.sample_stamp)
                for edge, direction, resolved in traversal
            ),
        )

    def resolved_records(self, path):
        if not isinstance(path, PathExpression):
            raise TypeError("path must be a PathExpression.")
        return tuple(
            self._buffers[view.edge_id].record_at_sample_stamp(view.sample_stamp)
            for view in path.edge_views
        )

    def lookup_kernel(
        self,
        target_frame,
        source_frame,
        stamp=None,
        policy=TemporalPolicy.EXACT,
        tolerance=0.0,
    ):
        from probtf.kernels import kernel_from_path

        path = self.lookup_path(target_frame, source_frame, stamp, policy, tolerance)
        return kernel_from_path(path, self.resolved_records(path))

