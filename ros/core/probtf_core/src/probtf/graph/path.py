import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from probtf.distributions import TransformDistributionStamped
from probtf.distributions.validation import frame_id
from probtf.graph.edge import EdgeView
from probtf.graph.status import DependencyUnresolvedError
from probtf.temporal import TemporalEvaluationResult


@dataclass(frozen=True)
class PathExpression:
    """A resolved source-to-target traversal over physical latent edges."""

    source_frame: str
    target_frame: str
    resolved_stamp: float
    edge_views: Tuple[EdgeView, ...]
    diagnostics: Tuple[str, ...] = ()
    edge_evaluations: Tuple[Optional[TemporalEvaluationResult], ...] = ()
    _record_snapshot: Tuple[TransformDistributionStamped, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

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
        evaluations = tuple(self.edge_evaluations)
        if evaluations and len(evaluations) != len(views):
            raise ValueError("edge_evaluations must be empty or align with edge_views.")
        if any(
            item is not None and not isinstance(item, TemporalEvaluationResult)
            for item in evaluations
        ):
            raise TypeError(
                "edge_evaluations must contain TemporalEvaluationResult objects or None."
            )
        records = tuple(self._record_snapshot)
        if records and len(records) != len(views):
            raise ValueError("_record_snapshot must be empty or align with edge_views.")
        if any(not isinstance(item, TransformDistributionStamped) for item in records):
            raise TypeError("_record_snapshot must contain TransformDistributionStamped records.")
        object.__setattr__(self, "edge_evaluations", evaluations)
        object.__setattr__(self, "_record_snapshot", records)

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
            tuple(reversed(self.edge_evaluations)),
            tuple(reversed(self._record_snapshot)),
        )

    def reduce_adjacent_inverses(self):
        stack = []
        diagnostic_stack = []
        evaluation_stack = []
        record_stack = []
        aligned_diagnostics = len(self.diagnostics) == len(self.edge_views)
        aligned_evaluations = len(self.edge_evaluations) == len(self.edge_views)
        aligned_records = len(self._record_snapshot) == len(self.edge_views)
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
                if aligned_evaluations:
                    evaluation_stack.pop()
                if aligned_records:
                    record_stack.pop()
            else:
                stack.append(view)
                if aligned_diagnostics:
                    diagnostic_stack.append(self.diagnostics[index])
                if aligned_evaluations:
                    evaluation_stack.append(self.edge_evaluations[index])
                if aligned_records:
                    record_stack.append(self._record_snapshot[index])
        return PathExpression(
            self.source_frame,
            self.target_frame,
            self.resolved_stamp,
            tuple(stack),
            tuple(diagnostic_stack) if aligned_diagnostics else self.diagnostics,
            tuple(evaluation_stack) if aligned_evaluations else self.edge_evaluations,
            tuple(record_stack) if aligned_records else self._record_snapshot,
        )

    @property
    def temporal_evaluations(self):
        """Return model/sample evaluations in path order, omitting plain edges."""

        return tuple(item for item in self.edge_evaluations if item is not None)

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
