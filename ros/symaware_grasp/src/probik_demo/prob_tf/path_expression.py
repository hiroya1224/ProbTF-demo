from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeView:
    edge_id: str
    direction: int

    def inverse(self):
        return EdgeView(self.edge_id, -self.direction)


class PathExpression:
    def __init__(self, views):
        self.views = list(views)

    def reduce_adjacent_inverses(self):
        stack = []
        for view in self.views:
            if stack and stack[-1].edge_id == view.edge_id and stack[-1].direction == -view.direction:
                stack.pop()
            else:
                stack.append(view)
        return PathExpression(stack)

    def assert_no_repeated_edge_ids(self):
        edge_ids = [view.edge_id for view in self.views]
        if len(edge_ids) != len(set(edge_ids)):
            raise NotImplementedError(
                "The same physical edge appears multiple times after path reduction. "
                "This requires dependency-aware higher-order propagation and is not supported in the initial prototype."
            )

    def reversed(self):
        return PathExpression([view.inverse() for view in reversed(self.views)])

    def __iter__(self):
        return iter(self.views)

    def __len__(self):
        return len(self.views)

    def __repr__(self):
        return "PathExpression(%s)" % self.views
