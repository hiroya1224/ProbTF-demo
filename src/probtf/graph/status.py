from enum import Enum


class GraphErrorCode(Enum):
    UNKNOWN_FRAME = "unknown_frame"
    DISCONNECTED = "disconnected"
    CYCLE = "cycle"
    MULTIPLE_PARENT = "multiple_parent"
    DUPLICATE_EDGE = "duplicate_edge"
    EDGE_MISMATCH = "edge_mismatch"
    TEMPORAL_OUT_OF_RANGE = "temporal_out_of_range"
    TEMPORAL_STALE = "temporal_stale"
    STATIC_EDGE_CONFLICT = "static_edge_conflict"
    AUTHORITY_CONFLICT = "authority_conflict"
    UNSUPPORTED_TEMPORAL_POLICY = "unsupported_temporal_policy"
    DEPENDENCY_UNRESOLVED = "dependency_unresolved"


class ProbTfGraphError(RuntimeError):
    def __init__(self, code, message):
        if not isinstance(code, GraphErrorCode):
            raise TypeError("code must be a GraphErrorCode.")
        super().__init__(message)
        self.code = code


class TopologyError(ProbTfGraphError):
    pass


class TemporalResolutionError(ProbTfGraphError):
    pass


class DependencyUnresolvedError(ProbTfGraphError):
    def __init__(self, repeated_edge_ids):
        repeated = tuple(repeated_edge_ids)
        super().__init__(
            GraphErrorCode.DEPENDENCY_UNRESOLVED,
            "Repeated latent edge dependencies are unresolved: {}.".format(", ".join(repeated)),
        )
        self.repeated_edge_ids = repeated
