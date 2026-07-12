from dataclasses import dataclass

from probtf.distributions import TransformDistributionStamped
from probtf.kernels.base import TransformKernelExpression


@dataclass(frozen=True)
class InverseEdgeKernel(TransformKernelExpression):
    """Lazy inverse view of the same physical latent edge.

    Conditionally, ``z_child = R(Q).T (z_parent - X)``.  No independent
    inverse distribution is constructed.
    """

    edge_record: TransformDistributionStamped

    def __post_init__(self):
        if not isinstance(self.edge_record, TransformDistributionStamped):
            raise TypeError("edge_record must be TransformDistributionStamped.")

    def latent_dependency_ids(self):
        return frozenset(
            (self.edge_record.edge_id,) + self.edge_record.provenance.derived_from_edge_ids
        )

