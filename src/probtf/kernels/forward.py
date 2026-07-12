from dataclasses import dataclass

from probtf.distributions import TransformDistributionStamped
from probtf.kernels.base import TransformKernelExpression


@dataclass(frozen=True)
class ForwardEdgeKernel(TransformKernelExpression):
    """Apply the physical law ``z_parent = R(Q) z_child + X``."""

    edge_record: TransformDistributionStamped

    def __post_init__(self):
        if not isinstance(self.edge_record, TransformDistributionStamped):
            raise TypeError("edge_record must be TransformDistributionStamped.")

    def latent_dependency_ids(self):
        return frozenset(
            (self.edge_record.edge_id,) + self.edge_record.provenance.derived_from_edge_ids
        )

