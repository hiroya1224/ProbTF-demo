from dataclasses import dataclass

from probtf.distributions import TransformDistributionStamped
from probtf.kernels.base import TransformKernelExpression
from probtf.temporal.provenance import temporal_dependency_ids


@dataclass(frozen=True)
class ForwardEdgeKernel(TransformKernelExpression):
    """Apply the physical law ``z_parent = R(Q) z_child + X``."""

    edge_record: TransformDistributionStamped

    def __post_init__(self):
        if not isinstance(self.edge_record, TransformDistributionStamped):
            raise TypeError("edge_record must be TransformDistributionStamped.")

    def latent_dependency_ids(self):
        identifiers = {self.edge_record.edge_id}
        identifiers.update(self.edge_record.provenance.derived_from_edge_ids)
        identifiers.update(temporal_dependency_ids(self.edge_record))
        for component in self.edge_record.distribution.components:
            identifiers.update(component.provenance.derived_from_edge_ids)
        return frozenset(identifiers)
