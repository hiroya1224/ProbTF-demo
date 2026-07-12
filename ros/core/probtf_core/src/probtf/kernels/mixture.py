from dataclasses import dataclass

from probtf.kernels.base import TransformKernelExpression
from probtf.kernels.forward import ForwardEdgeKernel
from probtf.kernels.inverse import InverseEdgeKernel


@dataclass(frozen=True)
class MixtureTransformKernel(TransformKernelExpression):
    """Preserve a multi-component edge without reducing or closing it."""

    edge_kernel: TransformKernelExpression

    def __post_init__(self):
        if not isinstance(self.edge_kernel, (ForwardEdgeKernel, InverseEdgeKernel)):
            raise TypeError("edge_kernel must be a forward or inverse edge kernel.")
        if len(self.edge_kernel.edge_record.distribution.components) < 2:
            raise ValueError("MixtureTransformKernel requires at least two raw components.")

    @property
    def edge_record(self):
        return self.edge_kernel.edge_record

    def latent_dependency_ids(self):
        return self.edge_kernel.latent_dependency_ids()

