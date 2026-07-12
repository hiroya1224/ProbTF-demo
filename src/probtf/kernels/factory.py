from probtf.graph import EdgeDirection, PathExpression
from probtf.kernels.composed import ComposedTransformKernel
from probtf.kernels.forward import ForwardEdgeKernel
from probtf.kernels.inverse import InverseEdgeKernel
from probtf.kernels.mixture import MixtureTransformKernel


def kernel_from_path(path, edge_records):
    if not isinstance(path, PathExpression):
        raise TypeError("path must be a PathExpression.")
    records = tuple(edge_records)
    if len(records) != len(path.edge_views):
        raise ValueError("edge_records must match path.edge_views.")

    kernels = []
    for view, record in zip(path.edge_views, records):
        if record.edge_id != view.edge_id or record.stamp != view.sample_stamp:
            raise ValueError("Resolved record does not match its EdgeView.")
        edge_kernel = (
            ForwardEdgeKernel(record)
            if view.direction is EdgeDirection.FORWARD
            else InverseEdgeKernel(record)
        )
        if len(record.distribution.components) > 1:
            edge_kernel = MixtureTransformKernel(edge_kernel)
        kernels.append(edge_kernel)
    return ComposedTransformKernel(tuple(kernels), path)

