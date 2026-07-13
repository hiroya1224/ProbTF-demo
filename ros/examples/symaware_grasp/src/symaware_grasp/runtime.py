"""ROS runtime lookup helpers for application messages naming v2 records."""

from probtf.kernels import (
    ComposedTransformKernel,
    ForwardEdgeKernel,
    MixtureTransformKernel,
)
from probtf.temporal import TemporalPolicy


def stamp_to_seconds(stamp):
    if hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        return float(stamp.secs) + float(stamp.nsecs) * 1e-9
    return float(stamp)


def lookup_direct_record(
    listener,
    target_frame,
    source_frame,
    stamp=None,
    policy=TemporalPolicy.EXACT,
    timeout=2.0,
):
    """Resolve one forward graph edge through the public listener lookup API."""

    if not listener.wait_for_lookup(
        target_frame,
        source_frame,
        stamp=stamp,
        policy=policy,
        timeout=timeout,
    ):
        raise RuntimeError(
            "Timed out waiting for ProbTF lookup {} <- {}.".format(
                target_frame,
                source_frame,
            )
        )
    kernel = listener.lookup_kernel(
        target_frame,
        source_frame,
        stamp=stamp,
        policy=policy,
    )
    if not isinstance(kernel, ComposedTransformKernel) or len(kernel.kernels) != 1:
        raise ValueError("Application belief messages must name one direct ProbTF edge.")
    edge_kernel = kernel.kernels[0]
    if isinstance(edge_kernel, MixtureTransformKernel):
        edge_kernel = edge_kernel.edge_kernel
    if not isinstance(edge_kernel, ForwardEdgeKernel):
        raise ValueError("Application belief messages must resolve in the forward direction.")
    return edge_kernel.edge_record


def lookup_message_record(listener, transform_message, timeout=2.0):
    return lookup_direct_record(
        listener,
        transform_message.header.frame_id,
        transform_message.child_frame_id,
        stamp=stamp_to_seconds(transform_message.header.stamp),
        policy=TemporalPolicy.EXACT,
        timeout=timeout,
    )
