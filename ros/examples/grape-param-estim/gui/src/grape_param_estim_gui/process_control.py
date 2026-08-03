"""Qt-free platform boundary for interrupting estimator workers."""

from __future__ import annotations

import os
from pathlib import Path
import signal
from typing import Callable, Mapping


def send_cooperative_interrupt(
    process_id: int,
    *,
    platform_name: str | None = None,
    signal_sender: Callable[[int, int], None] | None = None,
) -> bool:
    """Send SIGINT to one direct child on POSIX, or report no support."""

    identifier = int(process_id)
    platform = os.name if platform_name is None else str(platform_name)
    if platform != "posix" or identifier <= 0:
        return False
    sender = os.kill if signal_sender is None else signal_sender
    sender(identifier, signal.SIGINT)
    return True


def finalize_cancelled_bundle(
    root: str | Path,
    reason: str,
    *,
    manifest_reader: Callable[[str | Path], Mapping[str, object]] | None = None,
    cancellation_marker: Callable[[str | Path, str], object] | None = None,
) -> bool:
    """Make an existing writing/cancelled worker manifest authoritative."""

    bundle_root = Path(root).expanduser().resolve()
    if not (bundle_root / "manifest.json").is_file():
        return False
    if manifest_reader is None and cancellation_marker is None:
        from grape_param_estim.artifact_io import read_json

        raw = read_json(bundle_root / "manifest.json")
        schema = raw.get("schema")
        if schema == "grape-param-estim/diagonal-wrench-q-estimate/v1":
            from grape_param_estim.diagonal_q_artifact import (
                mark_diagonal_q_artifact_cancelled,
                read_diagonal_q_manifest,
            )

            reader = read_diagonal_q_manifest
            marker = mark_diagonal_q_artifact_cancelled
        elif schema == (
            "grape-param-estim/fixed-q-augmented-parameter-estimate/v1"
        ):
            from grape_param_estim.augmented_parameter_artifact import (
                mark_augmented_parameter_artifact_cancelled,
                read_augmented_parameter_manifest,
            )

            reader = read_augmented_parameter_manifest
            marker = mark_augmented_parameter_artifact_cancelled
        else:
            from grape_param_estim.artifact_io import (
                mark_bundle_cancelled,
                read_manifest,
            )

            reader = read_manifest
            marker = mark_bundle_cancelled
    elif manifest_reader is None or cancellation_marker is None:
        from grape_param_estim.artifact_io import (  # local cross-env boundary
            mark_bundle_cancelled,
            read_manifest,
        )

        reader = read_manifest if manifest_reader is None else manifest_reader
        marker = (
            mark_bundle_cancelled
            if cancellation_marker is None
            else cancellation_marker
        )
    else:
        reader = manifest_reader
        marker = cancellation_marker
    manifest = reader(bundle_root)
    status = manifest.get("status")
    if status == "cancelled":
        return True
    if status != "writing":
        return False
    marker(bundle_root, str(reason))
    return True


__all__ = ["finalize_cancelled_bundle", "send_cooperative_interrupt"]
