"""Qt-free platform boundary for interrupting estimator workers."""

from __future__ import annotations

import os
from pathlib import Path
import signal
from typing import Callable, Mapping


_BATCH_CHECKPOINT_SCHEMA = "grape-param-estim/batch-estimation-checkpoint/v1"


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
    if manifest_reader is None and cancellation_marker is None:
        from grape_param_estim.artifact_io import read_json, write_json_atomic

        manifest_path = bundle_root / "manifest.json"
        if not manifest_path.is_file():
            checkpoint_root = bundle_root.parent / (
                ".{}-batch-checkpoint".format(bundle_root.name)
            )
            checkpoint_manifest = checkpoint_root / "manifest.json"
            if not checkpoint_manifest.is_file():
                return False
            raw = read_json(checkpoint_manifest)
            if raw.get("schema") != _BATCH_CHECKPOINT_SCHEMA:
                return False
            status = raw.get("status")
            if status == "cancelled":
                return True
            if status not in {"core_complete", "sampling"}:
                return False
            cancellation_reason = str(reason)
            if not cancellation_reason or cancellation_reason.strip() != cancellation_reason:
                raise ValueError("cancellation reason must be canonical text")
            updated = dict(raw)
            updated["status"] = "cancelled"
            updated["cancellation_reason"] = cancellation_reason
            write_json_atomic(checkpoint_manifest, updated)
            return True

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
