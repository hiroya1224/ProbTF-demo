"""Request-file command line worker for lightweight rosbag inspection."""

import argparse
from pathlib import Path
import signal
import sys
from typing import Callable, Optional, Sequence

from grape_param_estim.inspection import (
    inspect_flights,
    load_inspection_request,
)
from grape_param_estim.progress import (
    CancellationToken,
    JsonlProgressWriter,
    ProgressCallback,
    ProgressCancelled,
)
from grape_param_estim.real_rosbag import RosbagArrayData


def run_request(
    request_path: str,
    output_path: str,
    progress_callback: Optional[ProgressCallback] = None,
    cancellation_token: Optional[CancellationToken] = None,
    arrays_loader: Optional[Callable[[str], RosbagArrayData]] = None,
) -> Path:
    """Load one strict request and write its inspection bundle."""

    cancellation = (
        CancellationToken()
        if cancellation_token is None
        else cancellation_token
    )
    if not isinstance(cancellation, CancellationToken):
        raise TypeError("cancellation_token must be a CancellationToken")
    cancellation.raise_if_cancelled()
    request = load_inspection_request(request_path)
    cancellation.raise_if_cancelled()
    return inspect_flights(
        request,
        output_path,
        arrays_loader=arrays_loader,
        progress_callback=progress_callback,
        cancellation_token=cancellation,
    )


def _signal_reason(signum: int) -> str:
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = str(signum)
    return "signal_{}".format(name)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Grape rosbag flights for the desktop GUI."
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)

    cancellation = CancellationToken()

    def request_cancel(signum, _frame):
        cancellation.cancel(_signal_reason(signum))

    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_cancel)
    try:
        output = run_request(
            arguments.request,
            arguments.output,
            progress_callback=JsonlProgressWriter(sys.stdout),
            cancellation_token=cancellation,
        )
        print(
            "inspection bundle complete: {}".format(output),
            file=sys.stderr,
        )
        return 0
    except ProgressCancelled as error:
        print(str(error), file=sys.stderr)
        return 2
    except Exception as error:  # pylint: disable=broad-except
        print("inspection failed: {}".format(error), file=sys.stderr)
        return 1
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_request"]
