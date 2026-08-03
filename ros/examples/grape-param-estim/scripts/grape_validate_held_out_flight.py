#!/usr/bin/env python3
"""Run posterior validation on a successful flight excluded from fitting."""

import argparse
import signal
import sys

from grape_param_estim.held_out_validation_cli import run_request
from grape_param_estim.progress import CancellationToken, ProgressCancelled


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    cancellation = CancellationToken()

    def request_cancel(signum, _frame):
        cancellation.cancel("signal_{}".format(signum))

    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, request_cancel)
    try:
        destination = run_request(
            arguments.request,
            arguments.output,
            cancellation_token=cancellation,
        )
    except ProgressCancelled as error:
        print(str(error), file=sys.stderr)
        return 130
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    print(str(destination))
    return 0


if __name__ == "__main__":
    sys.exit(main())
