#!/usr/bin/env python3
"""Entry point for the minimal recorded-control estimators."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument(
        "--method",
        choices=("deterministic", "probabilistic"),
        default="deterministic",
    )
    selection, remaining = selector.parse_known_args(raw_arguments)
    if selection.method == "deterministic":
        from deterministic_estimator import main as selected_main
    else:
        from probabilistic_estimator import main as selected_main
    return selected_main(remaining)


if __name__ == "__main__":
    sys.exit(main())
