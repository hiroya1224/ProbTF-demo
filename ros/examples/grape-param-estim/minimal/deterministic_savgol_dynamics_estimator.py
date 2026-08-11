#!/usr/bin/env python3
"""Compatibility entry point for the dimensionless SG experiment."""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from dimensionless_savgol_experiment import main as _core_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--mode" not in arguments:
        arguments = ["--mode", "fit"] + arguments
    return _core_main(arguments)


if __name__ == "__main__":
    sys.exit(main())
