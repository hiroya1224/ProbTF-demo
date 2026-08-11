#!/usr/bin/env python3
"""Compatibility entry point for dimensionless SG window ablation."""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from dimensionless_savgol_experiment import main as _core_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    prefix = ["--mode", "ablation"]
    if "--include-minimum-window" not in arguments:
        prefix.append("--include-minimum-window")
    return _core_main(prefix + arguments)


if __name__ == "__main__":
    sys.exit(main())
