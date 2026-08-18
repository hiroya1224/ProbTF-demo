#!/usr/bin/env python3
"""Run the isolated first-order estimator for failure1, failure2, and success."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
CASES = ("failure1", "failure2", "success")


def main() -> int:
    forwarded = sys.argv[1:]
    for case in CASES:
        command = [
            sys.executable,
            str(HERE / "estimate.py"),
            "--case",
            case,
            *forwarded,
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
