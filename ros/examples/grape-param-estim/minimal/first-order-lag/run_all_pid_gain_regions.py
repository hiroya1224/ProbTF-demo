#!/usr/bin/env python3
"""Run first-order PID gain-region analysis for all three recorded cases."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"


def _run(case: str, forwarded: list[str], *, success_overlay: bool) -> int:
    command = [
        sys.executable,
        str(HERE / "pid_gain_region.py"),
        "--estimate-json",
        str(OUTPUTS / case / "estimate.json"),
    ]
    if success_overlay:
        command.extend(
            (
                "--success-json",
                str(OUTPUTS / "success" / "estimate.json"),
            )
        )
    command.extend(forwarded)
    return int(subprocess.run(command, check=False).returncode)


def main() -> int:
    forwarded = sys.argv[1:]
    for case in ("failure1", "failure2"):
        result = _run(case, forwarded, success_overlay=True)
        if result != 0:
            return result
    return _run("success", forwarded, success_overlay=False)


if __name__ == "__main__":
    raise SystemExit(main())
