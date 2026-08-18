#!/usr/bin/env python3
"""Trace first-order PID stability contours for the three recorded bags."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
CASES = ("failure1", "failure2", "success")
GROUPS = ("xy", "z", "roll_pitch", "yaw")
DEFAULT_CASE_WORKERS = 1


def _run_case(
    case: str,
    groups: Sequence[str],
    forwarded: Sequence[str],
) -> int:
    for group in groups:
        command = [
            sys.executable,
            str(HERE / "pid_gain_contour.py"),
            "--estimate-json",
            str(OUTPUTS / case / "estimate.json"),
            "--group",
            group,
        ]
        if case != "success":
            command.extend(
                (
                    "--success-json",
                    str(OUTPUTS / "success" / "estimate.json"),
                )
            )
        command.extend(forwarded)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


def _write_case_summary(case: str, groups: Sequence[str]) -> None:
    rows = []
    for group in groups:
        path = OUTPUTS / case / "pid_gain_contour" / group / "gain_contour.json"
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    summary = {
        "schema": "grape-param-estim/first-order-lag-pid-gain-contour/v1-summary",
        "case_name": case,
        "groups": list(groups),
        "rows": rows,
    }
    path = OUTPUTS / case / "pid_gain_contour" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse(argv: Sequence[str]) -> tuple[int, tuple[str, ...], list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--case-workers", type=int, default=DEFAULT_CASE_WORKERS)
    parser.add_argument("--group", action="append", choices=GROUPS)
    parser.add_argument("--all-groups", action="store_true")
    parsed, forwarded = parser.parse_known_args(argv)
    if parsed.case_workers <= 0:
        parser.error("--case-workers must be positive")
    if parsed.all_groups and parsed.group:
        parser.error("--all-groups and --group cannot be used together")
    groups = (
        GROUPS
        if parsed.all_groups
        else tuple(parsed.group or ("roll_pitch",))
    )
    return int(parsed.case_workers), groups, forwarded


def main() -> int:
    case_workers, groups, forwarded = _parse(sys.argv[1:])
    failures = []
    with ThreadPoolExecutor(max_workers=min(case_workers, len(CASES))) as executor:
        pending = {
            executor.submit(_run_case, case, groups, forwarded): case
            for case in CASES
        }
        for future in as_completed(pending):
            case = pending[future]
            return_code = int(future.result())
            if return_code != 0:
                failures.append((case, return_code))
    if failures:
        for case, return_code in failures:
            print(f"{case} failed with return code {return_code}", file=sys.stderr)
        return failures[0][1]
    for case in CASES:
        _write_case_summary(case, groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
