#!/usr/bin/env python3
"""Run first-order PID gain-region analysis for all three recorded cases."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from core import GAIN_REGION_SCHEMA


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
CASES = ("failure1", "failure2", "success")
GROUPS = ("xy", "z", "roll_pitch", "yaw")
DEFAULT_WORKERS = min(12, os.cpu_count() or 1)


def _run_group(
    case: str,
    group: str,
    forwarded: Sequence[str],
    *,
    success_overlay: bool,
) -> int:
    command = [
        sys.executable,
        str(HERE / "pid_gain_region.py"),
        "--estimate-json",
        str(OUTPUTS / case / "estimate.json"),
        "--group",
        group,
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


def _write_case_summary(case: str, groups: Sequence[str]) -> None:
    estimate_path = (OUTPUTS / case / "estimate.json").resolve()
    estimate = json.loads(estimate_path.read_text(encoding="utf-8"))
    output = OUTPUTS / case / "pid_gain_region"
    rows = [
        json.loads(
            (output / group / "gain_region.json").read_text(encoding="utf-8")
        )
        for group in groups
    ]
    summary = {
        "schema": GAIN_REGION_SCHEMA + "-summary",
        "case_name": case,
        "estimate_json": str(estimate_path),
        "first_order_time_constant_seconds": float(
            estimate["actuator_model"]["thrust_time_constant_seconds"]
        ),
        "alpha": float(rows[0]["alpha"]),
        "covariance": str(rows[0]["covariance"]),
        "rows": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_arguments(argv: Sequence[str]) -> tuple[int, tuple[str, ...], list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--group", action="append", choices=GROUPS)
    parsed, forwarded = parser.parse_known_args(argv)
    if parsed.workers <= 0:
        parser.error("--workers must be positive")
    groups = tuple(parsed.group or GROUPS)
    return int(parsed.workers), groups, forwarded


def main() -> int:
    workers, groups, forwarded = _parse_arguments(sys.argv[1:])
    tasks = [
        (case, group, case != "success")
        for case in CASES
        for group in groups
    ]
    failures: list[tuple[str, str, int]] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        pending = {
            executor.submit(
                _run_group,
                case,
                group,
                forwarded,
                success_overlay=success_overlay,
            ): (case, group)
            for case, group, success_overlay in tasks
        }
        for future in as_completed(pending):
            case, group = pending[future]
            result = int(future.result())
            if result != 0:
                failures.append((case, group, result))
    if failures:
        for case, group, result in failures:
            print(
                f"{case}/{group} failed with return code {result}",
                file=sys.stderr,
            )
        return failures[0][2]
    for case in CASES:
        _write_case_summary(case, groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
