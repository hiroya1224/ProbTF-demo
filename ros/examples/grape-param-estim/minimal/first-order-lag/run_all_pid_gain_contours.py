#!/usr/bin/env python3
"""Run exact adaptive single-group PID survival maps for the failure bags."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
FAILURE_CASES = ("failure1", "failure2")
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
            *forwarded,
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


def _load_group_masks(case: str, group: str) -> tuple[dict, np.ndarray, np.ndarray]:
    directory = OUTPUTS / case / "pid_gain_contour_unconditional" / group
    payload = json.loads((directory / "gain_contour.json").read_text(encoding="utf-8"))
    with np.load(directory / "gain_contour.npz", allow_pickle=False) as archive:
        stable = np.asarray(archive["recorded_failure_stable_mask"], dtype=bool)
        valid = np.asarray(archive["recorded_failure_pole_valid_mask"], dtype=bool)
    return payload, stable, valid


def _write_case_summary(case: str, groups: Sequence[str]) -> None:
    rows = []
    reference = None
    population = None
    controller = None
    for group in groups:
        payload, stable, valid = _load_group_masks(case, group)
        if payload["schema"] != "grape-param-estim/first-order-lag-pid-group-stability/v6":
            raise RuntimeError("Old conditional maps cannot be combined with unconditional maps")
        current = (stable, valid)
        if reference is not None and not all(np.array_equal(a, b) for a, b in zip(reference, current)):
            raise RuntimeError("Recorded failure controller results differ between groups")
        if population is not None and population != payload["sample_coordinates_sha256"]:
            raise RuntimeError("Plant sample population/order differs between groups")
        if controller is not None and controller != payload["fixed_controller_gains"]:
            raise RuntimeError("Fixed full failure controller differs between groups")
        reference = current
        population = payload["sample_coordinates_sha256"]
        controller = payload["fixed_controller_gains"]
        rows.append(payload)
    if not rows:
        raise ValueError("No group maps were generated")
    summary = {
        "schema": "grape-param-estim/first-order-lag-pid-group-stability/v6-summary",
        "case_name": case, "groups": list(groups),
        "sample_count": int(reference[0].size),
        "sample_coordinates_sha256": population,
        "fixed_controller_gains": controller,
        "recorded_failure_controller": rows[0]["recorded_failure_controller"],
        "group_slices": [{"group": group, "gain_domain": row["gain_domain"],
                          "projections": row["projections"]}
                         for group, row in zip(groups, rows)],
        "interpretation": "Every slice uses every original plant sample and the same full failure-record controller for its other nine gains. Arbitrary combinations of separately favorable group triples are not certified; evaluate all 12 gains jointly.",
    }
    output = OUTPUTS / case / "pid_gain_contour_unconditional"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse(argv: Sequence[str]) -> tuple[int, tuple[str, ...], tuple[str, ...], list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--case-workers", type=int, default=DEFAULT_CASE_WORKERS)
    parser.add_argument("--case", action="append", choices=FAILURE_CASES)
    parser.add_argument("--group", action="append", choices=GROUPS)
    parsed, forwarded = parser.parse_known_args(argv)
    if parsed.case_workers <= 0:
        parser.error("--case-workers must be positive")
    cases = tuple(parsed.case or FAILURE_CASES)
    groups = tuple(parsed.group or GROUPS)
    return int(parsed.case_workers), cases, groups, forwarded


def main() -> int:
    case_workers, cases, groups, forwarded = _parse(sys.argv[1:])
    if case_workers == 1:
        for case in cases:
            return_code = _run_case(case, groups, forwarded)
            if return_code != 0:
                print(
                    f"{case} failed with return code {return_code}",
                    file=sys.stderr,
                )
                return return_code
        for case in cases:
            _write_case_summary(case, groups)
        return 0

    failures = []
    with ThreadPoolExecutor(max_workers=min(case_workers, len(cases))) as executor:
        pending = {
            executor.submit(_run_case, case, groups, forwarded): case
            for case in cases
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
    for case in cases:
        _write_case_summary(case, groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
