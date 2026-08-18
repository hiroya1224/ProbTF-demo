#!/usr/bin/env python3
"""Run adaptive single-group PID break maps for the recorded failure bags."""

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
            "--success-json",
            str(OUTPUTS / "success" / "estimate.json"),
            "--group",
            group,
            *forwarded,
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


def _load_group_masks(case: str, group: str) -> tuple[dict, np.ndarray, np.ndarray]:
    directory = OUTPUTS / case / "pid_gain_contour" / group
    payload = json.loads((directory / "gain_contour.json").read_text(encoding="utf-8"))
    with np.load(directory / "gain_contour.npz", allow_pickle=False) as archive:
        baseline = np.asarray(archive["success_baseline_stable_mask"], dtype=bool)
        caused = np.asarray(archive["caused_break_mask"], dtype=bool)
    return payload, baseline, caused


def _write_case_summary(case: str, groups: Sequence[str]) -> None:
    rows = []
    baseline_reference = None
    caused_masks = []
    for group in groups:
        payload, baseline, caused = _load_group_masks(case, group)
        if baseline_reference is None:
            baseline_reference = baseline
        elif not np.array_equal(baseline_reference, baseline):
            raise RuntimeError(
                "all-success baseline mask changed across PID groups; sample alignment is invalid"
            )
        rows.append(payload)
        caused_masks.append(caused)

    assert baseline_reference is not None
    cause_matrix = np.column_stack(caused_masks)
    cause_multiplicity = np.sum(cause_matrix, axis=1)
    group_summary = []
    for index, (group, row) in enumerate(zip(groups, rows)):
        caused = cause_matrix[:, index]
        unique = caused & (cause_multiplicity == 1)
        group_summary.append(
            {
                "group": group,
                "caused_break_count": int(np.count_nonzero(caused)),
                "caused_break_fraction_of_all_samples": float(np.mean(caused)),
                "unique_single_group_culprit_count": int(np.count_nonzero(unique)),
                "unique_single_group_culprit_fraction_of_all_samples": float(np.mean(unique)),
                "recorded_failure_group_gain": row["recorded_failure_group_gain"],
                "recorded_success_group_gain": row["recorded_success_group_gain"],
            }
        )

    summary = {
        "schema": "grape-param-estim/first-order-lag-pid-group-break/v2-summary",
        "case_name": case,
        "sample_count": int(baseline_reference.size),
        "groups": list(groups),
        "all_success_baseline": {
            "stable_count": int(np.count_nonzero(baseline_reference)),
            "unstable_count": int(np.count_nonzero(~baseline_reference)),
            "stable_fraction": float(np.mean(baseline_reference)),
        },
        "group_only_failure_interventions": group_summary,
        "single_group_cause_overlap": {
            "samples_broken_by_no_single_group": int(np.count_nonzero(cause_multiplicity == 0)),
            "samples_broken_by_exactly_one_single_group": int(np.count_nonzero(cause_multiplicity == 1)),
            "samples_broken_by_multiple_single_groups": int(np.count_nonzero(cause_multiplicity > 1)),
        },
        "interpretation": (
            "Each caused-break count uses the same plant samples. Other PID groups are fixed "
            "at success-flight gains, and only the named group is restored to this failure "
            "bag's recorded gain. A unique culprit sample is broken by exactly one of the "
            "four single-group interventions."
        ),
    }
    output = OUTPUTS / case / "pid_gain_contour"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
