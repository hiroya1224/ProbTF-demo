#!/usr/bin/env python3
"""Evaluate frozen LOBO observations and write auditable selection results."""

import argparse
from pathlib import Path
import subprocess
import sys

from grape_param_estim.selection import (
    load_selection_observations,
    load_selection_protocol,
    run_selection,
    write_selection_outputs,
)


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument(
        "--observations",
        help="JSON observations; omitted means no Phase-2 results are claimed.",
    )
    parser.add_argument("--output", required=True, help="Markdown output path.")
    parser.add_argument("--json-output", help="Optional machine-readable result.")
    parser.add_argument("--source-commit")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _source_commit(explicit):
    if explicit:
        return str(explicit)
    try:
        repository = Path(__file__).resolve().parents[4]
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repository),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            cwd=str(repository),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return commit + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def main():
    arguments = _arguments()
    protocol = load_selection_protocol(arguments.protocol)
    observations = (
        load_selection_observations(arguments.observations)
        if arguments.observations
        else ()
    )
    result = run_selection(
        protocol,
        observations,
        source_commit=_source_commit(arguments.source_commit),
    )
    write_selection_outputs(
        result,
        arguments.output,
        arguments.json_output,
        overwrite=arguments.overwrite,
    )
    print(
        "wrote {} observations, selection_complete={}, result_hash={}".format(
            len(observations),
            result["selection_complete"],
            result["result_hash"],
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("{}: {}".format(type(error).__name__, error), file=sys.stderr)
        sys.exit(2)
