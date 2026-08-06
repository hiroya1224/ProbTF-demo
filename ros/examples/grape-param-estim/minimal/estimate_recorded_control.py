#!/usr/bin/env python3
"""Entry point for the minimal recorded-control estimators."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


DEFAULT_METHOD = "deterministic_multiple_shooting"


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument(
        "--method",
        choices=(
            "deterministic_multiple_shooting",
            "deterministic",
            "deterministic_sobol",
            "deterministic_tempered",
            "deterministic_continuation",
            "deterministic_q",
            "probabilistic",
        ),
        default=DEFAULT_METHOD,
    )
    selection, remaining = selector.parse_known_args(raw_arguments)
    if selection.method == "deterministic_multiple_shooting":
        from deterministic_multiple_shooting_estimator import main as selected_main
    elif selection.method == "deterministic":
        from deterministic_estimator import main as selected_main
    elif selection.method == "deterministic_sobol":
        from deterministic_sobol_estimator import main as selected_main
    elif selection.method == "deterministic_tempered":
        from deterministic_tempered_estimator import main as selected_main
    elif selection.method == "deterministic_continuation":
        from deterministic_continuation_estimator import main as selected_main
    elif selection.method == "deterministic_q":
        from deterministic_q_estimator import main as selected_main
    else:
        from probabilistic_estimator import main as selected_main
    return selected_main(remaining)


if __name__ == "__main__":
    sys.exit(main())
