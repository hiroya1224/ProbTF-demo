#!/usr/bin/env python3
"""Entry point for the minimal recorded-control estimators."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


DEFAULT_METHOD = "deterministic_spline_dynamics"


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument(
        "--method",
        choices=(
            "deterministic_savgol_dimensionless",
            "deterministic_savgol_dynamics",
            "deterministic_spline_dynamics",
            "deterministic_multiple_shooting",
            "deterministic_multiple_shooting_multi",
            "generalized_profiling_multi",
            "deterministic_smooth_lag_multiple_shooting",
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
    config_requested = any(
        value == "--config" or value.startswith("--config=")
        for value in remaining
    )

    if selection.method in (
        "deterministic_savgol_dimensionless",
        "deterministic_savgol_dynamics",
    ):
        from dimensionless_savgol_experiment import main as selected_main

        remaining = ["--mode", "fit"] + remaining
    elif selection.method == "deterministic_spline_dynamics":
        from deterministic_spline_dynamics_estimator import (
            main as selected_main,
        )
    elif (
        selection.method == "deterministic_multiple_shooting_multi"
        or (
            selection.method == "deterministic_multiple_shooting"
            and config_requested
        )
    ):
        from legacies.deterministic_multi_bag_multiple_shooting_estimator import (
            main as selected_main,
        )
    elif selection.method == "generalized_profiling_multi":
        from legacies.deterministic_multi_bag_generalized_profiling_estimator import (
            main as selected_main,
        )
    elif selection.method == "deterministic_multiple_shooting":
        from legacies.deterministic_multiple_shooting_estimator import (
            main as selected_main,
        )
    elif selection.method == "deterministic_smooth_lag_multiple_shooting":
        from legacies.deterministic_smooth_lag_multiple_shooting_estimator import (
            main as selected_main,
        )
    elif selection.method == "deterministic":
        from legacies.deterministic_estimator import main as selected_main
    elif selection.method == "deterministic_sobol":
        from legacies.deterministic_sobol_estimator import (
            main as selected_main,
        )
    elif selection.method == "deterministic_tempered":
        from legacies.deterministic_tempered_estimator import (
            main as selected_main,
        )
    elif selection.method == "deterministic_continuation":
        from legacies.deterministic_continuation_estimator import (
            main as selected_main,
        )
    elif selection.method == "deterministic_q":
        from legacies.deterministic_q_estimator import (
            main as selected_main,
        )
    else:
        from legacies.probabilistic_estimator import main as selected_main
    return selected_main(remaining)


if __name__ == "__main__":
    sys.exit(main())
