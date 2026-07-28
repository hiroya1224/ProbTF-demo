#!/usr/bin/env python3
"""Estimate effective command-response parameters from one failed ROS bag."""

import argparse
from pathlib import Path
import sys

from grape_param_estim.effective_estimator import (
    load_config,
    run_from_bag,
    write_result,
)


def _default_config() -> Path:
    source = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "failure_estimator.yaml"
    )
    if source.is_file():
        return source
    try:
        import rospkg
    except ImportError as exc:
        raise RuntimeError("pass --config or install python3-rospkg") from exc
    return (
        Path(rospkg.RosPack().get_path("grape_param_estim"))
        / "config"
        / "failure_estimator.yaml"
    )


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace an existing output JSON.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = _arguments(argv)
    config = load_config(
        _default_config()
        if arguments.config is None
        else arguments.config
    )
    result = run_from_bag(arguments.bag, config)
    destination = write_result(
        arguments.output, result, overwrite=arguments.force
    )
    print("wrote {}".format(destination))
    print(
        "selected alignment lag: {:.3f} s".format(
            result["selected_alignment_lag_s"]
        )
    )
    for axis, diagnostics in result["channels"].items():
        name = diagnostics["gain_parameter"]
        parameter = result["parameters"][name]
        print(
            "{}: {}={:.6g} [{:.6g}, {:.6g}] ({})".format(
                axis,
                name,
                parameter["estimate"],
                parameter["ci95"][0],
                parameter["ci95"][1],
                diagnostics["information_grade"],
            )
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(
            "{}: {}".format(type(error).__name__, error),
            file=sys.stderr,
        )
        sys.exit(2)
