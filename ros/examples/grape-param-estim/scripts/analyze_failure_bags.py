#!/usr/bin/env python3
"""Automatically detect, estimate and visualize failed-flight episodes."""

import argparse
from pathlib import Path
import sys

from grape_param_estim.automatic_analysis import (
    analyze_bags,
    load_automatic_config,
)
from grape_param_estim.effective_estimator import write_result


def _default_config() -> Path:
    source = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "automatic_failure_analysis.yaml"
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
        / "automatic_failure_analysis.yaml"
    )


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bag",
        type=Path,
        nargs="+",
        required=True,
        help="one or more ROS 1 bags in trial order",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace analysis.json and report.html if they exist",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = _arguments(argv)
    config = load_automatic_config(
        _default_config()
        if arguments.config is None
        else arguments.config
    )
    output_dir = arguments.output_dir.expanduser().resolve()
    json_path = output_dir / "analysis.json"
    html_path = output_dir / "report.html"
    existing = [
        path for path in (json_path, html_path) if path.exists()
    ]
    if existing and not arguments.force:
        raise FileExistsError(
            "output exists: {}; pass --force to replace it".format(
                ", ".join(str(path) for path in existing)
            )
        )

    result = analyze_bags(arguments.bag, config)
    from grape_param_estim.browser_report import render_browser_report

    write_result(json_path, result, overwrite=arguments.force)
    render_browser_report(
        result, html_path, overwrite=arguments.force
    )
    estimated = sum(
        episode["status"] == "estimated"
        for bag in result["bags"]
        for episode in bag["episodes"]
    )
    unidentifiable = sum(
        episode["status"] == "not_identifiable"
        for bag in result["bags"]
        for episode in bag["episodes"]
    )
    print("wrote {}".format(json_path))
    print("wrote {}".format(html_path))
    print(
        "bags={} estimated_episodes={} not_identifiable_episodes={}".format(
            result["bag_count"], estimated, unidentifiable
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
