#!/usr/bin/env python3
"""Run the diagnostic-only Grape real-bag vertical slice."""

import argparse
from pathlib import Path
import sys

from grape_param_estim.grape_bag_adapter import (
    analyze_configured_bags,
    load_vertical_slice_config,
)


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--bag-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--episode",
        action="append",
        help="Episode ID to run; repeatable. Defaults to all configured episodes.",
    )
    parser.add_argument(
        "--source-commit",
        help="Explicit implementation commit; defaults to the current git state.",
    )
    parser.add_argument(
        "--analysis-bag-root",
        help=(
            "Optional directory for immutable-source derived analysis bags. "
            "No bag is written by default."
        ),
    )
    return parser.parse_args()


def main():
    arguments = _arguments()
    config = load_vertical_slice_config(arguments.config)
    selected = (
        set(arguments.episode)
        if arguments.episode
        else {item["episode_id"] for item in config["episodes"]}
    )
    known = {item["episode_id"] for item in config["episodes"]}
    unknown = sorted(selected - known)
    if unknown:
        raise ValueError("unknown episode(s): {}".format(", ".join(unknown)))
    bag_root = Path(arguments.bag_root).expanduser().resolve()
    episodes = [
        episode
        for episode in config["episodes"]
        if episode["episode_id"] in selected
    ]
    destinations = analyze_configured_bags(
        bag_root=bag_root,
        episodes=episodes,
        config=config,
        output_root=arguments.output_root,
        source_commit=arguments.source_commit,
        analysis_bag_root=arguments.analysis_bag_root,
    )
    for episode, destination in zip(episodes, destinations):
        print("{} -> {}".format(episode["episode_id"], destination))
        if arguments.analysis_bag_root:
            analysis_bag = (
                Path(arguments.analysis_bag_root).expanduser().resolve()
                / episode["episode_id"]
                / "{}.analysis.bag".format(destination.name)
            )
            print(
                "{} analysis bag -> {}".format(
                    episode["episode_id"],
                    analysis_bag,
                )
            )
            print(
                "{} analysis manifest -> {}".format(
                    episode["episode_id"],
                    analysis_bag.with_suffix(".json"),
                )
            )
    print(
        "completed {} diagnostic-only episode(s); "
        "exact controller status=ORACLE_UNAVAILABLE, workflow=EXPERIMENTAL".format(
            len(destinations)
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("{}: {}".format(type(error).__name__, error), file=sys.stderr)
        sys.exit(2)
