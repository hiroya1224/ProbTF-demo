#!/usr/bin/env python3
"""Audit whether Grape bags contain sufficient factual controller replay data."""

import argparse
from pathlib import Path
import re
import sys

from grape_param_estim.data import (
    audit_controller_replay_inventory,
    build_replay_audit_bundle,
    read_bag_topic_inventory,
    write_replay_audit_bundle,
)


_DEFAULT_EPISODES = ("4", "7", "8")
_EPISODE_PATTERN = re.compile(r"20260612_grape_hovering_(\d+)_")


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--bag",
        action="append",
        help="Bag path to audit; repeat for multiple episodes.",
    )
    source.add_argument(
        "--bag-root",
        help="Directory containing the 20260612 Grape hovering bags.",
    )
    parser.add_argument(
        "--episode",
        action="append",
        help="Episode number under --bag-root; defaults to 4, 7, and 8.",
    )
    parser.add_argument(
        "--output",
        default="controller_replay_audit.json",
        help="Output JSON path (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace an existing output artifact.",
    )
    return parser.parse_args(argv)


def _episode_id(path):
    match = _EPISODE_PATTERN.search(path.name)
    if match:
        return "20260612-{:02d}".format(int(match.group(1)))
    return path.stem


def _resolve_bags(arguments):
    if arguments.bag:
        if arguments.episode:
            raise ValueError("--episode is only valid with --bag-root")
        paths = tuple(
            Path(value).expanduser().resolve() for value in arguments.bag
        )
    else:
        root = Path(arguments.bag_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(str(root))
        episodes = arguments.episode or list(_DEFAULT_EPISODES)
        selected = []
        for value in episodes:
            number = int(value)
            if number <= 0:
                raise ValueError("episode numbers must be positive")
            matches = sorted(
                root.glob(
                    "20260612_grape_hovering_{}_*".format(number) + ".bag"
                )
            )
            if len(matches) != 1:
                raise RuntimeError(
                    "episode {} resolved to {} bags under {}".format(
                        number, len(matches), root
                    )
                )
            selected.append(matches[0].resolve())
        paths = tuple(selected)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("bag paths must be non-empty and unique")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return paths


def main(argv=None):
    arguments = _arguments(argv)
    paths = _resolve_bags(arguments)
    audits = []
    for path in paths:
        inventory = read_bag_topic_inventory(path)
        audit = audit_controller_replay_inventory(
            inventory, episode_id=_episode_id(path)
        )
        audits.append(audit)
        print(
            "{}: {} (available={}, derivable={}, missing={})".format(
                audit.episode_id,
                audit.decision,
                sum(item.status == "AVAILABLE" for item in audit.fields),
                sum(item.status == "DERIVABLE" for item in audit.fields),
                sum(item.status == "MISSING" for item in audit.fields),
            )
        )
    bundle = build_replay_audit_bundle(audits)
    destination = write_replay_audit_bundle(
        bundle, arguments.output, overwrite=arguments.force
    )
    print(
        "wrote {} (exact_replay_ready={})".format(
            destination, bundle["overall_exact_replay_ready"]
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("{}: {}".format(type(error).__name__, error), file=sys.stderr)
        sys.exit(2)
