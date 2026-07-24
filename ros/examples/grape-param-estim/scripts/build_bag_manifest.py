#!/usr/bin/env python3
"""Create a reproducible inventory for Grape rosbag episodes."""

import argparse
from pathlib import Path
import sys

from grape_param_estim.manifest import (
    build_manifest,
    load_metadata_yaml,
    write_manifest_yaml,
)


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bags",
        nargs="+",
        help="Bag files or directories (directories are searched recursively).",
    )
    parser.add_argument("--metadata", help="YAML labels/splits/assumptions by filename.")
    parser.add_argument("--output", required=True, help="Output manifest YAML.")
    return parser.parse_args()


def _bag_paths(arguments):
    paths = []
    for item in arguments:
        path = Path(item).expanduser().resolve()
        if path.is_dir():
            paths.extend(path.rglob("*.bag"))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(str(path))
    unique = sorted(set(paths))
    if not unique:
        raise ValueError("no .bag files found")
    return unique


def main():
    arguments = _arguments()
    metadata = load_metadata_yaml(arguments.metadata) if arguments.metadata else {}
    paths = _bag_paths(arguments.bags)
    manifest = build_manifest(paths, metadata)
    write_manifest_yaml(manifest, arguments.output)
    print(
        "wrote {} bags to {} (manifest_hash={})".format(
            len(paths), Path(arguments.output).resolve(), manifest["manifest_hash"]
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("{}: {}".format(type(error).__name__, error), file=sys.stderr)
        sys.exit(2)
