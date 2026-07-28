#!/usr/bin/env python3
"""Materialize future replay messages into all exact closed-loop inputs."""

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys

import rospkg

from grape_param_estim.data.replay_materializer import (
    MATERIALIZED_EXACT_FILES,
    extract_canonical_replay_stream,
    materialize_exact_replay_inputs,
    write_canonical_replay_stream,
)
from grape_param_estim.plant_assimilation import (
    load_assimilation_config,
    prepare_episodes,
)


def _default_config():
    return (
        Path(rospkg.RosPack().get_path("grape_param_estim"))
        / "config"
        / "plant_assimilation.yaml"
    )


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream",
        type=Path,
        help=(
            "Optional content-addressed canonical JSON extraction seam. "
            "When omitted, read ReplayMetadata/ReplayFrame directly from "
            "each configured bag."
        ),
    )
    parser.add_argument(
        "--write-stream",
        type=Path,
        help=(
            "When extracting directly from bags, atomically preserve the "
            "canonical JSON stream at this new path."
        ),
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--bag-root",
        type=Path,
        required=True,
        help=(
            "Bag root used by estimate_grape_plant.py; observations and "
            "non-controller grids are prepared from these exact bags."
        ),
    )
    parser.add_argument(
        "--exact-replay-executable",
        type=Path,
        required=True,
        help="Built C++ pc_exact replay executable.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--exact-oracle-timeout-s",
        type=float,
        default=30.0,
    )
    return parser.parse_args(argv)


def _require_closed_loop_policy(config):
    if config.get("mode") != "closed_loop_plant_identification":
        raise ValueError(
            "replay materialization requires a closed-loop assimilation config"
        )
    controller = config.get("controller")
    if not isinstance(controller, Mapping):
        raise ValueError("closed-loop config lacks a controller policy")
    if (
        controller.get("snapshot_policy")
        not in ("injected_hash_bound", "frozen_from_bag")
        or controller.get("nominal_model_policy") != "frozen"
        or controller.get("require_factual_replay_pass") is not True
    ):
        raise ValueError(
            "closed-loop config is not eligible for frozen, factual exact "
            "controller evidence"
        )


def main(argv=None):
    arguments = _arguments(argv)
    config_path = (
        _default_config() if arguments.config is None else arguments.config
    )
    config, config_sha256 = load_assimilation_config(config_path)
    _require_closed_loop_policy(config)
    prepared = prepare_episodes(arguments.bag_root, config)
    if arguments.stream is not None and arguments.write_stream is not None:
        raise ValueError("--write-stream is only valid without --stream")
    stream = arguments.stream
    if stream is None:
        stream = extract_canonical_replay_stream(
            arguments.bag_root, config
        )
        if arguments.write_stream is not None:
            write_canonical_replay_stream(
                stream,
                arguments.write_stream,
                prepared=prepared,
                assimilation_config_sha256=config_sha256,
            )
    destination = materialize_exact_replay_inputs(
        stream_path=stream,
        prepared=prepared,
        assimilation_config_sha256=config_sha256,
        exact_replay_executable=arguments.exact_replay_executable,
        output_root=arguments.output_root,
        run_id=arguments.run_id,
        timeout_s=arguments.exact_oracle_timeout_s,
    )
    exact_arguments = {
        "--exact-replay-executable": str(
            arguments.exact_replay_executable.expanduser().resolve()
        ),
        "--controller-fixture-bundle": str(
            destination / MATERIALIZED_EXACT_FILES[0]
        ),
        "--controller-snapshot-bundle": str(
            destination / MATERIALIZED_EXACT_FILES[1]
        ),
        "--controller-state-bundle": str(
            destination / MATERIALIZED_EXACT_FILES[2]
        ),
        "--factual-conformance-report": str(
            destination / MATERIALIZED_EXACT_FILES[5]
        ),
    }
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_directory": str(destination),
                "materialized_files": list(MATERIALIZED_EXACT_FILES),
                "estimate_grape_plant_exact_arguments": exact_arguments,
            },
            sort_keys=True,
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
