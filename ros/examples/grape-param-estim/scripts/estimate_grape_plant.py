#!/usr/bin/env python3
"""Infer a Grape plant posterior in open-loop or exact closed-loop mode."""

import argparse
from collections.abc import Mapping
from pathlib import Path
import sys

import rospkg

from grape_param_estim.alternative_backends import (
    ExactOracleIdentity,
    PersistentSubprocessExactControllerOracle,
    evaluate_exact_oracle_conformance,
)
from grape_param_estim.controller import (
    evaluate_exact_closed_loop_gate,
)
from grape_param_estim.controller.exact_inputs import (
    inject_controller_states,
    load_controller_state_bundle,
    load_episode_conformance_bundle,
    load_fixture_bundle,
    load_snapshot_bundle,
    require_episode_alignment,
)
from grape_param_estim.controller.exact_grid_alignment import (
    align_prepared_exact_grids,
)
from grape_param_estim.controller.external_oracle import (
    StatefulExactOracleControllerBackend,
    controller_backend_identity,
)
from grape_param_estim.episode import stable_hash
from grape_param_estim.forward import ClosedLoopGateError
from grape_param_estim.plant_assimilation import (
    ExactClosedLoopDependencies,
    load_assimilation_config,
    prepare_episodes,
    repository_source_identity,
    write_assimilation_run,
)


_EXACT_PATH_OPTIONS = (
    ("exact_replay_executable", "--exact-replay-executable"),
    ("controller_fixture_bundle", "--controller-fixture-bundle"),
    ("controller_snapshot_bundle", "--controller-snapshot-bundle"),
    ("controller_state_bundle", "--controller-state-bundle"),
    ("factual_conformance_report", "--factual-conformance-report"),
)


def _default_config():
    return (
        Path(rospkg.RosPack().get_path("grape_param_estim"))
        / "config"
        / "plant_assimilation.yaml"
    )


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--bag-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", default="open_loop_effective_v1")
    parser.add_argument("--particle-count", type=int)
    parser.add_argument(
        "--exact-replay-executable",
        type=Path,
        help=(
            "Built C++ pc_exact replay executable; closed-loop mode only."
        ),
    )
    parser.add_argument(
        "--controller-fixture-bundle",
        type=Path,
        help=(
            "Hash-bound per-episode controller replay fixture bundle."
        ),
    )
    parser.add_argument(
        "--controller-snapshot-bundle",
        type=Path,
        help="Hash-bound per-episode frozen controller snapshot bundle.",
    )
    parser.add_argument(
        "--controller-state-bundle",
        type=Path,
        help=(
            "Hash-bound explicit controller state for every episode/sample."
        ),
    )
    parser.add_argument(
        "--factual-conformance-report",
        type=Path,
        help=(
            "Typed per-episode exact-oracle conformance bundle, with "
            "each report bound to its runtime fixture and snapshot."
        ),
    )
    parser.add_argument(
        "--exact-oracle-timeout-s",
        type=float,
        default=None,
        help=(
            "Per-request timeout for the persistent exact replay process "
            "(default: 30 seconds; closed-loop mode only)."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Read/hash/smooth every episode but do not run SMC or write artifacts.",
    )
    values = list(sys.argv[1:] if argv is None else argv)
    return parser.parse_args([item for item in values if ":=" not in item])


def _validate_exact_mode_arguments(config, arguments):
    """Reject partial, ignored, or impossible exact-controller inputs."""

    closed_loop = (
        config["mode"] == "closed_loop_plant_identification"
    )
    present = {
        option: getattr(arguments, destination) is not None
        for destination, option in _EXACT_PATH_OPTIONS
    }
    if not closed_loop:
        supplied = tuple(
            option for option, is_present in present.items() if is_present
        )
        if arguments.exact_oracle_timeout_s is not None:
            supplied += ("--exact-oracle-timeout-s",)
        if supplied:
            raise ValueError(
                "exact-controller options are invalid in open-loop mode: "
                "{}".format(", ".join(supplied))
            )
        return
    missing = tuple(
        option for option, is_present in present.items() if not is_present
    )
    if missing:
        raise ClosedLoopGateError(
            "closed-loop mode requires explicit exact-controller inputs: "
            "{}".format(", ".join(missing))
        )
    timeout = (
        30.0
        if arguments.exact_oracle_timeout_s is None
        else float(arguments.exact_oracle_timeout_s)
    )
    if not 0.0 < timeout < float("inf"):
        raise ValueError(
            "--exact-oracle-timeout-s must be finite and positive"
        )
    controller = config.get("controller")
    if not isinstance(controller, Mapping):
        raise ClosedLoopGateError(
            "closed-loop config lacks a controller policy"
        )
    if (
        controller.get("snapshot_policy")
        not in ("injected_hash_bound", "frozen_from_bag")
        or controller.get("nominal_model_policy") != "frozen"
        or controller.get("require_factual_replay_pass") is not True
    ):
        raise ClosedLoopGateError(
            "closed-loop config is not eligible for injected, frozen, "
            "factual exact-controller evidence"
        )


def _persistent_backend_factory(
    executable,
    identity,
    timeout_s=30.0,
):
    """Start one exact process and return fresh adapters sharing it."""

    if not isinstance(identity, ExactOracleIdentity):
        raise TypeError(
            "persistent exact replay requires an ExactOracleIdentity"
        )
    oracle = PersistentSubprocessExactControllerOracle(
        (
            str(Path(executable).expanduser().resolve()),
            "--artifact-sha256",
            identity.artifact_sha256,
        ),
        identity,
        timeout_s=float(timeout_s),
    )

    def factory():
        return StatefulExactOracleControllerBackend(oracle)

    return oracle, factory


def _load_closed_loop_runtime(config, arguments, prepared):
    """Load exact evidence, inject state, and construct shared runtime."""

    fixtures, fixture_bundle_sha256 = load_fixture_bundle(
        arguments.controller_fixture_bundle
    )
    prepared = align_prepared_exact_grids(prepared, fixtures)
    snapshots, snapshot_bundle_sha256 = load_snapshot_bundle(
        arguments.controller_snapshot_bundle
    )
    states, state_bundle_sha256 = load_controller_state_bundle(
        arguments.controller_state_bundle
    )
    require_episode_alignment(
        prepared, fixtures, snapshots, states
    )
    prepared_with_states = inject_controller_states(
        prepared, states, state_bundle_sha256
    )
    conformance_bundle = load_episode_conformance_bundle(
        arguments.factual_conformance_report,
        fixtures,
        snapshots,
    )
    conformance = conformance_bundle.representative_report
    if not isinstance(conformance.identity, ExactOracleIdentity):
        raise ClosedLoopGateError(
            "factual conformance report lacks an exact oracle identity"
        )
    backend_identity = controller_backend_identity(
        conformance.identity
    )
    gate_report = evaluate_exact_closed_loop_gate(
        backend_identity,
        conformance,
        required_fidelity=config["controller"]["fidelity"],
    )
    if not gate_report.passed:
        raise ClosedLoopGateError(
            "factual exact-controller gate rejected the run: {}".format(
                "; ".join(gate_report.reasons)
            )
        )
    timeout = (
        30.0
        if arguments.exact_oracle_timeout_s is None
        else float(arguments.exact_oracle_timeout_s)
    )
    oracle, factory = _persistent_backend_factory(
        arguments.exact_replay_executable,
        conformance.identity,
        timeout_s=timeout,
    )
    try:
        for episode_id, evidence in (
            conformance_bundle.episodes.items()
        ):
            live_report = evaluate_exact_oracle_conformance(
                oracle,
                evidence.request_payload,
                evidence.conformance_fixture,
            )
            if (
                not live_report.passed
                or live_report.evidence_sha256
                != evidence.conformance_evidence_sha256
                or stable_hash(live_report.to_mapping())
                != stable_hash(
                    evidence.conformance_report.to_mapping()
                )
            ):
                raise ClosedLoopGateError(
                    "live exact-oracle preflight does not reproduce "
                    "the supplied factual evidence for episode {}".format(
                        episode_id
                    )
                )
        dependencies = ExactClosedLoopDependencies(
            controller_backend_factory=factory,
            fixtures=fixtures,
            snapshots=snapshots,
            gate_report=gate_report,
            conformance_bundle=conformance_bundle,
        )
        evidence_hashes = {
            "fixture_bundle_sha256": fixture_bundle_sha256,
            "snapshot_bundle_sha256": snapshot_bundle_sha256,
            "controller_state_bundle_sha256": state_bundle_sha256,
            "factual_conformance_evidence_sha256": (
                conformance.evidence_sha256
            ),
            "factual_conformance_bundle_sha256": (
                conformance_bundle.content_sha256
            ),
        }
        evidence_hashes["exact_input_bundle_sha256"] = stable_hash(
            evidence_hashes
        )
        return (
            prepared_with_states,
            dependencies,
            oracle,
            evidence_hashes,
        )
    except BaseException:
        oracle.close()
        raise


def _write_run_with_oracle_cleanup(*, oracle=None, **write_arguments):
    """Write one run and always release its persistent oracle process."""

    try:
        return write_assimilation_run(**write_arguments)
    finally:
        if oracle is not None:
            oracle.close()


def main(argv=None):
    arguments = _arguments(argv)
    config_path = (
        _default_config() if arguments.config is None else arguments.config
    )
    config, config_sha256 = load_assimilation_config(config_path)
    _validate_exact_mode_arguments(config, arguments)
    if arguments.particle_count is not None:
        if arguments.particle_count < 32:
            raise ValueError("--particle-count must be at least 32")
        config = dict(config)
        config["inference"] = dict(config["inference"])
        config["inference"]["particle_count"] = arguments.particle_count
        config["runtime_override"] = {
            "source_config_sha256": config_sha256,
            "particle_count": arguments.particle_count,
        }
        config_sha256 = stable_hash(config)
    package_root = Path(
        rospkg.RosPack().get_path("grape_param_estim")
    ).resolve()
    repository_root = package_root.parents[2]
    source_commit, clean = repository_source_identity(repository_root)
    if not clean and not arguments.prepare_only:
        raise RuntimeError(
            "plant assimilation artifacts require a clean Git worktree"
        )
    prepared = prepare_episodes(arguments.bag_root, config)
    for episode_id, item in prepared.items():
        print(
            "{}: role={}, commands={}, observations={}, episode_hash={}".format(
                episode_id,
                item.observations.role,
                item.commands.timestamps.size,
                item.observations.timestamps.size,
                item.data.normalized_episode_sha256,
            )
        )
    if arguments.prepare_only:
        print(
            "prepared {} episodes; no posterior/artifact was written".format(
                len(prepared)
            )
        )
        return 0
    oracle = None
    dependencies = None
    if config["mode"] == "closed_loop_plant_identification":
        (
            prepared,
            dependencies,
            oracle,
            exact_evidence,
        ) = _load_closed_loop_runtime(config, arguments, prepared)
        config = dict(config)
        runtime_override = dict(config.get("runtime_override", {}))
        runtime_override["exact_input_bundle_sha256"] = (
            exact_evidence["exact_input_bundle_sha256"]
        )
        config["runtime_override"] = runtime_override
        config_sha256 = stable_hash(config)
    destination = _write_run_with_oracle_cleanup(
        oracle=oracle,
        config=config,
        config_sha256=config_sha256,
        prepared=prepared,
        output_root=arguments.output_root,
        run_id=arguments.run_id,
        source_commit=source_commit,
        closed_loop_dependencies=dependencies,
    )
    print("wrote {}".format(destination))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("{}: {}".format(type(error).__name__, error), file=sys.stderr)
        sys.exit(2)
