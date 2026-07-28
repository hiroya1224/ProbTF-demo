#!/usr/bin/env python3
"""Verify the frozen inverse-dynamics baseline and its canonical listing hash."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


BASELINE_SCHEMA = "grape_inverse_dynamics_baseline/v1"
LISTING_SCHEMA = "grape_inverse_dynamics_baseline_payload_listing/v1"
LISTING_METHOD = (
    "UTF-8 lines episode_id<TAB>artifact_key<TAB>sha256<LF>, with episodes "
    "and *_sha256 artifact keys lexicographically sorted"
)
ARTIFACT_FILES = {
    "artifact_manifest_sha256": "artifact_manifest.json",
    "summary_sha256": "summary.json",
    "trajectory_particles_sha256": "trajectory_particles.npz",
}
MODEL_ID_ROLE = (
    "Phase 0 legacy baseline family label; it does not relabel the immutable "
    "real-bag payloads."
)
PAYLOAD_MODEL_ID_LOCATIONS = (
    "summary.json:model_id",
    "summary.json:effective_response.model_id",
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value, name):
    result = str(value).lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError("{} must be a lowercase SHA-256".format(name))
    return result


def canonical_payload_listing(manifest):
    """Return the path-independent canonical listing bytes described in JSON."""

    if manifest.get("combined_payload_listing_schema") != LISTING_SCHEMA:
        raise ValueError("unsupported combined payload listing schema")
    if manifest.get("combined_payload_listing_method") != LISTING_METHOD:
        raise ValueError("combined payload listing method is not canonical")
    runs = manifest.get("runs")
    if not isinstance(runs, dict) or not runs:
        raise ValueError("baseline manifest requires a non-empty runs mapping")
    lines = []
    for episode_id in sorted(runs):
        run = runs[episode_id]
        if not isinstance(run, dict):
            raise ValueError("baseline run entries must be mappings")
        artifact_keys = sorted(
            key for key in run if str(key).endswith("_sha256")
        )
        if tuple(artifact_keys) != tuple(sorted(ARTIFACT_FILES)):
            raise ValueError(
                "{} must list exactly {}".format(
                    episode_id, ", ".join(sorted(ARTIFACT_FILES))
                )
            )
        for artifact_key in artifact_keys:
            digest = _digest(
                run[artifact_key],
                "{}.{}".format(episode_id, artifact_key),
            )
            lines.append(
                "{}\t{}\t{}\n".format(
                    episode_id, artifact_key, digest
                )
            )
    return "".join(lines).encode("utf-8")


def canonical_payload_listing_sha256(manifest):
    return hashlib.sha256(canonical_payload_listing(manifest)).hexdigest()


def _verify_run_artifact_manifest(run_directory, expected_run_id):
    path = run_directory / "artifact_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_id") != expected_run_id:
        raise ValueError("{} run_id mismatch".format(path))
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("{} has no payload listing".format(path))
    for name, metadata in sorted(files.items()):
        target = run_directory / name
        if not target.is_file():
            raise FileNotFoundError(str(target))
        if _sha256_file(target) != _digest(
            metadata.get("sha256"), "{} sha256".format(target)
        ):
            raise ValueError("{} hash mismatch".format(target))
        if target.stat().st_size != int(metadata.get("size_bytes", -1)):
            raise ValueError("{} size mismatch".format(target))


def verify_baseline_manifest(path):
    manifest_path = Path(path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != BASELINE_SCHEMA:
        raise ValueError("unsupported inverse-dynamics baseline schema")
    if manifest.get("model_id") != "inverse_dynamics_baseline_v1":
        raise ValueError("baseline model_id mismatch")
    if manifest.get("model_id_role") != MODEL_ID_ROLE:
        raise ValueError("baseline model_id_role mismatch")
    declared_payload_model_ids = manifest.get(
        "frozen_payload_model_ids"
    )
    if (
        not isinstance(declared_payload_model_ids, list)
        or not declared_payload_model_ids
        or any(not str(item) for item in declared_payload_model_ids)
    ):
        raise ValueError("frozen_payload_model_ids must be a non-empty list")
    if tuple(
        manifest.get("frozen_payload_model_id_locations", ())
    ) != PAYLOAD_MODEL_ID_LOCATIONS:
        raise ValueError("frozen payload model ID locations mismatch")
    listing_hash = canonical_payload_listing_sha256(manifest)
    expected_listing_hash = _digest(
        manifest.get("combined_payload_listing_sha256"),
        "combined_payload_listing_sha256",
    )
    if listing_hash != expected_listing_hash:
        raise ValueError(
            "combined payload listing hash mismatch: expected {}, got {}".format(
                expected_listing_hash, listing_hash
            )
        )

    package_root = manifest_path.parent.parent
    result_root = package_root / str(manifest["frozen_result_root"])
    verified = []
    observed_payload_model_ids = set()
    for episode_id, run in sorted(manifest["runs"].items()):
        run_id = str(run.get("run_id", ""))
        if not run_id:
            raise ValueError("{} run_id is required".format(episode_id))
        run_directory = result_root / episode_id / run_id
        if not run_directory.is_dir():
            raise FileNotFoundError(str(run_directory))
        for artifact_key, filename in sorted(ARTIFACT_FILES.items()):
            target = run_directory / filename
            actual = _sha256_file(target)
            expected = _digest(
                run[artifact_key],
                "{}.{}".format(episode_id, artifact_key),
            )
            if actual != expected:
                raise ValueError("{} hash mismatch".format(target))
        summary = json.loads(
            (run_directory / "summary.json").read_text(encoding="utf-8")
        )
        try:
            summary_model_ids = {
                str(summary["model_id"]),
                str(summary["effective_response"]["model_id"]),
            }
        except (KeyError, TypeError) as error:
            raise ValueError(
                "{} has no declared frozen payload model IDs".format(
                    run_directory / "summary.json"
                )
            ) from error
        if len(summary_model_ids) != 1:
            raise ValueError(
                "{} payload model IDs disagree".format(
                    run_directory / "summary.json"
                )
            )
        observed_payload_model_ids.update(summary_model_ids)
        _verify_run_artifact_manifest(run_directory, run_id)
        verified.append(episode_id)
    if observed_payload_model_ids != set(declared_payload_model_ids):
        raise ValueError(
            "frozen payload model IDs mismatch: declared {}, observed {}".format(
                sorted(declared_payload_model_ids),
                sorted(observed_payload_model_ids),
            )
        )
    return {
        "schema": BASELINE_SCHEMA,
        "model_id": manifest["model_id"],
        "model_id_role": manifest["model_id_role"],
        "frozen_payload_model_ids": sorted(observed_payload_model_ids),
        "verified_runs": verified,
        "combined_payload_listing_sha256": listing_hash,
    }


def _default_manifest():
    source_candidate = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "inverse_dynamics_baseline.json"
    )
    if source_candidate.is_file():
        return source_candidate
    try:
        import rospkg
    except ImportError as exc:
        raise RuntimeError(
            "cannot locate inverse_dynamics_baseline.json; pass --manifest"
        ) from exc
    try:
        return (
            Path(rospkg.RosPack().get_path("grape_param_estim"))
            / "config"
            / "inverse_dynamics_baseline.json"
        )
    except rospkg.ResourceNotFound as exc:
        raise RuntimeError(
            "cannot locate grape_param_estim; pass --manifest"
        ) from exc


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_default_manifest(),
        help="Baseline JSON path (default: package config).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    result = verify_baseline_manifest(_arguments(argv).manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("{}: {}".format(type(error).__name__, error), file=sys.stderr)
        sys.exit(2)
