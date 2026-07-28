#!/usr/bin/env python3
"""Reproduce and verify the frozen legacy synthetic-sanity CLI baseline."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


BASELINE_SCHEMA = "grape_legacy_synthetic_sanity_baseline/v1"
INPUT_FILES = {
    "estimator_config": "config/estimator.yaml",
    "evaluator": "scripts/evaluate_sanity.py",
    "generator": "scripts/generate_sanity_bag.py",
    "estimator": "scripts/estimate_grape_bag.py",
    "truth_config": "config/sanity_truth.yaml",
}


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


def canonical_summary_bytes(payload):
    """Return path-independent canonical JSON bytes for a summary mapping."""

    if not isinstance(payload, dict):
        raise TypeError("canonical summary payload must be a mapping")
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_summary_sha256(payload):
    return hashlib.sha256(canonical_summary_bytes(payload)).hexdigest()


def normalize_generator_summary(payload):
    result = dict(payload)
    result["output_bag"] = "$INPUT_BAG"
    return result


def normalize_estimator_summary(payload):
    result = dict(payload)
    result["input_bag"] = "$INPUT_BAG"
    result["output_bag"] = "$ANALYSIS_BAG"
    return result


def normalize_evaluation_summary(payload):
    result = dict(payload)
    result["analysis_bag"] = "$ANALYSIS_BAG"
    estimate = dict(result["estimate"])
    estimate["source_bag"] = "$INPUT_BAG"
    result["estimate"] = estimate
    return result


def _load_manifest(path):
    manifest_path = Path(path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != BASELINE_SCHEMA:
        raise ValueError("unsupported synthetic-sanity baseline schema")
    if manifest.get("model_id") != "inverse_dynamics_baseline_v1":
        raise ValueError("synthetic-sanity model_id mismatch")
    return manifest_path, manifest


def verify_input_hashes(path):
    """Verify every source/config input bound by the frozen manifest."""

    manifest_path, manifest = _load_manifest(path)
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(INPUT_FILES):
        raise ValueError(
            "synthetic baseline must bind exactly {}".format(
                ", ".join(sorted(INPUT_FILES))
            )
        )
    package_root = manifest_path.parent.parent
    verified = {}
    for key, relative_path in sorted(INPUT_FILES.items()):
        entry = inputs[key]
        if not isinstance(entry, dict):
            raise ValueError("inputs.{} must be a mapping".format(key))
        if entry.get("path") != relative_path:
            raise ValueError("inputs.{} path mismatch".format(key))
        target = package_root / relative_path
        actual = _sha256_file(target)
        expected = _digest(entry.get("sha256"), "inputs.{}.sha256".format(key))
        if actual != expected:
            raise ValueError("{} hash mismatch".format(target))
        verified[key] = actual
    return package_root, manifest, verified


def _run(command, package_root, environment):
    completed = subprocess.run(
        command,
        cwd=str(package_root),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed ({!r}):\n{}".format(
                command, completed.stderr.strip()
            )
        )
    return completed.stdout


def _last_json_line(text):
    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("command output has no final JSON object")


def reproduce_synthetic_baseline(path, verify_expected=True):
    """Run the legacy three-command flow and return canonical summary hashes."""

    package_root, manifest, input_hashes = verify_input_hashes(path)
    run = manifest.get("run")
    if not isinstance(run, dict):
        raise ValueError("synthetic baseline run must be a mapping")
    seed = int(run["seed"])
    duration = float(run["duration_s"])
    rate = float(run["rate_hz"])
    particle_count = int(run["particle_count"])
    if (
        seed < 0
        or duration < 2.0
        or rate < 20.0
        or particle_count < 32
        or run.get("evaluation_mode") != "report_only"
    ):
        raise ValueError("synthetic baseline run settings are invalid")

    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(
        prefix="grape-synthetic-baseline-"
    ) as directory:
        temporary = Path(directory)
        input_bag = temporary / "input.bag"
        analysis_bag = temporary / "analysis.bag"
        generator_stdout = _run(
            [
                sys.executable,
                str(package_root / INPUT_FILES["generator"]),
                "--output-bag",
                str(input_bag),
                "--truth-config",
                str(package_root / INPUT_FILES["truth_config"]),
                "--seed",
                str(seed),
                "--duration",
                str(duration),
                "--rate",
                str(rate),
            ],
            package_root,
            environment,
        )
        estimator_stdout = _run(
            [
                sys.executable,
                str(package_root / INPUT_FILES["estimator"]),
                "--input-bag",
                str(input_bag),
                "--output-bag",
                str(analysis_bag),
                "--config",
                str(package_root / INPUT_FILES["estimator_config"]),
                "--seed",
                str(seed),
                "--particle-count",
                str(particle_count),
            ],
            package_root,
            environment,
        )
        evaluator_stdout = _run(
            [
                sys.executable,
                str(package_root / INPUT_FILES["evaluator"]),
                "--analysis-bag",
                str(analysis_bag),
                "--report-only",
            ],
            package_root,
            environment,
        )

    generator = normalize_generator_summary(json.loads(generator_stdout))
    estimator = normalize_estimator_summary(
        _last_json_line(estimator_stdout)
    )
    evaluation = normalize_evaluation_summary(json.loads(evaluator_stdout))
    summaries = {
        "generator_summary_sha256": canonical_summary_sha256(generator),
        "estimator_summary_sha256": canonical_summary_sha256(estimator),
        "evaluation_summary_sha256": canonical_summary_sha256(evaluation),
    }
    expected = manifest.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("synthetic baseline expected block is required")
    if verify_expected:
        for key, actual in sorted(summaries.items()):
            if actual != _digest(expected.get(key), "expected.{}".format(key)):
                raise ValueError(
                    "{} mismatch: expected {}, got {}".format(
                        key, expected.get(key), actual
                    )
                )
        invariants = {
            "analysis_message_count": int(estimator["analysis_messages"]),
            "observation_count": int(
                evaluation["estimate"]["observation_count"]
            ),
            "estimate_model": str(evaluation["estimate"]["model"]),
            "truth_model": str(evaluation["truth"]["model"]),
            "report_only": bool(evaluation["report_only"]),
        }
        for key, actual in sorted(invariants.items()):
            if expected.get(key) != actual:
                raise ValueError(
                    "expected.{} mismatch: expected {!r}, got {!r}".format(
                        key, expected.get(key), actual
                    )
                )
    return {
        "schema": BASELINE_SCHEMA,
        "model_id": manifest["model_id"],
        "input_sha256": input_hashes,
        "summary_sha256": summaries,
        "generator": generator,
        "estimator": estimator,
        "evaluation": evaluation,
    }


def _default_manifest():
    source_candidate = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "synthetic_sanity_baseline.json"
    )
    if source_candidate.is_file():
        return source_candidate
    try:
        import rospkg
    except ImportError as exc:
        raise RuntimeError(
            "cannot locate synthetic_sanity_baseline.json; pass --manifest"
        ) from exc
    try:
        return (
            Path(rospkg.RosPack().get_path("grape_param_estim"))
            / "config"
            / "synthetic_sanity_baseline.json"
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
        help="Synthetic baseline JSON path (default: package config).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    result = reproduce_synthetic_baseline(_arguments(argv).manifest)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "model_id": result["model_id"],
                "summary_sha256": result["summary_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("{}: {}".format(type(error).__name__, error), file=sys.stderr)
        sys.exit(2)
