#!/usr/bin/env python3
"""Evaluate a complete successful controller after the failure-only maps finish.

Replay the original ordered plant coordinates without drawing or selecting any
new plants. The failure controller alone supplies the evaluator's warm start.
This command never writes into data/ or modifies a gain-volume calculation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1]
CASES = ("failure1", "failure2")
GROUPS = ("xy", "z", "roll_pitch", "yaw")
SCHEMA = "grape-param-estim/successful-controller-comparison/v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stability_summary(radii):
    values = np.asarray(radii, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("expected the complete nonempty ordered pole vector")
    valid = np.isfinite(values)
    stable = valid & (values < 1.)
    return {"sample_count": int(values.size), "stable_count": int(stable.sum()),
            "valid_count": int(valid.sum()), "unresolved_count": int((~valid).sum()),
            "stable_fraction": float(stable.sum() / values.size),
            "spectral_radius": [float(x) if finite else None for x, finite in zip(values, valid)]}


def completed_inputs(data):
    """Read both completed maps before the optional successful gains are read."""
    manifests = {}
    for case in CASES:
        root = data / case
        manifest = json.loads((root / "manifest.json").read_text())
        if json.loads((root / "status.json").read_text()).get("status") != "complete":
            raise ValueError(f"failure-only maps are not complete: {case}")
        if manifest["arguments"].get("success_json") is not None:
            raise ValueError("the source maps must not depend on successful-flight input")
        for group in GROUPS:
            for size in (33, 65):
                status = json.loads((root / group / f"grid_{size}" / "status.json").read_text())
                if not status.get("complete") or status.get("pending") != 0:
                    raise ValueError(f"failure-only volume is not complete: {case}/{group}/{size}")
        manifests[case] = manifest
    return manifests


def compare(publication, reference_path, workers=1):
    publication, reference_path = Path(publication).resolve(), Path(reference_path).resolve()
    data = publication / "data"
    manifests = completed_inputs(data)
    # The existing numeric files still match the frozen map computation. Never
    # silently use a changed physical model to annotate the original figures.
    repo = HERE.parents[4]
    source_hashes = {}
    for original, expected in manifests[CASES[0]]["input_sha256"].items():
        if "/frozen_source/" in original:
            relative = original.split("/frozen_source/", 1)[1]
            current = repo / relative
            if not current.is_file() or sha256(current) != expected:
                raise ValueError(f"numeric source differs from the map snapshot: {relative}")
            source_hashes[relative] = expected
    reference = json.loads(reference_path.read_text())
    gains = np.asarray(reference["gains"], dtype=float)
    if (reference.get("schema") != "grape-param-estim/successful-controller-position-reference/v1"
            or reference.get("gain_order") != list(GROUPS)
            or gains.shape != (4, 3) or not np.all(np.isfinite(gains))):
        raise ValueError("expected an explicit finite full-12 successful-gain reference")
    # Import only after checking completion and source identity. No estimator,
    # sampler, successful-fit parameters, or map warm-start cache is used.
    sys.path.insert(0, str(HERE / "automation"))
    from pid_gain_voxels_fast import (
        GAIN_CAPS, FastGainEvaluator, load_estimate_json, _recorded_gain_matrix,
        load_vehicle_model, quotient_to_scale_free_plants, ScaleFreePlant,
        actuator_parameters_from_estimate,
    )
    result = {"schema": SCHEMA, "role": "independent_full_controller",
              "gain_order": list(GROUPS), "gains": gains.tolist(),
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "criterion": "count(finite spectral radius < 1) / ALL original N plants",
              "conditioned_on_success": False,
              "reference": {"path": str(reference_path), "sha256": sha256(reference_path)},
              "numerical_source": {"files_matching_frozen_snapshot": len(source_hashes),
                                   "sha256": source_hashes},
              "comparison_script_sha256": sha256(__file__), "cases": {}}
    for case in CASES:
        root, manifest = data / case, manifests[case]
        estimate_path, model_path = root / "inputs/original_estimate.json", root / "inputs/vehicle_model.json"
        estimate, model = load_estimate_json(estimate_path), load_vehicle_model(model_path)
        baseline = _recorded_gain_matrix(estimate)
        np.testing.assert_array_equal(baseline, manifest["fixed_controller_gains"])
        samples = root / "plant_samples.npz"
        with np.load(samples, allow_pickle=False) as saved:
            quotient = np.ascontiguousarray(saved["quotient_coordinates"], dtype=np.float64)
        if quotient.shape != (int(manifest["plant_samples"]), 13):
            raise ValueError("saved ordered sample population differs from the source map")
        plants = quotient_to_scale_free_plants(estimate, quotient, model.parameters, ScaleFreePlant)
        evaluator = FastGainEvaluator(reference_gains=baseline, plants=plants, vehicle_model=model,
            actuator_parameters=actuator_parameters_from_estimate(estimate),
            controller_dt=manifest["controller_dt_seconds"], workers=workers)
        try:
            failure = evaluator.evaluate_many([(baseline / GAIN_CAPS).ravel()], progress=False)[0]
            row = evaluator.evaluate_many([(gains / GAIN_CAPS).ravel()],
                                         initial_trims=failure.trim_vectors.copy(), progress=False)[0]
        finally:
            evaluator.close()
        np.testing.assert_array_equal(row.gains, gains)
        saved_failure = np.asarray(json.loads((root / "references.json").read_text())["recorded_case"]["spectral_radius"], dtype=float)
        actual_failure = np.asarray(failure.spectral_radius)
        common = np.isfinite(saved_failure) & np.isfinite(actual_failure)
        summary = stability_summary(row.spectral_radius)
        summary["failure_replay"] = {
            "same_ordered_radii": bool(np.array_equal(saved_failure, actual_failure, equal_nan=True)),
            "maximum_finite_absolute_error": float(np.max(np.abs(saved_failure[common] - actual_failure[common]))) if common.any() else None,
            "same_validity": bool(np.array_equal(np.isfinite(saved_failure), np.isfinite(actual_failure))),
            "stable_count": stability_summary(actual_failure)["stable_count"]}
        coordinate_hash = hashlib.sha256(quotient.tobytes()).hexdigest()
        summary["provenance"] = {
            "plant_samples_sha256": sha256(samples), "quotient_coordinates_sha256": coordinate_hash,
            "sample_coordinates_sha256": coordinate_hash,
            "estimate_sha256": sha256(estimate_path), "vehicle_model_sha256": sha256(model_path),
            "map_manifest_sha256": sha256(root / "manifest.json"),
            "map_status_sha256": sha256(root / "status.json"), "map_fingerprint": manifest["fingerprint"],
            "controller_dt_seconds": manifest["controller_dt_seconds"],
            "failure_reference_gains": baseline.tolist(),
            "warm_start": "independent evaluator; only freshly recomputed recorded-failure trims and cache",
            "resampled": False, "sample_order_changed": False}
        result["cases"][case] = summary
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    payload = compare(args.publication, args.reference, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({case: {key: row[key] for key in ("stable_count", "sample_count", "stable_fraction", "unresolved_count", "failure_replay")}
                      for case, row in payload["cases"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
