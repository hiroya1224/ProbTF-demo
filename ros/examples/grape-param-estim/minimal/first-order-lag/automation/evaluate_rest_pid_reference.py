#!/usr/bin/env python3
"""Post-map comparison of both reference triples under both rest-PID contexts.

Each row is an actual full twelve-gain controller. The existing failure-only
maps, successful-controller comparison, ordered plants and physical model stay
unchanged. Only the failure controller supplies initial trims and warm cache.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from evaluate_success_reference import HERE, CASES, GROUPS, completed_inputs, sha256, stability_summary

SCHEMA = "grape-param-estim/rest-pid-controller-comparison/v1"
CONTEXTS = ("success", "failure")
POINTS = ("failure", "success")


def controller_for_point(failure, success, group, point, rest_pid):
    """Take the selected triple from point, and the other nine from context."""
    choices = {"failure": np.asarray(failure, dtype=float), "success": np.asarray(success, dtype=float)}
    if any(a.shape != (4, 3) or not np.all(np.isfinite(a)) for a in choices.values()):
        raise ValueError("controller gain matrices must be finite 4 by 3 arrays")
    result = choices[rest_pid].copy()
    result[GROUPS.index(group)] = choices[point][GROUPS.index(group)]
    return result


def replay_diagnostic(expected, actual):
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    if expected.shape != actual.shape:
        raise ValueError("comparison pole vectors have different original populations")
    common = np.isfinite(expected) & np.isfinite(actual)
    return {"same_ordered_radii": bool(np.array_equal(expected, actual, equal_nan=True)),
            "same_validity": bool(np.array_equal(np.isfinite(expected), np.isfinite(actual))),
            "maximum_finite_absolute_error": float(np.max(np.abs(expected[common] - actual[common]))) if common.any() else None}


def compare(publication, reference_path, workers=1):
    publication, reference_path = Path(publication).resolve(), Path(reference_path).resolve()
    data = publication / "data"
    manifests = completed_inputs(data)
    old_path = publication / "successful_controller_comparison.json"
    old = json.loads(old_path.read_text())
    repo = HERE.parents[4]
    code_hashes = old["numerical_source"]["sha256"]
    if any(not (repo / rel).is_file() or sha256(repo / rel) != expected for rel, expected in code_hashes.items()):
        raise ValueError("numeric implementation differs from the frozen map comparison")
    reference = json.loads(reference_path.read_text())
    success = np.asarray(reference["gains"], dtype=float)
    if (reference.get("schema") != "grape-param-estim/successful-controller-position-reference/v1"
            or reference.get("gain_order") != list(GROUPS)
            or success.shape != (4, 3) or not np.all(np.isfinite(success))):
        raise ValueError("expected a finite full-12 successful-gain reference")
    np.testing.assert_array_equal(success, old["gains"])
    from pid_gain_voxels_fast import (
        GAIN_CAPS, FastGainEvaluator, load_estimate_json, _recorded_gain_matrix,
        load_vehicle_model, quotient_to_scale_free_plants, ScaleFreePlant,
        actuator_parameters_from_estimate,
    )
    result = {"schema": SCHEMA, "role": "paired_rest_pid_contexts", "gain_order": list(GROUPS),
              "successful_gains": success.tolist(), "conditioned_on_success": False,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "criterion": "count(finite spectral radius < 1) / ALL original N plants",
              "reference": {"path": str(reference_path), "sha256": sha256(reference_path)},
              "previous_full_controller_comparison_sha256": sha256(old_path),
              "numerical_source": {"files_matching_frozen_snapshot": len(code_hashes), "sha256": code_hashes},
              "comparison_script_sha256": sha256(__file__), "cases": {}}
    for case in CASES:
        root, manifest = data / case, manifests[case]
        estimate_path, model_path = root / "inputs/original_estimate.json", root / "inputs/vehicle_model.json"
        estimate, model = load_estimate_json(estimate_path), load_vehicle_model(model_path)
        failure = _recorded_gain_matrix(estimate)
        np.testing.assert_array_equal(failure, manifest["fixed_controller_gains"])
        samples = root / "plant_samples.npz"
        with np.load(samples, allow_pickle=False) as saved:
            quotient = np.ascontiguousarray(saved["quotient_coordinates"], dtype=np.float64)
        if quotient.shape != (int(manifest["plant_samples"]), 13):
            raise ValueError("saved sample count or order differs from the complete map")
        coordinate_hash = hashlib.sha256(quotient.tobytes()).hexdigest()
        if (sha256(samples) != old["cases"][case]["provenance"]["plant_samples_sha256"]
                or coordinate_hash != old["cases"][case]["provenance"]["quotient_coordinates_sha256"]):
            raise ValueError("original plant sample identity changed")
        controllers = {}
        for group in GROUPS:
            for point in POINTS:
                for context in CONTEXTS:
                    gains = controller_for_point(failure, success, group, point, context)
                    controllers.setdefault(tuple(gains.ravel()), gains)
        plants = quotient_to_scale_free_plants(estimate, quotient, model.parameters, ScaleFreePlant)
        evaluator = FastGainEvaluator(reference_gains=failure, plants=plants, vehicle_model=model,
            actuator_parameters=actuator_parameters_from_estimate(estimate),
            controller_dt=manifest["controller_dt_seconds"], workers=workers)
        try:
            base = evaluator.evaluate_many([(failure / GAIN_CAPS).ravel()], progress=False)[0]
            # One wave: all task seeds and nearest-cache seeds are constructed
            # while the sole cached controller is the recorded failure.
            rows = evaluator.evaluate_many([(g / GAIN_CAPS).ravel() for g in controllers.values()],
                                            initial_trims=base.trim_vectors.copy(), progress=False)
        finally:
            evaluator.close()
        evaluated = {}
        for expected, row in zip(controllers.values(), rows):
            np.testing.assert_array_equal(row.gains, expected)
            evaluated[tuple(expected.ravel())] = {"full_controller_gains": expected.tolist(),
                                                **stability_summary(row.spectral_radius)}
        groups = {}
        for group in GROUPS:
            groups[group] = {}
            for point in POINTS:
                point_gains = failure if point == "failure" else success
                groups[group][point] = {"gain": point_gains[GROUPS.index(group)].tolist(), "rest_pid": {}}
                for context in CONTEXTS:
                    gains = controller_for_point(failure, success, group, point, context)
                    groups[group][point]["rest_pid"][context] = evaluated[tuple(gains.ravel())]
        saved_failure = json.loads((root / "references.json").read_text())["recorded_case"]["spectral_radius"]
        result["cases"][case] = {"sample_count": len(plants), "failure_gains": failure.tolist(),
            "distinct_full_controller_count": len(controllers), "groups": groups,
            "full_controller_replay": {
                "failure": replay_diagnostic(saved_failure, evaluated[tuple(failure.ravel())]["spectral_radius"]),
                "success": replay_diagnostic(old["cases"][case]["spectral_radius"], evaluated[tuple(success.ravel())]["spectral_radius"])},
            "provenance": {"plant_samples_sha256": sha256(samples),
                "quotient_coordinates_sha256": coordinate_hash, "sample_coordinates_sha256": coordinate_hash,
                "estimate_sha256": sha256(estimate_path), "vehicle_model_sha256": sha256(model_path),
                "map_manifest_sha256": sha256(root / "manifest.json"), "map_status_sha256": sha256(root / "status.json"),
                "map_fingerprint": manifest["fingerprint"], "controller_dt_seconds": manifest["controller_dt_seconds"],
                "resampled": False, "sample_order_changed": False,
                "warm_start": "fresh failure-only evaluator cache and failure trims; all other controllers in one wave"}}
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
    for case, row in payload["cases"].items():
        for group, points in row["groups"].items():
            print(case, group, {point: {context: value["stable_count"] for context, value in item["rest_pid"].items()}
                                for point, item in points.items()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
