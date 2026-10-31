#!/usr/bin/env python3
"""Select reproducible results, excluding raw bags and duplicated intermediates."""
from pathlib import Path
import argparse
import json
import shutil

import numpy as np

from run import CASES, HERE, digest, require_separate_output, write_json


def compact_arrays(arrays):
    """Keep all state covariance/cross information through its full Gram factor."""
    duplicate = {branch + "_" + key
                 for branch in ("model", "imu_dead_reckoning")
                 for key in ("covariance_state", "position_covariance_factor_fc", "rotation_covariance_factor")}
    duplicate.update("orientation_right_rotation_covariance_" + key for key in ("recursive", "output"))
    duplicate.add("model_applied_residual_wrench_parameter_jacobian")
    return {key: arrays[key] for key in arrays if key not in duplicate and not key.startswith("native_")}


def publish(source, destination):
    source, destination = Path(source).resolve(), require_separate_output(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("publication destination must be new or empty")
    manifest = json.loads((source / "manifest.json").read_text())
    if any(row["status"] != "completed" for row in manifest["cases"]):
        raise ValueError("only completed runs can be published")
    destination.mkdir(parents=True, exist_ok=True)
    selected_pdf = {"report.pdf", "residual_wrench.pdf", "sensor_reconstruction.pdf", "trajectory_replay.pdf"}
    for result in manifest["cases"]:
        case = result["case"]
        if case not in CASES:
            raise ValueError("unknown case in run manifest")
        current, target = source / case, destination / case
        target.mkdir()
        for path in current.iterdir():
            if path.suffix in (".json", ".csv", ".png") or path.name in selected_pdf:
                shutil.copy2(path, target / path.name)
                if path.suffix == ".csv":
                    (target / path.name).write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        # Keep numerical curves and every parameter covariance/factor, without
        # uploading regenerable optimizer Jacobians/whitening intermediates.
        fit_keys = {"time", "fit_valid_mask", "residual", "raw_residual_wrench", "modeled_wrench",
                    "required_wrench", "actual_thrust", "actual_gimbal", "corrected_gyro",
                    "corrected_angular_acceleration", "corrected_specific_force",
                    "predicted_specific_force", "predicted_angular_acceleration", "tau_covariance_seconds"}
        with np.load(current / "fit_arrays.npz", allow_pickle=False) as arrays:
            np.savez_compressed(target / "fit_results.npz", **{
                key: arrays[key] for key in arrays.files
                if key in fit_keys or key.startswith("parameter_covariance")})
        shutil.copy2(current / "sensor_measurement_response.npz", target / "sensor_measurement_response.npz")
        with np.load(current / "arrays.npz", allow_pickle=False) as arrays:
            np.savez_compressed(target / "arrays.npz", **compact_arrays(arrays))
    shutil.copytree(source / "inputs", destination / "inputs")
    shutil.copy2(source / manifest["combined_report"], destination / manifest["combined_report"])
    # Keep run timing/source provenance without falsely labelling excluded
    # intermediate files as published artifacts.
    write_json(destination / "run_provenance.json", {
        key: manifest[key] for key in ("schema", "completed_utc", "cases", "source_sha256")})
    files = [{"path": str(p.relative_to(destination)), "bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(destination.rglob("*")) if p.is_file()]
    result = {"schema": "grape-param-estim/imu-first-order-lag-publication/v2",
              "model_trajectory": "identified model plus original-clock IMU-derived residual wrench in mean dynamics",
              "source_manifest_sha256": digest(source / "manifest.json"),
              "publication_source_sha256": {"publish_results.py": digest(HERE / "publish_results.py")},
              "csv_serialization": "CRLF converted to LF for Git; all numeric text unchanged",
              "combined_report": manifest["combined_report"], "files": files,
              "total_bytes": sum(row["bytes"] for row in files),
              "native_arrays": "inputs/<case>.npz; original extracted sensor/command/mocap clocks and indices",
              "state_covariance_recovery": "C_state[k] = state_covariance_factor[k] @ state_covariance_factor[k].T; order in replay_metadata.json",
              "shared_wrench_state_factor_columns": {
                  "state": {"parameter": [0, 16], "initial": [16, 25], "additional_wrench": [25, 31]},
                  "applied_wrench": {"parameter": [0, 16], "additional_wrench": [16, 22]},
                  "joint_cross": "Insert 9 zero initial columns into applied-wrench factor after column 16; C_state,wrench = L_state @ L_wrench_aligned.T"},
              "removed_only": ["duplicated native arrays (archived inputs retained)",
                               "duplicated covariance_state and marginal factor views (full state Gram factor retained)",
                               "duplicate right_rotation covariance aliases", "SVG versions of PNG/PDF figures",
                               "optimizer Jacobians/whitening/noise intermediates (inputs/code/full parameter covariance retained)",
                               "residual-wrench linearization intermediate (applied mean and full covariance factor retained)",
                               "PDF body/appendix intermediates (included in report.pdf and combined PDF)"],
              "raw_bags_included": False, "monte_carlo": False}
    write_json(destination / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = publish(args.source, args.destination)
    print(json.dumps({"files": len(result["files"]), "total_bytes": result["total_bytes"]}))


if __name__ == "__main__":
    main()
