#!/usr/bin/env python3
"""Separate native-IMU identification, Bingham reconstruction and uncertainty report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import traceback

os.environ.setdefault("MPLCONFIGDIR", "/tmp/grape-imu-first-order-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np

HERE = Path(__file__).resolve().parent
CASES = ("failure1", "failure2", "success")
SCHEMA = "grape-param-estim/imu-first-order-lag-run/v1"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def require_separate_output(output):
    """A new data source must never overwrite the mocap implementation/results."""
    selected = Path(output).resolve()
    mocap = HERE.parent / "first-order-lag"
    try:
        selected.relative_to(mocap.resolve())
    except ValueError:
        return selected
    raise ValueError("IMU outputs must be outside the existing mocap first-order-lag directory")


def numeric_arrays(value):
    return {key: item for key, item in value.items()
            if isinstance(item, np.ndarray) and item.dtype.kind != "O"}


def read_inputs(root, case):
    from imu_data import save_case
    path = Path(root) / (case + ".npz")
    metadata = Path(root) / (case + ".json")
    if not path.is_file() or not metadata.is_file():
        save_case(case, Path(root))
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    data["metadata"] = json.loads(metadata.read_text())
    return data, {"arrays_sha256": digest(path), "metadata_sha256": digest(metadata),
                  "arrays_path": str(path.resolve()), "metadata_path": str(metadata.resolve())}


def validate_fit_inputs(fit, data, model_path):
    from imu_model import input_provenance
    current = input_provenance(data, model_path)
    cached = fit["estimate"]["provenance"]
    if cached.get("identification_input_arrays") != current["identification_input_arrays"]:
        raise ValueError("cached fit does not identify these exact native IMU inputs")
    if cached.get("source_files", {}).get("vehicle_model", {}).get("sha256") != digest(model_path):
        raise ValueError("cached fit uses a different vehicle model")


def run_case(case, inputs, output, model_path, fit_cache=None):
    from imu_model import fit_case, load_fit_cache
    from imu_orientation import estimate_orientation
    from imu_replay import replay
    from imu_reports import write_reports

    directory = require_separate_output(Path(output) / case)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = {"schema": SCHEMA, "case": case, "status": "running", "stage": "inputs"}
    status_path = directory / "status.json"
    write_json(status_path, status)
    timing = {}

    def stage(name):
        status.update(stage=name, elapsed_seconds=time.perf_counter() - started)
        write_json(status_path, status)
        print(json.dumps({"case": case, "stage": name, "elapsed_seconds": status["elapsed_seconds"]}), flush=True)

    try:
        data, source = read_inputs(inputs, case)
        archived_inputs = Path(output) / "inputs"
        archived_inputs.mkdir(parents=True, exist_ok=True)
        for suffix in (".npz", ".json"):
            original = Path(inputs) / (case + suffix)
            target = archived_inputs / original.name
            if original.resolve() != target.resolve():
                shutil.copy2(original, target)
        source["archived_arrays"] = "inputs/" + case + ".npz"
        source["archived_metadata"] = "inputs/" + case + ".json"
        stage("identify_body_and_three_time_constants")
        then = time.perf_counter()
        fit = (fit_case(data, model_path) if fit_cache is None
               else load_fit_cache(Path(fit_cache) / case, model_path))
        timing["identification_wall_seconds"] = time.perf_counter() - then
        if fit_cache is not None:
            # A reused fit must identify exactly these observations, never a
            # different recording or a later mocap-derived initial guess.
            validate_fit_inputs(fit, data, model_path)
            source["fit_cache"] = str((Path(fit_cache) / case).resolve())
            source["fit_cache_estimate_sha256"] = digest(Path(fit_cache) / case / "estimate.json")
        estimate = fit["estimate"]
        estimate["provenance"] = {**estimate.get("provenance", {}), "native_inputs": source,
                                   "vehicle_model_sha256": digest(model_path)}
        write_json(directory / "estimate.json", estimate)
        np.savez_compressed(directory / "fit_arrays.npz", **numeric_arrays(fit["arrays"]))

        stage("gyro_acceleration_recurrence_and_output_only_magnetic_fusion")
        then = time.perf_counter()
        sensor_tau_covariance = np.asarray(fit["tau_covariance_seconds"])[1:3, 1:3]
        orientation = estimate_orientation(data, float(fit["taus"][1]), float(fit["taus"][2]),
                                           lag_covariance_2x2=sensor_tau_covariance)
        timing["orientation_wall_seconds"] = time.perf_counter() - then
        write_json(directory / "orientation_metadata.json", orientation["metadata"])

        stage("deterministic_uncertain_reconstruction")
        then = time.perf_counter()
        reconstructed = replay(data, fit, orientation)
        timing["reconstruction_wall_seconds"] = time.perf_counter() - then
        write_json(directory / "replay_metadata.json", reconstructed["metadata"])
        arrays = {"native_" + key: value for key, value in numeric_arrays(data).items()}
        arrays.update({"orientation_" + key: value for key, value in numeric_arrays(orientation).items()})
        for branch in ("model", "imu_dead_reckoning"):
            arrays.update({branch + "_" + key: value
                           for key, value in numeric_arrays(reconstructed[branch]).items()})
        np.savez_compressed(directory / "arrays.npz", **arrays)

        stage("figures_and_parameter_covariance_appendix")
        then = time.perf_counter()
        report = write_reports(directory, data, fit, orientation, reconstructed)
        timing["report_wall_seconds"] = time.perf_counter() - then
        estimate["orientation_method"] = orientation["metadata"]
        estimate["reconstruction_method"] = reconstructed["metadata"]
        estimate["evaluation"] = report.get("statistics", {})
        write_json(directory / "estimate.json", estimate)
        write_json(directory / "report_metadata.json", report)
        timing["total_wall_seconds"] = time.perf_counter() - started
        write_json(directory / "timing.json", timing)
        status.update(status="completed", stage="completed", timing=timing,
                      report="report.pdf", numerical_source="IMU; mocap only initialization/evaluation")
        write_json(status_path, status)
        return {"case": case, "directory": str(directory), "status": "completed", "report": report}
    except Exception as error:
        status.update(status="failed", exception_type=type(error).__name__, message=str(error),
                      traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started)
        write_json(status_path, status)
        raise


def combine_reports(output, results):
    """All case figures first, all actual-value/covariance appendices last."""
    output = Path(output)
    pages_body, pages_appendix = [], []
    with tempfile.TemporaryDirectory(prefix="imu-first-order-pdf-") as temporary:
        temporary = Path(temporary)
        for result in results:
            case = result["case"]
            report = output / case / "report.pdf"
            info = subprocess.check_output(["pdfinfo", str(report)], text=True)
            count = int(next(line.split(":", 1)[1].strip() for line in info.splitlines() if line.startswith("Pages:")))
            boundary = int(result["report"]["appendix_start_page"])
            if not 1 <= boundary <= count:
                raise ValueError("parameter appendix boundary is outside its completed report")
            pattern = temporary / (case + "-%d.pdf")
            subprocess.run(["pdfseparate", str(report), str(pattern)], check=True)
            pages_body.extend(temporary / (case + f"-{page}.pdf") for page in range(1, boundary))
            pages_appendix.extend(temporary / (case + f"-{page}.pdf") for page in range(boundary, count + 1))
        target = output / "imu_first_order_lag_comparison.pdf"
        subprocess.run(["pdfunite", *map(str, pages_body + pages_appendix), str(target)], check=True)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--inputs-root", type=Path, default=HERE / "inputs")
    parser.add_argument("--output", type=Path, default=HERE / "outputs")
    parser.add_argument("--vehicle-model", type=Path, default=HERE.parent / "grape_vehicle_model.json")
    parser.add_argument("--fit-cache", type=Path, help="Optional completed IMU fit cache; never mocap estimates")
    args = parser.parse_args(argv)
    output = require_separate_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = [run_case(case, args.inputs_root, output, args.vehicle_model, args.fit_cache)
               for case in args.cases]
    combined = combine_reports(output, results)
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                          "sha256": digest(path)})
    manifest = {"schema": SCHEMA, "completed_utc": datetime.now(timezone.utc).isoformat(),
                "cases": results, "combined_report": combined.name, "files": files,
                "total_bytes": sum(row["bytes"] for row in files),
                "mocap_outputs_overwritten": False, "monte_carlo": False,
                "source_sha256": {str(path.relative_to(HERE)): digest(path)
                                  for path in sorted(HERE.rglob("*.py"))}}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"combined_report": str(combined), "total_bytes": manifest["total_bytes"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
