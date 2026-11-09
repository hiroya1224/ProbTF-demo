#!/usr/bin/env python3
"""Publish completed IMU-only PID maps without caches or duplicate PDFs.

This copies the selected source artifacts byte for byte.  It performs no
identification, uncertainty propagation, pole calculation, or plot rendering.
The individual source PDFs are concatenated into one PDF, preserving their
vector labels and existing rasterized surfaces.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import subprocess

import numpy as np


CASES = ("failure1", "failure2")
MODES = ("3d", "plane")
GROUPS = ("xy", "z", "roll_pitch", "yaw")
GRID_SIZES = (33, 65)
MAX_FILE_BYTES = 50_000_000


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def selected_grids(source):
    """Validate every selected artifact before creating a publication."""
    jobs, grids = [], []
    for case in CASES:
        model_hash = None
        for mode in MODES:
            job = source / case / mode
            manifest = read_json(job / "manifest.json")
            status = read_json(job / "status.json")
            if status.get("status") != "completed":
                raise ValueError("cannot publish unfinished job: " + str(job))
            if (manifest.get("case") != case or manifest.get("mode") != mode
                    or not str(manifest.get("assessment_method", "")).startswith("analytic")
                    or manifest.get("numerical_eigenvalues_computed") is not False
                    or manifest.get("success_inputs_used") is not False
                    or manifest.get("monte_carlo") is not False):
                raise ValueError("source is not the requested analytic IMU-only assessment: " + str(job))
            current_model_hash = digest(job / "model_inputs.npz")
            if model_hash is not None and current_model_hash != model_hash:
                raise ValueError("the two views have different model inputs: " + case)
            model_hash = current_model_hash
            jobs.append((case, mode, job))
            for size in GRID_SIZES:
                for group in GROUPS:
                    directory = job / group / ("grid_" + str(size))
                    stem = "pid_3d" if mode == "3d" else "pid_i1"
                    assessment = read_json(directory / "assessment.json")
                    plot = read_json(directory / "plot_metadata.json")
                    if (assessment.get("numerical_eigenvalues_computed") is not False
                            or plot.get("numerical_poles_computed") is not False
                            or plot.get("analytic_certificate") is not True
                            or assessment.get("strict_stability_upper_bound") != 0.0
                            or plot.get("law_or_statistics_changed") is not False):
                        raise ValueError("unexpected assessment semantics: " + str(directory))
                    with np.load(directory / "criterion_grid.npz", allow_pickle=False) as arrays:
                        shape = tuple(len(arrays[key]) for key in ("p_axis", "i_axis", "d_axis"))
                        decided = arrays["criterion_decided"]
                        rate = arrays["strict_stability_fraction"]
                        if (shape != tuple(assessment["actual_shape"])
                                or rate.shape != shape or decided.shape != shape
                                or not np.all(decided) or not np.all(rate == 0.0)
                                or np.any(arrays["numerical_pole_evaluated"])):
                            raise ValueError("source is not a completed analytic zero grid: " + str(directory))
                        if mode == "plane" and not np.array_equal(arrays["i_axis"], [1.0]):
                            raise ValueError("plane must have exactly I=1: " + str(directory))
                    for name in (stem + ".png", stem + ".pdf", "assessment.json",
                                 "plot_metadata.json", "criterion_grid.npz"):
                        if not (directory / name).is_file():
                            raise FileNotFoundError(directory / name)
                    grids.append({"case": case, "mode": mode, "group": group,
                                  "requested_grid_size": size, "actual_shape": list(shape),
                                  "source": directory, "stem": stem})
    return jobs, grids


def combine_pdf(source_pdfs, destination):
    """Concatenate existing pages, retaining vector content."""
    pdfunite = shutil.which("pdfunite")
    if pdfunite:
        subprocess.run([pdfunite] + [str(path) for path in source_pdfs]
                       + [str(destination)], check=True)
        return "pdfunite (source PDF pages retained)"
    try:
        from pypdf import PdfWriter
    except ImportError:
        try:
            from PyPDF2 import PdfWriter
        except ImportError as error:
            raise RuntimeError("PDF concatenation needs pdfunite, pypdf, or PyPDF2") from error
    writer = PdfWriter()
    for path in source_pdfs:
        if hasattr(writer, "append"):
            writer.append(str(path))
        else:
            from PyPDF2 import PdfReader
            for page in PdfReader(str(path)).pages:
                writer.add_page(page)
    with destination.open("wb") as stream:
        writer.write(stream)
    return "PdfWriter (source PDF pages retained)"


def publish(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("source and publication must be separate directories")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("publication destination must be new or empty")
    jobs, grids = selected_grids(source)
    destination.mkdir(parents=True, exist_ok=True)
    copied = {}

    def copy(path, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hash = digest(path)
        if digest(target) != source_hash:
            raise IOError("copy hash mismatch: " + str(target))
        copied[str(target.relative_to(destination))] = {
            "source": str(path), "source_sha256": source_hash}

    job_records = []
    for case, mode, job in jobs:
        for name in ("manifest.json", "status.json"):
            copy(job / name, destination / case / mode / name)
        common_model = destination / case / "model_inputs.npz"
        if not common_model.exists():
            copy(job / "model_inputs.npz", common_model)
        job_records.append({"case": case, "mode": mode,
                            "source_directory": str(job),
                            "model_inputs": str(common_model.relative_to(destination)),
                            "source_model_inputs_sha256": digest(job / "model_inputs.npz")})
    pages, source_pdfs = [], []
    for page_number, grid in enumerate(grids, 1):
        relative = Path(grid["case"]) / grid["mode"] / grid["group"] / ("grid_" + str(grid["requested_grid_size"]))
        for name in (grid["stem"] + ".png", "criterion_grid.npz", "assessment.json", "plot_metadata.json"):
            copy(grid["source"] / name, destination / relative / name)
        pdf = grid["source"] / (grid["stem"] + ".pdf")
        source_pdfs.append(pdf)
        pages.append({"page": page_number, "case": grid["case"], "mode": grid["mode"],
                      "group": grid["group"], "requested_grid_size": grid["requested_grid_size"],
                      "actual_shape": grid["actual_shape"], "directory": str(relative),
                      "source_pdf": str(pdf), "source_pdf_sha256": digest(pdf)})
    pdf_name = "imu_only_pid_maps.pdf"
    pdf_method = combine_pdf(source_pdfs, destination / pdf_name)
    with (destination / "page_index.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("page", "case", "mode", "group", "requested_grid_size", "actual_shape", "directory"))
        for page in pages:
            writer.writerow(tuple(page[key] for key in ("page", "case", "mode", "group", "requested_grid_size"))
                            + ("x".join(map(str, page["actual_shape"])), page["directory"]))
    (destination / "ARTIFACTS.md").write_text(
        "# IMU-only PID map artifacts\n\n"
        "`imu_only_pid_maps.pdf` contains the 32 selected source pages; `page_index.csv` gives their order. "
        "Individual PNGs, compressed grid data, and assessment/plot metadata are byte-exact copies. "
        "Only the final requested 33/65 refinements are included; actual axis sizes (including inserted failure-gain anchors) are recorded per grid.\n\n"
        "The strict full-state stability fraction is zero by the analytic translation-observability certificate. "
        "No numerical poles or hover equilibria were calculated. The three unanchored position modes already rule out "
        "strict full-state spectral radius below one. This is not a flight-failure probability or a computation of the remaining poles.\n\n"
        "No augmented observer trajectory was simulated: the certificate applies to any causal observer that receives only the declared "
        "IMU measurements and no world-position anchor. See [the model definition](../../IMU_ONLY_PID_MAPS.md) for its scope.\n\n"
        "Each case has one shared `model_inputs.npz`. Per-case/view manifests and status files preserve the original run history, "
        "including references to intermediate grids which are deliberately not published. Plot metadata preserves original source filenames; "
        "its individual PDF entry is represented by the corresponding page of the combined PDF. "
        "No success plant parameters, raw bags, caches, or diagnostic alternative models are included.\n\n"
        "From the `imu-first-order-lag` directory, publish a completed run into a new destination with:\n\n"
        "```sh\npython3 publish_imu_pid_maps.py /tmp/grape-imu-only-pid-20261106 publication/imu_only_pid_maps_20261106\n```\n\n"
        "The [publisher](../../publish_imu_pid_maps.py) changes no plotted values. "
        "Use the [runner](../../run_imu_pid_maps.py) and [model definition](../../IMU_ONLY_PID_MAPS.md) to recreate the source run.\n")
    files = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            row = {"path": str(path.relative_to(destination)), "bytes": path.stat().st_size, "sha256": digest(path)}
            if row["bytes"] >= MAX_FILE_BYTES:
                raise ValueError("publication artifact exceeds the 50 MB limit: " + row["path"])
            row.update(copied.get(row["path"], {}))
            files.append(row)
    result = {
        "schema": "grape/imu-only-pid-map-publication/v1", "source_directory": str(source),
        "publication_script_sha256": digest(Path(__file__)), "combined_pdf": pdf_name,
        "combined_pdf_method": pdf_method, "page_count": len(pages), "pages": pages,
        "selected_requested_grid_sizes": list(GRID_SIZES), "jobs": job_records,
        "assessment_method": "analytic_translation_observability",
        "numerical_poles_computed": False, "numerical_hover_equilibria_computed": False,
        "strict_full_state_stability_fraction": 0.0, "flight_failure_probability_claimed": False,
        "source_artifacts_byte_exact": True, "success_plant_parameters_read": False,
        "individual_pdfs_excluded_as_duplicates": True, "intermediate_grids_excluded": True,
        "files": files, "total_bytes_excluding_this_manifest": sum(row["bytes"] for row in files),
        "maximum_file_bytes": max(row["bytes"] for row in files),
        "size_limit_bytes_per_file": MAX_FILE_BYTES,
    }
    write_json(destination / "published_artifacts.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = publish(args.source, args.destination)
    print(json.dumps({"destination": str(args.destination), "pages": result["page_count"],
                      "artifact_files": len(result["files"]) + 1,
                      "bytes_excluding_manifest": result["total_bytes_excluding_this_manifest"],
                      "maximum_file_bytes": result["maximum_file_bytes"]}))


if __name__ == "__main__":
    main()
