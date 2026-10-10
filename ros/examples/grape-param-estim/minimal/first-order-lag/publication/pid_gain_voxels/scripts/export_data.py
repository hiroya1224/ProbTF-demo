#!/usr/bin/env python3
"""Export a compact, lossless plotting/decision archive from completed local runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone
import numpy as np

from data_io import ROOT, CASES, GROUPS, LABELS, SOURCE_GROUPS, load_grid, survival_from_bits

DATASET = "pid_gain_voxels_fast_interior_success_context"
FILE_LIMIT = 50_000_000


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="original outputs directory")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    rows, all_reports = [], {}
    for case in CASES:
        source = args.source / case / DATASET
        dest = args.output / "data" / case
        dest.mkdir(parents=True, exist_ok=True)
        state = json.loads((source / "status.json").read_text())
        if state["status"] != "complete":
            raise ValueError("run is incomplete: %s" % source)
        manifest = json.loads((source / "manifest.json").read_text())
        references = json.loads((source / "references.json").read_text())
        baseline = np.asarray(references["success"]["spectral_radius"])
        baseline_mask = np.isfinite(baseline) & (baseline < 1)
        report = {"schema": "pid-paper-data/v1", "case": case, "source": str(source),
                  "source_fingerprint": manifest["fingerprint"], "completed_utc": state["updated_utc"],
                  "exported_utc": datetime.now(timezone.utc).isoformat(),
                  "samples": len(baseline), "baseline_count": int(baseline_mask.sum()),
                  "bit_order": "little", "axis_order": ["P", "I", "D"],
                  "criterion": "1 - breaks / count(finite rho(success) < 1)",
                  "unresolved_policy": "counts as a break in the fixed conditioning set",
                  "preserved": ["exact float64 node survival and gain axes", "per-sample stable/finite decisions",
                                "exact float64 worst radius and absolute margin", "reference sample radii"],
                  "omitted": ["per-node per-sample float64 radii; retained in original local runs"], "grids": {}}
        for filename in ("manifest.json", "references.json", "plant_samples.npz", "reference_trims.npz"):
            shutil.copy2(source / filename, dest / filename)
        shutil.copy2(args.source / case / "estimate.json", dest / "estimate.json")
        for group in GROUPS:
            for size in (33, 65):
                native = source / group / ("grid_%d" % size)
                status = json.loads((native / "status.json").read_text())
                evaluated = np.load(native / "evaluated.npy", mmap_mode="r")
                if not status.get("complete") or not np.all(evaluated):
                    raise ValueError("incomplete nodes: %s" % native)
                radii = np.load(native / "spectral_radius.npy", mmap_mode="r")
                valid = np.isfinite(radii)
                stable = valid & (radii < 1)
                stable_bits = np.packbits(stable, axis=-1, bitorder="little")
                valid_bits = np.packbits(valid, axis=-1, bitorder="little")
                with np.load(native / "conditional_survival.npz", allow_pickle=False) as saved:
                    arrays = {key: saved[key].copy() for key in saved.files}
                if not np.array_equal(arrays["baseline_stable_mask"], baseline_mask):
                    raise ValueError("baseline ordering mismatch")
                recovered = survival_from_bits(stable_bits, valid_bits, baseline_mask)
                for key, values in zip(("survival_fraction", "caused_break_count", "conditional_unresolved_count"), recovered):
                    if not np.array_equal(values, arrays[key]):
                        raise ValueError("packed decisions disagree with native %s" % key)
                arrays.update(stable_bits=stable_bits, valid_bits=valid_bits,
                              sample_count=np.asarray(len(baseline), dtype=np.int32))
                for name in ("delta", "worst_spectral_radius", "worst_sample_index", "pole_valid_count", "near_kink_count"):
                    arrays[name] = np.load(native / (name + ".npy"), mmap_mode="r")
                destination = dest / group / ("grid_%d.npz" % size)
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination, **arrays)
                if destination.stat().st_size >= FILE_LIMIT:
                    raise ValueError("archive exceeds the 50 MB file cap")
                with np.load(destination, allow_pickle=False) as restored:
                    for key, value in arrays.items():
                        if not np.array_equal(value, restored[key], equal_nan=True):
                            raise ValueError("export roundtrip differs: " + key)
                    if not np.array_equal(np.unpackbits(restored["stable_bits"], axis=-1, count=len(baseline), bitorder="little"), stable):
                        raise ValueError("stable flags differ after unpacking")
                    if not np.array_equal(np.unpackbits(restored["valid_bits"], axis=-1, count=len(baseline), bitorder="little"), valid):
                        raise ValueError("finite flags differ after unpacking")
                j = SOURCE_GROUPS.index(group)
                for name in ("success", "recorded_case"):
                    gain = references[name]["gains"][j]
                    index = tuple(int(np.flatnonzero(arrays[key] == value)[0]) for key, value in zip(("p", "i", "d"), gain))
                    if name == "success" and not np.array_equal(radii[index], baseline):
                        raise ValueError("success sample radii differ from the baseline")
                    arrays[name + "_spectral_radius"] = radii[index].copy()
                # Preserve all exact sample radii at both reference nodes too.
                np.savez_compressed(destination, **arrays)
                record = {"shape": list(evaluated.shape), "nodes": int(evaluated.size),
                          "archive": str(destination.relative_to(args.output)), "bytes": destination.stat().st_size,
                          "sha256": sha256(destination), "native_radii_sha256": sha256(native / "spectral_radius.npy"),
                          "native_conditional_sha256": sha256(native / "conditional_survival.npz"),
                          "stable_and_finite_roundtrip_exact": True,
                          "all_native_plot_arrays_roundtrip_exact": True}
                report["grids"][group + "/" + str(size)] = record
                print(json.dumps({"case": case, "group": group, "grid": size, "bytes": record["bytes"]}), flush=True)
                del radii, valid, stable, arrays, stable_bits, valid_bits
        write_json(dest / "export.json", report)
        all_reports[case] = report
        for group in GROUPS:
            grid = load_grid(case, group, root=args.output)
            for name in ("success", "recorded"):
                mark = grid.mark(name)
                gain = mark.pop("gain")
                rows.append({"case": case, "group": group, "reference": name,
                             "P": gain[0], "I": gain[1], "D": gain[2], "baseline_count": grid.baseline_count, **mark})
    with (args.output / "reference_gains.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    success = args.output / "data" / "successful_flight_estimate.json"
    shutil.copy2(args.source / "success" / "estimate.json", success)
    write_json(args.output / "validation" / "data_export.json", {
        "tested_grids": 16, "exact_roundtrip": True,
        "checks": ["every stable/finite flag vs original float64 rho", "every native plotting/diagnostic array",
                   "every node conditional survival and unresolved count", "reference nodes exact sample order"],
        "cases": all_reports})


if __name__ == "__main__":
    main()
