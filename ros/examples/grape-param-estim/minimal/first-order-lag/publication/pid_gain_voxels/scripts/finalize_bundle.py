#!/usr/bin/env python3
"""Validate the focused comparison and create a capped file inventory."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from collections import Counter
import numpy as np

from data_io import ROOT, CASES, GROUPS, LABELS, load_grid


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def included_files(root):
    return sorted(path for path in root.rglob("*") if path.is_file()
                  and "__pycache__" not in path.parts and "custom_figures" not in path.parts
                  and ".venv" not in path.parts and path.suffix != ".pyc"
                  and path.name not in ("files.tsv", "inventory.json"))




def resolution_report(root):
    rows = []
    for case in CASES:
        for group in GROUPS:
            coarse = load_grid(case, group, 33, root)
            fine = load_grid(case, group, 65, root)
            indices = [np.searchsorted(b, a) for a, b in zip(coarse.axes, fine.axes)]
            for a, b, idx in zip(coarse.axes, fine.axes, indices):
                if not np.array_equal(a, b[idx]):
                    raise ValueError("coarse axes are not nested")
            selected = np.ix_(*indices)
            diff = np.abs(coarse.survival - fine.survival[selected])
            row = {"case": case, "group": group, "coarse_nodes": coarse.survival.size,
                   "fine_nodes": fine.survival.size, "shared_nodes": diff.size,
                   "shared_survival_values_equal": int(np.count_nonzero(diff == 0)),
                   "shared_classifications_equal": int(np.count_nonzero(coarse.classes() == fine.classes()[selected])),
                   "max_shared_survival_difference": float(diff.max()),
                   "shared_all_sample_stable_bits_equal": bool(np.array_equal(coarse.arrays["stable_bits"], fine.arrays["stable_bits"][selected])),
                   "coarse_green_nodes": int(np.count_nonzero(coarse.classes() == 3)),
                   "fine_green_nodes": int(np.count_nonzero(fine.classes() == 3)),
                   "coarse_red_nodes": int(np.count_nonzero(coarse.classes() == 2)),
                   "fine_red_nodes": int(np.count_nonzero(fine.classes() == 2))}
            rows.append(row)
    with (root / "validation" / "resolution_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def validate_figures(root, figures):
    if len(figures) != 2 or {row['view'] for row in figures} != {'PD', '3D'}:
        raise ValueError('expected the all-axis comparison and its shared-plane view')
    for row in figures:
        if row['groups'] != list(GROUPS) or row['cases'] != list(CASES) or row['panel_count'] != 8:
            raise ValueError('comparison scope differs')
        for case in CASES:
            for group in GROUPS:
                grid = load_grid(case, group, 65, root)
                for name in ('success', 'recorded'):
                    if row['references'][case][group][name] != grid.mark(name):
                        raise ValueError('Plotted reference differs from its exact source node')
                if grid.mark('success')['classification'] != 3:
                    raise ValueError('The all-axis success-green claim is not supported')
                expected = 2 if group == 'roll_pitch' else 3
                if grid.mark('recorded')['classification'] != expected:
                    raise ValueError('Recorded reference classification changed')
    checks = []
    from PIL import Image
    for row in figures:
        base = root / "figures" / row["figure"]
        for ext in ("pdf", "svg", "png"):
            if not base.with_suffix("." + ext).is_file():
                raise ValueError("missing figure output")
        text = subprocess.check_output(["pdftotext", str(base.with_suffix(".pdf")), "-"], text=True)
        if "gain" not in text or "Success" not in text or "Failure" not in text:
            raise ValueError("PDF axes/legend text is missing: " + str(base))
        svg = base.with_suffix(".svg").read_text()
        if "<text" not in svg:
            raise ValueError("SVG text is not editable: " + str(base))
        with Image.open(base.with_suffix(".png")) as image:
            dpi = image.info.get("dpi")
            if dpi is None or min(dpi) < 399:
                raise ValueError("PNG resolution is below 400 dpi")
            checks.append({"figure": row["figure"], "png_pixels": list(image.size), "png_dpi": list(dpi),
                           "pdf_text_extractable": True, "svg_text_editable": True})
    return checks


def inventory(root):
    records = [{"file": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}
               for path in included_files(root)]
    if any(row["bytes"] >= 50_000_000 for row in records):
        raise ValueError("file exceeds the strict decimal 50 MB limit")
    with (root / "files.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "bytes", "sha256"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    sizes = Counter()
    for row in records:
        sizes[Path(row["file"]).parts[0]] += row["bytes"]
    report = {"schema": "pid-paper-file-inventory/v1", "limit_bytes_exclusive": 50_000_000,
              "file_count": len(records), "total_bytes": sum(row["bytes"] for row in records),
              "largest_file": max(records, key=lambda row: row["bytes"]),
              "directory_bytes": dict(sizes), "excluded_from_self_hashing": ["files.tsv", "inventory.json"],
              "ignored": ["__pycache__", "*.pyc", ".venv", "custom_figures"]}
    (root / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    for name in ("files.tsv", "inventory.json"):
        if (root / name).stat().st_size >= 50_000_000:
            raise ValueError("inventory exceeds the file cap")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    if not args.inventory_only:
        figures = json.loads((args.root / "figures" / "figure_index.json").read_text())
        subprocess.run([__import__('sys').executable, str(args.root/'scripts/build_interactive.py'), '--root', str(args.root)], check=True)
        page = (args.root/'index.html').read_text()
        if page.count('<img ') != 1 or '<article' in page:
            raise ValueError('comparison page must show one primary figure')
        resolution = resolution_report(args.root)
        checks = validate_figures(args.root, figures)
        report = {"passed": True, "figure_sets": len(figures), "figure_files": 3 * len(figures),
                  "primary_figure": "figures/all_pid_success_failure_3D.png",
                  "primary_image_count": 1, "cases_separate": True, "panels_per_figure": 8, "all_success_green": True,
                  "comparison": {case: {group: {name: load_grid(case, group, 65, args.root).mark(name)
                                                 for name in ('success', 'recorded')} for group in GROUPS} for case in CASES},
                  "finest_grid_shapes": {case: {group: list(load_grid(case, group, 65, args.root).survival.shape)
                                                 for group in GROUPS} for case in CASES},
                  "resolution_comparison": resolution, "figure_checks": checks}
        path = args.root / "validation" / "bundle_validation.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
    inventory(args.root)


if __name__ == "__main__":
    main()
