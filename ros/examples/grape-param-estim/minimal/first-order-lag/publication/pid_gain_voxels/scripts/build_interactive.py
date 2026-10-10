#!/usr/bin/env python3
"""Build the focused success/failure comparison page from saved node data."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
from pathlib import Path
import numpy as np
from data_io import ROOT, CASES, load_grid


def face_counts(classes):
    return {str(code): sum(int(np.count_nonzero(np.diff((classes == code).astype(np.int8), axis=axis)))
                          + int(np.count_nonzero(np.take(classes == code, 0, axis=axis)))
                          + int(np.count_nonzero(np.take(classes == code, -1, axis=axis))) for axis in range(3))
            for code in (2, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--settings", type=Path, default=ROOT / "figure_settings.json")
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    data = {"alpha": settings["alpha"], "styles": settings, "cases": []}
    for case in CASES:
        grid = load_grid(case, "roll_pitch", 65, args.root)
        classes = grid.classes(settings["alpha"])
        cut_index = grid.index(grid.gain("success"))[1]
        cut_i = float(grid.axes[1][cut_index])
        edges = [a.copy() for a in grid.edges]
        edges[1][cut_index] = cut_i
        marks = {name: grid.mark("success" if name == "success" else "recorded") for name in ("success", "failure")}
        for name in marks:
            marks[name]["classification"] = int(classes[grid.index(marks[name]["gain"])])
        data["cases"].append({"case": case, "baseline_count": grid.baseline_count,
                              "axes": [a.tolist() for a in grid.axes], "edges": [a.tolist() for a in edges],
                              "cut_index": cut_index, "cut_i": cut_i,
                              "shape": list(grid.survival.shape), "nodes": int(grid.survival.size),
                              "breaks": base64.b64encode(grid.arrays["caused_break_count"].astype(np.uint8).tobytes()).decode("ascii"),
                              "marks": marks, "faces": face_counts(classes[:, cut_index:, :])})
    scripts = Path(__file__).parent
    template = (scripts / "pid_voxel_viewer.html").read_text()
    js = (scripts / "pid_voxel_viewer.js").read_text()
    from plotly.offline import get_plotlyjs
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = template.replace("__DATA__", encoded).replace("__VIEWER__", js).replace("__PLOTLY__", get_plotlyjs())
    html = html.replace("__THRESHOLD__", format(100 * settings["alpha"], ".5g"))
    agrees = all(g["marks"]["success"]["classification"] == 3 and g["marks"]["failure"]["classification"] == 2 for g in data["cases"])
    html = html.replace("__CLAIM__", "Successは緑の領域、Failureは赤の領域に入る" if agrees else "SuccessとFailureのPIDゲインを比較する")
    for old, name in (("#169b46", "green"), ("#d62728", "red")):
        html = html.replace(old, settings[name])
    output = args.output or args.root / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    if output.stat().st_size >= 50_000_000:
        raise ValueError("HTML exceeds 50 MB")
    if args.output is None:
        interactive = args.root / "interactive"
        interactive.mkdir(exist_ok=True)
        (interactive / "index.html").write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=../index.html#3d"><a href="../index.html#3d">Success / Failure PID comparison</a>\n')
        (interactive / "viewer.json").write_text(json.dumps({"view": "two-case Roll/Pitch success/failure comparison",
            "html": "index.html", "sha256": hashlib.sha256(html.encode()).hexdigest(),
            "bytes": output.stat().st_size, "primary": "exact shared I=1 P-D slice", "cases": list(CASES)}, indent=2) + "\n")
    print(json.dumps({"html": str(output), "bytes": output.stat().st_size, "cases": list(CASES)}))


if __name__ == "__main__":
    main()
