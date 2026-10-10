#!/usr/bin/env python3
"""Build the offline 33/65-grid comparison from the portable archives."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pid-paper-matplotlib")
import numpy as np
from matplotlib import colormaps, colors
from plotly.offline import get_plotlyjs
from data_io import ROOT, CASES, GROUPS, LABELS, load_grid


def encode_bytes(values):
    if np.any(values < 0) or np.any(values > 255):
        raise ValueError("viewer requires at most 255 sample counts")
    return base64.b64encode(values.astype(np.uint8).tobytes()).decode("ascii")


def face_counts(classes):
    result = {}
    for code in (2, 3):
        mask = classes == code
        result[str(code)] = sum(int(np.count_nonzero(np.diff(mask.astype(np.int8), axis=axis)))
                                + int(np.count_nonzero(np.take(mask, 0, axis=axis)))
                                + int(np.count_nonzero(np.take(mask, -1, axis=axis))) for axis in range(3))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--settings", type=Path, default=ROOT / "figure_settings.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    alpha = settings["alpha"]
    payload = {"sizes": [33, 65], "groups": list(GROUPS), "alpha": alpha, "styles": settings,
               "colorscale": [[float(value), colors.to_hex(colormaps["RdYlGn"](float(value)))]
                              for value in np.linspace(0, 1, 256)], "cases": {}}
    for case in CASES:
        data = {"source": "data/" + case + " (all plotting values embedded in this offline HTML)", "groups": {}}
        for group in GROUPS:
            selected = {"label": LABELS[group], "grids": {}}
            for size in (33, 65):
                grid = load_grid(case, group, size, args.root)
                arrays = grid.arrays
                data.update(baseline_count=grid.baseline_count, samples=int(arrays["sample_count"]),
                            completed_utc=grid.metadata["completed_utc"])
                selected.update(success=grid.gain("success").tolist(), failure=grid.gain("recorded").tolist())
                classes = grid.classes(alpha)
                marks = {}
                for label, key in (("success", "success"), ("failure", "recorded")):
                    marks[label] = grid.mark(key)
                    marks[label]["classification"] = int(classes[grid.index(grid.gain(key))])
                selected["grids"][str(size)] = {
                    "axes": [a.tolist() for a in grid.axes], "edges": [a.tolist() for a in grid.edges],
                    "shape": list(grid.survival.shape), "nodes": grid.survival.size,
                    "counts": {"green": int(np.count_nonzero(classes == 3)), "red": int(np.count_nonzero(classes == 2)),
                               "unresolved": int(np.count_nonzero(arrays["conditional_unresolved_count"] > 0))},
                    "breaks": encode_bytes(arrays["caused_break_count"]),
                    "unresolved": encode_bytes(arrays["conditional_unresolved_count"]) if np.any(arrays["conditional_unresolved_count"]) else None,
                    "alpha": alpha, "marks": marks, "faces": face_counts(classes),
                    "source_npz": "data/%s/%s/grid_%d.npz" % (case, group, size)}
            data["groups"][group] = selected
        payload["cases"][case] = data
    directory = Path(__file__).parent
    template = (directory / "pid_voxel_viewer.html").read_text()
    script = (directory / "pid_voxel_viewer.js").read_text()
    threshold = format(100 * alpha, ".5g") + "%"
    template = template.replace("95%", threshold)
    script = script.replace("95%", threshold)
    template = template.replace('value="0.24"', 'value="%s"' % settings["green_opacity"])
    template = template.replace('value="0.16"', 'value="%s"' % settings["red_opacity"])
    for old, new in (("#2070c0", settings["success_color"]), ("#9c27b0", settings["recorded_color"]),
                     ("#168b42", settings["green"]), ("#bf2529", settings["red"])):
        template = template.replace(old, new)
        script = script.replace(old, new)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = template.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", encoded).replace("__VIEWER__", script)
    output = args.output or args.root / "interactive" / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    if output.stat().st_size >= 50_000_000:
        raise ValueError("viewer exceeds the file cap")
    report = {"html": "interactive/index.html", "bytes": output.stat().st_size,
              "sha256": hashlib.sha256(html.encode()).hexdigest(), "grids": 16,
              "input": "portable NPZ archives; no original run paths needed", "alpha": alpha}
    (output.parent / "viewer.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
