#!/usr/bin/env python3
"""Create the offline gallery, numerical summaries and capped file inventory."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
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


def build_gallery(root, figures):
    cards = []
    for row in figures:
        path = "figures/" + row["figure"]
        title = (row["group"] if row["group"] != "all" else "All groups") + " · " + Path(row["figure"]).name
        cards.append('<article data-case="%s" data-group="%s" data-kind="%s"><h2>%s</h2>'
                     '<a href="%s.png"><img loading="lazy" src="%s.png" alt="%s"></a>'
                     '<nav><a href="%s.pdf">PDF</a><a href="%s.svg">SVG</a><a href="%s.png">PNG</a></nav>'
                     '<p>%s</p></article>' %
                     (row["case"], row["group"], row["kind"], html.escape(title), path, path,
                      html.escape(title), path, path, path, html.escape(row["caption"])))
    page = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PIDゲインの俯瞰図・断面図</title><style>
*{box-sizing:border-box}body{margin:0;font:15px system-ui,sans-serif;background:#f5f7fa;color:#182633}
main{max-width:1450px;margin:auto;padding:22px}h1{font-size:25px}p{line-height:1.7}a{color:#1763a4}
.toolbar,article{background:white;border:1px solid #d9e1e8;border-radius:7px;padding:15px}.toolbar{display:flex;gap:20px;flex-wrap:wrap;align-items:center;margin:20px 0;position:sticky;top:0;z-index:2}
label{display:flex;gap:8px;align-items:center}select{padding:6px;font:inherit}#cards{display:grid;grid-template-columns:1fr 1fr;gap:20px}
article[hidden]{display:none}h2{font-size:15px;overflow-wrap:anywhere}img{width:100%;height:auto;display:block}nav{display:flex;gap:20px}article p{font-size:13px;color:#4b5966}
@media(max-width:800px){#cards{grid-template-columns:1fr}}
</style><main><h1>PIDゲインの俯瞰図・断面図</h1>
<p>failure 1・2の推定分布を別々に表示します。緑は条件付き生存率95%以上、赤は95%未満です。青の◆が成功設定、紫の×が記録された失敗飛行の選択グループのゲインです。他の9ゲインは成功設定に固定しています。</p>
<p><a href="interactive/index.html">半透明ボクセルを操作する（33・65点比較）</a> · <a href="README.md">再描画・編集手順</a> · <a href="reference_gains.csv">記録ゲインと生存率</a> · <a href="figures/figure_index.csv">図の索引</a></p>
<div class="toolbar"><label>ケース<select id="case"><option value="failure1">failure 1</option><option value="failure2">failure 2</option></select></label>
<label>PID<select id="group"><option value="any">全グループ</option><option value="all">全グループの一覧図</option><option value="roll_pitch">Roll / Pitch</option><option value="yaw">Yaw</option><option value="xy">XY</option><option value="z">Z</option></select></label>
<label>図<select id="kind"><option value="any">全種類</option><option value="overview">三次元俯瞰図</option><option value="reference_slice">記録ゲインを通る断面</option><option value="sheet">断面の一覧図</option></select></label><span id="count"></span></div>
<section id="cards">__CARDS__</section></main><script>
function update(){const c=document.getElementById('case').value,g=document.getElementById('group').value,k=document.getElementById('kind').value;let n=0;
 for(const a of document.querySelectorAll('article')){const kind=k==='any'||a.dataset.kind===k||(k==='overview'&&a.dataset.kind.startsWith('overview'))||(k==='sheet'&&['slice_sheet','slices_all'].includes(a.dataset.kind));a.hidden=!(a.dataset.case===c&&(g==='any'||a.dataset.group===g)&&kind);if(!a.hidden)n++}document.getElementById('count').textContent=n+'組を表示';}
for(const id of ['case','group','kind'])document.getElementById(id).addEventListener('change',update);update();</script></html>
'''
    (root / "index.html").write_text(page.replace("__CARDS__", "\n".join(cards)))


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
    expected = Counter(overview_3d=16, overview_all=4, reference_slice=34, slice_sheet=24, slices_all=2)
    if Counter(row["kind"] for row in figures) != expected:
        raise ValueError("incomplete figure coverage")
    checks = []
    from PIL import Image
    for row in figures:
        base = root / "figures" / row["figure"]
        for ext in ("pdf", "svg", "png"):
            if not base.with_suffix("." + ext).is_file():
                raise ValueError("missing figure output")
        text = subprocess.check_output(["pdftotext", str(base.with_suffix(".pdf")), "-"], text=True)
        if "gain" not in text or "Successful-flight" not in text:
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
        build_gallery(args.root, figures)
        resolution = resolution_report(args.root)
        checks = validate_figures(args.root, figures)
        report = {"passed": True, "figure_sets": len(figures), "figure_files": 3 * len(figures),
                  "counts": dict(Counter(row["kind"] for row in figures)), "cases_separate": True,
                  "finest_grid_shapes": {case: {group: list(load_grid(case, group, 65, args.root).survival.shape)
                                                 for group in GROUPS} for case in CASES},
                  "resolution_comparison": resolution, "figure_checks": checks}
        path = args.root / "validation" / "bundle_validation.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
    inventory(args.root)


if __name__ == "__main__":
    main()
