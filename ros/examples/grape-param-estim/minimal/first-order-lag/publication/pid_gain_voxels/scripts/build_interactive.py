#!/usr/bin/env python3
"""Build the focused success/failure comparison page from saved node data."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
from pathlib import Path
import numpy as np
from data_io import ROOT, CASES, GROUPS, LABELS, load_grid
from rate_colors import rate_colorscale


def face_counts(rates, cut=False):
    total = sum(2 * int(np.count_nonzero(np.diff(rates, axis=axis)))
                + np.take(rates, 0, axis=axis).size + np.take(rates, -1, axis=axis).size
                for axis in range(3))
    front = rates.shape[0] * rates.shape[2] if cut else 0
    return {'surface': total - front, 'cut': front}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--settings", type=Path, default=ROOT / "figure_settings.json")
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    scale = rate_colorscale(settings)
    data = {"colorscale": scale, "rate_range": [0, 1], "styles": settings, "cases": []}
    for case in CASES:
        for group in GROUPS:
            grid = load_grid(case, group, 65, args.root)
            cut_index = grid.index(grid.gain('success'))[1]
            cut_i = float(grid.axes[1][cut_index])
            marks = {name:grid.mark('success' if name=='success' else 'recorded') for name in ('success','failure')}
            for name in marks:
                if marks[name]['gain'][1] != cut_i:
                    raise ValueError('Both references must lie on the exact I cut')
            data['cases'].append({'case':case,'group':group,'label':LABELS[group],'baseline_count':grid.baseline_count,
                'axes':[a.tolist() for a in grid.axes],'edges':[a.tolist() for a in grid.edges],
                'cut_index':cut_index,'cut_i':cut_i,'shape':list(grid.survival.shape),'nodes':int(grid.survival.size),
                'breaks':base64.b64encode(grid.arrays['caused_break_count'].astype(np.uint8).tobytes()).decode('ascii'),
                'marks':marks,'faces':{'full':face_counts(grid.survival),'cut':face_counts(grid.survival[:,cut_index:,:],cut=True)}})
    scripts = Path(__file__).parent
    template = (scripts / "pid_voxel_viewer.html").read_text()
    js = (scripts / "pid_voxel_viewer.js").read_text()
    from plotly.offline import get_plotlyjs
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = template.replace("__DATA__", encoded).replace("__VIEWER__", js).replace("__PLOTLY__", get_plotlyjs())
    html = html.replace('__RATE_GRADIENT__', ','.join(color+' '+format(100*q,'.5g')+'%' for q,color in scale))
    all_success_green = all(g['marks']['success']['survival']==1 for g in data['cases'])
    html = html.replace('__CLAIM__','すべてのPID軸でSuccessは緑の領域に入る' if all_success_green else '全PID軸のSuccessとFailureを比較する')
    html = html.replace('__SUMMARY__','Success：両ケース・全4軸とも緑（8/8、rate : 100.0%）' if all_success_green else '各点のrateは図中の表示を参照してください')
    for token,name in (('__SUCCESS__','success_color'),('__FAILURE__','recorded_color')):
        html = html.replace(token,settings[name])
    output = args.output or args.root / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    if output.stat().st_size >= 50_000_000:
        raise ValueError("HTML exceeds 50 MB")
    if args.output is None:
        interactive = args.root / "interactive"
        interactive.mkdir(exist_ok=True)
        (interactive / "index.html").write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=../index.html#3d"><a href="../index.html#3d">Success / Failure PID comparison</a>\n')
        (interactive / "viewer.json").write_text(json.dumps({"view": "all-axis two-case success/failure comparison",
            "html": "index.html", "sha256": hashlib.sha256(html.encode()).hexdigest(),
            "bytes": output.stat().st_size, "primary": "all eight P/I/D volumes with exact success and failure references", "cases": list(CASES), "groups": list(GROUPS), "all_success_green": all_success_green,
            "color_encoding": "continuous rate; pastel RdYlGn on [0,1]; no threshold"}, indent=2) + "\n")
    print(json.dumps({"html": str(output), "bytes": output.stat().st_size, "cases": list(CASES), "groups": list(GROUPS), "all_success_green": all_success_green}))


if __name__ == "__main__":
    main()
