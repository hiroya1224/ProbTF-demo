#!/usr/bin/env python3
"""Redraw saved reconstructions after a display-only request; never estimate."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tempfile

import numpy as np

from run_requested_comparison import (CASES, LABELS, SOURCE, read_case,
                                      load_angular_carrier, write_json)
from focused_reports import write_requested_figures


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refresh(output, source=SOURCE, cases=CASES):
    output, source = Path(output), Path(source)
    renderer = Path(__file__).with_name('focused_reports.py')
    manifest_path = output/'render_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'schema':'grape/immutable-reconstruction-display-refresh/v1','cases':{}}
    for case in cases:
        folder = output/case
        metadata_path = folder/'comparison_metadata.json'
        metadata = json.loads(metadata_path.read_text())
        if metadata['native_rows'] != metadata['figures']['native_count']:
            raise ValueError('Saved native row counts disagree')
        data = read_case(source, case)
        model, _ = load_angular_carrier(source, case, data)
        estimate = json.loads((source/case/'estimate.json').read_text())
        tracks, track_hashes = {}, {}
        for method, label in LABELS.items():
            path = folder/(method+'_full.npz')
            track_hashes[path.name] = sha(path)
            with np.load(path, allow_pickle=False) as archive:
                tracks[label] = {k:archive[k].copy() for k in archive.files
                                 if not k.startswith('orientation__')}
        with np.load(folder/'figure_data.npz', allow_pickle=False) as archive:
            old_arrays = {k:archive[k].copy() for k in archive.files}
        report = write_requested_figures(folder, case, data, model,
            estimate['time_constants_seconds'], tracks, show_model_uncertainty=False)
        if set(old_arrays) != set(report['arrays']) or not all(
                np.array_equal(old_arrays[k], report['arrays'][k]) for k in old_arrays):
            raise ValueError('A display-only refresh changed a plotted numerical array')
        if any(sha(folder/name) != expected for name, expected in track_hashes.items()):
            raise ValueError('A display-only refresh changed a reconstruction archive')
        record = dict(renderer_sha256=sha(renderer),
            refresh_script_sha256=sha(__file__), saved_reconstruction_sha256=track_hashes,
            all_figure_data_arrays_bit_identical=True,
            model_position_bands_shown=False, imu_attitude_bands_shown=True,
            computed_covariance_retained=True, estimation_repeated=False)
        metadata['figures'], metadata['plot_files'] = report['metadata'], report['files']
        metadata['display_refresh'] = record
        write_json(metadata_path, metadata)
        manifest['cases'][case] = record
    if all(case in manifest['cases'] for case in CASES):
        with tempfile.TemporaryDirectory(prefix='requested-render-') as temporary:
            joined = Path(temporary)/'all.pdf'
            subprocess.run(['pdfunite',*[str(output/c/'requested_results.pdf') for c in CASES],
                            str(joined)], check=True)
            joined.replace(output/'requested_sensor_trajectory_torque_comparison.pdf')
        manifest['combined_report_pages'] = 15
    write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    args = parser.parse_args()
    print(json.dumps(refresh(args.output, args.source, args.cases), indent=2))


if __name__ == '__main__':
    main()
