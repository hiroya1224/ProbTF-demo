#!/usr/bin/env python3
"""Build an offline viewer from completed, all-sample PID stability data."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from matplotlib import colormaps, colors
from plotly.offline import get_plotlyjs

HERE = Path(os.environ.get('PID_VOXEL_CODE_DIR', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_voxel_survival import sample_stability

GROUPS = ('xy', 'z', 'roll_pitch', 'yaw')
LABELS = {'xy':'XY', 'z':'Z', 'roll_pitch':'Roll / Pitch', 'yaw':'Yaw'}
DATASET = 'pid_gain_voxels_fast_interior_failure_context'


def midpoint_edges(axis):
    axis = np.asarray(axis, dtype=float)
    if axis.ndim != 1 or len(axis) < 2 or np.any(np.diff(axis) <= 0):
        raise ValueError('axes must be strictly increasing')
    return np.r_[axis[0], (axis[:-1]+axis[1:])/2, axis[-1]]


def encode_bytes(array):
    values = np.asarray(array)
    if np.any(values < 0) or np.any(values > np.iinfo(np.uint32).max):
        raise ValueError('invalid sample counts')
    return base64.b64encode(values.astype('<u4').tobytes()).decode('ascii')


def face_counts(classes):
    counts = {}
    for code in (2, 3):
        mask = np.asarray(classes)==code
        count = 0
        for axis in range(3):
            count += int(np.count_nonzero(np.diff(mask.astype(np.int8), axis=axis)))
            count += int(np.count_nonzero(np.take(mask,0,axis=axis)))
            count += int(np.count_nonzero(np.take(mask,-1,axis=axis)))
        counts[str(code)] = count
    return counts


def load_payload(source, sizes=None, *, cases=('failure1', 'failure2'), groups=GROUPS, dataset=DATASET):
    source = Path(source)
    roots = {case: source/case/dataset for case in cases}
    manifests = {case: json.loads((root/'manifest.json').read_text()) for case, root in roots.items()}
    if sizes is None:
        available = [set(value['arguments']['grid_sizes']) for value in manifests.values()]
        sizes = sorted(set.intersection(*available))
    sizes = sorted(set(int(size) for size in sizes))
    if not sizes:
        raise ValueError('no completed comparison grid sizes selected')
    payload = {'sizes': sizes, 'groups': list(groups), 'cases': {}, 'count_encoding': 'uint32-little-endian',
               'colorscale': [[float(v), colors.to_hex(colormaps['RdYlGn'](float(v)))]
                              for v in np.linspace(0, 1, 256)]}
    summary = {'schema': 'grape-param-estim/pid-voxel-viewer/v2',
               'criterion': 'stable finite samples / all original plant samples',
               'fixed_context': 'other nine gains from the same failure recording',
               'voxel_geometry': 'node-centred midpoint bins, not certified cells', 'cases': {}}
    for case, root in roots.items():
        state = json.loads((root/'status.json').read_text())
        if state['status'] != 'complete':
            raise ValueError(f'{case} has not completed')
        refs = json.loads((root/'references.json').read_text())
        manifest = manifests[case]
        recorded = np.asarray(refs['recorded_case']['gains'], dtype=float)
        sample_count = int(manifest['plant_samples'])
        data = {'source': str(root), 'sample_count': sample_count,
                'completed_utc': state['updated_utc'], 'groups': {},
                'full_controller_references': {}}
        for name, reference in refs.items():
            if name not in ('recorded_case', 'success_full_controller'):
                continue
            radii = np.asarray(reference['spectral_radius'], dtype=float)
            if radii.shape != (sample_count,):
                raise ValueError(f'full-controller sample count differs: {case}/{name}')
            valid = np.isfinite(radii)
            count = int(np.count_nonzero(valid & (radii < 1)))
            data['full_controller_references'][name] = {
                'gains': reference['gains'], 'stable_count': count,
                'sample_count': sample_count, 'stable_fraction': count/sample_count,
                'unresolved_count': int(np.count_nonzero(~valid))}
        report = {'source': str(root), 'fingerprint': manifest['fingerprint'],
                  'sample_count': sample_count, 'groups': {}}
        for group in groups:
            j = GROUPS.index(group)
            selected = {'label': LABELS[group], 'failure': recorded[j].tolist(), 'grids': {}}
            group_report = {}
            for size in sizes:
                path = root/group/f'grid_{size}'
                status = json.loads((path/'status.json').read_text())
                metadata = json.loads((path/'metadata.json').read_text())
                if not np.array_equal(np.asarray(metadata['full_fixed_controller_gains']), recorded):
                    raise ValueError(f'not a recorded-failure controller section: {path}')
                if not status.get('complete') or status.get('pending'):
                    raise ValueError(f'incomplete grid: {path}')
                archive = path/'volume.npz'
                if archive.is_file():
                    with np.load(archive) as saved_volume:
                        done = saved_volume['evaluated'].copy()
                        radii = saved_volume['spectral_radius'].copy()
                else:
                    done = np.load(path/'evaluated.npy', mmap_mode='r')
                    radii = np.load(path/'spectral_radius.npy', mmap_mode='r')
                if not np.all(done):
                    raise ValueError(f'uncommitted samples: {path}')
                if radii.shape[-1] != sample_count:
                    raise ValueError(f'plant sample count differs: {path}')
                checked = sample_stability(radii, done, .95)
                with np.load(path/'stability_fraction.npz') as saved:
                    axes = [saved[name].copy() for name in ('p', 'i', 'd')]
                    if int(saved['sample_count']) != sample_count:
                        raise ValueError(f'saved denominator differs: {path}')
                    fields = {'stable_fraction': checked.stable_fraction, 'stable_count': checked.stable_count,
                              'valid_count': checked.valid_count, 'unresolved_count': checked.unresolved_count,
                              'classification': checked.alpha_classification}
                    for key, values in fields.items():
                        if not np.array_equal(saved[key], values, equal_nan=True):
                            raise ValueError(f'saved all-sample statistic differs ({key}): {path}')
                gain = selected['failure']
                index = tuple(int(np.flatnonzero(axis == gain[k])[0]) for k, axis in enumerate(axes))
                mark = {'gain': gain, 'stable_fraction': float(checked.stable_fraction[index]),
                        'stable_count': int(checked.stable_count[index]),
                        'unresolved': int(checked.unresolved_count[index]),
                        'classification': int(checked.alpha_classification[index])}
                counts = {'green': int(np.count_nonzero(checked.alpha_classification == 3)),
                          'red': int(np.count_nonzero(checked.alpha_classification == 2)),
                          'unresolved': int(np.count_nonzero(checked.unresolved_count > 0))}
                grid = {'axes': [axis.tolist() for axis in axes],
                        'edges': [midpoint_edges(axis).tolist() for axis in axes],
                        'shape': list(done.shape), 'nodes': int(done.size), 'counts': counts,
                        'nonstable': encode_bytes(sample_count - checked.stable_count),
                        'unresolved': encode_bytes(checked.unresolved_count),
                        'alpha': .95, 'marks': {'failure': mark},
                        'faces': face_counts(checked.alpha_classification),
                        'source_npz': str(path/'stability_fraction.npz')}
                selected['grids'][str(size)] = grid
                group_report[str(size)] = {key: grid[key] for key in ('shape', 'nodes', 'counts', 'marks', 'faces', 'source_npz')}
            for a, b in zip(selected['grids'][str(sizes[0])]['axes'], selected['grids'][str(sizes[-1])]['axes']):
                if not np.all(np.isin(a, b)):
                    raise ValueError('comparison grids must remain nested')
            data['groups'][group] = selected
            report['groups'][group] = group_report
        payload['cases'][case] = data
        summary['cases'][case] = report
    return payload, summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=HERE/'outputs')
    parser.add_argument('--output',type=Path,default=HERE/'visualizations/pid_gain_voxels_failure_context')
    parser.add_argument('--sizes', type=int, nargs='+', default=None)
    parser.add_argument('--cases', nargs='+', default=['failure1', 'failure2'])
    parser.add_argument('--groups', nargs='+', choices=GROUPS, default=list(GROUPS))
    parser.add_argument('--dataset', default=DATASET)
    args=parser.parse_args()
    payload,summary=load_payload(args.source, args.sizes, cases=args.cases, groups=args.groups, dataset=args.dataset)
    template=Path(__file__).with_name('pid_voxel_viewer.html').read_text()
    script=Path(__file__).with_name('pid_voxel_viewer.js').read_text()
    encoded=json.dumps(payload,separators=(',',':'),ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    cards = ''.join(f'<section class="card"><div id="head{size}" class="cardhead"></div><div id="plot{size}" class="plot"></div></section>' for size in payload['sizes'])
    slices = ''.join(f'<section class="card"><div class="cardhead">{size}点のP・D断面</div><div id="slice{size}" class="slice"></div></section>' for size in payload['sizes'])
    html=template.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',encoded).replace('__VIEWER__',script).replace('__VOLUME_CARDS__',cards).replace('__SLICE_CARDS__',slices)
    args.output.mkdir(parents=True,exist_ok=True)
    temporary=args.output/'index.tmp.html'
    temporary.write_text(html)
    os.replace(temporary,args.output/'index.html')
    summary['viewer']=str((args.output/'index.html').resolve())
    summary['html_sha256']=hashlib.sha256(html.encode()).hexdigest()
    (args.output/'viewer.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'viewer':summary['viewer'],'bytes':len(html.encode()),
                      'cases':list(payload['cases']),'sizes':payload['sizes']}))


if __name__=='__main__':
    main()
