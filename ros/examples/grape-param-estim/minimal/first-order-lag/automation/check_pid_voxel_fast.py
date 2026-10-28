#!/usr/bin/env python3
"""Re-evaluate saved failure-context voxel poles on the same ordered plants.

--source accepts a flat data/failure1 layout, the normal outputs layout, or a
single dataset root. Only the recorded failure controller supplies warm seeds.
The source arrays and pole model are never modified.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np

from pid_gain_voxels_fast import (
    HERE, PID_GROUPS, GAIN_CAPS, FastGainEvaluator, load_estimate_json,
    _recorded_gain_matrix, load_vehicle_model, quotient_to_scale_free_plants,
    ScaleFreePlant, actuator_parameters_from_estimate, load_configuration,
    coordinates_for_nodes, atomic_json, utc_now,
)


def dataset_root(source, case, dataset):
    source = Path(source).expanduser().resolve()
    for root in (source/case/dataset, source/case, source):
        if (root/'manifest.json').is_file():
            return root
    raise ValueError(f'no saved dataset for {case} under {source}')


def check(case, workers, count, *, source=None, grid_size=5,
          dataset='pid_gain_voxels_fast_interior_failure_context', groups=PID_GROUPS):
    if count <= 0:
        raise ValueError('nodes-per-group must be positive')
    root = dataset_root(HERE/'outputs' if source is None else source, case, dataset)
    manifest = json.loads((root/'manifest.json').read_text())
    estimate_path = root/'estimate.json'
    if not estimate_path.is_file():
        estimate_path = Path(manifest['arguments']['estimate_json']).expanduser()
    estimate = load_estimate_json(estimate_path)
    model = load_vehicle_model(Path(estimate['input']['vehicle_model']))
    base = np.asarray(manifest['fixed_controller_gains'])
    np.testing.assert_array_equal(base, _recorded_gain_matrix(estimate))
    quotient = np.load(root/'plant_samples.npz')['quotient_coordinates']
    plants = quotient_to_scale_free_plants(estimate, quotient, model.parameters, ScaleFreePlant)
    result = {'case':case, 'source_dataset':str(root),
              'source_fingerprint':manifest['fingerprint'], 'samples':len(plants),
              'workers':workers, 'grid_size':grid_size,
              'warm_start':'recorded failure controller only', 'groups':{}}
    evaluator = FastGainEvaluator(reference_gains=base, plants=plants, vehicle_model=model,
        actuator_parameters=actuator_parameters_from_estimate(estimate),
        controller_dt=manifest['controller_dt_seconds'], workers=workers)
    try:
        trim_path = root/'reference_trims.npz'
        if trim_path.is_file():
            with np.load(trim_path) as saved:
                warm = saved['recorded_case'].copy()
        else:
            warm = evaluator.evaluate_many([(base/GAIN_CAPS).ravel()], progress=False)[0].trim_vectors.copy()
        original_warm = warm.copy()
        for group in groups:
            saved_grid = root/group/f'grid_{grid_size}'
            metadata = json.loads((saved_grid/'metadata.json').read_text())
            np.testing.assert_array_equal(metadata['full_fixed_controller_gains'], base)
            with np.load(saved_grid/'volume.npz') as volume:
                axes = tuple(volume[name].copy() for name in ('p','i','d'))
                done = volume['evaluated'].copy()
                saved_radii = volume['spectral_radius'].copy()
            if not np.all(done):
                raise ValueError(f'grid is not complete: {saved_grid}')
            if saved_radii.shape != done.shape+(len(plants),):
                raise ValueError(f'saved plant count differs: {saved_grid}')
            for actual_axis, saved_axis in zip(axes, metadata['axes']):
                np.testing.assert_array_equal(actual_axis, saved_axis)
            candidates = np.argwhere(done)
            indices = candidates[np.linspace(0,len(candidates)-1,min(count,len(candidates)),dtype=int)]
            # This anchor is the same complete failure controller in every panel.
            gain = base[PID_GROUPS.index(group)]
            anchor = [[int(np.flatnonzero(axis==gain[j])[0]) for j,axis in enumerate(axes)]]
            indices = np.unique(np.vstack([indices,anchor]),axis=0)
            points = coordinates_for_nodes(indices, axes, base, PID_GROUPS.index(group))
            expected = saved_radii[tuple(indices.T)]
            evaluator._cache.clear()
            started = time.monotonic()
            rows = evaluator.evaluate_many(points, initial_trims=warm, progress=False)
            seconds = time.monotonic()-started
            actual = np.array([row.spectral_radius for row in rows])
            expected_valid = np.isfinite(expected)
            actual_valid = np.isfinite(actual)
            if np.any(expected_valid&~actual_valid):
                raise AssertionError(f'{case} {group}: lost resolved saved poles')
            error = float(np.max(np.abs(actual[expected_valid]-expected[expected_valid]))) if np.any(expected_valid) else 0.0
            if error>1e-9:
                raise AssertionError(f'{case} {group}: radius mismatch {error}')
            if np.any((actual[expected_valid]<1)!=(expected[expected_valid]<1)):
                raise AssertionError(f'{case} {group}: changed stability on resolved saved samples')
            payload = {'nodes':len(indices),'seconds':seconds,
                       'compared_resolved_poles':int(expected_valid.sum()),
                       'additional_resolved_poles':int(np.count_nonzero(actual_valid&~expected_valid)),
                       'maximum_absolute_radius_error':error,
                       'pid_triples':np.array([row.gains[PID_GROUPS.index(group)] for row in rows]).tolist(),
                       'saved_indices':indices.tolist()}
            result['groups'][group]=payload
            print(json.dumps({'case':case,'group':group,**{k:v for k,v in payload.items() if k not in ('pid_triples','saved_indices')}}),flush=True)
    finally:
        evaluator.close()
    np.testing.assert_array_equal(warm,original_warm)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=HERE/'outputs')
    parser.add_argument('--dataset',default='pid_gain_voxels_fast_interior_failure_context')
    parser.add_argument('--grid-size',type=int,default=5)
    parser.add_argument('--cases',nargs='+',default=['failure1','failure2'])
    parser.add_argument('--groups',nargs='+',choices=PID_GROUPS,default=PID_GROUPS)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--nodes-per-group',type=int,default=16)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    payload={'started_utc':utc_now(),'criterion':'unchanged exact poles on all resolved saved samples',
             'results':[check(case,args.workers,args.nodes_per_group,source=args.source,
                              grid_size=args.grid_size,dataset=args.dataset,groups=args.groups)
                        for case in args.cases]}
    payload['completed_utc']=utc_now()
    atomic_json(args.output,payload)
    print(str(args.output),flush=True)
