#!/usr/bin/env python3
"""Compare only the old/new identifier union in the SAME Gaussian store.

No alternate Gaussian implementation or compressed provenance is introduced.
The legacy list union is a benchmark oracle, patched only for each timed call.
"""
from pathlib import Path
import argparse
import json
import platform
import sys
import sysconfig
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if sys.prefix != sys.base_prefix: sys.path.insert(0,sysconfig.get_path('purelib'))
sys.path.insert(0,str(ROOT/'ros/core/probtf_core/src'))
import numpy as np
from probtf.dependency import GaussianLatentStore,GaussianObservationFactor
from probtf.dependency import store as store_module
from probtf.provenance import Provenance


def legacy_unique(values):
    output = []
    for value in values:
        if value not in output: output.append(value)
    return tuple(output)


def identifiers(count):
    return tuple('final_demo-for-debug.bag:/imu/joint1_parent_imu:'+str(1794274955496092672+5_000_000*i)
                 for i in range(count))


def benchmark(count,repetitions):
    values=identifiers(count);old_ids=values[:-66];incoming=values[-74:]
    result={};posteriors=[]
    for name,unique in (('legacy_list_union',legacy_unique),('ordered_string_set_union',store_module._unique)):
        unions=[];updates=[]
        for _ in range(repetitions):
            start=time.perf_counter();merged=unique(old_ids+incoming);unions.append(time.perf_counter()-start)
            assert merged==values
            store=GaussianLatentStore()
            store.put_factor('shared',np.zeros(12),np.eye(12),0.,Provenance(source_ids=old_ids))
            observation=GaussianObservationFactor('one_new_supported_row',('shared',),[.1,-.2,.3],
                (np.eye(3,12),),np.eye(3)*.05,1.,Provenance(source_ids=incoming))
            with patch.object(store_module,'_unique',unique):
                start=time.perf_counter();store.apply_observation(observation);updates.append(time.perf_counter()-start)
            posterior=store.snapshot().factor('shared');assert posterior.provenance.source_ids==values
        posteriors.append(posterior)
        result[name]=dict(unique_union_seconds=unions,unique_union_median_s=float(np.median(unions)),
            same_store_apply_observation_seconds=updates,same_store_apply_observation_median_s=float(np.median(updates)))
    np.testing.assert_array_equal(posteriors[0].mean,posteriors[1].mean)
    np.testing.assert_array_equal(posteriors[0].covariance,posteriors[1].covariance)
    assert posteriors[0].provenance==posteriors[1].provenance
    return dict(raw_source_ids=count,previous_ids=len(old_ids),incoming_ids=len(incoming),
        incoming_overlap=8,repetitions=repetitions,measurements=result,
        union_speedup=result['legacy_list_union']['unique_union_median_s']/result['ordered_string_set_union']['unique_union_median_s'],
        same_store_update_speedup=result['legacy_list_union']['same_store_apply_observation_median_s']/result['ordered_string_set_union']['same_store_apply_observation_median_s'],
        posterior_mean_covariance_bitwise_equal=True,complete_ordered_provenance_equal=True)


def sequential_control(count,rows):
    """Actual new-store cumulative metadata path, with every raw ID retained."""
    values=identifiers(count);store=GaussianLatentStore()
    store.put_factor('shared',np.zeros(12),np.eye(12),0.)
    batches=np.array_split(np.arange(count),rows);started=time.perf_counter()
    for i,batch in enumerate(batches):
        provenance=Provenance(source_ids=tuple(values[j] for j in batch))
        store.apply_observation(GaussianObservationFactor('row_'+str(i),('shared',),[.1,-.2,.3],
            (np.eye(3,12),),np.eye(3)*.05,float(i+1),provenance))
    elapsed=time.perf_counter()-started;posterior=store.snapshot().factor('shared')
    assert posterior.provenance.source_ids==values
    return dict(rows=rows,raw_source_ids=count,elapsed_s=elapsed,all_source_ids_retained_in_order=True,
        scope='New implementation only; full sequential legacy runtime was not measured or inferred from single-call speedup')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=int,default=25000)
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/gaussian_provenance_25000.json')
    args=parser.parse_args()
    if args.sources<100 or args.repetitions<1:raise ValueError('At least100 sources and one repetition required')
    report=dict(kind='Metadata-only performance control; SAME existing Gaussian update',
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
            gaussian_store_module=store_module.__file__),
        benchmark=benchmark(args.sources,args.repetitions),sequential_new_store=sequential_control(args.sources,370),
        limits=['No provenance compression, truncation, identifier replacement, numerical Gaussian change or alternate Gaussian store.',
            'The source payload is synthetic realistic-length raw identifier strings, not a replay of physical force measurements.',
            'An individual source union is expected O(N) for normalized plain strings. Repeated cumulative snapshots still copy/validate growing metadata; fixed per-step memory/time is not claimed.',
            'Arbitrary non-plain-string inputs retain the original equality-only fallback and can remain quadratic.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
