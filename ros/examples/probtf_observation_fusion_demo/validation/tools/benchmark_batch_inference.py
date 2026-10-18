#!/usr/bin/env python3
"""Compare the actual serial and frontier-batch workers on one recorded prefix.

The original DB and actual ledger are read only. Each mode gets an isolated
copy of exactly the selected raw rows/BLOBs for normal claim/complete status
changes. Default coverage is only the first 0.1 acquisition seconds, not the
entire recorded run. Full-run replay is explicitly requested.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import resource
import sqlite3
import time


PACKAGE = Path(__file__).resolve().parents[2]
TIMING_KEYS = {'seconds', 'inference_seconds', 'total_seconds', 'work_seconds',
               'claimed_wall_stamp', 'completed_wall_stamp', 'processing_s', 'ready_to_claim_s'}


def rows(path):
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {key:clean(item) for key,item in value.items() if key not in TIMING_KEYS}
    if isinstance(value, (tuple,list)):
        return [clean(item) for item in value]
    return value


def source_hashes():
    result = {str(path.relative_to(PACKAGE)):sha(path)
              for root in (PACKAGE/'src',PACKAGE/'config',PACKAGE/'native') for path in sorted(root.rglob('*'))
              if path.is_file() and path.suffix in ('.py','.yaml','.cpp','.h')}
    from probtf_observation_fusion_demo.auxiliary_equilibrium_native import _library_path
    native = _library_path(None)
    result['native_auxiliary_backend:'+str(native)] = sha(native)
    return result


def select_prefix(run, prefix_seconds, max_stamp, full_recorded):
    connection=sqlite3.connect('file:'+str(run/'observations.sqlite3')+'?mode=ro',uri=True)
    connection.row_factory=sqlite3.Row
    try:
        connection.execute('PRAGMA query_only=ON')
        epoch,last=connection.execute('SELECT MIN(stamp),MAX(stamp) FROM events').fetchone()
        if epoch is None:
            raise ValueError('The recorded DB is empty.')
        cutoff=last if full_recorded else max_stamp if max_stamp is not None else epoch+prefix_seconds
        selected=[]
        blocker=None
        for row in connection.execute('SELECT sequence,event_id,kind,stamp,command_json,payload_json,payload_hash FROM events WHERE stamp<=? ORDER BY stamp,sequence',(cutoff,)):
            if row['payload_hash'] is None:
                blocker=dict(event_id=row['event_id'],sequence=row['sequence'],stamp=row['stamp'],reason='original raw payload unavailable')
                break
            if row['payload_json'] is None:
                raise ValueError('A raw payload hash has no payload metadata.')
            selected.append(dict(row))
        if not selected:
            raise ValueError('No complete raw-event prefix is available through the requested frontier.')
        total=connection.execute('SELECT COUNT(*) FROM events').fetchone()[0]
        return selected,dict(original_event_count=total,selected_event_count=len(selected),
            requested_cutoff_stamp=cutoff,acquisition_epoch=epoch,
            selected_first_stamp=selected[0]['stamp'],selected_last_stamp=selected[-1]['stamp'],
            original_last_stamp=last,all_recorded_events_selected=len(selected)==total,
            full_recorded_requested=full_recorded,blocked_by_unavailable_raw_event=blocker,
            raw_event_ids=[row['event_id'] for row in selected],
            kinds=dict(Counter(row['kind'] for row in selected)),
            rule='All raw rows in acquisition stamp/immutable sequence order through the inclusive cutoff; no newest-only selection.')
    finally:
        connection.close()


def clone_prefix(run, destination, selected):
    from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore
    started=time.perf_counter()
    source=sqlite3.connect('file:'+str(run/'observations.sqlite3')+'?mode=ro',uri=True)
    source.row_factory=sqlite3.Row
    blob_bytes=blob_count=0
    try:
        source.execute('PRAGMA query_only=ON')
        with SensorEventStore(destination) as target, target._transaction():
            for selected_row in selected:
                row=source.execute('SELECT * FROM events WHERE event_id=?',(selected_row['event_id'],)).fetchone()
                target.connection.execute('''INSERT INTO events(sequence,event_id,kind,stamp,command_json,status,
                    received_wall_stamp,ready_wall_stamp,payload_json,payload_hash,attempts)
                    VALUES(?,?,?,?,?,'pending',?,?,?,?,0)''',
                    (row['sequence'],row['event_id'],row['kind'],row['stamp'],row['command_json'],
                     row['received_wall_stamp'],row['ready_wall_stamp'],row['payload_json'],row['payload_hash']))
                for blob_index,value in source.execute('SELECT blob_index,value FROM payload_blobs WHERE event_id=? ORDER BY blob_index',(row['event_id'],)):
                    blob_count+=1;blob_bytes+=len(value)
                    target.connection.execute('INSERT INTO payload_blobs(event_id,blob_index,value) VALUES(?,?,?)',
                                              (row['event_id'],blob_index,value))
            target_rows=target.connection.execute('SELECT sequence,event_id,kind,stamp,command_json,payload_json,payload_hash FROM events ORDER BY stamp,sequence').fetchall()
            if [dict(row) for row in target_rows]!=selected:
                raise AssertionError('Working DB copy changed an original event identity or raw payload descriptor.')
    finally:
        source.close()
    return dict(seconds=time.perf_counter()-started,event_rows=len(selected),blob_count=blob_count,
        blob_bytes=blob_bytes,original_metadata_and_raw_bytes_copied_without_resampling=True,
        bookkeeping_status_reset_only=True,copy_setup_time_in_throughput=False)


def worker_entry(requests,replies,description,urdf,templates,output):
    from probtf_observation_fusion_demo.async_workers import event_inference_worker
    started=time.perf_counter()
    try:
        event_inference_worker(requests,replies,description,urdf,templates,output,
                               'per_effective_sensor_event')
    finally:
        usage=resource.getrusage(resource.RUSAGE_SELF)
        replies.put(dict(kind='benchmark_worker_usage',wall_seconds=time.perf_counter()-started,
            user_cpu_seconds=usage.ru_utime,system_cpu_seconds=usage.ru_stime,
            peak_RSS_MiB=usage.ru_maxrss/1024.,peak_RSS_convention='Linux ru_maxrss in KiB'))


def run_mode(mode,run,work,selected,actions,templates,timeout):
    from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore
    work.mkdir(parents=True,exist_ok=False)
    database=work/'observations.sqlite3'
    copy_record=clone_prefix(run,database,selected)
    context=mp.get_context('spawn')
    requests,replies=context.Queue(),context.Queue()
    worker=context.Process(target=worker_entry,args=(requests,replies,(run/'robot.urdf').read_text(),
        str(run/'robot.urdf'),templates,str(work)))
    counters=dict(raw_events=0,evaluated_events=0,disabled_or_command_notifications=0,
                  worker_requests=0,worker_replies=0,raw_blob_bytes_sent_to_worker=0)
    started=time.perf_counter();worker.start()
    error=None;usage=None
    def receive():
        message=replies.get(timeout=timeout)
        counters['worker_replies']+=1
        if message.get('failed'):
            raise RuntimeError(message.get('traceback','Worker failed.'))
        return message
    try:
        with SensorEventStore(database) as store:
            if mode=='batch':
                batch=store.claim_batch('benchmark',max_stamp=selected[-1]['stamp'],snapshot_id='benchmark-fixed-prefix')
                if batch.event_ids!=tuple(row['event_id'] for row in selected):
                    raise AssertionError('The batch does not cover the complete selected raw prefix.')
                counters['raw_events']=len(batch)
                counters['worker_requests']+=1
                requests.put(dict(kind='sensor_event_batch',batch=batch,database=str(database),ledger_steps=actions,
                                  joint_body_batch=False))
                while True:
                    message=receive()
                    if message.get('kind')=='inference_batch_completed':
                        counters['evaluated_events']=message['evaluated_event_count']
                        counters['disabled_or_command_notifications']=len(batch)-message['evaluated_event_count']
                        break
            else:
                enabled={item['sensor'] for item in templates}
                previous_index=None
                for selected_row in selected:
                    event=store.claim_next('benchmark',through_stamp=selected[-1]['stamp'])
                    if event is None or event.event_id!=selected_row['event_id']:
                        raise AssertionError('Serial selection skipped or reordered a raw event.')
                    counters['raw_events']+=1
                    sensor={'imu':'imu','image':'marker'}.get(event.kind)
                    use=(previous_index is None and event.kind=='joint') or sensor in enabled
                    if use:
                        index=int(event.payload['frame_index'])
                        steps=() if previous_index is None else actions[previous_index:index]
                        requests.put((event,index,steps));counters['worker_requests']+=1
                        counters['raw_blob_bytes_sent_to_worker']+=sum(len(blob) for blob in event.blobs)
                        message=receive()
                        if message['event_id']!=event.event_id:
                            raise AssertionError('The serial worker reply belongs to another event.')
                        previous_index=index;counters['evaluated_events']+=1
                    else:
                        counters['disabled_or_command_notifications']+=1
                    store.complete(event.event_id,'benchmark',result_metadata={'offline_benchmark':True})
            statuses=store.stats()
        finished=time.perf_counter()
        requests.put(None)
        while usage is None:
            message=receive()
            if message.get('kind')=='benchmark_worker_usage':usage=message
        worker.join(timeout=10.)
        if worker.is_alive():raise RuntimeError('Worker did not exit after its sentinel.')
    except Exception as exc:
        error=dict(type=type(exc).__name__,reason=str(exc))
        finished=time.perf_counter()
        worker.join(timeout=3.)
        if worker.is_alive():worker.terminate()
        worker.join(timeout=10.)
        with SensorEventStore(database) as store:
            statuses=store.stats()
        while usage is None:
            try:
                message=replies.get_nowait()
            except Empty:
                break
            if message.get('kind')=='benchmark_worker_usage':usage=message
    finally:
        requests.cancel_join_thread();replies.cancel_join_thread()
        requests.close();replies.close()
    return dict(mode=mode,status='passed' if error is None else 'failed',error=error,
        measured_wall_seconds=finished-started,
        raw_events_per_second=(None if error is not None else counters['raw_events']/(finished-started)),
        evaluated_events_per_second=(None if error is not None else counters['evaluated_events']/(finished-started)),
        raw_events_completed=statuses['counts'].get('completed',0),
        failed_trial_is_performance_measurement=False if error is not None else None,
        counters=counters,worker_usage=usage,database_status=statuses,
        working_database_copy=copy_record,worker_exitcode=worker.exitcode,
        output_directory=str(work),fresh_process=True,OS_page_cache_controlled=False,
        scope='Actual production event_inference_worker; includes worker initialization, event IPC, processing and per-event logs; excludes prefix-copy setup and final process teardown.')


def semantic_record(row):
    stiffness=row.get('stiffness',{})
    posterior=stiffness.get('posterior') or stiffness.get('retained_prior') or {}
    return clean(dict(event=row.get('sensor_event'),observation_ids=row.get('observation_ids'),
        stages=row.get('stages'),internal_projection=row.get('internal_projection'),
        frozen_body_charts=row.get('frozen_body_charts'),
        body_bayes_factor_surprise=row.get('bayes_factor_surprise'),
        candidate_measure=row.get('candidate_measure'),
        body_density_ratio=stiffness.get('body_marginal'),
        factor_ids=posterior.get('factor_ids'),density_id=posterior.get('density_id'),
        K_posterior=posterior,K_evidence=stiffness.get('bayes_factor_surprise'),
        K_proposal=stiffness.get('command_proposal'),
        proposal_selection=stiffness.get('proposal_selection')))


def compare_outputs(serial,batch):
    from itertools import zip_longest
    records=[];mismatches=[]
    for left,right in zip_longest(rows(serial/'estimation.jsonl'),rows(batch/'estimation.jsonl')):
        if left is None or right is None:
            mismatches.append(dict(reason='Different number of evaluated event records.'));break
        a,b=semantic_record(left),semantic_record(right)
        event=a['event']['event_id']
        categories={key:digest(a[key])==digest(b[key]) for key in a}
        numeric=[]
        def differences(x,y,path=''):
            if isinstance(x,dict) and isinstance(y,dict):
                for key in x.keys()&y.keys():differences(x[key],y[key],path+'/'+key)
            elif isinstance(x,list) and isinstance(y,list) and len(x)==len(y):
                for index,(u,v) in enumerate(zip(x,y)):differences(u,v,path+'/'+str(index))
            elif (isinstance(x,(float,int)) and not isinstance(x,bool)
                    and isinstance(y,(float,int)) and not isinstance(y,bool)):
                if x!=y:numeric.append((abs(float(x)-float(y)),path,float(x),float(y)))
        differences(a,b)
        record=dict(event_id=event,semantic_categories_bitwise_equal=categories,
            all_semantic_categories_bitwise_equal=all(categories.values()),
            maximum_numeric_absolute_difference=0. if not numeric else max(item[0] for item in numeric),
            largest_numeric_differences=[dict(absolute_difference=delta,path=path,serial=x,batch=y)
                for delta,path,x,y in sorted(numeric,reverse=True)[:5]])
        records.append(record)
        if not record['all_semantic_categories_bitwise_equal']:
            mismatches.append(dict(event_id=event,categories=[key for key,value in categories.items() if not value]))
    if not records:
        mismatches.append(dict(reason='No evaluated event records were produced; empty outputs are not equivalence evidence.'))
    return dict(evaluated_events_compared=len(records),all_bitwise_equal=bool(records) and not mismatches,
        mismatches=mismatches,events=records,
        scope='Every evaluated event: complete joint covariance/transforms, all observation/body/goal/factor/density IDs, all cross blocks, K proposal and every evidence value/score/Hessian. Explicit processing-time fields excluded; acquisition/source times retained.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',choices=('marker','imu_marker'))
    parser.add_argument('--prefix-seconds',type=float,default=.1)
    parser.add_argument('--max-stamp',type=float)
    parser.add_argument('--full-recorded',action='store_true')
    parser.add_argument('--reply-timeout-seconds',type=float,default=300.)
    options=parser.parse_args()
    run=options.run.resolve();output=options.output.resolve();output.mkdir(parents=True,exist_ok=False)
    if not math.isfinite(options.prefix_seconds) or options.prefix_seconds<0.:
        parser.error('--prefix-seconds must be finite and nonnegative.')
    case=options.case or run.name
    if case not in ('marker','imu_marker'):parser.error('Specify --case marker or imu_marker.')
    selected,coverage=select_prefix(run,options.prefix_seconds,options.max_stamp,options.full_recorded)
    import yaml
    from probtf_observation_fusion_demo.async_state import ActionStep
    end_index=max(json.loads(row['payload_json'])['frame_index'] for row in selected)
    actions=[]
    for row in rows(run/'commands.jsonl'):
        step=ActionStep(**row['step'])
        if step.index>=end_index:break
        if step.index!=len(actions):raise ValueError('Original actual command ledger has a gap.')
        actions.append(step)
    if len(actions)!=end_index:raise ValueError('Original actual ledger does not reach the selected frontier.')
    templates=tuple(yaml.safe_load((PACKAGE/'config'/(case+'.yaml')).read_text())['observations'])
    original_db_sha=sha(run/'observations.sqlite3')
    before=source_hashes()
    result=dict(schema='probtf_serial_batch_inference_benchmark_v1',original_run=str(run),case=case,
        coverage=coverage,actual_ledger_intervals=len(actions),original_database_open_mode='ro',
        raw_arrays_resampled=False,monte_carlo_used=False,source_hashes_before=before,
        source_URDF_sha256=sha(run/'robot.urdf'),source_ledger_sha256=sha(run/'commands.jsonl'),
        source_database_sha256_before=original_db_sha,results={})
    for mode in ('serial','batch'):
        result['results'][mode]=run_mode(mode,run,output/mode,selected,tuple(actions),templates,
                                        options.reply_timeout_seconds)
        (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result['results'][mode],indent=2),flush=True)
    result['comparison']=compare_outputs(output/'serial',output/'batch')
    result['speedup_wall_ratio']=(result['results']['serial']['measured_wall_seconds']/result['results']['batch']['measured_wall_seconds']
        if all(row['status']=='passed' for row in result['results'].values()) else None)
    after=source_hashes()
    result['source_unchanged_during_benchmark']=after==before
    result['source_files_changed_during_benchmark']=[name for name in sorted(set(before)|set(after))
                                                   if before.get(name)!=after.get(name)]
    result['timing_is_provisional_due_to_source_change']=not result['source_unchanged_during_benchmark']
    result['original_database_bytes_unchanged']=sha(run/'observations.sqlite3')==original_db_sha
    result['benchmark_script_sha256']=sha(Path(__file__))
    result['all_checks_passed']=(all(row['status']=='passed' for row in result['results'].values())
        and result['comparison']['all_bitwise_equal'] and result['source_unchanged_during_benchmark']
        and result['original_database_bytes_unchanged'])
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(summary=str(output/'summary.json'),all_checks_passed=result['all_checks_passed'],
        speedup_wall_ratio=result['speedup_wall_ratio'],comparison=result['comparison']),indent=2),flush=True)
    if not result['all_checks_passed']:raise SystemExit(1)


if __name__=='__main__':
    main()
