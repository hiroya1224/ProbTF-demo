import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

BASE=Path('/tmp/probtf_1244f3f_rviz_rerun_20261018')
OUTPUT=Path('/tmp/probtf_1244_objective_analysis_20261018')


def reverse_lines(path):
    with path.open('rb') as stream:
        stream.seek(0,2);end=stream.tell();tail=b''
        while end:
            count=min(end,1<<20);end-=count;stream.seek(end)
            pieces=(stream.read(count)+tail).split(b'\n');tail=pieces[0]
            for line in reversed(pieces[1:]):
                if line:yield line
        if tail:yield tail


def extract():
    OUTPUT.mkdir(exist_ok=True)
    for case in ('marker','imu_marker'):
        directory=BASE/case
        targets=(112.582,158.474,179.) if case=='marker' else (123.575,140.,179.)
        samples={}
        for line in (directory/'commands.jsonl').open():
            row=json.loads(line)
            t=row['step']['end_stamp']-1792080000.
            for target in targets:
                if target not in samples or abs(t-target)<abs(samples[target]['time']-target):
                    samples[target]=dict(time=t,command_record=row)
        packets={sample['command_record']['reaching_execution']['gradient_packet_id'] for sample in samples.values()}
        adoptions={}
        for line in (directory/'gradient_adoptions.jsonl').open():
            row=json.loads(line)
            if row.get('accepted') and row['packet']['packet_id'] in packets:
                adoptions[row['packet']['packet_id']]=row
        ids={row['request_id'] for row in adoptions.values()}
        controls={}
        # One backwards pass, ending at the earliest selected request.
        for line in reverse_lines(directory/'control.jsonl'):
            match=re.search(br'"request_id"\s*:\s*(\d+)',line[:300])
            if match and int(match[1]) in ids:
                row=json.loads(line)
                controls[row['request_id']]=row
            if len(controls)==len(ids):break
        for target,sample in samples.items():
            pid=sample['command_record']['reaching_execution']['gradient_packet_id']
            adoption=adoptions[pid];control=controls[adoption['request_id']]
            cp=control['prediction']['command_path']
            references=[]
            for path in (directory/'stiffness_jobs').glob('*.json'):
                item=json.loads(path.read_text())
                if item['source_stamp']==cp['source_stamp']:references.append(item)
            assert len(references)==1
            sample.update(target_time=target,case=case,adoption=adoption,
                control=control,matching_job_reference=references[0])
            out=OUTPUT/(case+'_'+str(int(target))+'_source.json')
            out.write_text(json.dumps(sample,separators=(',',':'),allow_nan=False))
            print(case,target,'time',sample['time'],'request',control['request_id'],
                'source',control['candidate_identity']['stamp']-1792080000.,
                'archive',references[0]['job_sequence'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Extract the six original actually used request snapshots; no inference.')
    parser.add_argument('--input-run',type=Path,default=BASE)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();BASE=args.input_run;OUTPUT=args.output
    extract()
