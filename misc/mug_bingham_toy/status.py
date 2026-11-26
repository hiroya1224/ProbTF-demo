#!/usr/bin/env python3
"""Read-only rendering/training progress, suitable for a tmux window."""
from pathlib import Path
import argparse
from datetime import datetime
import json
import time


HERE=Path(__file__).resolve().parent


def show(run):
    print(datetime.now().astimezone().isoformat(timespec='seconds'),flush=True)
    for split,expected in (('train',10000),('val',1000),('test',1000)):
        folder=HERE/'data'/split
        count=sum(1 for _ in (folder/'images').glob('*.png'))
        ready=(folder/'metadata.json').exists() and (folder/'poses.npz').exists()
        print(f'{split}: {count}/{expected} images; labels complete={ready}',flush=True)
    print(f'Run: {run}',flush=True)
    log=run/'train.log'
    if log.exists():
        lines=log.read_text().splitlines()
        progress=[line for line in lines if line.startswith('{')]
        if progress:
            print('Latest batch/epoch: '+progress[-1],flush=True)
    candidates=[run/'metrics.jsonl',run/'status.json',run/'report.json',run/'final_report.json',run/'test_metrics.json']
    for path in candidates:
        if path.exists():
            try:
                if path.suffix=='.jsonl':
                    lines=path.read_text().splitlines()
                    value=json.loads(lines[-1]) if lines else {}
                else:
                    value=json.loads(path.read_text())
                print(path.name+': '+json.dumps(value,ensure_ascii=False),flush=True)
            except (ValueError,OSError):
                print(path.name+': writing',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--run',type=Path,default=HERE/'runs/vgg16_10000_20261126')
    args=parser.parse_args()
    while True:
        if args.watch:
            print('\033[2J\033[H',end='')
        show(args.run)
        if not args.watch:
            break
        time.sleep(5)
