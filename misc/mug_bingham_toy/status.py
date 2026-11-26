#!/usr/bin/env python3
"""Read-only rendering/training progress, suitable for a tmux window."""
from pathlib import Path
import argparse
from datetime import datetime
import hashlib
import json
import time


HERE=Path(__file__).resolve().parent
DEFAULT_DATA=HERE/'data/cylinder_center_20261127'
DEFAULT_RUN=HERE/'runs/vgg16_10000_cylinder_center_20261127'
SPLITS=(('train',10000,1234),('val',1000,1235),('test',1000,1236))
MESH_METADATA='mug_cylinder_center_metadata.json'
COLORED_MESH='mug_cylinder_centered_colored.ply'


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def validate_launch(data,run,assets=None):
    """Reject the wrong pivot, stale mesh metadata, or an existing trained run.

    This is a concrete input/provenance check, not a numerical quality gate.
    train.py subsequently verifies the complete image/label bytes itself.
    """
    data,run=Path(data),Path(run)
    assets=HERE/'assets' if assets is None else Path(assets)
    mesh=json.loads((assets/MESH_METADATA).read_text())
    if mesh.get('center_native') != [40.,40.,45.]:
        raise ValueError('Mesh origin must be the cylindrical center [40,40,45]')
    if mesh.get('colored_mesh') != COLORED_MESH:
        raise ValueError('Mesh metadata must identify the cylindrical-center colored asset')
    if mesh.get('colored_mesh_sha256') != sha256(assets/COLORED_MESH):
        raise ValueError('Corrected colored asset bytes differ from their metadata')
    if mesh.get('original_sha256') != sha256(assets/mesh['original_stl']):
        raise ValueError('Original STL bytes differ from corrected mesh metadata')
    rendered=json.loads((data/'render_metadata.json').read_text())
    if rendered.get('image_size') != [640,360]:
        raise ValueError('Expected whole 640x360 images')
    if rendered.get('mesh') != mesh:
        raise ValueError('Dataset was rendered from different mesh/origin metadata')
    for split,count,seed in SPLITS:
        folder=data/split
        metadata=json.loads((folder/'metadata.json').read_text())
        if (metadata.get('count'),metadata.get('seed')) != (count,seed):
            raise ValueError(f'{split}: expected {count} images with pose seed {seed}')
        if metadata.get('mesh') != mesh:
            raise ValueError(f'{split}: mesh/origin metadata differs from corrected assets')
        if metadata.get('poses_sha256') != sha256(folder/'poses.npz'):
            raise ValueError(f'{split}: pose-label bytes differ from render metadata')
        if sum(1 for _ in (folder/'images').glob('*.png')) != count:
            raise ValueError(f'{split}: complete image count does not match metadata')
    artifacts=('best.pt','last.pt','config.json','provenance.json','metrics.jsonl',
               'test_metrics.json','test_predictions.npz')
    if any((run/name).exists() for name in artifacts):
        raise ValueError('This fresh launch requires a run without prior training artifacts')
    return {'launch':'ready from scratch','data':str(data.resolve()),'run':str(run.resolve()),
            'center_native':mesh['center_native'],'colored_mesh_sha256':mesh['colored_mesh_sha256'],
            'epochs':50,'batch_size':16,'training_seed':20261126}


def show(run,data=DEFAULT_DATA):
    print(datetime.now().astimezone().isoformat(timespec='seconds'),flush=True)
    print(f'Data: {data}',flush=True)
    for split,expected,_ in SPLITS:
        folder=data/split
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
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--check-launch',action='store_true',help='validate corrected data and a fresh output directory, then exit')
    args=parser.parse_args()
    if args.check_launch:
        try:
            result=validate_launch(args.data,args.run)
        except (OSError,ValueError,KeyError) as error:
            parser.exit(1,f'Launch refused: {error}\n')
        print(json.dumps(result),flush=True)
        raise SystemExit(0)
    while True:
        if args.watch:
            print('\033[2J\033[H',end='')
        show(args.run,args.data)
        if not args.watch:
            break
        time.sleep(5)
