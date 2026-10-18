#!/usr/bin/env python3
"""Lossless storage of this task's stopped /tmp diagnostic runs only.

Every uncompressed SHA is verified by a gzip roundtrip before replacing the
original. No directory removal and no modification of user/source files.
Restore: for each manifest entry, gzip-decompress compressed_path to path and
verify sha256. Numerical measurements were completed before compression.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path


ROOT=Path('/tmp/probtf_uncertainty_ablation_20261018/live')


def digest(path,compressed=False):
    hasher=hashlib.sha256()
    opener=gzip.open if compressed else open
    with opener(path,'rb')as stream:
        while True:
            data=stream.read(1024*1024)
            if not data:break
            hasher.update(data)
    return hasher.hexdigest()


def compress_case(directory,*,confirmed_stopped=False,minimum_bytes=1000000):
    directory=Path(directory).resolve()
    if directory.parent!=ROOT or directory.is_symlink():
        raise ValueError('Only a direct case directory in this diagnostic live root is allowed.')
    if not confirmed_stopped or not(directory/'diagnostic_summary.json').is_file():
        raise ValueError('A completed summary and explicit stopped-worker assertion are required.')
    setup=json.loads((directory/'diagnostic_setup.json').read_text())
    if not setup.get('wall_clock_async')or setup.get('intervention')is None:
        raise ValueError('Not an identified uncertainty diagnostic run.')
    manifest=directory/'lossless_compression.json'
    record=json.loads(manifest.read_text())if manifest.exists()else dict(
        method='gzip level1, exact original bytes verified by decompression SHA before unlink',
        root=str(directory),files=[],original_bytes=0,compressed_bytes=0)
    # Explicit file types: source/RViz/settings and small summaries stay put.
    suffixes={'.jsonl','.json','.sqlite3','.pickle','.bin','.png','.npz'}
    candidates=sorted(p for p in directory.rglob('*')if p.is_file()and not p.is_symlink()
        and p.suffix in suffixes and p.stat().st_size>=minimum_bytes
        and p.name not in ('diagnostic_summary.json','diagnostic_setup.json','lossless_compression.json')
        and 'frozen_source'not in p.parts)
    for path in candidates:
        target=path.with_name(path.name+'.gz')
        if target.exists():
            raise ValueError('Refusing to overwrite existing compressed artifact: '+str(target))
        temporary=target.with_name(target.name+'.writing')
        original_size=path.stat().st_size;original_sha=digest(path)
        with path.open('rb')as source,temporary.open('wb')as destination:
            with gzip.GzipFile(filename='',fileobj=destination,mode='wb',compresslevel=1,mtime=0)as zipped:
                while True:
                    data=source.read(1024*1024)
                    if not data:break
                    zipped.write(data)
        if digest(temporary,compressed=True)!=original_sha:
            raise ValueError('Lossless gzip verification failed: '+str(path))
        compressed_size=temporary.stat().st_size
        os.replace(temporary,target)
        record['files'].append(dict(path=str(path.relative_to(directory)),
            compressed_path=str(target.relative_to(directory)),sha256=original_sha,
            original_bytes=original_size,compressed_bytes=compressed_size))
        record['original_bytes']+=original_size;record['compressed_bytes']+=compressed_size
        pending=manifest.with_name(manifest.name+'.writing')
        pending.write_text(json.dumps(record,indent=2)+'\n');os.replace(pending,manifest)
        # This exact original is retained and SHA-verified in target. The
        # transformation is reversible and restricted to our own artifacts.
        path.unlink()
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--confirmed-stopped',action='store_true')
    args=parser.parse_args()
    value=compress_case(args.directory,confirmed_stopped=args.confirmed_stopped)
    print(json.dumps({key:value[key]for key in ('root','original_bytes','compressed_bytes')},indent=2))
