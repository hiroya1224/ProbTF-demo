#!/usr/bin/env python3
"""Copy only the completed experiment's selected small scientific artifacts."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Use an empty publication destination')
    # An incomplete trial, cache or smoke directory is not a publication.
    for name in ('summary.json', 'fit.json', 'identifiability.json', 'data_flow_verification.json',
                 'fit_report.pdf', 'comparison_table.csv', 'INDEX.md', 'RESULTS.md'):
        if not (source/name).is_file():
            raise FileNotFoundError(source/name)
    files = [p for p in sorted(source.rglob('*')) if p.is_file()
             and p.suffix in ('.json', '.npz', '.csv', '.pdf', '.png', '.md')]
    if any(p.stat().st_size >= 50_000_000 for p in files):
        raise ValueError('Do not publish artifacts >=50 MB')
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in files:
        relative = path.relative_to(source)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha(target) != sha(path):
            raise IOError('Publication differs from completed source')
        copied.append(dict(path=str(relative), bytes=path.stat().st_size, sha256=sha(path)))
    manifest = dict(schema='grape/mocap-refined-dynamics-publication/v1',
        source=str(source), files=copied, bytes_excluding_manifest=sum(p['bytes'] for p in copied),
        raw_bags_videos_or_intermediate_caches_included=False,
        original_archived_inputs_reused=True, old_results_overwritten=False)
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(destination=str(destination), files=len(copied)+1,
                         bytes_excluding_manifest=manifest['bytes_excluding_manifest'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    publish(args.source, args.destination)
