#!/usr/bin/env python3
"""Publish only selected small diagnostic arrays, figures and original clocks."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(source,destination):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Use a new output directory to preserve previous results')
    destination.mkdir(parents=True,exist_ok=True)
    copied=[]
    def copy(path,relative):
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        if sha(path)!=sha(target): raise IOError('Published copy differs from original')
        copied.append(dict(path=str(relative),source=str(path),bytes=target.stat().st_size,sha256=sha(target)))
    for name in ('input.npz','input.json','roundtrip_step_sensitivity.json'):
        copy(source/name,Path(name))
    for path in sorted((source/'results').iterdir()):
        if path.is_file() and path.suffix in ('.npz','.json','.csv','.png','.pdf'):
            copy(path,Path(path.name))
    summary=json.loads((destination/'summary.json').read_text())
    lines=['# SG微分・gyro照合と再積分の数値結果','',
        '元の時計・測定値は変更していない。時間ずれだけを動かし、bias・scale・軸回転は合わせていない。',
        'lagの正はgyroがmocap由来角速度より後という規約。','',
        '| SG窓 [ms] | 主区間lag [ms] | 広域lag [ms] | 生比較のvector RMSE [deg/s] | lag適用後 [deg/s] |',
        '|---:|---:|---:|---:|---:|']
    import math
    for r in summary['results']:
        factor=180/math.pi
        values=[r['sg_window_seconds']*1000,r['primary']['best_lag_seconds']*1000,
            r['wide']['best_lag_seconds']*1000,r['primary']['before']['vector_rmse_rad_s']*factor,
            r['primary']['after']['vector_rmse_rad_s']*factor]
        lines.append('| '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines.extend(['','| SG窓 [ms] | 全レート再積分 RMSE [deg] | 最大 [deg] | 終端 [deg] | 0.5 Hz保持 RMSE [deg] |',
                  '|---:|---:|---:|---:|---:|'])
    for r in summary['results']:
        a=r['roundtrip']['full_rate'];b=r['roundtrip']['sparse_0p5hz']
        values=[r['sg_window_seconds']*1000,a['geodesic_rmse_deg'],a['geodesic_max_deg'],a['geodesic_final_deg'],b['geodesic_rmse_deg']]
        lines.append('| '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines.extend(['','[方法・評価範囲](../../README.md)','',
        'gyro_comparison.pdf：原時計の比較、時間ずれ適用後、lag走査。',
        'sg_roundtrip.pdf：全レート再積分、0.5 Hz保持、誤差曲線。',
        '同じmocapから微分・積分した非因果的な整合検査であり、独立したIMU推定の精度ではない。',''])
    (destination/'RESULTS.md').write_text('\n'.join(lines))
    files=[]
    for p in sorted(destination.iterdir()):
        if p.is_file():
            if p.stat().st_size>=50_000_000: raise ValueError('Do not include files >=50 MB')
            files.append(dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)))
    manifest=dict(schema='grape/sg-clock-roundtrip-publication/v1',files=files,source_copies=copied,
        source_sha256={p.name:sha(p) for p in HERE.glob('*.py')},
        bytes_excluding_manifest=sum(f['bytes'] for f in files),original_prediction_changed=False,
        model_or_filter_parameters_retuned=False,raw_bag_or_video_included=False)
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(output=str(destination),files=len(files)+1,bytes_excluding_manifest=manifest['bytes_excluding_manifest'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args();publish(args.source,args.destination)
