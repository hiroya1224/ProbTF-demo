#!/usr/bin/env python3
"""Publish the rate/noise cases separately, reusing the unchanged archived input."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(source,destination):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):raise ValueError('Use a new publication directory')
    destination.mkdir(parents=True,exist_ok=True);copied=[]
    def copy(path,relative):
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        if sha(path)!=sha(target):raise IOError('Published bytes differ from source')
        copied.append(dict(path=str(relative),source=str(path),bytes=target.stat().st_size,sha256=sha(target)))
    for path in sorted((source/'results').rglob('*')):
        if path.is_file() and path.suffix in ('.npz','.json','.csv','.pdf','.png'):
            copy(path,path.relative_to(source/'results'))
    for path in sorted(source.glob('sg_benchmark*')):
        if path.is_file() and path.suffix in ('.npz','.json','.pdf','.png'):copy(path,Path('sg_diagnostic')/path.name)
    copy(source/'display_approximation_audit.json',Path('display_approximation_audit.json'))
    summary=json.loads((destination/'summary.json').read_text())
    index=['# 取得頻度・角速度ノイズの比較','',
        '各周波数・各ノイズは別フォルダ／別PDF。sigmaは各軸の1観測あたりの標準偏差deg/s。',
        '初期姿勢・5秒区間・制御入力・公称モデル・固定した過去余剰トルク平均は共通。','',
        '| sigma [deg/s] | 0.5 Hz | 2 Hz | 5 Hz |','|---:|---|---|---|']
    def label(value):return format(value,'g').replace('.','p')
    for sigma in summary['sigmas_deg_s']:
        links=[f'[PDF](rate_{label(rate)}_hz/sigma_{label(sigma)}_deg_s/comparison.pdf)' for rate in summary['rates_hz']]
        index.append('| '+format(sigma,'g')+' | '+' | '.join(links)+' |')
    index.extend(['','## sigma=2.5 deg/sのSO(3) RMSE [deg]','',
        '| 取得頻度 | 観測数 | 一定角速度 | 線形強制 | 非線形 |','|---:|---:|---:|---:|---:|'])
    for row in summary['results']:
        if row['sigma_deg_s']!=2.5:continue
        values=[row['rate_hz'],row['original_gyro_count'],*[row['methods'][m]['geodesic_rmse_deg'] for m in ('constant','linear_forced','nonlinear')]]
        index.append('| '+' | '.join(f'{value:.3f}' for value in values)+' |')
    index.extend(['','[sigma=2.5の一覧図](rate_overview_sigma_2p5.pdf) / [全条件のCSV](comparison_table.csv)','',
        '[SG由来角速度の周波数対照](sg_diagnostic/sg_benchmark.pdf)は、力学モデル・ノイズ伝播を使わない別の運動学診断。',
        '状態の候補法則は同じ6点増分因子の積であり、回転モーメントは全交差を含めて伝播。RPYの中心・帯は同じ回転モーメントからのdelta-method表示近似。',
        '[方法・仮定・検証](../../README.md)',''])
    (destination/'INDEX.md').write_text('\n'.join(index))
    files=[]
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            if path.stat().st_size>=50_000_000:raise ValueError('Do not include artifacts >=50 MB')
            files.append(dict(path=str(path.relative_to(destination)),bytes=path.stat().st_size,sha256=sha(path)))
    manifest=dict(schema='grape/rate-noise-publication/v1',files=files,source_copies=copied,
        source_sha256={p.name:sha(p) for p in HERE.glob('*.py')},
        bytes_excluding_manifest=sum(f['bytes'] for f in files),
        archived_input_reused=summary['input_source'],archived_torque_reused=summary['torque_source'],
        raw_bags_videos_intermediates_copied=False,original_experiment_overwritten=False)
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(destination=str(destination),files=len(files)+1,bytes_excluding_manifest=manifest['bytes_excluding_manifest'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args();publish(args.source,args.destination)
