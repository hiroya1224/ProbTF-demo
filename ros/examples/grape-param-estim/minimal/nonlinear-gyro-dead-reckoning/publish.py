#!/usr/bin/env python3
"""Select the small replayable inputs, four separate cases and figure tables."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(source,destination):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Use a new publication directory to preserve previous results')
    summary=json.loads((source/'results/summary.json').read_text())
    destination.mkdir(parents=True,exist_ok=True)
    copied=[]
    def copy(path,relative):
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        if sha(path)!=sha(target): raise IOError('Published copy differs from source')
        copied.append(dict(path=str(relative),source=str(path),sha256=sha(target),bytes=target.stat().st_size))
    for name in ('input.npz','input.json','input_model_diagnostic.npz',
                 'input_model_diagnostic.json','input_model_diagnostic.png','rk4_step_audit.json',
                 'input_mapping_audit.json'):
        copy(source/name,Path(name))
    for name in ('summary.json','comparison_table.csv','noise_overview.png','noise_overview.pdf',
                 'torque_profile.npz','torque_profile.json'):
        copy(source/'results'/name,Path(name))
    for case in summary['results']:
        label=format(case['sigma_deg_s'],'g').replace('.','p')
        relative=Path('sigma_'+label+'_deg_s')
        for name in ('comparison.pdf','rpy.png','omega.png','uncertainty.png','metrics.json',
                     'constant.npz','linear_forced.npz','nonlinear.npz'):
            copy(source/'results'/relative/name,relative/name)
    lines=['# 固定した5秒区間の数値結果','',
        'SO(3)測地角のRMSEと、非線形法の終端RPYの±2σ半幅。全単位deg。',
        'σは各軸の角速度標準偏差（deg/s）であり、測定値そのものへ乱数を追加していない。','',
        '| σ [deg/s] | 一定角速度 RMSE | 線形強制 RMSE | 非線形 RMSE | 非線形終端 Roll 2σ | Pitch 2σ | Yaw 2σ |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for case in summary['results']:
        methods=case['methods'];width=methods['nonlinear']['rpy_final_two_sigma_deg']
        values=[case['sigma_deg_s'],*[methods[m]['geodesic_rmse_deg'] for m in
            ('constant','linear_forced','nonlinear')],*width]
        lines.append('| '+' | '.join(f'{value:.3f}' for value in values)+' |')
    lines.extend(['','非線形法はこの固定入力モデルでは実測再現を改善していない。公称J・指令トルク・過去余剰平均の不整合が残る。',
        'ノイズを広げると帯は広がるが、平均の大きなずれは解消しない。',
        'RPYの軸別RMSE・周期性を考慮した終端誤差は `comparison_table.csv` と各条件の `metrics.json` に保存した。',
        '帯は同じ有限6点求積積法則の記述的な±2標準偏差であり、Gaussianや95%被覆を保証しない。','',
        '0.5 Hz以外の評価gyroは角速度の参照図と入力モデルの事後診断だけに使う。予測へは戻していない。',
        '各条件のPDFは3ページで、RPY・角速度・広がりと姿勢誤差を順に示す。','',
        '[方法と既存処理の説明](../../README.md)',''])
    (destination/'RESULTS.md').write_text('\n'.join(lines))
    files=[]
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            if path.stat().st_size>=50_000_000: raise ValueError('Do not publish artifacts >=50 MB')
            files.append(dict(path=str(path.relative_to(destination)),bytes=path.stat().st_size,sha256=sha(path)))
    manifest=dict(schema='grape/nonlinear-gyro-publication/v1',source=str(source),
        scope='new independent experiment; previous IMU results/core unchanged',
        raw_bag_or_video_included=False,noise_cases=len(summary['results']),
        code_sha256={p.name:sha(p) for p in HERE.glob('*.py')},files=files,
        byte_exact_source_copies=copied,total_bytes_excluding_manifest=sum(f['bytes'] for f in files),
        maximum_file_bytes=max(f['bytes'] for f in files))
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(destination=str(destination),files=len(files)+1,
                         bytes_excluding_manifest=manifest['total_bytes_excluding_manifest'])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args();publish(args.source,args.destination)


if __name__=='__main__':main()
