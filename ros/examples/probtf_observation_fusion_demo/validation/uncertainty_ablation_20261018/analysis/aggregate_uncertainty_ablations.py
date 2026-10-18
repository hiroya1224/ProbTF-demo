#!/usr/bin/env python3
"""Aggregate saved ablation audits; no inference, controller or worker replay.

JSON/Markdown stay bounded: no full event IDs, candidate arrays or cutoff
records. Partial trials remain partial. Standard-library only. Run with an
optional experiment root argument; output defaults to ROOT/analysis.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import time


DEFAULT_ROOT = Path('/tmp/probtf_uncertainty_ablation_20261018')


def open_text(path):
    return path.open() if path.is_file() else gzip.open(str(path)+'.gz', 'rt')


def exists(path):
    return path.is_file() or Path(str(path)+'.gz').is_file()


def load(path):
    if not exists(path):
        return None
    with open_text(path) as stream:
        return json.load(stream)


def statistic(audit, name):
    value = audit.get(name)
    return value if isinstance(value, dict) and value.get('n', 0) else None


def first_reach(run, summary):
    """Use saved prefix metrics; stream only if this older run omitted them."""
    when = summary.get('first_reach_time_s')
    arc = summary.get('arc_to_first_reach_m')
    if when is not None and arc is not None:
        source = 'saved diagnostic trajectory summary'
        error = None
    elif summary.get('reached') and exists(run/'physical_trajectory.jsonl'):
        arc, previous, error, when = 0., None, None, None
        with open_text(run/'physical_trajectory.jsonl') as stream:
            for line in stream:
                row = json.loads(line)
                tip = row['tip']
                if previous is not None:
                    arc += math.dist(tip, previous)
                previous = tip
                if row['reached']:
                    when, error = row['time'], row['physical_error_m']
                    break
        source = 'original physical trajectory streamed only through first reach'
    else:
        return None
    if when is None:
        return None
    return dict(time_s=when, arc_before_first_reach_m=arc,
        physical_error_at_first_reach_m=error,
        remaining_observed_duration_s=summary['physical_duration_s']-when,
        remaining_arc_m=summary['tip_arc_m']-arc,
        final_frame_reached=summary.get('final_frame_reached'), source=source,
        reached_once_is_not_sustained_reaching=True)


def case_record(run, analysis):
    setup = load(run/'diagnostic_setup.json')
    summary = load(run/'diagnostic_summary.json')
    if not setup or setup.get('case') not in ('marker', 'imu_marker') or not summary:
        return None
    audit_path = analysis/(run.name+'_compact_audit.json')
    if not exists(audit_path):
        audit_path = analysis/(run.name+'_diagnostic_summary_extension.json')
    audit = load(audit_path) or {}
    cutoff = load(run/'diagnostic_physical_cutoff.json')
    recovery = load(run/'diagnostic_raw_recovery.json')
    complete = bool(summary.get('completed_requested_duration',
        False if cutoff is None else cutoff.get('finished', False)))
    start = None
    if cutoff is not None:
        start = cutoff['physical_stamp']-summary['physical_duration_s']
    elif audit.get('last_recorded_physical_frame'):
        start = (audit['last_recorded_physical_frame']['stamp']
                 -summary['physical_duration_s'])
    detection = audit.get('raw_preview_detection', {})
    if not detection and exists(run/'camera_previews.jsonl'):
        count = missing = 0
        first_missing = last_stamp = None
        with open_text(run/'camera_previews.jsonl') as stream:
            for line in stream:
                row = json.loads(line)
                count += 1
                last_stamp = row['stamp']
                if not row['visible']:
                    missing += 1
                    if first_missing is None:
                        first_missing = row['stamp']
        detection = dict(saved_count=count, missing_count=missing,
                         first_missing_stamp=first_missing, last_saved_stamp=last_stamp)
    row = dict(case=run.name, sensor_configuration=setup['case'],
        intervention=setup['intervention'], trial_truth_assisted=setup.get('physical_truth_received'),
        completed_requested_duration=complete, requested_duration_s=setup['duration_s'],
        observed_physical_duration_s=summary['physical_duration_s'],
        failure_message=summary.get('failure_message'),
        reached_once=summary.get('reached'), final_frame_reached=summary.get('final_frame_reached'),
        final_error_m=summary['final_error_m'], minimum_error_m=summary['minimum_error_m'],
        tip_arc_m=summary['tip_arc_m'], net_shortening_per_arc=summary['net_shortening_per_arc'],
        first_reach=first_reach(run, summary),
        physical_cutoff_available=cutoff is not None,
        ages_at_physical_cutoff=None, actual_Hz=None, K_jobs_at_physical_cutoff=None,
        raw_at_physical_cutoff=None, raw_after_run_recovery=None,
        executed_gradient_source_age_s=None, executed_K_source_age_s=None,
        saved_control_body_observation_age_s=None,
        finite_VI=dict(saved_branches=audit.get('variational_branches'),
            finite_nonstationary_branches=audit.get('finite_nonstationary_branch_count'),
            gradient_inf=statistic(audit, 'final_gradient_inf'),
            marker_position_residual_m=statistic(audit, 'final_fitted_position_residual_m'),
            gravity_tangent2_residual=statistic(audit, 'final_fitted_gravity_tangent2_residual_norm'),
            scope='K-worker saved prefix, not guaranteed to cover all completed body updates'),
        raw_detection=dict(saved_count=detection.get('saved_count'),
            missing_count=detection.get('missing_count'),
            first_missing_time_s=(None if start is None or detection.get('first_missing_stamp') is None
                else detection['first_missing_stamp']-start),
            last_saved_time_s=(None if start is None or detection.get('last_saved_stamp') is None
                else detection['last_saved_stamp']-start),
            scope='saved raw preview window only, not necessarily all acquired images'),
        prepared_marker_interventions=audit.get('prepared_marker_interventions'),
        source_changed=summary.get('source_changed'),
        inputs=dict(summary=str(run/'diagnostic_summary.json'),
            audit=str(audit_path) if exists(audit_path) else None,
            cutoff=str(run/'diagnostic_physical_cutoff.json') if cutoff is not None else None))
    if cutoff is not None:
        db = cutoff['sensor_event_database']
        row.update(ages_at_physical_cutoff=dict(
            body_observation_s=cutoff.get('observation_age_s'),
            gradient_source_s=cutoff.get('gradient_source_age_s')),
            executed_gradient_source_age_s=statistic(audit, 'actual_interval_gradient_source_age_s'),
            executed_K_source_age_s=statistic(audit, 'actual_interval_K_source_age_s'),
            saved_control_body_observation_age_s=statistic(audit, 'control_body_observation_age_s'),
            actual_Hz={name:cutoff.get('actual_'+name+'_hz') for name in ('motion', 'camera', 'control')},
            K_jobs_at_physical_cutoff=dict(completed=cutoff['stiffness_jobs_completed'],
                submitted=cutoff['stiffness_jobs_submitted']),
            raw_at_physical_cutoff=dict(acquired_requests=cutoff['acquisition_requests'],
                archived=cutoff['acquisition_archived'], reserved_database_rows=db['total'],
                database_completed_rows=db['counts'].get('completed', 0),
                database_pending_rows=db['counts'].get('pending', 0),
                database_awaiting_payload_rows=db['counts'].get('awaiting_payload', 0)))
    else:
        row['missing_cutoff_reason'] = ('Trial ended before requested physical cutoff; '
            'end-of-request body/K ages, Hz and processed counts are unavailable, never zero-filled.')
    if recovery is not None:
        row['raw_after_run_recovery'] = dict(all_raw_archived=recovery['all_raw_archived'],
            total_requests=recovery['raw_request_count'], recovered_event_count=recovery['recovered_event_count'],
            database_completed_rows=recovery['after']['counts'].get('completed', 0),
            database_pending_rows=recovery['after']['counts'].get('pending', 0),
            results_used_in_completed_motion=recovery['results_used_in_completed_motion'])
    if setup['intervention'] in ('oracle_body', 'oracle_body_goal', 'all_oracle'):
        row['age_interpretation'] = ('Recursive estimator ages remain recorded; control body '
            'centers are truth-replaced independently at each control capture.')
    elif setup['intervention'] == 'true_K':
        row['age_interpretation'] = ('K source age names the original estimator packet for '
            'availability/provenance; the substituted true K value is constant.')
    return row


def number(value, digits=2):
    return '—' if value is None else format(value, '.'+str(digits)+'f')


def median(record):
    return number(None if record is None else record.get('median'), 3)


def markdown(data):
    cases = data['cases']
    text = ['# 真値介入試行の実行時監査', '',
        f"現在の保存資料は完了{data['completed_case_count']}試行・途中終了{data['partial_case_count']}試行。"
        '未完了の動作を180秒試行へ付け替えない。既存compact監査・終端manifest・raw回収記録を読むだけで、推定や制御を再実行しない。', '',
        '| 試行 | 実時間 s | 状態 | 終端 mm | 軌跡 m | 終端身体観測age s | 勾配age中央値 s | K source age中央値 s |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in cases:
        ages = row['ages_at_physical_cutoff'] or {}
        text.append('| '+row['case']+' | '+number(row['observed_physical_duration_s'], 3)
            +' | '+('完了' if row['completed_requested_duration'] else '途中終了')
            +' | '+number(1000*row['final_error_m'])+' | '+number(row['tip_arc_m'], 3)
            +' | '+number(ages.get('body_observation_s'))
            +' | '+median(row['executed_gradient_source_age_s'])
            +' | '+median(row['executed_K_source_age_s'])+' |')
    text += ['', 'K source ageは取得情報の出自であり、true_Kの真値が古くなったという意味ではない。'
        'oracle_body・oracle_body_goal・all_oracleでは制御身体の中心を各captureで真値へ置換する。'
        'その再帰推定のageを、そのまま実行身体中心の古さと呼ばない。', '',
        '| 試行 | 物理終端raw保存/取得 | DB処理済み/取得 | K完了/提出 | 保存VI非定常/全枝 | raw欠落/保存preview |',
        '|---|---:|---:|---:|---:|---:|']
    for row in cases:
        raw, jobs = row['raw_at_physical_cutoff'], row['K_jobs_at_physical_cutoff']
        vi, det = row['finite_VI'], row['raw_detection']
        text.append('| '+row['case']+' | '+('—' if raw is None else f"{raw['archived']}/{raw['acquired_requests']}")
            +' | '+('—' if raw is None else f"{raw['database_completed_rows']}/{raw['acquired_requests']}")
            +' | '+('—' if jobs is None else f"{jobs['completed']}/{jobs['submitted']}")
            +f" | {vi['finite_nonstationary_branches']}/{vi['saved_branches']}"
            +f" | {det['missing_count']}/{det['saved_count']} |")
    text += ['', 'DB処理済みには無効センサや時間被覆選択で融合しなかった行も含む。'
        'VIは独立K workerが保存したprefixであり、全身体更新数とは限らない。'
        'raw previewの保存窓は試行間で異なるため、180秒全体の欠落率ではない。'
        'IMUの2次元重力方向残差とマーカーの位置残差を混ぜていない。', '',
        'raw回収は動作終了後であり、回収結果を終了済みの制御へ使っていない。'
        '回収件数はdrain後の不足数なので、物理終端の取得−保存の差とも一致しない。', '']
    for row in cases:
        prefix = row['first_reach']
        if prefix is not None:
            text.append(row['case']+'：最初の到達は'+number(prefix['time_s'], 6)
                +'秒、そこまでの軌跡は'+number(prefix['arc_before_first_reach_m'], 6)
                +' m。その後の観測済み'+number(prefix['remaining_observed_duration_s'], 3)
                +'秒の移動量は'+number(prefix['remaining_arc_m'], 6)+' m。'
                +('最終frameも到達条件内。' if prefix['final_frame_reached'] else '最終frameは到達条件外。'))
        if row['failure_message']:
            text.append(row['case']+'の途中終了理由：`'+row['failure_message']+'`。'
                '終端manifestがないため、終端age・Hz・処理数は不明としている。')
    text += ['', '各一回の非同期試行では軌跡・検出継続・処理負荷・観測frontierも変わる。'
        '身体と目標の共通位置ずれを片側だけ訂正する介入もあるため、終端距離だけで不定性を順位付けしない。'
        '未完了baselineは比較に含めない。', '', '詳細は `CROSS_CASE_RUNTIME_SUMMARY.json`。']
    return '\n'.join(text)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.root/'analysis'
    output.mkdir(exist_ok=True)
    cases = [case_record(path, args.root/'analysis') for path in sorted((args.root/'live').iterdir())]
    cases = [row for row in cases if row is not None]
    data = dict(audit_wall_stamp=time.time(), completed_case_count=sum(
        row['completed_requested_duration'] for row in cases), partial_case_count=sum(
        not row['completed_requested_duration'] for row in cases), cases=cases,
        audit_changed_trial_laws=False,
        notes=['All data are original saved trial summaries or bounded audit statistics.',
            'Partial trials lack requested end-of-run ages/Hz/counts; these are unknown, not zero.',
            'No incomplete IMU baseline is used as a completed comparison.'])
    (output/'CROSS_CASE_RUNTIME_SUMMARY.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    (output/'CROSS_CASE_RUNTIME_SUMMARY.md').write_text(markdown(data))
    print(json.dumps(dict(completed=data['completed_case_count'], partial=data['partial_case_count'],
        cases=[row['case'] for row in cases]), sort_keys=True))


if __name__ == '__main__':
    main()
