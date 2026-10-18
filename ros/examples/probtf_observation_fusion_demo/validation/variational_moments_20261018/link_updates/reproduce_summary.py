import hashlib,json,math
from pathlib import Path
out=Path('/tmp/probtf_variational_top12_physical_audit_20261018')
d=json.loads((out/'details.json').read_text())
summary={k:d[k]for k in('schema','source_run','selection_sha256','runtime_source_manifest_sha256','runtime_git_HEAD','analysis_source_sha256','current_postrun_native_symmetry_repair_used_for_queries','native_repair_scope','physical_truth_used_only_for_posthoc_comparison','production_changed','optimizer_or_detector_run','scope','analysis_sources_unchanged','limits')}
summary['analysis_library_sha256']=d['analysis_body_backend']['library_sha256']
summary['runtime_body_backend']=d['runtime_native_backends'].get('body',d['runtime_native_backends'])
summary['cases']={}
for case,content in d['cases'].items():
 rows=[]
 for e in content['events']:
  bf=e['bayes_factor_surprise'];dominant=max(e['branches'],key=lambda b:b['posterior_log_weight'])
  branches=[]
  for b in e['branches']:
   branches.append(dict({k:b[k]for k in('parent_label','posterior_log_weight','initial_F','final_F','final_KL','final_gradient_inf','completed_iterations','stationarity_verified','finite_nonstationary_iterate_used','raw_gaussian_sha256')},native=[dict(sensor=n['sensor'],mean_closure_max_abs=n['saved_raw_forward_vs_independent_core_mean_max_abs'],covariance_closure_max_abs=n['saved_raw_forward_vs_independent_core_covariance_max_abs'],NLL_closure_abs=n['saved_raw_NLL_vs_independent_core_abs'],original_parent_fit=n['original_parent'],raw_VI_fit=n['raw'],canonical_fit=n['canonical'],physical_point_in_native_likelihood=n['physical_truth_in_native_likelihood'])for n in b['native']]))
  ep=e['original_prediction']['K_epochs'][-1]if e['original_prediction']['K_epochs']else None
  rows.append(dict(job_sequence=e['job_sequence'],event_id=e['event']['event_id'],original_sequence=e['event']['sequence'],acquisition_elapsed_s=e['event']['stamp']-1792080000.,frame_index=e['event']['frame_index'],payload_hash=e['event']['payload_hash'],archive_sha256=e['archive_sha256'],initial_observation_time=e['initial_observation_time'],original_frame_stamp_command_verified=e['frame_stamp_command_exact'],physical_tip_FK_vs_saved_m=e['physical_tip_crosscheck_error_m'],adoption_elapsed_s=e['body_adoption_elapsed_s'],body_adoption_source_age_s=e['body_adoption_source_age_s'],adopted_at_or_after_physical_end=e['body_ready_after_physical_cutoff'],retained_K_packet_id=e['body_ready_adoption'].get('retained_K_packet_id')if e['body_ready_adoption']else None,prior_state_id=e['prior_state_id'],canonical_state_id=e['canonical_state_id'],posterior_reset_probability=bf['posterior_change_probability'],posterior_keep_log_probability=bf['posterior_keep_log_probability'],posterior_reset_log_probability=bf['posterior_change_log_probability'],log_bayes_factor=bf['log_bayes_factor'],rod_tip=e['links']['rod_tip'],tool0=e['links']['tool0'],relative_camera_marker=e['relative_camera_marker_truth'],map=e['map_truth'],per_link_posterior_truth_errors={k:v['posterior']for k,v in e['links'].items()},prediction_latest_original_K_epoch=ep,branches=branches))
 allnative=[n for e in rows for b in e['branches']for n in b['native']]
 allbranches=[b for e in rows for b in e['branches']]
 summary['cases'][case]=dict(selected_count=len(rows),initial_count=sum(e['initial_observation_time']for e in rows),noninitial_count=sum(not e['initial_observation_time']for e in rows),selected_branches=len(allbranches),stationary_branches=sum(b['stationarity_verified']for b in allbranches),all_original_frame_stamp_command_verified=all(e['original_frame_stamp_command_verified']for e in rows),physical_tip_FK_vs_saved_max_m=max(e['physical_tip_FK_vs_saved_m']for e in rows),raw_mean_closure_max_abs=max(n['mean_closure_max_abs']for n in allnative),raw_covariance_closure_max_abs=max(n['covariance_closure_max_abs']for n in allnative),raw_NLL_closure_max_abs=max(n['NLL_closure_abs']for n in allnative),events=rows)
summary['supplementary_KL_reverse_reconstruction']=[dict(case=case,job_sequence=e['job_sequence'],parent_label=b['parent_label'],saved_KL=b['final_KL'],KL_from_saved_full_P=b['independent_KL'],absolute_discrepancy=b['saved_KL_vs_independent_abs'],conditioning=b['independent_KL_reconstruction_conditioning'])for case,c in d['cases'].items()for e in c['events']for b in e['branches']if b.get('saved_KL_vs_independent_abs',0)>1e-5]
summary['supplementary_KL_limit']='The optimizer computes KL stably from its original whitened Cholesky coordinates. Reconstructing them from the saved nearly singular full covariance can amplify roundoff; original Cholesky coordinates are not retained in this archive. The supplementary differences are disclosed, not certified as a defect or erased. They do not affect the independent core mu/S/NLL comparison.'
(out/'expanded_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
compact={k:v for k,v in summary.items()if k not in('cases','runtime_body_backend')}
compact['cases']={}
for case,content in summary['cases'].items():
 data={k:v for k,v in content.items()if k!='events'}
 data['events']=[]
 for e in content['events']:
  native=max(e['branches'],key=lambda b:b['posterior_log_weight'])['native'][0]
  data['events'].append(dict(job_sequence=e['job_sequence'],event_id=e['event_id'],frame_index=e['frame_index'],payload_hash=e['payload_hash'],archive_sha256=e['archive_sha256'],acquisition_elapsed_s=e['acquisition_elapsed_s'],adoption_elapsed_s=e['adoption_elapsed_s'],body_adoption_source_age_s=e['body_adoption_source_age_s'],adopted_at_or_after_physical_end=e['adopted_at_or_after_physical_end'],initial_observation_time=e['initial_observation_time'],prior_state_id=e['prior_state_id'],canonical_state_id=e['canonical_state_id'],posterior_reset_probability=e['posterior_reset_probability'],posterior_keep_log_probability=e['posterior_keep_log_probability'],posterior_reset_log_probability=e['posterior_reset_log_probability'],rod_tip_update_m=e['rod_tip']['same_q_modal_update_m'],rod_tip_O_P_mean_update_m=e['rod_tip']['same_q_O_P_mean_update_m'],rod_tip_prior_modal_truth_error_m=e['rod_tip']['prior']['modal_position_error_m'],rod_tip_posterior_modal_truth_error_m=e['rod_tip']['posterior']['modal_position_error_m'],rod_tip_posterior_O_P_mean_truth_error_m=e['rod_tip']['posterior']['O_P_mean_position_error_m'],rod_tip_posterior_modal_angle_truth_error_rad=e['rod_tip']['posterior']['modal_rotation_error_rad'],tool0_prior_modal_truth_error_m=e['tool0']['prior']['modal_position_error_m'],tool0_posterior_modal_truth_error_m=e['tool0']['posterior']['modal_position_error_m'],map_posterior_modal_truth_error_m=e['map']['posterior']['modal_position_error_m'],camera_marker_posterior_modal_truth_error_m=e['relative_camera_marker']['posterior']['modal_position_error_m'],camera_marker_posterior_modal_angle_truth_error_rad=e['relative_camera_marker']['posterior']['modal_rotation_error_rad'],canonical_native_fit=native['canonical_fit'],physical_point_in_native_likelihood=native['physical_point_in_native_likelihood'],prediction_latest_original_K_epoch=e['prediction_latest_original_K_epoch'],branch_diagnostics=[{k:b[k]for k in('parent_label','posterior_log_weight','initial_F','final_F','final_KL','final_gradient_inf','completed_iterations','stationarity_verified')}for b in e['branches']]))
 compact['cases'][case]=data
(out/'summary.json').write_text(json.dumps(compact,indent=2,allow_nan=False)+'\n')
lines=['# 180秒 VI 実走の取得時身体更新と真値の後解析','',
'対象は `/tmp/probtf_variational_commonchart_180s_live_20261018`。保存されたリンクmodal更新幅上位6件を各ケースから選び、計12件だけ調べた。全試行の精度統計ではない。初期観測の補正を別表示し、物理運動と同じ取得時刻・指令での推定補正を分けた。','',
'両物理動作は180秒を完了したが、IMUのpost-motion drainは内部modal gapの丸めskew判定で停止した。原runを完了済み推定と偽らない。解析は、その後の内部対称積の修正済みnativeで保存されたGaussianを照会する。最適化や検出器は再実行せず、原mean/P・全cross・枝質量を変更していない。','',
'## 独立照会の一致','',
'保存VI最終Gaussianを同じ原prior chartから再表現し、Python coreの平均・共分散伝播と固定native観測chartで独立に照会した。元VIで評価した平均・共分散・期待NLLと一致した。旧方式で問題になった「計算中の観測分布と実際に再合成した分布のずれ」は、この12件では再現していない。一方、これは推定精度や最適化収束の証明ではない。','',
'| ケース | 最大平均差 | 最大共分散差 | Stationary枝数 |','|---|---:|---:|---:|']
for case,c in summary['cases'].items():lines.append(f"| {case} | {c['raw_mean_closure_max_abs']:.3g} | {c['raw_covariance_closure_max_abs']:.3g} | {c['stationary_branches']}/{c['selected_branches']} |")
lines+=['','## 取得時の棒先端精度','',
'各原physical frameの保存実関節角をsaved URDFでFKし、同じ取得stamp/qと厳密一致した12組だけを比較した。最近傍時刻への丸めはしていない。棒先端FKと保存された実棒先端位置の差は全件0。真値はこの後解析だけで使用した。以下はmodal棒先端の位置であり、補正Gaussian平均の誤差はJSONに別保存した。','',
'| ケース/job | 取得秒 | 同じq/tの棒先端補正 mm | 真値誤差 prior→post mm | map真値誤差 post mm | Reset確率 | 採用秒 |','|---|---:|---:|---:|---:|---:|---:|']
for case,c in summary['cases'].items():
 for e in sorted(c['events'],key=lambda e:e['acquisition_elapsed_s']):
  tip=e['rod_tip'];init=' (初期)'if e['initial_observation_time']else'';aft=' (動作後)'if e['adopted_at_or_after_physical_end']else''
  lines.append(f"| {case}/{e['job_sequence']}{init} | {e['acquisition_elapsed_s']:.3f} | {tip['same_q_modal_update_m']*1000:.2f} | {tip['prior']['modal_position_error_m']*1000:.2f}→{tip['posterior']['modal_position_error_m']*1000:.2f} | {e['map']['posterior']['modal_position_error_m']*1000:.2f} | {e['posterior_reset_probability']:.3g} | {e['adoption_elapsed_s']:.3f}{aft} |")
lines+=['','## 身体が正しく見えることと、実際の精度の違い','',
'IMU job251では画像とのカメラ相対位置平均残差は2.74mm、相対姿勢modalの真位置誤差は3.55mmまで小さくなった。しかし根元基準の実棒先端誤差は205.97mm、保持marker mapの位置誤差は250.90mmだった。画像相対の整合だけから各リンク・手先の絶対姿勢が正しいとはいえない。姿勢を見た目で評価するだけでは、この系統的ずれを見落とし得る。','',
'同じK epoch（推定source5.641秒）のIMU job249→画像job251で逆向きの補正が起きた。重力観測249は棒先端真値誤差209.65→130.28mmへ改善した一方、画像251は123.43→205.97mmへ悪化した。両者の取得間には実際の運動もあるため、単純な差分を同一物理状態の反復とは扱わない。各更新単体は、それぞれ同じq/tの前後を比較している。重力の局所2方向と画像相対の制約、保持map・全cross・priorの整合が十分でない可能性があるが、この12件だけで可観測性や制御振動の全原因を確定したわけではない。','',
'原画像観測自体も常に真値付近ではない。Marker job55の中央観測は真相対位置から42.28mm/0.229rad、job73は7.12mm/0.314radずれていた。他方IMU job251のnative位置誤差は1.10mmだった。観測外れと、正しい相対観測でも残る根元基準のずれを混同しない。','',
'大きい更新はcanonical単峰化だけの追加ずれでもなかった。上位事象の多くではKeep/Resetの一方が支配的で、そのraw VI枝の同じcore分布がcanonicalになっていた。独立raw/core平均照会が一致することは、選ばれた枝の精度や負Fによる質量近似の適切さを保証しない。','',
'## 遅延と収束の留保','',
'IMU job451（取得168.351秒）は180秒以後の採用なので、物理動作中の振動を引き起こした事象として扱わない。全branchを一括して比較すると、Marker上位6件の12枝はすべて有限非stationary、IMU上位6件では12枝中1枝だけstationaryだった。例えば451はKeep/ResetのFが3047/2149、勾配最大成分が7.88/1866。有限計算を継続する規約に沿った結果だが、収束済みや精度良好とは説明できない。K出自と身体採用時計はJSONで別記録した。','',
'補助検算として保存full Pを再白色化してKLを逆算すると、IMU441 Keepで記録KL647.978に対して0.3173、451 Keepで0.000664の差が出た。441の再構成白色共分散の最小固有値は約2.86e−14で、条件数は約1e14だった。元optimizerは白色Cholesky座標から直接KLを計算しているため、保存Pからの逆再構成と同じ精度を保証できない。この差は隠さずJSONに残したが、本体Fの実装誤りと断定せず、独立平均・共分散・NLLの照会一致とも分けている。','',
'コード・数値閾値・候補・観測・制御は変更していない。巨大行列やrawは追加せず、詳細はこの/tmpの`details.json`、選定用小要約は`summary.json`に保存した。']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
manifest={str(p):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in(out/'summary.json',out/'REPORT.md',Path('/tmp/probtf_variational_top12_physical_audit_20261018.py'),Path(__file__))}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(out/'summary.json', (out/'summary.json').stat().st_size)
print(out/'REPORT.md',(out/'REPORT.md').stat().st_size)
