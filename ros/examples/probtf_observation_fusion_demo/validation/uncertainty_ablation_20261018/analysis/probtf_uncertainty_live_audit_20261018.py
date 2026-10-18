import json,math,pathlib,sqlite3,collections,time,hashlib
import numpy as np
ROOT=pathlib.Path('/tmp/probtf_uncertainty_ablation_20261018')
OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
START=1792080000.

def rows(path):
 with path.open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def stats(values):
 a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
 return dict(n=int(len(a)),min=float(a.min()),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max()))if len(a)else dict(n=0)
def summary_case(name):
 p=ROOT/'live'/name
 cutoff=json.loads((p/'diagnostic_physical_cutoff.json').read_text());summary=json.loads((p/'diagnostic_summary.json').read_text())
 vi=[];events={};fit=[];count=0
 for d in rows(p/'estimation.jsonl'):
  count+=1;ev=d.get('sensor_event',{});key=ev.get('event_id')
  detail=d.get('variational_observation_update',{})
  branches=detail.get('branch_diagnostics',[])
  if branches:events[key]=dict(event_id=key,t=ev.get('acquisition_stamp',START)-START,
    bayes_change=d.get('bayes_factor_surprise',{}).get('posterior_change_probability'))
  for b in branches:
   row={k:b.get(k)for k in ['parent_label','label','finite_nonstationary_iterate_used','final_gradient_inf','stationarity_verified','optimizer_success','completed_iterations','final_KL','final_objective','final_expected_negative_log_likelihood_terms','prior_support_rank']}
   row.update(event_id=key,t=ev.get('acquisition_stamp',START)-START)
   vi.append(row)
   for v in b.get('source_evaluator_diagnostics',{}).get('propagated_observations',[]):
    residual=np.asarray(v['predicted_mean'])-np.asarray(v['observed_mean'])
    fit.append(dict(event_id=key,t=row['t'],label=b['label'],position_residual_m=float(np.linalg.norm(residual[:3])),rotation_chart_residual_rad=float(np.linalg.norm(residual[3:])),dimension=len(residual)))
 c=sqlite3.connect('file:'+str(p/'observations.sqlite3')+'?mode=ro',uri=True)
 db=dict(status_counts=c.execute('select status,count(*) from events group by status').fetchall(),kind_status_counts=c.execute('select kind,status,count(*),min(stamp),max(stamp) from events group by kind,status').fetchall());c.close()
 ad=list(rows(p/'body_ready_adoptions.jsonl'))
 bodylag=[r['physical_stamp']-r['body_source_stamp']for r in ad]
 cmdages=[];Kages=[];dt=[]
 for d in rows(p/'commands.jsonl'):
  step=d['step'];ex=d['reaching_execution'];duration=step['end_stamp']-step['start_stamp'];age=ex.get('gradient_source_age_s')
  if age is not None:cmdages.append((float(age),duration))
  if step.get('source_stamp')is not None:Kages.append((step['end_stamp']-step['source_stamp'],duration))
 targets=[0.,30.,60.,90.,120.,150.,179.];selected={t:None for t in targets};errors={t:float('inf')for t in targets}
 controlages=[];goalages=[];ratios=[];statuses=collections.Counter();controlcount=0
 for d in rows(p/'control.jsonl'):
  controlcount+=1;ctrl=d['controller'];statuses[ctrl['status']]+=1;t=ctrl['source_stamp']-START
  if ctrl.get('body_observation_source_stamp')is not None:controlages.append(ctrl['source_stamp']-ctrl['body_observation_source_stamp'])
  if ctrl.get('goal_observation_source_stamp')is not None:goalages.append(ctrl['source_stamp']-ctrl['goal_observation_source_stamp'])
  fac=ctrl.get('selection_factors',{});tg=fac.get('task_gradient');vg=fac.get('view_gradient')
  if tg is not None and vg is not None and np.linalg.norm(tg)>0:ratios.append(2e6*np.linalg.norm(vg)/np.linalg.norm(tg))
  for target in targets:
   if abs(t-target)<errors[target]:
    errors[target]=abs(t-target)
    selected[target]=dict(index=d['index'],request_id=d['request_id'],t=t,stamp=ctrl['source_stamp'],command=ctrl['source_command'],origins=[q for q in d['snapshot']['queries']if q['axis']=='origin'],gradient=ctrl.get('gradient'),task_gradient=tg,view_gradient=vg,selected_K=ctrl.get('selected_K'),body_observation_stamp=ctrl.get('body_observation_source_stamp'),goal_observation_stamp=ctrl.get('goal_observation_source_stamp'),objective_at_source=ctrl.get('objective_at_source'))
 trajectory={i:d for i,d in enumerate(rows(p/'physical_trajectory.jsonl'))}
 goalfile=ROOT/'live/marker_oracle_goal/diagnostic_control_truth.jsonl'
 truth_by_request={d['request_id']:d for d in rows(goalfile)}
 truegoal=np.asarray(next(iter(truth_by_request.values()))['goal_position'])
 samples=[]
 # Only a selected sample needs FK; use the exact archived URDF, never solve
 # dynamics/equilibrium or derive a new physical posture from K.
 from probtf_observation_fusion_demo.moving_scene import EquilibriumBody
 model=EquilibriumBody(p/'robot.urdf',[350.,125.,95.,60.,40.,30.])
 robot=model.robot;robot._fk_update(np.asarray(summary.get('initial_command',[-.15,-1.85,2.35,0.,-.95,0.])))
 base_inv=np.linalg.inv(np.asarray(robot.data.oMf[robot.base_fid].homogeneous))
 for target,row in selected.items():
  phys=trajectory[row['index']]
  if phys['stamp']!=row['stamp']or phys['command']!=row['command']:raise ValueError('Control sample lost exact physical index/stamp/command')
  robot._fk_update(np.asarray(phys['actual_joints']))
  origins={q['frame_label']:q for q in row.pop('origins')}
  H=np.asarray(origins['hand']['mean']);G=np.asarray(origins['switch_goal']['mean']);xH=np.asarray(phys['tip'])
  Hmode=np.asarray(origins['hand']['mode']);Gmode=np.asarray(origins['switch_goal']['mode'])
  deltaH=H-xH;deltaG=G-truegoal
  row.update(target_t=target,physical_hand=xH.tolist(),physical_goal=truegoal.tolist(),estimated_hand_mean=H.tolist(),estimated_goal_mean=G.tolist(),estimated_hand_mode=Hmode.tolist(),estimated_goal_mode=Gmode.tolist(),hand_mean_error_m=float(np.linalg.norm(deltaH)),goal_mean_error_m=float(np.linalg.norm(deltaG)),hand_mode_error_m=float(np.linalg.norm(Hmode-xH)),goal_mode_error_m=float(np.linalg.norm(Gmode-truegoal)),estimated_mean_distance_m=float(np.linalg.norm(H-G)),physical_distance_m=float(np.linalg.norm(xH-truegoal)),relative_mean_error_m=float(np.linalg.norm(deltaH-deltaG)),hand_error_vector=deltaH.tolist(),goal_error_vector=deltaG.tolist(),common_offset_cosine=None if not np.linalg.norm(deltaH)*np.linalg.norm(deltaG)else float(np.dot(deltaH,deltaG)/np.linalg.norm(deltaH)/np.linalg.norm(deltaG)),hand_position_sigma_eigen_m=np.sqrt(np.linalg.eigvalsh(origins['hand']['covariance'])).tolist(),goal_position_sigma_eigen_m=np.sqrt(np.linalg.eigvalsh(origins['switch_goal']['covariance'])).tolist())
  link_errors={}
  for label,q in origins.items():
   if label not in ('hand','marker','switch_goal'):
    fid=robot.get_frame_id('camera_optical_frame'if label=='camera'else label)
    pose=base_inv@np.asarray(robot.data.oMf[fid].homogeneous)
    link_errors[label]=dict(mean_error_m=float(np.linalg.norm(np.asarray(q['mean'])-pose[:3,3])),mode_error_m=float(np.linalg.norm(np.asarray(q['mode'])-pose[:3,3])))
  row['link_position_errors']=link_errors
  samples.append(row)
 def weighted_mean(pairs):return sum(x*w for x,w in pairs)/sum(w for x,w in pairs)if pairs and sum(w for x,w in pairs)else None
 keys=['actual_motion_hz','actual_camera_hz','actual_control_hz','actual_estimate_hz','actual_update_hz','estimate_stamp','physical_stamp','observation_age_s','gradient_source_age_s','acquisition_requests','acquisition_archived','stiffness_jobs_submitted','stiffness_jobs_completed','body_estimation_drained','stiffness_estimation_drained','active_gradient_K_est']
 out=dict(case=name,physical_summary=summary,physical_cutoff={k:cutoff.get(k)for k in keys},posthoc_database_snapshot=db,body_adoptions=len(ad),body_result_adoption_age_s=stats(bodylag),effective_events_in_K_saved_estimation=len(events),estimation_saved_rows=count,variational_branches=len(vi),finite_nonstationary_branch_count=sum(bool(v['finite_nonstationary_iterate_used'])for v in vi),stationary_branch_count=sum(bool(v['stationarity_verified'])for v in vi),final_gradient_inf=stats([v['final_gradient_inf']for v in vi]),final_fitted_position_residual_m=stats([v['position_residual_m']for v in fit]),final_fitted_rotation_chart_residual_rad=stats([v['rotation_chart_residual_rad']for v in fit]),control_artifact_rows=controlcount,control_status_counts=dict(statuses),control_body_observation_age_s=stats(controlages),control_goal_observation_age_s=stats(goalages),actual_interval_gradient_source_age_s=stats([x for x,w in cmdages]),actual_interval_gradient_source_age_time_weighted_mean_s=weighted_mean(cmdages),actual_interval_K_source_age_s=stats([x for x,w in Kages]),actual_interval_K_source_age_time_weighted_mean_s=weighted_mean(Kages),view_task_gradient_norm_ratio=stats(ratios),events=list(events.values()),variational_branch_diagnostics=vi,variational_fit_diagnostics=fit,selected_control_samples=samples,notes=['Physical truth used only for this offline scoring.', 'estimation.jsonl is written by independent K worker and may omit already completed body updates still queued for K; do not treat its rows as complete body-event coverage.', 'Posthoc SQLite can include recovery appended captures; physical-cutoff counts are separate.', 'Selected control samples exactly match original physical frame index, timestamp and sent command.', 'A corrected Gaussian mean is not itself a rigid link posture.'])
 (OUT/(name+'_diagnostic_summary_extension.json')).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 print(name,json.dumps({k:out[k]for k in ['body_adoptions','effective_events_in_K_saved_estimation','variational_branches','finite_nonstationary_branch_count','final_gradient_inf','final_fitted_position_residual_m','control_body_observation_age_s','actual_interval_gradient_source_age_s','view_task_gradient_norm_ratio']},indent=2),flush=True)
 print('samples',json.dumps([{k:r[k]for k in ['t','hand_mean_error_m','goal_mean_error_m','estimated_mean_distance_m','physical_distance_m','common_offset_cosine']}for r in samples],indent=2),flush=True)
 return out
for name in ['marker_baseline','marker_oracle_goal']:summary_case(name)
