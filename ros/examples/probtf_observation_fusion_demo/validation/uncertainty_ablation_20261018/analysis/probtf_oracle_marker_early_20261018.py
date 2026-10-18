import json,gzip,pathlib,hashlib,pickle,sys,math
import numpy as np
from probtf_observation_fusion_demo.async_artifacts import capture_candidate_identity
from probtf_observation_fusion_demo.async_state import ActionStep,predict_result
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_control import control_components
from probtf_observation_fusion_demo.body_native import pose_query
from probtf_observation_fusion_demo.moving_scene import EquilibriumBody
from probtf.geometry import quat_to_rotmat
ROOT=pathlib.Path('/tmp/probtf_uncertainty_ablation_20261018');START=1792080000.
def open_text(p):return gzip.open(str(p)+'.gz','rt')if not p.exists()else p.open()
def rows(p):
 with open_text(p)as f:
  for l in f:
   if l.strip():yield json.loads(l)
def restore(p,row):
 pred=row['prediction'];cp=pred['command_path'];refs=[]
 for path in (p/'stiffness_jobs').glob('*.json'):
  ref=json.loads(path.read_text())
  if ref['source_stamp']==cp['source_stamp']and not ref['initialize_only']:refs.append(ref)
 assert len(refs)==1, (p.name,row['request_id'],cp['source_stamp'],len(refs))
 ref=refs[0];path=pathlib.Path(ref['path'])
 payload=path.read_bytes()if path.exists()else gzip.open(str(path)+'.gz','rb').read()
 assert hashlib.sha256(payload).hexdigest()==ref['payload_sha256']
 job=pickle.loads(payload);original=job['result'];assert original.prior.state_id==cp['source_prior_state_id']
 ref=cp['ledger_reference'];begin,end=ref['start_index'],ref['end_index_exclusive'];steps=[]
 for d in rows(p/ref['path']):
  st=d['step'];i=st['index']
  if i>=end:break
  if begin<=i<end:steps.append(ActionStep(**st))
 axes=cp['axes'];ids=tuple(x['edge_id']for x in axes);actions=tuple(dict(joint_index=x['joint_index'],axis=x['axis'])for x in axes)
 value=predict_result(original,steps,target_stamp=pred['present_stamp'],frame_index=pred['frame_index'],body_ids=ids,actions=actions,ledger_reference=ref)
 assert capture_candidate_identity(value,parameter_identity=row['candidate_identity']['parameter'])==row['candidate_identity']
 return value
out=dict(diagnostic_only=True,physical_truth_used_only_for_posthoc_comparison=True,cases={})
for name in ['marker_baseline','marker_oracle_marker']:
 p=ROOT/'live'/name;selected={.4:None,.65:None,1.:None,2.:None,5.:None,10.:None};errs={t:float('inf')for t in selected}
 for r in rows(p/'control.jsonl'):
  g=r['controller'];t=g['source_stamp']-START
  if t>12.:break
  if g['status']!='ok':continue
  for target in selected:
   if abs(t-target)<errs[target]:selected[target]=r;errs[target]=abs(t-target)
 physical={}
 for i,d in enumerate(rows(p/'physical_trajectory.jsonl')):
  if d['time']>12.:break
  physical[i]=d
 base=EquilibriumBody(p/'robot.urdf',[350,125,95,60,40,30]);robot=base.robot;robot._fk_update(np.asarray(physical[0]['actual_joints']));invbase=np.linalg.inv(np.asarray(robot.data.oMf[robot.base_fid].homogeneous))
 truegoal=np.array([1.16,-.04,.72]) # replaced from known recorded physical scene below
 # The exact same fixed world goal is recorded by the independent goal-oracle
 # run; reading this score reference does not change any inference input.
 truegoal=np.asarray(next(rows(ROOT/'live/marker_oracle_goal/diagnostic_control_truth.jsonl'))['goal_position'])
 samples=[]
 for target,row in selected.items():
  if row is None:continue
  ctrl=row['controller'];t=ctrl['source_stamp']-START;frame=physical[row['index']]
  assert frame['stamp']==ctrl['source_stamp']and frame['command']==ctrl['source_command']
  snap=row['snapshot'];origin={q['frame_label']:q for q in snap['queries']if q['axis']=='origin'}
  H=np.asarray(origin['hand']['mean']);G=np.asarray(origin['switch_goal']['mean']);truthH=np.asarray(frame['tip']);
  try:
   result=restore(p,row);_,goals=control_components(result);restore_status='exact candidate identity match';restored=True
   path=tuple((name,not reverse)for name,reverse in reversed(result.marker_path))+tuple(result.hand_path)
  except AssertionError as e:
   goals=();restore_status='unavailable: source is an uninformative/missing event without a K-job archive; '+str(e);restored=False
  covs=[];relmeans=[];relerrors=[]
  goaloffset=np.array([0.,-.24,.045])
  for item in goals:
   q=pose_query(item.state,path,query_id='diagnostic_relative_at_original_source',frame='marker')
   covs.append(np.linalg.eigvalsh(q.covariance[:3,:3]).tolist());relmeans.append(q.mean[:3].tolist());relerrors.append(float(np.linalg.norm(q.mean[:3]-goaloffset)))
  curvature=np.asarray(ctrl['curvature']);eig=np.linalg.eigvalsh(curvature);g=np.asarray(ctrl['gradient']);task=np.asarray(ctrl['task_gradient']);view=np.asarray(ctrl.get('weighted_view_gradient',np.zeros(6)))
  samples.append(dict(target_t=target,t=t,request_id=row['request_id'],full_candidate_identity_reconstructed_exactly=restored,relative_covariance_restore_status=restore_status,physical_error_m=frame['physical_error_m'],hand_mean_error_m=float(np.linalg.norm(H-truthH)),goal_mean_error_m=float(np.linalg.norm(G-truegoal)),estimated_root_distance_m=float(np.linalg.norm(H-G)),hand_error_vector=(H-truthH).tolist(),goal_error_vector=(G-truegoal).tolist(),relative_position_covariance_eigenvalues_m2=covs,relative_position_mean=relmeans,estimated_relative_mean_distances_m=relerrors,selected_gradient_norm=float(np.linalg.norm(g)),task_gradient_norm=float(np.linalg.norm(task)),weighted_view_gradient_norm=float(np.linalg.norm(view)),local_H_eigenvalues=eig.tolist(),minimum_modal_damping_ratio=float(8/(2*np.sqrt(1e-6*eig[-1]))),highest_undamped_frequency_rad_s=float(np.sqrt(1e-6*eig[-1])),source_objective=ctrl['objective_at_source'],selected_K=ctrl.get('selected_K'),body_observation_age_s=ctrl['source_stamp']-ctrl['body_observation_source_stamp']))
 branches=[];observation_checks=[]
 for d in rows(p/'estimation.jsonl'):
  event=d.get('sensor_event',{});t=event.get('acquisition_stamp',START)-START
  if t>12.:break
  vi=d.get('variational_observation_update',{})
  for b in vi.get('branch_diagnostics',[]):
   fits=[]
   for fit in b.get('source_evaluator_diagnostics',{}).get('propagated_observations',[]):
    residual=np.asarray(fit['predicted_mean'])-np.asarray(fit['observed_mean'])
    fits.append(dict(position_mean_residual_m=float(np.linalg.norm(residual[:3])),rotation_chart_mean_residual_rad=float(np.linalg.norm(residual[3:])),predicted_position_covariance_eigenvalues_m2=np.linalg.eigvalsh(np.asarray(fit['predicted_covariance'])[:3,:3]).tolist()))
   branches.append(dict(event_id=event.get('event_id'),t=t,parent_label=b['parent_label'],finite_nonstationary_iterate_used=b['finite_nonstationary_iterate_used'],final_gradient_inf=b['final_gradient_inf'],completed_iterations=b['completed_iterations'],native_mean_fit=fits))
  if name=='marker_oracle_marker':
   for native in vi.get('native_likelihoods',[]):
    capture=json.loads((p/'diagnostic_marker_truth'/(event['event_id']+'.json')).read_text())
    rel=np.linalg.solve(capture['camera_pose'],capture['marker_pose'])
    observation_checks.append(dict(event_id=event['event_id'],t=t,exact_original_stamp=(capture['stamp']==event['acquisition_stamp']),exact_original_command=(capture['command']==event['sent_command']),translation_residual_m=float(np.linalg.norm(np.asarray(native['observed_mean'])[:3]-rel[:3,3])),rotation_matrix_maximum_error=float(np.max(np.abs(quat_to_rotmat(native['rotation_wxyz'])-rel[:3,:3]))),native_covariance_eigenvalues=np.linalg.eigvalsh(native['covariance']).tolist()))
 out['cases'][name]=dict(selected_controls=samples,early_VI_branches=branches,oracle_native_truth_checks=observation_checks)
 print(name,[(r['t'],r['physical_error_m'],r['selected_gradient_norm'],r['minimum_modal_damping_ratio'],r['hand_mean_error_m'],r['goal_mean_error_m'])for r in samples],flush=True)
out['notes']=['Selected early control candidates were reconstructed from their original immutable K-job source and actual executed ledger; candidate identity matches exactly.', 'No new sensor update, optimization or K selection was run.', 'Damping ratios describe each frozen affine local model, not whole switched closed-loop stability.', 'The finite covariance native oracle removes image ambiguity when detection succeeds; body remains variationally estimated and may not converge.']
(ROOT/'analysis/oracle_marker_early.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
