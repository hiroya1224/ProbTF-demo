"""Saved-data/6D-affine audit only; no posterior, FK, equilibrium or plant replay."""
from pathlib import Path
import json,hashlib,math,xml.etree.ElementTree as ET,sys,importlib.util
import numpy as np
D=Path('/tmp/probtf_uncertainty_ablation_20261018/live/imu_marker_oracle_marker');O=Path(__file__).parent;ORIGIN=1792080000.
def rows(name):
 with(D/name).open()as f:
  for line in f:
   if line.strip():yield json.loads(line)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def cosine(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b);return None if d==0 else float(a@b/d)
setup=json.load((D/'diagnostic_setup.json').open());failure=list(rows('running_gradient_fail_safe.jsonl'))[-1];adopts=[r for r in rows('gradient_adoptions.jsonl')if r.get('accepted')and r['packet']['status']=='ok'];packets=[]
for i,r in enumerate(adopts):
 p=r['packet'];g=np.asarray(p['gradient']);H=np.asarray(p['curvature']);lam=np.linalg.eigvalsh(H);u=np.asarray(r['position']);v=np.asarray(r['velocity']);ng=np.asarray(r['model_gradient_at_adoption']);sf=p['selection_factors'];switch=None
 if i:
  old=adopts[i-1]['packet'];og=np.asarray(old['gradient'])+np.asarray(old['curvature'])@(u-np.asarray(old['source_command']));switch=dict(old_local_gradient=og.tolist(),new_local_gradient=ng.tolist(),cosine=cosine(og,ng),gradient_jump_norm=float(np.linalg.norm(ng-og)),acceleration_jump_norm=float(1e-6*np.linalg.norm(ng-og)),H_eigenmax_ratio=float(lam[-1]/np.linalg.eigvalsh(old['curvature'])[-1]),K_changed=p['selected_K']!=old['selected_K'])
 branches=[]
 # The signed covariance eigenvalues in log do not alter anything here.
 for mode in p['local_model_diagnostics'].get('task_components',[]):branches.append(mode)
 packets.append(dict(request_id=r['request_id'],packet_id=p['packet_id'],source_s=p['source_stamp']-ORIGIN,adoption_s=r['adoption_stamp']-ORIGIN,source_age_at_adoption_s=r['gradient_source_age_s'],source_command=p['source_command'],command_at_adoption=r['position'],velocity_at_adoption=r['velocity'],source_to_adoption_command_distance=float(np.linalg.norm(u-p['source_command'])),source_gradient=g.tolist(),source_gradient_norm=float(np.linalg.norm(g)),source_acceleration_from_rest_norm=float(1e-6*np.linalg.norm(g)),task_gradient_norm=float(np.linalg.norm(sf['task_gradient'])),view_gradient_norm=float(np.linalg.norm(sf['weighted_view_gradient'])),local_gradient_norm_at_adoption=float(np.linalg.norm(ng)),H=H.tolist(),H_eigenvalues=lam.tolist(),fixed_model_min_damping_ratio=float(8/(2*np.sqrt(1e-6*lam[-1]))),fastest_natural_frequency_Hz=float(np.sqrt(1e-6*lam[-1])/(2*math.pi)),local_equilibrium=p['local_model_diagnostics']['unconstrained_equilibrium_command'],local_equilibrium_within_joint_limits=p['local_model_diagnostics']['unconstrained_equilibrium_within_joint_positions'],selected_K=p['selected_K'],density_id=p['parameter_density_id'],body_ids=p['body_state_ids'],goal_ids=p['goal_state_ids'],source_observation_stamps=p['observation_source_stamps'],switch=switch))
bounds=[];names=[]
for j in ET.parse(D/'robot.urdf').getroot().findall('joint'):
 if j.get('type')in('revolute','prismatic'):
  l=j.find('limit');bounds.append((float(l.get('lower')),float(l.get('upper'))));names.append(j.get('name'))
bounds=np.asarray(bounds);proposal=np.asarray(failure['proposed_command']);violations=[]
for i,x in enumerate(proposal):
 if x<bounds[i,0]or x>bounds[i,1]:violations.append(dict(joint_index=i,joint_name=names[i],lower=bounds[i,0],upper=bounds[i,1],last_safe_position=failure['last_safe_command'][i],last_safe_velocity=failure['last_safe_velocity'][i],proposed_position=float(x)))
commands=[];Kchanges=[];lastK=None
for r in rows('commands.jsonl'):
 step=r['step'];ex=r['reaching_execution'];K=step['selected_K'];t=step['end_stamp']-ORIGIN
 if K!=lastK:Kchanges.append(dict(first_command_time_s=t,index=step['index'],selected_K=K,density_id=step['density_id'],K_source_s=None if step['source_stamp']is None else step['source_stamp']-ORIGIN));lastK=K
 if ex.get('gradient_packet_id'):
  commands.append(dict(index=step['index'],time_s=t,command=step['end_command'],velocity=r['velocity'],source_age_s=ex['gradient_source_age_s'],source_command_distance=ex['gradient_source_command_distance_l2'],gradient=ex['gradient'],acceleration=ex.get('acceleration'),gradient_packet_id=ex['gradient_packet_id'],position_before=ex['position_before'],velocity_before=ex['velocity_before'],dt=ex['dt']))
localpath=Path('/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/probtf_observation_fusion_demo/src/probtf_observation_fusion_demo/local_quadratic_tracking.py');spec=importlib.util.spec_from_file_location('marker_failure_affine_reference',localpath);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);p=packets[-1];model=mod.FrozenLocalQuadraticModel(p['source_command'],p['source_gradient'],p['H']);u,v=model.advance(failure['position_before'],failure['velocity_before'],failure['dt'],failure['eta'],failure['gamma']);replay=dict(failed_tick_proposal_max_abs_difference=float(np.max(abs(u-proposal))),proposed_velocity=v.tolist(),source_path=str(localpath),source_sha256=sha(localpath),new_inference=0,new_gradient_evaluations=0,new_FK=0,new_equilibrium_solves=0,new_physics=0)
physical=list(rows('physical_trajectory.jsonl'));interventions=[]
for raw in rows('diagnostic_marker_interventions.jsonl'):
 record={k:v for k,v in raw.items()if k!='raw_observations'}
 record['raw_native_payload_omitted_from_summary_only']=True
 record['raw_native_markers']=[dict(child_frame_id=o.get('child_frame_id'),authority=o.get('authority'),component_count=len(o.get('components',[])))for o in raw['raw_observations']]
 reconstructed=np.linalg.solve(np.asarray(raw['true_camera_pose']),np.asarray(raw['true_marker_pose']))
 record['camera_marker_transform_chain_max_abs_error']=float(np.max(abs(reconstructed-np.asarray(raw['exact_camera_relative_pose']))))
 interventions.append(record)
Kjobs=[]
for rr in rows('stiffness.jsonl'):
 sr=rr['stiffness_record'];post=sr.get('posterior')or{};Kjobs.append(dict(job_sequence=rr['job_sequence'],source_s=rr['source_stamp']-ORIGIN,packet_id=rr.get('packet_id'),density_id=post.get('density_id'),status=sr.get('status'),observation_ids=sr.get('observation_ids'),factor_count=len(post.get('factor_ids',[]))))
sourcechecks=[]
for name,want in setup['source_sha256'].items():sourcechecks.append(dict(path=name,expected=want,actual=sha(name),equal=sha(name)==want))
helpers=[]
for name,want in setup.get('diagnostic_sha256',{}).items():
 path=D/'diagnostic_source'/Path(name).name
 if path.exists():helpers.append(dict(path=str(path),expected=want,actual=sha(path),equal=sha(path)==want))
files=['diagnostic_setup.json','diagnostic_summary.json','diagnostic_failure.txt','running_gradient_fail_safe.jsonl','gradient_adoptions.jsonl','commands.jsonl','diagnostic_marker_interventions.jsonl','physical_trajectory.jsonl','stiffness_ready_adoptions.jsonl','robot.urdf'];output=dict(schema='imu_oracle_marker_saved_affine_joint_limit_failure_v1',source_directory=str(D),read_only=True,summary=json.load((D/'diagnostic_summary.json').open()),failure=failure,violations=violations,successful_gradient_packets=packets,all_successful_packet_K_equal=len({tuple(p['selected_K'])for p in packets})==1,K_change_history=Kchanges,K_job_context=Kjobs,K_adoption_history=list(rows('stiffness_ready_adoptions.jsonl')),selected_commands=commands,peak_velocity=max(commands,key=lambda v:np.linalg.norm(v['velocity'])),peak_local_gradient=max(commands,key=lambda v:np.linalg.norm(v['gradient'])),first_physical=physical[0],last_physical=physical[-1],marker_interventions=interventions,affine_replay=replay,source_checks_count=len(sourcechecks),source_changed=[v for v in sourcechecks if not v['equal']],frozen_helper_checks=helpers,input_files=[dict(path=str(D/name),bytes=(D/name).stat().st_size,sha256=sha(D/name))for name in files],last_complete_candidate_artifact_source_s=.28581905364990234,last_complete_candidate_artifact_used_for_failed_packet=False,full_posterior_or_J_decomposition_not_replayed=True)
(O/'failure_audit.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(violations=violations,replay=replay,Kchanges=Kchanges,packets=packets,marker_interventions=interventions),indent=2))
