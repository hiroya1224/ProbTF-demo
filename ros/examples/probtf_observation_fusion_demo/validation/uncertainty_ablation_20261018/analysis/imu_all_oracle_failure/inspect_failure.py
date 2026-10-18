"""Bounded read-only audit of saved fail-safe inputs; no inference or dynamics replay."""
from pathlib import Path
import json,hashlib,collections,math
import numpy as np
D=Path('/tmp/probtf_uncertainty_ablation_20261018/live/imu_marker_all_oracle');O=Path(__file__).parent;ORIGIN=1792080000.
def rows(name):
 with(D/name).open()as f:
  for l in f:
   if l.strip():yield json.loads(l)
def cos(a,b):
 den=np.linalg.norm(a)*np.linalg.norm(b);return None if den==0 else float(np.dot(a,b)/den)
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
fail=list(rows('running_gradient_fail_safe.jsonl'))[-1];last=fail['last_safe_command'];laststamp=fail['interval_start_stamp'];bounds=np.array([[-3.1416,3.1416],[-2.3562,2.3562],[-2.61799,2.61799],[-3.1416,3.1416],[-2.61799,2.61799],[-3.1416,3.1416]])
# Extract actual bounds from original URDF, not this explanatory fallback.
import xml.etree.ElementTree as ET
bounds=[];joint_names=[]
for e in ET.parse(D/'robot.urdf').getroot().findall('joint'):
 if e.get('type')in('revolute','prismatic'):
  l=e.find('limit');bounds.append((float(l.get('lower')),float(l.get('upper'))));joint_names.append(e.get('name'))
bounds=np.asarray(bounds)
accepted=[r for r in rows('gradient_adoptions.jsonl')if r.get('accepted')];packets=[]
for index,r in enumerate(accepted):
 p=r['packet'];H=np.asarray(p['curvature']);g=np.asarray(p['gradient']);u=np.asarray(r['position']);v=np.asarray(r['velocity']);lam=np.linalg.eigvalsh(H);sf=p['selection_factors'];lg=np.asarray(r['model_gradient_at_adoption']);prev=None
 if index:
  old=accepted[index-1]['packet'];gp=np.asarray(old['gradient'])+np.asarray(old['curvature'])@(u-np.asarray(old['source_command']));prev=dict(previous_request_id=accepted[index-1]['request_id'],previous_packet_id=old['packet_id'],gradient_before_switch=gp.tolist(),gradient_after_switch=lg.tolist(),gradient_switch_cosine=cos(gp,lg),gradient_jump_norm=float(np.linalg.norm(lg-gp)),acceleration_jump_norm_rad_s2=float(1e-6*np.linalg.norm(lg-gp)),K_changed=old['selected_K']!=p['selected_K'],density_changed=old['parameter_density_id']!=p['parameter_density_id'],H_relative_Frobenius_change=float(np.linalg.norm(H-np.asarray(old['curvature']))/np.linalg.norm(old['curvature'])))
 out=dict(request_id=r['request_id'],packet_id=p['packet_id'],candidate_id=r['candidate_id'],source_s=p['source_stamp']-ORIGIN,adoption_s=r['adoption_stamp']-ORIGIN,gradient_age_at_adoption_s=r['gradient_source_age_s'],source_command=p['source_command'],command_at_adoption=u.tolist(),velocity_at_adoption=v.tolist(),velocity_norm_at_adoption=float(np.linalg.norm(v)),source_gradient=g.tolist(),source_gradient_norm=float(np.linalg.norm(g)),task_gradient_norm=float(np.linalg.norm(sf['task_gradient'])),view_gradient_norm=float(np.linalg.norm(sf['weighted_view_gradient'])),local_gradient_norm_at_adoption=float(np.linalg.norm(lg)),H=H.tolist(),H_eigenvalues=lam.tolist(),minimum_fixed_model_damping_ratio=float(8/(2*np.sqrt(1e-6*lam[-1]))),fastest_natural_frequency_Hz=float(np.sqrt(1e-6*lam[-1])/(2*math.pi)),local_equilibrium=p['local_model_diagnostics']['unconstrained_equilibrium_command'],local_equilibrium_within_joint_bounds=p['local_model_diagnostics']['unconstrained_equilibrium_within_joint_positions'],selected_K=p['selected_K'],density_id=p['parameter_density_id'],body_ids=p['body_state_ids'],goal_ids=p['goal_state_ids'],source_observation_stamps=p['observation_source_stamps'],source_action_packet_id=r['source_action_packet_id'],switch=prev)
 packets.append(out)
selected=[r for r in packets if r['adoption_s']>=90.];ids={r['packet_id']for r in selected};requests={r['request_id']for r in selected};lastcommands=[];intervalstats={pid:dict(intervals=0,max_local_gradient_norm=0.,max_command_speed=0.,max_command_component_speed=0.,max_acceleration_norm=0.,max_source_age=0.,max_source_command_distance=0.)for pid in ids};peak=[];Kchanges=[];previousK=None;withoutK=[]
for r in rows('commands.jsonl'):
 s=r['step'];t=s['end_stamp']-ORIGIN;ex=r['reaching_execution'];pid=ex.get('gradient_packet_id');K=s['selected_K']
 if K is None:
  increment_norm=float(np.linalg.norm(np.asarray(s['end_command'])-s['start_command']))
  if increment_norm>0.:withoutK.append(dict(end_s=t,command_increment_norm=increment_norm))
 if K!=previousK:Kchanges.append(dict(time_s=t,index=s['index'],selected_K=K,density_id=s['density_id'],K_source_s=None if s.get('source_stamp')is None else s['source_stamp']-ORIGIN));previousK=K
 if t>=90:
  v=np.asarray(r['velocity']);g=np.asarray(ex['gradient']);a=np.asarray(ex['acceleration']);small=dict(index=s['index'],start_s=s['start_stamp']-ORIGIN,end_s=t,command=s['end_command'],velocity=r['velocity'],velocity_before=ex['velocity_before'],gradient=g.tolist(),source_gradient=ex['source_gradient'],local_acceleration=a.tolist(),source_age_s=ex['gradient_source_age_s'],source_command_distance=ex['gradient_source_command_distance_l2'],gradient_packet_id=pid,density_id=s['density_id'],selected_K=K,K_source_s=s['source_stamp']-ORIGIN,active_auxiliary_limits=s['provenance']['active_joint_limits']);peak.append(small)
  if t>=98:lastcommands.append(small)
  if pid in intervalstats:
   st=intervalstats[pid];st['intervals']+=1
   for key,value in [('max_local_gradient_norm',np.linalg.norm(g)),('max_command_speed',np.linalg.norm(v)),('max_command_component_speed',np.max(abs(v))),('max_acceleration_norm',np.linalg.norm(a)),('max_source_age',ex['gradient_source_age_s']),('max_source_command_distance',ex['gradient_source_command_distance_l2'])]:st[key]=max(st[key],float(value))
truths=[]
for r in rows('diagnostic_control_truth.jsonl'):
 if r['request_id']in requests:
  truths.append(dict(request_id=r['request_id'],intervention=r['intervention'],source_s=r['stamp']-ORIGIN,source_command=r['command'],physical_joint_position=r['physical_joint_position'],goal_position=r['goal_position'],true_K=r['true_K'],has_all_local_link_truth=all('link_'+str(i)in r['link_poses']for i in range(1,7))))
physical=[];first_reach=None;minrow=None
for r in rows('physical_trajectory.jsonl'):
 if minrow is None or r['physical_error_m']<minrow['physical_error_m']:minrow=r
 if first_reach is None and r['physical_error_m']<=.006:first_reach=dict(time_s=r['time'],error_m=r['physical_error_m'])
 if r['time']>=90:
  physical.append(dict(time_s=r['time'],physical_distance_m=r['physical_error_m'],command=r['command'],actual_joints=r['actual_joints'],physical_joint_velocities=r['joint_velocities'],reached=r['reached']))
setup=json.loads((D/'diagnostic_setup.json').read_text());sourcechecks=[]
for filename,want in setup['source_sha256'].items():
 p=Path(filename);got=digest(p);sourcechecks.append(dict(path=filename,expected=want,actual=got,equal=want==got))
helperchecks=[]
for filename,want in setup.get('diagnostic_sha256',{}).items():
 p=D/'diagnostic_source'/Path(filename).name
 if p.exists():
  got=digest(p);helperchecks.append(dict(path=str(p),expected=want,actual=got,equal=want==got))
originalfiles=['diagnostic_setup.json','diagnostic_failure.txt','running_gradient_fail_safe.jsonl','gradient_adoptions.jsonl','commands.jsonl','diagnostic_control_truth.jsonl','stiffness_ready_adoptions.jsonl','physical_trajectory.jsonl','robot.urdf'];hashes=[dict(path=str(D/f),bytes=(D/f).stat().st_size,sha256=digest(D/f))for f in originalfiles]
changes=[r for r in sourcechecks if not r['equal']];lastpacket=packets[-1];proposal=np.asarray(fail['proposed_command']);violations=[dict(joint_index=i,joint_name=joint_names[i],lower=bounds[i,0],upper=bounds[i,1],last_safe_command=last[i],proposed_command=proposal[i],last_lower_margin=last[i]-bounds[i,0],proposal_lower_violation=max(0.,bounds[i,0]-proposal[i]))for i in range(len(bounds))if proposal[i]<bounds[i,0]or proposal[i]>bounds[i,1]]
# Analytic6D command model only: no candidate/physics/equilibrium replay.
import importlib.util,sys
localpath=Path('/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/probtf_observation_fusion_demo/src/probtf_observation_fusion_demo/local_quadratic_tracking.py')
spec=importlib.util.spec_from_file_location('failure_local_quadratic_reference',localpath);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
model=mod.FrozenLocalQuadraticModel(lastpacket['source_command'],lastpacket['source_gradient'],lastpacket['H'])
pu,pv=model.advance(fail['position_before'],fail['velocity_before'],fail['dt'],fail['eta'],fail['gamma'])
lastadopt=accepted[-1];duration=fail['interval_end_stamp']-lastadopt['adoption_stamp'];continuous_u,continuous_v=model.advance(lastadopt['position'],lastadopt['velocity'],duration,fail['eta'],fail['gamma'])
analytic=dict(model_source=str(localpath),model_source_sha256=digest(localpath),single_fail_tick_proposal_replay_max_abs=float(np.max(abs(pu-proposal))),single_fail_tick_proposed_velocity=pv.tolist(),from_adoption_duration_s=duration,from_adoption_predicted_proposal=continuous_u.tolist(),from_adoption_proposal_error_max_abs=float(np.max(abs(continuous_u-proposal))),from_adoption_small_difference_reason='Absolute stamp subtraction versus actual monotonic per-tick dt accumulation; one-tick replay uses exact stored dt',new_body_queries=0,new_gradient_evaluations=0,new_equilibrium_solves=0,physical_dynamics_replayed=False)
output=dict(first_reach=first_reach,minimum_physical_distance=dict(time_s=minrow['time'],error_m=minrow['physical_error_m']),analytic_command_only_reproduction=analytic,schema='saved_imu_all_oracle_real_joint_limit_failure_audit_v1',source_directory=str(D),read_only=True,no_replay=True,no_new_equilibrium=True,truth_only_diagnostic_case=True,frozen_diagnostic_source_hash_checks=helperchecks,source_hash_checks_count=len(sourcechecks),source_changed=changes,input_hashes=hashes,actual_joint_bounds=bounds.tolist(),failure=fail,violations=violations,selected_packets=selected,packet_interval_statistics=intervalstats,last_second_commands=lastcommands,peak_velocity_interval=max(peak,key=lambda r:np.linalg.norm(r['velocity'])),peak_acceleration_interval=max(peak,key=lambda r:np.linalg.norm(r['local_acceleration'])),K_unavailable_motion=dict(moving_interval_count=len(withoutK),max_command_increment_norm=max((v['command_increment_norm']for v in withoutK),default=0.),records=withoutK),K_change_history=Kchanges,truth_for_selected_requests=truths,last_physical=physical[-1],physical_since90_summary=dict(count=len(physical),minimum_distance_m=min(r['physical_distance_m']for r in physical),maximum_distance_m=max(r['physical_distance_m']for r in physical)),last_packet=lastpacket)
(O/'failure_audit.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(violations=violations,source_changed=changes,Kchange=Kchanges[-6:],lastpacket={k:lastpacket[k]for k in('request_id','source_s','adoption_s','source_gradient_norm','H_eigenvalues','minimum_fixed_model_damping_ratio','fastest_natural_frequency_Hz','local_equilibrium_within_joint_bounds','selected_K','switch')},lastphysical=physical[-1],lastcommands=lastcommands[-2:]),indent=2))
