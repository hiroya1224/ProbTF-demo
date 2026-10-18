"""Diagnostic-only paired uncertainty interventions on originally used immutable laws.
Truth is injected ONLY into this offline counterfactual harness, never production.
"""
from dataclasses import replace
from pathlib import Path
import hashlib,json,sys,re,math,copy
import numpy as np,yaml
from scipy.special import logsumexp
sys.path.insert(0,'/tmp/probtf_1244_objective_analysis_20261018')
from decompose_objective import restore,derivative
from probtf_observation_fusion_demo.async_artifacts import load_stiffness_job
from extract_sources import reverse_lines
from probtf.spherical_law.first_order import BinghamTransform,JointTransformState,bingham_from_rotation_coordinates
from probtf.geometry import rotmat_to_quat
from probtf_observation_fusion_demo.moving_control import control_components,command_state,_relative_mixture_log_parts
from probtf_observation_fusion_demo.joint_batch_fusion import log_evidence_parts_difference
from probtf_observation_fusion_demo.visibility_factor import VisibilityFactor
from probtf_observation_fusion_demo.local_curvature import build_local_curvature
from probtf_observation_fusion_demo.body_native import pose_query
from probtf_switch_press_demo.scene import inverse,transform
from deflecomp_core.robot.pinocchio_robot import RobotArm
BASE=Path('/tmp/probtf_1244f3f_rviz_rerun_20261018');OUT=Path(__file__).parent
CONFIG=Path('/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/probtf_observation_fusion_demo/config')
ORIGIN=1792080000.
TARGETS={'marker':(30.,60.,90.,112.582,158.474,179.),'imu_marker':(30.,60.,90.,123.575,140.,179.)}

def extract():
 out=[]
 for case,targets in TARGETS.items():
  directory=BASE/case;samples={}
  for line in (directory/'commands.jsonl').open():
   row=json.loads(line);t=row['step']['end_stamp']-ORIGIN
   if row.get('reaching_execution',{}).get('gradient_packet_id') is None:continue
   for target in targets:
    if target not in samples or abs(t-target)<abs(samples[target]['time']-target):samples[target]=dict(time=t,command_record=row)
  packets={s['command_record']['reaching_execution']['gradient_packet_id']for s in samples.values()};adopts={}
  for line in(directory/'gradient_adoptions.jsonl').open():
   row=json.loads(line)
   if row.get('accepted')and row['packet']['packet_id']in packets:adopts[row['packet']['packet_id']]=row
  ids={r['request_id']for r in adopts.values()};controls={}
  for line in reverse_lines(directory/'control.jsonl'):
   m=re.search(br'"request_id"\s*:\s*(\d+)',line[:300])
   if m and int(m[1])in ids:
    row=json.loads(line);controls[row['request_id']]=row
   if len(controls)==len(ids):break
  refs={}
  for p in(directory/'stiffness_jobs').glob('*.json'):
   r=json.loads(p.read_text());refs.setdefault(r['source_stamp'],[]).append(r)
  physical={}
  wanted={r['gradient_evaluation']['source_stamp']for r in controls.values()}
  for line in(directory/'physical_trajectory.jsonl').open():
   r=json.loads(line)
   if r['stamp']in wanted:physical[r['stamp']]=r
  for target,s in samples.items():
   a=adopts[s['command_record']['reaching_execution']['gradient_packet_id']];c=controls[a['request_id']]
   cp=c['prediction']['command_path']; candidates=refs[cp['source_stamp']]
   matches=[r for r in candidates if load_stiffness_job(r)['result'].prior.state_id==cp['source_prior_state_id']]
   assert matches, (case,target,cp['source_prior_state_id'])
   s.update(case=case,target_time=target,adoption=a,control=c,matching_job_reference=matches[0])
   stamp=c['gradient_evaluation']['source_stamp'];p=physical[stamp]
   assert p['command']==c['gradient_evaluation']['source_command']
   s['physical_source']=p
   out.append(s)
 return out

class Action:
 def __init__(self,mapping,K=None):
  self.mapping=mapping;self.K=None if K is None else np.asarray(K);self.source=mapping.source_command
  if K is None:self.anchor=mapping.anchor
  else:self.anchor=mapping.identifier._refine_equilibrium(mapping.anchor,self.source,self.K)[0]
  self.cache={tuple(self.source):self.anchor}
 def theta(self,q):
  if self.K is None:return self.mapping._evaluate(q)[0]
  key=tuple(q)
  if key not in self.cache:self.cache[key]=self.mapping.identifier._refine_equilibrium(self.anchor,np.asarray(q),self.K)[0]
  return self.cache[key]
 def push(self,state,q,body_ids,actions):
  return command_state(state,q,body_ids,actions,joint_increment=self.theta(q)-self.anchor)

def changed(state,condition,body_ids,body_oracle,marker_oracle):
 if condition in('baseline','true_K'):return state
 P=np.asarray(state.covariance).copy();ids=set(body_ids);scales=np.ones(P.shape[0]);centers={}
 if condition.startswith('body_width_'):
  scale=float(condition.split('_')[-1])
  for i,e in enumerate(state.transforms):
   if e.edge_id in ids:scales[6*i:6*i+6]=scale
 elif condition.startswith('goal_width_'):
  scale=float(condition.split('_')[-1])
  for i,e in enumerate(state.transforms):
   if e.edge_id=='marker':scales[6*i:6*i+6]=scale
 elif condition in('oracle_body_centers','oracle_body_translation','oracle_body_rotation'):
  for e in state.transforms:
   if e.edge_id in body_oracle:
    tr,rr=body_oracle[e.edge_id];centers[e.edge_id]=(e.translation if condition=='oracle_body_rotation' else tr,e.rotation_wxyz if condition=='oracle_body_translation' else rr)
 elif condition in('oracle_goal_center','oracle_goal_translation','oracle_goal_rotation'):
  e=next(e for e in state.transforms if e.edge_id=='marker');tr,rr=marker_oracle
  centers={'marker':(e.translation if condition=='oracle_goal_rotation' else tr,e.rotation_wxyz if condition=='oracle_goal_translation' else rr)}
 elif condition in('oracle_all_centers','oracle_all_centers_narrow_trueK'):
  centers=dict(body_oracle,marker=marker_oracle)
  if condition=='oracle_all_centers_narrow_trueK':
   for i,e in enumerate(state.transforms):
    if e.edge_id in ids or e.edge_id=='marker':scales[6*i:6*i+6]=.01
 elif condition in('body_translation_width_0.1','body_rotation_width_0.1','goal_translation_width_0.1','goal_rotation_width_0.1'):
  for i,e in enumerate(state.transforms):
   if (condition.startswith('body_')and e.edge_id in ids)or(condition.startswith('goal_')and e.edge_id=='marker'):
    begin=6*i+(3 if '_rotation_'in condition else 0);scales[begin:begin+3]=.1
 else:raise ValueError(condition)
 P=scales[:,None]*P*scales[None,:]
 transforms=[]
 for i,e in enumerate(state.transforms):
  translation,rotation=centers.get(e.edge_id,(e.translation,e.rotation_wxyz));S=P[6*i+3:6*i+6,6*i+3:6*i+6];St=P[6*i:6*i+3,6*i:6*i+3]
  transforms.append(BinghamTransform(e.edge_id,translation,bingham=None if e.bingham is None else bingham_from_rotation_coordinates(rotation,S),translation_covariance=St,rotation_wxyz=rotation if e.bingham is None else None))
 return JointTransformState(tuple(transforms),P,version=state.version,stamp=state.stamp,command=state.command,component_id=state.component_id)

def analyze(s):
 result,mapping,controller=restore(s);_,components=control_components(result);saved=s['control']['gradient_evaluation'];q=np.asarray(saved['source_command']);p=s['physical_source'];theta=np.asarray(p['actual_joints'])
 scene=yaml.safe_load((CONFIG/'switch_press_scene.yaml').read_text());TM=inverse(transform(scene['base_translation']))@transform(scene['marker']['world_translation'],scene['marker']['world_rpy']);goal=(TM@np.r_[scene['marker']['goal_offset'],1])[:3];trueK=scene['plant']['stiffness'];markeroracle=(TM[:3,3],rotmat_to_quat(TM[:3,:3]))
 robot=RobotArm(str(BASE/s['case']/'robot.urdf'),tip_link='tool0',base_link='base_link');robot._fk_update(theta);bodyoracle={};parent='base_link'
 for eid in controller.body_ids:
  Mp=robot.data.oMf[robot.get_frame_id(parent)];Mc=robot.data.oMf[robot.get_frame_id(eid)];M=Mp.inverse()*Mc;bodyoracle[eid]=(np.array(M.translation).copy(),rotmat_to_quat(np.array(M.rotation)));parent=eid
 fid=robot.get_frame_id('rod_tip')
 def fk(th):
  robot._fk_update(th);M=robot.data.oMf[robot.base_fid].inverse()*robot.data.oMf[fid];return np.array(M.translation).copy()
 x=fk(theta);np.testing.assert_allclose(x,p['tip'],atol=1e-12,rtol=0)
 JT=np.column_stack([(fk(theta+np.eye(6)[j]*1e-6)-fk(theta-np.eye(6)[j]*1e-6))/(2e-6)for j in range(6)])
 originalaction=Action(mapping);trueaction=Action(mapping,trueK);h=1e-5
 Dtrue=np.column_stack([(trueaction.theta(q+np.eye(6)[j]*h)-trueaction.theta(q-np.eye(6)[j]*h))/(2*h)for j in range(6)])
 Jphysical=JT@Dtrue;radial=(goal-x)/np.linalg.norm(goal-x)
 visibility=VisibilityFactor(yaml.safe_load((CONFIG/'moving_estimation.yaml').read_text())['control']['visibility_factor'])
 camera_marker=tuple((e,not inv)for e,inv in reversed(result.camera_path))+tuple(result.marker_path)
 conditions=('baseline','oracle_goal_center','goal_width_0.1','goal_width_0.01','oracle_body_centers','body_width_0.1','body_width_0.01','true_K','oracle_goal_translation','oracle_goal_rotation','oracle_body_translation','oracle_body_rotation','body_translation_width_0.1','body_rotation_width_0.1','goal_translation_width_0.1','goal_rotation_width_0.1','oracle_all_centers','oracle_all_centers_narrow_trueK')
 previous=OUT/(s['case']+'_'+str(int(s['target_time']))+'.json')
 output=json.loads(previous.read_text())['conditions'] if previous.exists() else {}
 for condition in conditions:
  if condition in output:continue
  cs=tuple(replace(c,state=changed(c.state,condition,controller.body_ids,bodyoracle,markeroracle))for c in components);action=trueaction if condition in('true_K','oracle_all_centers_narrow_trueK')else originalaction
  cache={}
  def value(command):
   key=tuple(command)
   if key in cache:return cache[key]
   states=[];_,branches,queries=controller._evaluate_command_objective(cs,np.asarray(command),action,trial_states=states);parts=_relative_mixture_log_parts(branches)
   vrecord=visibility.evaluate(cs,tuple(pose_query(st,camera_marker,frame='camera_optical_frame')for st in states));v=vrecord['weighted_penalty']
   weights=np.exp([b['log_weight']for b in branches]);weights/=sum(weights);mu=np.array([b['relative_position_mean']for b in branches]);dist=np.sqrt(weights@np.sum((mu-controller.goal_offset)**2,axis=1))
   cache[key]=(parts,float(v),float(dist),queries,branches,vrecord);return cache[key]
  a=value(q);g=np.zeros(6);gt=np.zeros(6);gv=np.zeros(6);gd=np.zeros(6)
  for j in range(6):
   plus=value(q+np.eye(6)[j]*h);minus=value(q-np.eye(6)[j]*h)
   gt[j]=-(log_evidence_parts_difference(plus[0],minus[0]))/(2*h);gv[j]=(plus[1]-minus[1])/(2*h);gd[j]=(plus[2]-minus[2])/(2*h)
  g=gt+gv;norm=np.linalg.norm(g);d=-g/norm;vel=Jphysical@d;inward=float(radial@vel);speed=float(np.linalg.norm(vel));tangent=float(np.linalg.norm(vel-inward*radial))
  coordinates=[]
  for j in range(6):
   coordinates.append(dict(joint_index=j,forward=dict(status='ok',step=float((q+np.eye(6)[j]*h)[j]-q[j])),backward=dict(status='ok',step=float((q-np.eye(6)[j]*h)[j]-q[j]))))
  task=dict(saved,selected_gradient=g.tolist(),finite_difference_diagnostics=dict(coordinates=coordinates))
  probes={key:dict(branches=v[4])for key,v in cache.items()};views={key:v[5]for key,v in cache.items()}
  settings=yaml.safe_load((CONFIG/'moving_estimation.yaml').read_text())['control']['local_quadratic_model']
  H,hd=build_local_curvature(task,probes,views,visibility,settings,lower=controller.lower,upper=controller.upper)
  localdelta=-np.linalg.solve(H,g);localunit=localdelta/np.linalg.norm(localdelta);localvel=Jphysical@localunit;localinward=float(radial@localvel);localspeed=float(np.linalg.norm(localvel))
  # A unit command norm makes direction comparisons independent of changed PDF scale.
  output[condition]=dict(selected_gradient=g.tolist(),task_gradient=gt.tolist(),view_gradient=gv.tolist(),gradient_norm=float(norm),negative_unit_gradient=d.tolist(),source_J=-sum(a[0]),source_weighted_view=a[1],source_estimated_mean_distance_m=a[2],estimated_distance_slope_m_per_unit_command=float(gd@d),oracle_quasistatic_physical_inward_m_per_unit_command=inward,oracle_quasistatic_physical_tangent_m_per_unit_command=tangent,oracle_quasistatic_physical_speed_m_per_unit_command=speed,inward_fraction_of_tip_motion=inward/speed,source_relative_position_covariance_eigenvalues=[np.linalg.eigvalsh(np.asarray(b['relative_position_covariance'])).tolist()for b in a[4]],full_joint_covariance_psd_min_eigenvalues=[float(np.linalg.eigvalsh(c.state.covariance)[0])for c in cs],new_sensor_updates=0,new_K_density_evaluations=0,local_curvature=H.tolist(),local_equilibrium_delta=localdelta.tolist(),local_equilibrium_within_joint_positions=hd['unconstrained_equilibrium_within_joint_positions'],local_equilibrium_unit_direction=localunit.tolist(),local_equilibrium_direction_inward_fraction=localinward/localspeed,local_equilibrium_direction_inward_m_per_unit_command=localinward,local_equilibrium_estimated_distance_slope_m_per_unit_command=float(gd@localunit))
 if s['control']['gradient_evaluation']['finite_difference_diagnostics']['coordinates']:
  diff=float(np.max(abs(np.asarray(saved['selected_gradient'])-output['baseline']['selected_gradient'])))
 else:diff=None
 answer=dict(case=s['case'],target_time=s['target_time'],sample_time_s=s['time'],source_time_s=saved['source_stamp']-ORIGIN,request_id=s['control']['request_id'],candidate_id=s['control']['candidate_id'],source_command=q.tolist(),actual_joints=theta.tolist(),trueK=trueK,estimatedK=saved['selected_K'],physical_source_stamp_exact=True,physical_source_command_exact=True,baseline_selected_gradient_replay_max_abs=diff,baseline_curvature_replay_max_abs=float(np.max(abs(np.asarray(saved['curvature'])-output['baseline']['local_curvature']))),original_candidate_identity_exact=True,original_ledger_chain_exact=True,conditions=output,physical_distance_at_source_m=float(np.linalg.norm(goal-x)))
 (OUT/(s['case']+'_'+str(int(s['target_time']))+'.json')).write_text(json.dumps(answer,indent=2,allow_nan=False)+'\n')
 print(s['case'],s['target_time'],'request',answer['request_id'],'replay',diff,'inwardfractions',{k:round(v['inward_fraction_of_tip_motion'],3)for k,v in output.items()},flush=True)
 return answer

def main():
 sources=extract();records=[]
 for s in sources:
  try:
   existing=OUT/(s['case']+'_'+str(int(s['target_time']))+'.json')
   records.append(json.loads(existing.read_text()) if existing.exists() and 'oracle_all_centers_narrow_trueK' in json.loads(existing.read_text())['conditions'] else analyze(s))
  except Exception as error:
   import traceback;traceback.print_exc();raise
 summary=dict(schema='diagnostic_single_factor_frozen_oracle_ablation_v1',source_run=str(BASE),source_head='efeb1d2',source_numeric_commit='1244f3f',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),production_changed=False,oracle_only_diagnostic=True,combined_ceiling_conditions=['oracle_all_centers','oracle_all_centers_narrow_trueK'],closed_loop_run=False,monte_carlo=False,conditions=dict(baseline='Original used body-goal joint law and original K action',oracle_goal_center='Replace root marker modal translation/rotation by physical truth; retain all original covariance/cross numbers in the new local chart',oracle_body_centers='Replace six body local modal centers by actual local transforms at exact source physical frame; retain all original covariance/cross numbers in new local charts',goal_width_01='Multiply goal standard coordinates by0.1 or0.01 via Pnew=D P D^T, including all cross; centers retained',body_width_01='Multiply six body standard coordinates by0.1 or0.01 via Pnew=D P D^T, including all cross; centers retained',true_K='Use simulator true K for the anchored auxiliary equilibrium action only; retain original body and goal law'),physical_direction_scope='FK Jacobian at recorded actual theta times true-K quasistatic equilibrium command Jacobian; dynamic lag/velocity are not simulated. This is a common frozen diagnostic transfer for all conditions, not actual next-tick motion or closed-loop result.',mean_oracle_scope='A counterfactual recentering with identical local-coordinate covariance/cross; not an exact Bayesian conditioning or push-forward of the original distribution.',records=records)
 for case in TARGETS:
  subset=[r for r in records if r['case']==case];agg={}
  for name in subset[0]['conditions']:
   fractions=np.array([r['conditions'][name]['inward_fraction_of_tip_motion']for r in subset]);slopes=np.array([r['conditions'][name]['oracle_quasistatic_physical_inward_m_per_unit_command']for r in subset]);base=np.array([r['conditions']['baseline']['inward_fraction_of_tip_motion']for r in subset]);agg[name]=dict(mean_inward_fraction=float(fractions.mean()),median_inward_fraction=float(np.median(fractions)),outward_count=int(sum(fractions<0)),mean_paired_fraction_change=float(np.mean(fractions-base)),improvement_count=int(sum(fractions>base+1e-9)),mean_inward_m_per_unit_command=float(slopes.mean()),sample_count=len(subset),mean_local_equilibrium_direction_inward_fraction=float(np.mean([r['conditions'][name]['local_equilibrium_direction_inward_fraction']for r in subset])),mean_local_equilibrium_paired_fraction_change=float(np.mean([r['conditions'][name]['local_equilibrium_direction_inward_fraction']-r['conditions']['baseline']['local_equilibrium_direction_inward_fraction']for r in subset])))
  summary.setdefault('aggregate',{})[case]=agg
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 print(json.dumps(summary['aggregate'],indent=2),flush=True)
if __name__=='__main__':main()
