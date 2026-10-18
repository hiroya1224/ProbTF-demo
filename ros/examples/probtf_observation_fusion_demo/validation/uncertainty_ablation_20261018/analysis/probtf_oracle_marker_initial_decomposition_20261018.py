import json,pathlib,gzip,math,yaml
import numpy as np
from scipy.special import logsumexp
from probtf_observation_fusion_demo.moving_control import MovingController,_relative_mixture_log_parts,control_components
from probtf_observation_fusion_demo.joint_batch_fusion import log_evidence_parts_difference
from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier
ns={};source=pathlib.Path('/tmp/probtf_oracle_marker_early_20261018.py').read_text().split('out=dict(diagnostic_only')[0];exec(source,ns)
ROOT=ns['ROOT'];START=ns['START'];rows=ns['rows'];restore=ns['restore'];EquilibriumBody=ns['EquilibriumBody'];MovingInference=ns['MovingInference']
OUT={};TERMS=('position_quadratic','position_logdet','position_normalizer','negative_bingham_quadratic','positive_bingham_logZ')
for name in ['marker_baseline','marker_oracle_marker']:
 p=ROOT/'live'/name
 for row in rows(p/'control.jsonl'):
  saved=row['controller']
  if saved['status']=='ok':break
 result=restore(p,row);_,components=control_components(result)
 inference=MovingInference((p/'robot.urdf').read_text(),());motion=inference.settings['control'];ctl=MovingController(inference,(p/'robot.urdf').read_text(),motion['goal_offset'],motion['goal_orientation_wxyz'],motion,integration_steps=motion['integration_steps'])
 q=np.asarray(saved['source_command']);assert saved['selected_K']is None
 def evaluate(command):
  _,branches,queries=ctl._evaluate_command_objective(components,command,None);parts=_relative_mixture_log_parts(branches)
  weights=np.exp(np.array([b['log_weight']for b in branches])-logsumexp([b['log_weight']for b in branches]));terms=[];means=[];covs=[];vs=[]
  for b in branches:
   S=np.asarray(b['relative_position_covariance']);e=np.asarray(b['relative_position_mean'])-ctl.goal_offset;inv=np.linalg.inv(S)
   terms.append([.5*float(e@inv@e),.5*np.linalg.slogdet(S)[1],1.5*math.log(2*math.pi),-float(ctl.goal_orientation@np.asarray(b['relative_orientation_bingham'])@ctl.goal_orientation),b['log_bingham_normalizer']]);means.append(b['relative_position_mean']);covs.append(S);vs.append(inv@e)
  return dict(parts=np.asarray(parts),terms=np.asarray(terms),means=np.asarray(means),covs=np.asarray(covs),vs=np.asarray(vs),weights=weights,distance=math.sqrt(weights@np.sum((np.asarray(means)-ctl.goal_offset)**2,axis=1)))
 base=evaluate(q);g=np.zeros(6);termg=np.zeros((5,6));gmean=np.zeros(6);gcov=np.zeros(6);dm=np.zeros((3,6));gd=np.zeros(6)
 for i in range(6):
  eps=np.zeros(6);eps[i]=1e-5;plus=evaluate(q+eps);minus=evaluate(q-eps)
  g[i]=-(log_evidence_parts_difference(plus['parts'],base['parts'])-log_evidence_parts_difference(minus['parts'],base['parts']))/(2e-5)
  termg[:,i]=base['weights']@((plus['terms']-minus['terms'])/(2e-5));dmu=(plus['means']-minus['means'])/(2e-5);dS=(plus['covs']-minus['covs'])/(2e-5)
  dm[:,i]=base['weights']@dmu;gmean[i]=base['weights']@np.einsum('ki,ki->k',base['vs'],dmu);gcov[i]=-.5*float(base['weights']@np.einsum('ki,kij,kj->k',base['vs'],dS,base['vs']));gd[i]=(plus['distance']-minus['distance'])/(2e-5)
 exact=np.asarray(saved['task_gradient']);difference=float(np.max(np.abs(g-exact)));H=np.asarray(saved['curvature']);selected=np.asarray(saved['gradient'])
 assert np.max(np.abs(g-exact))/np.max(np.abs(exact))<1e-8
 frame=None
 for i,r in enumerate(rows(p/'physical_trajectory.jsonl')):
  if i==row['index']:frame=r;break
 assert frame['stamp']==saved['source_stamp']and frame['command']==saved['source_command']
 K=np.array([350.,125.,95.,60.,40.,30.]);settings=yaml.safe_load(pathlib.Path('/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/probtf_observation_fusion_demo/config/stiffness_identification.yaml').read_text());identifier=StiffnessIdentifier(p/'robot.urdf',inference.body_ids,settings)
 theta=np.asarray(frame['actual_joints']);hessian,_=identifier.sensitivity.equilibrium_jacobians(theta,q,K);D=np.linalg.solve(hessian,np.diag(K));robot=identifier.robot
 tipfid=robot.get_frame_id('rod_tip');robot._fk_update(theta);invbase=np.linalg.inv(np.asarray(robot.data.oMf[robot.base_fid].homogeneous))
 def tip(at):robot._fk_update(at);return(invbase@np.asarray(robot.data.oMf[tipfid].homogeneous))[:3,3].copy()
 J=np.zeros((3,6))
 for i in range(6):
  eps=np.zeros(6);eps[i]=1e-6;J[:,i]=(tip(theta+eps)-tip(theta-eps))/(2e-6)
 truegoal=np.asarray(next(rows(ROOT/'live/marker_oracle_goal/diagnostic_control_truth.jsonl'))['goal_position']);error=np.asarray(frame['tip'])-truegoal
 directions={}
 for label,direction in [('negative_gradient',-selected),('local_equilibrium_displacement',-np.linalg.solve(H,selected))]:
  unit=direction/np.linalg.norm(direction);direct=J@unit;eq=J@D@unit
  directions[label]=dict(direction=direction.tolist(),direction_norm=float(np.linalg.norm(direction)),unit_direction=unit.tolist(),estimated_mean_distance_slope_m_per_rad=float(gd@unit),complete_task_J_slope_per_rad=float(g@unit),term_J_slopes={n:float(v@unit)for n,v in zip(TERMS,termg)},position_mean_term_J_slope=float(gmean@unit),position_covariance_term_J_slope=float(gcov@unit),true_body_direct_FK=dict(position_velocity=direct.tolist(),distance_slope_m_per_rad=float(error@direct/np.linalg.norm(error)),inward_cosine=float(-error@direct/np.linalg.norm(error)/np.linalg.norm(direct))),true_K_quasistatic_FK=dict(position_velocity=eq.tolist(),distance_slope_m_per_rad=float(error@eq/np.linalg.norm(error)),inward_cosine=float(-error@eq/np.linalg.norm(error)/np.linalg.norm(eq))))
 OUT[name]=dict(request_id=row['request_id'],source_t=saved['source_stamp']-START,gradient_replay_inf_difference=difference,selected_gradient_norm=float(np.linalg.norm(selected)),initial_acceleration_norm_rad_s2=1e-6*float(np.linalg.norm(selected)),minimum_modal_damping_ratio=float(8/(2*np.sqrt(1e-6*np.linalg.eigvalsh(H)[-1]))),source_terms={n:float(base['weights']@base['terms'][:,i])for i,n in enumerate(TERMS)},gradient_parts={n:dict(norm=float(np.linalg.norm(v)),vector=v.tolist())for n,v in zip(TERMS,termg)},position_mean_gradient=dict(norm=float(np.linalg.norm(gmean)),vector=gmean.tolist()),position_covariance_gradient=dict(norm=float(np.linalg.norm(gcov)),vector=gcov.tolist()),position_quadratic_chain_split_residual_inf=float(np.max(np.abs(gmean+gcov-termg[0]))),source_mean_distance_m=base['distance'],directions=directions,true_K_command_to_equilibrium_jacobian=D.tolist(),true_K_equilibrium_residual_norm=float(np.linalg.norm(identifier.robot.tau_gravity(theta)+K*(theta-q))),notes=['No K packet existed at this selected source in either run; original control used its declared direct command action.','True K/FK directional probes are posthoc only, not a measured instantaneous dynamic response.','The full original complete-J gradient was reconstructed using 13 deterministic command queries with the same immutable body/goal candidate.'])
 print(name,json.dumps({k:OUT[name][k]for k in ['source_t','gradient_replay_inf_difference','position_mean_gradient','position_covariance_gradient','directions']},indent=2),flush=True)
path=ROOT/'analysis/oracle_marker_early.json';early=json.loads(path.read_text());early['initial_complete_J_directional_decomposition']=OUT;path.write_text(json.dumps(early,indent=2,allow_nan=False)+'\n')
