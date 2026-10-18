"""Same-full-J directional chain decomposition with all truth centers (offline only)."""
import json,math,hashlib
from pathlib import Path
import numpy as np,yaml
from scipy.special import logsumexp
from run_frozen import extract,restore,changed,Action,BASE,OUT,CONFIG
from probtf.geometry import rotmat_to_quat
from probtf.spherical_law.support import covariance_support
from probtf_observation_fusion_demo.moving_control import control_components,_relative_mixture_log_parts
from probtf_observation_fusion_demo.joint_batch_fusion import log_evidence_parts_difference
from probtf_observation_fusion_demo.visibility_factor import VisibilityFactor
from probtf_observation_fusion_demo.body_native import pose_query,point_query
from probtf_switch_press_demo.scene import transform,inverse
from deflecomp_core.robot.pinocchio_robot import RobotArm
from dataclasses import replace
scene=yaml.safe_load((CONFIG/'switch_press_scene.yaml').read_text());TM=inverse(transform(scene['base_translation']))@transform(scene['marker']['world_translation'],scene['marker']['world_rpy']);goal=(TM@np.r_[scene['marker']['goal_offset'],1])[:3];trueK=scene['plant']['stiffness'];markeroracle=(TM[:3,3],rotmat_to_quat(TM[:3,:3]));view=VisibilityFactor(yaml.safe_load((CONFIG/'moving_estimation.yaml').read_text())['control']['visibility_factor']);records=[]
for source in extract():
 if source['case']!='marker' or int(source['target_time'])not in(90,158):continue
 result,mapping,controller=restore(source);_,components=control_components(result);q=np.asarray(source['control']['gradient_evaluation']['source_command']);phys=source['physical_source'];theta=np.asarray(phys['actual_joints']);robot=RobotArm(str(BASE/'marker/robot.urdf'),tip_link='tool0',base_link='base_link');robot._fk_update(theta);bodyoracle={};parent='base_link'
 for eid in controller.body_ids:
  Mp=robot.data.oMf[robot.get_frame_id(parent)];Mc=robot.data.oMf[robot.get_frame_id(eid)];M=Mp.inverse()*Mc;bodyoracle[eid]=(np.array(M.translation).copy(),rotmat_to_quat(np.array(M.rotation)));parent=eid
 fid=robot.get_frame_id('rod_tip')
 def fk(th):
  robot._fk_update(th);M=robot.data.oMf[robot.base_fid].inverse()*robot.data.oMf[fid];return np.array(M.translation).copy(),rotmat_to_quat(np.array(M.rotation))
 physx,physr=fk(theta);np.testing.assert_allclose(physx,phys['tip'],atol=1e-12,rtol=0)
 previous=json.loads((OUT/('marker_'+str(int(source['target_time']))+'.json')).read_text())
 for condition in('oracle_all_centers','oracle_all_centers_narrow_trueK'):
  cs=tuple(replace(c,state=changed(c.state,condition,controller.body_ids,bodyoracle,markeroracle))for c in components);action=Action(mapping,trueK)if condition.endswith('trueK')else Action(mapping);row=previous['conditions'][condition];delta=np.asarray(row['local_equilibrium_delta']);direction=delta/np.linalg.norm(delta);h=1e-5
  camera_marker=tuple((e,not inv)for e,inv in reversed(result.camera_path))+tuple(result.marker_path)
  def evaluate(command):
   states=[];_,branches,queries=controller._evaluate_command_objective(cs,np.asarray(command),action,trial_states=states);parts=_relative_mixture_log_parts(branches);responsibility=np.exp([b['log_weight']+log_evidence_parts_difference(b['log_density_parts'],parts)for b in branches]);responsibility/=responsibility.sum();v=view.evaluate(cs,tuple(pose_query(st,camera_marker,frame='camera_optical_frame')for st in states))['weighted_penalty'];terms=[];mu=[];S=[];vprecision=[]
   for b in branches:
    cov=np.asarray(b['relative_position_covariance']);support=covariance_support(cov);e=np.asarray(b['relative_position_mean'])-controller.goal_offset;pr=support.inverse@e
    terms.append([.5*float(e@pr),.5*support.log_pdet,.5*support.rank*math.log(2*math.pi),-float(controller.goal_orientation@np.asarray(b['relative_orientation_bingham'])@controller.goal_orientation),b['log_bingham_normalizer']]);mu.append(b['relative_position_mean']);S.append(cov);vprecision.append(pr)
   weights=np.exp([b['log_weight']for b in branches]);weights/=weights.sum();distance=math.sqrt(weights@np.sum((np.asarray(mu)-controller.goal_offset)**2,axis=1));physicalx,_=fk(theta+action.theta(command)-action.anchor);physicaldistance=float(np.linalg.norm(physicalx-goal))
   return dict(parts=parts,terms=np.asarray(terms),responsibility=responsibility,mu=np.asarray(mu),S=np.asarray(S),precision_error=np.asarray(vprecision),view=v,distance=distance,physical_distance=physicaldistance)
  baseline=evaluate(q);plus=evaluate(q+h*direction);minus=evaluate(q-h*direction);dmu=(plus['mu']-minus['mu'])/(2*h);dS=(plus['S']-minus['S'])/(2*h);v=baseline['precision_error'];w=baseline['responsibility'];mean=float(w@np.einsum('ki,ki->k',v,dmu));cov=-.5*float(w@np.einsum('ki,kij,kj->k',v,dS,v));terms=w@((plus['terms']-minus['terms'])/(2*h));dv=(plus['view']-minus['view'])/(2*h);full=-(log_evidence_parts_difference(plus['parts'],minus['parts']))/(2*h)+dv
  original=cs[0].state;hand=pose_query(original,result.hand_path,frame='base_link');goalquery=point_query(original,result.marker_path,controller.goal_offset,query_id='oracle_geometry',frame='base_link');marker=pose_query(original,result.marker_path,frame='base_link');qm=marker.rotation_wxyz
  geometry=dict(core_hand_mode=hand.mode[:3].tolist(),actual_FK_hand=physx.tolist(),hand_mode_truth_max_abs_m=float(np.max(abs(hand.mode[:3]-physx))),core_goal_mode=goalquery.mode.tolist(),actual_goal=goal.tolist(),goal_mode_truth_max_abs_m=float(np.max(abs(goalquery.mode-goal))),hand_mode_orientation_abs_dot=float(abs(hand.rotation_wxyz@physr)),marker_mode_orientation_abs_dot=float(abs(qm@markeroracle[1])),core_hand_corrected_mean=hand.mean[:3].tolist(),core_goal_corrected_mean=goalquery.mean.tolist(),hand_correction_m=(hand.mean[:3]-hand.mode[:3]).tolist(),goal_correction_m=(goalquery.mean-goalquery.mode).tolist())
  r=dict(source_time_s=source['control']['gradient_evaluation']['source_stamp']-1792080000.,request_id=source['control']['request_id'],condition=condition,local_equilibrium_direction=direction.tolist(),local_equilibrium_delta_norm=float(np.linalg.norm(delta)),unit_direction_is_same_as_frozen_sweep=True,geometry_truth_checks=geometry,full_selected_objective_directional_derivative=full,source_gradient_dot_unit_direction=float(np.asarray(row['selected_gradient'])@direction),estimated_marker_relative_mean_distance_derivative_m_per_unit_command=(plus['distance']-minus['distance'])/(2*h),oracle_anchored_FK_true_goal_distance_derivative_m_per_unit_command=(plus['physical_distance']-minus['physical_distance'])/(2*h),source_physical_distance_m=baseline['physical_distance'],source_estimated_relative_mean_distance_m=baseline['distance'],derivative_terms_per_unit_command=dict(position_mean_in_quadratic=mean,position_covariance_in_quadratic=cov,position_quadratic_total=float(terms[0]),position_logdet=float(terms[1]),position_normalizer=float(terms[2]),bingham_negative_quadratic=float(terms[3]),bingham_logZ=float(terms[4]),view=float(dv)),position_chain_split_error=float(mean+cov-terms[0]),term_sum_error=float(np.sum(terms)+dv-full),finite_directional_step=h,new_observations=0,physical_truth_only_offline=True)
  records.append(r);print(r['request_id'],condition,'truth geometry',geometry['hand_mode_truth_max_abs_m'],geometry['goal_mode_truth_max_abs_m'],'dist slope',r['oracle_anchored_FK_true_goal_distance_derivative_m_per_unit_command'],'J slope',full,'terms',r['derivative_terms_per_unit_command'],flush=True)
(OUT/'oracle_outward_decomposition.json').write_text(json.dumps(dict(schema='all_truth_full_objective_outward_direction_decomposition_v1',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=records),indent=2,allow_nan=False)+'\n')
