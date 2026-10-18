"""Posthoc frozen K-increment vs actual motion; no new equilibrium solve.

Oracle anchoring is diagnostic only: it applies the recorded K-model joint
increments to the saved actual starting configuration. It is NOT the
controller's own candidate prediction, and includes no simulated true K.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from deflecomp_core.robot.pinocchio_robot import RobotArm
from probtf_1244_trajectory_efficiency_20261018 import rows,stats
BASE=Path('/tmp/probtf_1244f3f_rviz_rerun_20261018');OUT=Path('/tmp/probtf_1244_trajectory_efficiency_20261018')
report=json.loads((OUT/'summary.json').read_text());goal=np.asarray(report['goal_in_root']);output=dict(schema='captured_K_increment_actual_alignment_v1',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_run=str(BASE),truth_only_posthoc=True,source_edited=False,new_equilibrium_solve=False,controller_prediction_reconstructed=False,meaning='Frozen auxiliary equilibrium increments are compared at an oracle actual initial configuration, to distinguish action sensitivity/dynamic mismatch from an erroneous absolute body anchor. This is not a claim that the controller used physical truth.',cases={})
for case,c in report['cases'].items():
 z=np.load(OUT/(case+'_motion.npz'));t=z['time'];x=z['tip'];theta=z['theta'];q=z['command'];delta=z['joint_increment'];dt=np.diff(t);r=(goal-x[:-1])/np.linalg.norm(goal-x[:-1],axis=1)[:,None]
 robot=RobotArm(str(BASE/case/'robot.urdf'),tip_link='tool0',base_link='base_link');fid=robot.get_frame_id('rod_tip')
 def fk(joints):
  robot._fk_update(np.asarray(joints));M=robot.data.oMf[robot.base_fid].inverse()*robot.data.oMf[fid];return np.asarray(M.translation).copy()
 pred=np.array([fk(th+d)-xx for th,d,xx in zip(theta[:-1],delta,x[:-1])]);actual=np.diff(x,axis=0);predrad=np.sum(pred*r,axis=1);actrad=np.sum(actual*r,axis=1);predtan=np.linalg.norm(pred-predrad[:,None]*r,axis=1);acttan=np.linalg.norm(actual-actrad[:,None]*r,axis=1);pn=np.linalg.norm(pred,axis=1);an=np.linalg.norm(actual,axis=1);cos=np.divide(np.sum(pred*actual,axis=1),pn*an,out=np.full(len(dt),np.nan),where=pn*an>0)
 sources=list(rows(OUT/(case+'_interval_sources.jsonl')));equilibria=[];differ=[];qsourceerr=[];sourcecheck=[];Jac=[]
 for i,row in enumerate(rows(BASE/case/'commands.jsonl')):
  s=row['step'];p=s.get('provenance')or{};eq=p.get('source_equilibrium');eq1=p.get('query_equilibrium')
  equilibria.append(eq)
  if eq is not None and eq1 is not None:
   differ.append(float(np.max(abs(np.asarray(eq1)-np.asarray(eq)-delta[i]))));qsourceerr.append(p['source_command']==q[i].tolist()and p['query_command']==q[i+1].tolist()and p['density_id']==s['density_id']and p['selected_K']==s['selected_K']);sourcecheck.append(i)
  Jac.append(p.get('command_jacobian'))
 windows=[]
 for lo,hi in((0,60),(60,120),(120,180),(100,180),(0,180)):
  w=np.maximum(0,np.minimum(t[1:],hi)-np.maximum(t[:-1],lo));k=w>0;valid=k&np.isfinite(cos);frac=w[k]/dt[k]
  windows.append(dict(start_s=lo,end_s=hi,oracle_anchored_captured_K_delta_tip_arc_m=float(pn[k]@frac),actual_tip_arc_m=float(an[k]@frac),predicted_mean_signed_inward_velocity_m_s=float((predrad[k]/dt[k])@w[k]/w.sum()),actual_mean_signed_inward_velocity_m_s=float((actrad[k]/dt[k])@w[k]/w.sum()),predicted_mean_tangent_velocity_m_s=float((predtan[k]/dt[k])@w[k]/w.sum()),actual_mean_tangent_velocity_m_s=float((acttan[k]/dt[k])@w[k]/w.sum()),opposite_predicted_actual_radial_sign_time_fraction=float(w[k][predrad[k]*actrad[k]<0].sum()/w.sum()),direction_cosine_timeweighted_mean=float(cos[valid]@w[valid]/w[valid].sum()),direction_cosine_actual_arc_weighted_mean=float(cos[valid]@(an[valid]*w[valid]/dt[valid])/(an[valid]*w[valid]/dt[valid]).sum()),delta_velocity_model_error_m_s=stats(np.linalg.norm(pred[k]-actual[k],axis=1)/dt[k],w[k]),one_interval_limit='quasistatic prediction vs dynamic physical response; sign mismatch alone cannot be attributed solely to wrong K'))
 selected=[]
 # .5 sec windows anchored at exact physical frame preceding each chosen tick;
 # plus representative smooth late intervals around150/175sec.
 chosen={v['index']for v in c['selected_intervals'].values()}
 chosen.update(int(np.searchsorted(t,s))for s in(100.,130.,150.,175.))
 for start in sorted(chosen):
  end=min(len(t)-1,int(np.searchsorted(t,t[start]+.5)));inc=delta[start:end].sum(axis=0);oracle_end=fk(theta[start]+inc);actual_end=x[end];begin=x[start];rad=(goal-begin)/np.linalg.norm(goal-begin);pa=oracle_end-begin;da=actual_end-begin;eq=equilibria[start];aux=None
  if eq is not None:aux=fk(np.asarray(eq)+inc)-fk(eq)
  sources_window=sources[start:end];packets=list(dict.fromkeys(s['gradient_packet_id']for s in sources_window));requests=list(dict.fromkeys(s['request_id']for s in sources_window));densities=list(dict.fromkeys(s['density_id']for s in sources_window));Ks=[]
  for s in sources_window:
   if s['selected_K']not in Ks:Ks.append(s['selected_K'])
  dn=float(np.linalg.norm(da));pnorm=float(np.linalg.norm(pa));acos=None if dn*pnorm==0 else float(da@pa/(dn*pnorm))
  selected.append(dict(start_index=start,end_index_exclusive=end,start_s=float(t[start]),end_s=float(t[end]),duration_s=float(t[end]-t[start]),first_request_id=sources[start]['request_id'],request_ids=requests,gradient_packet_ids=packets,density_ids=densities,selected_K_values=Ks,K_estimated_source_stamps=list(dict.fromkeys(s['K_source_stamp']for s in sources_window)),exact_original_interval_provenance_verified=all(qsourceerr[sourcecheck.index(i)]for i in range(start,end)if i in sourcecheck),
    actual_delta_tip_m=da.tolist(),oracle_anchored_K_delta_tip_m=pa.tolist(),auxiliary_FK_own_anchor_delta_tip_m=None if aux is None else aux.tolist(),actual_signed_start_radial_delta_m=float(rad@da),predicted_signed_start_radial_delta_m=float(rad@pa),auxiliary_own_anchor_signed_radial_delta_m=None if aux is None else float(rad@aux),actual_exact_distance_shortening_m=float(np.linalg.norm(goal-begin)-np.linalg.norm(goal-actual_end)),predicted_exact_distance_shortening_m=float(np.linalg.norm(goal-begin)-np.linalg.norm(goal-oracle_end)),actual_tangential_delta_m=float(np.linalg.norm(da-(da@rad)*rad)),predicted_tangential_delta_m=float(np.linalg.norm(pa-(pa@rad)*rad)),actual_predicted_displacement_cosine=acos,actual_predicted_tip_displacement_error_m=float(np.linalg.norm(pa-da)),actual_delta_theta=(theta[end]-theta[start]).tolist(),captured_K_joint_increment_sum=inc.tolist(),joint_increment_error_norm=float(np.linalg.norm(theta[end]-theta[start]-inc))))
 grid=np.arange(100.,180.,.025);cum=np.vstack((np.zeros(3),np.cumsum(pred,axis=0)))
 xp=np.array([np.interp(grid,t,cum[:,j])for j in range(3)]).T;xa=np.array([np.interp(grid,t,x[:,j])for j in range(3)]).T
 vp=np.diff(xp,axis=0)/.025;va=np.diff(xa,axis=0)/.025;lags=[]
 for shift in range(41):
  pp=vp[:len(vp)-shift]if shift else vp;aa=va[shift:]if shift else va
  lags.append(dict(actual_delay_s=.025*shift,vector_velocity_normalized_inner_product=float(np.sum(pp*aa)/np.sqrt(np.sum(pp*pp)*np.sum(aa*aa)))))
 output['cases'][case]=dict(posthoc_lag_scan=dict(start_s=100.,end_s=180.,resampling_step_s=.025,actual_delay_search_s=[0.,1.],zero_lag=lags[0],maximum=max(lags,key=lambda a:a['vector_velocity_normalized_inner_product']),values=lags,interpretation='Only a diagnostic correlation of recorded quasistatic increments and actual dynamic response; irregular samples are linearly interpolated, no plant parameter/control gain is fitted or changed.'),exact_recorded_increment_vs_endpoint_equilibrium_max_abs=max(differ),all_recorded_command_K_density_provenance_exact=all(qsourceerr),K_supported_interval_count=len(sourcecheck),windows=windows,selected_half_second_windows=selected)
 np.savez(OUT/(case+'_action_alignment.npz'),oracle_anchored_K_delta_tip=pred,actual_delta_tip=actual,cosine=cos)
 print(case,'maxeqerr',max(differ),'windows',json.dumps(windows[-2]),flush=True)
(OUT/'action_alignment_full.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
for case in output['cases'].values():
 scan=case['posthoc_lag_scan'];scan['tested_delay_count']=len(scan.pop('values'))
(OUT/'action_alignment.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n');print(OUT/'action_alignment.json')
