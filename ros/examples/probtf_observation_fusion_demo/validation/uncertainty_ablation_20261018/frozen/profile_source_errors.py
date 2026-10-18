"""Read-only original-source absolute estimation errors; truth only offline."""
import json,hashlib
from pathlib import Path
import numpy as np,yaml
from run_frozen import extract,restore,OUT,BASE,CONFIG
from probtf_observation_fusion_demo.body_native import pose_query,point_query
from probtf_observation_fusion_demo.moving_control import control_components
from probtf_switch_press_demo.scene import inverse,transform
scene=yaml.safe_load((CONFIG/'switch_press_scene.yaml').read_text());TM=inverse(transform(scene['base_translation']))@transform(scene['marker']['world_translation'],scene['marker']['world_rpy']);goal=(TM@np.r_[scene['marker']['goal_offset'],1.])[:3]
records=[]
for s in extract():
 result,mapping,controller=restore(s);_,cs=control_components(result);hand=pose_query(cs[0].state,result.hand_path,frame='base_link');gs=[point_query(c.state,result.marker_path,controller.goal_offset,frame='base_link',query_id='offline_error')for c in cs];w=np.array([c.weight for c in cs]);g=sum(v.mean*ww for v,ww in zip(gs,w));physical=np.asarray(s['physical_source']['tip']);h=hand.mean[:3];m=hand.mode[:3];goalerror=g-goal;bodyerror=h-physical;relerror=(h-g)-(physical-goal)
 rec=dict(case=s['case'],request_id=s['control']['request_id'],source_time_s=s['control']['gradient_evaluation']['source_stamp']-1792080000.,actual_tip=physical.tolist(),true_goal=goal.tolist(),estimated_tip_corrected_mean=h.tolist(),estimated_tip_mode=m.tolist(),estimated_goal_corrected_mean=g.tolist(),body_error_m=bodyerror.tolist(),body_error_norm_m=float(np.linalg.norm(bodyerror)),goal_error_m=goalerror.tolist(),goal_error_norm_m=float(np.linalg.norm(goalerror)),relative_error_m=relerror.tolist(),relative_error_norm_m=float(np.linalg.norm(relerror)),physical_target_distance_m=float(np.linalg.norm(physical-goal)),estimated_target_distance_m=float(np.linalg.norm(h-g)),estimated_vs_physical_root_error_bearing_cosine=float((h-g)@(physical-goal)/(np.linalg.norm(h-g)*np.linalg.norm(physical-goal))),body_goal_error_cosine=float(bodyerror@goalerror/(np.linalg.norm(bodyerror)*np.linalg.norm(goalerror))),corrected_body_sigma_eigenvalues=np.linalg.eigvalsh(hand.covariance[:3,:3]).tolist(),corrected_goal_sigma_eigenvalues=[np.linalg.eigvalsh(v.covariance).tolist()for v in gs])
 records.append(rec);print(rec['case'],round(rec['source_time_s'],1),'body mm',round(rec['body_error_norm_m']*1000,2),'goal mm',round(rec['goal_error_norm_m']*1000,2),'relative mm',round(rec['relative_error_norm_m']*1000,2),'err cos',round(rec['body_goal_error_cosine'],3),flush=True)
(OUT/'source_absolute_errors.json').write_text(json.dumps(dict(schema='frozen_source_absolute_error_diagnostic_v1',oracle_only_posthoc=True,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=records),indent=2,allow_nan=False)+'\n')
