"""Actual Cartesian radial/tangential motion, all intervals; no control input."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np,yaml
from probtf_switch_press_demo.scene import inverse,transform
ORIGIN=1792080000.
def rows(path):
 with Path(path).open()as s:
  for l in s:
   if l.strip():yield json.loads(l)
def stats(a,w):
 a=np.asarray(a);w=np.asarray(w);order=np.argsort(a);cs=np.cumsum(w[order]);total=cs[-1]
 return dict(mean=float(w@a/total),rms=float(np.sqrt(w@(a*a)/total)),p50=float(a[order][np.searchsorted(cs,.5*total)]),p95=float(a[order][np.searchsorted(cs,.95*total)]),max=float(a.max()))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=Path('/tmp/probtf_1244f3f_rviz_rerun_20261018'));p.add_argument('--output',type=Path,default=Path('/tmp/probtf_1244_trajectory_efficiency_20261018'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 config=yaml.safe_load((a.run/'frozen_source/ros/examples/probtf_observation_fusion_demo/config/switch_press_scene.yaml').read_text());T=inverse(transform(config['base_translation']))@transform(config['marker']['world_translation'],config['marker']['world_rpy']);goal=(T@np.r_[config['marker']['goal_offset'],1.])[:3]
 result=dict(schema='actual_radial_tangential_motion_v1',source_run=str(a.run),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),goal_in_root=goal.tolist(),truth_only_posthoc=True,source_edited=False,velocity_clipping_added=False,configured_old_servo_component_speed_reference_rad_s=.048,old_servo_speed_limit_active_in_running_gradient=False,cases={})
 for case in('marker','imu_marker'):
  phy=list(rows(a.run/case/'physical_trajectory.jsonl'));t=np.array([r['time']for r in phy]);stamp=np.array([r['stamp']for r in phy]);x=np.array([r['tip']for r in phy]);theta=np.array([r['actual_joints']for r in phy]);q=np.array([r['command']for r in phy]);d=np.linalg.norm(x-goal,axis=1);error=np.max(abs(d-np.array([r['physical_error_m']for r in phy])));del phy
  dt=np.diff(t);dx=np.diff(x,axis=0);r=(goal-x[:-1])/d[:-1,None];radial=np.sum(r*dx,axis=1);tangent=np.linalg.norm(dx-radial[:,None]*r,axis=1);arc=np.linalg.norm(dx,axis=1);progress=d[:-1]-d[1:];v=dx/dt[:,None]
  adopted={r['packet']['packet_id']:r for r in rows(a.run/case/'gradient_adoptions.jsonl')if r.get('accepted')};requests=[];packets=[];increments=[];K=[];density=[];Kstamp=[];qvelocity=[];source_gradient=[];current_gradient=[]
  compact=[]
  for i,row in enumerate(rows(a.run/case/'commands.jsonl')):
   s=row['step'];g=row.get('reaching_execution')or{}
   if not(s['index']==i and s['start_stamp']==stamp[i] and s['end_stamp']==stamp[i+1]and list(s['start_command'])==q[i].tolist()and list(s['end_command'])==q[i+1].tolist()):raise ValueError('Exact actual command/physical frame lineage failed')
   packets.append(g.get('gradient_packet_id'));requests.append(adopted.get(g.get('gradient_packet_id'),{}).get('request_id'));increments.append(s['joint_increment']);K.append(s['selected_K']);density.append(s['density_id']);Kstamp.append(s.get('source_stamp'));qvelocity.append(row['velocity']);source_gradient.append((g.get('source_gradient')or g.get('gradient')or[float('nan')]*6));current_gradient.append((g.get('gradient')or[float('nan')]*6))
   compact.append(dict(index=i,start_s=t[i],end_s=t[i+1],request_id=requests[-1],gradient_packet_id=packets[-1],density_id=density[-1],selected_K=K[-1],K_source_stamp=Kstamp[-1],gradient_source_stamp=g.get('source_stamp'),source_command=g.get('source_command'),body_source_state_ids=g.get('body_state_ids')))
  if len(compact)!=len(dt):raise ValueError('Command interval count differs from physical trajectory')
  qvelocity=np.asarray(qvelocity);increments=np.asarray(increments);source_gradient=np.asarray(source_gradient);current_gradient=np.asarray(current_gradient)
  windows=[]
  for lo,hi in((0,60),(60,120),(120,180),(100,180),(0,180)):
   weights=np.maximum(0,np.minimum(t[1:],hi)-np.maximum(t[:-1],lo));keep=weights>0;fraction=weights[keep]/dt[keep];duration=weights.sum();arclen=float(arc[keep]@fraction);p0=np.array([np.interp(lo,t,x[:,j])for j in range(3)]);p1=np.array([np.interp(hi,t,x[:,j])for j in range(3)]);net=float(np.linalg.norm(goal-p0)-np.linalg.norm(goal-p1));signed=progress[keep];vv=qvelocity[keep]
   windows.append(dict(start_s=lo,end_s=hi,actual_duration_s=float(duration),start_distance_m=float(np.linalg.norm(goal-p0)),end_distance_m=float(np.linalg.norm(goal-p1)),net_inward_progress_m=net,Cartesian_arc_m=arclen,net_progress_per_arc=net/arclen if arclen else None,absolute_start_radial_arc_m=float(np.abs(radial[keep])@fraction),start_tangential_arc_m=float(tangent[keep]@fraction),inward_duration_s=float(weights[keep][signed>0].sum()),outward_duration_s=float(weights[keep][signed<0].sum()),zero_distance_change_duration_s=float(weights[keep][signed==0].sum()),outward_time_fraction=float(weights[keep][signed<0].sum()/duration),tip_speed_m_s=stats(arc[keep]/dt[keep],weights[keep]),signed_inward_speed_m_s=stats(radial[keep]/dt[keep],weights[keep]),tangential_speed_m_s=stats(tangent[keep]/dt[keep],weights[keep]),command_component_abs_speed_rad_s=stats(np.max(abs(vv),axis=1),weights[keep]),command_any_component_above_dormant_048_fraction=float(weights[keep][np.any(abs(vv)>.048,axis=1)].sum()/duration)))
  # Representative observed intervals from the final100..180s, with packet provenance.
  selection={}
  late=(t[:-1]>=100)&(t[1:]<=180)
  masks={'largest_tangential_speed':tangent/dt,'largest_outward_speed':-progress/dt,'largest_inward_speed':progress/dt,'largest_actual_speed':arc/dt}
  for name,score in masks.items():
   index=int(np.argmax(np.where(late,score,-np.inf)));selection[name]=dict(compact[index],actual_delta_tip=dx[index].tolist(),actual_radial_progress_m=float(progress[index]),start_radial_projection_m=float(radial[index]),tangential_displacement_m=float(tangent[index]),tip_speed_m_s=float(arc[index]/dt[index]),signed_inward_speed_m_s=float(radial[index]/dt[index]),tangential_speed_m_s=float(tangent[index]/dt[index]))
  np.savez(a.output/(case+'_motion.npz'),time=t,stamp=stamp,tip=x,theta=theta,command=q,command_velocity=qvelocity,joint_increment=increments,source_gradient=source_gradient,current_gradient=current_gradient,radial_projection=radial,tangential_displacement=tangent,distance=d)
  (a.output/(case+'_interval_sources.jsonl')).write_text(''.join(json.dumps(x,allow_nan=False)+'\n'for x in compact))
  result['cases'][case]=dict(physical_rows=len(t),command_intervals=len(compact),command_frame_lineage_exact=True,goal_distance_crosscheck_max_abs_m=float(error),windows=windows,selected_intervals=selection)
  print(case,json.dumps(windows),flush=True)
 (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(a.output/'summary.json')
if __name__=='__main__':main()
