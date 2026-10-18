from pathlib import Path
from dataclasses import replace
import json
import numpy as np,yaml
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.moving_inference import MovingInference

P=Path('/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/probtf_observation_fusion_demo')
out=Path('/tmp/probtf_hand_imu_variational_test_audit_20261018');out.mkdir(exist_ok=True)
scene=MovingSwitchScene(out/'scene',imu_target_link='tool0')
frame=scene.step(0.,1000.);inputs=scene.sensor_inputs(frame)
obs=yaml.safe_load((P/'config/imu_hand.yaml').read_text())['observations']
inference=MovingInference(scene.robot_description,obs)
first=inference.update(inputs,0)
previous=first.stages[-1].components[0].state
second=inference.update(replace(inputs,stamp=1000.05,time=.05),1)
current=second.stages[-1].components[0].state
record=dict(method=inference.observation_update_method,edges={},branches={},prior_carried=np.array_equal(second.prior.covariance,previous.covariance))
for edge in inference.body_ids:
 i=current.edge_ids.index(edge);s=slice(6*i+3,6*i+6)
 query=second.local_estimates[0].posterior.query
 record['edges'][edge]=dict(initial_trace=float(np.trace(first.prior.covariance[s,s])),first_trace=float(np.trace(previous.covariance[s,s])),second_trace=float(np.trace(current.covariance[s,s])),second_cov_change=float(np.linalg.norm(current.covariance[s,s]-previous.covariance[s,s])),gravity_jac_rank=int(np.linalg.matrix_rank(query.jacobian[:,s])),gravity_prior_cross=float(np.linalg.norm(query.cross_covariance[s,:])))
for name,result in [('first',first),('second',second)]:
 record['branches'][name]=[{k:b[k]for k in ('parent_label','source_prior_state_id','initial_objective','final_objective','initial_distribution','final_KL','final_gradient_inf','completed_iterations','valid_evaluator_calls','stationarity_verified','posterior_log_weight')}for b in result.records['variational_observation_update']['branch_diagnostics']]
path=out/'summary.json';path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
