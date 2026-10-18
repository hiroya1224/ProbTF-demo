from pathlib import Path
import hashlib,json,pickle,time
import numpy as np
from probtf.geometry import quat_to_rotmat,rotmat_to_quat,quaternion_to_rotation_vector
from probtf.spherical_law.first_order import reexpress_state,pose_query
from probtf.spherical_law.support import covariance_support
from probtf_observation_fusion_demo.moment_reverse import MomentQueryEvaluator,backend_record
from probtf_observation_fusion_demo.sensors import _rotation_log_right_jacobian_inverse

indexpath=Path('/tmp/probtf_propagation_comparison_fixtures_20261018/index.json')
fixture=json.loads(indexpath.read_text())['cases']['M93']
payload=Path(fixture['archive_reference']['payload_path']).read_bytes()
assert hashlib.sha256(payload).hexdigest()==fixture['archive_reference']['payload_sha256']
result=pickle.loads(payload)['result'];state=result.frozen_body_branches[0].reference_state
native=fixture['native_likelihood_recorded_by_old_operator'][0]
reference=rotmat_to_quat(np.array(native['rotation']));path=tuple(tuple(v)for v in native['path'])
y=np.r_[native['translation'],np.zeros(3)];W=np.linalg.inv(np.array(native['covariance']))
support=covariance_support(state.covariance);L=support.basis*np.sqrt(support.eigenvalues)
m=L@(.1*np.sin(np.arange(L.shape[1])+.4));P=.8*state.covariance
evaluator=MomentQueryEvaluator(state,path,kind='pose',reference_rotation_wxyz=reference)
before=backend_record()
source=(state.state_id,state.covariance.tobytes())
def core(mm,PP):
 trial=reexpress_state(state,mm,PP)
 q=pose_query(trial,path)
 phi=quaternion_to_rotation_vector(rotmat_to_quat(quat_to_rotmat(reference).T@quat_to_rotmat(q.rotation_wxyz)))
 D=np.eye(6);D[3:,3:]=_rotation_log_right_jacobian_inverse(phi)
 return D@q.mean+np.r_[np.zeros(3),phi],D@q.covariance@D.T
actual=evaluator.evaluate(m,P,likelihood_mean=y,likelihood_precision=W)
mu,S=core(m,P)
def corecost(mm,PP):
 a,b=core(mm,PP);return float(.5*((a-y)@W@(a-y)+np.trace(W@b)))
directions=[]
for i in range(4):
 dm=L@np.sin(np.arange(L.shape[1])+.5+i)
 direction=np.cos(np.arange(L.shape[1])+i)
 dP=L@np.diag(direction*.2)@L.T
 h=2e-5
 fd=(corecost(m+h*dm,P+h*dP)-corecost(m-h*dm,P-h*dP))/(2*h)
 analytic=float(actual.gradient_mean@dm+np.sum(actual.gradient_covariance*dP))
 directions.append(dict(analytic=analytic,core_central_difference=fd,absolute_error=abs(analytic-fd),relative_error=abs(analytic-fd)/max(1.,abs(fd))))
def timing(call,count):
 for unused in range(3):call()
 rows=[]
 for unused in range(count):
  t=time.perf_counter();c=time.process_time();call();rows.append((time.perf_counter()-t,time.process_time()-c))
 return dict(repeats=count,median_wall_s=float(np.median(np.array(rows)[:,0])),median_cpu_s=float(np.median(np.array(rows)[:,1])),min_wall_s=min(r[0]for r in rows))
report=dict(scope='Single archived full-cross 11-edge/66-coordinate geometry. Native computes moments AND analytic reverse likelihood gradient; Python comparison computes only existing reexpression/query moments. This is not whole-inference throughput or a live accuracy result.',original_fixture='M93 historical prior',fixture_index_sha256=hashlib.sha256(indexpath.read_bytes()).hexdigest(),payload_sha256=hashlib.sha256(payload).hexdigest(),joint_dimension=state.mean.size,support_rank=L.shape[1],source_state_id=state.state_id,backend=before,tape_nodes=actual.diagnostics['tape_nodes'],mean_max_abs_vs_core=float(np.max(np.abs(actual.mean-mu))),covariance_max_abs_vs_core=float(np.max(np.abs(actual.covariance-S))),cost_abs_vs_core=abs(actual.cost-corecost(m,P)),gradient_directions=directions,native_moments_and_reverse=timing(lambda:evaluator.evaluate(m,P,likelihood_mean=y,likelihood_precision=W),50),core_moments_only=timing(lambda:core(m,P),25),source_unchanged=source==(state.state_id,state.covariance.tobytes()),backend_unchanged=before==backend_record(),stochastic_sampling=False,production_edited=False)
report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
out=Path('/tmp/probtf_moment_reverse_benchmark_20261018.json');out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
