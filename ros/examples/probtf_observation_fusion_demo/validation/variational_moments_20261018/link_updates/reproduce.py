"""Bounded read-only original-acquisition VI/jump/physical-oracle audit.

No optimizer/detector/control run. Truth is read only AFTER choosing events
from inferred same-q update size; it never enters an inference operation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import yaml
from scipy.linalg import cho_solve
from probtf.geometry import quat_to_rotmat,rotmat_to_quat,quaternion_to_rotation_vector
from probtf.spherical_law.first_order import pose_query as core_pose_query, direction_query as core_direction_query
from probtf.spherical_law.support import covariance_support
from probtf_observation_fusion_demo.async_artifacts import load_stiffness_job
from probtf_observation_fusion_demo.body_native import backend_record,pose_query,reexpress_state
from probtf_observation_fusion_demo.moment_observation import NativeMomentLikelihood
from probtf_observation_fusion_demo.sensors import _rotation_log_right_jacobian_inverse
from deflecomp_core.robot.pinocchio_robot import RobotArm
from probtf_switch_press_demo.scene import inverse,transform

ORIGIN=1792080000.

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def maximum(value):return float(np.max(np.abs(value),initial=0.))
def rotvec(R):return quaternion_to_rotation_vector(rotmat_to_quat(R))
def angle(R0,R1):return float(np.linalg.norm(rotvec(R0.T@R1)))
def lines(path):
    with Path(path).open()as stream:
        for line in stream:
            if line.strip():yield json.loads(line)

def fixed_query(state,native):
    """Independent core Python query and fixed-observation chart transport."""
    if native.sensor=='marker':
        q=core_pose_query(state,native.path)
        phi=rotvec(quat_to_rotmat(native.rotation_wxyz).T@quat_to_rotmat(q.rotation_wxyz))
        D=np.eye(6);D[3:,3:]=_rotation_log_right_jacobian_inverse(phi)
        offset=np.r_[np.zeros(3),phi]
    else:
        q=core_direction_query(state,native.path,native.root_direction)
        D=native.basis.T;offset=np.zeros(2)
    return D@q.mean+offset,D@q.covariance@D.T,q

def fixed_fit(mu,P,native):
    e=mu-native.observed_mean
    root=np.linalg.cholesky(native.covariance)
    precision=cho_solve((root,True),np.eye(root.shape[0]))
    quad=.5*float(e@precision@e);trace=.5*float(np.trace(precision@P))
    normalizer=float(np.log(np.diag(root)).sum()+.5*len(e)*math.log(2*math.pi))
    return dict(coordinates='fixed native pose chart'if native.sensor=='marker'else'observed-fixed direction tangent',
        position_mean_error_m=float(np.linalg.norm(e[:3]))if native.sensor=='marker'else None,
        rotation_chart_mean_error_rad=float(np.linalg.norm(e[3:]))if native.sensor=='marker'else None,
        tangent_mean_error=float(np.linalg.norm(e))if native.sensor=='imu'else None,
        mean_whitened_residual_sq=2*quad,expected_whitened_residual_sq=2*(quad+trace),
        expected_negative_log_likelihood=math.fsum((quad,trace,normalizer)),
        NLL_terms=[quad,trace,normalizer])

def oracle_native_fit(T,native):
    if native.sensor=='marker':
        mu=np.r_[T[:3,3],rotvec(quat_to_rotmat(native.rotation_wxyz).T@T[:3,:3])]
        out=fixed_fit(mu,np.zeros((6,6)),native)
        out['position_mean_error_m']=float(np.linalg.norm(T[:3,3]-native.observed_mean[:3]))
        return out
    truth=T[:3,:3].T@native.root_direction
    out=fixed_fit(native.basis.T@truth,np.zeros((2,2)),native)
    out['gravity_direction_angle_rad']=float(np.arctan2(np.linalg.norm(np.cross(truth,native.observed_direction)),truth@native.observed_direction))
    return out

def kl(prior,posterior):
    support=covariance_support(prior.covariance,'saved original prior')
    W=(support.basis/np.sqrt(support.eigenvalues)).T
    a=W@(posterior.mean-prior.mean);V=W@posterior.covariance@W.T
    root=np.linalg.cholesky(.5*V+.5*V.T)
    values=np.linalg.eigvalsh(.5*V+.5*V.T)
    audit=dict(original_support_rank=support.rank,original_prior_positive_eigenvalues_min=float(support.eigenvalues.min()),
        original_prior_positive_eigenvalues_max=float(support.eigenvalues.max()),
        reconstructed_white_covariance_eigenvalues_min=float(values.min()),
        reconstructed_white_covariance_eigenvalues_max=float(values.max()),
        reconstructed_white_covariance_skew_max_abs=maximum(V-V.T),
        meaning='Supplementary reverse reconstruction from saved full P; original optimizer computes KL directly from its whitened Cholesky coordinates. An ill-conditioned reconstruction is not a bitwise validation of that KL.')
    return .5*float(np.trace(V)+a@a-len(a)-2*np.log(np.diag(root)).sum()),audit

def metric(state,path,T):
    q=pose_query(state,path)
    R=quat_to_rotmat(q.rotation_wxyz)
    return dict(modal_position_error_m=float(np.linalg.norm(q.mode[:3]-T[:3,3])),
        O_P_mean_position_error_m=float(np.linalg.norm(q.mean[:3]-T[:3,3])),
        modal_rotation_error_rad=angle(T[:3,:3],R),
        position_covariance_trace_m2=float(np.trace(q.covariance[:3,:3])),
        rotation_covariance_trace_rad2=float(np.trace(q.covariance[3:,3:])))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('/tmp/probtf_variational_commonchart_180s_live_20261018'))
    p.add_argument('--output',type=Path,default=Path('/tmp/probtf_variational_top12_physical_audit_20261018'))
    p.add_argument('--physical-motion-completed',required=True,action='store_true');args=p.parse_args()
    if not(args.run/'physical_finished.json').is_file():raise ValueError('Both physical cutoffs must exist.')
    args.output.mkdir(parents=True,exist_ok=True)
    selected=json.loads((args.run/'variational_diagnostics.json').read_text())
    freeze=json.loads((args.run/'source_freeze.json').read_text())
    configpath=args.run/'frozen_source/ros/examples/probtf_observation_fusion_demo/config/switch_press_scene.yaml'
    config=yaml.safe_load(configpath.read_text())
    Tmarker=inverse(transform(config['base_translation']))@transform(config['marker']['world_translation'],config['marker']['world_rpy'])
    paths_to_hash=[Path(__file__),configpath,
        Path('/home/leus/catkin_ws/src/ProbTF-demo/ros/core/probtf_core/src/probtf/spherical_law/first_order.py')]
    hashes={str(p):digest(p)for p in paths_to_hash}
    report=dict(schema='bounded_original_acquisition_variational_physical_audit_v1',
        source_run=str(args.run),selection_sha256=digest(args.run/'variational_diagnostics.json'),
        runtime_source_manifest_sha256=digest(args.run/'source_freeze.json'),runtime_git_HEAD=freeze['git_HEAD'],
        analysis_source_sha256=hashes,analysis_body_backend=backend_record(),runtime_native_backends=freeze['native_backends'],
        current_postrun_native_symmetry_repair_used_for_queries=True,
        native_repair_scope='Internal mathematically symmetric modal-gap / GPG.T rotation-block assembly only; saved original m/P/logweights are not changed.',
        physical_truth_used_only_for_posthoc_comparison=True,production_changed=False,
        optimizer_or_detector_run=False,scope='Exactly the six inferred-modal-jump-ranked events per case from the saved diagnostics, including initial corrections. No time-nearest truth lookup.',cases={})
    for case,selected_case in selected['cases'].items():
        directory=args.run/case;top=selected_case['top_modal_link_updates']
        indices={e['frame_index']for e in top};physical={}
        for index,row in enumerate(lines(directory/'physical_trajectory.jsonl')):
            if index in indices:physical[index]=row
            if index>=max(indices):break
        adoptions={r['stiffness_job_sequence']:r for r in lines(directory/'body_ready_adoptions.jsonl')}
        robot=RobotArm(str(directory/'robot.urdf'),tip_link='tool0',base_link='base_link')
        refs={int(path.name.split('_')[0]):path for path in(directory/'stiffness_jobs').glob('*.json')}
        rows=[]
        for selection in top:
            seq=selection['job_sequence'];ref=json.loads(refs[seq].read_text());job=load_stiffness_job(ref);result=job['result'];event=job['events'][0]
            prior=result.frozen_body_branches[0].reference_state;post=result.stages[-1].components[0].state
            actual=physical[event['frame_index']]
            if not(event['stamp']==actual['stamp']==prior.stamp==post.stamp and tuple(event['command'])==tuple(actual['command'])==prior.command==post.command):
                raise ValueError('Original frame/stamp/command mismatch: '+case+':'+str(seq))
            if event['event_id']!=selection['event_id']or event['payload_hash']!=selection['payload_hash']:raise ValueError('Selection provenance mismatch')
            robot._fk_update(np.asarray(actual['actual_joints']));B=np.linalg.inv(np.asarray(robot.data.oMf[robot.base_fid].homogeneous))
            fk={name:B@np.asarray(robot.data.oMf[robot.get_frame_id(name)].homogeneous)for name in list(result.link_paths)+['rod_tip','camera_optical_frame']}
            Tcm=inverse(fk['camera_optical_frame'])@Tmarker
            adoption=adoptions.get(seq);adopt_elapsed=None if adoption is None else adoption['physical_stamp']-ORIGIN
            row=dict(job_sequence=seq,event=dict(event),initial_observation_time=event['stamp']==ORIGIN,
                archive_sha256=ref['payload_sha256'],physical_truth_frame_index=event['frame_index'],
                frame_stamp_command_exact=True,physical_tip_crosscheck_error_m=float(np.linalg.norm(fk['rod_tip'][:3,3]-np.asarray(actual['tip']))),
                body_ready_adoption=adoption,body_ready_after_physical_cutoff=(adopt_elapsed is not None and adopt_elapsed>=180.),
                body_adoption_elapsed_s=adopt_elapsed,body_adoption_source_age_s=None if adoption is None else adoption['physical_stamp']-event['stamp'],
                prior_state_id=prior.state_id,canonical_state_id=post.state_id,
                bayes_factor_surprise=result.records.get('bayes_factor_surprise'),links={},branches=[])
            for name,path in list(result.link_paths.items())+[('rod_tip',result.hand_path),('camera_optical_frame',result.camera_path)]:
                q0=pose_query(prior,path);q1=pose_query(post,path)
                row['links'][name]=dict(same_q_modal_update_m=float(np.linalg.norm(q1.mode[:3]-q0.mode[:3])),
                    same_q_O_P_mean_update_m=float(np.linalg.norm(q1.mean[:3]-q0.mean[:3])),
                    same_q_modal_rotation_update_rad=angle(quat_to_rotmat(q0.rotation_wxyz),quat_to_rotmat(q1.rotation_wxyz)),
                    prior=metric(prior,path,fk[name]),posterior=metric(post,path,fk[name]))
            row['map_truth']=dict(prior=metric(prior,result.marker_path,Tmarker),posterior=metric(post,result.marker_path,Tmarker))
            row['relative_camera_marker_truth']=dict(prior=metric(prior,result.camera_marker_path,Tcm),
                posterior=metric(post,result.camera_marker_path,Tcm))
            update=result.records['variational_observation_update']
            natives={n['label']:NativeMomentLikelihood(**{k:n[k]for k in NativeMomentLikelihood.__dataclass_fields__ if k in n})for n in update['native_likelihoods']}
            parents={b.label:b for b in result.frozen_body_branches}
            for diagnostic in update['branch_diagnostics']:
                parent=parents[diagnostic['parent_label']];child=next(x for x in parent.posterior_components if x.label==diagnostic['label'])
                state=reexpress_state(parent.reference_state,child.distribution.mean,child.distribution.covariance,version=post.version,component_id=child.label)
                before_hash=hashlib.sha256(child.distribution.mean.tobytes()+child.distribution.covariance.tobytes()).hexdigest()
                b=dict(label=child.label,parent_label=parent.label,posterior_log_weight=child.log_weight,
                    posterior_weight=float(math.exp(child.log_weight)),raw_gaussian_sha256=before_hash,
                    original_prior_state_id=parent.reference_state.state_id,
                    initial_F=diagnostic['initial_objective'],final_F=diagnostic['final_objective'],
                    final_KL=diagnostic['final_KL'],final_gradient_inf=diagnostic['final_gradient_inf'],
                    completed_iterations=diagnostic['completed_iterations'],stationarity_verified=diagnostic['stationarity_verified'],
                    finite_nonstationary_iterate_used=diagnostic['finite_nonstationary_iterate_used'],
                    initial_distribution=diagnostic['initial_distribution'],native=[],links={})
                for name,path in [('tool0',result.link_paths['tool0']),('rod_tip',result.hand_path),('camera_optical_frame',result.camera_path)]:
                    b['links'][name]=metric(state,path,fk[name])
                nll=[]
                for final in diagnostic['source_evaluator_diagnostics']['propagated_observations']:
                    native=natives[final['native_component_label']];mu,P,_=fixed_query(state,native)
                    old_mu,old_P,_=fixed_query(parent.reference_state,native);canonical_mu,canonical_P,_=fixed_query(post,native)
                    fit=fixed_fit(mu,P,native);nll.extend(fit['NLL_terms'])
                    if not(final['source_state_id']==parent.reference_state.state_id and final['source_stamp']==event['stamp'] and tuple(final['source_command'])==prior.command):raise ValueError('Final evaluator source differs from original acquisition.')
                    truth_pose=Tcm if native.sensor=='marker'else None
                    # For IMU, inverse path's first edge is the sensor's final
                    # local edge; composing its reversed forward path gives
                    # the actual target link. Here that edge is link_3 (or
                    # whichever explicit target was originally captured).
                    if native.sensor=='imu':
                        target=native.path[0][0]
                        truth_pose=fk[target]if target in fk else B@np.asarray(robot.data.oMf[robot.get_frame_id(target)].homogeneous)
                    n=dict(label=native.label,sensor=native.sensor,
                        original_source_exact=True,
                        saved_raw_forward_vs_independent_core_mean_max_abs=maximum(mu-np.asarray(final['predicted_mean'])),
                        saved_raw_forward_vs_independent_core_covariance_max_abs=maximum(P-np.asarray(final['predicted_covariance'])),
                        saved_raw_NLL_vs_independent_core_abs=abs(fit['expected_negative_log_likelihood']-final['expected_negative_log_likelihood']),
                        raw=fixed_fit(mu,P,native),original_parent=fixed_fit(old_mu,old_P,native),canonical=fixed_fit(canonical_mu,canonical_P,native),
                        canonical_minus_raw_mean_norm=float(np.linalg.norm(canonical_mu-mu)),
                        canonical_minus_raw_covariance_max_abs=maximum(canonical_P-P),
                        physical_truth_in_native_likelihood=oracle_native_fit(truth_pose,native))
                    b['native'].append(n)
                try:
                    independent_KL,conditioning=kl(parent.prior,child.distribution);independent_F=math.fsum([independent_KL]+nll)
                    b.update(independent_KL=independent_KL,independent_F=independent_F,
                        independent_KL_reconstruction_conditioning=conditioning,
                        saved_KL_vs_independent_abs=abs(independent_KL-diagnostic['final_KL']),
                        saved_F_vs_independent_abs=abs(independent_F-diagnostic['final_objective']))
                except(np.linalg.LinAlgError,ValueError)as error:b['independent_KL_error']=str(error)
                if before_hash!=hashlib.sha256(child.distribution.mean.tobytes()+child.distribution.covariance.tobytes()).hexdigest():raise ValueError('A saved raw Gaussian changed during read-only analysis.')
                row['branches'].append(b)
            prediction=result.records.get('prediction_command_action')or{}
            row['original_prediction']=dict(source_stamp=prediction.get('source_stamp'),target_stamp=prediction.get('target_stamp'),
                actual_interval_count=len(prediction.get('steps',())),K_epochs=[])
            for step in prediction.get('steps',()):
                epoch=dict(density_id=step.get('density_id'),selected_K=step.get('selected_K'),source_stamp=step.get('source_stamp'),start_stamp=step['start_stamp'],end_stamp=step['end_stamp'])
                epochs=row['original_prediction']['K_epochs']
                if epochs and all(epochs[-1][k]==epoch[k]for k in('density_id','selected_K','source_stamp')):epochs[-1]['end_stamp']=epoch['end_stamp']
                else:epochs.append(epoch)
            rows.append(row);print(case,seq,'tool0 truth prior/post mm',row['links']['tool0']['prior']['modal_position_error_m']*1000,row['links']['tool0']['posterior']['modal_position_error_m']*1000,'reset',row['bayes_factor_surprise']['posterior_change_probability'],flush=True)
        report['cases'][case]=dict(selected_count=len(rows),events=rows)
        (args.output/'details.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    report['analysis_sources_unchanged']=all(digest(path)==value for path,value in hashes.items())
    report['limits']=[
        'Selection ranks saved acquisition-time posterior corrections, not online displayed current-time error, physical motion, or command oscillation.',
        'Physical truth is a separate posthoc FK comparison using the exact original frame, command and timestamp; it is not an inference input.',
        'Prior here means continued Keep law before the selected observation; Reset prior and each raw VI posterior are separately recorded.',
        'The two physical motions completed 180 seconds, but the original IMU drain later failed on internal modal-gap floating symmetry; original sources were not all assimilated.',
        'Current analysis uses the post-run native internal-symmetry repair. It does not rerun the optimizer, change saved m/P/weights, or claim the original runtime used this repair.',
        'A finite nonstationary VI iterate and negative F evidence remain approximations; forward/core consistency does not prove accurate inference or a correct observation model.',
        'The rod-tip position error includes accumulated orientation and tool-length effects; modal and corrected O(P) mean are both reported.']
    (args.output/'details.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(args.output/'details.json',flush=True)

if __name__=='__main__':main()
