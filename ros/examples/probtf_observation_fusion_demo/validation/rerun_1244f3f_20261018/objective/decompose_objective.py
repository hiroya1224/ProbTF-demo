"""Read-only actual-request full-J decomposition on its immutable candidate."""
from dataclasses import replace
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import yaml

from probtf.spherical_law.support import covariance_support
from probtf_observation_fusion_demo.async_artifacts import capture_candidate_identity,load_stiffness_job
from probtf_observation_fusion_demo.async_state import ActionStep,predict_result
from probtf_observation_fusion_demo.joint_batch_fusion import log_evidence_parts_difference
from probtf_observation_fusion_demo.moving_control import MovingController,control_components,_relative_mixture_log_parts
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.stiffness_command import StiffnessCommandMap
from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier
from probtf_observation_fusion_demo.visibility_factor import VisibilityFactor
from probtf_observation_fusion_demo.body_native import pose_query

BASE=Path('/tmp/probtf_1244f3f_rviz_rerun_20261018')
OUT=Path('/tmp/probtf_1244_objective_analysis_20261018')
PACKAGE=Path('/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/probtf_observation_fusion_demo')
TERM_NAMES=('position_quadratic','position_logdet','position_normalizer','bingham_quadratic','bingham_normalizer')


def derivative(baseline,plus,minus,hp,hm):
    if hp is not None and hm is not None:
        return (hm*(plus-baseline)/hp-hp*(minus-baseline)/hm)/(hp+hm)
    return (plus-baseline)/hp if hp is not None else (baseline-minus)/hm


def cosine(a,b):
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return None if denominator==0 else float(np.dot(a,b)/denominator)


def summarize_gradient(g,radial,velocity):
    return dict(vector=g.tolist(),norm=float(np.linalg.norm(g)),
        cosine_with_distance_gradient=cosine(g,radial),
        distance_derivative_under_negative_unit_gradient=(None if np.linalg.norm(g)==0 else -float(radial@g)/np.linalg.norm(g)),
        directional_derivative_at_actual_velocity=float(g@velocity))


def restore(source):
    directory=BASE/source['case'];row=source['control'];pred=row['prediction'];cp=pred['command_path']
    reference=dict(source['matching_job_reference'])
    if not Path(reference['path']).exists():
        reference['path']=str(directory/'stiffness_jobs'/Path(reference['path']).name)
    original=load_stiffness_job(reference)['result']
    assert original.prior.state_id==cp['source_prior_state_id']
    ref=cp['ledger_reference'];begin,end=ref['start_index'],ref['end_index_exclusive']
    steps=[];chain=hashlib.sha256(b'ProbTF actual-command ledger v1').hexdigest();source_chain=chain if begin==0 else None
    for line in (directory/ref['path']).open():
        record=json.loads(line)['step'];index=record['index']
        if index>=end:break
        step=ActionStep(**record)
        chain=hashlib.sha256(bytes.fromhex(chain)+json.dumps(step.record(),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if index+1==begin:source_chain=chain
        if begin<=index<end:steps.append(step)
    assert chain==ref['prefix_chain_sha256'] and source_chain==ref['source_prefix_chain_sha256']
    axes=cp['axes'];ids=tuple(a['edge_id'] for a in axes);actions=tuple(dict(joint_index=a['joint_index'],axis=a['axis']) for a in axes)
    predicted=predict_result(original,steps,target_stamp=pred['present_stamp'],frame_index=pred['frame_index'],body_ids=ids,actions=actions,ledger_reference=ref)
    identity=capture_candidate_identity(predicted,parameter_identity=row['candidate_identity']['parameter'])
    assert identity==row['candidate_identity']
    # Reattach only the *original* stiffness action record: no observation,
    # parameter density, representative or equilibrium is recomputed here.
    predicted=replace(predicted,records=dict(predicted.records,stiffness=row['stiffness']))
    inference=MovingInference((directory/'robot.urdf').read_text(),())
    settings=yaml.safe_load((PACKAGE/'config/stiffness_identification.yaml').read_text())
    identifier=StiffnessIdentifier(directory/'robot.urdf',ids,settings)
    identifier.last_record=predicted.records['stiffness']
    mapping=StiffnessCommandMap(identifier,predicted,predicted.records['stiffness']['action_command_proposal'])
    motion=inference.settings['control']
    controller=MovingController(inference,(directory/'robot.urdf').read_text(),motion['goal_offset'],motion['goal_orientation_wxyz'],motion,integration_steps=motion['integration_steps'])
    return predicted,mapping,controller


def analyze(path):
    source=json.loads(path.read_text());row=source['control'];saved=row['gradient_evaluation']
    result,mapping,controller=restore(source)
    _,components=control_components(result)
    view_settings=yaml.safe_load((PACKAGE/'config/moving_estimation.yaml').read_text())['control']['visibility_factor']
    view=VisibilityFactor(view_settings)
    camera_marker=tuple((edge,not inverse) for edge,inverse in reversed(result.camera_path))+tuple(result.marker_path)
    cache={}

    def evaluate(command):
        key=tuple(command)
        if key in cache:return cache[key]
        states=[]
        _,branches,queries=controller._evaluate_command_objective(components,np.asarray(command),mapping,trial_states=states)
        logparts=_relative_mixture_log_parts(branches)
        responsibility_logs=[b['log_weight']+log_evidence_parts_difference(b['log_density_parts'],logparts) for b in branches]
        responsibility=np.exp(responsibility_logs-logsumexp(responsibility_logs))
        nativeweights=np.exp(np.asarray([b['log_weight'] for b in branches])-logsumexp([b['log_weight'] for b in branches]))
        terms=[];squared=[];modal_squared=[];trace=[];positions=[];modal_positions=[];covariances=[];precise_errors=[]
        for b,query in zip(branches,queries):
            covariance=np.asarray(b['relative_position_covariance']);support=covariance_support(covariance)
            d=np.asarray(b['relative_position_mean'])-controller.goal_offset
            terms.append([.5*float(d@support.inverse@d),.5*support.log_pdet,
                .5*support.rank*math.log(2*math.pi),
                -float(controller.goal_orientation@np.asarray(b['relative_orientation_bingham'])@controller.goal_orientation),b['log_bingham_normalizer']])
            squared.append(float(d@d));trace.append(float(np.trace(covariance)));positions.append(b['relative_position_mean'])
            modal_positions.append(query.mode[:3]);modal_squared.append(float(np.sum((query.mode[:3]-controller.goal_offset)**2)))
            covariances.append(covariance);precise_errors.append(support.inverse@d)
        queries_view=tuple(pose_query(state,camera_marker,frame='camera_optical_frame') for state in states)
        visibility=view.evaluate(components,queries_view)
        item=dict(J=-math.fsum(logparts),Jparts=np.array([-logparts[0],-logparts[1]]),
            terms=np.asarray(terms),responsibility=responsibility,nativeweights=nativeweights,
            mean_distance=math.sqrt(nativeweights@np.asarray(squared)),
            modal_distance=math.sqrt(nativeweights@np.asarray(modal_squared)),
            rms_distance=math.sqrt(nativeweights@(np.asarray(squared)+np.asarray(trace))),
            relative_mean=nativeweights@np.asarray(positions),
            relative_mode=nativeweights@np.asarray(modal_positions),
            positions=np.asarray(positions),covariances=np.asarray(covariances),
            precision_errors=np.asarray(precise_errors),
            V=visibility['weighted_penalty'],branches=branches,visibility=visibility)
        item['selected_objective']=item['J']+item['V'];cache[key]=item;return item

    def gradients(command,original_stencil=False):
        baseline=evaluate(command);n=len(command);g=np.zeros(n);terms=np.zeros((len(TERM_NAMES),n));gd=np.zeros(n);gmodal=np.zeros(n);grms=np.zeros(n);gv=np.zeros(n);dm=np.zeros((3,n));dmode=np.zeros((3,n));gmean=np.zeros(n);gcov=np.zeros(n)
        for index in range(n):
            hp=hm=1.e-5
            if original_stencil:
                coordinate=saved['finite_difference_diagnostics']['coordinates'][index]
                hp=coordinate['forward']['step'] if coordinate['forward']['status']=='ok' else None
                hm=-coordinate['backward']['step'] if coordinate['backward']['status']=='ok' else None
            plus=minus=None
            if hp is not None:
                point=np.asarray(command).copy();point[index]+=hp;plus=evaluate(point)
            if hm is not None:
                point=np.asarray(command).copy();point[index]-=hm;minus=evaluate(point)
            p=baseline if plus is None else plus;m=baseline if minus is None else minus
            # Subtract high/low J parts before differencing. All other
            # component terms are kept separate at the original native mass.
            dp=log_evidence_parts_difference(p['Jparts'],baseline['Jparts'])
            dmj=log_evidence_parts_difference(m['Jparts'],baseline['Jparts'])
            g[index]=derivative(0.,dp,dmj,hp,hm)
            branch_gradient=derivative(baseline['terms'],p['terms'],m['terms'],hp,hm)
            terms[:,index]=baseline['responsibility']@branch_gradient
            gd[index]=derivative(baseline['mean_distance'],p['mean_distance'],m['mean_distance'],hp,hm)
            gmodal[index]=derivative(baseline['modal_distance'],p['modal_distance'],m['modal_distance'],hp,hm)
            grms[index]=derivative(baseline['rms_distance'],p['rms_distance'],m['rms_distance'],hp,hm)
            gv[index]=derivative(baseline['V'],p['V'],m['V'],hp,hm)
            dm[:,index]=derivative(baseline['relative_mean'],p['relative_mean'],m['relative_mean'],hp,hm)
            dmode[:,index]=derivative(baseline['relative_mode'],p['relative_mode'],m['relative_mode'],hp,hm)
            dmu=derivative(baseline['positions'],p['positions'],m['positions'],hp,hm)
            dS=derivative(baseline['covariances'],p['covariances'],m['covariances'],hp,hm)
            v=baseline['precision_errors'];w=baseline['responsibility']
            gmean[index]=w@np.einsum('ki,ki->k',v,dmu)
            gcov[index]=-.5*float(w@np.einsum('ki,kij,kj->k',v,dS,v))
        return dict(task=g,terms=terms,distance=gd,modal_distance=gmodal,rms=grms,view=gv,mean_jacobian=dm,modal_jacobian=dmode,position_mean_term=gmean,position_covariance_term=gcov)

    q=np.asarray(saved['source_command']);current=np.asarray(source['command_record']['step']['end_command']);velocity=np.asarray(source['command_record']['velocity'])
    source_value=evaluate(q);source_grad=gradients(q,True)
    np.testing.assert_allclose(source_value['J'],saved['objective_at_source'],rtol=0,atol=1.e-8)
    np.testing.assert_allclose(source_grad['task'],saved['task_gradient'],rtol=1.e-8,atol=1.e-5)
    np.testing.assert_allclose(source_value['V'],saved['visibility_factor']['weighted_penalty'],rtol=1.e-8,atol=1.e-5)
    current_value=evaluate(current);current_grad=gradients(current)
    H=np.asarray(saved['curvature']);delta=current-q
    original_g=np.asarray(saved['selected_gradient']);effective=original_g+H@delta
    actual_effective=np.asarray(source['command_record']['reaching_execution']['gradient'])
    np.testing.assert_allclose(effective,actual_effective,rtol=1.e-12,atol=1.e-8)
    directions={}
    for name,direction in (('negative_source_selected_gradient',-original_g),
            ('negative_source_Fisher_inverse_gradient',-np.linalg.solve(H,original_g)),
            ('actual_command_velocity',velocity)):
        norm=float(np.linalg.norm(direction));unit=direction/norm
        value=dict(command_direction=direction.tolist(),direction_norm=norm,
            unit_command_direction=unit.tolist(),source_selected_J_slope_per_unit_command=float(original_g@unit),
            source_corrected_mean_distance_slope_per_unit_command=float(source_grad['distance']@unit),
            source_modal_distance_slope_per_unit_command=float(source_grad['modal_distance']@unit),
            source_RMS_distance_slope_per_unit_command=float(source_grad['rms']@unit))
        for prefix,jacobian,error in (('corrected_mean',source_grad['mean_jacobian'],source_value['relative_mean']-controller.goal_offset),
                ('modal',source_grad['modal_jacobian'],source_value['relative_mode']-controller.goal_offset)):
            task_velocity=jacobian@unit;radial=float(error@task_velocity/np.linalg.norm(error));tangent=task_velocity-radial*error/np.linalg.norm(error)
            value[prefix+'_radial_rate_per_unit_command']=radial
            value[prefix+'_tangential_rate_per_unit_command']=float(np.linalg.norm(tangent))
        directions[name]=value
    output=dict(case=source['case'],request_id=row['request_id'],candidate_id=row['candidate_id'],
        actual_used_gradient_packet_id=source['adoption']['packet']['packet_id'],
        physical_sample_stamp=source['command_record']['step']['end_stamp'],
        physical_sample_offset_s=source['time'],source_stamp=saved['source_stamp'],
        source_command=q.tolist(),current_command=current.tolist(),velocity=velocity.tolist(),
        source_K=saved['selected_K'],density_id=saved['parameter_density_id'],
        source_goal_offset=controller.goal_offset.tolist(),
        body_source_state_ids=saved['body_state_ids'],goal_source_state_ids=saved['goal_state_ids'],
        full_candidate_identity_reconstructed_exact=True,complete_actual_ledger_chain_verified=True,
        source_task_gradient_maximum_replay_difference=float(np.max(np.abs(source_grad['task']-saved['task_gradient']))),
        original_gradient_effective_force_maximum_difference=float(np.max(np.abs(effective-actual_effective))),
        metadata=dict(native_component_count=len(source_value['branches']),source_responsibility=source_value['responsibility'].tolist(),
            native_weights=source_value['nativeweights'].tolist(),source_observation_stamp=saved['body_observation_source_stamp'],
            K_observation_stamp=json.loads(row['candidate_identity']['parameter']['action_packet_ref']['action_json'])['estimated_source_stamp'],
            probe_count=len(cache),new_sensor_updates=0,new_K_density_evaluations=0,physical_truth_received=False),
        frozen_source=dict(J=source_value['J'],weighted_view=source_value['V'],selected=source_value['selected_objective'],
            mean_distance_m=source_value['mean_distance'],rms_distance_m=source_value['rms_distance'],
            modal_distance_m=source_value['modal_distance'],relative_mean=source_value['relative_mean'].tolist(),relative_mode=source_value['relative_mode'].tolist(),
            mean_correction_norm_m=float(np.linalg.norm(source_value['relative_mean']-source_value['relative_mode'])),
            position_covariance_eigenvalues=[np.linalg.eigvalsh(S).tolist() for S in source_value['covariances']],
            position_covariance_eigenvectors=[np.linalg.eigh(S)[1].tolist() for S in source_value['covariances']],
            position_precision_matrices=[covariance_support(S).inverse.tolist() for S in source_value['covariances']],
            corrected_position_error=(source_value['relative_mean']-controller.goal_offset).tolist(),
            corrected_error_components_in_covariance_eigenbasis=[(np.linalg.eigh(S)[1].T@(source_value['positions'][k]-controller.goal_offset)).tolist() for k,S in enumerate(source_value['covariances'])],
            relative_mean_command_jacobian=source_grad['mean_jacobian'].tolist(),relative_mode_command_jacobian=source_grad['modal_jacobian'].tolist(),
            per_native_terms=source_value['terms'].tolist(),
            gradient_parts={name:summarize_gradient(g,source_grad['distance'],velocity) for name,g in zip(TERM_NAMES,source_grad['terms'])},
            task=summarize_gradient(source_grad['task'],source_grad['distance'],velocity),
            view=summarize_gradient(source_grad['view'],source_grad['distance'],velocity),
            position_mean_term=summarize_gradient(source_grad['position_mean_term'],source_grad['distance'],velocity),
            position_covariance_term=summarize_gradient(source_grad['position_covariance_term'],source_grad['distance'],velocity),
            position_quadratic_chain_split_error_inf=float(np.max(np.abs(source_grad['position_mean_term']+source_grad['position_covariance_term']-source_grad['terms'][0]))),
            source_distance_gradient=source_grad['distance'].tolist(),source_rms_gradient=source_grad['rms'].tolist(),
            term_gradient_sum_error_inf=float(np.max(np.abs(source_grad['terms'].sum(axis=0)-source_grad['task'])))),
        at_actual_command=dict(J=current_value['J'],weighted_view=current_value['V'],selected_objective=current_value['selected_objective'],
            mean_distance_m=current_value['mean_distance'],rms_distance_m=current_value['rms_distance'],
            modal_distance_m=current_value['modal_distance'],relative_mean=current_value['relative_mean'].tolist(),relative_mode=current_value['relative_mode'].tolist(),
            mean_correction_norm_m=float(np.linalg.norm(current_value['relative_mean']-current_value['relative_mode'])),
            per_native_terms=current_value['terms'].tolist(),
            task=summarize_gradient(current_grad['task'],current_grad['distance'],velocity),
            gradient_parts={name:summarize_gradient(g,current_grad['distance'],velocity) for name,g in zip(TERM_NAMES,current_grad['terms'])},
            position_mean_term=summarize_gradient(current_grad['position_mean_term'],current_grad['distance'],velocity),
            position_covariance_term=summarize_gradient(current_grad['position_covariance_term'],current_grad['distance'],velocity),
            view=summarize_gradient(current_grad['view'],current_grad['distance'],velocity),
            selected=summarize_gradient(current_grad['task']+current_grad['view'],current_grad['distance'],velocity),
            effective_local_gradient=summarize_gradient(effective,current_grad['distance'],velocity),
            actual_vs_local_gradient_cosine=cosine(current_grad['task']+current_grad['view'],effective),
            distance_derivative_at_actual_velocity_m_per_s=float(current_grad['distance']@velocity),
            modal_distance_derivative_at_actual_velocity_m_per_s=float(current_grad['modal_distance']@velocity),
            rms_derivative_at_actual_velocity_m_per_s=float(current_grad['rms']@velocity),
            relative_position_velocity=(current_grad['mean_jacobian']@velocity).tolist(),
            distance_gradient=current_grad['distance'].tolist(),rms_gradient=current_grad['rms'].tolist()),
        local_model=dict(regularization_added=saved['local_model_diagnostics']['regularization_added'],
            task_curvature=saved['local_model_diagnostics']['task_curvature'],
            view_curvature=saved['local_model_diagnostics']['view_curvature'],curvature=H.tolist(),
            actual_increment_l2=float(np.linalg.norm(delta)),actual_increment=delta.tolist(),
            force_changes=dict(task=(np.asarray(saved['local_model_diagnostics']['task_curvature'])@delta).tolist(),
                view=(np.asarray(saved['local_model_diagnostics']['view_curvature'])@delta).tolist(),
                regularization=(saved['local_model_diagnostics']['regularization_added']*delta).tolist())),
        diagnostic_directions=directions,
        diagnostic_direction_comparison_is_not_a_new_controller_or_acceptance_rule=True)
    out=OUT/(path.name.replace('_source.json','_decomposition.json'))
    out.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(source['case'],row['request_id'],'norms',[(name,round(x['norm'],3)) for name,x in output['frozen_source']['gradient_parts'].items()],
        'task-distance cos',output['frozen_source']['task']['cosine_with_distance_gradient'],
        'at-current local/actual cos',output['at_actual_command']['actual_vs_local_gradient_cosine'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run',type=Path,default=BASE)
    parser.add_argument('--source-dir',type=Path,default=OUT)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();BASE=args.input_run;OUT=args.output
    OUT.mkdir(parents=True,exist_ok=True)
    for case,target in (('marker',112),('marker',158),('marker',179),('imu_marker',123),('imu_marker',140),('imu_marker',179)):
        analyze(args.source_dir/(case+'_'+str(target)+'_source.json'))
