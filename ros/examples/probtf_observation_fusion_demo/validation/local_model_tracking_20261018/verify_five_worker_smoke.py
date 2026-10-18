#!/usr/bin/python3
"""Production five-worker spawn pipeline smoke, not a performance/reaching claim."""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
from probtf_observation_fusion_demo.event_motion import AsyncMotionSession,_gradient_identity
from probtf_observation_fusion_demo.display import verify_wire_identity
from probtf_observation_fusion_demo.async_artifacts import identity_hash
from probtf_observation_fusion_demo.local_quadratic_tracking import FrozenLocalQuadraticModel
from probtf_observation_fusion_demo.running_gradient import step_running_gradient
from visualization_msgs.msg import MarkerArray

PACKAGE=Path(__file__).resolve().parents[2];SOURCE=PACKAGE/'src/probtf_observation_fusion_demo'

def write(path,value):path.write_text(json.dumps(_gradient_identity(value),indent=2,allow_nan=False)+'\n')
def readrows(path):
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if line.strip():yield json.loads(line)

def source_hashes():return {str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.rglob('*.py')}

def audit(session):
    counts=dict(control_candidates=len(session.control_candidates),artifacts_received=len(session.control_artifacts),wire_candidates_verified=0,wire_bytes_verified=0,core_queries_verified=0,ready_display_identities_verified=0)
    errors=[];display=session.inference_description.common['display']
    for identity,artifact in session.control_artifacts.items():
        try:
            ready=session.control_candidates[identity]['message'];assert ready['candidate_identity']==artifact['candidate_identity'];assert identity_hash(ready['candidate_identity'])==identity
            snapshot=artifact['snapshot'];wire=artifact['markers_wire'];markers=MarkerArray().deserialize(wire);assert verify_wire_identity(snapshot,markers,seed=display['seed'],point_size=display['point_size'])==wire
            evaluation=ready['gradient_evaluation'];assert snapshot.stamp==evaluation['source_stamp'];assert tuple(snapshot.command)==tuple(evaluation['source_command'])
            if evaluation['status']=='ok':
                assert {p.query.state_id for p in snapshot.points if p.frame_label=='hand'}==set(evaluation['body_state_ids']);assert {p.query.state_id for p in snapshot.points if p.frame_label=='switch_goal'}==set(evaluation['goal_state_ids'])
            counts['wire_candidates_verified']+=1;counts['ready_display_identities_verified']+=1;counts['wire_bytes_verified']+=len(wire);counts['core_queries_verified']+=len(snapshot.points)
        except Exception as error:errors.append(dict(candidate_id=identity,error=str(error)))
    adoptions=list(readrows(session.output/'gradient_adoptions.jsonl'));accepted={r['gradient_packet']['packet_id']:r for r in adoptions if r.get('accepted')};max_u=max_v=0.;nonretro_errors=[];used=set();intervals=0;zero_resets=0;base_stamp=session.frames[0].stamp;time_tolerance=8.*np.spacing(base_stamp)
    for row in adoptions:
        if row.get('accepted'):
            if row.get('state_before')!=row.get('state_after'):nonretro_errors.append(dict(request_id=row.get('request_id'),error='adoption reset u/v'))
            else:zero_resets+=1
    for row in readrows(session.output/'commands.jsonl'):
        execution=row.get('reaching_execution') or {};identity=execution.get('gradient_packet_id')
        if identity is None:continue
        adoption=accepted[identity];packet=adoption['gradient_packet'];used.add(identity);step=row['step'];start=step['start_stamp'];received=adoption.get('received_motion_elapsed_s')
        if start+time_tolerance<adoption['adoption_stamp']:nonretro_errors.append(dict(packet_id=identity,error='packet used before activation',start=start,activation=adoption['adoption_stamp']))
        if received is not None and start+time_tolerance<base_stamp+received:nonretro_errors.append(dict(packet_id=identity,error='packet used before receipt',start=start,receipt=base_stamp+received))
        u0=execution['position_before'];v0=execution['velocity_before'];dt=execution['dt'];eta=execution['eta'];gamma=execution['gamma']
        if packet.get('curvature') is None or not np.any(np.asarray(packet['curvature'])):u,v=step_running_gradient(u0,v0,packet['gradient'],dt,eta,gamma)
        else:u,v=FrozenLocalQuadraticModel(packet['source_command'],packet['gradient'],packet['curvature']).advance(u0,v0,dt,eta,gamma)
        max_u=max(max_u,float(np.max(np.abs(u-np.asarray(step['end_command'])))));max_v=max(max_v,float(np.max(np.abs(v-np.asarray(row['velocity'])))));intervals+=1
    body=list(readrows(session.output/'body_ready_adoptions.jsonl'));K=list(readrows(session.output/'stiffness_ready_adoptions.jsonl'));observation_K=[r for r in K if r['job_sequence']>0 and r.get('parameter_packet_id') is not None]
    return dict(counts=counts,wire_errors=errors,body_ready_adoptions=body,stiffness_ready_adoptions=K,body_ready_before_first_K=bool(body) and (not K or body[0]['physical_stamp']<=K[0]['physical_stamp']),body_ready_before_first_K_includes_initialization_job_zero=True,body_ready_before_first_observation_K=(None if not body or not observation_K else body[0]['physical_stamp']<=observation_K[0]['physical_stamp']),first_observation_K_definition='job_sequence > 0 and parameter_packet_id present',all_body_records_did_not_wait_for_K=all(r['body_waited_for_K_evidence'] is False for r in body),accepted_gradient_count=len(accepted),accepted_gradient_states_unchanged=zero_resets==len(accepted),actually_used_gradient_packets=len(used),executed_gradient_intervals=intervals,maximum_command_equation_residual=max_u,maximum_velocity_equation_residual=max_v,nonretroactive_errors=nonretro_errors,clock_comparison_tolerance_s=time_tolerance)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('/tmp/probtf_5worker_spawn_smoke_'+time.strftime('%Y%m%d_%H%M%S')));parser.add_argument('--drain-limit',type=float,default=35.);args=parser.parse_args();args.output.mkdir(exist_ok=False);before=source_hashes();write(args.output/'source_start.json',dict(source_sha256=before,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),purpose='real 5-worker pipeline smoke only; no runtime source edit; not throughput/reaching validation'))
    session=None;error=None;termination=None;physical_done=None;initial=None;progress=[];start=time.monotonic();report=0.;manifest=None;checks=None
    print('OUTPUT',args.output,flush=True)
    try:
        session=AsyncMotionSession(PACKAGE/'config/imu_marker.yaml',args.output/'run',duration=3.,control_mode='running_gradient')
        while True:
            tick=time.monotonic();session.advance();session.raw_publications.clear()
            if session.started is None and session.current_display is not None:
                initial=dict(raw_observation_ids=session.latest_result.records.get('observation_ids'),gradient_packet_present=session.gradient_tracker.last_packet is not None,body_state_id=session.latest_result.prior.state_id,display_candidate_id=session.current_display.get('candidate_id'),ready_candidate_id=None if session.current_control is None else session.current_control.get('candidate_id'),worker_count=len(session.workers),worker_pids=[worker.pid for worker in session.workers],source_stamp=session.frame.stamp)
                assert session.start();write(args.output/'initial_prior_before_start.json',initial)
            if session.motion_finished and physical_done is None:
                physical_done=time.monotonic();write(args.output/'physical_end_manifest.json',session.manifest());print('PHYSICAL_DONE',flush=True)
            elapsed=time.monotonic()-start
            if elapsed>=report:
                m=session.manifest();small={key:m.get(key) for key in ('simulated_elapsed_s','finished','body_estimation_drained','estimation_drained','completed_inference_batches','pending_stiffness_job_sequences','observation_age_s','estimate_stamp','latest_estimated_K','inference_busy','control_busy','stiffness_failure','pending_display_artifact_count')};small['wall_s']=elapsed;progress.append(small);print(json.dumps(_gradient_identity(small)),flush=True);report=elapsed+5.
            display_current=session.current_display is not None and session.current_control is not None and session.current_display.get('candidate_id')==session.current_control.get('candidate_id')
            if session.estimation_finished and not session.control_busy and not session.pending_control_activations and display_current:
                termination='physical_and_all_workers_drained';break
            if session.started is None and elapsed>60.:
                termination='initial_display_not_ready_within_validation_window';break
            if physical_done is not None and time.monotonic()-physical_done>=args.drain_limit:
                termination='validation_drain_observation_window_ended';break
            time.sleep(max(.001,session.period-(time.monotonic()-tick)))
        manifest=session.manifest();checks=audit(session)
    except Exception:
        error=traceback.format_exc();termination='real_exception';print(error,flush=True)
    finally:
        if session is not None:
            if manifest is None:manifest=session.manifest()
            session.close()
        after=source_hashes();write(args.output/'source_end.json',dict(source_sha256=after));result=dict(schema='probtf_5worker_actual_spawn_smoke_v1',output=str(args.output),termination=termination,error=error,wall_s=time.monotonic()-start,initial=initial,manifest=manifest,checks=checks,progress=progress,source_changed_during_smoke=[name for name,sha in before.items() if after.get(name)!=sha],source_added_during_smoke=sorted(set(after)-set(before)),source_edit_performed_by_smoke=False,raw_deleted=False,performance_or_reaching_improvement_claimed=False)
        write(args.output/'result.json',result);print('RESULT',args.output/'result.json',termination,flush=True)
    return 1 if error else 0

if __name__=='__main__':raise SystemExit(main())
