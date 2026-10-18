"""Replay saved events through the current production estimator, with no hooks."""
import argparse
from dataclasses import replace
import hashlib,json,sqlite3,time
from pathlib import Path
from threading import RLock

import cv2
import yaml

from probtf_observation_fusion_demo.async_state import ActionStep,RecordedCommandPath
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore
from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier

PACKAGE=Path(__file__).resolve().parents[2]

def rows(path):
 with Path(path).open() as stream:
  for raw in stream:
   if raw.strip():yield json.loads(raw)

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 run=a.run.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
 failed=next(rows(run/'estimation_failure.jsonl'));target=failed['event_id'];expected={}
 for row in rows(run/'estimation.jsonl'):
  s=row['stiffness'];posterior=s.get('posterior') or s.get('retained_prior') or {}
  expected[row['sensor_event']['event_id']]=dict(source_prior_state_id=s['source_prior_state_id'],body_posterior_state_ids=s['body_posterior_state_ids'],body_ratio_hash=digest(s.get('body_marginal')),factor_ids=posterior.get('factor_ids'),density_id=posterior.get('density_id'))
 s=failed['stiffness'];expected[target]=dict(source_prior_state_id=s['source_prior_state_id'],body_posterior_state_ids=s['body_posterior_state_ids'],body_ratio_hash=digest(s.get('body_marginal')),factor_ids=s['retained_prior']['factor_ids']+s['pending_factor_ids'],density_id=None)
 selections=list(rows(run/'assimilation_selection.jsonl'))
 target_position=next(i for i,s in enumerate(selections) if s['event_id']==target);selections=selections[:target_position+1]
 store=object.__new__(SensorEventStore);store._lock=RLock();store.connection=sqlite3.connect('file:'+str(run/'observations.sqlite3')+'?mode=ro',uri=True);store.connection.row_factory=sqlite3.Row
 original_target=store.connection.execute('SELECT * FROM events WHERE event_id=?',(target,)).fetchone()
 extra=[]
 for r in store.connection.execute('SELECT event_id,sequence,kind,stamp,payload_json FROM events WHERE stamp>? OR (stamp=? AND sequence>?) ORDER BY stamp,sequence',(original_target['stamp'],original_target['stamp'],original_target['sequence'])):
  payload=json.loads(r['payload_json']) if r['payload_json'] else None
  if payload is None:break
  extra.append(dict(event_id=r['event_id'],selected_capture_index=int(payload['frame_index']),after_original_failure=True))
  if r['kind'] in ('imu','image'):break
 selections+=extra;end_index=max(s['selected_capture_index'] for s in selections)
 actions=[]
 for row in rows(run/'commands.jsonl'):
  if row['step']['index']>=end_index:break
  actions.append(ActionStep(**row['step']))
 templates=tuple(yaml.safe_load((PACKAGE/'config/imu_marker.yaml').read_text())['observations']);settings=yaml.safe_load((PACKAGE/'config/stiffness_identification.yaml').read_text())
 inference=MovingInference((run/'robot.urdf').read_text(),templates);identifier=StiffnessIdentifier(run/'robot.urdf',inference.body_ids,settings)
 runtime_source=PACKAGE/'src/probtf_observation_fusion_demo';sources_before={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in runtime_source.glob('*.py')}
 result_rows=[];previous=previous_index=None;started=time.monotonic()
 summary=dict(source=str(run),original_failed_event=target,offline_replay_only=True,original_DB_read_only=True,
  production_estimator_unmodified_by_script=True,monkeypatches_or_forced_polish_used=False,
  original_body_candidates_and_actual_command_ledger_used=True,
  numerical_quality_diagnostics_are_not_acceptance_gates=True,runtime_source_sha256=sources_before,events=result_rows)
 try:
  for selection in selections:
   original=store._event(store.connection.execute('SELECT * FROM events WHERE event_id=?',(selection['event_id'],)).fetchone());event=replace(original,status='processing');index=selection['selected_capture_index']
   path=None if previous is None else RecordedCommandPath(previous,actions[previous_index:index],body_ids=inference.body_ids,actions=inference.actions)
   t=time.monotonic();result=inference.update_sensor_event(event,event_index=index,command_map=path,probability_unit='per_effective_sensor_event')
   record=identifier.update(result);proposal=identifier.command_proposal(result);result.records['stiffness']=record
   posterior=record.get('posterior') or record.get('retained_prior') or {};ref=expected.get(event.event_id)
   matches=None if ref is None else dict(source_prior_state_id=record['source_prior_state_id']==ref['source_prior_state_id'],body_posterior_state_ids=record['body_posterior_state_ids']==ref['body_posterior_state_ids'],complete_body_ratio=digest(record.get('body_marginal'))==ref['body_ratio_hash'],all_factors_retained_once=posterior.get('factor_ids')==ref['factor_ids'])
   branch={}
   for name in ('keep_evidence','reset_evidence'):
    integral=(record.get('bayes_factor_surprise') or {}).get(name)
    if integral is not None:
     d=integral['diagnostics'];branch[name]=dict(method=integral['method'],log_value=integral['log_value'],**{key:d.get(key) for key in ('optimizer_success','mode_score_inf','gradient_tolerance','stationarity_verified','finite_nonstationary_point_accepted','numerical_quality_checks_are_update_gates','Hessian_steps','Hessian_step','two_step_stability_checked','two_step_stability_verified','positive_definiteness_verified','stationarity_polish')})
   row=dict(event_id=event.event_id,sequence=event.sequence,stamp=event.stamp,frame_index=index,after_original_failure=selection.get('after_original_failure',False),seconds=time.monotonic()-t,status=record['status'],failure_reason=record.get('failure_reason'),posterior_defined=record.get('posterior') is not None,update_applied=record['update_applied'],matches_original=matches,
    cumulative_density_id=posterior.get('density_id'),original_density_id=None if ref is None else ref['density_id'],density_id_same_as_original=(None if ref is None or ref['density_id'] is None else posterior.get('density_id')==ref['density_id']),
    factor_ids=posterior.get('factor_ids'),pending_factor_ids=record.get('pending_factor_ids'),proposal_available=proposal is not None,proposal_status=record.get('proposal_status'),selected_K=None if proposal is None else proposal['selected_K'],evidence=branch)
   result_rows.append(row);summary.update(seconds=time.monotonic()-started,all_original_body_and_factor_id_checks_passed=all(all(r['matches_original'].values()) for r in result_rows if r['matches_original'] is not None),all_processed_events_have_defined_posterior=all(r['posterior_defined'] for r in result_rows))
   (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({key:row[key] for key in ('event_id','status','seconds','matches_original','proposal_available')}),flush=True)
   if not row['posterior_defined']:
    (out/'unexpected_failure.json').write_text(json.dumps(record,indent=2)+'\n');break
   previous,previous_index=result,index
  sources_after={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in runtime_source.glob('*.py')};summary['runtime_source_unchanged_during_replay']=sources_after==sources_before
  summary['original_failed_event_now_updated']=any(r['event_id']==target and r['update_applied'] and r['posterior_defined'] and r['proposal_available'] for r in result_rows)
  summary['subsequent_effective_event_updated']=any(r['after_original_failure'] and r['update_applied'] and r['posterior_defined'] and r['proposal_available'] for r in result_rows)
  (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 finally:store.connection.close()

if __name__=='__main__':main()
