#!/usr/bin/env python3
"""Read-only saved core-query -> wire audit; run only after demo motion ends."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def lines(path):
    if path.exists():
        with path.open() as stream:
            for raw in stream:
                if raw.strip():
                    yield json.loads(raw)


def restore_snapshot(record):
    import numpy as np
    from probtf_observation_fusion_demo.display import DisplayPoint, DisplaySnapshot
    points=[]
    for item in record['queries']:
        query=SimpleNamespace(query_id=item['query_id'],component_id=item['source_component_id'],
            state_id=item['source_state_id'],frame=item['source_frame'],stamp=item['source_stamp'],
            version=item['source_version'],command=tuple(item['source_command']),
            approximation_order=item['approximation_order'],
            **{name:np.asarray(item[name],dtype=float) for name in ('mode','mean','covariance','within','between')})
        points.append(DisplayPoint(query,item['frame_label'],item['axis'],item['component_id'],
            item['weight'],item['sample_count'],item.get('log_weight')))
    return DisplaySnapshot(tuple(points),record['frame_id'],record['stamp'],tuple(record['command']),
        record['version'],record['stage'],tuple(record['observation_ids']),record['summary'],
        show_labels=record['show_labels'],stiffness_record=record.get('stiffness'),
        goal_memory=record.get('fixed_goal_memory'),behavior=record.get('behavior'),
        surprise_record=record.get('bayes_factor_surprise'),candidate_measure=record.get('candidate_measure'),
        sensor_event=record.get('sensor_event'))


def audit_run(run):
    import numpy as np
    from visualization_msgs.msg import MarkerArray
    from probtf_observation_fusion_demo.display import verify_wire_identity, snapshot_record
    from probtf_observation_fusion_demo.event_motion import _gradient_identity
    run=Path(run).resolve()
    manifest_path=run/'motion_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    adoptions={}
    for row in lines(run/'gradient_adoptions.jsonl'):
        packet=row.get('gradient_packet',row.get('packet',{}))
        adoptions[row.get('request_id')]=dict(accepted=row.get('accepted'),packet_id=packet.get('packet_id'),
            gradient=packet.get('gradient'),source_stamp=packet.get('source_stamp'),
            source_command=packet.get('source_command'),body_state_ids=packet.get('body_state_ids'),
            goal_state_ids=packet.get('goal_state_ids'),parameter_density_id=packet.get('parameter_density_id'),
            selected_K=packet.get('selected_K'),selection_factors_hash=canonical_hash(packet.get('selection_factors')))
    counts=dict(control_records=0,snapshots_verified=0,wire_markers_verified=0,wire_gaussian_markers_verified=0,
        wire_gaussian_points_verified=0,wire_axis_markers_verified=0,core_point_queries_verified=0,
        successful_selected_gradients=0,active_view_packets=0,unit_view_packets=0,
        accepted_packets_matched=0,rejected_packets_matched=0,waiting_control_records=0)
    errors=[]
    files_digest=hashlib.sha256()
    request_ids=set()
    for row in lines(run/'control.jsonl'):
        counts['control_records']+=1
        request_id=row.get('request_id')
        request_ids.add(request_id)
        prefix=run/row['artifact_prefix']
        try:
            saved_raw=prefix.with_suffix('.json').read_bytes()
            saved=json.loads(saved_raw)
            wire_raw=prefix.with_suffix('.markers.bin').read_bytes()
            files_digest.update(hashlib.sha256(saved_raw).digest())
            files_digest.update(hashlib.sha256(wire_raw).digest())
            snapshot=restore_snapshot(saved)
            actual=MarkerArray().deserialize(wire_raw)
            assert hashlib.sha256(wire_raw).hexdigest()==row['markers_wire_sha256']
            payload=verify_wire_identity(snapshot,actual,seed=saved['render_seed'],point_size=saved['point_size'])
            assert payload==wire_raw
            rebuilt=snapshot_record(snapshot)
            for key in ('queries','candidate_measure','goal_axes','goal_poses'):
                assert rebuilt.get(key)==saved.get(key),key+' roundtrip differs'
                assert row['snapshot'].get(key)==saved.get(key),key+' control sidecar differs'
            assert snapshot.command==tuple(row['snapshot']['command'])
            assert snapshot.stamp==row['snapshot']['stamp']
            counts['snapshots_verified']+=1
            counts['wire_markers_verified']+=len(actual.markers)
            for marker in actual.markers:
                if marker.ns.startswith('gaussian/'):
                    counts['wire_gaussian_markers_verified']+=1
                    counts['wire_gaussian_points_verified']+=len(marker.points)
                if marker.ns.startswith(('axes/','goal_axes/')):
                    counts['wire_axis_markers_verified']+=1
            counts['core_point_queries_verified']+=len(snapshot.points)
            for point in snapshot.points:
                assert point.query.stamp==snapshot.stamp
                assert point.query.command==snapshot.command
                assert point.query.version==snapshot.version
            evaluation=row.get('gradient_evaluation') or {}
            adoption=adoptions.get(request_id)
            if adoption:
                expected=hashlib.sha256(json.dumps(_gradient_identity(evaluation),sort_keys=True,
                    separators=(',',':'),allow_nan=False).encode()).hexdigest()
                assert adoption['packet_id']==expected,'adopted immutable packet identity differs'
                for field in ('gradient','source_stamp','source_command','body_state_ids','goal_state_ids',
                              'parameter_density_id','selected_K'):
                    assert adoption[field]==evaluation.get(field),'adopted packet field '+field+' differs'
                assert adoption['selection_factors_hash']==canonical_hash(evaluation.get('selection_factors'))
                counts['accepted_packets_matched' if adoption['accepted'] else 'rejected_packets_matched']+=1
            if evaluation.get('status')!='ok':
                counts['waiting_control_records']+=1
                continue
            counts['successful_selected_gradients']+=1
            assert evaluation['source_stamp']==snapshot.stamp
            assert tuple(evaluation['source_command'])==snapshot.command
            body_ids={p.query.state_id for p in snapshot.points if p.frame_label=='hand'}
            goal_ids={p.query.state_id for p in snapshot.points if p.frame_label=='switch_goal'}
            assert body_ids==set(evaluation['body_state_ids'])
            assert goal_ids==set(evaluation['goal_state_ids'])
            view=evaluation['visibility_factor']
            if view['status']=='unit_factor':
                counts['unit_view_packets']+=1
                assert view['log_factor']==0. and view['weighted_penalty']==0.
                np.testing.assert_array_equal(evaluation['selected_gradient'],evaluation['task_gradient'])
            else:
                counts['active_view_packets']+=1
                assert view['status']=='ok'
                ids=view['source_ids']
                for field in ('body_state_ids','goal_state_ids','source_command','source_stamp',
                              'parameter_density_id','selected_K','body_observation_source_stamp',
                              'goal_observation_source_stamp'):
                    assert ids[field]==evaluation[field],'task/view source field '+field+' differs'
                assert ids['evaluation_command']==evaluation['source_command']
                assert len(view['per_mode_record'])==len(evaluation['source_branches'])
                for mode,branch in zip(view['per_mode_record'],evaluation['source_branches']):
                    assert mode['source_state_id']==branch['state_id']
                    assert mode['source_stamp']==branch['query_stamp']==evaluation['source_stamp']
                    assert mode['source_command']==branch['query_command']==evaluation['source_command']
                    assert mode['source_version']==branch['query_version']
                np.testing.assert_array_equal(evaluation['selected_gradient'],
                    np.asarray(evaluation['task_gradient'])+view['weight']*np.asarray(evaluation['view_gradient']))
                assert evaluation['log_select_factor']==evaluation['log_task_factor']+view['log_factor']
        except Exception as error:
            errors.append(dict(request_id=request_id,artifact_prefix=row.get('artifact_prefix'),
                               exception_type=type(error).__name__,reason=str(error)))
    missing_adoptions=[key for key in adoptions if key not in request_ids]
    status='passed' if not errors and not missing_adoptions and counts['snapshots_verified']>0 else 'failed'
    return dict(run_directory=str(run),status=status,counts=counts,errors=errors,
        adoption_requests_without_saved_control_record=missing_adoptions,
        saved_snapshot_and_wire_ordered_digest_sha256=files_digest.hexdigest(),
        implementation_unchanged_during_execution=manifest.get('implementation_unchanged_during_execution'),
        implementation_sha256=manifest.get('implementation_sha256'),
        proof_scope='All saved point-query means/covariances/masses regenerate exact full MarkerArray bytes; task/view/packet numerical gradient and source identities cross-checked.',
        limitation='Point-query sidecars do not contain the complete joint transform density, so this audit does not independently re-run core propagation or re-estimate raw observations.',
        raw_estimation_or_sensor_payloads_read=False,read_only=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reports=[audit_run(path) for path in args.run_dirs]
    result=dict(schema='probtf_saved_candidate_wire_identity_v1',runs=reports,
        all_passed=all(row['status']=='passed' for row in reports))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),all_passed=result['all_passed'],
        runs=[dict(run=row['run_directory'],status=row['status'],counts=row['counts'],errors=row['errors'][:3])
              for row in reports]),indent=2))
    if not result['all_passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
