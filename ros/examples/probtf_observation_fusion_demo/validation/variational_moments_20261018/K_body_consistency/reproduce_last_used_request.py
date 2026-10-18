"""Replay only archived law plus actual action ledger; no new inference/K solve."""
import argparse
import hashlib
import json
from pathlib import Path

from probtf_observation_fusion_demo.async_artifacts import (
    capture_candidate_identity, identity_hash, load_stiffness_job,
)
from probtf_observation_fusion_demo.async_state import ActionStep, predict_result
from probtf_observation_fusion_demo.stiffness_body_consistency import KBodyConsistencyEvaluator
from probtf_observation_fusion_demo import stiffness_body_consistency



def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run(case, base, source_dir, output_directory):
    directory = base/case
    source_path = source_dir/(case+'_control_source.json')
    source = json.loads(source_path.read_text())
    identity = source['candidate_identity']
    prediction = source['prediction']
    command_path = prediction['command_path']
    assert len(source['matching_job_references']) == 1
    reference = dict(source['matching_job_references'][0])
    if not Path(reference['path']).exists():
        reference['path'] = str(directory/'stiffness_jobs'/Path(reference['path']).name)
    archived = load_stiffness_job(reference)
    original = archived['result']
    assert original.prior.state_id == command_path['source_prior_state_id']
    assert original.prior.stamp == command_path['source_stamp']
    assert list(original.prior.command) == command_path['source_command']
    ledger = command_path['ledger_reference']
    start, end = ledger['start_index'], ledger['end_index_exclusive']
    steps = []
    chain = hashlib.sha256(b'ProbTF actual-command ledger v1').hexdigest()
    source_chain = chain if start == 0 else None
    for line in (directory/ledger['path']).open():
        item = json.loads(line)['step']
        if item['index'] >= end:
            break
        step = ActionStep(**item)
        encoded = json.dumps(step.record(),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        chain = hashlib.sha256(bytes.fromhex(chain)+encoded).hexdigest()
        if item['index']+1 == start:
            source_chain = chain
        if start <= item['index'] < end:
            steps.append(step)
    assert len(steps) == ledger['step_count']
    assert chain == ledger['prefix_chain_sha256']
    assert source_chain == ledger['source_prefix_chain_sha256']
    axes = command_path['axes']
    body_ids = tuple(item['edge_id'] for item in axes)
    actions = tuple(dict(joint_index=item['joint_index'], axis=item['axis']) for item in axes)
    predicted = predict_result(original, steps, target_stamp=prediction['present_stamp'],
        frame_index=prediction['frame_index'], body_ids=body_ids, actions=actions,
        ledger_reference=ledger)
    actual = capture_candidate_identity(predicted, parameter_identity=identity['parameter'])
    assert actual == identity, 'Reconstructed candidate law/source differs; no diagnostic is reported.'
    assert identity_hash(actual) == identity_hash(identity)
    adopted = source['adopted_packet']
    assert adopted['source_stamp'] == actual['stamp']
    assert adopted['source_command'] == actual['command']
    assert adopted['body_state_ids'] == [item['state_id'] for item in actual['body']]
    assert adopted['goal_state_ids'] == [item['state_id'] for item in actual['goals']]
    assert adopted['selected_K'] == actual['parameter']['selected_K']
    assert adopted['parameter_density_id'] == actual['parameter']['density_id']
    evaluator = KBodyConsistencyEvaluator(directory/'robot.urdf')
    diagnostic = evaluator.evaluate(predicted, actual, request=source['request'])
    assert diagnostic['candidate_id'] == identity_hash(identity)
    output = dict(scope='last trajectory request whose gradient packet was used by the final physical command interval',
        physical_rerun=False, new_observation_update=False, new_K_solve=False,
        reconstruction=dict(full_candidate_identity_exact=True,
            archived_body_job_sequence=reference['job_sequence'],
            archived_body_job_path=reference['path'], archived_body_job_sha256=reference['payload_sha256'],
            source_control_extract_sha256=file_hash(source_path),
            actual_used_gradient_packet_id=adopted['packet_id'],
            actual_used_packet_source_identity_exact=True,
            original_source_stamp=original.prior.stamp,
            actual_ledger_start_index=start, actual_ledger_end_index_exclusive=end,
            actual_ledger_step_count=len(steps), ledger_reference=ledger,
            complete_actual_ledger_chain_verified=True,
            no_truth_input=True, no_new_equilibrium_solve=True,
            implementation_source_hashes={name:file_hash(Path(stiffness_body_consistency.__file__).parent/name)
                for name in ('stiffness_body_consistency.py','async_state.py','body_native.py')},
            run_implementation_is_before_this_new_diagnostic=True),
        diagnostic=diagnostic)
    (output_directory/(case+'_last_used_request_diagnostic.json')).write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(case, diagnostic['status'], json.dumps(diagnostic.get('summary')), 'K_age',diagnostic.get('K_source_age_s'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run', type=Path, default=Path('/tmp/probtf_variational_commonchart_180s_live_20261018'),
        help='Original complete run: robot.urdf, commands.jsonl and immutable stiffness_jobs are required.')
    parser.add_argument('--source-dir', type=Path, default=Path(__file__).resolve().parent,
        help='Directory containing the two selected original control-source extracts.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=('marker','imu_marker'), default=('marker','imu_marker'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        run(case, args.input_run, args.source_dir, args.output)
