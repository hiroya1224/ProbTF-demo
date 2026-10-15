"""Subscribe, trigger one owned demo run, and audit its native transform laws."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import time
import traceback

import numpy as np
import yaml

ROOT = Path('/home/leus/catkin_ws/src/ProbTF-demo')
PACKAGE = ROOT / 'ros/examples/probtf_switch_press_demo'
parser = argparse.ArgumentParser()
parser.add_argument('--run-dir', type=Path, required=True)
parser.add_argument('--report', type=Path, required=True)
parser.add_argument('--deadline', type=float, default=180.)
args = parser.parse_args()
os.environ.update(ROS_MASTER_URI='http://localhost:11764/', ROS_IP='127.0.0.1')
os.environ.pop('ROS_HOSTNAME', None)

import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from probtf_msgs.msg import ProbabilisticTransformArray, ProbabilisticTransformStamped
from probtf.candidates import load_initial_transform_law
from probtf.distributions import (
    BinghamOrientation, ConditionalGaussianTranslation, OrientationKind,
    TransformComponent, TransformDistribution, TransformDistributionStamped,
)
from probtf.geometry import quat_to_rotmat
from probtf_switch_press_demo.beliefs import BODY_FRAMES, left_compose_record
from probtf_switch_press_demo.kinematics import CandidateKinematics
from probtf_switch_press_demo.trial import record_parameters


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(type(value).__name__)


report = dict(passed=False, run_dir=str(args.run_dir), master_uri=os.environ['ROS_MASTER_URI'],
              audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              errors=[], capture_errors=[], status_changes=[])
roots, hands, goals, descriptors = {}, {}, {}, []
counts = dict(root_batches=0, hand_packets=0, goal_packets=0)
queue = SimpleQueue()


def stamp_ns(seconds):
    return rospy.Time.from_sec(float(seconds)).to_nsec()


def message_metadata(message):
    metadata = json.loads(message.provenance.detail)
    assert metadata['display_result_kind'] == 'frozen_law_query'
    assert metadata['temporal_prediction'] is False
    assert abs(metadata['source_observation_stamp']-message.header.stamp.to_sec()) <= 3e-7
    return metadata


def compact_support(message):
    metadata = message_metadata(message)
    assert message.header.frame_id == 'base_link'
    ids, weights, poses = [], [], []
    for component in message.components:
        assert component.orientation.kind == component.orientation.DIRAC
        assert not any(component.translation.residual_covariance_upper)
        assert not any(component.translation.rotation_coupling)
        q = component.orientation.reference_quaternion
        p = component.translation.mean_at_reference
        ids.append(component.component_id)
        weights.append(component.weight)
        poses.append([p.x, p.y, p.z, q.w, q.x, q.y, q.z])
    return dict(parent=message.header.frame_id, child=message.child_frame_id,
                source_stamp_ns=message.header.stamp.to_nsec(), metadata=metadata,
                derived_from_edge_ids=list(message.provenance.derived_from_edge_ids),
                ids=ids, weights=weights, poses=poses)


def compact_goal(message):
    metadata = message_metadata(message)
    from probtf_ros.v2_conversions import transform_distribution_from_msg
    # At most two native observation modes per source stamp; never retain
    # the full repeated ROS message history.
    record = transform_distribution_from_msg(message)
    return dict(source_stamp_ns=message.header.stamp.to_nsec(), metadata=metadata,
                parameters=record_parameters(record))


def drain():
    while True:
        try:
            kind, message = queue.get_nowait()
        except Empty:
            return
        try:
            if kind == 'status':
                if not report['status_changes'] or report['status_changes'][-1]['text'] != message.data:
                    report['status_changes'].append(dict(receipt=time.time(), text=message.data))
            elif kind == 'descriptor':
                value = json.loads(message.data)
                if not descriptors or descriptors[-1] != value:
                    descriptors.append(value)
            elif kind == 'root':
                counts['root_batches'] += 1
                if not message.transforms:
                    continue
                assert len(message.transforms) == len(BODY_FRAMES)
                records = [compact_support(item) for item in message.transforms]
                assert {item['child'] for item in records} == {'root_predicted/'+f for f in BODY_FRAMES}
                query = records[0]['metadata']['query_id']
                assert all(item['metadata']['query_id'] == query for item in records)
                item = roots.setdefault(query, dict(records=records, source_stamps_ns=[],
                                                    first_heartbeat_ns=message.header.stamp.to_nsec(),
                                                    last_heartbeat_ns=message.header.stamp.to_nsec(), receipts=0))
                item['last_heartbeat_ns'] = message.header.stamp.to_nsec()
                item['receipts'] += 1
                source_stamp = records[0]['source_stamp_ns']
                if source_stamp not in item['source_stamps_ns']:
                    item['source_stamps_ns'].append(source_stamp)
                for before, after in zip(item['records'], records):
                    assert before['child'] == after['child']
                    assert before['ids'] == after['ids']
                    np.testing.assert_allclose(before['weights'], after['weights'], rtol=0., atol=1e-15)
                    assert before['metadata']['source_law_id'] == after['metadata']['source_law_id']
                    assert before['metadata']['source_revision'] == after['metadata']['source_revision']
                    np.testing.assert_allclose(before['poses'], after['poses'], rtol=0., atol=1e-12)
            elif kind == 'hand':
                counts['hand_packets'] += 1
                record = compact_support(message)
                meta = record['metadata']
                key = meta['query_id']+'|'+meta['reference_query_id']
                variants = hands.setdefault(key, {})
                stamp = str(record['source_stamp_ns'])
                if stamp in variants:
                    assert variants[stamp]['ids'] == record['ids']
                    np.testing.assert_allclose(variants[stamp]['weights'], record['weights'], rtol=0., atol=1e-15)
                    np.testing.assert_allclose(variants[stamp]['poses'], record['poses'], rtol=0., atol=1e-12)
                else:
                    variants[stamp] = record
            elif kind == 'goal':
                counts['goal_packets'] += 1
                key = str(message.header.stamp.to_nsec())
                if key not in goals:
                    goals[key] = compact_goal(message)
                else:
                    message_metadata(message)
        except Exception:
            report['capture_errors'].append(dict(kind=kind, error=traceback.format_exc()))


def record_from_parameters(value):
    components = []
    for item in value['components']:
        orientation = BinghamOrientation(OrientationKind(item['orientation_kind']),
            item['inverse_concentration'], np.asarray(item['orientation_shape']), np.asarray(item['reference_wxyz']))
        components.append(TransformComponent(item['id'], item['weight'], orientation,
            ConditionalGaussianTranslation(np.asarray(item['mean_at_reference']),
                np.asarray(item['residual_covariance']), np.asarray(item['rotation_coupling']))))
    return TransformDistributionStamped(value['parent'], value['child'], value['stamp'],
        'archived_native_goal', 'offline_audit', TransformDistribution(tuple(components)))


def compare_support(actual, expected, ids, weights, source, expected_query, reference_query=None):
    assert actual['ids'] == list(ids)
    np.testing.assert_allclose(actual['weights'], weights, rtol=0., atol=1e-15)
    assert abs(sum(actual['weights'])-1.) < 1e-13
    meta = actual['metadata']
    assert meta['source_law_id'] == source.law_id and meta['source_revision'] == source.revision
    assert source.law_id in actual['derived_from_edge_ids']
    assert meta['query_id'] == expected_query
    assert meta['result_kind'] == source.query_metadata()['result_kind']
    assert meta['integration_method'] == source.query_metadata()['integration_method']
    if reference_query is not None:
        assert meta['reference_query_id'] == reference_query
    positions = np.asarray(actual['poses'])[:, :3]
    rotations = np.array([quat_to_rotmat(row[3:]) for row in actual['poses']])
    translation_error = float(np.max(np.abs(positions-expected[:, :3, 3])))
    rotation_error = float(np.max(np.abs(rotations-expected[:, :3, :3])))
    assert translation_error < 2e-10 and rotation_error < 2e-10
    return translation_error, rotation_error


def offline_audit():
    settings = yaml.safe_load((args.run_dir/'settings.yaml').read_text())
    summary = json.loads((args.run_dir/'summary.json').read_text())
    updates = json.loads((args.run_dir/'applied_ik_updates.json').read_text())
    cycles = json.loads((args.run_dir/'cycles.json').read_text())
    servo = json.loads((args.run_dir/'servo.json').read_text())
    archived = json.loads((args.run_dir/'initial_transform_law.json').read_text())['law']
    report['runtime_summary'] = summary
    path = Path(settings['control']['initial_transform_law'])
    if not path.is_absolute():
        path = PACKAGE/path
    source = load_initial_transform_law(path)
    assert json.loads(json.dumps(source.descriptor())) == archived['source_law']
    assert descriptors and all(value['source_law'] == archived['source_law'] for value in descriptors)
    candidates = CandidateKinematics(args.run_dir/'robot.urdf', source,
        settings['rod']['length'], source.reference_command)
    assert candidates.law.model_id == archived['model_id']
    assert all(value['model_id'] == archived['model_id'] for value in descriptors)
    expected_root, expected_hand, expected_goals = {}, {}, {}
    expected_stamps = {}
    specifications = list(updates)
    for cycle in cycles:
        if 'joint_measure_after' in cycle:
            specifications.append(dict(reference_command=cycle['command_before'], increment=cycle['increment'],
                source_stamp=cycle['source_stamp'], display_reference_hand_base=cycle['reference_hand_base'],
                joint_measure_before_query=cycle['joint_measure_before']['query'],
                joint_measure_after_query=cycle['joint_measure_after']['query'], goal_record=cycle['goal_record']))
    for item in specifications:
        prediction = candidates.predict(np.asarray(item['reference_command']), np.asarray(item['increment']), item['source_stamp'])
        after = prediction.joint_after.metadata['query_id']
        before = prediction.joint_before.metadata['query_id']
        assert after == item['joint_measure_after_query']['query_id']
        assert before == item['joint_measure_before_query']['query_id']
        expected_root[after] = prediction.joint_after.root_transforms
        source_stamp = stamp_ns(item['source_stamp'])
        expected_stamps.setdefault(after, set()).add(source_stamp)
        anchor = np.asarray(item['display_reference_hand_base'])
        expected_hand[(after+'|'+before, str(source_stamp))] = anchor @ prediction.relative_transforms[:, -1]
        expected_goals[str(source_stamp)] = left_compose_record(record_from_parameters(item['goal_record']), anchor, 'base_link')
        # Independently multiply every local tuple, retaining row correspondence.
        product = np.broadcast_to(np.eye(4), (len(candidates.ids), 4, 4)).copy()
        for index in range(len(BODY_FRAMES)):
            product = product @ prediction.joint_after.local_transforms[:, index]
            np.testing.assert_allclose(product, prediction.joint_after.root_transforms[:, index], rtol=0., atol=1e-13)
    maximum_translation, maximum_rotation = 0., 0.
    root_laws, hand_laws = 0, 0
    for query, item in roots.items():
        assert query in expected_root, 'captured root query absent from applied/final journal: '+query
        assert set(item['source_stamps_ns']) <= expected_stamps[query]
        for record in item['records']:
            index = BODY_FRAMES.index(record['child'].split('/', 1)[1])
            errors = compare_support(record, expected_root[query][:, index], candidates.ids,
                candidates.weights, source, query)
            maximum_translation = max(maximum_translation, errors[0])
            maximum_rotation = max(maximum_rotation, errors[1])
            root_laws += 1
    for key, variants in hands.items():
        for stamp, record in variants.items():
            assert (key, stamp) in expected_hand, 'captured hand query absent from applied/final journal'
            after, before = key.split('|')
            errors = compare_support(record, expected_hand[key, stamp], candidates.ids,
                candidates.weights, source, after, before)
            maximum_translation = max(maximum_translation, errors[0])
            maximum_rotation = max(maximum_rotation, errors[1])
            hand_laws += 1
    maximum_goal_error = 0.
    for stamp, item in goals.items():
        assert stamp in expected_goals, 'captured goal stamp absent from applied/final journal'
        expected = record_parameters(expected_goals[stamp])
        actual = item['parameters']
        assert actual['parent'] == expected['parent'] == 'base_link'
        assert stamp_ns(actual['stamp']) == stamp_ns(expected['stamp'])
        assert len(actual['components']) == len(expected['components'])
        for observed, wanted in zip(actual['components'], expected['components']):
            assert observed['id'] == wanted['id'] and observed['orientation_kind'] == wanted['orientation_kind']
            for field in ('weight', 'inverse_concentration', 'orientation_shape', 'mean_at_reference',
                          'residual_covariance', 'rotation_coupling'):
                error = float(np.max(np.abs(np.asarray(observed[field])-np.asarray(wanted[field]))))
                maximum_goal_error = max(maximum_goal_error, error)
                np.testing.assert_allclose(observed[field], wanted[field], rtol=2e-12, atol=2e-11)
            np.testing.assert_allclose(quat_to_rotmat(observed['reference_wxyz']),
                                       quat_to_rotmat(wanted['reference_wxyz']), rtol=0., atol=2e-12)
    assert roots and hands and goals and not report['capture_errors']
    mismatches = [name for name, digest in summary['source_sha256'].items()
                  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest]
    report['source_hash_mismatches'] = mismatches
    assert not mismatches
    moving = [tick for tick in servo if np.max(np.abs(tick['velocity'])) > 1e-6]
    moving_pending = [tick for tick in moving if tick['ik_pending']]
    first_received = min((cycle['received_time'] for cycle in cycles), default=float('inf'))
    first_solve_moving = [tick for tick in moving_pending if tick['time'] < first_received]
    report['audit'] = dict(captured_root_queries=len(roots), checked_root_laws=root_laws,
        checked_hand_laws=hand_laws, checked_goal_laws=len(goals), source_atom_count=len(candidates.ids),
        maximum_translation_error_m=maximum_translation, maximum_rotation_matrix_error=maximum_rotation,
        maximum_goal_parameter_error=maximum_goal_error, applied_ik_updates=len(updates),
        servo_ticks=len(servo), moving_ticks=len(moving), moving_ticks_while_ik_pending=len(moving_pending),
        moving_ticks_before_first_solve_returned=len(first_solve_moving),
        body_source_law_id=source.law_id, body_source_revision=source.revision,
        repeated_root_queries_with_new_heartbeat=sum(item['receipts']>1 and item['last_heartbeat_ns']>item['first_heartbeat_ns'] for item in roots.values()),
        source_and_render_times_distinct=any(item['last_heartbeat_ns'] not in item['source_stamps_ns'] for item in roots.values()))
    times = np.array([tick['time'] for tick in servo])
    report['audit']['maximum_servo_tick_interval_s'] = float(np.max(np.diff(times))) if len(times)>1 else None
    assert moving_pending and first_solve_moving, 'no measured continuous movement during the first IK solve'
    assert report['audit']['source_and_render_times_distinct']
    assert summary['reached'] and not summary['collided'], 'actual run did not reach collision-free'
    assert summary['final_error_m'] < settings['reach_tolerance']
    assert summary['minimum_clearance_m'] >= settings['motion']['clearance']


try:
    assert args.run_dir.is_dir() and not (args.run_dir/'summary.json').exists()
    rospy.init_node('native_transform_live_audit', anonymous=True, disable_signals=True)
    subscriptions = [rospy.Subscriber(topic, message_type, lambda message, k=kind: queue.put((k, message)), queue_size=100)
        for kind, topic, message_type in (
            ('root', '/switch_press/probtf_batch', ProbabilisticTransformArray),
            ('hand', '/switch_press/predicted_tip', ProbabilisticTransformStamped),
            ('goal', '/switch_press/predicted_goal', ProbabilisticTransformStamped),
            ('descriptor', '/switch_press/body_law', String), ('status', '/switch_press/status', String))]
    setup_deadline = time.monotonic()+30.
    while not descriptors and time.monotonic()<setup_deadline:
        drain()
        time.sleep(.05)
    assert descriptors, 'no latched governing law descriptor'
    rospy.wait_for_service('/switch_press/replay', timeout=10.)
    response = rospy.ServiceProxy('/switch_press/replay', Trigger)()
    assert response.success, response.message
    report['trigger'] = dict(time=time.time(), message=response.message)
    print('Triggered owned master 11764: '+str(args.run_dir), flush=True)
    deadline = time.monotonic()+args.deadline
    next_update = time.monotonic()+15.
    while not (args.run_dir/'camera_final_detection.png').exists() and time.monotonic()<deadline:
        drain()
        if time.monotonic()>=next_update:
            print(json.dumps(dict(counts=counts, root_queries=len(roots), hand_queries=len(hands),
                                  latest_status=report['status_changes'][-1] if report['status_changes'] else None)), flush=True)
            next_update += 15.
        time.sleep(.02)
    assert (args.run_dir/'camera_final_detection.png').exists(), 'trial exceeded audit deadline before final archives completed'
    finish = time.monotonic()+.5
    while time.monotonic()<finish:
        drain()
        time.sleep(.02)
    offline_audit()
    report['passed'] = True
except BaseException:
    report['errors'].append(traceback.format_exc())
finally:
    report['capture_counts'] = counts
    report['governing_descriptors'] = descriptors
    args.report.parent.mkdir(parents=True, exist_ok=True)
    capture_path = args.report.with_name(args.report.stem+'_captured_laws.json.gz')
    with gzip.open(capture_path, 'wt', encoding='utf-8') as handle:
        json.dump(dict(roots=roots, hands=hands, goals=goals), handle, default=native, allow_nan=False)
    report['compact_capture_archive'] = str(capture_path)
    args.report.write_text(json.dumps(report, default=native, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'], report=str(args.report), audit=report.get('audit'), errors=report['errors'])), flush=True)
if not report['passed']:
    raise SystemExit(1)
