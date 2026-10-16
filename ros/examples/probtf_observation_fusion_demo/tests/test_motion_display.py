"""Physical references use their recorded frame, including on replay hold."""

from types import SimpleNamespace
from copy import deepcopy
import io
from pathlib import Path

import cv2
import numpy as np
import pytest
import rospy
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import Marker, MarkerArray

from probtf_observation_fusion_demo.motion_display import MotionDisplay
from probtf_observation_fusion_demo.model import load_inputs
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.node import build_snapshot


class Capture:
    def __init__(self):
        self.messages = None

    def sendTransform(self, messages):
        self.messages = messages

    def publish(self, message):
        self.messages = message


def _pose(message):
    result = np.eye(4)
    value = message.transform
    result[:3, 3] = [value.translation.x, value.translation.y, value.translation.z]
    result[:3, :3] = Rotation.from_quat([
        value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w]).as_matrix()
    return result


def test_replay_robot_tf_and_physical_path_do_not_use_future_frame(tmp_path, monkeypatch):
    scene = MovingSwitchScene(tmp_path)
    first = scene.step(0.0, 1792080000.0)
    second = scene.step(1.0, 1792080001.0, command=scene.initial_command + .02)
    # Preparing the whole movie leaves FK at a later physical state.  Replaying
    # the first frame must still draw its original physical tip and no future path.
    scene.monitor.reached = True
    display = MotionDisplay.__new__(MotionDisplay)
    display.scene, display.case, display.sensor_names = scene, "no_sensors", ()
    display.tag_texture = cv2.imread(str(scene.texture_path), cv2.IMREAD_GRAYSCALE)
    display.transforms, display.publisher = Capture(), Capture()
    display.marker_id = 0
    monkeypatch.setattr(rospy.Time, "now", staticmethod(lambda: rospy.Time.from_sec(1234.5)))
    snapshot = SimpleNamespace(observation_ids=(), stamp=first.stamp)
    display.publish(first, 0, snapshot, 0)

    sent = {message.child_frame_id: message for message in display.transforms.messages}
    for prefix, poses in (("", first.link_poses), ("rigid/", first.rigid_link_poses)):
        for name, expected in poses.items():
            if name == "base_link":
                continue
            message = sent[prefix + name]
            assert message.header.frame_id == "base_link"
            np.testing.assert_allclose(_pose(message), expected, atol=1e-14)
    assert not np.allclose(first.link_poses["rod_tip"], second.link_poses["rod_tip"])
    assert not np.allclose(first.link_poses["rod_tip"], first.rigid_link_poses["rod_tip"])
    markers = display.publisher.messages.markers
    assert not any(marker.ns == "actual_path" for marker in markers)
    actual_tip = next(marker for marker in markers if marker.ns == "actual_tip")
    np.testing.assert_allclose([actual_tip.pose.position.x, actual_tip.pose.position.y,
                               actual_tip.pose.position.z], first.tip, atol=1e-14)
    # The final monitor cannot turn the lamp green in an earlier movie frame.
    lamp = next(marker for marker in markers if marker.ns == "scene" and marker.id == 2)
    assert lamp.color.g < 0.2
    assert not any(marker.ns.startswith(("gaussian/", "axes/")) for marker in markers)

    display.publish(second, 1, SimpleNamespace(observation_ids=(), stamp=second.stamp), 1)
    path = next(marker for marker in display.publisher.messages.markers if marker.ns == "actual_path")
    np.testing.assert_allclose([[point.x, point.y, point.z] for point in path.points],
                               [first.tip, second.tip], atol=1e-14)


@pytest.fixture(scope='module')
def stiffness_display_input(tmp_path_factory):
    scene = MovingSwitchScene(tmp_path_factory.mktemp('stiffness_display'))
    frame = scene.step(0.0, 1792080000.0)
    config = Path(__file__).resolve().parents[1]/'config'
    _, observations = load_inputs(config/'common.yaml', config/'imu_marker.yaml')
    inference = MovingInference(scene.robot_description, observations)
    result = inference.update(scene.sensor_inputs(frame), 0)
    stage = result.stages[-1]
    snapshot = build_snapshot(result, stage, inference.common)
    # Explicit test data for a display contract, not an added estimation prior.
    prior_k = np.array([140., 130., 120., 110., 100., 90.])
    posterior_k = np.array([82., 93., 104., 115., 126., 137.])
    covariance = np.diag(np.arange(1., 7.)**2/10000.)
    covariance[0, 1] = covariance[1, 0] = .00005
    record = {
        'source_stamp': result.prior.stamp,
        'source_command': list(result.prior.command),
        'source_prior_state_id': result.prior.state_id,
        'source_prior_version': result.prior.version,
        'body_posterior_state_ids': [item.state.state_id for item in stage.components],
        'body_posterior_versions': [item.state.version for item in stage.components],
        'body_posterior_weights': [item.weight for item in stage.components],
        'body_posterior_labels': [item.label for item in stage.components],
        'observation_ids': list(snapshot.observation_ids),
        'prior': {'x_est': np.log(prior_k).tolist(), 'P_est': (.02*np.eye(6)).tolist(),
                  'K_median': prior_k.tolist()},
        'posterior': {'x_est': np.log(posterior_k).tolist(), 'P_est': covariance.tolist(),
                      'K_median': posterior_k.tolist()},
        'status': 'test update', 'information_rank': 3,
    }
    return scene, frame, result, snapshot, record


def test_stiffness_hud_uses_same_raw_snapshot_and_updater_values_on_actual_ros_wire(stiffness_display_input, monkeypatch):
    scene, frame, result, snapshot, record = stiffness_display_input
    original = deepcopy(record)
    display = MotionDisplay.__new__(MotionDisplay)
    display.scene, display.case, display.sensor_names = scene, 'imu_marker', ('imu', 'marker')
    display.tag_texture = cv2.imread(str(scene.texture_path), cv2.IMREAD_GRAYSCALE)
    display.transforms, display.publisher = Capture(), Capture()
    display.marker_id = 0
    monkeypatch.setattr(rospy.Time, 'now', staticmethod(lambda: rospy.Time.from_sec(8000.)))
    display.publish(frame, 0, snapshot, 0, stiffness_record=record)
    buffer = io.BytesIO()
    display.publisher.messages.serialize(buffer)
    wire = MarkerArray().deserialize(buffer.getvalue())
    hud = next(marker for marker in wire.markers if marker.ns == 'estimated_stiffness')
    assert hud.type == Marker.TEXT_VIEW_FACING
    assert hud.header.frame_id == snapshot.frame_id
    assert hud.header.stamp == rospy.Time.from_sec(snapshot.stamp)
    assert 'Prior K median [Nm/rad]: 140, 130, 120, 110, 100, 90' in hud.text
    assert 'Posterior K median: 82, 93, 104, 115, 126, 137' in hud.text
    assert 'sigma(log K): 0.01, 0.02, 0.03, 0.04, 0.05, 0.06' in hud.text
    assert 'Information rank 3/6 | test update' in hud.text
    assert record == original
    assert record['source_prior_state_id'] == result.prior.state_id
    assert record['body_posterior_state_ids'] == [item.state.state_id for item in result.stages[-1].components]
    # Physical stiffness is never consulted when rendering these numbers.
    saved = scene.plant.stiffness.copy()
    try:
        scene.plant.stiffness[:] = 1e9
        actual = MotionDisplay.stiffness_marker(snapshot, record)
        assert actual.text == hud.text
    finally:
        scene.plant.stiffness[:] = saved


@pytest.mark.parametrize('field', [
    'source_stamp', 'source_command', 'body_posterior_state_ids',
    'body_posterior_versions', 'body_posterior_weights', 'body_posterior_labels',
    'observation_ids',
])
def test_stiffness_hud_rejects_mismatched_source_before_publishing(stiffness_display_input, field):
    _, _, _, snapshot, record = stiffness_display_input
    changed = deepcopy(record)
    if field == 'source_stamp':
        changed[field] += .05
    elif field == 'source_command':
        changed[field][0] += .01
    elif field == 'body_posterior_versions':
        changed[field][0] += 1
    elif field == 'body_posterior_weights':
        changed[field][0] *= .5
    else:
        changed[field][0] += ':unrelated'
    with pytest.raises(AssertionError):
        MotionDisplay.stiffness_marker(snapshot, changed)


def test_stiffness_hud_retains_all_body_mixture_components(stiffness_display_input):
    _, _, _, snapshot, record = stiffness_display_input
    assert len(record['body_posterior_state_ids']) >= 2
    changed = deepcopy(record)
    for key in ('body_posterior_labels', 'body_posterior_state_ids',
                'body_posterior_versions', 'body_posterior_weights'):
        changed[key].pop()
    with pytest.raises(AssertionError, match='lost a displayed body mixture component'):
        MotionDisplay.stiffness_marker(snapshot, changed)


def test_stiffness_hud_observation_order_is_permutation_invariant(stiffness_display_input):
    _, _, _, snapshot, record = stiffness_display_input
    expected = MotionDisplay.stiffness_marker(snapshot, record)
    reordered = deepcopy(record)
    reordered['observation_ids'].reverse()
    actual = MotionDisplay.stiffness_marker(snapshot, reordered)
    assert actual.text == expected.text
    # Equality of sorted IDs still checks their multiplicities.
    duplicated = deepcopy(record)
    duplicated['observation_ids'].append(duplicated['observation_ids'][0])
    with pytest.raises(AssertionError):
        MotionDisplay.stiffness_marker(snapshot, duplicated)


def test_stiffness_source_must_match_the_actual_frame_command_time(stiffness_display_input):
    _, frame, _, snapshot, record = stiffness_display_input
    display = MotionDisplay.__new__(MotionDisplay)
    display.transforms, display.publisher = Capture(), Capture()
    stale = SimpleNamespace(stamp=frame.stamp+.05, command=frame.command)
    with pytest.raises(AssertionError, match='different command times'):
        display.publish(stale, 0, snapshot, 0, stiffness_record=record)
    assert display.transforms.messages is None
    assert display.publisher.messages is None


def _failed_stiffness_record(record):
    failed = deepcopy(record)
    failed.update(status='failed_support_or_laplace', update_applied=False,
        posterior=None, retained_prior=failed['prior'], prior_retained=True,
        information_rank=None, information_eigenvalues=None,
        failure_reason='No justified stationary Laplace mode; '+('full reason retained in source record. '*8))
    return failed


def test_failed_stiffness_hud_discloses_exact_prior_and_unavailable_information_on_ros_wire(stiffness_display_input):
    _, _, _, snapshot, record = stiffness_display_input
    failed = _failed_stiffness_record(record)
    original = deepcopy(failed)
    marker = MotionDisplay.stiffness_marker(snapshot, failed)
    stream = io.BytesIO()
    MarkerArray([marker]).serialize(stream)
    actual = MarkerArray().deserialize(stream.getvalue()).markers[0]
    assert actual.header.stamp == rospy.Time.from_sec(snapshot.stamp)
    assert actual.header.frame_id == snapshot.frame_id
    assert actual.text.startswith('K UPDATE UNAVAILABLE; prior retained\n')
    assert 'Retained prior K median [Nm/rad]: 140, 130, 120, 110, 100, 90' in actual.text
    assert 'Retained sigma(log K): 0.1414, 0.1414, 0.1414, 0.1414, 0.1414, 0.1414' in actual.text
    assert 'Information rank: unavailable' in actual.text
    assert 'Posterior' not in actual.text and 'Information rank 0' not in actual.text
    assert 'Reason: No justified stationary Laplace mode;' in actual.text
    assert failed['failure_reason'] not in actual.text
    assert failed == original  # The complete failure evidence remains intact.
    changed_source = deepcopy(failed)
    changed_source['source_command'][0] += .001
    with pytest.raises(AssertionError):
        MotionDisplay.stiffness_marker(snapshot, changed_source)


@pytest.mark.parametrize('change', ['missing_retention', 'different_prior', 'fake_posterior',
                                  'fake_information_rank', 'fake_information_eigenvalues', 'fake_applied'])
def test_failed_stiffness_hud_rejects_fabricated_posterior_or_information(stiffness_display_input, change):
    _, _, _, snapshot, record = stiffness_display_input
    failed = _failed_stiffness_record(record)
    if change == 'missing_retention':
        failed.pop('prior_retained')
    elif change == 'different_prior':
        failed['retained_prior'] = deepcopy(failed['prior'])
        failed['retained_prior']['P_est'][0][0] += .001
    elif change == 'fake_posterior':
        failed['posterior'] = failed['prior']
    elif change == 'fake_information_rank':
        failed['information_rank'] = 0
    elif change == 'fake_information_eigenvalues':
        failed['information_eigenvalues'] = [0.]*6
    else:
        failed['update_applied'] = True
    with pytest.raises(ValueError, match='failed stiffness update'):
        MotionDisplay.stiffness_marker(snapshot, failed)


def test_density_factor_hud_keeps_continuous_memory_separate_from_numerical_proposal(stiffness_display_input):
    from copy import deepcopy
    _, _, _, snapshot, source = stiffness_display_input
    record = deepcopy(source)
    initial = dict(record['prior'])
    prior = dict(density_id='prior_density_000000000000', factor_ids=[],
        initial_components=[dict(label='explicit_prior', weight=1., log_weight=0.,
                                 x_est=initial['x_est'], P_est=initial['P_est'])],
        normalization_constant=None, global_moments=None)
    posterior = dict(prior, density_id='posterior_density_000000000', factor_ids=['actual_frozen_ratio'])
    record.update(prior=prior, posterior=posterior, status='updated_density_factor',
        proposal_approximation=None, proposal_status='deterministic_density_candidate',
        command_proposal=dict(selected_K=[1., 2., 3., 4., 5., 6.]),
        information_rank=None, information_eigenvalues=None)
    source_logs = {}
    for point in snapshot.points:
        source_logs[point.component_id] = point.log_weight
    record['body_posterior_log_weights'] = list(source_logs.values())
    marker = MotionDisplay.stiffness_marker(snapshot, record)
    stream = io.BytesIO()
    MarkerArray([marker]).serialize(stream)
    actual = MarkerArray().deserialize(stream.getvalue()).markers[0]
    assert 'Factors: 0 -> 1' in actual.text
    assert 'normalization unevaluated' in actual.text
    assert 'Numerical proposal K [Nm/rad]: 1, 2, 3, 4, 5, 6' in actual.text
    assert 'Proposal only; global moments not evaluated' in actual.text
    assert actual.header.stamp.to_sec() == snapshot.stamp
    assert record['posterior']['global_moments'] is None
    record['body_posterior_log_weights'][0] -= 1.
    with pytest.raises(AssertionError, match='log masses'):
        MotionDisplay.stiffness_marker(snapshot, record)


def test_density_factor_hud_does_not_invent_parameters_when_proposal_is_unavailable(stiffness_display_input):
    from copy import deepcopy
    _, _, _, snapshot, source = stiffness_display_input
    record = deepcopy(source)
    density = dict(density_id='actual_density', factor_ids=['retained_ratio'],
                   normalization_constant=None, global_moments=None)
    record.update(prior=density, posterior=density, status='identity_density_ratio',
                  proposal_approximation=None, proposal_status='unavailable',
                  proposal_failure_reason='No supported auxiliary proposal at this time.')
    marker = MotionDisplay.stiffness_marker(snapshot, record)
    assert 'Factors: 1 -> 1' in marker.text
    assert 'No numerical proposal; global moments not evaluated' in marker.text
    assert record['proposal_failure_reason'] in marker.text
    assert 'median' not in marker.text
