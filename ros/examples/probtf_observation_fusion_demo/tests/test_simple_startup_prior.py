"""Publish the unobserved rigid-model law before physical reaching starts."""

from pathlib import Path

import numpy as np
import pytest

from probtf.geometry import quat_to_rotmat
from probtf.spherical_law.first_order import point_query, pose_query
from probtf_observation_fusion_demo.async_publication import control_key
from probtf_observation_fusion_demo.node import build_snapshot
from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore
from probtf_observation_fusion_demo.simple_motion import SimpleMotionSession
from test_async_execution import Context
from test_initial_prior import _paths, _rigid_poses


CONFIG = Path(__file__).resolve().parents[1]/'config'
EPOCH = 1792080000.


def initial_display(request_id=1):
    return dict(kind='display', request_id=request_id, event_id='joint:initial',
        capture_index=0, index=0, snapshot=None,
        diagnostics={'original_observation_stamp': EPOCH})


def test_headless_start_holds_initial_display_for_three_seconds_without_resetting_timer(tmp_path):
    clock = [100.]
    session = SimpleMotionSession(CONFIG/'imu_marker.yaml', tmp_path/'headless_startup',
        context=Context(), clock=lambda: clock[0], duration=10., body_surprise_enabled=False)
    try:
        original_command = session.command.copy()
        assert session.startup_prior_hold_s == pytest.approx(3.)
        assert session.initial_prior_display_wall is None
        assert session.initial_prior_published_wall is None
        assert not session.initial_prior_publication_required
        assert not session.start()

        session.control_replies.put(initial_display())
        session.poll()
        assert session.initial_prior_display_wall == 100.
        assert not session.start()
        clock[0] = 101.
        session.control_replies.put(initial_display(request_id=2))
        session.poll()
        assert session.initial_prior_display_wall == 100.
        assert not session.start()

        clock[0] = 102.999
        session.advance()
        assert not session.start()
        assert session.started is None
        assert session.frame.time == 0.
        np.testing.assert_array_equal(session.command, original_command)
        assert not session.actions

        clock[0] = 103.
        assert session.start()
        assert session.started == 103.
        assert not session.start(), 'Startup is a single transition.'
        clock[0] = 103.2
        session.advance()
        assert session.frame.time == pytest.approx(.2)
    finally:
        session.close()


def test_ros_start_requires_matching_published_prior_ack_and_holds_from_that_ack(tmp_path):
    clock = [100.]
    session = SimpleMotionSession(CONFIG/'imu_marker.yaml', tmp_path/'ros_startup',
        context=Context(), clock=lambda: clock[0], duration=10., body_surprise_enabled=False)
    try:
        original_command = session.command.copy()
        session.require_initial_prior_publication()
        assert session.initial_prior_publication_required
        display = initial_display()
        correct_key = control_key(display)
        session.notify_initial_prior_published(correct_key)
        assert session.initial_prior_published_wall is None
        session.control_replies.put(display)
        session.poll()
        assert session.initial_prior_display_wall == 100.

        clock[0] = 110.
        assert not session.start(), 'Wall time before ROS publication cannot satisfy the hold.'
        session.notify_initial_prior_published((999,)+correct_key[1:])
        assert session.initial_prior_published_wall is None
        assert not session.start()
        assert session.frame.time == 0.
        np.testing.assert_array_equal(session.command, original_command)

        clock[0] = 111.
        session.notify_initial_prior_published(correct_key)
        assert session.initial_prior_published_wall == 111.
        clock[0] = 112.
        session.notify_initial_prior_published(correct_key)
        assert session.initial_prior_published_wall == 111., 'Repeated publication must not restart the hold.'
        clock[0] = 113.999
        assert not session.start()
        assert session.started is None
        assert session.frame.time == 0.
        assert not session.actions
        clock[0] = 114.
        assert session.start()
        assert session.started == 114.
    finally:
        session.close()


def test_startup_joint_notification_preserves_direct_rigid_prior_and_next_observation_source(tmp_path):
    session = SimpleMotionSession(CONFIG/'imu_marker.yaml', tmp_path/'law_startup',
        context=Context(), duration=10., body_surprise_enabled=False)
    try:
        inference = session.inference_description
        command = session.command.copy()
        direct = inference.prior_at(command, EPOCH, 0)
        with SensorEventStore(tmp_path/'startup_events.sqlite3') as store:
            notification = store.append('joint:initial', 'joint', EPOCH, command,
                {'frame_index': 0, 'time': 0.})
            initial = inference.update_sensor_event(notification, event_index=0)
            current = initial.stages[-1].components[0].state
            assert current is inference.components[0].state
            assert current.state_id == direct.state_id
            np.testing.assert_array_equal(current.mean, direct.mean)
            np.testing.assert_array_equal(current.covariance, direct.covariance)
            for edge, prior_edge in zip(current.transforms, direct.transforms):
                assert edge.edge_id == prior_edge.edge_id
                np.testing.assert_array_equal(edge.translation, prior_edge.translation)
                np.testing.assert_array_equal(edge.rotation_wxyz, prior_edge.rotation_wxyz)
                if edge.bingham is not None:
                    np.testing.assert_array_equal(edge.bingham, prior_edge.bingham)
            assert initial.records['observation_ids'] == []
            assert not initial.records['simple_inference']['goal_observed']
            assert not initial.records['simple_inference']['physical_truth_received']
            snapshot = build_snapshot(initial, initial.stages[-1], inference.common)
            assert {point.query.state_id for point in snapshot.points} == {current.state_id}
            assert snapshot.stamp == EPOCH
            assert tuple(snapshot.command) == tuple(command)

            rigid = _rigid_poses(session.scene.urdf, command, _paths(inference))
            for name, path in _paths(inference).items():
                point, pose = point_query(current, path), pose_query(current, path)
                np.testing.assert_allclose(point.mode, rigid[name].translation, atol=1e-12, rtol=0.)
                np.testing.assert_allclose(quat_to_rotmat(pose.rotation_wxyz),
                    rigid[name].rotation, atol=1e-12, rtol=0.)
                assert point.state_id == current.state_id
            # Core's propagated expectation is retained even when it differs
            # from a modal pose. No display-only recentering is performed.
            hand = point_query(current, inference.paths['hand'])
            np.testing.assert_array_equal(hand.mean, hand.mode+hand.correction)
            assert np.linalg.norm(hand.mean-hand.mode) > .001
            assert np.linalg.norm(session.frame.tip-hand.mode) > .001

            # The physical initial sag supplies only the existing IMU
            # observation, after the untouched rigid-model prior is retained.
            imu = store.append('imu:initial', 'imu', EPOCH, command,
                dict(time=0., frame_index=0,
                    gravity_in_sensor=session.frame.gravity_in_sensor,
                    imu_target_link=session.scene.imu_target_link,
                    imu_sigma_rad=session.scene.config['imu']['direction_sigma_rad']))
            updated = inference.update_sensor_event(imu, event_index=0)
            assert updated.records['prediction'][0]['source_state_id'] == current.state_id
            assert updated.records['observation_ids']
            assert updated.records['simple_inference']['physical_truth_received'] is False
    finally:
        session.close()
