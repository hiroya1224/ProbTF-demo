"""The faster physical FK path preserves realized poses and exposure history."""
import numpy as np
import pinocchio as pin

from probtf_observation_fusion_demo.dynamic_physics import NativeDynamics
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene


def _independent_poses(robot, data, joints):
    pin.forwardKinematics(robot.model, data, joints)
    pin.updateFramePlacements(robot.model, data)
    base_inverse = np.linalg.inv(np.asarray(data.oMf[robot.base_fid].homogeneous))
    names = (['base_link'] + ['link_{}'.format(i+1) for i in range(6)]
             + ['tool0', 'press_rod', 'rod_tip', 'camera_optical_frame'])
    return {name: base_inverse @ np.asarray(data.oMf[robot.model.getFrameId(name)].homogeneous)
            for name in names}


def test_cached_frames_and_camera_only_fk_match_independent_fk_bitwise(tmp_path):
    scene = MovingSwitchScene(tmp_path)
    initial = scene.step(0., render_camera=False)
    native = NativeDynamics(scene.urdf, scene.config['plant']['stiffness'],
                            **scene.config['plant']['dynamics'])
    native.reset(initial.joints, command=initial.command)
    command = initial.command.copy(); command[3:5] += [.03, -.02]
    trace = native.advance(.025, initial.command, command)
    data = scene.plant.robot.model.createData()
    retained = None
    for joints in trace['positions']:
        independent = _independent_poses(scene.plant.robot, data, joints)
        complete = scene._link_poses(joints)
        camera = scene._camera_pose(joints)
        for name in independent:
            np.testing.assert_array_equal(complete[name], independent[name])
            assert not complete[name].flags.writeable
        np.testing.assert_array_equal(camera, complete['camera_optical_frame'])
        assert not camera.flags.writeable
        if retained is None:
            retained = camera
    # Later FK updates must not mutate a camera pose already sent to rendering.
    np.testing.assert_array_equal(retained,
        _independent_poses(scene.plant.robot, data, initial.joints)['camera_optical_frame'])
    native.close()


def test_exposure_keeps_each_accepted_physical_camera_endpoint(tmp_path, monkeypatch):
    scene = MovingSwitchScene(tmp_path)
    initial = scene.step(0., render_camera=False)
    captured = []
    advance = scene.dynamics.advance
    def record_advance(*args):
        path = advance(*args)
        captured.append(path)
        return path
    monkeypatch.setattr(scene.dynamics, 'advance', record_advance)
    command = initial.command.copy(); command[3:5] += [.03, -.02]
    frame = scene.step(.2, command=command, render_camera=False)
    path = captured[0]
    first = max(0, np.searchsorted(path['times'],
        frame.time-scene.config['camera']['exposure_time'], side='left')-1)
    expected_times = path['times'][first:]
    assert len(expected_times) > 32
    np.testing.assert_array_equal([time for time, _ in scene.camera_history], expected_times)
    data = scene.plant.robot.model.createData()
    for (_, camera), joints in zip(scene.camera_history, path['positions'][first:]):
        expected = _independent_poses(scene.plant.robot, data, joints)['camera_optical_frame']
        np.testing.assert_array_equal(camera, expected)
        assert not camera.flags.writeable
    np.testing.assert_array_equal(scene.camera_history[-1][1], frame.camera_pose)
    assert scene.config['camera']['motion_blur_samples'] == 7
    assert scene.config['camera']['exposure_time'] == .09
