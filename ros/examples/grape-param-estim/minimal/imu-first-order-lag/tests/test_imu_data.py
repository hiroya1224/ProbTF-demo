from pathlib import Path
import copy
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import imu_data


def test_unbiased_noise_is_isotropic_not_covariance_of_the_mean():
    samples = np.array([[1., 2., 4.], [3., 1., 5.], [-1., 4., 7.], [0., -2., 6.]])
    before = samples.copy()
    used, diagnostic = imu_data.isotropic_noise(samples)
    expected = np.cov(samples, rowvar=False, ddof=1)
    np.testing.assert_array_equal(diagnostic, expected)
    np.testing.assert_array_equal(used, np.eye(3) * (np.trace(expected) / 3))
    np.testing.assert_array_equal(samples, before)
    assert used[0, 0] != np.trace(expected) / (3 * len(samples))


def test_initial_alignment_uses_only_one_pose_bracket_and_strict_past_velocity():
    time = np.arange(0., 2.01, .2)
    position = np.column_stack([time, time**2, .01 * time**3])
    quaternion = Rotation.from_euler("z", .1 * time).as_quat()
    source = np.arange(30, 30 + len(time))
    initial = imu_data.initial_conditions(time, position, quaternion, source, .9)
    assert np.all(initial["initial_velocity_source_time"] < .9)
    np.testing.assert_array_equal(initial["initial_mocap_bracket_source_indices"], [34, 35])
    changed_position, changed_quaternion = position.copy(), quaternion.copy()
    changed_position[time > 1.] = 1000
    changed_quaternion[time > 1.] = Rotation.from_euler("x", 2.).as_quat()
    changed = imu_data.initial_conditions(time, changed_position, changed_quaternion, source, .9)
    for key in initial:
        np.testing.assert_array_equal(changed[key], initial[key])


def row(stamp, index, **fields):
    return {"stamp_ns": round(stamp * 1e9), "record_ns": round(stamp * 1e9),
            "source_index": index, "selected_record_index": index, **fields}


def synthetic_streams():
    native_time = np.arange(.1, 2., .2)
    native = [row(t, i, gyro=[.01 * i, .02 * (-1)**i, .001 * i],
                  accel=[.01 * i, .02 * (-1)**i, 9.8 + .01 * i],
                  mag=[20. + .01 * i, .02 * (-1)**i, 40.])
              for i, t in enumerate(native_time)]
    pose_time = np.arange(0., 2.41, .2)
    pose = [row(t, i, position=[t, .01 * t**2, .2],
                quaternion=Rotation.from_euler("z", .1 * t).as_quat().tolist(), frame="world")
            for i, t in enumerate(pose_time)]
    command_time = np.arange(0., 2.41, .2)
    streams = {"imu": native, "mocap": pose,
               "state": [row(0., 0, state=0), row(1., 1, state=1)],
               "converted": [row(.1, 0, frame="gimbalrotor/fc")],
               "static": [row(0., 0, transforms=[("gimbalrotor/main_body", "gimbalrotor/fc",
                                                   imu_data.SENSOR_POSITION.tolist(), [0., 0., 0., 1.])])],
               "pid": [row(t, i) for i, t in enumerate(command_time)]}
    for topic in ["rotor", "joint", "gimbal_command"]:
        streams[topic] = [row(t, i, value=[1., 2., 3., 4.]) for i, t in enumerate(command_time)]
    return streams


def test_future_mocap_changes_evaluation_only_not_observations_calibration_or_initial_state():
    streams = synthetic_streams()
    config = {"start_seconds": 1., "end_seconds": 2.}
    original = imu_data.build_case_data("synthetic", config, streams, 0, {})
    modified = copy.deepcopy(streams)
    for item in modified["mocap"]:
        if item["stamp_ns"] > 1.2e9:
            item["position"] = [100., 200., 300.]
            item["quaternion"] = Rotation.from_euler("y", 2.).as_quat().tolist()
    changed = imu_data.build_case_data("synthetic", config, modified, 0, {})
    assert not np.array_equal(original["mocap_position"], changed["mocap_position"])
    assert not np.array_equal(original["mocap_quaternion_xyzw"], changed["mocap_quaternion_xyzw"])
    for key in original:
        if key == "metadata" or key.startswith("mocap_"):
            continue
        np.testing.assert_array_equal(changed[key], original[key], err_msg=key)
    assert np.all(original["initial_velocity_source_time"] < original["time"][0])
    assert original["metadata"]["magnetometer"]["onboard_fused_quaternion_used"] is False
    assert original["rotor_command_time"][0] < original["time"][0]


def test_magnetic_hold_is_not_a_new_independent_observation():
    vectors = np.array([[1., 2., 3.], [1., 2., 3.], [2., 2., 3.], [2., 2., 3.]])
    np.testing.assert_array_equal(imu_data.changed_magnetometer_mask(vectors), [True, False, True, False])
