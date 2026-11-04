#!/usr/bin/env python3
"""Exercise the real launch and native wire messages on a PRIVATE ROS master.

Run after building probtf_imu_demo and sourcing the workspace. No hardware is
used and the user's ROS master/topics are not touched. This verifies the
published law against a separate ROS-free replay, not only topic existence.
"""

import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
from xmlrpc.client import ServerProxy

import numpy as np


def wait_for(predicate, message, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except (OSError, ConnectionError):
            pass
        time.sleep(0.02)
    raise AssertionError(message)


def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def run():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    os.environ["ROS_MASTER_URI"] = "http://127.0.0.1:{}".format(port)
    os.environ["ROS_IP"] = "127.0.0.1"
    os.environ.pop("ROS_HOSTNAME", None)
    import rospy
    from sensor_msgs.msg import Imu
    from probtf_msgs.msg import ProbabilisticTransformStamped
    from probtf_ros.v2_conversions import transform_distribution_from_msg
    from probtf_estimators.imu_pair_pipeline import ImuRelativePosePipeline, imu_measurement_covariance
    from probtf.geometry import quat_to_rotmat, rpy_to_quat
    from probtf_estimators.imu_relative_pose import skew

    def vector(message, values):
        message.x, message.y, message.z = values

    pipeline = ImuRelativePosePipeline("parent", "child")
    messages, expected = [], {}
    rotation = quat_to_rotmat(rpy_to_quat(0.4, -0.3, 0.2))
    position = np.array([0.12, -0.06, 0.03])
    for index in range(100):
        for role, offset in (("parent", 0.0), ("child", 0.0006)):
            message = Imu()
            message.header.frame_id = role + ("__aabbcc" if role == "parent" else "__ddeeff")
            message.header.stamp = rospy.Time.from_sec(1800000000.0 + index * 0.002 + offset)
            stamp = message.header.stamp.to_sec()
            elapsed = stamp - 1800000000.0
            omega = np.array([0.4 + elapsed, 0.2 + elapsed ** 2, 0.7 - elapsed])
            alpha = np.array([1.0, 2 * elapsed, -1.0])
            force = np.array([0.1 * elapsed, 0.3, 9.81])
            if role == "child":
                force = rotation.T @ (force + (skew(omega) @ skew(omega) + skew(alpha)) @ position)
                omega = rotation.T @ omega
            vector(message.angular_velocity, omega)
            vector(message.linear_acceleration, force)
            # The actual Fire receiver uses unknown covariances (all zeros).
            message.orientation_covariance[0] = -1.0
            messages.append(message)
            record = pipeline.update(
                message.header.frame_id, stamp, omega, force,
                imu_measurement_covariance(message.angular_velocity_covariance, 1e-4),
                imu_measurement_covariance(message.linear_acceleration_covariance, 1e-3),
            )
            if record is not None:
                expected[record.stamp] = record

    received = {}
    master = launch = None
    with tempfile.TemporaryDirectory(prefix="probtf-imu-ros-") as directory:
        logs = Path(directory)
        os.environ["ROS_LOG_DIR"] = str(logs / "ros")
        try:
            with (logs / "master.log").open("w") as master_log, (logs / "launch.log").open("w") as launch_log:
                master = subprocess.Popen(["roscore", "-p", str(port)], stdout=master_log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                server = ServerProxy(os.environ["ROS_MASTER_URI"])
                wait_for(lambda: server.getUri("/probtf_imu_smoke")[0] == 1, "private ROS master did not start")
                rospy.init_node("probtf_imu_smoke", anonymous=True, disable_signals=True)

                def receive(message):
                    # Check actual generated-message serialization as well as
                    # the TCPROS transport and lossless native conversion.
                    wire = io.BytesIO()
                    message.serialize(wire)
                    restored = ProbabilisticTransformStamped().deserialize(wire.getvalue())
                    record = transform_distribution_from_msg(restored)
                    received[record.stamp] = record

                subscriber = rospy.Subscriber("/smoke/probtf", ProbabilisticTransformStamped, receive, queue_size=1000)
                publisher = rospy.Publisher("/smoke/imu", Imu, queue_size=1000)
                launch_environment = dict(os.environ)
                # Reproduce the workstation's old system NumPy on PYTHONPATH;
                # the venv node must still select its own numerical packages.
                launch_environment["PYTHONPATH"] = "/usr/lib/python3/dist-packages:" + os.environ.get("PYTHONPATH", "")
                launch = subprocess.Popen([
                    "roslaunch", "probtf_imu_demo", "imu_relative_pose.launch",
                    "parent_frame_id:=parent", "child_frame_id:=child",
                    "imu_topic:=/smoke/imu", "probtf_topic:=/smoke/probtf",
                    "node_name:=smoke_imu_relative_pose",
                ], stdout=launch_log, stderr=subprocess.STDOUT, start_new_session=True,
                   env=launch_environment)
                wait_for(lambda: publisher.get_num_connections() == 1 and subscriber.get_num_connections() == 1,
                         "launched IMU node did not connect")
                unrelated = Imu()
                unrelated.header.frame_id = "other_imu__112233"
                unrelated.angular_velocity_covariance[0] = -1
                unrelated.linear_acceleration_covariance[0] = -1
                for index, message in enumerate(messages):
                    publisher.publish(message)
                    if index % 10 == 0:
                        publisher.publish(unrelated)
                    # Pacing tests the input interface, not estimator rate limiting.
                    time.sleep(0.001)
                wait_for(lambda: len(received) >= len(expected), "missing posterior outputs after input drain", timeout=20)
                assert set(received) == set(expected), (len(received), len(expected))
                max_mean_error = 0.0
                for stamp, reference in expected.items():
                    actual = received[stamp]
                    assert actual.parent_frame_id == reference.parent_frame_id
                    assert actual.child_frame_id == reference.child_frame_id
                    assert actual.edge_id == reference.edge_id
                    assert actual.authority == "/smoke_imu_relative_pose"
                    assert actual.representative is None
                    assert actual.provenance == reference.provenance
                    assert actual.approximation == reference.approximation
                    left, right = actual.distribution.components[0], reference.distribution.components[0]
                    assert left.raw_weight == right.raw_weight == 1
                    assert left.provenance == right.provenance
                    assert left.approximation == right.approximation
                    assert left.orientation.kind == right.orientation.kind
                    np.testing.assert_allclose(left.orientation.shape_matrix, right.orientation.shape_matrix, atol=1e-11)
                    np.testing.assert_allclose(left.orientation.inverse_concentration, right.orientation.inverse_concentration)
                    np.testing.assert_allclose(left.orientation.reference_quaternion_wxyz, right.orientation.reference_quaternion_wxyz)
                    np.testing.assert_allclose(left.translation.mean_at_reference, right.translation.mean_at_reference)
                    np.testing.assert_allclose(left.translation.residual_covariance, right.translation.residual_covariance)
                    np.testing.assert_array_equal(left.translation.rotation_coupling, right.translation.rotation_coupling)
                    max_mean_error = max(max_mean_error, float(np.max(np.abs(
                        left.translation.mean_at_reference - right.translation.mean_at_reference
                    ))))
                assert not rospy.has_param("/robot_description")
                assert not rospy.has_param("/symbolic_robot_description")
                assert "A NumPy version" not in (logs / "launch.log").read_text()
                print(json.dumps({"input_imu_samples": len(messages), "posterior_records": len(received),
                                  "max_position_mean_wire_error": max_mean_error,
                                  "uses_urdf": False}, indent=2))
        except Exception:
            for name in ("launch.log", "master.log"):
                path = logs / name
                if path.exists():
                    print(path.read_text()[-5000:])
            raise
        finally:
            stop(launch)
            if rospy.core.is_initialized():
                rospy.signal_shutdown("smoke complete")
            stop(master)


if __name__ == "__main__":
    run()
