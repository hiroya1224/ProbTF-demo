#!/usr/bin/env python3
"""Exercise the real launch and native wire messages on a PRIVATE ROS master.

Run after building relative_coords_from_paired_imu and sourcing the workspace. No hardware is
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
import sys
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


def publish_stream(topic, path):
    """Independent TCPROS sender; keep the connection until the test stops it."""
    import rospy
    from sensor_msgs.msg import Imu

    rospy.init_node("independent_imu_sender", anonymous=True)
    publisher = rospy.Publisher(topic, Imu, queue_size=0, tcp_nodelay=True)
    wait_for(lambda: publisher.get_num_connections() == 1, "independent publisher did not connect")
    for index, encoded in enumerate(json.loads(Path(path).read_text())):
        publisher.publish(Imu().deserialize(bytes.fromhex(encoded)))
        # Differing, deliberately irregular delivery timings, not a shared
        # callback or alternating send loop between the two IMU streams.
        if index % 7 == 0:
            time.sleep(0.003)
    rospy.spin()


def compare_record(actual, reference, case):
    assert actual.parent_frame_id == reference.parent_frame_id
    assert actual.child_frame_id == reference.child_frame_id
    assert actual.edge_id == reference.edge_id
    assert actual.authority == "/smoke_imu_" + case
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
    from relative_coords_from_paired_imu.imu_pair_pipeline import ImuRelativePosePipeline, imu_measurement_covariance
    from probtf.geometry import quat_to_rotmat, rpy_to_quat
    from relative_coords_from_paired_imu.imu_relative_pose import skew

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
            records = pipeline.update_all(
                message.header.frame_id, stamp, omega, force,
                imu_measurement_covariance(message.angular_velocity_covariance, 1e-4),
                imu_measurement_covariance(message.linear_acceleration_covariance, 1e-3),
            )
            for record in records:
                expected[record.stamp] = record

    parents, children = messages[::2], messages[1::2]
    batched = [message for start in range(0, len(parents), 10)
               for group in (parents[start:start + 10], children[start:start + 10]) for message in group]
    cases = {"interleaved": messages, "batch10": batched,
             "parent_ahead": parents + children, "child_ahead": children + parents,
             "independent_publishers": None}
    received = {case: {} for case in cases}
    received_order = {case: [] for case in cases}
    master = None
    launches, senders = [], []
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

                subscribers, publishers = {}, {}
                for case in cases:
                    def receive_case(message, case=case):
                        wire = io.BytesIO()
                        message.serialize(wire)
                        restored = ProbabilisticTransformStamped().deserialize(wire.getvalue())
                        record = transform_distribution_from_msg(restored)
                        received_order[case].append(record.stamp)
                        received[case][record.stamp] = record
                    subscribers[case] = rospy.Subscriber(
                        "/smoke/{}/probtf".format(case), ProbabilisticTransformStamped,
                        receive_case, queue_size=None,
                    )
                    if cases[case] is not None:
                        publishers[case] = rospy.Publisher("/smoke/{}/imu".format(case), Imu, queue_size=0)
                launch_environment = dict(os.environ)
                # Reproduce the workstation's old system NumPy on PYTHONPATH;
                # the venv node must still select its own numerical packages.
                launch_environment["PYTHONPATH"] = "/usr/lib/python3/dist-packages:" + os.environ.get("PYTHONPATH", "")
                for case in cases:
                    launches.append(subprocess.Popen([
                        "roslaunch", "relative_coords_from_paired_imu", "imu_relative_pose.launch",
                        "parent_frame_id:=parent", "child_frame_id:=child",
                        "imu_topic:=/smoke/{}/imu".format(case), "probtf_topic:=/smoke/{}/probtf".format(case),
                        "node_name:=smoke_imu_{}".format(case),
                    ], stdout=launch_log, stderr=subprocess.STDOUT, start_new_session=True,
                       env=launch_environment))
                wait_for(lambda: all(publisher.get_num_connections() == 1 for publisher in publishers.values())
                         and all(subscriber.get_num_connections() == 1 for subscriber in subscribers.values()),
                         "launched IMU node did not connect")
                unrelated = Imu()
                unrelated.header.frame_id = "other_imu__112233"
                unrelated.angular_velocity_covariance[0] = -1
                unrelated.linear_acceleration_covariance[0] = -1
                for case, publisher in publishers.items():
                    for index, message in enumerate(cases[case]):
                        publisher.publish(message)
                        if index % 10 == 0:
                            publisher.publish(unrelated)
                for role, data in (("parent", parents), ("child", children)):
                    encoded = []
                    for message in data:
                        wire = io.BytesIO()
                        message.serialize(wire)
                        encoded.append(wire.getvalue().hex())
                    path = logs / (role + ".json")
                    path.write_text(json.dumps(encoded))
                    senders.append(subprocess.Popen([
                        sys.executable, str(Path(__file__).resolve()), "--publish-stream",
                        "/smoke/independent_publishers/imu", str(path),
                    ], stdout=launch_log, stderr=subprocess.STDOUT, start_new_session=True))
                wait_for(lambda: all(len(values) >= len(expected) for values in received.values()),
                         "missing posterior outputs after asynchronous input drain", timeout=30)
                max_mean_error = 0.0
                for case, values in received.items():
                    assert set(values) == set(expected), (case, len(values), len(expected))
                    assert received_order[case] == list(expected), "Dropped, repeated, or reordered updates: " + case
                    for stamp, reference in expected.items():
                        actual = values[stamp]
                        compare_record(actual, reference, case)
                        max_mean_error = max(max_mean_error, float(np.max(np.abs(
                            actual.distribution.components[0].translation.mean_at_reference
                            - reference.distribution.components[0].translation.mean_at_reference
                        ))))
                assert not rospy.has_param("/robot_description")
                assert not rospy.has_param("/symbolic_robot_description")
                assert "A NumPy version" not in (logs / "launch.log").read_text()
                assert "input/update failed" not in (logs / "launch.log").read_text()
                print(json.dumps({"input_imu_samples_per_case": len(messages),
                                  "posterior_records_per_case": {case: len(values) for case, values in received.items()},
                                  "max_position_mean_wire_error": max_mean_error,
                                  "uses_urdf": False}, indent=2))

        except Exception:
            for name in ("launch.log", "master.log"):
                path = logs / name
                if path.exists():
                    print(path.read_text()[-5000:])
            raise
        finally:
            for process in senders + launches:
                stop(process)
            if rospy.core.is_initialized():
                rospy.signal_shutdown("smoke complete")
            stop(master)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--publish-stream":
        publish_stream(sys.argv[2], sys.argv[3])
    else:
        run()
