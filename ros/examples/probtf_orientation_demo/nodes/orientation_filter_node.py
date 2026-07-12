#!/usr/bin/env python3

import threading

import numpy as np
import rospy
from sensor_msgs.msg import Imu, MagneticField

from probtf.bingham import bingham_mode, canonical_bingham_parameter
from probtf_estimators.evidence_fusion import TransformEvidence
from probtf_estimators.orientation_imu import (
    OrientationBinghamFilter,
    gravity_bingham_evidence,
    magnetic_bingham_evidence,
)
from probtf_estimators.ros_conversions import transform_evidence_to_msg
from probtf_msgs.msg import ProbabilisticTF, TransformEvidence as TransformEvidenceMsg


def _vector3(message):
    return np.array([message.x, message.y, message.z], dtype=float)


def _covariance(values, fallback_variance):
    covariance = np.asarray(values, dtype=float).reshape(3, 3)
    if covariance[0, 0] < 0.0 or not np.all(np.isfinite(covariance)):
        return np.eye(3, dtype=float) * float(fallback_variance)
    covariance = 0.5 * (covariance + covariance.T)
    if np.min(np.linalg.eigvalsh(covariance)) < 0.0:
        return np.eye(3, dtype=float) * float(fallback_variance)
    return covariance


class OrientationFilterNode:
    def __init__(self):
        initial = np.asarray(
            rospy.get_param(
                "~initial_bingham",
                np.diag([0.0, -5.0, -5.0, -5.0]).reshape(-1).tolist(),
            ),
            dtype=float,
        ).reshape(4, 4)
        self.orientation_filter = OrientationBinghamFilter(
            initial,
            integration_steps=rospy.get_param("~integration_steps", 60),
            max_iterations=rospy.get_param("~max_iterations", 40),
        )
        self.parent_frame_id = rospy.get_param("~parent_frame_id", "world").lstrip("/")
        self.child_frame_id = rospy.get_param("~child_frame_id", "").lstrip("/")
        self.reference_gravity = np.asarray(
            rospy.get_param("~reference_gravity", [0.0, 0.0, 1.0]),
            dtype=float,
        )
        self.reference_magnetic = np.asarray(
            rospy.get_param("~reference_magnetic", [1.0, 0.0, 0.0]),
            dtype=float,
        )
        self.gravity_concentration = float(rospy.get_param("~gravity_concentration", 50.0))
        self.magnetic_concentration = float(rospy.get_param("~magnetic_concentration", 20.0))
        self.default_gyro_variance = float(rospy.get_param("~default_gyro_variance", 1e-4))
        self.maximum_dt = float(rospy.get_param("~maximum_dt", 0.2))
        self.maximum_magnetic_age = float(rospy.get_param("~maximum_magnetic_age", 0.2))
        self.use_magnetometer = bool(rospy.get_param("~use_magnetometer", True))
        self.last_imu_stamp = None
        self.latest_magnetic = None
        self.lock = threading.Lock()

        self.prediction_publisher = rospy.Publisher(
            "~prediction",
            TransformEvidenceMsg,
            queue_size=10,
        )
        self.gravity_publisher = rospy.Publisher(
            "~gravity_evidence",
            TransformEvidenceMsg,
            queue_size=10,
        )
        self.magnetic_publisher = rospy.Publisher(
            "~magnetic_evidence",
            TransformEvidenceMsg,
            queue_size=10,
        )
        self.posterior_publisher = rospy.Publisher(
            "~posterior",
            ProbabilisticTF,
            queue_size=10,
        )
        self.imu_subscriber = rospy.Subscriber("~imu", Imu, self._update_imu, queue_size=50)
        self.magnetic_subscriber = rospy.Subscriber(
            "~magnetic_field",
            MagneticField,
            self._update_magnetic,
            queue_size=20,
        )

    def _update_magnetic(self, message):
        with self.lock:
            self.latest_magnetic = (
                message.header.frame_id.lstrip("/"),
                message.header.stamp.to_sec(),
                _vector3(message.magnetic_field),
            )

    def _evidence(self, source_id, kind, parameter, stamp, sequence):
        return TransformEvidence(
            source_id=source_id,
            parent_frame_id=self.parent_frame_id,
            child_frame_id=self.child_frame_id,
            evidence_kind=kind,
            orientation_bingham=parameter,
            timestamp=stamp,
            sequence=sequence,
        )

    def _update_imu(self, message):
        with self.lock:
            child_frame_id = self.child_frame_id or message.header.frame_id.lstrip("/")
            if not child_frame_id:
                rospy.logwarn_throttle(2.0, "Orientation filter requires a child frame ID")
                return
            if self.child_frame_id and child_frame_id != message.header.frame_id.lstrip("/"):
                rospy.logwarn_throttle(2.0, "Ignoring IMU message from an unexpected frame")
                return
            self.child_frame_id = child_frame_id
            stamp = message.header.stamp.to_sec()
            dt = 0.0 if self.last_imu_stamp is None else stamp - self.last_imu_stamp
            if dt < 0.0:
                rospy.logwarn_throttle(2.0, "Ignoring out-of-order IMU message")
                return
            if dt > self.maximum_dt:
                rospy.logwarn_throttle(2.0, "IMU gap exceeds maximum_dt; skipping gyro propagation")
                dt = 0.0
            self.last_imu_stamp = stamp

            try:
                gravity = gravity_bingham_evidence(
                    self.reference_gravity,
                    _vector3(message.linear_acceleration),
                    self.gravity_concentration,
                )
                magnetic = self._current_magnetic_evidence(stamp)
                update = self.orientation_filter.update(
                    angular_velocity=_vector3(message.angular_velocity),
                    dt=dt,
                    angular_velocity_covariance=_covariance(
                        message.angular_velocity_covariance,
                        self.default_gyro_variance,
                    ),
                    gravity_evidence=gravity,
                    magnetic_evidence=magnetic,
                )
            except (TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                rospy.logwarn_throttle(2.0, "ProbTF orientation update rejected: %s", error)
                return

            prediction = self._evidence(
                "gyro_prediction",
                "prediction",
                update.prediction_parameter,
                stamp,
                message.header.seq,
            )
            gravity_evidence = self._evidence(
                "gravity",
                "likelihood",
                update.gravity_evidence,
                stamp,
                message.header.seq,
            )
            self.prediction_publisher.publish(
                transform_evidence_to_msg(
                    prediction,
                    message_type=TransformEvidenceMsg,
                    time_factory=rospy.Time.from_sec,
                )
            )
            self.gravity_publisher.publish(
                transform_evidence_to_msg(
                    gravity_evidence,
                    message_type=TransformEvidenceMsg,
                    time_factory=rospy.Time.from_sec,
                )
            )
            source_ids = ["gyro_prediction", "gravity"]
            if update.magnetic_evidence is not None:
                magnetic_evidence = self._evidence(
                    "magnetic",
                    "likelihood",
                    update.magnetic_evidence,
                    stamp,
                    message.header.seq,
                )
                self.magnetic_publisher.publish(
                    transform_evidence_to_msg(
                        magnetic_evidence,
                        message_type=TransformEvidenceMsg,
                        time_factory=rospy.Time.from_sec,
                    )
                )
                source_ids.append("magnetic")
            self.posterior_publisher.publish(
                self._posterior_message(
                    update.posterior_parameter,
                    message.header,
                    source_ids,
                )
            )

    def _current_magnetic_evidence(self, imu_stamp):
        if not self.use_magnetometer or self.latest_magnetic is None:
            return None
        frame_id, stamp, magnetic_field = self.latest_magnetic
        if frame_id and frame_id != self.child_frame_id:
            return None
        if abs(imu_stamp - stamp) > self.maximum_magnetic_age:
            return None
        return magnetic_bingham_evidence(
            self.reference_magnetic,
            magnetic_field,
            self.magnetic_concentration,
        )

    def _posterior_message(self, parameter, header, source_ids):
        parameter = canonical_bingham_parameter(parameter)
        mode = bingham_mode(parameter)
        output = ProbabilisticTF()
        output.header.seq = header.seq
        output.header.stamp = header.stamp
        output.header.frame_id = self.parent_frame_id
        output.parent_frame_id = self.parent_frame_id
        output.child_frame_id = self.child_frame_id
        output.edge_id = "{}__to__{}".format(self.parent_frame_id, self.child_frame_id)
        output.source_id = "orientation_filter"
        output.evidence_source_ids = source_ids
        output.has_position = False
        output.has_orientation = True
        output.orientation_bingham.matrix = parameter.reshape(-1).tolist()
        output.orientation_mode.w = float(mode[0])
        output.orientation_mode.x = float(mode[1])
        output.orientation_mode.y = float(mode[2])
        output.orientation_mode.z = float(mode[3])
        output.approximation_type = "gyro_moment_prediction_with_vector_likelihoods"
        output.closure_approximation = True
        return output


if __name__ == "__main__":
    rospy.init_node("probtf_orientation_filter")
    OrientationFilterNode()
    rospy.spin()
