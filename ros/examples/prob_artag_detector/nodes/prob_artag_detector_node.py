#!/usr/bin/env python3

import threading

import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image

from prob_artag_detector import (
    ArucoCornerDetector,
    CameraModel,
    PoseMixtureEstimator,
    draw_debug_image,
)
from probtf_msgs.msg import ProbabilisticTransformStamped
from probtf_ros import transform_distribution_to_msg


class ProbArtagDetectorNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.camera_lock = threading.Lock()
        self.processing_lock = threading.Lock()
        self.camera_model = None
        self.configured_camera_frame = str(rospy.get_param("~camera_frame_id", "")).strip("/")
        self.tag_frame_prefix = str(rospy.get_param("~tag_frame_prefix", "apriltag_"))
        self.detector = ArucoCornerDetector(
            family=rospy.get_param("~family", "DICT_APRILTAG_36h11"),
            corner_sigma_px=rospy.get_param("~corner_sigma_px", 0.5),
            corner_refinement=rospy.get_param("~corner_refinement", True),
        )
        self.estimator = PoseMixtureEstimator(
            tag_size_m=rospy.get_param("~tag_size_m", 0.12),
            max_iterations=rospy.get_param("~max_iterations", 30),
            convergence_tolerance=rospy.get_param("~convergence_tolerance", 1e-9),
            min_depth=rospy.get_param("~min_depth", 1e-6),
            dedup_translation_tolerance=rospy.get_param(
                "~dedup_translation_tolerance", 1e-7
            ),
            dedup_rotation_tolerance_rad=rospy.get_param(
                "~dedup_rotation_tolerance_rad", 1e-7
            ),
            finite_difference_step=rospy.get_param("~finite_difference_step", 1e-6),
            verify_jacobian=rospy.get_param("~verify_jacobian", False),
            spd_tolerance=rospy.get_param("~spd_tolerance", 1e-12),
            translation_prior_mean=rospy.get_param(
                "~translation_prior_mean", [0.0, 0.0, 0.0]
            ),
            translation_prior_variance=rospy.get_param(
                "~translation_prior_variance", 1e6
            ),
        )
        self.probtf_publisher = rospy.Publisher(
            rospy.get_param("~probtf_topic", "/probtf"),
            ProbabilisticTransformStamped,
            queue_size=10,
        )
        self.publish_debug = bool(rospy.get_param("~publish_debug_image", True))
        self.debug_publisher = (
            rospy.Publisher("~debug_image", Image, queue_size=1)
            if self.publish_debug
            else None
        )
        self.camera_subscriber = rospy.Subscriber(
            "~camera_info", CameraInfo, self._camera_info, queue_size=1
        )
        self.image_subscriber = rospy.Subscriber(
            "~image", Image, self._image, queue_size=1, buff_size=2 ** 24
        )

    def _camera_info(self, message):
        try:
            camera = CameraModel.from_camera_info(message)
        except (TypeError, ValueError) as error:
            rospy.logwarn_throttle(2.0, "Rejected camera calibration: %s", error)
            return
        with self.camera_lock:
            self.camera_model = camera

    def _image(self, message):
        if not self.processing_lock.acquire(False):
            rospy.logwarn_throttle(2.0, "Dropping image while the previous tag frame is processing")
            return
        try:
            with self.camera_lock:
                camera = self.camera_model
            if camera is None:
                rospy.logwarn_throttle(2.0, "Waiting for CameraInfo before estimating AprilTag poses")
                return
            try:
                image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            except CvBridgeError as error:
                rospy.logwarn_throttle(2.0, "cv_bridge rejected image: %s", error)
                return
            try:
                observations = self.detector.detect(image)
            except (TypeError, ValueError, RuntimeError) as error:
                rospy.logwarn_throttle(2.0, "AprilTag detection failed: %s", error)
                return

            camera_frame = self.configured_camera_frame or message.header.frame_id.strip("/")
            if not camera_frame:
                rospy.logwarn_throttle(2.0, "Image requires a camera optical frame ID")
                return
            stamp = message.header.stamp.to_sec()
            results = []
            for observation in observations:
                child_frame = "{}{}".format(self.tag_frame_prefix, observation.marker_id)
                try:
                    result = self.estimator.estimate(
                        observation,
                        camera,
                        parent_frame_id=camera_frame,
                        child_frame_id=child_frame,
                        stamp=stamp,
                        edge_id="{}__to__{}".format(camera_frame, child_frame),
                        authority=rospy.get_name().strip("/"),
                    )
                    self.probtf_publisher.publish(
                        transform_distribution_to_msg(
                            result.record,
                            time_factory=rospy.Time.from_sec,
                        )
                    )
                    results.append(result)
                except (TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    rospy.logwarn_throttle(
                        2.0,
                        "Probabilistic pose for tag %d was rejected: %s",
                        observation.marker_id,
                        error,
                    )
                    results.append(None)
            if self.debug_publisher is not None:
                debug = draw_debug_image(
                    image,
                    observations,
                    results,
                    camera_model=camera,
                    axis_length=0.5 * self.estimator.tag_size_m,
                )
                debug_message = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
                debug_message.header = message.header
                self.debug_publisher.publish(debug_message)
        finally:
            self.processing_lock.release()


def main():
    rospy.init_node("prob_artag_detector")
    ProbArtagDetectorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
