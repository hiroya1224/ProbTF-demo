#!/usr/bin/env python3

import threading

import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, String
from visualization_msgs.msg import MarkerArray

from prob_artag_detector import (
    AdaptiveCornerCovariance,
    ArucoCornerDetector,
    CameraModel,
    PoseMixtureEstimator,
    approximate_camera_model,
    draw_debug_image,
)
from prob_artag_detector.ros_markers import build_pose_mixture_markers
from probtf_msgs.msg import ProbabilisticTransformStamped
from probtf_ros import transform_distribution_to_msg


class ProbArtagDetectorNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.camera_lock = threading.Lock()
        self.processing_lock = threading.Lock()
        self.camera_model = None
        self.camera_model_source = None
        self.camera_info_frame = ""
        self.calibration_status_hint = ""
        self.configured_camera_frame = str(rospy.get_param("~camera_frame_id", "")).strip("/")
        self.fallback_camera_frame = str(
            rospy.get_param("~fallback_camera_frame_id", "camera_optical_frame")
        ).strip("/")
        self.fallback_calibration_enabled = bool(
            rospy.get_param("~fallback_calibration_enabled", False)
        )
        self.fallback_horizontal_fov_deg = float(
            rospy.get_param("~fallback_horizontal_fov_deg", 60.0)
        )
        fallback_fx = float(rospy.get_param("~fallback_fx_px", 0.0))
        fallback_fy = float(rospy.get_param("~fallback_fy_px", 0.0))
        fallback_cx = float(rospy.get_param("~fallback_cx_px", -1.0))
        fallback_cy = float(rospy.get_param("~fallback_cy_px", -1.0))
        if not np.all(
            np.isfinite(
                [
                    self.fallback_horizontal_fov_deg,
                    fallback_fx,
                    fallback_fy,
                    fallback_cx,
                    fallback_cy,
                ]
            )
        ):
            raise ValueError("fallback calibration parameters must be finite.")
        self.fallback_fx_px = fallback_fx if fallback_fx > 0.0 else None
        self.fallback_fy_px = fallback_fy if fallback_fy > 0.0 else None
        self.fallback_cx_px = fallback_cx if fallback_cx >= 0.0 else None
        self.fallback_cy_px = fallback_cy if fallback_cy >= 0.0 else None
        self.require_calibration_resolution_match = bool(
            rospy.get_param("~require_calibration_resolution_match", True)
        )
        self._fallback_log_keys = set()
        if self.fallback_calibration_enabled:
            # Fail at startup for invalid numeric parameters, before image callbacks.
            approximate_camera_model(
                2,
                2,
                self.fallback_horizontal_fov_deg,
                self.fallback_fx_px,
                self.fallback_fy_px,
                self.fallback_cx_px,
                self.fallback_cy_px,
            )
        self.tag_frame_prefix = str(rospy.get_param("~tag_frame_prefix", "apriltag_"))
        self.corner_sigma_px = float(rospy.get_param("~corner_sigma_px", 0.5))
        self.adaptive_covariance_enabled = bool(
            rospy.get_param("~adaptive_covariance", True)
        )
        self.detector = ArucoCornerDetector(
            family=rospy.get_param("~family", "DICT_APRILTAG_36h11"),
            corner_sigma_px=self.corner_sigma_px,
            corner_refinement=rospy.get_param("~corner_refinement", True),
            adaptive_covariance=self.adaptive_covariance_enabled,
            bootstrap_samples=rospy.get_param("~bootstrap_samples", 8),
            bootstrap_noise_std=rospy.get_param("~bootstrap_noise_std", 6.0),
            bootstrap_dither_px=rospy.get_param("~bootstrap_dither_px", 0.35),
            bootstrap_blur_sigma_px=rospy.get_param(
                "~bootstrap_blur_sigma_px", 0.35
            ),
            bootstrap_covariance_shrinkage=rospy.get_param(
                "~bootstrap_covariance_shrinkage", 0.25
            ),
            bootstrap_min_success_ratio=rospy.get_param(
                "~bootstrap_min_success_ratio", 0.5
            ),
            bootstrap_dropout_sigma_px=rospy.get_param(
                "~bootstrap_dropout_sigma_px", 2.0
            ),
            bootstrap_seed=rospy.get_param("~bootstrap_seed", 0),
        )
        self.temporal_covariance_enabled = bool(
            rospy.get_param("~temporal_covariance_enabled", True)
        )
        self.corner_uncertainty = (
            AdaptiveCornerCovariance(
                minimum_sigma_px=self.corner_sigma_px,
                maximum_excess_sigma_px=rospy.get_param(
                    "~temporal_max_excess_sigma_px", 5.0
                ),
                warmup_samples=rospy.get_param("~temporal_warmup_samples", 8),
                temporal_half_life_sec=rospy.get_param(
                    "~temporal_half_life_sec", 0.5
                ),
                shrinkage=rospy.get_param("~temporal_shrinkage", 0.25),
                motion_gate_px=rospy.get_param(
                    "~temporal_motion_gate_px", 1.5
                ),
                motion_gate_edge_fraction=rospy.get_param(
                    "~temporal_motion_gate_edge_fraction", 0.02
                ),
                affine_motion_fraction=rospy.get_param(
                    "~temporal_affine_motion_fraction", 0.9
                ),
                freeze_affine_motion=rospy.get_param(
                    "~temporal_freeze_affine_motion", False
                ),
                huber_chi2=rospy.get_param("~temporal_huber_chi2", 20.09),
                hard_outlier_px=rospy.get_param(
                    "~temporal_hard_outlier_px", 8.0
                ),
                hard_outlier_edge_fraction=rospy.get_param(
                    "~temporal_hard_outlier_edge_fraction", 0.15
                ),
                max_gap_sec=rospy.get_param("~temporal_max_gap_sec", 0.5),
                max_dt_ratio=rospy.get_param("~temporal_max_dt_ratio", 2.5),
                track_ttl_sec=rospy.get_param(
                    "~temporal_track_ttl_sec", 2.0
                ),
            )
            if self.temporal_covariance_enabled
            else None
        )
        self.corner_uncertainty_context = None
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
        self.publish_markers = bool(rospy.get_param("~publish_markers", True))
        self.marker_lifetime_sec = float(rospy.get_param("~marker_lifetime_sec", 0.5))
        self.marker_axis_length_m = float(
            rospy.get_param("~marker_axis_length_m", 0.5 * self.estimator.tag_size_m)
        )
        self.marker_tag_thickness_m = float(
            rospy.get_param("~marker_tag_thickness_m", 0.003)
        )
        self.maximum_uncertainty_scale_m = float(
            rospy.get_param("~maximum_uncertainty_scale_m", 0.75)
        )
        marker_dimensions = (
            self.marker_lifetime_sec,
            self.marker_axis_length_m,
            self.marker_tag_thickness_m,
            self.maximum_uncertainty_scale_m,
        )
        if not np.all(np.isfinite(marker_dimensions)) or min(marker_dimensions) <= 0.0:
            raise ValueError("marker dimensions and lifetime must be positive and finite.")
        self.marker_publisher = (
            rospy.Publisher("~markers", MarkerArray, queue_size=1)
            if self.publish_markers
            else None
        )
        self.camera_subscriber = rospy.Subscriber(
            "~camera_info", CameraInfo, self._camera_info, queue_size=1
        )
        self.calibration_status_subscriber = rospy.Subscriber(
            "~calibration_status", String, self._calibration_status, queue_size=1
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
            self.camera_model_source = "camera_info"
            self.camera_info_frame = str(message.header.frame_id).strip("/")

    def _calibration_status(self, message):
        status = str(message.data).strip()
        if not status:
            return
        with self.camera_lock:
            self.calibration_status_hint = status

    def _fallback_camera(self, width, height):
        key = (int(width), int(height))
        camera = approximate_camera_model(
            width,
            height,
            self.fallback_horizontal_fov_deg,
            self.fallback_fx_px,
            self.fallback_fy_px,
            self.fallback_cx_px,
            self.fallback_cy_px,
        )
        if key not in self._fallback_log_keys:
            rospy.logwarn(
                "Using approximate zero-distortion calibration for %dx%d "
                "(horizontal FOV %.1f deg, fx=%.1f px). Metric depth is only "
                "approximate until a valid CameraInfo is supplied.",
                width,
                height,
                self.fallback_horizontal_fov_deg,
                camera.camera_matrix[0, 0],
            )
            self._fallback_log_keys.add(key)
        return camera

    def _camera_for_image(self, width, height, image_frame):
        image_frame = str(image_frame).strip("/")
        with self.camera_lock:
            camera = self.camera_model
            source = self.camera_model_source
            camera_info_frame = self.camera_info_frame
            calibration_status_hint = self.calibration_status_hint

        if (
            camera is not None
            and source == "camera_info"
            and self.require_calibration_resolution_match
            and (camera.width != int(width) or camera.height != int(height))
        ):
            if not self.fallback_calibration_enabled:
                raise ValueError(
                    "CameraInfo resolution {}x{} does not match image {}x{}.".format(
                        camera.width, camera.height, width, height
                    )
                )
            rospy.logwarn_throttle(
                2.0,
                "CameraInfo resolution %dx%d does not match image %dx%d; using "
                "the approximate fallback for this frame.",
                camera.width,
                camera.height,
                width,
                height,
            )
            camera = self._fallback_camera(width, height)
            source = "fallback_resolution_mismatch"

        if camera is None or source == "fallback":
            if not self.fallback_calibration_enabled:
                return None
            if (
                camera is None
                or camera.width != int(width)
                or camera.height != int(height)
            ):
                camera = self._fallback_camera(width, height)
                with self.camera_lock:
                    # Never overwrite a valid CameraInfo that arrived concurrently.
                    if self.camera_model_source != "camera_info":
                        self.camera_model = camera
                        self.camera_model_source = "fallback"
            source = "fallback"

        if (
            not self.configured_camera_frame
            and image_frame
            and camera_info_frame
            and image_frame != camera_info_frame
            and source == "camera_info"
        ):
            raise ValueError(
                "Image frame {!r} does not match CameraInfo frame {!r}; set "
                "~camera_frame_id only when this override is intentional.".format(
                    image_frame, camera_info_frame
                )
            )
        camera_frame = (
            self.configured_camera_frame
            or image_frame
            or camera_info_frame
        )
        if not camera_frame and source.startswith("fallback"):
            camera_frame = self.fallback_camera_frame
        if not camera_frame:
            raise ValueError("Image requires a camera optical frame ID.")
        if source == "camera_info" and calibration_status_hint:
            calibration_source = calibration_status_hint
        elif source.startswith("fallback") and calibration_status_hint.startswith(
            "fallback"
        ):
            calibration_source = calibration_status_hint
        else:
            calibration_source = source
        return camera, camera_frame, calibration_source

    def _image(self, message):
        if not self.processing_lock.acquire(False):
            rospy.logwarn_throttle(2.0, "Dropping image while the previous tag frame is processing")
            return
        try:
            try:
                image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            except CvBridgeError as error:
                rospy.logwarn_throttle(2.0, "cv_bridge rejected image: %s", error)
                return
            try:
                camera_selection = self._camera_for_image(
                    image.shape[1], image.shape[0], message.header.frame_id
                )
            except ValueError as error:
                rospy.logwarn_throttle(2.0, "Camera calibration rejected image: %s", error)
                return
            if camera_selection is None:
                rospy.logwarn_throttle(
                    2.0,
                    "Waiting for valid CameraInfo before estimating AprilTag poses "
                    "(or enable ~fallback_calibration_enabled).",
                )
                return
            camera, camera_frame, calibration_source = camera_selection
            stamp = message.header.stamp.to_sec()
            corner_diagnostics = ()
            try:
                observations = self.detector.detect(image)
                if self.corner_uncertainty is not None:
                    uncertainty_context = (
                        camera_frame,
                        calibration_source,
                        int(camera.width),
                        int(camera.height),
                        tuple(np.asarray(camera.camera_matrix).reshape(-1)),
                        tuple(np.asarray(camera.distortion).reshape(-1)),
                    )
                    if uncertainty_context != self.corner_uncertainty_context:
                        self.corner_uncertainty.reset()
                        self.corner_uncertainty_context = uncertainty_context
                    # Preserve diagnostics for the image overlay without making
                    # them part of the probabilistic observation contract.
                    updates = tuple(
                        self.corner_uncertainty.update(
                            observation,
                            stamp,
                            spatial_covariance=observation.image_covariance,
                        )
                        for observation in observations
                    )
                    observations = tuple(update[0] for update in updates)
                    corner_diagnostics = tuple(update[1] for update in updates)
            except (
                TypeError,
                ValueError,
                RuntimeError,
                np.linalg.LinAlgError,
            ) as error:
                rospy.logwarn_throttle(2.0, "AprilTag detection failed: %s", error)
                return

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
                        camera_model_source_id="camera_model:{}".format(
                            calibration_source.replace(" ", "_")
                        ),
                        camera_model_is_approximate=calibration_source.startswith(
                            "fallback"
                        ),
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
            output_header = Header(
                seq=message.header.seq,
                stamp=message.header.stamp,
                frame_id=camera_frame,
            )
            if self.marker_publisher is not None:
                try:
                    self.marker_publisher.publish(
                        build_pose_mixture_markers(
                            results,
                            output_header,
                            self.estimator.tag_size_m,
                            axis_length_m=self.marker_axis_length_m,
                            tag_thickness_m=self.marker_tag_thickness_m,
                            maximum_uncertainty_scale_m=(
                                self.maximum_uncertainty_scale_m
                            ),
                            lifetime=rospy.Duration.from_sec(
                                self.marker_lifetime_sec
                            ),
                        )
                    )
                except (TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    rospy.logwarn_throttle(2.0, "RViz marker generation failed: %s", error)
            if self.debug_publisher is not None:
                try:
                    debug = draw_debug_image(
                        image,
                        observations,
                        results,
                        camera_model=camera,
                        axis_length=0.5 * self.estimator.tag_size_m,
                        status_text="calibration={} corner_cov={}{}".format(
                            calibration_source,
                            "bootstrap" if self.adaptive_covariance_enabled else "fixed",
                            "+temporal" if self.temporal_covariance_enabled else "",
                        ),
                        corner_diagnostics=corner_diagnostics,
                    )
                    debug_message = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
                    debug_message.header = output_header
                    self.debug_publisher.publish(debug_message)
                except (TypeError, ValueError, CvBridgeError, RuntimeError) as error:
                    rospy.logwarn_throttle(2.0, "Debug image generation failed: %s", error)
        finally:
            self.processing_lock.release()


def main():
    rospy.init_node("prob_artag_detector")
    ProbArtagDetectorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
