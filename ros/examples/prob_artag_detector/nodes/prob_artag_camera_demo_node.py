#!/usr/bin/env python3

"""Small ROS webcam source with a calibration-YAML or explicit FOV fallback."""

from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, String

from prob_artag_detector import (
    CameraCalibration,
    approximate_camera_model,
    load_camera_calibration,
)


def _video_device(value):
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    if not text:
        raise ValueError("video_device must not be empty.")
    return text


def _calibration_path(value):
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("package://"):
        import rospkg

        relative = text[len("package://") :]
        package_name, separator, package_path = relative.partition("/")
        if not separator or not package_name or not package_path:
            raise ValueError("package:// camera_info_url must include a package and path.")
        try:
            package_root = rospkg.RosPack().get_path(package_name)
        except rospkg.ResourceNotFound as error:
            raise ValueError(
                "camera_info_url package {!r} was not found.".format(package_name)
            ) from error
        return Path(package_root) / package_path
    parsed = urlparse(text)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError("file:// camera_info_url may only use localhost.")
        return Path(unquote(parsed.path)).expanduser()
    if parsed.scheme:
        raise ValueError("camera_info_url supports plain, file://, or package:// paths.")
    return Path(text).expanduser()


class ProbArtagCameraDemoNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.publisher = rospy.Publisher("~image_raw", Image, queue_size=1)
        self.info_publisher = rospy.Publisher("~camera_info", CameraInfo, queue_size=1)
        self.calibration_status_publisher = rospy.Publisher(
            "~calibration_status", String, queue_size=1, latch=True
        )
        self.device = _video_device(rospy.get_param("~video_device", "/dev/video0"))
        self.frame_id = str(
            rospy.get_param("~camera_frame_id", "camera_optical_frame")
        ).strip("/")
        if not self.frame_id:
            raise ValueError("camera_frame_id must not be empty.")
        self.camera_name = str(rospy.get_param("~camera_name", "prob_artag_camera"))
        self.width = int(rospy.get_param("~image_width", 640))
        self.height = int(rospy.get_param("~image_height", 480))
        self.fps = float(rospy.get_param("~framerate", 30.0))
        requested_fourcc = str(rospy.get_param("~fourcc", "MJPG")).strip()
        self.fourcc = (
            ""
            if requested_fourcc.lower() in ("", "none", "default")
            else requested_fourcc.upper()
        )
        self.loop_video = bool(rospy.get_param("~loop_video", False))
        self.reconnect_interval = float(rospy.get_param("~reconnect_interval", 1.0))
        self.fallback_horizontal_fov_deg = float(
            rospy.get_param("~fallback_horizontal_fov_deg", 60.0)
        )
        if self.width < 0 or self.height < 0:
            raise ValueError("image_width and image_height must be non-negative.")
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("framerate must be positive and finite.")
        if not np.isfinite(self.reconnect_interval) or self.reconnect_interval <= 0.0:
            raise ValueError("reconnect_interval must be positive and finite.")
        if self.fourcc and len(self.fourcc) != 4:
            raise ValueError("fourcc must be empty or exactly four characters.")
        approximate_camera_model(2, 2, self.fallback_horizontal_fov_deg)

        self.calibration = self._load_calibration(
            rospy.get_param("~camera_info_url", "")
        )
        self.capture = None
        self.sequence = 0
        self._fallback_sizes = set()
        self._resolution_warning_sizes = set()
        self._last_calibration_status = None
        rospy.on_shutdown(self.close)

    def _load_calibration(self, value):
        try:
            path = _calibration_path(value)
            if path is None:
                rospy.logwarn(
                    "No camera_info_url supplied; no CameraInfo will be published. "
                    "The detector will use its explicit FOV-based fallback."
                )
                return None
            calibration = load_camera_calibration(path)
            rospy.loginfo(
                "Loaded camera calibration %s (%dx%d, %s).",
                path,
                calibration.model.width,
                calibration.model.height,
                calibration.distortion_model,
            )
            return calibration
        except (OSError, TypeError, ValueError) as error:
            rospy.logwarn(
                "Camera calibration could not be loaded (%s); no CameraInfo will "
                "be published and the detector will use its FOV-based fallback.",
                error,
            )
            return None

    def _open(self):
        self.close()
        is_v4l = isinstance(self.device, int) or str(self.device).startswith("/dev/video")
        backend = cv2.CAP_V4L2 if is_v4l else cv2.CAP_ANY
        capture = cv2.VideoCapture(self.device, backend)
        if not capture.isOpened():
            capture.release()
            rospy.logwarn_throttle(
                5.0,
                "Cannot open camera source %r; retrying every %.1f s.",
                self.device,
                self.reconnect_interval,
            )
            return False
        if self.fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        if self.width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        self.capture = capture
        rospy.loginfo(
            "Opened camera source %r at requested %dx%d @ %.1f Hz (fourcc=%s).",
            self.device,
            self.width,
            self.height,
            self.fps,
            self.fourcc or "driver default",
        )
        return True

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _calibration_for_frame(self, width, height):
        if (
            self.calibration is not None
            and self.calibration.model.width == int(width)
            and self.calibration.model.height == int(height)
        ):
            return self.calibration, "calibrated"
        size = (int(width), int(height))
        if self.calibration is not None and size not in self._resolution_warning_sizes:
            rospy.logwarn(
                "Calibration resolution %dx%d does not match captured frame %dx%d; "
                "not publishing CameraInfo for this frame.",
                self.calibration.model.width,
                self.calibration.model.height,
                width,
                height,
            )
            self._resolution_warning_sizes.add(size)
        model = approximate_camera_model(
            width, height, self.fallback_horizontal_fov_deg
        )
        if size not in self._fallback_sizes:
            rospy.logwarn(
                "Detector-side fallback for %dx%d: hfov=%.1f deg, fx=%.1f px; "
                "CameraInfo remains unpublished. Calibrate the camera before "
                "using metric depth quantitatively.",
                width,
                height,
                self.fallback_horizontal_fov_deg,
                model.camera_matrix[0, 0],
            )
            self._fallback_sizes.add(size)
        # Do not publish an inferred model as ordinary CameraInfo: downstream
        # consumers cannot distinguish it from a calibrated camera.  The
        # detector independently constructs the same explicit fallback from the
        # image size and records that fact in ProbTF provenance.
        return None, "fallback"

    @staticmethod
    def _camera_info_message(calibration, header):
        model = calibration.model
        message = CameraInfo()
        message.header = header
        message.width = model.width
        message.height = model.height
        message.distortion_model = calibration.distortion_model
        message.D = model.distortion.tolist()
        message.K = model.camera_matrix.reshape(-1).tolist()
        message.R = np.eye(3, dtype=float).reshape(-1).tolist()
        fx = float(model.camera_matrix[0, 0])
        fy = float(model.camera_matrix[1, 1])
        cx = float(model.camera_matrix[0, 2])
        cy = float(model.camera_matrix[1, 2])
        message.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return message

    def _publish_calibration_status(self, calibration, source):
        if source == "fallback":
            status = "fallback_hfov_{:.3f}deg".format(
                self.fallback_horizontal_fov_deg
            )
        else:
            status = "calibrated_{}".format(calibration.camera_name)
        if status != self._last_calibration_status:
            self.calibration_status_publisher.publish(String(data=status))
            self._last_calibration_status = status

    def _read(self):
        success, frame = self.capture.read()
        if success:
            return frame
        if self.loop_video and not (
            isinstance(self.device, int) or str(self.device).startswith("/dev/video")
        ):
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
            success, frame = self.capture.read()
            if success:
                return frame
        rospy.logwarn_throttle(
            2.0, "Camera source %r stopped returning frames; reopening.", self.device
        )
        self.close()
        return None

    def spin(self):
        rate = rospy.Rate(self.fps)
        while not rospy.is_shutdown():
            if self.capture is None and not self._open():
                rospy.sleep(self.reconnect_interval)
                continue
            frame = self._read()
            if frame is None:
                rospy.sleep(self.reconnect_interval)
                continue
            height, width = frame.shape[:2]
            calibration, source = self._calibration_for_frame(width, height)
            header = Header(
                seq=self.sequence,
                stamp=rospy.Time.now(),
                frame_id=self.frame_id,
            )
            self.sequence += 1
            try:
                image_message = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            except CvBridgeError as error:
                rospy.logwarn_throttle(2.0, "Cannot convert camera frame: %s", error)
                rate.sleep()
                continue
            image_message.header = header
            self._publish_calibration_status(calibration, source)
            if calibration is not None:
                self.info_publisher.publish(
                    self._camera_info_message(calibration, header)
                )
            self.publisher.publish(image_message)
            rospy.logdebug("Published camera frame with %s calibration", source)
            rate.sleep()


def main():
    rospy.init_node("prob_artag_camera")
    try:
        node = ProbArtagCameraDemoNode()
    except (TypeError, ValueError) as error:
        rospy.logfatal("Invalid real-camera demo configuration: %s", error)
        raise
    node.spin()


if __name__ == "__main__":
    main()
