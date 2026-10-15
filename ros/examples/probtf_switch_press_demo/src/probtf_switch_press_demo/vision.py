"""Planar image rendering, then the existing detector and single-image estimator."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from prob_artag_detector.camera import ippe_square_object_points, project_points
from prob_artag_detector.detector import ArucoCornerDetector
from prob_artag_detector.estimator import PoseMixtureEstimator
from prob_artag_detector.models import CameraModel
from prob_artag_detector.visualization import draw_debug_image
from probtf_switch_press_demo.scene import inverse


def camera_model(settings):
    width, height, focal = (settings[key] for key in ("width", "height", "focal_length"))
    return CameraModel(np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]]),
                       np.zeros(5), width, height)


class PlanarCamera:
    """Use a real shipped AprilTag texture; annotations never leave this renderer."""
    def __init__(self, texture_path, marker_size, settings, seed):
        self.texture = cv2.imread(str(texture_path), cv2.IMREAD_GRAYSCALE)
        if self.texture is None:
            raise ValueError("AprilTag texture was not found")
        self.camera = camera_model(settings)
        self.settings = settings
        self.rng = np.random.default_rng(seed)
        # The shipped texture has a one-module quiet zone around an 8-module tag.
        self.texture_points = ippe_square_object_points(marker_size * 10 / 8)
        self.clean_image = None
        self.diagnostics = {}

    def sharp_image(self, world_camera, world_marker):
        relative = inverse(world_camera) @ world_marker
        image = np.full((self.camera.height, self.camera.width), 255, dtype=np.uint8)
        points_camera = self.texture_points @ relative[:3, :3].T + relative[:3, 3]
        if np.all(points_camera[:, 2] > 0.02):
            corners = project_points(self.texture_points, relative[:3, :3], relative[:3, 3], self.camera)
            height, width = self.texture.shape
            pixels = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
            homography = cv2.getPerspectiveTransform(pixels, np.float32(corners))
            image = cv2.warpPerspective(self.texture, homography, (self.camera.width, self.camera.height),
                                        flags=cv2.INTER_LINEAR, borderValue=255)
        return image

    def render(self, world_camera, world_marker, history=()):
        self.clean_image = self.sharp_image(world_camera, world_marker)
        exposure = self.settings["exposure_time"]
        count = self.settings["motion_blur_samples"]
        cameras = [world_camera]
        if len(history) >= 2 and exposure > 0 and count > 1:
            times = np.array([item[0] for item in history])
            poses = np.array([item[1] for item in history])
            sampled_times = np.clip(np.linspace(times[-1] - exposure, times[-1], count), times[0], times[-1])
            rotations = Slerp(times, Rotation.from_matrix(poses[:, :3, :3]))(sampled_times).as_matrix()
            cameras = []
            for stamp, rotation in zip(sampled_times, rotations):
                camera = np.eye(4)
                camera[:3, :3] = rotation
                camera[:3, 3] = [np.interp(stamp, times, poses[:, axis, 3]) for axis in range(3)]
                cameras.append(camera)
        image = np.mean([self.sharp_image(camera, world_marker) for camera in cameras], axis=0).astype(np.float32)
        motion = 0.
        if len(cameras) > 1:
            projected = []
            for camera in (cameras[0], cameras[-1]):
                relative = inverse(camera) @ world_marker
                projected.append(project_points(self.texture_points, relative[:3, :3], relative[:3, 3], self.camera))
            motion = float(np.max(np.linalg.norm(projected[1] - projected[0], axis=1)))
        if self.settings["blur_sigma"] > 0:
            image = cv2.GaussianBlur(image, (0, 0), self.settings["blur_sigma"])
        image += self.rng.normal(0, self.settings["noise_std"], image.shape).astype(np.float32)
        self.diagnostics = dict(exposure_s=exposure, temporal_samples=len(cameras),
                                motion_displacement_px=motion, gaussian_sigma_px=self.settings["blur_sigma"],
                                noise_std=self.settings["noise_std"])
        return np.clip(np.rint(image), 0, 255).astype(np.uint8)


class ImageObservation:
    """Receives pixels and intrinsics, never a true pose or renderer corner array."""
    def __init__(self, marker_settings, camera_settings):
        self.marker_id = marker_settings["id"]
        self.camera = camera_model(camera_settings)
        self.detector = ArucoCornerDetector(corner_sigma_px=marker_settings["corner_sigma_px"],
                                            adaptive_covariance=marker_settings["adaptive_covariance"],
                                            bootstrap_noise_std=camera_settings["noise_std"])
        self.estimator = PoseMixtureEstimator(marker_settings["size"],
            translation_prior_variance=marker_settings["translation_prior_variance"])
        self.last_observation = None

    def estimate(self, image, stamp):
        observations = tuple(item for item in self.detector.detect(image) if item.marker_id == self.marker_id)
        self.last_observation = observations[0] if observations else None
        if not observations:
            return None, draw_debug_image(image, (), status_text="marker missing: hold command")
        result = self.estimator.estimate(observations[0], self.camera, "camera_optical_frame", "switch_marker",
                                         stamp, camera_model_source_id="configured_pinhole")
        debug = draw_debug_image(image, observations[:1], (result,), self.camera,
                                 status_text="fresh single-image estimate")
        return result, debug
