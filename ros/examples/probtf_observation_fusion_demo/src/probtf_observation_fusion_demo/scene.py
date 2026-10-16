"""Observation generation only. Physical poses never enter the estimator API."""

from pathlib import Path

import cv2
import numpy as np
import rospkg

from prob_artag_detector.camera import ippe_square_object_points, project_points
from prob_artag_detector.models import CameraModel
from probtf.geometry import quat_to_rotmat, rotation_vector_to_quaternion


def camera_model(settings):
    return CameraModel(np.asarray(settings["matrix"], dtype=float),
                       np.asarray(settings["distortion"], dtype=float),
                       int(settings["width"]), int(settings["height"]))


def _pose(data):
    value = np.eye(4)
    value[:3, 3] = data["translation"]
    value[:3, :3] = quat_to_rotmat(rotation_vector_to_quaternion(data["rotation_vector"]))
    return value


def render_image(common, texture_path=None):
    """Return pixels alone; no projected/true corners cross this boundary."""
    physical = common["physical_scene"]
    camera_pose = np.eye(4)
    for item in physical["body_transforms"]:
        camera_pose = camera_pose @ _pose(item)
    mount = common["known_geometry"]["camera_mount"]
    camera_mount = np.eye(4)
    camera_mount[:3, 3] = mount["translation"]
    camera_mount[:3, :3] = quat_to_rotmat(mount["rotation_wxyz"])
    camera_pose = camera_pose @ camera_mount
    relative = np.linalg.solve(camera_pose, _pose(physical["marker_transform"]))
    model = camera_model(common["camera"])
    if texture_path is None:
        texture_path = Path(rospkg.RosPack().get_path("prob_artag_detector")) / "samples" / "apriltag_36h11_id_000.png"
    texture = cv2.imread(str(texture_path), cv2.IMREAD_GRAYSCALE)
    if texture is None:
        raise ValueError("AprilTag texture is unavailable: {}".format(texture_path))
    # The shipped bitmap contains a one-module border around the 8-module tag.
    object_points = ippe_square_object_points(common["marker"]["size"] * 10.0 / 8.0)
    corners = project_points(object_points, relative[:3, :3], relative[:3, 3], model)
    if np.any((object_points @ relative[:3, :3].T + relative[:3, 3])[:, 2] <= 0):
        raise ValueError("The configured physical marker is behind the camera.")
    height, width = texture.shape
    pixels = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    homography = cv2.getPerspectiveTransform(pixels, np.float32(corners))
    image = cv2.warpPerspective(texture, homography, (model.width, model.height),
                                flags=cv2.INTER_LINEAR, borderValue=255)
    sigma = float(physical["image_blur_sigma_px"])
    if sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    image.setflags(write=False)
    return image
