"""A coplanar AprilTag board rendered through the existing camera pipeline.

The small tags form one texture before projection. Temporal integration,
Gaussian blur and physical image noise are consequently applied once to the
complete board, using the unchanged switch camera implementation.
"""
import cv2
import numpy as np

from probtf_switch_press_demo.vision import PlanarCamera

from .marker_board import board_layout


def marker_texture(marker_id, pixels_per_module=16):
    """Native 36h11 bitmap with its one-module white quiet zone."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    pixels_per_module = int(pixels_per_module)
    if pixels_per_module < 1:
        raise ValueError('An AprilTag module needs a positive pixel size.')
    size = 8*pixels_per_module
    if hasattr(cv2.aruco, 'generateImageMarker'):
        tag = cv2.aruco.generateImageMarker(dictionary, int(marker_id), size)
    else:
        tag = cv2.aruco.drawMarker(dictionary, int(marker_id), size)
    return cv2.copyMakeBorder(tag, pixels_per_module, pixels_per_module,
        pixels_per_module, pixels_per_module, cv2.BORDER_CONSTANT, value=255)


def board_bitmap(settings, pixels_per_module=16):
    """Return layout, composite texture, metric outer size and native tiles.

    Coordinates are in the unchanged central marker frame: image columns
    increase with its x axis, image rows decrease with its y axis.
    """
    layout = board_layout(settings)
    size = layout[0].size_m
    if any(marker.size_m != size for marker in layout):
        raise ValueError('The planar board renderer requires equal small tag sizes.')
    centers = np.asarray([marker.center_to_marker_translation[:2] for marker in layout])
    quiet_size = 1.25*size
    low = centers.min(axis=0)-quiet_size/2
    high = centers.max(axis=0)+quiet_size/2
    # board_layout fixes the central frame and a symmetric regular layout.
    if not np.allclose(low, -high, rtol=0., atol=1e-14):
        raise ValueError('The board bitmap must be centered in the central marker frame.')
    scale = 8*int(pixels_per_module)/size
    dimensions = np.rint((high-low)*scale).astype(int)
    texture = np.full((dimensions[1], dimensions[0]), 255, dtype=np.uint8)
    tiles = {}
    for marker in layout:
        tile = marker_texture(marker.marker_id, pixels_per_module)
        center = np.asarray(marker.center_to_marker_translation)
        column = int(round((center[0]-quiet_size/2-low[0])*scale))
        row = int(round((high[1]-center[1]-quiet_size/2)*scale))
        height, width = tile.shape
        texture[row:row+height, column:column+width] = np.minimum(
            texture[row:row+height, column:column+width], tile)
        tile.setflags(write=False)
        tiles[marker.marker_id] = tile
    texture.setflags(write=False)
    return layout, texture, tuple(high-low), tiles


class BoardCamera(PlanarCamera):
    """One board image per camera pose, retaining the old noise/exposure law."""
    def __init__(self, texture_path, board_settings, settings, seed):
        # Reuse intrinsics, physical image noise and all exposure behavior.
        super().__init__(texture_path, board_settings['marker_size'], settings, seed)
        self.layout, self.texture, self.board_size, self.marker_textures = board_bitmap(board_settings)
        width, height = self.board_size
        self.texture_points = np.array([[-width/2, height/2, 0.],
            [width/2, height/2, 0.], [width/2, -height/2, 0.],
            [-width/2, -height/2, 0.]])

    def render(self, world_camera, world_marker, history=()):
        image = super().render(world_camera, world_marker, history)
        self.diagnostics.update(marker_board=True,
            marker_ids=[marker.marker_id for marker in self.layout],
            board_size_m=list(self.board_size),
            board_projection='one coplanar composite texture per temporal exposure pose',
            gaussian_blur_applications=1 if self.settings['blur_sigma'] > 0 else 0,
            image_noise_applications=1)
        return image
