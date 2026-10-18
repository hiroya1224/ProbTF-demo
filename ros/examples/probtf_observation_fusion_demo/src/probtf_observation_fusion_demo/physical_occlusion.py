"""Opaque fixed geometry for physical camera pixels, never an observation input."""

import cv2
import numpy as np
from functools import lru_cache

from .physical_camera_native import occlusion_mask


def marker_occluder_settings(camera_settings):
    """Resolve one marker-fixed XY rectangle; dimensions are in metres."""
    supplied = camera_settings.get('marker_occluder', {})
    enabled = supplied.get('enabled', False)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError('Physical marker occlusion enabled must be a boolean.')
    center = np.asarray(supplied.get('center', [0., -.155, .28]), dtype=float)
    size = np.asarray(supplied.get('size', [.36, .14]), dtype=float)
    intensity = float(supplied.get('intensity', 96))
    thickness = float(supplied.get('thickness', .004))
    if (center.shape != (3,) or size.shape != (2,)
            or not np.all(np.isfinite(center)) or not np.all(np.isfinite(size))
            or np.any(size <= 0.) or not np.isfinite(thickness) or thickness <= 0.
            or not np.isfinite(intensity) or intensity != int(intensity)
            or not 0. <= intensity <= 255.):
        raise ValueError('Physical occluder geometry and its byte intensity must be finite and valid.')
    return dict(enabled=bool(enabled), center=center.tolist(), size=size.tolist(),
                intensity=int(intensity), thickness=thickness,
                frame='marker', geometry='fixed parallel XY rectangle')


def _clip_near(vertices, near):
    """Clip a convex camera-space polygon to the camera's near half-space."""
    output = []
    previous = vertices[-1]
    for current in vertices:
        was_inside, is_inside = previous[2] >= near, current[2] >= near
        if was_inside != is_inside:
            fraction = (near-previous[2])/(current[2]-previous[2])
            output.append(previous+fraction*(current-previous))
        if is_inside:
            output.append(current)
        previous = current
    return np.asarray(output)


def _pixels(vertices, matrix):
    homogeneous = vertices @ matrix.T
    return homogeneous[:, :2]/homogeneous[:, 2:3]


@lru_cache(maxsize=4)
def _pinhole_rays(matrix_values, width, height):
    """Cache only the fixed intrinsic pixel rays, never a physical pose."""
    inverse = np.linalg.inv(np.asarray(matrix_values).reshape(3, 3))
    columns, rows = np.arange(width)[None, :], np.arange(height)[:, None]
    rays = tuple(inverse[i, 0]*columns+inverse[i, 1]*rows+inverse[i, 2]
                 for i in range(3))
    for ray in rays:
        ray.setflags(write=False)
    return rays


def occlude_marker_image(image, world_camera, world_marker, camera, texture_points,
                         occluder, near=.02):
    """Composite a sheet using ray/plane depths and rectangle intersections.

    The marker and opaque sheet are fixed in the same physical marker frame.
    A visible board surface wins over a sheet behind it; outside that board,
    the renderer's white background has infinite depth. This function uses
    no elapsed time, hand distance, detector state, inferred law or RNG.
    """
    center, size = np.asarray(occluder['center']), np.asarray(occluder['size'])
    lower, upper = center[:2]-.5*size, center[:2]+.5*size
    corners = np.array([[lower[0], upper[1], center[2]],
        [upper[0], upper[1], center[2]], [upper[0], lower[1], center[2]],
        [lower[0], lower[1], center[2]]])
    camera_marker = np.linalg.solve(world_camera, world_marker)
    marker_camera = np.linalg.solve(world_marker, world_camera)
    matrix = np.asarray(camera.camera_matrix)
    height, width = image.shape[:2]
    board_camera = np.asarray(texture_points) @ camera_marker[:3, :3].T+camera_marker[:3, 3]
    board_mask = np.zeros((height, width), dtype=np.uint8)
    # This is exactly the existing planar renderer's board visibility rule.
    if np.all(board_camera[:, 2] > near):
        projected_board = _pixels(board_camera, matrix)
        cv2.fillConvexPoly(board_mask, np.rint(projected_board).astype(np.int32), 1)
    board_pixels = int(np.count_nonzero(board_mask))
    diagnostic = dict(board_pixels=board_pixels, occluded_board_pixels=0,
                      occluder_pixels=0, occluded_board_fraction=0.)
    camera_corners = corners @ camera_marker[:3, :3].T+camera_marker[:3, 3]
    clipped = _clip_near(camera_corners, near)
    if len(clipped) < 3:
        return image, diagnostic
    projected = _pixels(clipped, matrix)
    x0, y0 = np.maximum(np.floor(projected.min(axis=0)).astype(int), [0, 0])
    x1, y1 = np.minimum(np.ceil(projected.max(axis=0)).astype(int)+1, [width, height])
    if x1 <= x0 or y1 <= y0:
        return image, diagnostic
    rays = _pinhole_rays(tuple(matrix.ravel()), width, height)
    rotation, origin = marker_camera[:3, :3], marker_camera[:3, 3]
    board = board_mask[y0:y1, x0:x1].astype(bool)
    opaque = occlusion_mask(rays, rotation, origin, center, lower, upper, near,
                            board_mask, (x0, y0, x1, y1))
    pixels = int(np.count_nonzero(opaque))
    if not pixels:
        return image, diagnostic
    output = np.array(image, copy=True)
    output[y0:y1, x0:x1][opaque] = occluder['intensity']
    covered = int(np.count_nonzero(opaque & board))
    diagnostic.update(occluder_pixels=pixels, occluded_board_pixels=covered,
        occluded_board_fraction=covered/board_pixels if board_pixels else 0.)
    return output, diagnostic
