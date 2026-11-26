"""Latitude/longitude sphere geometry for display of the saved Bingham law.

Camera +Z and -Z are the poles. Latitude zero is the camera XY equator:
(x,y,z)=(cos(latitude)*cos(longitude), cos(latitude)*sin(longitude),
sin(latitude)). Geometry and colors are display-only. The analytic density
evaluator and learned A are used unchanged.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from figure_geometry import axis_direction_log_density, SPHERICAL_DENSITY_PROVENANCE


GRID_COLOR = ".70"
GRID_ALPHA = .70
GRID_LINEWIDTH = .65


def _positive_integer(value, minimum, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


@lru_cache(maxsize=8)
def latitude_longitude_mesh(latitude_steps: int = 48, longitude_steps: int = 96) -> dict:
    """Return a closed unit sphere with Z poles and latitude/longitude rings.

    Each latitude/longitude cell is split internally into two fill triangles;
    no triangle edges are intended to be drawn. Use the separate graticule for
    the visible latitude and longitude lines. Both poles are single vertices,
    and the longitude seam is joined without duplicate vertices or open edges.
    """
    latitude_steps = _positive_integer(latitude_steps, 2, "latitude_steps")
    longitude_steps = _positive_integer(longitude_steps, 3, "longitude_steps")
    latitudes = np.linspace(np.pi/2, -np.pi/2, latitude_steps+1)[1:-1]
    longitudes = np.arange(longitude_steps)*2*np.pi/longitude_steps
    latitude, longitude = np.meshgrid(latitudes, longitudes, indexing="ij")
    rings = np.stack((np.cos(latitude)*np.cos(longitude),
                      np.cos(latitude)*np.sin(longitude), np.sin(latitude)), axis=-1)
    vertices = np.concatenate(([[0., 0., 1.]], rings.reshape((-1, 3)), [[0., 0., -1.]]))
    faces = []
    for column in range(longitude_steps):
        following = (column+1) % longitude_steps
        faces.append((0, 1+column, 1+following))
    for row in range(latitude_steps-2):
        first = 1+row*longitude_steps
        second = first+longitude_steps
        for column in range(longitude_steps):
            following = (column+1) % longitude_steps
            faces.extend(((first+column, second+column, second+following),
                          (first+column, second+following, first+following)))
    south = len(vertices)-1
    last_ring = south-longitude_steps
    for column in range(longitude_steps):
        following = (column+1) % longitude_steps
        faces.append((last_ring+column, south, last_ring+following))
    faces = np.asarray(faces, dtype=np.int64)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    inward = np.einsum("ni,ni->n", normals, triangles.sum(axis=1)) < 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    triangles = vertices[faces]
    centers = triangles.sum(axis=1)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    determinant = np.abs(np.einsum("ni,ni->n", triangles[:, 0],
                                   np.cross(triangles[:, 1], triangles[:, 2])))
    denominator = 1+np.einsum("ni,ni->n", triangles[:, 0], triangles[:, 1]) \
        + np.einsum("ni,ni->n", triangles[:, 1], triangles[:, 2]) \
        + np.einsum("ni,ni->n", triangles[:, 2], triangles[:, 0])
    areas = 2*np.arctan2(determinant, denominator)
    result = {
        "vertices": vertices, "faces": faces, "centers": centers,
        "camera_xy": vertices[:, :2].copy(), "face_depth": centers[:, 2].copy(),
        "front_faces": centers[:, 2] < 0,
        "limb_faces": (triangles[:, :, 2].min(axis=1) < 0) & (triangles[:, :, 2].max(axis=1) > 0),
        "areas": areas,
    }
    # Callers can add density fields to a shallow dictionary copy, but cannot
    # accidentally alter the cached geometric arrays used by other panels.
    for value in result.values():
        value.setflags(write=False)
    return result


def latitude_longitude_graticule(latitude_spacing_degrees: int = 30,
                                 longitude_spacing_degrees: int = 30,
                                 points_per_circle: int = 181, *,
                                 return_xyz: bool = False) -> list[np.ndarray]:
    """Return latitude circles and longitude meridians, with camera Z poles.

    By default the polylines contain camera (x,y), for an image-down Y axis.
    Coincident front/back latitude projections are drawn once, and each
    projected meridian runs from pole to equator once without retracing itself.
    ``return_xyz=True`` retains camera Z for geometric checks or 3D displays.
    Every meridian starts at +Z and ends at -Z. No icosahedral or triangle
    edges form part of this visible wire grid.
    """
    latitude_spacing_degrees = _positive_integer(latitude_spacing_degrees, 1, "latitude_spacing_degrees")
    longitude_spacing_degrees = _positive_integer(longitude_spacing_degrees, 1, "longitude_spacing_degrees")
    points_per_circle = _positive_integer(points_per_circle, 3, "points_per_circle")
    if latitude_spacing_degrees >= 90 or longitude_spacing_degrees >= 360:
        raise ValueError("Grid spacing must leave latitude circles and longitude meridians")
    lines = []
    phase = np.linspace(0., 2*np.pi, points_per_circle)
    positive_latitudes = np.arange(latitude_spacing_degrees, 90, latitude_spacing_degrees)
    latitude_degrees = np.concatenate((-positive_latitudes[::-1], [0], positive_latitudes))
    for degrees in latitude_degrees:
        latitude = np.radians(degrees)
        lines.append(np.column_stack((np.cos(latitude)*np.cos(phase),
                                       np.cos(latitude)*np.sin(phase),
                                       np.full(len(phase), np.sin(latitude)))))
    latitude = np.linspace(np.pi/2, -np.pi/2, points_per_circle)
    for degrees in np.arange(0, 360, longitude_spacing_degrees):
        longitude = np.radians(degrees)
        line = np.column_stack((np.cos(latitude)*np.cos(longitude),
                                np.cos(latitude)*np.sin(longitude), np.sin(latitude)))
        line[0], line[-1] = [0., 0., 1.], [0., 0., -1.]
        lines.append(line)
    if return_xyz:
        return lines
    projected, seen_latitudes = [], set()
    for degrees, line in zip(latitude_degrees, lines):
        key = abs(int(degrees))
        if key not in seen_latitudes:
            projected.append(line[:, :2])
            seen_latitudes.add(key)
    radius = np.linspace(0., 1., (points_per_circle+1)//2)
    for degrees in np.arange(0, 360, longitude_spacing_degrees):
        longitude = np.radians(degrees)
        projected.append(radius[:, None]*np.array([np.cos(longitude), np.sin(longitude)]))
    return projected


def latitude_longitude_density_mesh(matrix_camera: np.ndarray, latitude_steps: int = 48,
                                    longitude_steps: int = 96,
                                    log_normalizer: float | None = None) -> dict:
    """Evaluate the same exact axis-density formula on the display mesh.

    RGB is each object's X/Y/Z density divided by that channel's mesh maximum,
    exactly as in the prior display. This changes display geometry, not A,
    normalization, concentration, covariance, training or network predictions.
    """
    mesh = dict(latitude_longitude_mesh(latitude_steps, longitude_steps))
    vertex_logs = axis_direction_log_density(matrix_camera, mesh["vertices"], log_normalizer)
    face_logs = axis_direction_log_density(matrix_camera, mesh["centers"], log_normalizer)
    vertex_offsets, face_offsets = vertex_logs.max(axis=0), face_logs.max(axis=0)
    mesh.update({
        "A_camera": np.array(matrix_camera, dtype=float, copy=True),
        "vertex_log_density": vertex_logs, "face_log_density": face_logs,
        "vertex_rgb": np.exp(vertex_logs-vertex_offsets),
        "face_rgb": np.exp(face_logs-face_offsets),
        "vertex_color_log_offsets": vertex_offsets, "face_color_log_offsets": face_offsets,
        "diagnostics": {
            "geometry": "latitude-longitude unit sphere; camera +Z and -Z poles; XY equator",
            "latitude_steps": int(latitude_steps), "longitude_steps": int(longitude_steps),
            "density_formula": "pi/(2 Z(A)) exp(T) I0(hypot((h11-h22)/2,h12))",
            "density_provenance": dict(SPHERICAL_DENSITY_PROVENANCE),
            "normalization": ("absolute density per unit spherical area" if log_normalizer is not None
                              else "common -log Z(A) omitted from raw log-density"),
            "log_normalizer_s3": None if log_normalizer is None else float(log_normalizer),
            "axis_order": ["object +x", "object +y", "object +z"],
            "color_channels": {"red": "object +x", "green": "object +y", "blue": "object +z"},
            "color_mapping": "each density channel divided by its mesh maximum; display only",
            "direction_frame": "OpenCV camera: x right, y image-down, z forward",
            "projection": "camera_xy=(x,y); display increasing y downward; viewer looks along +z",
            "front_hemisphere": "camera z < 0; all far-hemisphere faces also retained",
            "vertex_count": int(len(mesh["vertices"])), "face_count": int(len(mesh["faces"])),
            "front_face_count": int(np.count_nonzero(mesh["front_faces"])),
            "far_face_count": int(np.count_nonzero(~mesh["front_faces"])),
            "limb_face_count": int(np.count_nonzero(mesh["limb_faces"])),
            "sphere_area": float(mesh["areas"].sum()),
            "matrix_modified": False, "monte_carlo": False,
        },
    })
    return mesh


@lru_cache(maxsize=4)
def _orthographic_pixel_directions(pixel_size: int) -> tuple[np.ndarray, np.ndarray]:
    pixel_size = _positive_integer(pixel_size, 8, "pixel_size")
    coordinate = -1.+(np.arange(pixel_size)+.5)*2./pixel_size
    x, y = np.meshgrid(coordinate, coordinate)
    inside = x*x+y*y <= 1.
    directions = np.column_stack((x[inside], y[inside], np.sqrt(1.-x[inside]**2-y[inside]**2)))
    inside.setflags(write=False)
    directions.setflags(write=False)
    return inside, directions


def camera_sphere_density_raster(matrix_camera: np.ndarray, pixel_size: int = 320,
                                  log_normalizer: float | None = None) -> dict:
    """Evaluate each orthographic pixel once on each spherical hemisphere.

    The two images share one maximum per object-axis density. Rasterization
    replaces overlapping translucent face painting; neither A nor the analytic
    S2 law changes. Pixels outside the unit disc are fully transparent.
    """
    inside, far_directions = _orthographic_pixel_directions(pixel_size)
    near_directions = far_directions.copy()
    near_directions[:, 2] *= -1.
    far_logs = axis_direction_log_density(matrix_camera, far_directions, log_normalizer)
    near_logs = axis_direction_log_density(matrix_camera, near_directions, log_normalizer)
    offsets = np.maximum(far_logs.max(axis=0), near_logs.max(axis=0))
    result = {"inside": inside, "far_directions": far_directions, "near_directions": near_directions,
              "far_log_density": far_logs, "near_log_density": near_logs,
              "color_log_offsets": offsets, "pixel_size": int(pixel_size)}
    for side, logs, fade in (("far", far_logs, .25), ("near", near_logs, 1.)):
        rgb = np.exp(logs-offsets)
        rgba = np.zeros((pixel_size, pixel_size, 4))
        rgba[inside, :3] = rgb
        rgba[inside, 3] = .9*rgb.max(axis=1)*fade
        result[side+"_rgba"] = rgba
    return result


def latitude_longitude_graticule_segments() -> dict[str, np.ndarray]:
    """Keep the wire depth while splitting each polyline into short segments.

    Camera Z>0 is the far hemisphere. Equatorial segments belong to the near
    layer. Front and back curves are not merged after their XY projection.
    """
    curves = latitude_longitude_graticule(return_xyz=True)
    segments = np.concatenate([np.stack((line[:-1], line[1:]), axis=1) for line in curves])
    far = segments[:, :, 2].mean(axis=1) > 0.
    return {"far": segments[far], "near": segments[~far]}


def draw_latitude_longitude_sphere(ax, mesh: dict, rotation: np.ndarray | None,
                                    colors, mode_linewidth: float = 1.8) -> dict:
    """Paint far support/density/wire, near layers, and opaque mode axes.

    The saved A is evaluated analytically at orthographic pixel centers.
    Each hemisphere is one RGBA image, avoiding translucent face overpainting.
    The two images share one density maximum per RGB channel; the mesh's raw
    RGB/log-density arrays remain unchanged. The far layer's opacity is
    uniformly one quarter of the near layer's opacity.
    Graticule segments retain camera Z until they are assigned to a layer,
    so the far wire stays behind the near sphere surface. Mode axes are
    fully opaque overlays, including axes pointing toward the far hemisphere.
    The returned artists expose their alpha and zorder for display tests.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    raster = camera_sphere_density_raster(
        mesh["A_camera"], log_normalizer=mesh["diagnostics"]["log_normalizer_s3"])
    segments = latitude_longitude_graticule_segments()
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=float)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("Mode rotation must be a finite 3 by 3 matrix")
    artists = {"raster_metadata": {
        "pixel_size": raster["pixel_size"],
        "color_log_offsets": raster["color_log_offsets"].tolist(),
        "extent": [-1., 1., 1., -1.],
        "color_mapping": "RGB_i = exp(log p_i - color_log_offsets_i); each axis shares one maximum across both raster hemispheres",
        "sample_coordinates_rule": "x=-1+(column+0.5)*2/pixel_size; y=-1+(row+0.5)*2/pixel_size; z=+/-sqrt(1-x*x-y*y); only x*x+y*y<=1 evaluated",
        "density_alpha": "0.9*max(RGB); far camera Z>0 multiplied by 0.25",
        "image_interpolation": "bilinear RGBA, clipped to unit circle; one image per hemisphere",
    }}
    for side, far, fade, zorder in (("far", True, .25, 1.), ("near", False, 1., 2.)):
        # An orthographic hemisphere projects to one exact unit disc. Painting
        # its neutral support once avoids translucent triangle edge overlaps.
        artists[side+"_support"] = Circle(
            (0., 0.), 1., facecolor=(.89, .89, .89, .14*fade), edgecolor="none", zorder=zorder)
        artists[side+"_density"] = ax.imshow(
            raster[side+"_rgba"], origin="upper", extent=(-1., 1., 1., -1.),
            interpolation="bilinear", zorder=zorder+.1)
        artists[side+"_density"].set_clip_path(Circle((0., 0.), 1., transform=ax.transData))
        artists[side+"_wire"] = LineCollection(
            segments[side][:, :, :2], colors=GRID_COLOR, alpha=GRID_ALPHA*fade,
            linewidths=GRID_LINEWIDTH, antialiaseds=True, zorder=zorder+.2)
        ax.add_patch(artists[side+"_support"])
        ax.add_collection(artists[side+"_wire"])
        rays = []
        if rotation is not None:
            for axis, color in enumerate(colors):
                if (rotation[2, axis] > 0.) == far:
                    line, = ax.plot([0., rotation[0, axis]], [0., rotation[1, axis]],
                                    color=color, linewidth=mode_linewidth, alpha=1., zorder=3.)
                    rays.append(line)
        artists[side+"_axes"] = rays
    return artists
