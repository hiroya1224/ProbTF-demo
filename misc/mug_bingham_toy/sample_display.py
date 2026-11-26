"""Display-only Bingham samples and a camera-aligned latitude/longitude sphere.

The existing native Kent BACG sampler draws one shared set of orientations.
Each quaternion rotates all three object axes, preserving their joint frame.
Sampling is confined to visualization and never enters training, likelihood,
normalization, covariance evaluation, or checkpoint selection.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np

from figure_geometry import quaternion_matrix
from sphere_display import (FAR_OPACITY_RATIO, GRID_ALPHA, GRID_COLOR,
                            GRID_LINEWIDTH, latitude_longitude_graticule_segments)


SAMPLE_COUNT = 1000
DEFAULT_SEED = 20261127
FRONT_POINT_ALPHA = .7
MODE_LINEWIDTH = 1.8
AXIS_COLORS = ("#c33130", "#247b44", "#2b64b2")
REPO = Path(__file__).resolve().parents[2]
THESIS_SOURCE = "/home/leus/dthesis/chapter/3-mathematical-foundations.tex"


def _sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            checksum.update(block)
    return checksum.hexdigest()


@lru_cache(maxsize=1)
def _native_sampler():
    try:
        return importlib.import_module("probtf._sampling")
    except ImportError:
        # The source package extends its path to the existing catkin-built
        # extension.  This imports the established sampler without rebuilding
        # or modifying ProbTF core or the current training environment.
        source = str(REPO / "ros/core/probtf_core/src")
        if source not in sys.path:
            sys.path.insert(0, source)
        return importlib.import_module("probtf._sampling")


@lru_cache(maxsize=1)
def _sampler_provenance():
    native = _native_sampler()
    source = REPO / "ros/core/probtf_core/src/bingham_sampler.cpp"
    binding = REPO / "ros/core/probtf_core/src/python_sampling.cpp"
    return {
        "algorithm": "Kent BACG normalized-Gaussian rejection sampler, equations (3.4)-(3.6)",
        "thesis_source": THESIS_SOURCE,
        "thesis_algorithm": "math:alg:bingham-bacg",
        "thesis_direction_construction": "/home/leus/dthesis/chapter/4-transform-law-propagation.tex:2419; x_j=Rot(q_j)v",
        "native_extension": str(native.__file__),
        "native_extension_sha256": _sha256(native.__file__),
        "native_sampler_source": str(source),
        "native_sampler_source_sha256": _sha256(source),
        "native_binding_source": str(binding),
        "native_binding_source_sha256": _sha256(binding),
    }


def camera_axes_from_quaternions(quaternions):
    """Return R(q), with layout (sample, camera_xyz, object_axis).

    The three columns of every matrix are the corresponding X/Y/Z axis tips.
    Hamilton wxyz quaternions q and -q produce identical tips.
    """
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if (quaternions.ndim != 2 or quaternions.shape[1] != 4
            or not np.isfinite(quaternions).all()
            or np.any(np.linalg.norm(quaternions, axis=1) == 0.)):
        raise ValueError("quaternions must be a finite nonzero (N,4) Hamilton wxyz array")
    return quaternion_matrix(quaternions)


def sample_camera_axes(A_camera, count=SAMPLE_COUNT, seed=DEFAULT_SEED):
    """Draw shared visual orientations from the unchanged raw camera-frame A.

    The native sampler gets the original full matrix directly.  Its internal
    B=lambda_max(A)I-A conversion is the thesis algorithm's sign/gauge change,
    not a concentration adjustment.  A local native mt19937_64 stream is
    seeded directly and does not consume NumPy/Python/Torch training RNGs.
    No normalizer, tangent approximation, or independent S2 sampler is used.

    quaternions has shape (N,4), Hamilton wxyz.  tips has shape (N,3,3), with
    camera coordinates on axis 1 and the object's X/Y/Z directions on axis 2.
    Thus tips[:,:,k] is the N-by-3 cloud for object axis k.
    """
    matrix = np.asarray(A_camera, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("A_camera must be a finite symmetric 4 by 4 matrix")
    if not np.allclose(matrix, matrix.T, rtol=0.,
                       atol=1e-12*max(1., float(np.max(np.abs(matrix))))):
        raise ValueError("A_camera must be symmetric")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("count must be a positive integer")
    if (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
            or seed < 0 or seed >= 1 << 64):
        raise ValueError("seed must be an unsigned 64-bit integer")
    count, seed = int(count), int(seed)
    quaternions = np.empty((count, 4), dtype=np.float64)
    optimal_b, proposals = _native_sampler().sample_bingham(
        np.ascontiguousarray(matrix), quaternions, seed)
    tips = camera_axes_from_quaternions(quaternions)
    metadata = dict(_sampler_provenance(),
        seed=seed, count=count, quaternion_order="Hamilton wxyz",
        tips_layout="sample,camera_xyz,object_xyz_axis",
        direction_frame="OpenCV camera: x right, y image-down, z forward",
        optimal_b=float(optimal_b), proposal_count=int(proposals),
        accepted_count=count, acceptance_fraction=float(count/proposals),
        orientations_shared_by_all_three_axes=True,
        colored_axis_tip_count=3*count,
        sampling_only_visualization=True,
        matrix_modified=False,
        training_rng_consumed=False,
        native_seed_stream="local mt19937_64; fixed seed preserves accepted sample prefixes")
    return {"A_camera": matrix.copy(), "quaternions": quaternions, "tips": tips,
            "seed": seed, "count": count, "optimal_b": float(optimal_b),
            "proposal_count": int(proposals), "metadata": metadata}


def draw_sample_sphere(ax, packet, rotation=None, colors=AXIS_COLORS, *,
                       front_alpha=FRONT_POINT_ALPHA, marker_size=.8,
                       mode_linewidth=MODE_LINEWIDTH):
    """Paint the sampled axis clouds in the same camera coordinates as RGB.

    Camera Z>0 is rear: its points and wires use half the near alpha.  The
    existing Z-polar latitude/longitude helpers supply visible gray wires;
    no triangulation edges or unrelated oblique camera are added.  Mode axes
    are fully opaque top overlays, including rays pointing toward the rear.
    rotation=None draws no rays, preserving a caller's nonunique-mode state.
    Returned point artist lists are ordered object X, Y, Z.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    tips = np.asarray(packet["tips"], dtype=np.float64)
    if (tips.ndim != 3 or tips.shape[1:] != (3, 3)
            or not np.isfinite(tips).all()):
        raise ValueError("packet tips must have shape (N,3,3)")
    if not 0. <= front_alpha <= 1. or marker_size <= 0. or mode_linewidth <= 0.:
        raise ValueError("Display alpha, marker size and mode linewidth must be valid")
    if len(colors) != 3:
        raise ValueError("colors must contain the object X/Y/Z colors")
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("rotation must be a finite mode rotation matrix")
    segments = latitude_longitude_graticule_segments()
    artists = {"far_points": [], "near_points": [], "mode_axes": []}
    counts = {"far": [], "near": []}
    for side, far, fade, zorder in (("far", True, FAR_OPACITY_RATIO, 1.),
                                     ("near", False, 1., 2.)):
        artists[side+"_support"] = Circle((0., 0.), 1.,
            facecolor=(.89, .89, .89, .14*fade), edgecolor="none", zorder=zorder)
        ax.add_patch(artists[side+"_support"])
        artists[side+"_wire"] = LineCollection(
            segments[side][:, :, :2], colors=GRID_COLOR, alpha=GRID_ALPHA*fade,
            linewidths=GRID_LINEWIDTH, antialiaseds=True, zorder=zorder+.1)
        ax.add_collection(artists[side+"_wire"])
        for axis, color in enumerate(colors):
            points = tips[:, :, axis]
            selected = points[(points[:, 2] > 0.) == far]
            selected = selected[np.argsort(selected[:, 2])[::-1]]
            artist = ax.scatter(selected[:, 0], selected[:, 1], s=marker_size,
                color=color, alpha=front_alpha*fade, linewidths=0.,
                rasterized=True, zorder=zorder+.2)
            artists[side+"_points"].append(artist)
            counts[side].append(int(len(selected)))
    if rotation is not None:
        for axis, color in enumerate(colors):
            line, = ax.plot([0., rotation[0, axis]], [0., rotation[1, axis]],
                            color=color, linewidth=mode_linewidth, alpha=1., zorder=3.)
            artists["mode_axes"].append(line)
    ax.set(aspect="equal", xlim=(-1.07, 1.07), ylim=(1.07, -1.07))
    ax.axis("off")
    metadata = {
        "visualization": "shared Bingham quaternion samples pushed to the three object axes",
        "projection": "camera (x,y) directly; x right and y image-down; viewer looks along +z",
        "front_point_alpha": float(front_alpha),
        "rear_point_alpha": float(front_alpha*FAR_OPACITY_RATIO),
        "far_opacity_ratio": float(FAR_OPACITY_RATIO),
        "rear_definition": "camera z > 0; equator belongs to near layer",
        "marker_area_points2": float(marker_size),
        "near_point_count_per_axis": counts["near"],
        "rear_point_count_per_axis": counts["far"],
        "axis_colors": {name: color for name, color in zip(("x", "y", "z"), colors)},
        "wire_geometry": "latitude-longitude; camera +/-Z poles and XY equator",
        "wire_gray": GRID_COLOR, "wire_front_alpha": float(GRID_ALPHA),
        "wire_rear_alpha": float(GRID_ALPHA*FAR_OPACITY_RATIO),
        "wire_linewidth_pt": float(GRID_LINEWIDTH),
        "mode_axis_alpha": 1., "mode_axis_linewidth_pt": float(mode_linewidth),
        "mode_axes_on_top": True,
        "mode_axis_count": len(artists["mode_axes"]),
        "sample_or_matrix_modified_by_draw": False,
    }
    artists["metadata"] = metadata
    artists["display_metadata"] = metadata
    return artists
