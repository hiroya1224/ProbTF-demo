"""Lazy pyrender mesh construction for one-sided textured marker planes."""

from typing import Iterable

import numpy as np

from .tag_texture import TagTexture


def create_tag_mesh(texture: TagTexture, marker_size_m: float):
    """Create a mesh whose coded square, excluding white margin, has the requested size."""
    try:
        import pyrender
    except ImportError as exc:
        raise RuntimeError("pyrender is required to construct render meshes") from exc
    plane_size = float(marker_size_m) * texture.plane_to_marker_scale
    half = 0.5 * plane_size
    positions = np.array(
        [[-half, -half, 0.0], [half, -half, 0.0],
         [half, half, 0.0], [-half, half, 0.0]], dtype=np.float32,
    )
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (4, 1))
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    image_texture = pyrender.Texture(source=texture.rgb, source_channels="RGB")
    material = pyrender.MetallicRoughnessMaterial(
        baseColorTexture=image_texture, metallicFactor=0.0, roughnessFactor=1.0,
        doubleSided=False, alphaMode="OPAQUE",
    )
    primitive = pyrender.Primitive(
        positions=positions, normals=normals, texcoord_0=uv,
        indices=indices, material=material,
    )
    return pyrender.Mesh([primitive], name="tag_{}_{}".format(texture.family, texture.marker_id))


def create_colored_quad(width_m: float, height_m: float,
                        color_rgb: Iterable[int]):
    try:
        import pyrender
    except ImportError as exc:
        raise RuntimeError("pyrender is required to construct render meshes") from exc
    width = 0.5 * float(width_m)
    height = 0.5 * float(height_m)
    positions = np.array(
        [[-width, -height, 0.0], [width, -height, 0.0],
         [width, height, 0.0], [-width, height, 0.0]], dtype=np.float32,
    )
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (4, 1))
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    color = np.asarray(color_rgb, dtype=np.float64) / 255.0
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[color[0], color[1], color[2], 1.0],
        metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True,
    )
    primitive = pyrender.Primitive(
        positions=positions, normals=normals, indices=indices, material=material,
    )
    return pyrender.Mesh([primitive], name="occluder")
