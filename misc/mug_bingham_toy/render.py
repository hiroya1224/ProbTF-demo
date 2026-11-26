#!/usr/bin/env python3
"""Fixed-camera mug renders and exact Hamilton quaternion pose labels.

Only dataset pose generation uses random numbers.  Rendering is a conventional
triangle rasterizer (pyrender/OpenGL), with no randomized lights or backgrounds.
Run with system Python, or a venv that includes numpy, trimesh, pyrender, Pillow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
COLOR_HEX = "#BC1543"
COLOR_RGB = np.array([188, 21, 67], dtype=np.uint8)
DEFAULT_SPLITS = {"train": (10000, 1234),
                  "val": (1000, 1235),
                  "test": (1000, 1236)}
R_CAMERA_WORLD = np.diag([1.0, -1.0, -1.0])
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 360
FOCAL_OVER_HEIGHT = 1.640625


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quaternion_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """Active object→destination rotation; Hamilton product, scalar first."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError("Quaternion shape must end in four")
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
                     2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
                     2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)),
                    axis=-1).reshape(q.shape[:-1] + (3, 3))


def uniform_quaternions(count: int, seed: int) -> np.ndarray:
    """Haar SO(3) from uniform S³, preserving both quaternion signs."""
    q = np.random.Generator(np.random.PCG64(seed)).standard_normal((count, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def prepare_assets(source: Path) -> dict:
    """Retain the original STL bytes and write an explicitly colored PLY."""
    import trimesh

    ASSETS.mkdir(parents=True, exist_ok=True)
    original = ASSETS / "mug_original.stl"
    if source.resolve() != original.resolve():
        if original.exists() and sha256(original) != sha256(source):
            raise FileExistsError("The existing original STL differs from source")
        shutil.copyfile(source, original)
    mesh = trimesh.load(original, force="mesh", process=True)
    bounds = mesh.bounds.copy()
    center = bounds.mean(axis=0)
    scale = 1.0 / np.max(bounds[1] - bounds[0])
    mesh.vertices = (mesh.vertices - center) * scale
    # PLY vertex RGBA stores the selected solid material without STL's missing
    # material ambiguity.  The renderer reads this exact asset.
    mesh.visual.vertex_colors = np.tile(np.r_[COLOR_RGB, 255], (len(mesh.vertices), 1))
    colored = ASSETS / "mug_centered_colored.ply"
    mesh.export(colored, encoding="binary_little_endian")
    metadata = {
        "source_filename": source.name,
        "original_stl": original.name,
        "original_sha256": sha256(original),
        "colored_mesh": colored.name,
        "colored_mesh_sha256": sha256(colored),
        "native_bounds": bounds.tolist(),
        "native_units": "unspecified STL length units; no physical unit inferred",
        "center_native": center.tolist(),
        "scale_native_to_object": float(scale),
        "coordinate_transform": "v_object = scale * (v_native - center_native)",
        "axes": "Original STL axes preserved; no semantic-axis rotation",
        "object_max_extent": 1.0,
        "object_bounding_radius": float(np.linalg.norm(mesh.vertices, axis=1).max()),
        "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "material_srgb_hex": COLOR_HEX,
        "material_srgb_rgba": [188, 21, 67, 255],
    }
    (ASSETS / "mesh_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _look_at_light(position: np.ndarray) -> np.ndarray:
    """OpenGL light local -z points toward object origin."""
    z = np.asarray(position, dtype=np.float64)
    z = z / np.linalg.norm(z)
    x = np.cross([0., 1., 0.], z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((x, y, z))
    pose[:3, 3] = position
    return pose


class MugRenderer:
    """White world, fixed optical camera, colored object at a fixed center."""
    def __init__(self, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT,
                 backend: str = "egl"):
        # Select before importing pyrender/OpenGL.  Explicit user overrides win.
        os.environ.setdefault("PYOPENGL_PLATFORM", backend)
        mesa = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
        if backend == "egl" and Path(mesa).exists():
            os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", mesa)
        import trimesh
        import pyrender

        self.pyrender = pyrender
        if width < 1 or height < 1:
            raise ValueError("Image width and height must be positive")
        self.width = width
        self.height = height
        self.distance = 2.5
        self.focal = FOCAL_OVER_HEIGHT * height
        # OpenGL viewport edges are integer coordinates, while image pixel
        # centers below use integer indices.  The symmetric image center is
        # therefore (dimension-1)/2; pyrender uses dimension/2 at pixel edges.
        self.K = np.array([[self.focal, 0., (width-1)/2],
                           [0., self.focal, (height-1)/2], [0., 0., 1.]])
        mesh = trimesh.load(ASSETS / "mug_centered_colored.ply", force="mesh")
        colors = np.asarray(mesh.visual.vertex_colors)
        if not np.all(colors == np.array([188, 21, 67, 255])):
            raise ValueError("Colored mesh material no longer matches #BC1543")
        # glTF baseColorFactor is linear RGB.  Remove PLY display colors before
        # material creation to avoid multiplying the material by the same color.
        srgb = colors[0, :3] / 255.0
        linear = np.where(srgb <= .04045, srgb / 12.92,
                          ((srgb + .055) / 1.055)**2.4)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=np.r_[linear, 1.], metallicFactor=0.,
            roughnessFactor=.65, doubleSided=False)
        self.scene = pyrender.Scene(bg_color=[1., 1., 1., 1.],
                                    ambient_light=[.50, .50, .50])
        self.object_node = self.scene.add(pyrender.Mesh.from_trimesh(
            mesh, material=material, smooth=True))
        camera = pyrender.IntrinsicsCamera(self.focal, self.focal,
                                          width/2, height/2, znear=.01, zfar=10.)
        self.T_world_gl_camera = np.eye(4)
        self.T_world_gl_camera[2, 3] = self.distance
        self.scene.add(camera, pose=self.T_world_gl_camera)
        # A deterministic broad cluster behind/above the camera approximates a
        # soft source; ambient fill illuminates the mouth in visible viewpoints.
        lights = [([-.9, 1.0, 3.5], .45), ([.9, 1.0, 3.5], .45),
                  ([-.9, 2.0, 3.5], .45), ([.9, 2.0, 3.5], .45)]
        for position, intensity in lights:
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
            self.scene.add(light, pose=_look_at_light(np.array(position)))
        self.offscreen = pyrender.OffscreenRenderer(width, height)

    def world_object_pose(self, q_object_to_camera: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = R_CAMERA_WORLD.T @ quaternion_matrix(q_object_to_camera)
        return pose

    def render(self, q_object_to_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.scene.set_pose(self.object_node, self.world_object_pose(q_object_to_camera))
        return self.offscreen.render(self.scene, flags=self.pyrender.RenderFlags.NONE)

    def close(self):
        self.offscreen.delete()

    def metadata(self) -> dict:
        import pyrender
        import trimesh
        return {
            "image_size": [self.width, self.height], "image_size_order": "width,height",
            "channels": "RGB uint8",
            "K": self.K.tolist(), "distance_object_units": self.distance,
            "vertical_fov_degrees": float(np.degrees(2*np.arctan(self.height/(2*self.focal)))),
            "t_object_to_camera": [0., 0., self.distance],
            "camera_coordinates": "OpenCV: x right, y down, z forward",
            "object_coordinates": "Centered/scaled original STL axes",
            "quaternion": "Hamilton wxyz, active object→OpenCV camera; q and -q equivalent",
            "pixel_projection": "u=fx*X_C/Z_C+cx; v=fy*Y_C/Z_C+cy",
            "pixel_coordinates": "Integer pixel centers; top-left center (0,0). K principal point=((width-1)/2,(height-1)/2). OpenGL viewport-edge cx/cy=(width/2,height/2).",
            "world_camera_rotation": R_CAMERA_WORLD.tolist(),
            "T_world_opengl_camera": self.T_world_gl_camera.tolist(),
            "object_center_world": [0., 0., 0.],
            "background": "white empty world, no ground plane or other geometry",
            "material_srgb_hex": COLOR_HEX,
            "material_roughness": .65, "material_metallic": 0.,
            "ambient_light": [.50, .50, .50],
            "lights": "four fixed white directional lights behind/above camera, intensity .45 each; shadow maps disabled",
            "rasterizer": "pyrender OpenGL offscreen triangle rasterization, smooth vertex normals",
            "backend": os.environ.get("PYOPENGL_PLATFORM"),
            "egl_vendor": os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"),
            "numpy_version": np.__version__, "pyrender_version": pyrender.__version__,
            "trimesh_version": trimesh.__version__,
            "randomness": "PCG64 standard_normal 4-vectors normalized onto S³; independent split seeds; no image/light randomization",
            "pixel_reproducibility": "Exact on the same software/driver stack; other GPU drivers may change rasterization rounding",
            "source_code_sha256": sha256(Path(__file__)),
        }


def render_split(renderer: MugRenderer, output: Path, split: str,
                 count: int, seed: int, overwrite: bool = False) -> dict:
    from PIL import Image

    folder = output / split
    images = folder / "images"
    if (folder / "poses.npz").exists() and not overwrite:
        raise FileExistsError(f"{folder} already has labels; use --overwrite")
    images.mkdir(parents=True, exist_ok=True)
    q = uniform_quaternions(count, seed)
    index = np.arange(count, dtype=np.int64)
    started = time.monotonic()
    image_hash = hashlib.sha256()
    for i, pose in enumerate(q):
        rgb, _ = renderer.render(pose)
        path = images / f"{i:06d}.png"
        Image.fromarray(rgb, "RGB").save(path, compress_level=3)
        image_hash.update(path.read_bytes())
        if (i+1) % 1000 == 0 or i+1 == count:
            print(f"{split}: {i+1}/{count}, {time.monotonic()-started:.1f}s", flush=True)
    translations = np.tile([0., 0., renderer.distance], (count, 1))
    np.savez_compressed(folder / "poses.npz", q_wxyz=q, index=index,
                        t_camera=translations, seed=np.array(seed, dtype=np.int64))
    info = {"split": split, "count": count, "seed": seed,
            "images": "images/{index:06d}.png", "labels": "poses.npz",
            "label_arrays": {"q_wxyz": "float64 [N,4]", "index": "int64 [N]",
                             "t_camera": "float64 [N,3]", "seed": "int64 scalar"},
            "poses_sha256": sha256(folder / "poses.npz"),
            "images_concat_sha256": image_hash.hexdigest(),
            "elapsed_seconds": time.monotonic()-started}
    (folder / "metadata.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, metavar="SOURCE_STL")
    parser.add_argument("--output", type=Path, default=HERE / "data")
    parser.add_argument("--split", choices=["all", *DEFAULT_SPLITS], default="all")
    parser.add_argument("--count", type=int, help="override count for one split")
    parser.add_argument("--seed", type=int, help="override seed for one split")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--backend", choices=["egl", "osmesa"], default="egl")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare_assets(args.prepare), indent=2))
    if args.count is not None and args.split == "all":
        parser.error("--count requires one --split")
    if args.seed is not None and args.split == "all":
        parser.error("--seed requires one --split")
    renderer = MugRenderer(width=args.width, height=args.height, backend=args.backend)
    args.output.mkdir(parents=True, exist_ok=True)
    splits = DEFAULT_SPLITS if args.split == "all" else {args.split: DEFAULT_SPLITS[args.split]}
    results = []
    try:
        metadata = renderer.metadata()
        metadata["mesh"] = json.loads((ASSETS / "mesh_metadata.json").read_text())
        (args.output / "render_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for split, (default_count, default_seed) in splits.items():
            results.append(render_split(renderer, args.output, split,
                           args.count if args.count is not None else default_count,
                           args.seed if args.seed is not None else default_seed,
                           args.overwrite))
    finally:
        renderer.close()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
