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
COLORED_MESH = ASSETS / "mug_cylinder_centered_colored.ply"
MESH_METADATA = ASSETS / "mug_cylinder_center_metadata.json"
CYLINDER_CENTER_NATIVE = np.array([40.0, 40.0, 45.0])
MATERIAL_ROUGHNESS = .4
POINT_LIGHT_POSITION = np.array([0.0, .8, 2.7])
POINT_LIGHT_INTENSITY = 20.0
IMAGE_BLUR_SIGMA_360P = 0.0
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
    """Retain STL bytes and center the colored mesh on the cup cylinder axis.

    The verified cylinder axis is native X=40, Y=40.  Native Z=45 is the
    midpoint of the cylinder's 0..90 height.  It is a geometric rotation
    center, rather than a whole-mesh bounding-box center or a mass estimate.
    """
    import trimesh

    ASSETS.mkdir(parents=True, exist_ok=True)
    original = ASSETS / "mug_original.stl"
    if source.resolve() != original.resolve():
        if original.exists():
            if sha256(original) != sha256(source):
                raise FileExistsError("The existing original STL differs from source")
        else:
            shutil.copyfile(source, original)
    mesh = trimesh.load(original, force="mesh", process=True)
    bounds = mesh.bounds.copy()
    center = CYLINDER_CENTER_NATIVE.copy()
    scale = 1.0 / np.max(bounds[1] - bounds[0])
    mesh.vertices = (mesh.vertices - center) * scale
    # PLY vertex RGBA stores the selected solid material without STL's missing
    # material ambiguity.  The renderer reads this exact asset.
    mesh.visual.vertex_colors = np.tile(np.r_[COLOR_RGB, 255], (len(mesh.vertices), 1))
    colored = COLORED_MESH
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
        "origin_definition": "cup cylinder axis and native height midpoint",
        "cylinder_axis_native": {"point": center.tolist(), "direction": [0., 0., 1.]},
        "center_provenance": "Original STL inner/outer circular rings have common XY center (40,40); native height is 0..90",
        "native_full_bounds_center": bounds.mean(axis=0).tolist(),
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
    MESH_METADATA.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


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

        # Mesa may enumerate an inaccessible hardware device before llvmpipe
        # on the host. Select the software device explicitly for this fixed
        # headless renderer, while respecting an explicit device override.
        if (backend == "egl" and os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES") == mesa
                and "EGL_DEVICE_ID" not in os.environ):
            from pyrender.platforms.egl import query_devices
            for index, device in enumerate(query_devices()):
                if device.name is None:
                    os.environ["EGL_DEVICE_ID"] = str(index)
                    break

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
        self.mesh_metadata = json.loads(MESH_METADATA.read_text())
        if self.mesh_metadata["colored_mesh"] != COLORED_MESH.name:
            raise ValueError("Mesh metadata does not select the cylinder-centered asset")
        if self.mesh_metadata["colored_mesh_sha256"] != sha256(COLORED_MESH):
            raise ValueError("Colored mesh differs from its recorded SHA256")
        mesh = trimesh.load(COLORED_MESH, force="mesh")
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
            roughnessFactor=MATERIAL_ROUGHNESS, doubleSided=False)
        self.scene = pyrender.Scene(bg_color=[1., 1., 1., 1.],
                                    ambient_light=[.50, .50, .50])
        self.object_node = self.scene.add(pyrender.Mesh.from_trimesh(
            mesh, material=material, smooth=True))
        camera = pyrender.IntrinsicsCamera(self.focal, self.focal,
                                          width/2, height/2, znear=.01, zfar=10.)
        self.T_world_gl_camera = np.eye(4)
        self.T_world_gl_camera[2, 3] = self.distance
        self.scene.add(camera, pose=self.T_world_gl_camera)
        # A true positional source behind and above the camera, with ambient
        # fill and moderate surface roughness to keep the highlight broad.
        light_pose = np.eye(4)
        light_pose[:3, 3] = POINT_LIGHT_POSITION
        self.point_light_node = self.scene.add(pyrender.PointLight(
            color=np.ones(3), intensity=POINT_LIGHT_INTENSITY), pose=light_pose)
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
            "image_postprocess": "none; raw rasterizer RGB before saving or network input",
            "image_blur_sigma_pixels": IMAGE_BLUR_SIGMA_360P * self.height / DEFAULT_HEIGHT,
            "K": self.K.tolist(), "distance_object_units": self.distance,
            "vertical_fov_degrees": float(np.degrees(2*np.arctan(self.height/(2*self.focal)))),
            "t_object_to_camera": [0., 0., self.distance],
            "camera_coordinates": "OpenCV: x right, y down, z forward",
            "object_coordinates": "Original STL axes, translated by native cylinder center (40,40,45), uniformly scaled",
            "quaternion": "Hamilton wxyz, active object→OpenCV camera; q and -q equivalent",
            "pixel_projection": "u=fx*X_C/Z_C+cx; v=fy*Y_C/Z_C+cy",
            "pixel_coordinates": "Integer pixel centers; top-left center (0,0). K principal point=((width-1)/2,(height-1)/2). OpenGL viewport-edge cx/cy=(width/2,height/2).",
            "world_camera_rotation": R_CAMERA_WORLD.tolist(),
            "T_world_opengl_camera": self.T_world_gl_camera.tolist(),
            "object_center_world": [0., 0., 0.],
            "mesh": dict(self.mesh_metadata),
            "mesh_metadata_file": MESH_METADATA.name,
            "mesh_metadata_sha256": sha256(MESH_METADATA),
            "background": "white empty world, no ground plane or other geometry",
            "material_srgb_hex": COLOR_HEX,
            "material_roughness": MATERIAL_ROUGHNESS, "material_metallic": 0.,
            "ambient_light": [.50, .50, .50],
            "lights": "one fixed white point light behind/above camera; shadow maps disabled",
            "point_light_position_world": POINT_LIGHT_POSITION.tolist(),
            "point_light_intensity_candela": POINT_LIGHT_INTENSITY,
            "rasterizer": "pyrender OpenGL offscreen triangle rasterization, smooth vertex normals",
            "backend": os.environ.get("PYOPENGL_PLATFORM"),
            "egl_device_id": os.environ.get("EGL_DEVICE_ID"),
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
            "mesh": dict(renderer.mesh_metadata),
            "elapsed_seconds": time.monotonic()-started}
    (folder / "metadata.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def validate_output_metadata(output: Path, splits, overwrite: bool = False) -> None:
    """Reject split replacement or mixed mesh identities before any write."""
    for split in splits:
        if (output / split / "poses.npz").exists() and not overwrite:
            raise FileExistsError(f"{output / split} already has labels; use --overwrite")
    metadata_path = output / "render_metadata.json"
    if metadata_path.exists():
        recorded = json.loads(metadata_path.read_text())
        selected = json.loads(MESH_METADATA.read_text())
        if recorded.get("mesh", {}).get("colored_mesh_sha256") != selected["colored_mesh_sha256"]:
            raise FileExistsError("Output records a different mesh; choose a new dataset directory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, metavar="SOURCE_STL")
    parser.add_argument("--output", type=Path, default=HERE / "data/cylinder_center_no_blur_20261127")
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
    splits = DEFAULT_SPLITS if args.split == "all" else {args.split: DEFAULT_SPLITS[args.split]}
    validate_output_metadata(args.output, splits, args.overwrite)
    renderer = MugRenderer(width=args.width, height=args.height, backend=args.backend)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        metadata = renderer.metadata()
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
