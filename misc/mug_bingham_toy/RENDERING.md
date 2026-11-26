# Rendering and exact pose labels

The renderer reads `assets/mug_centered_colored.ply`. Its solid vertex RGBA is
`(188, 21, 67, 255)`, corresponding to sRGB `#BC1543`. The original source STL is
retained byte-for-byte as `assets/mug_original.stl`, with SHA-256
`b594ed7482a3e1148de7407d6b690c03a9a0b44a3d5d1d06b62e4e5c9bc243eb`.
STL has no reliable physical unit or material, so neither is inferred from it.

The original bounds are `[0,0,0]` to `[112.11322021484375,80,90]`. Only a fixed
translation and positive uniform scale are applied:

```text
v_object = 0.008919554697329088 * (v_native - [56.056610107421875,40,45])
```

The native axes are preserved. No opening axis is guessed or rotated into a
preferred orientation. The longest object extent is 1, and the bounding radius
is approximately 0.637481. The material is nonmetallic, roughness 0.65. The
renderer converts the asset's selected sRGB color to linear RGB for the glTF
material; illumination produces ordinary shading rather than constant pixels.

The empty world background is pure white. The object center remains at the
world origin; the camera remains at `[0,0,2.5]` looking toward the origin. At
640×360 pixels, focal length is 590.625 pixels in both axes. The vertical field
of view is approximately 33.9 degrees. Bright white ambient
fill and four fixed directional lights above/behind the camera illuminate the
rim and interior when the opening is visible. Their cluster gives broad
shading; shadow maps are disabled, so this is deliberately simple illumination,
not a physically accurate area-light simulation. A camera view directed at the
bottom of an opaque mug naturally cannot show its interior.

`q_wxyz[i]` is the active object-to-OpenCV-camera rotation, with Hamilton
quaternions in scalar-first order. OpenCV camera axes are x right, y down,
z forward. Translation is always `[0,0,2.5]` in normalized object units. The
normalized source mesh coordinates are the object frame used for all labels.
The camera/world axis rotation is `R_camera_world = diag(1,-1,-1)`.
The actual object node receives
`R_world_object = R_camera_world.T @ R(q_wxyz)`, so labels describe the actual
rendered object-to-camera transform. A quaternion and its negative produce
the same image.

The saved `K` uses integer pixel centers, so its principal point is
`(319.5,179.5)` at resolution 640×360. Pyrender receives the equivalent OpenGL
viewport-edge principal point `(320,180)`. Projection is
`u=fx*X_camera/Z_camera+cx`, `v=fy*Y_camera/Z_camera+cy` with the saved `K`.

The fixed camera and lights apply to every split. Orientations are independent
normalized four-dimensional standard-normal vectors, uniform on S³ and hence
Haar-uniform on SO(3). Both quaternion signs are retained. Split settings were
fixed before training:

| Split | Images | PCG64 seed |
| --- | ---: | ---: |
| train | 10,000 | 1234 |
| val | 1,000 | 1235 |
| test | 1,000 | 1236 |

Each split has `images/000000.png`, etc., and `poses.npz` with float64
`q_wxyz[N,4]`, int64 `index[N]`, float64 `t_camera[N,3]`, and an int64 scalar
`seed`. `metadata.json` records count, label hash, concatenated image-file hash,
and generation time. `data/render_metadata.json` records geometry, camera,
material, rendering stack, driver selection, coordinate conventions, and code
hash. Pixel reproducibility is guaranteed only on the same rendering stack;
GPU drivers can change rasterization rounding.

Image width and height are explicit renderer settings (`--width`, `--height`),
defaulting to 640 and 360. Focal length scales with image height to retain the
same vertical field of view at any resolution; the horizontal field of view
follows the chosen aspect ratio. Geometry and renderer consistency tests use
an explicit 128×128 fixture for speed and also verify the default 360p output.

Generate the dataset with a Python interpreter containing numpy, trimesh,
pyrender, and Pillow:

```bash
/usr/bin/python3 misc/mug_bingham_toy/render.py \
  --output misc/mug_bingham_toy/data
```

The asset preparation command, needed only when reconstructing the local
assets from the originally supplied STL, is:

```bash
/usr/bin/python3 misc/mug_bingham_toy/render.py \
  --prepare /home/leus/Downloads/mug.STL \
  --output /tmp/mug-render-preview --split train --count 12 --seed 1234
```

The default headless renderer is Mesa EGL. Explicit environment selections are
respected; `--backend osmesa` can select OSMesa if installed. No display or
Blender application is required. Meaningful source, geometry, quaternion,
projection, antipodal, and saved-label/pixel checks run with:

```bash
/usr/bin/python3 -m unittest discover \
  -s misc/mug_bingham_toy/tests -p test_render.py -v
```
