# Probabilistic AprilTag synthetic renderer

This standalone Python package generates deliberately simple AprilTag images
and exact geometry fixtures for the probabilistic detector.  RGB rendering and
ground truth are separate paths: `corners_px` is always projected from the
saved transforms, never recovered from rendered pixels.

## Conventions

- `T_A_B` maps coordinates from frame B into frame A.
- `C` is an OpenCV/ROS optical camera: x right, y down, z forward.
- A tag has x toward its printed right, y toward its printed top, and z along
  its front-face normal.
- `SOLVEPNP_IPPE_SQUARE` corners are TL, TR, BR, BL:
  `(-s/2,+s/2)`, `(+s/2,+s/2)`, `(+s/2,-s/2)`, `(-s/2,-s/2)`.
- pyrender's OpenGL camera conversion is isolated in `coordinates.py` as
  `T_W_GL = T_W_C @ diag(1,-1,-1,1)`.

`size_m` is the coded black-bordered square.  The white detection margin lies
outside it, so the rendered support plane is correspondingly larger.

## Install and run

```bash
cd src/ProbTF-demo/misc/prob_artag_renderer
python3 -m pip install -e '.[render,test]'
prob-artag-render-single --output /tmp/tag-one --seed 7
prob-artag-render-scene --output /tmp/tag-scene --seed 7 --count 4
prob-artag-generate-dataset --output /tmp/tag-sequence --seed 7 \
  --scenario multi_tag --frames 10
pytest -q
```

Direct scripts with the same behavior are under `scripts/`.

The default headless backend is EGL.  Before importing pyrender the package
sets `PYOPENGL_PLATFORM=egl`.  On GLVND systems it also selects
`/usr/share/glvnd/egl_vendor.d/50_mesa.json` when present; an existing
`__EGL_VENDOR_LIBRARY_FILENAMES` value is always preserved.  Override either
setting explicitly when a different GPU vendor is required, or select
`--backend osmesa` where OSMesa is installed.

## Dataset layout

Each generated frame contains:

```text
frames/000000/
  rgb.png
  depth.npy
  instance_id.png
  metadata.json
```

Depth is metric `float32`; instance zero is background and positive values map
to the tag entries in metadata.  Metadata includes camera calibration,
`T_W_C`, `T_W_M`, `T_C_M`, ordered image corners, corner depths, projected
size, front-facing state, and rendered visible fraction.

The six sampler scenarios are `frontal`, `moderate`, `oblique`, `small`,
`occluded`, and `multi_tag`.  A fixed seed reproduces poses, IDs, annotations,
degradation noise, and (on the same rendering stack) pixels.  Sequence mode
keeps the world tags fixed while moving the camera smoothly.

All eight degradations in `config/default_scene.yaml` are independent:
Gaussian noise, defocus blur, motion blur, brightness/contrast, radial
distortion, partial occlusion, background clutter, and rolling shutter.  The
radial model is reflected in the saved camera model and projected corners;
rolling shutter is recorded as an image-space degradation because it has no
single global-shutter pose.
