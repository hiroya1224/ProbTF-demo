# prob_artag_detector

ROS 1 wrapper and ROS-independent estimator for a calibrated single-camera
AprilTag observation. Each detection publishes one dynamic
`probtf_msgs/ProbabilisticTransformStamped` edge on `/probtf`.

The edge direction is explicit: `header.frame_id` is the OpenCV/ROS optical
camera frame and `child_frame_id` is `apriltag_<id>`. Therefore each component
maps marker coordinates into camera coordinates,
`z_C = R(Q) z_M + X` (`T_C_M`).

## Estimation contract

- OpenCV `ArucoDetector` supplies ID and corners in top-left, top-right,
  bottom-right, bottom-left order.
- The matching IPPE object-point order is
  `(-L/2,+L/2)`, `(+L/2,+L/2)`, `(+L/2,-L/2)`, `(-L/2,-L/2)`.
- Every `solvePnPGeneric(..., SOLVEPNP_IPPE_SQUARE)` candidate is refined using
  the same full 8-by-8 image-space Mahalanobis objective and the same proper,
  broad translation prior (`translation_prior_variance: 1.0e6` m2 by default).
- The local Gauss-Newton Hessian is split into `S`, `B`, rotational Schur
  precision, 3-by-9 `vec(R)` coupling `C`, and a right-chart Bingham law.
- Invalid covariance, failed cheirality, singular/non-SPD Hessians, and duplicate
  modes are rejected explicitly; no hidden covariance regularization is added.
- Laplace log masses include both the mode objective and local Hessian volume,
  then are converted to the linear component weights required by ProbTF v2.

Set `translation_prior_variance: null` only when an intentionally improper
translation law is required. `translation_prior_mean` remains a global camera-
frame mean shared by every IPPE branch; it is never centered separately on each
seed.

For nonzero OpenCV distortion coefficients the estimator uses a central finite
difference pose Jacobian. Undistorted pinhole cameras use the analytic
right-perturbation Jacobian and can enable finite-difference verification with
`verify_jacobian:=true`.

## Adaptive corner uncertainty

`config/default.yaml` treats `corner_sigma_px` as a covariance floor, not as the
complete error model.  With `adaptive_covariance: true`, each accepted tag is
redetected after a deterministic set of small subpixel, intensity-noise, and
blur perturbations.  The resulting ordered corner samples produce a full
8-by-8 covariance, including correlations between corners.  A weak, aliased, or
jagged edge therefore broadens the pose law even when the unperturbed detector
returns a sharp-looking quadrilateral.

The optional temporal stage uses the second difference of each tag's corner
track.  This cancels constant-velocity image motion and estimates residual
frame-to-frame localization jitter.  By default, coherent corner jitter also
widens the law; this is conservative when the camera or tag really
accelerates.  Set `temporal_freeze_affine_motion: true` to freeze innovations
explained by an affine warp, accepting that coherent detector jitter cannot
then be distinguished from physical motion.  It never averages old corners
into the reported position: the current frame remains the observation mean,
and history is applied as a positive-semidefinite covariance excess capped by
`temporal_max_excess_sigma_px`.
Invalid/non-positive spatial covariance is still rejected.  There is no way to
perfectly distinguish coherent detector bias from arbitrary physical motion
using one monocular corner track, so the same-frame bootstrap remains the
primary uncertainty source.  In particular, the constant-velocity second
difference is sensitive to frame-scale jitter and acceleration, but can remove
a slowly drifting coherent corner bias together with genuine smooth motion.

The debug-image label reports
`sigma=sqrt(trace(image_covariance)/8)` in pixels and the temporal update state
(`warmup`, `accepted`, `motion_frozen`, `outlier`, or `gap_reset`).  These are
compact diagnostics only; the estimator and ProbTF message retain the full
matrix.  To compare with the old fixed model, set both:

```yaml
adaptive_covariance: false
temporal_covariance_enabled: false
```

If a particular camera still looks overconfident, first raise
`bootstrap_noise_std` to match its observed intensity noise, then
`bootstrap_dither_px`; raise `corner_sigma_px` only when a larger unconditional
floor is justified.  `bootstrap_dropout_sigma_px` controls the penalty when a
perturbed re-detection disappears.  `bootstrap_samples` trades runtime for
stability (8 is the real-camera default).  `temporal_half_life_sec` controls
how quickly repeated jitter is forgotten.

The micro-bootstrap models corner localization uncertainty.  It does not model
an incorrect focal length, lens distortion, rolling shutter, tag-size error, or
motion blur outside the perturbation family.  Use a real camera calibration for
quantitative covariance.

## Run

```bash
roslaunch prob_artag_detector prob_artag_detector.launch \
  image_topic:=/camera/image_raw \
  camera_info_topic:=/camera/camera_info
```

The optional `~debug_image` overlay is for inspection only and is never fed back
into pose estimation.

## Sample tags

Ready-to-use `DICT_APRILTAG_36h11` tags are included under
[`samples/`](samples/README.md). For the exact marker used by the real-camera
smoke test, open `samples/apriltag_36h11_id_021.png`. The other shipped IDs are
0, 7, and 17.

These are AprilTag 36h11 markers, not ordinary ArUco markers or another
AprilTag family. Each PNG includes a one-module white quiet zone so it can be
displayed directly on a monitor or printed without adding a border.

## Real USB-camera demo

The package contains a self-contained OpenCV/V4L camera source, so the demo does
not require a separate camera-driver package:

```bash
source /home/leus/catkin_ws/devel/setup.bash
roslaunch prob_artag_detector prob_artag_real_camera_demo.launch \
  video_device:=/dev/video0
```

The launch starts all of the following in one command:

- `prob_artag_camera_demo_node.py`: USB camera to `Image`, plus `CameraInfo` when calibrated;
- `prob_artag_detector_node.py`: ordered corners to the full ProbTF mixture;
- `probtf_bridge`: highest-weight component mode to ordinary TF for display only;
- `probtf_rviz/ProbabilisticTF`: native joint transform-mixture samples for every latest tag edge;
- a REP-103 `camera_link` to `camera_optical_frame` static transform;
- RViz with the debug image, TF, every IPPE mode, mode weights, tag planes,
  ProbTF-owned representative axes, and two-sigma conditional translation
  ellipsoids. These ellipsoids condition on each orientation mode; they are not
  the full translation marginal. A magenta ellipsoid and label indicate display
  clipping at `maximum_uncertainty_scale_m`.

Useful launch overrides are:

```bash
roslaunch prob_artag_detector prob_artag_real_camera_demo.launch \
  video_device:=/dev/video2 \
  image_width:=1280 image_height:=720 framerate:=30 \
  config:=/absolute/path/to/my_detector.yaml
```

The detector's `tag_size_m` must equal the physical black-border square size of
the printed tag. Copy `config/default.yaml`, set the measured size there, and
pass that file with `config:=...`; an incorrect value directly scales the
estimated translation.

`/dev/v4l/by-id/...` is preferable to `/dev/videoN` when a stable device name is
available. Set `fourcc:=none` if a camera does not accept the default `MJPG` request.
The estimator never mirrors its raw input: image reflection would invalidate the
published pinhole model and usually makes the AprilTag code itself undecodable.

The default topics are:

| Topic | Type | Meaning |
|---|---|---|
| `/prob_artag_camera/image_raw` | `sensor_msgs/Image` | raw webcam frame |
| `/prob_artag_camera/camera_info` | `sensor_msgs/CameraInfo` | published only for a matching calibration YAML |
| `/prob_artag_camera/calibration_status` | `std_msgs/String` | calibrated/fallback provenance hint |
| `/prob_artag_demo/probtf` | `probtf_msgs/ProbabilisticTransformStamped` | complete pose mixture |
| `/prob_artag_detector/debug_image` | `sensor_msgs/Image` | corners, all retained axes, calibration source |
| `/prob_artag_detector/markers` | `visualization_msgs/MarkerArray` | all modes and their local uncertainty |
| `/tf` | `tf2_msgs/TFMessage` | highest-weight mode, for RViz compatibility only |

### Calibration and fallback

Pass a normal ROS camera-calibration YAML when one is available:

```bash
roslaunch prob_artag_detector prob_artag_real_camera_demo.launch \
  camera_info_url:=file:///absolute/path/to/camera.yaml
```

Plain filesystem paths and `package://package/path.yaml` URLs are also accepted.
Only `plumb_bob` and `rational_polynomial` models are consumed by this OpenCV
projection path; fisheye/equidistant calibration is rejected explicitly. The
input must be the full, unbinned, unrectified `image_raw` matching the YAML.
Nontrivial CameraInfo binning or ROI is rejected instead of silently reusing an
incorrect intrinsic matrix.

When the file is absent, unreadable, or has a resolution different from the
captured frame, the camera node deliberately does not publish an inferred model
as ordinary `CameraInfo`. Instead, the detector constructs a zero-distortion
pinhole fallback from the actual image size and `fallback_horizontal_fov_deg`
(60 degrees by default). This also handles an external driver whose uncalibrated
CameraInfo contains a zero matrix. The camera node logs a prominent warning and
the debug image reports `calibration=fallback...`. This is suitable for bringing
up the pipeline, but its metric depth and covariance are only approximate;
calibrate the real camera before quantitative use.

### Using the standard `usb_cam` package instead

The detector remains camera-driver agnostic. If `usb_cam` is installed, start it
in one terminal:

```bash
sudo apt install ros-noetic-usb-cam
rosrun usb_cam usb_cam_node \
  _video_device:=/dev/video0 \
  _camera_frame_id:=camera_optical_frame
```

Then disable the built-in source in a second terminal:

```bash
roslaunch prob_artag_detector prob_artag_real_camera_demo.launch \
  start_camera:=false \
  image_topic:=/usb_cam/image_raw \
  camera_info_topic:=/usb_cam/camera_info
```

If that driver already publishes `camera_link -> camera_optical_frame`, also set
`publish_camera_tf:=false` to avoid duplicate static TF publishers.

Fallback records include a `camera_model:fallback_hfov_...` source ID and an
explicit warning that intrinsic uncertainty is not represented in the local
Hessian covariance. The demo uses the scoped `/prob_artag_demo/probtf` topic so
an approximate fallback cannot silently contaminate an unrelated global graph.
Pass `probtf_topic:=/probtf` explicitly when integration with the global ProbTF
graph is intended.

The ordinary tag TF is intentionally a lossy display projection selected with
`tf_export_policy=highest_weight_component_mode`. The configured ProbTF topic
(`/prob_artag_demo/probtf` by default here) and RViz markers retain every accepted
planar branch; downstream probabilistic consumers must use that ProbTF record,
not the representative `/tf` edge.

The RViz `Probabilistic AprilTags` display subscribes directly to the scoped
ProbTF stream and keeps the latest edge for every detected tag ID. Its RGB
clouds use full joint transform samples, preserving discrete IPPE branches and
the translation/rotation correlation inside every component. Its `Axis Length`
property controls both the cloud endpoints and the central representative
axes. To avoid a fixed-length axis being drawn on top, ordinary TF axes are
hidden and MarkerArray mode axes are not published by default. Set
`publish_mode_axes: true` to restore the latter as a separate fixed-length
diagnostic using `marker_axis_length_m`. The `Probabilistic Tag Modes`
MarkerArray remains enabled for branch weights, tag planes, and each branch's
conditional translation covariance. Neither visualization is fed back into
estimation.
