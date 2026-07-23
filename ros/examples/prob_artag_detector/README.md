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
- RViz with the debug image, TF, every IPPE mode, mode weights, tag planes, axes,
  and two-sigma conditional translation ellipsoids. These ellipsoids condition on
  each orientation mode; they are not the full translation marginal. A magenta
  ellipsoid and label indicate display clipping at `maximum_uncertainty_scale_m`.

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
the translation/rotation correlation inside every component. The separate
`Probabilistic Tag Modes` MarkerArray remains enabled because it labels the
branch weights and shows each branch's conditional translation covariance
explicitly. Neither visualization is fed back into estimation.
