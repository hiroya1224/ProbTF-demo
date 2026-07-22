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
