# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `overlap_corrected`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- valid eigen samples: `27/27`

## Gain-scale sensitivity

| group | center | local linear 1-sigma | relative | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|
| xy | 1.1520448 | 0.0003772969 | 0.033% | 1.1515747 | 1.154526 |
| z | 1.1697648 | 9.4940763e-05 | 0.008% | 1.1696206 | 1.1698806 |
| roll_pitch | 3.5287743 | 0.034023237 | 0.964% | 3.4774616 | 3.76433 |
| yaw | 3.3788679 | 0.092844752 | 2.748% | 3.2753771 | 3.594216 |

The `local linear 1-sigma` column is reconstructed from the
centered +/- eigen-direction evaluations. A large value means that
the corresponding static PID correction is sensitive to the
identified-plant ridge even when the point estimate itself is fixed.

## Dominant covariance directions

### xy

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 4 | -0.00023028903 | 37.25% |
| 0 | +0.00017828208 | 22.33% |
| 3 | +0.00014582879 | 14.94% |
| 2 | +0.00011801973 | 9.78% |
| 1 | +0.00011703358 | 9.62% |

### z

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 1 | +6.5001041e-05 | 46.87% |
| 2 | -4.4100698e-05 | 21.58% |
| 4 | -3.1399079e-05 | 10.94% |
| 8 | -2.5759353e-05 | 7.36% |
| 3 | -2.0625947e-05 | 4.72% |

### roll_pitch

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | +0.023924092 | 49.44% |
| 3 | -0.019500478 | 32.85% |
| 1 | +0.012865595 | 14.30% |
| 0 | +0.0045533186 | 1.79% |
| 4 | +0.0042794175 | 1.58% |

### yaw

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | -0.079709729 | 73.71% |
| 2 | +0.041447939 | 19.93% |
| 1 | +0.016533106 | 3.17% |
| 3 | -0.016469322 | 3.15% |
| 4 | -0.0017502669 | 0.04% |

## Monte Carlo

Disabled for this run. The deterministic 27-point eigen-direction
analysis is the primary result.

