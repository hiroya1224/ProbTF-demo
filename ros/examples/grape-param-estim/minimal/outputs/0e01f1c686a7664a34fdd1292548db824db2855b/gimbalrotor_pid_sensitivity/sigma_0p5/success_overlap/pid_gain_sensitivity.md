# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `overlap_corrected`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- valid eigen samples: `27/27`

## Gain-scale sensitivity

| group | center | local linear 1-sigma | relative | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|
| xy | 1.2145134 | 0.0001063908 | 0.009% | 1.2144673 | 1.2145591 |
| z | 1.2249249 | 4.4777693e-05 | 0.004% | 1.2249058 | 1.2249439 |
| roll_pitch | 4.1559028 | 0.015521991 | 0.373% | 4.1490245 | 4.162752 |
| yaw | 2.4909143 | 0.0075630512 | 0.304% | 2.4879048 | 2.4939158 |

The `local linear 1-sigma` column is reconstructed from the
centered +/- eigen-direction evaluations. A large value means that
the corresponding static PID correction is sensitive to the
identified-plant ridge even when the point estimate itself is fixed.

## Dominant covariance directions

### xy

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 5 | -9.176221e-05 | 74.39% |
| 1 | -3.0226296e-05 | 8.07% |
| 6 | +2.5246354e-05 | 5.63% |
| 2 | -2.4659232e-05 | 5.37% |
| 3 | -1.9609582e-05 | 3.40% |

### z

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 1 | -3.8046053e-05 | 72.19% |
| 6 | +1.5601037e-05 | 12.14% |
| 2 | -1.4139015e-05 | 9.97% |
| 5 | -9.7392489e-06 | 4.73% |
| 9 | -2.3884613e-06 | 0.28% |

### roll_pitch

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | -0.013727478 | 78.21% |
| 0 | -0.0053742111 | 11.99% |
| 5 | -0.0043598723 | 7.89% |
| 4 | +0.001347041 | 0.75% |
| 1 | +0.001272988 | 0.67% |

### yaw

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 3 | -0.0060110068 | 63.17% |
| 1 | +0.0034465411 | 20.77% |
| 4 | -0.0021393472 | 8.00% |
| 5 | -0.0020302969 | 7.21% |
| 2 | -0.00063350285 | 0.70% |

## Monte Carlo

Disabled for this run. The deterministic 27-point eigen-direction
analysis is the primary result.

