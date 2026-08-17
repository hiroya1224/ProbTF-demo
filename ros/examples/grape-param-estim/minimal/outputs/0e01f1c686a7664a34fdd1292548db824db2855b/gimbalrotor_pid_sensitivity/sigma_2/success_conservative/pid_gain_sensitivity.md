# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `conservative_fusion`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- valid eigen samples: `27/27`

## Gain-scale sensitivity

| group | center | local linear 1-sigma | relative | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|
| xy | 1.2145134 | 0.00089758283 | 0.074% | 1.2132787 | 1.2155419 |
| z | 1.2249249 | 0.00051425782 | 0.042% | 1.2239501 | 1.2257241 |
| roll_pitch | 4.1559028 | 0.0593639 | 1.428% | 4.0601768 | 4.2427137 |
| yaw | 2.4909143 | 0.065007757 | 2.610% | 2.3697361 | 2.6237748 |

The `local linear 1-sigma` column is reconstructed from the
centered +/- eigen-direction evaluations. A large value means that
the corresponding static PID correction is sensitive to the
identified-plant ridge even when the point estimate itself is fixed.

## Dominant covariance directions

### xy

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 4 | -0.00056580584 | 39.74% |
| 2 | -0.00048843712 | 29.61% |
| 6 | -0.00028319432 | 9.95% |
| 1 | +0.00028210306 | 9.88% |
| 9 | -0.0002048202 | 5.21% |

### z

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 4 | -0.00044350271 | 74.38% |
| 9 | -0.000205815 | 16.02% |
| 1 | +9.5761251e-05 | 3.47% |
| 2 | -8.0880939e-05 | 2.47% |
| 7 | -6.0635527e-05 | 1.39% |

### roll_pitch

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 3 | +0.045634228 | 59.09% |
| 6 | -0.028703516 | 23.38% |
| 2 | -0.01771464 | 8.90% |
| 0 | -0.011461239 | 3.73% |
| 5 | -0.0077799606 | 1.72% |

### yaw

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | -0.059634618 | 84.15% |
| 1 | +0.018292881 | 7.92% |
| 0 | +0.012525585 | 3.71% |
| 5 | -0.00953844 | 2.15% |
| 6 | -0.006646842 | 1.05% |

## Monte Carlo

Disabled for this run. The deterministic 27-point eigen-direction
analysis is the primary result.

