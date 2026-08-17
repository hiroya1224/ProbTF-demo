# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `overlap_corrected`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- valid eigen samples: `27/27`

## Gain-scale sensitivity

| group | center | local linear 1-sigma | relative | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|
| xy | 1.1787878 | 0.00010225762 | 0.009% | 1.1787608 | 1.1794632 |
| z | 1.1789924 | 0.00015293091 | 0.013% | 1.1789451 | 1.1792373 |
| roll_pitch | 1.7244864 | 0.20588133 | 11.939% | 1.7176222 | 3.2522643 |
| yaw | 1.9977971 | 0.052337987 | 2.620% | 1.9818527 | 3.1410974 |

The `local linear 1-sigma` column is reconstructed from the
centered +/- eigen-direction evaluations. A large value means that
the corresponding static PID correction is sensitive to the
identified-plant ridge even when the point estimate itself is fixed.

## Dominant covariance directions

### xy

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | +7.8830333e-05 | 59.43% |
| 2 | -5.1848657e-05 | 25.71% |
| 3 | +2.4614028e-05 | 5.79% |
| 4 | +2.2494172e-05 | 4.84% |
| 9 | +1.1926938e-05 | 1.36% |

### z

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | +9.0288015e-05 | 34.86% |
| 1 | +8.9615659e-05 | 34.34% |
| 2 | -6.4626694e-05 | 17.86% |
| 4 | +4.5823919e-05 | 8.98% |
| 5 | +2.4237879e-05 | 2.51% |

### roll_pitch

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | +0.20532853 | 99.46% |
| 3 | +0.013745423 | 0.45% |
| 2 | -0.0047305309 | 0.05% |
| 1 | -0.0031623179 | 0.02% |
| 4 | -0.0024279313 | 0.01% |

### yaw

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | -0.031937627 | 37.24% |
| 1 | -0.030397135 | 33.73% |
| 0 | -0.02659395 | 25.82% |
| 3 | +0.0090251772 | 2.97% |
| 4 | -0.0023195147 | 0.20% |

## Monte Carlo

Disabled for this run. The deterministic 27-point eigen-direction
analysis is the primary result.

