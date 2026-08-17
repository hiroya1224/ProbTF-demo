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
| xy | 1.1787878 | 0.00014364362 | 0.012% | 1.178668 | 1.1800572 |
| z | 1.1789924 | 0.00015592236 | 0.013% | 1.1787732 | 1.1793553 |
| roll_pitch | 1.7244864 | 0.17479472 | 10.136% | 1.6971316 | 2.8368088 |
| yaw | 1.9977971 | 0.075274336 | 3.768% | 0.12295845 | 2.0839516 |

The `local linear 1-sigma` column is reconstructed from the
centered +/- eigen-direction evaluations. A large value means that
the corresponding static PID correction is sensitive to the
identified-plant ridge even when the point estimate itself is fixed.

## Dominant covariance directions

### xy

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | +0.00012812765 | 79.56% |
| 2 | -5.1861672e-05 | 13.04% |
| 3 | +2.4717221e-05 | 2.96% |
| 4 | +2.2485142e-05 | 2.45% |
| 9 | +1.1926938e-05 | 0.69% |

### z

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | +9.5181257e-05 | 37.26% |
| 1 | +8.9701587e-05 | 33.10% |
| 2 | -6.4636196e-05 | 17.18% |
| 4 | +4.582091e-05 | 8.64% |
| 5 | +2.4238319e-05 | 2.42% |

### roll_pitch

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | +0.1741407 | 99.25% |
| 3 | +0.013744967 | 0.62% |
| 2 | -0.0047479008 | 0.07% |
| 1 | -0.0032776371 | 0.04% |
| 4 | -0.0024278311 | 0.02% |

### yaw

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | -0.059322371 | 62.11% |
| 1 | -0.032262032 | 18.37% |
| 2 | -0.031909428 | 17.97% |
| 3 | +0.0090253542 | 1.44% |
| 4 | -0.002319453 | 0.09% |

## Monte Carlo

Disabled for this run. The deterministic 27-point eigen-direction
analysis is the primary result.

