# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `conservative_fusion`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- valid eigen samples: `27/27`

## Gain-scale sensitivity

| group | center | local linear 1-sigma | relative | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|
| xy | 1.1787878 | 0.001374904 | 0.117% | 1.1776889 | 1.1800089 |
| z | 1.1789924 | 0.0014436748 | 0.122% | 1.1779002 | 1.1800855 |
| roll_pitch | 1.7244864 | 0.31216716 | 18.102% | 1.6685932 | 2.8067449 |
| yaw | 1.9977971 | 0.20962051 | 10.493% | 0.15142721 | 2.1985612 |

The `local linear 1-sigma` column is reconstructed from the
centered +/- eigen-direction evaluations. A large value means that
the corresponding static PID correction is sensitive to the
identified-plant ridge even when the point estimate itself is fixed.

## Dominant covariance directions

### xy

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 8 | +0.001099122 | 63.91% |
| 7 | +0.00067354997 | 24.00% |
| 3 | +0.00027322806 | 3.95% |
| 0 | +0.00025722351 | 3.50% |
| 2 | -0.00018322854 | 1.78% |

### z

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 8 | +0.001092609 | 57.28% |
| 7 | +0.0006674185 | 21.37% |
| 1 | +0.00048640409 | 11.35% |
| 2 | -0.00025881395 | 3.21% |
| 3 | +0.00021289956 | 2.17% |

### roll_pitch

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | +0.30672376 | 96.54% |
| 3 | +0.056510913 | 3.28% |
| 2 | -0.012431 | 0.16% |
| 4 | +0.0034825091 | 0.01% |
| 1 | -0.0025568814 | 0.01% |

### yaw

| direction | 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 1 | -0.13944897 | 44.26% |
| 0 | -0.11864429 | 32.04% |
| 2 | -0.094354495 | 20.26% |
| 3 | +0.037882161 | 3.27% |
| 5 | +0.007181948 | 0.12% |

## Optional Monte Carlo stress test

| group | mean | std | q16 | median | q84 | q2.5 | q97.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| xy | 1.1793662 | 0.0014617213 | 1.1779967 | 1.1793165 | 1.1808896 | 1.1766029 | 1.182296 |
| z | 1.1789841 | 0.0014346036 | 1.1775729 | 1.1790403 | 1.180443 | 1.1762412 | 1.1816439 |
| roll_pitch | 2.6516655 | 0.51839328 | 2.0741437 | 2.6742137 | 3.1511483 | 1.7459976 | 3.7013694 |
| yaw | 3.2774117 | 2.5898885 | 0.49387266 | 2.711135 | 6.0111614 | 0.080522323 | 8.9298295 |

Monte Carlo here is a nonlinear stress test of the selected local
covariance model. Do not relabel these quantiles as posterior
credible intervals without an independent probabilistic
calibration argument.

