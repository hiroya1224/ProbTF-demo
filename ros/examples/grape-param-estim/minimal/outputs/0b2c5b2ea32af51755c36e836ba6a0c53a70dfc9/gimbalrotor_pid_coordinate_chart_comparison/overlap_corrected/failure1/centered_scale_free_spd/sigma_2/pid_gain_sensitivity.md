# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `overlap_corrected`
- coordinate mode: `centered_scale_free_spd`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- finite sigma multiple: `2`
- local derivative sigma fraction: `1e-05`
- valid finite eigen samples: `26/27`
- valid local-derivative directions: `13/13`

## Gain-scale sensitivity

| group | center | infinitesimal 1-sigma | relative | finite secant 1-sigma | finite/local | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|---:|---:|
| xy | 1.1520448 | 0.00038954053 | 0.034% | incomplete | incomplete | 1.1516089 | 1.163777 |
| z | 1.1697648 | 9.4750936e-05 | 0.008% | incomplete | incomplete | 1.169655 | 1.1701256 |
| roll_pitch | 3.5287743 | 0.03469435 | 0.983% | incomplete | incomplete | 3.4313442 | 3.8343649 |
| yaw | 3.3788679 | 0.074874366 | 2.216% | incomplete | incomplete | 1.5582703 | 7.6066585 |

The infinitesimal value uses a dedicated small centered finite
difference and is independent of the requested finite sigma
excursion up to numerical differentiation error. The finite
secant value uses the requested +/- k-sigma points. A
`finite/local` ratio far from one therefore measures chart/output
nonlinearity over that finite excursion rather than changing the
definition of the local covariance.

Sampled `A_real` rank loss or a large condition number is retained
as a diagnostic. A sample is only marked invalid after the
floating-point calculation itself becomes non-finite or otherwise
mathematically undefined.

## Dominant local covariance directions

### xy

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 4 | -0.00021203885 | 29.63% |
| 0 | +0.00020379077 | 27.37% |
| 3 | +0.00017722697 | 20.70% |
| 1 | -0.00011699598 | 9.02% |
| 2 | +0.00010697858 | 7.54% |

### z

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 1 | -6.4861794e-05 | 46.86% |
| 2 | -4.3211112e-05 | 20.80% |
| 4 | -3.3488112e-05 | 12.49% |
| 8 | +2.7141778e-05 | 8.21% |
| 3 | -1.9192448e-05 | 4.10% |

### roll_pitch

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | +0.025020499 | 52.01% |
| 3 | -0.018518597 | 28.49% |
| 1 | -0.012800371 | 13.61% |
| 0 | +0.0081831379 | 5.56% |
| 4 | +0.0019028942 | 0.30% |

### yaw

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | -0.057664225 | 59.31% |
| 2 | +0.042389727 | 32.05% |
| 1 | -0.016748758 | 5.00% |
| 3 | -0.013710687 | 3.35% |
| 4 | -0.0038504477 | 0.26% |

## Monte Carlo

Disabled for this run. The deterministic eigen-direction
analysis remains available.

