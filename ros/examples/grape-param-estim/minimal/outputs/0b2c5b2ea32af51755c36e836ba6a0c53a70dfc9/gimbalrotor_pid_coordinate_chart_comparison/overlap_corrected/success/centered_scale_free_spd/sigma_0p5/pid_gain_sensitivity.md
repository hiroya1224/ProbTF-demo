# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `overlap_corrected`
- coordinate mode: `centered_scale_free_spd`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- finite sigma multiple: `0.5`
- local derivative sigma fraction: `1e-05`
- valid finite eigen samples: `25/27`
- valid local-derivative directions: `13/13`

## Gain-scale sensitivity

| group | center | infinitesimal 1-sigma | relative | finite secant 1-sigma | finite/local | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|---:|---:|
| xy | 1.2145134 | 0.00010639105 | 0.009% | incomplete | incomplete | 1.2144689 | 1.2205777 |
| z | 1.2249249 | 4.4777719e-05 | 0.004% | incomplete | incomplete | 1.2249171 | 1.2251946 |
| roll_pitch | 4.1559028 | 0.015522 | 0.373% | incomplete | incomplete | 0.23609443 | 4.1623169 |
| yaw | 2.4909143 | 0.0075630516 | 0.304% | incomplete | incomplete | -0.0049162181 | 2.4936486 |

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
| 5 | -8.8485907e-05 | 69.17% |
| 0 | -3.1163849e-05 | 8.58% |
| 2 | -2.7887803e-05 | 6.87% |
| 6 | +2.5610325e-05 | 5.79% |
| 4 | -2.2182167e-05 | 4.35% |

### z

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | -3.8420267e-05 | 73.62% |
| 6 | +1.5686086e-05 | 12.27% |
| 2 | -1.2884194e-05 | 8.28% |
| 5 | -9.2121422e-06 | 4.23% |
| 3 | +3.2325143e-06 | 0.52% |

### roll_pitch

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | -0.012861043 | 68.65% |
| 1 | -0.0053789475 | 12.01% |
| 3 | +0.0042805099 | 7.60% |
| 5 | -0.0041264244 | 7.07% |
| 4 | +0.0030662223 | 3.90% |

### yaw

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 4 | -0.0054707692 | 52.32% |
| 3 | +0.0036581482 | 23.40% |
| 0 | +0.0033434981 | 19.54% |
| 5 | -0.0014110216 | 3.48% |
| 2 | -0.00079256066 | 1.10% |

## Monte Carlo

Disabled for this run. The deterministic eigen-direction
analysis remains available.

