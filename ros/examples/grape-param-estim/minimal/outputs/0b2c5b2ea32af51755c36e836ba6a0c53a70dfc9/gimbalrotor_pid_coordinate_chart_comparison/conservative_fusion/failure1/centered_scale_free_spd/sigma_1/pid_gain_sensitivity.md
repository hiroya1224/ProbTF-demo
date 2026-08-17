# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `conservative_fusion`
- coordinate mode: `centered_scale_free_spd`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- finite sigma multiple: `1`
- local derivative sigma fraction: `1e-05`
- valid finite eigen samples: `25/27`
- valid local-derivative directions: `13/13`

## Gain-scale sensitivity

| group | center | infinitesimal 1-sigma | relative | finite secant 1-sigma | finite/local | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|---:|---:|
| xy | 1.1520448 | 0.0021862551 | 0.190% | incomplete | incomplete | 1.150498 | 1.1579671 |
| z | 1.1697648 | 0.0020644383 | 0.176% | incomplete | incomplete | 1.1680495 | 1.1714824 |
| roll_pitch | 3.5287743 | 0.076779354 | 2.176% | incomplete | incomplete | 3.4165261 | 3.5866621 |
| yaw | 3.3788679 | 0.19808149 | 5.862% | incomplete | incomplete | 3.2837336 | 7.7485451 |

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
| 8 | +0.0015473357 | 50.09% |
| 5 | -0.00089562436 | 16.78% |
| 6 | +0.00076970538 | 12.40% |
| 0 | +0.000644647 | 8.69% |
| 4 | +0.00055231181 | 6.38% |

### z

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 8 | +0.0017164311 | 69.13% |
| 5 | -0.00084153174 | 16.62% |
| 6 | +0.00069952745 | 11.48% |
| 0 | +0.00022337711 | 1.17% |
| 7 | +0.00018156181 | 0.77% |

### roll_pitch

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | -0.060803578 | 62.71% |
| 4 | -0.03319823 | 18.70% |
| 3 | +0.027906797 | 13.21% |
| 0 | -0.01492116 | 3.78% |
| 1 | +0.0054506684 | 0.50% |

### yaw

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | -0.16848991 | 72.35% |
| 2 | -0.095374594 | 23.18% |
| 3 | +0.038648655 | 3.81% |
| 4 | -0.012713267 | 0.41% |
| 5 | -0.0070501065 | 0.13% |

## Monte Carlo

Disabled for this run. The deterministic eigen-direction
analysis remains available.

