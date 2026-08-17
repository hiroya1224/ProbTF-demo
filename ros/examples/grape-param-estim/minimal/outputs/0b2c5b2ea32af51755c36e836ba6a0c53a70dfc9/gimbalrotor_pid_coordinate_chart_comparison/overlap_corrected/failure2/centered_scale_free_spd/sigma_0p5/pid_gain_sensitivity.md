# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
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
| xy | 1.1787878 | 8.4206124e-05 | 0.007% | incomplete | incomplete | 1.1787605 | 1.1796407 |
| z | 1.1789924 | 0.00017125451 | 0.015% | incomplete | incomplete | 1.1789587 | 1.1790558 |
| roll_pitch | 1.7244864 | 0.015122422 | 0.877% | incomplete | incomplete | 1.4233777 | 1.7313533 |
| yaw | 1.9977971 | 0.049890956 | 2.497% | incomplete | incomplete | 0.68267307 | 2.013845 |

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
| 0 | -5.3386673e-05 | 40.20% |
| 2 | +5.2419691e-05 | 38.75% |
| 4 | -2.6247182e-05 | 9.72% |
| 3 | -1.8055113e-05 | 4.60% |
| 9 | -1.1504053e-05 | 1.87% |

### z

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | -0.00011842662 | 47.82% |
| 1 | -8.9377439e-05 | 27.24% |
| 2 | +6.5027228e-05 | 14.42% |
| 4 | -4.470615e-05 | 6.81% |
| 5 | +2.9687641e-05 | 3.01% |

### roll_pitch

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 3 | -0.013719211 | 82.30% |
| 2 | +0.0053364942 | 12.45% |
| 1 | +0.0032011358 | 4.48% |
| 0 | +0.0011865669 | 0.62% |
| 5 | +0.00055836828 | 0.14% |

### yaw

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | +0.0320531 | 41.28% |
| 1 | +0.030512256 | 37.40% |
| 0 | -0.021646982 | 18.83% |
| 3 | -0.0078331284 | 2.47% |
| 5 | -0.00071811318 | 0.02% |

## Monte Carlo

Disabled for this run. The deterministic eigen-direction
analysis remains available.

