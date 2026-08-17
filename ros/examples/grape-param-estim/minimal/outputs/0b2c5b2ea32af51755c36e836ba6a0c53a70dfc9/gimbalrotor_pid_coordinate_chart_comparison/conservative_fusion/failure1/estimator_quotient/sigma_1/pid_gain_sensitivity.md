# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `conservative_fusion`
- coordinate mode: `estimator_quotient`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- finite sigma multiple: `1`
- local derivative sigma fraction: `1e-05`
- valid finite eigen samples: `27/27`
- valid local-derivative directions: `13/13`

## Gain-scale sensitivity

| group | center | infinitesimal 1-sigma | relative | finite secant 1-sigma | finite/local | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|---:|---:|
| xy | 1.1520448 | 0.0021862551 | 0.190% | 0.0021686875 | 0.99196452 | 1.150362 | 1.154915 |
| z | 1.1697648 | 0.0020644383 | 0.176% | 0.0020643579 | 0.99996102 | 1.1679529 | 1.1715796 |
| roll_pitch | 3.5287743 | 0.076779356 | 2.176% | 0.080799539 | 1.0523602 | 3.4681521 | 3.7996002 |
| yaw | 3.3788679 | 0.19808149 | 5.862% | 0.24205699 | 1.2220071 | 3.2123158 | 3.6494622 |

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
| 8 | +0.0016837073 | 59.31% |
| 5 | +0.00086536003 | 15.67% |
| 0 | +0.00064367855 | 8.67% |
| 3 | +0.00047786789 | 4.78% |
| 6 | +0.0004734632 | 4.69% |

### z

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 8 | +0.0018133442 | 77.15% |
| 5 | +0.00079282976 | 14.75% |
| 6 | +0.00036348121 | 3.10% |
| 7 | +0.00027619076 | 1.79% |
| 0 | +0.00022388628 | 1.18% |

### roll_pitch

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 2 | -0.057576401 | 56.23% |
| 4 | +0.038794327 | 25.53% |
| 3 | +0.028410444 | 13.69% |
| 0 | -0.014894592 | 3.76% |
| 1 | +0.005374086 | 0.49% |

### yaw

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 0 | -0.1685264 | 72.38% |
| 2 | -0.092333752 | 21.73% |
| 3 | +0.042485834 | 4.60% |
| 4 | +0.021461097 | 1.17% |
| 5 | +0.00587342 | 0.09% |

## Monte Carlo

Disabled for this run. The deterministic eigen-direction
analysis remains available.

