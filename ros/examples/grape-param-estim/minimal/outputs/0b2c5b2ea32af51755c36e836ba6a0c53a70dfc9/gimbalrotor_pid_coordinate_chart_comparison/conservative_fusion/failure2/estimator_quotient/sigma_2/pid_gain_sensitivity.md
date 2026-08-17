# Gimbalrotor PID postprocess sensitivity

This is a local sensitivity analysis of the static PID gain correction.
The selected estimator covariance defines perturbation size; it is not
reported as a calibrated posterior probability.

- estimator result: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json`
- arrays: `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_2_nominal_pseudo_conditioning_production_20260817/cases/prior_free/arrays.npz`
- covariance mode: `conservative_fusion`
- coordinate mode: `estimator_quotient`
- center coordinate source: `quotient_basis @ quotient_coordinate (zero common-scale gauge representative)`
- characteristic length: `0.191584994 m`
- finite sigma multiple: `2`
- local derivative sigma fraction: `1e-05`
- valid finite eigen samples: `27/27`
- valid local-derivative directions: `13/13`

## Gain-scale sensitivity

| group | center | infinitesimal 1-sigma | relative | finite secant 1-sigma | finite/local | eigen-point min | eigen-point max |
|---|---:|---:|---:|---:|---:|---:|---:|
| xy | 1.1787878 | 0.0013486732 | 0.114% | 0.0013802245 | 1.0233943 | 1.1765904 | 1.1809868 |
| z | 1.1789924 | 0.0014498283 | 0.123% | 0.0014465576 | 0.99774404 | 1.1768091 | 1.1811795 |
| roll_pitch | 1.7244864 | 0.059730548 | 3.464% | 0.2156123 | 3.6097493 | 1.6139664 | 2.721996 |
| yaw | 1.9977971 | 0.16111235 | 8.064% | 0.2352728 | 1.4603027 | 0.073647653 | 2.6224101 |

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
| 8 | +0.0010991226 | 66.42% |
| 7 | +0.0006735504 | 24.94% |
| 3 | +0.00027125537 | 4.05% |
| 2 | -0.00018279888 | 1.84% |
| 6 | +0.00015014241 | 1.24% |

### z

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 8 | +0.0010926089 | 56.79% |
| 7 | +0.00066741908 | 21.19% |
| 1 | +0.00048559536 | 11.22% |
| 2 | -0.00025869832 | 3.18% |
| 0 | +0.00022446439 | 2.40% |

### roll_pitch

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 3 | +0.056516618 | 89.53% |
| 0 | +0.014036552 | 5.52% |
| 2 | -0.012286761 | 4.23% |
| 4 | +0.0034828361 | 0.34% |
| 1 | -0.0033607447 | 0.32% |

### yaw

| direction | local 1-sigma scale effect | variance contribution |
|---:|---:|---:|
| 1 | -0.12304965 | 58.33% |
| 2 | -0.094597467 | 34.47% |
| 3 | +0.03787844 | 5.53% |
| 0 | +0.01876303 | 1.36% |
| 5 | +0.0071823735 | 0.20% |

## Monte Carlo

Disabled for this run. The deterministic eigen-direction
analysis remains available.

