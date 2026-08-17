# Gimbalrotor PID Monte Carlo proposal

The estimator Gaussian is sampled in its native common-scale quotient
coordinate and propagated through the nonlinear static PID map. The PID
output is retained as an empirical distribution rather than re-Gaussianized.

- covariance mode: `overlap_corrected`
- coordinate mode: `estimator_quotient`
- samples: `10000/10000` valid
- seed: `0`

## Gain-scale distribution

| group | point | median | 16–84% | 2.5–97.5% | std | nonpositive |
|---|---:|---:|---:|---:|---:|---:|
| xy | 1.2145134 | 1.2145283 | [1.2144181, 1.2146352] | [1.214318, 1.2147411] | 0.00010863017 | 0.000% |
| z | 1.2249249 | 1.2249236 | [1.2248798, 1.2249685] | [1.2248371, 1.2250117] | 4.4769369e-05 | 0.000% |
| roll_pitch | 4.1559028 | 4.1557937 | [4.1403818, 4.1710844] | [4.1249266, 4.1857614] | 0.015485042 | 0.000% |
| yaw | 2.4909143 | 2.4931641 | [2.4851708, 2.5016405] | [2.4776917, 2.5101023] | 0.0083955843 | 0.000% |

## Proposed PID gains

### xy

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 4 | 4.858113 | [4.8576726, 4.858541] | [4.857272, 4.8589644] |
| i_gain | 0.1 | 0.12145283 | [0.12144181, 0.12146352] | [0.1214318, 0.12147411] |
| d_gain | 2 | 2.4290565 | [2.4288363, 2.4292705] | [2.428636, 2.4294822] |

### z

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 5 | 6.1246181 | [6.1243991, 6.1248426] | [6.1241853, 6.1250584] |
| i_gain | 1 | 1.2249236 | [1.2248798, 1.2249685] | [1.2248371, 1.2250117] |
| d_gain | 2.5 | 3.0623091 | [3.0621996, 3.0624213] | [3.0620927, 3.0625292] |

### roll_pitch

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 13 | 54.025319 | [53.824963, 54.224097] | [53.624046, 54.414899] |
| i_gain | 1 | 4.1557937 | [4.1403818, 4.1710844] | [4.1249266, 4.1857614] |
| d_gain | 20 | 83.115875 | [82.807636, 83.421688] | [82.498532, 83.715229] |

### yaw

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 6 | 14.958985 | [14.911025, 15.009843] | [14.86615, 15.060614] |
| i_gain | 1 | 2.4931641 | [2.4851708, 2.5016405] | [2.4776917, 2.5101023] |
| d_gain | 2 | 4.9863282 | [4.9703415, 5.003281] | [4.9553835, 5.0202046] |

## Joint gain-scale correlation

| | xy | z | roll_pitch | yaw |
|---|---:|---:|---:|---:|
| xy | 1 | 0.56849 | 0.37575 | 0.34609 |
| z | 0.56849 | 1 | 0.29651 | -0.27218 |
| roll_pitch | 0.37575 | 0.29651 | 1 | 0.12961 |
| yaw | 0.34609 | -0.27218 | 0.12961 | 1 |

## Numerical diagnostics

- source-threshold rank histogram: `{'6': 10000}`
- infinite allocation-condition fraction: `0`
- max finite allocation condition number: `3.9379523721089984`
- warnings: `[]`

The ranges above are empirical quantiles of the selected estimator
Gaussian approximation after nonlinear PID postprocessing. No sample is
discarded merely because `A_real` loses the source SVD-threshold rank or
has a large condition number; only a genuinely undefined/non-finite
floating-point evaluation is marked invalid.
