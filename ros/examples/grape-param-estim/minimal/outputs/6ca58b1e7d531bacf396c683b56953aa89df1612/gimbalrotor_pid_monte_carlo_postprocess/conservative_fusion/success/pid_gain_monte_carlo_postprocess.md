# Gimbalrotor PID Monte Carlo proposal

The estimator Gaussian is sampled in its native common-scale quotient
coordinate and propagated through the nonlinear static PID map. The PID
output is retained as an empirical distribution rather than re-Gaussianized.

- covariance mode: `conservative_fusion`
- coordinate mode: `estimator_quotient`
- samples: `10000/10000` valid
- seed: `0`

## Gain-scale distribution

| group | point | median | 16–84% | 2.5–97.5% | std | nonpositive |
|---|---:|---:|---:|---:|---:|---:|
| xy | 1.2145134 | 1.2146696 | [1.213734, 1.2155643] | [1.2126902, 1.216395] | 0.00093619899 | 0.000% |
| z | 1.2249249 | 1.2249125 | [1.2243907, 1.2254089] | [1.2238354, 1.2258368] | 0.00051309098 | 0.000% |
| roll_pitch | 4.1559028 | 4.1583122 | [4.0976665, 4.2178085] | [4.039337, 4.2760622] | 0.060241984 | 0.000% |
| yaw | 2.4909143 | 2.5148331 | [2.4444053, 2.5887045] | [2.3775545, 2.6753152] | 0.075807108 | 0.000% |

## Proposed PID gains

### xy

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 4 | 4.8586784 | [4.8549362, 4.8622571] | [4.8507609, 4.86558] |
| i_gain | 0.1 | 0.12146696 | [0.1213734, 0.12155643] | [0.12126902, 0.1216395] |
| d_gain | 2 | 2.4293392 | [2.4274681, 2.4311286] | [2.4253804, 2.43279] |

### z

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 5 | 6.1245624 | [6.1219537, 6.1270445] | [6.1191771, 6.129184] |
| i_gain | 1 | 1.2249125 | [1.2243907, 1.2254089] | [1.2238354, 1.2258368] |
| d_gain | 2.5 | 3.0622812 | [3.0609769, 3.0635222] | [3.0595886, 3.064592] |

### roll_pitch

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 13 | 54.058059 | [53.269665, 54.83151] | [52.511381, 55.588808] |
| i_gain | 1 | 4.1583122 | [4.0976665, 4.2178085] | [4.039337, 4.2760622] |
| d_gain | 20 | 83.166245 | [81.953331, 84.35617] | [80.78674, 85.521244] |

### yaw

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 6 | 15.088998 | [14.666432, 15.532227] | [14.265327, 16.051891] |
| i_gain | 1 | 2.5148331 | [2.4444053, 2.5887045] | [2.3775545, 2.6753152] |
| d_gain | 2 | 5.0296661 | [4.8888106, 5.1774089] | [4.755109, 5.3506303] |

## Joint gain-scale correlation

| | xy | z | roll_pitch | yaw |
|---|---:|---:|---:|---:|
| xy | 1 | 0.79015 | 0.31259 | 0.71267 |
| z | 0.79015 | 1 | 0.24733 | 0.22063 |
| roll_pitch | 0.31259 | 0.24733 | 1 | 0.21313 |
| yaw | 0.71267 | 0.22063 | 0.21313 | 1 |

## Numerical diagnostics

- source-threshold rank histogram: `{'6': 10000}`
- infinite allocation-condition fraction: `0`
- max finite allocation condition number: `4.435612719002686`
- warnings: `[]`

The ranges above are empirical quantiles of the selected estimator
Gaussian approximation after nonlinear PID postprocessing. No sample is
discarded merely because `A_real` loses the source SVD-threshold rank or
has a large condition number; only a genuinely undefined/non-finite
floating-point evaluation is marked invalid.
