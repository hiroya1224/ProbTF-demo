# Gimbalrotor PID Monte Carlo proposal

The estimator Gaussian is sampled in its native common-scale quotient
coordinate and propagated through the nonlinear static PID map. The PID
output is retained as an empirical distribution rather than re-Gaussianized.

- covariance mode: `conservative_fusion`
- coordinate mode: `estimator_quotient`
- samples: `9993/10000` valid
- seed: `0`

## Gain-scale distribution

| group | point | median | 16–84% | 2.5–97.5% | std | nonpositive |
|---|---:|---:|---:|---:|---:|---:|
| xy | 1.1787878 | 1.179349 | [1.1777765, 1.1808313] | [1.1760844, 1.1822109] | 0.0015687999 | 0.000% |
| z | 1.1789924 | 1.1789183 | [1.1773895, 1.1803442] | [1.1757691, 1.1816937] | 0.0015207002 | 0.000% |
| roll_pitch | 1.7244864 | 2.511884 | [1.943682, 3.0652474] | [1.6826654, 3.6008057] | 0.54260376 | 0.000% |
| yaw | 1.9977971 | 1.9650952 | [0.15585571, 5.0653005] | [0.056304623, 8.7544805] | 2.5499875 | 0.000% |

## Proposed PID gains

### xy

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 3 | 3.538047 | [3.5333294, 3.5424938] | [3.5282531, 3.5466328] |
| i_gain | 0.1 | 0.1179349 | [0.11777765, 0.11808313] | [0.11760844, 0.11822109] |
| d_gain | 1 | 1.179349 | [1.1777765, 1.1808313] | [1.1760844, 1.1822109] |

### z

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 5 | 5.8945916 | [5.8869477, 5.901721] | [5.8788457, 5.9084686] |
| i_gain | 1 | 1.1789183 | [1.1773895, 1.1803442] | [1.1757691, 1.1816937] |
| d_gain | 2.5 | 2.9472958 | [2.9434738, 2.9508605] | [2.9394229, 2.9542343] |

### roll_pitch

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 10 | 25.11884 | [19.43682, 30.652474] | [16.826654, 36.008057] |
| i_gain | 1 | 2.511884 | [1.943682, 3.0652474] | [1.6826654, 3.6008057] |
| d_gain | 8 | 20.095072 | [15.549456, 24.521979] | [13.461324, 28.806446] |

### yaw

| gain | recorded | median proposal | 16–84% | 2.5–97.5% |
|---|---:|---:|---:|---:|
| p_gain | 4 | 7.8603809 | [0.62342283, 20.261202] | [0.22521849, 35.017922] |
| i_gain | 1 | 1.9650952 | [0.15585571, 5.0653005] | [0.056304623, 8.7544805] |
| d_gain | 2 | 3.9301905 | [0.31171141, 10.130601] | [0.11260925, 17.508961] |

## Joint gain-scale correlation

| | xy | z | roll_pitch | yaw |
|---|---:|---:|---:|---:|
| xy | 1 | 0.97788 | 0.28646 | -0.053541 |
| z | 0.97788 | 1 | 0.20998 | -0.041316 |
| roll_pitch | 0.28646 | 0.20998 | 1 | 0.32676 |
| yaw | -0.053541 | -0.041316 | 0.32676 | 1 |

## Numerical diagnostics

- source-threshold rank histogram: `{'4': 2265, '5': 111, '6': 7617}`
- infinite allocation-condition fraction: `0`
- max finite allocation condition number: `6.366845480125827e+17`
- warnings: `['numerically_undefined_samples_present', 'source_threshold_rank_loss_present']`

The ranges above are empirical quantiles of the selected estimator
Gaussian approximation after nonlinear PID postprocessing. No sample is
discarded merely because `A_real` loses the source SVD-threshold rank or
has a large condition number; only a genuinely undefined/non-finite
floating-point evaluation is marked invalid.
