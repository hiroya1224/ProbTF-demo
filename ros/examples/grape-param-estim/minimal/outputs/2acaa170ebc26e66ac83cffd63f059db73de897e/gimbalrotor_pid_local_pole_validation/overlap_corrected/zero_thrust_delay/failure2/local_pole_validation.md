# Gimbalrotor local sampled-data pole validation

- Flight outcome: `crashed`
- Covariance: `overlap_corrected`
- Delay: `zero_thrust_delay` (0 s; thrust only)
- Controller dt: 0.00500774384 s
- Recorded roll/pitch PID: P=10, I=1, D=8

## Center plant

- Equilibrium valid: `True`
- One-step trim defect: 3.17421289e-12
- Spectral radius: `0.999831284367488`
- Stable: `True`

## Monte Carlo distribution

- Pole-valid samples: 512/512
- Stable fraction among pole-valid: `0.5703125`
- Spectral-radius median: 0.999831284
- Spectral-radius 16–84%: [0.999831284, 1.00031221]
- Spectral-radius 2.5–97.5%: [0.999831284, 1.00036554]

## Warnings

- `constant_controller_dt_approximates_recorded_timing_jitter`
