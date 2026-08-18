# Gimbalrotor local sampled-data pole validation

- Flight outcome: `successful`
- Covariance: `overlap_corrected`
- Delay: `zero_thrust_delay` (0 s; thrust only)
- Controller dt: 0.00999832153 s
- Recorded roll/pitch PID: P=13, I=1, D=20

## Center plant

- Equilibrium valid: `True`
- One-step trim defect: 2.29531502e-15
- Spectral radius: `0.9997469563288534`
- Stable: `True`

## Monte Carlo distribution

- Pole-valid samples: 512/512
- Stable fraction among pole-valid: `1.0`
- Spectral-radius median: 0.999746956
- Spectral-radius 16–84%: [0.999746956, 0.999746956]
- Spectral-radius 2.5–97.5%: [0.999746956, 0.999746956]

## Warnings

- `constant_controller_dt_approximates_recorded_timing_jitter`
