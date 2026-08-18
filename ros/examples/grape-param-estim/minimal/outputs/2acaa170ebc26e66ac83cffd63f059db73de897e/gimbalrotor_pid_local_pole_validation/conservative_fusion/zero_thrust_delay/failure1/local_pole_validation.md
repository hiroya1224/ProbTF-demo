# Gimbalrotor local sampled-data pole validation

- Flight outcome: `crashed`
- Covariance: `conservative_fusion`
- Delay: `zero_thrust_delay` (0 s; thrust only)
- Controller dt: 0.00499534607 s
- Recorded roll/pitch PID: P=20, I=1, D=8

## Center plant

- Equilibrium valid: `True`
- One-step trim defect: 1.34202545e-15
- Spectral radius: `0.9998825721107053`
- Stable: `True`

## Monte Carlo distribution

- Pole-valid samples: 512/512
- Stable fraction among pole-valid: `1.0`
- Spectral-radius median: 0.999882653
- Spectral-radius 16–84%: [0.999880389, 0.999884625]
- Spectral-radius 2.5–97.5%: [0.999878347, 0.999886749]

## Warnings

- `constant_controller_dt_approximates_recorded_timing_jitter`
