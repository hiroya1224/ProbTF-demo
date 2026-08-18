# Gimbalrotor local sampled-data pole validation

- Flight outcome: `crashed`
- Covariance: `overlap_corrected`
- Delay: `fitted_thrust_delay` (0.271089196 s; thrust only)
- Controller dt: 0.00500774384 s
- Recorded roll/pitch PID: P=10, I=1, D=8

## Center plant

- Equilibrium valid: `True`
- One-step trim defect: 3.17421289e-12
- Spectral radius: `1.0030123256502241`
- Stable: `False`

## Monte Carlo distribution

- Pole-valid samples: 512/512
- Stable fraction among pole-valid: `0.322265625`
- Spectral-radius median: 1.00022071
- Spectral-radius 16–84%: [0.999831284, 1.0012546]
- Spectral-radius 2.5–97.5%: [0.999831284, 1.00289653]

## Warnings

- `constant_controller_dt_approximates_recorded_timing_jitter`
