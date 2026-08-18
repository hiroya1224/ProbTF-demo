# Gimbalrotor local sampled-data pole validation

- Flight outcome: `crashed`
- Covariance: `overlap_corrected`
- Delay: `fitted_thrust_delay` (0.197199225 s; thrust only)
- Controller dt: 0.00499534607 s
- Recorded roll/pitch PID: P=20, I=1, D=8

## Center plant

- Equilibrium valid: `True`
- One-step trim defect: 1.34202545e-15
- Spectral radius: `1.0002715005539535`
- Stable: `False`

## Monte Carlo distribution

- Pole-valid samples: 512/512
- Stable fraction among pole-valid: `0.033203125`
- Spectral-radius median: 1.00024455
- Spectral-radius 16–84%: [1.0001453, 1.00026889]
- Spectral-radius 2.5–97.5%: [0.99998225, 1.00027776]

## Warnings

- `constant_controller_dt_approximates_recorded_timing_jitter`
