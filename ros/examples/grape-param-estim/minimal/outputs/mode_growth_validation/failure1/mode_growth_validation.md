# Gimbalrotor unstable-mode growth validation

- Case: `failure1`
- Actual outcome: `crashed`
- Fit window: [0, 5.98856] s relative to first valid pose sample

## Pole prediction

- discrete pole: `1.00005951 +0.020592672i`
- |z|: `1.0002715`
- growth sigma: `0.0543433229 1/s`
- frequency: `0.655964052 Hz`
- doubling time: `12.754965` s
- matched orientation axis xyz: `[0.05208738 0.99619209 0.0699158 ]`

## Recorded attitude fit

- growth sigma: `0.204860003 1/s`
- frequency: `0.611273623 Hz`
- doubling time: `3.3835164` s
- fit R^2: `0.967666352`
- residual RMS: `0.0621568176 rad`

## Comparison

- observed - predicted growth: `0.15051668 1/s`
- observed / predicted growth: `3.76973641`
- observed - predicted frequency: `-0.0446904297 Hz`
- observed / predicted frequency: `0.931870612`

The flight label is metadata only; it does not enter the fit.
