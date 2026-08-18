# Gimbalrotor unstable-mode growth validation

- Case: `failure2`
- Actual outcome: `crashed`
- Fit window: [0, 5.49616] s relative to first valid pose sample

## Pole prediction

- discrete pole: `1.00282224 +0.0195265717i`
- |z|: `1.00301233`
- growth sigma: `0.600629303 1/s`
- frequency: `0.618764221 Hz`
- doubling time: `1.15403491` s
- matched orientation axis xyz: `[-0.04909348  0.99688733  0.06168855]`

## Recorded attitude fit

- growth sigma: `0.394687117 1/s`
- frequency: `0.553188527 Hz`
- doubling time: `1.75619408` s
- fit R^2: `0.843172945`
- residual RMS: `0.0584830049 rad`

## Comparison

- observed - predicted growth: `-0.205942185 1/s`
- observed / predicted growth: `0.657122647`
- observed - predicted frequency: `-0.0655756946 Hz`
- observed / predicted frequency: `0.894021515`

The flight label is metadata only; it does not enter the fit.
