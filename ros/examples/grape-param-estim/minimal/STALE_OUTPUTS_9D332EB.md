# Outputs committed before the dimensionless SG replacement

The existing directories below `output/savgol_window_ablation/` were generated
by several incompatible source generations:

- prior-included deterministic SG;
- prior-free dimensional SG with SciPy TRF;
- experimental custom truncated-SVD SG.

They are not outputs of `dimensionless_savgol_experiment.py` and must not be
used to validate the new implementation.

The replacement writes only below:

```text
output/dimensionless_savgol_experiment/
```

Old outputs are intentionally not deleted automatically because they are tracked
experiment records.  Regenerate every window before comparing parameter or
residual-wrench values.
