# Dimensionless geometric Savitzky--Golay experiment

This document describes `dimensionless_savgol_experiment.py`.  The script is a
flat experimental replacement for the broken SG deterministic/confidence chain
at commit `9d332eb`.

## Responsibility boundary

The deterministic estimator is prior-free.  `--prior-json` is read only when a
local posterior is requested.  It is never appended to the deterministic
least-squares residual.

```text
raw pose + recorded commands
        |
        v
dimensionless deterministic SG fit
        |
        +--> residual acceleration, raw residual wrench, Jacobian spectrum
        |
        +--> local data likelihood
                      |
external prior -------+--> local posterior
```

## Fixed reference scales

For reference mass \(M_*\) and inertia \(J_*\),

```math
L_*=\sqrt{\frac{\operatorname{tr}J_*}{3M_*}},\qquad
T_*=\sqrt{\frac{L_*}{g}},
```

and

```math
F_*=\frac{M_*L_*}{T_*^2},\qquad
N_*=\frac{M_*L_*^2}{T_*^2}.
```

All deterministic residuals use these fixed scales.  The optimizer does not
change the scales when the candidate parameters change.

## Physical chart

The 14 deterministic coordinates are dimensionless:

1. log mass scale;
2. six coordinates of a symmetric matrix \(S\);
3. CoG displacement divided by \(L_*\);
4. four independent log force-effectiveness scales.

Let \(\bar\Sigma_0\) be the reference mass-distribution second moment divided by
\(M_*L_*^2\), and let \(B_0=\bar\Sigma_0^{1/2}\).  The candidate is

```math
\bar\Sigma=B_0\exp(S)B_0,\qquad
\bar J=\operatorname{tr}(\bar\Sigma)I-\bar\Sigma.
```

This guarantees positive inertia and strict principal-moment triangle
inequalities.

The common scaling

```math
(m,J,f_1,\ldots,f_4)
\mapsto
(\lambda m,\lambda J,\lambda f_1,\ldots,\lambda f_4)
```

is retained.  In this chart it is the globally straight direction

```text
(1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1).
```

The estimator verifies analytically and by finite differences that this is a
null direction of the dimensionless acceleration objective when fixed drag is
zero.

## Objective

At every valid centered raw-pose SG time, the script computes dimensionless
body linear acceleration and angular acceleration.  The residual is

```math
r_k=
\begin{bmatrix}
\bar a_{\mathrm{obs},k}-\bar F_k/\bar m\\
\bar\alpha_{\mathrm{obs},k}
-\bar J^{-1}
(\bar\tau_k-\bar\omega_k\times\bar J\bar\omega_k)
\end{bmatrix}.
```

Each bag contributes the sample mean of \(\frac{1}{2}\lVert r_k\rVert^2\);
configured bag weights are normalized to sum to one.

## Optimization

The active solver is SciPy `least_squares` with

```text
method="trf"
tr_solver="exact"
x_scale=1.0
loss="linear"
```

There is no custom SVD threshold in the deterministic step.  Because every coordinate has already been nondimensionalized, no adaptive
Jacobian-column rescaling is used. The exact scale ridge is straight in the
chart and is handled by the dense SVD trust-region solve.  Broad finite coordinate bounds are floating-point domain guards, not a
probabilistic prior.

By default, rotor and gimbal lags are fixed to their data-derived median
publish periods; lag is not the target of this experiment. Pass `--search-lags`
to enable the smooth/strict split-lag search. During that optional smooth solve,
lag coordinates are divided by \(T_*\). Reported lags are seconds.

## Residual wrench

The script stores

```math
W_{\mathrm{res}}=W_{\mathrm{required}}-W_{\mathrm{modeled}}
```

in physical N/N m and in fixed-reference dimensionless units.  The absolute
physical wrench uses the representative point returned on the scale ridge.
The generalized-acceleration residual and its information spectrum are
scale-invariant.

## Confidence and posterior

Confidence uses the same dimensionless generalized-acceleration residual and
analytic Jacobian as the deterministic objective.  It does not use absolute
residual wrench to claim information about the common physical scale.

A separate empirical 6x6 residual covariance is estimated for each bag, and the
per-bag residual/Jacobian are whitened.  Data information is formed before any
prior is introduced.  When `--prior-json` is supplied, its physical Gaussian
factor is linearized into the 14-D dimensionless chart and added to the data
information matrix.

## Output

For one window \(W\):

```text
output/dimensionless_savgol_experiment/W_<W>s/
    result.json
    confidence.json                  # unless --skip-confidence
    parameters.txt
    summary.pdf
    <bag-id>_dynamics.npz
    <bag-id>_diagnostic.pdf
```

Ablation mode additionally writes:

```text
output/dimensionless_savgol_experiment/
    ablation.json
    ablation.pdf
```

Legacy forward replay and external-wrench optimization are intentionally not
called by this experiment.  They were a separate source of stale mixed-version
outputs and are not required to evaluate the dimensionless deterministic core.
