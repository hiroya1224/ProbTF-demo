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

The deterministic objective is prior-free. Numerical stabilization is separate
from scientific ridge analysis.

The analytically known common mass/inertia/thrust scale gauge is enforced only
inside each optimization step by the hard KKT constraint

```math
v_{\mathrm{scale}}^{T}p=0.
```

The physical step is computed with adaptive Levenberg--Marquardt damping and a
trust radius. Unknown weak directions are not removed with a singular-value
cutoff. Therefore a direction that is weakly identifiable remains in the
optimization problem; LM damping only suppresses numerically dangerous large
steps. Machine precision is used only when reporting numerical rank.

The old `--log-scale-bound`, `--matrix-log-bound`, and `--cog-bound` options are
retained for command-line compatibility but do not define active physical box
constraints. A deliberately broad numerical trial guard rejects pathological
trial points before model evaluation. A rejected trial increases LM damping and
shrinks the trust radius, so the safety guard cannot become an artificial
boundary optimum.

Rotor and gimbal lag search is enabled by default. The smooth stage estimates
continuous split rotor/gimbal lags, followed by strict ZOH lag screening and
physical refinement. `--skip-lag-search` fixes both lags to their data-derived
initial values. During the smooth solve, lag coordinates are divided by
\(T_*\); reported lags are seconds.

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

## Output and persistent diagnostics

Every completed window writes raw ridge analysis from the unstabilized final
data Jacobian:

```text
ridge.json
ridge.pdf
```

`ridge.json` contains the raw data information matrix, singular values, right
singular vectors, machine-precision rank, and the known common-scale gauge
diagnostic. LM damping and the KKT constraint are not added to this information
matrix. No scientific singular-value threshold is used.

Outputs are isolated by an experiment namespace derived from the config,
vehicle-model, and optional-prior contents:

```text
output/dimensionless_savgol_experiment/<config>_<hash>/W_<...>s/
```

The window root contains:

```text
result.json
arguments.json
timing.json
parameters.txt
parameters.pdf
summary.pdf
ridge.json
ridge.pdf
delay_profile.json
delay_profile.pdf
confidence.json            # unless --skip-confidence
confidence.pdf             # unless --skip-confidence
DATA_DICTIONARY.md
bags/
```

Each bag has an independent directory:

```text
bags/<bag-id>/
    result.json
    diagnostic.json
    diagnostic.pdf
    savgol_fit.pdf
    trajectory.pdf
    trajectory_free.pdf
    trajectory_3d.pdf
    sensor_consistency.pdf
    sensor_consistency_free.pdf
    raw_residual_wrench.pdf
    external_wrench.pdf
    external_wrench.json
    savgol_dynamics.npz
    rollout_diagnostics.npz
```

`external_wrench.*` is the raw inverse-dynamics residual wrench used only for
diagnostics and diagnostic replay. Its magnitude is not part of the
deterministic parameter objective.

Ablation mode writes `ablation.json` and `ablation.pdf` in the experiment
namespace and regenerates each window directory independently, preventing stale
files from another config/bag run from being mixed into the current result.

### Timing

`timing.json` and `result.json["timing"]` record wall-clock durations for
config/model setup, rosbag loading per bag and in total, Savitzky--Golay/problem
construction per bag and in total, lag and physical optimization, each adaptive
LM solve (including objective-evaluation and KKT linear-solve time), strict lag
screening, final raw-Jacobian ridge analysis, confidence/posterior construction,
free and raw-residual-wrench rollouts, per-bag report/file output, root
report/file output, and total execution.
