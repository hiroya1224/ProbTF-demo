# First-order rotor-lag experiment

This directory is intentionally isolated from the existing pure-delay / strict-ZOH-lag estimator.

The actuator model is

\[
\tau_f \dot f(t) + f(t) = u_{\mathrm{ZOH}}(t),
\]

with **zero pure transport delay**.  Command ZOH remains because the controller publishes discrete thrust commands; only the actuator response model changes.  The response between command switches is evaluated with the exact first-order solution.

The existing geometric-Savitzky--Golay trajectory, Newton--Euler residual, 14-D physical chart, common-scale gauge, and post-fit physical covariance are reused.  The fifteenth optimized coordinate is

\[
\eta = \log \tau_f,
\]

and its residual Jacobian is analytic through the first-order recurrence.  As in the previous production estimator, post-fit plant covariance contains only the 13-D common-scale quotient; the actuator time constant is held at its point estimate during downstream PID Monte Carlo.

## 1. Estimate each bag

```bash
python3 minimal/first-order-lag/estimate.py --case failure1
python3 minimal/first-order-lag/estimate.py --case failure2
python3 minimal/first-order-lag/estimate.py --case success
```

or

```bash
python3 minimal/first-order-lag/run_all_estimates.py
```

Outputs are written to

```text
minimal/first-order-lag/outputs/<case>/estimate.json
```

The JSON is the downstream contract.  It contains the fitted `thrust_time_constant_seconds`, the 13-D quotient Gaussian approximation, recorded PID gains, controller timing, actuator limits, and provenance.  Downstream scripts do not read the old pure-delay `result.json`, `arrays.npz`, or `arguments.json`.

The default tau multistart is tied to the recorded command period:

```text
1x, 4x, 16x, 64x controller-command period
```

A single start can be forced with `--initial-tau`.

## 2. PID gain-region exploration

After all three JSON files exist, for example:

```bash
python3 minimal/first-order-lag/pid_gain_region.py \
  --estimate-json minimal/first-order-lag/outputs/failure1/estimate.json \
  --success-json minimal/first-order-lag/outputs/success/estimate.json
```

and similarly for `failure2`.  For the successful bag itself:

```bash
python3 minimal/first-order-lag/pid_gain_region.py \
  --estimate-json minimal/first-order-lag/outputs/success/estimate.json
```

All three can be run after estimation with:

```bash
python3 minimal/first-order-lag/run_all_pid_gain_regions.py
```

The downstream pole model has no delay queue.  Its local state is therefore the original 26-state rigid-body/controller/actuator state, with the four actual-thrust states following the fitted first-order dynamics.

For each of `xy`, `z`, `roll_pitch`, and `yaw`, the script writes PI / ID / DP projections.  The plotted field is currently a local log-gain spectral-radius surrogate, while the recorded PID and optional successful-flight PID are also evaluated with the exact 26-state pole model.  This keeps the estimator JSON contract independent of later replacement by adaptive boundary tracing / D-decomposition-style refinement.

The gain-region default uses 512 fixed plant samples.  The expensive pole map is only 26-state in this model because no pure-delay queue is present.  `--samples 128` is available for a quick diagnostic run.
