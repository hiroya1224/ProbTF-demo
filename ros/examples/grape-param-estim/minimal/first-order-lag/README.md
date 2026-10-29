# First-order rotor-lag experiment

## 2026-10-28: withdraw success-conditioned gain maps

The independent 33→65 runs for failure1 and failure2 have now been launched in
tmux. See [the run record and retained grid data](HIGH_RESOLUTION_RUNS_20261028.md).
This records job startup, not completion.

The [completion watcher](PUBLICATION_WATCH_20261029.md) will generate the historical
paper layout and normally commit/push the independent results after both jobs
finish. Cross-failure fusion was cancelled by the user.

The October 27 audit identified a change of statistical target that was not
compatible with predicting gains from a failure record. The previous maps
selected plants that were stable with the known successful controller and fixed
the other nine gains to that successful controller. Those maps and their
success-100% claims are withdrawn as independent failure-only predictions.
Historical publication data are retained and explicitly labeled; they are not
silently reinterpreted as corrected results.

The corrected map uses the full, unchanged ordered plant sample set from one
failure estimate. For a complete 12-gain controller K it reports

\[
Q(K)=\frac{1}{N}\sum_{n=1}^N
\mathbf 1\{\text{valid finite }\rho(F_n(K))<1\}.
\]

Every original sample remains in N. An unresolved trim or nonfinite radius is
reported separately and is not counted stable. No sample is selected by its
behavior at successful-flight gains. No refit or change of the original
estimated plant Gaussian is introduced by this correction.

A three-gain volume or two-gain slice holds all other gains at **this failure
record's gains**. Its domain, anchors, evaluation order, and trim initialization
use no successful-flight data. Favorable points from different slices cannot be
combined and called stable without evaluating the resulting full 12-gain
controller. Q is a local linear stability fraction of finite model samples,
not a real-flight success probability or a guarantee throughout voxel cells.

### Current commands

Estimation remains `estimate.py --case failure1` (or `failure2`); the model is
still the exact first-order actuator response with zero pure transport delay.
For a corrected adaptive slice, from this directory:

```bash
/usr/bin/python3 pid_gain_contour.py \
  --estimate-json outputs/failure1/estimate.json --samples 128
```

For an exact finite 3-D grid, using the existing fixed 128 Sobol plant samples:

```bash
/usr/bin/python3 automation/pid_gain_voxels_fast.py \
  --estimate-json outputs/failure1/estimate.json \
  --samples 128 --workers 16 --grid-sizes 3 5 9 \
  --output outputs/failure1/pid_gain_voxels_fast_interior_failure_context
```

Use the failure2 estimate and a separate output directory for failure2. The
new default domains use declared scale factors of this record's gains; they do
not expand to include a known success point. Use a new output directory: old
success-context checkpoints are not compatible with the corrected statistic
and fixed-controller context.

`--success-json` is optional. When supplied, it is read only after all map
queries and used for an independent full-12-gain reference evaluation. It does
not change the plant population, grid, fixed gains, or warm-start cache, and no
projected "Success" marker is presented as a complete controller on a slice.

Primary voxel data are `stability_fraction.npz/json`, containing
`stable_fraction`, `stable_count`, `valid_count`, `unresolved_count`, and the
fixed `sample_count`. The 95% classification is a finite-sample reference;
100% means every original sample was valid and stable. Original per-sample
spectral radii remain available.

The existing full-12-gain search in `pid_safe_margin_full.py` remains a distinct
calculation. It already used successful-flight gains only after its search;
this correction does not replace that search with combinations of independent
slice points.

See [the correction report](AUDIT_CORRECTION_20261028.md) for tests, the actual
recalculation resolution, sample identity checks, and the distinction between
withdrawn historical figures and newly evaluated failure-context results.

<details>
<summary>Historical instructions before the October 28 correction (withdrawn where they use success context)</summary>

The following text is preserved to document the previous implementation. Its
success-context commands and claims are not instructions for the corrected
failure-only prediction pipeline.

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

## 3. Exact three-dimensional PID volumes

`pid_gain_voxels.py` evaluates P, I, and D together for each of `roll_pitch`,
`xy`, `z`, and `yaw`. The other nine gains are fixed to the recorded successful
controller. This makes the complete success controller an actual evaluated
node in every volume. A failure triple combined with the other nine success
gains is a hybrid controller; it is not labeled as the recorded failed flight.

The criterion is the exact empirical robust pole margin
`delta = 1 - max_i spectral_radius(F_i)` on 128 fixed scrambled-Sobol posterior
plant samples, using `conservative_fusion` covariance and seed 0. There is no
local gain surrogate, hidden-axis max projection, or stable-baseline
conditioning. Successful flight and this model's classification are separate:
the existing failure1-posterior calculation gives the complete success
controller a margin near `-1.96e-5`. The new volumes keep that result visible.

From the repository root, start a separate detached job:

```bash
tmux new-session -d -s pid-voxel-3d -c "$PWD" \
  'bash ros/examples/grape-param-estim/minimal/first-order-lag/run_pid_gain_voxels.sh'
tmux set-option -t pid-voxel-3d remain-on-exit on
tmux attach -t pid-voxel-3d
```

Run `failure2` concurrently in its own session and output directory, using
the same 128 plant samples, seed, covariance choice, grid schedule, and 16
workers per job:

```bash
tmux new-session -d -s pid-voxel-3d-failure2 -c "$PWD" \
  'env PID_VOXEL_OUTPUT="$PWD/ros/examples/grape-param-estim/minimal/first-order-lag/outputs/failure2/pid_gain_voxels_success_context" bash ros/examples/grape-param-estim/minimal/first-order-lag/run_pid_gain_voxels.sh --estimate-json ros/examples/grape-param-estim/minimal/first-order-lag/outputs/failure2/estimate.json --workers 16'
tmux set-option -t pid-voxel-3d-failure2 remain-on-exit on
tmux attach -t pid-voxel-3d-failure2
```

The posterior distribution comes from each respective failure bag; the other
nine controller gains are fixed to the same successful controller in both
jobs. The plant samples are drawn separately from those different posteriors.

Detach with `Ctrl-b`, then `d`. The wrapper uses `/usr/bin/python3`, 16 worker
processes, one BLAS thread per process, and `nice 10`. It writes a durable
`run.log` and does not use or terminate existing tmux sessions. For a shorter
run, pass `--grid-sizes 3 5 9 17` to the wrapper. The default refines through
3, 5, 9, 17, and 33 regular nodes per axis, inserting recorded P/I/D values
exactly. Actual axis sizes can therefore exceed the named regular grid size.
The same exact nodes are reused between levels; the plant sample set never
changes during refinement.

The default output is
`outputs/failure1/pid_gain_voxels_success_context/`. Open `index.html` for the
available 3D plots and download links; it updates as new batches finish.
Each `<group>/grid_<size>/` contains:

- `pid_3d.html`: standalone interactive P/I/D plot, with no CDN dependency.
- `pid_3d.png`: static scientific plot for export.
- `volume.vtr`: ParaView rectilinear grid with node-associated values.
- `volume.npz`: complete level, containing `p`, `i`, `d`, margins, per-plant
  spectral radii, masks, and classification. Array order is `[P, I, D]`, with
  plant sample as the last axis of `spectral_radius`.
- Individual `.npy` memmaps: partial data checkpointed after every batch.
- `metadata.json` and `status.json`: axis coordinates, fixed controller,
  calculation fingerprint, counts, and completion state.

Classification is 0 pending, 1 unresolved, 2 non-positive margin (red), and 3
positive margin (green). Pending nodes are omitted from plots; unresolved
nodes are gray. These are exact values at grid nodes, not a certification of
every point inside a voxel cell. The VTK file uses a rectilinear grid because
inserting the exact recorded gains makes the axes nonuniform. No interpolated
stability surface is asserted.

At the output root, `manifest.json`, `plant_samples.npz`, and `references.json`
retain input/source hashes, the exact ordered sample coordinates, all fixed
gains, and full-controller reference evaluations. `status.json` reports the
current group, level, progress, process ID, and measured time per node.

`Ctrl-c` in the job pane checkpoints the current batch and stops. Re-running
the same wrapper resumes the saved data. Input, code, covariance, seed, sample
count, and grid schedule must match; incompatible checkpoints are rejected.
An output lock prevents two processes from writing the same dataset.

To evaluate a different bag posterior, choose a separate output directory:

```bash
PID_VOXEL_OUTPUT=/tmp/pid-voxels-success-posterior \
  bash ros/examples/grape-param-estim/minimal/first-order-lag/run_pid_gain_voxels.sh \
  --estimate-json ros/examples/grape-param-estim/minimal/first-order-lag/outputs/success/estimate.json
```

The exploration caps reuse the previous PID-term clamp numbers; those numbers
are not enforced limits on controller gains.

## 4. Publish a file catalog when both calculations finish

The observer in `automation/watch_pid_voxel_catalog.py` waits for both
`failure1` and `failure2`, checks every scheduled volume and its exported
files, and waits for both calculation output locks to be released. It then
builds a catalog with local absolute paths, repository-relative paths, sizes,
SHA-256 checksums, calculation settings, and completed grid counts.

Start the observer independently of the numerical jobs:

```bash
tmux new-session -d -s pid-voxel-catalog -c "$PWD" \
  'bash ros/examples/grape-param-estim/minimal/first-order-lag/automation/run_pid_voxel_catalog.sh'
tmux set-option -t pid-voxel-catalog remain-on-exit on
```

After both jobs complete, it commits and pushes exactly these three files
to `origin/main`:

```text
minimal/first-order-lag/catalogs/pid_gain_voxels_success_context/catalog.json
minimal/first-order-lag/catalogs/pid_gain_voxels_success_context/files.tsv
minimal/first-order-lag/catalogs/pid_gain_voxels_success_context/README.md
```

Those paths are relative to `ros/examples/grape-param-estim/`. Calculation
data stay in their existing local output directories. The commit is built
in a detached temporary worktree on the latest remote branch; the primary
working tree, local branch, and staged changes are preserved. The catalog
can later be brought into the primary checkout with the usual pull.

Observer settings, progress, logs, and the publication commit ID are saved
under `outputs/pid_voxel_catalog_automation/`. `prepared/` retains a local
copy of the completed catalog. A partial preview can be generated without
publishing anything:

```bash
/usr/bin/python3 minimal/first-order-lag/automation/watch_pid_voxel_catalog.py --preview
```

The observer polls every 60 seconds, pins both current calculation
fingerprints and the destination URLs, and never publishes a partial
preview. Network failures and ordinary push rejections are retried using
the latest remote branch; the observer does not force-push. Terminal errors
are recorded in its `status.json` and `run.log`.


## 5. PID voxel domains with interior recorded gains

`run_pid_gain_voxels.sh` now selects `automation/pid_gain_voxels_interior.py` and defaults to a fresh `outputs/failure1/pid_gain_voxels_interior_success_context` directory. The frozen original driver and existing calculations retain their original domains.

| Group | P | I | D |
| --- | --- | --- | --- |
| XY | 1.5–6 | 0.025–0.3 | 0.25–3 |
| Z | 2–10 | 0.25–3 | 0.75–5 |
| Roll/Pitch | 5–25 | 0.25–3 | 4–28 |
| Yaw | 2–10 | 0.25–3 | 0.5–5 |

`automation/pid_voxel_domains.json` defines these physical bounds. Success, failure1 and failure2 recorded triples are all exact grid anchors and strictly interior at every refinement level. Validation rejects boundary anchors and requires at least 12.5% relative clearance on each axis; the current bounds provide at least 16.7%. Failure2 contributes reference anchors without changing the chosen case's posterior.

Validate the domains without creating a calculation or output directory:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 automation/pid_gain_voxels_interior.py --validate-domain-only
```

The gain normalization scales remain independent of the search bounds. Values beyond an old normalization scale, including Roll/Pitch D=28, reach the exact controller model without clipping. Source, domain configuration and recorded-anchor inputs enter the checkpoint fingerprint; a new domain cannot silently reinterpret an old checkpoint.

The new driver's plots retain the historical conditional-survival criterion: the fixed all-success-stable sample set, unresolved intervention samples counted as breaks, a 95% boundary and continuous RdYlGn colors on [0,1]. `conditional_survival.npz` is the primary classification export; the existing absolute-margin arrays remain numerical diagnostics. Recorded success and selected-case failure group gains are shown as separate markers.

## 6. Fast 65-point runs and naive cross-check data

`automation/pid_gain_voxels_fast.py` evaluates the same exact model and 128 fixed posterior samples with gain-aware hover-trim seeds. When I changes, the first six seed coordinates are scaled to preserve `K_I * integral_error`; orientation and thrust coordinates stay unchanged. The original equilibrium solver, fallbacks, analytic Jacobian and pole calculation are reused. Seeds from the nearest cached point receive the same correction. Zero I keeps the conservative original fallback, although the new search boxes have positive I throughout.

The refinement schedule is 3, 5, 9, 17, 33, 65 regular points per axis. Recorded gain anchors add a few points:

| Group | Final P × I × D shape | Final evaluated nodes |
| --- | --- | ---: |
| Roll/Pitch | 66 × 66 × 67 | 291,852 |
| XY | 67 × 66 × 67 | 296,274 |
| Z | 65 × 66 × 66 | 283,140 |
| Yaw | 65 × 66 × 66 | 283,140 |

Each case evaluates 1,154,406 distinct final-grid nodes; coarser results are inherited into nested grids. Failure1 and failure2 use separate posterior samples, output directories and worker pools. The new directories are:

```text
outputs/failure1/pid_gain_voxels_fast_interior_success_context/
outputs/failure2/pid_gain_voxels_fast_interior_success_context/
```

Start or resume each case from the repository root, using a different tmux pane for each:

```bash
code_dir="$PWD/ros/examples/grape-param-estim/minimal/first-order-lag"
PID_VOXEL_OUTPUT="$code_dir/outputs/failure1/pid_gain_voxels_fast_interior_success_context" \
  bash "$code_dir/run_pid_gain_voxels_fast.sh" \
  --estimate-json "$code_dir/outputs/failure1/estimate.json" --workers 16

PID_VOXEL_OUTPUT="$code_dir/outputs/failure2/pid_gain_voxels_fast_interior_success_context" \
  bash "$code_dir/run_pid_gain_voxels_fast.sh" \
  --estimate-json "$code_dir/outputs/failure2/estimate.json" --workers 16
```

The default preview interval is 15 minutes; completed refinement levels are exported immediately. Each output has `status.json`, `run.log`, `index.html` and the per-group grid data described above. Input/source fingerprints prevent a fast checkpoint from being mistaken for a naive one.

`automation/check_pid_voxel_fast.py` compares fast exact poles with the saved naive grid_17 cohorts for both cases and all four groups. The pre-launch check covered 142 gain nodes and 18,176 plant poles, with maximum radius difference approximately 8.4e-12 and identical stability decisions. Run metadata and this report are saved under `outputs/pid_voxel_fast_runs/`.

The frozen `pid_gain_voxels.py` and `outputs/{failure1,failure2}/pid_gain_voxels_success_context/` remain available for cross-checks. The already running naive failure2 job continues on its original schedule. The existing catalog observer remains attached to those original datasets.

## 7. Publish the fast-run file names after completion

`automation/watch_pid_voxel_fast_catalog.py` independently waits for both fast datasets to finish all scheduled levels through grid_65. It checks the pinned calculation fingerprints, every completed volume, the primary conditional-survival exports and released output locks. It then creates:

```text
catalogs/pid_gain_voxels_fast_interior_success_context/README.md
catalogs/pid_gain_voxels_fast_interior_success_context/files.tsv
catalogs/pid_gain_voxels_fast_interior_success_context/catalog.json
```

These paths are relative to this `first-order-lag` directory. The Markdown contains both absolute saved directories and all artifact file names; the TSV adds sizes and SHA-256 checksums, and the JSON retains calculation conditions. The observer commits these three inventory files and pushes them to `origin/main` using a detached worktree on the current remote tip. Ordinary network errors and push rejections are retried. The calculation data stay at their existing saved paths, and the main worktree/index are preserved.

The running observer has its own tmux session, `pid-voxel-fast-catalog`. Settings, progress, logs, a preview and the eventual publication commit ID are under `outputs/pid_voxel_fast_catalog_automation/`. The original naive-run observer has its separate control directory and catalog destination.

Start or resume the fast-run observer from the repository root:

```bash
tmux new-session -d -s pid-voxel-fast-catalog -c "$PWD" \
  'bash ros/examples/grape-param-estim/minimal/first-order-lag/automation/run_pid_voxel_fast_catalog.sh'
tmux set-option -t pid-voxel-fast-catalog remain-on-exit on
```

To inspect the current unpublished file list without publishing partial data:

```bash
/usr/bin/python3 ros/examples/grape-param-estim/minimal/first-order-lag/automation/watch_pid_voxel_fast_catalog.py --preview
```

## 8. Transparent voxel visualization of the completed fast runs

`visualizations/pid_gain_voxels_fast_interior/index.html` is an offline, interactive viewer of the completed 33- and 65-point grids. Failure1 and failure2 are selected separately; their posterior samples and conditioning baselines are kept separate. Roll/Pitch, Yaw, XY and Z can each be viewed.

The default colors are green for success-conditioned survival >=95% and red for <95%, using the same historical fixed baseline sample set. The previous continuous RdYlGn colors on [0,1] are available in the color selector. Successful gains and the selected case's recorded group gains remain exact markers. Both red and green regions have adjustable opacity, and the I selector provides a cross-section or cutaway.

Each calculated node has a display voxel whose edges are the midpoints to adjacent P/I/D nodes, clipped at the calculated domain endpoints. This respects the nonuniform axes introduced by recorded gain anchors. The renderer removes same-class internal faces and draws the remaining voxel faces and optional grid lines, with no fitted hull or interpolated stability surface. Voxel colors represent the node's saved value.

The builder checks the primary saved conditional-survival arrays against the original per-sample spectral radii and checks exact success anchors before constructing the viewer. The scientific datasets and already published file catalogs are unchanged. `viewer.json` retains source directories, fingerprints, group counts and recorded-reference values.

Rebuild from this `first-order-lag` directory:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /usr/bin/python3 automation/build_pid_voxel_viewer.py
node automation/test_pid_voxel_viewer.cjs visualizations/pid_gain_voxels_fast_interior/index.html
```

</details>
