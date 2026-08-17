# Single-bag measured-gimbal SG estimator

`single_bag_savgol_estimator.py` estimates one bag at a time. The default
scientific path uses a 1.0 s, degree-5 geometric pose SG; the same centered
irregular-time SG for all four measured gimbal angles; identity acceleration
weighting; one estimated rotor-command lag; `epsilon_k = 2^-k` continuation
through `k=9`; and exact strict-ZOH cell refinement. Full SG covariance is
still calculated for reference diagnostics.

Each file in `bag_jsons/` contains only `bag_path`, `start_seconds`, and
`end_seconds`. A default run is:

```bash
python3 minimal/single_bag_savgol_estimator.py \
  --bag-json minimal/bag_jsons/single_rosbag_1.json \
  --vehicle-model minimal/grape_vehicle_model.json
```

The estimator remains prior-free unless `--prior-json` is supplied. Optional
v1 Gaussian factors are restricted to the exact common-scale-invariant
physical quotient: CoG, inertia/mass, and the four force-effectiveness/mass
ratios. For example:

```bash
python3 minimal/single_bag_savgol_estimator.py \
  --bag-json minimal/bag_jsons/single_rosbag_1.json \
  --vehicle-model minimal/grape_vehicle_model.json \
  --prior-json minimal/config/priors/pseudo_conditioning/group/cog_all_nominal.json
```

The schema, SI units, supported components, and covariance rules are documented
in [`config/priors/README.md`](config/priors/README.md). An ordinary prior is
external physical information with a defensible target and covariance. The
configs under `config/priors/pseudo_conditioning/` instead use deliberately
tight artificial standard deviations to probe parameter compensation; they
are not calibrated uncertainty and are not required for convergence.

The lag domain is inferred from recorded rotor-command prehistory. There is no
manual lag bound or gimbal lag. The primary lag result is the exact strict-ZOH
interval `(lower, upper]`, with its representative and final smooth lag also
saved.

`single_bag_savgol_ablation.py` provides the 21 fixed cases defined in
`../new_implementation_idea.md`. Optional focused sweeps are declared in
`single_bag_savgol_sweep.json`. Every completed case writes JSON, NPZ, and a
ten-page PDF beneath `minimal/outputs/<source-commit>/`.

Run all 21 fixed cases independently on all three production bags, followed by
the three-bag quotient/cross-evaluation consensus:

```bash
./minimal/run_full_ablation.sh
```

Run the 12-case focused validation (eight tied pose/gimbal SG window and
covariance combinations, plus three initial-lag multipliers, alongside the
default case) with:

```bash
GRAPE_ABLATION_FOCUSED_VALIDATION=true ./minimal/run_full_ablation.sh
```

Use `--dry-run` to print commands. The bag jobs and failure-isolated case jobs
run concurrently; `GRAPE_ABLATION_CASE_WORKERS` and
`GRAPE_ABLATION_NUMERIC_THREADS` control concurrency. Optimizer evaluation
ceilings default to 2000 and normal termination remains tolerance-driven.

`single_bag_cross_bag_consensus.py` remains post-fit: it does not create a
multi-bag estimator. It compares `J/m`, `f/m`, and CoG in the fixed 13-D scale
quotient and profiles only each target bag's rotor lag during cross-evaluation.

## Nominal pseudo-conditioning ablation

`single_bag_prior_ablation.py` runs the explicit manifest at
`config/prior_ablation/nominal_pseudo_conditioning.json`: one prior-free
baseline, 13 scalar factors, and three group factors. All 17 cases use the same
normal estimator entry point and default initialization; no case is warm-started
from another. Case failures are isolated. A completed strict point estimate is
retained and marked `point_estimate_completed` if only post-fit uncertainty
construction fails; no covariance is fabricated and no closure tolerance is
relaxed.

One-bag example:

```bash
python3 minimal/single_bag_prior_ablation.py \
  --bag-json minimal/bag_jsons/single_rosbag_1.json \
  --vehicle-model minimal/grape_vehicle_model.json \
  --manifest minimal/config/prior_ablation/nominal_pseudo_conditioning.json
```

Run the fixed 51-run production study (17 independent cases on each of the
three bags), then build the three-bag point-spread summary, with:

```bash
./minimal/run_prior_ablation.sh
```

Use `--dry-run` to inspect the exact commands or `--resume-existing` to retain
terminal per-case outputs from an interrupted fixed-ID production run.
`GRAPE_PRIOR_ABLATION_CASE_WORKERS` controls process concurrency; numeric
libraries default to one thread per process to prevent oversubscription.
The production wrapper raises both optimizer evaluation safety ceilings to
10000 while retaining tolerance-driven termination. For numerical
investigation, override them with `GRAPE_PRIOR_ABLATION_SMOOTH_MAX_NFEV` and
`GRAPE_PRIOR_ABLATION_STRICT_MAX_NFEV`.
Outputs live under
`minimal/outputs/<source-commit>/prior_ablation/<run-id>/` and include the
resolved prior targets, source SHA256 values, separate data/prior/total
objectives, six-page per-bag summaries, and a three-bag physical point-spread
report.

## Gimbalrotor PID static postprocess

`gimbalrotor_pid_postprocess.py` is a downstream, proposal-only calculation.
It combines one prior-free scale-free plant result with the existing nominal
Gimbalrotor controller allocation at zero gimbal, then fits multiplicative
scales for the `xy`, `z`, `roll_pitch`, and `yaw` PID groups. P, I, and D are
scaled together. The baseline gains are reconstructed from the selected ROS
bag's four recorded dynamic-reconfigure streams; YAML gain values are never a
fallback for what the flight used. The supplied controller YAML is only the
proposal-file template and controller-mode contract. The physical estimator,
controller limits, source controller YAML, and controller's nominal plant
model are not modified.

```bash
source /home/leus/catkin_ws/devel/setup.bash
python3 minimal/gimbalrotor_pid_postprocess.py \
  --result minimal/outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1/prior_ablation/single_rosbag_1_nominal_pseudo_conditioning_production_20260817/cases/prior_free/result.json \
  --bag-json minimal/bag_jsons/single_rosbag_1.json \
  --vehicle-model minimal/grape_vehicle_model.json \
  --controller-yaml /home/leus/catkin_ws/src/jsk_aerial_robot/robots/gimbalrotor/config/grape/GimbalrotorControl.yaml \
  --output-dir /tmp/grape-pid-proposal
```

The result contains raw and mixed-unit-normalized effectiveness matrices,
four gain scales, the exact bag-recorded baseline gain snapshot and its event
provenance, an overlay YAML, and a full proposal YAML. Unlike the purely static
matrix calculation, the postprocessor opens the bag to obtain those gains.
It reports the identified rotor lag but does not turn delay into an unsupported
static D-gain rule. Large scales or strong coupling are marked
`review_required`; generated gains are not flight-approved deployment values.
