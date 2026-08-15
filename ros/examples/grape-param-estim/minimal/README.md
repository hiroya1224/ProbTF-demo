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
