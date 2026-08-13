# Single-bag geometric SG estimator

`single_bag_savgol_estimator.py` fits one rosbag with the prior-free,
full-covariance acceleration objective specified in
`../single_bag_savgol_ablation_implementation_plan.md`.

Each file in `bag_jsons/` contains only `bag_path`, `start_seconds`, and
`end_seconds`. Other algorithm settings are supplied explicitly on the CLI;
the JSON loader ignores any other member. A typical invocation is:

```bash
python3 minimal/single_bag_savgol_estimator.py \
  --bag-json minimal/bag_jsons/single_rosbag_1.json \
  --vehicle-model MODEL.json --sg-window WINDOW \
  --lag-mode split_estimated --lag-bounds LOWER UPPER \
  --initial-rotor-lag ROTOR --initial-gimbal-lag GIMBAL
```

The direct `--bag BAG.bag --bag-start START --bag-end END` form remains
available. Run the command once per JSON; no joint multi-bag objective is
formed.

Both smooth and strict optimizer evaluation ceilings default to 2000. Normal
termination is controlled by `gtol`, `ftol`, and `xtol`; the ceiling is a
failure guard rather than the expected stopping condition.

`single_bag_savgol_ablation.py` exposes every fixed case and reads optional,
explicit sweep values from `--sweep-config`.  Outputs are written beneath
`minimal/outputs/<source-commit>/`.

Run all 29 fixed cases independently for all three bag JSONs with:

```bash
./minimal/run_full_ablation.sh
```

The embedded lag seeds are the median recorded command periods in each
selected bag interval. They are not read from the
discarded JSON options. Use `--dry-run` to print the three commands without
starting the estimators.

The previous `minimal/` tree, including its tracked `output/`, is retained at
`minimal/legacies/pre_single_bag_rewrite_7fecffe/`.  New modules never import
that snapshot at runtime.
