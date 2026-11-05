# IMU-only PID map artifacts

`imu_only_pid_maps.pdf` contains the 32 selected source pages; `page_index.csv` gives their order. Individual PNGs, compressed grid data, and assessment/plot metadata are byte-exact copies. Only the final requested 33/65 refinements are included; actual axis sizes (including inserted failure-gain anchors) are recorded per grid.

The strict full-state stability fraction is zero by the analytic translation-observability certificate. No numerical poles or hover equilibria were calculated. The three unanchored position modes already rule out strict full-state spectral radius below one. This is not a flight-failure probability or a computation of the remaining poles.

No augmented observer trajectory was simulated: the certificate applies to any causal observer that receives only the declared IMU measurements and no world-position anchor. See [the model definition](../../IMU_ONLY_PID_MAPS.md) for its scope.

Each case has one shared `model_inputs.npz`. Per-case/view manifests and status files preserve the original run history, including references to intermediate grids which are deliberately not published. Plot metadata preserves original source filenames; its individual PDF entry is represented by the corresponding page of the combined PDF. No success plant parameters, raw bags, caches, or diagnostic alternative models are included.

From the `imu-first-order-lag` directory, publish a completed run into a new destination with:

```sh
python3 publish_imu_pid_maps.py /tmp/grape-imu-only-pid-20261106 publication/imu_only_pid_maps_20261106
```

The [publisher](../../publish_imu_pid_maps.py) changes no plotted values. Use the [runner](../../run_imu_pid_maps.py) and [model definition](../../IMU_ONLY_PID_MAPS.md) to recreate the source run.
