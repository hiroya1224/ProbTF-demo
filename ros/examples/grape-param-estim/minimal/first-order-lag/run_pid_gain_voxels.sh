#!/usr/bin/env bash
# Run in a new tmux session.  Extra arguments are passed to the Python driver.
set -euo pipefail

code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${PID_VOXEL_OUTPUT:-${code_dir}/outputs/failure1/pid_gain_voxels_interior_failure_context}"
mkdir -p -- "$output_dir"
export MPLCONFIGDIR="$output_dir/.mpl-cache"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

nice -n 10 /usr/bin/python3 -u "$code_dir/automation/pid_gain_voxels_interior.py" \
    --output "$output_dir" "$@" 2>&1 | tee -a "$output_dir/run.log"
