#!/usr/bin/env bash
set -euo pipefail

automation_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
control_dir="${PID_VOXEL_CATALOG_CONTROL:-${automation_dir}/../outputs/pid_voxel_catalog_automation}"
mkdir -p -- "$control_dir"
export GIT_TERMINAL_PROMPT=0
nice -n 15 /usr/bin/python3 -u "$automation_dir/watch_pid_voxel_catalog.py" \
    --control "$control_dir" "$@" 2>&1 | tee -a "$control_dir/run.log"
