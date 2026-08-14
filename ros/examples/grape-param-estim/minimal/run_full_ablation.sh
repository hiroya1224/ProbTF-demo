#!/usr/bin/env bash

set -eo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
workspace_root="$(cd -- "${project_root}/../../../../.." && pwd)"
catkin_setup="${GRAPE_CATKIN_SETUP:-${workspace_root}/devel/setup.bash}"

usage() {
  printf 'Usage: %s [--dry-run]\n' "${0##*/}"
}

dry_run=false
case "${1:-}" in
  "") ;;
  --dry-run) dry_run=true ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
if (( $# > 1 )); then
  usage >&2
  exit 2
fi

if [[ ! -f "${catkin_setup}" ]]; then
  printf 'catkin setup not found: %s\n' "${catkin_setup}" >&2
  printf 'Set GRAPE_CATKIN_SETUP to the correct setup.bash path.\n' >&2
  exit 1
fi

# Catkin setup scripts are not guaranteed to be nounset-safe.
source "${catkin_setup}"
set -u

bag_ids=(
  single_rosbag_1
  single_rosbag_2
  single_rosbag_succeeded
)
declare -A rotor_lag=(
  [single_rosbag_1]=0.0049977302551269531
  [single_rosbag_2]=0.0049576759338378906
  [single_rosbag_succeeded]=0.010000109672546387
)
declare -A gimbal_lag=(
  [single_rosbag_1]=0.0049986839294433594
  [single_rosbag_2]=0.0049571990966796875
  [single_rosbag_succeeded]=0.010000109672546387
)

estimator="${script_dir}/single_bag_savgol_ablation.py"
vehicle_model="${script_dir}/grape_vehicle_model.json"
bag_json_dir="${script_dir}/bag_jsons"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_stamp="${GRAPE_ABLATION_RUN_STAMP:-${run_stamp}}"
case_workers="${GRAPE_ABLATION_CASE_WORKERS:-3}"
numeric_threads="${GRAPE_ABLATION_NUMERIC_THREADS:-6}"
resume_existing="${GRAPE_ABLATION_RESUME_EXISTING:-false}"
overall_status=0

if [[ ! "${case_workers}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'GRAPE_ABLATION_CASE_WORKERS must be a positive integer\n' >&2
  exit 2
fi
if [[ ! "${numeric_threads}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'GRAPE_ABLATION_NUMERIC_THREADS must be a positive integer\n' >&2
  exit 2
fi
if [[ "${resume_existing}" != false && "${resume_existing}" != true ]]; then
  printf 'GRAPE_ABLATION_RESUME_EXISTING must be true or false\n' >&2
  exit 2
fi

export OMP_NUM_THREADS="${numeric_threads}"
export OPENBLAS_NUM_THREADS="${numeric_threads}"
export MKL_NUM_THREADS="${numeric_threads}"
export NUMEXPR_NUM_THREADS="${numeric_threads}"

for required_file in "${estimator}" "${vehicle_model}"; do
  if [[ ! -f "${required_file}" ]]; then
    printf 'required file not found: %s\n' "${required_file}" >&2
    exit 1
  fi
done

for bag_id in "${bag_ids[@]}"; do
  bag_json="${bag_json_dir}/${bag_id}.json"
  if [[ ! -f "${bag_json}" ]]; then
    printf 'bag JSON not found: %s\n' "${bag_json}" >&2
    exit 1
  fi
done

run_bag() {
  local bag_id="$1"
  local bag_json="${bag_json_dir}/${bag_id}.json"
  local -a ablation_command=(
    python3 "${estimator}"
    --bag-json "${bag_json}"
    --vehicle-model "${vehicle_model}"
    --sg-window 0.5
    --lag-bounds 0.0 0.2
    --initial-rotor-lag "${rotor_lag[$bag_id]}"
    --initial-gimbal-lag "${gimbal_lag[$bag_id]}"
    --fixed-rotor-lag "${rotor_lag[$bag_id]}"
    --fixed-gimbal-lag "${gimbal_lag[$bag_id]}"
    --kkt-scale-offsets -1.0 0.0 1.0
    --smooth-max-nfev 2000
    --strict-max-nfev 2000
    --cases all
    --case-workers "${case_workers}"
    --ablation-run-id "${bag_id}_${run_stamp}"
  )
  if [[ "${resume_existing}" == true ]]; then
    ablation_command+=(--resume-existing)
  fi

  printf '[%s] full ablation (%d fixed cases, %s workers)\n' \
    "${bag_id}" 29 "${case_workers}"
  if [[ "${dry_run}" == true ]]; then
    printf '  %q' "${ablation_command[@]}"
    printf '\n'
    return 0
  fi
  "${ablation_command[@]}"
}

if [[ "${dry_run}" == true ]]; then
  for bag_id in "${bag_ids[@]}"; do
    run_bag "${bag_id}"
  done
  exit 0
fi

declare -A bag_pid=()

terminate_children() {
  trap - INT TERM
  if (( ${#bag_pid[@]} )); then
    kill "${bag_pid[@]}" 2>/dev/null || true
    for pid in "${bag_pid[@]}"; do
      wait "${pid}" 2>/dev/null || true
    done
  fi
  exit 130
}

trap terminate_children INT TERM
for bag_id in "${bag_ids[@]}"; do
  run_bag "${bag_id}" &
  bag_pid["${bag_id}"]=$!
done

for bag_id in "${bag_ids[@]}"; do
  if ! wait "${bag_pid[$bag_id]}"; then
    printf '[%s] runner failed; other bags will continue.\n' \
      "${bag_id}" >&2
    overall_status=1
  fi
done
trap - INT TERM

exit "${overall_status}"
