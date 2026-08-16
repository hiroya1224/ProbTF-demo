#!/usr/bin/env bash

set -eo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
workspace_root="$(cd -- "${project_root}/../../../../.." && pwd)"
catkin_setup="${GRAPE_CATKIN_SETUP:-${workspace_root}/devel/setup.bash}"

usage() {
  printf 'Usage: %s [--dry-run] [--resume-existing]\n' "${0##*/}"
}

dry_run=false
resume_existing=false
while (( $# )); do
  case "$1" in
    --dry-run) dry_run=true ;;
    --resume-existing) resume_existing=true ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ! -f "${catkin_setup}" ]]; then
  printf 'catkin setup not found: %s\n' "${catkin_setup}" >&2
  printf 'Set GRAPE_CATKIN_SETUP to the correct setup.bash path.\n' >&2
  exit 1
fi

# Catkin setup scripts are not guaranteed to be nounset-safe.
source "${catkin_setup}"
set -u

bag_ids=(single_rosbag_1 single_rosbag_2 single_rosbag_succeeded)
run_ids=(
  single_rosbag_1_nominal_pseudo_conditioning_production_20260817
  single_rosbag_2_nominal_pseudo_conditioning_production_20260817
  single_rosbag_succeeded_nominal_pseudo_conditioning_production_20260817
)
aggregate_run_id=three_bag_nominal_pseudo_conditioning_production_20260817
runner="${script_dir}/single_bag_prior_ablation.py"
aggregator="${script_dir}/three_bag_prior_ablation_summary.py"
vehicle_model="${script_dir}/grape_vehicle_model.json"
manifest="${script_dir}/config/prior_ablation/nominal_pseudo_conditioning.json"
bag_json_dir="${script_dir}/bag_jsons"
case_workers="${GRAPE_PRIOR_ABLATION_CASE_WORKERS:-4}"
numeric_threads="${GRAPE_PRIOR_ABLATION_NUMERIC_THREADS:-1}"
source_revision="$(git -C "${project_root}" rev-parse HEAD)"
output_namespace="${script_dir}/outputs/${source_revision}/prior_ablation"

if [[ ! "${case_workers}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'GRAPE_PRIOR_ABLATION_CASE_WORKERS must be a positive integer\n' >&2
  exit 2
fi
if [[ ! "${numeric_threads}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'GRAPE_PRIOR_ABLATION_NUMERIC_THREADS must be a positive integer\n' >&2
  exit 2
fi

export OMP_NUM_THREADS="${numeric_threads}"
export OPENBLAS_NUM_THREADS="${numeric_threads}"
export MKL_NUM_THREADS="${numeric_threads}"
export NUMEXPR_NUM_THREADS="${numeric_threads}"

for required_file in "${runner}" "${aggregator}" "${vehicle_model}" "${manifest}"; do
  if [[ ! -f "${required_file}" ]]; then
    printf 'required file not found: %s\n' "${required_file}" >&2
    exit 1
  fi
done
for bag_id in "${bag_ids[@]}"; do
  if [[ ! -f "${bag_json_dir}/${bag_id}.json" ]]; then
    printf 'bag JSON not found: %s\n' "${bag_json_dir}/${bag_id}.json" >&2
    exit 1
  fi
done

run_bag() {
  local index="$1"
  local bag_id="${bag_ids[$index]}"
  local run_id="${run_ids[$index]}"
  local -a command=(
    python3 "${runner}"
    --bag-json "${bag_json_dir}/${bag_id}.json"
    --vehicle-model "${vehicle_model}"
    --manifest "${manifest}"
    --case-workers "${case_workers}"
    --ablation-run-id "${run_id}"
  )
  if [[ "${resume_existing}" == true ]]; then
    command+=(--resume-existing)
  fi
  printf '[%s] 17 independent estimator cases (%s workers)\n' "${bag_id}" "${case_workers}"
  if [[ "${dry_run}" == true ]]; then
    printf '  %q' "${command[@]}"
    printf '\n'
    return 0
  fi
  "${command[@]}"
}

run_aggregate() {
  local -a command=(
    python3 "${aggregator}"
    --ablation-directory
    "${output_namespace}/${run_ids[0]}"
    "${output_namespace}/${run_ids[1]}"
    "${output_namespace}/${run_ids[2]}"
    --aggregate-run-id "${aggregate_run_id}"
  )
  printf '[aggregate] three-bag point-spread summary\n'
  if [[ "${dry_run}" == true ]]; then
    printf '  %q' "${command[@]}"
    printf '\n'
    return 0
  fi
  if [[ "${resume_existing}" == true && -f "${output_namespace}/${aggregate_run_id}/status.json" ]]; then
    printf '[aggregate] existing completed directory retained: %s\n' "${output_namespace}/${aggregate_run_id}"
    return 0
  fi
  "${command[@]}"
}

if [[ "${dry_run}" == true ]]; then
  for index in "${!bag_ids[@]}"; do
    run_bag "${index}"
  done
  run_aggregate
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

for index in "${!bag_ids[@]}"; do
  run_bag "${index}" &
  bag_pid["${bag_ids[$index]}"]=$!
done

overall_status=0
for bag_id in "${bag_ids[@]}"; do
  if ! wait "${bag_pid[$bag_id]}"; then
    printf '[%s] runner failed; other bag jobs were allowed to finish.\n' "${bag_id}" >&2
    overall_status=1
  fi
done
trap - INT TERM

if (( overall_status == 0 )); then
  run_aggregate
fi
exit "${overall_status}"
