#!/usr/bin/env bash
set -euo pipefail
TOY_MUG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOY_MUG_PYTHON="${TOY_MUG_PYTHON:-/home/leus/posecnn_ws/.venv/bin/python}"
TOY_MUG_DATA="${TOY_MUG_DATA:-${TOY_MUG_DIR}/data/cylinder_center_20261127}"
TOY_MUG_RUN="${TOY_MUG_RUN:-${TOY_MUG_DIR}/runs/vgg16_10000_cylinder_center_20261127}"
if (( $# )); then
    echo "run.sh starts the fixed cylindrical-center experiment from scratch and accepts no training overrides or --resume." >&2
    exit 2
fi
mkdir -p "${TOY_MUG_RUN}"
cd "${TOY_MUG_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR="${TOY_MUG_RUN}/.mpl"
exec > >(tee -a "${TOY_MUG_RUN}/train.log") 2>&1
while [[ ! -f "${TOY_MUG_DATA}/train/metadata.json" || ! -f "${TOY_MUG_DATA}/val/metadata.json" || ! -f "${TOY_MUG_DATA}/test/metadata.json" ]]; do
    /usr/bin/python3 status.py --data "${TOY_MUG_DATA}" --run "${TOY_MUG_RUN}"
    sleep 10
done
/usr/bin/python3 status.py --check-launch --data "${TOY_MUG_DATA}" --run "${TOY_MUG_RUN}"
exec "${TOY_MUG_PYTHON}" -u train.py --data "${TOY_MUG_DATA}" \
    --output "${TOY_MUG_RUN}" --epochs 50 --batch-size 16 --device cuda --seed 20261126
