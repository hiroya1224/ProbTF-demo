#!/usr/bin/env bash
set -euo pipefail
TOY_MUG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOY_MUG_PYTHON="${TOY_MUG_PYTHON:-/home/leus/posecnn_ws/.venv/bin/python}"
TOY_MUG_RUN="${TOY_MUG_RUN:-${TOY_MUG_DIR}/runs/vgg16_10000_20261126}"
mkdir -p "${TOY_MUG_RUN}"
cd "${TOY_MUG_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR="${TOY_MUG_RUN}/.mpl"
exec > >(tee -a "${TOY_MUG_RUN}/train.log") 2>&1
while [[ ! -f data/train/metadata.json || ! -f data/val/metadata.json || ! -f data/test/metadata.json ]]; do
    /usr/bin/python3 status.py
    sleep 10
done
exec "${TOY_MUG_PYTHON}" -u train.py --data "${TOY_MUG_DIR}/data" \
    --output "${TOY_MUG_RUN}" --epochs 50 --batch-size 16 --device cuda --seed 20261126 "$@"
