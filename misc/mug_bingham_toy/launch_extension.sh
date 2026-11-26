#!/usr/bin/env bash
set -euo pipefail
MUG_EXTENSION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MUG_EXTENSION_RUN="${MUG_EXTENSION_DIR}/runs/vgg16_20000_cylinder_center_no_blur_20261127"
MUG_EXTENSION_DATA="${MUG_EXTENSION_DIR}/data/cylinder_center_no_blur_20000_20261127"
MUG_EXTENSION_RESULTS="${MUG_EXTENSION_DIR}/results/toyexample_vgg16_20000_cylinder_center_no_blur_100epoch"
MUG_EXTENSION_PYTHON="/home/leus/posecnn_ws/.venv/bin/python"
cd "${MUG_EXTENSION_DIR}"
mkdir -p "${MUG_EXTENSION_RUN}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR="${MUG_EXTENSION_RUN}/.mpl"
exec > >(tee -a "${MUG_EXTENSION_RUN}/train.log") 2>&1
while [[ ! -f "${MUG_EXTENSION_DATA}/train/metadata.json" ]]; do
    echo 'Waiting for completion of the 20,000-image dataset.'
    sleep 10
done
"${MUG_EXTENSION_PYTHON}" -u extend_training.py --learning-rate 1e-5 \
    --data "${MUG_EXTENSION_DATA}" --output "${MUG_EXTENSION_RUN}" \
    --monitor-inputs "${MUG_EXTENSION_RESULTS}/inputs.npz"
"${MUG_EXTENSION_PYTHON}" toyexample.py infer \
    --run "${MUG_EXTENSION_RUN}" --output "${MUG_EXTENSION_RESULTS}"
/usr/bin/python3 toyexample.py plot --output "${MUG_EXTENSION_RESULTS}" --samples 4096
