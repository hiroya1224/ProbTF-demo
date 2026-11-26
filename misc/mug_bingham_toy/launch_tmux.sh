#!/usr/bin/env bash
set -euo pipefail
TOY_MUG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOY_MUG_RUN="${TOY_MUG_DIR}/runs/vgg16_10000_cylinder_center_no_blur_20261127"
TOY_MUG_PYTHON="/home/leus/posecnn_ws/.venv/bin/python"
cd "${TOY_MUG_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR="${TOY_MUG_RUN}/.mpl"
case "${1:-launch}" in
    render)
        /usr/bin/python3 -u render.py --output data/cylinder_center_no_blur_20261127 2>&1 | tee "${TOY_MUG_RUN}/render.log"
        ;;
    train)
        bash run.sh
        /usr/bin/python3 toyexample.py render
        "${TOY_MUG_PYTHON}" toyexample.py infer
        /usr/bin/python3 toyexample.py plot
        ;;
    status)
        exec /usr/bin/python3 -u status.py --watch
        ;;
    live)
        export DISPLAY="${DISPLAY:-:1}"
        export QT_QPA_PLATFORM=xcb
        exec /usr/bin/python3 live_training_view.py \
            --run "${TOY_MUG_DIR}/results/toyexample_vgg16_10000_cylinder_center_no_blur" \
            --inputs "${TOY_MUG_DIR}/results/toyexample_vgg16_10000_cylinder_center_no_blur/inputs.npz" --samples 4096
        ;;
    front)
        exec "${TOY_MUG_PYTHON}" -u front_view_watch.py
        ;;
    launch)
        if tmux has-session -t mug-bingham-toy 2>/dev/null; then
            echo 'mug-bingham-toy already exists; inspect it before starting another job.' >&2
            exit 1
        fi
        mkdir -p "${TOY_MUG_RUN}"
        tmux new-session -d -s mug-bingham-toy -n render -c "${TOY_MUG_DIR}" /bin/bash "${TOY_MUG_DIR}/launch_tmux.sh" render
        tmux set-window-option -t mug-bingham-toy:render remain-on-exit on
        tmux new-window -d -t mug-bingham-toy -n train -c "${TOY_MUG_DIR}" /bin/bash "${TOY_MUG_DIR}/launch_tmux.sh" train
        tmux set-window-option -t mug-bingham-toy:train remain-on-exit on
        tmux new-window -d -t mug-bingham-toy -n status -c "${TOY_MUG_DIR}" /bin/bash "${TOY_MUG_DIR}/launch_tmux.sh" status
        tmux new-window -d -t mug-bingham-toy -n live -c "${TOY_MUG_DIR}" /bin/bash "${TOY_MUG_DIR}/launch_tmux.sh" live
        echo 'Started cylindrical-center regeneration and fresh training: tmux a -t mug-bingham-toy'
        ;;
    *)
        echo 'Usage: bash launch_tmux.sh [launch|render|train|status|live]' >&2
        exit 2
        ;;
esac
