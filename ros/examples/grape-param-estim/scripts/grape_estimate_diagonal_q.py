#!/usr/bin/env python3

import os

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

from grape_param_estim.diagonal_q_stage_cli import main


if __name__ == "__main__":
    main()
