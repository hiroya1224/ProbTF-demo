#!/usr/bin/env python3
"""Use all native gyro records in the frozen real Student-t comparison."""
from pathlib import Path
import argparse

import run
import full_report


HERE = Path(__file__).resolve().parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'full_config.json')
    parser.add_argument('--output', type=Path, default=HERE/'publication/bag8_full_gyro')
    args = parser.parse_args()
    run.run(args)
    full_report.render(args.output)
