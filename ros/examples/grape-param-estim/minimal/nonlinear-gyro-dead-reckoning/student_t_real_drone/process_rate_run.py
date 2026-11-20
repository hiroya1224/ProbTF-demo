#!/usr/bin/env python3
"""Previously used gyro selections with the same Student-t and Q sigma 1."""
from pathlib import Path
import argparse

import process_run


HERE = Path(__file__).resolve().parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'process_rate_config.json')
    parser.add_argument('--output',type=Path,default=HERE/'publication/bag8_selected_gyro_q1')
    process_run.run(parser.parse_args())
