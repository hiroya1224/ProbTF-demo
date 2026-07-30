#!/usr/bin/env python3
"""Launch the interactive Grape failed-bag analysis application."""

import sys

from grape_param_estim.failure_analysis_gui import main


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(
            "{}: {}".format(type(error).__name__, error),
            file=sys.stderr,
        )
        sys.exit(2)
