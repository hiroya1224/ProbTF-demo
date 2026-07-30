#!/usr/bin/env python3
"""Launch the Grape Streamlit application through ``rosrun``."""

import argparse
import importlib.util
import os
from pathlib import Path
import sys

import rospkg


def main() -> None:
    package_path = Path(rospkg.RosPack().get_path("grape_param_estim"))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(package_path / "config" / "default.yaml"),
        help="YAML file used to initialise the GUI",
    )
    arguments = parser.parse_args()
    if importlib.util.find_spec("streamlit") is None:
        parser.error(
            "Streamlit is not installed. Run: "
            "python3 -m pip install --user -r {}/requirements.txt".format(
                package_path
            )
        )
    application = (
        package_path / "src" / "grape_param_estim" / "app.py"
    )
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(application),
            "--",
            "--config",
            str(Path(arguments.config).expanduser()),
        ],
    )


if __name__ == "__main__":
    main()
