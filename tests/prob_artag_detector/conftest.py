import sys
from pathlib import Path


PACKAGE_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "ros"
    / "examples"
    / "prob_artag_detector"
    / "src"
)
sys.path.insert(0, str(PACKAGE_SOURCE))
