from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
for source in (PACKAGE_ROOT / "gui" / "src", PACKAGE_ROOT / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
