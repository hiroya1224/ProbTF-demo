from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
BINGHAM_SOURCE = ROOT / "third_party" / "BinghamNLL" / "src"
BINGHAM_PACKAGE_DIR = "third_party/BinghamNLL/src/bingham"

if not (BINGHAM_SOURCE / "bingham" / "__init__.py").is_file():
    raise RuntimeError(
        "BinghamNLL is not initialized. Run "
        "'git submodule update --init --recursive' before installing."
    )

packages = find_packages(where="src") + find_packages(where=str(BINGHAM_SOURCE))

setup(
    packages=packages,
    package_dir={
        "": "src",
        "bingham": BINGHAM_PACKAGE_DIR,
    },
)
