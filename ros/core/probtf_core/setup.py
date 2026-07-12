from pathlib import Path

from setuptools import find_packages, setup


PACKAGE_ROOT = Path(__file__).resolve().parent
PROBTF_ROOT = (PACKAGE_ROOT / "../../../src").resolve()
BINGHAM_ROOT = (PACKAGE_ROOT / "../../../third_party/BinghamNLL/src").resolve()

probtf_packages = [
    package
    for package in find_packages(where=str(PROBTF_ROOT))
    if package == "probtf" or package.startswith("probtf.")
]
bingham_packages = [
    package
    for package in find_packages(where=str(BINGHAM_ROOT))
    if package == "bingham" or package.startswith("bingham.")
]

setup(
    name="probtf_core",
    version="0.1.0",
    packages=probtf_packages + bingham_packages + ["probtf_ros"],
    package_dir={
        "probtf": str(PROBTF_ROOT / "probtf"),
        "bingham": str(BINGHAM_ROOT / "bingham"),
        "probtf_ros": "src/probtf_ros",
    },
)
