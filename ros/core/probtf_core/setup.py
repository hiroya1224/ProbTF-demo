from pathlib import Path

from setuptools import find_packages, setup


PACKAGE_ROOT = Path(__file__).resolve().parent
PROBTF_SOURCE = "../../../src"
BINGHAM_SOURCE = "../../../third_party/BinghamNLL/src"
PROBTF_ROOT = (PACKAGE_ROOT / PROBTF_SOURCE).resolve()
BINGHAM_ROOT = (PACKAGE_ROOT / BINGHAM_SOURCE).resolve()

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
        "probtf": PROBTF_SOURCE + "/probtf",
        "bingham": BINGHAM_SOURCE + "/bingham",
        "probtf_ros": "src/probtf_ros",
    },
)
