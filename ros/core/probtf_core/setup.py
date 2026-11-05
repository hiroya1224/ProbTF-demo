import os
from runpy import run_path

from setuptools import find_packages, setup

CORE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

setup(
    name="probtf_core",
    version="0.1.0",
    install_requires=["numpy>=1.20", "scipy>=1.6", "PyYAML>=5.3"],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[run_path(os.path.join(CORE_DIRECTORY, "native_build.py"))[
        "sampling_extension"](".")],
)
