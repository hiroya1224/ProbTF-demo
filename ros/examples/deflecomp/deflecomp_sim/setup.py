from setuptools import find_packages, setup


SOURCE_DIR = "../../../../src"

setup(
    name="deflecomp_sim",
    version="0.1.0",
    packages=find_packages(where=SOURCE_DIR, include=["deflecomp_sim", "deflecomp_sim.*"]),
    package_dir={"": SOURCE_DIR},
)
