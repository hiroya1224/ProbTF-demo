from setuptools import find_packages, setup


SOURCE_DIR = "../../../../src"

setup(
    name="deflecomp_examples",
    version="0.1.0",
    packages=find_packages(where=SOURCE_DIR, include=["deflecomp_examples", "deflecomp_examples.*"]),
    package_dir={"": SOURCE_DIR},
)
