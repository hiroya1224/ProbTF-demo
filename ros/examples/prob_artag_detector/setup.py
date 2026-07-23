from setuptools import find_packages, setup


setup(
    name="prob_artag_detector",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
