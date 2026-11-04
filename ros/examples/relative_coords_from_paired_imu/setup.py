from setuptools import find_packages, setup

setup(
    name="relative_coords_from_paired_imu",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
