from setuptools import find_packages, setup

from catkin_pkg.python_setup import generate_distutils_setup


setup_args = generate_distutils_setup(
    packages=sorted(find_packages("src")),
    package_dir={"": "src"},
)

setup(**setup_args)
