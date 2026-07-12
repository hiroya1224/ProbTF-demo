from setuptools import setup

from catkin_pkg.python_setup import generate_distutils_setup


setup_args = generate_distutils_setup(packages=["deflecomp_sim"], package_dir={"": "src"})

setup(**setup_args)
