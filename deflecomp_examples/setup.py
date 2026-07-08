from setuptools import setup

from catkin_pkg.python_setup import generate_distutils_setup


setup_args = generate_distutils_setup(
    packages=["deflecomp_examples", "deflecomp_examples.ik", "deflecomp_examples.utils"],
    package_dir={"": "src"},
)

setup(**setup_args)
