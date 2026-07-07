from setuptools import setup

from catkin_pkg.python_setup import generate_distutils_setup


setup_args = generate_distutils_setup(
    packages=["probik_demo", "probik_demo.prob_tf"],
    package_dir={"": "src"},
)

setup(**setup_args)
