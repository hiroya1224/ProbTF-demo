from setuptools import setup

from catkin_pkg.python_setup import generate_distutils_setup


setup_args = generate_distutils_setup(
    packages=[
        "deflecomp_core",
        "deflecomp_core.control",
        "deflecomp_core.estimator",
        "deflecomp_core.model",
        "deflecomp_core.observation",
        "deflecomp_core.pipeline",
        "deflecomp_core.robot",
        "deflecomp_core.utils",
    ],
    package_dir={"": "src"},
)

setup(**setup_args)
