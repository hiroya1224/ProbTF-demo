"""Shared setuptools configuration for the ROS-free native sampler."""

import os

from setuptools import Extension


def sampling_extension(core_directory):
    return Extension(
        "probtf._sampling",
        sources=[os.path.join(core_directory, "src", filename)
                 for filename in ("bingham_sampler.cpp", "python_sampling.cpp")],
        include_dirs=[os.path.join(core_directory, "include"),
                      os.environ.get("EIGEN3_INCLUDE_DIR", "/usr/include/eigen3")],
        language="c++",
        extra_compile_args=["-std=c++14", "-O3"],
    )
