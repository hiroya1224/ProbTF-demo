from setuptools import find_packages, setup


PACKAGE_ROOTS = (
    "ros/core/probtf_core/src",
    "ros/examples/deflecomp/deflecomp_core/src",
    "ros/examples/deflecomp/deflecomp_examples/src",
    "ros/examples/deflecomp/deflecomp_sim/src",
    "ros/examples/grape-param-estim/src",
    "ros/examples/symaware_grasp/src",
)

packages = []
package_dir = {}
for source_root in PACKAGE_ROOTS:
    discovered = find_packages(where=source_root)
    packages.extend(discovered)
    for package in discovered:
        top_level = package.split(".", 1)[0]
        package_dir.setdefault(top_level, "{}/{}".format(source_root, top_level))

setup(
    packages=packages,
    package_dir=package_dir,
)
