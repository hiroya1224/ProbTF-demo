#!/usr/bin/env python3
import os

import rospkg

from deflecomp_examples.offline_demo import main


def resolve_default_urdf() -> str:
    package = rospkg.RosPack().get_path("deflecomp_description")
    return os.path.join(package, "urdf", "simple6r.urdf")


if __name__ == "__main__":
    main(default_urdf=resolve_default_urdf())
