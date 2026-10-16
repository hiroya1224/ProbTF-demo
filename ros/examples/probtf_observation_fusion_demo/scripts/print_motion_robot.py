#!/usr/bin/python3
"""Publish known robot geometry to roslaunch before RViz starts."""
from pathlib import Path
from tempfile import TemporaryDirectory
import rospkg
import yaml
from probtf_switch_press_demo.scene import write_robot

packages = rospkg.RosPack()
package = Path(packages.get_path('probtf_observation_fusion_demo'))
config = yaml.safe_load((package/'config/switch_press_scene.yaml').read_text())
source = Path(packages.get_path('symaware_grasp'))/'urdf/simple_six_dof_arm.urdf'
with TemporaryDirectory(prefix='probtf_motion_robot_') as scratch:
    print(write_robot(source, Path(scratch)/'robot.urdf', config).read_text())
