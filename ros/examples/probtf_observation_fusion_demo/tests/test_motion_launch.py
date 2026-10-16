"""Prevent the motion launches from silently selecting the old static demo."""
from pathlib import Path

import roslaunch


def test_motion_and_static_entry_selection_uses_roslaunch_boolean_arguments():
    launch = Path(__file__).resolve().parents[1] / 'launch/imu_marker.launch'
    for enabled, expected in ((True, 'observation_fusion_motion.py'),
                              (False, 'observation_fusion_demo.py')):
        config = roslaunch.config.ROSLaunchConfig()
        roslaunch.xmlloader.XmlLoader().load(
            str(launch), config, argv=['moving:='+str(enabled).lower()], verbose=False)
        node = next(node for node in config.nodes if node.name == 'observation_fusion_demo')
        assert node.type == expected
        view = next(node for node in config.nodes if node.name == 'observation_fusion_rviz')
        assert ('observation_fusion_motion.rviz' in view.args) == enabled
        assert ('/observation_fusion/robot_description' in config.params) == enabled
