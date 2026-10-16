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
        assert config.params['/observation_fusion_demo/execution_mode'].value == 'async'


def test_each_motion_entry_defaults_to_async_and_diagnostics_require_explicit_selection():
    package = Path(__file__).resolve().parents[1]
    for case in ('no_sensors', 'imu', 'marker', 'imu_marker'):
        for args, expected in (([], 'async'), (['execution_mode:=live'], 'live'), (['execution_mode:=prepared_playback'], 'prepared_playback')):
            config = roslaunch.config.ROSLaunchConfig()
            roslaunch.xmlloader.XmlLoader().load(str(package/'launch'/(case+'.launch')),
                                                config, argv=args, verbose=False)
            assert config.params['/observation_fusion_demo/execution_mode'].value == expected


def test_bounded_marker_experiment_uses_the_explicit_live_update_limit():
    package = Path(__file__).resolve().parents[1]
    config = roslaunch.config.ROSLaunchConfig()
    roslaunch.xmlloader.XmlLoader().load(str(package/'launch/marker.launch'),
        config, argv=['max_updates:=6', 'execution_mode:=live'], verbose=False)
    assert config.params['/observation_fusion_demo/execution_mode'].value == 'live'
    assert config.params['/observation_fusion_demo/max_updates'].value == 6
