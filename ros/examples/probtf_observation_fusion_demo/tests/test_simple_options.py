"""The optional launch switch preserves explicit false and the YAML default."""

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from probtf_observation_fusion_demo.motion_node import optional_body_surprise


@pytest.mark.parametrize('value, expected', ((None, None), ('', None),
    (True, True), (False, False), ('true', True), ('false', False),
    ('TRUE', True), ('False', False)))
def test_optional_switch_preserves_false_and_yaml_default(value, expected):
    assert optional_body_surprise(value) is expected


@pytest.mark.parametrize('value', ('off', 'invalid', 1, 0))
def test_invalid_switch_does_not_silently_enable_body_surprise(value):
    with pytest.raises(ValueError, match='body_surprise'):
        optional_body_surprise(value)


@pytest.mark.parametrize('name', ('marker', 'imu_marker', 'imu', 'imu_hand', 'no_sensors'))
def test_case_launch_selects_current_mode_and_forwards_optional_switch(name):
    root = Path(__file__).resolve().parents[1]/'launch'
    launch = ET.parse(root/(name+'.launch')).getroot()
    options = {item.get('name'): item.get('default') for item in launch.findall('arg')}
    assert options['execution_mode'] == 'simple'
    assert options['body_surprise'] == ''
    included = launch.find('include')
    forwarded = {item.get('name'): item.get('value') for item in included.findall('arg')}
    assert forwarded['body_surprise'] == '$(arg body_surprise)'
    common = ET.parse(root/'common.launch').getroot()
    parameter = common.find("./node/param[@name='body_surprise']")
    assert parameter.get('type') == 'str'
    assert parameter.get('value') == '$(arg body_surprise)'
