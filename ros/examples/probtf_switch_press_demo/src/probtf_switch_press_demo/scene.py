"""Known geometry and a generated, mass-bearing extension of the demo arm."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def transform(translation, rpy=(0.0, 0.0, 0.0)):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    value[:3, 3] = translation
    return value


def inverse(value):
    result = np.eye(4)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ value[:3, 3]
    return result


def apply(value, points):
    return np.asarray(points) @ value[:3, :3].T + value[:3, 3]


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if min(config["rod"].values()) <= 0:
        raise ValueError("rod dimensions and mass must be positive")
    base = np.asarray(config["base_translation"], dtype=float)
    if base.shape != (3,) or not np.all(np.isfinite(base)):
        raise ValueError("base_translation must contain three finite coordinates")
    for key in ("noise_std", "blur_sigma", "exposure_time"):
        if not np.isfinite(config["camera"][key]) or config["camera"][key] < 0:
            raise ValueError(key + " must be nonnegative and finite")
    if not isinstance(config["camera"]["motion_blur_samples"], int) or config["camera"]["motion_blur_samples"] < 1:
        raise ValueError("motion_blur_samples must be a positive integer")
    if not config["control"].get("initial_transform_law"):
        raise ValueError("control.initial_transform_law must name the configured link transform law")
    motion = config["motion"]
    if not 0 < motion["spatial_resolution"] < motion["clearance"]:
        raise ValueError("spatial resolution must be positive and below clearance")
    for key in ("frame_period", "max_joint_velocity", "max_joint_acceleration", "tracking_time"):
        if not np.isfinite(motion[key]) or motion[key] <= 0:
            raise ValueError(key + " must be positive and finite")
    if config["reach_tolerance"] <= 0:
        raise ValueError("reach tolerance must be positive")
    return config


def write_robot(source, destination, config):
    """Keep the existing six-axis arm; replace the gripper with a rigid rod."""
    root = ET.parse(str(source)).getroot()
    root.set("name", "switch_press_arm")
    flange = root.find("./link[@name='tool0']")
    for element in list(flange):
        flange.remove(element)
    rod = ET.SubElement(root, "link", name="press_rod")
    length, radius, mass = (config["rod"][key] for key in ("length", "radius", "mass"))
    # Fill the existing 75 mm gap from link_6's mounting face to tool0.
    for link, span, width, weight, center, color in (
        (flange, .075, .026, .3, -.0375, "0.22 0.25 0.30 1"),
        (rod, length, radius, mass, length / 2, "0.12 0.65 0.72 1"),
    ):
        for kind in ("visual", "collision"):
            shape = ET.SubElement(link, kind)
            ET.SubElement(shape, "origin", xyz="{} 0 0".format(center), rpy="0 1.5707963267948966 0")
            ET.SubElement(ET.SubElement(shape, "geometry"), "cylinder", radius=str(width), length=str(span))
            if kind == "visual":
                ET.SubElement(ET.SubElement(shape, "material", name=link.get("name")), "color", rgba=color)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz="{} 0 0".format(center), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=str(weight))
        transverse = weight * (3 * width ** 2 + span ** 2) / 12
        ET.SubElement(inertial, "inertia", ixx=str(weight * width ** 2 / 2),
                      iyy=str(transverse), izz=str(transverse), ixy="0", ixz="0", iyz="0")
    for name, parent, child, xyz, rpy in (
        ("press_rod_joint", "tool0", "press_rod", [0, 0, 0], [0, 0, 0]),
        ("rod_tip_joint", "press_rod", "rod_tip", [length, 0, 0], [0, 0, 0]),
        ("camera_joint", "tool0", "camera_optical_frame", config["camera"]["translation"], config["camera"]["rpy"]),
    ):
        if child != "press_rod":
            ET.SubElement(root, "link", name=child)
        joint = ET.SubElement(root, "joint", name=name, type="fixed")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        ET.SubElement(joint, "origin", xyz=" ".join(map(str, xyz)), rpy=" ".join(map(str, rpy)))
    camera = root.find("./link[@name='camera_optical_frame']")
    camera_size = np.array(config["camera"]["body_size"])
    camera_mass = config["camera"]["mass"]
    for kind in ("visual", "collision"):
        shape = ET.SubElement(camera, kind)
        ET.SubElement(shape, "origin", xyz="0 0 {}".format(-camera_size[2] / 2))
        ET.SubElement(ET.SubElement(shape, "geometry"), "box", size=" ".join(map(str, camera_size)))
        if kind == "visual":
            ET.SubElement(ET.SubElement(shape, "material", name="camera"), "color", rgba="0.15 0.18 0.24 1")
    inertial = ET.SubElement(camera, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 {}".format(-camera_size[2] / 2))
    ET.SubElement(inertial, "mass", value=str(camera_mass))
    moments = camera_mass * (np.sum(camera_size ** 2) - camera_size ** 2) / 12
    ET.SubElement(inertial, "inertia", ixx=str(moments[0]), iyy=str(moments[1]), izz=str(moments[2]),
                  ixy="0", ixz="0", iyz="0")
    ET.ElementTree(root).write(str(destination), encoding="utf-8", xml_declaration=True)
    return Path(destination)
