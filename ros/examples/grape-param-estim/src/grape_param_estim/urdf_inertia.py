"""Compose URDF link inertias in a selected base-link frame.

This module uses only the Python standard library and NumPy; it does not
depend on ``urdf``, ``rospy``, the ROS parameter server, or Pinocchio.  URDF
joint transforms are traversed at the supplied joint configuration and every
link inertia in the connected robot is re-expressed in the selected base-link
frame before the composite center-of-mass inertia is recovered.  The selected
frame may be an internal link; it does not have to be the URDF tree root.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

from grape_param_estim.dynamics import (
    inertia_to_parameters,
    validate_physical_parameters,
)


XmlInput = Union[str, bytes, os.PathLike, ET.Element, ET.ElementTree]


def _finite_floats(text: Optional[str], count: int, name: str, default: Sequence[float]) -> np.ndarray:
    if text is None:
        values = np.asarray(default, dtype=float)
    else:
        try:
            values = np.asarray([float(item) for item in text.split()], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("{} must contain {} numeric values".format(name, count)) from exc
    if values.shape != (count,) or not np.all(np.isfinite(values)):
        raise ValueError("{} must contain {} finite values".format(name, count))
    return values


def _required_name(element: ET.Element, attribute: str, context: str) -> str:
    value = str(element.attrib.get(attribute, "")).strip()
    if not value:
        raise ValueError("{} requires a non-empty '{}' attribute".format(context, attribute))
    return value


def _root_element(source: XmlInput) -> ET.Element:
    if isinstance(source, ET.ElementTree):
        root = source.getroot()
    elif isinstance(source, ET.Element):
        root = source
    elif isinstance(source, bytes):
        try:
            root = ET.fromstring(source)
        except ET.ParseError as exc:
            raise ValueError("invalid URDF XML") from exc
    elif isinstance(source, (str, os.PathLike)):
        text = os.fspath(source)
        if isinstance(text, str) and text.lstrip().startswith("<"):
            try:
                root = ET.fromstring(text)
            except ET.ParseError as exc:
                raise ValueError("invalid URDF XML") from exc
        else:
            path = Path(text)
            if not path.is_file():
                raise ValueError("URDF path does not exist: {}".format(path))
            try:
                root = ET.parse(str(path)).getroot()
            except ET.ParseError as exc:
                raise ValueError("invalid URDF XML in {}".format(path)) from exc
    else:
        raise TypeError("urdf_xml must be XML text, a path, Element, or ElementTree")
    if root.tag != "robot":
        raise ValueError("URDF root element must be <robot>")
    return root


def _rpy_rotation(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float
    )
    rotation_y = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float
    )
    rotation_z = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float
    )
    # URDF rpy is fixed-axis roll, pitch, yaw, equivalent to Rz * Ry * Rx.
    return rotation_z @ rotation_y @ rotation_x


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("revolute/continuous joint axis must have non-zero norm")
    unit = axis / norm
    x_value, y_value, z_value = unit
    skew = np.array(
        [
            [0.0, -z_value, y_value],
            [z_value, 0.0, -x_value],
            [-y_value, x_value, 0.0],
        ],
        dtype=float,
    )
    sine = math.sin(float(angle))
    cosine = math.cos(float(angle))
    return np.eye(3, dtype=float) + sine * skew + (1.0 - cosine) * (skew @ skew)


def _origin(element: Optional[ET.Element], context: str) -> Tuple[np.ndarray, np.ndarray]:
    if element is None:
        return np.eye(3, dtype=float), np.zeros(3, dtype=float)
    xyz = _finite_floats(element.attrib.get("xyz"), 3, context + " xyz", (0.0, 0.0, 0.0))
    rpy = _finite_floats(element.attrib.get("rpy"), 3, context + " rpy", (0.0, 0.0, 0.0))
    return _rpy_rotation(rpy), xyz


@dataclass(frozen=True)
class CompositeInertia:
    """Composite inertial properties expressed in ``base_link`` coordinates."""

    base_link: str
    mass: float
    center_of_mass: np.ndarray
    inertia_com: np.ndarray
    reachable_links: Tuple[str, ...]

    def __post_init__(self) -> None:
        base_link = str(self.base_link).strip()
        if not base_link:
            raise ValueError("base_link must be non-empty")
        mass = float(self.mass)
        center = np.asarray(self.center_of_mass, dtype=float)
        inertia = np.asarray(self.inertia_com, dtype=float)
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("composite mass must be finite and positive")
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("center_of_mass must be a finite three-vector")
        if inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
            raise ValueError("inertia_com must be a finite 3x3 matrix")
        inertia = 0.5 * (inertia + inertia.T)
        parameters = inertia_to_parameters(mass, center, inertia)
        validate_physical_parameters(parameters)
        links = tuple(str(name).strip() for name in self.reachable_links)
        if any(not name for name in links) or len(set(links)) != len(links):
            raise ValueError("reachable_links must contain unique non-empty names")

        center_copy = np.array(center, copy=True)
        inertia_copy = np.array(inertia, copy=True)
        center_copy.setflags(write=False)
        inertia_copy.setflags(write=False)
        object.__setattr__(self, "base_link", base_link)
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "center_of_mass", center_copy)
        object.__setattr__(self, "inertia_com", inertia_copy)
        object.__setattr__(self, "reachable_links", links)

    @property
    def cog(self) -> np.ndarray:
        """Alias for ``center_of_mass``."""

        return self.center_of_mass

    @property
    def parameters(self) -> np.ndarray:
        """Return the public ten-element dynamics parameter vector."""

        output = inertia_to_parameters(self.mass, self.center_of_mass, self.inertia_com)
        output.setflags(write=False)
        return output


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin_rotation: np.ndarray
    origin_translation: np.ndarray
    axis: np.ndarray
    mimic_joint: Optional[str]
    mimic_multiplier: float
    mimic_offset: float


def _parse_links(root: ET.Element) -> Dict[str, ET.Element]:
    links: Dict[str, ET.Element] = {}
    for element in root.findall("link"):
        name = _required_name(element, "name", "<link>")
        if name in links:
            raise ValueError("duplicate URDF link '{}'".format(name))
        links[name] = element
    if not links:
        raise ValueError("URDF must contain at least one link")
    return links


def _parse_joints(root: ET.Element, links: Mapping[str, ET.Element]) -> Tuple[_Joint, ...]:
    joints = []
    names = set()
    child_links = set()
    supported = {"fixed", "revolute", "continuous", "prismatic"}
    for element in root.findall("joint"):
        name = _required_name(element, "name", "<joint>")
        if name in names:
            raise ValueError("duplicate URDF joint '{}'".format(name))
        names.add(name)
        kind = _required_name(element, "type", "joint '{}'".format(name)).lower()
        if kind not in supported:
            raise ValueError(
                "joint '{}' has unsupported type '{}'; only fixed, revolute, "
                "continuous, and prismatic joints are supported".format(name, kind)
            )
        parent_element = element.find("parent")
        child_element = element.find("child")
        if parent_element is None or child_element is None:
            raise ValueError("joint '{}' requires parent and child elements".format(name))
        parent = _required_name(parent_element, "link", "joint '{}' parent".format(name))
        child = _required_name(child_element, "link", "joint '{}' child".format(name))
        if parent not in links or child not in links:
            raise ValueError("joint '{}' references an unknown link".format(name))
        if parent == child:
            raise ValueError("joint '{}' cannot connect a link to itself".format(name))
        if child in child_links:
            raise ValueError("link '{}' has more than one parent joint".format(child))
        child_links.add(child)
        rotation, translation = _origin(element.find("origin"), "joint '{}' origin".format(name))
        axis_element = element.find("axis")
        axis = _finite_floats(
            None if axis_element is None else axis_element.attrib.get("xyz"),
            3,
            "joint '{}' axis".format(name),
            (1.0, 0.0, 0.0),
        )
        if kind != "fixed" and float(np.linalg.norm(axis)) <= 1.0e-12:
            raise ValueError("joint '{}' axis must have non-zero norm".format(name))

        mimic = element.find("mimic")
        mimic_joint = None
        mimic_multiplier = 1.0
        mimic_offset = 0.0
        if mimic is not None:
            mimic_joint = _required_name(mimic, "joint", "joint '{}' mimic".format(name))
            try:
                mimic_multiplier = float(mimic.attrib.get("multiplier", "1"))
                mimic_offset = float(mimic.attrib.get("offset", "0"))
            except ValueError as exc:
                raise ValueError("joint '{}' mimic values must be numeric".format(name)) from exc
            if not np.isfinite(mimic_multiplier) or not np.isfinite(mimic_offset):
                raise ValueError("joint '{}' mimic values must be finite".format(name))
        joints.append(
            _Joint(
                name,
                kind,
                parent,
                child,
                rotation,
                translation,
                axis,
                mimic_joint,
                mimic_multiplier,
                mimic_offset,
            )
        )
    known_names = {joint.name for joint in joints}
    for joint in joints:
        if joint.mimic_joint is not None and joint.mimic_joint not in known_names:
            raise ValueError(
                "joint '{}' mimics unknown joint '{}'".format(joint.name, joint.mimic_joint)
            )
    return tuple(joints)


def _joint_values(
    joints: Sequence[_Joint],
    q: Optional[Union[Mapping[str, float], Sequence[float]]],
) -> Dict[str, float]:
    movable = tuple(joint for joint in joints if joint.kind != "fixed")
    if q is None:
        supplied: Dict[str, float] = {}
    elif isinstance(q, Mapping):
        unknown = set(q) - {joint.name for joint in movable}
        if unknown:
            raise ValueError("q contains unknown or fixed joints: {}".format(sorted(unknown)))
        try:
            supplied = {str(name): float(value) for name, value in q.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("q values must be numeric") from exc
    else:
        try:
            values = np.asarray(q, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("q must be a joint-name mapping or numeric sequence") from exc
        if values.shape != (len(movable),):
            raise ValueError(
                "q sequence must contain {} movable-joint values in URDF order".format(
                    len(movable)
                )
            )
        supplied = dict(zip((joint.name for joint in movable), values.tolist()))
    if not all(np.isfinite(value) for value in supplied.values()):
        raise ValueError("q values must be finite")

    resolved: Dict[str, float] = {}
    resolving = set()
    by_name = {joint.name: joint for joint in joints}

    def resolve(joint: _Joint) -> float:
        if joint.kind == "fixed":
            return 0.0
        if joint.name in resolved:
            return resolved[joint.name]
        if joint.name in resolving:
            raise ValueError("mimic joint cycle involving '{}'".format(joint.name))
        resolving.add(joint.name)
        if joint.name in supplied:
            value = supplied[joint.name]
        elif joint.mimic_joint is not None:
            value = (
                joint.mimic_multiplier * resolve(by_name[joint.mimic_joint])
                + joint.mimic_offset
            )
        else:
            value = 0.0
        resolving.remove(joint.name)
        resolved[joint.name] = float(value)
        return float(value)

    for joint in movable:
        resolve(joint)
    return resolved


def _joint_transform(joint: _Joint, value: float) -> Tuple[np.ndarray, np.ndarray]:
    if joint.kind in ("revolute", "continuous"):
        rotation = joint.origin_rotation @ _axis_rotation(joint.axis, value)
        translation = joint.origin_translation
    elif joint.kind == "prismatic":
        unit_axis = joint.axis / np.linalg.norm(joint.axis)
        rotation = joint.origin_rotation
        translation = joint.origin_translation + joint.origin_rotation @ (unit_axis * value)
    else:
        rotation = joint.origin_rotation
        translation = joint.origin_translation
    return rotation, translation


def _link_inertia(
    link: ET.Element,
    rotation_base_link: np.ndarray,
    translation_base_link: np.ndarray,
) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    inertial = link.find("inertial")
    if inertial is None:
        return None
    mass_element = inertial.find("mass")
    inertia_element = inertial.find("inertia")
    if mass_element is None or inertia_element is None:
        raise ValueError(
            "link '{}' inertial requires mass and inertia elements".format(
                link.attrib.get("name", "")
            )
        )
    try:
        mass = float(mass_element.attrib["value"])
    except (KeyError, ValueError) as exc:
        raise ValueError("link inertial mass must be numeric") from exc
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("link inertial mass must be finite and positive")
    entries = {}
    for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
        try:
            entries[name] = float(inertia_element.attrib[name])
        except (KeyError, ValueError) as exc:
            raise ValueError("link inertia requires numeric '{}'".format(name)) from exc
    if not all(np.isfinite(value) for value in entries.values()):
        raise ValueError("link inertia entries must be finite")
    inertia_local = np.array(
        [
            [entries["ixx"], entries["ixy"], entries["ixz"]],
            [entries["ixy"], entries["iyy"], entries["iyz"]],
            [entries["ixz"], entries["iyz"], entries["izz"]],
        ],
        dtype=float,
    )
    eigenvalues = np.linalg.eigvalsh(inertia_local)
    if eigenvalues[0] <= 0.0 or np.sum(eigenvalues[:2]) <= eigenvalues[2]:
        raise ValueError(
            "link '{}' center-of-mass inertia is not physically valid".format(
                link.attrib.get("name", "")
            )
        )
    rotation_link_inertial, translation_link_inertial = _origin(
        inertial.find("origin"),
        "link '{}' inertial origin".format(link.attrib.get("name", "")),
    )
    rotation_base_inertial = rotation_base_link @ rotation_link_inertial
    center_base = (
        translation_base_link
        + rotation_base_link @ translation_link_inertial
    )
    inertia_base = (
        rotation_base_inertial
        @ inertia_local
        @ rotation_base_inertial.T
    )
    return mass, center_base, 0.5 * (inertia_base + inertia_base.T)


def composite_inertia_from_urdf(
    urdf_xml: XmlInput,
    q: Optional[Union[Mapping[str, float], Sequence[float]]] = None,
    base_link: Optional[str] = None,
) -> CompositeInertia:
    """Aggregate all robot link inertias in ``base_link`` coordinates.

    Parameters
    ----------
    urdf_xml:
        URDF XML text, a filesystem path, or an ElementTree object.
    q:
        Optional joint-name mapping.  Missing movable joints default to zero.
        A numeric sequence is also accepted in movable-joint URDF order.
    base_link:
        Frame in which the result is expressed.  It may be any link in the
        robot tree.  When omitted, the unique root link (a link with no parent
        joint) is selected.
    """

    root = _root_element(urdf_xml)
    links = _parse_links(root)
    joints = _parse_joints(root, links)
    child_links = {joint.child for joint in joints}
    root_links = sorted(set(links) - child_links)
    if len(root_links) != 1:
        raise ValueError("URDF must have one root link; found {}".format(root_links))
    root_link = root_links[0]
    if base_link is None:
        resolved_base = root_link
    else:
        resolved_base = str(base_link).strip()
        if resolved_base not in links:
            raise ValueError("unknown base_link '{}'".format(resolved_base))

    values = _joint_values(joints, q)
    children: Dict[str, list] = {name: [] for name in links}
    for joint in joints:
        children[joint.parent].append(joint)

    transforms_root: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        root_link: (np.eye(3, dtype=float), np.zeros(3, dtype=float))
    }
    visiting = set()

    def visit(link_name: str) -> None:
        if link_name in visiting:
            raise ValueError("URDF joint graph contains a cycle at '{}'".format(link_name))
        visiting.add(link_name)
        rotation_root_parent, translation_root_parent = transforms_root[link_name]
        for joint in children[link_name]:
            if joint.child in transforms_root:
                raise ValueError("URDF joint graph revisits link '{}'".format(joint.child))
            rotation_parent_child, translation_parent_child = _joint_transform(
                joint, values.get(joint.name, 0.0)
            )
            transforms_root[joint.child] = (
                rotation_root_parent @ rotation_parent_child,
                translation_root_parent
                + rotation_root_parent @ translation_parent_child,
            )
            visit(joint.child)
        visiting.remove(link_name)

    visit(root_link)
    if len(transforms_root) != len(links):
        missing = sorted(set(links) - set(transforms_root))
        raise ValueError(
            "URDF joint graph is disconnected or cyclic; unreachable links: {}".format(
                missing
            )
        )

    # Compute the whole tree before changing frames so an internal base-link
    # still includes inertial contributions from its ancestors and siblings.
    rotation_root_base, translation_root_base = transforms_root[resolved_base]
    rotation_base_root = rotation_root_base.T
    transforms: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for link_name, (rotation_root_link, translation_root_link) in transforms_root.items():
        transforms[link_name] = (
            rotation_base_root @ rotation_root_link,
            rotation_base_root @ (translation_root_link - translation_root_base),
        )

    total_mass = 0.0
    total_first_moment = np.zeros(3, dtype=float)
    total_inertia_origin = np.zeros((3, 3), dtype=float)
    for link_name, (rotation, translation) in transforms.items():
        item = _link_inertia(links[link_name], rotation, translation)
        if item is None:
            continue
        mass, center, inertia_com = item
        total_mass += mass
        total_first_moment += mass * center
        center_squared = float(center @ center)
        total_inertia_origin += inertia_com + mass * (
            center_squared * np.eye(3, dtype=float) - np.outer(center, center)
        )
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("URDF contains no positive-mass inertial links")
    center_of_mass = total_first_moment / total_mass
    center_squared = float(center_of_mass @ center_of_mass)
    inertia_composite = total_inertia_origin - total_mass * (
        center_squared * np.eye(3, dtype=float)
        - np.outer(center_of_mass, center_of_mass)
    )
    inertia_composite = 0.5 * (inertia_composite + inertia_composite.T)
    return CompositeInertia(
        base_link=resolved_base,
        mass=total_mass,
        center_of_mass=center_of_mass,
        inertia_com=inertia_composite,
        reachable_links=tuple(transforms),
    )


def urdf_inertial_parameters(
    urdf_xml: XmlInput,
    q: Optional[Union[Mapping[str, float], Sequence[float]]] = None,
    base_link: Optional[str] = None,
) -> np.ndarray:
    """Return composite URDF properties in the public ten-parameter order."""

    return composite_inertia_from_urdf(urdf_xml, q=q, base_link=base_link).parameters


# Concise compatibility alias useful to offline configuration loaders.
load_urdf_inertia = composite_inertia_from_urdf


__all__ = [
    "CompositeInertia",
    "composite_inertia_from_urdf",
    "load_urdf_inertia",
    "urdf_inertial_parameters",
]
