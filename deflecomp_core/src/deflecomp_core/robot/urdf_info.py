from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class UrdfJointInfo:
    name: str
    joint_type: str
    parent: str
    child: str
    mimic_joint: Optional[str]
    velocity_limit: Optional[float]

    @property
    def is_fixed(self) -> bool:
        return self.joint_type == "fixed"

    @property
    def is_mimic(self) -> bool:
        return self.mimic_joint is not None

    @property
    def is_controllable(self) -> bool:
        return (not self.is_fixed) and (not self.is_mimic) and (
            self.velocity_limit is None or self.velocity_limit > 0.0
        )


@dataclass(frozen=True)
class UrdfModelInfo:
    root_link: str
    link_names: List[str]
    leaf_links: List[str]
    depth_by_link: Dict[str, int]
    joint_map: Dict[str, UrdfJointInfo]
    movable_joint_names: List[str]
    controllable_joint_names: List[str]
    controllable_child_links: List[str]


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_urdf_model_info(urdf_path: str) -> UrdfModelInfo:
    root = ET.parse(urdf_path).getroot()

    link_names = [link.attrib["name"] for link in root.findall("link") if "name" in link.attrib]
    joint_map: Dict[str, UrdfJointInfo] = {}
    child_to_parent: Dict[str, str] = {}
    outgoing_children: Dict[str, List[str]] = {name: [] for name in link_names}

    for joint_elem in root.findall("joint"):
        name = joint_elem.attrib["name"]
        joint_type = joint_elem.attrib.get("type", "fixed")
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        parent = parent_elem.attrib["link"] if parent_elem is not None else ""
        child = child_elem.attrib["link"] if child_elem is not None else ""
        mimic_elem = joint_elem.find("mimic")
        limit_elem = joint_elem.find("limit")
        mimic_joint = None if mimic_elem is None else mimic_elem.attrib.get("joint")
        velocity_limit = None if limit_elem is None else _parse_float(limit_elem.attrib.get("velocity"))

        joint = UrdfJointInfo(
            name=name,
            joint_type=joint_type,
            parent=parent,
            child=child,
            mimic_joint=mimic_joint,
            velocity_limit=velocity_limit,
        )
        joint_map[name] = joint
        if parent:
            outgoing_children.setdefault(parent, []).append(child)
        if child:
            child_to_parent[child] = parent

    if "base_link" in link_names:
        root_link = "base_link"
    else:
        root_candidates = [name for name in link_names if name not in child_to_parent]
        root_link = root_candidates[0] if root_candidates else (link_names[0] if link_names else "")

    depth_by_link: Dict[str, int] = {root_link: 0}
    queue = [root_link]
    while queue:
        parent = queue.pop(0)
        parent_depth = depth_by_link.get(parent, 0)
        for child in outgoing_children.get(parent, []):
            if child in depth_by_link:
                continue
            depth_by_link[child] = parent_depth + 1
            queue.append(child)

    leaf_links = [
        name for name in link_names if not outgoing_children.get(name)
    ]
    leaf_links.sort(key=lambda name: (depth_by_link.get(name, -1), name))

    movable_joint_names = [joint.name for joint in joint_map.values() if not joint.is_fixed]
    controllable_joint_names = [joint.name for joint in joint_map.values() if joint.is_controllable]
    controllable_child_links = [
        joint_map[name].child for name in controllable_joint_names if joint_map[name].child
    ]

    return UrdfModelInfo(
        root_link=root_link,
        link_names=link_names,
        leaf_links=leaf_links,
        depth_by_link=depth_by_link,
        joint_map=joint_map,
        movable_joint_names=movable_joint_names,
        controllable_joint_names=controllable_joint_names,
        controllable_child_links=controllable_child_links,
    )


def infer_base_link(info: UrdfModelInfo, preferred: Optional[str] = None) -> str:
    if preferred and preferred in info.link_names:
        return preferred
    if "base_link" in info.link_names:
        return "base_link"
    return info.root_link


def infer_tip_link(info: UrdfModelInfo, preferred: Optional[str] = None) -> str:
    if preferred and preferred in info.link_names:
        return preferred
    if info.controllable_child_links:
        return info.controllable_child_links[-1]
    if info.leaf_links:
        return info.leaf_links[-1]
    return info.root_link


def infer_imu_frames(
    info: UrdfModelInfo,
    preferred: Optional[Sequence[str]] = None,
    count: int = 3,
) -> List[str]:
    requested = list(preferred or [])
    resolved: List[str] = []
    for name in requested:
        if name in info.link_names and name not in resolved:
            resolved.append(name)

    if count <= len(resolved):
        return resolved[:count]

    preferred_pool = [name for name in info.controllable_child_links if name not in resolved]
    non_dummy_pool = [name for name in preferred_pool if "dummy" not in name.lower()]
    pool = non_dummy_pool if non_dummy_pool else preferred_pool

    if len(pool) < count - len(resolved):
        for name in reversed(info.leaf_links):
            if name not in resolved and name not in pool:
                pool.append(name)

    need = max(0, count - len(resolved))
    resolved.extend(pool[-need:])
    if not resolved and info.root_link:
        resolved.append(info.root_link)
    return resolved


def infer_joint_types(info: UrdfModelInfo, joint_names: Sequence[str]) -> List[str]:
    joint_types: List[str] = []
    for name in joint_names:
        joint = info.joint_map.get(name)
        joint_types.append(joint.joint_type if joint is not None else "linear")
    return joint_types
