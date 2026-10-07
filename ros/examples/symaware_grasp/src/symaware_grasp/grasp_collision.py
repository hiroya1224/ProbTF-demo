"""Conservative separation of URDF collision bodies from the bottle.

Robot cylinders are enclosed in oriented boxes. Bottle cylinders retain their
exact support function. A positive separating-axis gap certifies separation;
an inconclusive or negative gap is rejected. No contact dynamics are assumed.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from symaware_grasp.grasp_trial import GRIPPER_JOINT_NAMES, gripper_joint_position


def origin_matrix(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return transform


def box_cylinder_gap(center, rotation, half_size, cylinder_center, cylinder_axis, radius, half_length):
    """Return a certified lower bound on clearance, in metres."""
    axis = np.array(cylinder_axis, dtype=float)
    axis /= np.linalg.norm(axis)
    helper = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(axis, helper); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
    radial = np.cos(angles)[:, None]*u + np.sin(angles)[:, None]*v
    axes = np.vstack((rotation.T, axis, np.cross(rotation.T, axis), radial))
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1e-9] / norms[norms > 1e-9, None]
    axial = np.clip(axes @ axis, -1, 1)
    cylinder_extent = radius*np.sqrt(np.maximum(0, 1-axial**2)) + half_length*np.abs(axial)
    box_extent = np.abs(axes @ rotation) @ half_size
    separation = np.abs(axes @ (center-cylinder_center)) - box_extent - cylinder_extent
    return float(np.max(separation))


class BottleCollisionModel:
    def __init__(self, urdf):
        text = Path(urdf).read_text() if isinstance(urdf, Path) else urdf
        root = ET.fromstring(text)
        self.joints = []
        for joint in root.findall("joint"):
            axis = joint.find("axis")
            self.joints.append((joint.get("name"), joint.get("type"),
                joint.find("parent").get("link"), joint.find("child").get("link"),
                origin_matrix(joint.find("origin")),
                np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else np.zeros(3)))
        self.bodies = []
        for link in root.findall("link"):
            for index, collision in enumerate(link.findall("collision")):
                box, cylinder = collision.find("geometry/box"), collision.find("geometry/cylinder")
                if box is not None:
                    half_size = np.fromstring(box.get("size"), sep=" ")/2
                elif cylinder is not None:
                    radius = float(cylinder.get("radius"))
                    half_size = np.array([radius, radius, float(cylinder.get("length"))/2])
                else:
                    raise ValueError("trial collision checking requires boxes and cylinders")
                self.bodies.append((link.get("name"), collision.get("name", str(index)),
                                    origin_matrix(collision.find("origin")), half_size))

    def transforms(self, theta, opening):
        values = dict(zip(["joint_{}".format(i+1) for i in range(6)], theta))
        values.update(dict.fromkeys(GRIPPER_JOINT_NAMES, gripper_joint_position(opening)))
        transforms, pending = {"base_link": np.eye(4)}, list(self.joints)
        while pending:
            progress = False
            for joint in pending[:]:
                name, kind, parent, child, origin, axis = joint
                if parent not in transforms:
                    continue
                motion = np.eye(4)
                if kind == "revolute":
                    motion[:3, :3] = Rotation.from_rotvec(axis*values[name]).as_matrix()
                elif kind == "prismatic":
                    motion[:3, 3] = axis*values[name]
                transforms[child] = transforms[parent] @ origin @ motion
                pending = [item for item in pending if item is not joint]
                progress = True
            if not progress:
                raise ValueError("URDF joint tree cannot be resolved")
        return transforms

    def clearance(self, theta, opening, position, axis=(0, 0, 1)):
        transforms = self.transforms(theta, opening)
        position, axis = np.asarray(position), np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        smallest, pair = float("inf"), None
        # Body, neck, cap match the RViz geometry.
        for link, name, local, half_size in self.bodies:
            body = transforms[link] @ local
            for bottle_part, height, radius, half_length in (
                ("body", 0, .05, .09), ("neck", .105, .025, .015), ("cap", .128, .029, .008)):
                gap = box_cylinder_gap(body[:3, 3], body[:3, :3], half_size,
                    position+height*axis, axis, radius, half_length)
                if gap < smallest:
                    smallest, pair = gap, "{}:{} / bottle_{}".format(link, name, bottle_part)
        return smallest, pair

    def validate_path(self, waypoints, opening, position, axis=(0, 0, 1), minimum=.0005):
        waypoints = np.asarray(waypoints)
        smallest, closest = float("inf"), None
        # Include joint interpolation between each pair, not only IK knots.
        for start, end in zip(waypoints[:-1], waypoints[1:]):
            count = max(2, int(np.ceil(np.max(np.abs(end-start))/.01))+1)
            for fraction in np.linspace(0, 1, count):
                gap, pair = self.clearance(start+fraction*(end-start), opening, position, axis)
                if gap < smallest:
                    smallest, closest = gap, pair
                if gap < minimum:
                    raise RuntimeError("path has insufficient bottle clearance ({:.2f} mm): {}".format(1000*gap, pair))
        return {"clearance_lower_bound_mm": 1000*smallest, "closest_pair": closest}
