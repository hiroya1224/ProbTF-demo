from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ImuFrameConfig:
    frame_id: str
    model_frame: str
    parent_frame: str
    xyz: np.ndarray
    rpy: np.ndarray
    R_model_imu: np.ndarray
    publish_static_tf: bool = False


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(value)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(float(roll)), np.sin(float(roll))
    cp, sp = np.cos(float(pitch)), np.sin(float(pitch))
    cy, sy = np.cos(float(yaw)), np.sin(float(yaw))
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return Rz @ Ry @ Rx


def quat_xyzw_to_matrix(quat_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n <= 1e-18:
        return np.eye(3, dtype=float)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quat_xyzw_from_matrix(R: np.ndarray) -> np.ndarray:
    Rm = np.asarray(R, dtype=float)
    trace = float(np.trace(Rm))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (Rm[2, 1] - Rm[1, 2]) / s
        qy = (Rm[0, 2] - Rm[2, 0]) / s
        qz = (Rm[1, 0] - Rm[0, 1]) / s
    else:
        idx = int(np.argmax([Rm[0, 0], Rm[1, 1], Rm[2, 2]]))
        if idx == 0:
            s = np.sqrt(max(0.0, 1.0 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2])) * 2.0
            qx = 0.25 * s
            qy = (Rm[0, 1] + Rm[1, 0]) / max(s, 1e-18)
            qz = (Rm[0, 2] + Rm[2, 0]) / max(s, 1e-18)
            qw = (Rm[2, 1] - Rm[1, 2]) / max(s, 1e-18)
        elif idx == 1:
            s = np.sqrt(max(0.0, 1.0 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2])) * 2.0
            qx = (Rm[0, 1] + Rm[1, 0]) / max(s, 1e-18)
            qy = 0.25 * s
            qz = (Rm[1, 2] + Rm[2, 1]) / max(s, 1e-18)
            qw = (Rm[0, 2] - Rm[2, 0]) / max(s, 1e-18)
        else:
            s = np.sqrt(max(0.0, 1.0 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1])) * 2.0
            qx = (Rm[0, 2] + Rm[2, 0]) / max(s, 1e-18)
            qy = (Rm[1, 2] + Rm[2, 1]) / max(s, 1e-18)
            qz = 0.25 * s
            qw = (Rm[1, 0] - Rm[0, 1]) / max(s, 1e-18)
    quat = np.array([qx, qy, qz, qw], dtype=float)
    return quat / (np.linalg.norm(quat) + 1e-18)


def _float_array(value: Any, size: int, default: Sequence[float]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=float)
    if isinstance(value, str):
        items = value.replace(";", ",").split(",")
    else:
        items = list(value)
    values = [float(item) for item in items if str(item).strip()]
    if len(values) != size:
        raise ValueError(f"expected {size} values, got {len(values)}: {value}")
    return np.asarray(values, dtype=float)


def _parse_legacy_item(item: str) -> ImuFrameConfig:
    parts = [part.strip() for part in item.split("@")]
    frame_id = parts[0]
    xyz = _float_array(parts[1], 3, [0.0, 0.0, 0.0]) if len(parts) >= 2 and parts[1] else np.zeros(3)
    rpy = _float_array(parts[2], 3, [0.0, 0.0, 0.0]) if len(parts) >= 3 and parts[2] else np.zeros(3)
    return ImuFrameConfig(
        frame_id=frame_id,
        model_frame=frame_id,
        parent_frame=frame_id,
        xyz=xyz,
        rpy=rpy,
        R_model_imu=rpy_to_matrix(*rpy),
        publish_static_tf=False,
    )


def _entry_name(entry: dict, keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = entry.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _parse_dict_entry(entry: dict) -> ImuFrameConfig:
    static_tf = entry.get("static_tf")
    static_tf_dict = static_tf if isinstance(static_tf, dict) else {}
    frame_id = (
        _entry_name(static_tf_dict, ("child_frame", "frame_id", "frame", "name"))
        or _entry_name(entry, ("frame_id", "frame", "name", "child_frame"))
    )
    if not frame_id:
        raise ValueError(f"IMU frame entry has no frame_id/name: {entry}")

    model_frame = (
        _entry_name(entry, ("model_frame", "parent_frame", "link_frame", "urdf_frame"))
        or _entry_name(static_tf_dict, ("parent_frame", "model_frame", "link_frame", "urdf_frame"))
        or frame_id
    )
    parent_frame = (
        _entry_name(static_tf_dict, ("parent_frame", "model_frame", "link_frame", "urdf_frame"))
        or _entry_name(entry, ("parent_frame", "model_frame", "link_frame", "urdf_frame"))
        or model_frame
    )

    xyz_value = static_tf_dict.get("xyz", static_tf_dict.get("translation", entry.get("xyz", entry.get("translation"))))
    rpy_value = static_tf_dict.get("rpy", static_tf_dict.get("rotation_rpy", entry.get("rpy", entry.get("rotation_rpy"))))
    quat_value = static_tf_dict.get("quat", static_tf_dict.get("quaternion", entry.get("quat", entry.get("quaternion"))))
    xyz = _float_array(xyz_value, 3, [0.0, 0.0, 0.0])
    rpy = _float_array(rpy_value, 3, [0.0, 0.0, 0.0])
    R_model_imu = quat_xyzw_to_matrix(_float_array(quat_value, 4, [0.0, 0.0, 0.0, 1.0])) if quat_value is not None else rpy_to_matrix(*rpy)
    publish_static_tf = _as_bool(entry.get("publish_static_tf", False)) or (
        isinstance(static_tf, dict) or _as_bool(static_tf)
    )

    return ImuFrameConfig(
        frame_id=frame_id,
        model_frame=model_frame,
        parent_frame=parent_frame,
        xyz=xyz,
        rpy=rpy,
        R_model_imu=R_model_imu,
        publish_static_tf=publish_static_tf,
    )


def parse_imu_frame_configs(value: Any) -> List[ImuFrameConfig]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = value.get("imu_frames", value.get("frames", []))
    if isinstance(value, str):
        if "@" in value:
            items = [item.strip() for item in value.split(";") if item.strip()]
        else:
            items = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
        return [_parse_legacy_item(item) for item in items]

    configs: List[ImuFrameConfig] = []
    for item in list(value):
        if isinstance(item, str):
            if item.strip():
                configs.append(_parse_legacy_item(item.strip()))
        elif isinstance(item, dict):
            configs.append(_parse_dict_entry(item))
        else:
            raise ValueError(f"unsupported IMU frame entry: {item!r}")
    return configs


def identity_imu_frame_config(frame_name: str) -> ImuFrameConfig:
    name = str(frame_name).strip()
    return ImuFrameConfig(
        frame_id=name,
        model_frame=name,
        parent_frame=name,
        xyz=np.zeros(3, dtype=float),
        rpy=np.zeros(3, dtype=float),
        R_model_imu=np.eye(3, dtype=float),
        publish_static_tf=False,
    )


def resolve_imu_frame_configs(robot: Any, value: Any, count: int = 3) -> List[ImuFrameConfig]:
    parsed = parse_imu_frame_configs(value)
    valid = [cfg for cfg in parsed if robot.has_frame(cfg.model_frame)]
    if valid:
        return valid

    preferred = [cfg.model_frame for cfg in parsed]
    suggested = robot.suggest_imu_frames(preferred=preferred, count=max(1, int(count)))
    return [identity_imu_frame_config(name) for name in suggested]
