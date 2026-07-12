"""Load robot-specific sensor mounting metadata without ROS dependencies."""

from collections.abc import Mapping, Sequence
import math
import numbers
from pathlib import Path

import numpy as np
import yaml

from probtf.geometry import rpy_to_quat
from probtf.models import SensorMount


_TOP_LEVEL_KEYS = {"sensors", "imu_frames", "static_transforms"}
_ENTRY_KEYS = {
    "source_id",
    "sensor_id",
    "frame_id",
    "frame",
    "name",
    "child_frame",
    "parent_frame_id",
    "parent_frame",
    "model_frame",
    "link_frame",
    "urdf_frame",
    "position_xyz",
    "xyz",
    "translation",
    "orientation_wxyz",
    "quaternion_wxyz",
    "orientation_xyzw",
    "quat",
    "quaternion",
    "rpy",
    "rotation_rpy",
    "transform",
    "static_tf",
    "publish_static_tf",
}
_TRANSFORM_KEYS = {
    "frame_id",
    "frame",
    "name",
    "child_frame",
    "parent_frame_id",
    "parent_frame",
    "model_frame",
    "link_frame",
    "urdf_frame",
    "position_xyz",
    "xyz",
    "translation",
    "orientation_wxyz",
    "quaternion_wxyz",
    "orientation_xyzw",
    "quat",
    "quaternion",
    "rpy",
    "rotation_rpy",
}


def _require_mapping(value, context):
    if not isinstance(value, Mapping):
        raise ValueError("{} must be a mapping".format(context))
    return value


def _check_keys(mapping, allowed, context):
    unknown = sorted(str(key) for key in set(mapping) - allowed)
    if unknown:
        raise ValueError("{} has unsupported keys: {}".format(context, ", ".join(unknown)))


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(name))
    return value.strip()


def _present_values(layers, aliases):
    values = []
    for layer_name, layer in layers:
        for alias in aliases:
            if alias in layer and layer[alias] is not None:
                values.append(("{}.{}".format(layer_name, alias), layer[alias]))
    return values


def _single_value(layers, aliases, field_name, default=None, allow_first_layer_override=False):
    values = _present_values(layers, aliases)
    if len(values) > 1:
        if allow_first_layer_override:
            first_layer_name = layers[0][0] + "."
            first_layer_values = [(key, value) for key, value in values if key.startswith(first_layer_name)]
            if len(first_layer_values) == 1:
                return first_layer_values[0][1]
        keys = ", ".join(key for key, _ in values)
        raise ValueError("{} is specified more than once ({})".format(field_name, keys))
    return values[0][1] if values else default


def _numeric_vector(value, size, name):
    if isinstance(value, str):
        items = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    elif isinstance(value, Sequence) or isinstance(value, np.ndarray):
        items = list(value)
    else:
        raise ValueError("{} must be a sequence of {} numbers".format(name, size))
    if len(items) != size:
        raise ValueError("{} must contain exactly {} numbers".format(name, size))
    if any(isinstance(item, bool) or not isinstance(item, numbers.Real) for item in items):
        try:
            converted = [float(item) for item in items]
        except (TypeError, ValueError) as error:
            raise ValueError("{} must contain only numbers".format(name)) from error
        if any(isinstance(item, bool) for item in items):
            raise ValueError("{} must contain only numbers".format(name))
    else:
        converted = [float(item) for item in items]
    if not all(math.isfinite(item) for item in converted):
        raise ValueError("{} must contain only finite numbers".format(name))
    return np.asarray(converted, dtype=float)


def _orientation(layers, context, allow_first_layer_override=False):
    rotation_specs = []
    for kind, aliases in (
        ("wxyz", ("orientation_wxyz", "quaternion_wxyz")),
        ("xyzw", ("orientation_xyzw", "quat", "quaternion")),
        ("rpy", ("rpy", "rotation_rpy")),
    ):
        for key, value in _present_values(layers, aliases):
            rotation_specs.append((kind, key, value))
    if len(rotation_specs) > 1:
        if allow_first_layer_override:
            first_layer_name = layers[0][0] + "."
            first_layer_specs = [spec for spec in rotation_specs if spec[1].startswith(first_layer_name)]
            if len(first_layer_specs) == 1:
                rotation_specs = first_layer_specs
        if len(rotation_specs) > 1:
            raise ValueError(
                "{}.orientation is ambiguous; use exactly one of {}".format(
                    context,
                    ", ".join(key for _, key, _ in rotation_specs),
                )
            )
    if not rotation_specs:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    kind, key, value = rotation_specs[0]
    if kind == "rpy":
        return rpy_to_quat(*_numeric_vector(value, 3, key))
    quaternion = _numeric_vector(value, 4, key)
    if np.linalg.norm(quaternion) <= 1e-12:
        raise ValueError("{} must not be a zero quaternion".format(key))
    if kind == "xyzw":
        quaternion = quaternion[[3, 0, 1, 2]]
    return quaternion


def _nested_transform(entry, context):
    transform = entry.get("transform")
    static_tf = entry.get("static_tf")
    if transform is not None and static_tf is not None:
        raise ValueError("{} must not define both transform and static_tf".format(context))
    nested = transform if transform is not None else static_tf
    if nested is None or isinstance(nested, bool):
        return {}, False
    nested = _require_mapping(nested, "{}.transform".format(context))
    _check_keys(nested, _TRANSFORM_KEYS, "{}.transform".format(context))
    return nested, static_tf is not None


def _parse_entry(entry, context, source_hint=None, require_source=False):
    entry = _require_mapping(entry, context)
    _check_keys(entry, _ENTRY_KEYS, context)
    nested, legacy_override = _nested_transform(entry, context)
    layers = (("{}.transform".format(context), nested), (context, entry))

    explicit_source = _single_value(((context, entry),), ("source_id", "sensor_id"), "{}.source_id".format(context))
    if source_hint is not None and explicit_source is not None:
        if _identifier(explicit_source, "{}.source_id".format(context)) != source_hint:
            raise ValueError("{}.source_id conflicts with its sensors mapping key".format(context))
    source_id = explicit_source if explicit_source is not None else source_hint

    frame_id = _single_value(
        layers,
        ("frame_id", "frame", "name", "child_frame"),
        "{}.frame_id".format(context),
        allow_first_layer_override=legacy_override,
    )
    if frame_id is None:
        raise ValueError("{}.frame_id is required".format(context))
    frame_id = _identifier(frame_id, "{}.frame_id".format(context))
    if source_id is None:
        if require_source:
            raise ValueError("{}.source_id is required".format(context))
        source_id = frame_id
    source_id = _identifier(source_id, "{}.source_id".format(context))

    parent_frame = _single_value(
        layers,
        ("parent_frame_id", "parent_frame"),
        "{}.parent_frame_id".format(context),
        allow_first_layer_override=legacy_override,
    )
    if parent_frame is None:
        parent_frame = _single_value(
            layers,
            ("model_frame", "link_frame", "urdf_frame"),
            "{}.model_frame".format(context),
            default=frame_id,
            allow_first_layer_override=legacy_override,
        )
    parent_frame = _identifier(parent_frame, "{}.parent_frame_id".format(context))

    position = _single_value(
        layers,
        ("position_xyz", "xyz", "translation"),
        "{}.position_xyz".format(context),
        default=(0.0, 0.0, 0.0),
        allow_first_layer_override=legacy_override,
    )
    position = _numeric_vector(position, 3, "{}.position_xyz".format(context))
    orientation = _orientation(layers, context, allow_first_layer_override=legacy_override)
    return SensorMount(
        source_id=source_id,
        frame_id=frame_id,
        parent_frame_id=parent_frame,
        position_xyz=position,
        orientation_wxyz=orientation,
    )


def _generic_sensor_entries(value):
    if isinstance(value, Mapping):
        for source_id, entry in value.items():
            source_id = _identifier(source_id, "sensors mapping key")
            yield entry, "sensors.{}".format(source_id), source_id
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, entry in enumerate(value):
            yield entry, "sensors[{}]".format(index), None
        return
    raise ValueError("sensors must be a list or mapping")


def _legacy_entries(value, section):
    if value is None:
        return
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("{} must be a list".format(section))
    for index, entry in enumerate(value):
        if isinstance(entry, str):
            parts = [part.strip() for part in entry.split("@")]
            if not parts[0] or len(parts) > 3:
                raise ValueError("{}[{}] is not a valid compact frame entry".format(section, index))
            entry = {"frame_id": parts[0]}
            if len(parts) > 1 and parts[1]:
                entry["xyz"] = parts[1]
            if len(parts) > 2 and parts[2]:
                entry["rpy"] = parts[2]
        yield entry, "{}[{}]".format(section, index), None


def parse_sensor_mounts(config):
    """Convert a loaded YAML mapping into validated ``SensorMount`` entries."""
    config = _require_mapping(config, "sensor configuration")
    if not set(config).issubset(_TOP_LEVEL_KEYS):
        unknown = sorted(str(key) for key in set(config) - _TOP_LEVEL_KEYS)
        raise ValueError("sensor configuration has unsupported keys: {}".format(", ".join(unknown)))

    mounts = []
    if "sensors" in config:
        for entry, context, source_hint in _generic_sensor_entries(config["sensors"]):
            mounts.append(_parse_entry(entry, context, source_hint=source_hint, require_source=True))
    for section in ("imu_frames", "static_transforms"):
        for entry, context, _ in _legacy_entries(config.get(section), section):
            mounts.append(_parse_entry(entry, context))

    source_ids = set()
    frame_ids = set()
    for mount in mounts:
        if mount.source_id in source_ids:
            raise ValueError("duplicate sensor source_id: {}".format(mount.source_id))
        if mount.frame_id in frame_ids:
            raise ValueError("duplicate sensor frame_id: {}".format(mount.frame_id))
        source_ids.add(mount.source_id)
        frame_ids.add(mount.frame_id)
    return tuple(mounts)


def load_sensor_mounts(source):
    """Load sensor mounts from a YAML path or an already-loaded mapping."""
    if isinstance(source, Mapping):
        config = source
    elif isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.is_file():
            raise ValueError("sensor configuration path is not a file: {}".format(path))
        with path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if config is None:
            config = {}
    else:
        raise TypeError("source must be a path or mapping")
    return parse_sensor_mounts(config)
