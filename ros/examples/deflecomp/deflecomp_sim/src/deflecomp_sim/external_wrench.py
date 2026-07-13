import numpy as np
import pinocchio as pin


def frame_wrench_in_world(
    robot,
    q,
    frame_name,
    force,
    torque,
    reference_frame="world",
):
    """Return the application point and wrench components in the model world frame."""
    if not robot.has_frame(frame_name):
        raise KeyError(frame_name)
    if reference_frame not in ("world", "local"):
        raise ValueError("reference_frame must be 'world' or 'local'")

    configuration = np.asarray(q, dtype=float).reshape(robot.nv)
    frame_id = robot.get_frame_id(frame_name)
    pin.forwardKinematics(robot.model, robot.data, configuration)
    pin.updateFramePlacements(robot.model, robot.data)

    placement_world_frame = robot.data.oMf[frame_id]
    application_point_world = np.asarray(
        placement_world_frame.translation,
        dtype=float,
    ).reshape(3).copy()
    force_world = np.asarray(force, dtype=float).reshape(3).copy()
    torque_world = np.asarray(torque, dtype=float).reshape(3).copy()
    if reference_frame == "local":
        rotation_world_frame = placement_world_frame.rotation
        force_world = rotation_world_frame @ force_world
        torque_world = rotation_world_frame @ torque_world
    return application_point_world, force_world, torque_world


def external_force_arrow_points(application_point_world, force_world, scale):
    """Return an arrow whose direction is the applied external-force direction."""
    marker_scale = float(scale)
    if not np.isfinite(marker_scale) or marker_scale < 0.0:
        raise ValueError("scale must be finite and non-negative")
    start = np.asarray(application_point_world, dtype=float).reshape(3).copy()
    force = np.asarray(force_world, dtype=float).reshape(3)
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(force)):
        raise ValueError("application point and force must be finite")
    return start, start + marker_scale * force


def generalized_external_wrench(
    robot,
    q,
    frame_name,
    force,
    torque,
    reference_frame="world",
):
    if not robot.has_frame(frame_name):
        raise KeyError(frame_name)
    if reference_frame not in ("world", "local"):
        raise ValueError("reference_frame must be 'world' or 'local'")

    configuration = np.asarray(q, dtype=float).reshape(robot.nv)
    frame_id = robot.get_frame_id(frame_name)
    pin.forwardKinematics(robot.model, robot.data, configuration)
    pin.computeJointJacobians(robot.model, robot.data, configuration)
    pin.updateFramePlacements(robot.model, robot.data)
    jacobian = pin.computeFrameJacobian(
        robot.model,
        robot.data,
        configuration,
        frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )

    force_world = np.asarray(force, dtype=float).reshape(3)
    torque_world = np.asarray(torque, dtype=float).reshape(3)
    if reference_frame == "local":
        rotation_world_frame = robot.data.oMf[frame_id].rotation
        force_world = rotation_world_frame @ force_world
        torque_world = rotation_world_frame @ torque_world
    return jacobian[:3, :].T @ force_world + jacobian[3:6, :].T @ torque_world
