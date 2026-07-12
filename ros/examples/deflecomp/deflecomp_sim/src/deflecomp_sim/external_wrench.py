import numpy as np
import pinocchio as pin


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

