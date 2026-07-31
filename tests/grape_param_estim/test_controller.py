import hashlib
import json
from pathlib import Path
import subprocess
import unittest

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.controller import (
    ControllerConfig,
    GrapeController,
    PIDConfig,
    acceleration_allocation_matrix,
)
from grape_param_estim.dynamics import actuator_wrench
from grape_param_estim.geometry import (
    euler_xyz_to_matrix,
    matrix_to_quaternion,
)
from grape_param_estim.system import (
    ActuatorState,
    ControllerState,
    GrapeGeometry,
    ReferenceState,
    RigidBodyState,
    VehicleParameters,
)


SOURCE_COMMIT = "9ae2159277489ef74892486291655deac2dc38dc"
EXPECTED = (
    {
        "thrust": (
            0.18079984188079834,
            0.6427966356277466,
            0.3262578845024109,
            0.601447343826294,
        ),
        "gimbal": (
            -1.2038956794078783,
            -1.3398663399254471,
            0.3208444757565898,
            1.196710598101624,
        ),
        "virtual": (
            0.16876645635419585,
            0.06485725615116969,
            0.6257329131961373,
            0.14712516834233938,
            -0.10289131838522045,
            0.30960877272453435,
            -0.5598524949860004,
            0.21978186796315444,
        ),
        "integral": (0.0016, 0.0016, 0.0016, 0.0, 0.0, 0.001),
    },
    {
        "thrust": (
            0.15085209906101227,
            0.6248741745948792,
            0.30432289838790894,
            0.5878070592880249,
        ),
        "gimbal": (
            -1.0578345075259028,
            -1.3454517431416697,
            0.21800326098613218,
            1.1798359661445885,
        ),
        "virtual": (
            0.13143664434840663,
            0.07403217619845835,
            0.6090755871553464,
            0.139623289624003,
            -0.06581913506473026,
            0.29711996996519574,
            -0.5434531782798048,
            0.22399943755661725,
        ),
        "integral": (0.0032, 0.0032, 0.0032, 0.0006, 0.0004, 0.002),
    },
)


SEQUENCE = (
    (
        (0.10, -0.20, 1.00),
        (0.04, -0.02, 0.01),
        (0.05, -0.03, 0.10),
        (0.02, -0.01, 0.03),
        (0.18, -0.12, 1.08),
        (0.02, 0.01, -0.03),
        (0.12, -0.08, 0.04),
        (0.08, -0.01, 0.15),
        (0.02, -0.03, 0.04),
        (0.01, -0.02, 0.015),
    ),
    (
        (0.11, -0.195, 1.005),
        (0.045, -0.018, 0.012),
        (0.052, -0.028, 0.104),
        (0.021, -0.012, 0.028),
        (0.19, -0.115, 1.085),
        (0.018, 0.012, -0.028),
        (0.10, -0.07, 0.035),
        (0.082, -0.008, 0.154),
        (0.018, -0.028, 0.038),
        (0.009, -0.018, 0.013),
    ),
)


def _geometry_payload(parameters, geometry):
    rotations = [
        euler_xyz_to_matrix((0.0, 0.0, yaw)).tolist()
        for yaw in geometry.arm_yaws
    ]
    return {
        "mass": parameters.mass,
        "inertia": parameters.inertia.tolist(),
        "moment_force_rate": geometry.moment_force_rate,
        "rotor_origins_from_cog": geometry.rotor_origins.tolist(),
        "rotor_directions": [
            int(value) for value in geometry.rotor_directions
        ],
        "thrust_coordinate_rotations": rotations,
    }


def _snapshot(configuration, parameters, geometry):
    pid = []
    for name, item in zip(
        ("x", "y", "z", "roll", "pitch", "yaw"),
        configuration.pid,
    ):
        pid.append(
            {
                "name": name,
                "p_gain": item.p_gain,
                "i_gain": item.i_gain,
                "d_gain": item.d_gain,
                "limit_sum": item.limit_sum,
                "limit_p": item.limit_p,
                "limit_i": item.limit_i,
                "limit_d": item.limit_d,
                "limit_error_p": item.limit_error_p,
                "limit_error_i": item.limit_error_i,
                "limit_error_d": item.limit_error_d,
            }
        )
    return {
        "snapshot_id": "phase1-golden",
        "pid": pid,
        "pose_config": {
            "need_yaw_d_control": True,
            "start_roll_pitch_integration_height": 0.01,
            "force_landing_descending_rate": -0.5,
        },
        "allocation_options": {
            "gimbal_dof": 1,
            "gimbal_calc_in_fc": False,
            "hovering_approximate": False,
            "underactuate": False,
            "gravity": 9.797,
        },
        "geometry": _geometry_payload(parameters, geometry),
    }


def _run_python_and_ticks():
    configuration = ControllerConfig.grape()
    parameters = VehicleParameters.nominal()
    geometry = GrapeGeometry.grape()
    controller = GrapeController(configuration, parameters, geometry)
    controller_state = ControllerState(np.zeros(6), False)
    outputs = []
    ticks = []
    for index, values in enumerate(SEQUENCE):
        (
            position,
            velocity,
            rpy,
            omega,
            target_position,
            target_velocity,
            target_acceleration,
            target_rpy,
            target_omega,
            target_alpha,
        ) = values
        rotation = euler_xyz_to_matrix(rpy)
        target_rotation = euler_xyz_to_matrix(target_rpy)
        target_omega_current = rotation.T @ target_rotation @ np.asarray(
            target_omega
        )
        state = RigidBodyState(
            position,
            matrix_to_quaternion(rotation),
            velocity,
            omega,
        )
        reference = ReferenceState(
            target_position,
            target_velocity,
            target_acceleration,
            target_rpy,
            target_omega,
            target_alpha,
        )
        command, controller_state = controller.step(
            state, reference, controller_state, 0.02
        )
        outputs.append((command, controller_state))
        ticks.append(
            {
                "stamp": 0.02 * index,
                "dt": 0.02,
                "position": list(position),
                "velocity": list(velocity),
                "rpy": list(rpy),
                "angular_velocity": list(omega),
                "target_position": list(target_position),
                "target_velocity": list(target_velocity),
                "target_acceleration": list(target_acceleration),
                "target_rpy": list(target_rpy),
                "target_angular_velocity": target_omega_current.tolist(),
                "target_angular_acceleration": list(target_alpha),
                "xy_control_mode": 0,
                "control_mode": [0] * 6,
                "integration_enabled": [
                    True,
                    True,
                    True,
                    bool(index),
                    bool(index),
                    True,
                ],
                "flight_state": 5,
                "force_landing": False,
                "reset": False,
                "initial_height": 0.0,
                "orientation": rotation.tolist(),
                "joint_positions": [],
            }
        )
    return configuration, parameters, geometry, outputs, ticks


class ControllerTests(unittest.TestCase):
    def test_articulated_q0_snapshot_matches_audited_controller_values(self):
        parameters, geometry = GrapeArticulatedModel().at(np.zeros(4))
        expected_parameters = VehicleParameters.nominal()
        expected_geometry = GrapeGeometry.grape()
        self.assertAlmostEqual(parameters.mass, expected_parameters.mass)
        np.testing.assert_allclose(
            parameters.inertia, expected_parameters.inertia, atol=1.0e-15
        )
        np.testing.assert_allclose(parameters.cog_offset, 0.0)
        np.testing.assert_allclose(
            geometry.rotor_origins,
            expected_geometry.rotor_origins,
            atol=1.0e-15,
        )

    def test_python_port_matches_frozen_cpp_golden(self):
        _, _, _, outputs, _ = _run_python_and_ticks()
        for index, ((command, state), expected) in enumerate(
            zip(outputs, EXPECTED)
        ):
            with self.subTest(tick=index):
                np.testing.assert_allclose(
                    command.thrust, expected["thrust"], atol=3.0e-8
                )
                np.testing.assert_allclose(
                    command.gimbal_angle, expected["gimbal"], atol=1.0e-12
                )
                np.testing.assert_allclose(
                    command.virtual_force,
                    expected["virtual"],
                    atol=1.0e-12,
                )
                np.testing.assert_allclose(
                    state.integral_error,
                    expected["integral"],
                    atol=1.0e-15,
                )

    def test_live_exact_cpp_oracle_matches_same_ticks(self):
        workspace = Path(__file__).resolve().parents[4]
        executable = (
            workspace
            / "devel/.private/gimbalrotor/lib/gimbalrotor"
            / "gimbalrotor_controller_replay"
        )
        if not executable.is_file():
            self.skipTest("exact C++ controller oracle is not built")
        configuration, parameters, geometry, outputs, ticks = (
            _run_python_and_ticks()
        )
        digest = hashlib.sha256(executable.read_bytes()).hexdigest()
        request = {
            "protocol": "grape.exact-controller-oracle/v1",
            "operation": "replay",
            "payload": {
                "snapshot": _snapshot(
                    configuration, parameters, geometry
                ),
                "jobs": [
                    {"reset_before_first_tick": True, "ticks": ticks}
                ],
            },
        }
        process = subprocess.run(
            [str(executable), "--artifact-sha256", digest],
            input=json.dumps(request) + "\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5.0,
            check=True,
        )
        self.assertEqual(process.stderr, "")
        reply = json.loads(process.stdout)
        self.assertTrue(reply["ok"])
        self.assertEqual(reply["identity"]["source_commit"], SOURCE_COMMIT)
        for index, (command, _state) in enumerate(outputs):
            np.testing.assert_allclose(
                command.thrust,
                reply["continuous"]["four_axis_command"][index][3:],
                atol=3.0e-8,
            )
            np.testing.assert_allclose(
                command.gimbal_angle,
                reply["continuous"]["gimbal_command"][index],
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                command.virtual_force,
                reply["continuous"]["vectoring_force"][index],
                atol=1.0e-12,
            )

    def test_live_cpp_oracle_matches_nonzero_articulated_geometry(self):
        workspace = Path(__file__).resolve().parents[4]
        executable = (
            workspace
            / "devel/.private/gimbalrotor/lib/gimbalrotor"
            / "gimbalrotor_controller_replay"
        )
        if not executable.is_file():
            self.skipTest("exact C++ controller oracle is not built")
        configuration, parameters, geometry, _outputs, ticks = (
            _run_python_and_ticks()
        )
        values = SEQUENCE[0]
        state = RigidBodyState(
            values[0],
            matrix_to_quaternion(euler_xyz_to_matrix(values[2])),
            values[1],
            values[3],
        )
        reference = ReferenceState(
            values[4], values[5], values[6], values[7], values[8], values[9]
        )
        angles = np.asarray((0.83, -0.71, 0.46, -0.92))
        articulated = GrapeArticulatedModel()
        dynamic_parameters, dynamic_geometry = articulated.at(angles)
        controller = GrapeController(
            configuration,
            parameters,
            geometry,
            articulated_model=articulated,
        )
        command, _state = controller.step(
            state,
            reference,
            ControllerState(np.zeros(6), False),
            0.02,
            angles,
        )
        tick = dict(ticks[0])
        tick["geometry"] = _geometry_payload(
            dynamic_parameters, dynamic_geometry
        )
        digest = hashlib.sha256(executable.read_bytes()).hexdigest()
        request = {
            "protocol": "grape.exact-controller-oracle/v1",
            "operation": "replay",
            "payload": {
                "snapshot": _snapshot(configuration, parameters, geometry),
                "jobs": [
                    {"reset_before_first_tick": True, "ticks": [tick]}
                ],
            },
        }
        process = subprocess.run(
            [str(executable), "--artifact-sha256", digest],
            input=json.dumps(request) + "\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5.0,
            check=True,
        )
        reply = json.loads(process.stdout)
        self.assertTrue(reply["ok"])
        np.testing.assert_allclose(
            command.thrust,
            reply["continuous"]["four_axis_command"][0][3:],
            atol=3.0e-8,
        )
        np.testing.assert_allclose(
            command.gimbal_angle,
            reply["continuous"]["gimbal_command"][0],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            command.virtual_force,
            reply["continuous"]["vectoring_force"][0],
            atol=1.0e-12,
        )

    def test_pid_state_persists_and_every_limit_clamps(self):
        limited = PIDConfig(
            10.0,
            10.0,
            10.0,
            limit_sum=0.7,
            limit_p=0.3,
            limit_i=0.2,
            limit_d=0.4,
            limit_error_p=0.5,
            limit_error_i=0.02,
            limit_error_d=0.1,
        )
        configuration = ControllerConfig(pid=(limited,) * 6)
        parameters = VehicleParameters.nominal()
        controller = GrapeController(
            configuration, parameters, GrapeGeometry.grape()
        )
        state = RigidBodyState(
            np.zeros(3),
            (0.0, 0.0, 0.0, 1.0),
            np.zeros(3),
            np.zeros(3),
        )
        reference = ReferenceState(
            np.full(3, 100.0),
            np.full(3, 100.0),
            np.zeros(3),
            np.full(3, 100.0),
            np.full(3, 100.0),
            np.zeros(3),
        )
        command, first = controller.step(
            state, reference, ControllerState(np.zeros(6), True), 0.1
        )
        _, second = controller.step(state, reference, first, 0.1)
        self.assertTrue(np.all(np.abs(command.desired_acceleration) <= 0.7))
        self.assertTrue(np.all(np.abs(first.integral_error) <= 0.02))
        np.testing.assert_allclose(first.integral_error, second.integral_error)
        self.assertGreaterEqual(first.integral_error[2], 0.0)

    def test_allocation_spans_and_reconstructs_all_six_axes(self):
        parameters = VehicleParameters.nominal()
        geometry = GrapeGeometry.grape()
        allocation = acceleration_allocation_matrix(parameters, geometry)
        self.assertEqual(np.linalg.matrix_rank(allocation), 6)
        desired = np.asarray((0.3, -0.2, 9.8, 0.1, -0.15, 0.2))
        virtual = np.linalg.pinv(allocation) @ desired
        np.testing.assert_allclose(
            allocation @ virtual, desired, atol=1.0e-12
        )
        thrust = np.empty(4)
        gimbal = np.empty(4)
        for rotor in range(4):
            lateral, axial = virtual[2 * rotor:2 * rotor + 2]
            thrust[rotor] = np.hypot(lateral, axial)
            gimbal[rotor] = np.arctan2(-lateral, axial)
        # The deployed controller linearises allocation at the measured
        # gimbal geometry.  The physical plant evaluates the finite-angle
        # wrench, so only a zero-angle axial check is exact in both models.
        hover_thrust = np.full(4, parameters.mass * 9.8 / 4.0)
        wrench = actuator_wrench(
            ActuatorState(hover_thrust, np.zeros(4)), parameters, geometry
        )
        np.testing.assert_allclose(wrench[:2], 0.0, atol=2.0e-4)
        self.assertAlmostEqual(wrench[2], parameters.mass * 9.8)


if __name__ == "__main__":
    unittest.main()
