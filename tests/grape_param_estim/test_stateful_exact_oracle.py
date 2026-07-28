import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

import numpy as np

from grape_param_estim.alternative_backends import (
    EXACT_ORACLE_PROTOCOL,
    PC_EXACT_ORACLE_CAPABILITIES,
    ExactOracleIdentity,
    ExactOracleProtocolError,
    ExactOracleReplayOutput,
    PersistentSubprocessExactControllerOracle,
    SubprocessExactControllerOracle,
    _output_from_reply,
)
from grape_param_estim.controller import (
    ControllerCoreInput,
    ControllerCoreState,
    ControllerSnapshot,
    ControllerStaticOptions,
)
from grape_param_estim.controller.contracts import deep_thaw
from grape_param_estim.controller.external_oracle import (
    StatefulExactOracleControllerBackend,
    build_exact_replay_payload,
)


def _identity():
    return ExactOracleIdentity(
        protocol=EXACT_ORACLE_PROTOCOL,
        backend_id="jsk_aerial_robot/gimbalrotor_controller_replay",
        implementation_language="c++",
        source_commit="2786cc3e",
        artifact_sha256="a" * 64,
        capabilities=PC_EXACT_ORACLE_CAPABILITIES,
        fidelity="pc_exact",
    )


def _snapshot(identity, *, underactuate=False, integral_gain=0.1):
    identity_matrix = np.eye(3).tolist()
    return ControllerSnapshot(
        backend_id=identity.backend_id,
        source_commit=identity.source_commit,
        artifact_sha256=identity.artifact_sha256,
        nominal_model_sha256="b" * 64,
        parameter_dump_sha256="c" * 64,
        controller_rate_hz=10.0,
        gains={
            "p_gain": [1.0] * 6,
            "i_gain": [float(integral_gain)] * 6,
            "d_gain": [0.0 if underactuate else 0.01] * 6,
        },
        limits={
            "limit_sum": [100.0] * 6,
            "limit_p": [100.0] * 6,
            "limit_i": [100.0] * 6,
            "limit_d": [100.0] * 6,
            "limit_err_p": [100.0] * 6,
            "limit_err_i": [100.0] * 6,
            "limit_err_d": [100.0] * 6,
        },
        static_options=ControllerStaticOptions(
            gimbal_dof=1,
            gimbal_calc_in_fc=False,
            hovering_approximate=False,
            underactuate=underactuate,
            need_yaw_d_control=True,
            integration_start_height=0.01,
            force_landing_descending_rate=-0.1,
            estimate_mode=0,
        ),
        nominal_mass=2.0,
        nominal_cog=[0.0, 0.0, 0.0],
        nominal_inertia=np.diag([0.3, 0.4, 0.5]),
        nominal_geometry={
            "gravity": 9.797,
            "moment_force_rate": 0.01,
            "rotor_origins_from_cog": [
                [0.3, 0.2, 0.0],
                [-0.3, 0.2, 0.0],
                [-0.3, -0.2, 0.0],
                [0.3, -0.2, 0.0],
            ],
            "rotor_directions": [1, -1, 1, -1],
            "thrust_coordinate_rotations": [
                identity_matrix,
                identity_matrix,
                identity_matrix,
                identity_matrix,
            ],
        },
    )


def _core_input(stamp):
    roll_pitch_enabled = float(stamp) > 0.0
    return ControllerCoreInput(
        stamp=stamp,
        dt=0.1,
        position=[0.0, 0.0, 1.0],
        velocity=[0.0, 0.0, 0.0],
        orientation=np.eye(3),
        angular_velocity=[0.0, 0.0, 0.0],
        target_position=[0.0, 0.0, 1.0],
        target_velocity=[0.0, 0.0, 0.0],
        target_acceleration=[0.0, 0.0, 9.797],
        target_orientation=np.eye(3),
        target_angular_velocity=[0.0, 0.0, 0.0],
        target_angular_acceleration=[0.0, 0.0, 0.0],
        control_mode=[0] * 6,
        integration_enabled=[
            True,
            True,
            True,
            roll_pitch_enabled,
            roll_pitch_enabled,
            True,
        ],
        flight_state=5,
        force_landing=False,
        initial_height=0.0,
        current_rpy=[0.0, 0.0, 0.0],
        target_rpy=[0.0, 0.0, 0.0],
    )


class _PersistentFakeOracle:
    is_exact = True
    transport_is_persistent = True

    def __init__(self, identity):
        self.identity = identity
        self.requests = []

    def replay(self, payload):
        request = copy.deepcopy(payload)
        self.requests.append(request)
        job = request["jobs"][0]
        tick = job["ticks"][0]
        pose = copy.deepcopy(job["initial_pose_state"])
        allocation = copy.deepcopy(job["initial_allocation_state"])
        pose["pid"][0]["error_i"] += 1.0
        pose["previous_stamp"] = tick["stamp"]
        pose["previous_flight_state"] = tick["flight_state"]
        pose["previous_xy_control_mode"] = tick["xy_control_mode"]
        pose["previous_force_landing"] = tick["force_landing"]
        pose["has_previous_force_landing"] = True
        pose["start_roll_pitch_integration"] = True
        if "state_previous_stamp" in tick:
            pose["previous_stamp"] = tick["state_previous_stamp"]
        pending_events = int(pose["pending_events"])
        pose["pending_events"] = 0
        allocation["target_gimbal_angles"] = [0.1, 0.2, 0.3, 0.4]
        allocation["target_roll"] += 0.25
        allocation["target_pitch"] -= 0.5
        return ExactOracleReplayOutput(
            identity=self.identity,
            continuous={
                "command_timestamp": np.asarray([[tick["stamp"]]]),
                "pid_terms": np.arange(24, dtype=float).reshape(1, 24),
                "four_axis_command": np.asarray(
                    [[0.25, -0.5, 0.75, 1.0, 2.0, 3.0, 4.0]]
                ),
                "vectoring_force": np.arange(
                    8, dtype=float
                ).reshape(1, 8),
                "gimbal_command": np.asarray(
                    [[0.1, 0.2, 0.3, 0.4]]
                ),
                "allocation_internal": np.asarray(
                    [[
                        allocation["target_roll"],
                        allocation["target_pitch"],
                        0.625,
                    ]]
                ),
                "torque_allocation_matrix_inverse": np.arange(
                    24, dtype=float
                ).reshape(1, 24),
                "job_tick": np.asarray([[0.0, 0.0]]),
            },
            events=np.asarray([34 | pending_events]),
            final_states=(
                {"pose": pose, "allocation": allocation},
            ),
        )


class StatefulExactOracleTests(unittest.TestCase):
    def test_reply_decoder_preserves_immutable_final_states(self):
        identity = _identity()
        parsed = {
            "continuous": {"pid_terms": [[0.0]]},
            "events": [0],
            "final_states": [
                {
                    "pose": {"previous_stamp": 1.0},
                    "allocation": {
                        "target_gimbal_angles": [0.1, 0.2]
                    },
                }
            ],
        }
        output = _output_from_reply(parsed, identity)
        parsed["final_states"][0]["pose"]["previous_stamp"] = 99.0
        self.assertEqual(
            deep_thaw(output.final_states[0])["pose"][
                "previous_stamp"
            ],
            1.0,
        )
        with self.assertRaises(TypeError):
            output.final_states[0]["pose"] = {}

    def test_adapter_round_trips_final_state_between_one_tick_jobs(self):
        identity = _identity()
        oracle = _PersistentFakeOracle(identity)
        backend = StatefulExactOracleControllerBackend(oracle)
        backend.reset(
            _snapshot(identity),
            ControllerCoreState(
                target_gimbal_angles=(0.0, 0.0, 0.0, 0.0)
            ),
        )
        first = backend.step(_core_input(0.0))
        second = backend.step(_core_input(0.1))

        second_request = oracle.requests[1]
        self.assertEqual(
            second_request["jobs"][0]["initial_pose_state"][
                "previous_stamp"
            ],
            0.0,
        )
        self.assertEqual(
            second_request["jobs"][0]["initial_pose_state"]["pid"][0][
                "error_i"
            ],
            1.0,
        )
        self.assertEqual(
            second_request["jobs"][0]["initial_allocation_state"][
                "target_roll"
            ],
            0.25,
        )
        self.assertEqual(first.events, (2, 32))
        self.assertTrue(first.saturated)
        self.assertEqual(first.stamp, 0.0)
        self.assertEqual(second.stamp, 0.1)
        self.assertEqual(first.base_thrust, (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(
            first.gimbal_angle, (0.1, 0.2, 0.3, 0.4)
        )
        self.assertEqual(
            len(first.torque_allocation_matrix_inverse), 8
        )
        self.assertEqual(first.candidate_yaw_term, 0.625)
        self.assertEqual(first.four_axis_angles[2], 0.75)
        self.assertEqual(first.command.four_axis_angles[2], 0.75)
        self.assertEqual(backend.state.previous_stamp, 0.1)

    def test_pending_reset_event_is_imported_consumed_and_exported(self):
        identity = _identity()
        oracle = _PersistentFakeOracle(identity)
        backend = StatefulExactOracleControllerBackend(oracle)
        backend.reset(
            _snapshot(identity),
            ControllerCoreState(
                target_gimbal_angles=(0.0, 0.0, 0.0, 0.0),
                pending_events=1,
            ),
        )
        first = backend.step(_core_input(0.0))
        backend.step(_core_input(0.1))
        self.assertEqual(first.events, (1, 2, 32))
        self.assertEqual(backend.state.pending_events, 0)
        self.assertEqual(
            oracle.requests[1]["jobs"][0]["initial_pose_state"][
                "pending_events"
            ],
            0,
        )

    def test_mid_episode_reset_precedes_integration_gate_validation(self):
        identity = _identity()
        reset_values = _core_input(0.2).to_mapping()
        reset_values["reset"] = True
        reset_values["position"][2] = 0.0
        reset_values["integration_enabled"][3] = False
        reset_values["integration_enabled"][4] = False

        payload = build_exact_replay_payload(
            _snapshot(identity),
            ControllerCoreState(
                start_roll_pitch_integration=True,
                target_gimbal_angles=(0.0, 0.0, 0.0, 0.0),
            ),
            (
                _core_input(0.1),
                ControllerCoreInput.from_mapping(reset_values),
            ),
        )

        ticks = payload["jobs"][0]["ticks"]
        self.assertTrue(ticks[1]["reset"])
        self.assertFalse(ticks[1]["integration_enabled"][3])
        self.assertFalse(ticks[1]["integration_enabled"][4])

    def test_one_shot_subprocess_transport_is_rejected(self):
        one_shot = object.__new__(SubprocessExactControllerOracle)
        with self.assertRaisesRegex(
            ExactOracleProtocolError, "one-shot subprocess"
        ):
            StatefulExactOracleControllerBackend(one_shot)

    def test_unsupported_per_axis_integration_fails_closed(self):
        identity = _identity()
        oracle = _PersistentFakeOracle(identity)
        backend = StatefulExactOracleControllerBackend(oracle)
        backend.reset(
            _snapshot(identity),
            ControllerCoreState(
                target_gimbal_angles=(0.0, 0.0, 0.0, 0.0)
            ),
        )
        values = _core_input(0.0).to_mapping()
        values["integration_enabled"][0] = False
        with self.assertRaisesRegex(
            ExactOracleProtocolError, "requires enabled X/Y/Z/YAW"
        ):
            backend.step(ControllerCoreInput.from_mapping(values))
        self.assertEqual(oracle.requests, [])

    def test_full_pid_config_event_persists_without_following_event(self):
        identity = _identity()
        oracle = _PersistentFakeOracle(identity)
        backend = StatefulExactOracleControllerBackend(oracle)
        backend.reset(
            _snapshot(identity),
            ControllerCoreState(
                target_gimbal_angles=(0.0, 0.0, 0.0, 0.0)
            ),
        )
        first_values = _core_input(0.0).to_mapping()
        first_values["pid_config"] = [
            [
                2.0,
                0.2,
                0.02,
                90.0,
                80.0,
                70.0,
                60.0,
                50.0,
                40.0,
                30.0,
            ]
            for _ in range(6)
        ]
        backend.step(ControllerCoreInput.from_mapping(first_values))
        backend.step(_core_input(0.1))

        self.assertEqual(
            oracle.requests[0]["jobs"][0]["ticks"][0][
                "pid_config"
            ][0],
            [
                2.0,
                0.2,
                0.02,
                90.0,
                80.0,
                70.0,
                60.0,
                50.0,
                40.0,
                30.0,
            ],
        )
        self.assertNotIn(
            "pid_config",
            oracle.requests[1]["jobs"][0]["ticks"][0],
        )
        self.assertEqual(
            oracle.requests[1]["snapshot"]["pid"][0]["p_gain"],
            2.0,
        )
        self.assertEqual(
            oracle.requests[1]["snapshot"]["pid"][0][
                "limit_error_d"
            ],
            30.0,
        )

    def test_state_previous_stamp_round_trips_distinct_completion_time(self):
        identity = _identity()
        oracle = _PersistentFakeOracle(identity)
        backend = StatefulExactOracleControllerBackend(oracle)
        backend.reset(
            _snapshot(identity),
            ControllerCoreState(
                target_gimbal_angles=(0.0, 0.0, 0.0, 0.0)
            ),
        )
        values = _core_input(0.0).to_mapping()
        values["state_previous_stamp"] = 0.0007
        output = backend.step(ControllerCoreInput.from_mapping(values))
        self.assertEqual(output.stamp, 0.0)
        self.assertEqual(backend.state.previous_stamp, 0.0007)

    def test_actual_cpp_server_protocol_round_trips_two_ticks(self):
        repository = Path(__file__).resolve().parents[2]
        executable = (
            repository.parents[1]
            / "devel/.private/gimbalrotor/lib/gimbalrotor"
            / "gimbalrotor_controller_replay"
        )
        if not executable.is_file():
            self.skipTest("built C++ controller replay executable unavailable")
        digest = hashlib.sha256(executable.read_bytes()).hexdigest()
        command = [
            str(executable),
            "--artifact-sha256",
            digest,
        ]
        handshake = subprocess.run(
            command,
            input=json.dumps(
                {
                    "protocol": EXACT_ORACLE_PROTOCOL,
                    "operation": "handshake",
                    "payload": {},
                }
            )
            + "\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5.0,
            check=True,
        )
        self.assertEqual(handshake.stderr, "")
        identity = ExactOracleIdentity.from_mapping(
            json.loads(handshake.stdout)["identity"]
        )
        with PersistentSubprocessExactControllerOracle(
            command,
            identity,
            timeout_s=5.0,
        ) as oracle:
            process_id = oracle._process.pid
            self.assertEqual(
                oracle.runtime_executable_sha256,
                digest,
            )
            backend = StatefulExactOracleControllerBackend(oracle)
            backend.reset(
                _snapshot(identity),
                ControllerCoreState(
                    target_gimbal_angles=(0.0, 0.0, 0.0, 0.0),
                    pending_events=1,
                ),
            )
            first = backend.step(_core_input(0.0))
            second = backend.step(_core_input(0.1))
            self.assertEqual(oracle._process.pid, process_id)
            self.assertIsNone(oracle._process.poll())
        self.assertEqual(first.stamp, 0.0)
        self.assertIn(1, first.events)
        self.assertEqual(second.stamp, 0.1)
        self.assertEqual(len(first.pid_result), 6)
        self.assertEqual(len(first.base_thrust), 4)
        self.assertEqual(len(first.gimbal_angle), 4)
        self.assertEqual(
            len(first.torque_allocation_matrix_inverse), 8
        )
        self.assertEqual(backend.state.previous_stamp, 0.1)
        self.assertEqual(backend.state.pending_events, 0)

    def test_actual_cpp_underactuated_feedback_uses_prior_allocation_state(self):
        repository = Path(__file__).resolve().parents[2]
        executable = (
            repository.parents[1]
            / "devel/.private/gimbalrotor/lib/gimbalrotor"
            / "gimbalrotor_controller_replay"
        )
        if not executable.is_file():
            self.skipTest(
                "built C++ controller replay executable unavailable"
            )
        digest = hashlib.sha256(executable.read_bytes()).hexdigest()
        command = [
            str(executable),
            "--artifact-sha256",
            digest,
        ]
        handshake = subprocess.run(
            command,
            input=json.dumps(
                {
                    "protocol": EXACT_ORACLE_PROTOCOL,
                    "operation": "handshake",
                    "payload": {},
                }
            )
            + "\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5.0,
            check=True,
        )
        identity = ExactOracleIdentity.from_mapping(
            json.loads(handshake.stdout)["identity"]
        )
        snapshot = _snapshot(
            identity, underactuate=True, integral_gain=0.0
        )
        initial_roll = 0.123456789012345
        initial_pitch = -0.234567890123456
        first_values = _core_input(0.0).to_mapping()
        first_values["target_position"] = [1.0, 2.0, 1.0]
        first_values["target_rpy"] = [0.0, 0.0, 0.5]
        second_values = _core_input(0.1).to_mapping()
        second_values["target_position"] = [1.0, 2.0, 1.0]
        second_values["target_rpy"] = [0.0, 0.0, 0.5]
        with PersistentSubprocessExactControllerOracle(
            command, identity, timeout_s=5.0
        ) as oracle:
            backend = StatefulExactOracleControllerBackend(oracle)
            backend.reset(
                snapshot,
                ControllerCoreState(
                    target_gimbal_angles=(0.0, 0.0, 0.0, 0.0),
                    target_roll=initial_roll,
                    target_pitch=initial_pitch,
                ),
            )
            first = backend.step(
                ControllerCoreInput.from_mapping(first_values)
            )
            second = backend.step(
                ControllerCoreInput.from_mapping(second_values)
            )

        self.assertAlmostEqual(first.pid_result[3], initial_roll)
        self.assertAlmostEqual(first.pid_result[4], initial_pitch)
        self.assertAlmostEqual(second.pid_result[3], first.target_roll)
        self.assertAlmostEqual(
            second.pid_result[4], first.target_pitch
        )
        self.assertEqual(first.four_axis_angles[2], 0.0)
        self.assertNotEqual(first.candidate_yaw_term, 0.0)
        self.assertNotEqual(
            first.command.four_axis_angles[2],
            first.candidate_yaw_term,
        )


if __name__ == "__main__":
    unittest.main()
